"""
EXP18 — FEDERATED CONFIRMATION of the noise -> counterfactual-validity effect
=============================================================================
Question (paper, Section "Federated confirmation"):
    Does DP-style Gaussian gradient noise applied on the CLIENTS (local
    training) still flatten the decision boundary and raise counterfactual
    validity after FedAvg aggregation?

Design
------
* K clients (default 10); IID partition of the P1 training split, or Dirichlet
  label skew with --alpha (smaller alpha = more non-IID).
* R rounds (default 20) x E local epochs (default 1): total local compute equals
  R*E full-data epochs, i.e. comparable to the 20-epoch centralised P1 runs.
* Conditions use EXACTLY the exp14 code path (common.train_condition) on each
  client: A_Standard, C_Noise_Only (default); D_Clip_Only / B_DP_SGD optional
  via --conditions.  Same seed -> same partition + same init for all conditions
  (paired design; Wilcoxon on seeds).
* Server: FedAvg (weighted by client sample count).  BatchNorm running
  statistics are averaged like weights; integer buffers are taken from the
  first client.
* Per run on the GLOBAL model: accuracy, macro-F1, CF validity (same generator
  as exp14: cf_samples attack instances x cf_max_iter iterations), boundary
  geometry (margin-gradient norm, Hutchinson Tr(H_x), softmax/logit margins),
  ECE, weight norm.
* Output: experiments_journal/exp18_federated_<dataset>/results.json + log.txt

Honesty note (goes into the paper): averaging K independently noised client
models divides the noise variance by ~K, so at the same nominal epsilon the
effective noise under FedAvg is milder than in centralised training.  The
script therefore also reports the centralised exp14 reference values if
experiments_journal/exp14_mechanism_<dataset>/results.json exists.

Usage
-----
python3 -m journal_extension.exp18_federated --dataset ciciot2023 --n-runs 3 \
    --clients 10 --rounds 20 --local-epochs 1 --batch-size 256 \
    --cf-samples 100 --cf-max-iter 300
# non-IID:   --alpha 0.5        # all four conditions: --conditions A,C,D,B
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from journal_extension import common as C
from journal_extension.exp14_mechanism import (_input_grad_and_curvature,
                                                _decision_margins, _ece,
                                                _weight_norm)

COND_ALIASES = {"A": "A_Standard", "B": "B_DP_SGD", "C": "C_Noise_Only", "D": "D_Clip_Only"}
METRICS = ["accuracy", "f1_macro", "cf_validity", "cf_proximity", "cf_sparsity",
           "margin_grad_norm", "hessian_trace", "margin_prob", "margin_logit",
           "ece", "weight_norm"]
PAIRED_TEST_METRICS = ["cf_validity", "accuracy", "margin_grad_norm", "hessian_trace",
                       "margin_prob", "ece"]


# -----------------------------------------------------------------------------
# data partitioning
# -----------------------------------------------------------------------------
def partition_indices(y: torch.Tensor, n_clients: int, alpha: Optional[float],
                      rng: np.random.RandomState, min_size: int = 64) -> List[np.ndarray]:
    """IID (alpha=None) or Dirichlet label-skew partition of indices 0..N-1."""
    y_np = y.numpy() if hasattr(y, "numpy") else np.asarray(y)
    n = len(y_np)
    if alpha is None:
        perm = rng.permutation(n)
        return [p for p in np.array_split(perm, n_clients)]
    classes = np.unique(y_np)
    for _ in range(50):                       # retry until every client is non-trivial
        parts: List[List[int]] = [[] for _ in range(n_clients)]
        for c in classes:
            idx = rng.permutation(np.where(y_np == c)[0])
            props = rng.dirichlet(alpha * np.ones(n_clients))
            cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
            for k, chunk in enumerate(np.split(idx, cuts)):
                parts[k].extend(chunk.tolist())
        if min(len(p) for p in parts) >= min_size:
            break
    return [rng.permutation(np.asarray(p)) for p in parts]


# -----------------------------------------------------------------------------
# FedAvg
# -----------------------------------------------------------------------------
def fedavg(states: List[Dict[str, torch.Tensor]], weights: List[float]) -> Dict[str, torch.Tensor]:
    tot = float(sum(weights))
    out: Dict[str, torch.Tensor] = {}
    for key, ref in states[0].items():
        if ref.is_floating_point():
            acc = torch.zeros_like(ref, dtype=torch.float32)
            for s, w in zip(states, weights):
                acc += (w / tot) * s[key].float()
            out[key] = acc.to(ref.dtype)
        else:                                   # e.g. BatchNorm num_batches_tracked
            out[key] = ref.clone()
    return out


def train_federated(global_model: nn.Module, parts: List[Tuple[torch.Tensor, torch.Tensor]],
                    *, rounds: int, local_epochs: int, use_clipping: bool, use_noise: bool,
                    noise_std: float, criterion, device: str, log=None, tag: str = "") -> nn.Module:
    """FedAvg with common.train_condition as the local trainer (exp14 code path)."""
    n_k = [len(p[0]) for p in parts]
    loaders = [DataLoader(TensorDataset(px, py), batch_size=C.BATCH_SIZE, shuffle=True)
               for px, py in parts]
    for r in range(rounds):
        states = []
        for loader in loaders:
            local = copy.deepcopy(global_model).to(device)
            C.train_condition(local, loader, use_clipping=use_clipping, use_noise=use_noise,
                              noise_std=noise_std, epochs=local_epochs, lr=C.LEARNING_RATE,
                              criterion=criterion, device=device)
            states.append({k: v.detach().cpu() for k, v in local.state_dict().items()})
            del local
        global_model.load_state_dict(fedavg(states, n_k))
        if log is not None and ((r + 1) % 5 == 0 or r == rounds - 1):
            log.log(f"    {tag} round {r+1}/{rounds} done")
    global_model.to(device).eval()
    return global_model


# -----------------------------------------------------------------------------
def _paired_tests(a: List[float], b: List[float]) -> Dict:
    """Wilcoxon (paired, same seeds) + Mann-Whitney; rank-biserial r for both."""
    from scipy import stats
    out = {"n": min(len(a), len(b))}
    if out["n"] >= 2 and any(x != y for x, y in zip(a, b)):
        d = np.asarray(b[:out["n"]]) - np.asarray(a[:out["n"]])
        try:
            w = stats.wilcoxon(a[:out["n"]], b[:out["n"]])
            ranks = stats.rankdata(np.abs(d[d != 0]))
            r_pos = ranks[d[d != 0] > 0].sum(); r_neg = ranks[d[d != 0] < 0].sum()
            out["wilcoxon_p"] = float(w.pvalue)
            out["wilcoxon_rank_biserial_r"] = float((r_pos - r_neg) / max(ranks.sum(), 1e-12))
        except ValueError as e:
            out["wilcoxon_p"] = None; out["wilcoxon_note"] = str(e)
        u = stats.mannwhitneyu(a, b, alternative="two-sided")
        out["mannwhitney_p"] = float(u.pvalue)
        out["mannwhitney_rank_biserial_r"] = float(1.0 - 2.0 * u.statistic / (len(a) * len(b)))
    return out


def _load_exp14_reference(dataset: str) -> Optional[Dict]:
    p = os.path.join(C.OUT_ROOT, f"exp14_mechanism_{dataset}", "results.json")
    if not os.path.exists(p):
        return None
    try:
        d = json.load(open(p, encoding="utf-8"))
        return {cond: {m: d["conditions"][cond][m]["mean"]
                       for m in ("accuracy", "cf_validity", "margin_grad_norm",
                                 "hessian_trace", "margin_prob", "ece")}
                for cond in d.get("conditions", {})}
    except Exception:
        return None


# -----------------------------------------------------------------------------
def run(dataset: str, n_runs: int = 3, max_samples=None, *, clients: int = 10,
        rounds: int = 20, local_epochs: int = 1, alpha: Optional[float] = None,
        conditions: str = "A,C", cf_samples: int = 100, cf_max_iter: int = C.CF_MAX_ITER,
        quick: bool = False, batch_size=None, epsilon: float = C.REFERENCE_EPSILON):
    log = C.Runlog("exp18_federated", dataset)
    device = C.get_device()
    C.set_batch_size(batch_size)
    if quick:
        max_samples = max_samples or 40000; rounds = 3; n_runs = 2; cf_samples = 20
        clients = min(clients, 4)
    conds = [COND_ALIASES.get(c.strip().upper(), c.strip()) for c in conditions.split(",")]
    for c in conds:
        if c not in C.CONDITIONS:
            raise ValueError(f"unknown condition {c}; use A,B,C,D")
    noise_std = C.sigma_from_epsilon(epsilon) * C.DP_CLIP_NORM
    log.log(f"device={device} n_runs={n_runs} clients={clients} rounds={rounds} "
            f"local_epochs={local_epochs} alpha={alpha} batch={C.BATCH_SIZE} "
            f"cf_samples={cf_samples} cf_max_iter={cf_max_iter} conditions={conds}")
    log.log(f"nominal eps={epsilon} -> per-client noise_std={noise_std:.4f} "
            f"(FedAvg of {clients} clients reduces effective noise variance ~1/{clients})")

    Xtr, Xte, ytr, yte, info = C.load_data(dataset, max_samples)
    log.log(f"train={len(Xtr)} test={len(Xte)} F={info['num_features']} C={info['num_classes']}")
    _, test_loader = C.make_loaders(Xtr, ytr, Xte, yte)
    cw = C.class_weights_tensor(ytr, info["num_classes"], device)
    criterion = nn.CrossEntropyLoss(weight=cw)

    agg = {c: {m: [] for m in METRICS} for c in conds}
    part_sizes: List[List[int]] = []

    for run_i in range(n_runs):
        seed = C.RANDOM_SEED + run_i
        rng = np.random.RandomState(seed)
        idx_parts = partition_indices(ytr, clients, alpha, rng)
        parts = [(Xtr[torch.as_tensor(ix)], ytr[torch.as_tensor(ix)]) for ix in idx_parts]
        part_sizes.append([len(p[0]) for p in parts])
        log.log(f"\n=== run {run_i+1}/{n_runs} (seed {seed}) client sizes "
                f"min/max={min(part_sizes[-1])}/{max(part_sizes[-1])} ===")
        for cond in conds:
            clip, noise = C.CONDITIONS[cond]
            C.set_seed(seed)                              # same init for every condition
            model = C.make_model("mlp", info).to(device)
            model = train_federated(model, parts, rounds=rounds, local_epochs=local_epochs,
                                    use_clipping=clip, use_noise=noise,
                                    noise_std=(noise_std if noise else 0.0),
                                    criterion=criterion, device=device, log=log, tag=cond)
            ev = C.evaluate_model(model, test_loader, device)
            attacks = C.get_attack_samples(model, Xte, yte, cf_samples, device)
            if attacks is not None:
                cf = C.eval_cf(model, attacks, info, device, max_samples=cf_samples,
                               max_iter=cf_max_iter)
            else:
                cf = {"validity": 0.0, "proximity": 0.0, "sparsity": 0.0}
            probe = attacks if attacks is not None else Xte
            _, g_margin, h_tr = _input_grad_and_curvature(model, probe, device)
            m_p, m_l = _decision_margins(model, probe, device)
            vals = {
                "accuracy": ev["accuracy"], "f1_macro": ev["f1_macro"],
                "cf_validity": cf["validity"], "cf_proximity": cf.get("proximity", 0.0),
                "cf_sparsity": cf.get("sparsity", 0.0),
                "margin_grad_norm": float(np.mean(g_margin)),
                "hessian_trace": float(np.mean(h_tr)),
                "margin_prob": float(np.mean(m_p)), "margin_logit": float(np.mean(m_l)),
                "ece": _ece(model, test_loader, device), "weight_norm": _weight_norm(model),
            }
            for m in METRICS:
                agg[cond][m].append(vals[m])
            log.log(f"  {cond}: acc={vals['accuracy']:.3f} f1={vals['f1_macro']:.3f} "
                    f"val={vals['cf_validity']:.3f} prox={vals['cf_proximity']:.3f} "
                    f"|dMargin/dx|={vals['margin_grad_norm']:.3f} TrH={vals['hessian_trace']:.3f} "
                    f"margin={vals['margin_prob']:.3f} ECE={vals['ece']:.3f}")
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    # ---- summary -----------------------------------------------------------
    summary = {
        "fl_config": {"clients": clients, "rounds": rounds, "local_epochs": local_epochs,
                      "alpha": alpha, "batch_size": C.BATCH_SIZE, "nominal_epsilon": epsilon,
                      "per_client_noise_std": noise_std, "cf_samples": cf_samples,
                      "cf_max_iter": cf_max_iter, "n_runs": n_runs,
                      "client_sizes_per_run": part_sizes},
        "conditions": {c: {m: {"mean": C.mean_std(agg[c][m])[0], "std": C.mean_std(agg[c][m])[1],
                               "raw": agg[c][m]} for m in METRICS} for c in conds},
        "contrasts": {},
        "centralised_reference_exp14": _load_exp14_reference(dataset),
        "effect_size_convention": "rank-biserial r (Wilcoxon paired on seeds; Mann-Whitney unpaired)",
    }
    if "A_Standard" in conds:
        for c in conds:
            if c == "A_Standard":
                continue
            summary["contrasts"][f"{c}_minus_A_Standard"] = {
                m: {"delta": float(np.mean(agg[c][m]) - np.mean(agg["A_Standard"][m])),
                    **(_paired_tests(agg["A_Standard"][m], agg[c][m]) if m in PAIRED_TEST_METRICS else {})}
                for m in METRICS}
    if "C_Noise_Only" in conds and "A_Standard" in conds:
        cc = summary["contrasts"]["C_Noise_Only_minus_A_Standard"]
        summary["hypothesis_check"] = {
            "basis": "FedAvg global model, C_Noise_Only vs A_Standard (paired seeds)",
            "validity_increases_with_noise": cc["cf_validity"]["delta"] > 0,
            "boundary_sharpness_decreases (margin_grad)": cc["margin_grad_norm"]["delta"] < 0,
            "curvature_decreases (Tr H)": cc["hessian_trace"]["delta"] < 0,
            "margin_decreases": cc["margin_prob"]["delta"] < 0,
            "accuracy_delta": cc["accuracy"]["delta"],
        }
        log.log("\n=== FEDERATED CONTRAST (C_Noise_Only - A_Standard, global model) ===")
        for m in METRICS:
            p = cc[m].get("wilcoxon_p", None)
            log.log(f"  {m:18s}: {cc[m]['delta']:+.4f}" + (f"  (Wilcoxon p={p:.4f})" if p is not None else ""))
        log.log(f"hypothesis_check: {summary['hypothesis_check']}")
    ref = summary["centralised_reference_exp14"]
    if ref:
        log.log("centralised exp14 reference (means): " +
                "; ".join(f"{c}: acc={v['accuracy']:.3f} val={v['cf_validity']:.3f} "
                          f"|dM/dx|={v['margin_grad_norm']:.2f}" for c, v in ref.items()))
    log.save_json(summary)
    return summary


def main():
    ap = argparse.ArgumentParser(description="EXP18 federated confirmation (FedAvg)")
    ap.add_argument("--dataset", default="ciciot2023",
                    choices=["ciciot2023", "edge-iiot", "toniot", "all"])
    ap.add_argument("--n-runs", type=int, default=3)
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--local-epochs", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=None,
                    help="Dirichlet label-skew concentration (omit = IID)")
    ap.add_argument("--conditions", default="A,C", help="subset of A,B,C,D (default A,C)")
    ap.add_argument("--epsilon", type=float, default=C.REFERENCE_EPSILON,
                    help="nominal budget for the noise level (default 2.0)")
    ap.add_argument("--cf-samples", type=int, default=100)
    ap.add_argument("--cf-max-iter", type=int, default=C.CF_MAX_ITER)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    dsets = ["ciciot2023", "edge-iiot", "toniot"] if a.dataset == "all" else [a.dataset]
    for d in dsets:
        run(d, n_runs=a.n_runs, max_samples=a.max_samples, clients=a.clients, rounds=a.rounds,
            local_epochs=a.local_epochs, alpha=a.alpha, conditions=a.conditions,
            cf_samples=a.cf_samples, cf_max_iter=a.cf_max_iter, quick=a.quick,
            batch_size=a.batch_size, epsilon=a.epsilon)


if __name__ == "__main__":
    main()
