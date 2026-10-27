"""
EXP20 - CERTIFIED DP-SGD ARM (Opacus)
=====================================
The main experiments inject DP-*style* noise (mini-batch clipping, unscaled
Gaussian noise, no accounting), so their sigma cannot be mapped to a certified
epsilon.  This experiment trains a genuinely (epsilon, delta)-DP detector with
Opacus (per-example clipping, C/B-scaled Gaussian noise, RDP accounting,
Poisson sampling) at target budgets and measures the SAME quantities as exp14:
accuracy, macro-F1, CF validity, boundary geometry (sharpness, curvature,
margins), the first-order distance d_lin, ECE.

Design notes
------------
* Opacus does not support BatchNorm; ModuleValidator.fix() replaces the
  detector's BatchNorm1d layers by GroupNorm.  To keep the comparison fair the
  NON-private baseline of this experiment uses the same GroupNorm model
  (arm "GN_baseline").  Numbers are therefore comparable within this
  experiment, and only approximately with the BatchNorm detector of exp14.
* Arms: GN_baseline (no DP) and DP at each --target-eps (default 2 and 8),
  delta = 1e-5, max_grad_norm = 1.0, batch 256, --epochs (default 20).
  The noise multiplier is chosen by Opacus for the target epsilon; the epsilon
  actually spent is read back from the accountant and stored.
* Queries: instances the detector predicts as non-target (exp14 rule).

Install: pip install opacus     (tested with opacus 1.6, torch 2.7)
Run:
  python3 -m journal_extension.exp20_opacus --dataset ciciot2023 --max-samples 200000 \
      --batch-size 256 --n-runs 3 --target-eps 2,8 --cf-samples 100 --cf-max-iter 300
"""
from __future__ import annotations
import argparse
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from journal_extension import common as C
from journal_extension.exp14_mechanism import (_input_grad_and_curvature, _decision_margins,
                                                _ece, _weight_norm)

TARGET = 0


def _geometry(model, X, device) -> Dict[str, float]:
    _, gm, htr = _input_grad_and_curvature(model, X, device)
    mp, ml = _decision_margins(model, X, device)
    gm_m = float(np.mean(gm)); ml_m = float(np.mean(ml))
    return {"margin_grad_norm": gm_m, "hessian_trace": float(np.mean(htr)),
            "margin_prob": float(np.mean(mp)), "margin_logit": ml_m, "d_lin": ml_m / max(gm_m, 1e-12)}


def make_gn_model(info, arch: str = "gn"):
    """Opacus-compatible detector.
       gn     : the paper's MLP with BatchNorm1d replaced by GroupNorm (ModuleValidator.fix)
       nonorm : the paper's MLP with BatchNorm1d replaced by Identity (no normalisation at all);
                Opacus accepts it unchanged, and the sigma-indexed treatment can be trained on the
                SAME network, which makes the two treatments comparable."""
    from opacus.validators import ModuleValidator
    m = C.make_model("mlp", info)
    if arch == "nonorm":
        for name, mod in list(m.named_modules()):
            for cname, child in list(mod.named_children()):
                if isinstance(child, nn.BatchNorm1d):
                    setattr(mod, cname, nn.Identity())
    else:
        m = ModuleValidator.fix(m)          # BatchNorm1d -> GroupNorm
    ModuleValidator.validate(m, strict=True)
    return m


_LAST_INFO = {}
def make_gn_model_like(core):
    """Fresh model with the same architecture as `core` (needs the dataset info and arch)."""
    return make_gn_model(_LAST_INFO["info"], _LAST_INFO.get("arch", "gn"))


def train_plain(model, loader, epochs, lr, criterion, device):
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad(); criterion(model(bx), by).backward(); opt.step()
    return model.eval()


def train_opacus(model, loader, epochs, lr, criterion, device, target_eps, delta, max_grad_norm, log):
    from opacus import PrivacyEngine
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    pe = PrivacyEngine(accountant="rdp")
    model, opt, loader = pe.make_private_with_epsilon(
        module=model, optimizer=opt, data_loader=loader, epochs=epochs,
        target_epsilon=target_eps, target_delta=delta, max_grad_norm=max_grad_norm)
    log.log(f"    Opacus: target eps={target_eps} delta={delta} -> noise_multiplier={opt.noise_multiplier:.4f} "
            f"(per-example clip C={max_grad_norm}, Poisson sampling)")
    for ep in range(epochs):
        for bx, by in loader:
            if len(bx) == 0:
                continue
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad(); criterion(model(bx), by).backward(); opt.step()
    eps_spent = pe.get_epsilon(delta)
    core = model._module if hasattr(model, "_module") else model
    # copy weights into a hook-free model (GradSampleModule hooks break input-gradient probes)
    try:
        model.remove_hooks()
    except Exception:  # noqa: BLE001
        pass
    fresh = make_gn_model_like(core).to(device)
    fresh.load_state_dict(core.state_dict())
    return fresh.eval(), float(eps_spent), float(opt.noise_multiplier)


def run(dataset: str, n_runs: int = 3, max_samples=None, epochs: int = C.EPOCHS,
        cf_samples: int = 100, cf_max_iter: int = C.CF_MAX_ITER, target_eps: str = "2,8",
        delta: float = C.DP_DELTA, max_grad_norm: float = 1.0, batch_size=None, quick: bool = False,
        arch: str = "gn", sigma_arms: bool = False, sigma: Optional[float] = None, out_suffix: str = ""):
    log = C.Runlog("exp20_opacus" + out_suffix, dataset)
    device = C.get_device(); C.set_batch_size(batch_size)
    if quick:
        max_samples = max_samples or 40000; epochs = 2; n_runs = 1; cf_samples = 20
    eps_list = [float(e) for e in target_eps.split(",")]
    noise_std = float(sigma) if sigma is not None else C.sigma_from_epsilon(C.REFERENCE_EPSILON) * C.DP_CLIP_NORM
    log.log(f"device={device} arch={arch} sigma_arms={sigma_arms} (sigma={noise_std:.4f}) n_runs={n_runs} "
            f"epochs={epochs} batch={C.BATCH_SIZE} target_eps={eps_list} delta={delta} C={max_grad_norm} "
            f"cf_samples={cf_samples} cf_max_iter={cf_max_iter}")
    Xtr, Xte, ytr, yte, info = C.load_data(dataset, max_samples)
    _LAST_INFO["info"] = info; _LAST_INFO["arch"] = arch
    train_loader, test_loader = C.make_loaders(Xtr, ytr, Xte, yte)
    cw = C.class_weights_tensor(ytr, info["num_classes"], device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    log.log(f"train={len(Xtr)} test={len(Xte)} F={info['num_features']} C={info['num_classes']}")

    base_name = "GN_baseline" if arch == "gn" else "NN_baseline"
    arms = [base_name] + (["SIG_C_noise_only", "SIG_B_clip_noise"] if sigma_arms else []) + [f"DP_eps{e:g}" for e in eps_list]
    agg: Dict[str, Dict[str, List[float]]] = {a: {} for a in arms}
    def push(a, k, v): agg[a].setdefault(k, []).append(float(v))

    for run_i in range(n_runs):
        seed = C.RANDOM_SEED + run_i
        log.log(f"\n=== run {run_i+1}/{n_runs} (seed {seed}) ===")
        for arm in arms:
            C.set_seed(seed); model = make_gn_model(info, arch).to(device)
            if arm == base_name:
                model = train_plain(model, train_loader, epochs, C.LEARNING_RATE, criterion, device)
                eps_spent, nm = float("inf"), 0.0
            elif arm.startswith("SIG_"):
                clip = arm == "SIG_B_clip_noise"
                C.train_condition(model, train_loader, use_clipping=clip, use_noise=True, noise_std=noise_std,
                                  epochs=epochs, lr=C.LEARNING_RATE, criterion=criterion, device=device)
                model.eval(); eps_spent, nm = float("inf"), noise_std
            else:
                e = float(arm.replace("DP_eps", ""))
                model, eps_spent, nm = train_opacus(model, train_loader, epochs, C.LEARNING_RATE, criterion,
                                                    device, e, delta, max_grad_norm, log)
            ev = C.evaluate_model(model, test_loader, device)
            queries = C.get_attack_samples(model, Xte, yte, cf_samples, device)
            if queries is None:
                log.log(f"  {arm}: no queries"); continue
            cf = C.eval_cf(model, queries, info, device, max_samples=cf_samples, max_iter=cf_max_iter)
            geo = _geometry(model, queries, device)
            ece = _ece(model, test_loader, device)
            push(arm, "accuracy", ev["accuracy"]); push(arm, "f1_macro", ev["f1_macro"])
            push(arm, "cf_validity", cf["validity"]); push(arm, "cf_proximity", cf.get("proximity", 0.0))
            push(arm, "ece", ece); push(arm, "eps_spent", eps_spent); push(arm, "noise_multiplier", nm)
            push(arm, "weight_norm", _weight_norm(model))
            for k, v in geo.items(): push(arm, k, v)
            log.log(f"  {arm:12s}: eps_spent={eps_spent:.3f} sigma_mult={nm:.3f} acc={ev['accuracy']:.3f} "
                    f"F1={ev['f1_macro']:.3f} val={cf['validity']:.3f} |dm/dx|={geo['margin_grad_norm']:.3f} "
                    f"TrH={geo['hessian_trace']:.3f} m={geo['margin_logit']:.3f} d_lin={geo['d_lin']:.3f} ECE={ece:.3f}")
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    summary = {"config": {"n_runs": n_runs, "epochs": epochs, "batch": C.BATCH_SIZE, "target_eps": eps_list,
                          "delta": delta, "max_grad_norm": max_grad_norm, "cf_samples": cf_samples,
                          "cf_max_iter": cf_max_iter, "arch": arch, "sigma_arms": sigma_arms, "sigma": noise_std,
                          "model": ("IDSMLP with GroupNorm (Opacus ModuleValidator.fix)" if arch == "gn"
                                    else "IDSMLP without normalisation (BatchNorm1d -> Identity)"),
                          "accountant": "RDP", "sampling": "Poisson (expected batch size = batch)"},
               "arms": {a: {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "raw": v}
                            for k, v in d.items()} for a, d in agg.items() if d},
               "contrasts": {}}
    base = agg.get(base_name, {})
    for arm in arms[1:]:
        if arm in agg and agg[arm] and base:
            summary["contrasts"][f"{arm}_minus_{base_name}"] = {
                k: float(np.mean(agg[arm][k]) - np.mean(base[k])) for k in base if k in agg[arm]}
    log.log(f"\n=== contrasts vs {base_name} (mean deltas) ===")
    for k, v in summary["contrasts"].items():
        log.log(f"  {k}: " + ", ".join(f"{m}={d:+.3f}" for m, d in v.items() if m in
                                        ("accuracy", "f1_macro", "cf_validity", "margin_grad_norm", "margin_logit", "d_lin", "ece")))
    log.save_json(summary)
    return summary


def main():
    ap = argparse.ArgumentParser(description="EXP20 certified DP-SGD arm (Opacus)")
    ap.add_argument("--dataset", default="ciciot2023", choices=["ciciot2023", "edge-iiot", "toniot"])
    ap.add_argument("--n-runs", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--target-eps", default="2,8", help="comma-separated target epsilons (delta=1e-5)")
    ap.add_argument("--delta", type=float, default=C.DP_DELTA)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--cf-samples", type=int, default=100)
    ap.add_argument("--cf-max-iter", type=int, default=C.CF_MAX_ITER)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--arch", default="gn", choices=["gn", "nonorm"],
                    help="gn = BatchNorm->GroupNorm (paper v1 arm); nonorm = BatchNorm removed (same net for both treatments)")
    ap.add_argument("--sigma-arms", action="store_true",
                    help="also train the sigma-indexed noise-only and clip+noise arms on the same architecture")
    ap.add_argument("--sigma", type=float, default=None, help="sigma for the sigma arms (default 2.4224)")
    ap.add_argument("--out-suffix", default="", help="output folder suffix, e.g. _nonorm")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    run(a.dataset, n_runs=a.n_runs, max_samples=a.max_samples, epochs=a.epochs, cf_samples=a.cf_samples,
        cf_max_iter=a.cf_max_iter, target_eps=a.target_eps, delta=a.delta, max_grad_norm=a.max_grad_norm,
        batch_size=a.batch_size, quick=a.quick, arch=a.arch, sigma_arms=a.sigma_arms, sigma=a.sigma,
        out_suffix=a.out_suffix)


if __name__ == "__main__":
    main()
