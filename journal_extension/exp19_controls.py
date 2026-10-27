"""
EXP19 - CONTROLS for the noise -> counterfactual-validity effect
================================================================
Controls that test the alternative explanations of the effect with the
existing pipeline (paper Section 6.3 and Appendix E):

  hist      per-condition test-set predicted-class histogram, share of
            predictions in the target (benign) class, accuracy AND macro-F1;
            query-selection rule report: queries are instances the
            detector predicts as non-target; we record the share already in the
            target class at iteration 0 (must be 0) and the share whose TRUE
            label is an attack, and report validity on the true-attack subset.
  random    matched-norm random-perturbation control: for every query
            and its best counterfactual, K random perturbations with the SAME
            L1 norm and the SAME number of changed features; the rate at which
            they flip the decision to the target is the "trivial validity".
  curves    validity-vs-iteration curves with early stopping DISABLED,
            recorded at fixed checkpoints, per condition (best-so-far validity,
            exactly as the generator scores it).
  deepfool  empirical minimal flipping distance to the target class
            (targeted DeepFool-style linearisation, up to 50 steps), compared
            with the first-order distance d_lin = m/||dm/dx||.
  smooth    non-DP smoothing arms trained with the SAME loop: label
            smoothing, input Gaussian noise augmentation, weight decay (AdamW),
            SAM (rho=0.05); validity + geometry + d_lin for each.
  discrete  gradient-free counterfactual search (random-restart local
            search sampling feature values from the training marginals, <=10
            features changed, generic range) - does the noise effect survive a
            non-gradient generator?
  transfer  evasion transfer: counterfactuals generated on one detector
            are fed to other detectors (same architecture other seed, 1D-CNN,
            and the noised/non-noised counterpart); flip rate = evasion success.

Every part reuses common.train_condition / common.eval_cf / the geometry
functions of exp14, so numbers are comparable with the main tables.

Part sets
---------
  diag      optimiser diagnostics of the generator (Adam moment ablation; effective step norm,
            gradient sign flips, |m_hat|/sqrt(v_hat), classes visited; early vs late, ok vs fail)
  core (default) = hist,random,curves,discrete,deepfool -> what the manuscript uses
                   (Section "Controls": trivial-validity baseline, class histogram,
                   budget curves, gradient-free generator, targeted first-order and
                   DeepFool distances to the benign class)
  all            = core + smooth,transfer (optional extras, not used in the paper)

Usage (CICIoT2023, protocol P1):
  python3 -m journal_extension.exp19_controls --dataset ciciot2023 --max-samples 200000 \
      --batch-size 256 --n-runs 3 --cf-samples 100 --cf-max-iter 300
  # extras:  --parts all     or     --parts smooth,transfer
"""
from __future__ import annotations
import argparse, copy, math, time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from journal_extension import common as C
from journal_extension.exp14_mechanism import (_input_grad_and_curvature, _decision_margins,
                                                _ece, _weight_norm)
from counterfactual.cf_generator import CounterfactualGenerator  # noqa: E402
from counterfactual.constraints import IoTConstraintManager  # noqa: E402

TARGET = 0          # benign class id in all three loaders
COND4 = ["A_Standard", "D_Clip_Only", "C_Noise_Only", "B_DP_SGD"]
CHECKPOINTS = [10, 25, 50, 75, 100, 150, 200, 300, 500, 750, 1000]


# =============================================================================
# helpers
# =============================================================================
def _ms(xs):
    xs = [float(x) for x in xs]
    return (float(np.mean(xs)) if xs else 0.0, float(np.std(xs)) if xs else 0.0)


def _first_order_distances(model, X, device, target: int = TARGET, max_pts: int = 256) -> Dict[str, float]:
    """Per-query first-order distances on the SAME points:
      d_lin_pt     = (z_top1 - z_top2) / ||grad_x (z_top1 - z_top2)||   (nearest boundary, as in exp14)
      d_lin_target = (z_top1 - z_target) / ||grad_x (z_top1 - z_target)||  (boundary to the CF target class)
    Reported as mean and median over queries; the targeted version is the quantity a
    counterfactual generator towards `target` actually has to cross."""
    model.eval(); d_pt, d_tg, m_tg, g_tg = [], [], [], []
    for i in range(min(max_pts, len(X))):
        x = X[i:i + 1].to(device).clone().detach().requires_grad_(True)
        logits = model(x)
        top2v, top2i = torch.topk(logits, 2, dim=1)
        k = int(top2i[0, 0].item())
        m2 = (top2v[0, 0] - top2v[0, 1])
        g2, = torch.autograd.grad(m2, x, retain_graph=True)
        d_pt.append(float(m2.item()) / (float(g2.norm().item()) + 1e-12))
        if k == target:      # already at the target: distance 0 by convention
            d_tg.append(0.0); m_tg.append(0.0); g_tg.append(float("nan")); continue
        mt = (logits[0, k] - logits[0, target])
        gt, = torch.autograd.grad(mt, x)
        gn = float(gt.norm().item()) + 1e-12
        d_tg.append(float(mt.item()) / gn); m_tg.append(float(mt.item())); g_tg.append(gn)
    return {"d_lin_pt_mean": float(np.mean(d_pt)), "d_lin_pt_median": float(np.median(d_pt)),
            "d_lin_target_mean": float(np.mean(d_tg)), "d_lin_target_median": float(np.median(d_tg)),
            "margin_target_logit": float(np.mean(m_tg)),
            "margin_target_grad_norm": float(np.nanmean(g_tg)) if np.isfinite(np.nanmean(g_tg)) else 0.0}


def _geometry(model, X, device) -> Dict[str, float]:
    _, gm, htr = _input_grad_and_curvature(model, X, device)
    mp, ml = _decision_margins(model, X, device)
    gm = np.asarray(gm); ml_mean = float(np.mean(ml))
    out = {"margin_grad_norm": float(np.mean(gm)), "hessian_trace": float(np.mean(htr)),
           "margin_prob": float(np.mean(mp)), "margin_logit": ml_mean,
           "d_lin": float(ml_mean / max(np.mean(gm), 1e-12))}
    out.update(_first_order_distances(model, X, device))
    return out


def _pred_hist(model, X, device, num_classes: int, bs: int = 4096) -> Tuple[np.ndarray, float]:
    model.eval(); counts = np.zeros(num_classes, dtype=np.int64)
    with torch.no_grad():
        for i in range(0, len(X), bs):
            p = model(X[i:i + bs].to(device)).argmax(1).cpu().numpy()
            counts += np.bincount(p, minlength=num_classes)
    return counts, float(counts[TARGET] / max(counts.sum(), 1))


def _attack_queries(model, Xte, yte, n, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Queries = instances the detector predicts as non-target (exp14 rule); return with true labels."""
    model.eval(); xs, ys = [], []
    with torch.no_grad():
        for i in range(len(Xte)):
            if model(Xte[i:i + 1].to(device)).argmax(1).item() != TARGET:
                xs.append(Xte[i]); ys.append(int(yte[i]))
                if len(xs) >= n:
                    break
    return (torch.stack(xs), torch.tensor(ys)) if xs else (None, None)


def _predict(model, X: torch.Tensor, device) -> np.ndarray:
    with torch.no_grad():
        return model(X.to(device)).argmax(1).cpu().numpy()


# =============================================================================
# training variants for the smoothing arms
# =============================================================================
def train_variant(model, loader, *, variant: str, epochs: int, lr: float, criterion_plain,
                  class_weights, device: str, input_noise_std: float = 0.3,
                  label_smoothing: float = 0.1, weight_decay: float = 1e-2, sam_rho: float = 0.05):
    """Same loop as common.train_condition (Adam, no clipping, no gradient noise) plus one smoother."""
    model.to(device).train()
    if variant == "G_WeightDecay":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    if variant == "E_LabelSmooth":
        # Per-sample class weight w_y on the smoothed cross-entropy; the uniform component is
        # NOT re-weighted. torch's CrossEntropyLoss(weight=w, label_smoothing=eps) multiplies the
        # uniform part by w_k, which with the inverse-frequency weights of
        # common.class_weights_tensor (rarest CICIoT2023 class ~1e3) puts almost all smoothing
        # mass on the rarest classes and collapses the detector (accuracy 0.000 in the
        # 18 Oct 2026 run). At eps=0 this loss equals the weighted CE of every other condition.
        def criterion(logits, target, _w=class_weights, _eps=label_smoothing):
            logp = F.log_softmax(logits, dim=1)
            nll = -logp.gather(1, target.unsqueeze(1)).squeeze(1)
            per = (1.0 - _eps) * nll + _eps * (-logp.mean(dim=1))
            wy = _w[target]
            return (wy * per).sum() / wy.sum()
    else:
        criterion = criterion_plain
    for _ in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            if variant == "F_InputNoise":
                bx = bx + torch.randn_like(bx) * input_noise_std
            if variant == "H_SAM":
                # Sharpness-Aware Minimisation (Foret et al. 2021), two-step
                loss = criterion(model(bx), by); opt.zero_grad(); loss.backward()
                grads = [p.grad.detach().clone() if p.grad is not None else None for p in model.parameters()]
                gnorm = torch.sqrt(sum((g ** 2).sum() for g in grads if g is not None)) + 1e-12
                eps_w = []
                with torch.no_grad():
                    for p, g in zip(model.parameters(), grads):
                        e = (sam_rho * g / gnorm) if g is not None else None
                        if e is not None:
                            p.add_(e)
                        eps_w.append(e)
                opt.zero_grad(); criterion(model(bx), by).backward()
                with torch.no_grad():
                    for p, e in zip(model.parameters(), eps_w):
                        if e is not None:
                            p.sub_(e)
                opt.step(); continue
            opt.zero_grad(); criterion(model(bx), by).backward(); opt.step()
    model.eval(); return model


# =============================================================================
# tracing generator: identical loop, early stopping disabled, history kept
# =============================================================================
class TracingGenerator(CounterfactualGenerator):
    def generate_traced(self, query_instance: torch.Tensor, target_class: int = TARGET,
                        checkpoints: List[int] = CHECKPOINTS) -> Dict[int, float]:
        query = query_instance.to(self.device)
        if query.dim() == 1:
            query = query.unsqueeze(0)
        cfs = self._initialize_cfs(query)
        optimizer = torch.optim.Adam(cfs, lr=self.learning_rate)
        best = 0.0; hist: Dict[int, float] = {}
        for it in range(1, self.max_iter + 1):
            optimizer.zero_grad()
            cf_batch = torch.stack(cfs)
            loss = (self.lambda_pred * self._prediction_loss(cf_batch, target_class) +
                    self.lambda_dist * self._distance_loss(cf_batch, query) +
                    self.lambda_div * self._diversity_loss(cf_batch) +
                    self.lambda_feas * self._feasibility_loss(cf_batch, query))
            loss.backward(); optimizer.step()
            with torch.no_grad():
                for i, cf in enumerate(cfs):
                    cfs[i].data = self.constraint_manager.apply_all(query.squeeze(0), cf)
                preds = self.model.predict(torch.stack(cfs))
                v = (preds == target_class).float().mean().item()
                best = max(best, v)            # best-so-far, as in the original scorer
            if it in checkpoints:
                hist[it] = best
        return hist


# =============================================================================
# targeted DeepFool-style minimal flipping distance
# =============================================================================
def deepfool_target_distance(model, X: torch.Tensor, device, target: int = TARGET,
                             max_steps: int = 50, overshoot: float = 0.02, max_pts: int = 100,
                             box: float = 5.0) -> Tuple[List[float], float]:
    """L2 norm of the iterative linearised minimal perturbation that makes `target` the top class."""
    model.eval(); dists = []; reached = 0
    for i in range(min(max_pts, len(X))):
        x0 = X[i:i + 1].to(device); x = x0.clone(); r_tot = torch.zeros_like(x0)
        for _ in range(max_steps):
            x = x.detach().requires_grad_(True)
            logits = model(x)
            k = int(logits.argmax(1).item())
            if k == target:
                break
            m = (logits[0, k] - logits[0, target])           # positive while not target
            g, = torch.autograd.grad(m, x)
            gn = float(g.norm().item()) + 1e-12
            r = -(float(m.item()) + 1e-4) / gn ** 2 * g         # step DOWN the margin to the linearised boundary
            r_tot = r_tot + r
            x = torch.clamp(x0 + (1 + overshoot) * r_tot, -box, box)
        with torch.no_grad():
            ok = int(model(x).argmax(1).item()) == target
        if ok:
            reached += 1; dists.append(float((x - x0).norm().item()))
    return dists, reached / max(1, min(max_pts, len(X)))


# =============================================================================
# sparse (k-feature) targeted DeepFool: minimal linearised perturbation restricted to the k
# coordinates with the largest |d(margin_target)/dx| at the query -- the geometry the
# sparsity-capped generator actually works in
# =============================================================================
def sparse_deepfool_target_distance(model, X: torch.Tensor, device, target: int = TARGET, k: int = 10,
                                    max_steps: int = 50, overshoot: float = 0.02, max_pts: int = 100,
                                    box: float = 5.0) -> Dict[str, object]:
    model.eval(); l2, l1, linf = [], [], []; reached = 0; n = min(max_pts, len(X))
    for i in range(n):
        x0 = X[i:i + 1].to(device); x = x0.clone(); r_tot = torch.zeros_like(x0); mask = None
        for _ in range(max_steps):
            x = x.detach().requires_grad_(True)
            logits = model(x)
            kk = int(logits.argmax(1).item())
            if kk == target:
                break
            m = (logits[0, kk] - logits[0, target])
            g, = torch.autograd.grad(m, x)
            if mask is None:                       # support chosen once, at the query
                idx = torch.topk(g.abs().flatten(), min(k, g.numel())).indices
                mask = torch.zeros_like(g).flatten(); mask[idx] = 1.0; mask = mask.view_as(g)
            gs = g * mask
            gn = float(gs.norm().item()) + 1e-12
            r_tot = r_tot - (float(m.item()) + 1e-4) / gn ** 2 * gs
            x = torch.clamp(x0 + (1 + overshoot) * r_tot, -box, box)
        with torch.no_grad():
            ok = int(model(x).argmax(1).item()) == target
        if ok:
            reached += 1; d = (x - x0).flatten()
            l2.append(float(d.norm().item())); l1.append(float(d.abs().sum().item())); linf.append(float(d.abs().max().item()))
    med = lambda v: float(np.median(v)) if v else float("nan")
    return {"l2_median": med(l2), "l1_median": med(l1), "linf_median": med(linf), "reach": reached / max(1, n), "k": k}


def query_top2_hist(model, X: torch.Tensor, device, num_classes: int) -> Dict[str, List[int]]:
    """Histogram of the top-1 and top-2 (runner-up) classes of the queries."""
    model.eval(); t1 = np.zeros(num_classes, dtype=np.int64); t2 = np.zeros(num_classes, dtype=np.int64)
    with torch.no_grad():
        top = torch.topk(model(X.to(device)), 2, dim=1).indices.cpu().numpy()
    t1 += np.bincount(top[:, 0], minlength=num_classes); t2 += np.bincount(top[:, 1], minlength=num_classes)
    return {"top1": t1.tolist(), "top2": t2.tolist()}


def _manager(F_: int, dataset: str, sparsity_cap: bool):
    mgr = IoTConstraintManager(feature_count=F_, dataset_name=dataset)
    if not sparsity_cap:
        mgr.constraints = [c for c in mgr.constraints if getattr(c, "name", "") != "sparsity"]
    return mgr


# name, sparsity cap, lambda_dist, optimiser, learning rate (None = default 0.05), Adam betas (None = (0.9, 0.999))
VARIANTS = [("cap_only", True, 0.0, "adam", None, None), ("dist_only", False, 0.5, "adam", None, None),
            ("none", False, 0.0, "adam", None, None), ("sgd", True, 0.5, "sgd", None, None),
            ("sgd_lr0.5", True, 0.5, "sgd", 0.5, None),
            ("sgd_lr0.2", True, 0.5, "sgd", 0.2, None),      # effective-step match: 0.05 x (3.74/0.97) ~ 0.19
            ("sgd_lr0.35", True, 0.5, "sgd", 0.35, None),    # match on the targeted gradient: 0.05 x (7.4/1.1) ~ 0.34
            ("signgd", True, 0.5, "signgd", None, None),     # scale-free step without Adam's history
            ("adam", True, 0.5, "adam", None, None),         # the default generator itself (reference row)
            ("adam_b1_0", True, 0.5, "adam", None, (0.0, 0.999)),       # no first-moment memory (RMSprop-like)
            ("adam_b2_0.9", True, 0.5, "adam", None, (0.9, 0.9)),       # short second-moment memory
            ("adam_b1_0_b2_0.9", True, 0.5, "adam", None, (0.0, 0.9))]  # neither memory long
VARIANT_NAMES = [v[0] for v in VARIANTS]


def _variant_generator(model, name: str, info: dict, device, cf_max_iter: int, trace: bool = False):
    """Build the generator of a named VARIANTS entry."""
    F_ = info["num_features"]; ds = info.get("dataset", "generic")
    for vname, cap, ld, opt, lr, betas in VARIANTS:
        if vname == name:
            kw = {}
            if lr is not None:
                kw["learning_rate"] = lr
            if betas is not None:
                kw["betas"] = betas
            return CounterfactualGenerator(model=model, feature_count=F_, num_cfs=C.NUM_CFS, use_dp=False,
                                           max_iter=cf_max_iter, device=device, lambda_dist=ld,
                                           constraint_manager=_manager(F_, ds, cap), optimizer=opt,
                                           trace=trace, **kw)
    raise ValueError(f"unknown generator variant {name}")


def generator_variants(model, queries: torch.Tensor, info: dict, device, cf_max_iter: int,
                       n: int = 100, which=None) -> Dict[str, Dict[str, float]]:
    """Validity of the generator under (cap, lambda_dist) variants:
       default   = sparsity cap 10 + L1 proximity (lambda_dist 0.5)  [what the paper reports]
       cap_only  = cap 10, lambda_dist 0
       dist_only = no cap, lambda_dist 0.5
       none      = no cap, lambda_dist 0   (pure prediction loss + diversity + range box)
       sgd       = default constraints, plain SGD instead of Adam (lr 0.05)
       sgd_lr0.5 = default constraints, plain SGD with lr 0.5
       sgd_lr0.2, sgd_lr0.35 = plain SGD with the step scaled by the sharpness ratio A/C
       signgd    = default constraints, sign-GD (lr 0.05): scale-free steps without Adam's history
       adam_b1_0, adam_b2_0.9, adam_b1_0_b2_0.9 = Adam with beta1 = 0 and/or beta2 = 0.9 (moment ablation)
    `which`: optional list of variant names to run (default: all).
    Each entry reports the mean fraction of valid candidates ('frac') and the share of queries
    with at least one valid candidate ('any')."""
    out = {}
    for name in VARIANT_NAMES:
        if which and name not in which:
            continue
        gen = _variant_generator(model, name, info, device, cf_max_iter)
        fr = []
        for qi in range(min(n, len(queries))):
            try:
                fr.append(gen.generate(queries[qi].to(device), target_class=TARGET).validity)
            except Exception:  # noqa: BLE001
                continue
        out[name] = {"frac": float(np.mean(fr)) if fr else float("nan"),
                     "any": float(np.mean([v > 0 for v in fr])) if fr else float("nan"), "n": len(fr)}
    return out


DIAG_PHASE = 50   # first / last DIAG_PHASE iterations of each run define the early and late phases


def generator_diagnostics(model, queries: torch.Tensor, info: dict, device, cf_max_iter: int,
                          n: int = 100, which=("adam", "adam_b1_0", "adam_b2_0.9", "signgd")) -> Dict[str, Dict[str, float]]:
    """Optimiser diagnostics of the generator: for each variant and query the
    generator runs with trace=True and records, per iteration, the effective step ||x_t - x_{t-1}||
    (after projection), the share of gradient coordinates whose sign flipped since the previous
    iteration, Adam's mean |m_hat|/sqrt(v_hat) (the factor that scales its step), the share of
    candidates sitting in a class other than the query's and the target, and the number of classes
    visited. Per query the early phase is the first DIAG_PHASE iterations and the late phase the last
    DIAG_PHASE iterations of that run (which ends at early stopping or the budget). Returned per
    variant: validity, iterations, the early/late means of every diagnostic over all queries and
    separately over the queries that found a valid counterfactual ('ok') and those that did not
    ('fail'), and the share of failing queries. Hypothesis under test: on the sharp non-private map
    the gradient direction oscillates (high sign-flip rate), m_hat averages toward zero while v_hat
    stays large (small Adam ratio), and the effective step collapses before a valid point is found;
    on the compressed noised map it does not, and sign steps, which keep no moment average, do not stall."""
    out = {}
    metrics = ("step_norm", "sign_flip", "adam_ratio", "share_third_class")
    for name in which:
        gen = _variant_generator(model, name, info, device, cf_max_iter, trace=True)
        rows = []
        for qi in range(min(n, len(queries))):
            try:
                r = gen.generate(queries[qi].to(device), target_class=TARGET)
            except Exception:  # noqa: BLE001
                continue
            t = gen.last_trace
            if not t or t["n_iter"] == 0:
                continue
            row = {"validity": float(r.validity), "n_iter": int(t["n_iter"]),
                   "classes_visited": float(t["classes_visited"])}
            for m_ in metrics:
                v = np.asarray(t[m_], dtype=float)
                v = v[~np.isnan(v)] if len(v) else v
                row[f"{m_}_early"] = float(v[:DIAG_PHASE].mean()) if len(v) else float("nan")
                row[f"{m_}_late"] = float(v[-DIAG_PHASE:].mean()) if len(v) else float("nan")
                row[f"{m_}_all"] = float(v.mean()) if len(v) else float("nan")
            rows.append(row)
        if not rows:
            out[name] = {"n": 0}; continue
        def agg(sel):
            d = {}
            if not sel:
                return d
            for k in rows[0]:
                vals = np.array([r_[k] for r_ in sel], dtype=float); vals = vals[~np.isnan(vals)]
                d[k] = float(vals.mean()) if len(vals) else float("nan")
            return d
        ok = [r_ for r_ in rows if r_["validity"] > 0]; fail = [r_ for r_ in rows if r_["validity"] == 0]
        res = {"n": len(rows), "fail_share": len(fail) / len(rows)}
        for grp, sel in (("all", rows), ("ok", ok), ("fail", fail)):
            for k, v in agg(sel).items():
                res[f"{k}_{grp}"] = v
        out[name] = res
    return out


def weight_noise_probe(model_A, Xte, yte, info, device, cf_max_iter: int, target_sharpness: float,
                       scales=(0.02, 0.05, 0.1, 0.2, 0.3, 0.5), n_queries: int = 50, seed: int = 0,
                       test_loader=None) -> Dict[str, object]:
    """Add N(0, s^2) to every parameter of the trained non-private model (BatchNorm running
    statistics untouched) and measure accuracy, sharpness, margin and validity at each s;
    report the scale whose sharpness is closest to `target_sharpness` (that of the noise-only
    detector of the same run). Tests 'partial randomisation' vs 'regularisation'."""
    rows = []
    g = torch.Generator().manual_seed(seed)
    for s_ in scales:
        m = copy.deepcopy(model_A).to(device).eval()
        with torch.no_grad():
            for p in m.parameters():
                p.add_(torch.randn(p.shape, generator=g).to(device) * s_)
        ev = C.evaluate_model(m, test_loader, device) if test_loader is not None else {"accuracy": float("nan"), "f1_macro": float("nan")}
        q, qy = _attack_queries(m, Xte, yte, n_queries, device)
        if q is None:
            rows.append({"scale": s_, "accuracy": ev["accuracy"], "note": "no queries"}); continue
        geo = _geometry(m, q, device)
        val = C.eval_cf(m, q, info, device, max_samples=n_queries, max_iter=cf_max_iter)["validity"]
        rows.append({"scale": s_, "accuracy": ev["accuracy"], "f1_macro": ev["f1_macro"], "cf_validity": val,
                     "margin_grad_norm": geo["margin_grad_norm"], "margin_logit": geo["margin_logit"],
                     "d_lin": geo["d_lin"], "d_lin_target_median": geo["d_lin_target_median"]})
    ok = [r for r in rows if "margin_grad_norm" in r]
    best = min(ok, key=lambda r: abs(math.log(max(r["margin_grad_norm"], 1e-9)) - math.log(max(target_sharpness, 1e-9)))) if ok else None
    return {"rows": rows, "target_sharpness": target_sharpness, "matched": best}


# =============================================================================
# gradient-free counterfactual search
# =============================================================================
def discrete_search_validity(model, queries: torch.Tensor, Xtr: torch.Tensor, device,
                             n_candidates: int = 4000, max_changes: int = 10, box: float = 5.0,
                             seed: int = 0) -> Dict[str, float]:
    """Random-restart local search: change <= max_changes features to values drawn from the
    training marginals (empirical, so every proposed value has been observed); keep the closest
    valid candidate. Validity = share of queries with at least one valid candidate."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    F_ = Xtr.shape[1]; Xtr_cpu = Xtr.cpu(); valid = []; prox = []; spars = []
    model.eval()
    for q in queries:
        q = q.cpu()
        ks = torch.randint(1, max_changes + 1, (n_candidates,), generator=g)
        cand = q.unsqueeze(0).repeat(n_candidates, 1)
        rows = torch.randint(0, len(Xtr_cpu), (n_candidates,), generator=g)
        for c in range(n_candidates):
            feats = torch.randperm(F_, generator=g)[: int(ks[c])]
            cand[c, feats] = Xtr_cpu[rows[c], feats]
        cand = torch.clamp(cand, -box, box)
        with torch.no_grad():
            preds = torch.cat([model(cand[i:i + 1024].to(device)).argmax(1).cpu()
                               for i in range(0, n_candidates, 1024)])
        ok = (preds == TARGET).nonzero().flatten()
        if len(ok) == 0:
            valid.append(0.0); continue
        d = (cand[ok] - q).abs().mean(1); j = int(ok[int(d.argmin())])
        valid.append(1.0); prox.append(float((cand[j] - q).abs().mean()))
        spars.append(int(((cand[j] - q).abs() > 0.01).sum()))
    return {"validity": float(np.mean(valid)), "proximity": float(np.mean(prox)) if prox else 0.0,
            "sparsity": float(np.mean(spars)) if spars else 0.0, "n": len(valid)}


# =============================================================================
# matched-norm random perturbation control
# =============================================================================
def random_control(model, query: torch.Tensor, cf: torch.Tensor, device, K: int = 20,
                   box: float = 5.0, gen: Optional[torch.Generator] = None) -> float:
    """K random perturbations with the same L1 norm and the same number of changed features as cf."""
    delta = (cf - query); changed = (delta.abs() > 0.01); k = int(changed.sum().item())
    l1 = float(delta.abs().sum().item())
    if k == 0 or l1 == 0:
        return 0.0
    F_ = query.numel(); hits = 0
    for _ in range(K):
        feats = torch.randperm(F_, generator=gen)[:k]
        w = torch.distributions.Dirichlet(torch.ones(k)).sample()
        signs = torch.randint(0, 2, (k,), generator=gen).float() * 2 - 1
        pert = torch.zeros(F_); pert[feats] = signs * w * l1
        x = torch.clamp(query + pert, -box, box)
        with torch.no_grad():
            hits += int(model(x.unsqueeze(0).to(device)).argmax(1).item() == TARGET)
    return hits / K


# =============================================================================
CORE_PARTS = ["hist", "random", "curves", "discrete", "deepfool"]
# hist2 = top-2 histogram, variants = (cap, lambda) variants, sparsedf = sparse DeepFool,
# weightnoise = weight-noise probe, diag = optimiser diagnostics
EXTRA_PARTS = ["hist2", "variants", "sparsedf", "weightnoise", "diag"]
ALL_PARTS = CORE_PARTS + EXTRA_PARTS + ["smooth", "transfer"]
SMOOTH_ARMS = ["E_LabelSmooth", "F_InputNoise", "G_WeightDecay", "H_SAM"]


def run(dataset: str, n_runs: int = 3, max_samples=None, epochs: int = C.EPOCHS,
        cf_samples: int = 100, cf_max_iter: int = C.CF_MAX_ITER, parts: str = "core",
        batch_size=None, quick: bool = False, sigma: Optional[float] = None,
        random_k: int = 20, curve_samples: int = 50, curve_max_iter: Optional[int] = None,
        out_suffix: str = "", variant_samples: int = 100, seed0: int = C.RANDOM_SEED,
        smooth_arms: str = "", arch: str = "bn", variants: str = "",
        diag_variants: str = "adam,adam_b1_0,adam_b2_0.9,signgd", diag_samples: int = 100,
        diag_conds: str = "A_Standard,C_Noise_Only"):
    log = C.Runlog("exp19_controls" + out_suffix, dataset)
    curve_max_iter = int(curve_max_iter) if curve_max_iter else int(cf_max_iter)
    device = C.get_device(); C.set_batch_size(batch_size)

    def _mlp(info_):
        """The paper's MLP; with arch='nonorm' every BatchNorm1d is replaced by Identity, i.e. the
        normalisation-free network of the certified arm (exp20 --arch nonorm), so the controls can
        be run on that architecture too; arch='cnn' gives the 1D-CNN of the architecture check."""
        if arch == "cnn":
            return C.make_model("cnn1d", info_)
        m_ = C.make_model("mlp", info_)
        if arch == "nonorm":
            for _n, mod in list(m_.named_modules()):
                for cname, child in list(mod.named_children()):
                    if isinstance(child, nn.BatchNorm1d):
                        setattr(mod, cname, nn.Identity())
        return m_
    if quick:
        max_samples = max_samples or 40000; epochs = 3; n_runs = 2; cf_samples = 20; curve_samples = 10
    parts_l = (ALL_PARTS if parts == "all" else CORE_PARTS if parts == "core"
               else [p.strip() for p in parts.split(",")])
    noise_std = float(sigma) if sigma is not None else C.sigma_from_epsilon(C.REFERENCE_EPSILON) * C.DP_CLIP_NORM
    which_variants = [v.strip() for v in variants.split(",") if v.strip()] or None
    diag_which = [v.strip() for v in diag_variants.split(",") if v.strip()]
    diag_conds_l = [c.strip() for c in diag_conds.split(",") if c.strip()]
    log.log(f"device={device} arch={arch} n_runs={n_runs} seed0={seed0} epochs={epochs} cf_samples={cf_samples} "
            f"cf_max_iter={cf_max_iter} curve_max_iter={curve_max_iter} batch={C.BATCH_SIZE} sigma={noise_std:.4f} parts={parts_l}"
            + (f" diag_variants={diag_which} diag_samples={diag_samples} diag_conds={diag_conds_l}" if "diag" in parts_l else ""))

    Xtr, Xte, ytr, yte, info = C.load_data(dataset, max_samples)
    K_cls = info["num_classes"]
    train_loader, test_loader = C.make_loaders(Xtr, ytr, Xte, yte)
    cw = C.class_weights_tensor(ytr, K_cls, device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    log.log(f"train={len(Xtr)} test={len(Xte)} F={info['num_features']} C={K_cls} "
            f"true-label class share (test): {np.round(np.bincount(yte.numpy(), minlength=K_cls)/len(yte), 3).tolist()}")

    out: Dict = {"config": {"parts": parts_l, "n_runs": n_runs, "seed0": seed0, "epochs": epochs, "cf_samples": cf_samples,
                            "cf_max_iter": cf_max_iter, "curve_max_iter": curve_max_iter, "variant_samples": variant_samples,
                            "batch": C.BATCH_SIZE, "sigma": noise_std, "device": device,
                            "query_rule": "instances predicted non-target by the evaluated detector "
                                          "(true label recorded); target class = 0 (benign)",
                            "random_control_K": random_k, "checkpoints": CHECKPOINTS,
                            "diag_variants": diag_which if "diag" in parts_l else None,
                            "diag_samples": diag_samples if "diag" in parts_l else None,
                            "diag_conds": diag_conds_l if "diag" in parts_l else None, "diag_phase": DIAG_PHASE}}

    # ---------------------------------------------------------------- models --
    def train_cond(cond, seed):
        clip, noise = C.CONDITIONS[cond]
        C.set_seed(seed); m = _mlp(info).to(device)
        C.train_condition(m, train_loader, use_clipping=clip, use_noise=noise,
                          noise_std=(noise_std if noise else 0.0), epochs=epochs,
                          lr=C.LEARNING_RATE, criterion=criterion, device=device)
        return m

    need_models = any(p in parts_l for p in ["hist", "random", "curves", "deepfool", "discrete", "transfer"] + EXTRA_PARTS)
    if "smooth" in parts_l and set(parts_l) <= {"smooth", "diag"}:
        need_models = False          # smooth,diag: diagnostics on the smoothing detectors only
    conds = (COND4 if any(p in parts_l for p in ["hist", "random", "deepfool"] + [p for p in EXTRA_PARTS if p != "diag"])
             else ["A_Standard", "C_Noise_Only"])
    if "diag" in parts_l:      # a diag-only run trains just the conditions it diagnoses
        conds = [c for c in COND4 if c in set(conds) | set(diag_conds_l)] if parts_l != ["diag"] else \
                [c for c in COND4 if c in diag_conds_l]

    per_cond: Dict[str, Dict[str, List]] = {c: {} for c in COND4 + ["E_LabelSmooth", "F_InputNoise", "G_WeightDecay", "H_SAM"]}
    def push(c, k, v): per_cond[c].setdefault(k, []).append(v)

    curves: Dict[str, List[Dict[int, float]]] = {}
    transfer_rows: List[Dict] = []

    if need_models:
        for run_i in range(n_runs):
            seed = seed0 + run_i
            models = {c: train_cond(c, seed) for c in conds}
            log.log(f"\n=== run {run_i+1}/{n_runs} (seed {seed}) ===")
            run_queries: Dict[str, torch.Tensor] = {}
            for c in conds:
                m = models[c]
                ev = C.evaluate_model(m, test_loader, device)
                hist_counts, share_target = _pred_hist(m, Xte, device, K_cls)
                queries, qy = _attack_queries(m, Xte, yte, cf_samples, device)
                if queries is None:
                    log.log(f"  {c}: no queries"); continue
                run_queries[c] = queries
                if "hist2" in parts_l:
                    h2 = query_top2_hist(m, queries, device, K_cls)
                    push(c, "query_top1_hist", h2["top1"]); push(c, "query_top2_hist", h2["top2"])
                at_target_it0 = float((_predict(m, queries, device) == TARGET).mean())   # must be 0
                true_attack = (qy != TARGET).numpy()
                geo = _geometry(m, queries, device)
                push(c, "accuracy", ev["accuracy"]); push(c, "f1_macro", ev["f1_macro"])
                push(c, "pred_hist", hist_counts.tolist()); push(c, "share_pred_target", share_target)
                push(c, "queries_at_target_it0", at_target_it0); push(c, "queries_true_attack_share", float(true_attack.mean()))
                for k_, v_ in geo.items(): push(c, k_, v_)
                push(c, "ece", _ece(m, test_loader, device))
                line = (f"  {c}: acc={ev['accuracy']:.3f} F1={ev['f1_macro']:.3f} predBenign={share_target:.3f} "
                        f"q@target(it0)={at_target_it0:.2f} q_trueAttack={true_attack.mean():.2f} d_lin={geo['d_lin']:.3f} "
                        f"d_lin_target(med)={geo['d_lin_target_median']:.3f}")

                # ---- generic CF generation (validity + per-query CFs) ----
                if any(p in parts_l for p in ["hist", "random", "transfer"]):
                    gen = CounterfactualGenerator(model=m, feature_count=info["num_features"], num_cfs=C.NUM_CFS,
                                                  use_dp=False, max_iter=cf_max_iter, device=device)
                    vals, vals_true, rnd, cf_store, prox_l, spars_l = [], [], [], [], [], []
                    g_cpu = torch.Generator().manual_seed(seed)
                    for qi in range(len(queries)):
                        try:
                            r = gen.generate(queries[qi].to(device), target_class=TARGET)
                        except Exception as ex:  # noqa: BLE001
                            log.log(f"   gen error {ex}"); continue
                        vals.append(r.validity); prox_l.append(float(r.proximity)); spars_l.append(float(r.sparsity))
                        if true_attack[qi]:
                            vals_true.append(r.validity)
                        cfs = [cc.detach().cpu() for cc in r.counterfactuals]
                        # best CF = the valid one closest to the query (else the first)
                        best = None
                        for cc in cfs:
                            if int(_predict(m, cc.unsqueeze(0), device)[0]) == TARGET:
                                if best is None or (cc - queries[qi]).abs().sum() < (best - queries[qi]).abs().sum():
                                    best = cc
                        if best is None:
                            best = cfs[0]
                        cf_store.append((queries[qi].clone(), best.clone(), r.validity))
                        if "random" in parts_l:
                            rnd.append(random_control(m, queries[qi].cpu(), best, device, K=random_k, gen=g_cpu))
                    push(c, "cf_validity", float(np.mean(vals)) if vals else 0.0)
                    push(c, "cf_validity_any", float(np.mean([v > 0 for v in vals])) if vals else 0.0)   # >=1 of 5 valid
                    push(c, "cf_validity_per_query", [float(v) for v in vals])
                    push(c, "cf_proximity", float(np.mean(prox_l)) if prox_l else float("nan"))   # mean |dx| per standardised feature
                    push(c, "cf_sparsity", float(np.mean(spars_l)) if spars_l else float("nan"))  # features changed by > 0.01
                    push(c, "cf_validity_true_attack_subset", float(np.mean(vals_true)) if vals_true else float("nan"))
                    line += (f" | val={np.mean(vals):.3f} (any {np.mean([v > 0 for v in vals]):.3f}; "
                             f"true-attack subset {np.mean(vals_true) if vals_true else float('nan'):.3f})")
                    if "random" in parts_l:
                        push(c, "random_matched_flip_rate", float(np.mean(rnd)) if rnd else 0.0)
                        line += f" randomFlip={np.mean(rnd):.3f}"
                    if "transfer" in parts_l:
                        per_cond[c].setdefault("_cf_store", []).append(cf_store)

                # ---- validity-iteration curves (early stopping off) ----
                if "curves" in parts_l and c in ("A_Standard", "C_Noise_Only", "B_DP_SGD", "D_Clip_Only"):
                    cps = [t for t in CHECKPOINTS if t <= curve_max_iter] or [curve_max_iter]
                    tg = TracingGenerator(model=m, feature_count=info["num_features"], num_cfs=C.NUM_CFS,
                                          use_dp=False, max_iter=max(cps), device=device)
                    hs = []
                    for qi in range(min(curve_samples, len(queries))):
                        torch.manual_seed(seed * 1000 + qi)
                        hs.append(tg.generate_traced(queries[qi], TARGET, checkpoints=cps))
                    curve = {t: float(np.mean([h[t] for h in hs])) for t in cps}
                    curves.setdefault(c, []).append(curve)
                    show = [t for t in CHECKPOINTS if t in curve][:: max(1, len(CHECKPOINTS) // 3)]
                    line += " | curve " + " ".join(f"{t}:{curve[t]:.2f}" for t in show)

                # ---- DeepFool distance ----
                if "deepfool" in parts_l:
                    dd, reach = deepfool_target_distance(m, queries, device)
                    push(c, "deepfool_l2_median", float(np.median(dd)) if dd else float("nan"))
                    push(c, "deepfool_reach_rate", reach)
                    line += f" | DF L2 med={np.median(dd) if dd else float('nan'):.3f} reach={reach:.2f}"

                # ---- discrete (gradient-free) generator ----
                if "discrete" in parts_l:
                    ds = discrete_search_validity(m, queries[:min(50, len(queries))], Xtr, device, seed=seed)
                    push(c, "discrete_validity", ds["validity"]); push(c, "discrete_proximity", ds["proximity"])
                    line += f" | discrete val={ds['validity']:.3f}"

                # ---- sparse (k=10) targeted DeepFool ----
                if "sparsedf" in parts_l:
                    sd = sparse_deepfool_target_distance(m, queries, device)
                    for k_ in ("l2_median", "l1_median", "linf_median", "reach"):
                        push(c, f"sparsedf_{k_}", sd[k_])
                    line += f" | sparseDF L2={sd['l2_median']:.3f} L1={sd['l1_median']:.3f} reach={sd['reach']:.2f}"

                # ---- generator variants (cap / proximity ablation) ----
                if "variants" in parts_l:
                    gv = generator_variants(m, queries, info, device, cf_max_iter, n=variant_samples, which=which_variants)
                    for name, r_ in gv.items():
                        push(c, f"variant_{name}_frac", r_["frac"]); push(c, f"variant_{name}_any", r_["any"])
                    line += " | variants " + " ".join(f"{k_}={v_['frac']:.2f}/{v_['any']:.2f}" for k_, v_ in gv.items())

                # ---- optimiser diagnostics (Adam moment ablation, step norms, sign flips) ----
                if "diag" in parts_l and c in diag_conds_l:
                    gd = generator_diagnostics(m, queries, info, device, cf_max_iter, n=diag_samples, which=diag_which)
                    for name, r_ in gd.items():
                        for k_, v_ in r_.items():
                            push(c, f"diag_{name}_{k_}", v_)
                    line += "\n     diag " + " | ".join(
                        f"{k_}: val={v_.get('validity_all', float('nan')):.3f} it={v_.get('n_iter_all', float('nan')):.0f} "
                        f"fail={v_.get('fail_share', float('nan')):.2f} "
                        f"step e/l={v_.get('step_norm_early_all', float('nan')):.3f}/{v_.get('step_norm_late_all', float('nan')):.3f} "
                        f"flip e/l={v_.get('sign_flip_early_all', float('nan')):.2f}/{v_.get('sign_flip_late_all', float('nan')):.2f} "
                        f"adamratio e/l={v_.get('adam_ratio_early_all', float('nan')):.2f}/{v_.get('adam_ratio_late_all', float('nan')):.2f} "
                        f"third={v_.get('share_third_class_all_all', float('nan')):.2f} "
                        f"[fail: step_l={v_.get('step_norm_late_fail', float('nan')):.3f} flip_l={v_.get('sign_flip_late_fail', float('nan')):.2f} "
                        f"adamratio_l={v_.get('adam_ratio_late_fail', float('nan')):.2f}]"
                        for k_, v_ in gd.items() if v_.get("n", 0))
                log.log(line)

            # ---- weight-noise probe on the non-private model of this run ----
            if "weightnoise" in parts_l and "A_Standard" in models and "C_Noise_Only" in models:
                tgt = per_cond["C_Noise_Only"]["margin_grad_norm"][-1]
                wn = weight_noise_probe(models["A_Standard"], Xte, yte, info, device, cf_max_iter, tgt,
                                        n_queries=min(50, cf_samples), seed=seed, test_loader=test_loader)
                out.setdefault("weight_noise", []).append({"run": run_i, **wn})
                mt = wn["matched"]
                log.log("  weight-noise probe (A + N(0,s^2) on parameters): " + "; ".join(
                    f"s={r['scale']}: acc={r.get('accuracy', float('nan')):.3f} sharp={r.get('margin_grad_norm', float('nan')):.2f} "
                    f"val={r.get('cf_validity', float('nan')):.3f}" for r in wn["rows"]) +
                    (f" | matched to C sharpness {tgt:.2f}: s={mt['scale']} val={mt['cf_validity']:.3f} acc={mt['accuracy']:.3f}" if mt else ""))

            # ---- evasion transfer ----
            if "transfer" in parts_l:
                C.set_seed(seed + 100); m_alt = _mlp(info).to(device)
                C.train_condition(m_alt, train_loader, use_clipping=False, use_noise=False, noise_std=0.0,
                                  epochs=epochs, lr=C.LEARNING_RATE, criterion=criterion, device=device)
                C.set_seed(seed); m_cnn = C.make_model("cnn1d", info).to(device)
                C.train_condition(m_cnn, train_loader, use_clipping=False, use_noise=False, noise_std=0.0,
                                  epochs=epochs, lr=C.LEARNING_RATE, criterion=criterion, device=device)
                targets = {"A_other_seed": m_alt, "CNN1D_A": m_cnn}
                for c in conds:
                    targets_c = dict(targets); targets_c.update({k: models[k] for k in models if k != c})
                    store = per_cond[c].get("_cf_store", [])
                    if not store:
                        continue
                    store = store[-1]
                    valid_pairs = [(q, cf) for (q, cf, v) in store if v > 0]
                    if not valid_pairs:
                        continue
                    Q = torch.stack([q for q, _ in valid_pairs]); CFS = torch.stack([cf for _, cf in valid_pairs])
                    row = {"run": run_i, "source": c, "n_valid_cfs": len(valid_pairs)}
                    for tname, tm in targets_c.items():
                        q_flagged = (_predict(tm, Q, device) != TARGET)      # target detector flags the original
                        if q_flagged.sum() == 0:
                            row[tname] = None; continue
                        evaded = (_predict(tm, CFS, device) == TARGET) & q_flagged
                        row[tname] = float(evaded.sum() / q_flagged.sum())
                    transfer_rows.append(row)
                    log.log(f"  transfer from {c}: " + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}=NA"
                                                                  for k, v in row.items() if k not in ("run", "source", "n_valid_cfs")))
            del models
            if device == "cuda":
                torch.cuda.empty_cache()

    # ---------------------------------------------------------------- smooth --
    if "smooth" in parts_l:
        arms = [a.strip() for a in smooth_arms.split(",") if a.strip()] if smooth_arms else list(SMOOTH_ARMS)
        out["config"]["smooth_arms"] = arms
        log.log(f"\n=== smoothing arms: {arms} | E: per-sample class-weighted label smoothing 0.1, "
                f"F: input noise std 0.3, G: AdamW wd 1e-2, H: SAM rho 0.05 ===")
        for run_i in range(n_runs):
            seed = seed0 + run_i
            for var in arms:
                C.set_seed(seed); m = _mlp(info).to(device)
                train_variant(m, train_loader, variant=var, epochs=epochs, lr=C.LEARNING_RATE,
                              criterion_plain=criterion, class_weights=cw, device=device)
                ev = C.evaluate_model(m, test_loader, device)
                queries, qy = _attack_queries(m, Xte, yte, cf_samples, device)
                if queries is None:
                    continue
                geo = _geometry(m, queries, device)
                cf = C.eval_cf(m, queries, info, device, max_samples=cf_samples, max_iter=cf_max_iter)
                push(var, "accuracy", ev["accuracy"]); push(var, "f1_macro", ev["f1_macro"])
                push(var, "cf_validity", cf["validity"]); push(var, "ece", _ece(m, test_loader, device))
                for k_, v_ in geo.items(): push(var, k_, v_)
                line = (f"  run {run_i+1} {var}: acc={ev['accuracy']:.3f} F1={ev['f1_macro']:.3f} "
                        f"val={cf['validity']:.3f} |dm/dx|={geo['margin_grad_norm']:.3f} m={geo['margin_logit']:.3f} "
                        f"d_lin={geo['d_lin']:.3f} ECE={per_cond[var]['ece'][-1]:.3f}")
                # optimiser diagnostics on the smoothing detectors (--parts smooth,diag): does the
                # Adam step factor fall along the path as on the non-private detector or stay up as
                # on the noised one?
                if "diag" in parts_l:
                    gd = generator_diagnostics(m, queries, info, device, cf_max_iter, n=diag_samples, which=diag_which)
                    for name, r_ in gd.items():
                        for k_, v_ in r_.items():
                            push(var, f"diag_{name}_{k_}", v_)
                    line += " | diag " + " | ".join(
                        f"{k_}: val={v_.get('validity_all', float('nan')):.3f} it={v_.get('n_iter_all', float('nan')):.0f} "
                        f"step e/l={v_.get('step_norm_early_all', float('nan')):.3f}/{v_.get('step_norm_late_all', float('nan')):.3f} "
                        f"adamratio e/l={v_.get('adam_ratio_early_all', float('nan')):.2f}/{v_.get('adam_ratio_late_all', float('nan')):.2f}"
                        for k_, v_ in gd.items() if v_.get("n", 0))
                log.log(line)
                del m

    # ---------------------------------------------------------------- summary -
    summary = {"config": out["config"], "conditions": {}, "curves": {}, "transfer": transfer_rows,
               "weight_noise": out.get("weight_noise", [])}
    for c, d in per_cond.items():
        if not d:
            continue
        summary["conditions"][c] = {}
        for k, v in d.items():
            if k.startswith("_"):
                continue
            if k in ("pred_hist", "query_top1_hist", "query_top2_hist"):
                arr = np.array(v, dtype=float)
                summary["conditions"][c][k] = {"mean_share": (arr.mean(0) / max(arr.sum(1).mean(), 1e-12)).round(4).tolist(),
                                               "counts_per_run": [list(map(int, r)) for r in v]}
            elif k == "cf_validity_per_query":
                summary["conditions"][c][k] = {"raw": v}
            else:
                mu, sd = _ms([x for x in v if x == x]); summary["conditions"][c][k] = {"mean": mu, "std": sd, "raw": v}
    for c, lst in curves.items():
        ts = sorted(lst[0].keys())
        summary["curves"][c] = {str(t): _ms([h[t] for h in lst]) for t in ts}
    # distance vs validity pooled over every (condition, run) of this experiment, for the
    # nearest-boundary d_lin (exp14 definition) and the targeted median distance
    from scipy import stats
    for key, name in [("d_lin", "dlin_vs_validity"), ("d_lin_target_median", "dlin_target_vs_validity"),
                      ("deepfool_l2_median", "deepfool_vs_validity")]:
        pts = []
        for c, d in per_cond.items():
            if "cf_validity" in d and key in d:
                for v, dl in zip(d["cf_validity"], d[key]):
                    if dl == dl: pts.append((c, dl, v))
        if len(pts) >= 4:
            dl = np.array([p[1] for p in pts]); vv = np.array([p[2] for p in pts])
            summary[name] = {"n": len(pts), "spearman_rho": float(stats.spearmanr(dl, vv).statistic),
                             "spearman_p": float(stats.spearmanr(dl, vv).pvalue),
                             "pearson_r": float(stats.pearsonr(dl, vv).statistic), "points": pts}
            log.log(f"\n{key} vs validity over {len(pts)} (condition,run) points: Spearman rho="
                    f"{summary[name]['spearman_rho']:.3f} (p={summary[name]['spearman_p']:.4f})")
    log.log("\n=== SUMMARY (means) ===")
    for c, d in summary["conditions"].items():
        keys = ["accuracy", "f1_macro", "cf_validity", "cf_validity_any", "random_matched_flip_rate", "share_pred_target",
                "d_lin", "d_lin_target_median", "deepfool_l2_median", "sparsedf_l2_median", "sparsedf_l1_median",
                "cf_proximity", "cf_sparsity",
                "discrete_validity", "variant_cap_only_frac", "variant_dist_only_frac", "variant_none_frac",
                "variant_sgd_frac", "variant_sgd_lr0.5_frac", "variant_sgd_lr0.2_frac", "variant_sgd_lr0.35_frac",
                "variant_signgd_frac", "variant_adam_frac", "variant_adam_b1_0_frac", "variant_adam_b2_0.9_frac",
                "variant_adam_b1_0_b2_0.9_frac",
                "diag_adam_validity_all", "diag_adam_fail_share", "diag_adam_step_norm_late_fail",
                "diag_adam_sign_flip_late_fail", "diag_adam_adam_ratio_late_fail",
                "diag_signgd_validity_all", "diag_signgd_fail_share", "diag_signgd_step_norm_late_fail",
                "diag_adam_b1_0_validity_all", "diag_adam_b2_0.9_validity_all"]
        log.log(f"  {c:15s} " + " ".join(f"{k}={d[k]['mean']:.3f}" for k in keys if k in d))
    log.save_json(summary)
    return summary


def main():
    ap = argparse.ArgumentParser(description="EXP19 controls")
    ap.add_argument("--dataset", default="ciciot2023", choices=["ciciot2023", "edge-iiot", "toniot"])
    ap.add_argument("--n-runs", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--cf-samples", type=int, default=100)
    ap.add_argument("--cf-max-iter", type=int, default=C.CF_MAX_ITER)
    ap.add_argument("--curve-samples", type=int, default=50)
    ap.add_argument("--random-k", type=int, default=20)
    ap.add_argument("--parts", default="core",
                    help="core (=hist,random,curves,discrete,deepfool; default), all, or a comma list incl. "
                         "hist2,variants,sparsedf,weightnoise,smooth,transfer")
    ap.add_argument("--sigma", type=float, default=None, help="noise std (default: 2.4224, the sigma used throughout)")
    ap.add_argument("--curve-max-iter", type=int, default=None, help="iterations for the validity curves (default = --cf-max-iter)")
    ap.add_argument("--variant-samples", type=int, default=100, help="queries per generator variant")
    ap.add_argument("--out-suffix", default="", help="suffix for the output folder, e.g. _k5 -> exp19_controls_k5_<dataset>")
    ap.add_argument("--seed0", type=int, default=C.RANDOM_SEED, help="first seed (runs use seed0..seed0+n_runs-1)")
    ap.add_argument("--smooth-arms", default="", help="comma list of smoothing arms for --parts smooth "
                                                        "(default: E_LabelSmooth,F_InputNoise,G_WeightDecay,H_SAM)")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--arch", default="bn", choices=["bn", "nonorm", "cnn"],
                    help="bn = the paper's MLP (default); nonorm = BatchNorm1d removed, the network of the certified arm; cnn = the 1D-CNN")
    ap.add_argument("--variants", default="", help="comma list of generator variants for --parts variants "
                    "(default all: cap_only,dist_only,none,sgd,sgd_lr0.5,sgd_lr0.2,sgd_lr0.35,signgd,adam,"
                    "adam_b1_0,adam_b2_0.9,adam_b1_0_b2_0.9)")
    ap.add_argument("--diag-variants", default="adam,adam_b1_0,adam_b2_0.9,signgd",
                    help="generator variants traced by --parts diag (optimiser diagnostics)")
    ap.add_argument("--diag-samples", type=int, default=100, help="queries per variant for --parts diag")
    ap.add_argument("--diag-conds", default="A_Standard,C_Noise_Only",
                    help="conditions diagnosed by --parts diag (comma list of A_Standard,D_Clip_Only,C_Noise_Only,B_DP_SGD)")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    run(a.dataset, n_runs=a.n_runs, max_samples=a.max_samples, epochs=a.epochs, cf_samples=a.cf_samples,
        cf_max_iter=a.cf_max_iter, parts=a.parts, batch_size=a.batch_size, quick=a.quick, sigma=a.sigma,
        random_k=a.random_k, curve_samples=a.curve_samples, curve_max_iter=a.curve_max_iter,
        out_suffix=a.out_suffix, variant_samples=a.variant_samples, seed0=a.seed0,
        smooth_arms=a.smooth_arms, arch=a.arch, variants=a.variants,
        diag_variants=a.diag_variants, diag_samples=a.diag_samples, diag_conds=a.diag_conds)


if __name__ == "__main__":
    main()
