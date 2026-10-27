"""Smoothing-arm table (exp19 --parts smooth) against the paired non-private / noise-only detectors.

Rows: A (non-private), label smoothing, input noise, weight decay, SAM, C (noise only).
Smoothing arms come from experiments_journal/exp19_controls_smooth3v2_ciciot2023/results.json,
A and C from the three-seed controls run of the same seeds (exp19_controls_ciciot2023).
Usage: python3 journal_paper/make_table_smooth.py [experiments_journal]
"""
import json
import os
import sys

import numpy as np
from scipy import stats

R = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("RESULTS_DIR", "experiments_journal")
SM = json.load(open(f"{R}/exp19_controls_smooth3v2_ciciot2023/results.json", encoding="utf-8"))
CO = json.load(open(f"{R}/exp19_controls_ciciot2023/results.json", encoding="utf-8"))

ROWS = [("A_Standard", CO, "A (non-private)"),
        ("E_LabelSmooth", SM, "Label smoothing ($\\varepsilon=0.1$)"),
        ("F_InputNoise", SM, "Input noise (std $0.3$)"),
        ("G_WeightDecay", SM, "Weight decay ($10^{-2}$)"),
        ("H_SAM", SM, "SAM ($\\rho=0.05$)"),
        ("C_Noise_Only", CO, "C (noise only)")]
COLS = [("accuracy", 3, False), ("f1_macro", 3, False), ("cf_validity", 3, True),
        ("margin_grad_norm", 2, True), ("hessian_trace", 2, False), ("margin_logit", 2, True),
        ("d_lin", 2, True), ("ece", 3, False)]          # (key, decimals, show std)

assert SM["config"]["seed0"] == 42 and SM["config"]["n_runs"] == 3, SM["config"]
assert CO["config"]["n_runs"] == 3


def raw(src, cond, key):
    return np.array(src["conditions"][cond][key]["raw"], dtype=float)


def cell(src, cond, key, nd, sd=True):
    v = src["conditions"][cond][key]
    return f"${v['mean']:.{nd}f}\\pm{v['std']:.{nd}f}$" if sd else f"${v['mean']:.{nd}f}$"


print("% ---- table rows (mean +- population std over seeds 42-44) ----")
for cond, src, label in ROWS:
    print(label + " & " + " & ".join(cell(src, cond, k, nd, sd) for k, nd, sd in COLS) + r" \\")

print("\n% ---- paired by seed, against A ----")
a_val, a_sh, a_m, a_d = (raw(CO, "A_Standard", k) for k in ("cf_validity", "margin_grad_norm", "margin_logit", "d_lin"))
for cond, src, label in ROWS[1:]:
    v, sh, m, d = (raw(src, cond, k) for k in ("cf_validity", "margin_grad_norm", "margin_logit", "d_lin"))
    dv = 100 * (v - a_val)
    print(f"% {cond:14s} dval(pp) per seed={np.round(dv, 1).tolist()} mean={dv.mean():+.1f} "
          f"higher in {int((dv > 0).sum())}/3 | sharp {a_sh.mean():.2f}->{sh.mean():.2f} "
          f"({100 * (sh.mean() / a_sh.mean() - 1):+.0f}%) | margin {a_m.mean():.2f}->{m.mean():.2f} "
          f"| d_lin {a_d.mean():.2f}->{d.mean():.2f} (lower in {int((d < a_d).sum())}/3)")

print("\n% ---- validity vs distance / sharpness over detectors ----")
pts = []
for cond, src, _ in ROWS:
    for v, d, sh, dt in zip(raw(src, cond, "cf_validity"), raw(src, cond, "d_lin"),
                            raw(src, cond, "margin_grad_norm"), raw(src, cond, "d_lin_target_median")):
        pts.append((cond, v, d, sh, dt))
for name, sel in [("smoothing arms only (12)", lambda c: c not in ("A_Standard", "C_Noise_Only")),
                  ("A + C + smoothing arms (18)", lambda c: True)]:
    P = [p for p in pts if sel(p[0])]
    v = np.array([p[1] for p in P]); d = np.array([p[2] for p in P])
    sh = np.array([p[3] for p in P]); dt = np.array([p[4] for p in P])
    r1 = stats.spearmanr(d, v); r2 = stats.spearmanr(sh, v); r3 = stats.spearmanr(dt, v)
    print(f"% {name}: rho(d_lin, val)={r1.statistic:.2f} p={r1.pvalue:.3g} | "
          f"rho(sharpness, val)={r2.statistic:.2f} p={r2.pvalue:.3g} | "
          f"rho(d_lin_target, val)={r3.statistic:.2f} p={r3.pvalue:.3g}")
