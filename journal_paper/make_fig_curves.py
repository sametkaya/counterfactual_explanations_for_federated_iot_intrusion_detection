"""Fig. curves: best-so-far validity vs iteration with early stopping disabled (exp19 controls).

Left:   CICIoT2023, mechanism run (seeds 42-44, the detectors of the three-seed controls),
        four conditions, checkpoints to 1,000 iterations.
Middle: Edge-IIoTset at batch 256, three-seed controls run (seeds 42-44), to 300 iterations.
Right:  non-private (A, dashed) and noise-only (C, solid) detectors of the normalisation-free
        network, the CNN and the batch-64 Edge-IIoTset detector, to 1,000 iterations
        (exp19 _nncurves, _cnncurves, _b64curves; seeds 42-44).
Usage: python3 journal_paper/make_fig_curves.py [experiments_journal]
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("RESULTS_DIR", "experiments_journal")
PANELS = [
    ("CICIoT2023 (3 seeds)", f"{R}/exp19_controls_mech_ciciot2023/results.json"),
    ("Edge-IIoTset, batch 256 (3 seeds)", f"{R}/exp19_controls_k3_edge-iiot/results.json"),
]
HARD = [  # right panel: A (dashed) vs C (solid) on the harder baselines
    ("no-norm", f"{R}/exp19_controls_nncurves_ciciot2023/results.json", "#009E73"),
    ("CNN", f"{R}/exp19_controls_cnncurves_ciciot2023/results.json", "#CC79A7"),
    ("Edge b64", f"{R}/exp19_controls_b64curves_edge-iiot/results.json", "#0072B2"),
]
COND = ["A_Standard", "D_Clip_Only", "C_Noise_Only", "B_DP_SGD"]
COLOR = {"A_Standard": "#666666", "D_Clip_Only": "#56B4E9", "C_Noise_Only": "#E69F00", "B_DP_SGD": "#D55E00"}
STYLE = {"A_Standard": ("-", "o"), "D_Clip_Only": ("--", "s"), "C_Noise_Only": ("-", "^"), "B_DP_SGD": ("-.", "D")}
LABEL = {"A_Standard": "A standard", "D_Clip_Only": "D clip only",
         "C_Noise_Only": "C noise only", "B_DP_SGD": "B clip + noise"}
TICKS = [10, 25, 50, 100, 300, 1000]


def series(curves, c):
    ts = sorted(int(t) for t in curves[c])
    m = np.array([curves[c][str(t)][0] for t in ts])
    s = np.array([curves[c][str(t)][1] for t in ts])
    return ts, m, s


def style_axis(ax, tmax, title):
    ax.set_xscale("log")
    ax.set_xticks([t for t in TICKS if t <= tmax])
    ax.set_xticklabels([str(t) for t in TICKS if t <= tmax])
    ax.minorticks_off()
    ax.set_xlim(8.5, tmax * 1.12)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_title(title, fontsize=8.5)
    ax.set_xlabel("optimisation iterations (log scale)", fontsize=8)
    if tmax > 300:
        ax.axvline(300, color="k", lw=0.7, ls=":")


plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.75), sharey=True,
                         gridspec_kw={"width_ratios": [1.2, 0.9, 1.2]})
for ax, (title, path) in zip(axes[:2], PANELS):
    curves = json.load(open(path, encoding="utf-8"))["curves"]
    for c in COND:
        ts, m, s = series(curves, c)
        ls, mk = STYLE[c]
        ax.plot(ts, m, ls=ls, marker=mk, ms=3.5, lw=1.5, color=COLOR[c], label=LABEL[c])
        ax.fill_between(ts, np.clip(m - s, 0, 1), np.clip(m + s, 0, 1), color=COLOR[c], alpha=0.13, lw=0)
    tmax = max(int(t) for t in curves[COND[0]])
    style_axis(ax, tmax, title)
axes[0].text(310, 0.04, "300 = budget\nused in the paper", fontsize=6.5, color="#444444", va="bottom")

ax = axes[2]
for name, path, col in HARD:
    curves = json.load(open(path, encoding="utf-8"))["curves"]
    for c, ls, mk in (("A_Standard", "--", "o"), ("C_Noise_Only", "-", "^")):
        ts, m, s = series(curves, c)
        ax.plot(ts, m, ls=ls, marker=mk, ms=3.0, lw=1.4, color=col,
                label=f"{name} {'A' if c == 'A_Standard' else 'C'}")
        ax.fill_between(ts, np.clip(m - s, 0, 1), np.clip(m + s, 0, 1), color=col, alpha=0.10, lw=0)
style_axis(ax, 1000, "Harder baselines: A dashed, C solid")

axes[0].set_ylabel("best-so-far validity (mean, ±1 sd)", fontsize=8)
axes[1].legend(fontsize=6.8, loc="upper left", frameon=False)
axes[2].legend(fontsize=6.2, loc="lower right", frameon=False, ncol=1, handlelength=2.0, labelspacing=0.3, borderaxespad=0.3)
fig.tight_layout(w_pad=1.0)
OUT = os.path.dirname(os.path.abspath(__file__))          # next to this script (journal_paper/)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"fig_curves.{ext}"), dpi=200)
print("saved", os.path.join(OUT, "fig_curves.pdf"))
