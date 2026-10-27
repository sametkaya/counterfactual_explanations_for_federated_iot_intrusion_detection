"""
Figures 1 and 2 of the journal manuscript (fig_paradox_epsilon, fig_mechanism_geometry).
The values are the means and standard deviations of experiments_journal/exp12b_epsilon_n10_*
(noise sweeps, ten seeds) and exp14_mechanism_* (boundary geometry, CICIoT2023 n=10,
Edge-IIoTset n=5), transcribed from their results.json; fig_dlin_scatter is made by
dlin_regression.py and fig_curves by make_fig_curves.py, both directly from the result files.
Outputs vector PDF (+PNG) next to this script.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.titlesize": 11,
    "axes.labelsize": 11, "legend.fontsize": 9.5, "xtick.labelsize": 10,
    "ytick.labelsize": 10, "axes.grid": True, "grid.alpha": 0.3,
    "grid.linewidth": 0.5, "axes.axisbelow": True, "figure.dpi": 150,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
      "red": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
      "yellow": "#F0E442", "gray": "#666666"}
COND = ["A\nstandard", "D\nclip only", "C\nnoise only", "B\nclip + noise"]
CCOL = [OI["gray"], OI["sky"], OI["orange"], OI["red"]]

def save(fig, name):
    fig.savefig(f"{name}.pdf"); fig.savefig(f"{name}.png", dpi=200); plt.close(fig)
    print("saved", name)

# =============================================================================
# F1 — noise sweeps under P1: CICIoT2023 (validity rises) vs Edge-IIoTset (not after Holm)
# x axis: Gaussian noise std sigma (0 = no noise). Same validity axis on both panels.
# =============================================================================
sig_lbl = ["0", "0.48", "0.97", "2.42", "4.84", "9.69"]
x = np.arange(len(sig_lbl))
sweeps = {
    "CICIoT2023  (n=10; 100 instances $\\times$ 300 it.)": dict(
        acc=[0.707, 0.589, 0.574, 0.357, 0.228, 0.171], acc_sd=[0.023, 0.028, 0.054, 0.140, 0.121, 0.089],
        val=[0.915, 0.978, 0.996, 0.997, 0.999, 0.996], val_sd=[0.031, 0.018, 0.004, 0.003, 0.002, 0.006],
        sig=["$p_{H}$=.01"]*5, sig_pos="bottom"),
    "Edge-IIoTset  (n=10; 25 instances $\\times$ 120 it.)": dict(
        acc=[0.991, 0.847, 0.432, 0.214, 0.110, 0.059], acc_sd=[0.001, 0.049, 0.175, 0.123, 0.074, 0.045],
        val=[0.721, 0.717, 0.801, 0.862, 0.855, 0.842], val_sd=[0.065, 0.064, 0.072, 0.119, 0.171, 0.203],
        sig=["$p_{H}$=.94", "$p_{H}$=.14", "$p_{H}$=.15", "$p_{H}$=.48", "$p_{H}$=.48"], sig_pos="top"),
}
VLIM = (50, 106)
fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.0))
for ax, (title, s) in zip(axes, sweeps.items()):
    acc = np.array(s["acc"])*100; val = np.array(s["val"])*100
    l1 = ax.errorbar(x, acc, yerr=np.array(s["acc_sd"])*100, marker="o", color=OI["blue"], capsize=3, lw=2, label="Accuracy")
    ax.set_ylim(0, 105); ax.set_xticks(x); ax.set_xticklabels(sig_lbl)
    ax.set_xlabel(r"Gaussian gradient-noise std $\sigma$ (0 = no noise)"); ax.set_title(title, fontsize=10.5)
    ax.set_ylabel("Accuracy (%)", color=OI["blue"]); ax.tick_params(axis="y", labelcolor=OI["blue"])
    ax2 = ax.twinx()
    l2 = ax2.errorbar(x, val, yerr=np.array(s["val_sd"])*100, marker="s", ls="--", color=OI["red"], capsize=3, lw=2, label="CF validity")
    ax2.set_ylim(*VLIM); ax2.grid(False)
    ax2.set_ylabel("Counterfactual validity (%)", color=OI["red"]); ax2.tick_params(axis="y", labelcolor=OI["red"])
    ax2.axhline(val[0], color=OI["red"], ls=":", lw=1)
    span = VLIM[1]-VLIM[0]
    ypos = VLIM[0] + 0.04*span if s["sig_pos"] == "bottom" else VLIM[1] - 0.06*span
    for xi, txt in zip(x[1:], s["sig"]):
        ax2.text(xi, ypos, txt, ha="center", fontsize=7.5, color=OI["red"])
    if ax is axes[0]:
        ax.legend(handles=[l1, l2], loc="center right", framealpha=0.9)
fig.tight_layout()
save(fig, "fig_paradox_epsilon")

# =============================================================================
# F3 — boundary geometry, both datasets: sharpness, curvature, logit margin, first-order distance
# =============================================================================
geo = {
    "CICIoT2023 (n=10)": {
        r"Sharpness $\|\partial m/\partial x\|$": ([3.97, 5.22, 0.95, 0.91], [0.44, 0.38, 0.11, 0.10]),
        r"Curvature Tr$(H_x)$":                    ([1.07, 1.25, 0.38, 0.39], [0.36, 0.40, 0.05, 0.07]),
        r"Logit margin $m$":                        ([5.25, 5.84, 0.68, 0.24], [0.30, 0.64, 0.14, 0.07]),
        r"Distance $m/\|\partial m/\partial x\|$":  ([1.33, 1.12, 0.71, 0.27], [0.13, 0.10, 0.10, 0.09]),
    },
    "Edge-IIoTset (n=5)": {
        r"Sharpness $\|\partial m/\partial x\|$": ([12.48, 16.10, 0.98, 0.93], [0.95, 0.55, 0.10, 0.06]),
        r"Curvature Tr$(H_x)$":                    ([30.8, 33.1, 0.55, 0.47], [18.2, 25.8, 0.13, 0.07]),
        r"Logit margin $m$":                        ([8.90, 10.69, 0.61, 0.23], [0.39, 0.64, 0.19, 0.04]),
        r"Distance $m/\|\partial m/\partial x\|$":  ([0.72, 0.67, 0.62, 0.25], [0.04, 0.06, 0.15, 0.06]),
    },
}
short = ["A", "D", "C", "B"]
fig, axes = plt.subplots(2, 4, figsize=(12.5, 6.2))
for r, (ds, panels) in enumerate(geo.items()):
    for c_, (title, (m, sd)) in enumerate(panels.items()):
        ax = axes[r, c_]
        ax.bar(short, m, yerr=sd, capsize=3, color=CCOL, edgecolor="black", lw=0.6)
        ax.set_ylim(0, (max(m) + max(sd)) * 1.28)
        if r == 0:
            ax.set_title(title, fontsize=10)
        if c_ == 0:
            ax.set_ylabel(ds + "\nvalue (mean $\\pm$ std)")
        if r == 1:
            ax.set_xlabel("Condition")
# annotate the key contrast
axes[0, 3].text(0.5, 0.92, "C vs A: $-47\\%$ ($p=.002$)", transform=axes[0, 3].transAxes, ha="center", fontsize=8.5, color=OI["red"])
axes[1, 3].text(0.5, 0.92, "C vs A: $-13\\%$ (n.s.)", transform=axes[1, 3].transAxes, ha="center", fontsize=8.5, color=OI["red"])
fig.text(0.5, -0.01, "A = standard,  D = clip only,  C = noise only,  B = clip + noise.   Noise flattens the boundary on both datasets; "
         "the first-order distance to the boundary shrinks only on CICIoT2023.", ha="center", fontsize=8.5, color=OI["gray"])
fig.tight_layout()
save(fig, "fig_mechanism_geometry")

