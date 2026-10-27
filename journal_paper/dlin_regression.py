"""Usage: python3 journal_paper/dlin_regression.py [experiments_journal]
d_lin vs validity across all (condition, run) points with per-run geometry.
Sources: exp14 (CIC n=10, Edge n=5), exp18 (CIC FedAvg n=3); later: exp19, exp20.
Writes fig_dlin_scatter.pdf/png and prints the statistics used in the paper."""
import json, sys, numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.ticker
import matplotlib.pyplot as plt

import os, sys
R = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("RESULTS_DIR", "experiments_journal")   # run from repo root
COND = ["A_Standard", "D_Clip_Only", "C_Noise_Only", "B_DP_SGD"]
LAB = {"A_Standard": "A (standard)", "D_Clip_Only": "D (clip only)", "C_Noise_Only": "C (noise only)", "B_DP_SGD": "B (clip+noise)"}
MK = {"A_Standard": "o", "D_Clip_Only": "s", "C_Noise_Only": "^", "B_DP_SGD": "D"}

def points(path, source, dataset, conds=COND):
    d = json.load(open(path))["conditions"]; out = []
    for c in conds:
        if c not in d: continue
        ml = d[c]["margin_logit"]["raw"]; gm = d[c]["margin_grad_norm"]["raw"]; v = d[c]["cf_validity"]["raw"]
        for i in range(len(v)):
            out.append(dict(source=source, dataset=dataset, cond=c, run=i, d_lin=ml[i] / max(gm[i], 1e-12),
                            margin=ml[i], grad=gm[i], validity=v[i]))
    return out

P = (points(f"{R}/exp14_mechanism_ciciot2023/results.json", "exp14", "CICIoT2023")
     + points(f"{R}/exp14_mechanism_edge-iiot/results.json", "exp14", "Edge-IIoTset")
     + points(f"{R}/exp18_federated_ciciot2023/results.json", "exp18", "CICIoT2023 (FedAvg)"))
# exp19 (controls, CICIoT2023): trained on CPU with the same protocol and seeds 42-44; the CPU RNG
# stream differs from the CUDA one, so these are independent detectors, not re-runs of exp14 (their
# validities differ per seed). Pooled as a fourth CICIoT2023 source. exp20 (Opacus) uses a
# group-normalised architecture whose geometry is not comparable and is NOT pooled (paper, Sec. 6.5).
p19 = f"{R}/exp19_controls_ciciot2023/results.json"
if os.path.exists(p19):
    d = json.load(open(p19))["conditions"]
    for c in COND:
        if c in d and "margin_logit" in d[c]:
            ml = d[c]["margin_logit"]["raw"]; gm = d[c]["margin_grad_norm"]["raw"]; val = d[c]["cf_validity"]["raw"]
            for i in range(len(val)):
                P.append(dict(source="exp19", dataset="CICIoT2023 (controls)", cond=c, run=i,
                              d_lin=ml[i]/max(gm[i], 1e-12), margin=ml[i], grad=gm[i], validity=val[i]))
x = np.array([p["d_lin"] for p in P]); y = np.array([p["validity"] for p in P])
print(f"n = {len(P)} points")
rho, prho = stats.spearmanr(x, y); r, pr = stats.pearsonr(np.log(x), y)
print(f"pooled: Spearman rho = {rho:.3f} (p = {prho:.2e}); Pearson r(validity, log d_lin) = {r:.3f} (p = {pr:.2e})")
# within-dataset
for ds in sorted(set(p["dataset"] for p in P)):
    xs = np.array([p["d_lin"] for p in P if p["dataset"] == ds]); ys = np.array([p["validity"] for p in P if p["dataset"] == ds])
    rh, ph = stats.spearmanr(xs, ys)
    print(f"  {ds:24s} n={len(xs):3d} Spearman rho={rh:+.3f} p={ph:.2e}  d_lin range [{xs.min():.2f},{xs.max():.2f}] validity range [{ys.min():.3f},{ys.max():.3f}]")
# linear fit of validity on log d_lin (pooled) with 95% CI on slope
X = np.log(x); res = stats.linregress(X, y)
print(f"OLS validity ~ log d_lin: slope={res.slope:.3f} (SE {res.stderr:.3f}), intercept={res.intercept:.3f}, R^2={res.rvalue**2:.3f}, p={res.pvalue:.2e}")
# partial: does d_lin explain validity beyond margin alone / gradient alone?
for name, z in [("log margin", np.log([p["margin"] for p in P])), ("log grad", np.log([p["grad"] for p in P]))]:
    rr = stats.linregress(z, y); print(f"OLS validity ~ {name}: R^2={rr.rvalue**2:.3f}")
# two-predictor
A = np.column_stack([np.ones(len(P)), np.log([p["margin"] for p in P]), np.log([p["grad"] for p in P])])
beta, *_ = np.linalg.lstsq(A, y, rcond=None); yhat = A @ beta; r2 = 1 - ((y-yhat)**2).sum()/((y-y.mean())**2).sum()
print(f"OLS validity ~ log margin + log grad: beta={beta.round(3)}, R^2={r2:.3f}  (d_lin hypothesis predicts beta_margin ~ -beta_grad)")

# figure: two panels, within-dataset fits
from matplotlib.lines import Line2D
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3), sharey=True)
panels = [("CICIoT2023", ["CICIoT2023", "CICIoT2023 (FedAvg)", "CICIoT2023 (controls)"]),
          ("Edge-IIoTset", ["Edge-IIoTset"])]
cols = {"CICIoT2023": "#1f77b4", "CICIoT2023 (FedAvg)": "#2ca02c", "CICIoT2023 (controls)": "#9467bd",
        "Edge-IIoTset": "#d62728"}
stats_out = {}
for ax, (title, dss) in zip(axes, panels):
    sel = [p for p in P if p["dataset"] in dss]
    for ds in dss:
        for c in COND:
            pts = [p for p in sel if p["dataset"] == ds and p["cond"] == c]
            if not pts: continue
            filled = c in ("C_Noise_Only", "B_DP_SGD")
            ax.scatter([p["d_lin"] for p in pts], [p["validity"] for p in pts], s=30, marker=MK[c],
                       facecolors=cols[ds] if filled else "none", edgecolors=cols[ds], linewidths=1.0, alpha=0.9, zorder=3)
    xs = np.array([p["d_lin"] for p in sel]); ys = np.array([p["validity"] for p in sel])
    rh, ph = stats.spearmanr(xs, ys); rr = stats.linregress(np.log(xs), ys)
    stats_out[title] = {"n": len(sel), "spearman_rho": rh, "spearman_p": ph, "slope": rr.slope, "se": rr.stderr, "r2": rr.rvalue**2, "p": rr.pvalue}
    xx = np.linspace(xs.min(), xs.max(), 100)
    ax.plot(xx, rr.intercept + rr.slope*np.log(xx), color="k", lw=1.0, ls="--", zorder=2)
    ax.set_xscale("log"); ax.grid(alpha=0.3, which="major")
    ax.set_xlim(0.15, 2.6); ax.set_xticks([0.2, 0.5, 1.0, 2.0]); ax.set_xticklabels(["0.2", "0.5", "1", "2"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_title(f"{title} ($n$={len(sel)})\nSpearman $\\rho$={rh:.2f} ($p$={ph:.1g}); OLS $R^2$={rr.rvalue**2:.2f}", fontsize=8.5)
    ax.set_xlabel(r"$d_{\mathrm{lin}} = m_{\mathrm{logit}}/\|\nabla_{\mathbf{x}} m\|$ (log scale)", fontsize=9)
axes[0].set_ylabel("counterfactual validity")
h = [Line2D([0],[0], marker=MK[c], color="w", markerfacecolor="grey" if c in ("C_Noise_Only","B_DP_SGD") else "none",
            markeredgecolor="grey", markersize=7, label=LAB[c]) for c in COND]
h += [Line2D([0],[0], marker="o", color="w", markerfacecolor=cols[ds], markeredgecolor=cols[ds], markersize=7, label=ds)
      for ds in cols if any(p["dataset"] == ds for p in P)]
h += [Line2D([0],[0], color="k", ls="--", lw=1, label="OLS of validity on $\\log d_{\\rm lin}$ (within dataset)")]
fig.legend(handles=h, fontsize=7, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.12, 1, 1))
for ext in ("pdf", "png"): fig.savefig(f"fig_dlin_scatter.{ext}", dpi=200, bbox_inches="tight")
print(json.dumps(stats_out, indent=1))
json.dump({"n": len(P), "pooled_spearman_rho": rho, "pooled_spearman_p": prho, "pooled_ols_log_dlin": {"slope": res.slope, "se": res.stderr, "intercept": res.intercept, "r2": res.rvalue**2, "p": res.pvalue}, "within": stats_out, "points": P}, open("dlin_regression.json", "w"), indent=1)
