"""LaTeX rows of the regime, generator-variant, actionability and certified-arm tables (journal manuscript):
  (1) Edge-IIoTset regime table: fine sigma grid at batch 256 and batch sweep at sigma=2.42
      (exp19_controls_s01/_s02/_s03/_b64/_b128_edge-iiot, _k3 for sigma=2.42 at batch 256);
  (2) generator-variant table (tab:mechctl): CICIoT2023 (exp19 _mech + _sgd), Edge-IIoTset (_var, _k3),
      normalisation-free network (_nonorm);
  (3) nonorm panel of the controls table (exp19_controls_nonorm_ciciot2023);
  (4) actionability rows for C and D (exp16_domain_cf_cd_*);
  (5) ten-seed certified arm on the normalisation-free network (exp20_opacus_nonorm10_ciciot2023).
Rounding is half-up on the decimal representation; std is the population std (numpy default),
as in the other table scripts.
Usage: python3 journal_paper/make_tables_regime.py [experiments_journal]"""
import json, os, sys
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
from scipy import stats

ROOT = sys.argv[1] if len(sys.argv) > 1 else "experiments_journal"
COND = [("A_Standard", "A standard"), ("D_Clip_Only", "D clip only"),
        ("C_Noise_Only", "C noise only"), ("B_DP_SGD", "B clip$+$noise")]


def J(name):
    return json.load(open(os.path.join(ROOT, name, "results.json"), encoding="utf-8"))


def r(x, nd):
    q = Decimal(1).scaleb(-nd)
    return str(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))


def raw(d, cond, key):
    v = d["conditions"][cond][key]
    return np.array(v["raw"] if isinstance(v, dict) else v, dtype=float)


def ms(x, nd):
    x = np.asarray(x, float)
    return f"${r(x.mean(), nd)}\\pm{r(x.std(), nd)}$"


def pdiff(a, c, scale=100, nd=1):
    """paired c-a: mean, 95% t-interval (ddof=1), number of seeds with c>a."""
    d = np.asarray(c, float) - np.asarray(a, float)
    n = len(d); m = d.mean(); se = d.std(ddof=1) / np.sqrt(n); t = stats.t.ppf(0.975, n - 1)
    return f"${r(m*scale, nd)}$ [${r((m-t*se)*scale, nd)}$, ${r((m+t*se)*scale, nd)}$]", int((d > 0).sum()), n


# ------------------------------------------------------------------ (1) Edge regime table
print("% ---- (1) tab:edgeregime  (Edge-IIoTset, three seeds 42-44, 100 queries x 300 it.)")
rows = [("0.1", 256, "exp19_controls_s01_edge-iiot"), ("0.2", 256, "exp19_controls_s02_edge-iiot"),
        ("0.3", 256, "exp19_controls_s03_edge-iiot"), ("2.42", 256, "exp19_controls_k3_edge-iiot"),
        ("2.42", 128, "exp19_controls_b128_edge-iiot"), ("2.42", 64, "exp19_controls_b64_edge-iiot")]
for sig, bs, name in rows:
    d = J(name)
    A, C = "A_Standard", "C_Noise_Only"
    dv, up, n = pdiff(raw(d, A, "cf_validity"), raw(d, C, "cf_validity"))
    print(f"{sig} & {bs} & ${r(raw(d,A,'accuracy').mean(),3)}$ / ${r(raw(d,A,'f1_macro').mean(),3)}$ & "
          f"${r(raw(d,C,'accuracy').mean(),3)}$ / ${r(raw(d,C,'f1_macro').mean(),3)}$ & "
          f"{ms(raw(d,A,'cf_validity'),3)} & {ms(raw(d,C,'cf_validity'),3)} & {dv} ({up}/{n}) & "
          f"${r(raw(d,A,'margin_grad_norm').mean(),1)}\\to{r(raw(d,C,'margin_grad_norm').mean(),2)}$ & "
          f"${r(raw(d,A,'margin_logit').mean(),1)}\\to{r(raw(d,C,'margin_logit').mean(),2)}$ & "
          f"${r(raw(d,A,'d_lin').mean(),2)}\\to{r(raw(d,C,'d_lin').mean(),2)}$ & "
          f"${r(raw(d,A,'d_lin_target_median').mean(),2)}\\to{r(raw(d,C,'d_lin_target_median').mean(),2)}$ \\\\")

# ------------------------------------------------------------------ (2) generator variants
print("\n% ---- (2) tab:mechctl panels: default | cap only | dist only | neither | SGD 0.05 | SGD 0.5 | curve 300 | 500 | 1000 | sparse DF")
mech = J("exp19_controls_mech_ciciot2023"); sgd = J("exp19_controls_sgd_ciciot2023")
core3 = J("exp19_controls_ciciot2023")   # three-seed controls: default validity on the same detectors
for ck, cl in COND:
    cv = mech["curves"][ck]; sd = raw(mech, ck, "sparsedf_l2_median"); reach = raw(mech, ck, "sparsedf_reach").mean()
    print(f"{cl} & ${r(raw(core3,ck,'cf_validity').mean(),3)}$ & {ms(raw(mech,ck,'variant_cap_only_frac'),3)} & "
          f"{ms(raw(mech,ck,'variant_dist_only_frac'),3)} & {ms(raw(mech,ck,'variant_none_frac'),3)} & "
          f"{ms(raw(sgd,ck,'variant_sgd_frac'),3)} & {ms(raw(sgd,ck,'variant_sgd_lr0.5_frac'),3)} & "
          f"${r(cv['300'][0],3)}$ & ${r(cv['500'][0],3)}$ & ${r(cv['1000'][0],3)}$ & "
          f"${r(sd.mean(),2)}\\pm{r(sd.std(),2)}$ (${r(reach,2)}$) \\\\")
print("% Edge-IIoTset panel (default and curve from _k3, variants from _var)")
k3 = J("exp19_controls_k3_edge-iiot"); var = J("exp19_controls_var_edge-iiot")
for ck, cl in COND:
    cv = k3["curves"][ck]
    print(f"{cl} & ${r(raw(k3,ck,'cf_validity').mean(),3)}$ & {ms(raw(var,ck,'variant_cap_only_frac'),3)} & "
          f"{ms(raw(var,ck,'variant_dist_only_frac'),3)} & {ms(raw(var,ck,'variant_none_frac'),3)} & "
          f"{ms(raw(var,ck,'variant_sgd_frac'),3)} & {ms(raw(var,ck,'variant_sgd_lr0.5_frac'),3)} & "
          f"${r(cv['300'][0],3)}$ & -- & -- & -- \\\\")
print("% normalisation-free panel (_nonorm)")
nn = J("exp19_controls_nonorm_ciciot2023")
for ck, cl in COND:
    print(f"{cl} & ${r(raw(nn,ck,'cf_validity').mean(),3)}$ & {ms(raw(nn,ck,'variant_cap_only_frac'),3)} & "
          f"{ms(raw(nn,ck,'variant_dist_only_frac'),3)} & {ms(raw(nn,ck,'variant_none_frac'),3)} & "
          f"{ms(raw(nn,ck,'variant_sgd_frac'),3)} & {ms(raw(nn,ck,'variant_sgd_lr0.5_frac'),3)} & -- & -- & -- & -- \\\\")
print("% check: _sgd run reproduces the _mech variant columns (max abs difference of means):")
mx = max(abs(raw(mech, ck, k).mean() - raw(sgd, ck, k).mean()) for ck, _ in COND
         for k in ["variant_cap_only_frac", "variant_dist_only_frac", "variant_none_frac"])
print(f"%   {mx:.4f}")

# ------------------------------------------------------------------ (3) nonorm panel of tab:controls
print("\n% ---- (3) tab:controls, normalisation-free panel: F1 | pred benign | validity | any | true-attack (val) | random | discrete | dlin_t | DF | sharpness")
for ck, cl in COND:
    print(f"{cl} & {ms(raw(nn,ck,'f1_macro'),3)} & ${r(raw(nn,ck,'share_pred_target').mean(),3)}$ & {ms(raw(nn,ck,'cf_validity'),3)} & "
          f"${r(raw(nn,ck,'cf_validity_any').mean(),3)}$ & ${r(raw(nn,ck,'queries_true_attack_share').mean(),2)}$ (${r(raw(nn,ck,'cf_validity_true_attack_subset').mean(),3)}$) & "
          f"{ms(raw(nn,ck,'random_matched_flip_rate'),3)} & {ms(raw(nn,ck,'discrete_validity'),3)} & "
          f"{ms(raw(nn,ck,'d_lin_target_median'),2)} & {ms(raw(nn,ck,'deepfool_l2_median'),2)} & ${r(raw(nn,ck,'margin_grad_norm').mean(),2)}$ \\\\")
print("% nonorm d_lin per condition:", {cl: r(raw(nn, ck, 'd_lin').mean(), 2) for ck, cl in COND})

# ------------------------------------------------------------------ (4) actionability, all four conditions (CPU runs _ab_cpu and _cd)
print("\n% ---- (4) tab:actionability rows (generic validity / feasible (sparsity) | domain actionable (sparsity) | pooled)")
for ds, lab in [("ciciot2023", "CICIoT2023"), ("edge-iiot", "Edge-IIoTset")]:
    ab = J(f"exp16_domain_cf_ab_cpu_{ds}"); cd = J(f"exp16_domain_cf_cd_{ds}")
    allm = {}
    for src in (ab, cd):
        allm.update(src["models"])
    for ck, cl in [("A_Standard", "A no noise"), ("D_Clip_Only", "D clip only"),
                   ("C_Noise_Only", "C noise only, $\\sigma=2.42$"), ("B_DP_SGD", "B clip$+$noise, $\\sigma=2.42$")]:
        runs = allm[ck]["runs"]
        gv = np.array([x["generic"]["validity"] for x in runs]); gf = sum(round(x["generic"]["feasibility_rate"] * x["generic"]["n"]) for x in runs)
        gs = np.mean([x["generic"]["mean_sparsity"] for x in runs]); n = sum(x["generic"]["n"] for x in runs)
        dv = np.array([x["domain"]["actionability"] for x in runs]); dsp = np.mean([x["domain"]["mean_sparsity"] for x in runs])
        pooled = sum(round(x["domain"]["actionability"] * x["domain"]["n"]) for x in runs)
        print(f"{lab}: {cl} & {ms(gv,3)} / ${gf}/{n}$ ({r(gs,1)}) & {ms(dv,3)} ({r(dsp,1)}) & ${pooled}/{n}$ \\\\")
    A = np.array([x["domain"]["actionability"] for x in allm["A_Standard"]["runs"]])
    Ag = np.array([x["generic"]["validity"] for x in allm["A_Standard"]["runs"]])
    for ck, cl in [("C_Noise_Only", "C"), ("D_Clip_Only", "D"), ("B_DP_SGD", "B")]:
        X = np.array([x["domain"]["actionability"] for x in allm[ck]["runs"]]); Xg = np.array([x["generic"]["validity"] for x in allm[ck]["runs"]])
        a_ = sum(round(v * 100) for v in A); c_ = sum(round(v * 100) for v in X)
        print(f"%   {lab} {cl}-A actionable (paired): {pdiff(A, X)}  Wilcoxon p={stats.wilcoxon(X, A, method='exact').pvalue:.4f}  "
              f"Fisher pooled p={stats.fisher_exact([[c_, 500 - c_], [a_, 500 - a_]]).pvalue:.2g}  generic: {pdiff(Ag, Xg)}  "
              f"per seed {[r(v,2) for v in X]} vs A {[r(v,2) for v in A]}")

# ------------------------------------------------------------------ (5) ten-seed nonorm arm
print("\n% ---- (5) tab:opacus-nn, ten seeds: accuracy | macro-F1 | validity | sharpness | curvature | margin | dlin | ECE")
d = J("exp20_opacus_nonorm10_ciciot2023")
arms = [("NN_baseline", "No-norm baseline"), ("SIG_C_noise_only", "C noise only, $\\sigma=2.42$"),
        ("SIG_B_clip_noise", "B clip$+$noise, $\\sigma=2.42$"), ("DP_eps8", "DP-SGD $\\epsilon=8$"), ("DP_eps2", "DP-SGD $\\epsilon=2$")]
for ak, al in arms:
    a = d["arms"][ak]
    g = lambda k: np.array(a[k]["raw"], float)
    print(f"{al} & {ms(g('accuracy'),3)} & {ms(g('f1_macro'),3)} & {ms(g('cf_validity'),3)} & {ms(g('margin_grad_norm'),2)} & "
          f"{ms(g('hessian_trace'),2)} & {ms(g('margin_logit'),2)} & {ms(g('d_lin'),2)} & ${r(g('ece').mean(),3)}$ \\\\")
base = d["arms"]["NN_baseline"]
for ak, al in arms[1:]:
    a = d["arms"][ak]
    for k in ["cf_validity", "accuracy", "f1_macro", "margin_grad_norm", "hessian_trace", "margin_logit", "d_lin", "ece", "margin_prob"]:
        sc = 100 if k in ("cf_validity", "accuracy", "f1_macro") else 1
        txt, up, n = pdiff(base[k]["raw"], a[k]["raw"], scale=sc, nd=(1 if sc == 100 else 2))
        rel = (np.mean(a[k]["raw"]) / np.mean(base[k]["raw"]) - 1) * 100
        print(f"%   {al:30s} {k:16s} {txt} ({up}/{n} up)  rel {r(rel,0)}%  Wilcoxon p={stats.wilcoxon(a[k]['raw'], base[k]['raw'], method='exact').pvalue:.3f}")
print("% per-seed validity:", {al: [r(v, 3) for v in d['arms'][ak]['cf_validity']['raw']] for ak, al in arms})
mg = sum((d["arms"][ak]["margin_logit"]["raw"] for ak, _ in arms), []); vv = sum((d["arms"][ak]["cf_validity"]["raw"] for ak, _ in arms), [])
dl = sum((d["arms"][ak]["d_lin"]["raw"] for ak, _ in arms), []); sh = sum((d["arms"][ak]["margin_grad_norm"]["raw"] for ak, _ in arms), [])
print(f"% Spearman over {len(vv)} points: margin {stats.spearmanr(mg, vv).statistic:.3f} (p={stats.spearmanr(mg, vv).pvalue:.1e}), "
      f"dlin {stats.spearmanr(dl, vv).statistic:.3f} (p={stats.spearmanr(dl, vv).pvalue:.1e}), sharpness {stats.spearmanr(sh, vv).statistic:.3f}")

# ------------------------------------------------------------------ (6) tab:optim: generator optimiser variants
print("\n% ---- (6) tab:optim: Adam 0.05 (default) | SGD 0.05 | SGD 0.2 | SGD 0.35 | SGD 0.5 | sign-GD 0.05")
sgd2c = J("exp19_controls_sgd2_ciciot2023"); sgd2e = J("exp19_controls_sgd2_edge-iiot")
for lab, dflt, s1, s2 in [("CICIoT2023", core3, sgd, sgd2c), ("Edge-IIoTset", k3, var, sgd2e), ("CICIoT2023, normalisation-free network", nn, nn, None)]:
    print(f"% {lab}")
    for ck, cl in COND:
        cells = [f"${r(raw(dflt,ck,'cf_validity').mean(),3)}$", ms(raw(s1, ck, "variant_sgd_frac"), 3)]
        if s2 is not None:
            cells += [ms(raw(s2, ck, "variant_sgd_lr0.2_frac"), 3), ms(raw(s2, ck, "variant_sgd_lr0.35_frac"), 3)]
        else:
            cells += ["--", "--"]
        cells.append(ms(raw(s1, ck, "variant_sgd_lr0.5_frac"), 3))
        cells.append(ms(raw(s2, ck, "variant_signgd_frac"), 3) if s2 is not None else "--")
        print(f"{cl} & " + " & ".join(cells) + " \\\\")
    for s_, keys in ((s1, ["variant_sgd_frac", "variant_sgd_lr0.5_frac"]), (s2, ["variant_sgd_lr0.2_frac", "variant_sgd_lr0.35_frac", "variant_signgd_frac"])):
        if s_ is None: continue
        for k in keys:
            txt, up, n = pdiff(raw(s_, "A_Standard", k), raw(s_, "C_Noise_Only", k))
            print(f"%   C-A {k}: {txt} ({up}/{n} up)   any-valid A {r(raw(s_,'A_Standard',k.replace('_frac','_any')).mean(),3)} C {r(raw(s_,'C_Noise_Only',k.replace('_frac','_any')).mean(),3)}")

# (7) LiRA rows: see make_table_lira.py (fixed-variance score from the stored raw statistics;
#     the runs this section used, _mem2 and _pos, were superseded by _mem3 and _canary).

# ------------------------------------------------------------------ (8) budget curves to 1,000 iterations (A and C)
print("\n% ---- (8) curves to 1,000 iterations: 300 | 500 | 1,000")
for name, lab in [("exp19_controls_mech_ciciot2023", "CICIoT2023 BN"), ("exp19_controls_nncurves_ciciot2023", "normalisation-free"),
                  ("exp19_controls_cnncurves_ciciot2023", "CNN"), ("exp19_controls_b64curves_edge-iiot", "Edge-IIoTset batch 64")]:
    d = J(name)
    for ck in ["A_Standard", "C_Noise_Only"]:
        cv = d["curves"][ck]
        print(f"% {lab} {ck}: " + " ".join(f"{it}:{r(cv[it][0],3)}" for it in ["300", "500", "1000"]) + f"  (acc {r(raw(d,ck,'accuracy').mean(),3)})")
