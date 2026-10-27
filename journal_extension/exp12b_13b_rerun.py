"""
EXP12b / EXP13b - n=10 RERUNS with corrected statistics
=======================================================
Gerekce: exp12/13 n=5 ile Wilcoxon'un ulasabilecegi en kucuk iki-yonlu
p=0.0625'tir, yani hicbir karsilastirma anlamli cikamaz; effect size olarak
da r ile Cohen's d birlikte kullanilmisti.

Bu modul:
  - exp12b: epsilon suprumu [0.5,1,2,5,10,inf], n=10 run, acc + cf_validity,
            Wilcoxon signed-rank (paired) + rank-biserial r (TUTARLI konvansiyon).
  - exp13b: 2x2 faktoriyel A/B/C/D, n=10 run, Kruskal-Wallis + Mann-Whitney U +
            rank-biserial r; faktoriyel ana etkiler (noise vs clipping).

Not: rank-biserial r [-1,1] araliginda, non-parametrik testlerle tutarli tek
effect-size olcusudur; makalede d yerine bunu kullanin.

Calistir:
    python -m journal_extension.exp12b_13b_rerun --dataset ciciot2023 --n-runs 10
"""
from __future__ import annotations
import argparse
import numpy as np
import torch.nn as nn
from scipy import stats

from journal_extension import common as C

EPSILONS = [0.5, 1.0, 2.0, 5.0, 10.0]


# --- effect sizes (non-parametric, tutarli) ----------------------------------
def rank_biserial_mwu(a, b):
    """Mann-Whitney U icin rank-biserial r = 1 - 2U/(n1 n2), [-1,1]."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return 0.0, 1.0
    U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    r = 1.0 - (2.0 * U) / (len(a) * len(b))
    return float(r), float(p)


def rank_biserial_wilcoxon(a, b):
    """Wilcoxon signed-rank icin matched-pairs rank-biserial r."""
    a, b = np.asarray(a), np.asarray(b)
    d = a - b
    d = d[d != 0]
    if len(d) < 1:
        return 0.0, 1.0
    try:
        W, p = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        return 0.0, 1.0
    ranks = stats.rankdata(np.abs(d))
    Rpos = ranks[d > 0].sum(); Rneg = ranks[d < 0].sum()
    tot = Rpos + Rneg
    r = float((Rpos - Rneg) / tot) if tot > 0 else 0.0
    return r, float(p)


def _train_eval(kind, value, info, tr_loader, te_loader, Xte, yte, criterion,
                device, epochs, cf_samples, run_i, cf_max_iter=C.CF_MAX_ITER):
    """kind='epsilon' -> DP-SGD @ eps; kind='condition' -> A/B/C/D."""
    C.set_seed(C.RANDOM_SEED + run_i)
    model = C.make_model("mlp", info).to(device)
    if kind == "epsilon":
        if value == float("inf"):
            clip = noise = False; nstd = 0.0
        else:
            clip = noise = True; nstd = C.sigma_from_epsilon(value) * C.DP_CLIP_NORM
    else:
        clip, noise = C.CONDITIONS[value]
        nstd = C.sigma_from_epsilon(C.REFERENCE_EPSILON) * C.DP_CLIP_NORM if noise else 0.0
    C.train_condition(model, tr_loader, use_clipping=clip, use_noise=noise,
                      noise_std=nstd, epochs=epochs, lr=C.LEARNING_RATE,
                      criterion=criterion, device=device)
    ev = C.evaluate_model(model, te_loader, device)
    attacks = C.get_attack_samples(model, Xte, yte, cf_samples, device)
    cf = C.eval_cf(model, attacks, info, device, max_samples=cf_samples,
                   max_iter=cf_max_iter) if attacks is not None else {"validity": 0.0}
    return ev["accuracy"], ev["f1_macro"], cf["validity"]


def run_exp12b(dataset, n_runs, max_samples, epochs, cf_samples, quick, cf_max_iter=C.CF_MAX_ITER):
    log = C.Runlog("exp12b_epsilon_n10", dataset)
    device = C.get_device()
    if quick:
        max_samples = max_samples or 40000; epochs = 3; n_runs = 3; cf_samples = 25
    Xtr, Xte, ytr, yte, info = C.load_data(dataset, max_samples)
    tr_loader, te_loader = C.make_loaders(Xtr, ytr, Xte, yte)
    cw = C.class_weights_tensor(ytr, info["num_classes"], device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    log.log(f"n_runs={n_runs} epsilons={EPSILONS}+inf cf_samples={cf_samples} cf_max_iter={cf_max_iter}")

    raw = {}
    for eps in EPSILONS + [float("inf")]:
        accs, f1s, vals = [], [], []
        for r in range(n_runs):
            a, f, v = _train_eval("epsilon", eps, info, tr_loader, te_loader, Xte, yte,
                                  criterion, device, epochs, cf_samples, r, cf_max_iter)
            accs.append(a); f1s.append(f); vals.append(v)
        raw[eps] = {"acc": accs, "f1": f1s, "val": vals}
        log.log(f"  eps={eps}: acc={np.mean(accs):.3f}±{np.std(accs):.3f} "
                f"val={np.mean(vals):.3f}±{np.std(vals):.3f}")

    base_acc, base_val = raw[float("inf")]["acc"], raw[float("inf")]["val"]
    res = {"n_runs": n_runs, "epsilons": {}, "stat_tests": {},
           "effect_size_convention": "rank-biserial r (Wilcoxon signed-rank, paired)"}
    for eps in EPSILONS + [float("inf")]:
        k = "inf" if eps == float("inf") else str(eps)
        res["epsilons"][k] = {
            "acc_mean": float(np.mean(raw[eps]["acc"])), "acc_std": float(np.std(raw[eps]["acc"])),
            "acc_ci95": [float(np.percentile(raw[eps]["acc"], 2.5)), float(np.percentile(raw[eps]["acc"], 97.5))],
            "f1_mean": float(np.mean(raw[eps]["f1"])),
            "val_mean": float(np.mean(raw[eps]["val"])), "val_std": float(np.std(raw[eps]["val"])),
            "raw_acc": raw[eps]["acc"], "raw_val": raw[eps]["val"],
        }
    for eps in EPSILONS:
        r_acc, p_acc = rank_biserial_wilcoxon(raw[eps]["acc"], base_acc)
        r_val, p_val = rank_biserial_wilcoxon(raw[eps]["val"], base_val)
        res["stat_tests"][f"eps_{eps}_vs_baseline"] = {
            "accuracy": {"test": "Wilcoxon signed-rank", "p_value": p_acc,
                         "significant": p_acc < 0.05, "rank_biserial_r": r_acc},
            "cf_validity": {"test": "Wilcoxon signed-rank", "p_value": p_val,
                            "significant": p_val < 0.05, "rank_biserial_r": r_val},
        }
        log.log(f"  eps={eps} vs inf: acc p={p_acc:.4f}(r={r_acc:+.2f}) "
                f"val p={p_val:.4f}(r={r_val:+.2f})")
    log.save_json(res)
    return res


def run_exp13b(dataset, n_runs, max_samples, epochs, cf_samples, quick, cf_max_iter=C.CF_MAX_ITER):
    log = C.Runlog("exp13b_paradox_n10", dataset)
    device = C.get_device()
    if quick:
        max_samples = max_samples or 40000; epochs = 3; n_runs = 3; cf_samples = 25
    Xtr, Xte, ytr, yte, info = C.load_data(dataset, max_samples)
    tr_loader, te_loader = C.make_loaders(Xtr, ytr, Xte, yte)
    cw = C.class_weights_tensor(ytr, info["num_classes"], device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    log.log(f"n_runs={n_runs} conditions={list(C.CONDITIONS)} cf_samples={cf_samples} cf_max_iter={cf_max_iter}")

    raw = {}
    for cond in C.CONDITIONS:
        accs, vals = [], []
        for r in range(n_runs):
            a, f, v = _train_eval("condition", cond, info, tr_loader, te_loader, Xte, yte,
                                  criterion, device, epochs, cf_samples, r, cf_max_iter)
            accs.append(a); vals.append(v)
        raw[cond] = {"acc": accs, "val": vals}
        log.log(f"  {cond}: acc={np.mean(accs):.3f}±{np.std(accs):.3f} "
                f"val={np.mean(vals):.3f}±{np.std(vals):.3f}")

    res = {"n_runs": n_runs, "conditions": {}, "stat_tests": {}, "factorial_analysis": {},
           "effect_size_convention": "rank-biserial r (Mann-Whitney U)"}
    for cond in C.CONDITIONS:
        res["conditions"][cond] = {
            "acc_mean": float(np.mean(raw[cond]["acc"])), "acc_std": float(np.std(raw[cond]["acc"])),
            "val_mean": float(np.mean(raw[cond]["val"])), "val_std": float(np.std(raw[cond]["val"])),
            "raw_acc": raw[cond]["acc"], "raw_val": raw[cond]["val"],
        }
    # Kruskal-Wallis (validity)
    groups = [raw[c]["val"] for c in C.CONDITIONS]
    try:
        H, pKW = stats.kruskal(*groups)
    except ValueError:
        H, pKW = 0.0, 1.0
    res["stat_tests"]["kruskal_wallis_validity"] = {"H": float(H), "p_value": float(pKW),
                                                    "significant": pKW < 0.05}
    for a_, b_ in [("A_Standard", "B_DP_SGD"), ("A_Standard", "C_Noise_Only"),
                   ("A_Standard", "D_Clip_Only"), ("B_DP_SGD", "C_Noise_Only")]:
        r, p = rank_biserial_mwu(raw[a_]["val"], raw[b_]["val"])
        res["stat_tests"][f"{a_}_vs_{b_}"] = {"test": "Mann-Whitney U", "p_value": p,
                                              "significant": p < 0.05, "rank_biserial_r": r}
    # faktoriyel ana etkiler (validity, yuzde puan)
    def m(c): return float(np.mean(raw[c]["val"]))
    noise_main = ((m("C_Noise_Only") + m("B_DP_SGD")) - (m("A_Standard") + m("D_Clip_Only"))) / 2 * 100
    clip_main = ((m("D_Clip_Only") + m("B_DP_SGD")) - (m("A_Standard") + m("C_Noise_Only"))) / 2 * 100
    interaction = (m("B_DP_SGD") - m("C_Noise_Only") - m("D_Clip_Only") + m("A_Standard")) * 100
    res["factorial_analysis"] = {"noise_main_effect_pp": noise_main,
                                 "clipping_main_effect_pp": clip_main,
                                 "interaction_pp": interaction}
    log.log(f"  KW H={H:.2f} p={pKW:.4f} | noise_main={noise_main:+.2f}pp "
            f"clip_main={clip_main:+.2f}pp interaction={interaction:+.2f}pp")
    log.save_json(res)
    return res


def run(dataset, which="both", n_runs=10, max_samples=None, epochs=C.EPOCHS,
        cf_samples=100, quick=False, batch_size=None, cf_max_iter=C.CF_MAX_ITER):
    C.set_batch_size(batch_size)
    if which in ("both", "12", "12b"):
        run_exp12b(dataset, n_runs, max_samples, epochs, cf_samples, quick, cf_max_iter)
    if which in ("both", "13", "13b"):
        run_exp13b(dataset, n_runs, max_samples, epochs, cf_samples, quick, cf_max_iter)


def main():
    ap = argparse.ArgumentParser(description="EXP12b/13b n=10 reruns")
    ap.add_argument("--dataset", default="ciciot2023",
                    choices=["ciciot2023", "edge-iiot", "toniot", "all"])
    ap.add_argument("--which", default="both", choices=["both", "12b", "13b"])
    ap.add_argument("--n-runs", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--cf-samples", type=int, default=100)
    ap.add_argument("--cf-max-iter", type=int, default=C.CF_MAX_ITER, help="CF optimizasyon iterasyonu (varsayilan 300; hiz icin 100-150)")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None, help="egitim batch boyutu (varsayilan 64)")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    dsets = ["ciciot2023", "edge-iiot", "toniot"] if a.dataset == "all" else [a.dataset]
    for d in dsets:
        run(d, which=a.which, n_runs=a.n_runs, epochs=a.epochs, cf_samples=a.cf_samples,
            max_samples=a.max_samples, quick=a.quick, batch_size=a.batch_size,
            cf_max_iter=a.cf_max_iter)


if __name__ == "__main__":
    main()
