"""
EXP17 - DP PARADOX ACROSS ARCHITECTURES (kapasite sinir kosulu)
===============================================================
Gerekce: P0 yalnizca MLP kullanir; etki mimariye ozgu olabilir.
Bu deney A/B/C/D 2x2 faktoriyeli birden fazla mimaride tekrar ederek
paradoksun (noise -> validity artisi) mimariden bagimsiz olup olmadigini
ve model kapasitesiyle nasil degistigini test eder.

Mimariler: mlp (baseline), wide_mlp (yuksek kapasite), cnn1d.
Cikti: her mimari icin kosul-bazli acc & cf_validity + noise ana etkisi.

Calistir:
    python -m journal_extension.exp17_architecture --dataset ciciot2023 --n-runs 5
"""
from __future__ import annotations
import argparse
import numpy as np
import torch.nn as nn

from journal_extension import common as C

ARCHS = ["mlp", "wide_mlp", "cnn1d"]


def run(dataset: str, archs=None, n_runs: int = 5, max_samples=None,
        epochs: int = C.EPOCHS, cf_samples: int = 100, quick: bool = False, batch_size=None,
        cf_max_iter: int = C.CF_MAX_ITER):
    log = C.Runlog("exp17_architecture", dataset)
    device = C.get_device()
    C.set_batch_size(batch_size)
    archs = archs or ARCHS
    if quick:
        max_samples = max_samples or 40000; epochs = 3; n_runs = 2; cf_samples = 25
        archs = ["mlp", "cnn1d"]
    Xtr, Xte, ytr, yte, info = C.load_data(dataset, max_samples)
    tr_loader, te_loader = C.make_loaders(Xtr, ytr, Xte, yte)
    cw = C.class_weights_tensor(ytr, info["num_classes"], device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    noise_std = C.sigma_from_epsilon(C.REFERENCE_EPSILON) * C.DP_CLIP_NORM
    log.log(f"device={device} archs={archs} n_runs={n_runs} noise_std={noise_std:.4f}")

    out = {"reference_epsilon": C.REFERENCE_EPSILON, "architectures": {}}
    for arch in archs:
        log.log(f"\n===== architecture: {arch} =====")
        conds = {c: {"acc": [], "val": []} for c in C.CONDITIONS}
        for cond, (clip, noise) in C.CONDITIONS.items():
            for run_i in range(n_runs):
                C.set_seed(C.RANDOM_SEED + run_i)
                model = C.make_model(arch, info).to(device)
                C.train_condition(model, tr_loader, use_clipping=clip, use_noise=noise,
                                  noise_std=(noise_std if noise else 0.0), epochs=epochs,
                                  lr=C.LEARNING_RATE, criterion=criterion, device=device)
                ev = C.evaluate_model(model, te_loader, device)
                attacks = C.get_attack_samples(model, Xte, yte, cf_samples, device)
                cf = C.eval_cf(model, attacks, info, device, max_samples=cf_samples,
                               max_iter=cf_max_iter) if attacks is not None else {"validity": 0.0}
                conds[cond]["acc"].append(ev["accuracy"])
                conds[cond]["val"].append(cf["validity"])
            am, asd = C.mean_std(conds[cond]["acc"]); vm, vsd = C.mean_std(conds[cond]["val"])
            log.log(f"  {arch}/{cond}: acc={am:.3f}±{asd:.3f} val={vm:.3f}±{vsd:.3f}")

        def cmean(cs, k):
            xs = [x for c in cs for x in conds[c][k]]
            return float(np.mean(xs)) if xs else 0.0
        noise_effect_val = cmean(["B_DP_SGD", "C_Noise_Only"], "val") - \
            cmean(["A_Standard", "D_Clip_Only"], "val")
        baseline_val = C.mean_std(conds["A_Standard"]["val"])[0]
        out["architectures"][arch] = {
            "conditions": {c: {"acc_mean": C.mean_std(conds[c]["acc"])[0],
                               "acc_std": C.mean_std(conds[c]["acc"])[1],
                               "val_mean": C.mean_std(conds[c]["val"])[0],
                               "val_std": C.mean_std(conds[c]["val"])[1],
                               "raw_acc": conds[c]["acc"], "raw_val": conds[c]["val"]}
                           for c in C.CONDITIONS},
            "noise_main_effect_validity": noise_effect_val,
            "baseline_validity": baseline_val,
            "paradox_present": noise_effect_val > 0.01 and baseline_val < 0.999,
        }
        log.log(f"  [{arch}] noise main effect on validity = {noise_effect_val:+.4f} "
                f"(baseline_val={baseline_val:.3f}) paradox={out['architectures'][arch]['paradox_present']}")

    log.save_json(out)
    return out


def main():
    ap = argparse.ArgumentParser(description="EXP17 DP paradox across architectures")
    ap.add_argument("--dataset", default="ciciot2023",
                    choices=["ciciot2023", "edge-iiot", "toniot", "all"])
    ap.add_argument("--archs", default="mlp,wide_mlp,cnn1d")
    ap.add_argument("--n-runs", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--cf-samples", type=int, default=100)
    ap.add_argument("--cf-max-iter", type=int, default=C.CF_MAX_ITER, help="CF optimizasyon iterasyonu (varsayilan 300)")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None, help="egitim batch boyutu (varsayilan 64)")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    dsets = ["ciciot2023", "edge-iiot", "toniot"] if a.dataset == "all" else [a.dataset]
    for d in dsets:
        run(d, archs=[s.strip() for s in a.archs.split(",") if s.strip()],
            n_runs=a.n_runs, epochs=a.epochs, cf_samples=a.cf_samples,
            max_samples=a.max_samples, quick=a.quick, batch_size=a.batch_size,
            cf_max_iter=a.cf_max_iter)


if __name__ == "__main__":
    main()
