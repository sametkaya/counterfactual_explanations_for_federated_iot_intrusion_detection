"""
EXP14 - DP Paradox MECHANISM ANALYSIS
=====================================
SIU 2026 bildirisindeki bulgu salt ampirikti. Bu deney gurultu -> validity
etkisini OLCULEBILIR mekanizmalara baglar:

Her kosul (A_Standard / B_DP_SGD / C_Noise_Only / D_Clip_Only) icin, n run:
  1) Egitim-ici global gradient-norm dagilimi (DP gradyani kucultur mu?)
  2) Karar sinirinda input-gradient normu  ||d loss / d x||   (sinir dikligi)
  3) Input-Hessian izi (Hutchinson)          Tr(H_x)          (egrilik/curvature)
  4) Karar marji (top1-top2 softmax ve logit)                 (sinir yumusakligi)
  5) ECE kalibrasyon hatasi
  6) Parametre L2 normu                                        (flatness vekili)
  7) CF validity  (mekanizma <-> validity korelasyonu icin)

Baslangic hipotezi: Gauss gurultusu iceren kosullar (B,C) daha DUSUK
input-grad normu, daha KUCUK egrilik ve daha KUCUK marj -> daha yumusak karar
siniri -> daha YUKSEK CF validity ("implicit regularization"). Deney bu
hipotezi test eder; olcumler tersini gostermistir (gurultulu detektorlerde
sinir daha dik, marj daha kucuk), bkz. makale Bolum 6.2.

Calistir:
    python -m journal_extension.exp14_mechanism --dataset ciciot2023 --n-runs 10
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
import torch.nn as nn

from journal_extension import common as C


# -----------------------------------------------------------------------------
# Mekanizma metrikleri
# -----------------------------------------------------------------------------
def _input_grad_and_curvature(model, X, device, n_probes: int = 3, max_pts: int = 256):
    """
    Attack noktalarinda sinir-geometrisi olcutleri:
      - loss_grad_norm : ||d CE/dx||  (loss tabanli; model cokerse buyur -> confound)
      - margin_grad_norm: ||d(z_top1 - z_top2)/dx||  (COKME-DAYANIKLI birincil olcut;
                          karar sinirinin yerel dikligi/Lipschitz vekili)
      - hess_trace     : Hutchinson ile Tr(H_x) (CE egriligi)
    Not: margin_grad_norm loss buyuklugunden bagimsizdir; 'boundary smoothing'
    iddiasi icin loss_grad_norm'dan daha guvenilirdir.
    """
    model.eval()
    ce = nn.CrossEntropyLoss()
    idx = torch.randperm(len(X))[:min(max_pts, len(X))]
    Xs = X[idx].to(device)
    loss_grads, margin_grads, hess_traces = [], [], []
    for i in range(len(Xs)):
        x = Xs[i:i+1].clone().detach().requires_grad_(True)
        logits = model(x)
        # margin gradyani (cokme-dayanikli sinir olcutu)
        top2 = torch.topk(logits, 2, dim=1).values
        margin = (top2[:, 0] - top2[:, 1]).sum()
        gm, = torch.autograd.grad(margin, x, retain_graph=True)
        margin_grads.append(float(gm.detach().norm(2).item()))
        # loss gradyani + Hutchinson egriligi
        y = logits.argmax(1).detach()
        loss = ce(logits, y)
        g, = torch.autograd.grad(loss, x, create_graph=True)
        loss_grads.append(float(g.detach().norm(2).item()))
        tr = 0.0
        for _ in range(n_probes):
            v = torch.randint(0, 2, x.shape, device=device, dtype=x.dtype) * 2 - 1  # Rademacher
            Hv, = torch.autograd.grad(g, x, grad_outputs=v, retain_graph=True)
            tr += float((Hv * v).sum().item())
        hess_traces.append(tr / n_probes)
    return loss_grads, margin_grads, hess_traces


def _decision_margins(model, X, device, max_pts: int = 2000):
    model.eval()
    idx = torch.randperm(len(X))[:min(max_pts, len(X))]
    with torch.no_grad():
        logits = model(X[idx].to(device))
        probs = torch.softmax(logits, dim=1)
        top2p, _ = torch.topk(probs, 2, dim=1)
        top2l, _ = torch.topk(logits, 2, dim=1)
    return (top2p[:, 0] - top2p[:, 1]).cpu().numpy(), (top2l[:, 0] - top2l[:, 1]).cpu().numpy()


def _ece(model, test_loader, device, n_bins: int = 15) -> float:
    model.eval(); confs, corrects = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            p = torch.softmax(model(bx.to(device)), dim=1)
            c, pred = p.max(1)
            confs.extend(c.cpu().numpy()); corrects.extend((pred.cpu() == by).numpy())
    confs, corrects = np.asarray(confs), np.asarray(corrects)
    bins = np.linspace(0, 1, n_bins + 1); ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (confs > lo) & (confs <= hi)
        if m.sum() > 0:
            ece += (m.mean()) * abs(corrects[m].mean() - confs[m].mean())
    return float(ece)


def _weight_norm(model) -> float:
    return float(sum(p.detach().norm(2).item() ** 2 for p in model.parameters()) ** 0.5)


# -----------------------------------------------------------------------------
def run(dataset: str, n_runs: int = 10, max_samples=None, epochs: int = C.EPOCHS,
        cf_samples: int = 100, quick: bool = False, batch_size=None,
        cf_max_iter: int = C.CF_MAX_ITER):
    log = C.Runlog("exp14_mechanism", dataset)
    device = C.get_device()
    C.set_batch_size(batch_size)
    if quick:
        max_samples = max_samples or 40000; epochs = 3; n_runs = 2; cf_samples = 25
    log.log(f"device={device} n_runs={n_runs} epochs={epochs} cf_samples={cf_samples} "
            f"batch={C.BATCH_SIZE}")

    Xtr, Xte, ytr, yte, info = C.load_data(dataset, max_samples)
    log.log(f"train={len(Xtr)} test={len(Xte)} F={info['num_features']} C={info['num_classes']}")
    train_loader, test_loader = C.make_loaders(Xtr, ytr, Xte, yte)
    cw = C.class_weights_tensor(ytr, info["num_classes"], device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    noise_std = C.sigma_from_epsilon(C.REFERENCE_EPSILON) * C.DP_CLIP_NORM
    log.log(f"reference eps={C.REFERENCE_EPSILON} -> noise_std={noise_std:.4f}")

    metrics = ["accuracy", "cf_validity", "train_grad_norm", "input_grad_norm",
               "margin_grad_norm", "hessian_trace", "margin_prob", "margin_logit",
               "ece", "weight_norm"]
    agg = {cond: {m: [] for m in metrics} for cond in C.CONDITIONS}

    for cond, (clip, noise) in C.CONDITIONS.items():
        log.log(f"\n--- condition {cond} (clip={clip}, noise={noise}) ---")
        for run_i in range(n_runs):
            C.set_seed(C.RANDOM_SEED + run_i)
            model = C.make_model("mlp", info).to(device)
            tinfo = C.train_condition(
                model, train_loader, use_clipping=clip, use_noise=noise,
                noise_std=(noise_std if noise else 0.0), epochs=epochs,
                lr=C.LEARNING_RATE, criterion=criterion, device=device,
                collect_grad_norms=True)
            ev = C.evaluate_model(model, test_loader, device)
            attacks = C.get_attack_samples(model, Xte, yte, cf_samples, device)
            cf = C.eval_cf(model, attacks, info, device, max_samples=cf_samples,
                           max_iter=cf_max_iter) if attacks is not None else {"validity": 0.0}
            g_in, g_margin, h_tr = _input_grad_and_curvature(
                model, attacks if attacks is not None else Xte, device)
            m_p, m_l = _decision_margins(model, attacks if attacks is not None else Xte, device)
            ece = _ece(model, test_loader, device)

            vals = {
                "accuracy": ev["accuracy"],
                "cf_validity": cf["validity"],
                "train_grad_norm": float(np.mean(tinfo["grad_norms"])) if tinfo["grad_norms"] else 0.0,
                "input_grad_norm": float(np.mean(g_in)),
                "margin_grad_norm": float(np.mean(g_margin)),
                "hessian_trace": float(np.mean(h_tr)),
                "margin_prob": float(np.mean(m_p)),
                "margin_logit": float(np.mean(m_l)),
                "ece": ece,
                "weight_norm": _weight_norm(model),
            }
            for m in metrics:
                agg[cond][m].append(vals[m])
            log.log(f"  run {run_i+1}/{n_runs}: acc={vals['accuracy']:.3f} "
                    f"val={vals['cf_validity']:.3f} |dMargin/dx|={vals['margin_grad_norm']:.3f} "
                    f"TrH={vals['hessian_trace']:.3f} margin={vals['margin_prob']:.3f} "
                    f"ECE={vals['ece']:.3f}")

    # ---- ozet + noise ana etkisi + COKME-KONTROLLU temiz kontrast ------------
    summary = {"conditions": {}, "noise_main_effect": {}, "clean_contrast_C_vs_A": {},
               "hypothesis_check": {}, "confound_note": ""}
    for cond in C.CONDITIONS:
        summary["conditions"][cond] = {m: {"mean": C.mean_std(agg[cond][m])[0],
                                           "std": C.mean_std(agg[cond][m])[1],
                                           "raw": agg[cond][m]} for m in metrics}

    def cmean(conds, m):
        xs = [x for c in conds for x in agg[c][m]]
        return float(np.mean(xs)) if xs else 0.0

    # 2x2 noise ana etkisi = ort(B,C) - ort(A,D)
    for m in metrics:
        noise_on = cmean(["B_DP_SGD", "C_Noise_Only"], m)
        noise_off = cmean(["A_Standard", "D_Clip_Only"], m)
        summary["noise_main_effect"][m] = {"noise_on": noise_on, "noise_off": noise_off,
                                           "delta": noise_on - noise_off}
    # TEMIZ kontrast: C_Noise_Only - A_Standard
    # (B_DP_SGD az epoch/yuksek noise'ta accuracy'i cokertip validity'yi trivial
    #  sekilde 1'e itebilir; clipping'siz-noise kosulu C, cokme confound'unu azaltir)
    for m in metrics:
        summary["clean_contrast_C_vs_A"][m] = cmean(["C_Noise_Only"], m) - cmean(["A_Standard"], m)

    accB = cmean(["B_DP_SGD"], "accuracy"); accA = cmean(["A_Standard"], "accuracy")
    collapsed = accB < 0.5 * max(accA, 1e-9)
    summary["confound_note"] = (
        f"B_DP_SGD accuracy={accB:.3f} vs A_Standard={accA:.3f}. "
        + ("UYARI: B cokmus gorunuyor (dusuk epoch/yuksek noise). Bu kosulda validity "
           "artisi trivial olabilir; mekanizma iddiasini clean_contrast_C_vs_A ve tam "
           "epoch kosumu uzerinden kur." if collapsed else
           "B makul accuracy koruyor; 2x2 ana etki guvenilir."))

    # hipotez: TEMIZ kontrast (C vs A) uzerinden — cokme-dayanikli margin_grad ile
    cc = summary["clean_contrast_C_vs_A"]
    summary["hypothesis_check"] = {
        "basis": "C_Noise_Only vs A_Standard (collapse-robust contrast)",
        "validity_increases_with_noise": cc["cf_validity"] > 0,
        "boundary_sharpness_decreases (margin_grad)": cc["margin_grad_norm"] < 0,
        "curvature_decreases (Tr H)": cc["hessian_trace"] < 0,
        "margin_decreases": cc["margin_prob"] < 0,
        "accuracy_preserved_in_C": cmean(["C_Noise_Only"], "accuracy") > 0.5 * accA,
        "interpretation": ("Temiz kontrastta (C vs A) validity artarken margin_grad/"
                           "curvature/margin dusuyorsa 'boundary smoothing' mekanizmasi, "
                           "accuracy cokmesi OLMADAN desteklenmis olur."),
    }
    log.log("\n=== NOISE MAIN EFFECT (ort B,C - ort A,D) ===")
    for m in metrics:
        log.log(f"  {m:18s}: {summary['noise_main_effect'][m]['delta']:+.4f}")
    log.log("=== CLEAN CONTRAST (C_Noise_Only - A_Standard) ===")
    for m in metrics:
        log.log(f"  {m:18s}: {cc[m]:+.4f}")
    log.log(f"confound_note: {summary['confound_note']}")
    log.log(f"hypothesis_check: {summary['hypothesis_check']}")
    log.save_json(summary)
    return summary


def main():
    ap = argparse.ArgumentParser(description="EXP14 DP Paradox mechanism analysis")
    ap.add_argument("--dataset", default="ciciot2023",
                    choices=["ciciot2023", "edge-iiot", "toniot", "all"])
    ap.add_argument("--n-runs", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--cf-samples", type=int, default=100)
    ap.add_argument("--cf-max-iter", type=int, default=C.CF_MAX_ITER, help="CF optimizasyon iterasyonu (varsayilan 300)")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None, help="egitim batch boyutu (varsayilan 64)")
    ap.add_argument("--quick", action="store_true", help="hizli duman testi")
    a = ap.parse_args()
    dsets = ["ciciot2023", "edge-iiot", "toniot"] if a.dataset == "all" else [a.dataset]
    for d in dsets:
        run(d, n_runs=a.n_runs, max_samples=a.max_samples, epochs=a.epochs,
            cf_samples=a.cf_samples, quick=a.quick, batch_size=a.batch_size,
            cf_max_iter=a.cf_max_iter)


if __name__ == "__main__":
    main()
