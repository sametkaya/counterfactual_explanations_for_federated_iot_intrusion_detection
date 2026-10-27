"""
EXP15 - STRONG MEMBERSHIP INFERENCE: offline LiRA + shadow models
=================================================================
Gerekce: P0'daki threshold/loss MIA (exp10) cok zayif kalir (~%50) ve DP'nin
faydasini ayirt edemez. Bu deney Carlini et al. (2022) "Likelihood Ratio Attack"in
OFFLINE surumunu ekler ve dusuk-FPR bolgesinde raporlar (TPR@0.1%FPR, TPR@1%FPR),
guvenlik topluluğunun standart olcutu.

Offline LiRA (ozet):
  - Audit kumesi A: yari uye (train'den), yari uye-degil (test'ten).
  - N gölge (shadow) model, A'yi ICERMEYEN bir referans havuzda egitilir -> her
    a in A icin tum golgeler OUT'tur.
  - Her a icin OUT guven dagilimindan Gauss(mu_out, sigma_out) uydurulur.
    Skor = Phi( (phi(a;target) - mu_out) / sigma_out ),  phi = logit(p_true);
    buyuk skor -> uye.
  - Target model DP'li ve DP'siz iki kez calisir; golgeler de ayni algoritmayla
    egitilir (dagilim eslesmesi icin).

Cikti: AUC, TPR@0.1%FPR, TPR@1%FPR, balanced accuracy — DP vs No-DP.

Positive control (--canaries N): 2N held-out records get a random wrong label; N of them are
added to the target's training set (members), N are kept out (non-members); every shadow is OUT
for all of them. A model that fits the N mislabelled members has memorised them, so the same
LiRA pipeline must give AUC >> 0.5 on the canary audit if the attack works. --no-dropout and a
small --batch-size make memorisation easier; --dedup drops exact duplicate feature rows before
the split (record-level splits of IoT traffic put duplicates on both sides, which equalises
train and test confidence and hides membership). The optimisation step count of the target is
logged and stored (steps = ceil(train/batch) * epochs).

Calistir:
    python -m journal_extension.exp15_lira_mia --dataset ciciot2023 --n-shadow 16
"""
from __future__ import annotations
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, roc_curve

from journal_extension import common as C


def _phi_logit_conf(model, X, y, device, batch: int = 512) -> np.ndarray:
    """phi = logit(p_true) = log(p/(1-p)), clamp'li (LiRA olcegi)."""
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = X[i:i+batch].to(device); yb = y[i:i+batch].to(device)
            p = torch.softmax(model(xb), dim=1)
            pt = p.gather(1, yb.view(-1, 1)).squeeze(1).clamp(1e-6, 1 - 1e-6)
            out.append(torch.log(pt / (1 - pt)).cpu().numpy())
    return np.concatenate(out)


def _tpr_at_fpr(y_true, scores, fpr_target: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    idx = np.searchsorted(fpr, fpr_target, side="right") - 1
    idx = max(0, min(idx, len(tpr) - 1))
    return float(tpr[idx])


def _train_target(info, loader, *, dp: bool, noise_std: float, epochs, lr, criterion, device, dropout: float = 0.3):
    C.set_seed(C.RANDOM_SEED)
    model = C.make_model("mlp", info, dropout=dropout).to(device)
    C.train_condition(model, loader, use_clipping=dp, use_noise=dp,
                      noise_std=(noise_std if dp else 0.0), epochs=epochs, lr=lr,
                      criterion=criterion, device=device)
    return model


def _run_attack(dp: bool, *, Xtr, ytr, Xte, yte, info, device, log,
                n_shadow, n_audit, shadow_train_size, shadow_epochs, noise_std, epochs,
                target_train_size=None, canaries: int = 0, dropout: float = 0.3):
    tag = "DP" if dp else "NoDP"
    cw = C.class_weights_tensor(ytr, info["num_classes"], device)
    criterion = nn.CrossEntropyLoss(weight=cw)

    # hedefin egitim kumesi: tum train (varsayilan) ya da --target-train-size kayitlik bir alt kume
    # (positive control: kucuk kume + cok epoch = ezberleme; uyeler bu alt kumeden cekilir, golgeler
    # alt kumenin DISINDAKI kayitlardan egitilir)
    if target_train_size and target_train_size < len(Xtr):
        t_idx = torch.randperm(len(Xtr))[:target_train_size]
    else:
        t_idx = torch.arange(len(Xtr))
    Xt, yt = Xtr[t_idx], ytr[t_idx]

    # audit set: yari uye (hedefin egitim kumesinden), yari uye-degil (test)
    half = n_audit // 2
    m_local = torch.randperm(len(Xt))[:half]
    m_idx = t_idx[m_local]
    te_perm = torch.randperm(len(Xte))
    n_idx = te_perm[:half]
    A_x = torch.cat([Xtr[m_idx], Xte[n_idx]], 0)
    A_y = torch.cat([ytr[m_idx], yte[n_idx]], 0)
    membership = np.concatenate([np.ones(len(m_idx)), np.zeros(len(n_idx))])

    # canaries (positive control): 2*canaries further held-out records with a random WRONG label;
    # the first half joins the target's training set, the second half stays out; all shadows are OUT
    can_x = can_y = None
    if canaries and canaries > 0:
        K_ = info["num_classes"]
        c_idx = te_perm[half:half + 2 * canaries]
        if len(c_idx) < 2 * canaries:
            raise ValueError("not enough held-out records for the canaries")
        g_ = torch.Generator().manual_seed(C.RANDOM_SEED + 7)
        can_x = Xte[c_idx].clone()
        true_y = yte[c_idx]
        shift = torch.randint(1, K_, (len(c_idx),), generator=g_)
        can_y = (true_y + shift) % K_                        # a label different from the true one
        can_in_x, can_in_y = can_x[:canaries], can_y[:canaries]
        Xt = torch.cat([Xt, can_in_x], 0); yt = torch.cat([yt, can_in_y], 0)
        can_membership = np.concatenate([np.ones(canaries), np.zeros(canaries)])

    # referans havuz: audit uyelerini ICERMEZ -> golgeler A icin OUT (alt kume kullanildiysa
    # havuz = alt kumenin disi, boylece golgeler hedefin hicbir egitim kaydini gormez)
    mask = torch.ones(len(Xtr), dtype=torch.bool); mask[m_idx] = False
    if target_train_size and target_train_size < len(Xtr):
        mask[t_idx] = False
    ref_idx = torch.arange(len(Xtr))[mask]

    # hedef model
    tr_loader, _ = C.make_loaders(Xt, yt, Xte, yte)
    steps_per_epoch = int(math.ceil(len(Xt) / C.BATCH_SIZE)); total_steps = steps_per_epoch * int(epochs)
    target = _train_target(info, tr_loader, dp=dp, noise_std=noise_std,
                           epochs=epochs, lr=C.LEARNING_RATE, criterion=criterion, device=device, dropout=dropout)
    # memorisation check: train vs test accuracy of the target
    _, te_loader = C.make_loaders(Xt, yt, Xte, yte)
    _, tr_as_test = C.make_loaders(Xt, yt, Xt, yt)      # unshuffled loader over the target's own training set
    tr_eval = C.evaluate_model(target, tr_as_test, device)
    te_eval = C.evaluate_model(target, te_loader, device)
    log.log(f"  [{tag}] target: train={len(Xt)} train_acc={tr_eval['accuracy']:.4f} test_acc={te_eval['accuracy']:.4f} "
            f"gap={tr_eval['accuracy'] - te_eval['accuracy']:+.4f} steps={total_steps} "
            f"({steps_per_epoch}/epoch x {epochs}, batch {C.BATCH_SIZE}, dropout {dropout})")
    canary_fit = None
    if can_x is not None:
        with torch.no_grad():
            target.eval()
            pin = target(can_x[:canaries].to(device)).argmax(1).cpu()
            pout = target(can_x[canaries:].to(device)).argmax(1).cpu()
        canary_fit = {"in_fit_rate": float((pin == can_y[:canaries]).float().mean()),      # memorised wrong labels
                      "out_fit_rate": float((pout == can_y[canaries:]).float().mean())}    # chance level for the held-out half
        log.log(f"  [{tag}] canaries: target predicts the wrong label of {canary_fit['in_fit_rate']:.3f} of the "
                f"{canaries} member canaries and {canary_fit['out_fit_rate']:.3f} of the {canaries} non-member canaries")
    phi_target = _phi_logit_conf(target, A_x, A_y, device)
    phi_target_can = _phi_logit_conf(target, can_x, can_y, device) if can_x is not None else None

    # golge modeller -> her a icin OUT phi dagilimi
    out_phis = np.zeros((n_shadow, len(A_x)), dtype=np.float32)
    out_phis_can = np.zeros((n_shadow, len(can_x)), dtype=np.float32) if can_x is not None else None
    for s in range(n_shadow):
        C.set_seed(1000 + s)
        sel = ref_idx[torch.randperm(len(ref_idx))[:min(shadow_train_size, len(ref_idx))]]
        sh_loader, _ = C.make_loaders(Xtr[sel], ytr[sel], Xte[:2], yte[:2])
        sh = C.make_model("mlp", info, dropout=dropout).to(device)
        C.train_condition(sh, sh_loader, use_clipping=dp, use_noise=dp,
                          noise_std=(noise_std if dp else 0.0), epochs=shadow_epochs,
                          lr=C.LEARNING_RATE, criterion=criterion, device=device)
        out_phis[s] = _phi_logit_conf(sh, A_x, A_y, device)
        if out_phis_can is not None:
            out_phis_can[s] = _phi_logit_conf(sh, can_x, can_y, device)
        log.log(f"  [{tag}] shadow {s+1}/{n_shadow} trained")

    mu_out = out_phis.mean(0); sigma_out = out_phis.std(0) + 1e-6
    # offline LiRA score (Carlini et al. 2022, Alg. 1, offline): Pr_{Z~N(mu_out, sigma_out^2)}[Z < phi_target]
    # = Phi(z); large when the target's confidence exceeds the OUT distribution -> member.
    # The direction matters: the one-sided p-value 1 - Phi(z) is the same statistic with the opposite
    # sign, its AUC is 1 - AUC and its TPR@FPR values belong to the reversed ranking.
    lira_scores = norm.cdf((phi_target - mu_out) / sigma_out)  # yuksek -> uye

    # baseline: dogrudan guven (confidence) skoru
    conf_scores = phi_target

    res = {"target": {"train_size": int(len(Xt)), "train_acc": float(tr_eval["accuracy"]),
                      "test_acc": float(te_eval["accuracy"]), "epochs": int(epochs),
                      "batch": int(C.BATCH_SIZE), "steps_per_epoch": steps_per_epoch, "steps": total_steps,
                      "dropout": dropout, "canaries": int(canaries or 0), "canary_fit": canary_fit},
           "raw": {"phi_target": np.round(phi_target, 4).tolist(), "mu_out": np.round(mu_out, 4).tolist(),
                   "sigma_out": np.round(sigma_out, 4).tolist(), "membership": membership.astype(int).tolist()}}
    for name, sc in [("lira_offline", lira_scores), ("confidence", conf_scores)]:
        res[name] = {
            "auc": float(roc_auc_score(membership, sc)),
            "tpr_at_0.1pct_fpr": _tpr_at_fpr(membership, sc, 0.001),
            "tpr_at_1pct_fpr": _tpr_at_fpr(membership, sc, 0.01),
            "tpr_at_5pct_fpr": _tpr_at_fpr(membership, sc, 0.05),
        }
        log.log(f"  [{tag}] {name}: AUC={res[name]['auc']:.4f} "
                f"TPR@0.1%={res[name]['tpr_at_0.1pct_fpr']:.4f} "
                f"TPR@1%={res[name]['tpr_at_1pct_fpr']:.4f}")
    if can_x is not None:
        mu_c = out_phis_can.mean(0); sd_c = out_phis_can.std(0) + 1e-6
        lira_can = norm.cdf((phi_target_can - mu_c) / sd_c)
        res["raw_canary"] = {"phi_target": np.round(phi_target_can, 4).tolist(), "mu_out": np.round(mu_c, 4).tolist(),
                             "sigma_out": np.round(sd_c, 4).tolist(), "membership": can_membership.astype(int).tolist()}
        for name, sc in [("lira_offline_canary", lira_can), ("confidence_canary", phi_target_can)]:
            res[name] = {
                "auc": float(roc_auc_score(can_membership, sc)),
                "tpr_at_0.1pct_fpr": _tpr_at_fpr(can_membership, sc, 0.001),
                "tpr_at_1pct_fpr": _tpr_at_fpr(can_membership, sc, 0.01),
                "tpr_at_5pct_fpr": _tpr_at_fpr(can_membership, sc, 0.05),
                "n": int(2 * canaries),
            }
            log.log(f"  [{tag}] {name} (positive control, {2 * canaries} mislabelled records, half members): "
                    f"AUC={res[name]['auc']:.4f} TPR@0.1%={res[name]['tpr_at_0.1pct_fpr']:.4f} "
                    f"TPR@1%={res[name]['tpr_at_1pct_fpr']:.4f} TPR@5%={res[name]['tpr_at_5pct_fpr']:.4f}")
    return res


def _dedup(Xtr, ytr, Xte, yte, log):
    """Drop exact duplicate feature rows (first occurrence kept, train before test), so that no record
    appears on both sides of the split; returns the filtered tensors and the counts removed."""
    X = torch.cat([Xtr, Xte], 0)
    _, first = np.unique(X.numpy(), axis=0, return_index=True)
    keep = np.zeros(len(X), dtype=bool); keep[first] = True
    ktr, kte = keep[:len(Xtr)], keep[len(Xtr):]
    log.log(f"  dedup: {int((~ktr).sum())} duplicate rows removed from train, {int((~kte).sum())} from test "
            f"({len(Xtr)}->{int(ktr.sum())}, {len(Xte)}->{int(kte.sum())})")
    return (Xtr[torch.from_numpy(ktr)], ytr[torch.from_numpy(ktr)], Xte[torch.from_numpy(kte)], yte[torch.from_numpy(kte)],
            {"train_removed": int((~ktr).sum()), "test_removed": int((~kte).sum())})


def run(dataset: str, n_shadow: int = 16, n_audit: int = 2000,
        shadow_train_size: int = 20000, shadow_epochs: int = 5,
        epochs: int = C.EPOCHS, max_samples=None, quick: bool = False, batch_size=None,
        out_suffix: str = "", target_train_size=None, canaries: int = 0, dedup: bool = False,
        no_dropout: bool = False):
    log = C.Runlog("exp15_lira_mia" + out_suffix, dataset)
    device = C.get_device()
    C.set_batch_size(batch_size)
    if quick:
        max_samples = max_samples or 40000; n_shadow = 4; n_audit = 400
        shadow_train_size = 8000; shadow_epochs = 2; epochs = 3
    log.log(f"device={device} n_shadow={n_shadow} n_audit={n_audit} "
            f"shadow_train_size={shadow_train_size} shadow_epochs={shadow_epochs} "
            f"target_epochs={epochs} target_train_size={target_train_size or 'all'} batch={C.BATCH_SIZE} "
            f"canaries={canaries} dedup={dedup} dropout={0.0 if no_dropout else 0.3}")

    Xtr, Xte, ytr, yte, info = C.load_data(dataset, max_samples)
    dedup_info = None
    if dedup:
        Xtr, ytr, Xte, yte, dedup_info = _dedup(Xtr, ytr, Xte, yte, log)
    noise_std = C.sigma_from_epsilon(C.REFERENCE_EPSILON) * C.DP_CLIP_NORM
    log.log(f"train={len(Xtr)} test={len(Xte)} noise_std(eps={C.REFERENCE_EPSILON})={noise_std:.4f}")

    out = {"reference_epsilon": C.REFERENCE_EPSILON, "attacks": {},
           "config": {"n_shadow": n_shadow, "n_audit": n_audit, "shadow_train_size": shadow_train_size,
                      "shadow_epochs": shadow_epochs, "target_epochs": epochs, "target_train_size": target_train_size,
                      "batch": C.BATCH_SIZE, "canaries": canaries, "dedup": dedup_info,
                      "dropout": 0.0 if no_dropout else 0.3, "max_samples": max_samples}}
    for dp in (False, True):
        out["attacks"]["DP" if dp else "NoDP"] = _run_attack(
            dp, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, info=info, device=device, log=log,
            n_shadow=n_shadow, n_audit=n_audit, shadow_train_size=shadow_train_size,
            shadow_epochs=shadow_epochs, noise_std=noise_std, epochs=epochs,
            target_train_size=target_train_size, canaries=canaries, dropout=(0.0 if no_dropout else 0.3))

    # DP etkinligi: LiRA AUC/TPR dususu
    a_no, a_dp = out["attacks"]["NoDP"]["lira_offline"], out["attacks"]["DP"]["lira_offline"]
    out["dp_effectiveness"] = {
        "lira_auc_nodp": a_no["auc"], "lira_auc_dp": a_dp["auc"],
        "auc_reduction": a_no["auc"] - a_dp["auc"],
        "tpr1pct_nodp": a_no["tpr_at_1pct_fpr"], "tpr1pct_dp": a_dp["tpr_at_1pct_fpr"],
        "dp_reduces_leakage": a_dp["auc"] < a_no["auc"],
    }
    log.log(f"\nDP effectiveness (LiRA): AUC {a_no['auc']:.4f} -> {a_dp['auc']:.4f} "
            f"(delta={out['dp_effectiveness']['auc_reduction']:+.4f})")
    log.save_json(out)
    return out


def main():
    ap = argparse.ArgumentParser(description="EXP15 offline LiRA + shadow MIA")
    ap.add_argument("--dataset", default="ciciot2023",
                    choices=["ciciot2023", "edge-iiot", "toniot", "all"])
    ap.add_argument("--n-shadow", type=int, default=16)
    ap.add_argument("--n-audit", type=int, default=2000)
    ap.add_argument("--shadow-train-size", type=int, default=20000)
    ap.add_argument("--shadow-epochs", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None, help="egitim batch boyutu (varsayilan 64)")
    ap.add_argument("--out-suffix", default="", help="output folder suffix, e.g. _mem -> exp15_lira_mia_mem_<dataset>")
    ap.add_argument("--target-train-size", type=int, default=None,
                    help="train the target on a random subset of this many records (memorisation regime)")
    ap.add_argument("--canaries", type=int, default=0,
                    help="positive control: N mislabelled held-out records added to the target's training set "
                         "and N kept out; LiRA is reported on these 2N canaries as well")
    ap.add_argument("--dedup", action="store_true", help="drop exact duplicate feature rows before the split")
    ap.add_argument("--no-dropout", action="store_true", help="train target and shadows without dropout")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    dsets = ["ciciot2023", "edge-iiot", "toniot"] if a.dataset == "all" else [a.dataset]
    for d in dsets:
        run(d, n_shadow=a.n_shadow, n_audit=a.n_audit, shadow_train_size=a.shadow_train_size,
            shadow_epochs=a.shadow_epochs, epochs=a.epochs, max_samples=a.max_samples,
            quick=a.quick, batch_size=a.batch_size, out_suffix=a.out_suffix,
            target_train_size=a.target_train_size, canaries=a.canaries, dedup=a.dedup,
            no_dropout=a.no_dropout)


if __name__ == "__main__":
    main()
