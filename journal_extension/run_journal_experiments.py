"""
MASTER RUNNER - journal extension experiments
=============================================
Tum protokol P1 deneylerini tek komuttan calistirir.

Ornekler
--------
# once hizli duman testi (kucuk veri, az epoch) — her sey caliyor mu?
python -m journal_extension.run_journal_experiments --exp all --dataset ciciot2023 --quick

# tam kosum, tek veri seti
python -m journal_extension.run_journal_experiments --exp all --dataset ciciot2023

# tek deney, tum veri setleri
python -m journal_extension.run_journal_experiments --exp 14 --dataset all --n-runs 10

# yalniz istatistik tekrarlari
python -m journal_extension.run_journal_experiments --exp 12b13b --dataset all --n-runs 10

Cikti: experiments_journal/<exp>_<dataset>/results.json + log.txt
"""
from __future__ import annotations
import argparse, time, traceback

from journal_extension import common as C
from journal_extension import (exp14_mechanism, exp15_lira_mia, exp16_domain_cf,
                               exp17_architecture, exp12b_13b_rerun, exp18_federated,
                               exp19_controls)
try:                                   # exp20 needs `pip install opacus`; keep the runner importable without it
    from journal_extension import exp20_opacus
except Exception as _ex:  # noqa: BLE001
    exp20_opacus = None; _EXP20_IMPORT_ERROR = _ex

DATASETS = ["ciciot2023", "edge-iiot", "toniot"]

# exp anahtari -> (aciklama, callable(dataset, args))
def _exp14(d, a): return exp14_mechanism.run(d, n_runs=a.n_runs, epochs=a.epochs,
                                             cf_samples=a.cf_samples, max_samples=a.max_samples,
                                             quick=a.quick, cf_max_iter=a.cf_max_iter)
def _exp15(d, a):
    kw = {}
    if a.shadow_epochs is not None: kw["shadow_epochs"] = a.shadow_epochs
    if a.shadow_train_size is not None: kw["shadow_train_size"] = a.shadow_train_size
    return exp15_lira_mia.run(d, n_shadow=a.n_shadow, epochs=a.epochs,
                              max_samples=a.max_samples, quick=a.quick, **kw)
def _exp16(d, a): return exp16_domain_cf.run(d, n_samples=a.cf_samples, epochs=a.epochs,
                                             max_samples=a.max_samples, quick=a.quick,
                                             cf_max_iter=a.cf_max_iter,
                                             semantics=a.exp16_semantics,
                                             model_conditions=a.exp16_models,
                                             snap_every=a.exp16_snap_every,
                                             n_runs=a.exp16_runs)
def _exp17(d, a): return exp17_architecture.run(d, n_runs=max(3, a.n_runs // 2), epochs=a.epochs,
                                                cf_samples=a.cf_samples, max_samples=a.max_samples,
                                                quick=a.quick, cf_max_iter=a.cf_max_iter)
def _exp1213(d, a): return exp12b_13b_rerun.run(d, which=a.which_1213, n_runs=a.n_runs, epochs=a.epochs,
                                                cf_samples=a.cf_samples, max_samples=a.max_samples,
                                                quick=a.quick, cf_max_iter=a.cf_max_iter)
def _exp18(d, a): return exp18_federated.run(d, n_runs=a.fl_runs, max_samples=a.max_samples,
                                             clients=a.fl_clients, rounds=a.fl_rounds,
                                             local_epochs=a.fl_local_epochs, alpha=a.fl_alpha,
                                             conditions=a.fl_conditions, cf_samples=a.cf_samples,
                                             cf_max_iter=a.cf_max_iter, quick=a.quick)
def _exp19(d, a): return exp19_controls.run(d, n_runs=a.ctrl_runs, max_samples=a.max_samples,
                                            epochs=a.epochs, cf_samples=a.cf_samples,
                                            cf_max_iter=a.cf_max_iter, parts=a.ctrl_parts,
                                            quick=a.quick, random_k=a.ctrl_random_k,
                                            curve_samples=a.ctrl_curve_samples)
def _exp20(d, a):
    if exp20_opacus is None:
        raise RuntimeError(f"exp20 needs opacus (pip install opacus): {_EXP20_IMPORT_ERROR}")
    return exp20_opacus.run(d, n_runs=a.ctrl_runs, max_samples=a.max_samples, epochs=a.epochs,
                            cf_samples=a.cf_samples, cf_max_iter=a.cf_max_iter,
                            target_eps=a.dp_target_eps, delta=a.dp_delta,
                            max_grad_norm=a.dp_max_grad_norm, quick=a.quick)

REGISTRY = {
    "14": ("DP Paradox mechanism analysis", _exp14),
    "15": ("Offline LiRA + shadow MIA", _exp15),
    "16": ("Domain-aware actionable CF", _exp16),
    "17": ("DP paradox across architectures", _exp17),
    "12b13b": ("n=10 reruns of exp12/13", _exp1213),
    "18": ("Federated (FedAvg) confirmation of the noise effect", _exp18),
    "19": ("Controls: trivialisation, curves, DeepFool, smoothing, discrete CF, transfer", _exp19),
    "20": ("Certified DP-SGD arm (Opacus, RDP accountant)", _exp20),
}
ORDER = ["14", "12b13b", "16", "17", "15", "18", "19", "20"]  # onerilen sira (getiri/maliyet)


def main():
    ap = argparse.ArgumentParser(description="Journal extension master runner")
    ap.add_argument("--exp", default="all",
                    help="virgullu liste: 14,15,16,17,12b13b,18,19,20 veya 'all'")
    ap.add_argument("--dataset", default="ciciot2023",
                    choices=["ciciot2023", "edge-iiot", "toniot", "all"])
    ap.add_argument("--n-runs", type=int, default=10)
    ap.add_argument("--n-shadow", type=int, default=16)
    ap.add_argument("--shadow-epochs", type=int, default=None,
                    help="exp15: golge model epoch sayisi (bos = modulun varsayilani; hedefle esit egitim onerilir)")
    ap.add_argument("--shadow-train-size", type=int, default=None,
                    help="exp15: golge model egitim boyutu (bos = modulun varsayilani)")
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--cf-samples", type=int, default=100)
    ap.add_argument("--cf-max-iter", type=int, default=C.CF_MAX_ITER, help="CF optimizasyon iterasyonu (varsayilan 300; hiz icin 100-150)")
    ap.add_argument("--which-1213", default="both", choices=["both", "12b", "13b"], help="exp12b13b hangi kisim (13b exp14 ile artik gereksiz)")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None, help="egitim batch boyutu (varsayilan 64; hiz icin 128/256)")
    ap.add_argument("--exp16-semantics", default="data", choices=["data", "name"],
                    help="exp16: tip cikarimi (data = v2 veriden; name = v1 addan)")
    ap.add_argument("--exp16-models", default="A,B", help="exp16: degerlendirilecek dedektorler (A,B)")
    ap.add_argument("--exp16-snap-every", type=int, default=10, help="exp16: hard-snap periyodu")
    ap.add_argument("--exp16-runs", type=int, default=1, help="exp16: tekrar (seed) sayisi (makalede 5)")
    ap.add_argument("--ctrl-runs", type=int, default=3, help="exp19/exp20: tekrar sayisi")
    ap.add_argument("--ctrl-parts", default="core",
                    help="exp19: core (=hist,random,curves,discrete) | all | virgullu liste")
    ap.add_argument("--ctrl-random-k", type=int, default=20, help="exp19: sorgu basina rastgele pertürbasyon sayisi")
    ap.add_argument("--ctrl-curve-samples", type=int, default=50, help="exp19: validity-iterasyon egrisi sorgu sayisi")
    ap.add_argument("--dp-target-eps", default="2,8", help="exp20: hedef epsilon listesi")
    ap.add_argument("--dp-delta", type=float, default=C.DP_DELTA, help="exp20: delta")
    ap.add_argument("--dp-max-grad-norm", type=float, default=1.0, help="exp20: per-example clip C")
    ap.add_argument("--fl-runs", type=int, default=3, help="exp18: tekrar sayisi (varsayilan 3)")
    ap.add_argument("--fl-clients", type=int, default=10, help="exp18: istemci sayisi")
    ap.add_argument("--fl-rounds", type=int, default=20, help="exp18: FedAvg tur sayisi")
    ap.add_argument("--fl-local-epochs", type=int, default=1, help="exp18: istemci basina yerel epoch")
    ap.add_argument("--fl-alpha", type=float, default=None, help="exp18: Dirichlet alpha (bos = IID)")
    ap.add_argument("--fl-conditions", default="A,C", help="exp18: kosullar, ornek A,C veya A,C,D,B")
    ap.add_argument("--quick", action="store_true", help="hizli duman testi")
    a = ap.parse_args()

    C.set_batch_size(a.batch_size)   # tum alt-deneylere yansir (ayni surec)
    exps = ORDER if a.exp.strip() == "all" else [e.strip() for e in a.exp.split(",")]
    dsets = DATASETS if a.dataset == "all" else [a.dataset]
    print(f"[runner] exps={exps} datasets={dsets} quick={a.quick} "
          f"batch={C.BATCH_SIZE} device={C.get_device()}")

    t0 = time.time(); status = []
    for e in exps:
        if e not in REGISTRY:
            print(f"[runner] UYARI: bilinmeyen exp '{e}', atlaniyor"); continue
        desc, fn = REGISTRY[e]
        for d in dsets:
            tag = f"exp{e}/{d}"; ts = time.time()
            print(f"\n[runner] >>> {tag} — {desc}")
            try:
                fn(d, a); status.append((tag, "OK", round(time.time() - ts, 1)))
            except Exception as ex:
                traceback.print_exc()
                status.append((tag, f"FAIL: {ex}", round(time.time() - ts, 1)))
    print(f"\n[runner] ===== SUMMARY ({round(time.time()-t0,1)}s) =====")
    for tag, st, dt in status:
        print(f"  {tag:28s} {st:10s} {dt}s")


if __name__ == "__main__":
    main()
