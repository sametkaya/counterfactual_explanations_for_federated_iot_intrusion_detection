"""
Journal extension experiments (protocol P1)
===========================================
SIU 2026 bildirisindeki gozlemi (gurultulu egitim -> daha yuksek
counterfactual validity) mekanizma olcumleri, denetim deneyleri (controls),
uyelik denetimi ve sertifikali DP-SGD kolu ile genisleten deneyler.

Modules
-------
- common            : ortak yardimcilar (veri, model, DP-SGD egitim, CF eval)
- exp14_mechanism   : DP Paradox mekanizma analizi (gradient/curvature/margin/ECE)
- exp15_lira_mia    : LiRA + shadow-model MIA (TPR@0.1%/1% FPR)
- exp16_domain_cf   : domain-aware kisitlar + ham uzayda CF + actionability
- exp17_architecture: ikinci mimari ile paradox kapasite sinir kosulu
- exp12b_13b_rerun  : exp12/13'un n=10 run ile tekrari
- exp18_federated   : FedAvg altinda tekrar
- exp19_controls    : denetimler, uretici varyantlari, butce egrileri, optimizer tanilari
- exp20_opacus      : sertifikali DP-SGD kolu (Opacus, RDP)
- run_journal_experiments : ana runner (argparse)

Ana runner:
    python -m journal_extension.run_journal_experiments --dataset all --exp all
"""
__all__ = [
    "common",
    "exp14_mechanism",
    "exp15_lira_mia",
    "exp16_domain_cf",
    "exp17_architecture",
    "exp12b_13b_rerun",
    "exp18_federated",
    "exp19_controls",
    "exp20_opacus",
]
