"""Table tab:diag (optimiser diagnostics of the generator, control (k)) from the exp19 _diag runs.
Usage: python3 journal_paper/make_table_diag.py [experiments_journal]"""
import json, os, sys
from decimal import Decimal, ROUND_HALF_UP

R = sys.argv[1] if len(sys.argv) > 1 else "experiments_journal"


def r(x, n=3):
    if x != x:
        return "--"
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-n), rounding=ROUND_HALF_UP))


for ds, f in [("CICIoT2023", "exp19_controls_diag_ciciot2023"), ("Edge-IIoTset", "exp19_controls_diag_edge-iiot")]:
    s = json.load(open(os.path.join(R, f, "results.json")))["conditions"]
    print(f"\\multicolumn{{7}}{{l}}{{\\emph{{{ds}}}}} \\\\")
    for c, cn in [("A_Standard", "A"), ("C_Noise_Only", "C")]:
        for v, vn in [("adam", "Adam (default)"), ("adam_b1_0", "Adam, $\\beta_1=0$"),
                      ("adam_b2_0.9", "Adam, $\\beta_2=0.9$"), ("signgd", "Sign steps")]:
            g = lambda k: s[c][f"diag_{v}_{k}"]["mean"]
            sd = s[c][f"diag_{v}_validity_all"]["std"]
            ratio = (f"${r(g('adam_ratio_early_all'), 2)}$ / ${r(g('adam_ratio_late_all'), 2)}$"
                     if f"diag_{v}_adam_ratio_early_all" in s[c] and g("adam_ratio_early_all") == g("adam_ratio_early_all") else "--")
            print(f"{cn} & {vn} & ${r(g('validity_all'))}\\pm{r(sd)}$ & ${r(g('n_iter_all'), 0)}$ & "
                  f"${r(g('step_norm_early_all'))}$ / ${r(g('step_norm_late_all'))}$ & "
                  f"${r(g('sign_flip_early_all'), 2)}$ / ${r(g('sign_flip_late_all'), 2)}$ & {ratio} \\\\")
    print("\\midrule")
