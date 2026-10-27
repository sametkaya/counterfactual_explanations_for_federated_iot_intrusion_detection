"""LaTeX for Table 5 (controls), Table 6 (generator constraints) and the Appendix D histograms,
generated from the exp19 results so that every number in them is traceable.
Usage: python3 journal_paper/make_tables_exp19.py [experiments_journal]
"""
import json
import os
import sys
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction

import numpy as np

R = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("RESULTS_DIR", "experiments_journal")
core = json.load(open(f"{R}/exp19_controls_ciciot2023/results.json", encoding="utf-8"))        # CIC, seeds 42-44
k10 = json.load(open(f"{R}/exp19_controls_k10_ciciot2023/results.json", encoding="utf-8"))     # CIC, seeds 42-51
k3 = json.load(open(f"{R}/exp19_controls_k3_edge-iiot/results.json", encoding="utf-8"))        # Edge, seeds 42-44
mech = json.load(open(f"{R}/exp19_controls_mech_ciciot2023/results.json", encoding="utf-8"))   # CIC, seeds 42-44

COND = ["A_Standard", "D_Clip_Only", "C_Noise_Only", "B_DP_SGD"]
NAME = {"A_Standard": "A standard", "D_Clip_Only": "D clip only",
        "C_Noise_Only": "C noise only", "B_DP_SGD": "B clip$+$noise"}
CIC_CLASSES = ["Benign", "DDoS", "DoS", "Recon", "Web", "BruteForce", "Spoofing", "Mirai"]
CIC_TRUE = [0.023, 0.723, 0.172, 0.015, 0.0, 0.0, 0.011, 0.055]        # log line of every run
EDGE_CLASSES = ["Normal", "DDoS UDP", "DDoS ICMP", "DDoS HTTP", "DDoS TCP", "SQL injection",
                "Vulnerability scanner", "Password", "Uploading", "Backdoor", "Port scanning",
                "XSS", "Ransomware", "Fingerprinting", "MITM"]
EDGE_TRUE = [0.728, 0.055, 0.052, 0.023, 0.022, 0.022, 0.023, 0.023, 0.017, 0.011, 0.01, 0.007,
             0.005, 0.0, 0.001]


def raw(d, c, k):
    return np.array(d["conditions"][c][k]["raw"], dtype=float)


def r(x, nd):
    """Round half-up after removing float noise (0.0555 stored as 0.05549999... -> 0.056)."""
    return str(Decimal(repr(round(float(x), 10))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def pm(d, c, k, nd=3):
    x = raw(d, c, k)
    return f"${r(x.mean(), nd)}\\pm{r(x.std(), nd)}$"


def mean(d, c, k, nd=3):
    return f"${r(raw(d, c, k).mean(), nd)}$"


def share_row(values, logged_true=False):
    """Shares in % with one decimal (half-up); a non-zero share below 0.05% is shown as <0.1.
    The logged true shares are rounded to three decimals, so a logged 0.0 there means
    'below 0.05%', not zero."""
    out = []
    for v in values:
        v = Fraction(v) if not isinstance(v, Fraction) else v
        if 0 < v < Fraction(5, 10000) or (logged_true and v == 0):
            out.append("$<\\!0.1$")
        else:
            pct = Decimal(v.numerator * 100) / Decimal(v.denominator)
            out.append(str(pct.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)))
    return out


def shares(d, c, key):
    """Exact pooled shares from the per-run counts (the stored mean_share is rounded to four
    decimals, which can move a one-decimal percentage by 0.1)."""
    runs = d["conditions"][c][key]["counts_per_run"]
    tot = sum(sum(run) for run in runs)
    return [Fraction(sum(run[i] for run in runs), tot) for i in range(len(runs[0]))]


# ------------------------------------------------------------------ Table 5
def table5():
    L = []
    L.append(r"\multicolumn{11}{l}{\emph{CICIoT2023}} \\")
    for c in COND:
        pb = raw(k10, c, "share_pred_target")
        pb_s = f"${r(np.median(pb), 3)}^{{\\ast}}$" if c == "B_DP_SGD" else f"${r(pb.mean(), 3)}$"
        ta = raw(k10, c, "queries_true_attack_share").mean()
        cells = [NAME[c], pm(k10, c, "f1_macro"), pb_s, pm(k10, c, "cf_validity"),
                 mean(k10, c, "cf_validity_any"),
                 f"${r(ta, 2)}$ (${r(raw(k10, c, 'cf_validity_true_attack_subset').mean(), 3)}$)",
                 pm(core, c, "random_matched_flip_rate"), pm(k10, c, "discrete_validity"),
                 pm(k10, c, "d_lin_target_median", 2), pm(core, c, "deepfool_l2_median", 2),
                 f"${r(raw(k10, c, 'margin_grad_norm').mean(), 2)}$"]
        L.append(" & ".join(cells) + r" \\")
    L.append(r"\midrule")
    L.append(r"\multicolumn{11}{l}{\emph{Edge-IIoTset (benign class: Normal)}} \\")
    for c in COND:
        ta = raw(k3, c, "queries_true_attack_share").mean()
        sh = raw(k3, c, "margin_grad_norm").mean()
        cells = [NAME[c], pm(k3, c, "f1_macro"), mean(k3, c, "share_pred_target"),
                 pm(k3, c, "cf_validity"), mean(k3, c, "cf_validity_any"),
                 f"${r(ta, 2)}$ (${r(raw(k3, c, 'cf_validity_true_attack_subset').mean(), 3)}$)",
                 "--", "--", pm(k3, c, "d_lin_target_median", 2), pm(k3, c, "deepfool_l2_median", 2),
                 f"${r(sh, 1)}$" if sh >= 10 else f"${r(sh, 2)}$"]
        L.append(" & ".join(cells) + r" \\")
    return "\n".join(L)


# ------------------------------------------------------------------ Table 6
def table6():
    L = []
    for c in COND:
        cv = mech["curves"][c]
        sd = raw(mech, c, "sparsedf_l2_median"); reach = raw(mech, c, "sparsedf_reach").mean()
        cells = [NAME[c], mean(core, c, "cf_validity"), pm(mech, c, "variant_cap_only_frac"),
                 pm(mech, c, "variant_dist_only_frac"), pm(mech, c, "variant_none_frac"),
                 f"${r(cv['300'][0], 3)}$", f"${r(cv['1000'][0], 3)}$",
                 f"${r(sd.mean(), 2)}\\pm{r(sd.std(), 2)}$ (${r(reach, 2)}$)"]
        L.append(" & ".join(cells) + r" \\")
    return "\n".join(L)


# ------------------------------------------------------------------ Appendix D
def hist_cic():
    L = [r"\multicolumn{9}{l}{\emph{Test-set predictions}} \\",
         "True share & " + " & ".join(share_row(CIC_TRUE, logged_true=True)) + r" \\"]
    for c in COND:
        L.append(NAME[c] + " & " + " & ".join(share_row(shares(k10, c, "pred_hist"))) + r" \\")
    for key, title in (("query_top1_hist", "Queries: top-1 class"), ("query_top2_hist", "Queries: top-2 (runner-up) class")):
        L.append(r"\midrule")
        L.append(r"\multicolumn{9}{l}{\emph{" + title + r"}} \\")
        for c in COND:
            L.append(NAME[c] + " & " + " & ".join(share_row(shares(k10, c, key))) + r" \\")
    return "\n".join(L)


def hist_edge():
    L = []
    for i, cls in enumerate(EDGE_CLASSES):
        cells = [cls, share_row([EDGE_TRUE[i]], logged_true=True)[0]]
        for key in ("pred_hist", "query_top1_hist", "query_top2_hist"):
            for c in COND:
                cells.append(share_row([shares(k3, c, key)[i]])[0])
        L.append(" & ".join(cells) + r" \\")
    return "\n".join(L)


if __name__ == "__main__":
    for name, fn in (("TABLE5", table5), ("TABLE6", table6), ("HIST_CIC", hist_cic), ("HIST_EDGE", hist_edge)):
        print(f"%%%%%% {name}")
        print(fn())
