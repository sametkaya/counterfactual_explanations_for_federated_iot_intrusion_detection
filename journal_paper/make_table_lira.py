"""Tables tab:lira and tab:canary: fixed-variance offline LiRA re-scored from the raw phi/mu/sigma stored
by exp15 (results['attacks'][tag]['raw'] / ['raw_canary']). Usage: python3 journal_paper/make_table_lira.py [experiments_journal]"""
import sys, os
R = sys.argv[1] if len(sys.argv) > 1 else "experiments_journal"
import json, numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, roc_curve
def tpr_at(y, s, f):
    fpr,tpr,_=roc_curve(y,s); i=np.searchsorted(fpr,f,side="right")-1; i=max(0,min(i,len(tpr)-1)); return tpr[i]
def stats(y,s): return (roc_auc_score(y,s), tpr_at(y,s,0.001), tpr_at(y,s,0.01), tpr_at(y,s,0.05))
def rescore(raw, floor=1e-6):
    phi=np.array(raw["phi_target"]); mu=np.array(raw["mu_out"]); sd=np.maximum(np.array(raw["sigma_out"]), floor); y=np.array(raw["membership"])
    z=(phi-mu)/sd
    zg=(phi-mu)/np.median(sd)          # global (fixed) variance variant
    return {"per": stats(y,z), "glob": stats(y,zg), "ties": float(np.mean(norm.cdf(z)>=1.0)), "sd0": float(np.mean(np.array(raw["sigma_out"])<=0.0005))}
def fmt(t): return f"AUC {t[0]:.3f} TPR {t[1]*100:.1f}/{t[2]*100:.1f}/{t[3]*100:.1f}"
rows={}
for d in ["exp15_lira_mia_v2_ciciot2023","exp15_lira_mia_v2_edge-iiot","exp15_lira_mia_mem3_ciciot2023","exp15_lira_mia_mem3_edge-iiot","exp15_lira_mia_canary_ciciot2023","exp15_lira_mia_canary_edge-iiot","exp15_lira_mia_canary_dedup_ciciot2023","exp15_lira_mia_canary_dedup_edge-iiot"]:
    r=json.load(open(os.path.join(R, d, "results.json")))
    print("==",d)
    for tag in ("NoDP","DP"):
        a=r["attacks"][tag]; t=a["target"]
        o=rescore(a["raw"])
        print(f"  {tag} audit   per-example: {fmt(o['per'])} | global-sigma: {fmt(o['glob'])} | ties {o['ties']:.3f} | train {t['train_acc']:.4f} test {t['test_acc']:.4f} gap {100*(t['train_acc']-t['test_acc']):+.1f} steps {t.get('steps')}")
        rows[(d,tag,"audit")]=(o,t)
        if "raw_canary" in a:
            o=rescore(a["raw_canary"]); cf=t.get("canary_fit")
            print(f"  {tag} canary  per-example: {fmt(o['per'])} | global-sigma: {fmt(o['glob'])} | ties {o['ties']:.3f} sd~0 {o['sd0']:.2f} | fit in {cf['in_fit_rate']:.3f} out {cf['out_fit_rate']:.3f}")
            rows[(d,tag,"canary")]=(o,t)
json.dump({f"{k[0]}|{k[1]}|{k[2]}": {"per":v[0]["per"],"glob":v[0]["glob"],"target":v[1]} for k,v in rows.items()}, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lira_rescored.json"), "w"), indent=1)
