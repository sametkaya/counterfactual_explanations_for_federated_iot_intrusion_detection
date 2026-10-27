"""Usage: python3 journal_paper/stats_tables.py [experiments_journal]
Paired differences with 95% CIs and Holm-adjusted p-values for the sweep and factorial tables."""
import json, numpy as np
from scipy import stats
import os, sys
R = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("RESULTS_DIR", "experiments_journal")   # run from repo root
SIG={"10.0":0.48,"5.0":0.97,"2.0":2.42,"1.0":4.84,"0.5":9.69}
def ci_paired(a,b):
    d=np.asarray(a)-np.asarray(b); n=len(d); m=d.mean(); se=d.std(ddof=1)/np.sqrt(n); t=stats.t.ppf(0.975,n-1)
    return m, m-t*se, m+t*se
def holm(ps):
    ps=np.asarray(ps); order=np.argsort(ps); adj=np.empty_like(ps); m=len(ps); running=0
    for rank,i in enumerate(order):
        val=min(1.0,(m-rank)*ps[i]); running=max(running,val); adj[i]=running
    return adj
for ds in ("ciciot2023","edge-iiot"):
    d=json.load(open(f"{R}/exp12b_epsilon_n10_{ds}/results.json"))
    base=d["epsilons"]["inf"]; print(f"\n=== sweep {ds} (n={d['n_runs']}) ===")
    rows=[]; pv=[]
    for eps in ("10.0","5.0","2.0","1.0","0.5"):
        e=d["epsilons"][eps]
        mv,lo,hi=ci_paired(e["raw_val"],base["raw_val"]); ma,alo,ahi=ci_paired(e["raw_acc"],base["raw_acc"])
        w=stats.wilcoxon(e["raw_val"],base["raw_val"]); pv.append(w.pvalue)
        rows.append((eps,SIG[eps],e["acc_mean"],e["acc_std"],e["f1_mean"],e["val_mean"],e["val_std"],mv,lo,hi,ma,alo,ahi,w.pvalue))
    adj=holm(pv)
    print("base: acc %.3f±%.3f F1 %.3f val %.3f±%.3f"%(base["acc_mean"],base["acc_std"],base["f1_mean"],base["val_mean"],base["val_std"]))
    for r,pa in zip(rows,adj):
        print(f"sigma={r[1]:5.2f} (eps={r[0]:>4}) acc={r[2]:.3f}±{r[3]:.3f} F1={r[4]:.3f} val={r[5]:.3f}±{r[6]:.3f} | Δval={100*r[7]:+.1f}pp [{100*r[8]:+.1f},{100*r[9]:+.1f}] Δacc={100*r[10]:+.1f}pp [{100*r[11]:+.1f},{100*r[12]:+.1f}] p={r[13]:.3f} Holm={pa:.3f}")
# factorial contrasts
for ds in ("ciciot2023","edge-iiot"):
    d=json.load(open(f"{R}/exp14_mechanism_{ds}/results.json"))["conditions"]
    print(f"\n=== factorial {ds} (n={len(d['A_Standard']['cf_validity']['raw'])}) ===")
    A=d["A_Standard"]
    for c in ("D_Clip_Only","C_Noise_Only","B_DP_SGD"):
        X=d[c]; out=[]
        for k in ("cf_validity","accuracy","margin_grad_norm","hessian_trace","margin_logit","ece"):
            m,lo,hi=ci_paired(X[k]["raw"],A[k]["raw"]); 
            n=len(X[k]["raw"]); w=stats.wilcoxon(X[k]["raw"],A[k]["raw"]).pvalue if n>=5 else float('nan')
            mw=stats.mannwhitneyu(X[k]["raw"],A[k]["raw"],alternative="two-sided").pvalue
            out.append(f"{k}: Δ={m:+.3f} [{lo:+.3f},{hi:+.3f}] W={w:.3f} MW={mw:.3f}")
        # d_lin per run
        dl=lambda Z: np.array(Z["margin_logit"]["raw"])/np.array(Z["margin_grad_norm"]["raw"])
        m,lo,hi=ci_paired(dl(X),dl(A)); w=stats.wilcoxon(dl(X),dl(A)).pvalue; mw=stats.mannwhitneyu(dl(X),dl(A),alternative="two-sided").pvalue
        out.append(f"d_lin(per-run): {dl(A).mean():.3f}->{dl(X).mean():.3f} Δ={m:+.3f} [{lo:+.3f},{hi:+.3f}] rel={100*m/dl(A).mean():+.0f}% W={w:.3f} MW={mw:.3f}")
        print(f"  {c} vs A:"); [print("     ",o) for o in out]
# exp18
d=json.load(open(f"{R}/exp18_federated_ciciot2023/results.json"))["conditions"]
A,Cc=d["A_Standard"],d["C_Noise_Only"]; print("\n=== exp18 FedAvg C vs A (n=3) ===")
for k in ("cf_validity","accuracy","f1_macro","margin_grad_norm","hessian_trace","margin_logit","ece"):
    m,lo,hi=ci_paired(Cc[k]["raw"],A[k]["raw"]); print(f"  {k}: {A[k]['mean']:.3f}->{Cc[k]['mean']:.3f} Δ={m:+.3f} [{lo:+.3f},{hi:+.3f}]")
dl=lambda Z: np.array(Z["margin_logit"]["raw"])/np.array(Z["margin_grad_norm"]["raw"])
m,lo,hi=ci_paired(dl(Cc),dl(A)); print(f"  d_lin: {dl(A).mean():.3f}->{dl(Cc).mean():.3f} Δ={m:+.3f} [{lo:+.3f},{hi:+.3f}]")
# exp13b?
d=json.load(open(f"{R}/exp13b_paradox_n10_ciciot2023/results.json")); print("\nexp13b keys:", list(d.keys()))
