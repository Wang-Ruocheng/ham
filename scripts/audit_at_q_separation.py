"""Frozen-Q phase-region distinguishability; labels used only for diagnostics."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from recurrent_rg.checkpoints import load_run
from recurrent_rg.data.ensembles import load_ensembles
from recurrent_rg.evaluation import collect_trajectories


def phase_labels(K):
    # Conservative diagnostic regions; not exact thermodynamic boundaries.
    y=np.full(len(K),-1)
    y[K<=.165]=0
    y[(K>=.200)&(K<=.214)]=1
    y[K>=.25]=2
    return y


def main():
    torch.set_num_threads(2)
    folder=Path('results/ashkin_teller2d_g2_fixed_decimation_1200_seed91')
    run=load_run(folder);m=run.model;m.eval();tr,te=load_ensembles(run.config['data']['cache'])
    arrays=[]
    for ds in (tr,te):
        arrays.append(collect_trajectories(m,ds.configurations,samples=128,max_rg_steps=2)['direct'])
    kt,kv=tr.couplings[:,0].numpy(),te.couplings[:,0].numpy()
    yt,yv=phase_labels(kt),phase_labels(kv)
    out=folder/'q_separation';out.mkdir(exist_ok=True)
    report=dict(protocol='Frozen model. Train-only PCA axis and phase thresholds; held-out coupling groups for evaluation. Phase regions are finite-size diagnostic assignments, not exact phase labels.',
                regions=dict(disordered='K<=.165',partial='.200<=K<=.214',ordered='K>=.25'),scales=[])
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    for k,ax in enumerate(axes):
        a,b=arrays[0][:,k],arrays[1][:,k]
        center=a.mean(0);_,_,vh=np.linalg.svd(a-center,full_matrices=False);axis=vh[0]
        if np.corrcoef((a-center)@axis,kt)[0,1]<0:axis=-axis
        st,sv=(a-center)@axis,(b-center)@axis
        # Two thresholds, trained only on confidently assigned training regions.
        unique=np.unique(st[yt>=0]);candidates=(unique[:-1]+unique[1:])/2
        best=None
        for i,t1 in enumerate(candidates):
            for t2 in candidates[i+1:]:
                pred=(st>t1).astype(int)+(st>t2).astype(int)
                score=np.mean([np.mean(pred[yt==p]==p) for p in range(3)])
                if best is None or score>best[0]:best=(score,t1,t2)
        _,t1,t2=best;pred=(sv>t1).astype(int)+(sv>t2).astype(int)
        confusion=np.array([[np.sum((yv==i)&(pred==j)) for j in range(3)] for i in range(3)])
        # Monotonic calibration of frozen g to K, train means only; diagnostic, not Q training.
        knots=np.unique(kt);smeans=np.array([st[kt==v].mean() for v in knots]);order=np.argsort(smeans)
        khat=np.interp(sv,smeans[order],knots[order]);near=(kv>=.15)&(kv<=.26)
        rec=dict(k=k,axis=axis.tolist(),train_balanced_accuracy=best[0],thresholds=[t1,t2],
                 test_confusion=confusion.tolist(),test_accuracy=float(np.mean(pred[yv>=0]==yv[yv>=0])),
                 test_counts=[int(np.sum(yv==i)) for i in range(3)],
                 spearman_test=float(spearmanr(sv,kv).statistic),
                 near_transition_K_MAE=float(np.abs(khat[near]-kv[near]).mean()),
                 calibration_note='Interpolation, not supervised neural training; near-transition MAE uses all held-out K in [.15,.26].')
        K=np.r_[kt,kv];g=np.concatenate([a,b]);s=np.r_[st,sv]
        probes=[]
        for target in (.153,.201,.274):
            v=np.unique(K)[np.argmin(abs(np.unique(K)-target))];values=g[K==v];scores=s[K==v]
            probes.append(dict(K=float(v),mean_g=values.mean(0).tolist(),std_score=float(scores.std(ddof=1)),score_range=[float(scores.min()),float(scores.max())],mean_score=float(scores.mean())))
        rec['representative_ensembles']=probes
        rec['adjacent_probe_gap_over_pooled_sd']=[float(abs(probes[j+1]['mean_score']-probes[j]['mean_score'])/np.sqrt((probes[j+1]['std_score']**2+probes[j]['std_score']**2)/2)) for j in range(2)]
        report['scales'].append(rec)
        for p,color,name in [(0,'tab:blue','Disordered region'),(1,'tab:orange','Partial-order region'),(2,'tab:green','Ordered region')]:
            mask=yt==p;ax.scatter(kt[mask],st[mask],color=color,s=15,alpha=.5)
            mask=yv==p;ax.scatter(kv[mask],sv[mask],color=color,s=45,marker='x',label=name)
        ax.scatter(kv[yv<0],sv[yv<0],color='.6',marker='x',s=20,label='Boundary regions excluded')
        ax.axhline(t1,c='black',ls=':',lw=1);ax.axhline(t2,c='black',ls=':',lw=1)
        ax.set(xlabel='K',ylabel='Frozen g projected on training PC1',title=f'k={k}: held-out region accuracy {rec["test_accuracy"]:.0%}')
        ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Fixed C, dim_g=2: Q distinguishes three representative phase regions\nDots: training; crosses: held-out couplings. Heuristic regions, not thermodynamic classification.')
    fig.savefig(out/'phase_separation.png',dpi=170)
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
