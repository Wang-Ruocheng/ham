"""Post-training physical-grid and latent-phase diagnostics; labels only for audit."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from recurrent_rg.runner import evaluate_run
from recurrent_rg.checkpoints import load_run
from recurrent_rg.data.ensembles import load_ensembles
from recurrent_rg.evaluation import collect_trajectories,closure_metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    out=a.run/'report';evaluate_run(a.run,output=out)
    run=load_run(a.run);tr,te=load_ensembles(run.config['data']['cache'])
    samples=run.config['evaluation']['samples']
    train=collect_trajectories(run.model,tr.configurations,samples=samples,max_rg_steps=2)
    test=collect_trajectories(run.model,te.configurations,samples=samples,max_rg_steps=2)
    labels=torch.cat([tr.couplings,te.couplings]).numpy();d=np.concatenate([train['direct'],test['direct']])
    cells=np.unique(labels,axis=0);means=np.stack([d[(labels==v).all(1)].mean(0) for v in cells])
    kc=np.log(1+np.sqrt(2))/2;phase=(cells[:,0]>kc).astype(int)+2*(cells[:,1]>kc).astype(int)
    fig,axs=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for k in range(3):
        for c in range(2):
            ax=axs[c,k];z=ax.scatter(cells[:,0],cells[:,1],c=means[:,k,c],marker='s',s=160,cmap='viridis')
            ax.axvline(kc,c='red',ls='--',lw=1);ax.axhline(kc,c='red',ls='--',lw=1)
            ax.set(xlabel='K1',ylabel='K2',title=f'g{c+1}, RG step {k}');fig.colorbar(z,ax=ax)
    fig.savefig(out/'g_on_K1K2.png',dpi=170)
    fig,axs=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    for k,ax in enumerate(axs):
        for i,name in enumerate(['both disordered','only layer 1 ordered','only layer 2 ordered','both ordered']):
            selected=phase==i;ax.scatter(*means[selected,k].T,s=18,label=name)
        ax.set(xlabel='g1',ylabel='g2',title=f'RG step {k}; exact phase labels for audit only');ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.savefig(out/'latent_phases.png',dpi=170)
    history=json.loads((a.run/'history_progress.json').read_text());fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    for split in ['train','test']:ax.plot([r['epoch'] for r in history],[r[split+'_total'] for r in history],label=split)
    ax.set_yscale('symlog',linthresh=.01)
    ax.set(xlabel='training step',ylabel='total loss');ax.legend();ax.grid(alpha=.2);fig.savefig(out/'loss.png',dpi=170)
    report=dict(critical_K1=kc,critical_K2=kc,heldout_closure=closure_metrics(test,reference_direct=train['direct']),
                covariance_eigenvalues_by_scale=[np.linalg.eigvalsh(np.cov(d[:,k].T,bias=True)).tolist() for k in range(3)],
                caveat='Thermodynamic phase labels annotate finite L data; no labels enter training. Latent separation alone does not establish correct RG basin boundaries.')
    (out/'benchmark.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
