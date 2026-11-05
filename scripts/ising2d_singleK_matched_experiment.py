"""Matched single-layer Ising control: discard the independent K1=0 layer."""
import json
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from recurrent_rg.data.ensembles import load_ensembles,EnsembleDataset
from recurrent_rg.config import RunConfig
from recurrent_rg.runner import run_training
from recurrent_rg.checkpoints import load_run
from recurrent_rg.evaluation import collect_trajectories,closure_metrics
from recurrent_rg.plotting.reports import plot_diagnostics

SOURCE=Path('data_cache/dual_ising2d_K1zero_L16_seed91')
CACHE=Path('data_cache/ising2d_singleK_matched_L16_seed91')
RUN=Path('results/ising2d_singleK_g2_joint_2400_seed91')
KC=np.log(1+np.sqrt(2))/2


def prepare():
    CACHE.mkdir(exist_ok=False)
    tr,te=load_ensembles(SOURCE/'ensembles.pt')
    def extract(ds):
        assert torch.all(ds.couplings[:,0]==0)
        # State code = 2*(sigma+1)/2 + (tau+1)/2.
        spins=(2*(ds.configurations.argmax(-1)%2)-1).to(torch.int8).unsqueeze(-1)
        return EnsembleDataset(spins,ds.couplings[:,1:]).validate()
    a,b=extract(tr),extract(te)
    torch.save(dict(schema_version=1,train=asdict(a),test=asdict(b)),CACHE/'ensembles.pt')
    load_ensembles(CACHE/'ensembles.pt')
    raw=np.load(SOURCE/'audit.npz');states=(2*(raw['states']%2)-1).astype(np.int8)
    np.savez_compressed(CACHE/'audit.npz',states=states,K=raw['K2'])
    meta=dict(source=str(SOURCE),operation='Exact tau marginal: 2*(joint_state % 2)-1; no resampling, same train/test groups and independent audit.',
        train_ensembles=len(a.configurations),test_ensembles=len(b.configurations),samples=128,L=16,field=0)
    (CACHE/'sampling.json').write_text(json.dumps(meta,indent=2)+'\n');print('DATA READY',meta,flush=True)


def report():
    run=load_run(RUN);m=run.model.eval();tr,te=load_ensembles(CACHE/'ensembles.pt')
    out=RUN/'report';out.mkdir(exist_ok=True)
    train=collect_trajectories(m,tr.configurations,samples=128,max_rg_steps=2,chunk_size=8)
    test=collect_trajectories(m,te.configurations,samples=128,max_rg_steps=2,chunk_size=8)
    plot_diagnostics(m,test,te.couplings.numpy(),['K'],out,training_inputs=train['direct'][:,:-1])
    labels=torch.cat([tr.couplings,te.couplings]).numpy().ravel();d=np.concatenate([train['direct'],test['direct']])
    grid=np.unique(labels);means=np.stack([d[labels==k].mean(0) for k in grid])
    fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for j,ax in enumerate(axs):
        for k in range(3):ax.plot(grid,means[:,k,j],label=f'Q at k={k}')
        ax.axvline(KC,c='black',ls='--',label='Kc');ax.set(xlabel='K',ylabel=f'g{j+1}');ax.legend();ax.grid(alpha=.2)
    fig.savefig(out/'K_vs_g.png',dpi=180)
    fig,ax=plt.subplots(figsize=(8,6),layout='constrained');v=means[:,1]-means[:,0]
    q=ax.quiver(*means[:,0].T,*v.T,grid,cmap='viridis',angles='xy',scale_units='xy',scale=1,minlength=0)
    ax.scatter(*means[:,0].T,c=grid,cmap='viridis',s=10);fig.colorbar(q,ax=ax,label='K')
    ax.set(xlabel='g1',ylabel='g2',title='Direct Q(D) -> Q(CD); actual displacements');ax.grid(alpha=.2)
    fig.savefig(out/'direct_Q_field.png',dpi=180)
    audit=np.load(CACHE/'audit.npz');x=torch.from_numpy(audit['states']).reshape(-1,128,16,16,1)
    t=collect_trajectories(m,x,samples=128,max_rg_steps=2,chunk_size=8,sampling_seed=22345)
    ad=t['direct'].reshape(5,4,2,3,2).mean(2);ar=t['rollout'].reshape(5,4,2,3,2).mean(2)
    fig,ax=plt.subplots(figsize=(9,6),layout='constrained')
    for i,value in enumerate(audit['K']):
        mean=ad[i].mean(0);r=ar[i].mean(0);se=ad[i].std(0,ddof=1)/2
        line=ax.errorbar(*mean.T,xerr=se[:,0],yerr=se[:,1],fmt='o-',capsize=3,label=f'K={value:.4f}: Q')[0]
        ax.plot(*r.T,'x--',c=line.get_color())
        for k,z in enumerate(mean):ax.annotate(str(k),z,xytext=(4,4),textcoords='offset points',fontsize=8)
    ax.set(xlabel='g1',ylabel='g2',title='Solid: direct Q; dashed: R rollout');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(out/'critical_trajectories.png',dpi=180)
    hist=json.loads((RUN/'history_progress.json').read_text());fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    for name in ['train','test']:ax.semilogy([h['epoch'] for h in hist],[h[name+'_total'] for h in hist],label=name)
    ax.set(xlabel='Training update',ylabel='Total loss');ax.legend();ax.grid(alpha=.2);fig.savefig(out/'loss.png',dpi=180)
    np.savez_compressed(out/'trajectories.npz',train_direct=train['direct'],test_direct=test['direct'],test_rollout=test['rollout'],couplings=grid,mean_direct=means,audit_direct=ad,audit_rollout=ar,audit_K=audit['K'])
    mask=np.repeat(np.isclose(audit['K'],KC)|np.isclose(abs(audit['K']-KC),.03),8)
    info=dict(test_closure=closure_metrics(test,reference_direct=train['direct']),critical_closure=closure_metrics({key:v[mask] for key,v in t.items()},reference_direct=train['direct']),
        critical_direct=ad[2].mean(0).tolist(),critical_rollout=ar[2].mean(0).tolist(),
        covariance_eigenvalues=[np.linalg.eigvalsh(np.cov(d[:,k].T,bias=True)).tolist() for k in range(3)],
        parameters={name:sum(p.numel() for p in module.parameters()) for name,module in [('Q',m.theory_encoder),('C',m.rg_kernel),('R',m.theory_rg)]},
        note='Exact single-layer marginal of previous K1=0 experiment; no random auxiliary layer. Binary input/output changes parameter count and categorical vs binary geometry statistics. QCR jointly trained from scratch; one seed; off-manifold R arrows extrapolate.')
    (out/'benchmark.json').write_text(json.dumps(info,indent=2)+'\n');print('REPORT COMPLETE',json.dumps(info),flush=True)


def main():
    torch.set_num_threads(2);prepare()
    run_training(RunConfig.load('configs/ising2d_singleK_g2_joint_seed91.json'),RUN)
    report()


if __name__=='__main__':main()
