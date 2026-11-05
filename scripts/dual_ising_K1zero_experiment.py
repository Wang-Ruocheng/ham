"""Sample K1=0, train QCR from scratch, then generate axis-specific diagnostics."""
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from recurrent_rg.data.dual_ising import sample_dual_ising_couplings
from recurrent_rg.data.ensembles import EnsembleDataset,grouped_split,load_ensembles
from recurrent_rg.config import RunConfig
from recurrent_rg.runner import run_training
from recurrent_rg.checkpoints import load_run
from recurrent_rg.evaluation import collect_trajectories,closure_metrics
from recurrent_rg.plotting.reports import plot_diagnostics
from train_dual_ising_dense_joint import install_dense_coarse_vjp

CACHE=Path('data_cache/dual_ising2d_K1zero_L16_seed91')
RUN=Path('results/dual_ising2d_K1zero_g2_joint_2400_seed91')
KC=np.log(1+np.sqrt(2))/2


def generate():
    CACHE.mkdir(exist_ok=False)
    offsets=np.linspace(-.08,.08,33)
    offsets=offsets[~np.isclose(offsets[:,None],[-.03,0,.03]).any(1)]
    grid=np.unique(np.r_[np.linspace(0,1,101),KC+offsets])
    xs=[];ys=[];checks=[]
    for i,k in enumerate(grid):
        for rep in range(6):
            x=sample_dual_ising_couplings(0.,k,16,128,1200000+i*10+rep,burn=2048,spacing=16,ordered=bool(rep%2))
            xs.append(torch.nn.functional.one_hot(torch.from_numpy(x.astype(np.int64)),4).to(torch.uint8));ys.append([0.,k])
            spin=2*(x//2)-1
            checks.append(float((spin*np.roll(spin,1,axis=1)).mean()))
        if (i+1)%10==0 or i+1==len(grid):
            progress=dict(stage='sampling',completed_cells=i+1,total_cells=len(grid))
            (CACHE/'progress.json').write_text(json.dumps(progress)+'\n');print(progress,flush=True)
    tr,te=grouped_split(EnsembleDataset(torch.stack(xs),torch.tensor(ys,dtype=torch.double)))
    torch.save(dict(schema_version=1,train=asdict(tr),test=asdict(te)),CACHE/'ensembles.pt');load_ensembles(CACHE/'ensembles.pt')
    audit_K=np.array([.2,KC-.03,KC,KC+.03,.8])
    audit=np.stack([np.stack([sample_dual_ising_couplings(0.,k,16,256,1400000+i*10+r,burn=2048,spacing=16,ordered=bool(r%2)) for r in range(4)]) for i,k in enumerate(audit_K)])
    np.savez_compressed(CACHE/'audit.npz',states=audit,K2=audit_K)
    assert not np.isclose(grid[:,None],audit_K[1:4]).any()
    info=dict(K1=0.,K2_grid=grid.tolist(),L=16,samples=128,replicas=6,burn=2048,spacing=16,
              train_ensembles=len(tr.configurations),test_ensembles=len(te.configurations),
              train_cells=len(torch.unique(tr.couplings,dim=0)),test_cells=len(torch.unique(te.couplings,dim=0)),
              layer1_nearest_neighbor_correlation_mean=float(np.mean(checks)),
              audit_note='Independent chains; Kc and Kc+-0.03 excluded from all training/grid-test cells. Endpoint probes .2 and .8 may match grid couplings, but not sample configurations.')
    (CACHE/'sampling.json').write_text(json.dumps(info,indent=2)+'\n');print('SAMPLING COMPLETE',info['train_ensembles'],info['test_ensembles'],flush=True)


def report():
    run=load_run(RUN);m=run.model.eval();tr,te=load_ensembles(CACHE/'ensembles.pt')
    out=RUN/'report';out.mkdir(exist_ok=True)
    a=collect_trajectories(m,tr.configurations,samples=128,max_rg_steps=2,chunk_size=4)
    b=collect_trajectories(m,te.configurations,samples=128,max_rg_steps=2,chunk_size=4)
    plot_diagnostics(m,b,te.couplings[:,1:].numpy(),['K2'],out,training_inputs=a['direct'][:,:-1])
    labels=torch.cat([tr.couplings,te.couplings])[:,1].numpy();direct=np.concatenate([a['direct'],b['direct']])
    grid=np.unique(labels);means=np.stack([direct[labels==k].mean(0) for k in grid])
    fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for j,ax in enumerate(axs):
        for k in range(3):ax.plot(grid,means[:,k,j],label=f'Q at k={k}')
        ax.axvline(KC,c='black',ls='--',label='Kc');ax.set(xlabel='K2 (K1=0)',ylabel=f'g{j+1}');ax.legend();ax.grid(alpha=.2)
    fig.savefig(out/'K2_vs_g.png',dpi=180)
    fig,ax=plt.subplots(figsize=(8,6),layout='constrained')
    disp=means[:,1]-means[:,0]
    q=ax.quiver(*means[:,0].T,*disp.T,grid,angles='xy',scale_units='xy',scale=1,minlength=0,cmap='viridis')
    ax.scatter(*means[:,0].T,c=grid,cmap='viridis',s=12);ax.set(xlabel='g1',ylabel='g2',title='Direct Q(D) -> Q(CD); actual displacement, no R')
    fig.colorbar(q,ax=ax,label='K2');ax.grid(alpha=.2);fig.savefig(out/'direct_Q_field.png',dpi=180)
    audit=np.load(CACHE/'audit.npz');x=torch.nn.functional.one_hot(torch.from_numpy(audit['states'].astype(np.int64)).reshape(-1,128,16,16),4).to(torch.uint8)
    t=collect_trajectories(m,x,samples=128,max_rg_steps=2,chunk_size=4,sampling_seed=22345)
    ad=t['direct'].reshape(5,4,2,3,2).mean(2);ar=t['rollout'].reshape(5,4,2,3,2).mean(2)
    fig,ax=plt.subplots(figsize=(9,6),layout='constrained')
    for i,value in enumerate(audit['K2']):
        d=ad[i].mean(0);r=ar[i].mean(0);se=ad[i].std(0,ddof=1)/2
        line=ax.errorbar(*d.T,xerr=se[:,0],yerr=se[:,1],fmt='o-',capsize=3,label=f'K2={value:.4f}: Q')[0]
        ax.plot(*r.T,'x--',c=line.get_color())
        for k,z in enumerate(d):ax.annotate(str(k),z,xytext=(4,4),textcoords='offset points',fontsize=8)
    ax.set(xlabel='g1',ylabel='g2',title='Solid: direct Q; dashed: R rollout; labels: scale k');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(out/'critical_trajectories.png',dpi=180)
    hist=json.loads((RUN/'history_progress.json').read_text());fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    for name in ['train','test']:ax.semilogy([h['epoch'] for h in hist],[h[name+'_total'] for h in hist],label=name)
    ax.set(xlabel='Training update',ylabel='Total loss');ax.legend();ax.grid(alpha=.2);fig.savefig(out/'loss.png',dpi=180)
    np.savez_compressed(out/'trajectories.npz',train_direct=a['direct'],test_direct=b['direct'],test_rollout=b['rollout'],couplings=grid,mean_direct=means,audit_direct=ad,audit_rollout=ar,audit_K2=audit['K2'])
    critical_mask=np.repeat(np.isclose(audit['K2'],KC) | np.isclose(abs(audit['K2']-KC),.03),8)
    critical_t={key:v[critical_mask] for key,v in t.items()}
    info=dict(test_closure=closure_metrics(b,reference_direct=a['direct']),critical_closure=closure_metrics(critical_t,reference_direct=a['direct']),
              critical_direct=ad[2].mean(0).tolist(),critical_rollout=ar[2].mean(0).tolist(),
              covariance_eigenvalues=[np.linalg.eigvalsh(np.cov(direct[:,k].T,bias=True)).tolist() for k in range(3)],
              note='One physical varying coupling, two latent dimensions; low-rank g alone is not a failure. C is trained jointly and stochastic. Off-data R arrows are extrapolation.')
    (out/'benchmark.json').write_text(json.dumps(info,indent=2)+'\n');print('REPORT COMPLETE',json.dumps(info),flush=True)


def main():
    torch.set_num_threads(2)
    generate()
    install_dense_coarse_vjp()
    run_training(RunConfig.load('configs/dual_ising2d_K1zero_g2_joint_seed91.json'),RUN)
    report()


if __name__=='__main__':main()
