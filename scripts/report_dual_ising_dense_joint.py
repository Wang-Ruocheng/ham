"""Physical and direct-vs-R audit of the from-scratch dense joint QCR run."""
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from recurrent_rg.checkpoints import load_run
from recurrent_rg.data.ensembles import load_ensembles
from recurrent_rg.evaluation import collect_trajectories,closure_metrics
from recurrent_rg.plotting.reports import plot_diagnostics


def main(root=Path('results/dual_ising2d_g2_dense_joint_fast_2400_seed91')):
    torch.set_num_threads(2)
    root=Path(root);out=root/'report';out.mkdir(exist_ok=True)
    run=load_run(root);m=run.model.eval();tr,te=load_ensembles(run.config['data']['cache'])
    train=collect_trajectories(m,tr.configurations,samples=128,max_rg_steps=2,chunk_size=4)
    test=collect_trajectories(m,te.configurations,samples=128,max_rg_steps=2,chunk_size=4)
    print('encoded train and test',flush=True)
    plot_diagnostics(m,test,te.couplings.numpy(),['K1','K2'],out,training_inputs=train['direct'][:,:-1])
    np.savez_compressed(out/'trajectories.npz',train_direct=train['direct'],test_direct=test['direct'],test_rollout=test['rollout'],train_couplings=tr.couplings.numpy(),test_couplings=te.couplings.numpy())
    kc=np.log(1+np.sqrt(2))/2
    labels=torch.cat([tr.couplings,te.couplings]).numpy();all_d=np.concatenate([train['direct'],test['direct']])
    cells=np.unique(labels,axis=0);means=np.stack([all_d[(labels==p).all(1)].mean(0) for p in cells]);phase=(cells[:,0]>kc).astype(int)+2*(cells[:,1]>kc)
    colors=['#1f77b4','#ff7f0e','#2ca02c','#d62728'];names=['both disordered','layer 1 ordered','layer 2 ordered','both ordered']
    fig,axs=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
    for k,ax in enumerate(axs):
        for j in range(4):ax.scatter(*means[phase==j,k].T,c=colors[j],s=12,label=names[j])
        ax.set(xlabel='g1',ylabel='g2',title=f'Direct Q at k={k}');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(out/'latent_phases.png',dpi=180)
    fig,axs=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for j in range(2):
        for k in range(3):
            ax=axs[j,k];sc=ax.scatter(*cells.T,c=means[:,k,j],s=20,cmap='viridis');ax.axvline(kc,c='r',ls='--');ax.axhline(kc,c='r',ls='--')
            ax.set(xlabel='K1',ylabel='K2',title=f'g{j+1}, k={k}');fig.colorbar(sc,ax=ax)
    fig.savefig(out/'g_on_K1K2.png',dpi=180)
    raw=np.load('results/dual_ising2d_g2_uniformK1K2_fixed_1200_seed91/critical_sampling_audit/sampled_states.npz')
    states=raw['states'];x=torch.nn.functional.one_hot(torch.from_numpy(states.astype(np.int64)).reshape(-1,128,16,16),4).to(torch.uint8)
    a=collect_trajectories(m,x,samples=128,max_rg_steps=2,chunk_size=4,sampling_seed=22345)
    ad=a['direct'].reshape(25,4,2,3,2).mean(2);ar=a['rollout'].reshape(25,4,2,3,2).mean(2)
    np.savez_compressed(out/'critical_audit.npz',direct=ad,rollout=ar,couplings=raw['couplings'])
    print('encoded independent critical audit',flush=True)
    center=np.flatnonzero(np.isclose(raw['couplings'],kc).all(1))[0]
    fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for axis,ax in enumerate(axs):
        for offset,color in zip([-.03,0,.03],['#0072b2','black','#d55e00']):
            p=np.array([kc,kc]);p[axis]+=offset;idx=np.flatnonzero(np.isclose(raw['couplings'],p).all(1))[0]
            d=ad[idx].mean(0);r=ar[idx].mean(0);se=ad[idx].std(0,ddof=1)/2
            ax.errorbar(*d.T,xerr=se[:,0],yerr=se[:,1],fmt='o-',c=color,capsize=2,label=f'Q, offset {offset:+.2f}')
            ax.plot(*r.T,'x--',c=color,label='R rollout' if offset==0 else None)
            for k,z in enumerate(d):ax.annotate(str(k),z,xytext=(3,4),textcoords='offset points',fontsize=7)
        ax.set(xlabel='g1',ylabel='g2',title=f'Vary K{axis+1}; other=Kc');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(out/'critical_trajectories.png',dpi=180)
    fig,ax=plt.subplots(figsize=(8,7),layout='constrained')
    for p,z in zip(phase,means):
        ax.plot(*z.T,c=colors[p],alpha=.45,lw=.7)
        for k,mark in enumerate(['o','s','^']):ax.scatter(*z[k],c=colors[p],s=9,marker=mark)
        for k in range(2):ax.annotate('',xy=z[k+1],xytext=z[k],arrowprops=dict(arrowstyle='->',color=colors[p],lw=.7,shrinkA=0,shrinkB=0))
    ax.plot(*ad[center].mean(0).T,'ko-',label='(Kc,Kc) independent sampling');ax.legend(fontsize=8)
    ax.set(xlabel='g1',ylabel='g2',title='Direct Q(C^k D); actual displacements, no R');ax.grid(alpha=.2)
    fig.savefig(out/'direct_Q_flow.png',dpi=180)
    history=json.loads((root/'history_progress.json').read_text());fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for split in ['train','test']:axs[0].semilogy([h['epoch'] for h in history],[h[split+'_total'] for h in history],label=split)
    weights={'anchor':2,'coarse_consistency':5,'separation':10,'commutator':5,'rollout':5,'kernel_noncollapse':2,'kernel_sign':1}
    weighted={}
    for key,w in weights.items():
        if 'test_'+key in history[-1]:
            vals=[w*h['test_'+key] for h in history];axs[1].semilogy([h['epoch'] for h in history],np.maximum(vals,1e-12),label=key);weighted[key]=vals[-1]
    for ax in axs:ax.set(xlabel='Training update',ylabel='Loss');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(out/'loss.png',dpi=180)
    patterns=torch.eye(4,dtype=torch.double).repeat(1,4)
    with torch.no_grad():prob=m.rg_kernel.field_head.logits(m.rg_kernel.local_encoder(patterns)).softmax(-1).numpy()
    report=dict(checkpoint=str(run.checkpoint),train_ensembles=len(tr.configurations),test_ensembles=len(te.configurations),
        parameters={name:sum(p.numel() for p in module.parameters()) for name,module in [('Q',m.theory_encoder),('C',m.rg_kernel),('R',m.theory_rg)]},
        test_closure=closure_metrics(test,reference_direct=train['direct']),critical_closure=closure_metrics(a,reference_direct=train['direct']),
        critical_direct_mean=ad[center].mean(0).tolist(),critical_rollout_mean=ar[center].mean(0).tolist(),
        critical_error_norm=np.linalg.norm((ar[center]-ad[center]).mean(0),axis=1).tolist(),
        covariance_eigenvalues=[np.linalg.eigvalsh(np.cov(all_d[:,k].T,bias=True)).tolist() for k in range(3)],
        uniform_block_output_probabilities=prob.tolist(),final_weighted_test_losses=weighted,
        caveat='All three modules initialized from scratch; joint gradients and learnable probabilistic C differ from prior fixed-C separated-gradient runs. New latent coordinates prohibit raw-coordinate comparisons to old Q. Error bars combine independent-chain and C-sampling variation; 2 observed coarse steps do not validate long-run basins.')
    (out/'benchmark.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
