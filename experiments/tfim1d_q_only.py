"""Frozen decimation, unsupervised Q; covariance rank versus curved 1D locus."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recurrent_rg.checkpoints import build_model,qcr_state_dict
from recurrent_rg.config import ModelConfig
from recurrent_rg.data.ensembles import load_ensembles
from recurrent_rg.data.statistics import two_spin_distribution
from experiments.ising2d_q_only import encode,objective


def statistics(model,x):
    hs,ps=[],[]
    with torch.no_grad():
        for chunk in x.split(16):
            hh,pp=[],[]
            for level in model.coarse_grain(chunk.double()):
                h,p=[],[]
                for q in level.chunk(4,dim=1):
                    patch=model.geometry.extract_patches(q).reshape(len(q),-1,3)
                    ids=((patch>0).long()*torch.tensor([1,2,4])).sum(-1)+torch.arange(len(q))[:,None]*8
                    h.append(torch.bincount(ids.flatten(),minlength=len(q)*8).reshape(len(q),8).double()/patch.shape[1])
                    p.append(two_spin_distribution(q))
                hh.append(torch.stack(h,1));pp.append(torch.stack(p,1))
            hs.append(torch.stack(hh,2));ps.append(torch.stack(pp,2))
    return torch.cat(hs),torch.cat(ps)


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=2000)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();torch.set_num_threads(2);torch.manual_seed(91)
    train,test=load_ensembles(args.cache);tx,vx=train.configurations,test.configurations
    ty=train.couplings.numpy().reshape(-1);vy=test.couplings.numpy().reshape(-1)
    depth=int(np.log2(tx.shape[2]//4));model=build_model(ModelConfig(dimension=1,theory_dim=2,hidden=60,fixed_kernel=True),depth)
    for param in model.parameters():param.requires_grad_(False)
    for param in model.theory_encoder.parameters():param.requires_grad_(True)
    frozen={n:v.clone() for n,v in model.state_dict().items() if not n.startswith('theory_encoder.')}
    tr=statistics(model,tx);te=statistics(model,vx)
    raw=model.theory_encoder(tx[:2,:tx.shape[1]//2].double());fast=encode(model,tr[0][:2,:2].mean(1))[:,0]
    torch.testing.assert_close(raw,fast,atol=1e-12,rtol=1e-10)
    params=list(model.theory_encoder.parameters())
    for a,b in zip(torch.autograd.grad(raw.square().sum(),params),torch.autograd.grad(fast.square().sum(),params)):
        torch.testing.assert_close(a,b,atol=1e-11,rtol=1e-9)
    opt=torch.optim.Adam(params,lr=.001);gen=torch.Generator().manual_seed(92);history=[]
    def evaluate(data):
        with torch.no_grad():vals=[objective(model,*data,torch.tensor(s))[1] for s in ([0,1,2,3],[0,2,1,3],[0,3,1,2])]
        r={k:float(torch.stack([v[k] for v in vals]).mean()) for k in vals[0]};r['total']=sum(r.values());return r
    for step in range(args.steps+1):
        if step:
            idx=torch.randperm(len(tx),generator=gen)[:64];partition=torch.randperm(4,generator=gen)
            opt.zero_grad();loss,_=objective(model,tr[0][idx],tr[1][idx],partition);assert torch.isfinite(loss);loss.backward();opt.step()
        if step%100==0 or step==args.steps:
            row=dict(step=step,train=evaluate(tr),test=evaluate(te));history.append(row);print(step,row,flush=True)
    for n,v in frozen.items():assert torch.equal(v,model.state_dict()[n]),n
    with torch.no_grad():
        gt=encode(model,tr[0].mean(1)).numpy();gv=encode(model,te[0].mean(1)).numpy()
        ga=encode(model,te[0][:,:2].mean(1)).numpy();gb=encode(model,te[0][:,2:].mean(1)).numpy()
    unique=np.unique(ty);means=np.array([gt[ty==h].mean(0) for h in unique]);metrics=[]
    for k in range(depth+1):
        eig,axes=np.linalg.eigh(np.cov(gt[:,k].T,bias=True))
        predicted=np.column_stack([np.interp(vy,unique,means[:,k,j]) for j in range(2)])
        curve_error=np.mean(np.sum((gv[:,k]-predicted)**2,axis=-1));total=np.var(gv[:,k],axis=0).sum()
        replicas=np.mean([np.mean(np.sum((gv[vy==h,k]-gv[vy==h,k].mean(0))**2,axis=-1)) for h in np.unique(vy)])
        noise=np.mean(((ga[:,k]-gb[:,k])@axes)**2,axis=0)/4
        metrics.append(dict(k=k,covariance_eigenvalues=eig.tolist(),weak_variance_fraction=float(eig[0]/eig.sum()),heldout_curve_R2=float(1-curve_error/total),within_coupling_variance_fraction=float(replicas/total),pc_signal_to_split_noise_variance=(np.var(gv[:,k]@axes,axis=0)/np.maximum(noise,1e-15)).tolist()))
    report=dict(seed=91,steps=args.steps,cache=str(args.cache.resolve()),train_shape=list(tx.shape),test_shape=list(vx.shape),optimized_parameters=sum(p.numel() for p in params),contract='Fixed positive decimation; Q only; no R calls, coupling supervision, augmentation or hard symmetry. Same objective as preceding 2D Ising Q-only pilot; all scales equally weighted; covariance penalty at k=0 only.',loss='20 split0 + 5 mean coarse split + 10 mean geometry(scale4,noise2) + 10 ||Cov(g0)-I||_F^2/2',metrics=metrics,history=history,seconds=time.perf_counter()-start,checks=dict(cached_forward_and_gradients=True,non_Q_weights_unchanged=True),diagnostic_caution='Curve R2 uses train-coupling mean g linearly interpolated to held-out couplings; replica variance uses three replicas per coupling. Covariance rank is not intrinsic dimension.')
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez(args.output/'encodings.npz',train=gt,test=gv,train_labels=ty,test_labels=vy)
    torch.save(dict(model=qcr_state_dict(model),metadata=dict(config=dict(dimension=1,theory_dim=2,hidden=60,levels=depth+1,fixed_kernel=True,kernel_layout='nonoverlapping',binary_lookup=True))),args.output/'q_only.pt')
    fig,axs=plt.subplots(2,3,figsize=(13,8),squeeze=False)
    for k,ax in enumerate(axs.flat):
        if k>depth:ax.axis('off');continue
        ax.plot(*means[:,k].T,'k-',lw=1,label='train coupling means')
        sc=ax.scatter(*gv[:,k].T,c=vy,cmap='viridis',s=18,vmin=.01,vmax=2)
        ax.set(title=f'k={k}; weak PC {100*metrics[k]["weak_variance_fraction"]:.1f}%',xlabel='g1',ylabel='g2',aspect='equal');ax.legend(fontsize=7);fig.colorbar(sc,ax=ax,label='h/J (held-out)')
    fig.tight_layout();fig.savefig(args.output/'Q_latent_by_scale.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for j,ax in enumerate(axs):
        ax.plot(unique,means[:,0,j],label='train coupling means');ax.scatter(vy,gv[:,0,j],s=15,label='held-out ensembles')
        ax.set(xlabel='h/J',ylabel=f'g{j+1}',title='Microscopic encoding');ax.legend()
    fig.tight_layout();fig.savefig(args.output/'coupling_vs_g.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,split in zip(axs,('train','test')):
        for key in history[0][split]:ax.plot([r['step'] for r in history],[r[split][key] for r in history],label=key)
        ax.set(title=split,xlabel='Q updates',ylabel='weighted loss',yscale='log');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(args.output/'loss.png',dpi=180);plt.close(fig)
    print(json.dumps(dict(metrics=metrics,seconds=report['seconds']),indent=2))


if __name__=='__main__':main()
