"""Unsupervised Q-only pilot with fixed positive decimation; no R forward calls."""
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recurrent_rg.checkpoints import build_model, qcr_state_dict
from recurrent_rg.config import ModelConfig
from recurrent_rg.data.statistics import plaquette_distribution
from recurrent_rg.losses import distribution_geometry_loss


def statistics(model, x):
    """Four disjoint sample quarters; exact Q sufficient statistics with fixed C."""
    hist, patterns, obs = [], [], []
    powers = 2**torch.arange(9)
    with torch.no_grad():
        for chunk in x.split(16):
            hs, ps, os = [], [], []
            for level in model.coarse_grain(chunk.double()):
                hq, pq = [], []
                for quarter in level.chunk(4, dim=1):
                    patch = model.geometry.extract_patches(quarter).reshape(len(chunk), -1, 9)
                    index = ((patch > 0).long()*powers).sum(-1)
                    index += torch.arange(len(chunk))[:, None]*512
                    counts = torch.bincount(index.flatten(), minlength=len(chunk)*512)
                    hq.append(counts.reshape(len(chunk),512).double()/patch.shape[1])
                    pq.append(plaquette_distribution(quarter))
                hs.append(torch.stack(hq,1)); ps.append(torch.stack(pq,1))
                dims = tuple(range(1,level.ndim))
                os.append(torch.stack((level.mean(dims), .5*((level*level.roll(-1,2)).mean(dims)+(level*level.roll(-1,3)).mean(dims))),-1))
            hist.append(torch.stack(hs,2)); patterns.append(torch.stack(ps,2)); obs.append(torch.stack(os,1))
    return torch.cat(hist), torch.cat(patterns), torch.cat(obs)


def encode(model, hist):
    width = model.geometry.patch_volume
    powers = 2**torch.arange(width)
    patterns = (2*((torch.arange(2**width)[:,None]//powers)%2)-1).double()
    return model.theory_encoder.head(hist @ model.theory_encoder.local_mlp(patterns))


def objective(model, hist, patterns, partition, isotropy_weight=10.):
    a, b = partition[:2], partition[2:]
    ga, gb = encode(model,hist[:,a].mean(1)), encode(model,hist[:,b].mean(1))
    g = .5*(ga+gb)
    split = (ga-gb).square().sum(-1).mean(0)
    geometry = torch.stack([distribution_geometry_loss(g[:,k],patterns[:,a,k].mean(1),patterns[:,b,k].mean(1))[0] for k in range(g.shape[1])])
    centered = g[:,0]-g[:,0].mean(0)
    covariance = centered.T@centered/len(g)
    isotropy = (covariance-torch.eye(2)).square().sum()/2
    terms = dict(split_micro=20*split[0], split_coarse=5*split[1:].mean(), geometry=10*geometry.mean(), isotropy=isotropy_weight*isotropy)
    return sum(terms.values()), terms


def readout(train, test, ytrain, ytest):
    def features(x):
        return np.column_stack((np.ones(len(x)),x,x[:,0]**2,x[:,0]*x[:,1],x[:,1]**2))
    pred = features(test)@np.linalg.lstsq(features(train),ytrain,rcond=1e-10)[0]
    r2 = 1-((pred-ytest)**2).sum(0)/((ytest-ytest.mean(0))**2).sum(0)
    return r2.tolist(), pred


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=2000)
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2); torch.manual_seed(91); start=time.perf_counter()
    tx,ty,vx,vy=torch.load(args.cache,weights_only=True)
    config=ModelConfig(dimension=2,theory_dim=2,hidden=60,fixed_kernel=True)
    model=build_model(config,2)
    for param in model.parameters(): param.requires_grad_(False)
    for param in model.theory_encoder.parameters(): param.requires_grad_(True)
    frozen={n:p.clone() for n,p in model.state_dict().items() if not n.startswith('theory_encoder.')}
    tr=statistics(model,tx); te=statistics(model,vx)
    # Verify cached forward AND parameter gradients against the ordinary encoder.
    raw=model.theory_encoder(tx[:2,:32].double())
    fast=encode(model,tr[0][:2,:2].mean(1))[:,0]
    torch.testing.assert_close(raw,fast,atol=1e-12,rtol=1e-10)
    params=list(model.theory_encoder.parameters())
    grad_raw=torch.autograd.grad(raw.square().sum(),params)
    grad_fast=torch.autograd.grad(fast.square().sum(),params)
    for a,b in zip(grad_raw,grad_fast): torch.testing.assert_close(a,b,atol=1e-11,rtol=1e-9)
    opt=torch.optim.Adam(params,lr=.001); history=[]
    gen=torch.Generator().manual_seed(92)
    def evaluate(data):
        with torch.no_grad():
            vals=[objective(model,*data[:2],torch.tensor(s))[1] for s in ([0,1,2,3],[0,2,1,3],[0,3,1,2])]
        r={key:float(torch.stack([v[key] for v in vals]).mean()) for key in vals[0]}
        r['total']=sum(r.values()); return r
    for step in range(args.steps+1):
        if step:
            idx=torch.randperm(len(tx),generator=gen)[:64]
            partition=torch.randperm(4,generator=gen)
            opt.zero_grad(); loss,_=objective(model,tr[0][idx],tr[1][idx],partition)
            assert torch.isfinite(loss)
            loss.backward(); opt.step()
        if step%100==0 or step==args.steps:
            row=dict(step=step,train=evaluate(tr),test=evaluate(te));history.append(row)
            print(step,row,flush=True)
    for n,v in frozen.items(): assert torch.equal(v,model.state_dict()[n]),n
    with torch.no_grad():
        gt=encode(model,tr[0].mean(1)).numpy(); gv=encode(model,te[0].mean(1)).numpy()
        ga=encode(model,te[0][:,:2].mean(1)).numpy();gb=encode(model,te[0][:,2:].mean(1)).numpy()
    metrics=[]; preds=[]
    for k in range(3):
        eig, axes=np.linalg.eigh(np.cov(gt[:,k].T,bias=True))
        signal=np.var(gv[:,k]@axes,axis=0)
        noise=np.mean(((ga[:,k]-gb[:,k])@axes)**2,axis=0)/4
        r2,pred=readout(gt[:,k],gv[:,k],ty.numpy(),vy.numpy());preds.append(pred)
        obs_r2,_=readout(gt[:,k],gv[:,k],tr[2][:,k].numpy(),te[2][:,k].numpy())
        # A curved one-dimensional representation can have full covariance rank.
        # Post-hoc readouts are diagnostics, not proof of sufficiency/intrinsic dimension.
        metrics.append(dict(k=k,train_covariance_eigenvalues=eig.tolist(),weak_variance_fraction=float(eig[0]/eig.sum()),test_pc_signal_to_split_noise_variance=(signal/np.maximum(noise,1e-15)).tolist(),heldout_quadratic_K_B_R2=r2,heldout_quadratic_m_bond_R2=obs_r2))
    report=dict(seed=91,steps=args.steps,cache=str(args.cache.resolve()),train_shape=list(tx.shape),test_shape=list(vx.shape),optimized_parameters=sum(p.numel() for p in params),contract='Unsupervised Q only; fixed positive decimation; no R calls, no labels, no augmentation; covariance regularization only k=0; four fixed disjoint sample quarters with random balanced re-pairing.',loss='20 split0 + 5 mean(split1,split2) + 10 mean(geometry0..2) + 10 ||Cov(g0)-I||_F^2/2; geometry unchanged scale4 noise2',metrics=metrics,history=history,seconds=time.perf_counter()-start,checks=dict(cached_forward_and_gradients=True,non_Q_parameters_unchanged=True))
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    torch.save(dict(model=qcr_state_dict(model),metadata=dict(config=dict(dimension=2,theory_dim=2,hidden=60,levels=3,fixed_kernel=True,kernel_layout='nonoverlapping',binary_lookup=True))),args.output/'q_only.pt')
    np.savez(args.output/'encodings.npz',train=gt,test=gv,train_labels=ty.numpy(),test_labels=vy.numpy())
    fig,axs=plt.subplots(2,3,figsize=(13,8))
    for k in range(3):
        for j,label in enumerate(('K','B')):
            sc=axs[j,k].scatter(*gv[:,k].T,c=vy[:,j],s=13,cmap='viridis' if j==0 else 'coolwarm')
            axs[j,k].set(title=f'k={k}, held-out, colored by {label}',xlabel='g1',ylabel='g2',aspect='equal');fig.colorbar(sc,ax=axs[j,k])
    fig.tight_layout();fig.savefig(args.output/'latent_KB.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(13,7))
    for k in range(3):
        for j,name in enumerate(('K','B')):
            ax=axs[j,k];ax.scatter(vy[:,j],preds[k][:,j],s=9);lims=[float(vy[:,j].min()),float(vy[:,j].max())];ax.plot(lims,lims,'k--')
            ax.set(title=f'k={k}, {name}: R2={metrics[k]["heldout_quadratic_K_B_R2"][j]:.3f}',xlabel='actual',ylabel='train-only quadratic readout')
    fig.tight_layout();fig.savefig(args.output/'readouts.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,split in zip(axs,('train','test')):
        for key in history[0][split]: ax.plot([r['step'] for r in history],[r[split][key] for r in history],label=key)
        ax.set(title=split,yscale='log',xlabel='Q updates',ylabel='weighted loss');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(args.output/'loss.png',dpi=170);plt.close(fig)
    print(json.dumps(dict(metrics=metrics,seconds=report['seconds']),indent=2),flush=True)


if __name__=='__main__': main()
