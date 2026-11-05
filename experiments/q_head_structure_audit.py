"""Frozen-feature, paired Q-head ablation. Does not alter QCR defaults."""
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from recurrent_rg.checkpoints import load_run, qcr_state_dict
from recurrent_rg.data.ensembles import load_ensembles
from recurrent_rg.data.statistics import categorical_plaquette_distribution
from recurrent_rg.runner import file_fingerprint
from experiments.q_only_cached_sweep import sample_counts
from experiments.q_only_candidates import validate_frozen_levels
from experiments.q_last_layer_audit import loss


class MonotoneSpline(torch.nn.Module):
    """Continuous piecewise-linear spline with positive slopes and linear tails."""
    def __init__(self):
        super().__init__()
        self.register_buffer('knots',torch.linspace(-4,4,17,dtype=torch.double))
        self.raw_slopes=torch.nn.Parameter(torch.full((18,),float(np.log(np.expm1(.999))),dtype=torch.double))

    def forward(self,x):
        slopes=torch.nn.functional.softplus(self.raw_slopes)+.001
        ramps=(x-self.knots).relu()-(-self.knots).relu()
        return slopes[0]*x+(ramps*(slopes[1:]-slopes[:-1])).sum(-1,keepdim=True)


class ProjectionHead(torch.nn.Module):
    def __init__(self,mean,whitener,linear,spline=False):
        super().__init__()
        self.register_buffer('mean',mean.clone());self.register_buffer('whitener',whitener.clone())
        self.linear=copy.deepcopy(linear)
        self.spline=MonotoneSpline() if spline else torch.nn.Identity()

    def forward(self,x):return self.spline(self.linear((x-self.mean)@self.whitener))


def paired_heads(mean,whitener,seed):
    torch.manual_seed(seed);linear=torch.nn.Linear(len(mean),1).double()
    raw=ProjectionHead(torch.zeros_like(mean),torch.eye(len(mean),dtype=torch.double),linear)
    white=ProjectionHead(mean,whitener,linear)
    with torch.no_grad():
        white.linear.weight.copy_(torch.linalg.solve(whitener,linear.weight.T).T)
        white.linear.bias.copy_(linear.bias+linear.weight@mean)
    spline=ProjectionHead(mean,whitener,white.linear,True)
    return dict(linear=raw,white_linear=white,white_spline=spline)


def run(source,cache_source,output,epochs):
    start=time.monotonic();torch.set_num_threads(2)
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    loaded=load_run(source);model=loaded.model;q=model.theory_encoder
    for param in model.parameters():param.requires_grad_(False)
    frozen={k:v.clone() for k,v in qcr_state_dict(model).items()}
    tr,te=load_ensembles(loaded.config['data']['cache']);datasets=dict(train=tr,test=te)
    cachefile=Path(cache_source)/'frozen_levels.pt';levels=torch.load(cachefile,weights_only=True)
    codes=(torch.arange(3**9)[:,None]//(3**torch.arange(9)))%3
    patterns=torch.nn.functional.one_hot(codes,3).flatten(1).double()
    with torch.no_grad():table=q.local_mlp(patterns).numpy()
    cache={}
    for split,ls in levels.items():
        validate_frozen_levels(ls);torch.testing.assert_close(ls[0],datasets[split].configurations)
        features=[];ps=[]
        for i in range(len(ls[0])):
            xs=[v[i:i+1].double() for v in ls]
            features.append(np.stack([sample_counts(x,model.geometry,3)@table for x in xs]))
            ps.append(torch.stack([categorical_plaquette_distribution(x.squeeze(0).unsqueeze(1)) for x in xs]))
        cache[split]=(torch.from_numpy(np.stack(features)),torch.stack(ps))
    def batch(split,ids,halves):
        f,p=cache[split]
        return (torch.stack([f[ids][:,:,h].mean(2) for h in halves],2),
                torch.stack([p[ids][:,:,h].mean(2) for h in halves],2))
    evals={s:batch(s,torch.arange(len(d.configurations)),[torch.arange(32),torch.arange(32,64)]) for s,d in datasets.items()}
    full={s:f.mean(2) for s,(f,p) in cache.items()}
    with torch.no_grad():torch.testing.assert_close(q.head(full['test'][:1,0]),q(te.configurations[:1].double()),atol=1e-10,rtol=1e-10)
    x=full['train'].reshape(-1,12);mean=x.mean(0);centered=x-mean
    cov=centered.T@centered/len(x);eig,V=torch.linalg.eigh(cov);epsilon=float(eig.max())*.001
    whitener=(V*(eig+epsilon).rsqrt())@V.T
    heads={}
    for seed in [91,92,93]:
        pair=paired_heads(mean,whitener,seed)
        with torch.no_grad():
            for h in pair.values():torch.testing.assert_close(h(full['test']),pair['linear'](full['test']),atol=1e-10,rtol=1e-10)
        heads.update({f'{name}_s{seed}':h for name,h in pair.items()})
    opts={n:torch.optim.Adam(h.parameters(),lr=.001) for n,h in heads.items()};history={n:[] for n in heads}
    gen=torch.Generator().manual_seed(91)
    def evaluate(step):
        with torch.no_grad():
            for n,h in heads.items():history[n].append(dict(epoch=step,**{s:loss(h,*ev)[1] for s,ev in evals.items()}))
        (out/'history.json').write_text(json.dumps(history,indent=2))
    evaluate(0);completed=0
    print('Exact cache/paired function initialization verified; 9 heads, shared frozen backbone.',flush=True)
    for step in range(1,epochs+1):
        if time.monotonic()-start>480:break
        ids=torch.randperm(len(tr.configurations),generator=gen)[:32];perm=torch.randperm(64,generator=gen)
        h,p=batch('train',ids,[perm[:32],perm[32:]])
        for name,head in heads.items():
            opts[name].zero_grad();value,_=loss(head,h,p)
            if not torch.isfinite(value):raise FloatingPointError(name)
            value.backward();opts[name].step()
        completed=step
        if step%200==0:evaluate(step);print(f'{step}/{epochs}; {time.monotonic()-start:.1f}s',flush=True)
    if history[next(iter(heads))][-1]['epoch']!=completed:evaluate(completed)
    # Labels first used below, for evaluation only.
    ks={s:d.couplings.reshape(len(d.configurations),-1)[:,0].numpy() for s,d in datasets.items()}
    metrics={};enc={}
    with torch.no_grad():
        for name,head in heads.items():
            g=head(full['test']).squeeze(-1).numpy();k=ks['test'];enc[name]=g
            vals=[]
            for j in range(3):
                unique=np.unique(k);means=np.array([g[k==v,j].mean() for v in unique]);ds=np.diff(means)
                direction=1 if spearmanr(k,g[:,j]).statistic>=0 else -1
                vals.append(dict(abs_spearman=float(abs(spearmanr(k,g[:,j]).statistic)),opposite_variation_fraction=float(np.maximum(-direction*ds,0).sum()/max(abs(ds).sum(),1e-12))))
            metrics[name]=dict(loss=history[name][-1],scales=vals,parameters=sum(p.numel() for p in head.parameters()))
            torch.save(dict(head_type=name.split('_s')[0],head_state=head.state_dict(),epoch=completed,source=str(loaded.checkpoint.resolve()),source_sha256=file_fingerprint(loaded.checkpoint)),out/f'{name}.pt')
        enc['original']=q.head(full['test']).squeeze(-1).numpy()
    for key,value in qcr_state_dict(model).items():torch.testing.assert_close(value,frozen[key],atol=0,rtol=0)
    report=dict(actual_steps=completed,requested_steps=epochs,elapsed_seconds=time.monotonic()-start,source=str(source),source_sha256=file_fingerprint(loaded.checkpoint),cache_sha256=file_fingerprint(cachefile),train_eigenvalues=eig.tolist(),whitening_epsilon=epsilon,whitening='Train full-ensemble features, all three scales equally weighted; one frozen transform for all scales; eigenvalues regularized by 0.001*lambda_max',initialization='Same function per seed across architectures, identity spline',objective='20 micro subset MSE+5 coarse subset MSE+10 relative_local geometry averaged across scales',results=metrics,labels_used_in_training=False,frozen_source_unchanged=True,limitations='Only heads trained; three head seeds on one backbone/C draw; original R not compatible with changed Q coordinates, no R training or fixed-point claims; standalone head artifacts require original backbone')
    (out/'metrics.json').write_text(json.dumps(report,indent=2));np.savez(out/'encodings.npz',K=k,**enc)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,3,figsize=(14,11));unique=np.unique(k)
    for row,seed in enumerate([91,92,93]):
        for col,kind in enumerate(['linear','white_linear','white_spline']):
            ax=axes[row,col];g=enc[f'{kind}_s{seed}']
            for j in range(3):ax.plot(unique,[g[k==v,j].mean() for v in unique],label=f'k={j}')
            ax.set(title=f'{kind}, seed={seed}',xlabel='K (evaluation only)',ylabel='g');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'K_to_g.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for name,rows in history.items():
        if not name.endswith('s91'):continue
        for ax,key in zip(axes,['total','anchor','geometry']):
            for split,style in [('train','-'),('test','--')]:ax.plot([r['epoch'] for r in rows],[np.mean(r[split][key]) for r in rows],style,label=name+' '+split)
            ax.set(title=key,xlabel='updates',yscale='log')
    axes[0].legend(fontsize=6);fig.tight_layout();fig.savefig(out/'loss_seed91.png',dpi=160);plt.close(fig)
    print(json.dumps(report,indent=2),flush=True)
