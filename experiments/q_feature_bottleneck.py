"""Explicit frozen-feature autoencoder diagnostic, never a default QCR recipe."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from recurrent_rg.checkpoints import load_run, qcr_state_dict
from recurrent_rg.data.ensembles import load_ensembles
from recurrent_rg.runner import file_fingerprint
from experiments.q_only_cached_sweep import sample_counts
from experiments.q_only_candidates import validate_frozen_levels


class FeatureBottleneck(torch.nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.encoder=torch.nn.Sequential(torch.nn.Linear(12,32),torch.nn.Tanh(),torch.nn.Linear(32,dim)).double()
        self.decoder=torch.nn.Sequential(torch.nn.Linear(dim,32),torch.nn.Tanh(),torch.nn.Linear(32,12)).double()

    def forward(self,x):return self.encoder(x)


def objective(model,x):
    target=x.detach();g=model(target);decoded=model.decoder(g)
    reconstruction=(decoded-target).square().mean()
    anchor=(g[:,0,0]-g[:,0,1]).square().sum(-1).mean()
    consistency=(g[:,1:,0]-g[:,1:,1]).square().sum(-1).mean()
    total=10*reconstruction+20*anchor+5*consistency
    return total,dict(total=float(total.detach()),reconstruction=float(reconstruction.detach()),anchor=float(anchor.detach()),consistency=float(consistency.detach()))


def linear_probe(a,b,y):
    mean=a.mean(0);sd=np.maximum(a.std(0),1e-10)
    A=np.column_stack([(a-mean)/sd,np.ones(len(a))]);B=np.column_stack([(b-mean)/sd,np.ones(len(b))])
    reg=np.eye(A.shape[1])*.001;reg[-1,-1]=0
    return B@np.linalg.solve(A.T@A+reg,A.T@y)


def run(source,cache_source,output,epochs):
    start=time.monotonic();torch.set_num_threads(2)
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    loaded=load_run(source);model=loaded.model;q=model.theory_encoder
    for p in model.parameters():p.requires_grad_(False)
    frozen={k:v.clone() for k,v in qcr_state_dict(model).items()}
    tr,te=load_ensembles(loaded.config['data']['cache']);datasets=dict(train=tr,test=te)
    cachefile=Path(cache_source)/'frozen_levels.pt'
    levels=torch.load(cachefile,weights_only=True)
    codes=(torch.arange(3**9)[:,None]//(3**torch.arange(9)))%3
    patterns=torch.nn.functional.one_hot(codes,3).flatten(1).double()
    with torch.no_grad():table=q.local_mlp(patterns).numpy()
    features={}
    for split,ls in levels.items():
        validate_frozen_levels(ls);torch.testing.assert_close(ls[0],datasets[split].configurations)
        features[split]=torch.from_numpy(np.stack([np.stack([sample_counts(v[i:i+1].double(),model.geometry,3)@table for v in ls]) for i in range(len(ls[0]))]))
    full_raw={s:f.mean(2) for s,f in features.items()}
    with torch.no_grad():torch.testing.assert_close(q.head(full_raw['test'][:1,0]),q(te.configurations[:1].double()),atol=1e-10,rtol=1e-10)
    train_z=full_raw['train'].reshape(-1,12);mean=train_z.mean(0);centered=train_z-mean
    eig,V=torch.linalg.eigh(centered.T@centered/len(centered));epsilon=float(eig.max())*.001
    whitening=(V*(eig+epsilon).rsqrt())@V.T
    features={s:(f-mean)@whitening for s,f in features.items()}
    full={s:f.mean(2) for s,f in features.items()}
    def batch(split,ids,halves):return torch.stack([features[split][ids][:,:,h].mean(2) for h in halves],2)
    evals={s:batch(s,torch.arange(len(d.configurations)),[torch.arange(32),torch.arange(32,64)]) for s,d in datasets.items()}
    nets={}
    for seed in [91,92,93]:
        for dim in [1,2]:
            torch.manual_seed(seed);nets[f'g{dim}_s{seed}']=FeatureBottleneck(dim)
    opts={n:torch.optim.Adam(net.parameters(),lr=.001) for n,net in nets.items()}
    history={n:[] for n in nets};gen=torch.Generator().manual_seed(91)
    def evaluate(step):
        with torch.no_grad():
            for name,net in nets.items():history[name].append(dict(epoch=step,**{s:objective(net,x)[1] for s,x in evals.items()}))
        (out/'history.json').write_text(json.dumps(history,indent=2))
    evaluate(0);completed=0
    print('Frozen features verified; six bottlenecks, no geometry or coupling supervision.',flush=True)
    for step in range(1,epochs+1):
        if time.monotonic()-start>480:break
        ids=torch.randperm(len(tr.configurations),generator=gen)[:32];perm=torch.randperm(64,generator=gen)
        x=batch('train',ids,[perm[:32],perm[32:]])
        for name,net in nets.items():
            opts[name].zero_grad();value,_=objective(net,x)
            if not torch.isfinite(value):raise FloatingPointError(name)
            value.backward();opts[name].step()
        completed=step
        if step%300==0:evaluate(step);print(f'{step}/{epochs}; {time.monotonic()-start:.1f}s',flush=True)
    if history[next(iter(nets))][-1]['epoch']!=completed:evaluate(completed)
    # Physical labels are first accessed here and are used for diagnostic evaluation only.
    ks={s:d.couplings.reshape(len(d.configurations),-1)[:,0].numpy() for s,d in datasets.items()}
    results={};enc={};k=ks['test']
    with torch.no_grad():
        for name,net in nets.items():
            gtrain=net(full['train']).numpy();g=net(full['test']).numpy();rec=net.decoder(torch.from_numpy(g)).numpy();enc[name]=g
            scales=[]
            for j in range(3):
                target=full['test'][:,j].numpy();baseline=full['train'][:,j].mean(0).numpy()
                nrmse=float(np.sqrt(np.mean((rec[:,j]-target)**2)/np.mean((target-baseline)**2)))
                pred=linear_probe(gtrain[:,j],g[:,j],ks['train'])
                mask0=abs(ks['train'])>.2;mask1=abs(k)>.2
                sp=linear_probe(gtrain[mask0,j],g[mask1,j],np.sign(ks['train'][mask0]))
                stats=dict(reconstruction_nrmse=nrmse,K_probe_r2=float(1-np.sum((pred-k)**2)/np.sum((k-k.mean())**2)),K_sign_linear_accuracy=float(np.mean(np.sign(sp)==np.sign(k[mask1]))),latent_std=np.std(g[:,j],axis=0).tolist())
                halfg=net(evals['test'])[:,j].numpy()
                stats['subset_relative_rms']=float(np.sqrt(np.mean(np.sum((halfg[:,0]-halfg[:,1])**2,axis=-1))/np.mean(np.sum((g[:,j]-g[:,j].mean(0))**2,axis=-1))))
                if g.shape[-1]==1:
                    stats['abs_spearman']=float(abs(spearmanr(k,g[:,j,0]).statistic))
                    unique=np.unique(k);curve=np.array([g[k==v,j,0].mean() for v in unique]);delta=np.diff(curve)
                    stats['reverse_variation_fraction']=float(min(np.maximum(delta,0).sum(),np.maximum(-delta,0).sum())/max(abs(delta).sum(),1e-12))
                scales.append(stats)
            results[name]=dict(loss=history[name][-1],scales=scales,parameters=sum(p.numel() for p in net.parameters()))
            torch.save(dict(bottleneck_dim=g.shape[-1],state=net.state_dict(),mean=mean,whitening=whitening,epoch=completed,source=str(loaded.checkpoint.resolve()),source_sha256=file_fingerprint(loaded.checkpoint),optimizer=opts[name].state_dict()),out/f'{name}.pt')
    for key,value in qcr_state_dict(model).items():torch.testing.assert_close(value,frozen[key],atol=0,rtol=0)
    report=dict(actual_steps=completed,requested_steps=epochs,elapsed_seconds=time.monotonic()-start,source=str(source),source_sha256=file_fingerprint(loaded.checkpoint),cache_sha256=file_fingerprint(cachefile),whitening_epsilon=epsilon,train_feature_eigenvalues=eig.tolist(),objective='10 mean reconstruction MSE over all scales/halves/features +20 micro subset squared L2 +5 coarse subset squared L2; no geometry',architecture='Frozen 12-dimensional pooled features; train-only shared regularized whitening; 12->32 tanh->d encoder; d->32 tanh->12 decoder',labels_used_in_training=False,source_unchanged=True,results=results,limitations='Three bottleneck initializations on one fixed backbone and C draw; dimensionality changes parameter count and consistency sums coordinates; latent scale not fixed so normalized subset error also reported; reconstruction does not establish autonomous RG closure; no R trained')
    (out/'metrics.json').write_text(json.dumps(report,indent=2));np.savez(out/'encodings.npz',K=k,**enc)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,3,figsize=(14,11));unique=np.unique(k)
    for row,seed in enumerate([91,92,93]):
        a=enc[f'g1_s{seed}'];b=enc[f'g2_s{seed}']
        for j in range(3):
            axes[row,0].plot(unique,[a[k==v,j,0].mean() for v in unique],label=f'k={j}')
            means=np.stack([b[k==v,j].mean(0) for v in unique])
            axes[row,1].plot(means[:,0],means[:,1],alpha=.5)
            sc=axes[row,1].scatter(means[:,0],means[:,1],c=unique,cmap='coolwarm',vmin=-.9,vmax=.9,marker=['o','s','^'][j],label=f'k={j}',s=20)
        for d in range(2):axes[row,2].plot(unique,[b[k==v,0,d].mean() for v in unique],label=f'g{d+1}, k=0')
        axes[row,0].set(title=f'dimg=1, seed={seed}',xlabel='K',ylabel='g');axes[row,1].set(title=f'dimg=2, seed={seed}',xlabel='g1',ylabel='g2');axes[row,2].set(title='dimg=2 microscopic coordinates',xlabel='K',ylabel='g')
        for ax in axes[row]:ax.legend(fontsize=7)
        fig.colorbar(sc,ax=axes[row,1],label='K (evaluation only)')
    fig.tight_layout();fig.savefig(out/'embeddings.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,7))
    for row,dim in enumerate([1,2]):
        for seed in [91,92,93]:
            rows=history[f'g{dim}_s{seed}']
            for ax,key in zip(axes[row],['total','reconstruction','anchor']):
                for s,style in [('train','-'),('test','--')]:ax.plot([r['epoch'] for r in rows],[r[s][key] for r in rows],style,label=f's{seed} {s}')
                ax.set(title=f'dimg={dim}: {key}',xlabel='updates',yscale='log')
        axes[row,0].legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'loss.png',dpi=150);plt.close(fig)
    print(json.dumps(report,indent=2),flush=True)
