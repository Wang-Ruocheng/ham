"""Explicit last-layer-only ablation, with a separate supervised diagnostic reference."""
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
from recurrent_rg.losses import distribution_geometry_loss
from recurrent_rg.runner import file_fingerprint
from experiments.q_only_cached_sweep import sample_counts
from experiments.q_only_candidates import OPTIONS, validate_frozen_levels


def loss(head, hidden, p):
    g=head(hidden)
    anchor=(g[:,0,0]-g[:,0,1]).square().mean()
    consistency=(g[:,1:,0]-g[:,1:,1]).square().mean()
    geometry=torch.stack([distribution_geometry_loss(g[:,k].mean(1),p[:,k,0],p[:,k,1],**OPTIONS)[0] for k in range(3)])
    total=20*anchor+5*consistency+10*geometry.mean()
    return total,dict(total=float(total.detach()),anchor=float(anchor.detach()),consistency=float(consistency.detach()),geometry=geometry.detach().tolist())


def run(source,cache_source,output,epochs):
    start=time.monotonic();torch.set_num_threads(2)
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    loaded=load_run(source);model=loaded.model;q=model.theory_encoder
    for param in model.parameters():param.requires_grad_(False)
    original=copy.deepcopy(q.head[-1])
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
        with torch.no_grad():
            f=torch.stack([f[ids][:,:,h].mean(2) for h in halves],2)
            p=torch.stack([p[ids][:,:,h].mean(2) for h in halves],2)
            return q.head[:2](f),p
    evals={s:batch(s,torch.arange(len(d.configurations)),[torch.arange(32),torch.arange(32,64)]) for s,d in datasets.items()}
    full={s:q.head[:2](f.mean(2)).detach() for s,(f,p) in cache.items()}
    with torch.no_grad():
        torch.testing.assert_close(original(full['test'][:1,0]),q(te.configurations[:1].double()),atol=1e-10,rtol=1e-10)
    ks={s:d.couplings.reshape(len(d.configurations),-1)[:,0].double() for s,d in datasets.items()}
    # This readout is diagnostic only: no supervised initialization enters the unsupervised arms.
    x=full['train'][:,0].numpy();y=ks['train'].numpy();mu=x.mean(0);sd=np.maximum(x.std(0),1e-10)
    design=np.column_stack([(x-mu)/sd,np.ones(len(x))]);reg=np.eye(design.shape[1])*1e-3;reg[-1,-1]=0
    coef=np.linalg.solve(design.T@design+reg,design.T@y)
    ref=copy.deepcopy(original)
    with torch.no_grad():
        ref.weight.copy_(torch.from_numpy(coef[:-1]/sd)[None]);ref.bias.fill_(coef[-1]-mu@(coef[:-1]/sd))
    reference={};scales=np.logspace(-2,2,81);scan=[]
    with torch.no_grad():
        for scale in scales:
            head=copy.deepcopy(ref);head.weight.mul_(scale);head.bias.mul_(scale)
            scan.append(loss(head,*evals['train'])[1]['total'])
        best=float(scales[np.argmin(scan)])
        for name,scale in [('K_readout',1.),('K_readout_train_loss_scaled',best)]:
            head=copy.deepcopy(ref);head.weight.mul_(scale);head.bias.mul_(scale)
            reference[name]=dict(scale=scale,loss={s:loss(head,*ev)[1] for s,ev in evals.items()},K_test_rmse=float(((ref(full['test'][:,0]).squeeze()-ks['test'])**2).mean().sqrt()))
    heads={'continue':copy.deepcopy(original)}
    for seed in [91,92,93]:
        torch.manual_seed(seed);heads[f'reset{seed}']=torch.nn.Linear(original.in_features,1).double()
    for h in heads.values():
        for param in h.parameters():param.requires_grad_(True)
    opts={n:torch.optim.Adam(h.parameters(),lr=.001) for n,h in heads.items()}
    history={n:[] for n in heads};gen=torch.Generator().manual_seed(91)
    def evaluate(step):
        with torch.no_grad():
            for n,h in heads.items():
                history[n].append(dict(epoch=step,**{s:loss(h,*ev)[1] for s,ev in evals.items()}))
        (out/'history.json').write_text(json.dumps(history,indent=2))
    initial={s:loss(original,*ev)[1] for s,ev in evals.items()}
    evaluate(0);completed=0
    print('Frozen-feature cache verified. Starting last-layer-only updates.',flush=True)
    for step in range(1,epochs+1):
        if time.monotonic()-start>480:break
        ids=torch.randperm(len(tr.configurations),generator=gen)[:32];perm=torch.randperm(64,generator=gen)
        hidden,p=batch('train',ids,[perm[:32],perm[32:]])
        for name,head in heads.items():
            opts[name].zero_grad();value,_=loss(head,hidden,p);value.backward();opts[name].step()
        completed=step
        if step%200==0:evaluate(step);print(f'{step}/{epochs}, elapsed {time.monotonic()-start:.1f}s',flush=True)
    if history['continue'][-1]['epoch']!=completed:evaluate(completed)
    metrics={};enc={};state=qcr_state_dict(model)
    with torch.no_grad():
        for name,head in heads.items():
            g=head(full['test']).squeeze(-1).numpy();k=ks['test'].numpy()
            metrics[name]=dict(loss=history[name][-1],abs_spearman=[float(abs(spearmanr(k,g[:,j]).statistic)) for j in range(3)])
            enc[name]=g
            saved={key:value.clone() for key,value in state.items()}
            saved['theory_encoder.head.2.weight']=head.weight.detach().clone();saved['theory_encoder.head.2.bias']=head.bias.detach().clone()
            for key in state:
                if key not in ['theory_encoder.head.2.weight','theory_encoder.head.2.bias']:torch.testing.assert_close(saved[key],state[key],atol=0,rtol=0)
            torch.save(dict(model=saved,metadata=dict(config=loaded.config,ablation='last layer only',source=str(source)),epoch=completed),out/f'{name}.pt')
        enc['original']=original(full['test']).squeeze(-1).numpy();enc['K_reference']=ref(full['test']).squeeze(-1).numpy()
    report=dict(actual_steps=completed,requested_steps=epochs,elapsed_seconds=time.monotonic()-start,source=str(source),source_sha256=file_fingerprint(loaded.checkpoint),cache_sha256=file_fingerprint(cachefile),initial=initial,reference=reference,results=metrics,protocol='Q final linear layer only; all preceding Q/C/R exactly unchanged; original Q loss 20 anchor+5 coarse consistency+10 mean geometry; three unsupervised random starts and continuation; labels used only by static diagnostic reference',limitations='One frozen backbone/C draw; finite training; supervised reference only fit at k=0, no guarantee for coarse scales; scale selected on training data only')
    (out/'metrics.json').write_text(json.dumps(report,indent=2));np.savez(out/'encodings.npz',K=k,**enc)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(14,8));order=np.argsort(k)
    for ax,(name,g) in zip(axes.flat,enc.items()):
        for j in range(3):ax.plot(k[order],g[order,j],label=f'k={j}')
        ax.set(title=name,xlabel='K (evaluation only)',ylabel='g');ax.legend()
    fig.tight_layout();fig.savefig(out/'K_to_g.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for name,rows in history.items():
        for ax,key in zip(axes,['total','anchor','geometry']):
            for split,style in [('train','-'),('test','--')]:
                ax.plot([r['epoch'] for r in rows],[np.mean(r[split][key]) for r in rows],style,label=name+' '+split)
            ax.set(title=key,xlabel='updates',yscale='log')
    axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(out/'loss.png',dpi=160);plt.close(fig)
    print(json.dumps(report,indent=2),flush=True)
