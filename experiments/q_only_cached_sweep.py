"""Explicit Q-only ablation on one shared frozen stochastic-C trajectory cache."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import time
import numpy as np
import torch
from scipy import sparse
from recurrent_rg.checkpoints import load_run, build_model, qcr_state_dict
from recurrent_rg.config import RunConfig
from recurrent_rg.data.ensembles import load_ensembles
from recurrent_rg.data.statistics import categorical_plaquette_distribution
from recurrent_rg.losses import distribution_geometry_loss
from recurrent_rg.runner import file_fingerprint


def sample_counts(x, geometry, states):
    """Per-configuration sparse 3x3 pattern counts; no learned features cached."""
    n=x.shape[1];s=x.argmax(-1);codes=torch.zeros_like(s)
    for j,offset in enumerate(geometry.patch_offsets()):
        codes += states**j*torch.roll(s,tuple(-a for a in offset),geometry.spatial_dims)
    flat=codes.numpy().reshape(n,-1)
    rows=np.repeat(np.arange(n),flat.shape[1])
    return sparse.csr_matrix((np.full(flat.size,1/flat.shape[1]),(rows,flat.ravel())),shape=(n,states**geometry.patch_volume))


def feature_table(q, patterns):
    return q.local_mlp(patterns)


def encode(q,patterns,hist):
    return q.head(hist @ feature_table(q,patterns))


def representation_loss(q,patterns,hist,plaquettes,weight,options):
    # [systems,scale,half,patterns] -> [systems,scale,half,1]
    g=encode(q,patterns,hist)
    anchor=(g[:,0,0]-g[:,0,1]).square().sum(-1).mean()
    consistency=(g[:,1:,0]-g[:,1:,1]).square().sum(-1).mean()
    geometry=torch.stack([distribution_geometry_loss(g[:,k].mean(1),plaquettes[:,k,0],plaquettes[:,k,1],**options)[0] for k in range(g.shape[1])]).mean()
    return 20*anchor+5*consistency+weight*geometry,dict(anchor=anchor,consistency=consistency,geometry=geometry)


def run(source,output,epochs):
    start=time.monotonic();output=Path(output);output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    loaded=load_run(source);cfg=RunConfig.from_dict(loaded.config)
    if cfg.model.categorical_states!=3 or cfg.model.theory_dim!=1 or cfg.model.dimension!=2:
        raise ValueError('Pilot currently requires scalar 2D three-state source')
    cfg=replace(cfg,model=replace(cfg.model,theory_head='nonlinear',reconstruction_hidden=0),
                training=replace(cfg.training,epochs=epochs,train_modules=('Q',)),
                loss=replace(cfg.loss,geometry_mode='relative_local',geometry_bounds='interval',geometry_distance='euclidean',geometry_tail_mix=0.,geometry_tail_add_weight=0.))
    torch.manual_seed(91);fresh=build_model(cfg.model,2)
    fresh.rg_kernel.load_state_dict(loaded.model.rg_kernel.state_dict())
    fresh.theory_rg.load_state_dict(loaded.model.theory_rg.state_dict())
    for p in fresh.parameters():p.requires_grad_(False)
    initial_q=copy.deepcopy(fresh.theory_encoder)
    for p in initial_q.parameters():p.requires_grad_(True)
    qs={w:copy.deepcopy(initial_q) for w in [10,30,100]}
    opts={w:torch.optim.Adam(q.parameters(),lr=.001) for w,q in qs.items()}
    tr,te=load_ensembles(cfg.data.cache);datasets={'train':tr,'test':te}
    cache={};cached_levels={};torch.manual_seed(9191)
    with torch.no_grad():
        for split,ds in datasets.items():
            count=[];pla=[];levels_all=[]
            for x in ds.configurations:
                levels=fresh.coarse_grain(x[None].double());levels_all.append([torch.nn.functional.one_hot(v.argmax(-1),3).to(torch.uint8) for v in levels])
                count.append([sample_counts(v,fresh.geometry,3) for v in levels])
                pla.append(torch.stack([categorical_plaquette_distribution(v.squeeze(0).unsqueeze(1)) for v in levels]).numpy())
            cache[split]=(count,np.stack(pla));cached_levels[split]=[torch.cat([ls[k] for ls in levels_all]) for k in range(3)]
    torch.save(cached_levels,output/'frozen_levels.pt')
    del cached_levels
    codes=(torch.arange(3**9)[:,None]//(3**torch.arange(9)))%3
    patterns=torch.nn.functional.one_hot(codes,3).flatten(1).double()
    def batch(split,ids,halves):
        counts,pl=cache[split]
        hist=np.array([[np.stack([np.asarray(counts[i][k][h].mean(0)).ravel() for h in halves]) for k in range(3)] for i in ids])
        p=np.stack([pl[ids][:,:,h].mean(2) for h in halves],axis=2)
        return torch.from_numpy(hist),torch.from_numpy(p)
    evals={s:batch(s,np.arange(len(ds.configurations)),[np.arange(32),np.arange(32,64)]) for s,ds in datasets.items()}
    with torch.no_grad():
        direct=initial_q(tr.configurations[:1].double())
        cached=encode(initial_q,patterns,evals['train'][0][:1,0].mean(1))
    torch.testing.assert_close(direct,cached,atol=1e-10,rtol=1e-10)
    # Verify gradients as well on the complete microscopic ensemble.
    qa=copy.deepcopy(initial_q);qb=copy.deepcopy(initial_q)
    a=qa(tr.configurations[:1].double()).sum();b=encode(qb,patterns,evals['train'][0][:1,0].mean(1)).sum()
    ga=torch.autograd.grad(a,tuple(qa.parameters()));gb=torch.autograd.grad(b,tuple(qb.parameters()))
    grad_error=max(float((a-b).abs().max()) for a,b in zip(ga,gb))
    if grad_error>1e-9:raise AssertionError('Cached gradient mismatch')
    options=dict(scale=4.,noise_multiplier=2.,mode='relative_local',relative_floor=.05,max_inverse_weight=100.,neighbors=4,local_fraction=.5)
    manifest=dict(config=cfg.to_dict(),source=str(source),source_checkpoint=str(loaded.checkpoint),source_sha256=file_fingerprint(loaded.checkpoint),data_sha256=file_fingerprint(cfg.data.cache),cached_levels_sha256=file_fingerprint(output/'frozen_levels.pt'),weights=[10,30,100],seed=91,coarse_draw_seed=9191,labels_used_in_training=False,gradient='Q only; no R or C updates',initialization='Fresh paired nonlinear Q; source C/R frozen; default interval relative_local',cache_forward_error=float(abs(direct-cached).max()),cache_gradient_error=grad_error,limitation='One frozen stochastic C draw; same cache for all arms, not an independent draw generalization test')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    histories={w:[] for w in qs};gen=torch.Generator().manual_seed(91)
    completed=0
    def evaluate(step):
        with torch.no_grad():
            for w,q in qs.items():
                row={'epoch':step}
                for split,(h,p) in evals.items():
                    loss,terms=representation_loss(q,patterns,h,p,w,options)
                    row.update({split+'_'+k:float(v) for k,v in dict(total=loss,**terms).items()})
                histories[w].append(row)
        (output/'history.json').write_text(json.dumps(histories,indent=2)+'\n')
    evaluate(0)
    print('Cache ready, exact forward/gradient verified; elapsed %.1fs'%(time.monotonic()-start),flush=True)
    for step in range(1,epochs+1):
        if time.monotonic()-start>500:break
        ids=torch.randperm(len(tr.configurations),generator=gen)[:32].numpy();perm=torch.randperm(64,generator=gen).numpy()
        h,p=batch('train',ids,[perm[:32],perm[32:]])
        for w,q in qs.items():
            opts[w].zero_grad(set_to_none=True)
            loss,_=representation_loss(q,patterns,h,p,w,options);loss.backward();opts[w].step()
        completed=step
        if step%30==0 or step==epochs:
            evaluate(step);print(f'epoch {step}/{epochs}, elapsed {time.monotonic()-start:.1f}s',flush=True)
    if histories[10][-1]['epoch']!=completed:evaluate(completed)
    results={}
    for w,q in qs.items():
        model=copy.deepcopy(fresh);model.theory_encoder=q
        arm_metadata=copy.deepcopy(manifest)
        arm_metadata['config']['loss']['weights']['separation']=w
        arm_metadata['separation_weight']=w
        torch.save(dict(model=qcr_state_dict(model),metadata=arm_metadata,epoch=completed,optimizer=opts[w].state_dict()),output/f'weight{w}.pt')
        with torch.no_grad():
            gs={s:encode(q,patterns,h.mean(2)).numpy()[:,:,0] for s,(h,p) in evals.items()}
            halves={s:encode(q,patterns,h).numpy()[:,:,:,0] for s,(h,p) in evals.items()}
        np.savez_compressed(output/f'encodings_weight{w}.npz',**gs,train_labels=tr.couplings.numpy(),test_labels=te.couplings.numpy(),train_halves=halves['train'],test_halves=halves['test'])
        per_scale=[]
        for k in range(3):
            K=np.r_[tr.couplings.numpy().ravel(),te.couplings.numpy().ravel()];v=np.r_[gs['train'][:,k],gs['test'][:,k]]
            knots=np.unique(K);means=np.array([v[K==x].mean() for x in knots]);d=np.diff(means)
            per_scale.append(dict(g_min=float(v.min()),g_max=float(v.max()),increasing_segments=int((d>0).sum()),decreasing_segments=int((d<0).sum()),endpoint_center_means=[float(means[j]) for j in [0,len(knots)//2,-1]]))
        near=[]
        for k in [0,1]:
            inp=gs['test'][:,k];nxt=gs['test'][:,k+1];sd=gs['train'][:,k].std();out_sd=gs['train'][:,k+1].std()
            i,j=np.triu_indices(len(inp),1);gap=abs(inp[i]-inp[j]);future=abs(nxt[i]-nxt[j])
            noise=.25*(halves['test'][:,k+1,0]-halves['test'][:,k+1,1])**2
            close=gap<.05*sd;conflict=close&(future>np.maximum(.25*out_sd,2*np.sqrt(noise[i]+noise[j])))
            nearest=abs(gs['test'][:,k,None]-gs['train'][None,:,k]).argmin(1)
            rmse=np.sqrt(np.mean((nxt-gs['train'][nearest,k+1])**2))/max(out_sd,1e-12)
            near.append(dict(close_pairs=int(close.sum()),conflicting_pairs=int(conflict.sum()),nearest_train_successor_nrmse=float(rmse)))
        results[w]=dict(scales=per_scale,near_pair_diagnostic=near,last_loss=histories[w][-1])
    results['metadata']=dict(completed_epochs=completed,requested_epochs=epochs,elapsed_seconds=time.monotonic()-start,C_R_unchanged=True,note='Near-pair thresholds heuristic, not proof of nonclosure. R frozen in old coordinates, not refitted.')
    (output/'metrics.json').write_text(json.dumps(results,indent=2)+'\n')
    plot(output,qs,fresh,results,completed)
    print(json.dumps(results),flush=True)


def plot(output,qs,fresh,results,completed):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for col,w in enumerate(qs):
        z=np.load(output/f'encodings_weight{w}.npz');K=np.r_[z['train_labels'].ravel(),z['test_labels'].ravel()];g=np.concatenate([z['train'],z['test']]);kn=np.unique(K)
        for k in range(3):axs[0,col].plot(kn,[g[K==x,k].mean() for x in kn],label=f'k={k}')
        axs[0,col].set(title=f'Q-only, separation weight {w}',xlabel='K',ylabel='g');axs[0,col].legend()
        for k in [0,1]:axs[1,col].scatter(z['test'][:,k],z['test'][:,k+1],c=z['test_labels'].ravel(),cmap='coolwarm',vmin=-.9,vmax=.9,marker=['o','x'][k],label=f'{k} to {k+1}')
        axs[1,col].set(xlabel='g_k',ylabel='g_(k+1)',title='Direct cached successors, test');axs[1,col].legend()
    for ax in axs.flat:ax.grid(alpha=.2)
    fig.suptitle(f'Paired Q-only sweep, {completed} updates; no coupling supervision')
    fig.savefig(output/'Q_comparison.png',dpi=160);plt.close(fig)
    histories=json.loads((output/'history.json').read_text());fig,axs=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for w in qs:
        for split,style in [('train','-'),('test','--')]:
            rows=histories[str(w)]
            for ax,key in zip(axs,['geometry','anchor','consistency']):
                ax.plot([r['epoch'] for r in rows],[r[split+'_'+key] for r in rows],style,label=f'w={w} {split}');ax.set(title=key+' (unweighted)',xlabel='epoch',yscale='log')
    for ax in axs:ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.savefig(output/'loss.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    lo=min(results[w]['scales'][0]['g_min'] for w in qs);hi=max(results[w]['scales'][0]['g_max'] for w in qs)
    x=torch.linspace(lo,hi,400,dtype=torch.double)[:,None]
    with torch.no_grad():r=fresh.advance_theory(x).numpy().ravel()
    x=x.numpy().ravel();axs[0].plot(x,r);axs[0].plot(x,x,'k--');axs[1].plot(x,r-x);axs[1].axhline(0,color='k',ls='--')
    axs[0].set(xlabel='g',ylabel='R(g)');axs[1].set(xlabel='g',ylabel='R(g)-g')
    fig.suptitle('Frozen SOURCE R only: NOT calibrated to the newly trained Q; NOT a phase diagram')
    fig.savefig(output/'frozen_R_reference_only.png',dpi=150);plt.close(fig)
