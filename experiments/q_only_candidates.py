"""Paired label-free representation candidates on an unchanged frozen-C cache."""
import copy
import json
import time
from pathlib import Path
from dataclasses import replace
import numpy as np
import torch
from scipy.sparse.csgraph import shortest_path, connected_components
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr
from recurrent_rg.checkpoints import load_run,build_model,qcr_state_dict
from recurrent_rg.config import RunConfig
from recurrent_rg.data.ensembles import load_ensembles
from recurrent_rg.data.statistics import categorical_plaquette_distribution
from recurrent_rg.losses import distribution_geometry_loss,distribution_reconstruction_loss
from recurrent_rg.runner import file_fingerprint
from experiments.q_only_cached_sweep import sample_counts,encode

ARMS=['baseline','collision','ranking','future','manifold']
OPTIONS=dict(scale=4.,noise_multiplier=2.,mode='relative_local',relative_floor=.05,max_inverse_weight=100.,neighbors=4,local_fraction=.5)

def validate_frozen_levels(levels):
    if any(not torch.all(v.sum(-1)==1) or not torch.all((v==0)|(v==1)) for v in levels):
        raise ValueError('Invalid categorical cache: every site must be exactly one-hot')

def normalized_distances(g):
    d=torch.cdist(g,g)
    mask=~torch.eye(len(g),device=g.device,dtype=torch.bool)
    return d/(d[mask].square().mean()+1e-12).sqrt()

def data_distances(pa,pb):
    with torch.no_grad():
        p=.5*(pa+pb);noise=.25*(pa-pb).square().sum(-1)
        d=torch.cdist(p,p);allow=2*(noise[:,None]+noise[None,:]).sqrt()
    return d,allow

def collision_loss(g,pa,pb):
    dist,allow=data_distances(pa,pb);dg=normalized_distances(g)
    with torch.no_grad():
        eligible=torch.triu(dist-allow>.05,diagonal=1)
        selected=eligible&(dg.detach()<.5)
    return (.5-dg[selected]).relu().square().mean() if selected.any() else g.sum()*0

def ranking_loss(g,pa,pb):
    dist,allow=data_distances(pa,pb);dg=normalized_distances(g)
    with torch.no_grad():
        n=len(g);d=dist.clone();d.fill_diagonal_(float('inf'))
        pos=d.topk(min(4,n-1),largest=False).indices
        a=torch.arange(n)[:,None,None].expand(n,pos.shape[1],n)
        j=pos[:,:,None].expand_as(a);k=torch.arange(n)[None,None,:].expand_as(a)
        valid=(k!=a)&(k!=j)&(dist[a,k]-dist[a,j]>allow[a,k]+allow[a,j]+.05)
    residual=.2+dg[a,j]-dg[a,k]
    return residual[valid].relu().mean() if valid.any() else g.sum()*0

def graph_targets(train_p,test_p,neighbors=4):
    """Train-only graph; test nodes attach without adding training shortcuts."""
    x=train_p.detach().numpy();y=test_p.detach().numpy()
    d=np.linalg.norm(x[:,None]-x[None,:],axis=-1);np.fill_diagonal(d,np.inf)
    ids=np.argsort(d,axis=1)[:,:neighbors]
    edges=np.full_like(d,np.inf)
    for i in range(len(x)):edges[i,ids[i]]=np.maximum(d[i,ids[i]],1e-12)
    edges=np.minimum(edges,edges.T);np.fill_diagonal(edges,0)
    adj=csr_matrix(np.where(np.isfinite(edges),edges,0))
    components,_=connected_components(adj,directed=False)
    path=shortest_path(adj,directed=False)
    attach=np.linalg.norm(y[:,None]-x[None,:],axis=-1);near=np.argsort(attach,axis=1)[:,:neighbors]
    test=np.full((len(y),len(y)),np.inf)
    for i in range(len(y)):
        for j in range(len(y)):
            test[i,j]=np.min(attach[i,near[i]][:,None]+path[np.ix_(near[i],near[j])]+attach[j,near[j]][None,:])
    np.fill_diagonal(test,0)
    return torch.from_numpy(path),torch.from_numpy(test),dict(components=int(components),train_finite_fraction=float(np.isfinite(path).mean()),test_finite_fraction=float(np.isfinite(test).mean()))

def manifold_loss(g,target):
    mask=torch.triu(torch.isfinite(target),diagonal=1)
    if not mask.any():return g.sum()*0
    td=target[mask];td=td/(td.square().mean()+1e-12).sqrt()
    return (normalized_distances(g)[mask]-td).square().mean()

def objective(arm,q,aux,patterns,h,p,graph):
    g=encode(q,patterns,h);direct=g.mean(2)
    anchor=(g[:,0,0]-g[:,0,1]).square().sum(-1).mean()
    consistency=(g[:,1:,0]-g[:,1:,1]).square().sum(-1).mean()
    terms=[]
    for k in range(3):
        if arm=='baseline':term=distribution_geometry_loss(direct[:,k],p[:,k,0],p[:,k,1],**OPTIONS)[0]
        elif arm=='collision':term=collision_loss(direct[:,k],p[:,k,0],p[:,k,1])
        elif arm=='ranking':term=ranking_loss(direct[:,k],p[:,k,0],p[:,k,1])
        elif arm=='manifold':term=manifold_loss(direct[:,k],graph[k])
        else:continue
        terms.append(term)
    if arm=='future':
        # Equal weight for each available prediction: 0->1, 1->2, 0->2.
        terms=[distribution_reconstruction_loss(aux[str(r)],direct[:,k],p[:,k+r].mean(1)) for r in [1,2] for k in range(3-r)]
    representation=torch.stack(terms).mean()
    return 20*anchor+5*consistency+10*representation,dict(anchor=anchor,consistency=consistency,representation=representation)

def run(source,cache_source,output,epochs):
    start=time.monotonic();output=Path(output);output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);loaded=load_run(source);cfg=RunConfig.from_dict(loaded.config)
    if cfg.model.categorical_states!=3 or cfg.model.theory_dim!=1:raise ValueError('Scalar spin1 pilot only')
    cfg=replace(cfg,model=replace(cfg.model,theory_head='nonlinear',reconstruction_hidden=0),training=replace(cfg.training,epochs=epochs,train_modules=('Q',)))
    torch.manual_seed(91);fresh=build_model(cfg.model,2)
    fresh.rg_kernel.load_state_dict(loaded.model.rg_kernel.state_dict());fresh.theory_rg.load_state_dict(loaded.model.theory_rg.state_dict())
    qs={name:copy.deepcopy(fresh.theory_encoder) for name in ARMS};aux={name:torch.nn.ModuleDict() for name in ARMS}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(92)
        aux['future']=torch.nn.ModuleDict({str(r):torch.nn.Sequential(torch.nn.Linear(1,32),torch.nn.Tanh(),torch.nn.Linear(32,81)).double() for r in [1,2]})
    opts={name:torch.optim.Adam(list(qs[name].parameters())+list(aux[name].parameters()),lr=.001) for name in ARMS}
    tr,te=load_ensembles(cfg.data.cache);datasets={'train':tr,'test':te}
    cache_source=Path(cache_source);previous=json.loads((cache_source/'manifest.json').read_text())
    cache_file=cache_source/'frozen_levels.pt'
    if previous['data_sha256']!=file_fingerprint(cfg.data.cache) or previous['source_sha256']!=file_fingerprint(loaded.checkpoint) or previous['cached_levels_sha256']!=file_fingerprint(cache_file):
        raise ValueError('Cache/source fingerprint mismatch')
    levels=torch.load(cache_file,weights_only=True);cache={}
    for split,ls in levels.items():
        validate_frozen_levels(ls)
        torch.testing.assert_close(ls[0],datasets[split].configurations)
        counts=[];ps=[]
        for i in range(len(ls[0])):
            xs=[v[i:i+1].double() for v in ls]
            counts.append([sample_counts(x,fresh.geometry,3) for x in xs])
            ps.append(torch.stack([categorical_plaquette_distribution(x.squeeze(0).unsqueeze(1)) for x in xs]).numpy())
        cache[split]=(counts,np.stack(ps))
    del levels
    codes=(torch.arange(3**9)[:,None]//(3**torch.arange(9)))%3
    patterns=torch.nn.functional.one_hot(codes,3).flatten(1).double()
    def batch(split,ids,halves):
        counts,pl=cache[split]
        hist=np.array([[np.stack([np.asarray(counts[i][k][h].mean(0)).ravel() for h in halves]) for k in range(3)] for i in ids])
        p=np.stack([pl[ids][:,:,h].mean(2) for h in halves],axis=2)
        return torch.from_numpy(hist),torch.from_numpy(p)
    evals={s:batch(s,np.arange(len(ds.configurations)),[np.arange(32),np.arange(32,64)]) for s,ds in datasets.items()}
    graphs={'train':[],'test':[]};graph_info=[]
    for k in range(3):
        a,b,info=graph_targets(evals['train'][1][:,k].mean(1),evals['test'][1][:,k].mean(1))
        graphs['train'].append(a);graphs['test'].append(b);graph_info.append(info)
    manifest=dict(config=cfg.to_dict(),arms=ARMS,source=str(loaded.checkpoint),source_sha256=file_fingerprint(loaded.checkpoint),cache_path=str(cache_file.resolve()),cache_sha256=file_fingerprint(cache_file),data_sha256=file_fingerprint(cfg.data.cache),seed=91,labels_used_in_training=False,weights=dict(anchor=20,consistency=5,representation=10),collision=dict(data_resolved_margin=.05,normalized_latent_margin=.5,selection='detached near pairs only',normalization='differentiable RMS pair distance'),ranking=dict(neighbors=4,data_gap_margin=.05,latent_margin=.2,normalization='differentiable RMS pair distance'),future=dict(hidden=32,targets=['0->1','1->2','0->2'],auxiliary_parameters=sum(p.numel() for p in aux['future'].parameters()),same_head_for_same_horizon=True),manifold=dict(neighbors=4,graph='training data only, no connectivity repair; disconnected pairs excluded',diagnostics=graph_info),limitations='Fixed candidate hyperparameters, not matched gradient norms; no physical labels; one seed/one frozen-C cache; R not fitted')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    histories={name:[] for name in ARMS};gen=torch.Generator().manual_seed(91)
    def evaluate(step):
        with torch.no_grad():
            for name in ARMS:
                row=dict(epoch=step)
                for split,(h,p) in evals.items():
                    loss,terms=objective(name,qs[name],aux[name],patterns,h,p,graphs[split])
                    row.update({split+'_'+key:float(val) for key,val in dict(total=loss,**terms).items()})
                histories[name].append(row)
        (output/'history.json').write_text(json.dumps(histories,indent=2)+'\n')
    evaluate(0);completed=0
    if previous.get('reference_run'):
        old_history=json.loads((Path(previous['reference_run'])/'history.json').read_text())['10'][0]
        errors=[abs(histories['baseline'][0][s+'_representation']-old_history[s+'_geometry']) for s in ['train','test']]
        if max(errors)>1e-10:raise AssertionError(f'Initial baseline does not match original cache computation: {errors}')
        print('Original baseline initial geometry reproduced: '+str(errors),flush=True)
    print('Shared cache verified; graph diagnostics '+json.dumps(graph_info),flush=True)
    for step in range(1,epochs+1):
        if time.monotonic()-start>560:break
        ids=torch.randperm(len(tr.configurations),generator=gen)[:32].numpy();perm=torch.randperm(64,generator=gen).numpy()
        h,p=batch('train',ids,[perm[:32],perm[32:]]);graph=[g[ids][:,ids] for g in graphs['train']]
        for name in ARMS:
            opts[name].zero_grad(set_to_none=True);loss,_=objective(name,qs[name],aux[name],patterns,h,p,graph)
            if not torch.isfinite(loss):raise FloatingPointError(name)
            loss.backward();opts[name].step()
        completed=step
        if step%100==0 or step==epochs:
            evaluate(step);print(f'epoch {step}/{epochs}; elapsed {time.monotonic()-start:.1f}s',flush=True)
    if histories['baseline'][-1]['epoch']!=completed:evaluate(completed)
    results={}
    for name in ARMS:
        armcfg=copy.deepcopy(manifest);armcfg['objective']=name
        model=copy.deepcopy(fresh);model.theory_encoder=qs[name]
        torch.save(dict(model=qcr_state_dict(model),metadata=armcfg,epoch=completed,auxiliary=aux[name].state_dict(),optimizer=opts[name].state_dict()),output/f'{name}.pt')
        with torch.no_grad():
            g={s:encode(qs[name],patterns,h.mean(2)).numpy()[:,:,0] for s,(h,p) in evals.items()}
            halves={s:encode(qs[name],patterns,h).numpy()[:,:,:,0] for s,(h,p) in evals.items()}
        np.savez_compressed(output/f'{name}_encodings.npz',**g,train_labels=tr.couplings.numpy(),test_labels=te.couplings.numpy(),train_halves=halves['train'],test_halves=halves['test'])
        k=np.r_[tr.couplings.numpy().ravel(),te.couplings.numpy().ravel()];kn=np.unique(k);allg=np.concatenate([g['train'],g['test']]);means=np.array([allg[k==v].mean(0) for v in kn])
        overlap=[]
        for level in range(3):
            neg=means[kn<0,level];pos=means[kn>0,level]
            overlap.append(float(max(0,min(neg.max(),pos.max())-max(neg.min(),pos.min()))/max(1e-12,min(np.ptp(neg),np.ptp(pos)))))
        near=[]
        for level in [0,1]:
            ids=abs(g['test'][:,level,None]-g['train'][None,:,level]).argmin(1)
            sd=g['train'][:,level+1].std()
            near.append(float(np.sqrt(np.mean((g['test'][:,level+1]-g['train'][ids,level+1])**2))/max(sd,1e-12)))
        collisions=[]
        for level in range(3):
            pa,pb=evals['test'][1][:,level,0],evals['test'][1][:,level,1];dist,allow=data_distances(pa,pb)
            dg=abs(g['test'][:,level,None]-g['test'][None,:,level]);close=torch.from_numpy(dg)<.05*g['train'][:,level].std()
            eligible=torch.triu(dist-allow>.05,diagonal=1)
            collisions.append(dict(eligible=int(eligible.sum()),collisions=int((eligible&close).sum())))
        results[name]=dict(abs_spearman=[float(abs(spearmanr(kn,means[:,level]).statistic)) for level in range(3)],branch_overlap_fraction=overlap,micro_span=float(np.ptp(allg[:,0])),micro_sd=float(g['train'][:,0].std()),subset_noise_relative=float(np.sqrt(np.mean((halves['test'][:,0,0]-halves['test'][:,0,1])**2))/max(g['train'][:,0].std(),1e-12)),nearest_train_successor_nrmse=near,distribution_collisions=collisions,last_loss=histories[name][-1])
    # Compare matched baseline to the previous weight-10 run when lengths match.
    if completed==2400 and (cache_source/'weight10.pt').exists():
        old=torch.load(cache_source/'weight10.pt',weights_only=True)['model'];new=qcr_state_dict(qs['baseline'])
        results['baseline_reproduction_max_Q_difference']=max(float((v-old['theory_encoder.'+key]).abs().max()) for key,v in new.items())
    results['metadata']=dict(completed_epochs=completed,requested_epochs=epochs,elapsed_seconds=time.monotonic()-start,near_metrics='Nearest-neighbor predictor, not trained R closure; sparse sampling can hide branch conflicts',C_R_unchanged=True)
    (output/'metrics.json').write_text(json.dumps(results,indent=2)+'\n')
    plot(output,results,fresh,completed)
    print(json.dumps(results),flush=True)

def plot(output,results,fresh,completed):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for ax,name in zip(axs.flat,ARMS):
        z=np.load(output/f'{name}_encodings.npz');K=np.r_[z['train_labels'].ravel(),z['test_labels'].ravel()];g=np.concatenate([z['train'],z['test']]);kn=np.unique(K)
        for level in range(3):ax.plot(kn,[g[K==k,level].mean() for k in kn],label=f'k={level}')
        ax.set(title=name,xlabel='K (evaluation only)',ylabel='g');ax.legend();ax.grid(alpha=.2)
    axs.flat[-1].axis('off');axs.flat[-1].text(.05,.5,'Same frozen C cache / Q initialization\n2400 requested updates / seed 91\nNo K supervision; R not fitted\nFuture arm adds 5474 decoder parameters',va='center')
    fig.suptitle(f'Q-only loss candidates, actual updates: {completed}')
    fig.savefig(output/'Q_comparison.png',dpi=160);plt.close(fig)
    h=json.loads((output/'history.json').read_text());fig,axs=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for ax,name in zip(axs.flat,ARMS):
        for split,ls in [('train','-'),('test','--')]:
            for key,color in [('representation','C0'),('anchor','C1'),('consistency','C2')]:
                ax.plot([r['epoch'] for r in h[name]],[r[split+'_'+key] for r in h[name]],ls,c=color,label=split+' '+key)
        ax.set(title=name,xlabel='epoch',ylabel='unweighted loss (log scale)',yscale='log');ax.legend(fontsize=6);ax.grid(alpha=.2)
    axs.flat[-1].axis('off');fig.suptitle('Loss units differ by objective; lower curves are not cross-arm rankings')
    fig.savefig(output/'loss.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(1,5,figsize=(17,3.5),layout='constrained')
    for ax,name in zip(axs,ARMS):
        z=np.load(output/f'{name}_encodings.npz')
        for k,mark in [(0,'o'),(1,'x')]:ax.scatter(z['test'][:,k],z['test'][:,k+1],c=z['test_labels'].ravel(),cmap='coolwarm',vmin=-.9,vmax=.9,marker=mark,label=f'{k}->{k+1}')
        ax.set(title=name,xlabel='g_k',ylabel='g_(k+1)');ax.legend(fontsize=7)
    fig.savefig(output/'direct_successors.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained');x=torch.linspace(-5,5,400,dtype=torch.double)[:,None]
    with torch.no_grad():r=fresh.advance_theory(x).numpy().ravel()
    x=x.numpy().ravel();axs[0].plot(x,r);axs[0].plot(x,x,'k--');axs[1].plot(x,r-x);axs[1].axhline(0,c='k',ls='--')
    axs[0].set(xlabel='g',ylabel='R(g)');axs[1].set(xlabel='g',ylabel='R(g)-g')
    fig.suptitle('Frozen SOURCE R; NOT calibrated to any new Q; NOT a phase diagram')
    fig.savefig(output/'frozen_R_reference_only.png',dpi=150);plt.close(fig)
