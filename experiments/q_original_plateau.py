"""Continue original full Q with unchanged representation loss until a measured plateau."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from recurrent_rg.checkpoints import load_run,qcr_state_dict
from recurrent_rg.data.ensembles import load_ensembles
from recurrent_rg.data.statistics import categorical_plaquette_distribution
from recurrent_rg.runner import file_fingerprint
from experiments.q_only_cached_sweep import sample_counts,encode,representation_loss
from experiments.q_only_candidates import OPTIONS,validate_frozen_levels


def plateau_reached(rows,lr):
    if lr>1e-5 or len(rows)<9:return False
    tail=rows[-9:]
    if tail[-1]['epoch']-tail[0]['epoch']<2000:return False
    if any(row.get('lr',float('inf'))>1e-5 for row in tail):return False
    first=tail[0]['best_train'];last=tail[-1]['best_train']
    return (first-last)/max(abs(first),1e-12)<.005


def run(source,cache_source,output,epochs):
    start=time.monotonic();torch.set_num_threads(2)
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    loaded=load_run(source);model=loaded.model;q=model.theory_encoder
    frozen={k:v.clone() for k,v in qcr_state_dict(model).items() if not k.startswith('theory_encoder.')}
    for p in model.parameters():p.requires_grad_(False)
    for p in q.parameters():p.requires_grad_(True)
    if loaded.model_config.theory_head!='nonlinear' or loaded.model_config.theory_dim!=1:raise ValueError('Original scalar nonlinear Q required')
    initial_epoch=int(loaded.payload['epoch'])
    optimizer=torch.optim.Adam(q.parameters(),lr=.001);optimizer.load_state_dict(loaded.payload['optimizer'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.3,patience=3,threshold=.002,threshold_mode='rel',min_lr=1e-6)
    previous_history=json.loads((Path(source).parent/'history.json').read_text())
    continuation=isinstance(previous_history,list)
    if continuation:scheduler.load_state_dict(loaded.payload['scheduler'])
    tr,te=load_ensembles(loaded.config['data']['cache']);datasets=dict(train=tr,test=te)
    cachefile=Path(cache_source)/'frozen_levels.pt';levels=torch.load(cachefile,weights_only=True);cache={}
    for split,ls in levels.items():
        validate_frozen_levels(ls);torch.testing.assert_close(ls[0],datasets[split].configurations)
        counts=[];ps=[]
        for i in range(len(ls[0])):
            xs=[v[i:i+1].double() for v in ls]
            counts.append([sample_counts(x,model.geometry,3) for x in xs]);ps.append(torch.stack([categorical_plaquette_distribution(x.squeeze(0).unsqueeze(1)) for x in xs]).numpy())
        cache[split]=(counts,np.stack(ps))
    del levels
    codes=(torch.arange(3**9)[:,None]//(3**torch.arange(9)))%3;patterns=torch.nn.functional.one_hot(codes,3).flatten(1).double()
    def batch(split,ids,halves):
        counts,p=cache[split]
        h=np.array([[np.stack([np.asarray(counts[i][k][half].mean(0)).ravel() for half in halves]) for k in range(3)] for i in ids])
        return torch.from_numpy(h),torch.from_numpy(np.stack([p[ids][:,:,half].mean(2) for half in halves],axis=2))
    evals={s:batch(s,np.arange(len(d.configurations)),[np.arange(32),np.arange(32,64)]) for s,d in datasets.items()}
    gen=torch.Generator().manual_seed(91)
    if continuation:gen.set_state(loaded.payload['batch_generator_state'])
    else:
        for _ in range(initial_epoch):torch.randperm(len(tr.configurations),generator=gen);torch.randperm(64,generator=gen)
    history=previous_history[:-1] if continuation else []
    best=min(row['best_train'] for row in previous_history) if continuation else float('inf')
    snapshots={}
    if continuation:
        old_snapshots=np.load(Path(source).parent/'encodings_history.npz')
        snapshots={key:old_snapshots[key] for key in old_snapshots.files if key!='K'}
    def evaluate(epoch,do_schedule=True):
        nonlocal best
        row=dict(epoch=epoch)
        with torch.no_grad():
            for split,(h,p) in evals.items():
                value,terms=representation_loss(q,patterns,h,p,10,OPTIONS)
                row[split]=dict(total=float(value),**{k:float(v) for k,v in terms.items()})
            g=encode(q,patterns,evals['test'][0].mean(2)).numpy()[:,:,0]
        if row['train']['total']<best:
            best=row['train']['total'];save(epoch,'best_train.pt')
        row['best_train']=best
        if do_schedule:scheduler.step(row['train']['total'])
        row['lr']=optimizer.param_groups[0]['lr'];history.append(row)
        snapshots[str(epoch)]=g
        (out/'history.json').write_text(json.dumps(history,indent=2))
        save(epoch,'latest.pt')
        print(f"epoch={epoch} train={row['train']['total']:.7f} test={row['test']['total']:.7f} lr={row['lr']:.2g} elapsed={time.monotonic()-start:.1f}s",flush=True)
    metadata=dict(config=loaded.config,source=str(source),source_sha256=file_fingerprint(loaded.checkpoint),cache_sha256=file_fingerprint(cachefile),objective='Original Q-only: 20 micro subset +5 coarse subset +10 relative_local geometry; all Q parameters trainable; C/R frozen',system_batch=32,seed=91,labels_used_in_training=False,optimizer='Original Adam moments resumed; original batch RNG replayed; new train-only ReduceLROnPlateau schedule',plateau='At lr<=1e-5, best fixed train-evaluation loss improves <0.5% over last 2000 updates',scope='Q-only; original R is not recalibrated to new coordinates')
    def save(epoch,name):
        torch.save(dict(model=qcr_state_dict(model),metadata=metadata,epoch=epoch,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),batch_generator_state=gen.get_state()),out/name)
    (out/'manifest.json').write_text(json.dumps(metadata,indent=2))
    evaluate(initial_epoch,False)
    old=previous_history[-1] if continuation else previous_history['10'][-1]
    for split in ['train','test']:
        expected=old[split]['total'] if continuation else old[split+'_total']
        if abs(history[-1][split]['total']-expected)>1e-9:raise AssertionError('Resumed Q loss does not match original valid run')
    if not continuation:scheduler.step(history[-1]['train']['total'])
    completed=initial_epoch;reason='step_limit'
    for epoch in range(initial_epoch+1,epochs+1):
        ids=torch.randperm(len(tr.configurations),generator=gen)[:32].numpy();perm=torch.randperm(64,generator=gen).numpy()
        h,p=batch('train',ids,[perm[:32],perm[32:]])
        optimizer.zero_grad(set_to_none=True);value,_=representation_loss(q,patterns,h,p,10,OPTIONS)
        if not torch.isfinite(value):raise FloatingPointError('Nonfinite Q loss')
        value.backward();optimizer.step();completed=epoch
        if (epoch-initial_epoch)%250==0:
            evaluate(epoch)
            if plateau_reached(history,optimizer.param_groups[0]['lr']):reason='training_plateau';break
    if history[-1]['epoch']!=completed:evaluate(completed)
    for k,v in frozen.items():torch.testing.assert_close(qcr_state_dict(model)[k],v,atol=0,rtol=0)
    ks={s:d.couplings.numpy().reshape(len(d.configurations),-1)[:,0] for s,d in datasets.items()};k=ks['test'];g=snapshots[str(completed)]
    per_scale=[]
    for j in range(3):
        unique=np.unique(k);means=np.array([g[k==v,j].mean() for v in unique]);delta=np.diff(means)
        per_scale.append(dict(abs_spearman=float(abs(spearmanr(k,g[:,j]).statistic)),reverse_variation_fraction=float(min(np.maximum(delta,0).sum(),np.maximum(-delta,0).sum())/max(abs(delta).sum(),1e-12))))
    elapsed=time.monotonic()-start
    prior_report=json.loads((Path(source).parent/'metrics.json').read_text()) if continuation else {}
    cumulative=elapsed+prior_report.get('cumulative_elapsed_seconds',prior_report.get('elapsed_seconds',0.))
    report=dict(initial_epoch=history[0]['epoch'],resumed_epoch=initial_epoch,final_epoch=completed,reason=reason,elapsed_seconds=elapsed,cumulative_elapsed_seconds=cumulative,initial=history[0],final=history[-1],scales=per_scale,C_R_unchanged=True,limitations='One seed and one frozen C cache; operational local training plateau, not proof of global optimum; fixed evaluation subsets; no R refit')
    (out/'metrics.json').write_text(json.dumps(report,indent=2));np.savez(out/'encodings_history.npz',K=k,**snapshots)
    plot(out,history,snapshots,k)
    print(json.dumps(report,indent=2),flush=True)


def plot(out,history,snapshots,k):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4));unique=np.unique(k)
    picks=[history[0]['epoch'],history[len(history)//2]['epoch'],history[-1]['epoch']]
    for ax,epoch in zip(axes,picks):
        g=snapshots[str(epoch)]
        for j in range(3):ax.plot(unique,[g[k==v,j].mean() for v in unique],label=f'k={j}')
        ax.set(title=f'epoch {epoch}',xlabel='K (evaluation only)',ylabel='g');ax.legend()
    fig.tight_layout();fig.savefig(out/'K_to_g.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(16,4))
    for ax,key in zip(axes[:3],['total','geometry','anchor']):
        for s,style in [('train','-'),('test','--')]:ax.plot([r['epoch'] for r in history],[r[s][key] for r in history],style,label=s)
        ax.set(title=key,xlabel='total Q updates');ax.legend()
    axes[3].step([r['epoch'] for r in history],[r['lr'] for r in history],where='post');axes[3].set(title='learning rate',yscale='log',xlabel='total Q updates')
    fig.tight_layout();fig.savefig(out/'loss.png',dpi=160);plt.close(fig)
