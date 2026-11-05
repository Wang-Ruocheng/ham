"""Paired short Q-only initialization comparison, no coupling supervision."""
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from recurrent_rg.checkpoints import load_run,build_model,qcr_state_dict
from recurrent_rg.data.ensembles import load_ensembles
from recurrent_rg.data.statistics import categorical_plaquette_distribution
from recurrent_rg.runner import file_fingerprint
from experiments.q_only_cached_sweep import sample_counts,encode,representation_loss
from experiments.q_only_candidates import OPTIONS,validate_frozen_levels


def initialize(q,kind,pooled=None):
    if kind in ('orthogonal','xavier_tanh'):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(91)
            for layer in q.modules():
                if isinstance(layer,torch.nn.Linear):
                    if kind=='orthogonal':torch.nn.init.orthogonal_(layer.weight,gain=1.)
                    else:torch.nn.init.xavier_uniform_(layer.weight,gain=5/3 if layer is not q.head[-1] else 1.)
                    torch.nn.init.zeros_(layer.bias)
    if kind=='covariance_aware':
        if pooled is None:raise ValueError('Train-only pooled features required')
        mean=pooled.mean(0);z=pooled-mean;eig,V=torch.linalg.eigh(z.T@z/len(z))
        W=(V*(eig+.001*eig.max()).rsqrt())@V.T
        with torch.no_grad():
            q.head[0].weight.copy_(q.head[0].weight@W)
            q.head[0].bias.sub_(q.head[0].weight@mean)


def calibrate_output(q,patterns,hist,target_std=.1):
    with torch.no_grad():
        g=encode(q,patterns,hist);mean=g.mean();sd=g.std(unbiased=False)
        if sd<1e-10:raise ValueError('Degenerate initial output')
        factor=target_std/sd;q.head[-1].weight.mul_(factor);q.head[-1].bias.copy_((q.head[-1].bias-mean)*factor)
        result=encode(q,patterns,hist)
        torch.testing.assert_close(result.mean(),torch.zeros((),dtype=torch.double),atol=1e-9,rtol=0)
        torch.testing.assert_close(result.std(unbiased=False),torch.tensor(target_std,dtype=torch.double),atol=1e-9,rtol=0)
        return dict(original_mean=float(mean),original_std=float(sd),scale=float(factor))


def run(source,cache_source,output,epochs):
    start=time.monotonic();torch.set_num_threads(2)
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    loaded=load_run(source);torch.manual_seed(91);fresh=build_model(loaded.model_config,2)
    fresh.rg_kernel.load_state_dict(loaded.model.rg_kernel.state_dict());fresh.theory_rg.load_state_dict(loaded.model.theory_rg.state_dict())
    for p in fresh.parameters():p.requires_grad_(False)
    base=copy.deepcopy(fresh.theory_encoder)
    for p in base.parameters():p.requires_grad_(True)
    tr,te=load_ensembles(loaded.config['data']['cache']);datasets=dict(train=tr,test=te)
    cachefile=Path(cache_source)/'frozen_levels.pt';levels=torch.load(cachefile,weights_only=True);cache={}
    for split,ls in levels.items():
        validate_frozen_levels(ls);torch.testing.assert_close(ls[0],datasets[split].configurations)
        counts=[];ps=[]
        for i in range(len(ls[0])):
            xs=[v[i:i+1].double() for v in ls];counts.append([sample_counts(x,fresh.geometry,3) for x in xs]);ps.append(torch.stack([categorical_plaquette_distribution(x.squeeze(0).unsqueeze(1)) for x in xs]).numpy())
        cache[split]=(counts,np.stack(ps))
    codes=(torch.arange(3**9)[:,None]//(3**torch.arange(9)))%3;patterns=torch.nn.functional.one_hot(codes,3).flatten(1).double()
    def batch(split,ids,halves):
        counts,p=cache[split]
        h=np.array([[np.stack([np.asarray(counts[i][k][half].mean(0)).ravel() for half in halves]) for k in range(3)] for i in ids])
        return torch.from_numpy(h),torch.from_numpy(np.stack([p[ids][:,:,half].mean(2) for half in halves],axis=2))
    evals={s:batch(s,np.arange(len(d.configurations)),[np.arange(32),np.arange(32,64)]) for s,d in datasets.items()}
    with torch.no_grad():pooled=(evals['train'][0].mean(2)@base.local_mlp(patterns)).reshape(-1,12)
    arms=['default','calibrated','orthogonal','xavier_tanh','covariance_aware'];qs={};calibration={}
    for name in arms:
        q=copy.deepcopy(base);initialize(q,name,pooled)
        if name!='default':calibration[name]=calibrate_output(q,patterns,evals['train'][0][:,0].mean(1))
        qs[name]=q
    opts={n:torch.optim.Adam(q.parameters(),lr=.001) for n,q in qs.items()};history={n:[] for n in arms};snapshots={}
    gen=torch.Generator().manual_seed(91)
    def evaluate(step):
        with torch.no_grad():
            for name,q in qs.items():
                row=dict(epoch=step)
                for split,(h,p) in evals.items():
                    value,terms=representation_loss(q,patterns,h,p,10,OPTIONS)
                    row[split]=dict(total=float(value),**{k:float(v) for k,v in terms.items()})
                history[name].append(row)
                snapshots[f'{name}_{step}']=encode(q,patterns,evals['test'][0].mean(2)).numpy()[:,:,0]
        (out/'history.json').write_text(json.dumps(history,indent=2))
    evaluate(0)
    previous=json.loads((Path(source).parent/'history.json').read_text())['baseline'][0]
    for split in ['train','test']:
        if abs(history['default'][0][split]['total']-previous[split+'_total'])>1e-9:raise AssertionError('Default initialization did not reproduce')
    print('Default initialization matched; 5 paired full-Q arms; no labels in initialization.',flush=True)
    for step in range(1,epochs+1):
        ids=torch.randperm(len(tr.configurations),generator=gen)[:32].numpy();perm=torch.randperm(64,generator=gen).numpy();h,p=batch('train',ids,[perm[:32],perm[32:]])
        for name,q in qs.items():
            opts[name].zero_grad(set_to_none=True);value,_=representation_loss(q,patterns,h,p,10,OPTIONS)
            if not torch.isfinite(value):raise FloatingPointError(name)
            value.backward();opts[name].step()
        if step%100==0 or step==epochs:evaluate(step);print(f'{step}/{epochs}; elapsed={time.monotonic()-start:.1f}s',flush=True)
    k=te.couplings.numpy().reshape(len(te.configurations),-1)[:,0];unique=np.unique(k);results={}
    for name,q in qs.items():
        vals={}
        for step in [0,epochs]:
            g=snapshots[f'{name}_{step}'];scales=[]
            for j in range(3):
                means=np.array([g[k==v,j].mean() for v in unique]);delta=np.diff(means)
                scales.append(dict(abs_spearman_K=float(abs(spearmanr(k,g[:,j]).statistic)),reverse_variation_fraction=float(min(np.maximum(delta,0).sum(),np.maximum(-delta,0).sum())/max(abs(delta).sum(),1e-12)),g_std=float(g[:,j].std())))
            vals[str(step)]=scales
        result_model=copy.deepcopy(fresh);result_model.theory_encoder=q
        for key,value in qcr_state_dict(fresh).items():
            if not key.startswith('theory_encoder.'):torch.testing.assert_close(qcr_state_dict(result_model)[key],value,atol=0,rtol=0)
        torch.save(dict(model=qcr_state_dict(result_model),metadata=dict(config=loaded.config,initialization=name,seed=91,labels_used_in_training=False),optimizer=opts[name].state_dict(),epoch=epochs),out/f'{name}.pt')
        results[name]=dict(scales=vals,initial_loss=history[name][0],final_loss=history[name][-1])
    report=dict(steps=epochs,elapsed_seconds=time.monotonic()-start,source=str(source),source_sha256=file_fingerprint(loaded.checkpoint),cache_sha256=file_fingerprint(cachefile),calibration=calibration,results=results,protocol='All original Q parameters trained; identical seed91/batches/Adam lr0.001/original Q loss; same frozen C/R and exact categorical cache; no labels used in training or initialization; non-default arms train-micro g std calibrated to0.1; covariance aware uses train ensembles all scales only, folded into first head affine weights; no architecture change',limitations='Single seed, short training; equal output scale does not equalize all gradients or condition numbers; R frozen in unrelated latent coordinates, not an RG phase diagram')
    (out/'metrics.json').write_text(json.dumps(report,indent=2));np.savez(out/'encodings_history.npz',K=k,**snapshots)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,5,figsize=(20,7))
    for col,name in enumerate(arms):
        for row,step in enumerate([0,epochs]):
            g=snapshots[f'{name}_{step}']
            for j in range(3):axes[row,col].plot(unique,[g[k==v,j].mean() for v in unique],label=f'k={j}')
            axes[row,col].set(title=f'{name}, step={step}',xlabel='K (evaluation only)',ylabel='g');axes[row,col].legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'initial_final_g.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for name in arms:
        for ax,key in zip(axes,['total','geometry','anchor']):
            for split,style in [('train','-'),('test','--')]:ax.plot([r['epoch'] for r in history[name]],[r[split][key] for r in history[name]],style,label=f'{name} {split}')
            ax.set(title=key,xlabel='updates',yscale='log')
    axes[0].legend(fontsize=6);fig.tight_layout();fig.savefig(out/'loss.png',dpi=160);plt.close(fig)
    print(json.dumps(report,indent=2),flush=True)
