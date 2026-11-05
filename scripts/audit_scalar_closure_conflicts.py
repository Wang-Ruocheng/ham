"""Read-only scalar checkpoint audit: near-identical inputs versus next-scale targets.

Repeated C draws isolate kernel randomness; iid within-cache bootstrap is a
sensitivity check, not a confidence interval for correlated MCMC samples.
No model fitting or coupling supervision is performed.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from recurrent_rg.checkpoints import load_run
from recurrent_rg.data.ensembles import load_ensembles


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--checkpoint',type=int)
    p.add_argument('--repeats',type=int,default=12)
    a=p.parse_args();torch.set_num_threads(2)
    r=load_run(a.run,a.checkpoint);m=r.model
    if r.model_config.theory_dim!=1:raise ValueError('Scalar Q only')
    train,test=load_ensembles(r.config['data']['cache'])
    data=torch.cat([train.configurations,test.configurations])
    labels=torch.cat([train.couplings,test.couplings]).numpy().ravel()
    splits=np.array(['train']*len(train.configurations)+['test']*len(test.configurations))
    out=a.run/('closure_conflicts' if a.checkpoint is None else f'closure_conflicts_{a.checkpoint}')
    out.mkdir(exist_ok=True)
    results={}
    for mode in ['fixed_cache','bootstrap_cache']:
        direct,full=[],[]
        with torch.random.fork_rng(devices=[]),torch.no_grad():
            for rep in range(a.repeats):
                torch.manual_seed(71900+rep)
                gen=torch.Generator().manual_seed(81900+rep)
                ds,fs=[],[]
                for raw in data.split(8):
                    if mode=='bootstrap_cache':
                        idx=torch.randint(raw.shape[1],(len(raw),raw.shape[1]),generator=gen)
                        raw=raw[torch.arange(len(raw))[:,None],idx]
                    levels=m.coarse_grain(raw.double())
                    half=raw.shape[1]//2
                    qs=[.5*(m.theory_encoder(x[:,:half])+m.theory_encoder(x[:,half:])) for x in levels]
                    ds.append(torch.stack(qs,1)[...,0].numpy())
                    fs.append(torch.stack([m.theory_encoder(x) for x in levels],1)[...,0].numpy())
                direct.append(np.concatenate(ds));full.append(np.concatenate(fs))
                print(f'{mode}: {rep+1}/{a.repeats}',flush=True)
        results[mode]=np.stack(direct)
        results[mode+'_full']=np.stack(full)
    np.savez_compressed(out/'repeated_encodings.npz',labels=labels,splits=splits,**results)
    z=results['fixed_cache'];mean=z.mean(0);noise=z.std(0,ddof=1)
    bs=results['bootstrap_cache'].std(0,ddof=1)
    n,levels=mean.shape;steps=levels-1
    x=mean[:,:-1].ravel();y=mean[:,1:].ravel()
    entity=np.repeat(np.arange(n),steps);scale=np.tile(np.arange(steps),n)
    xsd=float(x.std());ysd=float(y.std());threshold=.02*xsd
    with torch.no_grad():pred=m.advance_theory(torch.from_numpy(x[:,None])).numpy().ravel()
    groups={}
    for name in ['same_scale_0','same_scale_1','cross_scale']:
        candidates=[]
        for i in range(len(x)):
            for j in range(i+1,len(x)):
                ei,ej=entity[i],entity[j];ki,kj=scale[i],scale[j]
                if name=='cross_scale':
                    if ki==kj:continue
                elif not ki==kj==int(name[-1]):continue
                if labels[ei]==labels[ej]:continue
                dx=abs(x[i]-x[j]);dy=abs(y[i]-y[j])
                if dx>threshold:continue
                diff=z[:,ei,ki+1]-z[:,ej,kj+1]
                sd=float(diff.std(ddof=1))
                def point(e,k,idx):
                    return dict(ensemble=int(e),split=str(splits[e]),K=float(labels[e]),k=int(k),
                        g=float(x[idx]),next_g=float(y[idx]),R_g=float(pred[idx]),
                        C_draw_sd_g=float(noise[e,k]),C_draw_sd_next=float(noise[e,k+1]),
                        bootstrap_sd_g=float(bs[e,k]),bootstrap_sd_next=float(bs[e,k+1]))
                candidates.append(dict(a=point(ei,ki,i),b=point(ej,kj,j),input_gap=float(dx),target_gap=float(dy),
                    required_secant_slope=float(dy/max(dx,1e-15)),
                    target_difference_draw_sd=sd,target_gap_over_draw_sd=float(dy/max(sd,1e-15)),
                    min_abs_target_difference_across_draws=float(abs(diff).min())))
        robust=[c for c in candidates if c['target_gap']>.1*ysd and c['target_gap_over_draw_sd']>3]
        robust.sort(key=lambda c:c['target_gap'],reverse=True)
        groups[name]=dict(near_input_pairs=len(candidates),large_gap_pairs=len(robust),examples=robust[:4])
    report=dict(checkpoint=str(r.checkpoint),repeats=a.repeats,representation='Mean of two half-ensemble Q encodings, matching training',
        input_sd=xsd,target_sd=ysd,near_input_threshold=threshold,large_target_threshold=.1*ysd,
        target_draw_noise_threshold='Target gap > 3 times SD of pairwise target differences across independent C draws',
        groups=groups,one_step_R_rmse_by_scale=np.sqrt(((pred.reshape(n,steps)-mean[:,1:])**2).mean(0)).tolist(),
        full_vs_split_mean_rmse=float(np.sqrt(((results['fixed_cache_full']-z)**2).mean())),
        limitations=['Near inputs are not exact equalities. Finite scatter cannot prove no deterministic interpolant exists.',
            'Bootstrap is iid within stored MCMC cache; ignores temporal autocorrelation and is sensitivity only.',
            'Examples selected by largest target gap among predeclared near-input pairs; no new training or kernel changes.'])
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(17,5),layout='constrained')
    grid=np.linspace(x.min()-.15,x.max()+.15,1200)
    with torch.no_grad():mapped=m.advance_theory(torch.from_numpy(grid[:,None])).numpy().ravel()
    for ax,ks,title in zip(axes,[[0],[1],[0,1]],['k=0 to 1','k=1 to 2','Both scales (shared R)']):
        for k in ks:
            for negative,color in [(True,'tab:blue'),(False,'tab:red')]:
                mask=labels<0 if negative else labels>=0
                ax.errorbar(mean[mask,k],mean[mask,k+1],xerr=noise[mask,k],yerr=noise[mask,k+1],
                    fmt='o' if k==0 else '^',ms=4,alpha=.65,color=color,
                    label=f'k={k}, K'+('<0' if negative else '>=0'))
        ax.plot(grid,mapped,'k-',lw=1.3,label='Learned R(g)');ax.plot(grid,grid,'k:',alpha=.4,label='Identity')
        ax.set(xlabel='Input g_k',ylabel='Target g_(k+1)',title=title);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Frozen checkpoint; error bars: SD of 12 C draws on fixed ensembles (not MCMC uncertainty)')
    fig.savefig(out/'input_target_branches.png',dpi=170);plt.close(fig)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
