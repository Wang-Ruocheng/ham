"""Frozen tail50 local drift audit; reuse existing repeated-encoding evidence."""
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from recurrent_rg.checkpoints import load_run

def main():
    torch.set_num_threads(2)
    run=Path('results/spin1_geometry_ablation_seed91/tail50_top10')
    out=run/'local_drift_audit';out.mkdir(exist_ok=True)
    model=load_run(run).model
    z=np.load(run/'closure_conflicts/repeated_encodings.npz')
    q=z['fixed_cache_full'];bs=z['bootstrap_cache_full'];K=z['labels'];mean=q.mean(0)
    grid=np.linspace(-.5,.8,1001)
    with torch.no_grad():
        drift=model.advance_theory(torch.from_numpy(grid[:,None])).numpy().ravel()-grid
        pred=model.advance_theory(torch.from_numpy(q[:,:,:-1,None])).numpy()[...,0]
    residual=pred-q[:,:,1:]
    fig,axes=plt.subplots(1,3,figsize=(16,4.5),layout='constrained')
    report={'definition':'literal full-ensemble Q; 12 C draws; bootstrap is within-cache sensitivity, not MCMC confidence',
            'valley':{'g':float(grid[drift.argmin()]),'drift':float(drift.min())},'scales':{}}
    for k,ax in enumerate(axes[:2]):
        delta=q[:,:,k+1]-q[:,:,k];d=delta.mean(0);sd=delta.std(0,ddof=1)
        for mask,color,label in [(K<0,'tab:blue','K<0'),(K>=0,'tab:red','K>=0')]:
            ax.errorbar(mean[mask,k],d[mask],yerr=sd[mask],fmt='o',ms=4,color=color,alpha=.7,label=label)
        ax.plot(grid,drift,'k-',label='R(g)-g');ax.axhline(0,c='grey',ls=':')
        ax.set(xlim=(-.5,.8),xlabel=f'g_{k}',ylabel='Next g - current g',title=f'Local drift, k={k}');ax.legend()
        rows=[]
        for i in np.where((mean[:,k]>=-.5)&(mean[:,k]<=.8))[0]:
            rows.append(dict(ensemble=int(i),split=str(z['splits'][i]),K=float(K[i]),g=float(mean[i,k]),
                direct_drift=float(d[i]),C_draw_sd=float(sd[i]),
                bootstrap_drift_sd=float((bs[:,i,k+1]-bs[:,i,k]).std(ddof=1)),
                R_drift=float((pred[:,i,k]-q[:,i,k]).mean())))
        report['scales'][k]={'local_points':rows,'all_rmse':float(np.sqrt((residual[:,:,k]**2).mean()))}
    ax=axes[2]
    for k in range(2):
        for mask,color in [(K<0,'tab:blue'),(K>=0,'tab:red')]:
            ax.scatter(mean[mask,k],mean[mask,k+1]-mean[mask,k],c=color,marker='o' if k==0 else '^',s=20,alpha=.6)
    wide=np.linspace(mean[:,:-1].min()-.1,mean[:,:-1].max()+.1,1000)
    with torch.no_grad():v=model.advance_theory(torch.from_numpy(wide[:,None])).numpy().ravel()-wide
    ax.plot(wide,v,'k-');ax.axhline(0,c='grey',ls=':');ax.set(title='Both scales, full input range',xlabel='g',ylabel='Drift')
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('Tail mix 0.5: direct drift vs learned R; bars = C-draw SD, not sampling uncertainty')
    fig.savefig(out/'drift_comparison.png',dpi=170)
    (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
