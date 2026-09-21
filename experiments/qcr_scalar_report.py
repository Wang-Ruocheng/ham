"""Standard scalar-QCR reports: learned R, coupling coordinates and closure."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.optimize import brentq
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments._utils import balanced_views
from recurrent_rg.model_nd import RecurrentRGNetwork1D, RecurrentRGNetwork2D


def scalar_report(run,cache,dimension=1,checkpoint_steps=None):
    run,cache=Path(run),Path(cache)
    summary=json.loads((run/'summary.json').read_text())
    cfg=summary['config']
    if cfg['theory_dim']!=1: return
    torch.set_num_threads(min(cfg.get('threads',2),2))
    tx,th,vx,vh=torch.load(cache,weights_only=True)
    raw,h=torch.cat([tx,vx]),torch.cat([th,vh]).numpy()
    knots=np.unique(h)
    label='h/J' if 'time_slices' in cfg else 'K = beta J'
    steps=sorted(set([s for s in (checkpoint_steps if checkpoint_steps is not None else [1000,cfg['epochs']]) if s<=cfg['epochs']
                     and (s==cfg['epochs'] or (run/f'training_step_{s}.pt').exists())]))
    results={}
    for step in steps:
        checkpoint=run/f'training_step_{step}.pt'
        state=torch.load(checkpoint if checkpoint.exists() else run/'qcr_model.pt',weights_only=True)
        constructor=RecurrentRGNetwork1D if dimension==1 else RecurrentRGNetwork2D
        model=constructor(levels=cfg['levels'],theory_dim=1,hidden=cfg['hidden'],
                          kernel_layout=cfg.get('kernel_layout','nonoverlapping'),
                          **({'learnable_kernel':not cfg.get('fixed_kernel',False)} if dimension==1 else {})).double().eval()
        inc=model.load_state_dict(state.get('model',state),strict=False)
        assert not inc.unexpected_keys and all(k.startswith('action_network.') for k in inc.missing_keys)
        model.theory_encoder.binary_lookup=True
        model.rg_kernel.binary_lookup=True
        dest=run/f'step{step}'; dest.mkdir(exist_ok=True)
        gs,rs=[],[]
        with torch.no_grad():
            for chunk in raw.split(8):
                levels=model.coarse_grain(balanced_views(chunk,cfg['ensemble_size']))
                g=torch.cat([model.theory_encoder(v) for v in levels],1)
                r=[g[:,:1]]
                for _ in range(cfg['levels']-1): r.append(model.advance_theory(r[-1]))
                gs.append(g.numpy()); rs.append(torch.cat(r,1).numpy())
        g,r=np.concatenate(gs),np.concatenate(rs)
        means=np.stack([g[h==v].mean(0) for v in knots])
        sd=np.stack([g[h==v].std(0,ddof=1) for v in knots])
        lo,hi=float(g.min()),float(g.max()); pad=max(.2,.2*(hi-lo))
        grid=np.linspace(lo-pad,hi+pad,1500)
        with torch.no_grad(): curve=model.advance_theory(torch.tensor(grid[:,None])).numpy()[:,0]
        def residual(x):
            with torch.no_grad(): return float(model.advance_theory(torch.tensor([[x]],dtype=torch.float64)))-x
        roots=[]
        for a,b,fa,fb in zip(grid[:-1],grid[1:],(curve-grid)[:-1],(curve-grid)[1:]):
            if fa*fb>=0: continue
            root=brentq(residual,a,b)
            x=torch.tensor([[root]],dtype=torch.float64,requires_grad=True)
            slope=float(torch.autograd.grad(model.advance_theory(x).sum(),x)[0])
            roots.append(dict(g=root,derivative=slope,stability='stable' if abs(slope)<1 else 'unstable',inside_observed_support=lo<=root<=hi))
        fig,axes=plt.subplots(1,2,figsize=(11,4.5))
        axes[0].plot(grid,curve,label='R(g)'); axes[0].plot(grid,grid,'k--',label='identity')
        axes[1].plot(grid,curve-grid); axes[1].axhline(0,color='black',ls='--')
        for ax in axes: ax.axvspan(lo,hi,alpha=.08,color='gray'); ax.grid(alpha=.2); ax.set_xlabel('g')
        for root in roots:
            color='tab:green' if root['stability']=='stable' else 'tab:red'
            axes[0].scatter(root['g'],root['g'],color=color,label=f"{root['g']:.3f}, {root['stability']}")
            axes[1].scatter(root['g'],0,color=color)
        axes[0].set_ylabel('R(g)'); axes[1].set_ylabel('R(g)-g'); axes[0].legend(fontsize=8)
        fig.suptitle(f'{label}, dim(g)=1, {step} updates; shading: observed encoding range')
        fig.tight_layout(); fig.savefig(dest/'R_vs_g.png',dpi=180); plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(12,4.8))
        axes[0].plot(knots,means[:,0],label='g0 = Q(D)')
        axes[0].fill_between(knots,means[:,0]-sd[:,0],means[:,0]+sd[:,0],alpha=.2)
        for k in range(cfg['levels']):
            axes[1].plot(knots,means[:,k],label=f'k={k}, L={cfg["sites"]//2**k}')
            axes[1].fill_between(knots,means[:,k]-sd[:,k],means[:,k]+sd[:,k],alpha=.12)
        for ax in axes:
            for root in roots: ax.axhline(root['g'],color='gray',ls='--',lw=.8)
            if dimension==2:
                critical = cfg.get('reference_critical_coupling', .5*np.log(1+np.sqrt(2)))
                ax.axvline(critical,color='black',ls=':',label='critical coupling (reference)')
            ax.set(xlabel=label,ylabel='g'); ax.grid(alpha=.2); ax.legend(fontsize=8)
        axes[0].set_title('Microscopic encoding'); axes[1].set_title('Actual Q(C^k D), not R rollout')
        fig.suptitle(f'{step} updates; means +/-1 replica SD, all saved samples plus flips')
        fig.tight_layout(); fig.savefig(dest/'coupling_vs_g.png',dpi=180); plt.close(fig)
        metrics={}
        for name,sl in [('train',slice(None,len(tx))),('test',slice(len(tx),None))]:
            rmse=np.sqrt(((g[sl]-r[sl])**2).mean(0)); spread=g[sl].std(0)
            metrics[name]=dict(rmse=rmse.tolist(),std=spread.tolist(),nrmse=(rmse/spread).tolist())
        report=dict(checkpoint=str(checkpoint),observed_support=[lo,hi],roots=roots,metrics=metrics,
                    coupling=knots.tolist(),g_mean=means.tolist(),g_replica_sd=sd.tolist())
        if dimension==1:
            with torch.no_grad(): report['C_truth']=model.rg_kernel(torch.tensor([[-1,-1],[-1,1],[1,-1],[1,1]],dtype=torch.float64).reshape(4,1,2,1)).flatten().tolist()
        else:
            patterns=2*((torch.arange(16)[:,None] // (2**torch.arange(3,-1,-1)))%2)-1
            with torch.no_grad(): report['C_truth']=model.rg_kernel(patterns.double().reshape(16,1,2,2,1)).flatten().tolist()
            report['C_pattern_order']=patterns.tolist()
            report['theoretical_Kc']=float(cfg.get('reference_critical_coupling', .5*np.log(1+np.sqrt(2))))
            report['g_at_Kc']=[float(np.interp(report['theoretical_Kc'],knots,means[:,k])) for k in range(cfg['levels'])]
        np.savetxt(dest/'coupling_vs_g.csv',np.column_stack([knots,means]),delimiter=',',header='coupling,'+','.join(f'g{k}' for k in range(cfg['levels'])),comments='')
        (dest/'report.json').write_text(json.dumps(report,indent=2)+'\n'); results[str(step)]=report
        print('REPORT',step,json.dumps(dict(metrics=metrics,roots=roots,C_truth=report.get('C_truth'))),flush=True)
    hist=summary['history']; fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for split in ['train','test']:
        axes[0].semilogy([v['epoch'] for v in hist],[v[f'{split}_total'] for v in hist],label=split)
    for step,res in results.items(): axes[1].plot(range(1,cfg['levels']),res['metrics']['test']['nrmse'][1:],'o-',label=step+' updates')
    axes[0].set(title='Train/test total loss',xlabel='Update',ylabel='Loss')
    axes[1].set(title='Full-test rollout NRMSE',xlabel='k',ylabel='RMSE / direct-Q std')
    for ax in axes: ax.grid(alpha=.2); ax.legend()
    fig.tight_layout(); fig.savefig(run/'loss_and_closure.png',dpi=180); plt.close(fig)
    (run/'scalar_report.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--run',required=True); parser.add_argument('--cache',required=True); parser.add_argument('--dimension',type=int,default=1)
    args=parser.parse_args(); scalar_report(args.run,args.cache,args.dimension)
