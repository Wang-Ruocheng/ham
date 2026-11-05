"""Initial-ensemble basin test, without fitting or changing Q/C/R."""
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from recurrent_rg.checkpoints import load_run
from recurrent_rg.data.ensembles import load_ensembles
from recurrent_rg.evaluation import collect_trajectories
from recurrent_rg.data.ashkin_teller import observables


def crossings(x,y):
    return [float(a+(b-a)*(-u)/(v-u)) for a,b,u,v in zip(x[:-1],x[1:],y[:-1],y[1:]) if u*v<0]


def main():
    torch.set_num_threads(2)
    folder=Path('results/ashkin_teller2d_g1_J0p1_uniformU_fixed_1200_seed91')
    run=load_run(folder);m=run.model.eval();tr,te=load_ensembles(run.config['data']['cache'])
    x=torch.cat([tr.configurations,te.configurations]);U=torch.cat([tr.couplings,te.couplings])[:,0].numpy()
    knots=np.unique(U)
    with torch.no_grad():
        t=collect_trajectories(m,x,samples=128,max_rg_steps=2)
        direct=t['direct'][...,0]
        z=torch.from_numpy(direct[:,0,None]);paths=[z[:,0].numpy()]
        for _ in range(20):
            z=m.advance_theory(z);paths.append(z[:,0].numpy())
        paths=np.stack(paths,1)
        grid=torch.linspace(direct[:len(tr.configurations),:2].min(),direct[:len(tr.configurations),:2].max(),1001,dtype=torch.double)[:,None].requires_grad_()
    with torch.enable_grad():
        derivative=torch.autograd.grad(m.advance_theory(grid).sum(),grid)[0].detach().numpy()
    root=json.loads((folder/'full_report/metrics.json').read_text())['fixed_points']['roots'][0]['g'][0]
    avg=lambda a:np.stack([a[U==v].mean(0) for v in knots])
    gd,gr=avg(direct),avg(paths)
    stats=[]
    for k in range(3):
        codes=x[:,:,::2**k,::2**k].argmax(-1).numpy()
        o=observables(codes.reshape(-1,*codes.shape[2:]))
        stats.append(np.stack([o[n].reshape(codes.shape[:2]).mean(1) for n in ['m_sigma','m_product']],-1))
    stats=avg(np.stack(stats,1))
    probes=[]
    for target in [.36,.40,.43,.46,.55,.60,.65]:
        i=int(np.argmin(abs(knots-target)));mask=U==knots[i]
        probes.append(dict(U=float(knots[i]),direct=gd[i].tolist(),rollout=gr[i,:7].tolist(),
                           initial_product_magnetization=float(stats[i,0,1]),
                           fraction_above_repeller_by_step=(paths[mask]>root).mean(0).tolist()))
    lo,hi=direct[:len(tr.configurations),:2].min(),direct[:len(tr.configurations),:2].max()
    fractions=avg((paths>root).astype(float))
    report=dict(repeller=float(root),initial_coordinate_crossing=crossings(knots,gd[:,0]-root),
                ensemble_side_changes_over_20_steps=int(np.any((paths>root)!=(paths[:,:1]>root),axis=1).sum()),
                mixed_replica_U=knots[(fractions[:,0]>0)&(fractions[:,0]<1)].tolist(),
                initial_upper_branch_fraction=fractions[:,0].tolist(),U_grid=knots.tolist(),
                boundary_by_rollout_step={str(k):crossings(knots,gr[:,k]-root) for k in [0,1,2,5,10,20]},
                direct_drift_zero_by_step={str(k):crossings(knots,np.diff(gd,axis=1)[:,k]) for k in [0,1]},
                direct_root_crossing_by_step={str(k):crossings(knots,gd[:,k]-root) for k in [0,1,2]},
                fraction_outside_training_input_range_by_step=((paths<lo)|(paths>hi)).mean(0).tolist(),
                sampled_R_derivative_range=[float(derivative.min()),float(derivative.max())],probes=probes,
                caveat='Only direct k=0,1,2 (L16,8,4) are supported. R rollout to20 is extrapolation, not physical infinite-scale verification. Local-order statistics at L16 do not locate thermodynamic criticality. Linear U interpolation; three replicas per point. Mean-rollout zero crossings can shift due to nonlinear averaging/interpolation and are NOT shifts of individual basin membership. Initial coupling labels used for diagnostics only.')
    out=folder/'initial_boundary';out.mkdir(exist_ok=True)
    np.savez_compressed(out/'arrays.npz',U=U,direct=direct,rollout=paths,knots=knots,order=stats)
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,axs=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    for k in range(3):
        axs[0,0].plot(knots,stats[:,k,1],label=f'product, k={k}')
        axs[0,0].plot(knots,stats[:,k,0],ls='--',alpha=.7,label=f'sigma, k={k}')
        axs[0,1].plot(knots,gd[:,k],'.-',label=f'direct k={k}')
    axs[0,1].axhline(root,color='k',ls=':',label='R repeller')
    axs[0,0].set(xlabel='initial U/T',ylabel='mean absolute magnetization',title='Configurations: finite-size order, not an exact critical point')
    axs[0,1].set(xlabel='initial U/T',ylabel='g',title='Q(C^k D), direct data only')
    for k in [0,1,2,5,10]:axs[1,0].plot(knots,gr[:,k],label=f'R rollout k={k}')
    axs[1,0].axhline(root,color='k',ls=':');axs[1,0].axvline(.6025977594496204,color='.5',ls='--')
    axs[1,0].set(xlabel='initial U/T',ylabel='g',title='Model basin split: long rollouts are extrapolation')
    for target in [.383,.434,.46,.587,.638]:
        i=int(np.argmin(abs(knots-target)));line=axs[1,1].plot(range(7),gr[i,:7],'--',label=f'U/T={knots[i]:.3f}')[0]
        axs[1,1].plot(range(3),gd[i],'o-',c=line.get_color())
    axs[1,1].axhline(root,color='k',ls=':');axs[1,1].axvspan(2,6,color='.9',zorder=-1)
    axs[1,1].set(xlabel='RG step',ylabel='g',title='Solid: direct Q(C^k D); dashed: R rollout')
    for ax in axs.flat:ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(out/'initial_boundary.png',dpi=170)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
