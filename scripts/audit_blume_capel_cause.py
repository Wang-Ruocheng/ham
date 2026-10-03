"""Read-only C/Q/R diagnostics; never modify/retrain checkpoint weights."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from recurrent_rg.checkpoints import load_run
from recurrent_rg.data.ensembles import load_ensembles
from recurrent_rg.rsmi import categorical_kernel_probabilities


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(2);r=load_run(a.run);m=r.model;train,test=load_ensembles(r.config['data']['cache'])
    out=a.run/'cause_audit';out.mkdir(exist_ok=True)
    allx=torch.cat([train.configurations,test.configurations]);t=torch.cat([train.couplings,test.couplings]).numpy().ravel();knots=np.unique(t)
    series=[];aligned=[];observable=[]
    with torch.no_grad(),torch.random.fork_rng(devices=[]):
        for seed in [311,312,313]:
            torch.manual_seed(seed);glist=[];alist=[];olist=[]
            for chunk in allx.split(12):
                levels=m.coarse_grain(chunk.double());gg=[];aa=[];oo=[]
                for k,x in enumerate(levels):
                    gg.append(m.theory_encoder(x)[:,0]);canonical=x[...,[1,0,2]] if k%2 else x
                    aa.append(m.theory_encoder(canonical)[:,0])
                    probs=x.mean((1,2,3));equal=(x*x.roll(1,2)).sum(-1).mean((1,2,3))
                    conn=(equal-probs.square().sum(-1))/(1-probs.square().sum(-1)).clamp_min(1e-9)
                    s=canonical[...,2]-canonical[...,0]
                    abs_m=s.mean((2,3)).abs().mean(1);rho=s.square().mean((1,2,3))
                    oo.append(torch.stack([conn,abs_m,rho],-1))
                glist.append(torch.stack(gg,1));alist.append(torch.stack(aa,1));olist.append(torch.stack(oo,1))
            series.append(torch.cat(glist).numpy());aligned.append(torch.cat(alist).numpy());observable.append(torch.cat(olist).numpy())
    draws=np.array(series);g=draws.mean(0);can=np.array(aligned).mean(0);obs=np.array(observable).mean(0)
    mean=np.stack([g[t==v].mean(0) for v in knots]);mean_can=np.stack([can[t==v].mean(0) for v in knots]);mean_obs=np.stack([obs[t==v].mean(0) for v in knots])
    with torch.no_grad():
        one=m.advance_theory(torch.from_numpy(mean[:,:-1,None]))[:,:,0].numpy()
        rollout=m.advance_theory(m.advance_theory(torch.from_numpy(mean[:,0,None])))[:,0].numpy()
    # Flexible 1D interpolation is an audit, not R retraining. Train-temperature
    # ensemble means provide pairs (g_k,g_{k+1}); held-out temperatures evaluate.
    train_t=np.unique(train.couplings.numpy());mask=np.isin(knots,train_t)
    x=mean[mask,:-1].ravel();y=mean[mask,1:].ravel();order=np.argsort(x);x,y=x[order],y[order]
    def oracle(z):return np.interp(z,x,y)
    held=mean[~mask];pred=oracle(held[:,:-1]);multi=oracle(oracle(held[:,0]))
    report=dict(source=str(a.run),sampling_seeds=[311,312,313],configs_per_ensemble=allx.shape[1],
        fixed_target_test=dict(R_one_mse_by_scale=np.mean((one[~mask]-held[:,1:])**2,axis=0).tolist(),
            interpolation_one_mse_by_scale=np.mean((pred-held[:,1:])**2,axis=0).tolist(),
            R_two_step_mse=float(np.mean((rollout[~mask]-held[:,2])**2)),
            interpolation_two_step_mse=float(np.mean((multi-held[:,2])**2)),
            interpolation_outside_train_range=int(((held[:,:-1]<x.min())|(held[:,:-1]>x.max())).sum()),
            note='Piecewise linear fit to train-temperature means; evaluated on held-out temperature means. No checkpoint changes. Smoothed targets, not logged stochastic minibatch loss.'),
        sampling_sd_by_scale=np.sqrt(draws.var(0).mean(0)).tolist(),
        source_ranges=[[float(mean[:,k].min()),float(mean[:,k].max())] for k in range(2)])
    # Interpolated same-g cross-scale collision check, only where source ranges overlap.
    lo=max(mean[:,0].min(),mean[:,1].min());hi=min(mean[:,0].max(),mean[:,1].max())
    if lo<hi:
        grid=np.linspace(lo,hi,101);values=[]
        for k in [0,1]:
            order=np.argsort(mean[:,k]);values.append(np.interp(grid,mean[order,k],mean[order,k+1]))
        gap=np.abs(values[0]-values[1]);report['cross_scale_overlap']=dict(interval=[float(lo),float(hi)],max_next_gap=float(gap.max()),mean_next_gap=float(gap.mean()),note='Mean-curve interpolation; not an exact irreducible-loss lower bound')
    tc=1.398
    def crossing(values):
        delta=values[:,2]-values[:,0];hits=[]
        for i in range(len(knots)-1):
            if delta[i]*delta[i+1]<0:hits.append(float(knots[i]-delta[i]*(knots[i+1]-knots[i])/(delta[i+1]-delta[i])))
        return hits
    report['direct_two_step_zero_drift_temperatures']=crossing(mean)
    report['at_reference_Tc']=dict(T=tc,direct=[float(np.interp(tc,knots,mean[:,k])) for k in range(3)],gauge_aligned=[float(np.interp(tc,knots,mean_can[:,k])) for k in range(3)])
    report['phase_examples']=[dict(T=float(knots[i]),g=mean[i].tolist(),aligned_g=mean_can[i].tolist(),color_invariant_connected_correlation=mean_obs[i,:,0].tolist(),canonical_abs_m=mean_obs[i,:,1].tolist(),canonical_occupancy=mean_obs[i,:,2].tolist()) for i in [0,14,28,42,63]]
    codes=(torch.arange(81)[:,None]//(3**torch.arange(4)))%3
    with torch.no_grad():
        table=categorical_kernel_probabilities(m.rg_kernel,torch.nn.functional.one_hot(codes,3).double().flatten(1)[None])[0]
        swap=torch.tensor([1,0,2]);inp=swap[codes]
        swapped=categorical_kernel_probabilities(m.rg_kernel,torch.nn.functional.one_hot(inp,3).double().flatten(1)[None])[0]
        report['C_swap_commutator_mean_abs']=float((swapped-table[:,swap]).abs().mean())
    # Checkpoint drift separates moving-Q targets from convergence of R itself.
    drift=[]
    base=train.configurations[:24,:64].double();previous=None
    for epoch in [800,1200,1600,2000,2400]:
        loaded=load_run(a.run,epoch).model
        with torch.no_grad(),torch.random.fork_rng(devices=[]):
            torch.manual_seed(444);d=torch.stack([loaded.theory_encoder(z)[:,0] for z in loaded.coarse_grain(base)],1)
            pre=loaded.advance_theory(d[:,:-1,None])[:,:,0]
            row=dict(epoch=epoch,one_mse=float((pre-d[:,1:]).square().mean()))
            if previous is not None:
                old,old_d=previous;old_pre=old.advance_theory(d[:,:-1,None])[:,:,0]
                row.update(target_change_rms=float((d-old_d).square().mean().sqrt()),old_R_on_new_targets_one_mse=float((old_pre-d[:,1:]).square().mean()))
                delta=d-old_d
                design=torch.stack([old_d.flatten(),torch.ones_like(old_d.flatten())],1)
                affine=torch.linalg.lstsq(design,d.flatten()).solution
                row.update(target_mean_shift=float(delta.mean()),
                    target_change_after_translation_rms=float((delta-delta.mean()).square().mean().sqrt()),
                    affine_scale=float(affine[0]),affine_offset=float(affine[1]),
                    target_change_after_affine_rms=float((design@affine-d.flatten()).square().mean().sqrt()))
            drift.append(row);previous=(loaded,d)
    report['checkpoint_drift_common_samples']=drift
    fig,axs=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    for k in range(3):
        axs[0,0].plot(knots,mean[:,k],label=f'raw k={k}');axs[0,1].plot(knots,mean_can[:,k],label=f'aligned k={k}')
        axs[1,0].plot(knots,mean_obs[:,k,0],label=f'k={k}')
    axs[0,0].set(title='Actual Q(C^k D)',ylabel='g');axs[0,1].set(title='Post-hoc odd-scale label swap before Q',ylabel='g')
    axs[1,0].set(title='Label-invariant nearest-neighbor correlation',ylabel='Connected same-state correlation')
    for ax in axs.flat[:3]:ax.set_xlabel('T/J');ax.axvline(tc,ls=':',c='purple');ax.legend();ax.grid(alpha=.2)
    ax=axs[1,1]
    for k in [0,1]:ax.scatter(mean[:,k],mean[:,k+1],s=12,label=f'k={k} targets')
    grid=np.linspace(mean[:,:-1].min(),mean[:,:-1].max(),500)
    with torch.no_grad():mapped=m.advance_theory(torch.from_numpy(grid[:,None]))[:,0].numpy()
    ax.plot(grid,mapped,'k-',label='Trained R');ax.plot(grid,oracle(grid),'--',label='Train-mean interpolation')
    ax.set(xlabel='g_k',ylabel='g_(k+1)',title='Fixed-target mapping audit');ax.legend();ax.grid(alpha=.2)
    fig.savefig(out/'cause_diagnostics.png',dpi=160);plt.close(fig)
    np.savez_compressed(out/'audit_arrays.npz',temperatures=knots,direct=mean,aligned=mean_can,observables=mean_obs)
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
