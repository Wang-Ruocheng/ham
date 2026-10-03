"""Single-coupling g2 report including the composed two-step RG map."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import root
from recurrent_rg.checkpoints import load_run
from recurrent_rg.data.ensembles import load_ensembles
from recurrent_rg.evaluation import collect_trajectories, closure_metrics
from recurrent_rg.plotting.reports import plot_diagnostics
from recurrent_rg.rsmi import categorical_kernel_probabilities


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--critical-coupling',type=float)
    args=parser.parse_args();torch.set_num_threads(2)
    loaded=load_run(args.run);model=loaded.model;cfg=loaded.config
    if cfg['model']['theory_dim']!=2 or len(cfg['data']['coupling_names'])!=1:
        raise ValueError('Requires dim g=2 and one scanned coupling')
    coupling=cfg['data']['coupling_names'][0]
    system=('2D Blume-Capel' if coupling=='T/J' and cfg['model']['categorical_states']==3
            else '1D TFIM' if coupling=='h/J' else '1D classical Ising')
    critical_value=args.critical_coupling
    if critical_value is None and coupling=='h/J':critical_value=1.
    train,test=load_ensembles(cfg['data']['cache'])
    opts=dict(samples=cfg['evaluation']['samples'],max_rg_steps=cfg['training']['max_rg_steps'],spin_flip_augment=cfg['data']['spin_flip_augment'])
    tr=collect_trajectories(model,train.configurations,**opts)
    te=collect_trajectories(model,test.configurations,**opts)
    out=args.run/'full_report';out.mkdir(exist_ok=True)
    np.savez_compressed(out/'encodings.npz',train=tr['direct'],test=te['direct'],test_rollout=te['rollout'],train_labels=train.couplings.numpy(),test_labels=test.couplings.numpy())
    plot_diagnostics(model,te,test.couplings.numpy(),(coupling,),out,training_inputs=tr['direct'][:,:-1])
    labels=np.concatenate([train.couplings.numpy(),test.couplings.numpy()]).reshape(-1)
    direct=np.concatenate([tr['direct'],te['direct']]);knots=np.unique(labels)
    means=np.array([direct[labels==v].mean(0) for v in knots])
    report=dict(closure=closure_metrics(te,reference_direct=tr['direct']),representation=[],parameters={n:sum(p.numel() for p in module.parameters()) for n,module in [('Q',model.theory_encoder),('C',model.rg_kernel),('R',model.theory_rg)]})
    for k in range(direct.shape[1]):
        eig=np.linalg.eigvalsh(np.cov(te['direct'][:,k].T,bias=True))
        report['representation'].append(dict(k=k,eigenvalues=eig.tolist(),weak_fraction=float(eig[0]/eig.sum())))
    two=lambda x:model.advance_theory(model.advance_theory(x))
    data=json.loads((out/'fixed_points.json').read_text());lo,hi=np.array(data['search_box'])
    search_results=[]
    for steps,fn in [(1,model.advance_theory),(2,two)]:
        found=[]
        def f(x):
            with torch.no_grad():return fn(torch.tensor(x,dtype=torch.double)).numpy()-x
        seeds=np.array(np.meshgrid(*[np.linspace(a,b,17) for a,b in zip(lo,hi)])).reshape(2,-1).T
        for seed in np.concatenate([seeds,means[:,0]]):
            fit=root(f,seed);point=fit.x
            if not fit.success or np.linalg.norm(f(point))>1e-7 or np.any(point<lo) or np.any(point>hi):continue
            if any(np.linalg.norm(point-np.array(r['g']))<1e-5 for r in found):continue
            x=torch.tensor(point,dtype=torch.double,requires_grad=True)
            j=torch.autograd.functional.jacobian(fn,x).detach().numpy();eig=np.linalg.eigvals(j);mag=abs(eig)
            kind='nonhyperbolic' if np.any(abs(mag-1)<1e-6) else ('stable' if np.all(mag<1) else ('unstable' if np.all(mag>1) else 'saddle'))
            residual=float(torch.linalg.vector_norm(model.advance_theory(x)-x).detach())
            distances=np.linalg.norm(means[:,0]-point,axis=1);nearest=int(np.argmin(distances))
            found.append(dict(g=point.tolist(),stability=kind,eigenvalues=[dict(real=float(v.real),imag=float(v.imag)) for v in eig],one_step_residual=residual,is_R_fixed_point=residual<1e-6,
                nearest_mean_coupling=float(knots[nearest]),coupling_name=coupling,nearest_mean_distance=float(distances[nearest]),coupling_note='Nearest microscopic mean curve sample, NOT an inverse interpolation',
                nearest_training_input_distance=float(np.linalg.norm(tr['direct'][:,:-1].reshape(-1,2)-point,axis=1).min())))
        search_results.append(found)
    report['fixed_points_R'],report['fixed_points_R2']=search_results
    cycles=[];paired=set()
    for i,r in enumerate(search_results[1]):
        if r['is_R_fixed_point'] or i in paired:continue
        with torch.no_grad():mapped=model.advance_theory(torch.tensor(r['g'],dtype=torch.double)).numpy()
        distances=[np.linalg.norm(mapped-np.array(other['g'])) for other in search_results[1]]
        j=int(np.argmin(distances))
        if j!=i and distances[j]<1e-6:
            with torch.no_grad():back=model.advance_theory(torch.tensor(search_results[1][j]['g'],dtype=torch.double)).numpy()
            reverse_error=float(np.linalg.norm(back-np.array(r['g'])))
            if reverse_error<1e-6:
                cycles.append(dict(root_indices_1based=[i+1,j+1],forward_error=float(distances[j]),reverse_error=reverse_error,stability=r['stability']))
                paired.update((i,j))
    report['two_cycles']=cycles
    report['search_box']=[lo.tolist(),hi.tolist()];report['search_note']='Bounded 17x17 plus microscopic-curve seeds; not exhaustive. R2-only roots can be period-two cycles.'
    xx,yy=np.meshgrid(np.linspace(lo[0],hi[0],31),np.linspace(lo[1],hi[1],31));points=np.column_stack([xx.ravel(),yy.ravel()])
    fig,axes=plt.subplots(1,2,figsize=(13,6),layout='constrained')
    for ax,fn,found,title in zip(axes,[model.advance_theory,two],search_results,[r'$R(g)-g$',r'$R^2(g)-g$']):
        with torch.no_grad():delta=fn(torch.from_numpy(points)).numpy()-points
        arrows=delta/np.maximum(np.linalg.norm(delta,axis=1,keepdims=True),1e-12)*.025*min(hi-lo)
        ax.scatter(*tr['direct'].reshape(-1,2).T,s=4,c='.65',alpha=.2,label='All-scale training encodings')
        sc=ax.scatter(*direct[:,0].T,c=labels,cmap='viridis',s=16,zorder=3)
        ax.plot(*means[:,0].T,'k-',lw=1,label=f'Microscopic mean g({coupling})')
        if critical_value is not None:
            if not knots.min() <= critical_value <= knots.max():raise ValueError('Critical reference outside sampled couplings')
            critical=np.array([np.interp(critical_value,knots,means[:,0,j]) for j in range(2)])
            ax.scatter(*critical,c='magenta',marker='D',s=70,edgecolors='black',zorder=5,label=f'{coupling}={critical_value:g} input interpolation')
        ax.quiver(*points.T,*arrows.T,color='.3',angles='xy',scale_units='xy',scale=1,width=.0025)
        for i,r in enumerate(found,1):
            ax.scatter(*r['g'],marker='*',s=120,c='white',edgecolors='black',zorder=6)
            suffix=' (2-cycle)' if not r['is_R_fixed_point'] else ''
            ax.annotate(f'F{i}: {r["stability"]}{suffix}',r['g'],xytext=(5,5),textcoords='offset points',fontsize=7)
        ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),aspect='equal',xlabel='g1',ylabel='g2',title=title);ax.legend(fontsize=7)
    fig.colorbar(sc,ax=axes,label=f'Microscopic {coupling}',shrink=.7)
    fig.suptitle(f'{system}, dim g=2 | {cfg["model"]["kernel_sampling"]} C | {cfg["training"]["epochs"]} updates | sign weight={cfg["loss"].get("kernel_sign_weight",0):g}\nNormalized discrete displacements; off-data arrows are extrapolation')
    fig.savefig(out/'R_one_vs_two_step.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(8,7),layout='constrained')
    for j,ax in enumerate(axes):
        for k in range(direct.shape[1]):ax.plot(knots,means[:,k,j],label=f'k={k}')
        if critical_value is not None:ax.axvline(critical_value,c='black',ls='--',lw=1)
        ax.set(xlabel=coupling,ylabel=f'g{j+1}');ax.legend(ncol=3,fontsize=8)
    fig.savefig(out/'coupling_by_scale.png',dpi=170);plt.close(fig)
    h=json.loads((args.run/'history_progress.json').read_text());weights=dict(cfg['loss']['weights']);weights.update(rollout=cfg['loss']['rollout_weight'],kernel_sign=cfg['loss'].get('kernel_sign_weight',0))
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,split in zip(axes,['train','test']):
        for key,w in weights.items():
            if w:ax.plot([r['epoch'] for r in h],[w*r[split+'_'+key] for r in h],label=key)
        ax.plot([r['epoch'] for r in h],[r.get(split+'_qcr_total',r[split+'_total']) for r in h],'k-',label='QCR total (MI excluded)' if split+'_qcr_total' in h[0] else 'total');ax.set_yscale('symlog',linthresh=.001);ax.set(xlabel='updates',ylabel='weighted loss',title=split,ylim=(0,None));ax.legend(fontsize=7)
    fig.savefig(out/'loss.png',dpi=170);plt.close(fig)
    states=cfg['model']['categorical_states']
    if states:
        width=cfg['model']['block_factor']**cfg['model']['dimension']
        codes=(torch.arange(states**width)[:,None]//(states**torch.arange(width)))%states
        patterns=torch.nn.functional.one_hot(codes,states).double().flatten(1)[None]
        with torch.no_grad():p=categorical_kernel_probabilities(model.rg_kernel,patterns)[0]
        report['kernel']=dict(patterns=codes.tolist(),probabilities=p.tolist())
    else:
        if cfg['model']['dimension']!=1:raise ValueError('Binary kernel table currently requires 1D')
        patterns=torch.tensor([[-1.,-1.],[1.,-1.],[-1.,1.],[1.,1.]],dtype=torch.double)
        with torch.no_grad():p=torch.sigmoid(2*model.rg_kernel.field_head.logits(model.rg_kernel.local_encoder(patterns))).flatten()
        report['kernel']=dict(patterns=patterns.tolist(),p_plus=p.tolist())
    report['sampling_repeats']=[closure_metrics(collect_trajectories(model,test.configurations,**opts,sampling_seed=s),reference_direct=tr['direct']) for s in range(5)]
    (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
