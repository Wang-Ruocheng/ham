"""Extra physical diagnostics for the RSMI-C -> frozen-C QR experiment."""
import argparse
import itertools
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import root as solve_root

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recurrent_rg.checkpoints import load_run
from recurrent_rg.data.ensembles import load_ensembles
from recurrent_rg.evaluation import collect_trajectories, closure_metrics, discrete_fixed_point
from recurrent_rg.rsmi import kernel_probabilities, reference_probabilities


def exact_chain_mi(probabilities, coupling, length=128):
    """Exact finite-ring I(H; E) for V=(s0,s1), E=(s[-2],s3).

    Integrate out the 124 other sites using the transfer-matrix identity.
    probabilities has shape [4,2], binary states ordered low-bit-first.
    This is a label-using diagnostic, never part of the optimizer.
    """
    spins=np.array(list(itertools.product((-1.,1.),repeat=4)))
    t=np.tanh(coupling)
    distances=np.array([2,1,2,length-5])
    joint=np.prod(1+spins*np.roll(spins,-1,axis=1)*t**distances,axis=1)/(16*(1+t**length))
    vi=((spins[:,1:3]+1)/2 @ np.array([1,2])).astype(int)
    ei=((spins[:,[0,3]]+1)/2 @ np.array([1,2])).astype(int)
    he=np.zeros((2,4))
    for n in range(len(spins)):he[:,ei[n]]+=joint[n]*probabilities[vi[n]]
    assert abs(he.sum()-1)<1e-10
    independent=he.sum(1,keepdims=True)*he.sum(0,keepdims=True)
    mask=he>0
    return float(np.sum(he[mask]*np.log2(he[mask]/independent[mask])))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--checkpoint',type=int,help='Optional numbered training checkpoint')
    p.add_argument('--flow-title',default='RSMI C frozen',help='Label the actual training protocol when reusing physical diagnostics')
    args=p.parse_args();torch.set_num_threads(2)
    run=load_run(args.run/'qr',args.checkpoint);cfg=run.config;model=run.model
    train,test=load_ensembles(cfg['data']['cache'])
    options=dict(samples=cfg['evaluation']['samples'],max_rg_steps=cfg['training']['max_rg_steps'],
                 spin_flip_augment=cfg['data']['spin_flip_augment'])
    tr=collect_trajectories(model,train.configurations,**options)
    te=collect_trajectories(model,test.configurations,**options)
    out=args.run/'qr'/'report'
    np.savez_compressed(out/'encodings.npz',train=tr['direct'],test=te['direct'],
                        train_labels=train.couplings.numpy(),test_labels=test.couplings.numpy())
    report=dict(closure=closure_metrics(te,reference_direct=tr['direct']))
    d=np.concatenate([tr['direct'],te['direct']]);labels=np.concatenate([train.couplings.numpy(),test.couplings.numpy()]).reshape(len(d),-1)
    report['covariance_eigenvalues']=[np.linalg.eigvalsh(np.atleast_2d(np.cov(d[:,k].T,bias=True))).tolist() for k in range(d.shape[1])]
    # Direct configuration diagnostic, independent of Q and R.
    xs=torch.cat([train.configurations,test.configurations])
    selected=(np.linspace(0,len(xs)-1,16).astype(int) if cfg['model']['dimension']==1
              else np.flatnonzero(np.isclose(labels[:,1],0)))
    magn=[]
    with torch.random.fork_rng(devices=[]),torch.no_grad():
        torch.manual_seed(12345)
        for chunk in xs[selected].split(16):
            chunk=chunk.double()
            if cfg['data']['spin_flip_augment']:chunk=torch.cat([chunk,-chunk],dim=1)
            levels=model.coarse_grain(chunk)
            magn.append(torch.stack([z.mean(tuple(range(1,z.ndim))) for z in levels],1))
    report['zero_field_magnetization']=dict(couplings=labels[selected].tolist(),
                                          mean_by_scale=torch.cat(magn).tolist())
    if cfg['model']['dimension']==1:
        patterns=(2*((torch.arange(4)[:,None]//(2**torch.arange(2)))%2)-1).double()[None]
        with torch.no_grad():
            prob=(reference_probabilities(patterns,'decimation') if cfg['model']['fixed_kernel'] else kernel_probabilities(model.rg_kernel,patterns))[0].numpy()
        js=np.unique(labels[:,0]);mi={}
        for name,table in [('learned',prob),('decimation',reference_probabilities(patterns,'decimation')[0].numpy()),
                           ('majority',reference_probabilities(patterns,'majority')[0].numpy())]:
            mi[name]=[exact_chain_mi(table,j) for j in js]
        report['exact_chain_MI_bits']=dict(J=js.tolist(),**mi)
        report['exact_chain_mean_MI_bits']={n:float(np.mean(v)) for n,v in mi.items()}
        candidates=[]
        for code in range(16):
            plus=((code//(2**np.arange(4)))%2).astype(float)
            table=np.stack([1-plus,plus],-1)
            candidates.append(dict(code=code,outputs=(2*plus-1).tolist(),
                mean_MI_bits=float(np.mean([exact_chain_mi(table,j) for j in js]))))
        report['all_16_deterministic_binary_kernels']=sorted(candidates,key=lambda a:-a['mean_MI_bits'])
        fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
        for name,values in mi.items():axes[0].plot(js,values,label=name)
        axes[0].set(xlabel='J',ylabel='Exact I(H;E), bits',title='Independent finite-ring transfer-matrix check');axes[0].legend()
        means=np.stack([d[labels[:,0]==j].mean(0) for j in js])
        for k in range(d.shape[1]):axes[1].plot(js,means[:,k,0],label=f'k={k}')
        axes[1].set(xlabel='J',ylabel='g',title='Q(C^k D), all ensemble means');axes[1].legend()
        fig.savefig(out/'exact_MI_and_coupling.png',dpi=170);plt.close(fig)
    else:
        ks,bs=np.unique(labels[:,0]),np.unique(labels[:,1]);j0=np.argmin(abs(bs));kc=.5*np.log(1+np.sqrt(2))
        means=np.array([[d[np.all(labels==[k,b],axis=1),0].mean(0) for b in bs] for k in ks])
        report['critical_microscopic_g']=[float(np.interp(kc,ks,means[:,j0,a])) for a in range(2)]
        fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
        for a,ax in enumerate(axes):
            im=ax.pcolormesh(bs,ks,means[:,:,a],shading='nearest',cmap='coolwarm')
            ax.axhline(kc,color='k',ls='--',lw=.8);ax.set(xlabel='B',ylabel='K',title=f'g{a+1}(K,B)');fig.colorbar(im,ax=ax)
            ax.set_xticks(np.linspace(bs.min(),bs.max(),5))
        fig.savefig(out/'g_KB_maps.png',dpi=170);plt.close(fig)
        roots=json.loads((out/'fixed_points.json').read_text())
        # Check for extra period-two points, rather than assuming that all
        # fixed points of the composed map are also one-step fixed points.
        lo,hi=np.array(roots['search_box'])
        twice=SimpleNamespace(advance_theory=lambda x:model.advance_theory(model.advance_theory(x)))
        def residual_two(x):
            with torch.no_grad():return twice.advance_theory(torch.tensor(x,dtype=torch.double)).numpy()-x
        seeds=np.array(np.meshgrid(*[np.linspace(a,b,17) for a,b in zip(lo,hi)])).reshape(2,-1).T
        two_roots=[]
        for seed in np.concatenate([seeds,d[:,0]]):
            fit=solve_root(residual_two,seed);point=fit.x
            if not fit.success or np.linalg.norm(residual_two(point))>1e-7 or np.any(point<lo) or np.any(point>hi):continue
            if any(np.linalg.norm(point-np.array(r['g']))<1e-5 for r in two_roots):continue
            record=discrete_fixed_point(twice,point)
            with torch.no_grad():partner=model.advance_theory(torch.tensor(point,dtype=torch.double)).numpy()
            record.update(one_step_residual=float(np.linalg.norm(partner-point)),
                          is_R_fixed_point=bool(np.linalg.norm(partner-point)<1e-6),partner_under_R=partner.tolist())
            two_roots.append(record)
        report['two_step_fixed_points']=two_roots
        report['two_step_search_note']='Same bounded box; 17x17 plus microscopic encodings as seeds. Not exhaustive; stars in plots mark R roots only.'
        report['root_microscopic_inverse_candidates']=[]
        for root in roots['roots']:
            candidates=[]
            for i in range(len(ks)-1):
                for j in range(len(bs)-1):
                    for ids in (((i,j),(i+1,j),(i+1,j+1)),((i,j),(i+1,j+1),(i,j+1))):
                        tri=np.array([means[a,b] for a,b in ids])
                        matrix=np.vstack([tri.T,np.ones(3)])
                        if abs(np.linalg.det(matrix))<1e-12:continue
                        bary=np.linalg.solve(matrix,np.r_[root['g'],1.])
                        if np.min(bary)>=-1e-9:
                            candidate=bary @ np.array([[ks[a],bs[b]] for a,b in ids])
                            if not any(np.linalg.norm(candidate-np.array(c))<1e-7 for c in candidates):candidates.append(candidate.tolist())
            report['root_microscopic_inverse_candidates'].append(dict(g=root['g'],candidates_KB=candidates,
                note='Inverse of piecewise-linear microscopic grid only; not effective couplings or evidence of criticality. Multiple values indicate folding.'))
        lower,upper=np.array(roots['search_box']);xx,yy=np.meshgrid(np.linspace(lower[0],upper[0],28),np.linspace(lower[1],upper[1],28))
        grid=np.column_stack([xx.ravel(),yy.ravel()]);fig,axes=plt.subplots(1,2,figsize=(13,6),layout='constrained')
        for n,ax in enumerate(axes,1):
            with torch.no_grad():
                z=torch.from_numpy(grid);pred=z
                for _ in range(n):pred=model.advance_theory(pred)
                delta=pred.numpy()-grid
            delta/=np.maximum(np.linalg.norm(delta,axis=1,keepdims=True),1e-12)
            ax.quiver(*grid.T,*delta.T,color='.5',alpha=.6)
            sc=ax.scatter(*d[:,0].T,c=labels[:,1],cmap='coolwarm',s=8)
            ax.plot(*means[:,j0].T,'k-',lw=1,label='B=0 microscopic curve')
            ax.scatter(*report['critical_microscopic_g'],marker='D',c='magenta',s=60,label='Q(Kc, B=0)')
            for i,r in enumerate(roots['roots'],1):
                ax.scatter(*r['g'],marker='*',s=110,c='gold',edgecolors='k')
                ax.annotate(f'F{i}: {r["stability"]}',r['g'],fontsize=8)
            ax.set(xlabel='g1',ylabel='g2',title=f'R^{n}(g)-g (normalized arrows)',xlim=(lower[0],upper[0]),ylim=(lower[1],upper[1]))
            ax.legend(fontsize=8);fig.colorbar(sc,ax=ax,label='Microscopic B')
        fig.suptitle(args.flow_title+'; off-data arrows are extrapolation. Stars: R roots only.')
        fig.savefig(out/'R_one_two_step.png',dpi=170);plt.close(fig)
        # One-step full view and local view; choose the root nearest the
        # diagnostic critical coordinate, without altering training/selection.
        center=np.array(report['critical_microscopic_g'])
        if roots['roots']:
            nearest=min(roots['roots'],key=lambda r:np.linalg.norm(np.array(r['g'])-center))
            focus=.5*(center+np.array(nearest['g']))
        else:focus=center
        fig,axes=plt.subplots(1,2,figsize=(13,5.5),layout='constrained')
        bounds=[(lower,upper),(focus-np.array([.55,.30]),focus+np.array([.55,.30]))]
        for ax,(lo,hi),title in zip(axes,bounds,['Full view','Central region']):
            xx,yy=np.meshgrid(np.linspace(lo[0],hi[0],27),np.linspace(lo[1],hi[1],27))
            grid=np.column_stack([xx.ravel(),yy.ravel()])
            with torch.no_grad():delta=model.advance_theory(torch.from_numpy(grid)).numpy()-grid
            delta/=np.maximum(np.linalg.norm(delta,axis=1,keepdims=True),1e-12)
            ax.quiver(*grid.T,*delta.T,color='.55',alpha=.65)
            sc=ax.scatter(*d[:,0].T,c=labels[:,1],cmap='coolwarm',s=12)
            ax.plot(*means[:,j0].T,'k-',lw=1,label='B=0 microscopic curve')
            ax.scatter(*center,marker='D',c='magenta',s=55,label='Q(Kc, B=0)')
            for i,r in enumerate(roots['roots'],1):
                point=np.array(r['g'])
                if np.all(point>=lo)&np.all(point<=hi):
                    ax.scatter(*point,marker='*',s=140,c='gold',edgecolors='k')
                    ax.annotate(f'F{i}: {r["stability"]}',point,xytext=(5,7),textcoords='offset points',fontsize=8)
            ax.set(xlabel='g1',ylabel='g2',title=title,xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]))
            ax.legend(fontsize=8);fig.colorbar(sc,ax=ax,label='Microscopic B')
        fig.suptitle('R(g)-g, normalized arrows; off-data field is extrapolation')
        fig.savefig(out/'R_flow_and_center.png',dpi=170);plt.close(fig)
    (out/'rsmi_physical_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
