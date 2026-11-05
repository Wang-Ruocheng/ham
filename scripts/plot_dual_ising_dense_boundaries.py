"""Map physical critical lines using cached, irregular coupling-grid Q encodings."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scipy.interpolate import LinearNDInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from recurrent_rg.checkpoints import load_run


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,default=Path('results/dual_ising2d_g2_dense_joint_fast_2400_seed91'))
    args=parser.parse_args();out=args.run/'report';torch.set_num_threads(2)
    model=load_run(args.run).model.eval()
    data=np.load(out/'trajectories.npz')
    labels=np.concatenate([data['train_couplings'],data['test_couplings']])
    direct=np.concatenate([data['train_direct'],data['test_direct']])
    cells=np.unique(labels,axis=0)
    means=np.stack([direct[(labels==p).all(1)].mean(0) for p in cells])
    kc=np.log(1+np.sqrt(2))/2;u=np.linspace(0,1,501)
    parameters=[np.column_stack([np.full_like(u,kc),u]),np.column_stack([u,np.full_like(u,kc)])]
    interpolators=[LinearNDInterpolator(cells,means[:,k]) for k in range(3)]
    curves=np.array([[fn(p) for p in parameters] for fn in interpolators])
    centers=np.array([fn([[kc,kc]])[0] for fn in interpolators])
    assert np.isfinite(curves).all() and np.isfinite(centers).all()
    audit=np.load(out/'critical_audit.npz');audit_mean=audit['direct'].mean(1)
    ai=np.flatnonzero(np.isclose(audit['couplings'],kc).all(1))[0]
    old=json.loads((out/'fixed_points.json').read_text());lo,hi=np.array(old['search_box'])
    xx,yy=np.meshgrid(np.linspace(lo[0],hi[0],30),np.linspace(lo[1],hi[1],30));grid=np.column_stack([xx.ravel(),yy.ravel()])
    with torch.no_grad():delta=model.advance_theory(torch.from_numpy(grid)).numpy()-grid
    norm=np.linalg.norm(delta,axis=1);unit=delta/np.maximum(norm[:,None],1e-12)
    def draw(ax,k,legend=False):
        arrows=ax.quiver(*grid.T,*unit.T,norm,cmap='viridis',alpha=.48)
        ax.scatter(*direct[:,k].T,s=5,c='.45',alpha=.22)
        for j,color in enumerate(['#d81b60','#0072b2']):
            line=ax.plot(*curves[k,j].T,c=color,lw=2.8,label=f'K{j+1}=Kc (interpolated)')[0]
            line.set_path_effects([pe.Stroke(linewidth=4.5,foreground='white'),pe.Normal()])
            for value in [0,.2,.7]:
                z=curves[k,j,round(value*500)]
                ax.annotate(f'K{2-j}={value:g}',z,xytext=(5,7 if j==0 else -13),textcoords='offset points',color=color,fontsize=7)
            mask=np.isclose(audit['couplings'][:,j],kc)
            ax.scatter(*audit_mean[mask,k].T,s=25,facecolors='white',edgecolors=color,lw=1.2,zorder=5,label=f'K{j+1}=Kc direct samples')
        ax.scatter(*centers[k],c='black',s=55,marker='D',zorder=6,label='Interpolated (Kc,Kc)')
        ax.scatter(*audit_mean[ai,k],facecolors='white',edgecolors='black',s=75,marker='D',zorder=7,label='Direct sampled (Kc,Kc)')
        seen=set()
        for root in old['roots']:
            kind=root['stability'];color={'stable':'#2ca02c','unstable':'#ff7f0e','saddle':'#9467bd'}[kind]
            ax.scatter(*root['g'],c=color,s=140,marker='*',edgecolors='black',lw=.4,zorder=6,label='R '+kind if kind not in seen else None);seen.add(kind)
        ax.set(xlabel='g1',ylabel='g2',xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),title=f'Theoretical critical lines mapped at k={k}')
        ax.grid(alpha=.13)
        if legend:ax.legend(fontsize=7,loc='lower left')
        return arrows
    fig,ax=plt.subplots(figsize=(10,8),layout='constrained');arrows=draw(ax,0,True)
    ax.set_title('R(g)-g with microscopic critical lines\nPiecewise-linear Q interpolation, NOT inferred R separatrices')
    fig.colorbar(arrows,ax=ax,label='|R(g)-g|; arrows normalized')
    fig.savefig(out/'R_with_theory_boundaries.png',dpi=190)
    fig,axs=plt.subplots(1,3,figsize=(18,6),layout='constrained')
    for k,ax in enumerate(axs):draw(ax,k,k==0)
    fig.savefig(out/'theory_boundaries_by_scale.png',dpi=170)
    np.savez_compressed(out/'theory_boundaries.npz',parameter=u,curves=curves,interpolated_crossings=centers,direct_crossings=audit_mean[ai])
    report=dict(critical_K=kc,cells=len(cells),method='Piecewise-linear Delaunay interpolation of replica-mean Q on original grid plus dense training cells; independent audit not used in interpolation.',
        interpolated_crossings=centers.tolist(),direct_crossings=audit_mean[ai].tolist(),
        interpolation_vs_direct_distance=np.linalg.norm(centers-audit_mean[ai],axis=1).tolist(),
        caveat='These curves locate initially critical ensembles in latent coordinates, not validated invariant basin boundaries. Sparse interpolation remains away from densified regions. Off-data arrows are extrapolation.')
    (out/'theory_boundaries.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
