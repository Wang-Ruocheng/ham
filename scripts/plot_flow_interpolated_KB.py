"""Color forward-interpolated microscopic K/B samples on a learned R field.

Interpolate only neighboring physical-grid triangles, never assign an inverse
coupling to arbitrary off-manifold latent points or coarse-scale encodings.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize,SymLogNorm
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recurrent_rg.checkpoints import load_run


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--artifacts',type=Path,help='Encodings/root directory; defaults to run')
    p.add_argument('--critical-reference',action='store_true')
    p.add_argument('--signed-reference',action='store_true',help='Also show K=0 and the negative zero-field critical input')
    args=p.parse_args()
    torch.set_num_threads(2);model=load_run(args.run).model
    artifacts=args.artifacts or args.run
    z=np.load(artifacts/'encodings.npz');g=np.concatenate((z['train'],z['test']))[:,0]
    labels=np.concatenate((z['train_labels'],z['test_labels']))
    ks=np.unique(labels[:,0]);bs=np.unique(labels[:,1])
    means=np.array([[g[np.all(labels==[k,b],axis=1)].mean(0) for b in bs] for k in ks])
    dense_g=[];dense_y=[]
    bary=np.array([[a/5,b/5,1-(a+b)/5] for a in range(6) for b in range(6-a)])
    for i in range(len(ks)-1):
        for j in range(len(bs)-1):
            for ids in ([(i,j),(i+1,j),(i+1,j+1)],[(i,j),(i+1,j+1),(i,j+1)]):
                points=np.array([means[a,b] for a,b in ids]);parameters=np.array([[ks[a],bs[b]] for a,b in ids])
                dense_g.append(bary@points);dense_y.append(bary@parameters)
    dense_g=np.concatenate(dense_g);dense_y=np.concatenate(dense_y)
    roots=json.loads((artifacts/'fixed_points.json').read_text())
    lo,hi=np.array(roots['search_box'])
    xx,yy=np.meshgrid(np.linspace(lo[0],hi[0],32),np.linspace(lo[1],hi[1],32));points=np.column_stack((xx.ravel(),yy.ravel()))
    with torch.no_grad():delta=model.advance_theory(torch.from_numpy(points)).numpy()-points
    arrows=delta/np.maximum(np.linalg.norm(delta,axis=1,keepdims=True),1e-12)*.033*min(hi-lo)
    norms=[Normalize(ks.min(),ks.max()),SymLogNorm(linthresh=.001,vmin=bs.min(),vmax=bs.max())]
    fig,axs=plt.subplots(1,2,figsize=(14,6.8),layout='constrained')
    parameter_grid=np.array([[k,b] for k in ks for b in bs]);flat=means.reshape(-1,2)
    for j,ax in enumerate(axs):
        cmap='viridis' if j==0 else 'coolwarm'
        sc=ax.scatter(*dense_g.T,c=dense_y[:,j],cmap=cmap,norm=norms[j],s=5,alpha=.65,linewidths=0,rasterized=True)
        ax.scatter(*flat.T,c=parameter_grid[:,j],cmap=cmap,norm=norms[j],s=16,edgecolors='black',linewidths=.25,zorder=3)
        ax.quiver(*points.T,*arrows.T,angles='xy',scale_units='xy',scale=1,color='.3',alpha=.65,width=.0025,zorder=4)
        ax.plot(*means[:,np.argmin(abs(bs))].T,'k-',lw=1.3,label='B=0 microscopic curve',zorder=5)
        if args.signed_reference:
            i0=np.argmin(abs(ks))
            if not np.isclose(ks[i0],0.):raise ValueError('Signed reference requires sampled K=0')
            ax.plot(*means[i0].T,color='#009e73',ls='--',lw=1.5,label='K=0 microscopic curve',zorder=5)
            minus_kc=-.5*np.log(1+np.sqrt(2))
            if not ks.min()<minus_kc<ks.max():raise ValueError('Grid does not bracket negative critical K')
            gminus=np.array([np.interp(minus_kc,ks,means[:,np.argmin(abs(bs)),d]) for d in range(2)])
            ax.scatter(*gminus,marker='s',s=75,c='cyan',edgecolors='black',zorder=8,label='Negative-K critical input')
        if args.critical_reference:
            kc=.5*np.log(1+np.sqrt(2));j0=np.argmin(abs(bs))
            gc=np.array([np.interp(kc,ks,means[:,j0,d]) for d in range(2)])
            ax.scatter(*gc,marker='D',s=90,c='magenta',edgecolors='black',zorder=8,label='Q(D at Kc, B=0): reference')
            ax.annotate('Critical input',gc,xytext=(7,-16),textcoords='offset points',fontsize=9,zorder=9)
        for n,item in enumerate(roots['roots'],1):
            ax.scatter(*item['g'],marker='*',s=150,c='white',edgecolors='black',linewidths=1,zorder=6)
            ax.annotate(f'F{n} ({item["stability"]})',item['g'],xytext=(6,5),textcoords='offset points',fontsize=8,zorder=7)
        ax.set(xlabel='g1',ylabel='g2',xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),aspect='equal',title='Colored by '+('K' if j==0 else 'B'))
        fig.colorbar(sc,ax=ax,shrink=.8,label='K' if j==0 else 'B (symmetric log colors)');ax.legend(fontsize=8,loc='upper right')
    fig.suptitle('R(g)-g directions with microscopic g(K,B) interpolation\nSmall dots: interpolation within physical-neighbor triangles; outlined dots: sampled cell means')
    filename='R_interpolated_KB_critical.png' if args.critical_reference else 'R_interpolated_KB.png'
    fig.savefig(artifacts/filename,dpi=180);plt.close(fig)
    (artifacts/'R_interpolated_KB_notes.json').write_text(json.dumps(dict(method='Piecewise linear forward interpolation of microscopic mean g(K,B) on neighboring physical parameter-grid triangles, five subdivisions per edge. Not inverse interpolation of arbitrary latent points.',caution='Overlapping mapped triangles can yield nonunique labels; blank regions are not assigned couplings. Colors are original microscopic K/B, not effective coarse couplings. Stars retain no assigned K/B.',fixed_point_labels={f'F{i}':r for i,r in enumerate(roots['roots'],1)}),indent=2)+'\n')
    print(artifacts/filename)


if __name__=='__main__':main()
