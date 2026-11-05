"""Empirical one-step Q field on observed ensembles, without fitting any R."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D


def main():
    out=Path('results/dual_ising2d_g2_dense_joint_fast_2400_seed91/report')
    data=np.load(out/'trajectories.npz')
    labels=np.concatenate([data['train_couplings'],data['test_couplings']])
    direct=np.concatenate([data['train_direct'],data['test_direct']])
    cells=np.unique(labels,axis=0)
    means=np.stack([direct[(labels==p).all(1)].mean(0) for p in cells])
    source=means[:,0];target=means[:,1];delta=target-source;speed=np.linalg.norm(delta,axis=1)
    boundaries=np.load(out/'theory_boundaries.npz')['curves'][0]
    audit=np.load(out/'critical_audit.npz');kc=np.log(1+np.sqrt(2))/2
    idx=np.flatnonzero(np.isclose(audit['couplings'],kc).all(1))[0]
    critical=audit['direct'][idx].mean(0);cd=critical[1]-critical[0]
    norm=Normalize(0,speed.max())
    fig,axs=plt.subplots(1,2,figsize=(15,7),layout='constrained')
    for ax in axs:
        ax.scatter(*source.T,c='.45',s=9,alpha=.5,zorder=3)
        ax.scatter(*target.T,facecolors='none',edgecolors='.5',s=13,lw=.5,alpha=.5)
        arrows=ax.quiver(*source.T,*delta.T,speed,cmap='viridis',norm=norm,
            angles='xy',scale_units='xy',scale=1,pivot='tail',width=.004,headwidth=3.8,headlength=4.8,minlength=0,zorder=4)
        for curve,color in zip(boundaries,['#d81b60','#0072b2']):ax.plot(*curve.T,c=color,lw=1.6,alpha=.8,zorder=2)
        ax.scatter(*critical[0],c='black',s=45,marker='D',zorder=6)
        ax.scatter(*critical[1],facecolors='white',edgecolors='black',s=50,marker='D',zorder=6)
        ax.quiver(*critical[0],*cd,color='black',angles='xy',scale_units='xy',scale=1,pivot='tail',width=.006,minlength=0,zorder=5)
        ax.set(xlabel='g1',ylabel='g2');ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.15)
    cloud=means[:,:2].reshape(-1,2);lo=cloud.min(0)-.07;hi=cloud.max(0)+.07
    axs[0].set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),title='All observed coupling cells')
    axs[1].set(xlim=(critical[0,0]-.35,critical[0,0]+.4),ylim=(critical[0,1]-.3,critical[0,1]+.25),title='Near the double-critical ensemble')
    handles=[Line2D([],[],color='#d81b60',label='K1=Kc mapped at k=0'),Line2D([],[],color='#0072b2',label='K2=Kc mapped at k=0'),
        Line2D([],[],color='.5',marker='o',ls='none',label='Q(D): filled / Q(CD): hollow'),
        Line2D([],[],color='black',marker='D',label='Direct sampled (Kc,Kc)')]
    axs[0].legend(handles=handles,fontsize=8,loc='lower left')
    fig.colorbar(arrows,ax=axs,label='Actual displacement |Q(CD)-Q(D)|',shrink=.8)
    fig.suptitle('Empirical Q(D) -> Q(CD), L=16 -> 8\nNo R; arrow endpoints and lengths are actual replica-mean displacements (not normalized)')
    fig.savefig(out/'QCD_empirical_field.png',dpi=190)
    np.savez_compressed(out/'QCD_empirical_field.npz',couplings=cells,source=source,target=target,displacement=delta,critical_source=critical[0],critical_target=critical[1])
    report=dict(coupling_cells=len(cells),critical_source=critical[0].tolist(),critical_target=critical[1].tolist(),critical_displacement=cd.tolist(),
        note='Observed source points only; no interpolation/extrapolation of vector field. Replica means from cached stochastic C evaluation. Colored boundary curves show k=0 encodings, not validated invariant boundaries.')
    (out/'QCD_empirical_field.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
