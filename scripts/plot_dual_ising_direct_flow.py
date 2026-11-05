"""Actual direct-Q trajectories; no R, interpolation, or arrow normalization."""
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    root=Path('results/dual_ising2d_g2_uniformK1K2_fixed_1200_seed91')
    out=root/'report'
    saved=torch.load(root/'R_only_refit_6000'/'fixed_targets.pt',weights_only=True)
    g=torch.cat([saved['train'],saved['test']]).numpy()
    labels=torch.cat([saved['train_couplings'],saved['test_couplings']]).numpy()
    cells=np.unique(labels,axis=0)
    means=np.stack([g[(labels==p).all(1)].mean(0) for p in cells])
    kc=np.log(1+np.sqrt(2))/2
    phase=(cells[:,0]>kc).astype(int)+2*(cells[:,1]>kc).astype(int)
    colors=['#1f77b4','#ff7f0e','#2ca02c','#d62728']
    names=['Both disordered','Only layer 1 ordered','Only layer 2 ordered','Both ordered']
    audit=np.load(root/'critical_sampling_audit'/'trajectories.npz')
    ci=np.flatnonzero(np.isclose(audit['couplings'],kc).all(1))[0]
    critical=audit['direct'][ci].mean(0)
    def trajectory(ax,z,color,alpha=.65):
        for k,style in enumerate(['-','--']):
            ax.annotate('',xy=z[k+1],xytext=z[k],arrowprops=dict(arrowstyle='-|>',lw=1.1,color=color,alpha=alpha,linestyle=style,shrinkA=0,shrinkB=0,mutation_scale=9))
        for k,marker in enumerate(['o','s','^']):
            ax.scatter(*z[k],c=color,s=12,marker=marker,alpha=alpha,zorder=3)
    fig,axs=plt.subplots(1,2,figsize=(15,7),layout='constrained')
    for i,z in enumerate(means):
        trajectory(axs[0],z,colors[phase[i]])
        if phase[i]==3:trajectory(axs[1],z,colors[3],.8)
    trajectory(axs[0],critical,'black',1)
    for k,z in enumerate(critical):axs[0].annotate(f'critical k={k}',z,xytext=(8,5),textcoords='offset points',fontsize=8)
    for value,offset in [(.5,(-55,-20)),(.6,(-70,12)),(.8,(-65,-28))]:
        idx=np.flatnonzero(np.isclose(cells,[value,value]).all(1))[0]
        axs[1].annotate(f'K1=K2={value}',means[idx,0],xytext=offset,textcoords='offset points',fontsize=9,arrowprops=dict(arrowstyle='-',color='.4'))
    axs[0].set(title='All 121 coupling cells; three-replica mean')
    red=means[phase==3].reshape(-1,2);lo=red.min(0)-.04;hi=red.max(0)+.04
    axs[1].set(title='Both-ordered region: enlarged',xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]))
    handles=[Line2D([],[],color=c,marker='o',lw=1,label=n) for c,n in zip(colors,names)]
    handles += [Line2D([],[],color='black',lw=1,label='Fresh (Kc,Kc) sampling')]
    handles += [Line2D([],[],color='.4',marker=m,ls='none',label=f'k={k}') for k,m in enumerate(['o','s','^'])]
    axs[0].legend(handles=handles,fontsize=8,loc='lower right')
    for ax in axs:
        ax.set(xlabel='g1',ylabel='g2');ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.15)
    fig.suptitle('Direct Q(C^k D), L=16 -> 8 -> 4; NO R predictions\nSolid arrows: k=0 -> 1; dashed arrows: k=1 -> 2. Actual endpoint displacements, NOT normalized.')
    fig.savefig(out/'direct_Q_flow.png',dpi=190)
    np.savez_compressed(out/'direct_Q_flow.npz',couplings=cells,direct_means=means,critical_direct=critical)
    print(out/'direct_Q_flow.png')


if __name__=='__main__':main()
