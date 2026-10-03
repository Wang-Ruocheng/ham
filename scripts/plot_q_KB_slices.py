"""Parameter-resolved microscopic Q maps, using saved encodings only."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, SymLogNorm
from matplotlib.cm import ScalarMappable


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    z=np.load(args.run/'encodings.npz')
    g=np.concatenate((z['train'],z['test']))[:,0]
    labels=np.concatenate((z['train_labels'],z['test_labels']))
    ks=np.unique(labels[:,0]);bs=np.unique(labels[:,1])
    means=np.array([[g[np.all(labels==[k,b],axis=1)].mean(0) for b in bs] for k in ks])
    out=args.run/'KB_slices';out.mkdir(exist_ok=True)
    bnorm=SymLogNorm(linthresh=.001,vmin=bs.min(),vmax=bs.max());knorm=Normalize(ks.min(),ks.max())
    selected_b=[-.5,-.05,-.005,-.001,0,.001,.005,.05,.5]
    fig,axs=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    for coord in range(2):
        ax=axs[coord,0]
        for b in selected_b:
            j=np.argmin(abs(bs-b));ax.plot(ks,means[:,j,coord],'-o',ms=3,lw=1,color='black' if b==0 else plt.cm.coolwarm(bnorm(b)),label=f'B={b:g}')
        ax.set(xlabel='K',ylabel=f'g{coord+1}',title=f'g{coord+1}(K), fixed B');ax.grid(alpha=.15)
        ax=axs[coord,1]
        for i in range(0,len(ks),2):ax.plot(bs,means[i,:,coord],'-o',ms=3,lw=1,color=plt.cm.viridis(knorm(ks[i])))
        ax.set_xscale('symlog',linthresh=.001)
        ax.set(xlabel='B (symmetric log axis)',ylabel=f'g{coord+1}',title=f'g{coord+1}(B), fixed K');ax.grid(alpha=.15)
    axs[0,0].legend(ncol=3,fontsize=8)
    fig.colorbar(ScalarMappable(norm=knorm,cmap='viridis'),ax=axs[:,1].tolist(),label='K',shrink=.8)
    fig.suptitle('Fixed positive decimation, covariance weight20: microscopic Q(D)\nEach point is a mean over replicas; train + held-out parameter cells')
    fig.savefig(out/'g_vs_KB.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for coord,ax in enumerate(axs):
        sc=ax.pcolormesh(ks,bs,means[:,:,coord].T,shading='nearest',cmap='viridis')
        ax.set_yscale('symlog',linthresh=.001)
        ax.axhline(0,color='white',ls='--',lw=.8)
        ax.set(xlabel='K',ylabel='B (symmetric log axis)',title=f'g{coord+1}(K,B) at k=0')
        fig.colorbar(sc,ax=ax,label=f'g{coord+1}')
    fig.savefig(out/'g_KB_heatmaps.png',dpi=180);plt.close(fig)
    np.savez(out/'grid.npz',K=ks,B=bs,g=means)
    print(out.resolve())


if __name__=='__main__':main()
