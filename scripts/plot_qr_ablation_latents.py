"""Separate raw-coordinate held-out latent plots for the four random-init arms."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, SymLogNorm


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    arms=[('iso_none','Covariance ON; Q only'),
          ('iso_both','Covariance ON; Q + R closure'),
          ('noiso_none','Covariance OFF; Q only'),
          ('noiso_both','Covariance OFF; Q + R closure')]
    data={name:np.load(args.run/name/'encodings.npz') for name,_ in arms}
    cloud=np.concatenate([z['test'][:,0] for z in data.values()])
    lo=cloud.min(0);hi=cloud.max(0);pad=.08*np.maximum(hi-lo,.1)
    lo-=pad;hi+=pad
    out=args.run/'separate_latents';out.mkdir(exist_ok=True)
    norms=[Normalize(.2,.68),SymLogNorm(linthresh=.001,vmin=-.5,vmax=.5)]
    cmaps=['viridis','coolwarm']
    def panel(ax,z,j):
        g=z['test'][:,0];labels=z['test_labels']
        scatter=ax.scatter(g[:,0],g[:,1],c=labels[:,j],cmap=cmaps[j],norm=norms[j],s=23,alpha=.85,linewidths=.2,edgecolors='white')
        ax.set(xlabel='g1',ylabel='g2',xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),aspect='equal')
        ax.grid(alpha=.15)
        return scatter
    for index,(name,title) in enumerate(arms,1):
        z=data[name];eig=np.linalg.eigvalsh(np.cov(z['test'][:,0].T,bias=True));fraction=100*eig[0]/eig.sum()
        fig,axs=plt.subplots(1,2,figsize=(11,5),layout='constrained')
        for j,ax in enumerate(axs):
            sc=panel(ax,z,j);ax.set_title('Colored by '+('K' if j==0 else 'B'))
            fig.colorbar(sc,ax=ax,shrink=.85,label='K' if j==0 else 'B (symmetric log color scale)')
        fig.suptitle(f'{index}. {title}\nHeld-out k=0, 2000 updates; weak-PC variance: {fraction:.5g}%',fontsize=12)
        fig.savefig(out/f'{name}.png',dpi=180);plt.close(fig)
    for j,label in enumerate(('K','B')):
        fig,axs=plt.subplots(2,2,figsize=(10,9),layout='constrained')
        for ax,(name,title) in zip(axs.flat,arms):
            sc=panel(ax,data[name],j);ax.set_title(title,fontsize=11)
        fig.colorbar(sc,ax=axs.ravel().tolist(),shrink=.75,label=label+(' (symmetric log)' if j else ''))
        fig.suptitle('Held-out microscopic encodings; same raw coordinates and limits')
        fig.savefig(out/f'comparison_{label}.png',dpi=180);plt.close(fig)
    print(out.resolve())


if __name__=='__main__':main()
