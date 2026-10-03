"""Post-hoc parameter-grid diagnostic; no network training or label gradients."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    z=np.load(args.run/'encodings.npz')
    g=np.concatenate((z['train'],z['test'])); labels=np.concatenate((z['train_labels'],z['test_labels']))
    ks=np.unique(labels[:,0]);bs=np.unique(labels[:,1])
    grid=np.array([[g[np.all(labels==[k,b],axis=1)].mean(0) for b in bs] for k in ks])
    fig,axs=plt.subplots(1,3,figsize=(14,4.6))
    for level,ax in enumerate(axs):
        for i in range(0,len(ks),2):
            ax.plot(*grid[i,:,level].T,color='tab:blue',lw=.7,alpha=.5)
        for j in range(len(bs)):
            ax.plot(*grid[:,j,level].T,color='tab:orange',lw=.7,alpha=.5)
        sc=ax.scatter(*grid[:,:,level].reshape(-1,2).T,c=np.tile(bs,len(ks)),cmap='coolwarm',norm=SymLogNorm(linthresh=.001,vmin=-.5,vmax=.5),s=9)
        ax.plot(*grid[:,np.argmin(abs(bs)),level].T,'k-',lw=1.5,label='B=0')
        ax.set(title=f'k={level}, all cells (replica means)',xlabel='g1',ylabel='g2',aspect='equal');ax.legend()
    fig.colorbar(sc,ax=axs.tolist(),label='B (symmetric log color scale)',shrink=.8)
    fig.savefig(args.run/'parameter_grid.png',dpi=180,bbox_inches='tight');plt.close(fig)
    # Same train-only quadratic probe, with either one training PC or both.
    diagnostics=[]
    for level in range(3):
        a=z['train'][:,level];b=z['test'][:,level]
        _,axes=np.linalg.eigh(np.cov(a.T,bias=True))
        a=a@axes;b=b@axes
        row={}
        for name,ids in [('weak_pc',[0]),('strong_pc',[1]),('both',[0,1])]:
            def features(x):
                v=x[:,ids]
                return np.column_stack([np.ones(len(x)),v]+[v[:,i]*v[:,j] for i in range(len(ids)) for j in range(i,len(ids))])
            pred=features(b)@np.linalg.lstsq(features(a),z['train_labels'],rcond=1e-10)[0]
            target=z['test_labels']
            row[name]=(1-((pred-target)**2).sum(0)/((target-target.mean(0))**2).sum(0)).tolist()
        diagnostics.append(row)
    (args.run/'pc_readout_diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    print(json.dumps(diagnostics,indent=2))


if __name__=='__main__': main()
