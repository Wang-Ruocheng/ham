"""Coupling maps and fixed-coordinate closure for a saved two-dimensional TFIM run."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    out=args.run;f=np.load(out/'vector_flow_data.npz');d=f['direct'];r=f['rollout'];h=f['h_over_J'];test=f['is_test']
    knots=np.unique(h);means=np.array([d[h==x].mean(0) for x in knots]);sd=np.array([d[h==x].std(0,ddof=1) for x in knots])
    fig,axs=plt.subplots(2,3,figsize=(13,7))
    for k,ax in enumerate(axs.ravel()):
        for j in range(2):
            ax.plot(knots,means[:,k,j],color=f'C{j}',label=f'g{j+1}')
            ax.fill_between(knots,means[:,k,j]-sd[:,k,j],means[:,k,j]+sd[:,k,j],color=f'C{j}',alpha=.13)
        ax.axvline(1,color='black',ls='--',lw=1);ax.set(title=f'k={k}, L={128//2**k}',xlabel='Original h/J',ylabel='Encoded g');ax.grid(alpha=.2);ax.legend()
    config=json.loads((out/'summary.json').read_text())['config']
    loss_label='Coordinate-sum loss' if config.get('coordinate_reduction','mean')=='sum' else 'Original loss'
    fig.suptitle(f'{loss_label}, dim g=2: coupling map; shading = SD across 3 replicas')
    fig.tight_layout();fig.savefig(out/'g_vs_h_over_J.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5));ax.plot(means[:,0,0],means[:,0,1],color='.65',lw=1)
    im=ax.scatter(means[:,0,0],means[:,0,1],c=knots,cmap='viridis',s=26)
    gc=np.array([np.interp(1,knots,means[:,0,j]) for j in range(2)])
    ax.scatter(*gc,marker='*',color='red',s=130,label='Interpolated h/J=1')
    ax.set(xlabel='g1',ylabel='g2',title='Microscopic coupling curve in latent plane');ax.legend();ax.set_aspect('equal',adjustable='datalim');fig.colorbar(im,ax=ax,label='h/J')
    fig.tight_layout();fig.savefig(out/'latent_h_over_J_curve.png',dpi=180);plt.close(fig)
    training=d[~test,0];center=training.mean(0);_,s,v=np.linalg.svd(training-center,full_matrices=False);scales=s/np.sqrt(len(training)-1)
    e=(r[test]-d[test])@v.T/scales
    diag={'microscopic_train_weak_variance_fraction':float(scales[1]**2/(scales**2).sum()),
          'microscopic_train_PC_standard_deviations':scales.tolist(),
          'test_fixed_PC_rollout_NRMSE_strong_weak':np.sqrt((e**2).mean(0)).tolist(),
          'note':'PCA fitted at k0 to training only; same transform for all k. Does not identify physical scaling eigenvectors.'}
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for i,x in enumerate([.5,1.,1.5]):
        near=knots[np.argmin(abs(knots-x))];mask=h==near
        a=(d[mask]-center)@v.T/scales;b=(r[mask]-center)@v.T/scales
        for j,ax in enumerate(axs):
            ax.errorbar(range(d.shape[1]),a.mean(0)[:,j],yerr=a.std(0,ddof=1)[:,j],fmt='o-',color=f'C{i}',label=f'h/J={near:.3f}',capsize=3)
            ax.plot(range(d.shape[1]),b.mean(0)[:,j],'x--',color=f'C{i}');ax.set(xlabel='k',ylabel=f'PC{j+1} / train SD',title='Direct (solid) vs R rollout (dashed)');ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(out/'closure_fixed_PCs.png',dpi=180);plt.close(fig)
    (out/'extra_diagnostics.json').write_text(json.dumps(diag,indent=2)+'\n');print(json.dumps(diag,indent=2))


if __name__=='__main__':main()
