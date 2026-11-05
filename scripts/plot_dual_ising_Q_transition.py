"""Plot scalar Q transition curves from saved diagnostic encodings, without R."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    with np.load(args.report/'encodings.npz') as data:
        direct=np.concatenate([data['train'],data['test']])
        labels=np.concatenate([data['train_labels'],data['test_labels']]).reshape(-1)
    if direct.shape[-1]!=1:raise ValueError('This plot requires dim g = 1')
    K=np.unique(labels)
    groups=[direct[labels==k,:,0] for k in K]
    means=np.stack([group.mean(0) for group in groups])
    std=np.stack([group.std(0) for group in groups])
    fig,ax=plt.subplots(figsize=(9,5.8),layout='constrained')
    for k in range(means.shape[1]):
        line,=ax.plot(K,means[:,k],'.-',lw=2,ms=4,label=rf'$Q(C^{k}D)$, $L={16//2**k}$')
        ax.fill_between(K,means[:,k]-std[:,k],means[:,k]+std[:,k],color=line.get_color(),alpha=.12)
    kc=np.log(1+np.sqrt(2))/2
    for i,(value,color) in enumerate(zip([kc,2*kc],['crimson','deepskyblue']),1):
        ax.axvline(value,c=color,ls='--',lw=1.5,label=rf'$K_{{c{i}}}={value:.4f}$')
    ax.set(xlabel=r'Initial $K=J_1/T$ ($K_2=K/2$)',ylabel=r'$g_k=Q(C^kD)$',
           title='Dual Ising: Q transition curves, dim g = 1\nDirect coarse-grained ensembles; no R iteration')
    ax.legend(fontsize=10);ax.grid(alpha=.2)
    fig.text(.5,-.015,'Lines: ensemble means at each K; shading: ±1 ensemble SD. Train + held-out data for diagnosis.',ha='center',fontsize=9)
    fig.savefig(args.report/'Q_transition_curves.png',dpi=180,bbox_inches='tight');plt.close(fig)
    np.savez_compressed(args.report/'Q_transition_curves.npz',K=K,mean=means,ensemble_std=std,critical_K=[kc,2*kc])


if __name__=='__main__':main()
