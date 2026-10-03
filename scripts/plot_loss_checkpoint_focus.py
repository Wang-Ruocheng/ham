"""Logged weighted losses, highlighting a checkpoint without retraining."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--highlight',type=int,default=800)
    p.add_argument('--zoom-start',type=int,default=400)
    args=p.parse_args()
    cfg=json.loads((args.run/'manifest.json').read_text())['config']
    history=json.loads((args.run/'history_progress.json').read_text())
    steps=np.array([row['epoch'] for row in history]);late=steps>=args.zoom_start
    weights=dict(cfg['loss']['weights']);weights.update(rollout=cfg['loss']['rollout_weight'])
    for term,key in [('kernel_sign','kernel_sign_weight'),('isotropy','isotropy_weight'),('orientation','orientation_weight'),('tangent','tangent_weight')]:
        if cfg['loss'].get(key,0):weights[term]=cfg['loss'][key]
    names={'anchor':'microscopic subset consistency','coarse_consistency':'coarse subset consistency',
           'separation':'distribution geometry','commutator':'one-step closure','rollout':'multistep closure',
           'kernel_noncollapse':'C variance floor','kernel_sign':'C color preservation'}
    fig,axs=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for split in ('train','test'):
        values=np.array([row[f'{split}_total'] for row in history])
        axs[0,0].plot(steps,values,label=split)
        axs[0,1].plot(steps[late],values[late],'-o',ms=3,label=split)
    axs[0,0].set(title='Total loss: all updates',yscale='log',ylabel='total loss')
    axs[0,1].set(title=f'Total loss: updates >= {args.zoom_start}',ylabel='total loss (linear scale)')
    for ax,split in zip(axs[1],('train','test')):
        for key,weight in weights.items():
            if weight:
                values=np.array([weight*row[f'{split}_{key}'] for row in history])
                ax.plot(steps[late],values[late],label=names.get(key,key))
        ax.set(title=f'{split}: weighted components',ylabel='contribution to total')
        ax.set_yscale('symlog',linthresh=.001);ax.set_ylim(bottom=0)
    for ax in axs.flat:
        ax.axvline(args.highlight,color='purple',ls='--',lw=1.3,label=f'{args.highlight} updates')
        ax.set_xlabel('updates');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Potts q=3, dim(g)=1, C preservation weight=0; seed91\nLogged losses on fixed 32 train / 32 test ensembles',fontsize=13)
    out=args.run/'full_report';out.mkdir(exist_ok=True)
    fig.savefig(out/'loss_checkpoint_focus.png',dpi=170);plt.close(fig)
    print(out/'loss_checkpoint_focus.png')


if __name__=='__main__':main()
