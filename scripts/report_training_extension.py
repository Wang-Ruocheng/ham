"""Loss curves and fixed-window summaries for a resumed training run."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--boundary', type=int, default=6000)
    args = p.parse_args()
    history = json.loads((args.run/'history_progress.json').read_text())
    cfg = json.loads((args.run/'manifest.json').read_text())['config']
    epochs = np.array([r['epoch'] for r in history])
    out = args.run/'extension_report';out.mkdir(exist_ok=True)
    weights = dict(cfg['loss']['weights'])
    weights.update(rollout=cfg['loss']['rollout_weight'], isotropy=cfg['loss']['isotropy_weight'])
    start = next(r for r in history if r['epoch']==args.boundary)
    end = history[-1]
    def metrics(row, split):
        return dict(total=row[f'{split}_total'], weighted_components={k:w*row[f'{split}_{k}'] for k,w in weights.items() if w})
    def window(last_epoch):
        rows = [r for r in history if last_epoch-2000 < r['epoch'] <= last_epoch]
        return dict(epochs=[r['epoch'] for r in rows], **{s:dict(mean=float(np.mean([r[f'{s}_total'] for r in rows])),
                      std=float(np.std([r[f'{s}_total'] for r in rows]))) for s in ['train','test']})
    report = dict(boundary=args.boundary, final_epoch=end['epoch'],
                  at_boundary={s:metrics(start,s) for s in ['train','test']},
                  final={s:metrics(end,s) for s in ['train','test']},
                  preceding_2000_update_window=window(args.boundary), final_2000_update_window=window(end['epoch']),
                  note='Loss on the same fixed train/test evaluation subsets; raw logged values, not minibatch training losses. Window std measures temporal fluctuation, not seed uncertainty.')
    fig, axes = plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax in axes:
        for split,color in [('train','#2166ac'),('test','#b2182b')]:
            ax.plot(epochs,[r[f'{split}_total'] for r in history],color=color,label=split,lw=1.4)
        ax.axvline(args.boundary,color='gray',ls='--',label='resume boundary')
        ax.set(xlabel='Training updates',ylabel='Total loss');ax.grid(alpha=.2);ax.legend()
    axes[0].set(title='Full training history')
    tail = [r for r in history if r['epoch'] >= max(0,args.boundary-2000)]
    values = [r[f'{s}_total'] for r in tail for s in ['train','test']]
    padding = max(.02,.1*np.ptp(values))
    axes[1].set(title='Late training (zoom)',xlim=(max(0,args.boundary-2000),end['epoch']),ylim=(min(values)-padding,max(values)+padding))
    fig.savefig(out/'total_loss.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,split in zip(axes,['train','test']):
        for key,w in weights.items():
            if w:ax.plot(epochs,[w*r[f'{split}_{key}'] for r in history],label=key)
        ax.axvline(args.boundary,color='gray',ls='--')
        ax.set(xlabel='Training updates',ylabel='Weighted component',title=split,yscale='symlog');ax.set_yscale('symlog',linthresh=.001)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(out/'components.png',dpi=180);plt.close(fig)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
