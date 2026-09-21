"""Compare the controlled rollout-loss ablation using saved diagnostics."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for column, (run, label) in enumerate([(args.baseline, 'One-step only'), (args.run, 'One-step + multistep')]):
        info = json.loads((run/'high_k_diagnosis.json').read_text())['original']
        xs = np.array(info['x'])
        records = [r for r in info['records'] if r['sites'] >= 4]
        steps = [r['k'] for r in records]
        for j, target in enumerate([.95, 1.05]):
            x = xs[np.argmin(abs(xs-target))]
            ids = np.flatnonzero(xs == x)
            for key, style in [('g', 'o-'), ('R_rollout', 'x--')]:
                values = [np.mean(np.array(r[key])[ids]) for r in records]
                axes[0,column].plot(steps, values, style, color=f'C{j}',
                                      label=f'h/J={x:.3f}, '+('Q(C^k D)' if key=='g' else 'R^k(Q(D))'))
        axes[0,column].axvspan(3.05,5.1,color='gray',alpha=.12)
        axes[0,column].set(title=label, xlabel='Coarse-graining step k', ylabel='g (coordinates not aligned across runs)')
        closure = json.loads((run/'qcr_closure_comparison.json').read_text())
        axes[1,0].plot([1,2,3], closure['metrics']['test']['rollout_rmse_over_direct_std'][1:], 'o-', label=label)
    summary = json.loads((args.run/'summary.json').read_text())
    for split in ['train','test']:
        axes[1,1].semilogy([h['epoch'] for h in summary['history']],
                          [h[f'{split}_total'] for h in summary['history']], label=split)
    axes[1,0].set(title='Full-test normalized rollout error', xlabel='Horizon k', ylabel='RMSE / std[Q(C^k D)]', xticks=[1,2,3])
    axes[1,1].set(title='New run: total loss including multistep term', xlabel='Update', ylabel='Loss')
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('TFIM dim(g)=1: same data, 9711 QCR parameters, seed 91, 500 updates')
    fig.text(.5,.015,'Shaded k>3 is beyond training depth. Top curves: three-replica means; bottom-left: all 63 test ensembles.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.96))
    fig.savefig(args.run/'rollout_ablation.png',dpi=180)


if __name__ == '__main__':
    main()
