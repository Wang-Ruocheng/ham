"""Audit the single-seed empirical-MI intervention against InfoNCE."""
import json
import argparse
from pathlib import Path
import math
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from recurrent_rg.checkpoints import load_run
from recurrent_rg.config import RunConfig
from recurrent_rg.runner import prepared_data
from recurrent_rg.rsmi import binary_chain_joint_histograms, histogram_mutual_information, kernel_probabilities


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('results/ising1d_g1_exactEmpiricalMISeparate_w2_2400_seed91/qr'))
    parser.add_argument('--baseline',type=Path,default=Path('results/ising1d_g1_rsmiSeparate_w2_2400_seed91/qr'))
    args=parser.parse_args()
    root,old=args.run,args.baseline
    report = root/'report'
    run = load_run(root, 2400)
    cfg = RunConfig.from_dict(run.config)
    train, test = prepared_data(cfg)
    patterns = (2*((torch.arange(4)[:, None]//(2**torch.arange(2))) % 2)-1).double()[None]
    with torch.no_grad():
        learned = kernel_probabilities(run.model.rg_kernel, patterns)[0]
        tables = {'learned': learned}
        for code in range(16):
            positive = ((code//(2**torch.arange(4))) % 2).double()
            tables[str(code)] = torch.stack((1-positive, positive), -1)
        scores = {}
        for split, data in [('train', train), ('test', test)]:
            hist = binary_chain_joint_histograms(data.configurations, flip=cfg.data.spin_flip_augment)
            scores[split] = {name: float(histogram_mutual_information(hist, table).mean()/math.log(2))
                             for name, table in tables.items()}
    (report/'empirical_MI_enumeration.json').write_text(json.dumps(scores, indent=2)+'\n')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout='constrained')
    for label, path in [('Baseline', old), ('Exact MI intervention', root)]:
        history = json.loads((path/'history_progress.json').read_text())
        cweight=json.loads((path/'manifest.json').read_text())['config']['loss']['weights']['kernel_noncollapse']
        for split, style in [('train', '-'), ('test', '--')]:
            steps = [r['epoch'] for r in history]
            axes[0].plot(steps, [r[f'{split}_qcr_total']-cweight*r[f'{split}_kernel_noncollapse'] for r in history], style, label=f'{label}: {split}')
            axes[1].plot(steps, [5*(r[f'{split}_commutator']+r[f'{split}_rollout']) for r in history], style)
            if path == root:
                axes[2].plot(steps, [r[f'{split}_rsmi_bits'] for r in history], style, label=f'Learned: {split}')
                axes[2].axhline(scores[split]['10'], linestyle=':', label=f'Decimation: {split}')
    for ax, title in zip(axes, ['Common QCR loss (MI/C terms excluded)', 'R: weighted one-step + rollout', 'Exact empirical MI (bits)']):
        ax.set(xlabel='Training update', title=title); ax.grid(alpha=.2)
    axes[0].set_yscale('log'); axes[1].set_yscale('log')
    axes[0].legend(fontsize=8); axes[2].legend(fontsize=8)
    fig.savefig(report/'loss_comparison.png', dpi=160)
    history=json.loads((root/'history_progress.json').read_text())
    weight=cfg.loss.weights.kernel_noncollapse
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    for split,style in [('train','-'),('test','--')]:
        ax.plot([r['epoch'] for r in history],[weight*r[f'{split}_kernel_noncollapse'] for r in history],style,label=split)
    ax.set(xlabel='Training update',ylabel='Weighted C variance-floor loss');ax.legend();ax.grid(alpha=.2)
    fig.savefig(report/'variance_loss.png',dpi=160)
    print(json.dumps({'empirical_MI': scores, 'elapsed_seconds': json.loads((root/'summary.json').read_text())['elapsed_seconds']}, indent=2))


if __name__ == '__main__':
    main()
