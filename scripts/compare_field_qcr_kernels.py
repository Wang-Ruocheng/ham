"""Paired-seed fixed/learned-C comparison without aligning arbitrary g axes."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learned', type=Path, required=True)
    parser.add_argument('--fixed', type=Path, required=True)
    args = parser.parse_args()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
    results = {}
    for label, path, color in [('Learned C', args.learned, 'C0'), ('Fixed decimation', args.fixed, 'C1')]:
        summary = json.loads((path/'summary.json').read_text())
        folder = path/f"step{summary['config']['epochs']}"
        report = json.loads((folder/'field_diagnostics.json').read_text())
        data = np.load(folder/'field_flow_data.npz')
        d, r = data['direct'][data['is_test']], data['rollout'][data['is_test']]
        pc_errors = []
        for k in range(1, d.shape[1]):
            eig, basis = np.linalg.eigh(np.cov(d[:, k].T, bias=True))
            error = np.sqrt(np.mean(((r[:, k]-d[:, k])@basis)**2, axis=0))
            pc_errors.append((error/np.sqrt(np.maximum(eig, 1e-30))).tolist())
        history = summary['history']
        for split, style in [('train', '--'), ('test', '-')]:
            axes[0].plot([h['epoch'] for h in history], [h[f'{split}_total'] for h in history],
                         style, c=color, label=f'{label}: {split}')
        steps = np.arange(1, d.shape[1])
        axes[1].plot(steps, report['metrics']['test']['rollout_nrmse'][1:], 'o-', c=color, label=label)
        for j, name, style in [(0, 'weak', 'o-'), (1, 'strong', 'x--')]:
            axes[2].plot(steps, np.array(pc_errors)[:, j], style, c=color, label=f'{label}: {name}')
        results[label] = dict(optimized_parameters=summary['optimized_parameter_count'],
                             train_loss=history[-1]['train_total'], test_loss=history[-1]['test_total'],
                             pooled_nrmse=report['metrics']['test']['rollout_nrmse'][1:],
                             pc_nrmse_weak_strong=pc_errors,
                             weak_pc_fraction=report['microscopic_second_pc_fraction'],
                             observable_readout=report['posthoc_quadratic_observable_readout_test_R2'],
                             fixed_points=report['fixed_points'])
    axes[0].set(title='Training / fixed evaluation subsets', xlabel='Optimizer step', ylabel='Total loss', yscale='log')
    axes[1].set(title='Held-out pooled closure', xlabel='RG step k', ylabel='NRMSE')
    axes[2].set(title='Held-out direction-resolved closure', xlabel='RG step k', ylabel='PC RMSE / its own SD', yscale='log')
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Same data, Q/R initialization, batches and optimizer; different coarse-graining kernels')
    fig.text(.5, .015, 'Each model uses its own g chart and generated coarse distributions. '
             'Small closure error alone does not establish a physical RG.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .94))
    fig.savefig(args.fixed/'kernel_control_comparison.png', dpi=190)
    plt.close(fig)
    (args.fixed/'kernel_control_comparison.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
