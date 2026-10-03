"""Compare checkpoints without letting the dominant latent PC hide closure error."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def analyze(path):
    arrays = np.load(path/'field_flow_data.npz')
    test = arrays['is_test']
    direct, rollout = arrays['direct'], arrays['rollout']
    pc_errors = []
    for k in range(1, direct.shape[1]):
        d, r = direct[test, k], rollout[test, k]
        values, axes = np.linalg.eigh(np.cov(d.T, bias=True))
        error = np.sqrt(np.mean(((r-d)@axes)**2, axis=0))
        pc_errors.append((error/np.sqrt(np.maximum(values, 1e-30))).tolist())
    report = json.loads((path/'field_diagnostics.json').read_text())
    history = json.loads((path.parent/'summary.json').read_text())['history']
    step = report['config']['checkpoint_step']
    row = next(h for h in history if h['epoch'] == step)
    return {'step': step, 'train_loss': row['train_total'], 'test_loss': row['test_total'],
            'test_rollout_nrmse': report['metrics']['test']['rollout_nrmse'][1:],
            'test_pc_nrmse_weak_strong': pc_errors,
            'weak_pc_fraction': report['microscopic_second_pc_fraction'],
            'replica_diagnostic': report['replica_diagnostic'],
            'observable_probe_test_R2': report['posthoc_quadratic_observable_readout_test_R2'],
            'fixed_points': report['fixed_points'],
            'kernel_uniform_inputs': report['kernel_uniform_inputs']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.run/'summary.json').read_text())
    steps = summary['config'].get('report_steps', sorted({1000, summary['config']['epochs']}))
    reports = [analyze(args.run/f'step{step}') for step in steps if (args.run/f'step{step}').exists()]
    history = summary['history']
    grid = np.load(args.run/f"step{reports[-1]['step']}"/'field_flow_data.npz')
    labels, is_test = grid['KB'], grid['is_test']
    train_cells = len(np.unique(labels[~is_test], axis=0))
    test_cells = len(np.unique(labels[is_test], axis=0))
    samples = summary['data']['train_shape'][1]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for split in ('train', 'test'):
        axes[0].plot([h['epoch'] for h in history], [h[f'{split}_total'] for h in history], label=split)
    axes[0].set(xlabel='Optimizer step', ylabel='Total loss', yscale='log', title='Fixed evaluation subsets')
    for report in reports:
        x = np.arange(1, len(report['test_rollout_nrmse'])+1)
        axes[1].plot(x, report['test_rollout_nrmse'], 'o-', label=f"step {report['step']}")
        pc = np.array(report['test_pc_nrmse_weak_strong'])
        for j, name in enumerate(('weak PC', 'strong PC')):
            axes[2].plot(x, pc[:, j], 'o-' if j == 0 else 'x--', label=f"{report['step']}: {name}")
    axes[1].set(xlabel='RG steps k', ylabel='RMSE / pooled coordinate SD', title='Full held-out ensembles')
    axes[2].set(xlabel='RG steps k', ylabel='PC RMSE / its own SD', yscale='log', title='Direction-resolved closure')
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(args.run/'loss_and_closure.png', dpi=200)
    plt.close(fig)
    (args.run/'checkpoint_comparison.json').write_text(json.dumps(reports, indent=2)+'\n')
    lines = [f"# 2D Ising (K,B), dim(g)=2: seed {summary['config']['seed']}", '',
             'K=beta J; B=beta h. Physical labels are excluded from all training losses.', '',
             '## Run and audit', '',
             f"- 32x32 periodic lattices; scales 32→16→8→4; {summary['config']['epochs']} Adam steps at lr=0.001.",
             f"- {summary['optimized_parameter_count']:,} optimized parameters in {summary['training_contract']['optimized_modules']}; hidden=60; system batch 32, sample batch 256.",
             f"- {len(grid['K_grid'])} K values × {len(grid['B_grid'])} B values; {len(labels):,} ensembles × {samples} samples = {len(labels)*samples:,} configurations.",
             f'- {train_cells} train cells / {test_cells} held-out cells (no shared physical parameter cells).',
             '- Finite-field Swendsen–Wang heat-bath sampling; no global-flip augmentation; Q/R have no hard Z2 constraint.',
             f"- Kernel: {'learned, no hard Z2 constraint' if summary['config'].get('learnable_kernel', True) else 'fixed positive decimation, intrinsically Z2-equivariant; identical Q/R initialization to learned-C baseline'}.",
             '- All-scale distribution geometry, one-step and multistep closure. Input-adaptive C variance floor, ratio 0.25 (constant diagnostic only if C is fixed).',
             '- Action network and RSMI are unused. Physical reference lines on figures are annotations, not learned boundaries.', '',
             '## Results', '', '| Step | Train loss | Test loss | Test rollout NRMSE k=1,2,3 | Weak-PC NRMSE k=1,2,3 |',
             '|---|---|---|---|---|']
    for r in reports:
        overall = ', '.join(f'{x:.4f}' for x in r['test_rollout_nrmse'])
        weak = ', '.join(f'{x[0]:.4f}' for x in r['test_pc_nrmse_weak_strong'])
        lines.append(f"| {r['step']} | {r['train_loss']:.6f} | {r['test_loss']:.6f} | {overall} | {weak} |")
    lines += ['', 'Small pooled error does not establish a closed two-dimensional physical RG: check the weak direction separately.',
              'Small second PCA variance is not proof of intrinsic rank one; compare replica noise and held-out observable probes.',
              'R is discrete: fixed-point stability uses |eigenvalue(dR/dg)|. Off-support roots are extrapolations.',
              ('The learned kernel may flip uniform spins, so one-step flows can alternate signs; inward arrows alone do not show contraction.'
               if summary['config'].get('learnable_kernel', True) else
               'Fixed C keeps one spin per 2x2 block without changing its sign; there is no trainable kernel or kernel-induced sign alternation.'), '',
              '## Figures', '', '[Loss and closure](loss_and_closure.png)']
    for r in reports:
        prefix = f"step{r['step']}"
        lines += ['', f"### Step {r['step']}", '', f'![R field]({prefix}/R_vector_field.png)',
                  f'![Coupling map]({prefix}/physical_parameter_map.png)',
                  f'![Latent grid]({prefix}/learned_parameter_grid.png)',
                  f'![Rescaled latent grid]({prefix}/whitened_latent_grid.png)']
    (args.run/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
