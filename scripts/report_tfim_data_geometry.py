"""Summarize the controlled single-seed data-geometry objective experiment."""
import json
import argparse
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recurrent_rg.model_nd import RecurrentRGNetwork1D


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=root/'results/joint_qcr_tfim_1d_g1_v2_h60_sb128_block2_data_geometry_seed91')
    parser.add_argument('--baseline', type=Path, default=root/'results/joint_qcr_tfim_1d_g1_v2_h60_sb128_block2')
    args = parser.parse_args()
    run, baseline = args.run, args.baseline
    summary = json.loads((run/'summary.json').read_text())
    closure = json.loads((run/'qcr_closure_comparison.json').read_text())
    old_closure = json.loads((baseline/'qcr_closure_comparison.json').read_text())
    local = json.loads((run/'local_g_critical_flow.json').read_text())
    cfg = summary['config']
    model = RecurrentRGNetwork1D(levels=4, hidden=60, theory_dim=1).double().eval()
    model.load_state_dict(torch.load(run/'qcr_model.pt', weights_only=True), strict=False)
    optimized = sum(p.numel() for module in [model.theory_encoder, model.rg_kernel, model.theory_rg] for p in module.parameters())
    pairs = torch.tensor([[-1,-1],[-1,1],[1,-1],[1,1]], dtype=torch.float64)
    with torch.no_grad():
        truth = model.rg_kernel(pairs.reshape(4,1,2,1)).flatten().tolist()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    history = summary['history']
    epochs = [h['epoch'] for h in history]
    for split in ['train', 'test']:
        axes[0,0].semilogy(epochs, [h[f'{split}_total'] for h in history], label=split)
    axes[0,0].set(title='New objective: training and test loss', xlabel='Update', ylabel='Total loss')
    axes[0,0].legend()
    weights = {'anchor':2, 'coarse_consistency':5, 'kernel_noncollapse':2, 'separation':10, 'commutator':5}
    labels = {'anchor':'microscopic split', 'coarse_consistency':'coarse split',
              'kernel_noncollapse':'C balance', 'separation':'data geometry', 'commutator':'QCR closure'}
    if cfg.get('rollout_weight', 0):
        weights['rollout'] = cfg['rollout_weight']
        labels['rollout'] = 'multistep closure'
    for name, weight in weights.items():
        if name == 'kernel_noncollapse':
            continue
        axes[0,1].semilogy(epochs, [max(weight*h[f'test_{name}'],1e-10) for h in history], label=labels[name])
    axes[0,1].set(title='Weighted test-loss components', xlabel='Update', ylabel='Contribution')
    axes[0,1].legend(fontsize=8)
    support = summary['learned_R']['trained_support']
    grid = np.linspace(support[0]-.15, support[1]+.15, 500)
    with torch.no_grad():
        prediction = model.advance_theory(torch.tensor(grid[:,None])).numpy()[:,0]
    axes[1,0].plot(grid, prediction, label='R(g)')
    axes[1,0].plot(grid, grid, 'k--', label='identity')
    for fp in summary['learned_R']['fixed_points_in_support']:
        axes[1,0].scatter(fp['g'], fp['g'], marker='o', s=50,
                          label=f"{fp['stability']}: {fp['g']:.3f}")
    axes[1,0].axvline(local['center'], color='gray', ls=':', label='interpolated g0(h/J=1)')
    axes[1,0].set(title='R map and numerical fixed points', xlabel='g', ylabel='R(g)')
    axes[1,0].legend(fontsize=8)
    for initial, flow in zip(local['initials'], local['R_rollouts']):
        axes[1,1].plot(range(len(flow)), flow, '.-', label=f'{initial:.3f}')
    axes[1,1].set(title='R-only rollout near critical microscopic encoding', xlabel='Iteration k', ylabel='g')
    axes[1,1].legend(title='initial g', fontsize=7, ncol=2)
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.suptitle('TFIM, dim(g)=1, seed 91: equal-scale data-driven separation', fontsize=14)
    fig.text(.5,.015,'Only k=0..3 are constrained by actual coarse-grained targets. New/old total losses are not directly comparable.', ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.96))
    fig.savefig(run/'experiment_overview.png',dpi=200)
    plt.close(fig)
    report = {'optimized_parameters':optimized, 'instantiated_parameters':sum(p.numel() for p in model.parameters()),
              'C_inputs':pairs.tolist(),'C_outputs':truth,
              'old_test_rollout_NRMSE':old_closure['metrics']['test']['rollout_rmse_over_direct_std'],
              'new_test_rollout_NRMSE':closure['metrics']['test']['rollout_rmse_over_direct_std'],
              'test_loss_components_weighted':{name:weight*history[-1][f'test_{name}'] for name,weight in weights.items()},
              'critical_g0':local['center'], 'critical_direct':closure['direct_at_1'],
              'nearby_R_rollout_endpoints':[r[-1] for r in local['R_rollouts']],
              'fixed_points':summary['learned_R']['fixed_points_in_support']}
    (run/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
