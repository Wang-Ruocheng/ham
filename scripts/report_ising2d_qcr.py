"""Additional loss, coupling calibration, and representation diagnostics for 2D QCR."""
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def project_curve(point, curve, knots):
    start, direction = curve[:-1], np.diff(curve, axis=0)
    t = np.clip(np.sum((point-start)*direction, axis=1)/np.maximum(np.sum(direction**2, axis=1), 1e-30), 0, 1)
    projected = start+t[:, None]*direction
    distance = np.linalg.norm(projected-point, axis=1)
    j = np.argmin(distance)
    return {'K': float(knots[j]+t[j]*(knots[j+1]-knots[j])),
            'distance': float(distance[j]), 'projected_g': projected[j].tolist(),
            'caution': 'Closest point on the microscopic curve, not an exact physical identification'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.run/'summary.json').read_text())
    flow = json.loads((args.run/'vector_flow_summary.json').read_text())
    data = np.load(args.run/'vector_flow_data.npz')
    coupling, direct = data['K'], data['direct']
    knots = np.unique(coupling)
    means = np.array([direct[coupling == k].mean(0) for k in knots])
    std = np.array([direct[coupling == k].std(0) for k in knots])
    kc = .5*np.log(1+np.sqrt(2))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for d in range(2):
        for level in range(direct.shape[1]):
            axes[d].plot(knots, means[:, level, d], label=f'k={level}', color=f'C{level}')
            axes[d].fill_between(knots, means[:, level, d]-std[:, level, d],
                                 means[:, level, d]+std[:, level, d], color=f'C{level}', alpha=.15)
        axes[d].axvline(kc, color='black', ls=':', label='$K_c$')
        axes[d].set(xlabel='$K=J/(k_B T)$', ylabel=f'$g^{{({d+1})}}$', title='Mean +/- replica SD (3 replicas)')
    for level in range(direct.shape[1]):
        axes[2].plot(means[:, level, 0], means[:, level, 1], label=f'k={level}', color=f'C{level}')
        critical = [np.interp(kc, knots, means[:, level, d]) for d in range(2)]
        axes[2].scatter(*critical, marker='D', color=f'C{level}', s=30)
    for i, root in enumerate(flow['fixed_points_in_plot_box']):
        axes[2].scatter(*root['g'], marker='*', color='black', s=90)
        axes[2].annotate(f'F{i+1}', root['g'], xytext=(4, 4), textcoords='offset points')
    axes[2].set(xlabel='$g^{(1)}$', ylabel='$g^{(2)}$', title='Encoded curves; diamonds = critical K')
    axes[2].set_aspect('equal', adjustable='datalim')
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(args.run/'g_vs_K_by_scale.png', dpi=200)
    plt.close(fig)
    history = summary['history']
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, name, weight in zip(axes.flat,
            ('total', 'anchor', 'coarse_consistency', 'kernel_noncollapse', 'separation', 'commutator'),
            (1, 2, 5, 2, 10, 5)):
        for split in ('train', 'test'):
            ax.plot([h['epoch'] for h in history], [weight*h[f'{split}_{name}'] for h in history], label=split)
        ax.set(title=f'{name} (weight {weight})', xlabel='Optimizer step', ylabel='Weighted contribution',
               yscale='log' if name in ('total', 'anchor') else 'linear')
        if name not in ('total', 'anchor'):
            ax.set_ylim(bottom=0)
            ax.ticklabel_format(axis='y', style='plain', useOffset=False)
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(args.run/'loss_components.png', dpi=200)
    plt.close(fig)
    roots = [dict(root, microscopic_projection=project_curve(np.array(root['g']), means[:, 0], knots))
             for root in flow['fixed_points_in_plot_box']]
    eig = np.array(flow['pooled_coordinate_diagnostic']['covariance_eigenvalues'])
    patterns = np.array([p['input'] for p in flow['kernel_truth_table']])
    outputs = np.array([p['output'] for p in flow['kernel_truth_table']])
    decimation_matches = [{'input_index': i, 'sign': sign} for i in range(4) for sign in (-1, 1)
                          if np.allclose(outputs, sign*patterns[:, i])]
    odd_violations = int(np.count_nonzero(~np.isclose(outputs, -outputs[::-1])))
    kernel_diagnostic = {'signed_decimation_matches': decimation_matches,
                         'spin_flip_oddness_violations_out_of_16': odd_violations,
                         'constant_output': bool(np.allclose(outputs, outputs[0]))}
    diagnostic = {'pooled_covariance_eigenvalues': eig.tolist(),
                  'second_pc_variance_fraction': float(eig[0]/eig.sum()),
                  'participation_rank': float(eig.sum()**2/(eig**2).sum()),
                  'fixed_points': roots, 'critical_g_by_scale': flow['g_at_critical_coupling_by_level'],
                  'kernel_diagnostic': kernel_diagnostic,
                  'final_losses': history[-1]}
    (args.run/'ising2d_diagnostics.json').write_text(json.dumps(diagnostic, indent=2)+'\n')
    np.savetxt(args.run/'g_vs_K_by_scale.csv', np.column_stack((knots, means.reshape(len(knots), -1))),
               delimiter=',', header='K,'+','.join(f'g{k}_{d}' for k in range(direct.shape[1]) for d in range(2)), comments='')
    cfg = summary['config']
    sampler_path = Path(cfg['data_cache']).with_suffix('.json')
    sampler = json.loads(sampler_path.read_text())
    last = history[-1]
    rows = []
    for level in range(1, cfg['levels']):
        rows.append(f"| {level} | {flow['metrics']['train']['rollout_normalized_rmse'][level]:.6f} | "
                    f"{flow['metrics']['test']['rollout_normalized_rmse'][level]:.6f} |")
    root_lines = []
    for i, root in enumerate(roots):
        root_lines.append(f"- F{i+1}: g={root['g']}, {root['stability']}; eigenvalues "
                          f"{root['R_jacobian_eigenvalues']}; distance to sampled R inputs "
                          f"{root['distance_to_sampled_R_input']:.6g}. Closest microscopic K="
                          f"{root['microscopic_projection']['K']:.6g}, distance="
                          f"{root['microscopic_projection']['distance']:.6g}; this is not an exact physical identification.")
    report = f'''# 2D classical Ising QCR, dim(g)=2, seed 91

## Scope and scale

Joint Q/C/R training, no coupling labels in loss, no RSMI, no action training, no explicit Z2 constraint on the network.
Nonoverlapping learned 2x2 C; 3x3 local Q; scales 32 -> 16 -> 8 -> 4.

| Quantity | Prior TFIM dim(g)=2 | This run |
|---|---:|---:|
| Coupling points | 64 | {len(knots)} |
| Replicas per point | 3 | {cfg['replicas']} |
| Configurations per replica | 128 | {cfg['ensemble_size']} |
| Train / test configurations | 16512 / 8064 | 16512 / 8064 |
| Spins per configuration | 128 | 1024 |
| Optimized Q+C+R parameters | 9613 | {flow['optimized_parameter_count']} |
| Total instantiated parameters (includes unused A) | 10027 | {flow['parameter_count']} |
| Optimizer updates | 500 | {cfg['epochs']} |
| System / original-sample batch | 32 / 128 | {cfg['system_batch']} / {cfg['sample_batch']} |

The optimized count also exceeds the earlier scalar TFIM run (9711).
Global flips augment the samples; they do not double the count of independent draws.
Hidden=59, learning rate=0.001. K range [0.20,0.68], 64 approximately uniform points with one replaced by exact Kc.
Every third coupling (index mod 3 = 1) is held out. Kc is a training coupling.

## Sampling and implementation

New cache: `{cfg['data_cache']}`. {sampler['sampler']}.
4096 fixed cluster updates for thermalization; {sampler['gap_cluster_updates']} between draws.
Pilot diagnostics use 1024 draws each from hot and cold starts at K=0.2, Kc, 0.68.
See the cache-side JSON for means, standard errors, autocorrelation and finite-chain ESS estimates.
The pilot suggests equilibration and low saved-sample autocorrelation for those observables, not a proof of independence.
The sampler was also checked against exhaustive 2x2 periodic-lattice probabilities.

Q and C use binary-pattern lookup evaluation of the same local networks, preserving first-order gradients through every C level.
This changes computation, not the model or objective. Actual seed-91 initial-loss gradient comparison had maximum absolute difference 5.8e-14.
As with other floating-point changes, nondifferentiable distance ties can choose different subgradients; bitwise training identity is not claimed.
An initial slower implementation was interrupted after roughly 130 updates; its checkpoints are separately retained in the `_preliminary_slow` directory. The reported run restarts seed 91 from initialization and completes all 500 updates; it is not a resumed or spliced trajectory.

## Loss and results

L = 2 LQ + 5 Lsplit + 2 LC + 10 Lsep + 5 Lcomm.
LQ includes split consistency, variance gauge, covariance penalty and local-pattern distance geometry.
The geometry reference is the 16-pattern 2x2 plaquette distribution. It is a local statistical anchor, not RSMI.

Final logged train loss: {last['train_total']:.9g}; test loss: {last['test_total']:.9g}.
Logged losses use fixed 24 systems and 48 original configurations, matching the prior experiment.
The following rollout metrics instead use all 128 original configurations plus flips and all systems.
NRMSE = per-coordinate RMSE divided by pooled coordinate SD at that scale.

| RG step | Train rollout NRMSE | Test rollout NRMSE |
|---|---:|---:|
{chr(10).join(rows)}

Pooled covariance eigenvalues: {eig.tolist()}.
Second principal-component variance fraction: {diagnostic['second_pc_variance_fraction']:.9g}.
Participation rank: {diagnostic['participation_rank']:.9g}.
Per-scale eigenvalues are in `vector_flow_summary.json`; pooled rank mixes scales and does not establish physical sufficiency.

Kernel truth-table diagnostic: {kernel_diagnostic}.
Oddness violations compare C(-s) with -C(s); no such constraint was imposed during training.

## Numerical fixed points

{chr(10).join(root_lines) if root_lines else 'No roots found within the plotted box with the stated search.'}

Root search is not exhaustive. Stability uses the discrete R Jacobian, not Delta-g eigenvalues.
Close proximity to samples does not establish transverse derivative accuracy, especially for thin encoded curves.
Do not require Q at microscopic Kc to equal a fixed-point coordinate: inspect its multiscale trajectory and closure first.
The last 4x4 scale has significant potential finite-size effects; no size-convergence claim is made.
No learned-action relative error is reported because this QCR run does not train an action model.

## Artifacts

- `R_vector_field.png`: discrete displacement field, observed states, actual and rolled-out trajectories.
- `vector_training_diagnostics.png`: loss and full-ensemble rollout errors.
- `loss_components.png`: weighted train/test loss terms.
- `g_vs_K_by_scale.png` and `.csv`: coupling calibration by scale, replica spread.
- `summary.json`, `vector_flow_summary.json`, `ising2d_diagnostics.json`: raw metrics.
- `qcr_model.pt`: final Q/C/R parameters (no action weights).

## Reproduce

```sh
python scripts/prepare_ising2d_qcr.py --output {cfg['data_cache']}
python experiments/joint_qcr_ising.py --dimension 2 --sites 32 --levels 4 --theory-dim 2 --hidden 59 --replicas 3 --ensemble-size 128 --epochs 500 --system-batch 32 --sample-batch 128 --eval-samples 48 --eval-systems 24 --seed 91 --threads 4 --binary-lookup --data-cache {cfg['data_cache']} --output-dir {args.run}
python scripts/plot_qcr_vector_flow.py --run {args.run} --cache {cfg['data_cache']}
python scripts/report_ising2d_qcr.py --run {args.run}
```

The preparation script refuses to overwrite an existing cache. Skip it to reuse the validated data.
'''
    (args.run/'REPORT.md').write_text(report)
    print(json.dumps(diagnostic, indent=2))


if __name__ == '__main__':
    main()
