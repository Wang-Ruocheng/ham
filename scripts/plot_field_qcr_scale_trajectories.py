"""Component-wise direct versus autonomous trajectories; no extrapolation."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True, help='Checkpoint report directory')
    args = parser.parse_args()
    data = np.load(args.run/'field_flow_data.npz')
    metadata = json.loads((args.run/'field_diagnostics.json').read_text())
    ks, bs = data['K_grid'], data['B_grid']
    kc = .5*np.log1p(np.sqrt(2))
    near_k = float(ks[np.argmin(abs(ks-kc))])
    near_b = float(bs[bs>1e-12].min())
    groups = [
        ('Temperature direction: $B=0$', [(.425, 0), (kc, 0), (.455, 0)]),
        (f'Field direction: K={near_k:.6f}', [(near_k, -near_b), (near_k, 0), (near_k, near_b)]),
        ('Across coexistence line: $K=0.56$', [(.56, -near_b), (.56, 0), (.56, near_b)]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    records = []
    horizon = data['direct'].shape[1]
    steps = np.arange(horizon)
    for column, (title, cases) in enumerate(groups):
        handles = []
        for index, (k, b) in enumerate(cases):
            k, b = float(ks[np.argmin(abs(ks-k))]), float(bs[np.argmin(abs(bs-b))])
            mask = np.isclose(data['KB'][:, 0], k, rtol=0, atol=1e-12) & np.isclose(data['KB'][:, 1], b, rtol=0, atol=1e-12)
            d, r = data['direct'][mask], data['rollout'][mask]
            assert len(d) == 3
            np.testing.assert_allclose(d[:, 0], r[:, 0], atol=1e-12)
            dm, rm = d.mean(0), r.mean(0)
            sd = d.std(0, ddof=1)
            label = f'K={k:.6f}, B={b:+.3f}' if column == 0 else f'B={b:+.3f}'
            color = f'C{index}'
            handles.append(Line2D([], [], color=color, lw=2, label=label))
            for component in range(2):
                ax = axes[component, column]
                ax.errorbar(steps, dm[:, component], yerr=sd[:, component], color=color,
                            fmt='o-', lw=1.8, ms=5, capsize=3, elinewidth=.8)
                ax.plot(steps, rm[:, component], 'x--', color=color, lw=1.7, ms=7)
            records.append(dict(group=title, K=k, B=b, replicas=len(d),
                                split='test' if data['is_test'][mask].all() else 'train',
                                direct_mean=dm.tolist(), rollout_mean=rm.tolist(),
                                direct_replica_sd=sd.tolist(),
                                direct=d.tolist(), rollout=r.tolist()))
        axes[0, column].set_title(title)
        axes[0, column].legend(handles=handles, fontsize=8, loc='best')
        for component in range(2):
            ax = axes[component, column]
            ax.set_ylabel(f'$g_{component+1}$')
            ax.set_xticks(steps, [f'{s}\n{32//2**s}x{32//2**s}' for s in steps])
            ax.grid(alpha=.2)
            ax.ticklabel_format(axis='y', useOffset=False)
            if component == 1:
                ax.set_xlabel('RG step k / lattice size')
    fig.suptitle(f"2D Ising QCR: seed {metadata['config']['seed']}, checkpoint {metadata['config']['checkpoint_step']}", fontsize=15)
    fig.legend(handles=[Line2D([], [], color='black', marker='o', label=r'Direct: $Q(C^kD_0)$'),
                        Line2D([], [], color='black', marker='x', ls='--', label=r'Autonomous: $R^k(Q(D_0))$')],
               loc='upper center', bbox_to_anchor=(.5, .952), ncol=2, frameon=False)
    fig.text(.5, .017, 'Means over 3 independent ensembles (256 configurations each). '
             'Error bars: direct-trajectory replica SD, not confidence intervals.\n'
             'R is iterated separately for each ensemble before averaging. Only trained scales k=0,...,3 are shown.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .065, 1, .9))
    fig.savefig(args.run/'g_components_vs_scale.png', dpi=190)
    plt.close(fig)
    report = dict(checkpoint_step=metadata['config']['checkpoint_step'],
                  scales=steps.tolist(), uncertainty='SD across 3 independent ensembles', cases=records)
    (args.run/'g_components_vs_scale.json').write_text(json.dumps(report, indent=2)+'\n')
    for rec in records:
        print(rec['group'], rec['K'], rec['B'], '\n direct=', np.round(rec['direct_mean'], 5),
              '\n rollout=', np.round(rec['rollout_mean'], 5))


if __name__ == '__main__':
    main()
