"""Component-wise coupling slices of actual Q(C^k D), not R rollouts."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--scale', type=int, default=0)
    args = parser.parse_args()
    arrays = np.load(args.run/'field_flow_data.npz')
    cfg = json.loads((args.run/'field_diagnostics.json').read_text())['config']
    if not 0 <= args.scale < arrays['direct'].shape[1]:
        parser.error('Requested scale is not available')
    ks, bs = arrays['K_grid'], arrays['B_grid']
    uniform_b = np.allclose(np.diff(bs), np.diff(bs)[0])
    means, std = np.empty((len(ks), len(bs), 2)), np.empty((len(ks), len(bs), 2))
    rows = []
    for i, k in enumerate(ks):
        for j, b in enumerate(bs):
            mask = np.isclose(arrays['KB'][:, 0], k, rtol=0, atol=1e-12) & np.isclose(arrays['KB'][:, 1], b, rtol=0, atol=1e-12)
            g = arrays['direct'][mask, args.scale]
            assert len(g) == 3
            means[i, j], std[i, j] = g.mean(0), g.std(0, ddof=1)
            rows.append(dict(K=k, B=b, scale=args.scale, replicas=len(g),
                             split='test' if arrays['is_test'][mask].all() else 'train',
                             g1_mean=means[i, j, 0], g2_mean=means[i, j, 1],
                             g1_replica_sd=std[i, j, 0], g2_replica_sd=std[i, j, 1]))
    kc = np.log1p(np.sqrt(2))/2
    kernel = 'learned C' if cfg.get('learnable_kernel', True) else 'fixed decimation'
    header = f"{kernel}; seed {cfg['seed']}; checkpoint {cfg['checkpoint_step']}; actual Q at scale k={args.scale}"
    specs = [
        ('g_vs_K', 'K', [bs[0], bs[len(bs)//2-1], 0., bs[len(bs)//2+1], bs[-1]] if uniform_b else [-.02, -.002, 0., .002, .02]),
        ('g_vs_B', 'B', [.2, .32, kc, .56, .68]),
        ('g_vs_K_Bzero', 'K', [0.]),
    ]
    paths = []
    for filename, variable, held_values in specs:
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.1))
        for n, held in enumerate(held_values):
            if variable == 'K':
                j = np.argmin(abs(bs-held))
                x, y, error = ks, means[:, j], std[:, j]
                label = f'B={bs[j]:+.3f}'
            else:
                i = np.argmin(abs(ks-held))
                x, y, error = bs, means[i], std[i]
                label = '$K=K_c$' if abs(ks[i]-kc) < 1e-10 else f'K={ks[i]:.2f}'
            color = plt.get_cmap('coolwarm')(n/(len(held_values)-1)) if len(held_values) > 1 else '#257393'
            if len(held_values) > 1 and n == len(held_values)//2:
                color = '#303030'
            for component, ax in enumerate(axes):
                ax.plot(x, y[:, component], 'o-', ms=3.5, lw=1.6, color=color, label=label)
                ax.fill_between(x, y[:, component]-error[:, component],
                                y[:, component]+error[:, component], color=color, alpha=.11)
        for component, ax in enumerate(axes):
            ax.set(ylabel=f'$g_{component+1}$', title=f'$g_{component+1}$ versus ${variable}$',
                   xlabel='$K=\\beta J$' if variable == 'K' else '$B=\\beta h$')
            if variable == 'K':
                ax.axvline(kc, color='.35', lw=1, ls='--', label='Zero-field $K_c$ (reference)')
            else:
                if not uniform_b:
                    ax.set_xscale('symlog', linthresh=.002)
                ax.axvline(0, color='.65', lw=.8, ls=':')
            ax.legend(fontsize=8, loc='best')
            ax.grid(alpha=.18)
            ax.ticklabel_format(axis='y', useOffset=False)
        fig.suptitle(header, fontsize=12)
        caption = ('Points: means of 3 independent ensembles; bands: +/-1 replica SD (not confidence intervals). '
                   'Lines connect sampled grid points.\n'
                   + (('B axis: linear.' if uniform_b else 'B axis: symmetric-log scale, linear for |B| <= 0.002.') if variable == 'B'
                      else 'The dashed Kc line is a zero-field reference, not a phase transition at nonzero B.'))
        fig.text(.5, .016, caption, ha='center', fontsize=8.5)
        fig.tight_layout(rect=(0, .08, 1, .94))
        path = args.run/f'{filename}_scale{args.scale}.png'
        fig.savefig(path, dpi=190)
        plt.close(fig)
        paths.append(str(path))
    path = args.run/f'coupling_slices_scale{args.scale}.csv'
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(figures=paths, data=str(path)), indent=2))


if __name__ == '__main__':
    main()
