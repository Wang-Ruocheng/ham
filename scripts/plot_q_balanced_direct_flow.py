"""Plot saved Q(D)->Q(CD) displacements, without fitting or evaluating R."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--arm', choices=['source', 'baseline', 'balanced'], default='balanced')
    args = parser.parse_args()
    manifest = json.loads((args.run/'manifest.json').read_text())
    if manifest['config']['data']['coupling_names'] != ['K1', 'K2']:
        raise ValueError('Requires explicitly documented K1/K2 data, not a K/B grid')
    saved = np.load(args.run / f'{args.arm}_encodings.npz')
    labels = np.concatenate([saved['train_labels'], saved['test_labels']])
    direct = np.concatenate([saved['train'], saved['test']])
    if labels.shape != (len(direct), 2) or direct.ndim != 3 or direct.shape[-1] != 2:
        raise ValueError('Requires the independent dual-Ising two-dimensional grid encodings')
    axes = [np.unique(labels[:, j]) for j in range(2)]
    cells = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 2)
    values = []
    for point in cells:
        selected = np.max(np.abs(labels-point), axis=1) < 1e-8
        if not selected.any():
            raise ValueError('Incomplete coupling grid')
        values.append(direct[selected].mean(0))
    means = np.stack(values)
    grid_values = means.reshape(len(axes[0]), len(axes[1]), *direct.shape[1:])
    kc = float(np.log(1+np.sqrt(2))/2)
    phase = (cells[:, 0] > kc).astype(int) + 2*(cells[:, 1] > kc).astype(int)
    colors = np.array(['#444444', '#0072b2', '#d55e00', '#009e73'])
    names = ['both disordered', 'only layer 1 ordered', 'only layer 2 ordered', 'both ordered']

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    steps = direct.shape[1]-1
    fig, axs = plt.subplots(1, steps, figsize=(7*steps, 6.5), layout='constrained', squeeze=False)
    reports = []
    for k, ax in enumerate(axs[0]):
        source, target = means[:, k], means[:, k+1]
        delta = target-source
        ax.quiver(*source.T, *delta.T, color=colors[phase], angles='xy',
                  scale_units='xy', scale=1, pivot='tail', width=.003,
                  headwidth=4, headlength=5, minlength=0, alpha=.75, zorder=3)
        ax.scatter(*source.T, c=colors[phase], s=11, alpha=.65, zorder=4)
        ax.scatter(*target.T, facecolors='none', edgecolors=colors[phase],
                   s=17, linewidths=.6, alpha=.7, zorder=4)
        interpolator = RegularGridInterpolator(tuple(axes), grid_values[:, :, k], bounds_error=True)
        for j, color in enumerate(['#b2182b', '#762a83']):
            axis = axes[1-j]
            parameters = np.column_stack([np.full(301, kc), np.linspace(axis[0], axis[-1], 301)])
            if j == 1:
                parameters = parameters[:, ::-1]
            curve = interpolator(parameters)
            ax.plot(*curve.T, color=color, ls='--', lw=1.4, alpha=.7)
        critical = interpolator([[kc, kc]])[0]
        ax.scatter(*critical, marker='D', c='black', s=38, zorder=5)
        ax.set(xlabel='g1', ylabel='g2', title=f'Direct Q flow: k={k} -> {k+1}', aspect='equal')
        ax.grid(alpha=.15)
        reports.append(dict(source_scale=k, target_scale=k+1,
                            bare_double_critical_encoding=critical.tolist(),
                            mean_displacement_norm_by_bare_phase={name: float(np.linalg.norm(delta[phase==i], axis=1).mean())
                                                                 for i, name in enumerate(names)}))
    legend = [Line2D([], [], color=c, marker='o', lw=1, label=n) for c, n in zip(colors, names)]
    legend += [Line2D([], [], color=c, ls='--', label=f'Bare K{j+1}=Kc mapped by Q; NOT separatrix')
               for j, c in enumerate(['#b2182b', '#762a83'])]
    legend += [Line2D([], [], color='black', marker='D', ls='none', label='Interpolated bare (Kc,Kc); NOT R root')]
    axs[0, 0].legend(handles=legend, fontsize=7, loc='best')
    fig.suptitle(f'{args.arm}: Q(C^(k+1)D)-Q(C^k D), actual endpoint displacements\n'
                 'Filled: source; hollow: target. No R fitting, no normalized arrows, no field extrapolation.')
    output = args.run / f'{args.arm}_direct_Q_flow.png'
    fig.savefig(output, dpi=160)
    plt.close(fig)
    report = dict(arm=args.arm, coupling_cells=len(cells), source=str(args.run), steps=reports,
                  caveat='Single cached stochastic-C draw, two-ensemble means per coupling. '
                         'Only observed displacement arrows; critical curves are bare-label encodings, '
                         'not inferred invariant boundaries or fixed points. R has not been refitted.')
    (args.run/f'{args.arm}_direct_Q_flow.json').write_text(json.dumps(report, indent=2)+'\n')
    np.savez_compressed(args.run/f'{args.arm}_direct_Q_flow.npz', couplings=cells,
                        direct_means=means, displacement=means[:, 1:]-means[:, :-1])
    print(output)


if __name__ == '__main__':
    main()
