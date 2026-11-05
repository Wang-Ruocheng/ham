"""Checkpoint-only diagnostics of the independent two-coupling Ising grid."""
import json

import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from recurrent_rg.evaluation import closure_metrics


def grid_encodings(labels, direct):
    """Average replicas, without flattening the two physical parameter axes."""
    labels, direct = np.asarray(labels), np.asarray(direct)
    if labels.ndim != 2 or labels.shape != (len(direct), 2):
        raise ValueError('Independent grid labels must have two columns')
    axes = [np.unique(labels[:, j]) for j in range(2)]
    values = np.empty((len(axes[0]), len(axes[1]), *direct.shape[1:]))
    for i, k1 in enumerate(axes[0]):
        for j, k2 in enumerate(axes[1]):
            selected = (labels[:, 0] == k1) & (labels[:, 1] == k2)
            if not selected.any():
                raise ValueError(f'Incomplete Cartesian grid at {(k1, k2)}')
            values[i, j] = direct[selected].mean(0)
    return axes, values


def critical_curves(axes, values, *, points=301):
    """Map exact bare critical lines into g at each scale, for annotation only."""
    kc = float(np.log(1 + np.sqrt(2)) / 2)
    if any(not a[0] <= kc <= a[-1] for a in axes):
        raise ValueError('Grid does not cover both Ising critical lines')
    u1 = np.linspace(axes[0][0], axes[0][-1], points)
    u2 = np.linspace(axes[1][0], axes[1][-1], points)
    parameters = [np.column_stack((np.full_like(u2, kc), u2)),
                  np.column_stack((u1, np.full_like(u1, kc)))]
    curves, centers = [], []
    for k in range(values.shape[2]):
        f = RegularGridInterpolator(tuple(axes), values[:, :, k], bounds_error=True)
        curves.append([f(p) for p in parameters])
        centers.append(f([[kc, kc]])[0])
    return kc, parameters, curves, np.asarray(centers)


def write_grid_report(run, train, test, train_data, test_data, out, roots, sampling, fingerprint):
    direct = np.concatenate((train['direct'], test['direct']))
    labels = np.concatenate((train_data.couplings.numpy(), test_data.couplings.numpy()))
    axes, values = grid_encodings(labels, direct)
    steps = direct.shape[1]
    kc, parameters, curves, centers = critical_curves(axes, values)
    cells = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 2)
    means = values.reshape(-1, steps, 2)
    np.savez_compressed(out / 'grid_encodings.npz', K1=axes[0], K2=axes[1], direct_mean=values,
                        critical_parameters=np.stack(parameters), critical_curves=np.asarray(curves),
                        double_critical_encodings=centers)
    fig, axs = plt.subplots(2, steps, figsize=(5 * steps, 8), squeeze=False, layout='constrained')
    for j in range(2):
        for k in range(steps):
            ax = axs[j, k]
            z = ax.pcolormesh(*axes, values[:, :, k, j].T, shading='nearest', cmap='viridis')
            ax.axvline(kc, color='crimson', ls=':')
            ax.axhline(kc, color='deepskyblue', ls=':')
            ax.set(xlabel='Initial K1', ylabel='Initial K2', title=f'g{j+1} = Q(C^{k}D)')
            fig.colorbar(z, ax=ax)
    fig.savefig(out / 'coupling_by_scale.png', dpi=160)
    plt.close(fig)
    phase = (cells[:, 0] > kc).astype(int) + 2 * (cells[:, 1] > kc).astype(int)
    phase_names = ['both disordered', 'only layer 1 ordered', 'only layer 2 ordered', 'both ordered']
    phase_colors = ['#444444', '#0072b2', '#d55e00', '#009e73']
    fig, axs = plt.subplots(1, steps, figsize=(5 * steps, 4.7), squeeze=False, layout='constrained')
    for k, ax in enumerate(axs[0]):
        for i, (name, color) in enumerate(zip(phase_names, phase_colors)):
            selected = phase == i
            ax.scatter(*means[selected, k].T, s=15, color=color, label=name)
        ax.set(xlabel='g1', ylabel='g2', title=f'Q(C^{k}D): exact bare phase labels')
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.savefig(out / 'latent_phases.png', dpi=160)
    plt.close(fig)
    lower, upper = np.asarray(roots['search_box'])
    mesh = np.meshgrid(*[np.linspace(a, b, 31) for a, b in zip(lower, upper)])
    grid = np.stack(mesh, axis=-1).reshape(-1, 2)
    with torch.no_grad():
        delta = run.model.advance_theory(torch.from_numpy(grid)).numpy() - grid
    speed = np.linalg.norm(delta, axis=1)
    direction = delta / np.maximum(speed[:, None], 1e-12)

    def draw(ax, k):
        quiver = ax.quiver(*grid.T, *direction.T, speed, cmap='viridis', alpha=.55)
        for i, color in enumerate(phase_colors):
            ax.scatter(*means[phase == i, k].T, s=9, color=color, alpha=.6)
        for j, color in enumerate(['crimson', 'deepskyblue']):
            ax.plot(*curves[k][j].T, color=color, lw=2,
                    label=f'Bare K{j+1}=Kc, mapped at k={k}')
        ax.scatter(*centers[k], marker='D', color='black', s=55, label='Mapped (Kc,Kc); NOT an R root')
        for i, root in enumerate(roots['roots']):
            ax.scatter(*root['g'], marker='*', color='gold', edgecolors='black', s=120)
            ax.annotate(f'F{i+1}: {root["stability"]}', root['g'], xytext=(4, 4),
                        textcoords='offset points', fontsize=7)
        ax.set(xlabel='g1', ylabel='g2', xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]))
        ax.grid(alpha=.15)
        ax.legend(fontsize=7)
        return quiver

    fig, ax = plt.subplots(figsize=(9, 7), layout='constrained')
    z = draw(ax, 0)
    ax.set_title('R(g)-g: normalized arrows; off-data flow is extrapolation\n'
                 'Mapped bare critical lines are NOT fitted R separatrices')
    fig.colorbar(z, ax=ax, label='|R(g)-g|')
    fig.savefig(out / 'R_with_theory_boundaries.png', dpi=160)
    plt.close(fig)
    fig, axs = plt.subplots(1, steps, figsize=(6 * steps, 5.5), squeeze=False, layout='constrained')
    for k, ax in enumerate(axs[0]):
        draw(ax, k)
        ax.set_title(f'Bare critical lines mapped at scale k={k}')
    fig.savefig(out / 'theory_boundaries_by_scale.png', dpi=160)
    plt.close(fig)
    report = dict(checkpoint=str(run.checkpoint), checkpoint_epoch=run.payload.get('epoch'),
                  data_sha256=fingerprint, grid_mode='independent', coupling_names=['K1', 'K2'],
                  grid_axes=[a.tolist() for a in axes], critical_K1=kc, critical_K2=kc,
                  double_critical_encodings_by_scale=centers.tolist(),
                  heldout_closure=closure_metrics(test, reference_direct=train['direct']),
                  covariance_eigenvalues_by_scale=[np.linalg.eigvalsh(np.cov(direct[:, k].T, bias=True)).tolist()
                                                  for k in range(steps)],
                  sampling_seed=12345, lattice_sizes=[sampling['L'] // run.model_config.block_factor ** k
                                                    for k in range(steps)], fixed_points=roots,
                  caveat='One stochastic C draw; bounded root search is not exhaustive. Bilinear interpolation '
                         'uses train/test cells only for evaluation. Mapped microscopic critical lines may drift '
                         'and are not stationary RG boundaries; off-data flow is extrapolation. No coupling labels '
                         'enter training; finite-L phase labels are thermodynamic reference annotations.')
    (out / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    (out / 'critical_markers.json').write_text(json.dumps(dict(critical_K=kc, grid_mode='independent',
        encodings_by_scale=centers.tolist(), grid_spacing=[np.diff(a).tolist() for a in axes]), indent=2) + '\n')
    return report
