"""Saved-checkpoint diagnostics for single-layer classical Ising K/B scans.

No training, new configurations, or MI estimation. Physical reference markers
are forward interpolations of Q, not fitted RG fixed points or separatrices.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

from recurrent_rg.checkpoints import load_run
from recurrent_rg.config import RunConfig
from recurrent_rg.evaluation import collect_trajectories, closure_metrics
from recurrent_rg.plotting import plot_diagnostics
from recurrent_rg.runner import file_fingerprint, prepared_data
from scripts.dual_ising_grid_diagnostics import grid_encodings


def field_reference(axes, values):
    """Interpolate the bare zero-field line and critical input, never invert g."""
    kc = float(np.log(1 + np.sqrt(2)) / 2)
    if not axes[0][0] <= kc <= axes[0][-1] or not axes[1][0] <= 0 <= axes[1][-1]:
        raise ValueError('K/B grid must bracket Kc and B=0')
    params = np.column_stack((np.linspace(axes[0][0], axes[0][-1], 301), np.zeros(301)))
    curves, centers = [], []
    for k in range(values.shape[2]):
        f = RegularGridInterpolator(tuple(axes), values[:, :, k], bounds_error=True)
        curves.append(f(params))
        centers.append(f([[kc, 0.]])[0])
    return kc, params, np.asarray(curves), np.asarray(centers)


def write_field_report(run, train, test, train_data, test_data, out, roots, sampling, fingerprint):
    direct = np.concatenate((train['direct'], test['direct']))
    labels = np.concatenate((train_data.couplings.numpy(), test_data.couplings.numpy()))
    axes, values = grid_encodings(labels, direct)
    kc, params, curves, centers = field_reference(axes, values)
    steps = direct.shape[1]
    np.savez_compressed(out / 'grid_encodings.npz', K=axes[0], B=axes[1], direct_mean=values,
                        zero_field_parameters=params, zero_field_curves=curves,
                        critical_encodings=centers)
    fig, axs = plt.subplots(2, steps, figsize=(5 * steps, 8), squeeze=False, layout='constrained')
    for j in range(2):
        for k in range(steps):
            ax = axs[j, k]
            z = ax.pcolormesh(*axes, values[:, :, k, j].T, shading='nearest', cmap='viridis')
            ax.axvline(kc, color='crimson', ls=':', label='Kc reference')
            ax.axhline(0., color='white', ls=':')
            ax.set(xlabel='Initial K', ylabel='Initial B', title=f'g{j+1} = Q(C^{k}D)')
            fig.colorbar(z, ax=ax)
    fig.savefig(out / 'coupling_by_scale.png', dpi=160)
    plt.close(fig)
    cells = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 2)
    means = values.reshape(-1, steps, 2)
    lower, upper = np.asarray(roots['search_box'])
    mesh = np.meshgrid(*[np.linspace(a, b, 31) for a, b in zip(lower, upper)])
    grid = np.stack(mesh, axis=-1).reshape(-1, 2)
    with torch.no_grad():
        delta = run.model.advance_theory(torch.from_numpy(grid)).numpy() - grid
    speed = np.linalg.norm(delta, axis=1)
    direction = delta / np.maximum(speed[:, None], 1e-12)

    def draw(ax, k):
        ax.quiver(*grid.T, *direction.T, color='.35', alpha=.55)
        z = ax.scatter(*means[:, k].T, c=cells[:, 1], cmap='coolwarm', s=10, alpha=.7)
        ax.plot(*curves[k].T, color='black', lw=1.6, label=f'Bare B=0, encoded at k={k}')
        ax.scatter(*centers[k], marker='D', color='magenta', s=60,
                   label='Q(Kc,B=0); NOT an R root')
        for i, root in enumerate(roots['roots']):
            ax.scatter(*root['g'], marker='*', color='gold', edgecolors='black', s=120)
            ax.annotate(f'F{i+1}: {root["stability"]}', root['g'], xytext=(4, 4),
                        textcoords='offset points', fontsize=8)
        ax.set(xlabel='g1', ylabel='g2', xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]))
        ax.grid(alpha=.15)
        ax.legend(fontsize=7)
        return z

    fig, ax = plt.subplots(figsize=(9, 7), layout='constrained')
    z = draw(ax, 0)
    ax.set_title('R(g)-g: normalized arrows; off-data flow is extrapolation\n'
                 'Bare zero-field curve and critical input are evaluation references')
    fig.colorbar(z, ax=ax, label='Initial B (not a coarse coupling)')
    fig.savefig(out / 'R_with_KB_reference.png', dpi=160)
    plt.close(fig)
    fig, axs = plt.subplots(1, steps, figsize=(6 * steps, 5.5), squeeze=False, layout='constrained')
    for k, ax in enumerate(axs[0]):
        draw(ax, k)
        ax.set_title(f'Bare B=0 curve encoded at k={k}')
    fig.savefig(out / 'zero_field_by_scale.png', dpi=160)
    plt.close(fig)
    root_projections = []
    for root in roots['roots']:
        distances = np.linalg.norm(means[:, 0] - root['g'], axis=1)
        nearest = np.argsort(distances)[:5]
        root_projections.append(dict(g=root['g'], nearest_microscopic_cells=[
            dict(K=float(cells[i, 0]), B=float(cells[i, 1]), distance=float(distances[i])) for i in nearest],
            caution='Nearest grid cells with nonzero residual, not a unique inverse coupling'))
    covariance = [np.linalg.eigvalsh(np.cov(direct[:, k].T, bias=True)) for k in range(steps)]
    report = dict(checkpoint=str(run.checkpoint), checkpoint_epoch=run.payload.get('epoch'),
                  data_sha256=fingerprint, coupling_names=['K', 'B'], grid_axes=[a.tolist() for a in axes],
                  critical_K=kc, critical_encodings_by_scale=centers.tolist(),
                  heldout_closure=closure_metrics(test, reference_direct=train['direct']),
                  covariance_eigenvalues_by_scale=[v.tolist() for v in covariance],
                  participation_rank_by_scale=[float(v.sum() ** 2 / np.sum(v ** 2)) if np.sum(v ** 2) else None
                                               for v in covariance],
                  sampling_seed=12345, lattice_sizes=[sampling['sites'] // run.model_config.block_factor ** k
                                                     for k in range(steps)],
                  fixed_points=roots, root_projections=root_projections,
                  caveat='Single stochastic-C draw. No labels in training; finite-size data. Root search is '
                         'bounded, not exhaustive. Bare reference curves may drift and are not fitted '
                         'separatrices. Off-data flow and roots are extrapolations. Nearest cells do not '
                         'establish a unique physical inverse.')
    (out / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--checkpoint', type=int)
    p.add_argument('--output', type=Path)
    p.add_argument('--threads', type=int, default=1)
    args = p.parse_args()
    if args.threads < 1:
        p.error('Positive thread count required')
    torch.set_num_threads(args.threads)
    run = load_run(args.run, args.checkpoint)
    cfg = RunConfig.from_dict(run.config)
    cfg.validate()
    if cfg.model.dimension != 2 or cfg.model.theory_dim != 2 or cfg.model.categorical_states:
        raise ValueError('Requires single-layer binary 2D Ising with dim g=2')
    if tuple(cfg.data.coupling_names) != ('K', 'B'):
        raise ValueError('Requires explicitly documented K/B labels for evaluation')
    sampling = json.loads((Path(cfg.data.cache).parent / 'sampling_audit.json').read_text())
    if sampling['grid'] != 'positive-uniform-B':
        raise ValueError('Requires the documented ferromagnetic K/B Cartesian cache')
    manifest = json.loads((args.run / 'manifest.json').read_text())
    fingerprint = file_fingerprint(cfg.data.cache)
    if fingerprint != manifest['data_sha256']:
        raise ValueError('Training data fingerprint changed')
    train_data, test_data = prepared_data(cfg)
    opts = dict(samples=cfg.evaluation.samples, max_rg_steps=cfg.evaluation.max_rg_steps,
                spin_flip_augment=False, sampling_seed=12345)
    train = collect_trajectories(run.model, train_data.configurations, **opts)
    test = collect_trajectories(run.model, test_data.configurations, **opts)
    out = args.output or args.run / ('full_report' if args.checkpoint is None else f'full_report_{args.checkpoint}')
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'encodings.npz', train=train['direct'], test=test['direct'],
                        train_rollout=train['rollout'], test_rollout=test['rollout'],
                        train_labels=train_data.couplings.numpy(), test_labels=test_data.couplings.numpy())
    np.savez_compressed(out / 'trajectories.npz', **test, couplings=test_data.couplings.numpy(),
                        training_inputs=train['direct'][:, :-1])
    plot_diagnostics(run.model, test, test_data.couplings.numpy(), ('K', 'B'), out,
                     training_inputs=train['direct'][:, :-1])
    roots = json.loads((out / 'fixed_points.json').read_text())
    print(json.dumps(write_field_report(run, train, test, train_data, test_data, out, roots, sampling, fingerprint),
                     indent=2))


if __name__ == '__main__':
    main()
