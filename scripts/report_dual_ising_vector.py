"""Vector dual-Ising ratio/grid diagnostics; critical labels annotate only.

Reads a saved checkpoint and cached data, without evaluating MI or updating
the model. Also supports isolated, single-thread live-training previews.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from recurrent_rg.checkpoints import load_run
from recurrent_rg.config import RunConfig
from recurrent_rg.evaluation import collect_trajectories, closure_metrics
from recurrent_rg.plotting import plot_diagnostics
from recurrent_rg.runner import file_fingerprint, prepared_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--checkpoint', type=int)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('Positive diagnostic thread count required')
    torch.set_num_threads(args.threads)
    selector = args.checkpoint
    if selector is None and (args.run/'checkpoint_last.pt').exists():
        selector = 'checkpoint_last.pt'
    run = load_run(args.run, selector)
    cfg = RunConfig.from_dict(run.config)
    cfg.validate()
    if cfg.model.theory_dim != 2:
        raise ValueError('Requires a two-dimensional theory representation')
    sampling = json.loads((Path(cfg.data.cache).parent / 'sampling.json').read_text())
    if sampling.get('model') != 'two uncoupled square Ising layers' or sampling.get('grid_mode') not in ('ratio_line', 'independent'):
        raise ValueError('Requires explicitly documented uncoupled dual-Ising data')
    scan = sampling.get('scan_axis', 'K1')
    critical = np.asarray(sampling['critical_K'], dtype=float)
    manifest = json.loads((args.run / 'manifest.json').read_text())
    fingerprint = file_fingerprint(cfg.data.cache)
    if fingerprint != manifest['data_sha256']:
        raise ValueError('Training cache fingerprint changed')
    train_data, test_data = prepared_data(cfg)
    opts = dict(samples=cfg.evaluation.samples, max_rg_steps=cfg.evaluation.max_rg_steps,
                spin_flip_augment=cfg.data.spin_flip_augment, sampling_seed=12345)
    train = collect_trajectories(run.model, train_data.configurations, **opts)
    test = collect_trajectories(run.model, test_data.configurations, **opts)
    out = args.output or args.run / ('full_report' if args.checkpoint is None else f'full_report_{args.checkpoint}')
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'encodings.npz', train=train['direct'], test=test['direct'],
                        train_rollout=train['rollout'], test_rollout=test['rollout'],
                        train_labels=train_data.couplings.numpy(), test_labels=test_data.couplings.numpy())
    np.savez_compressed(out / 'trajectories.npz', **test, couplings=test_data.couplings.numpy(),
                        training_inputs=train['direct'][:, :-1])
    names = ('K1', 'K2') if sampling['grid_mode'] == 'independent' else (scan,)
    plot_diagnostics(run.model, test, test_data.couplings.numpy(), names, out,
                     training_inputs=train['direct'][:, :-1])
    roots = json.loads((out / 'fixed_points.json').read_text())
    if sampling['grid_mode'] == 'independent':
        from scripts.dual_ising_grid_diagnostics import write_grid_report
        report = write_grid_report(run, train, test, train_data, test_data, out, roots, sampling, fingerprint)
        print(json.dumps(report, indent=2))
        return
    direct = np.concatenate((train['direct'], test['direct']))
    labels = np.concatenate((train_data.couplings.numpy(), test_data.couplings.numpy())).reshape(-1)
    knots = np.unique(labels)
    means = np.stack([direct[labels == value].mean(0) for value in knots])
    markers = np.stack([np.column_stack([np.interp(critical, knots, means[:, k, j])
                                        for j in range(2)]) for k in range(means.shape[1])])
    colors = ['crimson', 'deepskyblue']
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.7), layout='constrained')
    for j in range(2):
        for k in range(means.shape[1]):
            axes[j].plot(knots, means[:, k, j], '.-', ms=3, label=f'Q(C^{k} D)')
        for value, color in zip(critical, colors):
            axes[j].axvline(value, color=color, ls=':', label=f'{scan}={value:.4f}')
        axes[j].set(xlabel=f'Initial {scan}', ylabel=f'g{j+1}')
    for k in range(means.shape[1]):
        axes[2].plot(*means[:, k].T, '.-', ms=3, label=f'k={k}')
        for i, point in enumerate(markers[k]):
            axes[2].scatter(*point, marker='x', color=colors[i], s=65)
    axes[2].set(xlabel='g1', ylabel='g2', title='Critical encodings by scale; not fixed-point labels')
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.savefig(out / 'coupling_by_scale.png', dpi=160)
    plt.close(fig)
    lower, upper = np.asarray(roots['search_box'])
    mesh = np.meshgrid(*[np.linspace(a, b, 31) for a, b in zip(lower, upper)])
    grid = np.stack(mesh, axis=-1).reshape(-1, 2)
    with torch.no_grad():
        delta = run.model.advance_theory(torch.from_numpy(grid)).numpy() - grid
    speed = np.linalg.norm(delta, axis=1)
    direction = delta / np.maximum(speed[:, None], 1e-12)
    fig, ax = plt.subplots(figsize=(9, 7), layout='constrained')
    ax.quiver(*grid.T, *direction.T, color='.65', alpha=.6)
    scatter = ax.scatter(*direct[:, 0].T, c=labels, cmap='viridis', s=14, alpha=.7)
    for k in range(means.shape[1]):
        ax.plot(*means[:, k].T, lw=1, alpha=.6, label=f'Q by initial {scan}: k={k}')
    for i, point in enumerate(markers[0]):
        ax.scatter(*point, marker='D', s=80, color=colors[i], edgecolors='black',
                   label=f'Microscopic Q({scan}={critical[i]:.4f})')
    for i, point in enumerate(roots['roots']):
        ax.scatter(*point['g'], marker='*', s=140, color='gold', edgecolors='black')
        ax.annotate(f'F{i+1}: {point["stability"]}', point['g'], xytext=(5, 5),
                    textcoords='offset points', fontsize=8)
    ax.set(xlabel='g1', ylabel='g2', xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]),
           title='R(g)-g: normalized arrows; off-data flow is extrapolation\n'
                 'Theoretical critical markers are not fitted roots or basin boundaries')
    ax.legend(fontsize=7)
    fig.colorbar(scatter, ax=ax, label=f'Initial {scan}')
    fig.savefig(out / 'R_with_two_transitions.png', dpi=160)
    plt.close(fig)
    report = dict(checkpoint=str(run.checkpoint), checkpoint_epoch=run.payload.get('epoch', args.checkpoint),
                  data_sha256=fingerprint, scan_axis=scan, critical_couplings=critical.tolist(),
                  critical_encodings_by_scale=markers.tolist(),
                  heldout_closure=closure_metrics(test, reference_direct=train['direct']),
                  covariance_eigenvalues_by_scale=[np.linalg.eigvalsh(np.cov(direct[:, k].T, bias=True)).tolist()
                                                  for k in range(direct.shape[1])],
                  sampling_seed=12345, fixed_points=roots,
                  caveat='One stochastic C draw. Bounded root search is not exhaustive. '
                         'Sparse or off-data directions are extrapolation; initial critical encodings can drift.')
    (out / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    (out / 'critical_markers.json').write_text(json.dumps(dict(critical_K=critical.tolist(),
        encodings_by_scale=markers.tolist(), checkpoint=args.checkpoint, scan_axis=scan), indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
