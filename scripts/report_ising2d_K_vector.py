"""Read-only vector-QCR diagnostics for single-layer, zero-field Ising K scans.

All roots and all Jacobian eigenvalues are retained. K labels annotate the
saved model only; no labels, MI calculation, or new configurations train it.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from recurrent_rg.checkpoints import load_run
from recurrent_rg.config import RunConfig
from recurrent_rg.evaluation import collect_trajectories, closure_metrics
from recurrent_rg.plotting import plot_diagnostics
from recurrent_rg.runner import file_fingerprint, prepared_data

KC = .4406867935


def full_r_jacobian(model, point, *, h=1e-5, atol=1e-7, rtol=1e-5):
    """Verify the full discrete map, including its identity skip, centrally."""
    point = np.asarray(point, dtype=np.float64)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError('A finite two-coordinate point is required')
    x = torch.tensor(point, dtype=torch.float64, requires_grad=True)
    jacobian = torch.autograd.functional.jacobian(model.advance_theory, x).detach().numpy()
    with torch.no_grad():
        residual = np.linalg.norm(model.advance_theory(x).numpy() - point)
        columns = []
        for j in range(2):
            offset = np.zeros(2)
            offset[j] = h
            plus = model.advance_theory(torch.from_numpy(point + offset)).numpy()
            minus = model.advance_theory(torch.from_numpy(point - offset)).numpy()
            columns.append((plus - minus) / (2 * h))
    finite = np.stack(columns, axis=1)
    if not np.isfinite(jacobian).all() or not np.isfinite(finite).all() or not np.isfinite(residual):
        raise ValueError('Nonfinite Jacobian or fixed-point residual')
    return dict(jacobian=jacobian.tolist(), finite_difference_jacobian=finite.tolist(),
                finite_difference_h=h, derivative_max_abs_error=float(np.max(abs(jacobian - finite))),
                derivative_verified=bool(np.allclose(jacobian, finite, atol=atol, rtol=rtol)),
                root_verified=bool(residual <= 1e-7), residual_norm=float(residual))


def eigenvalue_records(jacobian, block_factor, *, verified=True, tolerance=1e-6):
    """No eigenvalue is selected as thermal by rank or proximity to theory."""
    if block_factor <= 1:
        raise ValueError('Spatial block factor must exceed one')
    values, vectors = np.linalg.eig(np.asarray(jacobian, dtype=np.float64))
    result = []
    for j, value in enumerate(values):
        magnitude = float(abs(value))
        y = math.log(magnitude) / math.log(block_factor) if magnitude else None
        real = abs(value.imag) <= tolerance
        relevant_positive = bool(real and value.real > 1 + tolerance)
        result.append(dict(real=float(value.real), imag=float(value.imag), magnitude=magnitude,
            eigenvector_real=vectors[:, j].real.tolist(), eigenvector_imag=vectors[:, j].imag.tolist(),
            magnitude_scaling_exponent=y,
            conditional_nu=float(math.log(block_factor) / math.log(value.real))
                if verified and relevant_positive else None,
            positive_real_relevant=relevant_positive, nu_certified=False,
            physical_sector='unidentified',
            interpretation='Conditional inverse-y for a validated positive real relevant direction; '
                'thermal identity and transverse data support remain unverified.'))
    return result


def curve_projections(knots, curve, point, *, exact_tolerance=1e-10):
    """Retain all exact inverse branches and residual-bearing nearest segments."""
    knots, curve, point = map(lambda value: np.asarray(value, dtype=np.float64),
                               (knots, curve, point))
    if len(knots) < 2 or curve.shape != (len(knots), 2) or point.shape != (2,):
        raise ValueError('Expected K knots, a two-coordinate curve, and one point')
    rows, exact, intervals = [], [], []
    for i in range(len(knots) - 1):
        start, delta = curve[i], curve[i + 1] - curve[i]
        norm2 = float(delta @ delta)
        fraction = float(np.clip((point - start) @ delta / norm2, 0, 1)) if norm2 else 0.
        projected = start + fraction * delta
        distance = float(np.linalg.norm(projected - point))
        k = float(knots[i] + fraction * (knots[i + 1] - knots[i]))
        tangent = delta / np.linalg.norm(delta) if norm2 else np.zeros(2)
        rows.append(dict(segment=i, K=k, distance=distance, projected_g=projected.tolist(),
                         tangent=tangent.tolist(), K_interval=[float(knots[i]), float(knots[i + 1])]))
        if distance <= exact_tolerance:
            if norm2:
                if not any(abs(k - previous) <= exact_tolerance for previous in exact):
                    exact.append(k)
            else:
                intervals.append([float(knots[i]), float(knots[i + 1])])
    rows.sort(key=lambda row: (row['distance'], row['segment']))
    nearest = rows[0]['distance']
    tied = [row for row in rows if abs(row['distance'] - nearest) <= exact_tolerance]
    return dict(exact_inverse_K_branches=sorted(exact), exact_inverse_K_intervals=intervals,
                nearest_distance=nearest, nearest_segments=tied,
                closest_segments=rows[:5],
                inverse_caution='K is a nearest piecewise-linear curve projection with a reported '
                    'residual, not a unique physical inverse. All exact branches/flat intervals are retained.')


def training_support(point, train_direct):
    """Distance and hull metadata do not certify support of Jacobian directions."""
    from scipy.spatial import Delaunay, QhullError, cKDTree
    inputs = np.asarray(train_direct, dtype=np.float64)[:, :-1].reshape(-1, 2)
    point = np.asarray(point, dtype=np.float64)
    sd = inputs.std(axis=0)
    scale = np.where(sd > 1e-12, sd, 1.)
    hull_member = None
    if np.linalg.matrix_rank(inputs - inputs.mean(0)) == 2:
        try:
            hull_member = bool(Delaunay(inputs).find_simplex(point) >= 0)
        except QhullError:
            pass
    return dict(nearest_training_input_distance=float(cKDTree(inputs).query(point)[0]),
                nearest_training_input_distance_in_coordinate_sd=float(cKDTree(inputs / scale).query(point / scale)[0]),
                training_input_coordinate_sd=sd.tolist(),
                zero_variance_coordinates=(sd <= 1e-12).tolist(),
                inside_training_coordinate_box=bool(np.all(point >= inputs.min(0)) and np.all(point <= inputs.max(0))),
                inside_training_convex_hull=hull_member, support_certified=False,
                support_caution='Box/hull membership does not certify local support. K-only data '
                    'form scale-dependent curves; transverse Jacobian directions may be extrapolation.')


def write_vector_report(loaded, train, test, train_labels, test_labels, fixed_points, out,
                        *, sampling_repeats=None, data_sha256=None, lattice_sizes=None):
    direct = np.concatenate((train['direct'], test['direct']))
    labels = np.concatenate((np.asarray(train_labels).reshape(-1), np.asarray(test_labels).reshape(-1)))
    if (direct.ndim != 3 or direct.shape[-1] != 2 or len(labels) != len(direct) or
            not np.isfinite(direct).all() or not np.isfinite(labels).all()):
        raise ValueError('Expected finite [ensembles, scales, 2] encodings and one K per ensemble')
    knots = np.unique(labels)
    if len(knots) < 2:
        raise ValueError('At least two K knots are required')
    means = np.stack([direct[labels == k].mean(0) for k in knots])
    block = loaded.model_config.block_factor
    reference = None
    if knots[0] <= KC <= knots[-1]:
        reference = np.stack([[np.interp(KC, knots, means[:, k, j]) for j in range(2)]
                              for k in range(means.shape[1])])
    root_records = []
    for index, saved in enumerate(fixed_points['roots']):
        point = np.asarray(saved['g'])
        derivative = full_r_jacobian(loaded.model, point)
        eigenvalues = eigenvalue_records(derivative['jacobian'], block,
            verified=derivative['derivative_verified'] and derivative['root_verified'])
        projections = [curve_projections(knots, means[:, k], point) for k in range(means.shape[1])]
        for eigenvalue in eigenvalues:
            vector = np.asarray(eigenvalue['eigenvector_real'])
            eigenvalue['nearest_curve_tangent_alignment_by_scale'] = [
                [float(abs(vector @ np.asarray(segment['tangent'])) / np.linalg.norm(vector))
                 if np.linalg.norm(vector) else None for segment in projection['nearest_segments']]
                if abs(eigenvalue['imag']) <= 1e-6 else None for projection in projections]
        magnitude = np.array([item['magnitude'] for item in eigenvalues])
        stability = ('nonhyperbolic' if np.any(abs(magnitude - 1) <= 1e-6) else
                     'stable' if np.all(magnitude < 1) else
                     'unstable' if np.all(magnitude > 1) else 'saddle')
        root_records.append(dict(saved, root_id=f'F{index + 1}', **derivative,
            stability=stability, eigenvalues=eigenvalues,
            curve_projections_by_scale=projections, **training_support(point, train['direct']),
            selected_thermal_eigenvalue=None, physical_root_certified=False))
    covariance = [np.linalg.eigvalsh(np.cov(train['direct'][:, k].T, bias=True)).clip(min=0)
                  for k in range(direct.shape[1])]
    reference_map = None
    if reference is not None:
        with torch.no_grad():
            reference_map = loaded.model.advance_theory(torch.from_numpy(reference)).numpy()
    report = dict(checkpoint=str(loaded.checkpoint), checkpoint_epoch=loaded.payload.get('epoch'),
        data_sha256=data_sha256, physical_model='ising2d', coupling_names=['K'], B=0.,
        block_factor=block, lattice_sizes=lattice_sizes, sampling_seed=12345,
        thermal_reference=dict(Kc=KC, thermal_lambda=block, nu=1.,
                               use='Evaluation reference only; never root/eigenvalue selection'),
        heldout_closure=closure_metrics(test, reference_direct=train['direct']),
        covariance_eigenvalues_by_scale=[values.tolist() for values in covariance],
        participation_rank_by_scale=[float(values.sum() ** 2 / (values @ values)) if values @ values > 0 else None
                                     for values in covariance],
        covariance_reference='TRAIN direct encodings only',
        critical_K_reference=KC,
        critical_encodings_by_scale=None if reference is None else reference.tolist(),
        critical_R_drift_by_scale=None if reference is None else (reference_map - reference).tolist(),
        critical_direct_scale_drift=None if reference is None else np.diff(reference, axis=0).tolist(),
        fixed_points=dict(fixed_points, roots=root_records), roots=root_records,
        sampling_repeats=sampling_repeats,
        magnetic_exponents=dict(y_h=None, beta=None, reason='B=0 K-only ensembles do not identify a magnetic perturbation.'),
        selection=dict(root_selection='Every bounded-search root and every eigenvalue retained',
                       selected_thermal_eigenvalue=None, theory_used_for_selection=False, test_used_for_selection=False),
        caveat='K labels annotate only. Positive relevant eigenvalues yield conditional arithmetic '
            'inverse-y values; neither thermal identity nor transverse support is certified. '
            'Microscopic Kc encodings may drift and are not assumed to be R roots. '
            'Single diagnostic C draw and closure repeats do not provide eigenvalue confidence intervals.')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (out / 'critical_exponents.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (out / 'fixed_points.json').write_text(json.dumps(report['fixed_points'], indent=2, allow_nan=False) + '\n')
    np.savez_compressed(out / 'K_encodings.npz', K=knots, mean_by_scale=means)
    draw_summary(loaded.model, knots, means, reference, direct, labels, report['fixed_points'], out)
    return report


def draw_summary(model, knots, means, reference, direct, labels, fixed_points, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lower, upper = np.asarray(fixed_points['search_box'])
    mesh = np.meshgrid(*[np.linspace(a, b, 27) for a, b in zip(lower, upper)])
    grid = np.stack(mesh, axis=-1).reshape(-1, 2)
    with torch.no_grad():
        delta = model.advance_theory(torch.from_numpy(grid)).numpy() - grid
    norm = np.linalg.norm(delta, axis=1)
    direction = delta / np.maximum(norm[:, None], 1e-12)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout='constrained')
    axes[0].quiver(*grid.T, *direction.T, color='.65', alpha=.5)
    points = axes[0].scatter(*direct[:, 0].T, c=labels, cmap='viridis', s=10, alpha=.5)
    for k in range(means.shape[1]):
        axes[0].plot(*means[:, k].T, label=f'Q(C^{k}D)')
        if reference is not None:
            axes[0].scatter(*reference[k], marker='D', color='magenta', s=35)
        for j in range(2):
            axes[j + 1].plot(knots, means[:, k, j], label=f'k={k}')
    for index, root in enumerate(fixed_points['roots']):
        axes[0].scatter(*root['g'], marker='*', color='gold', edgecolors='black', s=100)
        suffix = '' if root['root_verified'] else '; unverified'
        axes[0].annotate(f'{root["root_id"]}: {root["stability"]}{suffix}', root['g'],
                         xytext=(6, 9 + 13 * (index % 3)), textcoords='offset points', fontsize=7)
    axes[0].set(xlabel='g1', ylabel='g2', xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]),
                title='R(g)-g; off-curve flow is extrapolation')
    fig.colorbar(points, ax=axes[0], label='Initial K')
    for j in range(2):
        axes[j + 1].axvline(KC, ls=':', color='magenta', label='Kc reference')
        axes[j + 1].set(xlabel='Initial K', ylabel=f'g{j + 1}', title='K -> Q(C^k D)')
    for ax in axes:
        ax.legend(fontsize=7)
        ax.grid(alpha=.15)
    fig.suptitle('Single-layer B=0 Ising; diamonds encode Kc and are not fitted fixed points')
    fig.savefig(out / 'vector_critical_summary.png', dpi=160)
    fig.savefig(out / 'R_with_K_reference.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for j, ax in enumerate(axes):
        for k in range(means.shape[1]):
            ax.plot(knots, means[:, k, j], label=f'k={k}')
        ax.axvline(KC, color='magenta', ls=':')
        ax.set(xlabel='Initial K', ylabel=f'g{j + 1}')
        ax.legend()
        ax.grid(alpha=.2)
    fig.savefig(out / 'coupling_by_scale.png', dpi=160)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--checkpoint', type=int)
    p.add_argument('--output', type=Path)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--sampling-repeats', type=int, default=5)
    args = p.parse_args(argv)
    if args.threads < 1 or args.sampling_repeats < 0:
        p.error('Invalid diagnostic threads/repeats')
    torch.set_num_threads(args.threads)
    selector = args.checkpoint
    if selector is None and (args.run / 'checkpoint_last.pt').exists():
        selector = 'checkpoint_last.pt'
    loaded = load_run(args.run, selector)
    cfg = RunConfig.from_dict(loaded.config)
    cfg.validate()
    if cfg.model.dimension != 2 or cfg.model.theory_dim != 2 or cfg.model.categorical_states:
        raise ValueError('Requires single-layer binary 2D Ising with dim g=2')
    if cfg.data.physical_model != 'ising2d' or tuple(cfg.data.coupling_names) != ('K',):
        raise ValueError('Requires explicitly documented single-layer K-only Ising data')
    if args.checkpoint is not None and loaded.payload.get('epoch') != args.checkpoint:
        raise ValueError('Loaded checkpoint epoch differs from requested numbered checkpoint')
    sampling = json.loads(Path(cfg.data.cache).with_suffix('.json').read_text())
    if sampling.get('B') != 0. or sampling.get('coupling_names') != ['K']:
        raise ValueError('Sampling sidecar must identify B=0 K-only ensembles')
    manifest = json.loads((args.run / 'manifest.json').read_text())
    fingerprint = file_fingerprint(cfg.data.cache)
    if fingerprint != manifest.get('data_sha256') or fingerprint != sampling.get('data_sha256'):
        raise ValueError('Cache bytes differ from training manifest or sampling sidecar')
    train_data, test_data = prepared_data(cfg)
    options = dict(samples=cfg.evaluation.samples, max_rg_steps=cfg.evaluation.max_rg_steps,
                   spin_flip_augment=cfg.data.spin_flip_augment, sampling_seed=12345)
    train = collect_trajectories(loaded.model, train_data.configurations, **options)
    test = collect_trajectories(loaded.model, test_data.configurations, **options)
    out = args.output or args.run / ('full_report' if args.checkpoint is None else f'full_report_{args.checkpoint}')
    out.mkdir(parents=True, exist_ok=True)
    plot_diagnostics(loaded.model, test, test_data.couplings.numpy(), ('K',), out,
                     training_inputs=train['direct'][:, :-1])
    roots = json.loads((out / 'fixed_points.json').read_text())
    np.savez_compressed(out / 'encodings.npz', train=train['direct'], test=test['direct'],
                        train_rollout=train['rollout'], test_rollout=test['rollout'],
                        train_labels=train_data.couplings.numpy(), test_labels=test_data.couplings.numpy())
    np.savez_compressed(out / 'trajectories.npz', **test, couplings=test_data.couplings.numpy(),
                        training_inputs=train['direct'][:, :-1])
    repeats = []
    for seed in range(args.sampling_repeats):
        draw_options = dict(options, sampling_seed=seed)
        draw = collect_trajectories(loaded.model, test_data.configurations, **draw_options)
        repeats.append(closure_metrics(draw, reference_direct=train['direct']))
    lattice_sizes = [train_data.configurations.shape[2] // cfg.model.block_factor ** k
                     for k in range(train['direct'].shape[1])]
    report = write_vector_report(loaded, train, test, train_data.couplings.numpy(),
        test_data.couplings.numpy(), roots, out, data_sha256=fingerprint, lattice_sizes=lattice_sizes,
        sampling_repeats=dict(seeds=list(range(args.sampling_repeats)), closure=repeats,
            note='Random-C closure variation of one model; not eigenvalue confidence intervals'))
    print(json.dumps(dict(checkpoint=report['checkpoint'], roots=report['roots']), indent=2, allow_nan=False))
    return report


if __name__ == '__main__':
    main()
