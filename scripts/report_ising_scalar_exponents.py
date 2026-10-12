"""Read-only scalar square-Ising exponent diagnostics from an existing report.

No configurations are loaded, no encodings are resampled, and no MI is
calculated. Every previously reported root is retained. The physical Kc
marker is a forward interpolation of saved encodings, not a selected R root.
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from recurrent_rg.checkpoints import load_run
from recurrent_rg.runner import file_fingerprint


KC = .4406867935


def exponent_from_multiplier(multiplier, block_factor, *, tolerance=1e-6):
    """Magnitude scaling is not proof that a learned coordinate is thermal."""
    if not math.isfinite(multiplier) or not math.isfinite(block_factor) or block_factor <= 1:
        raise ValueError('Finite multiplier and spatial block factor > 1 required')
    magnitude = abs(multiplier)
    y = math.log(magnitude) / math.log(block_factor) if magnitude else None
    stability = ('nonhyperbolic' if abs(magnitude - 1) <= tolerance else
                 'stable' if magnitude < 1 else 'unstable')
    nu = None
    if stability == 'stable':
        reason = 'Stable directions are irrelevant; their negative y is not a thermal nu.'
    elif stability == 'nonhyperbolic':
        reason = 'A marginal multiplier does not determine a finite thermal nu.'
    elif multiplier < 0:
        reason = ('A negative multiplier reverses perturbations. Magnitude scaling does not '
                  'certify a period-two orbit, a thermal direction, or a physical nu.')
    else:
        nu = 1. / y
        reason = ('Conditional nu=1/y only if this relevant direction is independently '
                  'established as the thermal critical scaling field.')
    return dict(multiplier=multiplier, y=y, magnitude_scaling_exponent=y,
                nu=nu, stability=stability,
                nu_certified=False, period_two_certified=False,
                sign_reversing=multiplier < 0, interpretation=reason)


def full_r_derivative(model, point, *, h=1e-5, atol=1e-7, rtol=1e-5):
    """Differentiate full R, including its identity skip, and verify centrally."""
    x = torch.tensor([[point]], dtype=torch.float64, requires_grad=True)
    with torch.enable_grad():
        mapped = model.advance_theory(x)
        multiplier = float(torch.autograd.grad(mapped.sum(), x)[0].item())
    with torch.no_grad():
        plus = model.advance_theory(torch.tensor([[point + h]], dtype=torch.float64)).item()
        minus = model.advance_theory(torch.tensor([[point - h]], dtype=torch.float64)).item()
    finite_difference = float((plus - minus) / (2 * h))
    residual = abs(float(mapped.detach().item()) - point)
    if not all(math.isfinite(v) for v in (multiplier, finite_difference, residual)):
        raise ValueError('Nonfinite full-R derivative or fixed-point residual')
    return dict(multiplier=multiplier, finite_difference_multiplier=finite_difference,
                finite_difference_h=h, derivative_abs_error=abs(multiplier - finite_difference),
                derivative_verified=bool(np.isclose(multiplier, finite_difference, atol=atol, rtol=rtol)),
                displacement_derivative=multiplier - 1., residual_norm=residual,
                root_verified=residual <= 1e-7)


def scalar_coupling_means(train, test, train_labels, test_labels):
    train, test = np.asarray(train, dtype=np.float64), np.asarray(test, dtype=np.float64)
    if (train.ndim != 3 or train.shape[-1] != 1 or train.shape[1] < 2 or
            test.ndim != 3 or train.shape[1:] != test.shape[1:] or
            not len(train) or not len(test)):
        raise ValueError('Saved scalar TRAIN/TEST encodings must have shape [N, scales, 1]')
    labels = []
    for direct, raw in ((train, train_labels), (test, test_labels)):
        raw = np.asarray(raw, dtype=np.float64)
        if raw.size != len(direct):
            raise ValueError('Expected exactly one K label per saved ensemble')
        labels.append(raw.reshape(-1))
    direct, couplings = np.concatenate((train, test)), np.concatenate(labels)
    if not np.isfinite(direct).all() or not np.isfinite(couplings).all():
        raise ValueError('Nonfinite saved encodings or labels')
    knots = np.unique(couplings)
    if len(knots) < 2:
        raise ValueError('At least two saved K knots are needed for interpolation')
    means = np.stack([direct[couplings == k, :, 0].mean(0) for k in knots])
    return knots, means


def inverse_coupling_branches(knots, microscopic_means, point):
    """Keep every segment crossing; a folded Q map need not have one inverse."""
    branches = []
    for ka, kb, ga, gb in zip(knots[:-1], knots[1:], microscopic_means[:-1], microscopic_means[1:]):
        if ga != gb and min(ga, gb) <= point <= max(ga, gb):
            branches.append(float(ka + (kb - ka) * (point - ga) / (gb - ga)))
    return branches


def inverse_coupling_plateau_intervals(knots, microscopic_means, point):
    """Report continuum inverses separately instead of choosing a K on a plateau."""
    return [[float(ka), float(kb)]
            for ka, kb, ga, gb in zip(knots[:-1], knots[1:], microscopic_means[:-1], microscopic_means[1:])
            if ga == gb == point]


def critical_encoding_reference(model, knots, means, *, kc=KC):
    reference = dict(Kc=kc, interpolation='Piecewise linear in initial K; mean of saved TRAIN/TEST ensembles',
                     R_root_equivalence_assumed=False, g_by_scale=None,
                     caution='Q(C^k D(Kc)) is an encoded microscopic input, not an R fixed point or a coarse K.')
    if not knots[0] <= kc <= knots[-1]:
        reference['reason'] = 'Saved K knots do not bracket Kc; no extrapolated critical encoding is reported.'
        return reference
    g = np.array([np.interp(kc, knots, means[:, k]) for k in range(means.shape[1])])
    with torch.no_grad():
        mapped = model.advance_theory(torch.from_numpy(g[:, None])).numpy()[:, 0]
    reference.update(g_by_scale=g.tolist(), R_at_g_by_scale=mapped.tolist(),
                     direct_scale_drift=np.diff(g).tolist(),
                     R_drift_by_scale=(mapped - g).tolist(),
                     numerical_fixed_point_by_scale=(np.abs(mapped - g) <= 1e-7).tolist(),
                     one_step_closure_residual=(mapped[:-1] - g[1:]).tolist())
    return reference


def all_saved_roots(fixed_points, metrics):
    """Merge metadata for matching roots, preserving the complete saved union."""
    roots = [dict(root) for root in fixed_points.get('roots', [])]
    for root in metrics.get('fixed_points', {}).get('roots', []):
        point = float(np.asarray(root['g']).reshape(-1)[0])
        matching = next((item for item in roots if np.isclose(
            float(np.asarray(item['g']).reshape(-1)[0]), point, atol=1e-9, rtol=0)), None)
        if matching is None:
            roots.append(dict(root))
        else:
            matching.update(root)
    return roots


def draw_summary(model, roots, knots, means, reference, training_support, bounds, out):
    grid = np.linspace(*bounds, 2001)
    with torch.no_grad():
        mapped = model.advance_theory(torch.from_numpy(grid[:, None])).numpy()[:, 0]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout='constrained')
    axes[0].plot(grid, mapped, label='Full R(g)')
    axes[0].plot(grid, grid, 'k--', label='Identity')
    axes[1].plot(grid, mapped - grid, label='R(g)-g')
    axes[1].axhline(0., color='black', ls='--', lw=.8)
    low, high = training_support
    for ax in axes[:2]:
        ax.axvspan(low, high, color='green', alpha=.08, label='TRAIN R-input range')
        if bounds[0] < low:
            ax.axvspan(bounds[0], min(low, bounds[1]), color='gray', alpha=.12, label='Outside TRAIN range')
        if bounds[1] > high:
            ax.axvspan(max(high, bounds[0]), bounds[1], color='gray', alpha=.12)
        ax.set_xlim(bounds)
    for k in range(means.shape[1]):
        axes[2].plot(knots, means[:, k], label=f'Q(C^{k}D)')
    axes[2].axvline(reference['Kc'], color='magenta', ls=':', label='Known input Kc (reference)')
    if reference['g_by_scale'] is not None:
        for k, (g, rg) in enumerate(zip(reference['g_by_scale'], reference['R_at_g_by_scale'])):
            label = 'Encoded Kc input (not an R root)' if k == 0 else None
            axes[0].scatter(g, rg, marker='D', color='magenta', s=35, zorder=6, label=label)
            axes[1].scatter(g, rg - g, marker='D', color='magenta', s=35, zorder=6, label=label)
            axes[2].scatter(reference['Kc'], g, marker='D', color='magenta', s=35, zorder=6, label=label)
    for i, root in enumerate(roots):
        g = root['g'][0]
        color = 'forestgreen' if root['stability'] == 'stable' else 'darkorange'
        label = 'Saved R roots' if i == 0 else None
        axes[0].scatter(g, g, marker='*', c=color, edgecolors='black', s=100, zorder=7, label=label)
        axes[1].scatter(g, 0., marker='*', c=color, edgecolors='black', s=100, zorder=7, label=label)
        suffix = '; extrapolation' if root['outside_training_support'] else ''
        axes[1].annotate(f'F{i + 1}: {root["stability"]}{suffix}', (g, 0.),
                         xytext=(4, 10 + 12 * (i % 2)), textcoords='offset points', fontsize=7)
        axes[2].axhline(g, color=color, ls='--', alpha=.3, lw=.8)
        branches = root['microscopic_inverse_coupling']
        if branches:
            axes[2].scatter(branches, [g] * len(branches), marker='*', c=color,
                            edgecolors='black', s=70, zorder=7)
    for ax, title, xlabel, ylabel in zip(axes,
            ('Full discrete map', 'Discrete RG displacement', 'Saved coupling-to-g maps'),
            ('g', 'g', 'Initial K=J/T, B=0'), ('R(g)', 'R(g)-g', 'g')):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.legend(fontsize=7)
        ax.grid(alpha=.2)
    fig.suptitle('All reported roots; no root or checkpoint selection. Scalar B=0 R does not determine y_h or beta.')
    fig.savefig(out / 'scalar_critical_summary.png', dpi=170)
    plt.close(fig)


def write_exponent_report(loaded, metrics, fixed_points, encodings, out):
    model = loaded.model.double().eval()
    spec = loaded.model_config
    if spec.dimension != 2 or spec.theory_dim != 1 or spec.categorical_states:
        raise ValueError('Requires binary 2D Ising with scalar g')
    if tuple(loaded.config['data']['coupling_names']) != ('K',):
        raise ValueError('Requires one explicitly documented K label for the B=0 scan')
    block = spec.block_factor
    train, test = encodings['train'], encodings['test']
    knots, means = scalar_coupling_means(train, test, encodings['train_labels'], encodings['test_labels'])
    inputs = np.asarray(train, dtype=np.float64)[:, :-1, 0]
    support = [float(inputs.min()), float(inputs.max())]
    bounds_raw = fixed_points.get('search_box', fixed_points.get('search_range'))
    if bounds_raw is None:
        bounds_raw = metrics.get('fixed_points', {}).get('search_box')
    if bounds_raw is None:
        raise ValueError('Existing fixed-point report must contain its original flow bounds')
    bounds = np.asarray(bounds_raw, dtype=np.float64).reshape(-1)
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
        raise ValueError('Invalid saved scalar flow bounds')

    roots = []
    reference = critical_encoding_reference(model, knots, means)
    for index, saved in enumerate(all_saved_roots(fixed_points, metrics)):
        point = float(np.asarray(saved['g']).reshape(-1)[0])
        if not math.isfinite(point):
            raise ValueError('Nonfinite saved root')
        derivative = full_r_derivative(model, point)
        exponent = exponent_from_multiplier(derivative['multiplier'], block)
        computed_branches = inverse_coupling_branches(knots, means[:, 0], point)
        branches = saved.get('microscopic_inverse_coupling', computed_branches)
        record = dict(saved)
        record.update(exponent)
        record.update(derivative)
        record.update(root_id=f'F{index + 1}', g=[point],
                      saved_stability=saved.get('stability'),
                      microscopic_inverse_coupling=list(branches),
                      inverse_branches_from_saved_encodings=computed_branches,
                      inverse_plateau_intervals_from_saved_encodings=inverse_coupling_plateau_intervals(
                          knots, means[:, 0], point),
                      inside_training_input_range=bool(support[0] <= point <= support[1]),
                      outside_training_support=not support[0] <= point <= support[1],
                      support_caution='Range membership does not certify dense sampling or thermal interpretation.',
                      distance_to_encoded_Kc_by_scale=(None if reference['g_by_scale'] is None else
                          [abs(point - g) for g in reference['g_by_scale']]))
        if not derivative['root_verified'] or not derivative['derivative_verified']:
            record['nu'] = None
            record['interpretation'] = 'Root or derivative validation failed; no exponent inference is reported.'
        if record['outside_training_support']:
            record['formal_nu_from_extrapolated_positive_multiplier'] = record['nu']
            record['nu'] = None
            record['interpretation'] += (' This root and its derivative lie outside the TRAIN input range; '
                                         'any inverse-y value is an algebraic extrapolation, not a reported thermal nu.')
        roots.append(record)

    report = dict(
        checkpoint=str(loaded.checkpoint), checkpoint_epoch=loaded.payload.get('epoch'),
        dtype='float64', block_factor=block, roots=roots,
        formulas=dict(multiplier='lambda = d(full R)/dg, including the identity skip',
                      y='y = ln(abs(lambda))/ln(b)',
                      nu='nu = 1/y only for a verified positive relevant thermal direction'),
        finite_difference_h=1e-5, training_input_range=support,
        training_input_range_source='Existing encodings.npz TRAIN direct[:, :-1, 0]',
        original_flow_bounds=bounds.tolist(),
        original_search_note=fixed_points.get('search_note', metrics.get('fixed_points', {}).get('search_note')),
        critical_reference=reference,
        coupling_means=dict(K=knots.tolist(), g_by_scale=means.tolist(),
                            source='Saved TRAIN and TEST encodings; evaluation only'),
        selection=dict(test_used_for_selection=False, theory_used_for_selection=False,
                       root_selection='All saved roots retained, including stable and extrapolated roots',
                       epoch_selection='One final saved model; no checkpoint comparison or selection'),
        magnetic_exponents=dict(y_h=None, beta=None,
            reason='This B=0 scalar R multiplier and saved encodings do not measure a magnetic scaling field; y_h or beta cannot be inferred here.'),
        caveats=[
            'An R fixed point and the encoded microscopic Kc cannot be assumed equivalent; even numerical coincidence does not certify thermal interpretation.',
            'Supported positive unstable multipliers provide conditional nu values only if g is a regular thermal coordinate and R is a faithful physical RG; neither assumption is certified here.',
            'Negative multipliers do not establish a periodic orbit or a thermal critical exponent.',
            'Saved root searches are bounded and may miss tangential roots; this report performs no new root search.',
            'No resampling, MI calculation, parameter updates, TEST-based selection, or theoretical root selection.',
        ],
    )
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    draw_summary(model, roots, knots, means, reference, support, bounds, out)
    (out / 'critical_exponents.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--checkpoint', type=int,
                        help='Numbered immutable checkpoint used for the existing preview report')
    parser.add_argument('--output', type=Path,
                        help='Existing report directory; defaults to run/full_report')
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    out = args.output if args.output is not None else args.run / 'full_report'
    metrics = json.loads((out / 'metrics.json').read_text())
    fixed_points = json.loads((out / 'fixed_points.json').read_text())
    with np.load(out / 'encodings.npz', allow_pickle=False) as arrays:
        encodings = {name: arrays[name].copy() for name in ('train', 'test', 'train_labels', 'test_labels')}
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = 'checkpoint_last.pt' if (args.run / 'checkpoint_last.pt').is_file() else None
    loaded = load_run(args.run, checkpoint)
    if args.checkpoint is not None and loaded.payload.get('epoch') != args.checkpoint:
        raise ValueError('Selected checkpoint epoch does not match its saved payload')
    report = write_exponent_report(loaded, metrics, fixed_points, encodings, out)
    report['input_sha256'] = {}
    for name in ('metrics.json', 'fixed_points.json', 'encodings.npz'):
        report['input_sha256'][name] = file_fingerprint(out / name)
    (out / 'critical_exponents.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(checkpoint=report['checkpoint'], roots=report['roots'],
                          critical_reference=report['critical_reference']), indent=2))
    return report


if __name__ == '__main__':
    main()
