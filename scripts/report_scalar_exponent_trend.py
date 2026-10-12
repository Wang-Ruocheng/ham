"""Summarize saved scalar exponent previews without models or resampling.

All roots are independent per-preview scatter points. Local F labels are
never treated as branch identities across training checkpoints.
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


KC = .4406867935


def finite_number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def scalar_point(root):
    value = root.get('g')
    if isinstance(value, list):
        return finite_number(value[0]) if len(value) == 1 else None
    return finite_number(value)


def merged_roots(critical, metrics):
    """Match metadata within one preview only, retaining its full root union."""
    roots = [dict(root) for root in critical.get('roots', [])]
    for saved in metrics.get('fixed_points', {}).get('roots', []):
        point = scalar_point(saved)
        match = next((root for root in roots if point is not None and
                      scalar_point(root) is not None and
                      abs(point - scalar_point(root)) <= 1e-9), None)
        if match is None:
            roots.append(dict(saved))
        else:
            combined = dict(saved)
            combined.update(match)
            match.clear()
            match.update(combined)
    return roots


def normalized_root(root, index, block):
    warnings = []
    multiplier = finite_number(root.get('multiplier'))
    if multiplier is None:
        warnings.append('Missing or nonfinite full-R multiplier; lambda point omitted.')
    nu = finite_number(root.get('nu'))
    outside = root.get('outside_training_support')
    if not isinstance(outside, bool):
        inside = root.get('inside_training_input_range')
        outside = not inside if isinstance(inside, bool) else None
    stability = root.get('stability', 'unknown')
    if nu is not None and (multiplier is None or multiplier <= 1 or
                           stability != 'unstable' or outside is not False or
                           root.get('root_verified') is not True or
                           root.get('derivative_verified') is not True):
        warnings.append('Saved nu is not eligible as a supported positive relevant conditional nu; omitted.')
        nu = None
    branches = []
    raw_branches = root.get('microscopic_inverse_coupling')
    if not isinstance(raw_branches, list):
        warnings.append('Missing microscopic inverse-coupling branches; K points omitted.')
    else:
        for branch in raw_branches:
            value = finite_number(branch)
            if value is None:
                warnings.append('Nonfinite inverse-coupling branch omitted.')
            else:
                branches.append(value)
    negative = multiplier is not None and multiplier < 0
    category = ('outside TRAIN' if outside else 'inside TRAIN' if outside is False else 'unknown support')
    category += '; ' + ('negative lambda; ' if negative else '') + stability
    return dict(local_root_id=root.get('root_id', f'F{index + 1}'),
                g=scalar_point(root), multiplier=multiplier, conditional_nu=nu,
                saved_nu=finite_number(root.get('nu')),
                stability=stability, outside_training_support=outside,
                root_verified=root.get('root_verified'),
                derivative_verified=root.get('derivative_verified'),
                residual_norm=finite_number(root.get('residual_norm')),
                derivative_abs_error=finite_number(root.get('derivative_abs_error')),
                sign_reversing=negative, category=category,
                microscopic_inverse_coupling=branches,
                inverse_plateau_intervals_from_saved_encodings=root.get(
                    'inverse_plateau_intervals_from_saved_encodings', []),
                lambda_minus_theory=None if multiplier is None else multiplier - block,
                conditional_nu_minus_theory=None if nu is None else nu - 1.,
                inverse_K_minus_Kc=[value - KC for value in branches],
                warnings=warnings)


def collect_trend(run):
    run = Path(run)
    snapshots, skipped = [], []
    for preview in sorted((run / 'monitor_previews').glob('epoch_*')):
        if not preview.is_dir():
            continue
        try:
            status = json.loads((preview / 'preview.json').read_text())
        except (OSError, ValueError):
            skipped.append(dict(preview=str(preview), reason='Missing or invalid preview.json'))
            continue
        if not isinstance(status, dict) or status.get('phase') != 'completed':
            skipped.append(dict(preview=str(preview), reason='Preview is not completed'))
            continue
        try:
            critical = json.loads((preview / 'critical_exponents.json').read_text())
            metrics = json.loads((preview / 'metrics.json').read_text())
        except (OSError, ValueError):
            skipped.append(dict(preview=str(preview), reason='Missing or invalid critical_exponents.json or metrics.json'))
            continue
        if not isinstance(critical, dict) or not isinstance(metrics, dict):
            skipped.append(dict(preview=str(preview), reason='Invalid report object'))
            continue
        epoch = finite_number(critical.get('checkpoint_epoch'))
        block = finite_number(critical.get('block_factor'))
        if epoch is None or epoch < 0 or not epoch.is_integer():
            skipped.append(dict(preview=str(preview), reason='Missing or invalid actual checkpoint_epoch; directory name is not used'))
            continue
        if block is None or block <= 1:
            skipped.append(dict(preview=str(preview), reason='Missing or invalid reported block_factor'))
            continue
        try:
            roots = merged_roots(critical, metrics)
            records = [normalized_root(root, index, block) for index, root in enumerate(roots)]
        except (AttributeError, TypeError, ValueError):
            skipped.append(dict(preview=str(preview), reason='Missing or malformed scalar root data'))
            continue
        snapshots.append(dict(preview=str(preview), checkpoint_epoch=int(epoch),
                              block_factor=block, roots=records,
                              root_count=len(records),
                              warnings=[] if records else ['No roots were reported at this checkpoint.']))
    snapshots.sort(key=lambda row: (row['checkpoint_epoch'], row['preview']))
    blocks = sorted({row['block_factor'] for row in snapshots})
    return dict(
        run=str(run), snapshots=snapshots, skipped=skipped,
        theory_references=dict(thermal_lambda=[dict(block_factor=b, multiplier=b) for b in blocks],
                               nu=1., Kc=KC, use='Plot references only; no root, branch, or epoch selection'),
        selection=dict(test_used_for_selection=False, theory_used_for_selection=False,
                       branch_matching='None: every root and inverse-K branch is an independent scatter point; F labels are local to a preview'),
        caveats=[
            'Conditional nu requires an independently justified thermal scaling interpretation; proximity to theory does not certify one.',
            'Stable, negative-multiplier and unsupported roots retain lambda/K diagnostics but do not supply thermal nu.',
            'Null nu values remain absent. No roots are selected or joined across epochs.',
            'Only completed previews with an actual reported checkpoint_epoch are read; missing data are listed under skipped or root warnings.',
            'Saved previews may use stochastic encodings; fluctuations are not independent training-seed uncertainty.',
        ],
    )


def plot_trend(report, out):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout='constrained')
    labeled = [set(), set(), set()]
    missing_nu = 0
    for snapshot in report['snapshots']:
        epoch = snapshot['checkpoint_epoch']
        for root in snapshot['roots']:
            category = root['category']
            color = ('purple' if root['sign_reversing'] else
                     'forestgreen' if root['stability'] == 'stable' else
                     'darkorange' if root['stability'] == 'unstable' else 'steelblue')
            marker = 'v' if root['sign_reversing'] else 'o' if root['stability'] == 'stable' else '^'
            outside = root['outside_training_support']
            style = dict(marker=marker, edgecolors=color,
                         facecolors='none' if outside is not False else color, s=45)
            values = ([root['multiplier']] if root['multiplier'] is not None else [],
                      [root['conditional_nu']] if root['conditional_nu'] is not None else [],
                      root['microscopic_inverse_coupling'])
            missing_nu += root['conditional_nu'] is None
            for panel, ys in enumerate(values):
                if ys:
                    label = category if category not in labeled[panel] else None
                    axes[panel].scatter([epoch] * len(ys), ys, label=label, **style)
                    labeled[panel].add(category)
            for interval in root['inverse_plateau_intervals_from_saved_encodings']:
                if isinstance(interval, list) and len(interval) == 2:
                    ends = [finite_number(value) for value in interval]
                    if all(value is not None for value in ends):
                        axes[2].vlines(epoch, *ends, color=color, alpha=.5)
    for item in report['theory_references']['thermal_lambda']:
        axes[0].axhline(item['multiplier'], color='black', ls=':',
                       label=f'Theory thermal lambda=b={item["block_factor"]:g}')
    axes[0].axhline(0., color='gray', lw=.7)
    axes[1].axhline(1., color='black', ls=':', label='Theory nu=1')
    axes[2].axhline(KC, color='black', ls=':', label=f'Input Kc={KC:.10f}')
    axes[1].text(.02, .03, f'nu absent for {missing_nu} reported roots',
                 transform=axes[1].transAxes, fontsize=8)
    for ax, ylabel, title in zip(axes, ('Full-R lambda', 'Conditional thermal nu', 'Interpolated microscopic K'),
            ('All saved root multipliers', 'Eligible values only; null stays absent', 'All inverse-coupling branches')):
        ax.set(xlabel='Actual checkpoint update', ylabel=ylabel, title=title)
        ax.grid(alpha=.2)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7)
    if not report['snapshots']:
        axes[0].text(.5, .5, 'No completed usable exponent previews',
                     ha='center', transform=axes[0].transAxes)
    fig.suptitle('Independent per-checkpoint scatter points; no cross-epoch root matching\n'
                 f'{len(report["snapshots"])} completed previews; {len(report["skipped"])} skipped. Hollow markers: outside/unknown TRAIN support.')
    fig.savefig(Path(out) / 'trend.png', dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args(argv)
    report = collect_trend(args.run.resolve())
    out = args.run.resolve() / 'monitor_previews' / 'exponent_trend'
    out.mkdir(parents=True, exist_ok=True)
    plot_trend(report, out)
    (out / 'trend.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(output=str(out), previews=len(report['snapshots']),
                          skipped=len(report['skipped'])), indent=2))
    return report


if __name__ == '__main__':
    main()
