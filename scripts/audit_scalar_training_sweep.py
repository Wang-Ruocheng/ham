"""Compare every root in final scalar Ising reports without model evaluation.

Example: python -m scripts.audit_scalar_training_sweep --run RUN_A --run RUN_B
         --output SWEEP_AUDIT

Inputs are completed runs and their existing full_report directories. Actual
wall-limited endpoints remain usable, but cannot establish convergence. This
script never trains, resamples C, computes MI, chooses a root, ranks strategies,
or selects checkpoints. Theory is an evaluation reference only.
"""
import argparse
import json
import math
from pathlib import Path
import textwrap
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from recurrent_rg.joint_plateau import joint_plateau_reached
from recurrent_rg.learning_rate import LearningRatePolicy
from recurrent_rg.runner import file_fingerprint
from scripts.report_scalar_exponent_trend import finite_number, merged_roots, normalized_root


KC = .4406867935
NU_RELATIVE_TOLERANCE = .05
PLATEAU_SETTINGS = dict(minimum=4000, window=2000, confirmations=3,
                        check_every=200, rtol=.01, rebound_rtol=.05, loss_floor=1e-3)


def read_object(path):
    result = json.loads(Path(path).read_text())
    if not isinstance(result, dict):
        raise ValueError(f'Expected a JSON object: {path}')
    return result


def integer_epoch(value, description):
    number = finite_number(value)
    if number is None or number < 1 or not number.is_integer():
        raise ValueError(f'Missing or invalid {description}')
    return int(number)


def training_components(history, cfg, manifest):
    """Reconstruct the independently routed losses from saved TRAIN metrics."""
    weights = cfg['loss']['weights']
    rows = []
    for row in history:
        try:
            rows.append(dict(
                epoch=row['epoch'],
                component_Q=weights['anchor'] * row['train_anchor'] +
                    weights['coarse_consistency'] * row['train_coarse_consistency'] +
                    weights['separation'] * row['train_separation'],
                component_C=manifest.get('rsmi_weight', 2.) * row['train_rsmi'] +
                    weights['kernel_noncollapse'] * row['train_kernel_noncollapse'] +
                    cfg['loss'].get('kernel_sign_weight', 0.) * row.get('train_kernel_sign', 0.),
                component_R=weights['commutator'] * row['train_commutator'] +
                    cfg['loss']['rollout_weight'] * row['train_rollout']))
        except (KeyError, TypeError, ValueError):
            return [], 'Saved TRAIN metrics cannot reconstruct every Q/C/R component.'
    if not rows or any(finite_number(row[key]) is None for row in rows
                       for key in ('epoch', 'component_Q', 'component_C', 'component_R')):
        return [], 'Missing or nonfinite saved TRAIN component history.'
    return rows, None


def root_diagnostic(saved, index, block):
    root = normalized_root(saved, index, block)
    # Preserve the entire original record, including all FD and search metadata.
    root['saved_record'] = dict(saved)
    for name in ('finite_difference_multiplier', 'finite_difference_h'):
        root[name] = finite_number(saved.get(name))
    root['inverse_branches_from_saved_encodings'] = list(
        saved.get('inverse_branches_from_saved_encodings', []))
    multiplier, nu = root['multiplier'], root['conditional_nu']
    if nu is not None:
        expected = math.log(block) / math.log(multiplier)
        if not math.isclose(nu, expected, rel_tol=1e-8, abs_tol=1e-10):
            root['warnings'].append('Saved nu disagrees with ln(b)/ln(lambda); conditional nu omitted.')
            nu = root['conditional_nu'] = None
    finite_difference, h = root['finite_difference_multiplier'], root['finite_difference_h']
    if (finite_difference is None or h is None or h <= 0 or multiplier is None or
            not math.isclose(multiplier, finite_difference, rel_tol=1e-5, abs_tol=1e-7) or
            root['residual_norm'] is None or root['residual_norm'] > 1e-7):
        root['warnings'].append('Missing or inconsistent saved root/finite-difference evidence; conditional nu omitted.')
        nu = root['conditional_nu'] = None
    root.update(
        conditional_nu_relative_error=None if nu is None else abs(nu - 1.),
        conditional_nu_within_prespecified_5pct=None if nu is None else abs(nu - 1.) <= NU_RELATIVE_TOLERANCE + 1e-12,
        lambda_relative_error=None if multiplier is None else abs(multiplier - block) / block,
        nu_certified=False, thermal_coordinate_certified=False,
        diagnostic_only=True)
    return root


def learning_rate_evidence(history, manifest, summary, payload, actual, requested):
    """Check the saved policy against actual LR logs; never infer missing logs."""
    sources = {}
    summary_state, checkpoint_state = summary.get('lr_scheduler_state'), payload.get('lr_scheduler_state')
    for name, value in (('manifest', manifest.get('learning_rate_policy')),
                        ('summary', summary_state.get('policy') if isinstance(summary_state, dict) else None),
                        ('checkpoint', checkpoint_state.get('policy') if isinstance(checkpoint_state, dict) else None)):
        if isinstance(value, dict):
            sources[name] = value
    logs = [dict(update=row['epoch'], learning_rate=finite_number(row.get('learning_rate')))
            for row in history if finite_number(row.get('learning_rate')) is not None]
    result = dict(policy_evidence=sources, logged_learning_rates=logs,
                  evidence_status='missing_policy_or_history',
                  schedule_active=None, first_scheduled_decay_update=None,
                  first_executed_decay_update=None, first_observed_changed_lr_log_update=None,
                  capped_before_first_decay=None, rate_at_actual_equals_initial=None,
                  initial_learning_rate=None, last_executed_learning_rate=None, warnings=[])
    if not sources or not logs or len(logs) != len(history):
        result['warnings'].append('Missing policy or complete LR history; activation is unknown, not inferred from old configs.')
        return result
    raw = next(iter(sources.values()))
    if any(value != raw for value in sources.values()):
        result.update(evidence_status='inconsistent_policy_evidence')
        result['warnings'].append('Saved LR policy descriptions disagree; activation is unknown.')
        return result
    try:
        policy = LearningRatePolicy(SimpleNamespace(
            lr_schedule=raw['schedule'], learning_rate=raw['base_learning_rate'],
            lr_milestones=raw['milestones'], lr_gamma=raw['gamma'],
            lr_min_factor=raw['minimum_factor'], lr_cosine_updates=raw['cosine_updates']))
        matched = all(math.isclose(row['learning_rate'], policy.rate_for_update(row['update']),
                                   rel_tol=1e-12, abs_tol=1e-15) for row in logs)
    except (KeyError, TypeError, ValueError):
        result.update(evidence_status='unusable_policy_evidence')
        result['warnings'].append('Saved LR policy cannot be validated; activation is unknown.')
        return result
    if not matched:
        result.update(evidence_status='history_disagrees_with_policy')
        result['warnings'].append('Recorded LR history disagrees with the saved policy; activation is unknown.')
        return result
    initial = policy.rate_for_update(1)
    first_change = None
    if policy.kind == 'multistep':
        first_change = next((milestone + 1 for milestone in policy.milestones
                             if policy.rate_for_update(milestone + 1) != initial), None)
    elif policy.kind == 'cosine' and policy.rate_for_update(policy.cosine_updates) != initial:
        # Monotone cosine schedules permit a bounded search even for long horizons.
        low, high = 2, policy.cosine_updates
        while low < high:
            middle = (low + high) // 2
            if policy.rate_for_update(middle) == initial:
                low = middle + 1
            else:
                high = middle
        first_change = low
    active = first_change is not None and actual >= first_change
    capped = actual < requested and first_change is not None and actual < first_change
    observed = [row['update'] for row in logs if row['learning_rate'] != initial]
    result.update(evidence_status='verified_against_recorded_history', schedule=policy.kind,
                  initial_learning_rate=initial, last_executed_learning_rate=policy.rate_for_update(actual),
                  first_scheduled_decay_update=first_change,
                  first_executed_decay_update=first_change if active else None,
                  first_observed_changed_lr_log_update=min(observed) if observed else None,
                  schedule_active=active, capped_before_first_decay=capped,
                  rate_at_actual_equals_initial=policy.rate_for_update(actual) == initial)
    if capped:
        result['warnings'].append(f'Endpoint is capped before scheduled LR change update {first_change}; no decay was executed.')
    return result


def timing_evidence(manifest, summary):
    """Retain recorded timing/cache evidence; no setup durations are estimated."""
    setup = {}
    for name, source in (('manifest', manifest), ('summary', summary)):
        number = finite_number(source.get('setup_seconds'))
        if number is not None:
            setup[f'{name}.setup_seconds'] = number
        timing = source.get('timing')
        if isinstance(timing, dict):
            number = finite_number(timing.get('setup_seconds'))
            if number is not None:
                setup[f'{name}.timing.setup_seconds'] = number
    execution = manifest.get('execution_optimizations')
    return dict(manifest_timing=manifest.get('timing'), summary_timing=summary.get('timing'),
                setup_seconds_by_source=setup,
                elapsed_seconds=summary.get('elapsed_seconds'),
                setup_and_training_seconds=summary.get('setup_and_training_seconds'),
                mi_statistics_cache=execution.get('mi_statistics_cache') if isinstance(execution, dict) else None)


def fairness_evidence(records):
    actual_equal = len({row['actual_updates'] for row in records}) == 1
    requested_equal = len({row['requested_updates'] for row in records}) == 1
    setup_values, cache_warm_states, warnings = [], [], []
    for row in records:
        evidence = row['timing_evidence']
        source = evidence['setup_seconds_by_source']
        setup_values.append(source.get('summary.timing.setup_seconds', source.get('summary.setup_seconds')))
        cache = evidence['mi_statistics_cache']
        state = ({split: item['hit'] for split, item in cache.items()}
                 if isinstance(cache, dict) and cache and
                 all(isinstance(item, dict) and isinstance(item.get('hit'), bool) for item in cache.values()) else None)
        cache_warm_states.append(state)
    setup_equal = (all(math.isclose(value, setup_values[0], rel_tol=1e-9, abs_tol=1e-9) for value in setup_values)
                   if all(value is not None for value in setup_values) else None)
    cache_equal = (all(state == cache_warm_states[0] for state in cache_warm_states)
                   if all(state is not None for state in cache_warm_states) else None)
    if not actual_equal:
        warnings.append('Actual update counts differ; capped endpoints are not an equal-update strategy comparison.')
    if cache_equal is False:
        warnings.append('Recorded MI-cache hit states differ: some runs built statistics and others reused them. Wall caps include setup, so equal caps do not establish equal optimization exposure or comparable steady-step speed.')
    elif setup_equal is False:
        warnings.append('Recorded setup durations differ. Wall caps include setup; compare actual update exposure separately from elapsed wall time.')
    if any(row['learning_rate_evidence']['capped_before_first_decay'] is True for row in records):
        warnings.append('At least one scheduled policy was capped before its first LR change; this arm did not exercise its decay strategy.')
    return dict(actual_updates_equal=actual_equal, requested_updates_equal=requested_equal,
                every_actual_equals_requested=all(row['actual_updates'] == row['requested_updates'] for row in records),
                measured_setup_seconds=setup_values, measured_setup_seconds_equal=setup_equal,
                recorded_MI_cache_hit_states=cache_warm_states, recorded_MI_cache_hit_states_equal=cache_equal,
                equal_wall_caps_certify_fair_comparison=False, warnings=warnings)


def collect_run(run, *, report_name='full_report'):
    run = Path(run).resolve()
    manifest = read_object(run / 'manifest.json')
    summary = read_object(run / 'summary.json')
    cfg = manifest['config']
    model, data = cfg['model'], cfg['data']
    if (model['dimension'] != 2 or model['theory_dim'] != 1 or
            model.get('categorical_states', 0) or tuple(data['coupling_names']) != ('K',)):
        raise ValueError(f'Requires a binary 2D Ising scalar K scan: {run}')
    requested = integer_epoch(cfg['training']['epochs'], 'requested epochs')
    actual = integer_epoch(summary.get('actual_epochs', requested), 'actual terminal epochs')
    if actual > requested:
        raise ValueError(f'Actual endpoint exceeds configured epochs: {run}')
    checkpoint = run / 'checkpoint_last.pt'
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if integer_epoch(payload.get('epoch'), 'final checkpoint epoch') != actual:
        raise ValueError(f'checkpoint_last.pt does not match the summary endpoint: {run}')
    if payload.get('metadata', {}).get('config', cfg) != cfg:
        raise ValueError(f'Final checkpoint configuration differs from manifest: {run}')
    report_dir = run / report_name
    critical = read_object(report_dir / 'critical_exponents.json')
    metrics = read_object(report_dir / 'metrics.json')
    if integer_epoch(critical.get('checkpoint_epoch'), 'reported actual checkpoint epoch') != actual:
        raise ValueError(f'Exponent report is not from the terminal checkpoint: {run}')
    if Path(critical.get('checkpoint', '')).name not in (
            'checkpoint_last.pt', f'training_step_{actual}.pt'):
        raise ValueError(f'Exponent report did not use the final checkpoint: {run}')
    block = finite_number(critical.get('block_factor'))
    if block is None or block <= 1 or block != model.get('block_factor'):
        raise ValueError(f'Reported block factor differs from the saved model: {run}')
    if any(finite_number(k) is None or k < 0 for k in critical.get('coupling_means', {}).get('K', [])):
        raise ValueError(f'Requires a positive-K evaluation scan: {run}')
    selection = critical.get('selection', {})
    if selection.get('test_used_for_selection') is not False or selection.get('theory_used_for_selection') is not False:
        raise ValueError(f'Exponent report does not declare selection-free evaluation: {run}')
    source_digests = critical.get('input_sha256', {})
    for name in ('metrics.json', 'fixed_points.json', 'encodings.npz'):
        if source_digests.get(name) != file_fingerprint(report_dir / name):
            raise ValueError(f'Exponent report source digest differs for {name}: {run}')
    history = payload.get('history', summary.get('history', []))
    if not isinstance(history, list) or not history or history[-1].get('epoch') != actual:
        raise ValueError(f'Final saved history does not end at the actual checkpoint: {run}')
    rows, component_warning = training_components(history, cfg, manifest)
    saved_plateau = summary.get('joint_plateau_diagnostics', {})
    settings = saved_plateau.get('settings', PLATEAU_SETTINGS)
    plateau, diagnostics = joint_plateau_reached(rows, **settings)
    wall_limited = bool(summary.get('stopped_by_wall_limit'))
    if wall_limited or actual < requested and not summary.get('stopped_by_train_plateau'):
        plateau = False
    roots = [root_diagnostic(root, index, block)
             for index, root in enumerate(merged_roots(critical, metrics))]
    files = dict(manifest=run / 'manifest.json', summary=run / 'summary.json',
                 checkpoint=checkpoint, exponents=report_dir / 'critical_exponents.json',
                 metrics=report_dir / 'metrics.json')
    return dict(
        run=str(run), run_name=run.name, report_directory=str(report_dir),
        checkpoint=str(checkpoint), checkpoint_sha256=file_fingerprint(checkpoint),
        requested_updates=requested, actual_updates=actual,
        endpoint='wall_limit' if wall_limited else
                 'train_plateau' if summary.get('stopped_by_train_plateau') else 'update_limit',
        not_converged=not plateau, train_plateau_established=plateau,
        train_plateau_diagnostic=diagnostics,
        training_components=rows, component_warning=component_warning,
        block_factor=block, seed=cfg['training']['seed'],
        learning_rate=cfg['training']['learning_rate'],
        learning_rate_policy=manifest.get('learning_rate_policy'),
        learning_rate_evidence=learning_rate_evidence(history, manifest, summary, payload, actual, requested),
        timing_evidence=timing_evidence(manifest, summary),
        config=cfg, data_sha256=manifest.get('data_sha256'),
        roots=roots, root_count=len(roots),
        closure=metrics.get('closure', {}),
        sampling_repeats=metrics.get('sampling_repeats'),
        critical_reference=critical.get('critical_reference'),
        training_input_range=critical.get('training_input_range'),
        original_flow_bounds=critical.get('original_flow_bounds'),
        input_sha256={name: file_fingerprint(path) for name, path in files.items()},
        source_report_sha256=source_digests,
        magnetic_exponents=dict(y_h=None, beta=None, reason='A scalar zero-field thermal scan does not measure a magnetic scaling field.'),
        warnings=[] if roots else ['No isolated roots were reported; this is retained as an outcome.'])


def collect_sweep(runs, *, report_name='full_report'):
    paths = [Path(run).resolve() for run in runs]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError('Supply at least one run, with no duplicate run paths')
    records = [collect_run(run, report_name=report_name) for run in paths]
    for index, row in enumerate(records):
        row['comparison_id'] = f'R{index + 1}'
    seeds = sorted({row['seed'] for row in records})
    fingerprints = {row['data_sha256'] for row in records}
    return dict(
        runs=records,
        protocol=dict(
            endpoint='The terminal checkpoint of each completed run, including explicitly labeled wall-limited endpoints.',
            root_handling='Keep every reported root and inverse-K branch; local F labels are not matched between runs.',
            convergence='TRAIN-only Q/C/R plateau; wall-limited/incomplete endpoints are not converged.',
            plateau_default_settings=PLATEAU_SETTINGS,
            conditional_nu_relative_tolerance=NU_RELATIVE_TOLERANCE,
            tolerance_use='Prespecified exploratory diagnostic only; not a root/checkpoint/strategy selection rule.',
            robust_success_requirements=['Independently justified regular thermal coordinate and faithful RG.',
                'Verified supported positive relevant root and full-R derivative.',
                'Converged objectives and consistent conditional nu across independent training seeds.',
                'Evaluation sample, stochastic-C draw, finite-size and root-search sensitivity checks.']),
        theory_references=dict(thermal_lambda='lambda=b', nu=1., Kc=KC, use='Evaluation reference only'),
        comparison=dict(same_data_fingerprint=len(fingerprints) == 1 and None not in fingerprints,
                        fairness=fairness_evidence(records),
                        same_model_configuration=all(row['config']['model'] == records[0]['config']['model'] for row in records),
                        same_loss_configuration=all(row['config']['loss'] == records[0]['config']['loss'] for row in records),
                        independent_training_seeds=seeds, has_independent_seeds=len(seeds) > 1,
                        robust_success_certified=False),
        selection=dict(test_used_for_selection=False, theory_used_for_selection=False,
                       checkpoint_selection='None; final actual endpoint only',
                       root_selection='None; all roots retained', strategy_ranking='None'),
        caveats=['All nu values are conditional and non-certified; proximity to theory does not establish a thermal direction.',
                 'TRAIN range membership alone does not establish local sampling density.',
                 'The saved bounded root search may miss tangential roots.',
                 'Sampling-repeat fluctuations are not independent training-seed uncertainty.',
                 'Different seeds under different strategies alone do not replicate a frozen training strategy.',
                 'No model evaluation, resampling, MI computation or parameter update occurs in this audit.'])


def plot_comparison(report, out):
    records = report['runs']
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), layout='constrained')
    names = [f'{row["comparison_id"]}: ' + '\n'.join(textwrap.wrap(row['run_name'], width=28)) +
             f'\n{row["actual_updates"]}/{row["requested_updates"]}' for row in records]
    for x, run in enumerate(records):
        color = f'C{x % 10}'
        for root in run['roots']:
            style = dict(edgecolors=color, facecolors=color if root['outside_training_support'] is False else 'none',
                         marker='v' if root['sign_reversing'] else 'o' if root['stability'] == 'stable' else '^', s=48)
            for ax, values in zip(axes[0], ([root['multiplier']], [root['conditional_nu']],
                                           root['microscopic_inverse_coupling'])):
                values = [value for value in values if value is not None]
                if values:
                    ax.scatter([x] * len(values), values, **style)
            for interval in root['inverse_plateau_intervals_from_saved_encodings']:
                if isinstance(interval, list) and len(interval) == 2 and all(finite_number(v) is not None for v in interval):
                    axes[0, 2].vlines(x, *interval, color=color, alpha=.5)
        axes[0, 0].scatter(x, run['block_factor'], marker='_', s=150, color='black')
        for name, style in (('one_step', '-o'), ('rollout', '--x')):
            values = run['closure'].get(name, {}).get('nrmse', [])
            if values:
                axes[1, 0].plot(range(len(values)), [math.nan if v is None else v for v in values], style,
                                color=color, label=f'{run["comparison_id"]}: {name}')
        history = run['training_components']
        for component, ax, style in (('component_Q', axes[1, 1], '-o'),
                                     ('component_C', axes[1, 2], '-o'), ('component_R', axes[1, 2], '--x')):
            ax.plot([row['epoch'] for row in history], [row[component] for row in history],
                    style, markersize=3, color=color, label=f'{run["comparison_id"]}: {component[-1]}')
    axes[0, 1].axhspan(.95, 1.05, color='gray', alpha=.12, label='Prespecified 5% diagnostic band')
    axes[0, 1].axhline(1., color='black', ls=':')
    axes[0, 2].axhline(KC, color='black', ls=':', label='Known input Kc reference')
    for ax, title, ylabel in zip(axes[0], ('All roots; black ticks lambda=b', 'Conditional nu; null stays absent',
                                        'Every inverse-K branch'), ('Full-R lambda', 'Conditional nu', 'Initial K')):
        ax.set(title=title, ylabel=ylabel, xticks=range(len(names)), xticklabels=names)
        ax.tick_params(axis='x', labelsize=7, rotation=15)
    for ax, title, xlabel, ylabel in zip(axes[1], ('TEST closure; TRAIN SD normalization', 'Saved TRAIN Q objective',
                                                'Saved TRAIN C and R objectives'), ('Scale index', 'Actual update', 'Actual update'),
                                        ('NRMSE', 'Weighted objective', 'Weighted objective')):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.set_yscale('symlog', linthresh=1e-3)
    for ax in axes.flat:
        ax.grid(alpha=.2)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=6)
    fig.suptitle('Final actual endpoints only; all roots retained; no ranking or selection\n'
                 'Hollow markers: outside/unknown TRAIN support. All scalar nu values are conditional, non-certified.')
    fig.savefig(Path(out) / 'comparison.png', dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report-name', default='full_report', help='Existing final-report subdirectory within each run')
    args = parser.parse_args(argv)
    if Path(args.report_name).is_absolute() or '..' in Path(args.report_name).parts:
        parser.error('--report-name must be a subdirectory within each run')
    report = collect_sweep(args.run, report_name=args.report_name)
    args.output.mkdir(parents=True, exist_ok=True)
    plot_comparison(report, args.output)
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(output=str(args.output), runs=len(report['runs']),
                          actual_updates=[row['actual_updates'] for row in report['runs']],
                          root_counts=[row['root_count'] for row in report['runs']]), indent=2))
    return report


if __name__ == '__main__':
    main()
