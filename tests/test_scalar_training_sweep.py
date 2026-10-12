import copy
import json
import math

import pytest
import torch

from recurrent_rg.config import TrainingConfig
from recurrent_rg.learning_rate import LearningRatePolicy
from recurrent_rg.runner import file_fingerprint
from scripts import audit_scalar_training_sweep as auditor


def save_object(path, obj):
    path.write_text(json.dumps(obj))


def make_run(tmp_path, name='run', *, actual=600, requested=600, seed=91, wall_limit=False):
    run = tmp_path / name
    report_dir = run / 'full_report'
    report_dir.mkdir(parents=True)
    cfg = dict(schema_version=5,
               model=dict(dimension=2, theory_dim=1, block_factor=3, categorical_states=0),
               data=dict(coupling_names=['K']),
               loss=dict(weights=dict(anchor=2., coarse_consistency=5., separation=10.,
                                      kernel_noncollapse=2., commutator=5.),
                         rollout_weight=5., kernel_sign_weight=1.),
               training=dict(epochs=requested, learning_rate=.001, seed=seed))
    history = [dict(epoch=actual, train_anchor=.1, train_coarse_consistency=.2,
                    train_separation=.3, train_kernel_noncollapse=.5, train_kernel_sign=.6,
                    train_rsmi=-.4, train_commutator=.7, train_rollout=.8, test_total=-1e9)]
    manifest = dict(config=cfg, data_sha256='same_data', rsmi_weight=2.)
    summary = dict(config=cfg, actual_epochs=actual, requested_epochs=requested,
                   stopped_by_wall_limit=wall_limit, history=history)
    save_object(run / 'manifest.json', manifest)
    save_object(run / 'summary.json', summary)
    torch.save(dict(epoch=actual, metadata=dict(config=cfg), history=history,
                    cuda_rng_state=torch.empty(0, dtype=torch.uint8), cuda_device_index=0),
               run / 'checkpoint_last.pt')
    roots = [dict(root_id='F1', g=[-1.], multiplier=-3., nu=None, stability='unstable',
                  microscopic_inverse_coupling=[.2, .35, .55]),
             dict(root_id='F2', g=[0.], multiplier=.5, nu=None, stability='stable',
                  microscopic_inverse_coupling=[.4]),
             dict(root_id='F3', g=[1.], multiplier=3., nu=1., stability='unstable',
                  microscopic_inverse_coupling=[.3, .5],
                  inverse_plateau_intervals_from_saved_encodings=[[.35, .4]])]
    for root in roots:
        root.update(root_verified=True, derivative_verified=True, residual_norm=0.,
                    derivative_abs_error=0., finite_difference_multiplier=root['multiplier'],
                    finite_difference_h=1e-5, outside_training_support=False)
    metrics = dict(closure=dict(one_step=dict(rmse=[.1, .2], nrmse=[.2, None]),
                               rollout=dict(rmse=[0., .1, .2], nrmse=[0., .2, .3])),
                   fixed_points=dict(roots=copy.deepcopy(roots)))
    critical = dict(checkpoint=str(run / 'checkpoint_last.pt'), checkpoint_epoch=actual,
                    block_factor=3, roots=roots, training_input_range=[-1.2, 1.2],
                    original_flow_bounds=[-1.5, 1.5], coupling_means=dict(K=[.2, .6]),
                    selection=dict(test_used_for_selection=False, theory_used_for_selection=False))
    save_object(report_dir / 'metrics.json', metrics)
    save_object(report_dir / 'fixed_points.json', dict(roots=roots))
    (report_dir / 'encodings.npz').write_bytes(b'saved encodings; no need to load them')
    critical['input_sha256'] = {name: file_fingerprint(report_dir / name)
                                for name in ('metrics.json', 'fixed_points.json', 'encodings.npz')}
    save_object(report_dir / 'critical_exponents.json', critical)
    return run, critical, metrics


def test_final_report_keeps_every_root_and_branch_and_no_magnetic_exponent(tmp_path, monkeypatch):
    run, critical, metrics = make_run(tmp_path)
    monkeypatch.setattr(torch.cuda, 'set_rng_state', lambda *a, **k: pytest.fail('Audit touched CUDA RNG'))
    report = auditor.collect_sweep([run])
    result = report['runs'][0]
    assert [root['g'] for root in result['roots']] == [-1., 0., 1.]
    assert result['roots'][0]['microscopic_inverse_coupling'] == [.2, .35, .55]
    assert result['roots'][2]['inverse_plateau_intervals_from_saved_encodings'] == [[.35, .4]]
    assert [root['conditional_nu'] for root in result['roots']] == [None, None, 1.]
    assert [root['conditional_nu_within_prespecified_5pct'] for root in result['roots']] == [None, None, True]
    assert all(not root['nu_certified'] for root in result['roots'])
    assert result['roots'][2]['finite_difference_multiplier'] == 3.
    assert result['roots'][2]['finite_difference_h'] == 1e-5
    assert result['closure'] == metrics['closure']
    assert result['not_converged']
    assert result['actual_updates'] == result['requested_updates'] == 600
    assert result['magnetic_exponents']['beta'] is None
    assert report['selection']['strategy_ranking'] == 'None'
    assert not report['comparison']['robust_success_certified']
    assert not report['comparison']['has_independent_seeds']


def test_wall_limit_is_retained_with_actual_endpoint_and_not_converged(tmp_path):
    run, _, _ = make_run(tmp_path, actual=200, requested=600, wall_limit=True)
    row = auditor.collect_run(run)
    assert row['actual_updates'] == 200 and row['requested_updates'] == 600
    assert row['endpoint'] == 'wall_limit' and row['not_converged']
    assert row['root_count'] == 3


def test_test_closure_and_theory_accuracy_never_choose_runs_or_roots(tmp_path):
    first, _, _ = make_run(tmp_path, 'first')
    second, critical, metrics = make_run(tmp_path, 'second', seed=92)
    critical['roots'][2].update(multiplier=9., finite_difference_multiplier=9., nu=.5)
    metrics['closure']['one_step']['nrmse'] = [1e9, 1e9]
    out = second / 'full_report'
    save_object(out / 'metrics.json', metrics)
    critical['input_sha256']['metrics.json'] = file_fingerprint(out / 'metrics.json')
    save_object(out / 'critical_exponents.json', critical)
    report = auditor.collect_sweep([second, first])
    assert [row['run_name'] for row in report['runs']] == ['second', 'first']
    assert [row['root_count'] for row in report['runs']] == [3, 3]
    assert report['runs'][0]['roots'][2]['conditional_nu'] == .5
    assert report['runs'][0]['roots'][2]['conditional_nu_within_prespecified_5pct'] is False
    assert report['comparison']['has_independent_seeds']
    assert not report['selection']['test_used_for_selection']
    assert not report['selection']['theory_used_for_selection']


@pytest.mark.parametrize('nu', [.95, 1.05])
def test_prespecified_accuracy_band_includes_boundaries(nu):
    multiplier = math.exp(math.log(3.) / nu)
    root = dict(g=[0.], multiplier=multiplier, finite_difference_multiplier=multiplier,
                finite_difference_h=1e-5, nu=nu, stability='unstable',
                root_verified=True, derivative_verified=True, residual_norm=0.,
                outside_training_support=False, microscopic_inverse_coupling=[])
    assert auditor.root_diagnostic(root, 0, 3)['conditional_nu_within_prespecified_5pct']


@pytest.mark.parametrize('change', [dict(nu=2.), dict(finite_difference_multiplier=12.),
                                    dict(finite_difference_h=0.), dict(residual_norm=.01),
                                    dict(outside_training_support=True)])
def test_invalid_or_extrapolated_root_keeps_record_without_nu(change):
    root = dict(g=[0.], multiplier=3., finite_difference_multiplier=3.,
                finite_difference_h=1e-5, nu=1., stability='unstable',
                root_verified=True, derivative_verified=True, residual_norm=0.,
                outside_training_support=False, microscopic_inverse_coupling=[.3, .5])
    root.update(change)
    result = auditor.root_diagnostic(root, 0, 3)
    assert result['conditional_nu'] is None
    assert result['conditional_nu_within_prespecified_5pct'] is None
    assert result['microscopic_inverse_coupling'] == [.3, .5]
    assert result['saved_record'] == root


def test_union_keeps_metrics_roots_missing_from_exponent_file(tmp_path):
    run, critical, _ = make_run(tmp_path)
    critical['roots'] = critical['roots'][:2]
    save_object(run / 'full_report' / 'critical_exponents.json', critical)
    result = auditor.collect_run(run)
    assert result['root_count'] == 3


def test_changed_report_sources_and_nonterminal_report_fail(tmp_path):
    run, critical, _ = make_run(tmp_path)
    critical['checkpoint_epoch'] = 200
    out = run / 'full_report'
    save_object(out / 'critical_exponents.json', critical)
    with pytest.raises(ValueError, match='terminal checkpoint'):
        auditor.collect_run(run)
    critical['checkpoint_epoch'] = 600
    save_object(out / 'critical_exponents.json', critical)
    (out / 'encodings.npz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='source digest'):
        auditor.collect_run(run)


def test_train_components_use_logged_train_losses_and_no_test_values():
    weights = dict(anchor=2., coarse_consistency=5., separation=10., kernel_noncollapse=2., commutator=5.)
    cfg = dict(loss=dict(weights=weights, rollout_weight=5., kernel_sign_weight=1.))
    row = dict(epoch=200, train_anchor=.1, train_coarse_consistency=.2,
               train_separation=.3, train_rsmi=-.4, train_kernel_noncollapse=.5,
               train_kernel_sign=.6, train_commutator=.7, train_rollout=.8, test_total=-1e99)
    rows, warning = auditor.training_components([row], cfg, dict(rsmi_weight=2.))
    assert warning is None
    assert rows[0] == pytest.approx(dict(epoch=200, component_Q=4.2, component_C=.8, component_R=7.5))
    row['test_total'] = 1e99
    assert auditor.training_components([row], cfg, dict(rsmi_weight=2.))[0] == rows


def test_cli_writes_comparison_and_summary_without_mutating_sources(tmp_path):
    first, _, _ = make_run(tmp_path, 'first')
    second, _, _ = make_run(tmp_path, 'second')
    before = {path: path.read_bytes() for run in (first, second) for path in run.rglob('*') if path.is_file()}
    out = tmp_path / 'audit'
    report = auditor.main(['--run', str(first), '--run', str(second), '--output', str(out)])
    saved = json.loads((out / 'summary.json').read_text())
    assert saved == report
    assert (out / 'comparison.png').stat().st_size > 0
    assert all(path.read_bytes() == value for path, value in before.items())


def test_duplicate_runs_are_not_counted_as_independent_evidence(tmp_path):
    run, _, _ = make_run(tmp_path)
    with pytest.raises(ValueError, match='duplicate'):
        auditor.collect_sweep([run, run])


def test_cap_before_first_milestone_does_not_claim_an_active_lr_strategy():
    policy = LearningRatePolicy(TrainingConfig(lr_schedule='multistep',
                                               lr_milestones=(6000, 9000), lr_gamma=.3))
    history = [dict(epoch=1, learning_rate=.001), dict(epoch=400, learning_rate=.001)]
    result = auditor.learning_rate_evidence(history, dict(learning_rate_policy=policy.description()),
                                           {}, {}, 400, 12000)
    assert result['evidence_status'] == 'verified_against_recorded_history'
    assert result['first_scheduled_decay_update'] == 6001
    assert result['first_executed_decay_update'] is None
    assert not result['schedule_active']
    assert result['capped_before_first_decay']
    assert result['rate_at_actual_equals_initial']


def test_decay_update_is_distinct_from_first_logged_decay_observation():
    policy = LearningRatePolicy(TrainingConfig(lr_schedule='multistep',
                                               lr_milestones=(6000, 9000), lr_gamma=.3))
    history = [dict(epoch=1, learning_rate=.001), dict(epoch=6200, learning_rate=.0003)]
    result = auditor.learning_rate_evidence(history, dict(learning_rate_policy=policy.description()),
                                           {}, {}, 6200, 12000)
    assert result['first_executed_decay_update'] == 6001
    assert result['first_observed_changed_lr_log_update'] == 6200
    assert result['schedule_active']
    assert not result['rate_at_actual_equals_initial']


@pytest.mark.parametrize('history', [[], [dict(epoch=400)],
                                    [dict(epoch=400, learning_rate=.0007)]])
def test_missing_or_inconsistent_lr_history_cannot_establish_activation(history):
    policy = LearningRatePolicy(TrainingConfig(lr_schedule='multistep', lr_milestones=(200,)))
    result = auditor.learning_rate_evidence(history, dict(learning_rate_policy=policy.description()),
                                           {}, {}, 400, 12000)
    assert result['schedule_active'] is None
    assert result['first_executed_decay_update'] is None
    assert result['warnings']


def test_cosine_first_change_and_constant_policy_are_reported_from_actual_evidence():
    for training, expected in ((TrainingConfig(lr_schedule='cosine', lr_cosine_updates=12000), 2),
                               (TrainingConfig(), None)):
        policy = LearningRatePolicy(training)
        history = [dict(epoch=1, learning_rate=policy.rate_for_update(1)),
                   dict(epoch=400, learning_rate=policy.rate_for_update(400))]
        result = auditor.learning_rate_evidence(history, dict(learning_rate_policy=policy.description()),
                                               {}, {}, 400, 12000)
        assert result['first_scheduled_decay_update'] == expected
        assert result['first_executed_decay_update'] == expected


def test_records_measured_timing_cache_warmth_and_unequal_update_fairness(tmp_path):
    first, _, _ = make_run(tmp_path, 'first', actual=200, requested=600, wall_limit=True)
    second, _, _ = make_run(tmp_path, 'second', actual=400, requested=600, wall_limit=True)
    for run, setup, hit in ((first, 120., False), (second, 2., True)):
        summary = json.loads((run / 'summary.json').read_text())
        summary['timing'] = dict(setup_seconds=setup, training_wall_seconds=30.)
        save_object(run / 'summary.json', summary)
        manifest = json.loads((run / 'manifest.json').read_text())
        manifest['execution_optimizations'] = dict(mi_statistics_cache=dict(train=dict(hit=hit)))
        save_object(run / 'manifest.json', manifest)
    report = auditor.collect_sweep([first, second])
    fairness = report['comparison']['fairness']
    assert fairness['requested_updates_equal'] and not fairness['actual_updates_equal']
    assert not fairness['every_actual_equals_requested']
    assert fairness['measured_setup_seconds'] == [120., 2.]
    assert not fairness['measured_setup_seconds_equal']
    assert fairness['recorded_MI_cache_hit_states'] == [dict(train=False), dict(train=True)]
    assert not fairness['recorded_MI_cache_hit_states_equal']
    assert any('built statistics' in warning for warning in fairness['warnings'])
    assert report['runs'][0]['timing_evidence']['summary_timing'] == dict(setup_seconds=120., training_wall_seconds=30.)
    assert report['runs'][0]['learning_rate_evidence']['schedule_active'] is None


def test_missing_timing_is_not_estimated_from_other_elapsed_fields(tmp_path):
    run, _, _ = make_run(tmp_path)
    summary = json.loads((run / 'summary.json').read_text())
    summary['setup_and_training_seconds'] = 80.
    save_object(run / 'summary.json', summary)
    result = auditor.collect_sweep([run])
    evidence = result['runs'][0]['timing_evidence']
    assert evidence['setup_and_training_seconds'] == 80.
    assert evidence['setup_seconds_by_source'] == {}
    assert result['comparison']['fairness']['measured_setup_seconds'] == [None]
