import json
from pathlib import Path

from scripts import report_scalar_exponent_trend as reporter


def saved_preview(run, folder, epoch, roots, *, phase='completed', block=2):
    path = run / 'monitor_previews' / folder
    path.mkdir(parents=True)
    (path / 'preview.json').write_text(json.dumps(dict(phase=phase)))
    (path / 'critical_exponents.json').write_text(json.dumps(dict(
        checkpoint_epoch=epoch, block_factor=block, roots=roots)))
    (path / 'metrics.json').write_text(json.dumps(dict(fixed_points=dict(roots=roots))))
    return path


def root(point, multiplier, nu, *, outside=False, branches=None, stability='unstable'):
    return dict(root_id='F1', g=[point], multiplier=multiplier, nu=nu,
                outside_training_support=outside, stability=stability,
                root_verified=True, derivative_verified=True,
                microscopic_inverse_coupling=branches if branches is not None else [.4, .5])


def test_actual_checkpoint_epochs_are_used_without_cross_step_f_labels(tmp_path):
    saved_preview(tmp_path, 'epoch_999999', 200,
                  [root(-1., 2.1, .93), root(0., .5, None, stability='stable')])
    saved_preview(tmp_path, 'epoch_000001', 400,
                  [root(1., 1.9, 1.08)])
    report = reporter.collect_trend(tmp_path)
    assert [row['checkpoint_epoch'] for row in report['snapshots']] == [200, 400]
    assert [row['root_count'] for row in report['snapshots']] == [2, 1]
    assert [record['local_root_id'] for row in report['snapshots'] for record in row['roots']] == ['F1', 'F1', 'F1']
    assert report['selection']['branch_matching'].startswith('None:')
    assert not report['selection']['theory_used_for_selection']


def test_stable_negative_outside_roots_and_every_inverse_branch_are_retained(tmp_path):
    saved_preview(tmp_path, 'epoch_000200', 200, [
        root(-1., -2., None, branches=[.1, .3, .5]),
        root(0., .5, None, stability='stable'),
        root(1., 2., None, outside=True),
        root(.5, 2.2, .88),
    ])
    report = reporter.collect_trend(tmp_path)
    records = report['snapshots'][0]['roots']
    assert len(records) == 4
    assert records[0]['sign_reversing'] and records[0]['conditional_nu'] is None
    assert records[0]['microscopic_inverse_coupling'] == [.1, .3, .5]
    assert records[1]['stability'] == 'stable' and records[1]['conditional_nu'] is None
    assert records[2]['outside_training_support'] and records[2]['conditional_nu'] is None
    assert records[3]['conditional_nu'] == .88
    assert report['theory_references']['thermal_lambda'] == [dict(block_factor=2., multiplier=2.)]


def test_incomplete_missing_data_and_missing_actual_epoch_are_skipped_with_reasons(tmp_path):
    saved_preview(tmp_path, 'epoch_000200', 200, [root(0., 2., 1.)], phase='exponents')
    missing = saved_preview(tmp_path, 'epoch_000400', 400, [root(0., 2., 1.)])
    (missing / 'metrics.json').unlink()
    saved_preview(tmp_path, 'epoch_000600', None, [root(0., 2., 1.)])
    good = saved_preview(tmp_path, 'epoch_000800', 800, [root(0., 2., 1.)])
    report = reporter.collect_trend(tmp_path)
    assert [row['preview'] for row in report['snapshots']] == [str(good)]
    assert len(report['skipped']) == 3
    reasons = ' '.join(row['reason'] for row in report['skipped'])
    assert 'not completed' in reasons and 'metrics.json' in reasons
    assert 'checkpoint_epoch' in reasons and 'directory name is not used' in reasons


def test_missing_nu_is_not_inferred_from_lambda_and_metrics_branches_are_merged(tmp_path):
    point = root(0., 2., None)
    point.pop('microscopic_inverse_coupling')
    preview = saved_preview(tmp_path, 'epoch_000200', 200, [point])
    (preview / 'metrics.json').write_text(json.dumps(dict(fixed_points=dict(roots=[
        dict(g=[0.], microscopic_inverse_coupling=[.2, .45, .6]),
        dict(g=[1.], stability='stable', microscopic_inverse_coupling=[]),
    ]))))
    records = reporter.collect_trend(tmp_path)['snapshots'][0]['roots']
    assert records[0]['conditional_nu'] is None
    assert records[0]['microscopic_inverse_coupling'] == [.2, .45, .6]
    assert len(records) == 2 and records[1]['multiplier'] is None
    assert records[1]['warnings']


def test_cli_outputs_are_isolated_and_existing_reports_are_unchanged(tmp_path):
    preview = saved_preview(tmp_path, 'epoch_000200', 200, [root(0., 2., 1.)])
    originals = {path: path.read_bytes() for path in preview.iterdir()}
    report = reporter.main(['--run', str(tmp_path)])
    out = tmp_path / 'monitor_previews' / 'exponent_trend'
    assert (out / 'trend.png').stat().st_size > 0
    assert json.loads((out / 'trend.json').read_text())['snapshots'] == report['snapshots']
    assert all(path.read_bytes() == value for path, value in originals.items())
    assert {path.name for path in out.iterdir()} == {'trend.png', 'trend.json'}
    source = Path(reporter.__file__).read_text()
    assert 'import torch' not in source and 'load_run' not in source


def test_empty_run_still_has_a_labeled_empty_trend(tmp_path):
    report = reporter.main(['--run', str(tmp_path)])
    assert report['snapshots'] == [] and report['skipped'] == []
    assert (tmp_path / 'monitor_previews' / 'exponent_trend' / 'trend.png').is_file()


def test_missing_or_failed_verification_does_not_supply_conditional_nu(tmp_path):
    missing = root(0., 2., 1.)
    missing.pop('derivative_verified')
    failed = root(1., 2., 1.)
    failed['root_verified'] = False
    saved_preview(tmp_path, 'epoch_000200', 200, [missing, failed])
    records = reporter.collect_trend(tmp_path)['snapshots'][0]['roots']
    assert len(records) == 2
    assert all(record['conditional_nu'] is None for record in records)
    assert all(record['multiplier'] == 2. for record in records)
    assert all(record['warnings'] for record in records)
