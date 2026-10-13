"""CPU orchestration audits: fixed updates, exact control, no retries or sampling in plans."""
from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import run_ising2d_exponent_ablation as pipeline


def arguments(tmp_path, **changes):
    values = dict(baseline_run=tmp_path / 'historical' / 'run', cache=tmp_path / 'L36.pt',
        expanded_cache=tmp_path / 'L108.pt', output=tmp_path / 'job', device='cuda:0',
        epochs=12000, seed=91, replicas=12, threads=2, mi_cache_dir=None)
    return Namespace(**(values | changes))


def cli(tmp_path, *extra):
    return ['--baseline-run', str(tmp_path / 'historical' / 'run'), '--cache', str(tmp_path / 'L36.pt'),
            '--expanded-cache', str(tmp_path / 'L108.pt'), '--output', str(tmp_path / 'job'), *extra]


def differences(left, right, prefix=''):
    results = []
    for key in left.keys() | right.keys():
        name = f'{prefix}.{key}' if prefix else key
        if isinstance(left.get(key), dict) and isinstance(right.get(key), dict):
            results.extend(differences(left[key], right[key], name))
        elif left.get(key) != right.get(key):
            results.append(name)
    return sorted(results)


def test_plan_changes_only_dim_or_cache_and_scale_range(tmp_path):
    plan = pipeline.build_plan(arguments(tmp_path))
    base, vector, expanded = plan['arms']
    assert [arm['name'] for arm in plan['arms']] == list(pipeline.ARM_NAMES)
    assert differences(base['config'], vector['config']) == ['model.theory_dim']
    assert differences(base['config'], expanded['config']) == [
        'data.cache', 'evaluation.max_rg_steps', 'training.max_rg_steps']
    assert base['lattice_sizes'] == vector['lattice_sizes'] == [36, 12, 4]
    assert expanded['lattice_sizes'] == [108, 36, 12, 4]
    assert base['start_updates'] == 9989 and vector['start_updates'] == expanded['start_updates'] == 0
    assert all(arm['final_updates'] == arm['config']['training']['epochs'] == 12000 for arm in plan['arms'])
    assert plan['mi_cache_dir'] == str(tmp_path / 'historical' / 'mi_cache')
    for arm in plan['arms']:
        cfg = arm['config']
        assert cfg['model']['block_factor'] == 3 and cfg['model']['kernel_initialization'] == 'random'
        assert cfg['model']['hidden'] == 96 and cfg['training']['gradient_mode'] == 'qc_r_separate'
        assert cfg['training']['system_batch'] == 16 and cfg['training']['sample_batch'] == 128
        assert cfg['training']['lr_milestones'] == [6000, 9000] and cfg['training']['lr_gamma'] == .3
        for record in arm['commands']:
            assert '--max-training-seconds' not in record['command']
            assert '--max-wall-seconds' not in record['command']
            assert '--allow-nondefault-recipe' not in record['command']
    assert '--resume-checkpoint' in base['commands'][1]['command']
    assert all('--resume-checkpoint' not in arm['commands'][1]['command'] for arm in (vector, expanded))
    assert vector['commands'][2]['command'][3] == 'scripts.report_ising2d_K_vector'
    assert base['commands'][-1]['command'][3] == 'scripts.report_ising_scalar_exponents'
    assert '--max-rg-steps' in plan['data_generation']['command']


def test_plan_is_read_only_and_does_not_need_remote_paths(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pipeline.subprocess, 'check_output', lambda *a, **k: pytest.fail('No subprocess'))
    monkeypatch.setattr(pipeline, 'check_data', lambda *a, **k: pytest.fail('No dataset reads'))
    monkeypatch.setattr(pipeline, 'check_source', lambda *a, **k: pytest.fail('No checkpoint reads'))
    plan = pipeline.main(cli(tmp_path, '--plan'))
    assert json.loads(capsys.readouterr().out) == plan
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('changes', [dict(epochs=9989), dict(seed=92), dict(replicas=3),
    dict(threads=1), dict(device='cuda'), dict(expanded_cache=Path('/tmp/data.npy')),
    dict(output=Path('/tmp/source'), baseline_run=Path('/tmp/source/run'))])
def test_locked_ablation_rejects_unauthorized_plan_variants(tmp_path, changes):
    with pytest.raises(ValueError):
        pipeline.build_plan(arguments(tmp_path, **changes))


def fake_source(tmp_path, monkeypatch, *, mutate=None):
    plan = pipeline.build_plan(arguments(tmp_path))
    source = Path(plan['baseline_run'])
    source.mkdir(parents=True)
    config = deepcopy(plan['arms'][0]['config'])
    manifest = dict(config=config, data_sha256=pipeline.SOURCE_DATA_SHA256, labels_used_in_training=False)
    summary = dict(config=config, actual_epochs=9989, requested_epochs=12000)
    checkpoint = dict(epoch=9989, metadata=dict(config=config, data_sha256=pipeline.SOURCE_DATA_SHA256,
        execution_optimizations=dict(selected_empirical_mi=True)), optimizer={}, batch_generator_state={},
        rsmi_generator_state={}, history=[], torch_rng_state={}, cuda_rng_state={}, cuda_device_index=0,
        lr_scheduler_state=dict(completed_updates=9989))
    if mutate:
        mutate(manifest, summary, checkpoint)
    (source / 'manifest.json').write_text(json.dumps(manifest))
    (source / 'summary.json').write_text(json.dumps(summary))
    (source / 'checkpoint_last.pt').write_text('untouched historical checkpoint')
    monkeypatch.setattr(pipeline.torch, 'load', lambda *a, **k: checkpoint)
    return plan, source, config


def test_control_source_requires_exact_config_rng_adam_and_actual_updates(tmp_path, monkeypatch):
    plan, source, config = fake_source(tmp_path, monkeypatch)
    before = {path: path.read_bytes() for path in source.iterdir()}
    audit = pipeline.check_source(source, config)
    assert audit['actual_epochs'] == 9989
    assert all(path.read_bytes() == content for path, content in before.items())
    assert len(audit['files']) == 3


@pytest.mark.parametrize('mutation,match', [
    (lambda m, s, c: s.update(actual_epochs=9000), '9989'),
    (lambda m, s, c: c.pop('optimizer'), 'Adam'),
    (lambda m, s, c: c.pop('torch_rng_state'), 'Adam'),
    (lambda m, s, c: c.update(cuda_device_index=1), 'CUDA'),
    (lambda m, s, c: c['metadata'].update(data_sha256='bad'), 'fingerprint'),
    (lambda m, s, c: m.update(config={}), 'full configuration')])
def test_control_source_rejects_nonexact_resume(tmp_path, monkeypatch, mutation, match):
    _, source, config = fake_source(tmp_path, monkeypatch, mutate=mutation)
    with pytest.raises(ValueError, match=match):
        pipeline.check_source(source, config)


def fake_workflow(tmp_path, monkeypatch, *, fail=None, mutate=None, expanded_exists=False, actual=12000):
    cache = tmp_path / 'L36.pt'
    cache.write_text('original data')
    cache.with_suffix('.json').write_text('{}')
    expanded = tmp_path / 'L108.pt'
    if expanded_exists:
        expanded.write_text('expanded data')
        expanded.with_suffix('.json').write_text('{}')
    _, source, _ = fake_source(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(pipeline.subprocess, 'check_output', lambda *a, **k: 'revision\n')

    def check_data(path, **kwargs):
        return dict(sha256=pipeline.file_fingerprint(path),
                    sampling_sidecar_sha256=pipeline.file_fingerprint(Path(path).with_suffix('.json')))

    def mock_stage(output, status, name, command):
        calls.append(name)
        if fail == name:
            raise RuntimeError(f'{name} failed')
        if name == 'expanded_scales_data_generation':
            expanded.write_text('expanded data')
            expanded.with_suffix('.json').write_text('{}')
        if name == 'baseline_control_training' and mutate:
            path = source / 'checkpoint_last.pt' if mutate == 'source' else cache
            path.write_text('unexpected mutation')

    def check_training(arm, data, **kwargs):
        if actual != 12000:
            raise ValueError('Training did not complete exactly 12000 cumulative updates; requires user action')
        return dict(actual_epochs=12000, requested_epochs=12000, new_updates=12000-arm['start_updates'])

    monkeypatch.setattr(pipeline, 'stage', mock_stage)
    monkeypatch.setattr(pipeline, 'check_data', check_data)
    monkeypatch.setattr(pipeline, 'check_training', check_training)
    monkeypatch.setattr(pipeline, 'check_reports', lambda arm: dict(checkpoint_epoch=12000, bounded_root_count=3))
    return calls, source


def test_serial_workflow_keeps_source_unchanged_and_generates_only_l108(tmp_path, monkeypatch):
    calls, source = fake_workflow(tmp_path, monkeypatch)
    before = {path: path.read_bytes() for path in source.iterdir()}
    status = pipeline.main(cli(tmp_path))
    assert status['status'] == 'complete'
    assert [name for name in calls if name.endswith('_training')] == [f'{arm}_training' for arm in pipeline.ARM_NAMES]
    assert calls.index('expanded_scales_data_generation') > calls.index('dimg2_loss_report')
    assert calls.count('expanded_scales_data_generation') == 1
    assert status['training']['baseline_control']['new_updates'] == 2011
    assert all(path.read_bytes() == content for path, content in before.items())
    assert (tmp_path / 'job' / 'configs' / 'dimg2.json').exists()


def test_existing_complete_expanded_pair_is_reused(tmp_path, monkeypatch):
    calls, _ = fake_workflow(tmp_path, monkeypatch, expanded_exists=True)
    status = pipeline.main(cli(tmp_path))
    assert 'expanded_scales_data_generation' not in calls
    assert status['stages'][0]['status'] == 'reused_existing_pair'


@pytest.mark.parametrize('failure', ['baseline_control_training', 'dimg2_phase_report',
                                   'expanded_scales_data_generation', 'expanded_scales_training'])
def test_errors_stop_without_advance_or_retry(tmp_path, monkeypatch, failure):
    calls, _ = fake_workflow(tmp_path, monkeypatch, fail=failure)
    with pytest.raises(RuntimeError, match=failure):
        pipeline.main(cli(tmp_path))
    assert calls[-1] == failure and calls.count(failure) == 1
    job = json.loads((tmp_path / 'job' / 'job.json').read_text())
    assert job['status'] == 'failed' and job['requires_user_action'] is True


@pytest.mark.parametrize('mutation', ['source', 'data'])
def test_changed_source_or_data_stops_before_reports(tmp_path, monkeypatch, mutation):
    calls, _ = fake_workflow(tmp_path, monkeypatch, mutate=mutation)
    with pytest.raises(ValueError, match='changed'):
        pipeline.main(cli(tmp_path))
    assert calls[-1] == 'baseline_control_training'


def test_incomplete_updates_are_not_reported_as_matched_or_automatically_continued(tmp_path, monkeypatch):
    calls, _ = fake_workflow(tmp_path, monkeypatch, actual=11000)
    with pytest.raises(ValueError, match='12000'):
        pipeline.main(cli(tmp_path))
    assert calls[-1] == 'baseline_control_training'


def test_existing_outputs_and_incomplete_data_pairs_are_never_overwritten(tmp_path):
    output = tmp_path / 'job'
    output.mkdir()
    with pytest.raises(FileExistsError):
        pipeline.main(cli(tmp_path))
    output.rmdir()
    (tmp_path / 'L36.pt').touch()
    with pytest.raises(ValueError, match='Incomplete'):
        pipeline.main(cli(tmp_path))
    assert not output.exists()


@pytest.mark.parametrize('returncode', [0, 7])
def test_stage_records_pid_elapsed_time_and_single_launch(tmp_path, monkeypatch, returncode):
    (tmp_path / 'logs').mkdir()
    status = dict(stages=[], current=None)
    calls = []
    class Process:
        pid = 123
        def wait(self):
            active = json.loads((tmp_path / 'job.json').read_text())['current']
            assert active['pid'] == 123
            return returncode
    def launch(command, **kwargs):
        calls.append(command)
        assert kwargs['env']['OMP_NUM_THREADS'] == '2'
        assert kwargs['stderr'] == pipeline.subprocess.STDOUT
        return Process()
    monkeypatch.setattr(pipeline.subprocess, 'Popen', launch)
    if returncode:
        with pytest.raises(RuntimeError, match='no automatic retry'):
            pipeline.stage(tmp_path, status, 'control_training', ['python', 'test'])
    else:
        pipeline.stage(tmp_path, status, 'control_training', ['python', 'test'])
    assert len(calls) == 1
    assert status['stages'][0]['returncode'] == returncode
    assert status['stages'][0]['elapsed_seconds'] >= 0


def production_pair(size, depth):
    values = pipeline.coupling_knots().tolist()
    options = dict(size=size, max_rg_steps=depth, seed=91, knots=64, samples=128,
                   replicas=12, equil=4096, gap_low=512, gap_high=128, pilot_samples=1024)
    metadata = dict(sampling_options=options, max_rg_steps=depth, block_factor=3, B=0.,
        coupling_names=['K'], seed=91, replicas=12, couplings=values, data_sha256='digest',
        lattice_sizes=[shape[0] for shape in pipeline.ScaleConfig((size, size), depth, block_factor=3).shapes()],
        ensembles=[])
    for i, k in enumerate(values):
        for replica in range(12):
            row = i * 12 + replica
            metadata['ensembles'].append(dict(K_index=i, replica=replica, K=k, test=i % 3 == 1,
                seed=pipeline.derived_seed(91, 1, row), global_flip_seed=pipeline.derived_seed(91, 2, row)))
    datasets = []
    for split in ('train', 'test'):
        knots = [k for i, k in enumerate(values) if (i % 3 == 1) == (split == 'test')]
        shape = (len(knots) * 12, 128, size, size, 1)
        metadata[split + '_K'] = knots
        metadata[split + '_shape'] = list(shape)
        datasets.append(SimpleNamespace(configurations=SimpleNamespace(shape=shape, dtype=torch.float64),
            couplings=torch.tensor([k for k in knots for _ in range(12)], dtype=torch.float64)))
    return metadata, datasets


@pytest.mark.parametrize('size,depth', [(36, 2), (108, 3)])
def test_data_validator_checks_actual_shape_and_depth_without_allocating_spins(tmp_path, monkeypatch, size, depth):
    metadata, datasets = production_pair(size, depth)
    if size == 36:
        metadata['sampling_options'].pop('max_rg_steps')
    cache = tmp_path / 'cache.pt'
    cache.with_suffix('.json').write_text(json.dumps(metadata))
    monkeypatch.setattr(pipeline, 'file_fingerprint', lambda path: 'digest')
    monkeypatch.setattr(pipeline, 'load_ensembles', lambda path: tuple(datasets))
    report = pipeline.check_data(cache, size=size, max_rg_steps=depth, expected_digest='digest')
    assert report['train_shape'] == [516, 128, size, size, 1]
    assert report['test_shape'] == [252, 128, size, size, 1]


@pytest.mark.parametrize('mutation', ['digest', 'options', 'grid', 'depth', 'seeds', 'labels', 'shape'])
def test_data_validator_rejects_unmatched_sampling_contract(tmp_path, monkeypatch, mutation):
    metadata, datasets = production_pair(108, 3)
    if mutation == 'digest':
        metadata['data_sha256'] = 'other'
    elif mutation == 'options':
        metadata['sampling_options']['equil'] = 512
    elif mutation == 'grid':
        metadata['couplings'][1] += .0001
    elif mutation == 'depth':
        metadata['max_rg_steps'] = 2
    elif mutation == 'seeds':
        metadata['ensembles'][0]['seed'] = metadata['ensembles'][1]['seed']
    elif mutation == 'labels':
        datasets[0].couplings[0] = .201
    else:
        datasets[0].configurations.shape = (516, 128, 36, 36, 1)
    cache = tmp_path / 'cache.pt'
    cache.with_suffix('.json').write_text(json.dumps(metadata))
    monkeypatch.setattr(pipeline, 'file_fingerprint', lambda path: 'digest')
    monkeypatch.setattr(pipeline, 'load_ensembles', lambda path: tuple(datasets))
    with pytest.raises(ValueError):
        pipeline.check_data(cache, size=108, max_rg_steps=3)


def training_artifacts(tmp_path, arm, *, actual=12000):
    run = Path(arm['run'])
    run.mkdir(parents=True)
    manifest = dict(config=arm['config'], config_source=arm['config_path'], data_sha256='digest',
        training_driver='rsmi_separate_qcr', precision='torch.float64', estimator='histogram',
        environment_mode='full_shell', buffer=1, independent_c=True, labels_used_in_training=False,
        rsmi_weight=2., execution_optimizations=dict(selected_empirical_mi=True,
            mi_statistics_cache=dict(train=dict(hit=True), test=dict(hit=True))))
    if arm['name'] == 'baseline_control':
        manifest['resumed_from'] = arm['commands'][1]['command'][-1]
    summary = dict(config=arm['config'], actual_epochs=actual, requested_epochs=12000,
        stopped_by_training_limit=False, stopped_by_wall_limit=False, timing=dict(training_wall_seconds=100.),
        lr_scheduler_state=dict(completed_updates=12000, next_learning_rate=.00009))
    (run / 'manifest.json').write_text(json.dumps(manifest))
    (run / 'summary.json').write_text(json.dumps(summary))
    return run, manifest, summary


@pytest.mark.parametrize('name', pipeline.ARM_NAMES)
def test_training_endpoint_validator_requires_exact12000_and_owned_manifest(tmp_path, name):
    arm = next(arm for arm in pipeline.build_plan(arguments(tmp_path))['arms'] if arm['name'] == name)
    training_artifacts(tmp_path, arm)
    report = pipeline.check_training(arm, dict(sha256='digest'), require_mi_hits=True)
    assert report['actual_epochs'] == 12000
    assert report['new_updates'] == (2011 if name == 'baseline_control' else 12000)


@pytest.mark.parametrize('mutation', ['cap', 'actual', 'config', 'source', 'mi_hit', 'selected', 'lr_clock', 'lr_rate'])
def test_training_validator_rejects_truncation_recipe_or_execution_changes(tmp_path, mutation):
    arm = pipeline.build_plan(arguments(tmp_path))['arms'][0]
    run, manifest, summary = training_artifacts(tmp_path, arm)
    if mutation == 'cap':
        summary['stopped_by_training_limit'] = True
    elif mutation == 'actual':
        summary['actual_epochs'] = 11000
    elif mutation == 'config':
        manifest['config']['model']['hidden'] = 60
        # Both objects share the initial dictionary; restore the planned arm only.
        arm = pipeline.build_plan(arguments(tmp_path))['arms'][0]
    elif mutation == 'source':
        manifest['resumed_from'] = '/unexpected/checkpoint.pt'
    elif mutation == 'mi_hit':
        manifest['execution_optimizations']['mi_statistics_cache']['test']['hit'] = False
    elif mutation == 'lr_clock':
        summary['lr_scheduler_state']['completed_updates'] = 11000
    elif mutation == 'lr_rate':
        summary['lr_scheduler_state']['next_learning_rate'] = .001
    else:
        manifest['execution_optimizations']['selected_empirical_mi'] = False
    (run / 'manifest.json').write_text(json.dumps(manifest))
    (run / 'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        pipeline.check_training(arm, dict(sha256='digest'), require_mi_hits=True)


@pytest.mark.parametrize('name', pipeline.ARM_NAMES)
def test_final_report_audit_retains_all_roots_and_requires_actual_endpoint(tmp_path, name):
    arm = next(arm for arm in pipeline.build_plan(arguments(tmp_path))['arms'] if arm['name'] == name)
    run = Path(arm['run'])
    (run / 'full_report').mkdir(parents=True)
    (run / 'report').mkdir()
    for filename in ['metrics.json', 'fixed_points.json', 'encodings.npz', 'coupling_by_scale.png',
                     'vector_critical_summary.png' if name == 'dimg2' else 'scalar_critical_summary.png']:
        (run / 'full_report' / filename).write_text('public artifact')
    (run / 'report/loss_comparison.png').write_text('public artifact')
    exponent_path = run / 'full_report/critical_exponents.json'
    exponent_path.write_text(json.dumps(dict(checkpoint_epoch=12000, roots=[dict(g=[1.]), dict(g=[2.])])))
    result = pipeline.check_reports(arm)
    assert result['bounded_root_count'] == 2 and result['checkpoint_epoch'] == 12000
    exponent_path.write_text(json.dumps(dict(checkpoint_epoch=10000, roots=[])))
    with pytest.raises(ValueError, match='12000'):
        pipeline.check_reports(arm)
    exponent_path.unlink()
    with pytest.raises(ValueError, match='artifacts'):
        pipeline.check_reports(arm)
