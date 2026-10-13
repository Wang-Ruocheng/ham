"""CPU-only checks for the larger-replica single-case detached workflow."""
from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from recurrent_rg.config import RunConfig
from scripts import run_scalar_ensemble_gpu_pipeline as pipeline


def arguments(tmp_path, **changes):
    values = dict(cache=tmp_path / 'cache.pt', output=tmp_path / 'job', device='cuda:0',
                  epochs=12000, seed=91, threads=2, replicas=12, policy='multistep',
                  training_wall_seconds=900., mi_cache_dir=None)
    return Namespace(**(values | changes))


def test_single_case_reuses_default_config_with_fixed_multistep_policy(tmp_path):
    plan = pipeline.build_plan(arguments(tmp_path))
    config = plan['config']
    assert config == pipeline.make_configs(tmp_path / 'cache.pt')['multistep']
    RunConfig.from_dict(config).validate()
    assert config['schema_version'] == 5
    assert config['model']['block_factor'] == 3
    assert config['model']['theory_dim'] == 1
    assert config['model']['hidden'] == 96
    assert config['model']['kernel_initialization'] == 'random'
    assert config['training']['epochs'] == 12000
    assert config['training']['system_batch'] == 16
    assert config['training']['sample_batch'] == 128
    assert config['training']['gradient_mode'] == 'qc_r_separate'
    assert config['training']['learning_rate'] == .001
    assert config['training']['lr_milestones'] == [6000, 9000]
    assert config['training']['lr_gamma'] == .3
    assert plan['replicas'] == 12
    assert plan['run'] == str(tmp_path / 'job' / 'run')
    assert plan['protocol']['labels_used_in_training'] is False
    assert plan['protocol']['sampling_and_mi_preprocessing_outside_training_wall_cap'] is True
    assert pipeline.build_plan(arguments(tmp_path, epochs=1000))['config']['training']['lr_milestones'] == [6000, 9000]


def test_preprocessing_precedes_one_fresh_training_and_three_final_reports(tmp_path):
    plan = pipeline.build_plan(arguments(tmp_path))
    commands = plan['commands']
    assert [entry['stage'] for entry in commands] == [
        'data_generation', 'mi_preprocessing', 'training', 'scalar_report', 'loss_report', 'exponent_report']
    assert commands[0]['command'][3] == 'scripts.prepare_ising2d_lr_data'
    assert commands[0]['command'][-2:] == ['--replicas', '12']
    assert commands[1]['command'][3] == 'scripts.prepare_qcr_mi_cache'
    assert '--device' not in commands[1]['command']
    assert commands[2]['command'][3] == 'experiments.rsmi_separate_qcr'
    assert commands[2]['command'][-2:] == ['--mi-cache-dir', plan['mi_cache_dir']]
    assert commands[2]['command'][commands[2]['command'].index('--max-training-seconds') + 1] == '900.0'
    assert '--max-wall-seconds' not in commands[2]['command']
    assert commands[-1]['command'][3] == 'scripts.report_ising_scalar_exponents'
    for entry in commands:
        assert not any(flag in entry['command'] for flag in (
            '--allow-nondefault-recipe', '--allow-legacy-training', '--source',
            '--resume-checkpoint', '--mi-initialize', '--joint-train-plateau', '--checkpoint'))
    assert '--mi-cache-dir' not in pipeline.build_plan(arguments(tmp_path, device='cpu'))['commands'][2]['command']


def test_plan_only_has_no_file_or_subprocess_side_effects(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pipeline.subprocess, 'check_output', lambda *a, **kw: pytest.fail('No subprocesses'))
    plan = pipeline.main(['--cache', str(tmp_path / 'absent.pt'), '--output', str(tmp_path / 'job'), '--plan-only'])
    assert not (tmp_path / 'job').exists()
    assert json.loads(capsys.readouterr().out) == plan


@pytest.mark.parametrize('changes', [dict(replicas=0), dict(epochs=2), dict(device='cuda'),
    dict(threads=0), dict(training_wall_seconds=float('inf')), dict(training_wall_seconds=-1.),
    dict(policy='cosine'), dict(cache=Path('/tmp/wrong.npy'))])
def test_invalid_plan_is_rejected(tmp_path, changes):
    with pytest.raises(ValueError):
        pipeline.build_plan(arguments(tmp_path, **changes))


def test_existing_output_or_incomplete_cache_fails_without_job_creation(tmp_path):
    output, cache = tmp_path / 'job', tmp_path / 'cache.pt'
    output.mkdir()
    with pytest.raises(FileExistsError):
        pipeline.main(['--cache', str(cache), '--output', str(output)])
    output.rmdir()
    cache.touch()
    with pytest.raises(ValueError, match='Incomplete'):
        pipeline.main(['--cache', str(cache), '--output', str(output)])
    assert not output.exists()


def fake_workflow(tmp_path, monkeypatch, *, existing=False, fail=None, mutate=None, actual=12000):
    cache, output = tmp_path / 'cache.pt', tmp_path / 'job'
    calls = []
    monkeypatch.setattr(pipeline.subprocess, 'check_output', lambda *a, **kw: 'abc123\n')

    def create_pair():
        cache.write_text('audited cache')
        cache.with_suffix('.json').write_text('{"audit":"sampling"}')

    if existing:
        create_pair()

    def data_check(path, **kwargs):
        assert kwargs == dict(replicas=12, seed=91)
        return dict(sha256=pipeline.file_fingerprint(cache),
                    sampling_sidecar_sha256=pipeline.file_fingerprint(cache.with_suffix('.json')))

    def mock_stage(out, status, name, command):
        calls.append((name, command))
        if name == fail:
            raise RuntimeError(f'{name} failed')
        if name == 'data_generation':
            create_pair()
        if name == 'training' and mutate:
            target = cache if mutate == 'cache' else cache.with_suffix('.json') if mutate == 'sidecar' else out / 'config.json'
            target.write_text('mutated after training')

    monkeypatch.setattr(pipeline, 'stage', mock_stage)
    monkeypatch.setattr(pipeline, 'check_data', data_check)
    monkeypatch.setattr(pipeline, 'check_training_manifest', lambda *a, **kw: dict(
        actual_epochs=actual, requested_epochs=12000, completed_requested_updates=actual == 12000))
    return cache, output, calls


def test_detached_job_generates_validates_warms_and_reports_once(tmp_path, monkeypatch):
    cache, output, calls = fake_workflow(tmp_path, monkeypatch)
    status = pipeline.main(['--cache', str(cache), '--output', str(output)])
    assert [name for name, _ in calls] == [
        'data_generation', 'mi_preprocessing', 'training', 'scalar_report', 'loss_report', 'exponent_report']
    assert status['status'] == 'complete' and status['git_commit'] == 'abc123'
    saved = json.loads((output / 'job.json').read_text())
    assert saved['training']['actual_epochs'] == 12000
    assert saved['config_sha256'] == pipeline.file_fingerprint(output / 'config.json')


def test_existing_complete_pair_is_validated_and_reused(tmp_path, monkeypatch):
    cache, output, calls = fake_workflow(tmp_path, monkeypatch, existing=True)
    status = pipeline.main(['--cache', str(cache), '--output', str(output)])
    assert calls[0][0] == 'mi_preprocessing'
    assert status['stages'][0]['name'] == 'data_generation'
    assert status['stages'][0]['status'] == 'reused_existing_pair'


def test_capped_job_finishes_reports_and_flags_user_action_without_retry(tmp_path, monkeypatch):
    cache, output, calls = fake_workflow(tmp_path, monkeypatch, actual=4321)
    status = pipeline.main(['--cache', str(cache), '--output', str(output)])
    assert status['status'] == 'requires_user_action'
    assert status['requires_user_action'] is True and status['workflow_complete'] is True
    assert status['training']['actual_epochs'] == 4321
    assert calls[-1][0] == 'exponent_report'
    assert [name for name, _ in calls].count('training') == 1


@pytest.mark.parametrize('failure', ['data_generation', 'mi_preprocessing', 'training'])
def test_stage_failure_is_durable_and_never_retried(tmp_path, monkeypatch, failure):
    cache, output, calls = fake_workflow(tmp_path, monkeypatch, fail=failure)
    with pytest.raises(RuntimeError, match=failure):
        pipeline.main(['--cache', str(cache), '--output', str(output)])
    assert calls[-1][0] == failure
    assert [name for name, _ in calls].count(failure) == 1
    assert 'scalar_report' not in [name for name, _ in calls]
    assert json.loads((output / 'job.json').read_text())['status'] == 'failed'


@pytest.mark.parametrize('returncode', [0, 7])
def test_stage_records_running_pid_and_completed_result(tmp_path, monkeypatch, returncode):
    output = tmp_path / 'job'
    output.mkdir()
    (output / 'logs').mkdir()
    status = dict(status='running', stages=[], current=None)
    seen = []

    class FakeProcess:
        pid = 12345

        def wait(self):
            running = json.loads((output / 'job.json').read_text())['current']
            assert running['name'] == 'mi_preprocessing'
            assert running['pid'] == self.pid and running['status'] == 'running'
            return returncode

    def launch(command, **kwargs):
        seen.append(command)
        assert kwargs['stderr'] == pipeline.subprocess.STDOUT
        return FakeProcess()

    monkeypatch.setattr(pipeline.subprocess, 'Popen', launch)
    if returncode:
        with pytest.raises(RuntimeError, match='no automatic retry'):
            pipeline.stage(output, status, 'mi_preprocessing', ['python', 'mock-preprocessor'])
    else:
        pipeline.stage(output, status, 'mi_preprocessing', ['python', 'mock-preprocessor'])
    assert len(seen) == 1
    final = json.loads((output / 'job.json').read_text())
    assert final['current'] is None
    assert final['stages'][0]['returncode'] == returncode
    assert final['stages'][0]['status'] == ('failed' if returncode else 'complete')


@pytest.mark.parametrize('mutation', ['cache', 'sidecar', 'config'])
def test_mutated_inputs_stop_before_reports(tmp_path, monkeypatch, mutation):
    cache, output, calls = fake_workflow(tmp_path, monkeypatch, mutate=mutation)
    with pytest.raises(ValueError, match='changed'):
        pipeline.main(['--cache', str(cache), '--output', str(output)])
    assert calls[-1][0] == 'training'
    assert json.loads((output / 'job.json').read_text())['status'] == 'failed'


def production_metadata(replicas=12, seed=91):
    values = pipeline.coupling_knots().tolist()
    metadata = dict(sampling_options=dict(size=36, seed=seed, knots=64, samples=128,
        replicas=replicas, equil=4096, gap_low=512, gap_high=128, pilot_samples=1024),
        seed=seed, replicas=replicas, lattice_sizes=[36, 12, 4], B=0., coupling_names=['K'],
        data_sha256='digest', couplings=values, ensembles=[])
    ensembles = []
    for split in ('train', 'test'):
        knots = [k for i, k in enumerate(values) if (i % 3 == 1) == (split == 'test')]
        metadata[split + '_K'] = knots
        shape = (len(knots) * replicas, 128, 36, 36, 1)
        metadata[split + '_shape'] = list(shape)
        labels = torch.tensor([k for k in knots for _ in range(replicas)], dtype=torch.float64)
        ensembles.append(SimpleNamespace(configurations=SimpleNamespace(shape=shape, dtype=torch.float64),
                                         couplings=labels))
    for i, k in enumerate(values):
        for replica in range(replicas):
            row = i * replicas + replica
            metadata['ensembles'].append(dict(K_index=i, replica=replica, K=k, test=i % 3 == 1,
                seed=pipeline.derived_seed(seed, 1, row), global_flip_seed=pipeline.derived_seed(seed, 2, row)))
    return metadata, ensembles


@pytest.mark.parametrize('column_labels', [False, True])
def test_larger_cache_validation_uses_516_252_ensembles_without_allocating_spins(tmp_path, monkeypatch, column_labels):
    cache = tmp_path / 'cache.pt'
    metadata, ensembles = production_metadata()
    if column_labels:
        for ensemble in ensembles:
            ensemble.couplings = ensemble.couplings[:, None]
    cache.with_suffix('.json').write_text(json.dumps(metadata))
    monkeypatch.setattr(pipeline, 'file_fingerprint', lambda path: 'digest')
    monkeypatch.setattr(pipeline, 'load_ensembles', lambda path: tuple(ensembles))
    report = pipeline.check_data(cache)
    assert report['train_shape'] == [516, 128, 36, 36, 1]
    assert report['test_shape'] == [252, 128, 36, 36, 1]
    assert report['train_K_count'] == 43 and report['test_K_count'] == 21
    assert report['independent_seed_count'] == 768
    # The historical three-arm pilot continues to reject this different cache.
    from scripts.run_scalar_lr_sweep import check_data as original_check
    with pytest.raises(ValueError, match='preregistered production sampling'):
        original_check(cache)


@pytest.mark.parametrize('mutation', ['digest', 'seed', 'replicas', 'grid', 'chain_seed', 'labels', 'dtype', 'shape'])
def test_data_validation_rejects_changed_contract(tmp_path, monkeypatch, mutation):
    metadata, ensembles = production_metadata()
    if mutation == 'digest':
        metadata['data_sha256'] = 'other'
    elif mutation == 'seed':
        metadata['sampling_options']['seed'] = 92
    elif mutation == 'replicas':
        metadata['sampling_options']['replicas'] = 3
    elif mutation == 'grid':
        metadata['couplings'][1] += .0001
    elif mutation == 'chain_seed':
        metadata['ensembles'][1]['seed'] = metadata['ensembles'][0]['seed']
    elif mutation == 'labels':
        ensembles[0].couplings[0] = ensembles[1].couplings[0]
    elif mutation == 'dtype':
        ensembles[0].configurations.dtype = torch.float32
    elif mutation == 'shape':
        ensembles[0].configurations.shape = (129, 128, 36, 36, 1)
    cache = tmp_path / 'cache.pt'
    cache.with_suffix('.json').write_text(json.dumps(metadata))
    monkeypatch.setattr(pipeline, 'file_fingerprint', lambda path: 'digest')
    monkeypatch.setattr(pipeline, 'load_ensembles', lambda path: tuple(ensembles))
    with pytest.raises(ValueError):
        pipeline.check_data(cache)


def write_training_artifacts(tmp_path, *, actual=12000):
    plan = pipeline.build_plan(arguments(tmp_path))
    config_path, run = Path(plan['config_path']), Path(plan['run'])
    config_path.parent.mkdir()
    config_path.write_text(json.dumps(plan['config']))
    run.mkdir()
    manifest = dict(config=deepcopy(plan['config']), config_source=str(config_path), data_sha256='digest',
        source=None, data_driven_initialization=None, training_driver='rsmi_separate_qcr',
        precision='torch.float64', estimator='histogram', environment_mode='full_shell', buffer=1,
        independent_c=True, labels_used_in_training=False, rsmi_weight=2.,
        execution_optimizations=dict(selected_empirical_mi=True,
            mi_statistics_cache={split: dict(hit=True) for split in ('train', 'test')}))
    (run / 'manifest.json').write_text(json.dumps(manifest))
    (run / 'summary.json').write_text(json.dumps(dict(actual_epochs=actual, requested_epochs=12000,
                                                    stopped_by_wall_limit=False,
                                                    stopped_by_training_limit=actual < 12000)))
    return run, config_path, manifest


def test_capped_endpoint_reports_actual_exposure_without_claiming_completion(tmp_path):
    run, config, _ = write_training_artifacts(tmp_path, actual=4321)
    report = pipeline.check_training_manifest(run, config, 'digest')
    assert report['actual_epochs'] == 4321
    assert report['requested_epochs'] == 12000
    assert report['completed_requested_updates'] is False
    assert report['stopped_by_wall_limit'] is False
    assert report['stopped_by_training_limit'] is True


@pytest.mark.parametrize('mutation', ['fingerprint', 'config', 'config_source', 'mi_hit', 'initialization', 'labels'])
def test_training_manifest_requires_owned_config_data_and_warm_cache(tmp_path, mutation):
    run, config, manifest = write_training_artifacts(tmp_path)
    if mutation == 'fingerprint':
        manifest['data_sha256'] = 'different'
    elif mutation == 'config':
        manifest['config']['model']['hidden'] = 48
    elif mutation == 'config_source':
        manifest['config_source'] = str(tmp_path / 'unowned.json')
    elif mutation == 'mi_hit':
        manifest['execution_optimizations']['mi_statistics_cache']['test']['hit'] = False
    elif mutation == 'initialization':
        manifest['resumed_from'] = 'checkpoint.pt'
    elif mutation == 'labels':
        manifest['labels_used_in_training'] = True
    (run / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        pipeline.check_training_manifest(run, config, 'digest')


@pytest.mark.parametrize('missing', [False, True])
def test_cuda_manifest_requires_selected_empirical_mi(tmp_path, missing):
    run, config, manifest = write_training_artifacts(tmp_path)
    if missing:
        manifest['execution_optimizations'].pop('selected_empirical_mi')
    else:
        manifest['execution_optimizations']['selected_empirical_mi'] = False
    (run / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='selected-ensemble empirical MI'):
        pipeline.check_training_manifest(run, config, 'digest')


def test_manifest_failure_stops_reports(tmp_path, monkeypatch):
    cache, output, calls = fake_workflow(tmp_path, monkeypatch)

    def reject(*a, **kw):
        raise ValueError('Training manifest differs')

    monkeypatch.setattr(pipeline, 'check_training_manifest', reject)
    with pytest.raises(ValueError, match='manifest differs'):
        pipeline.main(['--cache', str(cache), '--output', str(output)])
    assert calls[-1][0] == 'training'
    assert json.loads((output / 'job.json').read_text())['status'] == 'failed'
