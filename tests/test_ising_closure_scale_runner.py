"""Execution contracts only: fake immutable bytes, no physical/model loading."""
import copy
import json

import pytest

from scripts import run_ising_closure_scale_ablation as runner


@pytest.fixture(autouse=True)
def fake_new_data_fingerprint(monkeypatch):
    import hashlib
    monkeypatch.setattr(runner, 'SOURCE_DATA_SHA',
        hashlib.sha256(b'new physical data; never tensor-loaded').hexdigest())


def completed_source(tmp_path):
    source = tmp_path / 'completed'
    for folder in ('data', 'mi_cache', 'run_control'):
        (source / folder).mkdir(parents=True)
    (source / 'data/ensembles.pt').write_bytes(b'new physical data; never tensor-loaded')
    data_sha = runner.file_sha(source / 'data/ensembles.pt')
    runner.save(source / 'pairs.json', dict(schema_version=1, data_sha256=data_sha))
    pair_sha = runner.file_sha(source / 'pairs.json')
    runner.save(source / 'dataset_receipt.json', dict(data_sha256=data_sha, pair_sha256=pair_sha,
        train_shape=[112, 64, 108, 108, 1], test_shape=[324, 64, 108, 108, 1],
        old_9GB_dataset_opened=False, warning_count=16))
    config = runner.make_config(source / 'data/ensembles.pt')
    runner.save(source / 'config.json', config)
    cache = {}
    for split in ('train', 'test'):
        key = {'train': 'a', 'test': 'b'}[split] * 64
        path = source / 'mi_cache' / f'mi-v1-{key}.pt'
        path.write_bytes(f'new data-only {split} MI; never tensor-loaded'.encode())
        cache[split] = dict(key=key, path=str(path), cache_version=1, hit=True)
    manifest = dict(config=config, data_sha256=data_sha, labels_used_in_training=False,
        auxiliary_future_parameters=0, execution_optimizations=dict(mi_statistics_cache=cache),
        q_future_physics=dict(mode='paired_control', weight=0., labels_used=False,
                              data_sha256=data_sha, pairs_sha256=pair_sha))
    runner.save(source / 'run_control/manifest.json', manifest)
    runner.save(source / 'run_control/summary.json', dict(actual_epochs=6000, config=config))
    runner.save(source / 'job.json', dict(experiment='ising_future_physics_Q_loss_b3_paired_v1',
        status='complete', workflow_complete=True, current=None, child_still_running=False,
        stages=[dict(name='report', returncode=0, status='complete')]))
    return source


def test_reuse_only_named_new_inputs_and_profiles(tmp_path):
    source = completed_source(tmp_path)
    # Never inspect, link, or initialize from an available source checkpoint.
    (source / 'run_control/checkpoint_last.pt').write_bytes(b'not an initialization')
    receipt = runner.validated_source(source)
    root = tmp_path / 'new'; root.mkdir()
    runner.prepare_inputs(root, source, receipt)
    assert runner.PROFILES == {'uniform': [1., 1.], 'late': [.2, 1.8]}
    uniform, late = (runner.read(root / f'config_{name}.json') for name in runner.PROFILES)
    assert uniform['loss'].pop('closure_scale_weights') == [1., 1.]
    assert late['loss'].pop('closure_scale_weights') == [.2, 1.8]
    assert uniform == late == runner.make_config(root / 'data/ensembles.pt')
    for relative in receipt['input_sha256']:
        assert (root / relative).stat().st_ino == (source / relative).stat().st_ino
    assert not (root / 'run_control').exists() and not (root / 'normalizer.json').exists()
    assert runner.read(root / 'benchmark_config.json')['training']['epochs'] == 40
    assert receipt['reused_model_or_optimizer'] is False
    assert receipt['source_checkpoints_opened'] is False


@pytest.mark.parametrize('mutation', ['running', 'data', 'pairs', 'supervised', 'future', 'recipe', 'mi_symlink'])
def test_reuse_rejects_wrong_or_changed_inputs(tmp_path, mutation):
    source = completed_source(tmp_path)
    if mutation == 'running':
        path = source / 'job.json'; value = runner.read(path); value['status'] = 'running'
    elif mutation in ('data', 'pairs'):
        path = source / ('data/ensembles.pt' if mutation == 'data' else 'pairs.json')
        if mutation == 'data':
            path.write_bytes(b'changed')
        else:
            value = runner.read(path); value['data_sha256'] = 'different'; runner.save(path, value)
    elif mutation in ('supervised', 'future'):
        path = source / 'run_control/manifest.json'; value = runner.read(path)
        if mutation == 'supervised': value['labels_used_in_training'] = True
        else: value['q_future_physics']['weight'] = 1
    elif mutation == 'recipe':
        path = source / 'config.json'; value = runner.read(path); value['model']['block_factor'] = 2
    else:
        cache = next((source / 'mi_cache').glob('*.pt'))
        content = cache.read_bytes(); cache.unlink()
        elsewhere = tmp_path / 'elsewhere'; elsewhere.write_bytes(content); cache.symlink_to(elsewhere)
    if mutation in ('running', 'supervised', 'future', 'recipe'): runner.save(path, value)
    with pytest.raises(ValueError): runner.validated_source(source)


def test_reuse_detects_change_between_validation_and_hardlink(tmp_path):
    source = completed_source(tmp_path)
    receipt = runner.validated_source(source)
    (source / 'data/ensembles.pt').write_bytes(b'changed after source receipt')
    root = tmp_path / 'new'; root.mkdir()
    with pytest.raises(ValueError, match='changed before immutable reuse'):
        runner.prepare_inputs(root, source, receipt)


def test_commands_are_fresh_unsupervised_cuda2_and_have_no_future_loss(tmp_path):
    command = runner.trainer_command(tmp_path, tmp_path / 'run_late', tmp_path / 'config_late.json', limit=1234)
    assert command[1:3] == ['-m', 'experiments.rsmi_separate_qcr']
    assert command[command.index('--device') + 1] == 'cuda:2'
    assert '--q-future-control' in command and '--q-future-pairs' in command
    assert '--q-future-weight' not in command and '--q-future-normalizer' not in command
    for forbidden in ('--source', '--resume-checkpoint', '--q-coupling-weight', '--allow-nondefault-recipe',
                      '--allow-legacy-training', '--mi-initialize'):
        assert forbidden not in command
    assert '--mi-test-cpu' in command and '--skip-final-report' in command


def arm_fixture(root, name, weights, *, actual=6000, initial=None):
    path = root / ('benchmark' if name == 'benchmark' else f'run_{name}')
    path.mkdir()
    initial = initial or dict(Q='a' * 64, C='b' * 64, R='c' * 64)
    scale = dict(weights=weights, normalized_weights=[w / sum(weights) for w in weights],
        target_levels=[1, 2], application=['one_step_commutator', 'microscopic_start_rollout'], labels_used=False)
    contract = dict(closure_scale_weighting=scale, initial_module_sha256=initial)
    runner.save(path / 'manifest.json', dict(**contract, labels_used_in_training=False,
        q_future_physics=dict(mode='paired_control', weight=0.), auxiliary_future_parameters=0,
        execution_optimizations=dict(mi_statistics_cache={s: dict(hit=True) for s in ('train', 'test')})))
    runner.save(path / 'summary.json', dict(contract=contract, actual_epochs=actual, requested_epochs=actual))
    return path


def test_validates_full_6000_pair_and_benchmark_initialization(tmp_path):
    arm_fixture(tmp_path, 'benchmark', [1., 1.], actual=40)
    arm_fixture(tmp_path, 'uniform', [1., 1.])
    arm_fixture(tmp_path, 'late', [.2, 1.8])
    assert runner.validate_arm(tmp_path, 'benchmark', 40)['actual_epochs'] == 40
    assert runner.validate_arm(tmp_path, 'uniform', 6000)['actual_epochs'] == 6000
    assert runner.validate_arm(tmp_path, 'late', 6000)['actual_epochs'] == 6000


def test_partial_work_is_budget_hold_never_silently_accepted(tmp_path):
    arm_fixture(tmp_path, 'uniform', [1., 1.], actual=3000)
    with pytest.raises(runner.execution.BudgetHold, match='固定6000'):
        runner.validate_arm(tmp_path, 'uniform', 6000)


def test_initialization_difference_fails_without_model_loading(tmp_path):
    arm_fixture(tmp_path, 'uniform', [1., 1.])
    arm_fixture(tmp_path, 'late', [.2, 1.8], initial=dict(Q='different', C='b' * 64, R='c' * 64))
    with pytest.raises(RuntimeError, match='initial fingerprints differ'):
        runner.validate_arm(tmp_path, 'late', 6000)


def test_normalized_weight_roundoff_does_not_reject_late_arm(tmp_path):
    path = arm_fixture(tmp_path, 'late', [.2, 1.8])
    manifest, summary = (runner.read(path / name) for name in ('manifest.json', 'summary.json'))
    manifest['closure_scale_weighting']['normalized_weights'] = [.1, .8999999999999999]
    summary['contract']['closure_scale_weighting'] = manifest['closure_scale_weighting']
    runner.save(path / 'manifest.json', manifest); runner.save(path / 'summary.json', summary)
    assert runner.validate_arm(tmp_path, 'late', 6000)['actual_epochs'] == 6000


def test_source_and_config_pins_checked_before_every_stage(tmp_path, monkeypatch):
    code = tmp_path / 'code'; (code / 'scripts').mkdir(parents=True)
    science = code / 'scripts/science.py'; science.write_text('# frozen science')
    root = tmp_path / 'experiment'; root.mkdir()
    runner.save(root / 'pairs.json', {'same': 'new input'})
    runner.save(root / 'config_uniform.json', {'loss': [1, 1]})
    runner.save(root / 'plan.json', dict(source_code_sha256=runner.execution.source_hashes(code),
        configuration_sha256={'config_uniform.json': runner.file_sha(root / 'config_uniform.json')}))
    job = dict(code_snapshot=str(code), input_reuse=dict(input_sha256={'pairs.json': runner.file_sha(root / 'pairs.json')}))
    calls = []
    monkeypatch.setattr(runner.execution, 'run_stage', lambda *a, **k: calls.append(a[2]))
    runner.run_stage(root, job, 'benchmark', ['unused'], gpu=True)
    assert calls == ['benchmark']
    science.write_text('# changed science')
    with pytest.raises(RuntimeError, match='Frozen science'):
        runner.run_stage(root, job, 'train_uniform', ['unused'], gpu=True)
    assert calls == ['benchmark']


def test_public_whitelist_excludes_all_pt_and_unknown_logs(tmp_path):
    (tmp_path / 'report').mkdir(); (tmp_path / 'logs').mkdir()
    for name in ('report/report.json', 'report/evaluation_arrays.npz', 'report/comparison.png',
                 'report/checkpoint.pt', 'logs/private.json', 'job.json'):
        (tmp_path / name).write_bytes(b'fake')
    assert runner.public_files(tmp_path) == ['job.json', 'report/comparison.png',
        'report/evaluation_arrays.npz', 'report/report.json']


def test_existing_output_never_restarts(tmp_path):
    root = tmp_path / 'existing'; root.mkdir()
    runner.save(root / 'job.json', dict(status='failed'))
    with pytest.raises(FileExistsError, match='fresh output'):
        runner.run(root, tmp_path / 'unused')
