"""Fixed supervised continuation stages; all process and GPU actions are mocked."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.rsmi_separate_qcr import add_supervision_contract, build_parser
from recurrent_rg.config import RunConfig
from recurrent_rg.learning_rate import LearningRatePolicy
from scripts import run_ising2d_k_supervised_continuation as pipeline
from scripts import report_ising_zh as zh
from scripts.run_ising2d_block2_latest import differences
from scripts.run_scalar_lr_sweep import make_configs


def dump(path, value):
    path.write_text(json.dumps(value))


def fixture_runs(tmp_path):
    source = tmp_path / 'pilot' / 'run'
    control = tmp_path / 'unsupervised' / 'run'
    source.mkdir(parents=True)
    control.mkdir(parents=True)
    cfg = make_configs(tmp_path / 'L108.pt')['multistep']
    cfg['model']['block_factor'] = 2
    cfg['training']['epochs'] = 3000
    caches = {}
    mi = control.parent / 'mi_cache'
    mi.mkdir()
    for split, key in pipeline.MI_KEYS.items():
        path = mi / f'mi-v1-{key}.pt'
        path.write_bytes(b'plan-only cache fixture')
        caches[split] = dict(hit=True, key=key, path=str(path), cache_version=1)
    manifest = dict(config=cfg, data_sha256=pipeline.DATA_SHA,
        labels_used_in_training=True, independent_c=True, execution_device='cuda:0',
        precision='torch.float64', training_driver='rsmi_separate_qcr',
        estimator='histogram', environment_mode='full_shell', buffer=1, rsmi_weight=2.,
        source=None, note='No critic or coupling labels.',
        gradient_QC='Q: representation losses only; C: original',
        execution_optimizations=dict(mi_statistics_cache=caches))
    add_supervision_contract(manifest, 10.)
    manifest['q_supervision']['coupling_names'] = ['K']
    policy = LearningRatePolicy(RunConfig.from_dict(cfg).training)
    epochs = [1, *range(200, 2801, 200), 2870]
    history = [dict(epoch=epoch, learning_rate=policy.rate_for_update(epoch)) for epoch in epochs]
    summary = dict(config=cfg, actual_epochs=2870, requested_epochs=3000,
        stopped_by_training_limit=True, training_limit_seconds=900.,
        stopped_by_wall_limit=False, history=history,
        lr_scheduler_state=policy.checkpoint_state(2870))
    dump(source / 'manifest.json', manifest)
    dump(source / 'summary.json', summary)
    (source / 'checkpoint_last.pt').write_bytes(b'preserved Adam and RNG checkpoint placeholder')
    dump(source.parent / 'job.json', dict(status='complete', workflow_complete=True))
    reference = source.parent / 'reports' / 'supervised_final'
    reference.mkdir(parents=True)
    dump(reference / 'critical_exponents.json', dict(checkpoint_epoch=2870, roots=[]))
    control_cfg = deepcopy(cfg)
    control_cfg['training']['epochs'] = 12000
    dump(control / 'manifest.json', dict(config=control_cfg,
        data_sha256=pipeline.DATA_SHA, labels_used_in_training=False))
    dump(control / 'summary.json', dict(config=control_cfg, actual_epochs=12000,
        requested_epochs=12000, stopped_by_training_limit=False, stopped_by_wall_limit=False))
    dump(control.parent / 'mi_preprocessing.json', dict(status='complete',
        data_sha256=pipeline.DATA_SHA, splits={key: dict(cache=row) for key, row in caches.items()}))
    return source, control, cfg, manifest, summary


def test_plan_preserves_recipe_changes_only_cumulative_epochs_and_chains_exact_resumes(tmp_path):
    source, control, cfg, _, _ = fixture_runs(tmp_path)
    plan = pipeline.build_plan(source, control, tmp_path / 'continued')
    previous = source / 'checkpoint_last.pt'
    assert [row['endpoint'] for row in plan['stages']] == [4000, 6000, 9000, 12000]
    for row in plan['stages']:
        assert differences(cfg, row['config']) == ['training.epochs']
        assert row['config']['training']['epochs'] == row['endpoint']
        assert row['resume_checkpoint'] == str(previous)
        command = row['training_command']
        assert command[3:6] == ['scripts.report_ising_zh', '--module', 'experiments.rsmi_separate_qcr']
        args = build_parser().parse_args(command[6:])
        assert args.resume_checkpoint == previous and args.source is None
        assert args.q_coupling_weight == 10. and args.device == 'cuda:0'
        assert args.skip_final_report and args.independent_c
        assert args.max_training_seconds is None and args.max_wall_seconds is None
        assert not args.allow_nondefault_recipe and not args.mi_initialize
        assert args.mi_cache_dir == control.parent / 'mi_cache'
        assert args.config == Path(row['config_path'])
        report = row['report_command']
        assert report[3:6] == ['scripts.report_ising_zh', '--module', 'scripts.report_k_supervised_continuation']
        assert report[report.index('--checkpoint') + 1] == str(row['endpoint'])
        assert report[report.index('--trend-root') + 1] == str(tmp_path / 'continued' / 'reports')
        assert report[report.index('--reference-report') + 1] == plan['reference_report']
        previous = Path(row['run']) / f'training_step_{row["endpoint"]}.pt'
    assert cfg['training']['epochs'] == 3000
    assert plan['protocol']['start_updates'] == 2870
    assert plan['protocol']['additional_updates'] == 9130
    assert plan['protocol']['wall_cap'] is None and not plan['protocol']['automatic_retry']
    policy = LearningRatePolicy(RunConfig.from_dict(plan['stages'][-1]['config']).training)
    assert [policy.rate_for_update(u) for u in (4000, 6000, 6001, 9000, 9001, 12000)] == [
        .001, .001, .0003, .0003, .001 * .3**2, .001 * .3**2]
    assert 'scripts.report_k_supervised_continuation' in zh.SUPPORTED_MODULES


def test_plan_is_readonly_and_never_loads_models_or_launches(tmp_path, monkeypatch, capsys):
    source, control, _, _, _ = fixture_runs(tmp_path)
    before = {path: path.read_bytes() for root in (source.parent, control.parent)
              for path in root.rglob('*') if path.is_file()}
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('No checkpoint loading in plan'))
    monkeypatch.setattr(pipeline, 'sha', lambda *a: pytest.fail('No checkpoint hashing in plan'))
    monkeypatch.setattr(pipeline.subprocess, 'Popen', lambda *a, **k: pytest.fail('No launch in plan'))
    monkeypatch.setattr(pipeline.subprocess, 'check_output', lambda *a, **k: pytest.fail('No GPU query in plan'))
    output = tmp_path / 'continued'
    plan = pipeline.main(['--source-run', str(source), '--control-run', str(control),
                          '--output', str(output), '--plan'])
    assert json.loads(capsys.readouterr().out) == plan
    assert not output.exists()
    assert all(path.read_bytes() == value for path, value in before.items())


@pytest.mark.parametrize('change', ['weight', 'scale', 'split', 'labels', 'epoch', 'device',
    'precision', 'job', 'checkpoint', 'reference', 'cache_key', 'cache_file'])
def test_invalid_source_or_missing_existing_cache_fails_before_output(tmp_path, change):
    source, control, _, manifest, summary = fixture_runs(tmp_path)
    if change == 'weight': manifest['q_supervision']['weight'] = 1.
    if change == 'scale': manifest['q_supervision']['scale'] = 1
    if change == 'split': manifest['q_supervision']['fitting_split'] = 'TEST'
    if change == 'labels': manifest['labels_used_in_training'] = False
    if change == 'epoch': summary['actual_epochs'] = 3000
    if change == 'device': manifest['execution_device'] = 'cuda:1'
    if change == 'precision': manifest['precision'] = 'torch.float32'
    if change == 'job': dump(source.parent / 'job.json', dict(status='running', workflow_complete=False))
    if change == 'checkpoint': (source / 'checkpoint_last.pt').unlink()
    if change == 'reference': (source.parent / 'reports/supervised_final/critical_exponents.json').unlink()
    if change == 'cache_key':
        path = control.parent / 'mi_preprocessing.json'
        receipt = json.loads(path.read_text()); receipt['splits']['train']['cache']['key'] = 'wrong'
        dump(path, receipt)
    if change == 'cache_file':
        (control.parent / 'mi_cache' / f'mi-v1-{pipeline.MI_KEYS["test"]}.pt').unlink()
    dump(source / 'manifest.json', manifest); dump(source / 'summary.json', summary)
    output = tmp_path / 'continued'
    with pytest.raises(ValueError):
        pipeline.build_plan(source, control, output)
    assert not output.exists()


@pytest.mark.parametrize('field,value', [
    ('model.hidden', 48), ('model.affine_reconstruction', False),
    ('model.kernel_layout', 'overlapping'), ('loss.weights.separation', 20.),
    ('loss.rollout_weight', 2.), ('training.system_batch', 8),
    ('training.lr_gamma', .5), ('evaluation.samples', 64),
    ('data.spin_flip_augment', True),
])
def test_hidden_source_configuration_changes_cannot_pass_as_latest_control(tmp_path, field, value):
    source, control, _, manifest, summary = fixture_runs(tmp_path)
    for record in (manifest, summary):
        # Source manifest/summary agreeing with each other is insufficient:
        # they must still match the complete frozen unsupervised configuration.
        config = deepcopy(record['config'])
        parent = config
        keys = field.split('.')
        for key in keys[:-1]:
            parent = parent[key]
        parent[keys[-1]] = value
        record['config'] = config
    dump(source / 'manifest.json', manifest)
    dump(source / 'summary.json', summary)
    output = tmp_path / 'continued'
    with pytest.raises(ValueError, match='entire completed latest b2 control'):
        pipeline.build_plan(source, control, output)
    assert not output.exists()


@pytest.mark.parametrize('problem', ['epoch', 'fingerprint', 'labels'])
def test_control_must_be_completed_unlabelled_original_data_run(tmp_path, problem):
    source, control, _, _, _ = fixture_runs(tmp_path)
    manifest = json.loads((control / 'manifest.json').read_text())
    summary = json.loads((control / 'summary.json').read_text())
    if problem == 'epoch': summary['actual_epochs'] = 11000
    if problem == 'fingerprint': manifest['data_sha256'] = '0' * 64
    if problem == 'labels': manifest['labels_used_in_training'] = True
    dump(control / 'manifest.json', manifest)
    dump(control / 'summary.json', summary)
    with pytest.raises(ValueError, match='entire completed latest b2 control'):
        pipeline.build_plan(source, control, tmp_path / 'continued')
    assert not (tmp_path / 'continued').exists()


def test_historical_output_existing_output_and_cuda_remapping_are_protected(tmp_path, monkeypatch):
    source, control, _, _, _ = fixture_runs(tmp_path)
    for output in (source, source / 'child', source.parent, control, control / 'child', control.parent):
        with pytest.raises(ValueError, match='separate'):
            pipeline.build_plan(source, control, output)
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    monkeypatch.setattr(pipeline, 'validate_source_payload', lambda plan: 'prevalidated-source')
    output = tmp_path / 'continued'; output.mkdir()
    sentinel = output / 'keep.txt'; sentinel.write_text('user data')
    with pytest.raises(FileExistsError):
        pipeline.main(['--source-run', str(source), '--control-run', str(control), '--output', str(output)])
    assert sentinel.read_text() == 'user data'
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1')
    with pytest.raises(ValueError, match='CUDA_VISIBLE_DEVICES'):
        pipeline.main(['--source-run', str(source), '--control-run', str(control),
                       '--output', str(tmp_path / 'other')])
    assert not (tmp_path / 'other').exists()


def test_wait_gpu_requires_two_idle_checks_and_never_changes_other_processes(tmp_path, monkeypatch):
    outputs, sleeps, calls = iter(['981\n', '', '782\n', '', '']), [], []
    statuses = []
    def query(command, **kwargs):
        calls.append(command)
        assert command == ['nvidia-smi', '-i', '0', '--query-compute-apps=pid',
                           '--format=csv,noheader,nounits']
        return next(outputs)
    monkeypatch.setattr(pipeline.subprocess, 'check_output', query)
    monkeypatch.setattr(pipeline.subprocess, 'Popen', lambda *a, **k: pytest.fail('No process launch while waiting'))
    monkeypatch.setattr(pipeline.time, 'sleep', sleeps.append)
    monkeypatch.setattr(pipeline, 'write_job', lambda out, job: statuses.append(deepcopy(job['current'])))
    job = {}
    pipeline.wait_gpu(tmp_path, job, 'report_4000')
    assert len(calls) == 5 and sleeps == [30, 10, 30, 10]
    assert [row['consecutive_idle'] for row in statuses] == [0, 1, 0, 1, 2]
    assert all(row['next_stage'] == 'report_4000' for row in statuses)
    assert statuses[0]['busy_pids'] == ['981']


def test_failed_stage_records_error_without_retry(tmp_path, monkeypatch):
    (tmp_path / 'logs').mkdir()
    calls = []
    monkeypatch.setattr(pipeline, 'wait_gpu', lambda *a: None)
    monkeypatch.setattr(pipeline, 'write_job', lambda *a: None)
    def spawn(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(pid=18, wait=lambda: 7)
    monkeypatch.setattr(pipeline.subprocess, 'Popen', spawn)
    job = dict(stages=[], current=None)
    with pytest.raises(RuntimeError, match='no automatic retry'):
        pipeline.stage(tmp_path, job, 'training_4000', ['mock-training'], '/owned/run', 4000)
    assert calls == [['mock-training']]
    assert job['current'] is None and job['stages'][0]['returncode'] == 7


def finished_segment(tmp_path):
    source, control, _, source_manifest, source_summary = fixture_runs(tmp_path)
    row = pipeline.build_plan(source, control, tmp_path / 'continued')['stages'][0]
    run = Path(row['run']); run.mkdir(parents=True)
    manifest = deepcopy(source_manifest)
    manifest.update(config=deepcopy(row['config']), resumed_from=row['resume_checkpoint'])
    policy = LearningRatePolicy(RunConfig.from_dict(row['config']).training)
    history = deepcopy(source_summary['history']) + [dict(epoch=epoch,
        learning_rate=policy.rate_for_update(epoch)) for epoch in range(3000, 4001, 200)]
    summary = dict(config=deepcopy(row['config']), actual_epochs=4000, requested_epochs=4000,
        stopped_by_training_limit=False, stopped_by_wall_limit=False, training_limit_seconds=None,
        history=history, lr_scheduler_state=policy.checkpoint_state(4000))
    dump(run / 'manifest.json', manifest); dump(run / 'summary.json', summary)
    (run / 'training_step_4000.pt').write_bytes(b'immutable endpoint fixture')
    return row, source, manifest, summary


def test_training_receipt_retains_inherited_cap_endpoint_history(tmp_path):
    row, source, _, summary = finished_segment(tmp_path)
    receipt = pipeline.check_training(row, source)
    assert receipt['actual_epochs'] == 4000 and receipt['learning_rate'] == .001
    assert 2870 in [item['epoch'] for item in summary['history']]
    assert summary['history'][-1]['epoch'] == 4000


@pytest.mark.parametrize('problem', ['history', 'cap', 'endpoint', 'resume', 'config', 'manifest_config', 'mi_miss',
                                    'missing_checkpoint', 'empty_mi', 'scheduler', 'learning_rate'])
def test_training_receipt_rejects_history_lr_cache_and_resume_changes(tmp_path, problem):
    row, source, manifest, summary = finished_segment(tmp_path)
    if problem == 'history': summary['history'][0]['learning_rate'] = .02
    if problem == 'cap': summary['stopped_by_training_limit'] = True
    if problem == 'endpoint': summary['actual_epochs'] = 3999
    if problem == 'resume': manifest['resumed_from'] = '/different/checkpoint.pt'
    if problem == 'config': summary['config']['model']['hidden'] = 97
    if problem == 'manifest_config': manifest['config']['model']['hidden'] = 97
    if problem == 'mi_miss': manifest['execution_optimizations']['mi_statistics_cache']['test']['hit'] = False
    if problem == 'missing_checkpoint': (Path(row['run']) / 'training_step_4000.pt').unlink()
    if problem == 'empty_mi': manifest['execution_optimizations']['mi_statistics_cache'] = {}
    if problem == 'scheduler': summary['lr_scheduler_state']['completed_updates'] = 3999
    if problem == 'learning_rate': summary['history'][-1]['learning_rate'] = .0003
    dump(Path(row['run']) / 'manifest.json', manifest)
    dump(Path(row['run']) / 'summary.json', summary)
    with pytest.raises(ValueError):
        pipeline.check_training(row, source)


@pytest.mark.parametrize('spawn_fails', [False, True])
def test_detach_reserves_once_before_spawn_and_rejects_ambiguous_retry(tmp_path, monkeypatch, capsys, spawn_fails):
    source, control, _, _, _ = fixture_runs(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    for name in ('experiments/rsmi_separate_qcr.py', 'scripts/report_ising_zh.py',
                 'scripts/report_k_supervised_pilot.py', 'scripts/report_k_supervised_continuation.py',
                 'scripts/run_ising2d_k_supervised_continuation.py'):
        path = tmp_path / name; path.parent.mkdir(exist_ok=True)
        path.write_bytes(b'code fingerprint fixture')
    output = tmp_path / 'continued'
    receipt = tmp_path / 'logs' / 'continued.launch.json'
    calls = []
    def spawn(command, **kwargs):
        calls.append(command)
        assert json.loads(receipt.read_text())['status'] == 'reserved'
        assert kwargs['start_new_session'] and '--detach' not in command
        assert not output.exists()
        if spawn_fails: raise OSError('ambiguous launch fixture')
        return SimpleNamespace(pid=320)
    monkeypatch.setattr(pipeline.subprocess, 'Popen', spawn)
    args = ['--source-run', str(source), '--control-run', str(control), '--output', str(output), '--detach']
    if spawn_fails:
        with pytest.raises(OSError, match='ambiguous'):
            pipeline.main(args)
    else:
        record = pipeline.main(args)
        assert json.loads(capsys.readouterr().out) == record == json.loads(receipt.read_text())
        assert record['parent_pid'] == 320 and len(record['code_sha256']) == 5
    with pytest.raises(FileExistsError):
        pipeline.main(args)
    assert len(calls) == 1 and not output.exists()


def source_payload_fixture(tmp_path):
    source, control, _, manifest, summary = fixture_runs(tmp_path)
    plan = pipeline.build_plan(source, control, tmp_path / 'continued')
    metadata = deepcopy(manifest)
    # Actual torch checkpoints retain the dataclass tuple; JSON stores a list.
    metadata['config']['training']['lr_milestones'] = (6000, 9000)
    payload = dict(epoch=2870, metadata=metadata, model={}, optimizer=dict(state={}, param_groups=[]),
        history=deepcopy(summary['history']), lr_scheduler_state=summary['lr_scheduler_state'],
        torch_rng_state=torch.get_rng_state(), cuda_rng_state=torch.tensor([1], dtype=torch.uint8),
        cuda_device_index=0, batch_generator_state=torch.tensor([2], dtype=torch.uint8),
        rsmi_generator_state=torch.tensor([3], dtype=torch.uint8))
    torch.save(payload, source / 'checkpoint_last.pt')
    return plan, payload


def test_source_payload_accepts_lossless_json_tuple_difference_and_preserves_checkpoint(tmp_path):
    plan, _ = source_payload_fixture(tmp_path)
    path = Path(plan['initial_checkpoint'])
    before = path.read_bytes()
    assert pipeline.validate_source_payload(plan) == pipeline.sha(path)
    assert path.read_bytes() == before
    assert not Path(plan['output']).exists()


@pytest.mark.parametrize('problem', ['epoch', 'metadata', 'device', 'scheduler', 'history', 'missing_rng'])
def test_source_payload_rejects_changed_adam_rng_metadata_or_history(tmp_path, problem):
    plan, payload = source_payload_fixture(tmp_path)
    if problem == 'epoch': payload['epoch'] = 3000
    if problem == 'metadata': payload['metadata']['q_supervision']['weight'] = 1.
    if problem == 'device': payload['cuda_device_index'] = 1
    if problem == 'scheduler': payload['lr_scheduler_state']['completed_updates'] = 2869
    if problem == 'history': payload['history'][-1]['epoch'] = 2869
    if problem == 'missing_rng': payload.pop('rsmi_generator_state')
    torch.save(payload, plan['initial_checkpoint'])
    with pytest.raises(ValueError):
        pipeline.validate_source_payload(plan)
    assert not Path(plan['output']).exists()


def test_source_and_deployed_code_guards_detect_changes(tmp_path):
    path = tmp_path / 'checkpoint.pt'; path.write_bytes(b'original source')
    digest = pipeline.sha(path)
    pipeline.ensure_source(dict(initial_checkpoint=str(path)), digest)
    pipeline.ensure_code({str(path): digest})
    path.write_bytes(b'changed source')
    with pytest.raises(ValueError, match='source checkpoint changed'):
        pipeline.ensure_source(dict(initial_checkpoint=str(path)), digest)
    with pytest.raises(ValueError, match='Deployed source changed'):
        pipeline.ensure_code({str(path): digest})


def test_report_command_forwarded_interface_parses_without_loading_model(tmp_path, monkeypatch):
    from scripts import report_k_supervised_continuation as report
    source, control, _, _, _ = fixture_runs(tmp_path)
    command = pipeline.build_plan(source, control, tmp_path / 'continued')['stages'][0]['report_command']
    class Parsed(Exception):
        pass
    def stop_before_load(*args, **kwargs):
        raise Parsed()
    monkeypatch.setattr(report, 'load_run', stop_before_load)
    with pytest.raises(Parsed):
        report.main(command[6:])
