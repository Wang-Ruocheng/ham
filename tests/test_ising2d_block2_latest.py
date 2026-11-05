"""Block-size-only plan, immutable source, explicit GPU and final training receipts."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from recurrent_rg.learning_rate import LearningRatePolicy
from recurrent_rg.config import RunConfig
from scripts import run_ising2d_block2_latest as pipeline
from scripts.run_scalar_lr_sweep import make_configs


def source_fixture(tmp_path):
    source = tmp_path / 'historical' / 'run'
    source.mkdir(parents=True)
    cfg = make_configs(tmp_path / 'L108.pt')['multistep']
    manifest = dict(config=cfg, data_sha256=pipeline.DATA_SHA)
    summary = dict(config=cfg, actual_epochs=12000, requested_epochs=12000,
                   stopped_by_training_limit=False, stopped_by_wall_limit=False)
    (source / 'manifest.json').write_text(json.dumps(manifest))
    (source / 'summary.json').write_text(json.dumps(summary))
    (source / 'full_report').mkdir()
    (source / 'full_report' / 'encodings.npz').write_bytes(b'preserved encodings')
    return source, cfg


def test_only_block_factor_changes_and_all_new_figures_are_wrapped(tmp_path):
    source, cfg = source_fixture(tmp_path)
    before = {path: path.read_bytes() for path in source.rglob('*') if path.is_file()}
    plan = pipeline.build_plan(source, tmp_path / 'job')
    assert pipeline.differences(cfg, plan['config']) == ['model.block_factor']
    assert plan['config']['model']['block_factor'] == 2
    assert plan['lattice_sizes'] == [108, 54, 27]
    assert plan['control_lattice_sizes'] == [108, 36, 12]
    assert plan['config']['data']['cache'] == cfg['data']['cache']
    assert plan['protocol']['wall_cap'] is None
    assert plan['protocol']['automatic_retry'] is False
    assert plan['protocol']['resume'] is False
    assert [row['stage'] for row in plan['commands']] == [
        'mi_preprocessing', 'training', 'phase_report', 'loss_report', 'exponent_report', 'local_response']
    for row in plan['commands']:
        command = row['command']
        assert not any(flag in command for flag in ('--allow-nondefault-recipe', '--allow-legacy-training',
            '--resume-checkpoint', '--source', '--max-wall-seconds', '--max-training-seconds'))
        assert 'prepare_ising2d_lr_data' not in ' '.join(command)
    assert plan['commands'][0]['command'][3] == 'scripts.prepare_qcr_mi_cache'
    assert all(row['command'][3] == 'scripts.report_ising_zh' for row in plan['commands'][1:])
    training = plan['commands'][1]['command']
    assert training[training.index('--module') + 1] == 'experiments.rsmi_separate_qcr'
    assert training[training.index('--device') + 1] == 'cuda:0'
    assert training[training.index('--mi-cache-dir') + 1] == str(tmp_path / 'job' / 'mi_cache')
    assert all(path.read_bytes() == content for path, content in before.items())
    assert not (tmp_path / 'job').exists()


def test_plan_never_loads_data_or_checkpoint_and_does_not_write(tmp_path, monkeypatch, capsys):
    source, _ = source_fixture(tmp_path)
    monkeypatch.setattr(pipeline, 'file_fingerprint', lambda *a: pytest.fail('No hashes in plan'))
    monkeypatch.setattr(pipeline, 'check_data', lambda *a, **k: pytest.fail('No data in plan'))
    monkeypatch.setattr(pipeline.torch, 'load', lambda *a, **k: pytest.fail('No checkpoint in plan'))
    monkeypatch.setattr(pipeline.subprocess, 'check_output', lambda *a, **k: pytest.fail('No subprocess in plan'))
    output = tmp_path / 'job'
    result = pipeline.main(['--source-run', str(source), '--output', str(output), '--plan'])
    assert json.loads(capsys.readouterr().out) == result
    assert not output.exists()


@pytest.mark.parametrize('key,value', [('data_sha256', 'wrong'), ('model.block_factor', 2),
    ('training.max_rg_steps', 3), ('evaluation.max_rg_steps', 3), ('training.seed', 92),
    ('training.epochs', 11999), ('model.kernel_initialization', 'majority')])
def test_invalid_source_rejected_without_output(tmp_path, key, value):
    source, _ = source_fixture(tmp_path)
    manifest = json.loads((source / 'manifest.json').read_text())
    if '.' in key:
        section, field = key.split('.')
        manifest['config'][section][field] = value
    else:
        manifest[key] = value
    (source / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        pipeline.build_plan(source, tmp_path / 'job')
    assert not (tmp_path / 'job').exists()


def test_incomplete_source_and_overwrite_or_nested_output_rejected(tmp_path):
    source, _ = source_fixture(tmp_path)
    for output in (source, source / 'new', source.parent):
        with pytest.raises(ValueError, match='separate'):
            pipeline.build_plan(source, output)
    output = tmp_path / 'job'
    output.mkdir()
    with pytest.raises(FileExistsError, match='overwrite'):
        pipeline.main(['--source-run', str(source), '--output', str(output)])
    summary = json.loads((source / 'summary.json').read_text())
    summary['actual_epochs'] = 11999
    (source / 'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='12000'):
        pipeline.build_plan(source, tmp_path / 'other')


def test_control_fingerprints_never_read_checkpoint(tmp_path):
    source, _ = source_fixture(tmp_path)
    receipt = pipeline.control_receipt(source)
    assert set(Path(name).relative_to(source).as_posix() for name in receipt['files']) == set(pipeline.CONTROL_FILES)
    assert 'checkpoint_last.pt' not in ' '.join(receipt['files'])


def training_fixture(tmp_path, *, selected=False):
    source, _ = source_fixture(tmp_path)
    plan = pipeline.build_plan(source, tmp_path / 'job')
    run = Path(plan['run'])
    run.mkdir(parents=True)
    config_path = Path(plan['config_path'])
    config_path.parent.mkdir()
    config_path.write_text(json.dumps(plan['config']))
    records = {split: dict(hit=True, path=str(tmp_path / f'{split}.pt'), key=split, cache_version=1)
               for split in ('train', 'test')}
    manifest = dict(config=plan['config'], data_sha256=pipeline.DATA_SHA,
        config_source=str(config_path), training_driver='rsmi_separate_qcr', precision='torch.float64',
        estimator='histogram', environment_mode='full_shell', buffer=1, independent_c=True,
        labels_used_in_training=False, rsmi_weight=2., execution_device='cuda:0', source=None,
        execution_optimizations=dict(selected_empirical_mi=selected, mi_statistics_cache=records))
    policy = LearningRatePolicy(RunConfig.from_dict(plan['config']).training)
    epochs = [1, *range(200, 12001, 200)]
    summary = dict(config=plan['config'], actual_epochs=12000, requested_epochs=12000,
        stopped_by_training_limit=False, stopped_by_wall_limit=False, training_limit_seconds=None,
        lr_scheduler_state=policy.checkpoint_state(12000), timing=dict(device='cuda:0'),
        history=[dict(epoch=epoch, learning_rate=policy.rate_for_update(epoch)) for epoch in epochs])
    for name, value in [('manifest', manifest), ('summary', summary)]:
        (run / f'{name}.json').write_text(json.dumps(value))
    return plan, dict(splits={split: dict(cache=row) for split, row in records.items()}), manifest, summary


@pytest.mark.parametrize('selected', [False, True])
def test_training_receipt_accepts_both_valid_empirical_mi_execution_routes(tmp_path, selected):
    plan, prep, _, _ = training_fixture(tmp_path, selected=selected)
    receipt = pipeline.check_training(plan, prep)
    assert receipt['actual_epochs'] == receipt['new_updates'] == 12000
    assert receipt['source'] is None and receipt['device'] == 'cuda:0'
    assert receipt['selected_empirical_mi'] is selected


@pytest.mark.parametrize('change', ['epochs', 'lr', 'device', 'resume', 'source', 'mi_hit', 'dtype'])
def test_training_receipt_rejects_wrong_actual_steps_lr_device_or_initialization(tmp_path, change):
    plan, prep, manifest, summary = training_fixture(tmp_path)
    if change == 'epochs': summary['actual_epochs'] = 11000
    if change == 'lr': summary['history'][-1]['learning_rate'] = .001
    if change == 'device': manifest['execution_device'] = 'cuda:1'
    if change == 'resume': manifest['resumed_from'] = '/old/checkpoint_last.pt'
    if change == 'source': manifest['source'] = '/old/run'
    if change == 'mi_hit': manifest['execution_optimizations']['mi_statistics_cache']['train']['hit'] = False
    if change == 'dtype': manifest['precision'] = 'torch.float32'
    for name, value in [('manifest', manifest), ('summary', summary)]:
        (Path(plan['run']) / f'{name}.json').write_text(json.dumps(value))
    with pytest.raises(ValueError):
        pipeline.check_training(plan, prep)


def test_gpu_query_uses_selected_index_for_memory_and_processes(monkeypatch):
    calls = []
    def query(command, **kwargs):
        calls.append(command)
        return 'GPU-id, Test GPU, 16000\n' if '--query-gpu=uuid,name,memory.free' in command else ''
    monkeypatch.setattr(pipeline.subprocess, 'check_output', query)
    snapshot = pipeline.gpu_snapshot(2)
    assert snapshot['index'] == 2 and snapshot['free_mib'] == 16000
    assert all(command[command.index('-i') + 1] == '2' for command in calls)


def test_gpu_requires_two_consecutive_unused_checks_sixty_seconds_apart(tmp_path, monkeypatch):
    sleeps, indexes = [], []
    snapshots = iter([([8], 16000), ([], 16000), ([], 12000), ([], 16000), ([], 16000)])
    def snapshot(gpu):
        indexes.append(gpu)
        pids, memory = next(snapshots)
        return dict(index=gpu, uuid='id', name='GPU', compute_pids=pids, free_mib=memory)
    monkeypatch.setattr(pipeline, 'gpu_snapshot', snapshot)
    monkeypatch.setattr(pipeline.time, 'sleep', sleeps.append)
    status = {}
    pipeline.wait_gpu(tmp_path, status, 0)
    assert indexes == [0] * 5 and sleeps == [60] * 4
    assert status['status'] == 'running'
    assert [row['consecutive_ready'] for row in status['gpu_checks']] == [0, 1, 0, 1, 2]
    assert json.loads((tmp_path / 'job.json').read_text())['execution_gpu']['index'] == 0
