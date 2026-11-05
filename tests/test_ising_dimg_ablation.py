"""Dimension-only fresh QCR arms, cache-only execution and immutable receipts."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from recurrent_rg.config import RunConfig
from recurrent_rg.learning_rate import LearningRatePolicy
from scripts import run_ising2d_dimg_ablation as pipeline
from scripts.run_scalar_lr_sweep import make_configs


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def source_fixture(tmp_path, monkeypatch):
    source = tmp_path / 'historical' / 'run'
    cache = tmp_path / 'physical.pt'
    cache.write_bytes(b'physical bytes must never be loaded or rehashed')
    write(cache.with_suffix('.json'), dict(data_sha256=pipeline.DATA_SHA))
    monkeypatch.setattr(pipeline, 'SIDECAR_SHA', pipeline.fingerprint(cache.with_suffix('.json')))
    cfg = make_configs(cache)['multistep']
    cfg['model']['block_factor'] = 2
    records = {}
    for split, key in pipeline.MI_KEYS.items():
        target = source.parent / 'mi_cache' / f'mi-v1-{key}.pt'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b'existing MI bytes')
        records[split] = dict(hit=True, path=str(target), key=key, cache_version=1)
    manifest = dict(config=cfg, data_sha256=pipeline.DATA_SHA, labels_used_in_training=False,
                    source=None, execution_optimizations=dict(mi_statistics_cache=records))
    write(source / 'manifest.json', manifest)
    write(source / 'summary.json', dict(config=cfg, actual_epochs=12000, requested_epochs=12000,
        stopped_by_training_limit=False, stopped_by_wall_limit=False))
    write(source.parent / 'job.json', dict(status='complete', workflow_complete=True,
        training=dict(actual_epochs=12000, manifest_sha256=pipeline.fingerprint(source / 'manifest.json'))))
    write(source.parent / 'mi_preprocessing.json', dict(status='complete', data_sha256=pipeline.DATA_SHA,
        model_initialized=False, optimizer_updates=0, splits={s: dict(cache=r) for s, r in records.items()}))
    return source, cfg


def test_only_theory_dimension_changes_all_arms_fresh_and_no_scalar_report(tmp_path, monkeypatch):
    source, cfg = source_fixture(tmp_path, monkeypatch)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('Plan cannot load tensors'))
    plan = pipeline.build_plan(source, tmp_path / 'new')
    assert [a['theory_dim'] for a in plan['arms']] == [1, 2, 3]
    assert [a['gpu'] for a in plan['arms']] == [0, 2, 3]
    for arm in plan['arms']:
        assert pipeline.differences(cfg, arm['config']) == ([] if arm['theory_dim'] == 1 else ['model.theory_dim'])
        assert arm['config']['model']['block_factor'] == 2
        assert arm['config']['loss']['coordinate_reduction'] == 'sum'
        assert arm['config']['training']['threads'] == 2
        command = arm['commands']['training']
        assert '--cached-training' in command
        assert command[command.index('--device') + 1] == arm['device']
        all_commands = ' '.join(' '.join(row) for row in arm['commands'].values())
        for flag in ('--source ', '--resume-checkpoint', '--allow-nondefault-recipe', '--allow-legacy-training',
                     '--max-wall-seconds', '--max-training-seconds', 'report_scalar', 'prepare_qcr_mi'):
            assert flag not in all_commands
        assert arm['commands']['response'][:3] == ['nice', '-n', '10']
    assert plan['protocol']['automatic_retry'] is False
    assert plan['protocol']['labels_used_in_training'] is False
    assert 'skip-final-report' in plan['protocol']['execution_delta']
    assert all(Path(path).read_bytes() == content for path, content in before.items())
    assert not (tmp_path / 'new').exists()


@pytest.mark.parametrize('mutation', ['label', 'resume', 'frozen', 'block', 'depth', 'seed', 'steps', 'threads', 'lr', 'objective'])
def test_rejects_nonmatching_historical_source(tmp_path, monkeypatch, mutation):
    source, _ = source_fixture(tmp_path, monkeypatch)
    path = source / 'manifest.json'
    row = json.loads(path.read_text())
    if mutation == 'label': row['labels_used_in_training'] = True
    if mutation == 'resume': row['resumed_from'] = 'old.pt'
    if mutation == 'frozen': row['frozen_C'] = True
    if mutation == 'block': row['config']['model']['block_factor'] = 3
    if mutation == 'depth': row['config']['training']['max_rg_steps'] = 3
    if mutation == 'seed': row['config']['training']['seed'] = 92
    if mutation == 'steps': row['config']['training']['epochs'] = 11999
    if mutation == 'threads': row['config']['training']['threads'] = 1
    if mutation == 'lr': row['config']['training']['lr_gamma'] = .1
    if mutation == 'objective': row['config']['loss']['weights']['separation'] = 11.
    write(path, row)
    with pytest.raises(ValueError):
        pipeline.build_plan(source, tmp_path / 'new')
    assert not (tmp_path / 'new').exists()


@pytest.mark.parametrize('gpus', [(0, 0, 2), (0, 1), (0, 2, -1), (True, 2, 3)])
def test_gpu_maps_must_be_three_distinct_indices(tmp_path, monkeypatch, gpus):
    source, _ = source_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='GPU'):
        pipeline.build_plan(source, tmp_path / 'new', gpus)


def test_incomplete_source_missing_cache_or_changed_receipt_rejected(tmp_path, monkeypatch):
    source, _ = source_fixture(tmp_path, monkeypatch)
    job = json.loads((source.parent / 'job.json').read_text())
    job['workflow_complete'] = False
    write(source.parent / 'job.json', job)
    with pytest.raises(ValueError): pipeline.build_plan(source, tmp_path / 'new')
    job['workflow_complete'] = True
    write(source.parent / 'job.json', job)
    cache = source.parent / 'mi_cache' / f"mi-v1-{pipeline.MI_KEYS['train']}.pt"
    cache.unlink()
    with pytest.raises(ValueError, match='普通文件'): pipeline.build_plan(source, tmp_path / 'new')


def test_no_overwrite_or_historical_nested_output(tmp_path, monkeypatch):
    source, _ = source_fixture(tmp_path, monkeypatch)
    for target in (source, source / 'new', source.parent):
        with pytest.raises(ValueError, match='分离'): pipeline.build_plan(source, target)
    output = tmp_path / 'new'
    output.mkdir()
    with pytest.raises(FileExistsError): pipeline.build_plan(source, output)


def test_immutability_checks_small_receipts_and_stat_but_not_physical_sha(tmp_path, monkeypatch):
    source, _ = source_fixture(tmp_path, monkeypatch)
    plan = pipeline.build_plan(source, tmp_path / 'new')
    assert plan['cache'] not in plan['source_sha256']
    pipeline.ensure_unchanged(plan, {}, {})
    Path(plan['cache']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='元数据'): pipeline.ensure_unchanged(plan, {}, {})


def training_fixture(tmp_path, monkeypatch, dim=3):
    source, _ = source_fixture(tmp_path, monkeypatch)
    plan = pipeline.build_plan(source, tmp_path / 'new')
    arm = plan['arms'][dim - 1]
    run = Path(arm['run'])
    write(Path(arm['config_path']), arm['config'])
    records = {split: dict(hit=True, key=key, cache_version=1,
        path=str(Path(plan['mi_cache_dir']) / f'mi-v1-{key}.pt')) for split, key in pipeline.MI_KEYS.items()}
    manifest = dict(config=arm['config'], config_source=arm['config_path'], data_sha256=pipeline.DATA_SHA,
        training_driver='rsmi_separate_qcr', precision='torch.float64', estimator='histogram',
        environment_mode='full_shell', buffer=1, independent_c=True, labels_used_in_training=False,
        rsmi_weight=2., execution_device=arm['device'], source=None,
        execution_optimizations=dict(mi_statistics_cache=records))
    policy = LearningRatePolicy(RunConfig.from_dict(arm['config']).training)
    summary = dict(config=arm['config'], actual_epochs=12000, requested_epochs=12000,
        stopped_by_training_limit=False, stopped_by_wall_limit=False, training_limit_seconds=None,
        final_report_skipped=True, lr_scheduler_state=policy.checkpoint_state(12000),
        history=[dict(epoch=n, learning_rate=policy.rate_for_update(n)) for n in [1, *range(200, 12001, 200)]])
    write(run / 'manifest.json', manifest)
    write(run / 'summary.json', summary)
    write(Path(arm['root']) / 'initialization.json', dict(fresh=True, source_checkpoint=None))
    (run / 'training_step_12000.pt').write_bytes(b'new final checkpoint')
    return plan, arm, manifest, summary


@pytest.mark.parametrize('dim', [1, 2, 3])
def test_final_training_receipt_full_budget_cached_unsupervised_and_skipped_old_report(tmp_path, monkeypatch, dim):
    plan, arm, _, _ = training_fixture(tmp_path, monkeypatch, dim)
    receipt = pipeline.check_training(arm, plan)
    assert receipt['actual_epochs'] == receipt['new_updates'] == 12000
    assert receipt['theory_dim'] == dim and receipt['source'] is None
    assert receipt['labels_used_in_training'] is False


@pytest.mark.parametrize('mutation', ['cached', 'steps', 'lr', 'label', 'skip'])
def test_final_training_receipt_rejects_any_budget_or_recipe_change(tmp_path, monkeypatch, mutation):
    plan, arm, manifest, summary = training_fixture(tmp_path, monkeypatch)
    if mutation == 'cached': manifest['execution_optimizations']['mi_statistics_cache']['train']['hit'] = False
    if mutation == 'steps': summary['actual_epochs'] = 11999
    if mutation == 'lr': summary['history'][-1]['learning_rate'] = .001
    if mutation == 'label': manifest['labels_used_in_training'] = True
    if mutation == 'skip': summary['final_report_skipped'] = False
    write(Path(arm['run']) / 'manifest.json', manifest)
    write(Path(arm['run']) / 'summary.json', summary)
    with pytest.raises(ValueError): pipeline.check_training(arm, plan)


def test_cache_miss_guard_blocks_builder_and_restores_patches(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from experiments import rsmi_separate_qcr as trainer
    from recurrent_rg import mi_cache
    from scripts import report_ising_zh
    original_builder = trainer.build_model
    def cache_miss(directory, identity, build):
        build()
        pytest.fail('Cache-miss must throw')
    monkeypatch.setattr(mi_cache, 'load_or_build_mi_statistics', cache_miss)
    def training(argv):
        assert '--skip-final-report' in argv
        mi_cache.load_or_build_mi_statistics('unused', {}, lambda: pytest.fail('MI rebuilding forbidden'))
    monkeypatch.setattr(trainer, 'main', training)
    monkeypatch.setattr(report_ising_zh, 'chinese_figures', nullcontext)
    args = SimpleNamespace(config=tmp_path / 'c.json', output=tmp_path / 'run', device='cuda:3',
                           mi_cache_dir=tmp_path / 'mi', initialization_receipt=tmp_path / 'init.json')
    with pytest.raises(RuntimeError, match='禁止重算'): pipeline.cached_training(args)
    assert mi_cache.load_or_build_mi_statistics is cache_miss
    assert trainer.build_model is original_builder


def test_cache_hit_guard_and_initialization_receipt_are_readonly_model_observation(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from experiments import rsmi_separate_qcr as trainer
    from recurrent_rg import mi_cache
    from scripts import report_ising_zh
    model = SimpleNamespace(rg_kernel=torch.nn.Linear(2, 1), theory_encoder=torch.nn.Linear(3, 2),
                            theory_rg=torch.nn.Linear(2, 2))
    model.state_dict = lambda: {'C': model.rg_kernel.weight, 'Q': model.theory_encoder.weight,
                               'R': model.theory_rg.weight}
    torch.manual_seed(91)
    before = torch.random.get_rng_state().clone()
    monkeypatch.setattr(trainer, 'build_model', lambda *a, **kw: model)
    monkeypatch.setattr(mi_cache, 'load_or_build_mi_statistics', lambda d, k, b: ('static', dict(hit=True)))
    def training(argv):
        assert trainer.build_model(None) is model
        assert mi_cache.load_or_build_mi_statistics('unused', {}, None)[0] == 'static'
    monkeypatch.setattr(trainer, 'main', training)
    monkeypatch.setattr(report_ising_zh, 'chinese_figures', nullcontext)
    args = SimpleNamespace(config=tmp_path / 'c.json', output=tmp_path / 'run', device='cuda:0',
                           mi_cache_dir=tmp_path / 'mi', initialization_receipt=tmp_path / 'init.json')
    pipeline.cached_training(args)
    row = json.loads(args.initialization_receipt.read_text())
    assert row['fresh'] is True and row['source_checkpoint'] is None
    assert len(row['C_tensor_sha256']) == 64 and len(row['R_tensor_sha256']) == 64
    assert torch.equal(before, torch.random.get_rng_state())


def test_execution_refuses_cuda_remapping_before_creating_output(tmp_path, monkeypatch):
    source, _ = source_fixture(tmp_path, monkeypatch)
    plan = pipeline.build_plan(source, tmp_path / 'new')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,2,3')
    with pytest.raises(ValueError, match='CUDA_VISIBLE_DEVICES'): pipeline.execute(plan)
    assert not (tmp_path / 'new').exists()


def test_smoke_is_separate_two_step_plan_with_no_formal_reports(tmp_path, monkeypatch):
    source, cfg = source_fixture(tmp_path, monkeypatch)
    plan = pipeline.build_smoke_plan(source, tmp_path / 'new_smoke')
    assert plan['smoke_only'] is True and 'comparison_command' not in plan
    assert plan['protocol']['updates'] == 2
    for arm in plan['arms']:
        assert set(arm['commands']) == {'training'}
        assert arm['config']['training']['epochs'] == 2
        assert arm['config']['training']['checkpoint_every'] == 1
        assert arm['config']['training']['log_every'] == 1
        assert arm['config']['training']['lr_milestones'] == [6000, 9000]
    assert json.loads((source / 'manifest.json').read_text())['config'] == cfg
    assert not (tmp_path / 'new_smoke').exists()


def test_scheduler_waits_twice_then_parallel_starts_and_does_not_retry_failed_children(tmp_path, monkeypatch):
    source, _ = source_fixture(tmp_path, monkeypatch)
    plan = pipeline.build_plan(source, tmp_path / 'new')
    monkeypatch.setattr(pipeline, 'CODE_FILES', ())
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    monkeypatch.setattr(pipeline.subprocess, 'check_output', lambda *a, **k: 'host')
    clock = iter(range(0, 10000, 61))
    monkeypatch.setattr(pipeline.time, 'monotonic', lambda: next(clock))
    monkeypatch.setattr(pipeline.time, 'sleep', lambda seconds: None)
    calls, queries = [], []
    def gpu(index):
        queries.append(index)
        return dict(index=index, uuid=str(index), name='test', free_mib=24000, compute_pids=[])
    monkeypatch.setattr(pipeline, 'gpu_snapshot', gpu)
    monkeypatch.setattr(pipeline, 'resources', lambda: dict(available_memory_bytes=90 * (1 << 30)))
    class FailedChild:
        pid = 77
        def poll(self): return 7
        def terminate(self): pytest.fail('Do not stop shared or other active processes')
        def kill(self): pytest.fail('Do not kill failed or other processes')
    def popen(command, **kwargs):
        calls.append(command)
        assert kwargs['env']['OMP_NUM_THREADS'] == '2'
        return FailedChild()
    monkeypatch.setattr(pipeline.subprocess, 'Popen', popen)
    assert pipeline.execute(plan) == 1
    assert len(calls) == 3 and all('--cached-training' in command for command in calls)
    assert queries == [0, 2, 3, 0, 2, 3]
    job = json.loads((tmp_path / 'new' / 'job.json').read_text())
    assert job['status'] == 'failed' and job['requires_user_action'] is True
    assert [row['returncode'] for row in job['stages']] == [7, 7, 7]
    assert all(row['status'] == 'failed' for row in job['arms'].values())
