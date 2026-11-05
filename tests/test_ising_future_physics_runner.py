"""Execution-only budget/state tests; no GPU, SSH, data or model loading."""
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_ising_future_physics as runner


def sample_summary(setup=80., checkpoint=2.):
    return dict(history=[{}, {}, {}], timing=dict(steady_step_seconds=[.1] * 35,
        setup_seconds=setup, logging_seconds=6., checkpoint_seconds=checkpoint))


def test_prior_failed_preflight_carries_both_gpu_parts_and_missing_cpu(tmp_path):
    source, prior = tmp_path / 'source', tmp_path / 'prior'
    prior.mkdir()
    job = dict(experiment='ising_future_physics_Q_loss_b3_paired_v1', source=str(source), status='failed',
        stages=[dict(name='prepare'), dict(name='calibration'), dict(name='benchmark')],
        gpu_seconds_completed=6.151228308, gpu_seconds_current_stage=533.6,
        CPU_seconds_completed=12.2, peak_own_gpu_mib=12955)
    runner.save(prior / 'job.json', job)
    value = runner.prior_execution_receipt(prior, source)
    assert value['gpu_seconds'] == pytest.approx(539.751228308)
    assert value['CPU_seconds'] == pytest.approx(545.8)
    assert value['previous_peak_own_gpu_mib'] == 12955
    job['stages'].append(dict(name='train_control'))
    runner.save(prior / 'job.json', job)
    with pytest.raises(ValueError, match='preflight'):
        runner.prior_execution_receipt(prior, source)


def test_prior_chain_uses_already_cumulative_totals_once_and_keeps_earliest_peak(tmp_path):
    source, prior = tmp_path / 'source', tmp_path / 'v2'
    prior.mkdir()
    job = dict(experiment='ising_future_physics_Q_loss_b3_paired_v1', source=str(source), status='failed',
        stages=[dict(name='prepare'), dict(name='calibration'), dict(name='benchmark')],
        gpu_seconds_completed=576.1316, gpu_seconds_current_stage=0., CPU_seconds_completed=580.,
        peak_own_gpu_mib=11591, campaign_peak_own_gpu_mib=12955,
        prior_execution=dict(gpu_seconds=539.751228308, CPU_seconds=545.8,
                             previous_peak_own_gpu_mib=12955))
    runner.save(prior / 'job.json', job)
    receipt = runner.prior_execution_receipt(prior, source)
    # The v2 completed totals already include v1; its nested receipt is not added again.
    assert receipt['gpu_seconds'] == pytest.approx(576.1316)
    assert receipt['CPU_seconds'] == pytest.approx(580.)
    assert receipt['previous_peak_own_gpu_mib'] == 12955
    job['peak_own_gpu_mib'] = 13001
    runner.save(prior / 'job.json', job)
    assert runner.prior_execution_receipt(prior, source)['previous_peak_own_gpu_mib'] == 13001


def test_projection_includes_actual_setup_checkpoint_and_31_logs_per_arm():
    estimate = runner.benchmark_projection(sample_summary())
    assert estimate['updates_per_arm'] == 6000 and estimate['logs_per_arm'] == 31
    assert estimate['setup_seconds_per_arm'] == 80
    assert estimate['checkpoint_seconds_per_arm'] == 12
    assert estimate['per_arm_seconds'] == pytest.approx(1.5 * (600 + 62 + 80 + 12))
    projected = runner.remaining_projection(dict(gpu_seconds_completed=539.751228308,
        CPU_seconds_completed=545.8), estimate, arms=2, collections=2)
    assert projected['projected_total_gpu_seconds'] == pytest.approx(539.751228308 + 2 * estimate['per_arm_seconds'] + 240)
    assert projected['projected_total_CPU_seconds'] == pytest.approx(545.8 + 2 * estimate['per_arm_seconds'] + 240 + 60)
    assert runner.benchmark_projection(sample_summary(setup=2))['setup_seconds_per_arm'] == 60


def test_remaining_gate_rejects_slower_actual_prefix_without_shortening(tmp_path):
    estimate = runner.benchmark_projection(sample_summary())
    job = dict(gpu_seconds_completed=539.751228308, CPU_seconds_completed=545.8)
    assert runner.budget_gate(tmp_path, job, estimate, arms=2, collections=2, before='train_control')
    job['gpu_seconds_completed'] = 2800
    assert not runner.budget_gate(tmp_path, job, estimate, arms=1, collections=1, before='train_future')
    assert job['status'] == 'budget_hold' and job['workflow_complete'] is False
    assert job['budget_projection']['updates_per_arm'] == 6000
    assert len(job['budget_projection_history']) == 2


def test_training_limit_preserves_setup_saves_later_arm_and_collections():
    estimate = runner.benchmark_projection(sample_summary())
    job = dict(gpu_seconds_completed=539.751228308)
    limit = runner.training_time_limit(job, estimate, later_arms=1, collections=2)
    accounted = (job['gpu_seconds_completed'] + limit + estimate['per_arm_seconds'] + 240 +
        1.5 * (estimate['setup_seconds_per_arm'] + estimate['checkpoint_seconds_per_arm']))
    assert accounted == pytest.approx(3600)
    command = runner.trainer_command(__import__('pathlib').Path('/new'), __import__('pathlib').Path('/new/run'),
        __import__('pathlib').Path('/new/config.json'), future=True, limit=limit)
    assert '--mi-test-cpu' in command and '--resume-checkpoint' not in command and '--epochs' not in command


class FakeProcess:
    pid = 123
    def __init__(self, timeout=False, auto_finish=False):
        self.returncode = None
        self.timeout, self.auto_finish = timeout, auto_finish
        self.terminated = False
        self.polls = 0
    def poll(self):
        self.polls += 1
        if self.auto_finish and self.polls >= 2:
            self.returncode = 0
        return self.returncode
    def terminate(self):
        self.terminated = True
        if not self.timeout: self.returncode = -15
    def wait(self, timeout):
        if self.timeout: raise subprocess.TimeoutExpired('own child', timeout)
        return self.returncode


def stage_fixture(tmp_path, monkeypatch, process, own_memory=9000):
    (tmp_path / 'logs').mkdir()
    monkeypatch.setattr(runner, 'wait_resources', lambda *a: None)
    monkeypatch.setattr(runner.subprocess, 'Popen', lambda *a, **k: process)
    monkeypatch.setattr(runner, 'snapshot', lambda: dict(processes=[dict(pid=123, used_mib=own_memory), dict(pid=456, used_mib=20000)]))
    clock = iter(range(100, 200))
    monkeypatch.setattr(runner.time, 'monotonic', lambda: next(clock))
    monkeypatch.setattr(runner.time, 'sleep', lambda seconds: None)
    monkeypatch.setattr(runner.resource, 'getrusage', lambda kind: SimpleNamespace(ru_utime=2., ru_stime=3.))
    return dict(stages=[], gpu_seconds_completed=539.751228308,
                prior_execution=dict(CPU_seconds=545.8), peak_own_gpu_mib=0)


def test_gpu_limit_finalizes_failed_child_once_and_preserves_prior(tmp_path, monkeypatch):
    process = FakeProcess()
    job = stage_fixture(tmp_path, monkeypatch, process)
    with pytest.raises(RuntimeError, match='memory limit'):
        runner.run_stage(tmp_path, job, 'benchmark', ['unused'], gpu=True)
    row = job['stages'][0]
    assert process.terminated and row['returncode'] == -15 and row['status'] == 'failed'
    assert row['ended_at'] and job['current'] is None and not job['child_still_running']
    assert job['gpu_seconds_completed'] == pytest.approx(539.751228308 + row['elapsed_seconds'])
    assert job['gpu_seconds_current_stage'] == 0
    assert job['CPU_seconds_completed'] == pytest.approx(555.8)
    assert job['peak_own_gpu_mib'] == 9000  # Other users' 20000 MiB ignored.


def test_termination_timeout_preserves_live_pid_and_unsettled_accounting(tmp_path, monkeypatch):
    process = FakeProcess(timeout=True)
    job = stage_fixture(tmp_path, monkeypatch, process)
    with pytest.raises(RuntimeError, match='memory limit'):
        runner.run_stage(tmp_path, job, 'benchmark', ['unused'], gpu=True)
    row = job['stages'][0]
    assert row['status'] == 'child_still_running' and row['returncode'] is None
    assert 'ended_at' not in row and row['termination_timed_out']
    assert job['current']['pid'] == 123 and job['child_still_running']
    assert job['gpu_seconds_completed'] == pytest.approx(539.751228308)
    assert job['gpu_seconds_current_stage'] == row['elapsed_seconds']
    assert not hasattr(process, 'kill')


def test_other_gpu_process_memory_does_not_terminate_owned_child(tmp_path, monkeypatch):
    process = FakeProcess(auto_finish=True)
    job = stage_fixture(tmp_path, monkeypatch, process, own_memory=100)
    runner.run_stage(tmp_path, job, 'benchmark', ['unused'], gpu=True)
    assert not process.terminated and job['stages'][0]['status'] == 'complete'


def test_cache_reuse_requires_same_new_data_and_complete_two_files(tmp_path):
    old, new = tmp_path / 'old', tmp_path / 'new'
    old.mkdir(); new.mkdir(); (old / 'mi_cache').mkdir()
    runner.save(old / 'dataset_receipt.json', dict(data_sha256='same-new-data'))
    runner.save(new / 'dataset_receipt.json', dict(data_sha256='different'))
    prior = dict(root=str(old), cache_source=str(old / 'mi_cache'))
    with pytest.raises(ValueError, match='different data fingerprints'):
        runner.reuse_own_statistics(new, prior)
    assert not (new / 'mi_cache').exists()
    runner.save(new / 'dataset_receipt.json', dict(data_sha256='same-new-data'))
    # Fake files are bytes only; this test never parses or creates any model/data tensor.
    (old / 'mi_cache/mi-v1-train.pt').write_bytes(b'new TRAIN statistics')
    (old / 'mi_cache/mi-v1-test.pt').write_bytes(b'new TEST statistics')
    runner.reuse_own_statistics(new, prior)
    for name in ('mi-v1-train.pt', 'mi-v1-test.pt'):
        assert (old / 'mi_cache' / name).stat().st_ino == (new / 'mi_cache' / name).stat().st_ino
    assert len(json.loads((new / 'statistics_reuse.json').read_text())['records']) == 2


def test_authorized5400_budget_is_used_by_gate_and_training_limit(tmp_path):
    estimate = runner.benchmark_projection(sample_summary(setup=100., checkpoint=10.))
    job = dict(gpu_seconds_completed=2200., CPU_seconds_completed=2200.)
    assert not runner.budget_gate(tmp_path, job, estimate, arms=2, collections=2, before='train_control')
    job['GPU_card_seconds'] = 5400
    assert runner.budget_gate(tmp_path, job, estimate, arms=2, collections=2, before='train_control')
    assert job['budget_projection']['allowed_gpu_seconds'] == 5400
    assert job['budget_projection']['safety_factor'] == 1.5
    limit = runner.training_time_limit(job, estimate, later_arms=1, collections=2)
    assert 2200. + limit + estimate['per_arm_seconds'] + 240 + 1.5 * (100 + 60) == pytest.approx(5400)
    with pytest.raises(ValueError, match='authorized5400'):
        runner.gpu_budget(dict(GPU_card_seconds=7200))


@pytest.mark.parametrize('limit,terminated', [(3600, True), (5400, False)])
def test_live_hard_limit_uses_authorized_campaign_budget(tmp_path, monkeypatch, limit, terminated):
    process = FakeProcess(auto_finish=not terminated)
    job = stage_fixture(tmp_path, monkeypatch, process, own_memory=100)
    job.update(gpu_seconds_completed=4000., GPU_card_seconds=limit)
    if terminated:
        with pytest.raises(runner.BudgetHold, match='card-time'):
            runner.run_stage(tmp_path, job, 'train_control', ['unused'], gpu=True)
    else:
        runner.run_stage(tmp_path, job, 'train_control', ['unused'], gpu=True)
    assert process.terminated is terminated


def completed_preflight(tmp_path):
    """Fake byte caches and JSON only; never a tensor, model or optimizer."""
    prior, code, source = tmp_path / 'prior', tmp_path / 'code', tmp_path / 'source'
    for folder in (prior / 'data', prior / 'benchmark', prior / 'mi_cache', code / 'scripts', code / 'recurrent_rg'):
        folder.mkdir(parents=True)
    (code / 'scripts/run_ising_future_physics.py').write_text('# original runner\n')
    (code / 'recurrent_rg/science.py').write_text('# frozen science\n')
    (prior / 'data/ensembles.pt').write_bytes(b'own new spins, not loaded')
    (prior / 'mi_cache/mi-v1-train.pt').write_bytes(b'own train MI, not loaded')
    (prior / 'mi_cache/mi-v1-test.pt').write_bytes(b'own test MI, not loaded')
    data_sha = runner.file_sha(prior / 'data/ensembles.pt')
    runner.save(prior / 'pairs.json', dict(data_sha256=data_sha))
    pair_sha = runner.file_sha(prior / 'pairs.json')
    runner.save(prior / 'normalizer.json', dict(data_sha256=data_sha, pairs_sha256=pair_sha,
        target_mean=[[0.] * 5] * 3, target_std=[[.01] * 5] * 3))
    norm = json.loads((prior / 'normalizer.json').read_text())
    runner.save(prior / 'dataset_receipt.json', dict(data_sha256=data_sha, pair_sha256=pair_sha))
    config = dict(schema_version=5, data=dict(cache=str(prior / 'data/ensembles.pt')),
        model=dict(block_factor=3), training=dict(epochs=6000, log_every=200, checkpoint_every=1000))
    runner.save(prior / 'config.json', config)
    config['training'].update(epochs=40, log_every=20, checkpoint_every=40)
    runner.save(prior / 'benchmark_config.json', config)
    future = dict(data_sha256=data_sha, pairs_sha256=pair_sha,
        normalizer_sha256=runner.file_sha(prior / 'normalizer.json'), weight=1, hidden=32,
        mode='future_physics', labels_used=False, paths=[[0,1],[1,2],[0,2]],
        target_mean=norm['target_mean'], target_std=norm['target_std'],
        encoding_execution=dict(encoding_batch_chunk=4, validated_binary_inputs=True))
    execution = dict(mi_test_device='cpu', future_encoding_batch_chunk=4)
    manifest = dict(config=config, data_sha256=data_sha, q_future_physics=future,
        execution_device='cuda:2', execution_optimizations=execution)
    runner.save(prior / 'benchmark/manifest.json', manifest)
    summary = sample_summary()
    summary.update(config=config, contract=dict(q_future_physics=future, execution_optimizations=execution),
        actual_epochs=40, requested_epochs=40, stopped_by_wall_limit=False, stopped_by_training_limit=False)
    runner.save(prior / 'benchmark/summary.json', summary)
    job = dict(experiment='ising_future_physics_Q_loss_b3_paired_v1', source=str(source), status='budget_hold',
        workflow_complete=False, current=None, child_still_running=False, gpu_seconds_current_stage=0.,
        stages=[dict(name=name, status='complete', returncode=0) for name in ('prepare','calibration','benchmark')],
        gpu_seconds_completed=622.0111813731492, CPU_seconds_completed=630.,
        device='cuda:2', gpu_uuid=runner.GPU_UUID,
        peak_own_gpu_mib=6668, campaign_peak_own_gpu_mib=12955,
        prior_execution=dict(gpu_seconds=576., CPU_seconds=580.))
    runner.save(prior / 'job.json', job)
    runner.save(prior / 'plan.json', dict(source_code_sha256=runner.source_hashes(code)))
    return prior, source, code


def test_validated_preflight_reuses_bytes_and_never_checkpoint_or_model(tmp_path):
    prior, source, code = completed_preflight(tmp_path)
    (prior / 'benchmark/checkpoint_last.pt').write_bytes(b'never open this model')
    (code / 'scripts/run_ising_future_physics.py').write_text('# budget-only runner change\n')
    receipt = runner.validated_preflight(prior, source, code)
    assert receipt['gpu_seconds'] == 622.0111813731492  # Already cumulative, not prior added twice.
    assert receipt['previous_peak_own_gpu_mib'] == 12955
    new = tmp_path / 'new'; new.mkdir()
    runner.reuse_preflight(new, receipt)
    for name in ('data/ensembles.pt', 'pairs.json', 'normalizer.json', 'dataset_receipt.json'):
        assert (prior / name).stat().st_ino == (new / name).stat().st_ino
    old_cfg, new_cfg = (json.loads((base / 'config.json').read_text()) for base in (prior,new))
    assert new_cfg['data'].pop('cache') == str(new / 'data/ensembles.pt')
    old_cfg['data'].pop('cache')
    assert new_cfg == old_cfg
    assert not (new / 'benchmark/checkpoint_last.pt').exists()
    record = json.loads((new / 'preflight_reuse.json').read_text())
    assert record['checkpoint_files_opened'] is False and record['model_or_optimizer_reused'] is False


@pytest.mark.parametrize('mutation,error', [
    ('science', 'science source'), ('incomplete', 'complete budget-held'),
    ('formal', 'formal training'), ('stopped', 'completed40'), ('sha', 'SHA'),
    ('config', 'unchanged6000'), ('wrong_gpu', 'complete budget-held'),
    ('excess_memory', 'complete budget-held'), ('raw_config', 'unchanged6000')])
def test_preflight_reuse_rejects_changed_science_or_incomplete_contract(tmp_path, mutation, error):
    prior, source, code = completed_preflight(tmp_path)
    if mutation == 'science':
        (code / 'recurrent_rg/science.py').write_text('# changed loss\n')
    elif mutation == 'formal':
        (prior / 'run_control').mkdir()
    elif mutation == 'sha':
        (prior / 'data/ensembles.pt').write_bytes(b'changed data')
    else:
        name = ('job.json' if mutation in ('incomplete', 'wrong_gpu', 'excess_memory') else
                'benchmark/summary.json' if mutation == 'stopped' else 'benchmark_config.json')
        obj = json.loads((prior / name).read_text())
        if mutation == 'incomplete': obj['stages'][2]['returncode'] = 1
        elif mutation == 'stopped': obj['stopped_by_training_limit'] = True
        elif mutation == 'wrong_gpu': obj['gpu_uuid'] = 'a different GPU'
        elif mutation == 'excess_memory': obj['peak_own_gpu_mib'] = 8193
        elif mutation == 'raw_config':
            obj['model']['block_factor'] = 2
            raw = json.loads((prior / 'config.json').read_text())
            raw['model']['block_factor'] = 2
            runner.save(prior / 'config.json', raw)
        else: obj['training']['epochs'] = 20
        runner.save(prior / name, obj)
    with pytest.raises(ValueError, match=error):
        runner.validated_preflight(prior, source, code)


def test_reuse_run_starts_only_two_fresh_arms_and_collection_report(tmp_path, monkeypatch):
    prior, source, code = completed_preflight(tmp_path)
    monkeypatch.chdir(code)
    monkeypatch.setattr(runner.os, 'setpriority', lambda *args: None)
    stages = []
    def stage(root, job, name, command, **kwargs):
        stages.append(name)
        assert not any(word in command for word in ('prepare','calibrate','--resume-checkpoint'))
        if name.startswith('train_'):
            output = Path(command[command.index('--output') + 1]); output.mkdir()
            runner.save(output / 'summary.json', dict(actual_epochs=6000, requested_epochs=6000))
        job['stages'].append(dict(name=name, status='complete', returncode=0))
    monkeypatch.setattr(runner, 'run_stage', stage)
    new = tmp_path / 'new'
    runner.run(new, source, prior, gpu_seconds=5400, reuse=True)
    assert stages == ['train_control','collect_control','train_future','collect_future','report']
    job = json.loads((new / 'job.json').read_text())
    assert job['status'] == 'complete' and job['GPU_card_seconds'] == 5400
    assert job['gpu_seconds_completed'] == 622.0111813731492
    assert '扩大' in job['budget_authorization']
