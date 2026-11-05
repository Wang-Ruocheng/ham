"""Versioned low-RAM scheduling and protection tests; no GPU or physical data."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_ising_highdim as original
from scripts import run_ising_highdim_low_ram as low
from test_ising_highdim_runner import source  # noqa: F401: shared tiny JSON-only fixture


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def handoff(tmp_path, monkeypatch):
    old, output = tmp_path / 'old', tmp_path / 'new'
    write_json(old / 'plan.json', {'science': 'frozen'})
    write_json(old / 'job.json', dict(stages=[], GPU_stage_seconds=0, workflow_complete=False))
    receipt = dict(status='superseded_before_any_stage', termination_confirmed=True,
        old_stages=[], training_updates=0, physical_sampling_updates=0, old_directory_preserved=True,
        new_root=str(output), old_plan_sha256=original.sha(old / 'plan.json'),
        parent_pid=17808, parent_starttime_ticks='100')
    write_json(old / 'low_ram_handoff.json', receipt)
    monkeypatch.setattr(low, 'process_identity', lambda pid: None)
    return old, output, receipt


def test_handoff_requires_no_historical_stage_and_inactive_old_identity(handoff, monkeypatch):
    old, output, receipt = handoff
    result = low.verify_handoff(old, output)
    assert result['old_stages'] == [] and result['termination_confirmed']
    monkeypatch.setattr(low, 'process_identity', lambda pid: dict(starttime_ticks='100', state='S'))
    with pytest.raises(RuntimeError, match='still alive'):
        low.verify_handoff(old, output)
    monkeypatch.setattr(low, 'process_identity', lambda pid: dict(starttime_ticks='101', state='S'))
    assert low.verify_handoff(old, output) == result  # unrelated reused PID is not targeted
    monkeypatch.setattr(low, 'process_identity', lambda pid: dict(starttime_ticks='100', state='Z'))
    assert low.verify_handoff(old, output) == result


@pytest.mark.parametrize('field,value', [('old_stages', ['smoke_dim1']), ('training_updates', 1),
    ('physical_sampling_updates', 1), ('termination_confirmed', False), ('old_directory_preserved', False),
    ('new_root', '/different-output'), ('old_plan_sha256', 'changed'), ('status', 'waiting')])
def test_handoff_rejects_unsafe_or_changed_evidence(handoff, field, value):
    old, output, receipt = handoff
    receipt[field] = value
    write_json(old / 'low_ram_handoff.json', receipt)
    with pytest.raises(RuntimeError):
        low.verify_handoff(old, output)


@pytest.fixture
def preflight_handoff(source, tmp_path, monkeypatch):
    old, output = tmp_path / 'v2', tmp_path / 'v3'
    plan = original.build_plan(source[0], old)
    plan.update(execution_version='lowram_v2', low_ram_execution=dict(own_gpu_hard_limit_gib=8),
        source_code_sha256={'scripts/frozen.py': 'unchanged'},
        configuration_sha256={str(old / 'configs/dim1.json'): 'unchanged'})
    write_json(old / 'plan.json', plan)
    plan = original.read(old / 'plan.json')
    job = dict(status='failed', workflow_complete=False, requires_user_action=True,
        error=low.PREVIOUS_PREFLIGHT_ERROR, current='smoke_dim1', completed_arms=[],
        physical_sampling_updates=0, labels_used_in_training=False, execution_version='lowram_v2',
        host=original.HOST, device='cuda:2', gpu_uuid=original.GPU_UUID,
        parent_pid=31814, stages=[dict(name='cuda_verification', status='complete', returncode=0, pid=32110),
        dict(name='smoke_dim1', status='failed', returncode=-15, pid=644,
             error=low.PREVIOUS_PREFLIGHT_ERROR)])
    write_json(old / 'job.json', job)
    smoke_profile = dict(status='failed', returncode=-15, child_pid=644, child_starttime_ticks='1159101296',
        violation=low.PREVIOUS_PREFLIGHT_ERROR.removeprefix('RuntimeError: '),
        monitor_error=low.PREVIOUS_PREFLIGHT_ERROR)
    write_json(old / 'profiles/smoke_dim1.json', smoke_profile)
    receipt = dict(schema_version=1, status='superseded_after_preflight_guard', termination_confirmed=True,
        old_directory_preserved=True, new_root=str(output),
        old_plan_sha256=original.sha(old / 'plan.json'), old_job_sha256=original.sha(old / 'job.json'),
        old_profile_sha256=original.sha(old / 'profiles/smoke_dim1.json'),
        parent_pid=31814, parent_starttime_ticks=None, child_pid=644, child_starttime_ticks='1159101296',
        formal_training_updates=0, physical_sampling_updates=0, checkpoint_resume=False,
        resource_change=dict(user_authorized=True, previous_own_gpu_limit_gib=8,
            own_gpu_limit_gib=14, minimum_free_gpu_gib=16, scientific_configuration_unchanged=True),
        old_stage_names=['cuda_verification', 'smoke_dim1'], old_error=low.PREVIOUS_PREFLIGHT_ERROR)
    write_json(old / 'preflight_handoff.json', receipt)
    monkeypatch.setattr(low, 'process_identity', lambda pid: None)
    checks = []
    monkeypatch.setattr(original, 'ensure_unchanged', lambda *args: checks.append(args))
    return old, output, receipt, plan, job, smoke_profile, checks


def test_preflight_handoff_pins_stopped_failure_and_keeps_every_scientific_setting(source, preflight_handoff):
    old, output, receipt, old_plan, _, _, checks = preflight_handoff
    handoff_record = low.verify_handoff(old, output)
    assert handoff_record['kind'] == 'failed_preflight_resource_only'
    assert handoff_record['old_job_sha256'] == receipt['old_job_sha256']
    assert handoff_record['formal_training_updates'] == 0 and not handoff_record['checkpoint_resume']
    assert checks[-1] == (old_plan, old / 'code', old_plan['source_code_sha256'], old_plan['configuration_sha256'])
    new_plan = low.build_plan(source[0], output, old)
    assert json.loads(json.dumps([arm['config'] for arm in new_plan['arms']])) == [arm['config'] for arm in old_plan['arms']]
    assert new_plan['execution_version'] == 'lowram_v3'
    assert new_plan['low_ram_execution']['own_gpu_hard_limit_gib'] == 14
    assert new_plan['protocol']['resource_gate']['gpu_free_mib'] == 16384
    assert not output.exists()


@pytest.mark.parametrize('field,value', [('schema_version', 2), ('status', 'failed'),
    ('termination_confirmed', False), ('old_directory_preserved', False), ('new_root', '/another-root'),
    ('old_plan_sha256', 'changed'), ('old_job_sha256', 'changed'), ('old_profile_sha256', 'changed'),
    ('old_stage_names', ['cuda_verification']), ('old_error', 'OOM'), ('formal_training_updates', 1),
    ('physical_sampling_updates', 1), ('checkpoint_resume', True), ('resource_change', {}),
    ('parent_pid', 99), ('child_pid', 99), ('child_starttime_ticks', '999')])
def test_preflight_handoff_rejects_unsafe_or_changed_receipt(preflight_handoff, field, value):
    old, output, receipt, *_ = preflight_handoff
    receipt[field] = value
    write_json(old / 'preflight_handoff.json', receipt)
    with pytest.raises(RuntimeError):
        low.verify_handoff(old, output)


@pytest.mark.parametrize('target,field,value', [('job', 'error', 'CUDA out of memory'),
    ('job', 'stages', [dict(name='train_dim1', status='running', pid=1)]),
    ('job', 'completed_arms', ['dim1']), ('job', 'all_smoke_checks_passed', True),
    ('job', 'measured_resource_gate', {}), ('job', 'labels_used_in_training', True),
    ('profile', 'violation', 'Other error'), ('profile', 'returncode', 1),
    ('plan', 'execution_version', 'lowram_v3'), ('plan', 'source_code_sha256', {})])
def test_preflight_handoff_rejects_changed_content_even_with_updated_hash(preflight_handoff, target, field, value):
    old, output, receipt, plan, job, smoke_profile, _ = preflight_handoff
    objects = {'job': job, 'plan': plan, 'profile': smoke_profile}
    files = {'job': 'job.json', 'plan': 'plan.json', 'profile': 'profiles/smoke_dim1.json'}
    objects[target][field] = value
    write_json(old / files[target], objects[target])
    receipt[f'old_{target}_sha256'] = original.sha(old / files[target])
    write_json(old / 'preflight_handoff.json', receipt)
    with pytest.raises(RuntimeError):
        low.verify_handoff(old, output)


@pytest.mark.parametrize('pid', [31814, 32110, 644])
def test_preflight_handoff_rejects_any_remaining_old_pid(preflight_handoff, monkeypatch, pid):
    old, output, *_ = preflight_handoff
    monkeypatch.setattr(low, 'process_identity', lambda observed: dict(pid=pid) if observed == pid else None)
    with pytest.raises(RuntimeError, match='still present'):
        low.verify_handoff(old, output)


@pytest.mark.parametrize('name', ['run', 'report', 'initialization.json'])
def test_preflight_handoff_rejects_old_formal_training_artifacts(preflight_handoff, name):
    old, output, *_ = preflight_handoff
    write_json(old / 'dim1' / name, {})
    with pytest.raises(RuntimeError, match='formal training evidence'):
        low.verify_handoff(old, output)


def test_preflight_handoff_refuses_existing_new_output_and_changed_frozen_code(preflight_handoff, monkeypatch):
    old, output, *_ = preflight_handoff
    output.mkdir()
    with pytest.raises(RuntimeError):
        low.verify_handoff(old, output)
    output.rmdir()

    def frozen_change(*args):
        raise RuntimeError('Frozen science code changed')

    monkeypatch.setattr(original, 'ensure_unchanged', frozen_change)
    with pytest.raises(RuntimeError, match='Frozen science code'):
        low.verify_handoff(old, output)


def test_resource_only_plan_rejects_different_prior_scientific_config(source, preflight_handoff):
    old, output, receipt, plan, *_ = preflight_handoff
    plan['arms'][0]['config']['loss']['closure_scale_weights'] = [1.8, .2]
    write_json(old / 'plan.json', plan)
    receipt['old_plan_sha256'] = original.sha(old / 'plan.json')
    write_json(old / 'preflight_handoff.json', receipt)
    with pytest.raises(RuntimeError, match='scientific configuration'):
        low.build_plan(source[0], output, old)


def test_plan_preserves_every_scientific_config_and_uses_new_profiled_wrapper(source, handoff):
    old, output, _ = handoff
    plan = low.build_plan(source[0], output, old)
    reference = original.build_plan(source[0], output)
    assert plan['execution_version'] == 'lowram_v3'
    assert plan['protocol']['resource_gate']['gpu_free_mib'] == 16384
    assert plan['low_ram_execution']['own_gpu_hard_limit_gib'] == 14
    for arm, ref in zip(plan['arms'], reference['arms']):
        assert arm['config'] == ref['config']
        assert arm['report'] == ref['report']
        for key in ('smoke', 'training'):
            expected = deepcopy(ref[key])
            expected[expected.index('scripts.run_ising_highdim')] = 'scripts.run_ising_highdim_low_ram'
            assert arm[key] == expected + ['--benchmark-warmup', '0']
    assert plan['cuda_verification'][3] == 'scripts.verify_low_ram_cuda'
    assert plan['cuda_verification'][plan['cuda_verification'].index('--output') + 1].endswith('cuda_verification.json')
    assert not output.exists()


def test_snapshot_uses_documented_low_ram_gate_but_not_weaker_other_gates(monkeypatch):
    base = dict(mem_available_gib=18., load1=30., cpu_count=24, io_full_avg10=19.9,
        disk_free_gib=20., compute_processes=[], gpu_free_mib=16384)
    monkeypatch.setattr(original, 'snapshot', lambda: dict(base))
    assert low.snapshot(18)['gpu_ready']
    assert not low.snapshot(18.01)['host_ready']
    for key, value in [('load1', 30.1), ('io_full_avg10', 20), ('disk_free_gib', 19.9),
                       ('compute_processes', ['other']), ('gpu_free_mib', 16383)]:
        old = base[key]
        base[key] = value
        assert not low.snapshot(18)['gpu_ready']
        base[key] = old


def test_admission_two_consecutive_gpu_checks_spaced_ten_seconds(tmp_path, monkeypatch):
    states, sleeps = iter([True, False, True, True]), []
    monkeypatch.setattr(low, 'snapshot', lambda required: dict(host_ready=True, gpu_ready=next(states),
        required_mem_available_gib=required))
    monkeypatch.setattr(low.time, 'sleep', sleeps.append)
    job = {}
    low.admit(tmp_path, job, gpu=True, required_available_gib=18)
    assert sleeps == [10, 60, 10]
    assert len(job['resource_checks']) == 4


def test_cpu_admission_does_not_require_selected_gpu_idle(tmp_path, monkeypatch):
    monkeypatch.setattr(low, 'snapshot', lambda required: dict(host_ready=True, gpu_ready=False))
    monkeypatch.setattr(low.time, 'sleep', lambda seconds: pytest.fail('CPU can proceed'))
    low.admit(tmp_path, {}, gpu=False, required_available_gib=18)


def test_proc_memory_units_and_anonymous_fallback(tmp_path, monkeypatch):
    fixture = tmp_path / 'mem'
    fixture.write_text('Rss: 100 kB\nPss: 50 kB\nAnonymous: 25 kB\nPss_File: 25 kB\nVmSize: 777 kB\n'
                       'Shared_Clean: 10 kB\nPrivate_Clean: 5 kB\nShared_Dirty: 20 kB\nPrivate_Dirty: 65 kB\n')
    parsed = low.proc_kib(fixture)
    assert parsed['Rss'] == 100 * 1024
    monkeypatch.setattr(low, 'proc_kib', lambda path: parsed)
    identity = dict(pid=123, starttime_ticks='1')
    monkeypatch.setattr(low, 'process_identity', lambda pid: identity)
    measured = low.tree_memory([identity])
    assert measured['pss_anon_bytes'] == 25 * 1024
    assert measured['pss_file_bytes'] == 25 * 1024
    assert measured['rss_bytes'] == 100 * 1024
    assert measured['clean_resident_bytes'] == 15 * 1024
    assert measured['dirty_resident_bytes'] == 85 * 1024


def test_gpu_memory_counts_only_own_tree_and_never_other_gpus(monkeypatch):
    listing = f'GPU-other, 99, 23000\n{original.GPU_UUID}, 10, 100\n{original.GPU_UUID}, 11, 200\n'
    monkeypatch.setattr(low.subprocess, 'check_output', lambda *args, **kwargs: listing)
    measured = low.external_gpu_memory({10})
    assert measured['own_gpu_bytes'] == 100 * 1024 ** 2
    assert measured['other_gpu_pids'] == [11]


@pytest.mark.parametrize('field,value,gpu', [('mem_available_gib', 7.99, False),
    ('pss_anon_bytes', 6 * low.GIB + 1, False), ('own_gpu_bytes', 14 * low.GIB + 1, True),
    ('other_gpu_pids', [42], True)])
def test_guard_only_limits_new_child_and_host_reserve(field, value, gpu):
    sample = dict(mem_available_gib=8, pss_anon_bytes=6 * low.GIB,
                  own_gpu_bytes=14 * low.GIB, other_gpu_pids=[])
    assert low.guard_violation(sample, gpu=gpu) is None
    sample[field] = value
    assert low.guard_violation(sample, gpu=gpu)


def test_gpu_guard_error_reports_new_cap_and_permits_previous_smoke_peak():
    sample = dict(mem_available_gib=16, pss_anon_bytes=3 * low.GIB,
                  own_gpu_bytes=11.01 * low.GIB, other_gpu_pids=[])
    assert low.guard_violation(sample, gpu=True) is None
    sample['own_gpu_bytes'] = 14 * low.GIB + 1
    assert low.guard_violation(sample, gpu=True).endswith('14 GiB')


def profile(anon_gib=2):
    return dict(status='complete', returncode=0, sample_count=10,
        host_mem_available_min_gib=10, peak=dict(pss_anon_bytes=anon_gib * low.GIB,
        rss_bytes=12 * low.GIB, pss_file_bytes=9 * low.GIB, own_gpu_bytes=4 * low.GIB))


def test_formal_gate_requires_all_smokes_and_measures_anon_not_file_rss():
    profiles = {f'smoke_dim{dim}': profile() for dim in original.DIMENSIONS}
    gate = low.measured_gate(profiles)
    assert gate['mem_available_gib'] == 18
    profiles['smoke_dim100'] = profile(6)
    assert low.measured_gate(profiles)['mem_available_gib'] == 19
    del profiles['smoke_dim1']
    with pytest.raises(RuntimeError, match='all four'):
        low.measured_gate(profiles)


@pytest.mark.parametrize('field,value', [('status', 'failed'), ('returncode', 1),
    ('sample_count', 0), ('host_mem_available_min_gib', 7.9), ('violation', 'reserve')])
def test_profile_rejects_failed_or_missing_measurements(field, value):
    record = profile()
    record[field] = value
    with pytest.raises(RuntimeError):
        low.profile_receipt(record)


def test_terminate_refuses_pid_reuse_wrong_uid_and_unowned_session(monkeypatch):
    identity = dict(pid=123, starttime_ticks='1', uid=low.os.getuid(), session=123, process_group=123)
    monkeypatch.setattr(low.os, 'killpg', lambda *args: pytest.fail('Must not signal unrelated process'))
    child = SimpleNamespace(pid=123)
    for replacement in (None, dict(identity, starttime_ticks='2'), dict(identity, uid=999999),
                        dict(identity, session=10), dict(identity, process_group=10)):
        monkeypatch.setattr(low, 'process_identity', lambda pid, replacement=replacement: replacement)
        assert low.terminate_own_session(child, identity)['signal_sent'] is False


def test_terminate_only_own_session_term_before_kill(monkeypatch):
    identity = dict(pid=123, starttime_ticks='1', uid=low.os.getuid(), session=123, process_group=123)
    monkeypatch.setattr(low, 'process_identity', lambda pid: identity)
    signals, waits = [], []
    monkeypatch.setattr(low.os, 'killpg', lambda pid, sig: signals.append((pid, sig)))

    def wait(timeout):
        waits.append(timeout)
        if len(waits) == 1:
            raise low.subprocess.TimeoutExpired('own child', timeout)
        return -9

    child = SimpleNamespace(pid=123, returncode=-9, wait=wait)
    result = low.terminate_own_session(child, identity)
    assert signals == [(123, low.signal.SIGTERM), (123, low.signal.SIGKILL)]
    assert waits == [5, 5] and result['returncode'] == -9


def test_monitor_records_file_separately_and_never_stops_healthy_child(tmp_path, monkeypatch):
    identity = dict(pid=123, starttime_ticks='1')
    observations = iter([None, None, 0])
    child = SimpleNamespace(pid=123, returncode=0, poll=lambda: next(observations))
    monkeypatch.setattr(low, 'process_identity', lambda pid: identity)
    monkeypatch.setattr(low, 'child_tree', lambda *args: [identity])
    monkeypatch.setattr(low, 'tree_memory', lambda tree: dict(pss_anon_bytes=2 * low.GIB,
        rss_bytes=12 * low.GIB, pss_file_bytes=9 * low.GIB, pids=[123]))
    monkeypatch.setattr(low, 'proc_kib', lambda path: dict(MemAvailable=12 * low.GIB))
    monkeypatch.setattr(low, 'external_gpu_memory', lambda pids: dict(own_gpu_bytes=4 * low.GIB, other_gpu_pids=[]))
    monkeypatch.setattr(low.time, 'sleep', lambda seconds: None)
    monkeypatch.setattr(low, 'terminate_own_session', lambda *args: pytest.fail('Healthy child'))
    result = low.monitor(child, gpu=True, profile_path=tmp_path / 'profile.json')
    assert result['sample_count'] == 2 and result['status'] == 'complete'
    assert result['peak']['pss_anon_bytes'] == 2 * low.GIB
    assert result['peak']['pss_file_bytes'] == 9 * low.GIB


def test_monitor_failclosed_preserves_violation_profile(tmp_path, monkeypatch):
    identity = dict(pid=123, starttime_ticks='1')
    child = SimpleNamespace(pid=123, returncode=-15, poll=lambda: None)
    monkeypatch.setattr(low, 'process_identity', lambda pid: identity)
    monkeypatch.setattr(low, 'child_tree', lambda *args: [identity])
    monkeypatch.setattr(low, 'tree_memory', lambda tree: dict(pss_anon_bytes=1 * low.GIB, rss_bytes=low.GIB, pids=[123]))
    monkeypatch.setattr(low, 'proc_kib', lambda path: dict(MemAvailable=7 * low.GIB))
    stopped = []
    monkeypatch.setattr(low, 'terminate_own_session', lambda *args: stopped.append(123) or dict(signal_sent=True))
    with pytest.raises(RuntimeError, match='reserve'):
        low.monitor(child, gpu=False, profile_path=tmp_path / 'profile.json')
    saved = original.read(tmp_path / 'profile.json')
    assert saved['status'] == 'failed' and saved['violation']
    assert stopped == [123]


def test_exact_required_cache_only_profile_entrypoint(tmp_path, monkeypatch):
    from scripts import run_ising2d_dimg_ablation as prior
    received = []
    monkeypatch.setattr(prior, 'cached_training', lambda args: received.append(args))
    low.main(['--cached-training', '--output', str(tmp_path / 'out'), '--config', 'config.json',
        '--device', 'cuda:2', '--mi-cache-dir', 'existing', '--initialization-receipt', 'fresh.json',
        '--benchmark-warmup', '0'])
    assert received[0].benchmark_warmup == 0
    with pytest.raises(SystemExit):
        low.main(['--cached-training', '--output', str(tmp_path / 'out'), '--device', 'cuda:0'])


def mock_execution(source, handoff, tmp_path, monkeypatch, fail_at=None):
    old, output, _ = handoff
    plan = low.build_plan(source[0], output, old)
    project = tmp_path / 'project'
    script = project / 'scripts' / 'placeholder.py'
    script.parent.mkdir(parents=True)
    script.write_text('# snapshot stand-in, never executed\n')
    monkeypatch.setattr(low, '__file__', str(project / 'scripts' / 'run_ising_highdim_low_ram.py'))
    monkeypatch.setattr(original, 'source_hashes', lambda root: {'scripts/placeholder.py': original.sha(script)})
    monkeypatch.setattr(low.socket, 'gethostname', lambda: original.HOST)
    monkeypatch.setattr(low.os, 'getpriority', lambda *args: 10)
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    calls = []

    def fake_stage(root, job, name, command, **kwargs):
        calls.append(name)
        if name.startswith('train_'):
            assert job.get('all_smoke_checks_passed') is True
            assert job.get('measured_resource_gate', {}).get('mem_available_gib') == 18
            assert calls[:5] == ['cuda_verification', 'smoke_dim1', 'smoke_dim5', 'smoke_dim10', 'smoke_dim100']
        if name == fail_at:
            job['stages'].append(dict(name=name, status='failed', returncode=7))
            raise RuntimeError(f'{name} returncode=7; no retry')
        job['stages'].append(dict(name=name, status='complete', returncode=0))
        if name.startswith('smoke_'):
            write_json(root / 'profiles' / f'{name}.json', profile())
        if name.startswith('report_') or name == 'comparison':
            folder = root / 'comparison' if name == 'comparison' else root / name.removeprefix('report_') / 'report'
            write_json(folder / 'report.json', dict(status='complete', report_complete=True))

    monkeypatch.setattr(low, 'stage', fake_stage)
    return plan, calls


def test_cuda_then_four_real_smokes_then_formal_and_reports_with_measured_gate(source, handoff, tmp_path, monkeypatch):
    plan, calls = mock_execution(source, handoff, tmp_path, monkeypatch)
    job = low.execute(plan)
    expected = ['cuda_verification', *[f'smoke_dim{dim}' for dim in original.DIMENSIONS]]
    expected += [name for dim in original.DIMENSIONS for name in (f'train_dim{dim}', f'report_dim{dim}')]
    assert calls == [*expected, 'comparison']
    assert job['status'] == 'complete' and job['workflow_complete']
    assert job['all_smoke_checks_passed'] and job['physical_sampling_updates'] == 0
    assert all(Path(name).suffix != '.pt' for name in job['public_files'])
    for arm in plan['arms']:
        config, smoke = original.read(arm['config_path']), original.read(arm['smoke_config_path'])
        expected_smoke = deepcopy(config)
        expected_smoke['training'].update(epochs=2, log_every=1, checkpoint_every=1)
        assert smoke == expected_smoke
    with pytest.raises(FileExistsError):
        low.execute(plan)


@pytest.mark.parametrize('failure', ['cuda_verification', 'smoke_dim100', 'train_dim5', 'report_dim10', 'comparison'])
def test_execution_any_failure_stops_later_stages_and_never_retries(source, handoff, tmp_path, monkeypatch, failure):
    plan, calls = mock_execution(source, handoff, tmp_path, monkeypatch, fail_at=failure)
    with pytest.raises(RuntimeError, match=failure):
        low.execute(plan)
    job = original.read(Path(plan['output']) / 'job.json')
    assert job['status'] == 'failed' and job['workflow_complete'] is False
    assert job['requires_user_action'] and calls[-1] == failure and calls.count(failure) == 1
    if failure == 'cuda_verification' or failure.startswith('smoke_'):
        assert 'all_smoke_checks_passed' not in job
        assert not any(name.startswith('train_') for name in calls)


def test_missing_smoke_profile_prevents_every_formal_arm(source, handoff, tmp_path, monkeypatch):
    plan, calls = mock_execution(source, handoff, tmp_path, monkeypatch)
    original_fake_stage = low.stage

    def missing_one(*args, **kwargs):
        original_fake_stage(*args, **kwargs)
        if args[2] == 'smoke_dim100':
            (Path(plan['output']) / 'profiles' / 'smoke_dim100.json').unlink()

    monkeypatch.setattr(low, 'stage', missing_one)
    with pytest.raises(FileNotFoundError):
        low.execute(plan)
    assert not any(name.startswith('train_') for name in calls)
