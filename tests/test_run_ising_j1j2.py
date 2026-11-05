"""Read-only runner contract tests; all subprocesses/GPU checks are mocked."""

import json
from types import SimpleNamespace

import pytest

from scripts import run_ising_j1j2 as runner


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.socket, 'gethostname', lambda: runner.HOST)
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    monkeypatch.setattr(runner, 'source_hashes', lambda root: {'frozen.py': 'a' * 64})
    root = tmp_path / 'campaign'
    runner.prepare(root)
    plan = runner.read(root / 'plan.json')
    return root, plan


def test_prepare_pins_current_recipe_labels_boundaries_and_registered_only_stageA(prepared):
    root, plan = prepared
    assert plan['physical']['size'] == 108
    assert plan['physical']['groups'] == 12
    assert plan['physical']['train_groups'] == 8
    assert plan['pilot']['production_reuse'] is False
    assert plan['future_stages']['automatic_execution'] is False
    assert set(plan['configs']) == {'dg5_seed91'}
    assert plan['protocol']['training_seeds'] == [91]
    assert plan['protocol']['labels_used_in_training'] is False
    assert plan['resource_policy']['automatic_retry'] is False
    assert plan['resource_policy']['gpu_sharing_authorized'] is True
    assert plan['resource_policy']['eligible_gpu_uuids'] == {'3': runner.GPU_UUIDS[3]}
    assert plan['resource_policy']['gpu_free_mib'] >= 16384
    for config in plan['configs'].values():
        assert config['schema_version'] == 5
        assert config['data']['coupling_names'] == []
        assert config['data']['spin_flip_augment'] is False
        assert config['model']['block_factor'] == 3
        assert config['model']['kernel_initialization'] == 'random'
        assert config['model']['fixed_kernel'] is False
        assert config['training']['epochs'] == 12000
        assert config['training']['gradient_mode'] == 'qc_r_separate'
        assert config['loss']['closure_objective'] == 'all_start_anchor'
        assert config['loss']['closure_scale_weights'] is None
    verified_plan, job = runner.verify(root)
    assert verified_plan == plan
    assert job['status'] == 'prepared'
    assert job['training'] == {}


def test_prepare_and_verification_reject_overwrite_config_source_and_plan_mutation(prepared, monkeypatch):
    root, plan = prepared
    with pytest.raises(FileExistsError):
        runner.prepare(root)
    config = root / 'configs/dg5_seed91.json'
    saved = config.read_text()
    value = json.loads(saved)
    value['training']['epochs'] = 3
    write_json(config, value)
    with pytest.raises(RuntimeError, match='configuration'):
        runner.verify(root)
    config.write_text(saved)
    monkeypatch.setattr(runner, 'source_hashes', lambda root: {'frozen.py': 'b' * 64})
    with pytest.raises(RuntimeError, match='source'):
        runner.verify(root)
    monkeypatch.setattr(runner, 'source_hashes', lambda root: {'frozen.py': 'a' * 64})
    plan['success']['latent_nrmse_max'] = .2
    write_json(root / 'plan.json', plan)
    with pytest.raises(RuntimeError, match='mismatch'):
        runner.verify(root)


def test_cuda_mask_or_different_host_are_rejected(prepared, monkeypatch, tmp_path):
    root, _ = prepared
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    with pytest.raises(ValueError):
        runner.prepare(tmp_path / 'masked')
    with pytest.raises(RuntimeError):
        runner.verify(root)
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES')
    monkeypatch.setattr(runner.socket, 'gethostname', lambda: 'unregistered-host')
    with pytest.raises(RuntimeError):
        runner.verify(root)


def test_stage_schedule_has_no_implicit_future_arms_or_legacy_entrypoints(prepared):
    root, plan = prepared
    stages = runner.stage_list(root, plan)
    names = [name for name, _, _ in stages]
    assert names[:2] == ['sampler_benchmark', 'independent_pilot']
    assert len([name for name in names if name.startswith('sample_')]) == 84
    assert names[-2:] == ['dg5_seed91_training', 'dg5_seed91_report']
    assert sum(gpu for _, _, gpu in stages) == 2
    for _, command, _ in stages:
        assert not {'--source', '--resume-checkpoint', '--allow-nondefault-recipe',
                    '--allow-legacy-training'} & set(command)


def test_launch_is_exclusive_and_never_restarts_existing_receipt(prepared, monkeypatch):
    root, _ = prepared
    calls = []
    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(pid=9234)
    monkeypatch.setattr(runner.subprocess, 'Popen', fake_popen)
    receipt = runner.launch(root)
    assert receipt['parent_pid'] == 9234
    assert receipt['automatic_retry'] is False
    assert receipt['resumed'] is False
    assert len(calls) == 1
    assert calls[0][1]['start_new_session'] is True
    with pytest.raises((FileExistsError, RuntimeError)):
        runner.launch(root)
    assert len(calls) == 1


def ready_snapshot(*ready):
    return dict(host_ready=True, gpus=[
        dict(index=index, uuid=uuid, free_mib=24576, pids=[], ready=index in ready)
        for index, uuid in runner.GPU_UUIDS.items()
    ])


def test_gpu_admission_requires_same_device_twice_and_pins_binding(prepared, monkeypatch):
    root, plan = prepared
    job = runner.read(root / 'job.json')
    snapshots = iter([ready_snapshot(), ready_snapshot(3), ready_snapshot(3)])
    waits = []
    monkeypatch.setattr(runner, 'resource_snapshot', lambda plan: next(snapshots))
    monkeypatch.setattr(runner.time, 'sleep', waits.append)
    runner.admission(root, plan, job, 'profile', gpu=True)
    binding = runner.read(root / 'device_binding.json')
    assert binding['device'] == 'cuda:3'
    assert binding['gpu_uuid'] == runner.GPU_UUIDS[3]
    assert waits == [30, 10]
    # A busy bound device cannot be replaced by another idle GPU.
    snapshots = iter([ready_snapshot(2), ready_snapshot(3), ready_snapshot(3)])
    monkeypatch.setattr(runner, 'resource_snapshot', lambda plan: next(snapshots))
    waits.clear()
    runner.admission(root, plan, job, 'training', gpu=True)
    assert waits == [30, 10]
    assert runner.read(root / 'device_binding.json') == binding


@pytest.mark.parametrize('free,utilization,host_load,expected', [
    (20480, 0, 12., True), (16383, 0, 12., False),
    (20480, 11, 12., False), (20480, 0, 31., False),
])
def test_snapshot_limits_sharing_to_gpu3_and_preserves_other_process_evidence(
        prepared, monkeypatch, free, utilization, host_load, expected):
    _, plan = prepared
    original_read = runner.Path.read_text
    def fake_read(path, *args, **kwargs):
        if str(path) == '/proc/meminfo':
            return 'MemAvailable: 100000000 kB\n'
        if str(path) == '/proc/pressure/io':
            return 'some avg10=0.00 avg60=0.00 avg300=0.00 total=0\nfull avg10=0.00 avg60=0.00 avg300=0.00 total=0\n'
        return original_read(path, *args, **kwargs)
    monkeypatch.setattr(runner.Path, 'read_text', fake_read)
    monkeypatch.setattr(runner.os, 'cpu_count', lambda: 24)
    monkeypatch.setattr(runner.os, 'getloadavg', lambda: (host_load, 0., 0.))
    def fake_query(command, **kwargs):
        if '--query-compute-apps=gpu_uuid,pid' in command:
            return f'{runner.GPU_UUIDS[3]}, 9876\n{runner.GPU_UUIDS[3]}, 9877\n'
        return '\n'.join(f'{index}, {uuid}, {free}, {utilization}'
                         for index, uuid in runner.GPU_UUIDS.items())
    monkeypatch.setattr(runner.subprocess, 'check_output', fake_query)
    snapshot = runner.resource_snapshot(plan)
    assert [row['index'] for row in snapshot['gpus'] if row['ready']] == ([3] if expected else [])
    assert snapshot['gpus'][3]['pids'] == [9876, 9877]


def profile_summary():
    return dict(timing=dict(steady_step_seconds=[.01, .012], setup_seconds=5.,
        logging_seconds=2., checkpoint_seconds=.2,
        cuda_max_memory_reserved_bytes=1024**3))


def test_profile_projection_counts_measured_usage_and_rejects_memory_or_budget(prepared):
    _, plan = prepared
    summary = profile_summary()
    result = runner.profile_projection(summary, 77., plan)
    assert result['budget_passed']
    assert result['memory_passed']
    assert result['profile_is_scientific_result'] is False
    assert result['projected_total_gpu_seconds'] == pytest.approx(
        77. + len(plan['configs']) * result['projected_gpu_seconds_per_arm'])
    summary['timing']['cuda_max_memory_reserved_bytes'] = 13 * 1024**3
    assert not runner.profile_projection(summary, 77., plan)['memory_passed']
    plan['resource_policy']['gpu_seconds'] = 1
    assert not runner.profile_projection(summary, 77., plan)['budget_passed']


@pytest.mark.parametrize('values', [[], [0.], [-1.], [float('inf')], [float('nan')]])
def test_profile_rejects_missing_or_nonfinite_steady_timing(prepared, values):
    _, plan = prepared
    summary = profile_summary()
    summary['timing']['steady_step_seconds'] = values
    with pytest.raises(ValueError):
        runner.profile_projection(summary, 0., plan)


def fake_training(root, plan, *, actual=12000, stopped=False):
    arm = 'dg5_seed91'
    folder = root / arm / 'run'
    config = plan['configs'][arm]
    summary = dict(actual_epochs=actual, requested_epochs=12000, config=config,
        stopped_by_training_limit=stopped, stopped_by_wall_limit=False,
        stopped_by_train_plateau=False,
        timing=dict(cuda_max_memory_reserved_bytes=1024**3))
    manifest = dict(config=config, labels_used_in_training=False, independent_c=True,
        source=None, resumed_from=None, q_supervision=None,
        data_driven_initialization=None, frozen_C=None, data_sha256='c' * 64)
    write_json(root / 'data/metadata.json', dict(data_sha256='c' * 64))
    write_json(folder / 'summary.json', summary)
    write_json(folder / 'manifest.json', manifest)
    (folder / f'training_step_{actual}.pt').write_bytes(b'not-a-real-model')
    return folder, summary, manifest


def test_training_endpoint_and_label_boundary_are_verified(prepared):
    root, plan = prepared
    folder, summary, manifest = fake_training(root, plan)
    result = runner.validate_training(root, plan, 'dg5_seed91')
    assert result['status'] == 'complete'
    assert result['actual_updates'] == 12000
    manifest['labels_used_in_training'] = True
    write_json(folder / 'manifest.json', manifest)
    with pytest.raises(RuntimeError):
        runner.validate_training(root, plan, 'dg5_seed91')


def test_safety_clock_partial_training_is_budget_hold_not_execution_failure(prepared):
    root, plan = prepared
    fake_training(root, plan, actual=37, stopped=True)
    result = runner.validate_training(root, plan, 'dg5_seed91')
    assert result['status'] == 'budget_hold'


def execute_mocks(root, plan, monkeypatch, stages, returncodes):
    monkeypatch.setattr(runner.os, 'nice', lambda value: 0)
    monkeypatch.setattr(runner.os, 'getpriority', lambda *args: 10)
    monkeypatch.setattr(runner, 'cpu_seconds', lambda: 0.)
    monkeypatch.setattr(runner, 'directory_bytes', lambda root: 0)
    monkeypatch.setattr(runner, 'admission', lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, 'stage_list', lambda root, plan: stages)
    codes = iter(returncodes)
    launched = []
    def fake_popen(command, **kwargs):
        launched.append(command)
        code = next(codes)
        return SimpleNamespace(pid=1234 + len(launched), wait=lambda: code)
    monkeypatch.setattr(runner.subprocess, 'Popen', fake_popen)
    return launched


def test_nonzero_child_preserves_returncode_and_never_starts_next_stage(prepared, monkeypatch):
    root, plan = prepared
    stages = [('unit_stage', ['mock-first'], False), ('later', ['must-not-run'], False)]
    launched = execute_mocks(root, plan, monkeypatch, stages, [7])
    with pytest.raises(RuntimeError, match='returned 7'):
        runner.execute(root)
    job = runner.read(root / 'job.json')
    assert launched == [['mock-first']]
    assert job['status'] == 'failed'
    assert job['workflow_complete'] is False
    assert job['stages'][0]['returncode'] == 7


def test_pilot_scientific_stop_is_not_failed_and_never_starts_production(prepared, monkeypatch):
    root, plan = prepared
    write_json(root / 'data/pilot/receipt.json', dict(status='scientific_stop'))
    stages = [('independent_pilot', ['mock-pilot'], False), ('sample', ['must-not-run'], False)]
    launched = execute_mocks(root, plan, monkeypatch, stages, [0])
    runner.execute(root)
    job = runner.read(root / 'job.json')
    assert launched == [['mock-pilot']]
    assert job['status'] == 'scientific_stop'
    assert job['workflow_complete'] is True
    assert job['requires_user_action'] is True
    assert job['stages'][0]['returncode'] == 0


def test_sampling_budget_receipt_blocks_production_without_training_failure(prepared, monkeypatch):
    root, plan = prepared
    write_json(root / 'data/benchmark/receipt.json',
               dict(status='complete', conservative_production_cpu_seconds=1e9))
    stages = [('sampler_benchmark', ['mock-benchmark'], False), ('pilot', ['must-not-run'], False)]
    launched = execute_mocks(root, plan, monkeypatch, stages, [0])
    runner.execute(root)
    job = runner.read(root / 'job.json')
    assert launched == [['mock-benchmark']]
    assert job['status'] == 'budget_hold'
    assert job['workflow_complete'] is False


def test_execution_lock_blocks_second_direct_executor_even_with_prepared_job(prepared, monkeypatch):
    root, plan = prepared
    write_json(root / 'execution.json', dict(parent_pid=4567, exclusive_execution_lock=True))
    launched = execute_mocks(root, plan, monkeypatch, [('one', ['must-not-run'], False)], [])
    with pytest.raises(FileExistsError):
        runner.execute(root)
    assert launched == []


def test_nonzero_report_safety_deadline_is_budget_hold_with_actual_returncode(prepared, monkeypatch):
    root, plan = prepared
    write_json(root / 'dg5_seed91/report/status.json', dict(status='budget_hold', report_complete=False))
    stages = [('dg5_seed91_report', ['mock-report'], False), ('later', ['must-not-run'], False)]
    launched = execute_mocks(root, plan, monkeypatch, stages, [1])
    runner.execute(root)
    job = runner.read(root / 'job.json')
    assert launched == [['mock-report']]
    assert job['status'] == 'budget_hold'
    assert job['workflow_complete'] is False
    assert job['stages'][0]['returncode'] == 1
    assert job['stages'][0]['status'] == 'analysis_budget_hold'


REPORT_FILES = {'report.json', 'evaluation_arrays.npz', 'loss_history.json',
                'pca_explained.png', 'R_displacement.png', 'coupling_to_g.png',
                'loss_curves.png', 'physical_C_statistics.png'}


def complete_report(folder):
    for name in REPORT_FILES:
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'fixture')
    write_json(folder / 'report.json', dict(status='complete', report_complete=True,
                                           checkpoint_step=12000, thermal_nu=None))
    status = dict(status='complete', report_complete=True, public_files=sorted(REPORT_FILES),
                  sha256={name: runner.sha(folder / name) for name in REPORT_FILES})
    write_json(folder / 'status.json', status)
    return status


def test_report_requires_complete_whitelist_and_digests_not_just_zero_exit(prepared):
    root, _ = prepared
    folder = root / 'dg5_seed91/report'
    status = complete_report(folder)
    assert runner.validate_report(folder)['status'] == 'complete'
    (folder / 'evaluation_arrays.npz').write_bytes(b'mutated')
    with pytest.raises(RuntimeError, match='digest'):
        runner.validate_report(folder)
    status['sha256'] = {}
    status['public_files'] = []
    write_json(folder / 'status.json', status)
    with pytest.raises(RuntimeError):
        runner.validate_report(folder)


def test_report_rejects_symlinked_outputs_even_with_matching_digest(prepared):
    root, _ = prepared
    folder = root / 'dg5_seed91/report'
    status = complete_report(folder)
    outside = root / 'outside.png'
    outside.write_bytes(b'fixture')
    plot = folder / 'pca_explained.png'
    plot.unlink()
    plot.symlink_to(outside)
    with pytest.raises(RuntimeError):
        runner.validate_report(folder)


def test_zero_exit_incomplete_report_is_still_failed(prepared, monkeypatch):
    root, plan = prepared
    write_json(root / 'dg5_seed91/report/status.json', dict(status='running', report_complete=False))
    stages = [('dg5_seed91_report', ['mock-incomplete-report'], False), ('later', ['must-not-run'], False)]
    launched = execute_mocks(root, plan, monkeypatch, stages, [0])
    with pytest.raises((RuntimeError, FileNotFoundError)):
        runner.execute(root)
    assert launched == [['mock-incomplete-report']]
    job = runner.read(root / 'job.json')
    assert job['status'] == 'failed'
    assert job['workflow_complete'] is False


def test_execution_profile_is_not_reported_as_formal_training_started():
    job = dict(training={'profile': dict(status='complete')})
    runner.hold(job, 'Profile gate rejected')
    assert job['training_started'] is False


def test_public_file_whitelist_excludes_models_mi_and_raw_npz(prepared):
    root, _ = prepared
    for path in ['data/physical.npz', 'data/training.pt', 'data/mi_cache/info.json',
                 'dg5_seed91/run/training_step_12000.pt', 'dg5_seed91/report/evaluation_arrays.npz',
                 'dg5_seed91/report/report.json', 'dg5_seed91/report/流场.png']:
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b'fixture')
    paths = {row['path'] for row in runner.public_files(root)}
    assert 'dg5_seed91/report/evaluation_arrays.npz' in paths
    assert 'dg5_seed91/report/流场.png' in paths
    assert 'data/physical.npz' not in paths
    assert 'data/training.pt' not in paths
    assert 'data/mi_cache/info.json' not in paths
    assert all(not path.endswith('.pt') for path in paths)
