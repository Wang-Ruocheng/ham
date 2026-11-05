"""Explicit continuation contract; no CUDA, physical data or MI tensor loads."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from recurrent_rg.config import RunConfig
from recurrent_rg.learning_rate import LearningRatePolicy
from scripts import run_ising_highdim as original
from scripts import run_ising_highdim_low_ram as low
from scripts import run_ising_highdim_resume as resume
from test_ising_highdim_runner import source  # noqa: F401


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value))


@pytest.fixture
def interrupted(source, tmp_path, monkeypatch):
    old, output = tmp_path / 'v3', tmp_path / 'v4'
    plan = original.build_plan(source[0], old)
    plan.update(execution_version='lowram_v3', source_code_sha256={'recurrent_rg/model.py': 'same'},
        configuration_sha256={'frozen_config': 'same'}, low_ram_execution={
            'own_gpu_hard_limit_gib': 14, 'host_available_hard_floor_gib': 8,
            'child_tree_pss_anon_hard_limit_gib': 6})
    write(old / 'plan.json', plan)
    plan = original.read(old / 'plan.json')
    stages = [dict(name=name, status='complete', returncode=0, pid=200 + index)
              for index, name in enumerate(resume.EXPECTED_OLD_STAGES)]
    stages[-1].update(status='failed', returncode=-15, error=resume.PREVIOUS_ERROR, pid=8588)
    job = dict(status='failed', workflow_complete=False, requires_user_action=True,
        error=resume.PREVIOUS_ERROR, current='train_dim1', completed_arms=[],
        all_smoke_checks_passed=True, host=original.HOST, device='cuda:2', gpu_uuid=original.GPU_UUID,
        physical_sampling_updates=0, labels_used_in_training=False, parent_pid=7339,
        execution_version='lowram_v3', stages=stages)
    write(old / 'job.json', job)
    profile = dict(status='failed', returncode=-15,
        violation=resume.PREVIOUS_ERROR.removeprefix('RuntimeError: '),
        monitor_error=resume.PREVIOUS_ERROR, child_pid=8588, child_starttime_ticks='100005')
    write(old / 'profiles/train_dim1.json', profile)
    manifest = dict(source=None, resumed_from=None, config=plan['arms'][0]['config'])
    write(old / 'dim1/run/manifest.json', manifest)
    write(old / 'dim1/run/history_progress.json', [dict(epoch=5200)])
    checkpoint = old / 'dim1/run/training_step_5000.pt'
    checkpoint.write_bytes(b'tiny explicitly selected model checkpoint; never a physical cache')
    for arm in plan['arms']:
        write(old / 'profiles' / f'smoke_{arm["name"]}.json', dict(status='complete', returncode=0,
            sample_count=2, host_mem_available_min_gib=9,
            peak=dict(pss_anon_bytes=3 * low.GIB, own_gpu_bytes=11 * low.GIB)))
        write(old / 'smoke' / arm['name'] / 'initialization.json', dict(fresh=True, source_checkpoint=None))
    write(old / 'cuda_verification.json', dict(passed=True))
    write(old / 'dim1/initialization.json', dict(fresh=True, source_checkpoint=None))
    handoff = dict(schema_version=1, status='approved_exact_checkpoint_continuation',
        user_authorized=True, termination_confirmed=True, old_directory_preserved=True,
        new_root=str(output), checkpoint_path=str(checkpoint), checkpoint_step=5000, last_logged_step=5200,
        parent_pid=7339, child_pid=8588, child_starttime_ticks='100005',
        formal_other_arms_started=False, physical_sampling_updates=0,
        resource_change=deepcopy(resume.RESOURCE_CHANGE))
    for key, path in dict(old_plan_sha256=old / 'plan.json', old_job_sha256=old / 'job.json',
            old_profile_sha256=old / 'profiles/train_dim1.json', old_manifest_sha256=old / 'dim1/run/manifest.json',
            checkpoint_sha256=checkpoint).items():
        handoff[key] = original.sha(path)
    write(old / 'resume_handoff.json', handoff)
    monkeypatch.setattr(low, 'process_identity', lambda pid: None)
    monkeypatch.setattr(original, 'ensure_unchanged', lambda *args: None)
    monkeypatch.setattr(low, 'validate_training', lambda *args, **kwargs: dict(passed=True))
    monkeypatch.setattr(low, 'cuda_receipt', lambda path: dict(passed=True))
    selected = dict(path=str(checkpoint), sha256=original.sha(checkpoint), actual_updates=5000,
        all_adam_parameter_steps=5000, adam_state_count=24, cuda_device_index=2,
        model_tensor_sha256='model', history_sha256='history', history=[], metadata=manifest,
        exact_rng_and_optimizer_present=True)
    monkeypatch.setattr(resume, 'checkpoint_receipt', lambda *args, **kwargs: deepcopy(selected))
    return old, output, plan, job, profile, handoff


def test_plan_only_continues_dim1_from_saved_not_logged_step(interrupted):
    old, output, old_plan, *_ = interrupted
    plan = resume.build_plan(old, output)
    assert plan['execution_version'] == 'resume_v4'
    assert resume.canonical([a['config'] for a in plan['arms']]) == resume.canonical([a['config'] for a in old_plan['arms']])
    assert plan['protocol']['resume'] and not plan['protocol']['automatic_retry']
    assert plan['protocol']['resource_gate']['gpu_free_mib'] == 16384
    assert plan['low_ram_execution']['host_available_hard_floor_gib'] == 6
    assert plan['low_ram_execution']['own_gpu_hard_limit_gib'] == 14
    for arm in plan['arms']:
        assert 'smoke' not in arm
        assert '--cached-training' in arm['training']
        assert 'scripts.run_ising_highdim_resume' in arm['training']
        assert arm['config']['training']['epochs'] == 12000
        if arm['name'] == 'dim1':
            assert arm['segment_start_update'] == 5000 and arm['new_updates'] == 7000
            assert arm['resume_from'] == str(old / 'dim1/run/training_step_5000.pt')
            assert '--resume-checkpoint' in arm['training'] and '--resume-receipt' in arm['training']
        else:
            assert arm['segment_start_update'] == 0 and arm['new_updates'] == 12000
            assert arm['resume_from'] is None and '--resume-checkpoint' not in arm['training']
    assert not output.exists()


@pytest.mark.parametrize('field,value', [('schema_version', 2), ('status', 'failed'),
    ('user_authorized', False), ('termination_confirmed', False), ('old_directory_preserved', False),
    ('new_root', '/different'), ('checkpoint_step', 5200), ('last_logged_step', 5000),
    ('formal_other_arms_started', True), ('physical_sampling_updates', 1), ('resource_change', {}),
    ('parent_pid', 111), ('child_pid', 111), ('child_starttime_ticks', 'notknown'),
    ('old_plan_sha256', 'changed'), ('old_job_sha256', 'changed'), ('old_profile_sha256', 'changed'),
    ('old_manifest_sha256', 'changed'), ('checkpoint_sha256', 'changed')])
def test_reject_changed_or_unauthorized_handoff(interrupted, field, value):
    old, output, *_, handoff = interrupted
    handoff[field] = value
    write(old / 'resume_handoff.json', handoff)
    with pytest.raises(RuntimeError):
        resume.verify_handoff(old, output)


def test_reject_any_recorded_pid_even_if_reused(interrupted, monkeypatch):
    old, output, *_ = interrupted
    monkeypatch.setattr(low, 'process_identity', lambda pid: dict(pid=pid) if pid == 7339 else None)
    with pytest.raises(RuntimeError, match='still present'):
        resume.verify_handoff(old, output)


@pytest.mark.parametrize('mutation', ['other_started', 'later_checkpoint', 'wrong_log', 'existing_output'])
def test_reject_wrong_source_or_duplicate(interrupted, mutation):
    old, output, *_ = interrupted
    if mutation == 'other_started':
        write(old / 'dim5/initialization.json', {'fresh': True})
    elif mutation == 'later_checkpoint':
        (old / 'dim1/run/training_step_6000.pt').write_bytes(b'new')
    elif mutation == 'wrong_log':
        write(old / 'dim1/run/history_progress.json', [{'epoch': 5000}])
    else:
        output.mkdir()
    with pytest.raises(RuntimeError):
        resume.verify_handoff(old, output)


def test_measured_gate_reuses_four_actual_profiles_and_keeps_margin(interrupted):
    old, output, *_ = interrupted
    plan = resume.build_plan(old, output)
    gate = resume.measured_gate(plan)
    assert gate['mem_available_gib'] == 16
    assert gate['host_reserve_gib'] == 6 and gate['measured_max_smoke_pss_anon_gib'] == 3
    assert gate['inherited_independent_preflight']


def test_only_host_reserve_relaxed_and_old_monitor_settings_restored():
    row = dict(mem_available_gib=6, pss_anon_bytes=3 * low.GIB, own_gpu_bytes=11 * low.GIB, other_gpu_pids=[])
    assert resume.guard_violation(row, gpu=True) is None
    assert 'below 6 GiB' in resume.guard_violation(dict(row, mem_available_gib=5.999), gpu=True)
    assert resume.guard_violation(dict(row, pss_anon_bytes=6 * low.GIB + 1), gpu=True)
    assert resume.guard_violation(dict(row, own_gpu_bytes=14 * low.GIB + 1), gpu=True)
    assert resume.guard_violation(dict(row, other_gpu_pids=[123]), gpu=True)
    previous = low.HOST_RESERVE_GIB, low.INITIAL_AVAILABLE_GIB, low.guard_violation
    with resume.continuation_resources():
        assert low.HOST_RESERVE_GIB == 6 and low.INITIAL_AVAILABLE_GIB == 16
        assert low.guard_violation is resume.guard_violation
    assert (low.HOST_RESERVE_GIB, low.INITIAL_AVAILABLE_GIB, low.guard_violation) == previous


def test_source_evidence_stays_pinned_between_stages(interrupted):
    old, output, *_ = interrupted
    plan = resume.build_plan(old, output)
    resume.ensure_source(plan)
    write(old / 'smoke/dim10/initialization.json', {'changed': True})
    with pytest.raises(RuntimeError, match='Inherited preflight'):
        resume.ensure_source(plan)


def test_no_silent_scientific_code_changes(interrupted, monkeypatch):
    old, output, *_ = interrupted
    plan = resume.build_plan(old, output)
    monkeypatch.setattr(original, 'source_hashes', lambda project: {'recurrent_rg/model.py': 'changed'})
    with pytest.raises(RuntimeError, match='cannot change model'):
        resume.validate_code_delta(plan, old / 'code')


def test_strict_synthetic_resume_receipt(tmp_path):
    path = tmp_path / 'verification.json'
    value = dict(status='complete', passed=True, synthetic_only=True, scientific_training=False,
        production_data_or_MI_opened=False, device='cuda:2', physical_uuid=original.GPU_UUID,
        dimension=1, total_updates=4, split_update=2)
    write(path, value)
    assert resume.cuda_receipt(path) == value
    value['production_data_or_MI_opened'] = True
    write(path, value)
    with pytest.raises(RuntimeError):
        resume.cuda_receipt(path)


def checkpoint_fixture(config, step=5000):
    policy = LearningRatePolicy(RunConfig.from_dict(config).training)
    metadata = dict(config=deepcopy(config), training_driver='rsmi_separate_qcr', precision='torch.float64',
        estimator='histogram', environment_mode='full_shell', buffer=1, independent_c=True,
        labels_used_in_training=False, rsmi_weight=2., execution_device='cuda:2', source=None)
    return dict(epoch=step, cuda_device_index=2, metadata=metadata,
        model={'one_parameter': torch.tensor([1.], dtype=torch.float64)},
        optimizer={'state': {0: {'step': torch.tensor(float(step)), 'exp_avg': torch.zeros(1)}},
                   'param_groups': [{'params': [0], 'lr': policy.rate_for_update(step)}]},
        lr_scheduler_state=policy.checkpoint_state(step),
        history=[dict(epoch=index) for index in sorted({1, step, *range(config['training']['log_every'], step + 1,
                                                                        config['training']['log_every'])})],
        **{key: torch.zeros(3, dtype=torch.uint8) for key in
           ('torch_rng_state', 'cuda_rng_state', 'batch_generator_state', 'rsmi_generator_state')})


def test_checkpoint_receipt_reads_only_selected_small_model_and_requires_all_adam_states(source, tmp_path, monkeypatch):
    # The real fixture is deliberately tiny and has no physical configurations.
    config = original.build_plan(source[0], tmp_path / 'unused')['arms'][0]['config']
    path = tmp_path / 'training_step_5000.pt'
    path.write_bytes(b'model-only')
    payload = checkpoint_fixture(config)
    calls = []
    monkeypatch.setattr(torch, 'load', lambda selected, **kwargs: calls.append((selected, kwargs)) or payload)
    receipt = resume.checkpoint_receipt(path, config, step=5000)
    assert receipt['all_adam_parameter_steps'] == 5000 and receipt['adam_state_count'] == 1
    assert calls == [(str(path), dict(map_location='cpu', weights_only=True))]
    payload['optimizer']['state'][0]['step'] = torch.tensor(4999.)
    with pytest.raises(RuntimeError, match='Every Adam'):
        resume.checkpoint_receipt(path, config, step=5000)


@pytest.mark.parametrize('mutation', ['wrong_device', 'missing_rng', 'nonfinite', 'bad_history', 'wrong_config'])
def test_checkpoint_receipt_rejects_inexact_state(source, tmp_path, monkeypatch, mutation):
    config = original.build_plan(source[0], tmp_path / 'unused')['arms'][0]['config']
    path = tmp_path / 'training_step_5000.pt'
    path.write_bytes(b'model-only')
    payload = checkpoint_fixture(config)
    if mutation == 'wrong_device': payload['cuda_device_index'] = 0
    elif mutation == 'missing_rng': payload.pop('rsmi_generator_state')
    elif mutation == 'nonfinite': payload['model']['one_parameter'][0] = float('nan')
    elif mutation == 'bad_history': payload['history'][-1]['epoch'] = 5200
    else: payload['metadata']['config']['training']['learning_rate'] = .2
    monkeypatch.setattr(torch, 'load', lambda *args, **kwargs: payload)
    with pytest.raises(RuntimeError):
        resume.checkpoint_receipt(path, config, step=5000)


def test_cached_wrapper_explicit_resume_arguments_forwarded(tmp_path, monkeypatch):
    from scripts import run_ising2d_dimg_ablation as prior
    captured = []
    monkeypatch.setattr(prior, 'cached_training', lambda args: captured.append(args))
    resume.main(['--cached-training', '--output', str(tmp_path / 'run'),
        '--config', str(tmp_path / 'config'), '--mi-cache-dir', str(tmp_path / 'mi'),
        '--initialization-receipt', str(tmp_path / 'initialization.json'),
        '--resume-checkpoint', str(tmp_path / 'source.pt'), '--resume-receipt', str(tmp_path / 'resume.json')])
    assert captured[0].resume_checkpoint == tmp_path / 'source.pt'
    assert captured[0].resume_receipt == tmp_path / 'resume.json'
    assert captured[0].benchmark_warmup == 0


@pytest.fixture
def endpoint(interrupted, monkeypatch):
    old, output, *_ = interrupted
    plan = resume.build_plan(old, output)
    arm = plan['arms'][0]
    run = Path(arm['root']) / 'run'
    run.mkdir(parents=True)
    policy = LearningRatePolicy(RunConfig.from_dict(arm['config']).training)
    source_manifest = dict(config=arm['config'], execution_optimizations=dict(reference=False,
        selected_empirical_mi=True, low_ram_execution=dict(enabled=True, source_dtype_unchanged=True),
        mi_statistics_cache={split: dict(key=key, hit=True, cache_version=1,
            path=str(Path(plan['mi_cache_dir']) / f'mi-v1-{key}.pt'), load_seconds=10)
            for split, key in plan['mi_cache_keys'].items()}))
    write(old / 'dim1/run/manifest.json', source_manifest)
    manifest = deepcopy(source_manifest)
    manifest.update(resumed_from=arm['resume_from'], config_source=arm['config_path'],
        resume_start_provenance=dict(source_epoch=5000), resume_start_module_sha256={'C': 'c', 'Q': 'q', 'R': 'r'})
    for row in manifest['execution_optimizations']['mi_statistics_cache'].values():
        row['load_seconds'] = .1  # Must not compare new wall-clock measurements.
    history = [dict(epoch=index, learning_rate=policy.rate_for_update(index))
        for index in sorted({1, 12000, *range(200, 12001, 200)})]
    summary = dict(config=arm['config'], history=history, actual_epochs=12000, requested_epochs=12000,
        final_report_skipped=True, lr_scheduler_state=policy.checkpoint_state(12000),
        timing=dict(enabled=True, updates=7000, cuda_max_memory_allocated_bytes=6 * low.GIB,
                    cuda_max_memory_reserved_bytes=11 * low.GIB))
    provenance = dict(status='complete', fresh=False, labels_used_in_training=False,
        source_checkpoint=arm['resume_from'], source_checkpoint_sha256=plan['interrupted_launch']['checkpoint']['sha256'],
        source_epoch=5000, resume_start_module_sha256=manifest['resume_start_module_sha256'])
    write(run / 'manifest.json', manifest)
    write(run / 'summary.json', summary)
    write(run.parent / 'resume.json', provenance)
    def saved(path, config, *, step):
        return dict(sha256=plan['interrupted_launch']['checkpoint']['sha256'] if step == 5000 else 'new-final',
                    all_adam_parameter_steps=step, history=[row for row in history if row['epoch'] <= step])
    monkeypatch.setattr(resume, 'checkpoint_receipt', saved)
    return plan, arm, manifest, summary, provenance


def test_resumed_endpoint_allows_new_cache_load_duration_but_not_new_science(endpoint):
    plan, arm, *_ = endpoint
    result = resume.validate_training(plan, arm)
    assert result['actual_updates'] == result['all_adam_parameter_steps'] == 12000
    assert result['new_updates'] == 7000 and result['segment_start_update'] == 5000


@pytest.mark.parametrize('mutation', ['selected_mi', 'reference', 'cache_key', 'timing', 'new_fresh',
    'resume_hash', 'resume_fresh', 'resume_epoch', 'resume_module', 'history', 'initial_provenance'])
def test_resumed_endpoint_rejects_changed_contract(endpoint, mutation):
    plan, arm, manifest, summary, provenance = endpoint
    run = Path(arm['root']) / 'run'
    if mutation == 'selected_mi': manifest['execution_optimizations']['selected_empirical_mi'] = False
    elif mutation == 'reference': manifest['execution_optimizations']['reference'] = True
    elif mutation == 'cache_key': manifest['execution_optimizations']['mi_statistics_cache']['train']['key'] = 'different'
    elif mutation == 'timing': summary['timing']['updates'] = 12000
    elif mutation == 'new_fresh': write(run.parent / 'initialization.json', {'fresh': True})
    elif mutation == 'resume_hash': provenance['source_checkpoint_sha256'] = 'wrong'
    elif mutation == 'resume_fresh': provenance['fresh'] = True
    elif mutation == 'resume_epoch': provenance['source_epoch'] = 5200
    elif mutation == 'resume_module': provenance['resume_start_module_sha256'] = {'R': 'wrong'}
    elif mutation == 'history': summary['history'][1]['epoch'] = 201
    else: manifest['random_construction_module_sha256'] = {'C': 'wrong'}
    write(run / 'manifest.json', manifest)
    write(run / 'summary.json', summary)
    write(run.parent / 'resume.json', provenance)
    with pytest.raises(RuntimeError):
        resume.validate_training(plan, arm)


@pytest.mark.parametrize('failure', [None, 'resume_cuda_verification', 'train_dim1', 'report_dim1', 'train_dim10'])
def test_execution_orders_one_continuation_then_three_fresh_arms_without_retry(interrupted, monkeypatch, failure):
    old, output, *_ = interrupted
    plan = resume.build_plan(old, output)
    source_file = Path(resume.__file__).resolve()
    monkeypatch.setattr(resume.socket, 'gethostname', lambda: original.HOST)
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    monkeypatch.setattr(resume.os, 'getpriority', lambda *args: 10)
    monkeypatch.setattr(resume, 'validate_code_delta', lambda *args: {'existing_changed_files': []})
    monkeypatch.setattr(original, 'source_hashes', lambda project: {
        'scripts/run_ising_highdim_resume.py': original.sha(source_file)})
    observed = []
    def fake_stage(root, job, name, command, *, gpu, **kwargs):
        assert low.HOST_RESERVE_GIB == 6 and low.guard_violation is resume.guard_violation
        observed.append(name)
        assert ('--resume-checkpoint' in command) == (name == 'train_dim1')
        job['stages'].append({'name': name})
        if name == failure:
            raise RuntimeError('selected stage failed; no retry')
    monkeypatch.setattr(resume, 'stage', fake_stage)
    expected = ['resume_cuda_verification', 'train_dim1', 'report_dim1', 'train_dim5', 'report_dim5',
                'train_dim10', 'report_dim10', 'train_dim100', 'report_dim100', 'comparison']
    old_floor = low.HOST_RESERVE_GIB
    if failure is None:
        job = resume.execute(plan)
        assert job['status'] == 'complete' and job['workflow_complete']
        assert job['completed_arms'] == ['dim1', 'dim5', 'dim10', 'dim100']
        assert observed == expected
        assert not (output / 'dim1/initialization.json').exists()
        assert (output / 'smoke/dim100/initialization.json').exists()
        assert all(Path(path).suffix in ('.json', '.png', '.npz') for path in job['public_files'])
    else:
        with pytest.raises(RuntimeError, match='selected stage failed'):
            resume.execute(plan)
        job = original.read(output / 'job.json')
        assert job['status'] == 'failed' and job['requires_user_action']
        assert observed == expected[:expected.index(failure) + 1]
    assert low.HOST_RESERVE_GIB == old_floor
    assert original.read(old / 'job.json')['status'] == 'failed'
