"""Read-only continuation contracts; no SSH, CUDA, physical data or MI loads."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts import run_ising_highdim as original
from scripts import run_ising_highdim_low_ram as low
from scripts import run_ising_highdim_resume as v4
from scripts import run_ising_highdim_resume_v5 as resume
from test_ising_highdim_runner import source  # noqa: F401


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_guard_only_relaxes_shared_host_floor_to_three_gib():
    row = dict(mem_available_gib=3., pss_anon_bytes=6 * low.GIB,
               own_gpu_bytes=14 * low.GIB, other_gpu_pids=[])
    assert resume.guard_violation(row, gpu=True) is None
    assert 'below 3 GiB' in resume.guard_violation(dict(row, mem_available_gib=2.999), gpu=True)
    assert resume.guard_violation(dict(row, pss_anon_bytes=6 * low.GIB + 1), gpu=True)
    assert resume.guard_violation(dict(row, own_gpu_bytes=14 * low.GIB + 1), gpu=True)
    assert resume.guard_violation(dict(row, other_gpu_pids=[567]), gpu=True)
    assert resume.guard_violation(dict(row, own_gpu_bytes=14 * low.GIB + 1,
                                     other_gpu_pids=[567]), gpu=False) is None
    assert resume.INITIAL_AVAILABLE_GIB == 16
    assert resume.RESOURCE_CHANGE == dict(previous_host_floor_gib=6, host_floor_gib=3,
        own_anon_limit_gib=6, own_gpu_limit_gib=14, minimum_free_gpu_gib=16,
        initial_available_gib=16, scientific_configuration_unchanged=True)


@pytest.mark.parametrize('fail', [False, True])
def test_continuation_resource_context_restores_all_monitor_settings(fail):
    previous = low.HOST_RESERVE_GIB, low.INITIAL_AVAILABLE_GIB, low.guard_violation
    try:
        with resume.continuation_resources():
            assert low.HOST_RESERVE_GIB == 3
            assert low.INITIAL_AVAILABLE_GIB == 16
            assert low.guard_violation is resume.guard_violation
            if fail:
                raise RuntimeError('simulated child failure')
    except RuntimeError as error:
        assert fail and str(error) == 'simulated child failure'
    assert (low.HOST_RESERVE_GIB, low.INITIAL_AVAILABLE_GIB, low.guard_violation) == previous


def test_model_checkpoint_validator_reuses_audited_v4_implementation():
    assert resume.checkpoint_receipt is v4.checkpoint_receipt


def completed_report(path):
    """A tiny public-file report fixture, not regenerated scientific output."""
    write(path / 'report.json', dict(status='complete', report_complete=True,
        theory_dim=1, checkpoint_epoch=12000))
    write(path / 'metrics.json', dict(theory_dim=1, checkpoint_epoch=12000,
        labels_used_in_training=False, thermal_nu=None,
        freeze_verification=dict(model_state_unchanged=True, model_gradients_none=True,
                                 checkpoint_unchanged=True)))
    for name in ('fixed_points.json', 'critical_exponents.json', 'status.json'):
        write(path / name, {})
    for name in ('encodings.npz', 'trajectories.npz', 'fixed_point_arrays.npz',
                 'R_vs_g.png', 'R_displacement.png', 'coupling_by_scale.png',
                 'coupling_all_coordinates.png', 'loss_comparison.png',
                 'representation_diagnostics.png', 'fixed_point_spectrum.png'):
        (path / name).write_bytes(b'tiny public report fixture: never model or physical data')
    return original.report_receipt(path, 1)


@pytest.fixture
def interrupted(source, tmp_path, monkeypatch):
    old, output, smoke = tmp_path / 'v4', tmp_path / 'v5', tmp_path / 'v3'
    monkeypatch.setattr(resume, 'OLD_ROOT', old)
    monkeypatch.setattr(resume, 'OUTPUT_ROOT', output)
    plan = original.build_plan(source[0], old)
    plan.update(schema_version=4, execution_version='resume_v4',
        source_code_sha256={'recurrent_rg/model.py': 'same',
                            'scripts/run_ising_highdim_resume.py': 'same-v4'},
        configuration_sha256={'frozen_config': 'same'},
        low_ram_execution=dict(own_gpu_hard_limit_gib=14,
            host_available_hard_floor_gib=6, child_tree_pss_anon_hard_limit_gib=6),
        interrupted_launch=dict(root=str(smoke), inherited_public_sha256={}))
    for arm in plan['arms']:
        write(Path(arm['config_path']), arm['config'])
    write(old / 'plan.json', plan)
    stages = [dict(name=name, status='complete', returncode=0, pid=200 + index)
              for index, name in enumerate(resume.EXPECTED_OLD_STAGES)]
    stages[-1].update(status='failed', returncode=-15, error=resume.PREVIOUS_ERROR, pid=6398)
    job = dict(status='failed', workflow_complete=False, requires_user_action=True,
        error=resume.PREVIOUS_ERROR, current='train_dim5', completed_arms=['dim1'],
        all_smoke_checks_passed=True, host=original.HOST, device='cuda:2', gpu_uuid=original.GPU_UUID,
        physical_sampling_updates=0, labels_used_in_training=False, parent_pid=14372,
        execution_version='resume_v4', stages=stages,
        receipts={'report_dim1': completed_report(old / 'dim1/report')})
    write(old / 'job.json', job)
    write(old / 'profiles/train_dim5.json', dict(status='failed', returncode=-15,
        violation=resume.PREVIOUS_ERROR.removeprefix('RuntimeError: '),
        monitor_error=resume.PREVIOUS_ERROR, child_pid=6398, child_starttime_ticks='100005'))
    for dim in (1, 5):
        arm = next(row for row in plan['arms'] if row['theory_dim'] == dim)
        run = old / f'dim{dim}/run'
        write(run / 'manifest.json', dict(source=None,
            resumed_from=str(smoke / 'dim1/run/training_step_5000.pt') if dim == 1 else None,
            config=arm['config'], data_sha256=plan['data_sha256'],
            execution_optimizations=dict(low_ram_execution=dict(enabled=True))))
        step = 12000 if dim == 1 else 8000
        (run / f'training_step_{step}.pt').write_bytes(b'tiny selected model-only checkpoint')
        write(run / 'history_progress.json', [dict(epoch=step if dim == 1 else 8400)])
        if dim == 1:
            write(run / 'summary.json', dict(actual_epochs=12000, requested_epochs=12000,
                                             config=arm['config']))
            write(run.parent / 'resume.json', dict(status='complete', fresh=False))
        else:
            write(run.parent / 'initialization.json', dict(fresh=True, source_checkpoint=None))
    for arm in plan['arms']:
        write(smoke / 'profiles' / f'smoke_{arm["name"]}.json', dict(status='complete', returncode=0,
            sample_count=2, host_mem_available_min_gib=9,
            peak=dict(pss_anon_bytes=3 * low.GIB, own_gpu_bytes=11 * low.GIB)))
        write(old / 'smoke' / arm['name'] / 'initialization.json', dict(fresh=True, source_checkpoint=None))
        write(smoke / 'smoke' / arm['name'] / 'initialization.json', dict(fresh=True, source_checkpoint=None))
    write(old / 'resume_cuda_verification.json', dict(status='complete', passed=True,
        synthetic_only=True, scientific_training=False, production_data_or_MI_opened=False,
        device='cuda:2', physical_uuid=original.GPU_UUID, dimension=1, total_updates=4, split_update=2))
    for name in resume.EXPECTED_OLD_STAGES[:-1]:
        write(old / 'profiles' / f'{name}.json', dict(status='complete', returncode=0,
            sample_count=2, host_mem_available_min_gib=7,
            peak=dict(pss_anon_bytes=3 * low.GIB, own_gpu_bytes=11 * low.GIB)))
    write(smoke / 'plan.json', dict(execution_version='lowram_v3', arms=plan['arms']))
    write(old / 'measured_resource_gate.json', dict(mem_available_gib=16, host_reserve_gib=6))
    completed = dict(actual_updates=12000, all_adam_parameter_steps=12000,
        labels_used_in_training=False,
        final_checkpoint_sha256=original.sha(old / 'dim1/run/training_step_12000.pt'))
    job['receipts']['train_dim1'] = completed
    write(old / 'job.json', job)
    monkeypatch.setattr(low, 'process_identity', lambda pid: None)
    monkeypatch.setattr(original, 'ensure_unchanged', lambda *args: None)
    monkeypatch.setattr(v4, 'ensure_source', lambda *args: None)
    monkeypatch.setattr(v4, 'validate_training', lambda *args: deepcopy(completed))
    monkeypatch.setattr(low, 'validate_training', lambda *args, **kwargs: dict(passed=True))

    def selected(path, config, *, step):
        return dict(path=str(Path(path).resolve()), sha256=original.sha(path), actual_updates=step,
            all_adam_parameter_steps=step, adam_state_count=24, cuda_device_index=2,
            model_tensor_sha256='model', history_sha256='history', history=[],
            metadata=original.read(Path(path).parent / 'manifest.json'),
            exact_rng_and_optimizer_present=True)

    monkeypatch.setattr(resume, 'checkpoint_receipt', selected)
    return old, output, smoke, plan, job


def approved_handoff(interrupted):
    old, output, *_ = interrupted
    receipt = resume.prepare_handoff(old, output)
    write(old / 'resume_v5_handoff.json', receipt)
    return receipt


def test_prepare_handoff_is_read_only_and_selects_saved_not_logged_update(interrupted):
    old, output, *_ = interrupted
    before = {str(path): path.read_bytes() for path in old.parent.rglob('*') if path.is_file()}
    receipt = resume.prepare_handoff(old, output)
    assert receipt['checkpoint_step'] == 8000
    assert receipt['last_logged_step'] == 8400
    assert receipt['checkpoint_path'] == str(old / 'dim5/run/training_step_8000.pt')
    assert receipt['resource_change'] == resume.RESOURCE_CHANGE
    assert {str(path): path.read_bytes() for path in old.parent.rglob('*') if path.is_file()} == before
    assert not output.exists()


def test_plan_preserves_all_science_inherits_dim1_and_resumes_only_dim5(interrupted):
    old, output, _, old_plan, _ = interrupted
    approved_handoff(interrupted)
    plan = resume.build_plan(old, output)
    assert plan['execution_version'] == 'resume_v5'
    assert resume.canonical([arm['config'] for arm in plan['arms']]) == resume.canonical(
        [arm['config'] for arm in old_plan['arms']])
    for name in ('source_run', 'source_sha256', 'immutable_stat_identity', 'data_sha256',
                 'mi_cache_dir', 'mi_cache_keys'):
        assert plan[name] == old_plan[name]
    assert plan['protocol']['resume'] and not plan['protocol']['automatic_retry']
    assert plan['protocol']['labels_used_in_training'] is False
    assert plan['protocol']['resource_gate']['gpu_free_mib'] == 16384
    assert plan['low_ram_execution']['host_available_hard_floor_gib'] == 3
    assert plan['low_ram_execution']['own_gpu_hard_limit_gib'] == 14
    arms = {arm['name']: arm for arm in plan['arms']}
    inherited = arms['dim1']
    assert inherited['inherited'] is True
    assert inherited['root'] == str(old / 'dim1')
    assert inherited['training'] is inherited['report'] is None
    assert inherited['config_path'] == old_plan['arms'][0]['config_path']
    for name, arm in arms.items():
        assert arm['config']['training']['epochs'] == 12000
        assert arm['config']['loss']['closure_scale_weights'] is None
        assert arm['config']['model']['hidden'] == 96
        if name == 'dim1':
            continue
        assert '--cached-training' in arm['training']
        assert 'scripts.run_ising_highdim_resume_v5' in arm['training']
        if name == 'dim5':
            assert arm['segment_start_update'] == 8000 and arm['new_updates'] == 4000
            assert arm['resume_from'] == str(old / 'dim5/run/training_step_8000.pt')
            assert '--resume-checkpoint' in arm['training']
        else:
            assert arm['segment_start_update'] == 0 and arm['new_updates'] == 12000
            assert arm['resume_from'] is None
            assert '--resume-checkpoint' not in arm['training']
    assert f'dim1={old}/dim1/report' in plan['compare']
    for name in ('dim5', 'dim10', 'dim100'):
        assert f'{name}={output}/{name}/report' in plan['compare']
    assert not output.exists()


@pytest.mark.parametrize('field,value', [('schema_version', 2), ('status', 'failed'),
    ('user_authorized', False), ('termination_confirmed', False), ('old_directory_preserved', False),
    ('new_root', '/different'), ('checkpoint_step', 8400), ('last_logged_step', 8000),
    ('physical_sampling_updates', 1), ('resource_change', {}),
    ('parent_pid', 111), ('child_pid', 111), ('child_starttime_ticks', 'notknown'),
    ('old_plan_sha256', 'changed'), ('old_job_sha256', 'changed'),
    ('old_profile_sha256', 'changed'), ('old_manifest_sha256', 'changed'),
    ('checkpoint_sha256', 'changed')])
def test_reject_tampered_or_unauthorized_handoff(interrupted, field, value):
    old, output, *_ = interrupted
    receipt = approved_handoff(interrupted)
    receipt[field] = value
    write(old / 'resume_v5_handoff.json', receipt)
    with pytest.raises(RuntimeError):
        resume.verify_handoff(old, output)


@pytest.mark.parametrize('pid', [14372, 6398, 200, 201, 202])
def test_reject_recorded_pid_even_when_it_could_have_been_reused(interrupted, monkeypatch, pid):
    old, output, *_ = interrupted
    approved_handoff(interrupted)
    monkeypatch.setattr(low, 'process_identity', lambda selected: dict(pid=pid) if selected == pid else None)
    with pytest.raises(RuntimeError):
        resume.verify_handoff(old, output)


@pytest.mark.parametrize('mutation', ['dim10_started', 'dim100_started', 'later_checkpoint',
    'wrong_log', 'existing_output', 'wrong_host', 'wrong_uuid', 'wrong_stage', 'wrong_exit',
    'missing_report', 'changed_report', 'changed_dim5_manifest'])
def test_reject_changed_old_execution_or_inherited_scientific_evidence(interrupted, mutation):
    old, output, _, _, job = interrupted
    approved_handoff(interrupted)
    if mutation in ('dim10_started', 'dim100_started'):
        write(old / mutation.removesuffix('_started') / 'initialization.json', dict(fresh=True))
    elif mutation == 'later_checkpoint':
        (old / 'dim5/run/training_step_9000.pt').write_bytes(b'model only')
    elif mutation == 'wrong_log':
        write(old / 'dim5/run/history_progress.json', [dict(epoch=8000)])
    elif mutation == 'existing_output':
        output.mkdir()
    elif mutation == 'missing_report':
        (old / 'dim1/report/R_displacement.png').unlink()
    elif mutation == 'changed_report':
        write(old / 'dim1/report/metrics.json', dict(changed=True))
    elif mutation == 'changed_dim5_manifest':
        write(old / 'dim5/run/manifest.json', dict(config={'changed': True}))
    else:
        if mutation == 'wrong_host': job['host'] = 'other-host'
        elif mutation == 'wrong_uuid': job['gpu_uuid'] = 'other-gpu'
        elif mutation == 'wrong_stage': job['stages'][-1]['name'] = 'train_dim100'
        else: job['stages'][-1]['returncode'] = -9
        write(old / 'job.json', job)
    with pytest.raises((RuntimeError, FileNotFoundError)):
        resume.verify_handoff(old, output)


def test_measured_gate_reuses_four_actual_profiles_and_preserves_admission_floor(interrupted):
    old, output, *_ = interrupted
    approved_handoff(interrupted)
    gate = resume.measured_gate(resume.build_plan(old, output))
    assert gate['mem_available_gib'] == 16
    assert gate['host_reserve_gib'] == 3
    assert gate['measured_max_smoke_pss_anon_gib'] == 3
    assert gate['inherited_independent_preflight'] is True


def test_scientific_code_change_rejected_even_when_checkpoint_and_plan_are_pinned(interrupted, monkeypatch):
    old, output, *_ = interrupted
    approved_handoff(interrupted)
    plan = resume.build_plan(old, output)
    monkeypatch.setattr(original, 'source_hashes', lambda project: {
        'recurrent_rg/model.py': 'changed', 'scripts/run_ising_highdim_resume.py': 'same-v4'})
    with pytest.raises(RuntimeError):
        resume.validate_code_delta(plan, old / 'code')


def test_stage_rechecks_inherited_source_before_and_after_validation(interrupted, monkeypatch):
    old, output, *_ = interrupted
    approved_handoff(interrupted)
    plan = resume.build_plan(old, output)
    observed = []
    monkeypatch.setattr(resume, 'ensure_source', lambda selected: observed.append('source'))

    def fake_stage(root, job, name, command, *, check, **kwargs):
        observed.append('stage')
        assert check() == {'valid': True}

    monkeypatch.setattr(low, 'stage', fake_stage)
    resume.stage(output, {}, 'train_dim5', [], gpu=True, plan=plan, fingerprints={}, configs={},
                 check=lambda: observed.append('endpoint') or {'valid': True})
    assert observed == ['source', 'stage', 'source', 'endpoint', 'source']


@pytest.fixture
def endpoint(interrupted, monkeypatch):
    from recurrent_rg.config import RunConfig
    from recurrent_rg.learning_rate import LearningRatePolicy

    old, output, _, old_plan, _ = interrupted
    source_path = old / 'dim5/run/manifest.json'
    previous = original.read(source_path)
    previous['execution_optimizations'].update(reference=False, selected_empirical_mi=True,
        mi_statistics_cache={split: dict(key=key, hit=True, cache_version=1,
            path=str(Path(old_plan['mi_cache_dir']) / f'mi-v1-{key}.pt'), load_seconds=10.)
            for split, key in old_plan['mi_cache_keys'].items()})
    write(source_path, previous)
    approved_handoff(interrupted)
    plan = resume.build_plan(old, output)
    arm = plan['arms'][1]
    run = Path(arm['root']) / 'run'
    run.mkdir(parents=True)
    policy = LearningRatePolicy(RunConfig.from_dict(arm['config']).training)
    manifest = deepcopy(previous)
    manifest.update(resumed_from=arm['resume_from'], config_source=arm['config_path'],
        resume_start_provenance=dict(source_epoch=8000),
        resume_start_module_sha256={'C': 'c', 'Q': 'q', 'R': 'r'})
    for row in manifest['execution_optimizations']['mi_statistics_cache'].values():
        row['load_seconds'] = .1
    history = [dict(epoch=index, learning_rate=policy.rate_for_update(index))
               for index in sorted({1, 12000, *range(200, 12001, 200)})]
    summary = dict(config=arm['config'], history=deepcopy(history), actual_epochs=12000,
        requested_epochs=12000, final_report_skipped=True,
        lr_scheduler_state=policy.checkpoint_state(12000),
        timing=dict(enabled=True, updates=4000, cuda_max_memory_allocated_bytes=6 * low.GIB,
                    cuda_max_memory_reserved_bytes=11 * low.GIB))
    provenance = dict(status='complete', fresh=False, labels_used_in_training=False,
        source_checkpoint=arm['resume_from'], source_epoch=8000,
        source_checkpoint_sha256=plan['interrupted_launch']['checkpoint']['sha256'],
        resume_start_module_sha256=manifest['resume_start_module_sha256'])
    write(run / 'manifest.json', manifest)
    write(run / 'summary.json', summary)
    write(run.parent / 'resume.json', provenance)

    def saved(path, config, *, step):
        return dict(sha256=plan['interrupted_launch']['checkpoint']['sha256'] if step == 8000 else 'new-final',
            all_adam_parameter_steps=step,
            history=deepcopy([row for row in history if row['epoch'] <= step]))

    monkeypatch.setattr(resume, 'checkpoint_receipt', saved)
    return plan, arm, manifest, summary, provenance


def test_dim5_endpoint_is_total_12000_with_only_4000_new_updates(endpoint):
    plan, arm, *_ = endpoint
    result = resume.validate_training(plan, arm)
    assert result['actual_updates'] == result['all_adam_parameter_steps'] == 12000
    assert result['segment_start_update'] == 8000
    assert result['new_updates'] == 4000
    assert result['source_checkpoint_sha256'] == plan['interrupted_launch']['checkpoint']['sha256']


@pytest.mark.parametrize('mutation', ['timing_reset', 'lr_reset', 'adam_endpoint', 'selected_mi',
    'different_cache', 'reference', 'fresh_initialization', 'resume_epoch', 'resume_sha',
    'resume_module', 'wrong_history', 'changed_source_history', 'initial_provenance'])
def test_dim5_endpoint_rejects_nonexact_continuation(endpoint, monkeypatch, mutation):
    plan, arm, manifest, summary, provenance = endpoint
    run = Path(arm['root']) / 'run'
    if mutation == 'timing_reset': summary['timing']['updates'] = 12000
    elif mutation == 'lr_reset': summary['history'][-1]['learning_rate'] = .001
    elif mutation == 'adam_endpoint': summary['actual_epochs'] = 4000
    elif mutation == 'selected_mi': manifest['execution_optimizations']['selected_empirical_mi'] = False
    elif mutation == 'different_cache':
        manifest['execution_optimizations']['mi_statistics_cache']['train']['key'] = 'different'
    elif mutation == 'reference': manifest['execution_optimizations']['reference'] = True
    elif mutation == 'fresh_initialization': write(run.parent / 'initialization.json', {'fresh': True})
    elif mutation == 'resume_epoch': provenance['source_epoch'] = 8400
    elif mutation == 'resume_sha': provenance['source_checkpoint_sha256'] = 'different'
    elif mutation == 'resume_module': provenance['resume_start_module_sha256'] = {'R': 'different'}
    elif mutation == 'wrong_history': summary['history'][1]['epoch'] = 201
    elif mutation == 'changed_source_history': summary['history'][0]['loss'] = 42.
    else: manifest['random_construction_module_sha256'] = {'C': 'different'}
    write(run / 'manifest.json', manifest)
    write(run / 'summary.json', summary)
    write(run.parent / 'resume.json', provenance)
    with pytest.raises(RuntimeError):
        resume.validate_training(plan, arm)


def test_completed_dim1_cannot_be_accidentally_validated_as_new_training(interrupted):
    old, output, *_ = interrupted
    approved_handoff(interrupted)
    plan = resume.build_plan(old, output)
    with pytest.raises(RuntimeError, match='inherited'):
        resume.validate_training(plan, plan['arms'][0])


@pytest.mark.parametrize('failure', [None, 'train_dim5', 'report_dim5', 'train_dim10', 'comparison'])
def test_execution_skips_completed_dim1_and_never_retries(interrupted, monkeypatch, failure):
    old, output, *_ = interrupted
    approved_handoff(interrupted)
    plan = resume.build_plan(old, output)
    before = {str(path): path.read_bytes() for path in old.rglob('*') if path.is_file()}
    source_file = Path(resume.__file__).resolve()
    monkeypatch.setattr(resume.socket, 'gethostname', lambda: original.HOST)
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    monkeypatch.setattr(resume.os, 'getpriority', lambda *args: 10)
    monkeypatch.setattr(resume, 'validate_code_delta', lambda *args: {'existing_changed_files': []})
    monkeypatch.setattr(original, 'source_hashes', lambda project: {
        'scripts/run_ising_highdim_resume_v5.py': original.sha(source_file)})
    observed = []

    def fake_stage(root, job, name, command, *, gpu, **kwargs):
        assert low.HOST_RESERVE_GIB == 3
        assert low.INITIAL_AVAILABLE_GIB == 16
        assert low.guard_violation is resume.guard_violation
        observed.append(name)
        assert ('--resume-checkpoint' in command) == (name == 'train_dim5')
        assert gpu == name.startswith('train_')
        job['stages'].append(dict(name=name))
        if name == failure:
            raise RuntimeError('selected stage failed; no retry')

    monkeypatch.setattr(resume, 'stage', fake_stage)
    expected = ['train_dim5', 'report_dim5', 'train_dim10', 'report_dim10',
                'train_dim100', 'report_dim100', 'comparison']
    old_floor = low.HOST_RESERVE_GIB
    if failure is None:
        job = resume.execute(plan)
        assert job['status'] == 'complete' and job['workflow_complete']
        assert job['completed_arms'] == ['dim1', 'dim5', 'dim10', 'dim100']
        assert job['inherited_completed_arms'] == ['dim1']
        assert job['resume_cuda_evidence_inherited'] is True
        assert job['discarded_unsaved_logged_updates'] == 400
        assert observed == expected
        assert not (output / 'dim1').exists()
        assert not (output / 'resume_cuda_verification.json').exists()
        assert (output / 'smoke/dim100/initialization.json').exists()
        assert all(Path(path).suffix in ('.json', '.png', '.npz') for path in job['public_files'])
    else:
        with pytest.raises(RuntimeError, match='selected stage failed'):
            resume.execute(plan)
        job = original.read(output / 'job.json')
        assert job['status'] == 'failed' and job['requires_user_action']
        assert observed == expected[:expected.index(failure) + 1]
    assert low.HOST_RESERVE_GIB == old_floor
    assert {str(path): path.read_bytes() for path in old.rglob('*') if path.is_file()} == before




