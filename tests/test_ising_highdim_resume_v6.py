"""Frozen v6 report optimization contracts; no SSH, GPU or model execution."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts import run_ising_highdim as original
from scripts import run_ising_highdim_low_ram as low
from scripts import run_ising_highdim_resume_v5 as v5
from scripts import run_ising_highdim_resume_v6 as resume
from test_ising_highdim_runner import source  # noqa: F401
from test_ising_highdim_resume_v5 import completed_report


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def file_contents(root):
    return {str(path): path.read_bytes() for path in root.rglob('*') if path.is_file()}


@pytest.fixture
def gnu_timeout_available(monkeypatch):
    """Mock only the remote Linux binary check; never execute a subprocess."""
    actual_is_file = Path.is_file
    monkeypatch.setattr(Path, 'is_file',
        lambda path: str(path) == '/usr/bin/timeout' or actual_is_file(path))


@pytest.fixture
def interrupted(source, tmp_path, monkeypatch):
    old, output = tmp_path / 'v5', tmp_path / 'v6'
    previous, smoke = tmp_path / 'v4', tmp_path / 'v3'
    interrupt_path = tmp_path / 'logs/resume_v5.report_optimization_interrupt.json'
    monkeypatch.setattr(resume, 'OLD_ROOT', old)
    monkeypatch.setattr(resume, 'OUTPUT_ROOT', output)
    monkeypatch.setattr(resume, 'INTERRUPT_RECEIPT_PATH', interrupt_path)
    plan = original.build_plan(source[0], old)
    plan.update(schema_version=5, execution_version='resume_v5',
        inherited_completed_arms=['dim1'],
        source_code_sha256={'recurrent_rg/model_nd.py': 'same-model',
            'scripts/run_ising_highdim_resume_v5.py': 'same-v5'},
        configuration_sha256={},
        low_ram_execution=dict(host_available_hard_floor_gib=3,
            initial_validation_available_gib=16, child_tree_pss_anon_hard_limit_gib=6,
            own_gpu_hard_limit_gib=14, gpu_admission_free_gib=16),
        interrupted_launch=dict(root=str(previous), smoke_root=str(smoke), inherited_public_sha256={}))
    plan['protocol']['resource_gate'].update(gpu_free_mib=16384, mem_available_gib='unchanged measured16GiB')
    for arm in plan['arms']:
        arm['inherited'] = arm['name'] == 'dim1'
        arm.pop('smoke')
        arm['resume_from'] = str(previous / 'dim5/run/training_step_8000.pt') if arm['name'] == 'dim5' else None
        arm['segment_start_update'] = 8000 if arm['name'] == 'dim5' else 0
        arm['new_updates'] = 4000 if arm['name'] == 'dim5' else 12000
        if arm['inherited']:
            arm.update(root=str(previous / 'dim1'),
                config_path=str(previous / 'configs/dim1.json'), training=None, report=None,
                segment_start_update=12000, new_updates=0)
        write(arm['config_path'], arm['config'])
        plan['configuration_sha256'][arm['config_path']] = original.sha(arm['config_path'])
        write(old / 'smoke' / arm['name'] / 'initialization.json', dict(fresh=True))
        write(smoke / 'profiles' / f'smoke_{arm["name"]}.json', dict(status='complete',
            returncode=0, sample_count=2, host_mem_available_min_gib=10.,
            peak=dict(pss_anon_bytes=3 * low.GIB, own_gpu_bytes=11 * low.GIB)))
    report = completed_report(previous / 'dim1/report')
    plan['arms'][0]['inherited_report_receipt'] = report
    plan['arms'][0]['inherited_training_receipt'] = dict(actual_updates=12000,
        all_adam_parameter_steps=12000, labels_used_in_training=False)
    run = old / 'dim5/run'
    manifest = dict(config=plan['arms'][1]['config'], source=None,
        resumed_from=plan['arms'][1]['resume_from'], data_sha256=plan['data_sha256'],
        execution_optimizations=dict(low_ram_execution=dict(enabled=True)))
    write(run / 'manifest.json', manifest)
    write(run / 'summary.json', dict(actual_epochs=12000, requested_epochs=12000,
        config=plan['arms'][1]['config'], final_report_skipped=True))
    write(run / 'history_progress.json', [dict(epoch=12000)])
    write(run.parent / 'resume.json', dict(status='complete', fresh=False, source_epoch=8000))
    checkpoint = run / 'training_step_12000.pt'
    checkpoint.write_bytes(b'tiny frozen selected model checkpoint, never a physical cache')
    training_receipt = dict(actual_updates=12000, all_adam_parameter_steps=12000,
        segment_start_update=8000, new_updates=4000, labels_used_in_training=False,
        final_checkpoint_sha256=original.sha(checkpoint),
        manifest_sha256=original.sha(run / 'manifest.json'))
    stages = [dict(name='train_dim5', status='complete', returncode=0, pid=17702),
        dict(name='report_dim5', status='failed', returncode=-15, pid=28359,
            error=resume.PREVIOUS_ERROR)]
    job = dict(status='failed', workflow_complete=False, requires_user_action=True,
        error=resume.PREVIOUS_ERROR, current='report_dim5', completed_arms=['dim1'],
        inherited_completed_arms=['dim1'], all_smoke_checks_passed=True,
        smoke_evidence_inherited=True, resume_cuda_evidence_inherited=True,
        host=original.HOST, device='cuda:2', gpu_uuid=original.GPU_UUID,
        actual_nice=10, physical_sampling_updates=0, labels_used_in_training=False,
        parent_pid=17577, execution_version='resume_v5', stages=stages,
        receipts={'train_dim5': training_receipt},
        inherited_receipts={'train_dim1': plan['arms'][0]['inherited_training_receipt'],
                            'report_dim1': report})
    write(old / 'profiles/train_dim5.json', dict(status='complete', returncode=0,
        sample_count=2, host_mem_available_min_gib=10.,
        child_pid=17702, child_starttime_ticks='1162058739',
        peak=dict(pss_anon_bytes=3 * low.GIB, own_gpu_bytes=11 * low.GIB)))
    write(old / 'profiles/report_dim5.json', dict(status='failed', returncode=-15,
        sample_count=2, host_mem_available_min_gib=10.,
        child_pid=28359, child_starttime_ticks='1162240832',
        peak=dict(pss_anon_bytes=3 * low.GIB, own_gpu_bytes=0)))
    gate = dict(mem_available_gib=16, host_reserve_gib=3,
        measured_max_smoke_pss_anon_gib=3, inherited_independent_preflight=True)
    write(old / 'measured_resource_gate.json', gate)
    job['measured_resource_gate'] = gate
    write(old / 'plan.json', plan)
    write(old / 'job.json', job)
    interrupt = dict(schema_version=1, status='interrupted', user_authorized=True,
        authorized_request='中断报告并优化高维时间', stage='report_dim5',
        parent=dict(pid=17577, starttime_ticks='1162058628', uid=1097,
                    session=17577, process_group=17577),
        report_child=dict(pid=28359, starttime_ticks='1162240832', uid=1097,
                          session=28359, process_group=28359),
        only_owned_report_session=True, completed_training_preserved=True,
        other_processes_untouched=True, dim5_checkpoint_unchanged=True,
        job_status='failed', job_error=resume.PREVIOUS_ERROR,
        child_present=False, parent_present=True,
        completed_dim5_checkpoint_sha256=original.sha(checkpoint))
    write(interrupt_path, interrupt)
    monkeypatch.setattr(low, 'process_identity', lambda pid: None)
    monkeypatch.setattr(original, 'ensure_unchanged', lambda *args: None)
    monkeypatch.setattr(v5, 'ensure_source', lambda *args: None)
    monkeypatch.setattr(v5, 'validate_training', lambda *args: deepcopy(training_receipt))

    def selected(path, config, *, step):
        return dict(path=str(Path(path).resolve()), sha256=original.sha(path), actual_updates=step,
            all_adam_parameter_steps=step, adam_state_count=24, cuda_device_index=2,
            model_tensor_sha256='model', history_sha256='history', history=[],
            metadata=manifest, exact_rng_and_optimizer_present=True)

    monkeypatch.setattr(resume, 'checkpoint_receipt', selected)
    return old, output, previous, plan, job, interrupt_path, interrupt


def approve(interrupted):
    old, output, *_ = interrupted
    receipt = resume.prepare_handoff(old, output)
    write(old / resume.HANDOFF_NAME, receipt)
    return receipt


def test_prepare_is_read_only_and_inherits_only_completed_training(interrupted):
    old, output, *_ = interrupted
    before = file_contents(old.parent)
    receipt = resume.prepare_handoff(old, output)
    assert receipt['old_plan_sha256'] == original.sha(old / 'plan.json')
    assert receipt['old_job_sha256'] == original.sha(old / 'job.json')
    assert receipt['old_manifest_sha256'] == original.sha(old / 'dim5/run/manifest.json')
    assert receipt['final_checkpoint_sha256'] == original.sha(old / 'dim5/run/training_step_12000.pt')
    assert receipt['inherited_dim5_training_receipt']['actual_updates'] == 12000
    assert receipt['inherited_dim5_training_receipt']['new_updates'] == 4000
    assert not output.exists() and file_contents(old.parent) == before


def test_plan_retains_all_science_and_never_retrains_completed_arms(interrupted):
    old, output, previous, old_plan, *_ = interrupted
    approve(interrupted)
    plan = resume.build_plan(old, output)
    assert plan['execution_version'] == 'resume_v6'
    assert v5.canonical([arm['config'] for arm in plan['arms']]) == v5.canonical(
        [arm['config'] for arm in old_plan['arms']])
    for key in ('source_run', 'source_sha256', 'immutable_stat_identity', 'data_sha256',
                'mi_cache_dir', 'mi_cache_keys'):
        assert plan[key] == old_plan[key]
    for key in ('host_available_hard_floor_gib', 'initial_validation_available_gib',
                'child_tree_pss_anon_hard_limit_gib', 'own_gpu_hard_limit_gib', 'gpu_admission_free_gib'):
        assert plan['low_ram_execution'][key] == old_plan['low_ram_execution'][key]
    assert plan['protocol']['resource_gate']['gpu_free_mib'] == 16384
    arms = {arm['name']: arm for arm in plan['arms']}
    assert arms['dim1']['root'] == str(previous / 'dim1')
    assert arms['dim1']['training'] is arms['dim1']['report'] is None
    assert arms['dim5']['training_inherited'] is True and arms['dim5']['training'] is None
    assert arms['dim5']['run_path'] == str(old / 'dim5/run')
    assert arms['dim5']['root'] == str(output / 'dim5')
    assert arms['dim5']['report'][arms['dim5']['report'].index('--run') + 1] == str(old / 'dim5/run')
    for name in ('dim10', 'dim100'):
        arm = arms[name]
        assert arm['new_updates'] == 12000 and arm['segment_start_update'] == 0
        assert arm['resume_from'] is None
        assert '--resume-checkpoint' not in arm['training']
        assert arm['config']['loss']['closure_scale_weights'] is None
        assert arm['config']['training']['epochs'] == 12000
    assert f'dim1={previous}/dim1/report' in plan['compare']
    for name in ('dim5', 'dim10', 'dim100'):
        assert f'{name}={output}/{name}/report' in plan['compare']
    assert not output.exists()


def test_every_new_report_has_inner_and_external_wall_clock_limits(interrupted):
    old, output, *_ = interrupted
    approve(interrupted)
    plan = resume.build_plan(old, output)
    assert resume.REPORT_SECONDS == 840 and resume.REPORT_WALL_SECONDS == 900
    for arm in plan['arms'][1:]:
        command = arm['report']
        assert command[:4] == ['/usr/bin/timeout', '--signal=TERM', '--kill-after=5s', '900s']
        assert 'scripts.report_ising_highdim_fast' in command
        assert command[command.index('--max-seconds') + 1] == '840'
        assert command[command.index('--threads') + 1] == '1'
        assert command[command.index('--sampling-repeats') + 1] == '5'
        assert command[command.index('--checkpoint') + 1] == '12000'


@pytest.mark.parametrize('field,value', [('user_authorized', False), ('status', 'running'),
    ('stage', 'train_dim5'), ('only_owned_report_session', False),
    ('completed_training_preserved', False), ('other_processes_untouched', False),
    ('dim5_checkpoint_unchanged', False), ('job_error', 'different'),
    ('completed_dim5_checkpoint_sha256', 'changed')])
def test_handoff_rejects_unauthorized_or_different_interrupt(interrupted, field, value):
    old, output, _, _, _, path, receipt = interrupted
    receipt[field] = value
    write(path, receipt)
    with pytest.raises((RuntimeError, ValueError)):
        resume.prepare_handoff(old, output)


@pytest.mark.parametrize('pid', [17577, 17702, 28359])
def test_handoff_rejects_any_recorded_live_or_reused_pid(interrupted, monkeypatch, pid):
    old, output, *_ = interrupted
    monkeypatch.setattr(low, 'process_identity', lambda selected: dict(pid=pid) if selected == pid else None)
    with pytest.raises(RuntimeError):
        resume.prepare_handoff(old, output)


@pytest.mark.parametrize('mutation', ['train_not_complete', 'wrong_error', 'wrong_host', 'wrong_gpu',
    'dim10_started', 'dim100_started', 'report_already_exists', 'missing_checkpoint',
    'changed_final_checkpoint', 'changed_inherited_dim1_report', 'existing_output'])
def test_handoff_rejects_changed_frozen_execution(interrupted, mutation):
    old, output, previous, _, job, *_ = interrupted
    approve(interrupted)
    if mutation == 'train_not_complete':
        job['stages'][0]['returncode'] = 1
        write(old / 'job.json', job)
    elif mutation == 'wrong_error':
        job['error'] = 'OOM'
        write(old / 'job.json', job)
    elif mutation == 'wrong_host':
        job['host'] = 'other-host'
        write(old / 'job.json', job)
    elif mutation == 'wrong_gpu':
        job['gpu_uuid'] = 'other-uuid'
        write(old / 'job.json', job)
    elif mutation in ('dim10_started', 'dim100_started'):
        write(old / mutation.removesuffix('_started') / 'initialization.json', dict(fresh=True))
    elif mutation == 'report_already_exists':
        write(old / 'dim5/report/report.json', dict(status='complete', report_complete=True))
    elif mutation == 'missing_checkpoint':
        (old / 'dim5/run/training_step_12000.pt').unlink()
    elif mutation == 'changed_final_checkpoint':
        (old / 'dim5/run/training_step_12000.pt').write_bytes(b'changed checkpoint fixture')
    elif mutation == 'changed_inherited_dim1_report':
        write(previous / 'dim1/report/metrics.json', dict(changed=True))
    else:
        output.mkdir()
    with pytest.raises((RuntimeError, ValueError, FileNotFoundError)):
        resume.verify_handoff(old, output)


def test_old_partial_report_is_preserved_without_becoming_a_completed_result(interrupted):
    old, output, *_ = interrupted
    write(old / 'dim5/report/status.json', dict(status='running', report_complete=False))
    (old / 'dim5/report/partial_arrays.npz').write_bytes(b'partial public fixture')
    before = file_contents(old.parent)
    receipt = resume.prepare_handoff(old, output)
    assert receipt['inherited_completed_arms'] == ['dim1']
    assert file_contents(old.parent) == before and not output.exists()


@pytest.mark.parametrize('field', ['old_plan_sha256', 'old_job_sha256', 'old_profile_sha256',
                                 'old_manifest_sha256', 'final_checkpoint_sha256'])
def test_frozen_handoff_hash_changes_are_rejected(interrupted, field):
    old, output, *_ = interrupted
    receipt = approve(interrupted)
    receipt[field] = 'changed'
    write(old / resume.HANDOFF_NAME, receipt)
    with pytest.raises(RuntimeError):
        resume.verify_handoff(old, output)


def test_code_delta_only_allows_three_new_modules_never_training_changes(interrupted, monkeypatch):
    old, output, _, old_plan, *_ = interrupted
    approve(interrupted)
    plan = resume.build_plan(old, output)
    expected = {'scripts/run_ising_highdim_resume_v6.py',
                'scripts/report_ising_highdim_fast.py', 'scripts/ising_highdim_fast_evaluation.py'}
    assert set(resume.ALLOWED_NEW_MODULES) == expected
    valid = dict(old_plan['source_code_sha256'], **{name: 'new-file' for name in expected})
    monkeypatch.setattr(original, 'source_hashes', lambda project: dict(valid))
    assert resume.validate_code_delta(plan, old / 'code')['existing_changed_files'] == []
    for invalid in (dict(valid, **{'recurrent_rg/model_nd.py': 'changed'}),
                    {name: digest for name, digest in valid.items() if name != 'scripts/ising_highdim_fast_evaluation.py'},
                    dict(valid, **{'scripts/unrequested_extra.py': 'new'})):
        monkeypatch.setattr(original, 'source_hashes', lambda project, invalid=invalid: dict(invalid))
        with pytest.raises(RuntimeError):
            resume.validate_code_delta(plan, old / 'code')


def test_prepare_rejects_changed_upstream_evidence_even_before_new_handoff(interrupted, monkeypatch):
    old, output, *_ = interrupted

    def changed(plan):
        raise RuntimeError('upstream frozen training evidence changed')

    monkeypatch.setattr(v5, 'ensure_source', changed)
    with pytest.raises(RuntimeError, match='upstream frozen'):
        resume.prepare_handoff(old, output)


def test_stage_rechecks_frozen_sources_before_and_after_endpoint_validation(interrupted, monkeypatch):
    old, output, *_ = interrupted
    approve(interrupted)
    plan = resume.build_plan(old, output)
    observed = []
    monkeypatch.setattr(resume, 'ensure_source', lambda selected: observed.append('source'))

    def successful(root, job, name, command, *, check, **kwargs):
        observed.append('stage')
        assert check() == {'valid': True}

    monkeypatch.setattr(low, 'stage', successful)
    resume.stage(output, {}, 'report_dim5', [], gpu=False, plan=plan, fingerprints={}, configs={},
                 check=lambda: observed.append('endpoint') or {'valid': True})
    assert observed == ['source', 'stage', 'source', 'endpoint', 'source']


@pytest.mark.parametrize('failure', [None, 'report_dim5', 'train_dim10', 'report_dim100', 'comparison'])
def test_execution_only_rereports_dim5_then_trains_fresh_remaining_arms(
        interrupted, monkeypatch, gnu_timeout_available, failure):
    old, output, *_ = interrupted
    approve(interrupted)
    plan = resume.build_plan(old, output)
    before = file_contents(old.parent)
    source_file = Path(resume.__file__).resolve()
    monkeypatch.setattr(resume.socket, 'gethostname', lambda: original.HOST)
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    monkeypatch.setattr(resume.os, 'getpriority', lambda *args: 10)
    monkeypatch.setattr(resume, 'validate_code_delta', lambda *args: dict(existing_changed_files=[]))
    monkeypatch.setattr(original, 'source_hashes', lambda project: {
        'scripts/run_ising_highdim_resume_v6.py': original.sha(source_file)})
    observed = []

    def fake_stage(root, job, name, command, *, gpu, **kwargs):
        assert low.HOST_RESERVE_GIB == 3 and low.INITIAL_AVAILABLE_GIB == 16
        assert low.guard_violation is v5.guard_violation
        assert name not in ('train_dim1', 'report_dim1', 'train_dim5')
        assert '--resume-checkpoint' not in command
        assert gpu == name.startswith('train_')
        observed.append(name)
        job['stages'].append(dict(name=name))
        if name == failure:
            raise RuntimeError('selected stage failed; no retry')

    monkeypatch.setattr(resume, 'stage', fake_stage)
    expected = ['report_dim5', 'train_dim10', 'report_dim10', 'train_dim100', 'report_dim100', 'comparison']
    previous_floor = low.HOST_RESERVE_GIB
    if failure is None:
        job = resume.execute(plan)
        assert job['status'] == 'complete' and job['workflow_complete']
        assert job['completed_arms'] == ['dim1', 'dim5', 'dim10', 'dim100']
        assert job['inherited_completed_arms'] == ['dim1']
        assert observed == expected
        assert not (output / 'dim1').exists()
        assert not (output / 'dim5/run').exists()
        assert all(Path(path).suffix in ('.json', '.png', '.npz') for path in job['public_files'])
    else:
        with pytest.raises(RuntimeError, match='selected stage failed'):
            resume.execute(plan)
        job = original.read(output / 'job.json')
        assert job['status'] == 'failed' and job['requires_user_action']
        assert observed == expected[:expected.index(failure) + 1]
    assert low.HOST_RESERVE_GIB == previous_floor
    assert {name: contents for name, contents in file_contents(old.parent).items()
            if not Path(name).is_relative_to(output)} == before


@pytest.mark.parametrize('name', ['report_dim5', 'report_dim10', 'report_dim100'])
def test_external_report_timeout_holds_queue_without_claiming_completion(
        interrupted, monkeypatch, gnu_timeout_available, name):
    old, output, *_ = interrupted
    approve(interrupted)
    plan = resume.build_plan(old, output)
    source_file = Path(resume.__file__).resolve()
    monkeypatch.setattr(resume.socket, 'gethostname', lambda: original.HOST)
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    monkeypatch.setattr(resume.os, 'getpriority', lambda *args: 10)
    monkeypatch.setattr(resume, 'validate_code_delta', lambda *args: dict(existing_changed_files=[]))
    monkeypatch.setattr(original, 'source_hashes', lambda project: {
        'scripts/run_ising_highdim_resume_v6.py': original.sha(source_file)})
    calls = []

    def fake_stage(root, job, stage_name, command, **kwargs):
        calls.append(stage_name)
        job['stages'].append(dict(name=stage_name, returncode=124 if stage_name == name else 0))
        if stage_name == name:
            raise resume.AnalysisBudgetHold('report returncode=124; no retry')

    monkeypatch.setattr(resume, 'stage', fake_stage)
    with pytest.raises(resume.AnalysisBudgetHold):
        resume.execute(plan)
    job = original.read(output / 'job.json')
    assert job['status'] == 'analysis_budget_hold'
    assert job['workflow_complete'] is False and job['requires_user_action'] is True
    assert calls[-1] == name and calls.count(name) == 1
    assert 'comparison' not in calls


@pytest.mark.parametrize('returncode', [124, 137])
def test_stage_translates_external_report_timeout_into_explicit_budget_hold(
        interrupted, monkeypatch, returncode):
    old, output, *_ = interrupted
    approve(interrupted)
    plan = resume.build_plan(old, output)
    checks = []
    monkeypatch.setattr(resume, 'ensure_source', lambda selected: checks.append('source'))

    def timed_out(root, job, name, command, **kwargs):
        job['stages'].append(dict(name=name, status='failed', returncode=returncode,
                                 elapsed_seconds=resume.REPORT_WALL_SECONDS))
        raise RuntimeError(f'{name} returncode={returncode}; no retry/repair/restart')

    monkeypatch.setattr(low, 'stage', timed_out)
    job = dict(stages=[])
    with pytest.raises(resume.AnalysisBudgetHold, match='incomplete'):
        resume.stage(output, job, 'report_dim5', [], gpu=False, plan=plan,
                     fingerprints={}, configs={}, check=lambda: pytest.fail('No result on timeout'))
    assert checks == ['source'] and len(job['stages']) == 1


def test_early_report_sigkill_is_not_misclassified_as_timeout(interrupted, monkeypatch):
    old, output, *_ = interrupted
    approve(interrupted)
    plan = resume.build_plan(old, output)
    monkeypatch.setattr(resume, 'ensure_source', lambda selected: None)

    def killed(root, job, name, command, **kwargs):
        job['stages'].append(dict(name=name, status='failed', returncode=137, elapsed_seconds=2.))
        raise RuntimeError('report terminated early; investigate OOM or external signal')

    monkeypatch.setattr(low, 'stage', killed)
    with pytest.raises(RuntimeError, match='terminated early') as error:
        resume.stage(output, dict(stages=[]), 'report_dim5', [], gpu=False,
                     plan=plan, fingerprints={}, configs={}, check=lambda: None)
    assert not isinstance(error.value, resume.AnalysisBudgetHold)


def test_training_failure_is_never_misclassified_as_report_budget_hold(interrupted, monkeypatch):
    old, output, *_ = interrupted
    approve(interrupted)
    plan = resume.build_plan(old, output)
    monkeypatch.setattr(resume, 'ensure_source', lambda selected: None)

    def failed(root, job, name, command, **kwargs):
        job['stages'].append(dict(name=name, status='failed', returncode=124))
        raise RuntimeError('training failed, not report timeout')

    monkeypatch.setattr(low, 'stage', failed)
    with pytest.raises(RuntimeError, match='training failed') as error:
        resume.stage(output, dict(stages=[]), 'train_dim10', [], gpu=True,
                     plan=plan, fingerprints={}, configs={}, check=lambda: None)
    assert not isinstance(error.value, resume.AnalysisBudgetHold)


def test_internal_report_deadline_progress_becomes_explicit_hold(interrupted, monkeypatch):
    old, output, *_ = interrupted
    approve(interrupted)
    plan = resume.build_plan(old, output)
    write(output / 'dim5/report_progress.json', dict(status='analysis_budget_hold'))
    monkeypatch.setattr(resume, 'ensure_source', lambda selected: None)

    def stopped(root, job, name, command, **kwargs):
        job['stages'].append(dict(name=name, status='failed', returncode=1, elapsed_seconds=840.))
        raise RuntimeError('report internal deadline stopped further analysis')

    monkeypatch.setattr(low, 'stage', stopped)
    with pytest.raises(resume.AnalysisBudgetHold, match='incomplete'):
        resume.stage(output, dict(stages=[]), 'report_dim5', [], gpu=False,
                     plan=plan, fingerprints={}, configs={}, check=lambda: None)
