"""Explicit v5 continuation: inherit completed dim1; resume dim5 at update8000.

Only the user-approved host MemAvailable hard floor changes, from6 to3GiB.
Scientific code/configuration, admission gates, GPU identity and own-process
limits remain frozen. No automatic retry, new physical sampling or MI build.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import socket
import sys
import time

from scripts import run_ising_highdim as original
from scripts import run_ising_highdim_low_ram as low
from scripts import run_ising_highdim_resume as v4

EXECUTION_VERSION = 'resume_v5'
OLD_ROOT = Path('/mnt/nas-new/home/yange/ham/results/ising2d_b2_highdim_seed91_vm225_20261021_resume_v4')
OUTPUT_ROOT = Path('/mnt/nas-new/home/yange/ham/results/ising2d_b2_highdim_seed91_vm225_20261021_resume_v5')
HOST_RESERVE_GIB = 3.
INITIAL_AVAILABLE_GIB = 16.
CHECKPOINT_STEP = 8000
LAST_LOGGED_STEP = 8400
PREVIOUS_ERROR = 'RuntimeError: Shared-host MemAvailable fell below 6 GiB reserve'
EXPECTED_OLD_STAGES = ['resume_cuda_verification', 'train_dim1', 'report_dim1', 'train_dim5']
RESOURCE_CHANGE = dict(previous_host_floor_gib=6, host_floor_gib=3,
    own_anon_limit_gib=6, own_gpu_limit_gib=14, minimum_free_gpu_gib=16,
    initial_available_gib=16, scientific_configuration_unchanged=True)
HANDOFF_NAME = 'resume_v5_handoff.json'
canonical = v4.canonical
checkpoint_receipt = v4.checkpoint_receipt


def guard_violation(sample, *, gpu):
    if sample['mem_available_gib'] < HOST_RESERVE_GIB:
        return 'Shared-host MemAvailable fell below 3 GiB reserve'
    # v4 retains the ORIGINAL v3 function, not the mutable low.guard_violation.
    # Keep every other protection without recursion or a second host-floor test.
    return v4._V3_GUARD(dict(sample, mem_available_gib=max(8., sample['mem_available_gib'])), gpu=gpu)


@contextmanager
def continuation_resources():
    values = {name: getattr(low, name) for name in
              ('HOST_RESERVE_GIB', 'INITIAL_AVAILABLE_GIB', 'guard_violation')}
    low.HOST_RESERVE_GIB, low.INITIAL_AVAILABLE_GIB = HOST_RESERVE_GIB, INITIAL_AVAILABLE_GIB
    low.guard_violation = guard_violation
    try:
        yield
    finally:
        for name, value in values.items():
            setattr(low, name, value)


def _public_report_receipt(job, name, path, dimension):
    receipt = original.report_receipt(path, dimension)
    recorded = job.get('receipts', {}).get(name)
    if not recorded or canonical(recorded) != canonical(receipt):
        raise RuntimeError('Completed inherited report differs from the authoritative v4 job receipt')
    return receipt


def prepare_handoff(interrupted_root, output):
    """Read-only audit and deterministic authorization receipt; caller saves once.

    Opens only small selected checkpoints and public evidence/code. Existing
    data/MI identities are verified through the unchanged stat-only contract.
    No previous job, plan, checkpoint, report or profile is rewritten.
    """
    root, output = Path(interrupted_root).resolve(), Path(output).resolve()
    if (root != OLD_ROOT.resolve() or output != OUTPUT_ROOT.resolve()
            or output == root or root in output.parents or output in root.parents
            or output.exists() or output.is_symlink()):
        raise RuntimeError('Require the unique approved v4-to-v5 handoff without overwrite')
    plan, job = original.read(root / 'plan.json'), original.read(root / 'job.json')
    profile_path = root / 'profiles/train_dim5.json'
    profile = original.read(profile_path)
    stages = job.get('stages', [])
    if (plan.get('execution_version') != 'resume_v4' or job.get('execution_version') != 'resume_v4'
            or Path(plan.get('output', '')).resolve() != root
            or job.get('status') != 'failed' or job.get('workflow_complete') is not False
            or job.get('requires_user_action') is not True or job.get('error') != PREVIOUS_ERROR
            or job.get('current') != 'train_dim5' or job.get('completed_arms') != ['dim1']
            or job.get('all_smoke_checks_passed') is not True
            or job.get('host') != original.HOST or job.get('device') != 'cuda:2'
            or job.get('gpu_uuid') != original.GPU_UUID
            or job.get('physical_sampling_updates') != 0 or job.get('labels_used_in_training') is not False
            or [row.get('name') for row in stages] != EXPECTED_OLD_STAGES
            or [(row.get('status'), row.get('returncode')) for row in stages]
                != [('complete', 0)] * 3 + [('failed', -15)]
            or stages[-1].get('error') != PREVIOUS_ERROR
            or profile.get('status') != 'failed' or profile.get('returncode') != -15
            or profile.get('violation') != PREVIOUS_ERROR.removeprefix('RuntimeError: ')
            or profile.get('monitor_error') != PREVIOUS_ERROR
            or profile.get('child_pid') != stages[-1].get('pid')
            or not str(profile.get('child_starttime_ticks', '')).isdigit()
            or plan.get('low_ram_execution', {}).get('host_available_hard_floor_gib') != 6
            or plan.get('low_ram_execution', {}).get('child_tree_pss_anon_hard_limit_gib') != 6
            or plan.get('low_ram_execution', {}).get('own_gpu_hard_limit_gib') != 14):
        raise RuntimeError('Prior launch is not the approved v4 dim5 shared-host-reserve interruption')
    pids = {job.get('parent_pid'), *(row.get('pid') for row in stages)}
    if any(type(pid) is not int or pid <= 0 or low.process_identity(pid) is not None for pid in pids):
        raise RuntimeError('A prior recorded parent/child PID is still present; fail closed')
    if [arm.get('name') for arm in plan.get('arms', [])] != [f'dim{dim}' for dim in original.DIMENSIONS]:
        raise RuntimeError('The four authorized dimensions changed')
    for arm in plan['arms']:
        if Path(arm['root']).resolve() != root / arm['name']:
            raise RuntimeError('Prior arm path changed')
        if arm['name'] in ('dim10', 'dim100') and any((Path(arm['root']) / name).exists()
                for name in ('run', 'report', 'initialization.json')):
            raise RuntimeError('A remaining fresh arm already started')
    progress = original.read(root / 'dim5/run/history_progress.json')
    if not progress or progress[-1].get('epoch') != LAST_LOGGED_STEP:
        raise RuntimeError('Interrupted logged endpoint differs from the approved rollback')
    saved = sorted(int(path.stem.removeprefix('training_step_'))
        for path in (root / 'dim5/run').glob('training_step_*.pt'))
    if not saved or saved[-1] != CHECKPOINT_STEP:
        raise RuntimeError('Latest saved dim5 update differs; never select another checkpoint')
    original.ensure_unchanged(plan, root / 'code', plan['source_code_sha256'], plan['configuration_sha256'])
    # v4 already pins the v3 smokes, initializations, CUDA equivalence, selected
    # dim1 source checkpoint5000 and all upstream code/config/public evidence.
    v4.ensure_source(plan)
    with v4.continuation_resources():
        for name in EXPECTED_OLD_STAGES[:-1]:
            low.profile_receipt(original.read(root / 'profiles' / f'{name}.json'))
        completed = v4.validate_training(plan, plan['arms'][0])
    if canonical(completed) != canonical(job.get('receipts', {}).get('train_dim1')):
        raise RuntimeError('Inherited completed dim1 training differs from the v4 authoritative receipt')
    report = _public_report_receipt(job, 'report_dim1', root / 'dim1/report', 1)
    selected_path = root / 'dim5/run/training_step_8000.pt'
    selected = checkpoint_receipt(selected_path, plan['arms'][1]['config'], step=CHECKPOINT_STEP)
    manifest_path = root / 'dim5/run/manifest.json'
    manifest = original.read(manifest_path)
    if (selected['metadata'].get('resumed_from')
            or canonical(selected['metadata']) != canonical(manifest)
            or manifest.get('data_sha256') != plan['data_sha256']
            or not manifest.get('execution_optimizations', {}).get('low_ram_execution', {}).get('enabled')):
        raise RuntimeError('Dim5 source must be the frozen original fresh run checkpoint/manifest')
    final = checkpoint_receipt(root / 'dim1/run/training_step_12000.pt', plan['arms'][0]['config'], step=original.UPDATES)
    if final['sha256'] != completed.get('final_checkpoint_sha256'):
        raise RuntimeError('Completed dim1 final checkpoint differs from its verified receipt')
    smoke_root = Path(plan['interrupted_launch']['root']).resolve()
    smoke_plan = original.read(smoke_root / 'plan.json')
    if smoke_plan.get('execution_version') != 'lowram_v3':
        raise RuntimeError('Inherited actual smoke evidence must come from frozen v3')
    inherited = dict(plan['interrupted_launch']['inherited_public_sha256'])
    for dim in original.DIMENSIONS:
        arm = next(item for item in smoke_plan['arms'] if item['theory_dim'] == dim)
        low.validate_training(smoke_plan, arm, smoke=True)
        low.profile_receipt(original.read(smoke_root / 'profiles' / f'smoke_dim{dim}.json'))
        for path in (smoke_root / 'profiles' / f'smoke_dim{dim}.json',
                     root / 'smoke' / f'dim{dim}' / 'initialization.json'):
            inherited[str(path)] = original.sha(path)
    v4.cuda_receipt(root / 'resume_cuda_verification.json')
    for path in (root / 'resume_cuda_verification.json', root / 'dim5/initialization.json',
                 root / 'dim1/resume.json', root / 'dim1/run/manifest.json',
                 root / 'dim1/run/summary.json', root / 'dim1/run/training_step_12000.pt',
                 root / 'measured_resource_gate.json', smoke_root / 'plan.json'):
        inherited[str(path)] = original.sha(path)
    for name, digest in report['files'].items():
        inherited[str(root / 'dim1/report' / name)] = digest
    for name in EXPECTED_OLD_STAGES[:-1]:
        path = root / 'profiles' / f'{name}.json'
        inherited[str(path)] = original.sha(path)
    for path, digest in inherited.items():
        if original.sha(path) != digest:
            raise RuntimeError('Inherited source evidence changed')
    return dict(schema_version=1, status='approved_exact_checkpoint_continuation',
        authorized_request='保护线降低到3G，续训练', user_authorized=True,
        termination_confirmed=True, old_directory_preserved=True, new_root=str(output),
        checkpoint_path=str(selected_path), checkpoint_step=CHECKPOINT_STEP,
        last_logged_step=LAST_LOGGED_STEP, discarded_unsaved_logged_updates=400,
        parent_pid=job['parent_pid'], child_pid=profile['child_pid'],
        child_starttime_ticks=str(profile['child_starttime_ticks']),
        fresh_remaining_arms_started=False, inherited_completed_arms=['dim1'],
        physical_sampling_updates=0, resource_change=deepcopy(RESOURCE_CHANGE),
        old_plan_sha256=original.sha(root / 'plan.json'), old_job_sha256=original.sha(root / 'job.json'),
        old_profile_sha256=original.sha(profile_path), old_manifest_sha256=original.sha(manifest_path),
        checkpoint_sha256=selected['sha256'], inherited_public_sha256=inherited,
        inherited_dim1_training_receipt=completed, inherited_dim1_report_receipt=report,
        smoke_root=str(smoke_root),
        checkpoint={key: value for key, value in selected.items() if key not in ('history', 'metadata')})


def verify_handoff(interrupted_root, output):
    root = Path(interrupted_root).resolve()
    path = root / HANDOFF_NAME
    receipt = original.read(path)
    expected = prepare_handoff(root, output)
    if canonical(receipt) != canonical(expected):
        raise RuntimeError('Approved v5 handoff differs from current frozen source evidence or authorization')
    return dict(root=str(root), receipt_sha256=original.sha(path), receipt=receipt,
        inherited_public_sha256=receipt['inherited_public_sha256'], checkpoint=receipt['checkpoint'],
        smoke_root=receipt['smoke_root'])


def ensure_source(plan):
    record = plan['interrupted_launch']
    root = Path(record['root'])
    if original.sha(root / HANDOFF_NAME) != record['receipt_sha256']:
        raise RuntimeError('Approved v5 continuation handoff changed')
    paths = dict(old_plan_sha256=root / 'plan.json', old_job_sha256=root / 'job.json',
        old_profile_sha256=root / 'profiles/train_dim5.json',
        old_manifest_sha256=root / 'dim5/run/manifest.json',
        checkpoint_sha256=root / 'dim5/run/training_step_8000.pt')
    for key, path in paths.items():
        if original.sha(path) != record['receipt'][key]:
            raise RuntimeError('Interrupted v4 or selected checkpoint evidence changed')
    for path, digest in record['inherited_public_sha256'].items():
        if original.sha(path) != digest:
            raise RuntimeError('Inherited completed arm or preflight evidence changed')
    old = original.read(root / 'plan.json')
    original.ensure_unchanged(old, root / 'code', old['source_code_sha256'], old['configuration_sha256'])
    v4.ensure_source(old)


def validate_code_delta(plan, project):
    old = original.read(Path(plan['interrupted_launch']['root']) / 'plan.json')['source_code_sha256']
    current = original.source_hashes(project)
    changed = [name for name, digest in old.items() if current.get(name) != digest]
    added = sorted(set(current) - set(old))
    if changed or added != ['scripts/run_ising_highdim_resume_v5.py']:
        raise RuntimeError('Only the new v5 orchestration module may be added; all v4 files stay identical')
    return dict(existing_changed_files=[], added_files=added, all_v4_files_unchanged=True,
        reason='Only explicit dim5 continuation/finished dim1 inheritance and host hard floor6->3GiB')


def measured_gate(plan):
    old = Path(plan['interrupted_launch']['smoke_root'])
    with continuation_resources():
        profiles = [low.profile_receipt(original.read(old / 'profiles' / f'smoke_dim{dim}.json'))
                    for dim in original.DIMENSIONS]
    peak = max(profile['peak']['pss_anon_bytes'] for profile in profiles) / low.GIB
    return dict(mem_available_gib=max(INITIAL_AVAILABLE_GIB, HOST_RESERVE_GIB + 1.5 * peak + 2),
        measured_max_smoke_pss_anon_gib=peak, host_reserve_gib=HOST_RESERVE_GIB,
        peak_multiplier=1.5, additional_margin_gib=2,
        formula='max(16, 3 + 1.5 * inherited_four_smoke_peak_pss_anon_gib + 2)',
        scope='All new formal training and CPU reports; continuous owned-process guards remain active',
        inherited_independent_preflight=True)


def build_plan(interrupted_root, output):
    record = verify_handoff(interrupted_root, output)
    old = original.read(Path(record['root']) / 'plan.json')
    plan = original.build_plan(old['source_run'], output)
    keys = ('source_run', 'source_sha256', 'immutable_stat_identity', 'data_sha256', 'mi_cache_dir', 'mi_cache_keys')
    if (canonical([arm['config'] for arm in plan['arms']]) != canonical([arm['config'] for arm in old['arms']])
            or any(canonical(plan[key]) != canonical(old[key]) for key in keys)):
        raise RuntimeError('Continuation must preserve every scientific configuration and input identity')
    plan.update(schema_version=5, execution_version=EXECUTION_VERSION, interrupted_launch=record,
                inherited_completed_arms=['dim1'])
    for arm, previous in zip(plan['arms'], old['arms']):
        arm.pop('smoke')
        arm['inherited'] = arm['name'] == 'dim1'
        if arm['inherited']:
            arm.update(root=previous['root'], config_path=previous['config_path'],
                training=None, report=None, resume_from=None, segment_start_update=original.UPDATES,
                new_updates=0, inherited_training_receipt=record['receipt']['inherited_dim1_training_receipt'],
                inherited_report_receipt=record['receipt']['inherited_dim1_report_receipt'])
            continue
        command = arm['training']
        command[command.index('scripts.run_ising_highdim')] = 'scripts.run_ising_highdim_resume_v5'
        command.extend(['--benchmark-warmup', '0'])
        arm['resume_from'] = record['checkpoint']['path'] if arm['name'] == 'dim5' else None
        arm['segment_start_update'] = CHECKPOINT_STEP if arm['resume_from'] else 0
        arm['new_updates'] = original.UPDATES - arm['segment_start_update']
        if arm['resume_from']:
            command.extend(['--resume-checkpoint', arm['resume_from'],
                            '--resume-receipt', str(Path(arm['root']) / 'resume.json')])
    root = Path(plan['output'])
    plan['compare'] = [sys.executable, '-u', '-m', 'scripts.report_ising_highdim', '--compare',
        *(f"{arm['name']}={arm['root']}/report" for arm in plan['arms']), '--output', str(root / 'comparison')]
    plan['protocol'].update(resume=True, automatic_retry=False,
        initialization='dim1继承完成的12000步模型与正式报告；dim5严格恢复8000步完整状态；dim10/100保持新初始化',
        source_usage='续训前仅小型回执和选定checkpoint允许读取，物理data/MI仅stat身份核验；正式训练复用原物理data与data-only MI缓存，不重hash/采样/重建',
        continuation='dim5累计终点12000，新增4000，重新执行未保存的8001..8400；不恢复日志状态',
        smoke='继承v3四维独立真实数据2步预检及v4合成CUDA严格续训等价回执；不额外预检/采样/训练')
    plan['protocol']['resource_gate'].update(mem_available_gib='measured formula with unchanged initial16GiB', gpu_free_mib=16384)
    plan['low_ram_execution'] = deepcopy(old['low_ram_execution'])
    plan['low_ram_execution'].update(initial_validation_available_gib=INITIAL_AVAILABLE_GIB,
        initial_validation_justification='Admission remains16GiB despite host hard floor6->3GiB',
        host_available_hard_floor_gib=HOST_RESERVE_GIB,
        resource_only_change='User-authorized host reserve6->3GiB; own anonymous6/GPU14GiB and all admission gates unchanged',
        formal_admission_formula='max(16, 3 + 1.5 * inherited_four_smoke_peak_pss_anon_gib + 2)')
    return plan


def validate_training(plan, arm):
    if arm.get('inherited'):
        raise RuntimeError('Completed dim1 must be inherited, never executed or revalidated as a new stage')
    if not arm['resume_from']:
        receipt = low.validate_training(plan, arm, smoke=False)
        saved = checkpoint_receipt(Path(arm['root']) / 'run/training_step_12000.pt', arm['config'], step=original.UPDATES)
        receipt['all_adam_parameter_steps'] = saved['all_adam_parameter_steps']
        return receipt
    from recurrent_rg.config import RunConfig
    from recurrent_rg.learning_rate import LearningRatePolicy
    from experiments.rsmi_separate_qcr import _INITIAL_MODULE_METADATA_FIELDS
    run = Path(arm['root']) / 'run'
    manifest, summary = original.read(run / 'manifest.json'), original.read(run / 'summary.json')
    source = original.read(Path(plan['interrupted_launch']['root']) / 'dim5/run/manifest.json')
    policy = LearningRatePolicy(RunConfig.from_dict(arm['config']).training)
    for key in ('training_driver', 'precision', 'estimator', 'environment_mode', 'buffer',
                'independent_c', 'labels_used_in_training', 'rsmi_weight', 'execution_device', 'source',
                'q_supervision', 'frozen_C', 'data_driven_initialization', 'q_future_physics',
                'data_sha256', 'initial_module_sha256', 'training_start_module_sha256',
                *_INITIAL_MODULE_METADATA_FIELDS):
        if manifest.get(key) != source.get(key):
            raise RuntimeError('Resumed dim5 science/initialization differs from selected source')
    def execution_semantics(value):
        return {key: item for key, item in value.get('execution_optimizations', {}).items()
                if key != 'mi_statistics_cache'}
    if execution_semantics(manifest) != execution_semantics(source):
        raise RuntimeError('Resumed execution semantics differ from selected source')
    if (manifest.get('resumed_from') != arm['resume_from']
            or canonical(manifest.get('config')) != canonical(arm['config'])
            or canonical(summary.get('config')) != canonical(arm['config'])
            or Path(manifest.get('config_source', '')).resolve() != Path(arm['config_path']).resolve()
            or manifest.get('resume_start_provenance', {}).get('source_epoch') != CHECKPOINT_STEP
            or not manifest.get('resume_start_module_sha256')
            or summary.get('actual_epochs') != original.UPDATES or summary.get('requested_epochs') != original.UPDATES
            or any(summary.get(key) for key in ('stopped_by_training_limit', 'stopped_by_wall_limit', 'stopped_by_train_plateau'))
            or summary.get('final_report_skipped') is not True
            or summary.get('lr_scheduler_state') != policy.checkpoint_state(original.UPDATES)
            or (run.parent / 'initialization.json').exists()):
        raise RuntimeError('Resumed dim5 endpoint/source/initialization distinction differs')
    provenance = original.read(run.parent / 'resume.json')
    if (provenance.get('status') != 'complete' or provenance.get('fresh') is not False
            or provenance.get('labels_used_in_training') is not False
            or provenance.get('source_checkpoint') != arm['resume_from']
            or provenance.get('source_checkpoint_sha256') != plan['interrupted_launch']['checkpoint']['sha256']
            or provenance.get('source_epoch') != CHECKPOINT_STEP
            or provenance.get('resume_start_module_sha256') != manifest['resume_start_module_sha256']):
        raise RuntimeError('Explicit continuation receipt differs from loaded checkpoint provenance')
    history = summary.get('history', [])
    expected = sorted({1, original.UPDATES, *range(arm['config']['training']['log_every'], original.UPDATES + 1,
                                                 arm['config']['training']['log_every'])})
    if ([row.get('epoch') for row in history] != expected
            or any(not math.isclose(row.get('learning_rate', -1.), policy.rate_for_update(row['epoch']),
                                   rel_tol=1e-12) for row in history)):
        raise RuntimeError('Resumed dim5 history/update/LR clock differs')
    for split, key in plan['mi_cache_keys'].items():
        record = manifest['execution_optimizations']['mi_statistics_cache'].get(split, {})
        if (record.get('hit') is not True or record.get('key') != key or record.get('cache_version') != 1
                or Path(record.get('path', '')).resolve() != Path(plan['mi_cache_dir']) / f'mi-v1-{key}.pt'):
            raise RuntimeError('Continuation must hit the exact unchanged data-only MI caches')
    timing = summary.get('timing', {})
    if (timing.get('enabled') is not True or timing.get('updates') != arm['new_updates']
            or any(type(timing.get(key)) is not int or not 0 < timing[key] <= low.GPU_LIMIT_GIB * low.GIB
                   for key in ('cuda_max_memory_allocated_bytes', 'cuda_max_memory_reserved_bytes'))):
        raise RuntimeError('Continuation profiling must count only4000newupdates and preserve14GiBcap')
    checkpoint = checkpoint_receipt(run / 'training_step_12000.pt', arm['config'], step=original.UPDATES)
    previous = checkpoint_receipt(arm['resume_from'], arm['config'], step=CHECKPOINT_STEP)
    if history[:len(previous['history'])] != previous['history']:
        raise RuntimeError('Checkpoint8000 history was not preserved verbatim')
    return dict(actual_updates=original.UPDATES, segment_start_update=CHECKPOINT_STEP,
        new_updates=arm['new_updates'], all_adam_parameter_steps=checkpoint['all_adam_parameter_steps'],
        labels_used_in_training=False, resumed_from=arm['resume_from'],
        source_checkpoint_sha256=previous['sha256'], final_checkpoint_sha256=checkpoint['sha256'],
        timing=timing, manifest_sha256=original.sha(run / 'manifest.json'))


def stage(root, job, name, command, *, gpu, plan, fingerprints, configs, check):
    ensure_source(plan)
    def checked():
        ensure_source(plan)
        result = check()
        ensure_source(plan)
        return result
    return low.stage(root, job, name, command, gpu=gpu, plan=plan,
                     fingerprints=fingerprints, configs=configs, check=checked)


def execute(plan):
    root = Path(plan['output'])
    if root.exists() or root.is_symlink():
        raise FileExistsError('Refuse duplicate continuation or overwrite')
    if socket.gethostname() != original.HOST or os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise RuntimeError('Require host225 and native physical GPU2 indexing')
    if os.getpriority(os.PRIO_PROCESS, 0) != 10:
        raise RuntimeError('Require scheduled nice10; no online renice')
    if verify_handoff(plan['interrupted_launch']['root'], root) != plan['interrupted_launch']:
        raise RuntimeError('Interrupted v4 evidence changed after planning')
    project = Path(__file__).resolve().parents[1]
    delta = validate_code_delta(plan, project)
    fingerprints = original.source_hashes(project)
    root.mkdir(parents=True, exist_ok=False)
    for folder in ('logs', 'configs', 'smoke', 'code', 'profiles'):
        (root / folder).mkdir()
    for name, digest in fingerprints.items():
        target = root / 'code' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        original.shutil.copy2(project / name, target)
        if original.sha(target) != digest:
            raise RuntimeError('Frozen code snapshot copy mismatch')
    original.save(root / 'code/fingerprints.json', fingerprints, exclusive=True)
    configs = {}
    old_root = Path(plan['interrupted_launch']['root'])
    for arm in plan['arms']:
        if arm['inherited']:
            configs[arm['config_path']] = original.sha(arm['config_path'])
        else:
            Path(arm['root']).mkdir()
            original.save(arm['config_path'], arm['config'], exclusive=True)
            configs[arm['config_path']] = original.sha(arm['config_path'])
        target = root / 'smoke' / arm['name']
        target.mkdir()
        original.shutil.copy2(old_root / 'smoke' / arm['name'] / 'initialization.json', target / 'initialization.json')
    plan.update(source_code_sha256=fingerprints, configuration_sha256=configs, source_code_delta=delta)
    original.save(root / 'plan.json', plan, exclusive=True)
    gate = measured_gate(plan)
    original.save(root / 'measured_resource_gate.json', gate, exclusive=True)
    original.save(root / 'inherited_evidence.json', plan['interrupted_launch'], exclusive=True)
    job = dict(status='running', workflow_complete=False, started_at=original.now(), parent_pid=os.getpid(),
        host=original.HOST, device='cuda:2', gpu_uuid=original.GPU_UUID, actual_nice=10,
        current=None, stages=[], GPU_stage_seconds=0., physical_sampling_updates=0,
        labels_used_in_training=False, completed_arms=['dim1'], inherited_completed_arms=['dim1'],
        public_files=[], error=None, requires_user_action=False, execution_version=EXECUTION_VERSION,
        interrupted_launch=plan['interrupted_launch'], measured_resource_gate=gate,
        all_smoke_checks_passed=True, smoke_evidence_inherited=True, resume_cuda_evidence_inherited=True,
        continued_from_actual_update=CHECKPOINT_STEP, discarded_unsaved_logged_updates=400,
        inherited_receipts={'train_dim1': plan['arms'][0]['inherited_training_receipt'],
                            'report_dim1': plan['arms'][0]['inherited_report_receipt']})
    original.save(root / 'job.json', job, exclusive=True)
    started, cpu_start = time.monotonic(), original.cpu_seconds()
    try:
        with continuation_resources():
            for arm in plan['arms']:
                if arm['inherited']:
                    continue
                stage(root, job, 'train_' + arm['name'], arm['training'], gpu=True, plan=plan,
                    fingerprints=fingerprints, configs=configs, check=lambda a=arm: validate_training(plan, a))
                stage(root, job, 'report_' + arm['name'], arm['report'], gpu=False, plan=plan,
                    fingerprints=fingerprints, configs=configs,
                    check=lambda a=arm: original.report_receipt(Path(a['root']) / 'report', a['theory_dim']))
                job['completed_arms'].append(arm['name'])
                original.save(root / 'job.json', job)
            stage(root, job, 'comparison', plan['compare'], gpu=False, plan=plan,
                fingerprints=fingerprints, configs=configs, check=lambda: original.report_receipt(root / 'comparison'))
        ensure_source(plan)
        public = ['job.json', 'plan.json', 'measured_resource_gate.json', 'inherited_evidence.json',
                  *[str(Path(path).relative_to(root)) for path in configs if root in Path(path).parents]]
        public += [str(path.relative_to(root)) for path in sorted(root.rglob('*'))
                   if path.is_file() and not path.is_symlink() and 'code' not in path.relative_to(root).parts
                   and path.suffix in ('.json', '.png', '.npz') and str(path.relative_to(root)) not in public]
        job.update(status='complete', workflow_complete=True, current=None, ended_at=original.now(), public_files=public)
    except BaseException as error:
        job.update(status='failed', workflow_complete=False, requires_user_action=True,
                   error=f'{type(error).__name__}: {error}', ended_at=original.now())
        raise
    finally:
        job.update(elapsed_seconds=time.monotonic() - started, CPU_core_seconds=original.cpu_seconds() - cpu_start)
        original.save(root / 'job.json', job)
    return job


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--interrupted-root', type=Path)
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--cached-training', action='store_true')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--device', default='cuda:2')
    parser.add_argument('--mi-cache-dir', type=Path)
    parser.add_argument('--initialization-receipt', type=Path)
    parser.add_argument('--resume-checkpoint', type=Path)
    parser.add_argument('--resume-receipt', type=Path)
    parser.add_argument('--benchmark-warmup', type=int, default=0)
    args = parser.parse_args(argv)
    if args.cached_training:
        if (args.device != 'cuda:2' or args.benchmark_warmup != 0
                or any(value is None for value in (args.config, args.mi_cache_dir, args.initialization_receipt))):
            parser.error('Require exact physicalGPU2/cache/config/provenance/profiling settings')
        from scripts.run_ising2d_dimg_ablation import cached_training
        return cached_training(args)
    if args.interrupted_root is None or args.resume_checkpoint is not None:
        parser.error('Require interrupted-root; only the pinned dim5 checkpoint may resume')
    plan = build_plan(args.interrupted_root, args.output)
    if args.plan:
        print(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False))
        return plan
    return execute(plan)


if __name__ == '__main__':
    main()
