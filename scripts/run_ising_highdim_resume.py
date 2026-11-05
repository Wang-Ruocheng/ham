"""Explicit, frozen continuation after the v3 shared-host reserve stop.

Only dim1 resumes the saved 5000-update checkpoint.  The other three arms stay
fresh, and every scientific configuration/input remains identical to v3.
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

EXECUTION_VERSION = 'resume_v4'
HOST_RESERVE_GIB = 6.
INITIAL_AVAILABLE_GIB = 16.
CHECKPOINT_STEP = 5000
LAST_LOGGED_STEP = 5200
PREVIOUS_ERROR = 'RuntimeError: Shared-host MemAvailable fell below 8 GiB reserve'
RESOURCE_CHANGE = dict(previous_host_floor_gib=8, host_floor_gib=6,
    own_anon_limit_gib=6, own_gpu_limit_gib=14, minimum_free_gpu_gib=16,
    initial_available_gib=16, scientific_configuration_unchanged=True)
EXPECTED_OLD_STAGES = ['cuda_verification', 'smoke_dim1', 'smoke_dim5',
                       'smoke_dim10', 'smoke_dim100', 'train_dim1']
ALLOWED_EXISTING_CODE_CHANGES = {'scripts/run_ising2d_dimg_ablation.py'}


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def guard_violation(sample, *, gpu):
    if sample['mem_available_gib'] < HOST_RESERVE_GIB:
        return 'Shared-host MemAvailable fell below 6 GiB reserve'
    # Delegate all other independently unchanged protections to the v3 guard.
    return _V3_GUARD(dict(sample, mem_available_gib=max(8., sample['mem_available_gib'])), gpu=gpu)


_V3_GUARD = low.guard_violation


@contextmanager
def continuation_resources():
    """Use v3's audited owned-session monitor without rewriting frozen v3 code."""
    values = {name: getattr(low, name) for name in
              ('HOST_RESERVE_GIB', 'INITIAL_AVAILABLE_GIB', 'guard_violation')}
    low.HOST_RESERVE_GIB, low.INITIAL_AVAILABLE_GIB = HOST_RESERVE_GIB, INITIAL_AVAILABLE_GIB
    low.guard_violation = guard_violation
    try:
        yield
    finally:
        for name, value in values.items():
            setattr(low, name, value)


def checkpoint_receipt(path, config, *, step):
    """Read only the explicitly selected small model checkpoint, never data/MI."""
    import torch
    from recurrent_rg.config import RunConfig
    from recurrent_rg.learning_rate import LearningRatePolicy
    from scripts.run_ising2d_dimg_ablation import _tensor_digest
    path = Path(path)
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 64 * 1024 ** 2:
        raise RuntimeError('Require the selected small regular QCR checkpoint')
    before = original.sha(path)
    payload = torch.load(str(path), map_location='cpu', weights_only=True)
    metadata = payload.get('metadata', {})
    required = dict(training_driver='rsmi_separate_qcr', precision='torch.float64',
        estimator='histogram', environment_mode='full_shell', buffer=1, independent_c=True,
        labels_used_in_training=False, rsmi_weight=2., execution_device='cuda:2', source=None)
    fields = ('model', 'optimizer', 'history', 'torch_rng_state', 'cuda_rng_state',
              'batch_generator_state', 'rsmi_generator_state', 'lr_scheduler_state')
    if (payload.get('epoch') != step or payload.get('cuda_device_index') != 2
            or any(name not in payload for name in fields)
            or any(metadata.get(key) != value for key, value in required.items())
            or canonical(metadata.get('config')) != canonical(config)
            or metadata.get('q_supervision') or metadata.get('frozen_C')
            or metadata.get('data_driven_initialization') or metadata.get('q_future_physics')):
        raise RuntimeError('Checkpoint model/configuration/precision/routing/RNG provenance differs')
    states = payload['optimizer'].get('state', {})
    groups = payload['optimizer'].get('param_groups', [])
    parameter_ids = [identifier for group in groups for identifier in group.get('params', [])]
    policy = LearningRatePolicy(RunConfig.from_dict(config).training)
    if (not states or set(parameter_ids) != set(states)
            or len(parameter_ids) != len(set(parameter_ids))
            or any(float(value.get('step', -1)) != step for value in states.values())
            or payload['lr_scheduler_state'] != policy.checkpoint_state(step)
            or any(not math.isclose(group.get('lr', -1), policy.rate_for_update(step), rel_tol=1e-12)
                   for group in groups)):
        raise RuntimeError('Every Adam parameter and learning-rate clock must match checkpoint update')
    if (not payload['model'] or any(not torch.isfinite(value).all() for value in payload['model'].values())
            or any(not torch.isfinite(value).all() for state in states.values()
                   for value in state.values() if isinstance(value, torch.Tensor))
            or any(not isinstance(payload[field], torch.Tensor) or payload[field].numel() == 0
                   for field in fields if field.endswith('_state') and field != 'lr_scheduler_state')):
        raise RuntimeError('Checkpoint state is missing or nonfinite')
    expected = sorted({1, step, *range(config['training']['log_every'], step + 1,
                                      config['training']['log_every'])})
    if [row.get('epoch') for row in payload['history']] != expected:
        raise RuntimeError('Checkpoint history does not end at its numbered actual update')
    if original.sha(path) != before:
        raise RuntimeError('Checkpoint changed during read-only validation')
    return dict(path=str(path.resolve()), sha256=before, actual_updates=step,
        all_adam_parameter_steps=step, adam_state_count=len(states), cuda_device_index=2,
        model_tensor_sha256=_tensor_digest(payload['model']),
        history_sha256=__import__('hashlib').sha256(canonical(payload['history']).encode()).hexdigest(),
        history=payload['history'], metadata=metadata, exact_rng_and_optimizer_present=True)


def verify_handoff(interrupted_root, output):
    root, output = Path(interrupted_root).resolve(), Path(output).resolve()
    receipt_path = root / 'resume_handoff.json'
    receipt = original.read(receipt_path)
    plan, job = original.read(root / 'plan.json'), original.read(root / 'job.json')
    profile_path = root / 'profiles/train_dim1.json'
    profile = original.read(profile_path)
    checkpoint = root / 'dim1/run/training_step_5000.pt'
    manifest_path = root / 'dim1/run/manifest.json'
    stages = job.get('stages', [])
    expected_hashes = dict(old_plan_sha256=root / 'plan.json', old_job_sha256=root / 'job.json',
        old_profile_sha256=profile_path, old_manifest_sha256=manifest_path, checkpoint_sha256=checkpoint)
    if (receipt.get('schema_version') != 1
            or receipt.get('status') != 'approved_exact_checkpoint_continuation'
            or any(receipt.get(key) is not True for key in
                ('user_authorized', 'termination_confirmed', 'old_directory_preserved'))
            or Path(receipt.get('new_root', '')).resolve() != output
            or output == root or root in output.parents or output in root.parents
            or output.exists() or output.is_symlink()
            or any(receipt.get(key) != original.sha(path) for key, path in expected_hashes.items())
            or Path(receipt.get('checkpoint_path', '')).resolve() != checkpoint
            or receipt.get('checkpoint_step') != CHECKPOINT_STEP
            or receipt.get('last_logged_step') != LAST_LOGGED_STEP
            or receipt.get('formal_other_arms_started') is not False
            or receipt.get('physical_sampling_updates') != 0
            or receipt.get('resource_change') != RESOURCE_CHANGE
            or plan.get('execution_version') != 'lowram_v3'
            or Path(plan.get('output', '')).resolve() != root
            or job.get('execution_version') != 'lowram_v3'
            or job.get('status') != 'failed' or job.get('workflow_complete') is not False
            or job.get('requires_user_action') is not True or job.get('error') != PREVIOUS_ERROR
            or job.get('current') != 'train_dim1' or job.get('completed_arms') != []
            or job.get('all_smoke_checks_passed') is not True
            or job.get('host') != original.HOST or job.get('device') != 'cuda:2'
            or job.get('gpu_uuid') != original.GPU_UUID
            or job.get('physical_sampling_updates') != 0 or job.get('labels_used_in_training') is not False
            or [row.get('name') for row in stages] != EXPECTED_OLD_STAGES
            or [(row.get('status'), row.get('returncode')) for row in stages]
                != [('complete', 0)] * 5 + [('failed', -15)]
            or stages[-1].get('error') != PREVIOUS_ERROR
            or receipt.get('parent_pid') != job.get('parent_pid')
            or receipt.get('child_pid') != stages[-1].get('pid')
            or profile.get('status') != 'failed' or profile.get('returncode') != -15
            or profile.get('violation') != PREVIOUS_ERROR.removeprefix('RuntimeError: ')
            or profile.get('monitor_error') != PREVIOUS_ERROR
            or profile.get('child_pid') != receipt.get('child_pid')
            or str(profile.get('child_starttime_ticks')) != str(receipt.get('child_starttime_ticks'))
            or not str(receipt.get('child_starttime_ticks', '')).isdigit()):
        raise RuntimeError('Prior launch is not the explicitly approved v3 host-reserve interruption')
    pids = {receipt['parent_pid'], *(row.get('pid') for row in stages)}
    if any(type(pid) is not int or pid <= 0 or low.process_identity(pid) is not None for pid in pids):
        raise RuntimeError('A prior recorded parent/child PID is still present; fail closed')
    if [arm.get('name') for arm in plan.get('arms', [])] != [f'dim{dim}' for dim in original.DIMENSIONS]:
        raise RuntimeError('Prior four-arm comparison changed')
    for arm in plan['arms']:
        if Path(arm['root']).resolve() != root / arm['name']:
            raise RuntimeError('Prior arm path changed')
        if arm['name'] != 'dim1' and any((Path(arm['root']) / name).exists()
                for name in ('run', 'report', 'initialization.json')):
            raise RuntimeError('Another formal arm already started')
    progress = original.read(root / 'dim1/run/history_progress.json')
    if not progress or progress[-1].get('epoch') != LAST_LOGGED_STEP:
        raise RuntimeError('Interrupted log endpoint differs from the approved checkpoint rollback')
    saved = sorted(int(path.stem.removeprefix('training_step_'))
        for path in (root / 'dim1/run').glob('training_step_*.pt'))
    if not saved or saved[-1] != CHECKPOINT_STEP:
        raise RuntimeError('Latest saved actual update differs; never silently select another checkpoint')
    original.ensure_unchanged(plan, root / 'code', plan['source_code_sha256'], plan['configuration_sha256'])
    inherited = {}
    for arm in plan['arms']:
        low.validate_training(plan, arm, smoke=True)
        name = 'smoke_' + arm['name']
        path = root / 'profiles' / f'{name}.json'
        low.profile_receipt(original.read(path))
        inherited[str(path)] = original.sha(path)
        init = root / 'smoke' / arm['name'] / 'initialization.json'
        inherited[str(init)] = original.sha(init)
    low.cuda_receipt(root / 'cuda_verification.json')
    inherited[str(root / 'cuda_verification.json')] = original.sha(root / 'cuda_verification.json')
    original_init = root / 'dim1/initialization.json'
    inherited[str(original_init)] = original.sha(original_init)
    selected = checkpoint_receipt(checkpoint, plan['arms'][0]['config'], step=CHECKPOINT_STEP)
    if (selected['metadata'].get('resumed_from')
            or canonical(selected['metadata']) != canonical(original.read(manifest_path))):
        raise RuntimeError('Selected source must be the original uninterrupted dim1 manifest')
    return dict(root=str(root), receipt_sha256=original.sha(receipt_path),
        receipt=receipt, inherited_public_sha256=inherited,
        checkpoint={key: value for key, value in selected.items() if key not in ('history', 'metadata')})


def ensure_source(plan):
    record = plan['interrupted_launch']
    root = Path(record['root'])
    if original.sha(root / 'resume_handoff.json') != record['receipt_sha256']:
        raise RuntimeError('Approved continuation handoff changed')
    paths = dict(old_plan_sha256=root / 'plan.json', old_job_sha256=root / 'job.json',
        old_profile_sha256=root / 'profiles/train_dim1.json',
        old_manifest_sha256=root / 'dim1/run/manifest.json',
        checkpoint_sha256=root / 'dim1/run/training_step_5000.pt')
    for key, path in paths.items():
        if original.sha(path) != record['receipt'][key]:
            raise RuntimeError('Interrupted-run or selected checkpoint evidence changed')
    for path, digest in record['inherited_public_sha256'].items():
        if original.sha(path) != digest:
            raise RuntimeError('Inherited preflight receipt changed')
    old = original.read(root / 'plan.json')
    original.ensure_unchanged(old, root / 'code', old['source_code_sha256'], old['configuration_sha256'])


def validate_code_delta(plan, project):
    old_root = Path(plan['interrupted_launch']['root'])
    old = original.read(old_root / 'plan.json')['source_code_sha256']
    current = original.source_hashes(project)
    changed = [name for name, digest in old.items() if current.get(name) != digest]
    if set(changed) - ALLOWED_EXISTING_CODE_CHANGES:
        raise RuntimeError('Exact continuation cannot change model, trainer, loss, data or execution implementation')
    return dict(existing_changed_files=sorted(changed), permitted_existing_changes=sorted(ALLOWED_EXISTING_CODE_CHANGES),
        reason='Only the cache-only wrapper adds explicit resume forwarding; scientific trainer/modules stay frozen')


def measured_gate(plan):
    old = Path(plan['interrupted_launch']['root'])
    profiles = [original.read(old / 'profiles' / f'smoke_dim{dim}.json') for dim in original.DIMENSIONS]
    peak = max(low.profile_receipt(profile)['peak']['pss_anon_bytes'] for profile in profiles) / low.GIB
    return dict(mem_available_gib=max(INITIAL_AVAILABLE_GIB, HOST_RESERVE_GIB + 1.5 * peak + 2),
        measured_max_smoke_pss_anon_gib=peak, host_reserve_gib=HOST_RESERVE_GIB,
        peak_multiplier=1.5, additional_margin_gib=2,
        formula='max(16, 6 + 1.5 * inherited_four_smoke_peak_pss_anon_gib + 2)',
        scope='All formal training and CPU reports; continuous own-process guards remain active',
        inherited_independent_preflight=True)


def build_plan(interrupted_root, output):
    record = verify_handoff(interrupted_root, output)
    old = original.read(Path(record['root']) / 'plan.json')
    plan = original.build_plan(old['source_run'], output)
    identity_keys = ('source_run', 'source_sha256', 'immutable_stat_identity', 'data_sha256', 'mi_cache_dir', 'mi_cache_keys')
    if (canonical([arm['config'] for arm in plan['arms']]) != canonical([arm['config'] for arm in old['arms']])
            or any(canonical(plan[key]) != canonical(old[key]) for key in identity_keys)):
        raise RuntimeError('Continuation must preserve every scientific configuration and input identity')
    plan.update(schema_version=4, execution_version=EXECUTION_VERSION, interrupted_launch=record)
    root = Path(plan['output'])
    for arm in plan['arms']:
        arm.pop('smoke')
        command = arm['training']
        command[command.index('scripts.run_ising_highdim')] = 'scripts.run_ising_highdim_resume'
        command.extend(['--benchmark-warmup', '0'])
        arm['resume_from'] = record['checkpoint']['path'] if arm['name'] == 'dim1' else None
        arm['segment_start_update'] = CHECKPOINT_STEP if arm['name'] == 'dim1' else 0
        arm['new_updates'] = original.UPDATES - arm['segment_start_update']
        if arm['resume_from']:
            command.extend(['--resume-checkpoint', arm['resume_from'],
                            '--resume-receipt', str(Path(arm['root']) / 'resume.json')])
    plan['resume_cuda_verification'] = [sys.executable, '-u', '-m', 'scripts.verify_highdim_resume',
        '--output', str(root / 'resume_cuda_verification.json'), '--device', 'cuda:2']
    plan['protocol'].update(resume=True, automatic_retry=False,
        initialization='dim1严格恢复5000步模型/Adam/LR/CPU与CUDA及批与MI RNG；其余维度保持原独立新初始化',
        source_usage='除原小型回执/物理数据/data-only MI外，仅dim1读取明确选定5000步checkpoint；不使用日志5200步的未保存状态',
        continuation='总终点仍12000；dim1新增7000步，重做5001..5200，不将日志未保存状态当checkpoint',
        smoke='复用四个已通过独立真实数据2步预检和CUDA等价回执；另做小型合成CUDA续训等价检查，不改正式科学设置')
    plan['protocol']['resource_gate'].update(mem_available_gib='measured formula with initial16GiB', gpu_free_mib=16384)
    plan['low_ram_execution'] = deepcopy(old['low_ram_execution'])
    plan['low_ram_execution'].update(initial_validation_available_gib=INITIAL_AVAILABLE_GIB,
        initial_validation_justification='6 GiB host reserve + 6 GiB own anonymous cap + 4 GiB extra headroom',
        host_available_hard_floor_gib=HOST_RESERVE_GIB,
        resource_only_change='User-authorized host reserve8->6GiB; exact checkpoint continuation; own anonymous6/GPU14GiB unchanged',
        formal_admission_formula='max(16, 6 + 1.5 * inherited_four_smoke_peak_pss_anon_gib + 2)')
    return plan


def validate_training(plan, arm):
    if not arm['resume_from']:
        receipt = low.validate_training(plan, arm, smoke=False)
        saved = checkpoint_receipt(Path(arm['root']) / 'run/training_step_12000.pt', arm['config'], step=original.UPDATES)
        receipt['all_adam_parameter_steps'] = saved['all_adam_parameter_steps']
        return receipt
    from recurrent_rg.config import RunConfig
    from recurrent_rg.learning_rate import LearningRatePolicy
    run = Path(arm['root']) / 'run'
    manifest, summary = original.read(run / 'manifest.json'), original.read(run / 'summary.json')
    source = original.read(Path(plan['interrupted_launch']['root']) / 'dim1/run/manifest.json')
    policy = LearningRatePolicy(RunConfig.from_dict(arm['config']).training)
    for key in ('training_driver', 'precision', 'estimator', 'environment_mode', 'buffer',
                'independent_c', 'labels_used_in_training', 'rsmi_weight', 'execution_device', 'source',
                'q_supervision', 'frozen_C', 'data_driven_initialization',
                'q_future_physics', 'data_sha256', 'initial_module_sha256', 'training_start_module_sha256'):
        if manifest.get(key) != source.get(key):
            raise RuntimeError('Resumed training recipe/provenance differs from selected source')
    from experiments.rsmi_separate_qcr import _INITIAL_MODULE_METADATA_FIELDS
    if any(manifest.get(key) != source.get(key) for key in _INITIAL_MODULE_METADATA_FIELDS):
        raise RuntimeError('Resumed source initialization provenance was not preserved')
    # Cache load/build durations are measured anew, not scientific execution
    # conventions.  Their exact semantic identities are checked below.
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
        raise RuntimeError('Resumed endpoint, exact source or initialization distinction differs')
    provenance = original.read(run.parent / 'resume.json')
    if (provenance.get('status') != 'complete' or provenance.get('fresh') is not False
            or provenance.get('labels_used_in_training') is not False
            or provenance.get('source_checkpoint') != arm['resume_from']
            or provenance.get('source_checkpoint_sha256') != plan['interrupted_launch']['checkpoint']['sha256']
            or provenance.get('source_epoch') != CHECKPOINT_STEP
            or provenance.get('resume_start_module_sha256') != manifest['resume_start_module_sha256']):
        raise RuntimeError('Explicit continuation receipt is missing or differs from loaded checkpoint provenance')
    history = summary.get('history', [])
    expected = sorted({1, original.UPDATES, *range(arm['config']['training']['log_every'], original.UPDATES + 1,
                                                 arm['config']['training']['log_every'])})
    if ([row.get('epoch') for row in history] != expected
            or any(not math.isclose(row.get('learning_rate', -1.), policy.rate_for_update(row['epoch']),
                                   rel_tol=1e-12) for row in history)):
        raise RuntimeError('Resumed history/update/LR clock differs')
    for split, key in plan['mi_cache_keys'].items():
        record = manifest['execution_optimizations']['mi_statistics_cache'].get(split, {})
        if (record.get('hit') is not True or record.get('key') != key or record.get('cache_version') != 1
                or Path(record.get('path', '')).resolve() != Path(plan['mi_cache_dir']) / f'mi-v1-{key}.pt'):
            raise RuntimeError('Continuation must hit the exact unchanged data-only MI caches')
    timing = summary.get('timing', {})
    if (timing.get('enabled') is not True or timing.get('updates') != arm['new_updates']
            or any(type(timing.get(key)) is not int or not 0 < timing[key] <= low.GPU_LIMIT_GIB * low.GIB
                   for key in ('cuda_max_memory_allocated_bytes', 'cuda_max_memory_reserved_bytes'))):
        raise RuntimeError('Continuation profiling must count only the7000newupdates and preserve14GiBcap')
    checkpoint = checkpoint_receipt(run / 'training_step_12000.pt', arm['config'], step=original.UPDATES)
    old_checkpoint = checkpoint_receipt(arm['resume_from'], arm['config'], step=CHECKPOINT_STEP)
    if history[:len(old_checkpoint['history'])] != old_checkpoint['history']:
        raise RuntimeError('Checkpoint history was not preserved verbatim')
    return dict(actual_updates=original.UPDATES, segment_start_update=CHECKPOINT_STEP,
        new_updates=arm['new_updates'], all_adam_parameter_steps=checkpoint['all_adam_parameter_steps'],
        labels_used_in_training=False, resumed_from=arm['resume_from'],
        source_checkpoint_sha256=old_checkpoint['sha256'], final_checkpoint_sha256=checkpoint['sha256'],
        timing=timing, manifest_sha256=original.sha(run / 'manifest.json'))


def cuda_receipt(path):
    receipt = original.read(path)
    if (receipt.get('status') != 'complete' or receipt.get('passed') is not True
            or receipt.get('synthetic_only') is not True or receipt.get('scientific_training') is not False
            or receipt.get('production_data_or_MI_opened') is not False
            or receipt.get('device') != 'cuda:2' or receipt.get('physical_uuid') != original.GPU_UUID
            or receipt.get('dimension') != 1 or receipt.get('total_updates') != 4
            or receipt.get('split_update') != 2):
        raise RuntimeError('Synthetic CUDA exact-continuation verification did not pass')
    return receipt


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
        raise RuntimeError('Interrupted-run evidence changed after planning')
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
        Path(arm['root']).mkdir()
        original.save(arm['config_path'], arm['config'], exclusive=True)
        configs[arm['config_path']] = original.sha(arm['config_path'])
        # JSON-only evidence of the original fresh constructions; no smoke
        # weights or optimizer state are reused by the three fresh arms.
        target = root / 'smoke' / arm['name']
        target.mkdir()
        original.shutil.copy2(old_root / 'smoke' / arm['name'] / 'initialization.json', target / 'initialization.json')
    plan.update(source_code_sha256=fingerprints, configuration_sha256=configs, source_code_delta=delta)
    original.save(root / 'plan.json', plan, exclusive=True)
    gate = measured_gate(plan)
    original.save(root / 'measured_resource_gate.json', gate, exclusive=True)
    job = dict(status='running', workflow_complete=False, started_at=original.now(), parent_pid=os.getpid(),
        host=original.HOST, device='cuda:2', gpu_uuid=original.GPU_UUID, actual_nice=10,
        current=None, stages=[], GPU_stage_seconds=0., physical_sampling_updates=0,
        labels_used_in_training=False, completed_arms=[], public_files=[], error=None,
        requires_user_action=False, execution_version=EXECUTION_VERSION,
        interrupted_launch=plan['interrupted_launch'], measured_resource_gate=gate,
        all_smoke_checks_passed=True, smoke_evidence_inherited=True,
        continued_from_actual_update=CHECKPOINT_STEP, discarded_unsaved_logged_updates=200)
    original.save(root / 'job.json', job, exclusive=True)
    started, cpu_start = time.monotonic(), original.cpu_seconds()
    try:
        with continuation_resources():
            stage(root, job, 'resume_cuda_verification', plan['resume_cuda_verification'], gpu=True,
                plan=plan, fingerprints=fingerprints, configs=configs,
                check=lambda: cuda_receipt(root / 'resume_cuda_verification.json'))
            for arm in plan['arms']:
                stage(root, job, 'train_' + arm['name'], arm['training'], gpu=True, plan=plan,
                    fingerprints=fingerprints, configs=configs, check=lambda a=arm: validate_training(plan, a))
                stage(root, job, 'report_' + arm['name'], arm['report'], gpu=False, plan=plan,
                    fingerprints=fingerprints, configs=configs,
                    check=lambda a=arm: original.report_receipt(Path(a['root']) / 'report', a['theory_dim']))
                job['completed_arms'].append(arm['name'])
                original.save(root / 'job.json', job)
            stage(root, job, 'comparison', plan['compare'], gpu=False, plan=plan,
                fingerprints=fingerprints, configs=configs, check=lambda: original.report_receipt(root / 'comparison'))
        public = ['job.json', 'plan.json', 'measured_resource_gate.json', 'resume_cuda_verification.json',
                  *[str(Path(path).relative_to(root)) for path in configs]]
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
        parser.error('Require interrupted-root; only the pinned dim1 checkpoint may resume')
    plan = build_plan(args.interrupted_root, args.output)
    if args.plan:
        print(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False))
        return plan
    return execute(plan)


if __name__ == '__main__':
    main()
