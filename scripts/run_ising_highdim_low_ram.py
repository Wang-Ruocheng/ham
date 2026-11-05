"""Versioned low-RAM execution of the unchanged b2 output-dimension experiment.

All four real-data two-update smokes precede training.  Only this launch's own
child sessions may be terminated by the explicit shared-host protection guard.
The superseded frozen launch, physical data and MI caches remain read-only.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from scripts import run_ising_highdim as original

GIB = 1024 ** 3
INITIAL_AVAILABLE_GIB = 18.
HOST_RESERVE_GIB = 8.
ANONYMOUS_LIMIT_GIB = 6.
GPU_LIMIT_GIB = 14.
GPU_ADMISSION_FREE_GIB = 16.
EXECUTION_VERSION = 'lowram_v3'
SAMPLE_SECONDS = .25
GPU_SAMPLE_SECONDS = 1.
PREVIOUS_PREFLIGHT_ERROR = 'RuntimeError: Own selected-GPU sampled memory exceeded 8 GiB'


def process_identity(pid):
    """Linux process identity; start ticks defeat reuse of a historical PID."""
    try:
        fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=int(pid), state=fields[0], ppid=int(fields[1]),
                    process_group=int(fields[2]), session=int(fields[3]),
                    starttime_ticks=fields[19],
                    uid=(Path('/proc') / str(pid)).stat().st_uid)
    except (FileNotFoundError, ProcessLookupError):
        return None


def verify_handoff(superseded_root, output):
    root = Path(superseded_root).resolve()
    if (root / 'preflight_handoff.json').exists():
        return verify_preflight_handoff(root, output)
    receipt_path = root / 'low_ram_handoff.json'
    receipt = original.read(receipt_path)
    old_job = original.read(root / 'job.json')
    if (receipt.get('status') != 'superseded_before_any_stage'
            or receipt.get('termination_confirmed') is not True
            or receipt.get('old_stages') != []
            or receipt.get('training_updates') != 0
            or receipt.get('physical_sampling_updates') != 0
            or receipt.get('old_directory_preserved') is not True
            or Path(receipt.get('new_root', '')).resolve() != Path(output).resolve()
            or receipt.get('old_plan_sha256') != original.sha(root / 'plan.json')
            or old_job.get('stages') != []
            or old_job.get('GPU_stage_seconds') != 0
            or old_job.get('workflow_complete') is not False):
        raise RuntimeError('Superseded launch was not stopped before every stage')
    identity = process_identity(receipt['parent_pid'])
    if (identity is not None
            and identity['starttime_ticks'] == str(receipt['parent_starttime_ticks'])
            and identity['state'] != 'Z'):
        raise RuntimeError('Superseded waiting parent is still alive')
    return dict(root=str(root), receipt_sha256=original.sha(receipt_path),
                old_plan_sha256=receipt['old_plan_sha256'],
                parent_pid=receipt['parent_pid'], parent_starttime_ticks=str(receipt['parent_starttime_ticks']),
                termination_confirmed=True, old_stages=[])


def verify_preflight_handoff(root, output):
    """Accept only the explicitly approved, stopped v2 VRAM preflight failure.

    This is a new initialization, never a checkpoint resume or automated retry.
    The independent receipt preserves the immutable v1 -> v2 handoff chain.
    """
    root, output = Path(root).resolve(), Path(output).resolve()
    receipt_path = root / 'preflight_handoff.json'
    receipt = original.read(receipt_path)
    old_plan, old_job = original.read(root / 'plan.json'), original.read(root / 'job.json')
    profile_path = root / 'profiles' / 'smoke_dim1.json'
    profile = original.read(profile_path)
    stage_names = ['cuda_verification', 'smoke_dim1']
    resource_change = dict(user_authorized=True, previous_own_gpu_limit_gib=8,
        own_gpu_limit_gib=GPU_LIMIT_GIB, minimum_free_gpu_gib=GPU_ADMISSION_FREE_GIB,
        scientific_configuration_unchanged=True)
    stages = old_job.get('stages', [])
    if (receipt.get('schema_version') != 1
            or receipt.get('status') != 'superseded_after_preflight_guard'
            or receipt.get('termination_confirmed') is not True
            or receipt.get('old_directory_preserved') is not True
            or Path(receipt.get('new_root', '')).resolve() != output
            or output == root or root in output.parents or output in root.parents
            or output.exists() or output.is_symlink()
            or receipt.get('old_plan_sha256') != original.sha(root / 'plan.json')
            or receipt.get('old_job_sha256') != original.sha(root / 'job.json')
            or receipt.get('old_profile_sha256') != original.sha(profile_path)
            or receipt.get('old_stage_names') != stage_names
            or receipt.get('old_error') != PREVIOUS_PREFLIGHT_ERROR
            or receipt.get('formal_training_updates') != 0
            or receipt.get('physical_sampling_updates') != 0
            or receipt.get('checkpoint_resume') is not False
            or receipt.get('resource_change') != resource_change
            or old_job.get('status') != 'failed'
            or old_job.get('workflow_complete') is not False
            or old_job.get('requires_user_action') is not True
            or old_job.get('error') != PREVIOUS_PREFLIGHT_ERROR
            or old_job.get('current') != 'smoke_dim1'
            or old_job.get('completed_arms') != []
            or old_job.get('physical_sampling_updates') != 0
            or old_job.get('labels_used_in_training') is not False
            or old_job.get('execution_version') != 'lowram_v2'
            or old_job.get('host') != original.HOST or old_job.get('device') != 'cuda:2'
            or old_job.get('gpu_uuid') != original.GPU_UUID
            or old_job.get('all_smoke_checks_passed', False) is not False
            or 'measured_resource_gate' in old_job
            or [row.get('name') for row in stages] != stage_names
            or [(row.get('status'), row.get('returncode')) for row in stages]
                != [('complete', 0), ('failed', -15)]
            or stages[-1].get('error') != PREVIOUS_PREFLIGHT_ERROR
            or receipt.get('parent_pid') != old_job.get('parent_pid')
            or receipt.get('child_pid') != stages[-1].get('pid')
            or profile.get('status') != 'failed' or profile.get('returncode') != -15
            or profile.get('violation') != PREVIOUS_PREFLIGHT_ERROR.removeprefix('RuntimeError: ')
            or profile.get('monitor_error') != PREVIOUS_PREFLIGHT_ERROR
            or profile.get('child_pid') != receipt.get('child_pid')
            or str(profile.get('child_starttime_ticks')) != str(receipt.get('child_starttime_ticks'))
            or old_plan.get('execution_version') != 'lowram_v2'
            or old_plan.get('low_ram_execution', {}).get('own_gpu_hard_limit_gib') != 8
            or any(old_plan.get('protocol', {}).get('resource_gate', {}).get(key) != value
                for key, value in (('host', original.HOST), ('gpu', original.GPU_INDEX),
                                   ('gpu_uuid', original.GPU_UUID)))
            or Path(old_plan.get('output', '')).resolve() != root
            or [arm.get('name') for arm in old_plan.get('arms', [])]
                != [f'dim{dim}' for dim in original.DIMENSIONS]
            or not old_plan.get('source_code_sha256') or not old_plan.get('configuration_sha256')):
        raise RuntimeError('Prior launch is not the approved zero-formal-training v2 VRAM preflight failure')
    # Missing historical parent start ticks are not guessed.  Require complete
    # absence of every recorded PID; a reused live PID also fails closed.
    old_pids = {receipt['parent_pid'], *(row.get('pid') for row in stages)}
    if any(not isinstance(pid, int) or pid <= 0 for pid in old_pids):
        raise RuntimeError('Missing prior preflight process identity')
    if any(process_identity(pid) is not None for pid in old_pids):
        raise RuntimeError('Prior preflight parent or child PID is still present')
    if not str(receipt.get('child_starttime_ticks', '')).isdigit():
        raise RuntimeError('Missing prior smoke child start-time identity')
    for arm in old_plan['arms']:
        arm_root = Path(arm['root']).resolve()
        if arm_root != root / arm['name']:
            raise RuntimeError('Prior formal arm path changed')
        if any((arm_root / name).exists() for name in ('run', 'report', 'initialization.json')):
            raise RuntimeError('Prior formal training evidence exists; no automatic restart or resume')
    original.ensure_unchanged(old_plan, root / 'code', old_plan['source_code_sha256'],
                              old_plan['configuration_sha256'])
    return dict(root=str(root), kind='failed_preflight_resource_only',
        receipt_sha256=original.sha(receipt_path), old_plan_sha256=receipt['old_plan_sha256'],
        old_job_sha256=receipt['old_job_sha256'], old_profile_sha256=receipt['old_profile_sha256'],
        parent_pid=receipt['parent_pid'], parent_starttime_ticks=receipt.get('parent_starttime_ticks'),
        child_pid=receipt['child_pid'], child_starttime_ticks=str(receipt['child_starttime_ticks']),
        termination_confirmed=True, old_stage_names=stage_names, formal_training_updates=0,
        checkpoint_resume=False, resource_change=resource_change)


def build_plan(source_run, output, superseded_root):
    plan = original.build_plan(source_run, output)
    plan['schema_version'] = 3
    plan['execution_version'] = EXECUTION_VERSION
    plan['superseded_launch'] = verify_handoff(superseded_root, output)
    if plan['superseded_launch'].get('kind') == 'failed_preflight_resource_only':
        old_plan = original.read(Path(superseded_root) / 'plan.json')
        # RunConfig emits tuples for a few shape fields; JSON stores those as
        # arrays.  Compare their exact canonical JSON representation so a
        # serialization-only tuple/list distinction is not a science change.
        keys = ('source_run', 'source_sha256', 'immutable_stat_identity', 'data_sha256',
                'mi_cache_dir', 'mi_cache_keys')
        def science(value):
            return json.dumps(dict(arms=[arm['config'] for arm in value['arms']],
                inputs={key: value[key] for key in keys}), sort_keys=True, allow_nan=False)
        if science(plan) != science(old_plan):
            raise RuntimeError('Resource-only v3 must retain every v2 scientific configuration and input identity')
    root = Path(plan['output'])
    for arm in plan['arms']:
        for key in ('training', 'smoke'):
            command = arm[key]
            command[command.index('scripts.run_ising_highdim')] = 'scripts.run_ising_highdim_low_ram'
            command.extend(['--benchmark-warmup', '0'])
    plan['cuda_verification'] = [sys.executable, '-u', '-m', 'scripts.verify_low_ram_cuda',
                                 '--output', str(root / 'cuda_verification.json'),
                                 '--device', 'cuda:2', '--updates', '2']
    plan['protocol']['resource_gate']['mem_available_gib'] = 'measured formula, not the historical 36 GiB'
    plan['protocol']['resource_gate']['gpu_free_mib'] = int(GPU_ADMISSION_FREE_GIB * 1024)
    plan['low_ram_execution'] = dict(
        scientific_configuration_unchanged=True, reference_execution=False,
        initial_validation_available_gib=INITIAL_AVAILABLE_GIB,
        initial_validation_justification='8 GiB host reserve + 6 GiB own anonymous cap + 4 GiB extra headroom',
        host_available_hard_floor_gib=HOST_RESERVE_GIB,
        child_tree_pss_anon_hard_limit_gib=ANONYMOUS_LIMIT_GIB,
        own_gpu_hard_limit_gib=GPU_LIMIT_GIB,
        gpu_admission_free_gib=GPU_ADMISSION_FREE_GIB,
        resource_only_change='User-approved 8 -> 14 GiB own VRAM cap; start requires 16 GiB free; no scientific change or checkpoint reuse',
        formal_admission_formula='max(18, 8 + 1.5 * max(all four real-data smoke PSSAnon peaks in GiB) + 2)',
        report_admission='same measured conservative threshold; report retains continuous hard guard',
        sampling_seconds=SAMPLE_SECONDS, external_gpu_sampling_seconds=GPU_SAMPLE_SECONDS,
        peak_interpretation='sampled child-tree sums, not guaranteed instantaneous peaks; PSSFile is file-backed resident memory, not certified all clean or reclaimable; clean/dirty resident sums include libraries and anonymous pages and overlap RSS/PSS categories',
        cuda_allocator_peak='authoritative allocator lifetime maxima from each trainer summary; external nvidia-smi sampled separately',
        limit_action='terminate only this driver own child session, TERM then at most 5 seconds to KILL; fail closed; never retry',
        report_gpu=False, requires_all_smokes_before_training=True)
    return plan


def snapshot(required_available_gib):
    row = original.snapshot()
    row['required_mem_available_gib'] = required_available_gib
    row['host_ready'] = (row['mem_available_gib'] >= required_available_gib
        and row['load1'] <= 1.25 * row['cpu_count']
        and row['io_full_avg10'] < 20 and row['disk_free_gib'] >= 20)
    row['gpu_ready'] = (row['host_ready'] and not row['compute_processes']
                        and row['gpu_free_mib'] >= GPU_ADMISSION_FREE_GIB * 1024)
    return row


def admit(root, job, *, gpu, required_available_gib):
    ready = 0
    while True:
        row = snapshot(required_available_gib)
        job.setdefault('resource_checks', []).append(dict(row, requires_gpu=gpu))
        job['resource_checks'] = job['resource_checks'][-120:]
        ready = ready + 1 if row['gpu_ready' if gpu else 'host_ready'] else 0
        original.save(root / 'job.json', job)
        if ready >= (2 if gpu else 1):
            return
        time.sleep(10 if ready else 60)


def proc_kib(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        key, separator, rest = line.partition(':')
        fields = rest.split()
        if separator and len(fields) == 2 and fields[1] == 'kB':
            result[key] = int(fields[0]) * 1024
    return result


def child_tree(root_pid, start_ticks):
    """Only current descendants of our live root; no historical PID targeting."""
    top = process_identity(root_pid)
    if top is None or top['starttime_ticks'] != start_ticks:
        return []
    pending, found = [top], []
    while pending:
        row = pending.pop()
        if row['uid'] != os.getuid() or row['state'] == 'Z':
            continue
        found.append(row)
        try:
            child_ids = (Path('/proc') / str(row['pid']) / 'task' / str(row['pid']) / 'children').read_text().split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        for child_id in child_ids:
            child = process_identity(int(child_id))
            if child is not None and child['ppid'] == row['pid']:
                pending.append(child)
    return found


def tree_memory(identities):
    names = dict(rss_bytes='Rss', pss_bytes='Pss', pss_anon_bytes='Pss_Anon',
                 pss_file_bytes='Pss_File', pss_shmem_bytes='Pss_Shmem')
    result = {name: 0 for name in names}
    result.update(clean_resident_bytes=0, dirty_resident_bytes=0)
    result['pids'] = []
    for row in identities:
        try:
            measured = proc_kib(Path('/proc') / str(row['pid']) / 'smaps_rollup')
            after = process_identity(row['pid'])
        except (FileNotFoundError, ProcessLookupError):
            continue
        if after is None or after['starttime_ticks'] != row['starttime_ticks']:
            continue
        if 'Pss_Anon' not in measured:
            # Older Linux kernels expose Anonymous but not split PSS.  This
            # conservative fallback does not pretend shared anon is private.
            if 'Anonymous' not in measured:
                raise RuntimeError('Cannot measure own child anonymous memory')
            measured['Pss_Anon'] = measured['Anonymous']
        if not measured.get('Rss') or 'Pss_Anon' not in measured:
            raise RuntimeError('Cannot measure own child anonymous memory')
        result['pids'].append(row['pid'])
        for name, field in names.items():
            result[name] += measured.get(field, 0)
        result['clean_resident_bytes'] += measured.get('Shared_Clean', 0) + measured.get('Private_Clean', 0)
        result['dirty_resident_bytes'] += measured.get('Shared_Dirty', 0) + measured.get('Private_Dirty', 0)
    return result


def external_gpu_memory(own_pids):
    listing = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
        '--format=csv,noheader,nounits'], text=True, timeout=10)
    own, other = 0, []
    for line in listing.splitlines():
        uuid, pid, used = [field.strip() for field in line.split(',')]
        if uuid != original.GPU_UUID:
            continue
        if not used.isdigit():
            raise RuntimeError('Cannot measure selected GPU compute memory')
        if int(pid) in own_pids:
            own += int(used) * 1024 ** 2
        else:
            other.append(int(pid))
    return dict(own_gpu_bytes=own, other_gpu_pids=other)


def guard_violation(sample, *, gpu):
    if sample['mem_available_gib'] < HOST_RESERVE_GIB:
        return 'Shared-host MemAvailable fell below 8 GiB reserve'
    if sample['pss_anon_bytes'] > ANONYMOUS_LIMIT_GIB * GIB:
        return 'Own child-tree anonymous memory exceeded 6 GiB'
    if gpu and sample.get('own_gpu_bytes', 0) > GPU_LIMIT_GIB * GIB:
        return f'Own selected-GPU sampled memory exceeded {GPU_LIMIT_GIB:g} GiB'
    if gpu and sample.get('other_gpu_pids'):
        return 'Another compute process appeared on selected GPU; no sharing'
    return None


def terminate_own_session(child, identity):
    """Signal only our newly-created session while its root identity matches."""
    current = process_identity(child.pid)
    if (current is None or current['starttime_ticks'] != identity['starttime_ticks']
            or current['uid'] != os.getuid() or current['session'] != child.pid
            or current['process_group'] != child.pid):
        return dict(signal_sent=False, reason='child already gone or identity no longer matches')
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=5)
        return dict(signal_sent=True, signal='SIGTERM', returncode=child.returncode)
    except subprocess.TimeoutExpired:
        current = process_identity(child.pid)
        if (current is not None and current['starttime_ticks'] == identity['starttime_ticks']
                and current['session'] == child.pid and current['process_group'] == child.pid):
            os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=5)
        return dict(signal_sent=True, signal='SIGTERM then SIGKILL', returncode=child.returncode)


def monitor(child, *, gpu, profile_path):
    identity = process_identity(child.pid)
    if identity is None:
        raise RuntimeError('Own child vanished before Linux memory monitor attached')
    profile = dict(status='running', child_pid=child.pid, child_starttime_ticks=identity['starttime_ticks'],
        sample_count=0, peak={}, host_mem_available_min_gib=None, gpu_sample_count=0,
        sampling_seconds=SAMPLE_SECONDS, peak_is_sampled=True, external_gpu_peak_is_sampled=True,
        samples=[], last_sample=None)
    last_gpu, last_save = 0., 0.
    gpu_row = dict(own_gpu_bytes=0, other_gpu_pids=[])
    try:
        while child.poll() is None:
            tree = child_tree(child.pid, identity['starttime_ticks'])
            row = tree_memory(tree)
            available = proc_kib('/proc/meminfo')['MemAvailable'] / GIB
            if gpu and time.monotonic() - last_gpu >= GPU_SAMPLE_SECONDS:
                # NVML may retain our leader briefly after /proc has reaped it;
                # its known spawned PID is never mislabeled as another user.
                gpu_row = external_gpu_memory({child.pid, *(item['pid'] for item in tree)})
                profile['gpu_sample_count'] += 1
                last_gpu = time.monotonic()
            row.update(at=original.now(), mem_available_gib=available, **gpu_row)
            profile['sample_count'] += 1
            profile['last_sample'] = row
            profile['host_mem_available_min_gib'] = min(available,
                profile['host_mem_available_min_gib'] if profile['host_mem_available_min_gib'] is not None else available)
            for key, value in row.items():
                if key.endswith('_bytes'):
                    profile['peak'][key] = max(value, profile['peak'].get(key, 0))
            violation = guard_violation(row, gpu=gpu)
            if violation:
                profile['violation'] = violation
                raise RuntimeError(violation)
            if time.monotonic() - last_save >= 10:
                profile['samples'].append(row)
                profile['samples'] = profile['samples'][-360:]
                original.save(profile_path, profile)
                last_save = time.monotonic()
            time.sleep(SAMPLE_SECONDS)
        profile.update(status='complete' if child.returncode == 0 else 'failed', returncode=child.returncode)
    except BaseException as error:
        profile.update(status='failed', monitor_error=f'{type(error).__name__}: {error}',
                       termination=terminate_own_session(child, identity), returncode=child.poll())
        raise
    finally:
        original.save(profile_path, profile)
    if profile['sample_count'] < 1:
        raise RuntimeError('No child memory samples recorded')
    return profile


def profile_receipt(profile):
    if (profile.get('status') != 'complete' or profile.get('returncode') != 0
            or profile.get('violation') or profile.get('sample_count', 0) < 1
            or not profile.get('peak', {}).get('pss_anon_bytes')
            or profile.get('host_mem_available_min_gib', 0) < HOST_RESERVE_GIB
            or profile['peak']['pss_anon_bytes'] > ANONYMOUS_LIMIT_GIB * GIB
            or profile['peak'].get('own_gpu_bytes', 0) > GPU_LIMIT_GIB * GIB):
        raise RuntimeError('Missing or failed low-RAM profile')
    return profile


def measured_gate(profiles):
    if set(profiles) != {f'smoke_dim{dim}' for dim in original.DIMENSIONS}:
        raise RuntimeError('Require all four real-data smoke profiles')
    peak = max(profile_receipt(profile)['peak']['pss_anon_bytes'] for profile in profiles.values()) / GIB
    return dict(mem_available_gib=max(INITIAL_AVAILABLE_GIB, HOST_RESERVE_GIB + 1.5 * peak + 2),
                measured_max_smoke_pss_anon_gib=peak, host_reserve_gib=HOST_RESERVE_GIB,
                peak_multiplier=1.5, additional_margin_gib=2,
                formula='max(18, 8 + 1.5 * measured_peak_pss_anon_gib + 2)',
                scope='formal training and CPU reports; continuous hard guards remain active')


def validate_training(plan, arm, *, smoke):
    receipt = original.validate_training(plan, arm, smoke=smoke)
    run = (Path(plan['output']) / 'smoke' / arm['name'] / 'run' if smoke else Path(arm['root']) / 'run')
    manifest = original.read(run / 'manifest.json')
    policy = manifest.get('execution_optimizations', {}).get('low_ram_execution', {})
    if policy.get('enabled') is not True or policy.get('source_dtype_unchanged') is not True:
        raise RuntimeError('Low-RAM execution policy was not active')
    timing = receipt['timing']
    if (timing.get('enabled') is not True or timing.get('updates') != (2 if smoke else original.UPDATES)
            or any(not isinstance(timing.get(key), int) or not 0 < timing[key] <= GPU_LIMIT_GIB * GIB
                for key in ('cuda_max_memory_allocated_bytes', 'cuda_max_memory_reserved_bytes'))):
        raise RuntimeError(f'CUDA allocator profiling missing or exceeds own {GPU_LIMIT_GIB:g} GiB limit')
    receipt['low_ram_execution'] = policy
    return receipt


def cuda_receipt(path):
    receipt = original.read(path)
    dimensions = receipt.get('dimensions', [])
    if (receipt.get('status') != 'complete' or receipt.get('passed') is not True
            or receipt.get('physical_uuid') != original.GPU_UUID
            or receipt.get('device') != 'cuda:2'
            or receipt.get('synthetic_only') is not True
            or receipt.get('production_data_or_MI_opened') is not False
            or [row.get('dimension') for row in dimensions] != list(original.DIMENSIONS)
            or any(row.get('passed') is not True or row.get('updates') != 2 for row in dimensions)
            or any(not isinstance(receipt.get(key), int) or not 0 < receipt[key] <= GPU_LIMIT_GIB * GIB
                for key in ('peak_cuda_allocated_bytes', 'peak_cuda_reserved_bytes'))):
        raise RuntimeError('CUDA low-RAM/reference execution equivalence failed')
    return receipt


def stage(root, job, name, command, *, gpu, plan, fingerprints, configs, check):
    original.ensure_unchanged(plan, root / 'code', fingerprints, configs)
    job.update(status='waiting_resources', current=name)
    original.save(root / 'job.json', job)
    required = job.get('measured_resource_gate', {}).get('mem_available_gib', INITIAL_AVAILABLE_GIB)
    admit(root, job, gpu=gpu, required_available_gib=required)
    original.ensure_unchanged(plan, root / 'code', fingerprints, configs)
    row = dict(name=name, status='running', command=command, gpu=gpu,
        started_at=original.now(), log=str(root / 'logs' / f'{name}.log'),
        profile=str(root / 'profiles' / f'{name}.json'), admission_available_gib=required)
    job['stages'].append(row)
    env = dict(os.environ, PYTHONPATH=str(root / 'code'), OMP_NUM_THREADS='2' if gpu else '1',
        MKL_NUM_THREADS='2' if gpu else '1', OPENBLAS_NUM_THREADS='2' if gpu else '1',
        NUMBA_NUM_THREADS='2' if gpu else '1', MPLCONFIGDIR=str(root / 'matplotlib_cache'),
        MPLBACKEND='Agg', PYTHONDONTWRITEBYTECODE='1')
    if gpu:
        env.pop('CUDA_VISIBLE_DEVICES', None)
    else:
        env['CUDA_VISIBLE_DEVICES'] = ''
    started = time.monotonic()
    child_identity = None
    try:
        with Path(row['log']).open('x') as log:
            child = subprocess.Popen(command, cwd=root / 'code', env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            child_identity = process_identity(child.pid)
            if child_identity is None:
                raise RuntimeError('Own child vanished before stage attached its identity')
            row['pid'] = child.pid
            job.update(status='running')
            original.save(root / 'job.json', job)
            profile = monitor(child, gpu=gpu, profile_path=Path(row['profile']))
            rc = child.wait()
        if rc:
            raise RuntimeError(f'{name} returncode={rc}; no retry/repair/restart')
        profile_receipt(profile)
        result = check()
        original.ensure_unchanged(plan, root / 'code', fingerprints, configs)
        row.update(status='complete', returncode=rc)
        job.setdefault('receipts', {})[name] = result
        job.setdefault('resource_profiles', {})[name] = dict(
            path=row['profile'], sha256=original.sha(row['profile']), peak=profile['peak'],
            host_mem_available_min_gib=profile['host_mem_available_min_gib'])
    except BaseException as error:
        row.update(status='failed', error=f'{type(error).__name__}: {error}')
        if 'child' in locals():
            if child.poll() is None and child_identity is not None:
                row['termination'] = terminate_own_session(child, child_identity)
            row['returncode'] = child.poll()
        raise
    finally:
        row.update(ended_at=original.now(), elapsed_seconds=time.monotonic() - started)
        job['GPU_stage_seconds'] += row['elapsed_seconds'] if gpu else 0.
        original.save(root / 'job.json', job)


def execute(plan):
    root = Path(plan['output'])
    if root.exists() or root.is_symlink():
        raise FileExistsError('Refuse duplicate launch, overwrite or resume')
    if socket.gethostname() != original.HOST or os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise RuntimeError('Require host225 and native physical GPU indexing')
    if os.getpriority(os.PRIO_PROCESS, 0) != 10:
        raise RuntimeError('Require explicitly scheduled nice10; do not renice online')
    handoff = verify_handoff(plan['superseded_launch']['root'], root)
    if handoff != plan['superseded_launch']:
        raise RuntimeError('Superseded launch evidence changed after planning')
    project = Path(__file__).resolve().parents[1]
    fingerprints = original.source_hashes(project)
    root.mkdir(parents=True, exist_ok=False)
    for folder in ('logs', 'configs', 'smoke', 'code', 'profiles'):
        (root / folder).mkdir()
    for name, digest in fingerprints.items():
        target = root / 'code' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        original.shutil.copy2(project / name, target)
        if original.sha(target) != digest:
            raise RuntimeError('Code snapshot copy mismatch')
    original.save(root / 'code/fingerprints.json', fingerprints, exclusive=True)
    configs = {}
    for arm in plan['arms']:
        Path(arm['root']).mkdir()
        (root / 'smoke' / arm['name']).mkdir()
        original.save(arm['config_path'], arm['config'], exclusive=True)
        smoke_config = deepcopy(arm['config'])
        smoke_config['training'].update(epochs=2, checkpoint_every=1, log_every=1)
        original.save(arm['smoke_config_path'], smoke_config, exclusive=True)
        for key in ('config_path', 'smoke_config_path'):
            configs[arm[key]] = original.sha(arm[key])
    plan.update(source_code_sha256=fingerprints, configuration_sha256=configs)
    original.save(root / 'plan.json', plan, exclusive=True)
    job = dict(status='running', workflow_complete=False, started_at=original.now(), parent_pid=os.getpid(),
        host=original.HOST, device='cuda:2', gpu_uuid=original.GPU_UUID, error=None, requires_user_action=False,
        actual_nice=os.getpriority(os.PRIO_PROCESS, 0), current=None, stages=[], GPU_stage_seconds=0.,
        physical_sampling_updates=0, labels_used_in_training=False, completed_arms=[], public_files=[],
        superseded_launch=plan['superseded_launch'], execution_version=EXECUTION_VERSION)
    original.save(root / 'job.json', job, exclusive=True)
    started, cpu_start = time.monotonic(), original.cpu_seconds()
    try:
        stage(root, job, 'cuda_verification', plan['cuda_verification'], gpu=True, plan=plan,
            fingerprints=fingerprints, configs=configs, check=lambda: cuda_receipt(root / 'cuda_verification.json'))
        for arm in plan['arms']:
            stage(root, job, 'smoke_' + arm['name'], arm['smoke'], gpu=True, plan=plan,
                fingerprints=fingerprints, configs=configs, check=lambda a=arm: validate_training(plan, a, smoke=True))
        profiles = {f'smoke_{arm["name"]}': original.read(root / 'profiles' / f'smoke_{arm["name"]}.json')
                    for arm in plan['arms']}
        job['measured_resource_gate'] = measured_gate(profiles)
        original.save(root / 'measured_resource_gate.json', job['measured_resource_gate'], exclusive=True)
        job['all_smoke_checks_passed'] = True
        original.save(root / 'job.json', job)
        for arm in plan['arms']:
            stage(root, job, 'train_' + arm['name'], arm['training'], gpu=True, plan=plan,
                fingerprints=fingerprints, configs=configs, check=lambda a=arm: validate_training(plan, a, smoke=False))
            stage(root, job, 'report_' + arm['name'], arm['report'], gpu=False, plan=plan,
                fingerprints=fingerprints, configs=configs,
                check=lambda a=arm: original.report_receipt(Path(a['root']) / 'report', a['theory_dim']))
            job['completed_arms'].append(arm['name'])
            original.save(root / 'job.json', job)
        stage(root, job, 'comparison', plan['compare'], gpu=False, plan=plan, fingerprints=fingerprints,
            configs=configs, check=lambda: original.report_receipt(root / 'comparison'))
        public = ['job.json', 'plan.json', 'measured_resource_gate.json', 'cuda_verification.json',
                  *[str(Path(name).relative_to(root)) for name in configs]]
        public += [str(path.relative_to(root)) for path in sorted((root / 'profiles').glob('*.json'))]
        for arm in plan['arms']:
            public += [f'{arm["name"]}/{name}' for name in
                ('initialization.json', 'run/manifest.json', 'run/summary.json', 'run/history_progress.json')]
            public += [str(path.relative_to(root)) for path in sorted((Path(arm['root']) / 'report').iterdir())
                if path.is_file() and not path.is_symlink() and path.suffix in ('.json', '.png', '.npz')]
        public += [str(path.relative_to(root)) for path in sorted((root / 'comparison').iterdir())
            if path.is_file() and not path.is_symlink() and path.suffix in ('.json', '.png', '.npz')]
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
    parser.add_argument('--source-run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--superseded-root', type=Path)
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--cached-training', action='store_true')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--device', default='cuda:2')
    parser.add_argument('--mi-cache-dir', type=Path)
    parser.add_argument('--initialization-receipt', type=Path)
    parser.add_argument('--benchmark-warmup', type=int, default=0)
    args = parser.parse_args(argv)
    if args.cached_training:
        if (args.device != 'cuda:2' or args.benchmark_warmup != 0
                or any(value is None for value in (args.config, args.mi_cache_dir, args.initialization_receipt))):
            parser.error('Require explicit cache-only low-RAM profiling on cuda:2')
        from scripts.run_ising2d_dimg_ablation import cached_training
        return cached_training(args)
    if args.source_run is None or args.superseded_root is None:
        parser.error('--source-run and --superseded-root required')
    plan = build_plan(args.source_run, args.output, args.superseded_root)
    if args.plan:
        import json
        print(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False))
        return plan
    return execute(plan)


if __name__ == '__main__':
    main()
