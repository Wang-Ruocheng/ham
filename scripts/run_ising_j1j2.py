"""Frozen, serial J1--J2 QCR campaign; no retry, migration, or legacy recipe.

Stage A only: sampler benchmark -> independent pilot quality gate -> 84 PT
groups -> production quality gate -> label-free cache -> data-only MI ->
20-update GPU execution profile -> one fresh dg5 seed91 12000-update run/report.
Dimension/baseline/large-size follow-ups are registered, not silently launched.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time

MODULE = 'scripts.run_ising_j1j2'
HOST = 'thinklab-105-225'
GPU_UUIDS = {
    0: 'GPU-5bba87f1-7d32-4dde-fe28-caba4468ab42',
    1: 'GPU-18fac97c-11ee-4624-d298-9074e8d7028c',
    2: 'GPU-713b07d0-118c-f939-c0a8-565390215e01',
    3: 'GPU-a167decd-ce49-6c8f-6bcf-136fde596683',
}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if exclusive:
        with path.open('x') as stream:
            stream.write(value)
    else:
        temporary = path.with_name(path.name + '.writing')
        temporary.write_text(value)
        temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes(code):
    return {str(path.relative_to(code)): sha(path)
            for folder in ('recurrent_rg', 'scripts', 'experiments')
            for path in sorted((code / folder).rglob('*.py'))
            if not path.name.startswith('._')}


def cpu_seconds():
    return sum(row.ru_utime + row.ru_stime for row in (
        resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)))


def directory_bytes(root):
    return sum(path.stat().st_size for path in Path(root).rglob('*')
               if path.is_file() and not path.is_symlink())


def coupling_grid():
    """Twelve retained uniform-K conditions plus sampling-only PT rungs.

    The sampler requires increasing temperatures, so the retained slots have
    decreasing K=J1/T.  Preserve the old auxiliary temperature ladder, but do
    not mistake its integer-index subsampling for a uniform coupling grid.
    No auxiliary-only rung is retained in the training/evaluation cache.
    """
    import numpy as np
    retained_k = np.linspace(.25, 2.5, 12)[::-1]
    retained_temperatures = 1. / retained_k
    temperatures = np.unique(np.concatenate((
        np.geomspace(.4, 4., 73), retained_temperatures)))
    saved_indices = np.searchsorted(temperatures, retained_temperatures)
    if not np.array_equal(temperatures[saved_indices], retained_temperatures):
        raise RuntimeError('Retained uniform-K temperatures are absent from PT ladder')
    retained_indices = set(saved_indices.tolist())
    return dict(
        temperatures=temperatures.tolist(),
        saved_temperature_indices=saved_indices.tolist(),
        retained_K_values=retained_k.tolist(),
        retained_temperature_values=retained_temperatures.tolist(),
        auxiliary_temperature_indices=[index for index in range(len(temperatures))
                                       if index not in retained_indices],
        training_coupling_grid='uniform_K',
    )


def prepare(root, *, cpu_hours=48., gpu_hours=8., directory_gib=20.):
    from experiments.prepare_ising_j1j2 import campaign_configs, make_config, protocol_contract
    from scripts.ising_qcr_evaluation import encoding_contract
    for value in (cpu_hours, gpu_hours, directory_gib):
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError('Positive finite resource budgets required')
    if socket.gethostname() != HOST or 'CUDA_VISIBLE_DEVICES' in os.environ:
        raise ValueError('Requires declared host and unmasked physical GPU ordinals')
    root, code = Path(root).resolve(), Path(__file__).resolve().parents[1]
    if root.exists() or root.is_symlink():
        raise FileExistsError('Never overwrite, resume, or recreate a campaign')
    cache = root / 'data/training.pt'
    configs = campaign_configs(cache, seeds=(91,))
    profile = make_config(cache, seed=191, updates=20)
    plan = dict(schema_version=5, experiment='ising2d_frustrated_J1J2_stageA',
        host=HOST, output=str(root), code_root=str(code), prepared_at=now(),
        source_sha256=source_hashes(code), configs=configs, profile_config=profile,
        physical=dict(size=108, rhos=[0., .2, .35, .7, 1., .3, .8],
            train_rhos=[0., .2, .35, .7, 1.], heldout_rhos=[.3, .8],
            **coupling_grid(),
            groups=12, train_groups=8, samples=128, burn_in=16384, spacing=128,
            master_seed=20261026, initials=['hot', 'ferro', 'stripe_x', 'stripe_y'],
            convention='H=-sum_NN ss + rho sum_diagonal ss; J1=kB=1',
            symmetric_saved_data='Independent exact global Z2 and quarter-turn rotation per saved state; raw diagnostics retained before symmetry; no extra independent samples'),
        pilot=dict(rhos=[0., .35, .7, 1.], burn_in=4096, spacing=32, samples=128,
            min_exchange=.01, min_ess=16, hot_cold_abs_threshold=.05,
            hot_cold_z_threshold=4., production_reuse=False,
            constant_series='warning; not evidence of mixing, not alone proof of failure'),
        protocol=dict(**protocol_contract(), scales=[108, 36, 12],
            encoding_protocol=encoding_contract(128),
            training_coupling_grid='uniform_K',
            retained_K_definition='12 equally spaced K=J1/T in [0.25,2.5] per rho; ordered by increasing T',
            auxiliary_temperatures_are_sampling_only=True,
            phase_claims_require_independent_physics=True, root_count_not_prespecified=True,
            execution_cpu_threads=dict(sampling=1, MI_preprocessing=2, training=2, report=1),
            sampler='single-thread lazy local MH four-color visits + PT; no NN-only Wolff'),
        success=dict(pre_registered=True, latent_nrmse_max=.1,
            resolved_drift_error_reduction_vs_identity_min=.5,
            caveat='Latent criteria alone insufficient; cross-method comparison requires common physical targets and independent grouped uncertainty',
            thermal_nu=None, no_first_order_verdict=True),
        future_stages=dict(automatic_execution=False, dimensions=[1, 2],
            confirmation_seeds=[94, 95], challenge_rhos=[.45, .55, .6],
            larger_sizes=[216, 324], baselines=['PCA_R', 'fixed_C_QR', 'Neural_MCRG_adapter']),
        resource_policy=dict(cpu_seconds=cpu_hours * 3600, gpu_seconds=gpu_hours * 3600,
            directory_bytes=int(directory_gib * 1024**3), own_gpu_peak_bytes=12 * 1024**3,
            minimum_host_available_gib=36, load_per_cpu_max=1.25,
            io_full_avg10_percent_max=20, gpu_free_mib=16384,
            two_gpu_checks_seconds_apart=10, cpu_threads=1, training_cpu_threads=2,
            nice=10, safety_factor=1.5, max_training_seconds=10800,
            gpu_write_reserve_seconds=120, report_seconds_per_arm=1800,
            gpu_binding='User-authorized sharing of GPU3 only, free>=16GiB and utilization<=10%, two checks >=10s apart, then immutable; no migration',
            gpu_sharing_authorized=True, gpu_utilization_percent_max=10,
            sharing_authorization='User: one seed only; low-utilization GPU may be shared',
            eligible_gpu_uuids={'3': GPU_UUIDS[3]},
            resource_wait='Wait; never interfere with another process', automatic_retry=False))
    plan['resource_policy']['budget_check_granularity'] = (
        'Cumulative CPU and directory limits are checked at stage boundaries; '
        'no in-flight child is killed. Sampling has a measured conservative '
        'whole-campaign gate, training has an execution profile and per-arm clock. '
        'A stage can cross a budget before the next boundary; no automatic continuation.')
    plan['protocol']['training_seeds'] = [91]
    plan['protocol']['single_seed_caution'] = 'No training-seed robustness claim; seeds92-95 are not launched'
    root.mkdir(parents=True, exist_ok=False)
    for folder in ('logs', 'configs', 'data'):
        (root / folder).mkdir()
    for arm, config in configs.items():
        save(root / f'configs/{arm}.json', config, exclusive=True)
    save(root / 'configs/profile.json', profile, exclusive=True)
    save(root / 'plan.json', plan, exclusive=True)
    save(root / 'job.json', dict(status='prepared', workflow_complete=False,
        current=None, stages=[], error=None, requires_user_action=False,
        plan_sha256=sha(root / 'plan.json'), gpu_elapsed_seconds=0., training={}), exclusive=True)
    return plan


def verify(root):
    root = Path(root).resolve()
    plan, job = read(root / 'plan.json'), read(root / 'job.json')
    if (socket.gethostname() != plan['host'] or plan['host'] != HOST
            or plan['output'] != str(root) or 'CUDA_VISIBLE_DEVICES' in os.environ
            or job['plan_sha256'] != sha(root / 'plan.json')):
        raise RuntimeError('Host/output/plan/CUDA-mask mismatch; no automatic repair')
    if source_hashes(Path(plan['code_root'])) != plan['source_sha256']:
        raise RuntimeError('Executable source changed after freezing')
    for arm, expected in [*plan['configs'].items(), ('profile', plan['profile_config'])]:
        if read(root / f'configs/{arm}.json') != expected:
            raise RuntimeError('Frozen scientific configuration changed')
    return plan, job


def resource_snapshot(plan):
    policy = plan['resource_policy']
    info = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = int(info['MemAvailable'].split()[0]) * 1024
    pressure = Path('/proc/pressure/io').read_text().splitlines()
    full = next(line for line in pressure if line.startswith('full '))
    io_full = float(next(word.split('=')[1] for word in full.split() if word.startswith('avg10=')))
    cpus, load = os.cpu_count(), os.getloadavg()[0]
    host_ready = (available >= policy['minimum_host_available_gib'] * 1024**3
        and load <= policy['load_per_cpu_max'] * cpus
        and io_full < policy['io_full_avg10_percent_max'])
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True).splitlines()
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'], text=True).splitlines()
    devices = []
    for line in rows:
        index, uuid, free, utilization = [part.strip() for part in line.split(',')]
        pids = [int(line.split(',')[1]) for line in processes if uuid in line]
        devices.append(dict(index=int(index), uuid=uuid, free_mib=int(free), pids=pids,
            utilization_percent=int(utilization),
            ready=host_ready and policy['eligible_gpu_uuids'].get(index) == uuid
                and int(free) >= policy['gpu_free_mib']
                and int(utilization) <= policy['gpu_utilization_percent_max']
                and (policy['gpu_sharing_authorized'] or not pids)))
    return dict(at=now(), mem_available_bytes=available, cpus=cpus, load1=load,
                io_full_avg10_percent=io_full, host_ready=host_ready, gpus=devices)


def admission(root, plan, job, name, *, gpu=False):
    candidate, confirmations = None, 0
    while True:
        row = resource_snapshot(plan)
        job.update(status='waiting_resources', current=name)
        job.setdefault('resource_checks', []).append(row)
        job['resource_checks'] = job['resource_checks'][-30:]
        save(root / 'job.json', job)
        if not gpu and row['host_ready']:
            return
        if gpu:
            binding = read(root / 'device_binding.json') if (root / 'device_binding.json').is_file() else None
            eligible = [device for device in row['gpus'] if device['ready']
                        and (binding is None or device['uuid'] == binding['gpu_uuid'])]
            chosen = eligible[0] if eligible else None
            if chosen is not None:
                if chosen['uuid'] != candidate:
                    candidate, confirmations = chosen['uuid'], 1
                else:
                    confirmations += 1
                if confirmations >= 2:
                    if binding is None:
                        save(root / 'device_binding.json', dict(at=now(), host=plan['host'],
                            device=f"cuda:{chosen['index']}", gpu_uuid=chosen['uuid'],
                            plan_sha256=sha(root / 'plan.json')), exclusive=True)
                    return
            else:
                candidate, confirmations = None, 0
        time.sleep(10 if confirmations else 30)


def profile_projection(summary, elapsed, plan):
    timing, policy = summary['timing'], plan['resource_policy']
    values = timing['steady_step_seconds']
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError('Finite positive measured steady steps required')
    per_arm = policy['safety_factor'] * (max(values) * 12000
        + timing['setup_seconds'] + timing['logging_seconds'] / 2 * 61
        + timing['checkpoint_seconds'] * 14) + policy['gpu_write_reserve_seconds']
    total = elapsed + len(plan['configs']) * per_arm
    return dict(status='complete', profile_is_scientific_result=False,
        projected_gpu_seconds_per_arm=per_arm, projected_total_gpu_seconds=total,
        budget_passed=total <= policy['gpu_seconds'] and per_arm <= policy['max_training_seconds'],
        own_reserved_peak_bytes=timing['cuda_max_memory_reserved_bytes'],
        memory_passed=timing['cuda_max_memory_reserved_bytes'] <= policy['own_gpu_peak_bytes'])


def train_child(root, arm):
    import torch
    from experiments.phi4_qcr import _device_receipt
    from experiments.prepare_ising_j1j2 import training_command
    from experiments import rsmi_separate_qcr as trainer
    from recurrent_rg import mi_cache
    from scripts.report_ising_zh import chinese_figures
    root = Path(root).resolve()
    plan, job = verify(root)
    if arm != 'profile' and arm not in plan['configs']:
        raise ValueError('Unregistered arm')
    run = root / ('profile_run' if arm == 'profile' else f'{arm}/run')
    if run.exists() or run.is_symlink():
        raise FileExistsError('No resume or retry')
    binding = read(root / 'device_binding.json')
    if binding['plan_sha256'] != sha(root / 'plan.json'):
        raise RuntimeError('Device binding does not belong to plan')
    torch.cuda.set_device(torch.device(binding['device']))
    save(run.with_name(run.name + '.device.json'),
         _device_receipt(torch.device(binding['device']), binding['gpu_uuid']), exclusive=True)
    receipt = read(root / 'data/mi_preprocessing.json')
    keys = {split: receipt['splits'][split]['cache']['key'] for split in ('train', 'test')}
    original = mi_cache.load_or_build_mi_statistics
    def cache_only(directory, key_data, build, **kwargs):
        def forbidden():
            raise RuntimeError('Frozen MI cache miss; no automatic rebuilding')
        value, record = original(directory, key_data, forbidden, **kwargs)
        if not record['hit'] or record['key'] != keys[key_data['split']]:
            raise RuntimeError('MI cache identity mismatch')
        return value, record
    remaining = plan['resource_policy']['gpu_seconds'] - job['gpu_elapsed_seconds']
    setup_reserve = 120.
    if (root / 'profile_run/summary.json').is_file():
        setup_reserve += 2 * read(root / 'profile_run/summary.json')['timing']['setup_seconds']
    clock = min(plan['resource_policy']['max_training_seconds'], remaining - setup_reserve)
    if clock <= 0:
        raise RuntimeError('Insufficient remaining GPU execution budget')
    command = training_command(sys.executable, root / f'configs/{arm}.json', run,
        device=binding['device'], mi_cache_dir=root / 'data/mi_cache', benchmark_warmup=5)
    argv = command[command.index('experiments.rsmi_separate_qcr') + 1:]
    argv += ['--max-training-seconds', str(clock)]
    mi_cache.load_or_build_mi_statistics = cache_only
    try:
        with chinese_figures():
            return trainer.main(argv)
    finally:
        mi_cache.load_or_build_mi_statistics = original


def validate_training(root, plan, arm):
    folder = root / ('profile_run' if arm == 'profile' else f'{arm}/run')
    summary, manifest = read(folder / 'summary.json'), read(folder / 'manifest.json')
    updates = 20 if arm == 'profile' else 12000
    expected = plan['profile_config'] if arm == 'profile' else plan['configs'][arm]
    if (summary['requested_epochs'] != updates or summary['config'] != expected or manifest['config'] != expected
            or manifest.get('labels_used_in_training') is not False
            or manifest.get('independent_c') is not True or manifest.get('source') is not None
            or manifest.get('resumed_from') is not None or manifest.get('q_supervision')
            or manifest.get('data_driven_initialization') or manifest.get('frozen_C')):
        raise RuntimeError('Fresh default endpoint/label boundary mismatch')
    if any(summary.get(name) for name in ('stopped_by_training_limit', 'stopped_by_wall_limit',
                                         'stopped_by_train_plateau')):
        return dict(status='budget_hold', reason='Fixed safety clock reached; no continuation')
    if summary['actual_epochs'] != updates or not (folder / f'training_step_{updates}.pt').is_file():
        raise RuntimeError('Missing exact requested endpoint without a declared safety stop')
    if manifest['data_sha256'] != read(root / 'data/metadata.json')['data_sha256']:
        raise RuntimeError('Training data identity differs from the assembled label-free cache')
    if summary['timing']['cuda_max_memory_reserved_bytes'] > plan['resource_policy']['own_gpu_peak_bytes']:
        return dict(status='budget_hold', reason='Own GPU peak exceeded policy')
    return dict(status='complete', actual_updates=updates, timing=summary['timing'],
                manifest_sha256=sha(folder / 'manifest.json'))


def stage_list(root, plan):
    def data_command(mode, *extra):
        return [sys.executable, '-u', '-m', MODULE, '--data-child', mode,
                '--output', str(root), *extra]
    stages = [('sampler_benchmark', data_command('benchmark'), False),
              ('independent_pilot', data_command('pilot'), False)]
    for rho_index in range(len(plan['physical']['rhos'])):
        for group in range(plan['physical']['groups']):
            stages.append((f'sample_r{rho_index:02d}_g{group:02d}',
                data_command('sample', '--rho-index', str(rho_index), '--group', str(group)), False))
    stages += [('production_quality_gate', data_command('quality'), False),
               ('label_free_cache', data_command('assemble'), False),
               ('MI_preprocessing', [sys.executable, '-u', '-m', 'scripts.prepare_qcr_mi_cache',
                    '--config', str(root / 'configs/dg5_seed91.json'),
                    '--output', str(root / 'data/mi_preprocessing.json'),
                    '--cache-dir', str(root / 'data/mi_cache')], False)]
    stages.append(('training_profile', [sys.executable, '-u', '-m', MODULE,
        '--train-child', 'profile', '--output', str(root)], True))
    for arm in plan['configs']:
        stages.append((f'{arm}_training', [sys.executable, '-u', '-m', MODULE,
            '--train-child', arm, '--output', str(root)], True))
        stages.append((f'{arm}_report', [sys.executable, '-u', '-m', 'scripts.report_ising_j1j2',
            '--run', str(root / f'{arm}/run'), '--data-metadata', str(root / 'data/metadata.json'),
            '--output', str(root / f'{arm}/report'), '--max-seconds',
            str(plan['resource_policy']['report_seconds_per_arm'])], False))
    return stages


def public_files(root):
    allowed = []
    for path in Path(root).rglob('*'):
        if not path.is_file() or path.is_symlink() or path.name == 'job.json' or 'mi_cache' in path.parts:
            continue
        if path.suffix in ('.json', '.png') or (path.suffix == '.npz' and 'report' in path.parts):
            allowed.append(path)
    return [dict(path=str(path.relative_to(root)), sha256=sha(path), bytes=path.stat().st_size)
            for path in sorted(allowed)]


def hold(job, reason, *, scientific=False):
    job.update(status='scientific_stop' if scientific else 'budget_hold',
               workflow_complete=scientific, requires_user_action=True,
               error=reason, stopped_at=now(),
               training_started=any(key != 'profile' for key in job['training']))


def validate_report(folder):
    folder = Path(folder)
    report, status = read(folder / 'report.json'), read(folder / 'status.json')
    if (status.get('status') != 'complete' or status.get('report_complete') is not True
            or report.get('status') != 'complete' or report.get('report_complete') is not True
            or report.get('checkpoint_step') != 12000 or report.get('thermal_nu') is not None):
        raise RuntimeError('Formal report did not complete its declared fixed-endpoint protocol')
    required = {'report.json', 'evaluation_arrays.npz', 'loss_history.json',
        'pca_explained.png', 'R_displacement.png', 'coupling_to_g.png',
        'loss_curves.png', 'physical_C_statistics.png'}
    identities = status.get('sha256', {})
    if (not required.issubset(identities) or set(status.get('public_files', [])) != set(identities)):
        raise RuntimeError('Report public file manifest is incomplete')
    for name, digest in identities.items():
        path = Path(name)
        if (path.is_absolute() or len(path.parts) != 1 or path.suffix not in ('.json', '.png', '.npz')
                or (folder / name).is_symlink()):
            raise RuntimeError('Unsafe report public-file identity')
        if sha(folder / name) != digest:
            raise RuntimeError('Report public file digest mismatch')
    return dict(status='complete', public_files_verified=len(status['sha256']))


def execute(root):
    root = Path(root).resolve()
    plan, job = verify(root)
    if job['status'] != 'prepared' or job['stages']:
        raise RuntimeError('Only a fresh prepared workflow can execute; never resume or duplicate')
    save(root / 'execution.json', dict(at=now(), parent_pid=os.getpid(),
        plan_sha256=sha(root / 'plan.json'), host=socket.gethostname(),
        exclusive_execution_lock=True, retry=False), exclusive=True)
    os.nice(plan['resource_policy']['nice'])
    job.update(parent_pid=os.getpid(), actual_nice=os.getpriority(os.PRIO_PROCESS, 0),
               started_at=now(), gpu_elapsed_seconds=0., cpu_seconds=0.)
    stages = stage_list(root, plan)
    try:
        for index, (name, command, gpu) in enumerate(stages):
            verify(root)
            policy = plan['resource_policy']
            if (cpu_seconds() >= policy['cpu_seconds'] or directory_bytes(root) >= policy['directory_bytes']
                    or job['gpu_elapsed_seconds'] >= policy['gpu_seconds']):
                hold(job, 'Fixed resource envelope reached; preserved partial output, no automatic extension')
                break
            if (root / 'profile_receipt.json').is_file():
                projection = read(root / 'profile_receipt.json')
                pending = sum(stage[2] for stage in stages[index:])
                if job['gpu_elapsed_seconds'] + pending * projection['projected_gpu_seconds_per_arm'] > policy['gpu_seconds']:
                    hold(job, 'Actual GPU usage plus remaining conservative projection exceeds budget')
                    break
            admission(root, plan, job, name, gpu=gpu)
            start, cpu_before = time.perf_counter(), cpu_seconds()
            with (root / f'logs/{name}.log').open('x') as stream:
                child = subprocess.Popen(command, cwd=plan['code_root'], stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT,
                    env=dict(os.environ, OMP_NUM_THREADS=str(2 if gpu else 1),
                        MKL_NUM_THREADS=str(2 if gpu else 1), OPENBLAS_NUM_THREADS='1',
                        NUMBA_NUM_THREADS='1', PYTHONUNBUFFERED='1', CUDA_DEVICE_ORDER='PCI_BUS_ID'))
                row = dict(name=name, command=command, pid=child.pid, started_at=now(),
                           status='running', returncode=None, gpu_stage=gpu)
                job['stages'].append(row)
                job.update(status='running', current=name)
                save(root / 'job.json', job)
                returncode = child.wait()
            elapsed = time.perf_counter() - start
            row.update(returncode=returncode, status='complete' if returncode == 0 else 'failed',
                       ended_at=now(), wall_seconds=elapsed, cpu_seconds=cpu_seconds()-cpu_before)
            job['cpu_seconds'] = cpu_seconds()
            if gpu:
                job['gpu_elapsed_seconds'] += elapsed
            save(root / 'job.json', job)
            if returncode and name.endswith('_report'):
                report_status = root / name.removesuffix('_report') / 'report/status.json'
                if report_status.is_file() and read(report_status).get('status') == 'budget_hold':
                    row['status'] = 'analysis_budget_hold'
                    hold(job, 'Bounded report reached its analysis budget; model preserved without retry')
                    break
            if returncode:
                raise RuntimeError(f'{name} returned {returncode}; no retry or automatic repair')
            if name == 'sampler_benchmark':
                receipt = read(root / 'data/benchmark/receipt.json')
                production = receipt['conservative_production_cpu_seconds']
                p, pilot = plan['physical'], plan['pilot']
                pilot_ladders = sum(2 if rho < .5 else 3 for rho in pilot['rhos'])
                pilot_ratio = (pilot_ladders * (pilot['burn_in'] + pilot['samples'] * pilot['spacing'])
                    / (len(p['rhos']) * p['groups'] * (p['burn_in'] + p['samples'] * p['spacing'])))
                projected_cpu = (cpu_seconds() + production * (1 + pilot_ratio)
                    + policy['gpu_seconds'] * policy['training_cpu_threads']
                    + len(plan['configs']) * policy['report_seconds_per_arm'])
                decision = dict(status='complete', production_cpu_seconds_conservative=production,
                    independent_pilot_cpu_seconds_conservative=production * pilot_ratio,
                    projected_total_cpu_seconds=projected_cpu, fixed_cpu_seconds=policy['cpu_seconds'],
                    budget_passed=projected_cpu <= policy['cpu_seconds'], profile_is_scientific_result=False)
                save(root / 'sampling_budget_receipt.json', decision, exclusive=True)
                if not decision['budget_passed']:
                    hold(job, 'Measured conservative sampling projection exceeds fixed CPU budget')
                    break
            elif name in ('independent_pilot', 'production_quality_gate'):
                receipt = read(root / ('data/pilot/receipt.json' if name == 'independent_pilot'
                                       else 'data/quality_gate.json'))
                if receipt['status'] == 'scientific_stop':
                    hold(job, 'Independent finite-chain sampling quality gate not passed; no automatic resampling', scientific=True)
                    break
            elif gpu:
                arm = 'profile' if name == 'training_profile' else name.removesuffix('_training')
                result = validate_training(root, plan, arm)
                job['training'][arm] = result
                if result['status'] != 'complete':
                    hold(job, result['reason'])
                    break
                if arm == 'profile':
                    projection = profile_projection(read(root / 'profile_run/summary.json'),
                                                    job['gpu_elapsed_seconds'], plan)
                    save(root / 'profile_receipt.json', projection, exclusive=True)
                    if not projection['budget_passed'] or not projection['memory_passed']:
                        hold(job, 'GPU execution profile failed resource projection/own-memory gate; formal training not started')
                        break
            elif name.endswith('_report'):
                validate_report(root / name.removesuffix('_report') / 'report')
        else:
            job.update(status='complete', workflow_complete=True, current=None, ended_at=now(),
                thermal_nu=None, future_stages_not_launched=True)
        job['public_files'] = public_files(root)
    except BaseException as error:
        job.update(status='failed', workflow_complete=False, requires_user_action=True,
                   error=f'{type(error).__name__}: {error}', ended_at=now())
        raise
    finally:
        save(root / 'job.json', job)


def launch(root):
    root = Path(root).resolve()
    plan, job = verify(root)
    if job['status'] != 'prepared' or job['stages']:
        raise RuntimeError('No duplicate launch')
    with (root / 'launch.json').open('x') as receipt, (root / 'logs/parent.log').open('x') as stream:
        process = subprocess.Popen([sys.executable, '-u', '-m', MODULE, '--execute', '--output', str(root)],
            cwd=plan['code_root'], stdin=subprocess.DEVNULL, stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True,
            env=dict(os.environ, PYTHONUNBUFFERED='1', CUDA_DEVICE_ORDER='PCI_BUS_ID'))
        result = dict(status='launched', at=now(), parent_pid=process.pid,
            host=plan['host'], root=str(root), plan_sha256=sha(root / 'plan.json'),
            gpu_binding=None, resumed=False, automatic_retry=False)
        receipt.write(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare', 'launch', 'execute'):
        mode.add_argument('--' + name, action='store_true')
    mode.add_argument('--data-child', choices=('benchmark', 'pilot', 'sample', 'quality', 'assemble'))
    mode.add_argument('--train-child')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu-hours', type=float, default=48.)
    parser.add_argument('--gpu-hours', type=float, default=8.)
    parser.add_argument('--directory-gib', type=float, default=20.)
    parser.add_argument('--rho-index', type=int)
    parser.add_argument('--group', type=int)
    args = parser.parse_args(argv)
    if args.prepare:
        result = prepare(args.output, cpu_hours=args.cpu_hours, gpu_hours=args.gpu_hours,
                         directory_gib=args.directory_gib)
    elif args.launch:
        result = launch(args.output)
    elif args.execute:
        return execute(args.output)
    elif args.train_child is not None:
        return train_child(args.output, args.train_child)
    else:
        from scripts import ising_j1j2_data as data
        plan, _ = verify(args.output)
        if args.data_child == 'sample':
            if args.rho_index is None or args.group is None:
                parser.error('sample requires --rho-index and --group')
            result = data.sample_group(plan, args.output, args.rho_index, args.group)
        else:
            function = dict(benchmark=data.benchmark, pilot=data.pilot,
                            quality=data.quality_gate, assemble=data.assemble)[args.data_child]
            result = function(plan, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    return result


if __name__ == '__main__':
    main()
