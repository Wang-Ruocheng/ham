"""Bounded, frozen-model exploratory L1080 probe; no training or old-data writes."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import socket
import subprocess
import sys
import time


EXPERIMENT = 'ising_b3_frozen_L1080_small_probe_v1'
HOST = 'thinklab-105-225'
GPU_UUID = 'GPU-713b07d0-118c-f939-c0a8-565390215e01'
BASE = Path('/mnt/nas-new/home/yange/ham')
SOURCE_ROOT = BASE / 'results/ising2d_positiveK_QCR_closure_scale_early_b3_seed91_vm225_20261018_v1'
CHECKPOINT = SOURCE_ROOT / 'run_early/checkpoint_last.pt'
CHECKPOINT_SHA = '0687c8f4663810688833c54de1e2e6d2424fa9ba5a2c2dbc43042549d4303f4d'
BASELINE_ROOT = BASE / 'results/ising2d_positiveK_QCR_finite_perturbation_R5_seed20261017_vm225'
FROZEN_CODE = BASE / 'tmp/ising_closure_scale_early_b3_seed91_20261018_code_v1'
SOURCE_RECEIPT = Path(str(SOURCE_ROOT) + '_posthoc_evaluation_v1') / 'evaluation_receipt.json'
CPU_LIMIT = 4 * 3600
GPU_LIMIT = 900
GPU_PEAK_MIB = 8192
OUTPUT_LIMIT = 2 * 1024**3
# Conservatively include the preceding import/shape-only checks in this task.
PRIOR_CPU_RESERVE = 120.
NEW_FILES = ('scripts/run_ising_large_size_probe.py', 'scripts/ising_large_size_probe_sampling.py',
             'scripts/ising_large_size_probe_evaluation.py', 'scripts/report_ising_large_size_probe.py',
             'scripts/ising_qcr_evaluation.py')


class BudgetHold(RuntimeError):
    pass


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    pending = path.with_name(path.name + '.pending')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    pending.replace(path)


def ordinary(path, root):
    path, root = Path(path), Path(root)
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Expected regular in-scope file: {path}')
    return path


def validate_sources():
    """Hash the tiny checkpoint and three explicit new-data cells, never MI."""
    receipt = read(SOURCE_RECEIPT)
    if (receipt.get('status') != 'complete' or receipt.get('actual_steps') != 6000
            or receipt.get('checkpoint') != str(CHECKPOINT)
            or receipt.get('checkpoint_sha256') != CHECKPOINT_SHA
            or receipt.get('training_updates') != 0):
        raise ValueError('Completed actual6000 frozen-source evaluation receipt mismatch')
    if sha(ordinary(CHECKPOINT, SOURCE_ROOT)) != CHECKPOINT_SHA:
        raise ValueError('Frozen source checkpoint fingerprint mismatch')
    pin = receipt.get('frozen_source_sha256', {})
    if not pin:
        raise ValueError('Missing frozen training-code provenance')
    for relative, digest in pin.items():
        if sha(ordinary(FROZEN_CODE / relative, FROZEN_CODE)) != digest:
            raise ValueError(f'Frozen training code changed: {relative}')
    source_job = read(BASELINE_ROOT / 'job.json')
    if (source_job.get('status') not in ('complete', 'scientific_stop')
            or source_job.get('workflow_complete') is not True):
        raise ValueError('The explicit finite-perturbation baseline must be completed')
    files = {}
    for index, coupling in ((0, .42), (14, .44), (7, .46)):
        directory = BASELINE_ROOT / 'data/conditions' / f'c{index:03d}'
        status, sidecar = read(directory / 'status.json'), read(directory / 'sidecar.json')
        point = sidecar['condition']
        if (status.get('status') != 'complete' or status.get('workflow_complete') is not True
                or point.get('size') != 108 or point.get('K0') != coupling
                or sidecar.get('delta6') != [0.] * 6 or sidecar.get('smoke') is not False):
            raise ValueError('Explicit L108 reference data contract mismatch')
        for name in ('status.json', 'sidecar.json', 'data.pt'):
            path = ordinary(directory / name, BASELINE_ROOT)
            files[str(path)] = sha(path)
        if files[str(directory / 'data.pt')] != sidecar.get('data_sha256'):
            raise ValueError('Explicit new L108 data fingerprint mismatch')
    return dict(checkpoint=str(CHECKPOINT), checkpoint_sha256=CHECKPOINT_SHA,
        reference_receipt=str(SOURCE_RECEIPT), reference_receipt_sha256=sha(SOURCE_RECEIPT),
        module_sha256=receipt['module_sha256_before'], frozen_code=str(FROZEN_CODE),
        frozen_code_sha256=pin, baseline_files_sha256=files,
        historical_formal_failure_preserved=True, old_9GB_opened=False, MI_opened=False)


def assert_frozen(root, plan):
    for relative, digest in plan['new_code_sha256'].items():
        if sha(ordinary(root / 'code' / relative, root / 'code')) != digest:
            raise RuntimeError('New probe code changed after launch')
    source = plan['source']
    if sha(CHECKPOINT) != source['checkpoint_sha256'] or sha(SOURCE_RECEIPT) != source['reference_receipt_sha256']:
        raise RuntimeError('Frozen model/receipt changed; no next stage started')
    for relative, digest in source['frozen_code_sha256'].items():
        if sha(ordinary(FROZEN_CODE / relative, FROZEN_CODE)) != digest:
            raise RuntimeError('Frozen source implementation changed; no next stage started')
    for path, digest in source['baseline_files_sha256'].items():
        if sha(ordinary(path, BASELINE_ROOT)) != digest:
            raise RuntimeError('L108 reference changed; no next stage started')
    for name, digest in source.get('source_json_sha256', {}).items():
        if sha(ordinary(root / 'source_record' / name, root / 'source_record')) != digest:
            raise RuntimeError('Historical source metadata copy changed; no next stage started')


def snapshot():
    from scripts.run_ising_future_physics import snapshot as read_snapshot
    return read_snapshot()


def host_ready(row):
    return (row['mem_available_gib'] >= 36 and row['load1'] <= 1.25 * row['cpu_count']
            and row['io_full_avg10'] < 20)


def gpu_ready(row):
    return (host_ready(row) and row['gpu_uuid'] == GPU_UUID
            and row['free_mib'] >= 12 * 1024 and not row['processes'])


def wait_resources(root, job, *, gpu):
    checks = []
    while True:
        began = time.monotonic()
        row = snapshot()
        row['query_seconds'] = time.monotonic() - began
        row['monotonic'] = time.monotonic()
        if row['gpu_uuid'] != GPU_UUID:
            raise RuntimeError('Pinned physical GPU UUID changed')
        checks.append(row)
        checks = checks[-2:]
        job.update(status='waiting_resources', resource_checks=checks)
        save(root / 'job.json', job)
        if (not gpu and host_ready(row)) or (gpu and len(checks) == 2 and all(gpu_ready(x) for x in checks)
                and checks[1]['monotonic'] - checks[0]['monotonic'] >= 10):
            if gpu:
                # Charge even the successful idle-check interval conservatively.
                job['gpu_seconds_completed'] += (checks[1]['monotonic'] - checks[0]['monotonic']
                    + checks[0]['query_seconds'] + checks[1]['query_seconds'])
            return
        time.sleep(10.1 if gpu else 30.)


def cpu_total():
    own, children = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return PRIOR_CPU_RESERVE + own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime


def child_cpu(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        # Fields after ')' start at original field3; utime/stime are14/15.
        return (int(fields[11]) + int(fields[12])) / os.sysconf('SC_CLK_TCK')
    except FileNotFoundError:
        return 0.


def output_bytes(root):
    return sum(path.stat().st_size for path in root.rglob('*') if path.is_file() and not path.is_symlink())


def budget_ok(cpu, gpu, size):
    return cpu <= CPU_LIMIT and gpu <= GPU_LIMIT and size <= OUTPUT_LIMIT


def cpu_projection_gate(benchmark, first_remaining, *, actual_cpu):
    """Include all observed parent/GPU-check CPU, not sampling receipts alone."""
    remaining = sum(row['projected_cpu_seconds'] + row['process_setup_cpu_seconds']
        for row in benchmark['budget_gate']['conditions'] if row['condition_index'] >= first_remaining)
    # A single-threaded GPU stage can conservatively consume its whole900s
    # CPU allowance, plus report/validation overhead. This is not an ETA.
    reserve = GPU_LIMIT + 180.
    projection = dict(actual_cpu_seconds=actual_cpu, remaining_sampling_cpu_seconds=remaining,
        evaluation_and_report_cpu_reserve_seconds=reserve,
        projected_total_cpu_seconds=actual_cpu + remaining + reserve,
        allowed_cpu_seconds=CPU_LIMIT)
    projection['passed'] = projection['projected_total_cpu_seconds'] <= CPU_LIMIT
    return projection


def command(root, module, *args):
    return [sys.executable, '-m', module, '--root', str(root), *map(str, args)]


def stage_environment(root):
    environment = dict(os.environ)
    environment['PYTHONPATH'] = os.pathsep.join((str(root / 'code'), str(FROZEN_CODE)))
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    return environment


def stage_budget_hold(root, name):
    """Use explicit stage receipts, never infer a hold from an arbitrary failure."""
    if name == 'sampling_preflight':
        path = root / 'benchmark/status.json'
    elif name.startswith('sample_c'):
        path = root / 'data' / name.removeprefix('sample_') / 'status.json'
    elif name == 'evaluation_preflight':
        path = root / 'gpu_preflight.json'
    elif name == 'frozen_collection':
        path = root / 'collection/status.json'
    else:
        return None
    if path.is_file():
        value = read(path)
        if value.get('status') == 'budget_hold':
            return value
    return None


def run_stage(root, job, plan, name, cmd, *, gpu=False):
    assert_frozen(root, plan)
    wait_resources(root, job, gpu=gpu)
    remaining = int(math.floor(CPU_LIMIT - cpu_total()))
    if remaining < 120 or job['gpu_seconds_completed'] >= GPU_LIMIT:
        raise BudgetHold('Insufficient fixed resource budget before next stage')
    cmd = list(cmd)
    if gpu and '--prior-gpu-seconds' in cmd:
        cmd[cmd.index('--prior-gpu-seconds') + 1] = str(job['gpu_seconds_completed'])
    row = dict(name=name, command=cmd, status='running', started_at=now(), returncode=None,
               gpu_stage=gpu, log=str(root / 'logs' / (name + '.log')))
    job['stages'].append(row)
    began, process, failure = time.monotonic(), None, None
    def child_limits():
        resource.setrlimit(resource.RLIMIT_CPU, (remaining, remaining + 1))
    try:
        with (root / 'logs' / (name + '.log')).open('x') as log:
            process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, preexec_fn=child_limits,
                cwd=root / 'code', env=stage_environment(root))
            row['pid'] = process.pid
            job.update(status='running', current=dict(name=name, pid=process.pid, log=row['log']))
            save(root / 'job.json', job)
            while process.poll() is None:
                elapsed = time.monotonic() - began
                cpu, size = cpu_total() + child_cpu(process.pid), output_bytes(root)
                gpu_used = job['gpu_seconds_completed'] + (elapsed if gpu else 0.)
                if gpu:
                    measured = snapshot()
                    own = [value['used_mib'] for value in measured['processes'] if value['pid'] == process.pid]
                    job['peak_own_gpu_mib'] = max([job['peak_own_gpu_mib'], *own])
                    job['gpu_seconds_current_stage'] = elapsed
                if not budget_ok(cpu, gpu_used, size) or job['peak_own_gpu_mib'] > GPU_PEAK_MIB:
                    failure = BudgetHold('Own probe CPU/GPU/memory/output cap reached; no reduced-data retry')
                    process.terminate()  # Only this newly launched probe child.
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        row['termination_timed_out'] = True
                    break
                job.update(CPU_seconds_observed=cpu, output_bytes=size)
                save(root / 'job.json', job)
                time.sleep(2)
            if process.poll() is not None:
                row['returncode'] = process.wait()
    finally:
        elapsed = time.monotonic() - began
        live = process is not None and process.poll() is None
        row.update(status='child_still_running' if live else 'failed' if failure or row['returncode'] != 0 else 'complete',
                   elapsed_seconds=elapsed, checked_at=now())
        if not live:
            row['ended_at'] = now()
        if gpu:
            if live:
                job['gpu_seconds_current_stage'] = elapsed
            else:
                job['gpu_seconds_completed'] += elapsed
                job['gpu_seconds_current_stage'] = 0.
        job.update(CPU_seconds_completed=cpu_total(), output_bytes=output_bytes(root), child_still_running=live)
        if not live:
            job['current'] = None
        save(root / 'job.json', job)
    if failure:
        raise failure
    hold = stage_budget_hold(root, name)
    if hold is not None:
        row.update(status='budget_hold', budget_receipt=hold)
        save(root / 'job.json', job)
        raise BudgetHold(f'{name} explicitly held by fixed budget gate; no retry or reduced-data run')
    if row['returncode'] == -signal.SIGXCPU:
        row['status'] = 'budget_hold'
        save(root / 'job.json', job)
        raise BudgetHold(f'{name} reached its own fixed CPU limit; no retry')
    if row['returncode'] != 0:
        raise RuntimeError(f'{name} exited {row["returncode"]}; no retry, inspect stage log')
    if not budget_ok(job['CPU_seconds_completed'], job['gpu_seconds_completed'], job['output_bytes']):
        raise BudgetHold('Fixed cumulative resource cap reached at stage boundary')


def public_files(root):
    """Explicit public suffixes: never export spins.npy or a model checkpoint."""
    return sorted(str(path.relative_to(root)) for path in root.rglob('*')
        if path.is_file() and not path.is_symlink() and 'code' not in path.relative_to(root).parts
        and path.suffix.lower() in ('.json', '.png', '.npz'))


def run(root):
    root = root.resolve()
    if socket.gethostname() != HOST:
        raise RuntimeError('Pinned evaluation host differs')
    if not root.is_dir() or not (root / 'code').is_dir() or (root / 'job.json').exists():
        raise FileExistsError('Fresh deployed probe root required; no resume/restart')
    if any(path.name not in ('code', 'logs', 'launch.json') for path in root.iterdir()):
        raise FileExistsError('Unexpected existing output; never overwrite a prior probe')
    if Path(__file__).resolve() != root / 'code/scripts/run_ising_large_size_probe.py':
        raise RuntimeError('Execute the new pinned runner from its own deployed code directory')
    os.chdir(root / 'code')
    sys.path[:0] = [str(root / 'code'), str(FROZEN_CODE)]
    os.environ.update(stage_environment(root))
    (root / 'logs').mkdir(exist_ok=True)
    os.setpriority(os.PRIO_PROCESS, 0, 10)
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['NUMBA_CACHE_DIR'] = str(root / 'runtime_cache/numba')
    os.environ['MPLCONFIGDIR'] = str(root / 'runtime_cache/matplotlib')
    job = dict(schema_version=1, experiment=EXPERIMENT, status='validating_source', workflow_complete=False,
        root=str(root), parent_pid=os.getpid(), code_snapshot=str(root / 'code'), frozen_code=str(FROZEN_CODE),
        started_at=now(), checkpoint=str(CHECKPOINT), training_updates=0, MI_computation=False,
        K=[.42, .44, .46], new_size=1080, baseline_size=108, replicas=2, samples=64, C_draws=5,
        current=None, stages=[], CPU_seconds_completed=PRIOR_CPU_RESERVE, gpu_seconds_completed=0.,
        gpu_seconds_current_stage=0., peak_own_gpu_mib=0, child_still_running=False,
        budget=dict(CPU_core_hours=4, GPU_card_seconds=900, GPU_peak_mib=8192, output_bytes=OUTPUT_LIMIT),
        authorization='用户要求1080尺度外试验，并明确少量系综/构型；冻结当前模型，不重训，不自动扩样',
        error=None, requires_user_action=False, thermal_nu=None)
    save(root / 'job.json', job)
    try:
        source = validate_sources()
        record = root / 'source_record'
        record.mkdir(exist_ok=False)
        source_json_sha256 = {}
        for name, original in (('manifest.json', SOURCE_ROOT / 'run_early/manifest.json'),
                               ('summary.json', SOURCE_ROOT / 'run_early/summary.json'),
                               ('job.json', SOURCE_ROOT / 'job.json'), ('plan.json', SOURCE_ROOT / 'plan.json')):
            ordinary(original, SOURCE_ROOT)
            with original.open('rb') as stream, (record / name).open('xb') as target:
                shutil.copyfileobj(stream, target)
            source_json_sha256[name] = sha(record / name)
        source['source_json_sha256'] = source_json_sha256
        plan = dict(schema_version=1, experiment=EXPERIMENT, source=source,
            new_code_sha256={name: sha(ordinary(root / 'code' / name, root / 'code')) for name in NEW_FILES},
            sampling=dict(K=[.42, .44, .46], size=1080, replicas=2, samples=64, equil=4096, gap=128,
                initial_conditions=['hot', 'cold'], extra_pilots=0, independent_production_chains=True,
                no_equilibrium_certificate=True, seed=20261018),
            evaluation=dict(sizes=[[108,36,12,4], [1080,360,120,40]], depth=3,
                trained_depth=2, estimator='mean_of_two_disjoint_Q_halves_v1', samples=64,
                C_draws=5, QCR_updates=0, MI=False, label_supervision=False),
            budget=job['budget'], qualification='Exploratory two-chain test, no significance claim or adaptive resampling')
        save(root / 'plan.json', plan)
        run_stage(root, job, plan, 'sampling_preflight', command(root, 'scripts.ising_large_size_probe_sampling', 'benchmark'))
        benchmark = read(root / 'benchmark/receipt.json')
        if benchmark.get('status') == 'budget_hold' or not benchmark.get('budget_gate', {}).get('passed'):
            raise BudgetHold('Measured CPU projection exceeds the fixed4-core-hour cap; production not started')
        job['cpu_projection'] = cpu_projection_gate(benchmark, 0, actual_cpu=cpu_total())
        if not job['cpu_projection']['passed']:
            raise BudgetHold('Conservative sampling plus evaluation CPU projection exceeds4 core-hours')
        run_stage(root, job, plan, 'sample_c000', command(root, 'scripts.ising_large_size_probe_sampling', 'sample', '--condition-index', 0))
        run_stage(root, job, plan, 'evaluation_preflight', command(root, 'scripts.ising_large_size_probe_evaluation',
            'benchmark', '--checkpoint', CHECKPOINT, '--snapshot-root', FROZEN_CODE,
            '--baseline-root', BASELINE_ROOT, '--device', 'cuda:2',
            '--prior-gpu-seconds', job['gpu_seconds_completed']), gpu=True)
        gpu_receipt = read(root / 'gpu_preflight.json')
        # Worker records a conservative remaining total; the parent charges prior actual usage.
        estimate = gpu_receipt.get('projected_remaining_gpu_seconds')
        if (type(estimate) not in (int, float) or not math.isfinite(estimate) or estimate < 0
                or job['gpu_seconds_completed'] + estimate > GPU_LIMIT):
            raise BudgetHold('Measured GPU projection does not fit900 seconds; no further sampling')
        for index in (1, 2):
            job['cpu_projection'] = cpu_projection_gate(benchmark, index, actual_cpu=cpu_total())
            if not job['cpu_projection']['passed']:
                raise BudgetHold('Actual CPU plus conservative remaining stages exceeds4 core-hours')
            run_stage(root, job, plan, f'sample_c{index:03d}', command(root, 'scripts.ising_large_size_probe_sampling',
                'sample', '--condition-index', index))
        run_stage(root, job, plan, 'frozen_collection', command(root, 'scripts.ising_large_size_probe_evaluation',
            'collect', '--checkpoint', CHECKPOINT, '--snapshot-root', FROZEN_CODE,
            '--baseline-root', BASELINE_ROOT, '--device', 'cuda:2',
            '--prior-gpu-seconds', job['gpu_seconds_completed']), gpu=True)
        run_stage(root, job, plan, 'report', command(root, 'scripts.report_ising_large_size_probe'))
        report_status = read(root / 'report/status.json')
        if report_status.get('status') != 'complete' or report_status.get('workflow_complete') is not True:
            raise RuntimeError('Official probe report did not complete')
        assert_frozen(root, plan)
        job.update(status='complete', workflow_complete=True, completed_at=now())
    except BudgetHold as error:
        job.update(status='budget_hold', requires_user_action=True, error=str(error), checked_at=now())
    except BaseException as error:
        job.update(status='failed', requires_user_action=True, error=f'{type(error).__name__}: {error}', checked_at=now())
    finally:
        job['public_files'] = public_files(root)
        save(root / 'job.json', job)
    return job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    if args.launch:
        if not root.is_dir() or (root / 'job.json').exists() or (root / 'launch.json').exists():
            raise FileExistsError('New deployed probe only; no duplicate launch')
        (root / 'logs').mkdir(exist_ok=False)
        cmd = [sys.executable, '-m', 'scripts.run_ising_large_size_probe', '--root', str(root)]
        with (root / 'logs/parent.log').open('x') as log:
            child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        receipt = dict(status='launched', pid=child.pid, command=cmd, root=str(root), started_at=now(),
                       training_updates=0, pinned_host=HOST)
        save(root / 'launch.json', receipt)
        print(json.dumps(receipt), flush=True)
    else:
        print(json.dumps(run(root)), flush=True)


if __name__ == '__main__':
    main()
