"""One isolated, bounded paired unsupervised future-physics experiment."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


GPU_UUID = 'GPU-713b07d0-118c-f939-c0a8-565390215e01'
GPU_SECONDS = 3600
CPU_SECONDS = 24 * 3600
COLLECTION_RESERVE = 120.
REPORT_RESERVE = 60.


class BudgetHold(RuntimeError):
    """A resource budget is exhausted; this is not a scientific/training failure."""


def gpu_budget(job):
    """One campaign limit, shared by admission gates and live accounting."""
    value = job.get('GPU_card_seconds', GPU_SECONDS)
    if type(value) is not int or value not in (3600, 5400):
        raise ValueError('GPU campaign budget must be the original3600 or authorized5400 seconds')
    return value


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path, obj):
    target = Path(path)
    temporary = target.with_suffix(target.suffix + '.pending')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(target)


def snapshot():
    memory = dict(row.split(':', 1) for row in Path('/proc/meminfo').read_text().splitlines())
    full = next(row for row in Path('/proc/pressure/io').read_text().splitlines() if row.startswith('full '))
    io = float(next(word.split('=')[1] for word in full.split() if word.startswith('avg10=')))
    line = subprocess.check_output(['nvidia-smi', '-i', '2', '--query-gpu=uuid,memory.free',
                                    '--format=csv,noheader,nounits'], text=True).strip()
    uuid, free = [part.strip() for part in line.split(',')]
    processes = subprocess.check_output(['nvidia-smi', '-i', '2', '--query-compute-apps=pid,used_memory',
                                         '--format=csv,noheader,nounits'], text=True).strip()
    rows = []
    for row in processes.splitlines():
        if row:
            pid, used = [part.strip() for part in row.split(',')]
            rows.append(dict(pid=int(pid), used_mib=int(used)))
    return dict(checked_at=now(), gpu_uuid=uuid, free_mib=int(free), processes=rows,
                mem_available_gib=int(memory['MemAvailable'].split()[0]) / 1024**2,
                load1=os.getloadavg()[0], cpu_count=os.cpu_count(), io_full_avg10=io)


def host_ready(row):
    return row['mem_available_gib'] >= 36 and row['load1'] <= 1.25 * row['cpu_count'] and row['io_full_avg10'] < 20


def ready(row):
    return host_ready(row) and row['gpu_uuid'] == GPU_UUID and row['free_mib'] >= 12 * 1024 and not row['processes']


def wait_resources(root, job, gpu):
    checks = []
    while True:
        row = snapshot()
        if row['gpu_uuid'] != GPU_UUID:
            raise RuntimeError('Pinned GPU2 physical UUID changed')
        checks.append(row); checks = checks[-2:]
        job.update(status='waiting_resources', resource_checks=checks)
        save(root / 'job.json', job)
        if (not gpu and host_ready(row)) or (gpu and len(checks) == 2 and all(ready(x) for x in checks)):
            job['status'] = 'running'; save(root / 'job.json', job); return
        time.sleep(10 if gpu else 30)


def source_hashes(code):
    records = {}
    for folder in ('recurrent_rg', 'experiments', 'scripts'):
        for path in sorted((code / folder).rglob('*.py')):
            if path.name.startswith('._'): continue
            records[str(path.relative_to(code))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return records


def prior_execution_receipt(prior, source):
    """Carry this experiment's unsuccessful execution check, never old science."""
    if prior is None:
        return dict(gpu_seconds=0., CPU_seconds=0., cache_source=None)
    prior = prior.resolve()
    job = json.loads((prior / 'job.json').read_text())
    if (job.get('experiment') != 'ising_future_physics_Q_loss_b3_paired_v1' or
            Path(job['source']).resolve() != source.resolve() or
            job.get('status') != 'failed' or
            any(row['name'].startswith('train_') for row in job['stages'])):
        raise ValueError('Only the preceding new-experiment preflight is eligible')
    gpu = job['gpu_seconds_completed'] + job.get('gpu_seconds_current_stage', 0.)
    # Missing benchmark rusage is bounded conservatively by its whole wall time
    # at one CPU thread, in addition to already measured completed CPU usage.
    cpu = job.get('CPU_seconds_completed', 0.) + job.get('gpu_seconds_current_stage', 0.)
    if any(not math.isfinite(value) or value < 0 for value in (gpu, cpu)):
        raise ValueError('Prior execution accounting must be finite and nonnegative')
    return dict(root=str(prior), job_sha256=hashlib.sha256((prior / 'job.json').read_bytes()).hexdigest(),
                gpu_seconds=gpu, CPU_seconds=cpu, cache_source=str(prior / 'mi_cache'),
                previous_peak_own_gpu_mib=max(job.get('campaign_peak_own_gpu_mib') or 0,
                                            job.get('peak_own_gpu_mib') or 0),
                reason='Preserved preflight; TEST MI placement repair, no production resume or scientific result')


def reuse_own_statistics(root, prior):
    if not prior.get('cache_source'):
        return
    old = Path(prior['root'])
    previous = json.loads((old / 'dataset_receipt.json').read_text())
    current = json.loads((root / 'dataset_receipt.json').read_text())
    if previous['data_sha256'] != current['data_sha256']:
        raise ValueError('Cannot reuse current-experiment MI statistics across different data fingerprints')
    cache = root / 'mi_cache'; cache.mkdir(exist_ok=False)
    records = []
    sources = sorted(Path(prior['cache_source']).glob('mi-v1-*.pt'))
    if len(sources) != 2:
        raise ValueError('Require exactly the completed TRAIN and TEST new-data statistics caches')
    for source in sources:
        if source.is_symlink() or not source.is_file():
            raise ValueError('Unexpected current-experiment statistics cache')
        destination = cache / source.name
        os.link(source, destination)
        records.append(dict(name=source.name, bytes=source.stat().st_size,
            sha256=file_sha(source), immutable_hardlink=True))
    save(root / 'statistics_reuse.json', dict(source=str(old), data_sha256=current['data_sha256'],
        records=records, note='Only complete new-data MI statistics. Normal cache identity validation still applies.'))


def validated_preflight(prior, source, code):
    """Read JSON/bytes only; reuse completed execution timing, never model state."""
    prior = prior.resolve()
    read = lambda relative: json.loads((prior / relative).read_text())
    job, plan = read('job.json'), read('plan.json')
    stages = job.get('stages', [])
    if (job.get('experiment') != 'ising_future_physics_Q_loss_b3_paired_v1' or
            Path(job.get('source', '')).resolve() != source.resolve() or
            job.get('status') != 'budget_hold' or job.get('workflow_complete') is not False or
            job.get('current') is not None or job.get('child_still_running') is not False or
            job.get('device') != 'cuda:2' or job.get('gpu_uuid') != GPU_UUID or
            type(job.get('peak_own_gpu_mib')) not in (int, float) or
            not math.isfinite(job['peak_own_gpu_mib']) or not 0 < job['peak_own_gpu_mib'] <= 8192 or
            [row.get('name') for row in stages] != ['prepare', 'calibration', 'benchmark'] or
            any(row.get('status') != 'complete' or row.get('returncode') != 0 for row in stages) or
            job.get('gpu_seconds_current_stage', 0.) != 0. or
            any((prior / name).exists() for name in ('run_control', 'run_future'))):
        raise ValueError('Reuse requires a complete budget-held preflight, with no formal training or live child')
    current_hashes, previous_hashes = source_hashes(code), plan.get('source_code_sha256', {})
    runner = 'scripts/run_ising_future_physics.py'
    if ({key:value for key, value in current_hashes.items() if key != runner} !=
            {key:value for key, value in previous_hashes.items() if key != runner} or
            runner not in current_hashes or runner not in previous_hashes):
        raise ValueError('Preflight science source hashes differ; only the budget runner may change')
    config, bench_config = read('config.json'), read('benchmark_config.json')
    expected = json.loads(json.dumps(config))
    expected['training'].update(epochs=40, log_every=20, checkpoint_every=40)
    summary, manifest = read('benchmark/summary.json'), read('benchmark/manifest.json')
    def specified_fields_match(raw, expanded):
        return (isinstance(expanded, dict) and all(key in expanded and specified_fields_match(value, expanded[key])
                for key, value in raw.items())) if isinstance(raw, dict) else raw == expanded
    if (bench_config != expected or config.get('training', {}).get('epochs') != 6000 or
            summary.get('actual_epochs') != 40 or summary.get('requested_epochs') != 40 or
            any(summary.get(key, False) for key in
                ('stopped_by_wall_limit', 'stopped_by_training_limit', 'stopped_by_train_plateau')) or
            summary.get('config') != manifest.get('config') or
            not specified_fields_match(bench_config, manifest.get('config')) or
            manifest.get('config', {}).get('training', {}).get('epochs') != 40):
        raise ValueError('Reuse requires the exact completed40-update benchmark and unchanged6000-update config')
    receipt, pairs, norm = read('dataset_receipt.json'), read('pairs.json'), read('normalizer.json')
    data_path = prior / 'data' / 'ensembles.pt'
    if Path(config.get('data', {}).get('cache', '')).resolve() != data_path.resolve():
        raise ValueError('Preflight config must name its own isolated new-data cache')
    hashes = {name:file_sha(prior / name) for name in
              ('data/ensembles.pt', 'pairs.json', 'normalizer.json')}
    data_sha, pair_sha, norm_sha = (hashes[name] for name in
        ('data/ensembles.pt', 'pairs.json', 'normalizer.json'))
    future = manifest.get('q_future_physics', {})
    execution = manifest.get('execution_optimizations', {})
    if (receipt.get('data_sha256') != data_sha or receipt.get('pair_sha256') != pair_sha or
            pairs.get('data_sha256') != data_sha or norm.get('data_sha256') != data_sha or
            norm.get('pairs_sha256') != pair_sha or manifest.get('data_sha256') != data_sha or
            future.get('data_sha256') != data_sha or future.get('pairs_sha256') != pair_sha or
            future.get('normalizer_sha256') != norm_sha or future.get('weight') != 1 or
            future.get('hidden') != 32 or future.get('mode') != 'future_physics' or
            future.get('labels_used') is not False or future.get('paths') != [[0,1],[1,2],[0,2]] or
            future.get('target_mean') != norm.get('target_mean') or
            future.get('target_std') != norm.get('target_std') or
            future.get('encoding_execution') != dict(encoding_batch_chunk=4, validated_binary_inputs=True) or
            execution.get('mi_test_device') != 'cpu' or execution.get('future_encoding_batch_chunk') != 4 or
            manifest.get('execution_device') != 'cuda:2' or
            summary.get('contract', {}).get('q_future_physics') != future or
            summary.get('contract', {}).get('execution_optimizations') != execution):
        raise ValueError('Preflight data/pairs/normalizer SHA or benchmark loss/execution contract mismatch')
    # Validate timing before creating any reused files. Never open its checkpoints.
    benchmark_projection(summary)
    gpu, cpu = job.get('gpu_seconds_completed'), job.get('CPU_seconds_completed')
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in (gpu, cpu)):
        raise ValueError('Preflight cumulative budget accounting must be finite and nonnegative')
    records = {name:file_sha(prior / name) for name in
        ('job.json', 'plan.json', 'config.json', 'benchmark_config.json', 'dataset_receipt.json',
         'benchmark/summary.json', 'benchmark/manifest.json')}
    records.update(hashes)
    return dict(root=str(prior), job_sha256=records['job.json'], gpu_seconds=gpu, CPU_seconds=cpu,
        cache_source=str(prior / 'mi_cache'), previous_peak_own_gpu_mib=max(
            job.get('campaign_peak_own_gpu_mib') or 0, job.get('peak_own_gpu_mib') or 0),
        file_sha256=records, reason='User-authorized budget extension; reuse validated execution timing and immutable inputs only')


def reuse_preflight(root, receipt):
    old = Path(receipt['root'])
    (root / 'data').mkdir()
    (root / 'benchmark').mkdir()
    for name in receipt['file_sha256']:
        if name in ('job.json', 'plan.json', 'config.json'):
            continue
        source = old / name
        if source.is_symlink() or not source.is_file() or file_sha(source) != receipt['file_sha256'][name]:
            raise ValueError('Validated preflight bytes changed before reuse')
        os.link(source, root / name)
    config = json.loads((old / 'config.json').read_text())
    if file_sha(old / 'config.json') != receipt['file_sha256']['config.json']:
        raise ValueError('Validated preflight config changed before reuse')
    config['data']['cache'] = str(root / 'data' / 'ensembles.pt')
    save(root / 'config.json', config)
    reuse_own_statistics(root, receipt)
    save(root / 'preflight_reuse.json', dict(**receipt, destination=str(root),
        config_change='Only data.cache absolute path points to the byte-identical hardlinked new-data file',
        model_or_optimizer_reused=False, checkpoint_files_opened=False,
        additional_prepare_calibration_benchmark_updates=0,
        note='No synthetic execution stages. Both formal arms initialize fresh; benchmark checkpoints are neither opened nor linked.'))
    return json.loads((root / 'benchmark' / 'summary.json').read_text())


def run_stage(root, job, name, command, *, gpu=False):
    job['current'] = dict(name=name, command=command, pid=None)
    wait_resources(root, job, gpu)
    row = dict(name=name, status='running', started_at=now(), command=command,
               gpu_stage=gpu, log=str(root / 'logs' / (name + '.log')))
    job['stages'].append(row); job['status'] = 'running'
    began = time.monotonic()
    process, failure = None, None
    try:
        with open(row['log'], 'x') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            row['pid'] = process.pid; job['current'] = dict(name=name, pid=process.pid, log=row['log'])
            save(root / 'job.json', job)
            while process.poll() is None:
                if gpu:
                    measured = snapshot()
                    own = [x for x in measured['processes'] if x['pid'] == process.pid]
                    peak = max([x['used_mib'] for x in own] + [0])
                    job['peak_own_gpu_mib'] = max(job.get('peak_own_gpu_mib', 0), peak)
                    job['campaign_peak_own_gpu_mib'] = max(job.get('campaign_peak_own_gpu_mib', 0), peak)
                    job['gpu_seconds_current_stage'] = time.monotonic() - began
                    if peak > 8192 or job['gpu_seconds_completed'] + time.monotonic() - began > gpu_budget(job):
                        failure = (RuntimeError('New experiment own GPU memory limit reached') if peak > 8192 else
                                   BudgetHold('Cumulative GPU card-time limit reached; fixed updates not shortened'))
                        # Only the child launched here is in scope; never kill on timeout.
                        process.terminate()
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            row['termination_timed_out'] = True
                        raise failure
                    save(root / 'job.json', job)
                time.sleep(2)
    except BaseException as error:
        failure = error
    finally:
        elapsed = time.monotonic() - began
        code = None if process is None else process.poll()
        live = process is not None and code is None
        row.update(returncode=code, elapsed_seconds=elapsed, checked_at=now(),
            status='child_still_running' if live else 'failed' if failure is not None or code != 0 else 'complete')
        if not live:
            row['ended_at'] = now()
        if failure is not None:
            row['error'] = f'{type(failure).__name__}: {failure}'
        if gpu:
            if live:
                job['gpu_seconds_current_stage'] = elapsed
            else:
                job['gpu_seconds_completed'] += elapsed
                job['gpu_seconds_current_stage'] = 0.
        child_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        own_usage = resource.getrusage(resource.RUSAGE_SELF)
        job['CPU_seconds_completed'] = (job.get('prior_execution', {}).get('CPU_seconds', 0.) +
            child_usage.ru_utime + child_usage.ru_stime + own_usage.ru_utime + own_usage.ru_stime)
        job['output_bytes'] = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
        job['child_still_running'] = live
        if not live: job['current'] = None
        save(root / 'job.json', job)
    if failure is not None:
        raise failure
    if row['returncode']:
        raise RuntimeError(f'{name} exited {row["returncode"]}; inspect {row["log"]}')
    if (job['CPU_seconds_completed'] > CPU_SECONDS or job['output_bytes'] > 10 * 1024**3
            or job['gpu_seconds_completed'] > gpu_budget(job)):
        raise BudgetHold('Cumulative CPU/GPU or new output-storage budget reached')


def benchmark_projection(summary):
    """Conservative remaining fixed-work costs, including setup and all writes."""
    timing = summary['timing']
    steps = timing['steady_step_seconds']
    if (not steps or not summary['history'] or any(not math.isfinite(x) or x <= 0 for x in steps)
            or any(not math.isfinite(timing[key]) or timing[key] < 0
                   for key in ('setup_seconds', 'logging_seconds', 'checkpoint_seconds'))):
        raise ValueError('Benchmark timing must contain finite measured steps/setup/logging/checkpoints')
    steady = max(sum(steps) / len(steps), sorted(steps)[int(.9 * (len(steps) - 1))])
    per_log = timing['logging_seconds'] / len(summary['history'])
    setup = max(60., timing['setup_seconds'])
    # Benchmark writes one numbered checkpoint plus the final checkpoint/model.
    # Six times that whole measured write bundle upper-bounds six numbered
    # checkpoints plus one final bundle; retain the outer 1.5 safety factor.
    checkpoints = 6 * timing['checkpoint_seconds']
    per_arm = 1.5 * (6000 * steady + 31 * per_log + setup + checkpoints)
    return dict(steady_step_seconds_conservative=steady, log_seconds_estimate=per_log,
        measured_setup_seconds=timing['setup_seconds'], setup_seconds_per_arm=setup,
        checkpoint_seconds_per_arm=checkpoints, logs_per_arm=31, updates_per_arm=6000,
        safety_factor=1.5, per_arm_seconds=per_arm, collection_seconds=COLLECTION_RESERVE,
        report_reserve_seconds=REPORT_RESERVE)


def remaining_projection(job, estimate, *, arms, collections):
    if arms not in (0, 1, 2) or collections not in (0, 1, 2):
        raise ValueError('Remaining work must be an explicit fixed two-arm suffix')
    remaining = arms * estimate['per_arm_seconds'] + collections * COLLECTION_RESERVE
    gpu = job['gpu_seconds_completed'] + job.get('gpu_seconds_current_stage', 0.)
    cpu = job.get('CPU_seconds_completed', job.get('prior_execution', {}).get('CPU_seconds', 0.))
    return dict(actual_gpu_seconds=gpu, actual_CPU_seconds=cpu, remaining_arms=arms,
        remaining_collections=collections, projected_remaining_seconds=remaining,
        projected_total_gpu_seconds=gpu + remaining, allowed_gpu_seconds=gpu_budget(job),
        projected_total_CPU_seconds=cpu + remaining + REPORT_RESERVE, allowed_CPU_seconds=CPU_SECONDS,
        CPU_projection_note='Single-thread child wall time conservatively bounds future CPU use', **estimate)


def budget_gate(root, job, estimate, *, arms, collections, before):
    projection = remaining_projection(job, estimate, arms=arms, collections=collections)
    projection['before_stage'] = before
    job['budget_projection'] = projection
    job.setdefault('budget_projection_history', []).append(projection)
    passed = (projection['projected_total_gpu_seconds'] <= gpu_budget(job) and
              projection['projected_total_CPU_seconds'] <= CPU_SECONDS)
    if not passed:
        job.update(status='budget_hold', requires_user_action=True, workflow_complete=False, current=None,
            error=f'{before}前，剩余固定两臂/采集/写盘工作保守预算不通过；未减6000步、未启动下一阶段')
    save(root / 'job.json', job)
    return passed


def training_time_limit(job, estimate, *, later_arms, collections):
    reserve = (later_arms * estimate['per_arm_seconds'] + collections * COLLECTION_RESERVE +
               estimate['safety_factor'] * (estimate['setup_seconds_per_arm'] + estimate['checkpoint_seconds_per_arm']))
    limit = gpu_budget(job) - job['gpu_seconds_completed'] - reserve
    if not math.isfinite(limit) or limit <= 0:
        raise BudgetHold('No training-only time remains after preserving the fixed remaining workflow')
    return limit


def trainer_command(root, output, config, *, future, benchmark=False, limit=2400):
    command = [sys.executable, '-m', 'experiments.rsmi_separate_qcr', '--config', str(config),
        '--output', str(output), '--device', 'cuda:2', '--q-future-pairs', str(root / 'pairs.json'),
        '--mi-cache-dir', str(root / 'mi_cache'), '--mi-test-cpu', '--skip-final-report',
        '--max-training-seconds', str(limit), '--benchmark-warmup', '5']
    if future:
        command += ['--q-future-weight', '1', '--q-future-hidden', '32',
                    '--q-future-normalizer', str(root / 'normalizer.json')]
    else:
        command += ['--q-future-control']
    return command


def run(root, source, prior=None, *, gpu_seconds=3600, reuse=False):
    root = root.resolve(); root.mkdir(parents=True, exist_ok=True)
    if (root / 'job.json').exists() or (root / 'config.json').exists():
        raise FileExistsError('A new experiment must have a fresh output; no automatic restart')
    (root / 'logs').mkdir(exist_ok=True)
    os.setpriority(os.PRIO_PROCESS, 0, 19)
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    code = Path.cwd().resolve()
    gpu_budget(dict(GPU_card_seconds=gpu_seconds))
    if reuse and (prior is None or gpu_seconds != 5400):
        raise ValueError('Preflight reuse requires its prior root and the authorized5400-second budget')
    prior_receipt = (validated_preflight(prior, source, code) if reuse else
                     prior_execution_receipt(prior, source))
    job = dict(schema_version=1, experiment='ising_future_physics_Q_loss_b3_paired_v1',
        status='running', workflow_complete=False, started_at=now(), parent_pid=os.getpid(),
        source=str(source), output=str(root), code_snapshot=str(code),
        authorization='用户明确要求实现未来物理预测loss并在无监督条件接入训练',
        GPU_card_seconds=gpu_seconds,
        budget_authorization=('用户本轮明确回复“扩大”，将GPU总预算从1卡时扩大至1.5卡时（5400秒）'
                              if gpu_seconds == 5400 else '原定GPU总预算1卡时（3600秒）'),
        current=None, stages=[], gpu_seconds_completed=prior_receipt['gpu_seconds'], peak_own_gpu_mib=0,
        CPU_seconds_completed=prior_receipt['CPU_seconds'],
        campaign_peak_own_gpu_mib=prior_receipt.get('previous_peak_own_gpu_mib') or 0,
        prior_execution=prior_receipt, mi_test_device='cpu',
        device='cuda:2', gpu_uuid=GPU_UUID, cpu_threads=1, nice=19,
        error=None, requires_user_action=False, thermal_nu=None)
    save(root / 'job.json', job)
    plan = dict(schema_version=1, source=str(source), config='fresh schema5 b3, g1, hidden96, 108→36→12',
        training=dict(arms=['paired_control', 'future_weight1'], updates_per_arm=6000, batch_chains=16,
                      samples_per_chain=64, seed=91, Adam_learning_rate=.001),
        objective=dict(Q='effective20 subset +5 coarse subset +10 shared affine reconstruction + optional1 future physical NMSE',
                       C='-2 empirical full-shell buffered MI +2 per-ensemble variance floor +1 uniform color NLL',
                       R='5 one-step +5 micro-start all-horizon closure', independent_gradients=True),
        future=dict(hidden=32, paths=[[0,1],[1,2],[0,2]], observations=['L/6关联', 'L/3关联', '最低轴向S', '第二轴向S', '最低对角S'],
                    calibration='initial random C; all112TRAIN chains;5C;fixed std floor .01; no pretraining or labels',
                    actual_per_update_C_draws=1),
        budget=dict(GPU_card_seconds=gpu_seconds, GPU_peak_mib=8192, CPU_core_hours=24, output_gib=10,
                    projection='all prior/current actual +1.5x measured steady updates/logging/cached setup/checkpoint writes;120s per remaining GPU collection;60s CPU-only report'),
        source_code_sha256=source_hashes(code), zero_weight_default_unchanged=True,
        prior_execution=prior_receipt, budget_authorization=job['budget_authorization'],
        execution=dict(TRAIN_MI_device='cuda:2', TEST_MI_device='cpu',
            estimator_unchanged=True, no_TEST_gradients=True),
        no_adaptive_hyperparameter_selection=True, source_experiments_immutable=True)
    save(root / 'plan.json', plan)
    python = sys.executable
    try:
        if job['gpu_seconds_completed'] >= gpu_budget(job) or job['CPU_seconds_completed'] >= CPU_SECONDS:
            raise BudgetHold('Prior execution already exhausted the campaign budget; no new stage started')
        if reuse:
            summary = reuse_preflight(root, prior_receipt)
        else:
            run_stage(root, job, 'prepare', [python, '-m', 'scripts.prepare_ising_future_physics', 'prepare',
                '--root', str(root), '--source', str(source)])
            reuse_own_statistics(root, prior_receipt)
            run_stage(root, job, 'calibration', [python, '-m', 'scripts.prepare_ising_future_physics', 'calibrate',
                '--root', str(root), '--device', 'cuda:2'], gpu=True)
            benchmark_config = json.loads((root / 'config.json').read_text())
            benchmark_config['training'].update(epochs=40, log_every=20, checkpoint_every=40)
            save(root / 'benchmark_config.json', benchmark_config)
            run_stage(root, job, 'benchmark', trainer_command(root, root / 'benchmark', root / 'benchmark_config.json',
                      future=True, benchmark=True, limit=600), gpu=True)
            summary = json.loads((root / 'benchmark' / 'summary.json').read_text())
        if summary['actual_epochs'] != 40:
            raise RuntimeError('Benchmark did not complete its fixed40 execution-check updates')
        estimate = benchmark_projection(summary)
        # Gate again after every completed arm/collection using actual cost;
        # never launch a suffix whose fixed6000 updates no longer fit.
        for index, (name, future) in enumerate([('control', False), ('future', True)]):
            if not budget_gate(root, job, estimate, arms=2-index, collections=2-index, before=f'train_{name}'):
                return
            remaining = training_time_limit(job, estimate, later_arms=1-index, collections=2-index)
            run_path = root / f'run_{name}'
            run_stage(root, job, f'train_{name}', trainer_command(root, run_path, root / 'config.json',
                      future=future, limit=remaining), gpu=True)
            trained = json.loads((run_path / 'summary.json').read_text())
            if trained['actual_epochs'] != 6000 or any(trained.get(key, False) for key in
                    ('stopped_by_wall_limit', 'stopped_by_training_limit', 'stopped_by_train_plateau')):
                job.update(status='budget_hold', requires_user_action=True, workflow_complete=False,
                           error=f'{name}保存实际{trained["actual_epochs"]}步，未达到固定6000步；不自动续训或重跑')
                save(root / 'job.json', job); return
            if not budget_gate(root, job, estimate, arms=1-index, collections=2-index, before=f'collect_{name}'):
                return
            run_stage(root, job, f'collect_{name}', [python, '-m', 'scripts.prepare_ising_future_physics', 'collect',
                '--root', str(root), '--run', str(run_path), '--device', 'cuda:2'], gpu=True)
        if not budget_gate(root, job, estimate, arms=0, collections=0, before='report'):
            return
        run_stage(root, job, 'report', [python, '-m', 'scripts.report_ising_future_physics', '--root', str(root)])
        job.update(status='complete', workflow_complete=True, completed_at=now(), current=None)
        save(root / 'job.json', job)
    except BudgetHold as error:
        job.update(status='budget_hold', workflow_complete=False, error=f'{type(error).__name__}: {error}',
                   requires_user_action=True, ended_at=now())
        save(root / 'job.json', job)
    except BaseException as error:
        job.update(status='failed', workflow_complete=False, error=f'{type(error).__name__}: {error}',
                   requires_user_action=True, ended_at=now())
        save(root / 'job.json', job)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--prior-execution', type=Path)
    parser.add_argument('--gpu-budget-seconds', type=int, choices=(3600, 5400), default=3600)
    parser.add_argument('--reuse-preflight', action='store_true',
                        help='Reuse completed budget-held preflight JSON timing and immutable inputs, never checkpoints')
    args = parser.parse_args()
    if args.reuse_preflight and (args.prior_execution is None or args.gpu_budget_seconds != 5400):
        parser.error('--reuse-preflight requires --prior-execution and the authorized --gpu-budget-seconds 5400')
    if args.launch:
        root = args.output.resolve(); root.mkdir(parents=True, exist_ok=True)
        receipt = root / 'launch.json'
        if receipt.exists() or (root / 'job.json').exists(): raise FileExistsError(root)
        (root / 'logs').mkdir(exist_ok=True)
        command = [sys.executable, '-m', 'scripts.run_ising_future_physics', '--output', str(root), '--source', str(args.source.resolve())]
        command += ['--gpu-budget-seconds', str(args.gpu_budget_seconds)]
        if args.prior_execution:
            command += ['--prior-execution', str(args.prior_execution.resolve())]
        if args.reuse_preflight:
            command += ['--reuse-preflight']
        with open(root / 'logs' / 'parent.log', 'x') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        record = dict(status='launched', pid=child.pid, output=str(root), command=command, started_at=now())
        save(receipt, record); print(json.dumps(record), flush=True)
    else:
        run(args.output, args.source, args.prior_execution,
            gpu_seconds=args.gpu_budget_seconds, reuse=args.reuse_preflight)


if __name__ == '__main__': main()
