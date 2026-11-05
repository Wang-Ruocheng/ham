"""二维 Ising B=0 论文消融第一阶段：五个全新训练 seed，串行、不重试。

只使用已存在的同一批物理数据和 b3 数据统计。后续 loss、Q 容量、
尺寸／深度写入计划但不会由本阶段偷偷启动。正式配置和代码先冻结。
"""
from __future__ import annotations

import argparse
from copy import deepcopy
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

from scripts.run_phi4_qcr_large import resource_snapshot, source_hashes

HOST = 'thinklab-105-225'
MODULE = 'scripts.run_ising_b0_multiseed'
SEEDS = (91, 92, 93, 94, 95)
UPDATES = 12000
PROFILE_UPDATES = 20
BASE = Path('/mnt/nas-new/home/yange/ham')
REFERENCE = BASE / 'results/ising2d_positiveK_QCR_dim_scale_ablation12000_seed91_vm225'
DATA_SHA = 'f0e319a771d220e948c70e97fda8bbd373d46fdbb280211f0cbd29ce9d36de54'
SIDECAR_SHA = '0d2d7591b37aa27757bde1d1d964059d38b64ffb6374df1d3d7724993e801e2a'
MI_RECEIPT_SHA = '07e9285509e9b2b2a4fd0af7ccbc1db74b5abc7c11a147496453f3f82915dd8d'
MI_KEYS = dict(train='d5ed3c9ebc197011e4a8eac6743f17af0d0658a003875812e98f175399f06881',
               test='44fa100926e1196aee716658f1ed81d95dff2015e14f2745b4b5fb7641ab6cd5')
GPU_UUIDS = {0: 'GPU-5bba87f1-7d32-4dde-fe28-caba4468ab42',
             1: 'GPU-18fac97c-11ee-4624-d298-9074e8d7028c',
             2: 'GPU-713b07d0-118c-f939-c0a8-565390215e01',
             3: 'GPU-a167decd-ce49-6c8f-6bcf-136fde596683'}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value, *, exclusive=False):
    path = Path(path)
    serialized = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if exclusive:
        with path.open('x') as stream:
            stream.write(serialized)
    else:
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(serialized)
        temporary.replace(path)


def stat_identity(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Existing regular file required: {path}')
    stat = path.stat()
    return dict(size=stat.st_size, inode=stat.st_ino, mtime_ns=stat.st_mtime_ns, device=stat.st_dev)


def configs(raw):
    """Same physical data and current locked recipe; only seed varies in stage A."""
    from recurrent_rg.config import RunConfig
    from experiments.rsmi_separate_qcr import build_parser, validate_recipe
    base = RunConfig.from_dict(raw).to_dict()
    if (base['data']['cache'] != str(BASE / 'data/ising2d_positiveK_L108_b3_replicas12_seed91.pt')
            or base['model']['block_factor'] != 3):
        raise ValueError('Expected the audited pure nearest-neighbor L108 B0 reference config')
    base['model'].update(theory_dim=5, local_features=12)
    base['training'].update(max_rg_steps=2, epochs=UPDATES, threads=2,
        checkpoint_every=1000, log_every=200, learning_rate=.001,
        lr_schedule='multistep', lr_milestones=[6000, 9000], lr_gamma=.3)
    base['evaluation'].update(max_rg_steps=2, samples=128, systems=24)
    base['loss']['closure_scale_weights'] = None
    result = {}
    for seed in SEEDS:
        cfg = deepcopy(base)
        cfg['training']['seed'] = seed
        parsed = RunConfig.from_dict(cfg)
        parsed.validate()
        args = build_parser().parse_args(['--config', 'unused', '--output', 'unused',
                                        '--ising-b0-ablation', 'none'])
        validate_recipe(args, parsed)
        result[str(seed)] = parsed.to_dict()
    return result


def prepare(root, *, gpu=1, gpu_hours=8., cpu_hours=24., directory_gib=20.):
    from scripts.ising_qcr_evaluation import encoding_contract
    if socket.gethostname() != HOST or 'CUDA_VISIBLE_DEVICES' in os.environ:
        raise ValueError('Requires the declared physical host and unmasked physical GPU indices')
    if gpu not in GPU_UUIDS:
        raise ValueError('Unknown physical GPU')
    if any(isinstance(v, bool) or not math.isfinite(v) or v <= 0
           for v in (gpu_hours, cpu_hours, directory_gib)):
        raise ValueError('Finite positive budgets required')
    root, code = Path(root).resolve(), Path(__file__).resolve().parents[1]
    if root.exists() or root.is_symlink():
        raise FileExistsError('No overwrite, resume, retry or duplicate preparation')
    raw_path = REFERENCE / 'configs/expanded_scales.json'
    receipt_path = REFERENCE / 'expanded_scales/mi_preprocessing.json'
    receipt = read(receipt_path)
    arm_configs = configs(read(raw_path))
    cache = Path(arm_configs['91']['data']['cache'])
    sidecar = read(cache.with_suffix('.json'))
    if (sha(cache.with_suffix('.json')) != SIDECAR_SHA or sha(receipt_path) != MI_RECEIPT_SHA
            or sidecar.get('data_sha256') != DATA_SHA or sidecar.get('B') != 0.
            or sidecar.get('coupling_names') != ['K'] or receipt.get('status') != 'complete'
            or receipt.get('data_sha256') != DATA_SHA or receipt.get('model_initialized') is not False
            or receipt.get('optimizer_updates') != 0):
        raise ValueError('Audited data/MI provenance differs; no automatic rebuild')
    identities = {str(cache): stat_identity(cache)}
    for split, key in MI_KEYS.items():
        record = receipt['splits'][split]['cache']
        path = Path(record['path'])
        if record['key'] != key or record['cache_version'] != 1 or path.name != f'mi-v1-{key}.pt':
            raise ValueError('Wrong block-3 MI cache')
        identities[str(path)] = stat_identity(path)
    mi_directory = str(Path(receipt['splits']['train']['cache']['path']).parent)
    if Path(receipt['splits']['test']['cache']['path']).parent != Path(mi_directory):
        raise ValueError('MI splits must share their exact audited directory')
    profile = deepcopy(arm_configs['91'])
    profile['training'].update(seed=191, epochs=PROFILE_UPDATES,
                              log_every=PROFILE_UPDATES, checkpoint_every=PROFILE_UPDATES)
    plan = dict(schema_version=5, stage='A_multiseed_only', prepared_at=now(), host=HOST,
        code_root=str(code), output=str(root), device=f'cuda:{gpu}', gpu_uuid=GPU_UUIDS[gpu],
        source_sha256=source_hashes(code), source_config=str(raw_path),
        source_config_sha256=sha(raw_path), data_sha256=DATA_SHA,
        sidecar_sha256=SIDECAR_SHA, mi_receipt=str(receipt_path), mi_receipt_sha256=MI_RECEIPT_SHA,
        mi_cache_dir=mi_directory, mi_keys=MI_KEYS, immutable_stat_identity=identities,
        configs=arm_configs, profile_config=profile,
        protocol=dict(seeds=list(SEEDS), training_updates=UPDATES,
            physical_sampling_seed=91, physical_data_fixed_across_training_seeds=True,
            labels_used_in_training=False, fresh_QCR=True, source_checkpoint=None,
            block_factor=3, latent_dimension=5, scales=[108, 36, 12],
            Q_local_features=12, hidden_Q_C_R=96, closure_scale_weights='uniform',
            learning_rate_policy='Adam .001; multistep 6000/9000, gamma .3',
            C='-2 empirical histogram MI +2 ensemble variance floor +1 color NLL',
            Q='shared affine motif reconstruction10; microscopic/coarse subset20/5',
            R='one-step5 + microscopic-start rollout5; independent gradients',
            MI='reuse exact b3 data-only statistics; cache miss fails closed',
            execution_only='mmap physical/MI caches, selected-batch float64; TEST MI on CPU',
            selection='all five fixed endpoints; no best-seed or Kc/exponent selection',
            automatic_retry=False, resume=False, thermal_nu=None,
            report='CPU single-thread exact binary tables; fixed TRAIN root seeds and full spectra',
            encoding_protocol=encoding_contract(128),
            encoding_caution='R evaluation uses the same half-subset-mean estimator as training; not Q(full128)'),
        future_stages=dict(automatic_execution=False,
            B=dict(order=2, paired_seeds=[91, 92, 93],
                arms=['c_no_mi', 'c_no_variance', 'c_no_color', 'q_no_reconstruction',
                      'q_no_subset', 'r_no_rollout'], baseline_reused=True,
                rule='exactly one component removed; all other model/data/update settings fixed'),
            C=dict(order=3, seeds=[91, 92, 93], local_features=[12, 24, 48], hidden=96,
                baseline_reused=True, caveat='same seed is not identical Q/R initialization after a capacity change'),
            D=dict(order=4, seeds=[91, 92, 93],
                size_control=[[36, 12, 4], [108, 36, 12]],
                depth_control=[[108, 36, 12], [108, 36, 12, 4]],
                caveat='existing cross-size sampler seed namespace is shared; not independent physical seeds')),
        resource_policy=dict(cpu_threads=2, report_cpu_threads=1, nice=10,
            gpu_seconds=gpu_hours*3600, cpu_seconds=cpu_hours*3600,
            directory_bytes=int(directory_gib*1024**3), gpu_peak_bytes=8*1024**3,
            minimum_host_available_gib=36, gpu_free_mib=12288,
            io_full_avg10_percent_max=20, load_per_cpu_max=1.25,
            two_gpu_checks_seconds_apart=10, profile_safety_factor=1.5,
            report_reserve_per_seed=1800, train_final_reserve_per_seed=120,
            max_training_seconds_per_arm=7200, resource_wait='no interference with any existing process'),
        data_quality=dict(equilibrium_certified=False, energy_ESS_min=8.30,
            energy_chains_ESS_below32=153, energy_chains_ESS_below16=20,
            critical_neighborhood_energy_ESS_min=33.83,
            critical_neighborhood_m2_ESS_min=39.81,
            MI_train_singleton_environment_fraction=.6356211780821491,
            MI_test_singleton_environment_fraction=.6400771025989619,
            caution='training-seed replication does not cover common sampling bias, finite size or empirical-MI bias'))
    root.mkdir(parents=True, exist_ok=False)
    (root / 'logs').mkdir()
    (root / 'configs').mkdir()
    for seed, cfg in arm_configs.items():
        save(root / 'configs' / f'seed{seed}.json', cfg, exclusive=True)
    save(root / 'configs/profile.json', profile, exclusive=True)
    save(root / 'plan.json', plan, exclusive=True)
    save(root / 'job.json', dict(status='prepared', workflow_complete=False, current=None,
        stages=[], error=None, requires_user_action=False, plan_sha256=sha(root / 'plan.json')),
        exclusive=True)
    return plan


def verify(root):
    root = Path(root).resolve()
    plan, job = read(root / 'plan.json'), read(root / 'job.json')
    if (socket.gethostname() != plan['host'] or plan['host'] != HOST
            or str(root) != plan['output'] or sha(root / 'plan.json') != job['plan_sha256']
            or 'CUDA_VISIBLE_DEVICES' in os.environ):
        raise RuntimeError('Frozen host/output/plan/GPU indexing mismatch')
    code = Path(plan['code_root'])
    if source_hashes(code) != plan['source_sha256']:
        raise RuntimeError('Frozen executable source differs; no automatic repair')
    for path, identity in plan['immutable_stat_identity'].items():
        if stat_identity(path) != identity:
            raise RuntimeError('Audited physical/MI stat identity changed')
    if (sha(Path(plan['configs']['91']['data']['cache']).with_suffix('.json')) != plan['sidecar_sha256']
            or sha(plan['mi_receipt']) != plan['mi_receipt_sha256']):
        raise RuntimeError('Data/MI public provenance changed')
    for seed, cfg in plan['configs'].items():
        if read(root / 'configs' / f'seed{seed}.json') != cfg:
            raise RuntimeError('Frozen formal configuration changed')
    if read(root / 'configs/profile.json') != plan['profile_config']:
        raise RuntimeError('Frozen profiling configuration changed')
    return plan, job


def directory_bytes(root):
    return sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file())


def cpu_seconds():
    return sum(x.ru_utime+x.ru_stime for x in
               (resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)))


def admission(root, plan, job, name, gpu):
    ready = 0
    while True:
        snapshot = resource_snapshot(plan)
        if snapshot['gpu_uuid'] != plan['gpu_uuid']:
            raise RuntimeError('Physical GPU UUID differs; no migration')
        ready = ready+1 if snapshot['gpu_ready' if gpu else 'host_ready'] else 0
        job.update(status='waiting_resources', current=name)
        job.setdefault('resource_checks', []).append(snapshot)
        job['resource_checks'] = job['resource_checks'][-60:]
        save(root / 'job.json', job)
        if ready >= (2 if gpu else 1):
            return
        time.sleep(10 if ready else 30)


def training_command(root, plan, seed):
    name = 'profile' if seed is None else f'seed{seed}'
    run = root / ('profile_run' if seed is None else f'seed{seed}/run')
    command = [sys.executable, '-u', '-m', MODULE, '--train-child', '--output', str(root),
        '--config', str(root / 'configs' / f'{name}.json'), '--run', str(run)]
    if seed is None:
        command += ['--profile']
    return command


def train_child(root, config, run, *, profile=False):
    """Normal trainer entry, with an exact cache-hit requirement only."""
    from experiments import rsmi_separate_qcr as trainer
    from recurrent_rg import mi_cache
    from scripts.report_ising_zh import chinese_figures
    from experiments.phi4_qcr import _device_receipt
    import torch
    root, config, run = Path(root).resolve(), Path(config).resolve(), Path(run).resolve()
    plan, job = verify(root)
    allowed = [(root / 'configs/profile.json', root / 'profile_run', True)] + [
        (root / f'configs/seed{seed}.json', root / f'seed{seed}/run', False) for seed in SEEDS]
    if (config, run, profile) not in allowed or run.exists() or run.is_symlink():
        raise ValueError('Child config/output/profile pair is not a fresh frozen stage')
    # Query the CUDA runtime ordinal, not a second nvidia-smi index lookup.
    torch.cuda.set_device(torch.device(plan['device']))
    device_receipt = _device_receipt(torch.device(plan['device']), plan['gpu_uuid'])
    Path(run).parent.mkdir(parents=True, exist_ok=True)
    save(Path(run).with_name(Path(run).name + '.device.json'), device_receipt, exclusive=True)
    original = mi_cache.load_or_build_mi_statistics
    def cache_only(directory, key_data, build, **kwargs):
        def forbidden():
            raise RuntimeError('Frozen b3 MI cache miss; rebuilding is forbidden')
        value, record = original(directory, key_data, forbidden, **kwargs)
        if record.get('hit') is not True or record.get('key') != plan['mi_keys'][key_data['split']]:
            raise RuntimeError('Wrong or newly built MI statistics')
        return value, record
    mi_cache.load_or_build_mi_statistics = cache_only
    setup_reserve = 120.
    if (Path(root) / 'profile_run/summary.json').is_file():
        setup_reserve += 2 * read(Path(root) / 'profile_run/summary.json')['timing']['setup_seconds']
    remaining = plan['resource_policy']['gpu_seconds'] - job.get('gpu_elapsed_seconds', 0.)
    clock = min(plan['resource_policy']['max_training_seconds_per_arm'], remaining-setup_reserve)
    if clock <= 0:
        raise RuntimeError('No remaining GPU time after setup/final-write reserve')
    args = ['--config', str(config), '--output', str(run), '--device', plan['device'],
        '--ising-b0-ablation', 'none', '--mi-cache-dir', plan['mi_cache_dir'],
        '--mi-test-cpu', '--skip-final-report', '--benchmark-warmup', '5',
        '--max-training-seconds', str(clock)]
    try:
        with chinese_figures():
            return trainer.main(args)
    finally:
        mi_cache.load_or_build_mi_statistics = original


def validate_training(root, plan, seed):
    run = root / ('profile_run' if seed is None else f'seed{seed}/run')
    summary, manifest = read(run / 'summary.json'), read(run / 'manifest.json')
    config = plan['profile_config'] if seed is None else plan['configs'][str(seed)]
    count = PROFILE_UPDATES if seed is None else UPDATES
    if (summary.get('config') != config or summary.get('actual_epochs') != count
            or summary.get('requested_epochs') != count or manifest.get('config') != config
            or manifest.get('labels_used_in_training') is not False
            or manifest.get('source') is not None or manifest.get('resumed_from') is not None
            or manifest.get('q_supervision') or manifest.get('frozen_C')
            or manifest.get('independent_c') is not True or manifest.get('estimator') != 'histogram'
            or manifest.get('ising_b0_ablation', {}).get('name') != 'none'
            or manifest.get('data_driven_initialization') or manifest.get('data_sha256') != DATA_SHA
            or summary.get('stopped_by_wall_limit') or summary.get('stopped_by_training_limit')
            or summary.get('stopped_by_train_plateau')
            or not (run / f'training_step_{count}.pt').is_file()
            or not manifest.get('training_start_module_sha256')):
        raise RuntimeError('Exact fresh label-free endpoint or provenance mismatch')
    records = manifest['execution_optimizations']['mi_statistics_cache']
    if any(not records[split]['hit'] or records[split]['key'] != key for split, key in MI_KEYS.items()):
        raise RuntimeError('MI cache-hit provenance mismatch')
    peak = summary['timing']['cuda_max_memory_reserved_bytes']
    if peak > plan['resource_policy']['gpu_peak_bytes']:
        raise RuntimeError('Own Torch GPU peak exceeded the frozen memory limit')
    return dict(actual_epochs=count, training_start_module_sha256=manifest['training_start_module_sha256'],
        manifest_sha256=sha(run / 'manifest.json'), summary_sha256=sha(run / 'summary.json'),
        cuda_max_memory_reserved_bytes=peak, timing=summary['timing'])


def profile_projection(summary, elapsed, plan):
    """Include setup, all 61 log events, checkpoints, and both CPU/GPU reserves."""
    timing, policy = summary['timing'], plan['resource_policy']
    steady = timing['steady_step_seconds']
    if not steady or any(not math.isfinite(v) or v <= 0 for v in steady):
        raise ValueError('Actual measured steady updates required')
    # Use the maximum of the 15 measured updates, not the most favorable mean.
    per_step = max(steady)
    log_events = UPDATES // 200 + 1
    projected_arm = policy['profile_safety_factor'] * (
        per_step * UPDATES + timing['setup_seconds']
        + timing['logging_seconds'] / 2 * log_events
        + timing['checkpoint_seconds'] * 14) + policy['train_final_reserve_per_seed']
    gpu = elapsed + len(SEEDS) * projected_arm
    cpu = gpu * policy['cpu_threads'] + len(SEEDS) * policy['report_reserve_per_seed']
    return dict(profile_is_scientific_result=False, steady_max_step_seconds=per_step,
        projected_seconds_per_training_arm=projected_arm,
        projected_total_gpu_seconds_conservative=gpu,
        projected_total_cpu_seconds_conservative=cpu,
        fixed_gpu_budget_seconds=policy['gpu_seconds'], fixed_cpu_budget_seconds=policy['cpu_seconds'],
        budget_passed=gpu <= policy['gpu_seconds'] and cpu <= policy['cpu_seconds']
            and projected_arm <= policy['max_training_seconds_per_arm'])


def execute(root):
    root = Path(root).resolve()
    plan, job = verify(root)
    if job['status'] != 'prepared' or job['stages']:
        raise RuntimeError('Only a newly prepared workflow may execute; never resume or retry')
    os.nice(10)
    job.update(parent_pid=os.getpid(), actual_nice=os.getpriority(os.PRIO_PROCESS, 0),
               started_at=now(), training={}, gpu_elapsed_seconds=0.)
    stages = [('training_profile', training_command(root, plan, None), True, None)]
    for seed in SEEDS:
        stages += [(f'seed{seed}_training', training_command(root, plan, seed), True, seed),
            (f'seed{seed}_report', [sys.executable, '-u', '-m', 'scripts.report_ising_b0_multiseed',
                '--run', str(root / f'seed{seed}/run'), '--output', str(root / f'seed{seed}/report'),
                '--threads', '1', '--sampling-repeats', '2'], False, seed)]
    stages += [('comparison', [sys.executable, '-u', '-m', 'scripts.report_ising_b0_multiseed',
        '--compare', *(f'{seed}={root}/seed{seed}/report' for seed in SEEDS),
        '--output', str(root / 'comparison')], False, None)]
    try:
        for stage_index, (name, command, gpu, seed) in enumerate(stages):
            verify(root)
            policy = plan['resource_policy']
            projection_path = root / 'profile_receipt.json'
            if projection_path.is_file():
                projected_arm = read(projection_path)['projected_seconds_per_training_arm']
                remaining_gpu_arms = sum(bool(stage[2]) for stage in stages[stage_index:])
                if job['gpu_elapsed_seconds'] + remaining_gpu_arms * projected_arm > policy['gpu_seconds']:
                    job.update(status='budget_hold', requires_user_action=True,
                        error='Actual elapsed GPU time plus remaining conservative projection exceeds the fixed budget')
                    break
            if (cpu_seconds() > policy['cpu_seconds'] or directory_bytes(root) > policy['directory_bytes']
                    or job['gpu_elapsed_seconds'] >= policy['gpu_seconds']):
                job.update(status='budget_hold', requires_user_action=True,
                           error='Fixed budget reached; outputs preserved, no reduced steps or retry')
                break
            admission(root, plan, job, name, gpu)
            started, cpu_before = time.perf_counter(), cpu_seconds()
            with (root / 'logs' / f'{name}.log').open('x') as stream:
                process = subprocess.Popen(command, cwd=plan['code_root'], stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT,
                    env=dict(os.environ, OMP_NUM_THREADS=str(2 if gpu else 1),
                             MKL_NUM_THREADS=str(2 if gpu else 1), PYTHONUNBUFFERED='1',
                             OPENBLAS_NUM_THREADS='1', NUMBA_NUM_THREADS='1',
                             CUDA_DEVICE_ORDER='PCI_BUS_ID'))
                row = dict(name=name, pid=process.pid, command=command, started_at=now(),
                           status='running', returncode=None, gpu_stage=gpu)
                job['stages'].append(row)
                job.update(status='running', current=name)
                save(root / 'job.json', job)
                returncode = process.wait()
            elapsed = time.perf_counter() - started
            row.update(status='complete' if returncode == 0 else 'failed', returncode=returncode,
                ended_at=now(), wall_seconds=elapsed, cpu_seconds=cpu_seconds()-cpu_before)
            if gpu:
                job['gpu_elapsed_seconds'] += elapsed
            job['cpu_seconds'] = cpu_seconds()
            if returncode == 75 and name.endswith('_report'):
                status = read(root / f'seed{seed}/report/status.json')
                if status.get('status') == 'analysis_budget_hold':
                    row['status'] = 'analysis_budget_hold'
                    job.update(status='analysis_budget_hold', requires_user_action=True,
                        error='Frozen report reached its bounded analysis time; no retry, training preserved')
                    break
            if returncode:
                raise RuntimeError(f'{name} returned {returncode}; preserve output without retry')
            verify(root)
            if gpu:
                run = root / ('profile_run' if seed is None else f'seed{seed}/run')
                result = read(run / 'summary.json')
                if result.get('stopped_by_training_limit') or result.get('stopped_by_wall_limit'):
                    job.update(status='budget_hold', requires_user_action=True,
                        error=f'{name} reached its fixed safety clock; partial progress preserved, no continuation')
                    break
            if name == 'training_profile':
                validate_training(root, plan, None)
                projection = profile_projection(read(root / 'profile_run/summary.json'), elapsed, plan)
                save(root / 'profile_receipt.json', projection, exclusive=True)
                if not projection['budget_passed']:
                    job.update(status='budget_hold', requires_user_action=True,
                        error='Measured conservative stage-A projection exceeds frozen budget; formal seeds not started')
                    break
            elif name.endswith('_training'):
                job['training'][str(seed)] = validate_training(root, plan, seed)
            elif name.endswith('_report') or name == 'comparison':
                folder = root / ('comparison' if name == 'comparison' else f'seed{seed}/report')
                report_receipt = read(folder / 'report.json')
                if report_receipt.get('status') != 'complete':
                    raise RuntimeError('Public report did not complete')
                if name.endswith('_report'):
                    metrics = read(folder / 'metrics.json')
                    if (metrics.get('encoding_protocol') != plan['protocol']['encoding_protocol']
                            or metrics.get('encoding_estimator') != plan['protocol']['encoding_protocol']['estimator']):
                        raise RuntimeError('Report encoding estimator differs from the frozen training-matched protocol')
        else:
            job.update(status='complete', workflow_complete=True, current=None, ended_at=now(),
                next_stage='B loss removals not automatically launched')
            job['public_files'] = public_files(root)
    except BaseException as error:
        job.update(status='failed', workflow_complete=False, requires_user_action=True,
                   error=f'{type(error).__name__}: {error}', ended_at=now())
        raise
    finally:
        save(root / 'job.json', job)


def public_files(root):
    """Only declared public receipts/figures/arrays; job is a separate dynamic receipt."""
    root = Path(root)
    candidates = [root / name for name in ('plan.json', 'launch.json', 'profile_receipt.json', 'profile_run.device.json')]
    candidates += [root / 'configs' / f'{name}.json' for name in ['profile', *(f'seed{s}' for s in SEEDS)]]
    for seed in SEEDS:
        candidates += [root / f'seed{seed}/run' / name for name in
                       ('manifest.json', 'summary.json', 'history_progress.json')]
        candidates.append(root / f'seed{seed}/run.device.json')
    for folder in [*(root / f'seed{seed}/report' for seed in SEEDS), root / 'comparison']:
        receipt = folder / 'report.json'
        if receipt.is_file():
            for name in [*read(receipt).get('public_files', []), 'report.json', 'status.json']:
                relative = Path(name)
                if (relative.is_absolute() or len(relative.parts) != 1
                        or relative.suffix not in ('.json', '.npz', '.png')):
                    raise ValueError('Report public file is outside its explicit allowed namespace')
                candidates.append(folder / relative)
    result = []
    for path in sorted(set(candidates)):
        if path.is_symlink():
            raise ValueError('Refuse a symlink in public artifacts')
        if path.is_file():
            result.append(dict(path=str(path.relative_to(root)), sha256=sha(path), bytes=path.stat().st_size))
    return result


def launch(root):
    root = Path(root).resolve()
    plan, job = verify(root)
    if job['status'] != 'prepared' or job['stages']:
        raise RuntimeError('Never launch an already started workflow')
    with (root / 'launch.json').open('x') as receipt, (root / 'logs/parent.log').open('x') as stream:
        process = subprocess.Popen([sys.executable, '-u', '-m', MODULE, '--execute', '--output', str(root)],
            cwd=plan['code_root'], stdin=subprocess.DEVNULL, stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True,
            env=dict(os.environ, PYTHONUNBUFFERED='1', CUDA_DEVICE_ORDER='PCI_BUS_ID'))
        value = dict(status='launched', at=now(), parent_pid=process.pid, output=str(root),
            plan_sha256=sha(root / 'plan.json'), device=plan['device'], gpu_uuid=plan['gpu_uuid'],
            fresh_experiment=True, resumed=False)
        receipt.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for flag in ('prepare', 'launch', 'execute', 'train-child'):
        mode.add_argument('--'+flag, action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--gpu-hours', type=float, default=8.)
    parser.add_argument('--cpu-hours', type=float, default=24.)
    parser.add_argument('--directory-gib', type=float, default=20.)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args(argv)
    if args.prepare:
        result = prepare(args.output, gpu=args.gpu, gpu_hours=args.gpu_hours,
                         cpu_hours=args.cpu_hours, directory_gib=args.directory_gib)
    elif args.launch:
        result = launch(args.output)
    elif args.execute:
        return execute(args.output)
    else:
        if args.config is None or args.run is None:
            parser.error('--train-child requires --config and --run')
        return train_child(args.output, args.config, args.run, profile=args.profile)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    main()
