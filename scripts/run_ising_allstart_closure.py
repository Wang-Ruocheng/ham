"""冻结的 b=2、四尺度、单 seed 全起点闭环消融；串行执行，不续训或重试。"""
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
GPU_UUID = 'GPU-5bba87f1-7d32-4dde-fe28-caba4468ab42'
MODULE = 'scripts.run_ising_allstart_closure'
ARMS = ('control', 'anchor')
UPDATES = 12000
PROFILE_UPDATES = 20


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
    return dict(size=stat.st_size, inode=stat.st_ino, mtime_ns=stat.st_mtime_ns,
                device=stat.st_dev)


def configs(data):
    """Current recipe, explicit authorized block-two/depth-three loss ablation."""
    from recurrent_rg.config import RunConfig
    from experiments.rsmi_separate_qcr import build_parser, validate_recipe
    raw = dict(schema_version=5,
        data=dict(cache=str(Path(data).resolve()), physical_model='ising2d',
                  coupling_names=['K'], spin_flip_augment=False),
        model=dict(dimension=2, theory_dim=5, hidden=96, width=3,
                   local_features=12, block_factor=2, affine_reconstruction=True,
                   kernel_layout='nonoverlapping', kernel_sampling='bernoulli',
                   kernel_initialization='random', fixed_kernel=False),
        loss=dict(kernel_noncollapse_mode='variance_floor_ensemble',
                  kernel_variance_ratio=1., kernel_sign_weight=1.,
                  closure_scale_weights=None),
        training=dict(max_rg_steps=3, epochs=UPDATES, system_batch=16,
            sample_batch=128, learning_rate=.001, seed=91, threads=2,
            checkpoint_every=1000, log_every=200, gradient_mode='qc_r_separate',
            train_modules=['Q', 'C', 'R'], lr_schedule='multistep',
            lr_milestones=[6000, 9000], lr_gamma=.3),
        evaluation=dict(max_rg_steps=3, samples=128, systems=24))
    result = {}
    for arm, mode in zip(ARMS, ('one_step_and_rollout', 'all_start_anchor')):
        item = deepcopy(raw)
        item['loss']['closure_objective'] = mode
        cfg = RunConfig.from_dict(item)
        cfg.validate()
        args = build_parser().parse_args(['--config', 'unused', '--output', 'unused',
                                         '--all-start-closure-ablation'])
        validate_recipe(args, cfg)
        # JSON-canonical lists, not dataclass tuples, in the immutable contract.
        result[arm] = json.loads(json.dumps(cfg.to_dict()))
    return result


def physical_contract(config):
    """Read labels only to validate the fixed data split, never for optimization."""
    import torch
    from recurrent_rg.config import RunConfig
    from recurrent_rg.runner import prepared_data
    torch.set_num_threads(2)
    train, test = prepared_data(RunConfig.from_dict(config))
    expected = {'train': [129, 128, 32, 32, 1], 'test': [63, 128, 32, 32, 1]}
    labels = {}
    for split, dataset in (('train', train), ('test', test)):
        if list(dataset.configurations.shape) != expected[split] or dataset.configurations.dtype != torch.float64:
            raise ValueError('Frozen physical input shape/dtype differs from the declared cache')
        values, counts = torch.unique(dataset.couplings.reshape(-1), return_counts=True)
        if not bool(torch.isfinite(values).all()) or not bool((values > 0).all()) or not bool((counts == 3).all()):
            raise ValueError('Expected three independent physical ensembles per positive K')
        labels[split] = values.tolist()
    if len(labels['train']) != 43 or len(labels['test']) != 21 or set(labels['train']) & set(labels['test']):
        raise ValueError('Expected disjoint 43 TRAIN /21 TEST K values')
    return dict(train_shape=expected['train'], test_shape=expected['test'],
        train_K_values=labels['train'], test_K_values=labels['test'],
        K_splits_disjoint=True, dtype='torch.float64', physical_replicas_per_K=3,
        labels_read_only_for_input_validation=True, labels_used_in_training=False)


def prepare(root, *, data, sidecar, data_sha256, sidecar_sha256,
            gpu_hours=2., cpu_hours=4., directory_gib=10.):
    preparation_cpu_started = cpu_seconds()
    if socket.gethostname() != HOST or 'CUDA_VISIBLE_DEVICES' in os.environ:
        raise ValueError('Requires declared host and unmasked physical GPU indices')
    if any(isinstance(v, bool) or not math.isfinite(v) or v <= 0
           for v in (gpu_hours, cpu_hours, directory_gib)):
        raise ValueError('Finite positive budgets required')
    root, data, sidecar = map(lambda x: Path(x).resolve(), (root, data, sidecar))
    if root.exists() or root.is_symlink():
        raise FileExistsError('No overwrite, resume, retry, or duplicate preparation')
    identities = {str(path): stat_identity(path) for path in (data, sidecar)}
    if sha(data) != data_sha256 or sha(sidecar) != sidecar_sha256:
        raise ValueError('Physical data/sidecar fingerprint mismatch')
    metadata = read(sidecar)
    if (metadata.get('sites') != 32 or metadata.get('seed') != 91
            or len(metadata.get('couplings', [])) != 64
            or metadata.get('sampler') != 'Wolff fixed cluster count, numba v2'):
        raise ValueError('Expected the declared historical L32, 64K, physical seed91 cache')
    arm_configs = configs(data)
    physical = physical_contract(arm_configs['control'])
    profiles = deepcopy(arm_configs)
    for cfg in profiles.values():
        cfg['training'].update(seed=191, epochs=PROFILE_UPDATES,
            checkpoint_every=PROFILE_UPDATES, log_every=PROFILE_UPDATES)
    code = Path(__file__).resolve().parents[1]
    plan = dict(schema_version=5, experiment='ising2d_B0_all_start_anchor_paired',
        prepared_at=now(), host=HOST, code_root=str(code), output=str(root),
        device='cuda:0', gpu_uuid=GPU_UUID, source_sha256=source_hashes(code),
        data=str(data), sidecar=str(sidecar), data_sha256=data_sha256,
        sidecar_sha256=sidecar_sha256, immutable_stat_identity=identities,
        physical_input_contract=physical,
        configs=arm_configs, profile_configs=profiles,
        mi_cache_dir=str(root / 'mi_cache'),
        protocol=dict(seed=91, profile_seed=191, physical_seed=91,
            scales=[32, 16, 8, 4], block_factor=2, latent_dimension=5,
            data_train_shape=[129, 128, 32, 32, 1], data_test_shape=[63, 128, 32, 32, 1],
            train_K_values=43, test_K_values=21, physical_replicas_per_K=3,
            train_test_K_disjoint=True, train_dtype='torch.float64',
            Q_local_features=12, hidden=96, fixed_updates=UPDATES,
            labels_used_in_training=False, fresh_QCR=True, source_checkpoint=None,
            matched_initialization=True, independent_gradients=True,
            closure_target_scale_weights='uniform',
            control='5 mean(one_step) +5 mean(microscopic_start_rollout)',
            anchor='10/3 sum_(t=1..3) mean_(s=0..t-1) ||R^(t-s)(g_s)-g_t||^2',
            prediction_prediction_penalty=False, future_loss=False,
            C='-2 empirical full-shell histogram MI +2 per-ensemble variance floor +1 color NLL',
            Q='shared affine reconstruction10; microscopic/coarse subset20/5',
            optimizer='Fresh Adam .001; multistep6000/9000 gamma.3',
            scientific_limit='Historical finite chains and final4x4; one training seed, not thermal exponent certification',
            thermal_nu=None, automatic_retry=False, resume=False),
        resource_policy=dict(cpu_threads=2, report_cpu_threads=1, nice=10,
            gpu_seconds=float(gpu_hours)*3600, cpu_seconds=float(cpu_hours)*3600,
            directory_bytes=int(directory_gib*1024**3), gpu_peak_bytes=8*1024**3,
            minimum_host_available_gib=36, gpu_free_mib=12288,
            io_full_avg10_percent_max=20, load_per_cpu_max=1.25,
            two_gpu_checks_seconds_apart=10, profile_safety_factor=1.5,
            report_reserve_seconds=1200, train_final_reserve_seconds=120,
            max_training_seconds_per_arm=7200,
            resource_wait='Wait without interfering with any existing process'))
    root.mkdir(parents=True, exist_ok=False)
    (root / 'logs').mkdir()
    (root / 'configs').mkdir()
    for arm in ARMS:
        save(root / f'configs/{arm}.json', arm_configs[arm], exclusive=True)
        save(root / f'configs/profile_{arm}.json', profiles[arm], exclusive=True)
    save(root / 'plan.json', plan, exclusive=True)
    save(root / 'job.json', dict(status='prepared', workflow_complete=False,
        current=None, stages=[], error=None, requires_user_action=False,
        preparation_cpu_seconds=cpu_seconds()-preparation_cpu_started,
        plan_sha256=sha(root / 'plan.json')), exclusive=True)
    return plan


def verify(root):
    root = Path(root).resolve()
    plan, job = read(root / 'plan.json'), read(root / 'job.json')
    if (socket.gethostname() != HOST or plan['host'] != HOST
            or str(root) != plan['output'] or plan['device'] != 'cuda:0'
            or plan['gpu_uuid'] != GPU_UUID or 'CUDA_VISIBLE_DEVICES' in os.environ
            or sha(root / 'plan.json') != job['plan_sha256']):
        raise RuntimeError('Frozen host/output/plan/GPU-index mismatch')
    if source_hashes(Path(plan['code_root'])) != plan['source_sha256']:
        raise RuntimeError('Frozen Python source differs; no automatic repair')
    for path, identity in plan['immutable_stat_identity'].items():
        if stat_identity(path) != identity:
            raise RuntimeError('Frozen physical data stat identity differs')
    if sha(plan['sidecar']) != plan['sidecar_sha256']:
        raise RuntimeError('Physical sidecar differs')
    for arm in ARMS:
        if (read(root / f'configs/{arm}.json') != plan['configs'][arm]
                or read(root / f'configs/profile_{arm}.json') != plan['profile_configs'][arm]):
            raise RuntimeError('Frozen formal/profile configuration differs')
    if job.get('mi_preprocessing_sha256'):
        if sha(root / 'mi_preprocessing.json') != job['mi_preprocessing_sha256']:
            raise RuntimeError('Frozen MI preprocessing public receipt differs')
        for path, identity in job['mi_stat_identity'].items():
            if stat_identity(path) != identity:
                raise RuntimeError('Frozen data-only MI cache stat identity differs')
    return plan, job


def cpu_seconds():
    return sum(row.ru_utime + row.ru_stime for row in (
        resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)))


def directory_bytes(root):
    return sum(path.stat().st_size for path in Path(root).rglob('*') if path.is_file())


def admission(root, plan, job, name, gpu):
    ready = 0
    while True:
        snapshot = resource_snapshot(plan)
        if snapshot['gpu_uuid'] != plan['gpu_uuid']:
            raise RuntimeError('Physical GPU UUID differs; no migration')
        ready = ready + 1 if snapshot['gpu_ready' if gpu else 'host_ready'] else 0
        job.update(status='waiting_resources', current=name)
        job.setdefault('resource_checks', []).append(snapshot)
        job['resource_checks'] = job['resource_checks'][-60:]
        save(root / 'job.json', job)
        if ready >= (2 if gpu else 1):
            return
        time.sleep(10 if ready else 30)


def training_command(root, arm, *, profile=False):
    name = f'profile_{arm}' if profile else arm
    return [sys.executable, '-u', '-m', MODULE, '--train-child', '--output', str(root),
            '--arm', arm, *(['--profile'] if profile else [])]


def mi_receipt(root, plan):
    receipt = read(root / 'mi_preprocessing.json')
    if (receipt.get('status') != 'complete' or receipt.get('model_initialized') is not False
            or receipt.get('optimizer_updates') != 0 or receipt.get('sampling_draws') != 0
            or receipt.get('data_sha256') != plan['data_sha256']):
        raise RuntimeError('MI preprocessing is not a complete model-free data-only stage')
    for split in ('train', 'test'):
        record = receipt['splits'][split]
        if record['data_shape'] != plan['protocol'][f'data_{split}_shape']:
            raise RuntimeError('Physical cache shape does not match L32x128 contract')
        if Path(record['cache']['path']).resolve().parent != Path(plan['mi_cache_dir']):
            raise RuntimeError('MI cache outside experiment-specific directory')
    return receipt


def train_child(root, arm, *, profile=False):
    from experiments import rsmi_separate_qcr as trainer
    from experiments.phi4_qcr import _device_receipt
    from recurrent_rg import mi_cache
    from scripts.report_ising_zh import chinese_figures
    import torch
    root = Path(root).resolve()
    plan, job = verify(root)
    if arm not in ARMS:
        raise ValueError('Unknown frozen arm')
    name = f'profile_{arm}' if profile else arm
    config, run = root / f'configs/{name}.json', root / name / 'run'
    if run.exists() or run.is_symlink():
        raise FileExistsError('Never overwrite, resume, or retry a training child')
    receipt = mi_receipt(root, plan)
    torch.cuda.set_device(torch.device(plan['device']))
    device_receipt = _device_receipt(torch.device(plan['device']), plan['gpu_uuid'])
    run.parent.mkdir(parents=True, exist_ok=True)
    save(run.with_name('run.device.json'), device_receipt, exclusive=True)
    original = mi_cache.load_or_build_mi_statistics
    def cache_only(directory, key_data, build, **kwargs):
        def forbidden():
            raise RuntimeError('Frozen b2 MI cache miss; rebuild forbidden during training')
        value, record = original(directory, key_data, forbidden, **kwargs)
        expected = receipt['splits'][key_data['split']]['cache']['key']
        if record.get('hit') is not True or record.get('key') != expected:
            raise RuntimeError('MI cache-hit fingerprint differs')
        return value, record
    mi_cache.load_or_build_mi_statistics = cache_only
    setup_reserve = 120.
    if (root / f'profile_{arm}/run/summary.json').is_file():
        setup_reserve += 2*read(root / f'profile_{arm}/run/summary.json')['timing']['setup_seconds']
    remaining = plan['resource_policy']['gpu_seconds'] - job.get('gpu_elapsed_seconds', 0.)
    clock = min(300. if profile else plan['resource_policy']['max_training_seconds_per_arm'],
                remaining - setup_reserve)
    if clock <= 0:
        raise RuntimeError('No GPU budget remains after setup/final-write reserve')
    argv = ['--config', str(config), '--output', str(run), '--device', plan['device'],
        '--all-start-closure-ablation', '--mi-cache-dir', plan['mi_cache_dir'],
        '--mi-test-cpu', '--skip-final-report', '--benchmark-warmup', '5',
        '--max-training-seconds', str(clock)]
    try:
        with chinese_figures():
            return trainer.main(argv)
    finally:
        mi_cache.load_or_build_mi_statistics = original


def validate_training(root, plan, arm, *, profile=False):
    name = f'profile_{arm}' if profile else arm
    run = root / name / 'run'
    summary, manifest = read(run / 'summary.json'), read(run / 'manifest.json')
    config = plan['profile_configs' if profile else 'configs'][arm]
    count = PROFILE_UPDATES if profile else UPDATES
    if (summary.get('config') != config or summary.get('actual_epochs') != count
            or summary.get('requested_epochs') != count or manifest.get('config') != config
            or manifest.get('labels_used_in_training') is not False
            or manifest.get('source') is not None or manifest.get('resumed_from') is not None
            or manifest.get('q_supervision') or manifest.get('frozen_C')
            or manifest.get('independent_c') is not True
            or manifest.get('estimator') != 'histogram'
            or manifest.get('data_driven_initialization')
            or manifest.get('data_sha256') != plan['data_sha256']
            or manifest.get('closure_ablation', {}).get('objective') != config['loss']['closure_objective']
            or summary.get('stopped_by_wall_limit') or summary.get('stopped_by_training_limit')
            or summary.get('stopped_by_train_plateau')
            or not (run / f'training_step_{count}.pt').is_file()
            or not manifest.get('training_start_module_sha256')):
        raise RuntimeError('Fresh label-free exact endpoint/provenance mismatch')
    expected = mi_receipt(root, plan)
    records = manifest['execution_optimizations']['mi_statistics_cache']
    for split in ('train', 'test'):
        if not records[split]['hit'] or records[split]['key'] != expected['splits'][split]['cache']['key']:
            raise RuntimeError('MI statistics provenance differs')
    peak = summary['timing']['cuda_max_memory_reserved_bytes']
    if peak > plan['resource_policy']['gpu_peak_bytes']:
        raise RuntimeError('Own GPU peak exceeded frozen memory budget')
    return dict(actual_epochs=count, training_start_module_sha256=manifest['training_start_module_sha256'],
        manifest_sha256=sha(run / 'manifest.json'), summary_sha256=sha(run / 'summary.json'),
        cuda_max_memory_reserved_bytes=peak, timing=summary['timing'])


def profile_projection(summaries, elapsed_gpu, elapsed_cpu, plan):
    """Separate one-off setup, measured steady updates, logging, and writes."""
    policy, arms = plan['resource_policy'], {}
    for arm in ARMS:
        timing = summaries[arm]['timing']
        steady = timing['steady_step_seconds']
        if not steady or any(not math.isfinite(v) or v <= 0 for v in steady):
            raise ValueError('Measured positive steady update times required')
        maximum = max(steady)
        projected = policy['profile_safety_factor']*(
            maximum*UPDATES + timing['setup_seconds']
            + timing['logging_seconds']/2*(UPDATES//200 + 1)
            + timing['checkpoint_seconds']*14) + policy['train_final_reserve_seconds']
        arms[arm] = dict(steady_max_step_seconds=maximum,
            projected_gpu_seconds=projected, profile_setup_seconds=timing['setup_seconds'],
            logging_seconds=timing['logging_seconds'], checkpoint_seconds=timing['checkpoint_seconds'])
    gpu = elapsed_gpu + sum(item['projected_gpu_seconds'] for item in arms.values())
    cpu = elapsed_cpu + policy['cpu_threads']*sum(item['projected_gpu_seconds'] for item in arms.values())
    cpu += policy['report_reserve_seconds']
    return dict(profile_is_scientific_result=False, arms=arms,
        projected_total_gpu_seconds_conservative=gpu,
        projected_total_cpu_seconds_conservative=cpu,
        actual_gpu_seconds_including_profiles=elapsed_gpu,
        actual_cpu_seconds_including_MI_and_profiles=elapsed_cpu,
        fixed_gpu_budget_seconds=policy['gpu_seconds'], fixed_cpu_budget_seconds=policy['cpu_seconds'],
        budget_passed=gpu <= policy['gpu_seconds'] and cpu <= policy['cpu_seconds']
            and all(item['projected_gpu_seconds'] <= policy['max_training_seconds_per_arm'] for item in arms.values()))


def public_files(root):
    """Explicit public receipts/plots/arrays only; never PT, MI, or checkpoints."""
    root = Path(root)
    candidates = [root / name for name in ('plan.json', 'launch.json', 'profile_receipt.json', 'mi_preprocessing.json')]
    for name in (*ARMS, *(f'profile_{a}' for a in ARMS)):
        candidates += [root / f'configs/{name}.json', root / name / 'run.device.json']
        candidates += [root / name / 'run' / file for file in ('manifest.json', 'summary.json', 'history_progress.json')]
    report = root / 'comparison/report.json'
    if report.is_file():
        for name in (*read(report).get('public_files', []), 'report.json', 'status.json'):
            relative = Path(name)
            if (relative.is_absolute() or len(relative.parts) != 1
                    or relative.suffix not in ('.json', '.png', '.npz')):
                raise ValueError('Report public artifact escapes its explicit namespace')
            candidates.append(report.parent / relative)
    result = []
    for path in sorted(set(candidates)):
        if path.is_symlink():
            raise ValueError('Symlink forbidden in public artifacts')
        if path.is_file():
            result.append(dict(path=str(path.relative_to(root)), sha256=sha(path), bytes=path.stat().st_size))
    return result


def execute(root):
    root = Path(root).resolve()
    plan, job = verify(root)
    if job['status'] != 'prepared' or job['stages']:
        raise RuntimeError('Only a newly prepared workflow executes; never resume or retry')
    # Exactly once: children inherit this nice value; neither launch nor children renice.
    os.nice(10)
    preparation_cpu = job.get('preparation_cpu_seconds', 0.)
    job.update(parent_pid=os.getpid(), actual_nice=os.getpriority(os.PRIO_PROCESS, 0),
        started_at=now(), training={}, gpu_elapsed_seconds=0., cpu_seconds=cpu_seconds())
    stages = [('mi_preprocessing', [sys.executable, '-u', '-m', 'scripts.prepare_qcr_mi_cache',
        '--config', str(root / 'configs/control.json'), '--output', str(root / 'mi_preprocessing.json'),
        '--cache-dir', plan['mi_cache_dir']], False, None, False)]
    stages += [(f'profile_{arm}', training_command(root, arm, profile=True), True, arm, True) for arm in ARMS]
    stages += [(f'{arm}_training', training_command(root, arm), True, arm, False) for arm in ARMS]
    stages += [('comparison_report', [sys.executable, '-u', '-m', 'scripts.report_ising_allstart_closure',
        '--run-control', str(root / 'control/run'), '--run-anchor', str(root / 'anchor/run'),
        '--output', str(root / 'comparison'), '--threads', '1', '--max-seconds',
        str(plan['resource_policy']['report_reserve_seconds'])], False, None, False)]
    try:
        for index, (name, command, gpu, arm, profile) in enumerate(stages):
            verify(root)
            policy = plan['resource_policy']
            if (root / 'profile_receipt.json').is_file():
                projection = read(root / 'profile_receipt.json')
                remaining = sum(projection['arms'][s[3]]['projected_gpu_seconds']
                    for s in stages[index:] if s[2] and not s[4])
                projected_cpu = preparation_cpu + cpu_seconds() + policy['cpu_threads']*remaining
                if name != 'comparison_report':
                    projected_cpu += policy['report_reserve_seconds']
                if (job['gpu_elapsed_seconds'] + remaining > policy['gpu_seconds']
                        or projected_cpu > policy['cpu_seconds']):
                    job.update(status='budget_hold', requires_user_action=True,
                        error='Actual elapsed usage plus remaining conservative projection exceeds frozen budget')
                    break
            if (preparation_cpu + cpu_seconds() > policy['cpu_seconds'] or directory_bytes(root) > policy['directory_bytes']
                    or job['gpu_elapsed_seconds'] >= policy['gpu_seconds']):
                job.update(status='budget_hold', requires_user_action=True,
                    error='Frozen budget reached; no reduced steps, continuation, or retry')
                break
            admission(root, plan, job, name, gpu)
            started, cpu_before = time.perf_counter(), cpu_seconds()
            with (root / f'logs/{name}.log').open('x') as stream:
                process = subprocess.Popen(command, cwd=plan['code_root'], stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT,
                    env=dict(os.environ, OMP_NUM_THREADS='2' if gpu or name == 'mi_preprocessing' else '1',
                        MKL_NUM_THREADS='2' if gpu or name == 'mi_preprocessing' else '1',
                        OPENBLAS_NUM_THREADS='1', NUMBA_NUM_THREADS='1', PYTHONUNBUFFERED='1',
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
            job['cpu_seconds'] = preparation_cpu + cpu_seconds()
            if returncode == 75 and name == 'comparison_report':
                row['status'] = 'analysis_budget_hold'
                job.update(status='analysis_budget_hold', requires_user_action=True,
                    error='Bounded analysis clock reached; training preserved, report not retried')
                break
            if returncode:
                raise RuntimeError(f'{name} returned {returncode}; preserve outputs without retry')
            verify(root)
            if name == 'mi_preprocessing':
                receipt = mi_receipt(root, plan)
                job['mi_preprocessing_sha256'] = sha(root / 'mi_preprocessing.json')
                job['mi_keys'] = {k: v['cache']['key'] for k, v in receipt['splits'].items()}
                job['mi_stat_identity'] = {v['cache']['path']: stat_identity(v['cache']['path'])
                    for v in receipt['splits'].values()}
            elif gpu:
                summary = read(root / (f'profile_{arm}' if profile else arm) / 'run/summary.json')
                if summary.get('stopped_by_training_limit') or summary.get('stopped_by_wall_limit'):
                    job.update(status='budget_hold', requires_user_action=True,
                        error=f'{name} reached fixed safety clock; no continuation')
                    break
                result = validate_training(root, plan, arm, profile=profile)
                job['training'][name] = result
                if arm == 'anchor':
                    other_name = 'profile_control' if profile else 'control_training'
                    if result['training_start_module_sha256'] != job['training'][other_name]['training_start_module_sha256']:
                        raise RuntimeError('Paired arms did not have identical fresh Q/C/R initialization')
                if name == 'profile_anchor':
                    summaries = {a: read(root / f'profile_{a}/run/summary.json') for a in ARMS}
                    projection = profile_projection(summaries, job['gpu_elapsed_seconds'],
                        preparation_cpu + cpu_seconds(), plan)
                    save(root / 'profile_receipt.json', projection, exclusive=True)
                    if not projection['budget_passed']:
                        job.update(status='budget_hold', requires_user_action=True,
                            error='Measured conservative paired projection exceeds frozen budget; formal training not started')
                        break
            elif name == 'comparison_report':
                if read(root / 'comparison/report.json').get('status') != 'complete':
                    raise RuntimeError('Formal comparison report did not complete')
            save(root / 'job.json', job)
        else:
            job.update(status='complete', workflow_complete=True, current=None, ended_at=now())
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
        raise RuntimeError('Never launch an already started workflow')
    with (root / 'launch.json').open('x') as receipt, (root / 'logs/parent.log').open('x') as stream:
        process = subprocess.Popen([sys.executable, '-u', '-m', MODULE, '--execute', '--output', str(root)],
            cwd=plan['code_root'], stdin=subprocess.DEVNULL, stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True,
            env=dict(os.environ, PYTHONUNBUFFERED='1', CUDA_DEVICE_ORDER='PCI_BUS_ID'))
        value = dict(status='launched', at=now(), parent_pid=process.pid,
            output=str(root), plan_sha256=sha(root / 'plan.json'), device=plan['device'],
            gpu_uuid=plan['gpu_uuid'], fresh_experiment=True, resumed=False)
        receipt.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for flag in ('prepare', 'launch', 'execute', 'train-child'):
        mode.add_argument('--' + flag, action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--input-data', type=Path)
    parser.add_argument('--input-sidecar', type=Path)
    parser.add_argument('--data-sha256')
    parser.add_argument('--sidecar-sha256')
    parser.add_argument('--gpu-hours', type=float, default=2.)
    parser.add_argument('--cpu-hours', type=float, default=4.)
    parser.add_argument('--directory-gib', type=float, default=10.)
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args(argv)
    if args.prepare:
        if any(getattr(args, key) is None for key in ('input_data', 'input_sidecar', 'data_sha256', 'sidecar_sha256')):
            parser.error('--prepare requires input data/sidecar and their SHA256 values')
        result = prepare(args.output, data=args.input_data, sidecar=args.input_sidecar,
            data_sha256=args.data_sha256, sidecar_sha256=args.sidecar_sha256,
            gpu_hours=args.gpu_hours, cpu_hours=args.cpu_hours, directory_gib=args.directory_gib)
    elif args.launch:
        result = launch(args.output)
    elif args.execute:
        return execute(args.output)
    else:
        if args.arm is None:
            parser.error('--train-child requires --arm')
        return train_child(args.output, args.arm, profile=args.profile)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    main()
