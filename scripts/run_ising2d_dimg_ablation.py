"""完全无监督 2×2 QCR 的 dimg=1/2/3 同预算消融；独占输出、不重试。

只复制已完成无监督控制的配置，不加载其权重。三臂各自随机初始化，
独立 GPU 可并行训练；CPU 正式报告和物理响应诊断全局串行。
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
import subprocess
import sys
import time
import traceback

from recurrent_rg.config import RunConfig
from recurrent_rg.learning_rate import LearningRatePolicy
from scripts.run_ising2d_block2_latest import DATA_SHA, SIDECAR_SHA, differences, gpu_snapshot


UPDATES = 12000
MI_KEYS = dict(train='53b4cb45bda6c9271a4d18348e9f20cf75bbcf9f63647e76e5ca401dc19f5bea',
               test='85992c4ab2ce55ab36a0c45842d3a6e9c695bdd418c4cc3688117d6a60ebd8b6')
SOURCE_FILES = ('manifest.json', 'summary.json', '../job.json', '../mi_preprocessing.json')
CODE_FILES = ('scripts/run_ising2d_dimg_ablation.py', 'scripts/report_ising_dimg_ablation.py',
    'scripts/diagnose_ising_dimg_response.py', 'scripts/diagnose_ising_mcrg.py',
    'scripts/report_ising_mcrg.py', 'scripts/report_ising_zh.py',
    'scripts/run_ising2d_block2_latest.py', 'experiments/rsmi_separate_qcr.py',
    'recurrent_rg/config.py', 'recurrent_rg/checkpoints.py', 'recurrent_rg/model_nd.py',
    'recurrent_rg/losses.py', 'recurrent_rg/training.py', 'recurrent_rg/learning_rate.py',
    'recurrent_rg/mi_cache.py', 'recurrent_rg/execution.py')


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1 << 20), b''):
            digest.update(part)
    return digest.hexdigest()


def save_json(path, value, *, exclusive=False):
    path = Path(path)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if exclusive:
        with path.open('x') as stream:
            stream.write(text)
    else:
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(text)
        temporary.replace(path)


def regular(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'需要现有普通文件，拒绝符号链接：{path}')
    return path


def stat_identity(path):
    row = regular(path).stat()
    return dict(size=row.st_size, mtime_ns=row.st_mtime_ns, inode=row.st_ino, device=row.st_dev)


def validate_source_config(manifest):
    cfg = deepcopy(manifest['config'])
    t, m, d = cfg['training'], cfg['model'], cfg['data']
    if (manifest.get('data_sha256') != DATA_SHA or manifest.get('labels_used_in_training') is not False
            or manifest.get('q_supervision') or manifest.get('source') is not None
            or manifest.get('resumed_from') is not None or manifest.get('frozen_C')
            or manifest.get('data_driven_initialization') or cfg.get('schema_version') != 5
            or m['dimension'] != 2 or m['theory_dim'] != 1 or m['block_factor'] != 2
            or m['hidden'] != 96 or m['width'] != 3 or m['fixed_kernel']
            or m['kernel_initialization'] != 'random' or m['kernel_sampling'] != 'bernoulli'
            or m['kernel_layout'] != 'nonoverlapping' or not m['affine_reconstruction']
            or t['max_rg_steps'] != 2 or cfg['evaluation']['max_rg_steps'] != 2
            or t['epochs'] != UPDATES or t['seed'] != 91 or t['threads'] != 2
            or t['gradient_mode'] != 'qc_r_separate' or t['train_modules'] != ['Q', 'C', 'R']
            or t['learning_rate'] != .001 or t['lr_schedule'] != 'multistep'
            or t['lr_milestones'] != [6000, 9000] or t['lr_gamma'] != .3
            or t['system_batch'] != 16 or t['sample_batch'] != 128
            or d['train_pool_replicas'] != 1 or d['test_pool_replicas'] != 1):
        raise ValueError('源必须是完整无监督、最新配方、2×2 深度二 seed91 的12000步控制')
    from experiments.rsmi_separate_qcr import build_parser, validate_recipe
    parsed = RunConfig.from_dict(cfg)
    parsed.validate()
    validate_recipe(build_parser().parse_args(['--config', 'unused', '--output', 'unused']), parsed)
    return cfg


def build_plan(source_run, output, gpus=(0, 2, 3)):
    """只读小型 JSON 和元数据；不加载权重、物理数据、MI 或编码。"""
    source, output = Path(source_run).resolve(), Path(output).resolve()
    if len(gpus) != 3 or len(set(gpus)) != 3 or any(type(g) is not int or g < 0 for g in gpus):
        raise ValueError('三臂需要三个互不重复的非负 GPU 索引')
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('新输出必须与历史源目录分离')
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'拒绝覆盖、续跑或重复启动：{output}')
    manifest = json.loads(regular(source / 'manifest.json').read_text())
    summary = json.loads(regular(source / 'summary.json').read_text())
    job = json.loads(regular(source.parent / 'job.json').read_text())
    prep_path = regular(source.parent / 'mi_preprocessing.json')
    prep = json.loads(prep_path.read_text())
    base = validate_source_config(manifest)
    if (summary.get('config') != base or type(summary.get('actual_epochs')) is not int
            or summary['actual_epochs'] != UPDATES or summary.get('requested_epochs') != UPDATES
            or summary.get('stopped_by_training_limit') or summary.get('stopped_by_wall_limit')
            or job.get('status') != 'complete' or job.get('workflow_complete') is not True
            or job.get('training', {}).get('actual_epochs') != UPDATES
            or job['training'].get('manifest_sha256') != fingerprint(source / 'manifest.json')
            or prep.get('status') != 'complete' or prep.get('data_sha256') != DATA_SHA
            or prep.get('model_initialized') is not False or prep.get('optimizer_updates') != 0):
        raise ValueError('源实验或其静态MI预处理没有正式完整完成，或回执不匹配')
    mi_dir = source.parent / 'mi_cache'
    records = manifest.get('execution_optimizations', {}).get('mi_statistics_cache', {})
    for split, key in MI_KEYS.items():
        expected = mi_dir / f'mi-v1-{key}.pt'
        record = records.get(split, {})
        old = prep.get('splits', {}).get(split, {}).get('cache', {})
        if (record.get('hit') is not True or record.get('key') != key or old.get('key') != key
                or record.get('cache_version') != 1 or old.get('cache_version') != 1
                or Path(record.get('path', '')).resolve() != expected
                or Path(old.get('path', '')).resolve() != expected):
            raise ValueError('源2×2 MI缓存身份或命中回执不匹配')
        regular(expected)
    cache = regular(base['data']['cache'])
    sidecar = json.loads(regular(cache.with_suffix('.json')).read_text())
    if sidecar.get('data_sha256') != DATA_SHA:
        raise ValueError('历史物理数据侧车声明不匹配；不重采样或审计')
    source_sha = {str((source / name).resolve()): fingerprint(source / name) for name in SOURCE_FILES}
    if fingerprint(cache.with_suffix('.json')) != SIDECAR_SHA:
        raise ValueError('历史采样侧车 SHA 不匹配')
    immutable_stats = {str(path): stat_identity(path) for path in
                       (cache, *(mi_dir / f'mi-v1-{key}.pt' for key in MI_KEYS.values()))}
    arms = []
    for dim, gpu in zip((1, 2, 3), gpus):
        cfg = deepcopy(base)
        cfg['model']['theory_dim'] = dim
        changed = differences(base, cfg)
        if changed != ([] if dim == 1 else ['model.theory_dim']):
            raise ValueError('只允许 model.theory_dim 改变')
        RunConfig.from_dict(cfg).validate()
        root, run = output / f'dim{dim}', output / f'dim{dim}' / 'run'
        config = output / 'configs' / f'dim{dim}.json'
        training = [sys.executable, '-u', '-m', 'scripts.run_ising2d_dimg_ablation',
            '--cached-training', '--config', str(config), '--output', str(run),
            '--device', f'cuda:{gpu}', '--mi-cache-dir', str(mi_dir),
            '--initialization-receipt', str(root / 'initialization.json')]
        report = [sys.executable, '-u', '-m', 'scripts.report_ising_dimg_ablation',
            '--run', str(run), '--output', str(root / 'report'), '--checkpoint', str(UPDATES),
            '--threads', '1', '--sampling-repeats', '5']
        response = ['nice', '-n', '10', sys.executable, '-u', '-m',
            'scripts.diagnose_ising_dimg_response', '--source-run', str(run), '--data', str(cache),
            '--output', str(root / 'response'), '--k-count', '3', '--c-repeats', '5', '--seed', '12345']
        arms.append(dict(name=f'dim{dim}', theory_dim=dim, gpu=gpu, device=f'cuda:{gpu}',
            root=str(root), run=str(run), config_path=str(config), config=cfg,
            changes=changed, commands=dict(training=training, report=report, response=response)))
    compare = [sys.executable, '-u', '-m', 'scripts.report_ising_dimg_ablation', '--compare',
               *(f"{arm['name']}={arm['root']}/report" for arm in arms), '--output', str(output / 'comparison')]
    return dict(schema_version=1, source_run=str(source), output=str(output), cache=str(cache),
        source_sha256=source_sha, immutable_stat_identity=immutable_stats, mi_cache_dir=str(mi_dir),
        mi_cache_keys=MI_KEYS, arms=arms, comparison_command=compare, protocol=dict(
            semantic_change='仅 model.theory_dim=1/2/3；显式保留2×2块大小与108→54→27',
            source_usage='只读取配置和既有数据/MI缓存身份；不加载或续训任何源权重',
            initialization='三臂Q/C/R全新seed91；记录各模块初始tensor SHA，不预设相同种子代表相同初始化或轨迹',
            labels_used_in_training=False, Kc_or_exponent_target=False,
            updates=UPDATES, seed=91, threads_per_training_arm=2,
            training_wall_cap=None, automatic_retry=False, resume=False,
            MI='复用既有2×2 TRAIN/TEST缓存；子进程cache-miss立即失败，绝不构建统计',
            data='复用同一L108数据，不重采样、不修改历史b3/depth3侧车、不重复完整数据审计',
            objectives='原Q/C/R损失、权重、独立梯度和学习率调度全部保持；坐标loss保持sum归约',
            execution_delta='三臂统一skip-final-report以避开历史plotter的dim>2限制；正式报告由新通用中文模块生成',
            parallelism='不同GPU训练并行；CPU报告与物理响应诊断全局串行且各单线程',
            selection='只比较最终实际12000；不根据理论指数选择checkpoint、根或训练终点',
            validation='完整DR含恒等项，全部特征值及差分；K切向对齐候选不冒充已认证热方向',
            limitations='单seed、有限尺度、B0；额外维度可能秩塌缩；相同更新数不证明收敛'))


def build_smoke_plan(source_run, output, gpus=(0, 2, 3)):
    """独立输出中的两更新执行检查，不生成或替代任何正式科学结果。"""
    plan = build_plan(source_run, output, gpus)
    plan['smoke_only'] = True
    plan['protocol'].update(updates=2, selection='仅验证CUDA反向/缓存命中；不得作为正式科学结果',
                            smoke_execution_changes=['training.epochs', 'training.log_every', 'training.checkpoint_every'])
    for arm in plan['arms']:
        arm['config']['training'].update(epochs=2, log_every=1, checkpoint_every=1)
        arm['commands'] = dict(training=arm['commands']['training'])
    plan.pop('comparison_command')
    return plan


def ensure_unchanged(plan, code_sha256, config_sha256):
    for path, sha in {**plan['source_sha256'], **code_sha256, **config_sha256}.items():
        if fingerprint(regular(path)) != sha:
            raise ValueError(f'源回执、代码或配置发生变化：{path}')
    for path, identity in plan['immutable_stat_identity'].items():
        if stat_identity(path) != identity:
            raise ValueError(f'共享数据或MI缓存元数据发生变化，不重算：{path}')


def _tensor_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def cached_training(args):
    """增加cache-miss禁止器；仅全新训练记录随机初始化，续训交回原入口。"""
    from experiments import rsmi_separate_qcr as trainer
    from recurrent_rg import mi_cache
    from scripts.report_ising_zh import chinese_figures
    original_cache, original_model = mi_cache.load_or_build_mi_statistics, trainer.build_model
    def cache_only(directory, key_data, build, **execution_options):
        def forbidden():
            raise RuntimeError('预声明MI缓存未命中；禁止重算或自动修复')
        value, receipt = original_cache(directory, key_data, forbidden, **execution_options)
        if receipt.get('hit') is not True:
            raise RuntimeError('要求既有MI缓存命中')
        return value, receipt
    def record_initialization(*a, **kw):
        model = original_model(*a, **kw)
        save_json(args.initialization_receipt, dict(status='complete', created_at=utc_now(),
            seed=91, fresh=True, source_checkpoint=None, labels_used_in_training=False,
            C_tensor_sha256=_tensor_digest(model.rg_kernel.state_dict()),
            Q_tensor_sha256=_tensor_digest(model.theory_encoder.state_dict()),
            R_tensor_sha256=_tensor_digest(model.theory_rg.state_dict()),
            full_model_tensor_sha256=_tensor_digest(model.state_dict())), exclusive=True)
        return model
    training_args = ['--config', str(args.config), '--output', str(args.output),
        '--device', args.device, '--mi-cache-dir', str(args.mi_cache_dir), '--skip-final-report']
    resume_checkpoint = getattr(args, 'resume_checkpoint', None)
    resume_receipt = getattr(args, 'resume_receipt', None)
    source_checkpoint_sha256 = None
    if resume_receipt is not None:
        if resume_checkpoint is None:
            raise ValueError('A resume receipt requires an explicit resume checkpoint')
        if Path(resume_receipt).exists():
            raise FileExistsError('Refusing to overwrite a continuation receipt')
        source_checkpoint_sha256 = fingerprint(regular(resume_checkpoint))
    if resume_checkpoint is not None:
        training_args += ['--resume-checkpoint', str(resume_checkpoint)]
    if getattr(args, 'benchmark_warmup', None) is not None:
        training_args += ['--benchmark-warmup', str(args.benchmark_warmup)]
    mi_cache.load_or_build_mi_statistics = cache_only
    # The trainer constructs a temporary model before loading the saved state.
    # That constructor is not the resumed run's scientific initialization.
    # The continuation driver preserves the original receipt separately.
    if resume_checkpoint is None:
        trainer.build_model = record_initialization
    try:
        with chinese_figures():
            result = trainer.main(training_args)
        if resume_receipt is not None:
            manifest = json.loads(regular(Path(args.output) / 'manifest.json').read_text())
            provenance = manifest.get('resume_start_provenance', {})
            if (manifest.get('resumed_from') != str(resume_checkpoint)
                    or not manifest.get('resume_start_module_sha256')
                    or type(provenance.get('source_epoch')) is not int
                    or fingerprint(regular(resume_checkpoint)) != source_checkpoint_sha256):
                raise RuntimeError('Exact continuation provenance or source checkpoint changed')
            save_json(resume_receipt, dict(status='complete', created_at=utc_now(), fresh=False,
                source_checkpoint=str(resume_checkpoint),
                source_checkpoint_sha256=source_checkpoint_sha256,
                source_epoch=provenance['source_epoch'],
                resume_start_module_sha256=manifest['resume_start_module_sha256'],
                resume_start_provenance=provenance,
                labels_used_in_training=manifest['labels_used_in_training'],
                source_initial_metadata_preserved=provenance.get('source_initial_metadata_preserved', []),
                scope='Loaded source state before resumed segment; not a new random initialization'), exclusive=True)
        return result
    finally:
        mi_cache.load_or_build_mi_statistics, trainer.build_model = original_cache, original_model


def check_training(arm, plan):
    run = Path(arm['run'])
    manifest = json.loads(regular(run / 'manifest.json').read_text())
    summary = json.loads(regular(run / 'summary.json').read_text())
    required = dict(training_driver='rsmi_separate_qcr', precision='torch.float64', estimator='histogram',
        environment_mode='full_shell', buffer=1, independent_c=True, labels_used_in_training=False,
        rsmi_weight=2., execution_device=arm['device'], source=None)
    if (any(manifest.get(k) != v for k, v in required.items())
            or manifest.get('q_supervision') or manifest.get('resumed_from') is not None
            or manifest.get('frozen_C') or manifest.get('data_driven_initialization')
            or manifest.get('config') != arm['config'] or summary.get('config') != arm['config']
            or manifest.get('data_sha256') != DATA_SHA
            or Path(manifest.get('config_source', '')).resolve() != Path(arm['config_path'])):
        raise ValueError(f"{arm['name']}训练配方、初始方式、标签或精度不符")
    records = manifest.get('execution_optimizations', {}).get('mi_statistics_cache', {})
    for split, key in MI_KEYS.items():
        row = records.get(split, {})
        if (row.get('hit') is not True or row.get('key') != key or row.get('cache_version') != 1
                or Path(row.get('path', '')).resolve() != Path(plan['mi_cache_dir']) / f'mi-v1-{key}.pt'):
            raise ValueError('新训练必须命中相同的静态2×2 MI缓存')
    if (type(summary.get('actual_epochs')) is not int or summary['actual_epochs'] != UPDATES
            or summary.get('requested_epochs') != UPDATES or summary.get('stopped_by_training_limit')
            or summary.get('stopped_by_wall_limit') or summary.get('training_limit_seconds') is not None
            or summary.get('final_report_skipped') is not True):
        raise ValueError('必须完成12000步且统一跳过旧内置报告，不得wall cap或早停')
    policy = LearningRatePolicy(RunConfig.from_dict(arm['config']).training)
    history = summary.get('history', [])
    expected = [1, *range(arm['config']['training']['log_every'], UPDATES + 1,
                         arm['config']['training']['log_every'])]
    if (summary.get('lr_scheduler_state') != policy.checkpoint_state(UPDATES)
            or [row.get('epoch') for row in history] != expected
            or any(not math.isclose(row.get('learning_rate', -1.), policy.rate_for_update(row['epoch']),
                                    rel_tol=1e-12, abs_tol=0.) for row in history)):
        raise ValueError('学习率或真实更新数不符')
    init = json.loads(regular(Path(arm['root']) / 'initialization.json').read_text())
    if init.get('fresh') is not True or init.get('source_checkpoint') is not None:
        raise ValueError('初始化回执不符')
    regular(run / f'training_step_{UPDATES}.pt')
    return dict(status='complete', actual_epochs=UPDATES, new_updates=UPDATES,
        device=arm['device'], theory_dim=arm['theory_dim'], source=None,
        labels_used_in_training=False, initialization=init, mi_cache_hits=records,
        manifest_sha256=fingerprint(run / 'manifest.json'), timing=summary.get('timing', {}))


def resources():
    memory = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        memory[key] = int(value.strip().split()[0]) * 1024
    return dict(available_memory_bytes=memory['MemAvailable'], cpu_count=os.cpu_count(),
                load_average=list(os.getloadavg()), checked_at=utc_now())


def check_report(arm, name):
    root = Path(arm['root']) / name
    primary = root / ('report.json' if name == 'report' else 'comparison.json')
    if not primary.is_file():
        raise ValueError(f'正式{name}缺少主JSON：{primary}')
    if name == 'report':
        marker = json.loads(primary.read_text())
        if (marker.get('status') != 'complete' or marker.get('report_complete') is not True
                or marker.get('checkpoint_epoch') != UPDATES
                or marker.get('theory_dim') != arm['theory_dim']):
            raise ValueError('正式报告完成标志不符')
        for item in ('metrics.json', 'fixed_points.json', 'critical_exponents.json', 'encodings.npz',
                     'R_vs_g.png', 'R_displacement.png', 'coupling_by_scale.png',
                     'loss_comparison.png', 'representation_diagnostics.png'):
            regular(root / item)
    else:
        comparison = json.loads(primary.read_text())
        marker = json.loads(regular(root / 'status.json').read_text())
        if (comparison.get('status') != 'complete' or comparison.get('theory_dim') != arm['theory_dim']
                or marker.get('status') != 'complete' or marker.get('workflow_complete') is not True):
            raise ValueError('物理响应诊断完成标志不符')
    return dict(status='complete', path=str(root), files={path.name: fingerprint(path)
        for path in sorted(root.iterdir()) if path.is_file() and path.suffix in ('.json', '.png', '.npz')})


def check_smoke(arm, plan):
    run = Path(arm['run'])
    manifest = json.loads(regular(run / 'manifest.json').read_text())
    summary = json.loads(regular(run / 'summary.json').read_text())
    if (summary.get('actual_epochs') != 2 or summary.get('requested_epochs') != 2
            or summary.get('final_report_skipped') is not True or manifest.get('config') != arm['config']
            or manifest.get('execution_device') != arm['device'] or manifest.get('precision') != 'torch.float64'
            or manifest.get('labels_used_in_training') is not False or manifest.get('source') is not None
            or manifest.get('resumed_from') is not None):
        raise ValueError('独立两步执行检查不符合预声明设置')
    for split, key in MI_KEYS.items():
        row = manifest.get('execution_optimizations', {}).get('mi_statistics_cache', {}).get(split, {})
        if row.get('hit') is not True or row.get('key') != key:
            raise ValueError('两步检查必须命中既有MI缓存')
    for step in (1, 2):
        regular(run / f'training_step_{step}.pt')
    return dict(status='complete', actual_epochs=2, smoke_only=True, device=arm['device'],
                theory_dim=arm['theory_dim'], initialization=json.loads(regular(
                    Path(arm['root']) / 'initialization.json').read_text()))


def execute(plan):
    root = Path(plan['output'])
    if root.exists() or root.is_symlink():
        raise FileExistsError(f'拒绝覆盖或隐式续跑：{root}')
    if os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('必须取消CUDA_VISIBLE_DEVICES，保持cuda:N与nvidia-smi索引一致')
    project = Path(__file__).resolve().parents[1]
    code_sha = {str(project / name): fingerprint(regular(project / name)) for name in CODE_FILES}
    root.mkdir(parents=True, exist_ok=False)
    (root / 'logs').mkdir()
    (root / 'configs').mkdir()
    config_sha = {}
    for arm in plan['arms']:
        Path(arm['root']).mkdir()
        save_json(arm['config_path'], arm['config'], exclusive=True)
        config_sha[arm['config_path']] = fingerprint(arm['config_path'])
    save_json(root / 'plan.json', plan, exclusive=True)
    job = dict(status='running', workflow_complete=False, parent_pid=os.getpid(), started_at=utc_now(),
        hostname=subprocess.check_output(['hostname'], text=True).strip(), current=None,
        source_sha256=plan['source_sha256'], code_sha256=code_sha, config_sha256=config_sha,
        protocol=plan['protocol'], stages=[], training={}, reports={}, gpu_checks=[],
        arms={arm['name']: dict(status='waiting_gpu', current=None) for arm in plan['arms']})
    save = lambda: save_json(root / 'job.json', job)
    save()
    active, completed, ready, last_check = {}, set(), {}, {}
    failed = False
    def launch(arm, name):
        ensure_unchanged(plan, code_sha, config_sha)
        path = root / 'logs' / f"{arm['name']}_{name}.log"
        environment = dict(os.environ, OMP_NUM_THREADS='2' if name == 'training' else '1',
            MKL_NUM_THREADS='2' if name == 'training' else '1',
            OPENBLAS_NUM_THREADS='2' if name == 'training' else '1', MPLBACKEND='Agg')
        with path.open('x') as stream:
            child = subprocess.Popen(arm['commands'][name], stdout=stream, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, env=environment)
        row = dict(name=name, arm=arm['name'], run=arm['run'], pid=child.pid, log=str(path),
                   command=arm['commands'][name], status='running', started_at=utc_now())
        active[(arm['name'], name)] = (child, row)
        job['arms'][arm['name']].update(status=name, current=row)
        save()
    try:
        while len(completed) < len(plan['arms']):
            for key, (child, row) in list(active.items()):
                code = child.poll()
                if code is None:
                    continue
                del active[key]
                row.update(returncode=code, status='complete' if code == 0 else 'failed', ended_at=utc_now())
                job['stages'].append(row)
                name, stage_name = key
                arm = next(a for a in plan['arms'] if a['name'] == name)
                job['arms'][name]['current'] = None
                if code:
                    failed = True
                    job.update(status='failed', requires_user_action=True,
                               error=f'{name}/{stage_name} returncode={code}；不重试，不停止其他已启动训练')
                    job['arms'][name]['status'] = 'failed'
                elif stage_name == 'training':
                    job['training'][name] = check_smoke(arm, plan) if plan.get('smoke_only') else check_training(arm, plan)
                    job['arms'][name]['status'] = 'complete' if plan.get('smoke_only') else 'awaiting_report'
                    if plan.get('smoke_only'):
                        completed.add(name)
                else:
                    job['reports'].setdefault(name, {})[stage_name] = check_report(arm, stage_name)
                    job['arms'][name]['status'] = 'awaiting_response' if stage_name == 'report' else 'complete'
                    if stage_name == 'response':
                        completed.add(name)
                save()
            if failed:
                if not active:
                    break
            else:
                for arm in plan['arms']:
                    state = job['arms'][arm['name']]
                    if state['status'] != 'waiting_gpu' or time.monotonic() - last_check.get(arm['name'], -math.inf) < 60:
                        continue
                    snapshot = gpu_snapshot(arm['gpu'])
                    last_check[arm['name']] = time.monotonic()
                    ready[arm['name']] = ready.get(arm['name'], 0) + 1 if not snapshot['compute_pids'] and snapshot['free_mib'] >= 12288 else 0
                    row = dict(arm=arm['name'], checked_at=utc_now(), consecutive_ready=ready[arm['name']], **snapshot)
                    job['gpu_checks'].append(row)
                    if ready[arm['name']] >= 2:
                        host = resources()
                        job['resources'] = host
                        if host['available_memory_bytes'] >= 36 * (1 << 30):
                            launch(arm, 'training')
                    save()
                if not any(stage_name != 'training' for _, stage_name in active):
                    for arm in plan['arms']:
                        state = job['arms'][arm['name']]['status']
                        if state in ('awaiting_report', 'awaiting_response'):
                            launch(arm, 'report' if state == 'awaiting_report' else 'response')
                            break
            job['current'] = dict(name='parallel', active=[row for _, row in active.values()]) if active else None
            save()
            if len(completed) < len(plan['arms']):
                time.sleep(5)
        if not failed:
            ensure_unchanged(plan, code_sha, config_sha)
            if not plan.get('smoke_only'):
                log = root / 'logs' / 'comparison.log'
                with log.open('x') as stream:
                    child = subprocess.Popen(plan['comparison_command'], stdout=stream, stderr=subprocess.STDOUT,
                        env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MPLBACKEND='Agg'))
                    row = dict(name='comparison', pid=child.pid, command=plan['comparison_command'], log=str(log), status='running')
                    job['current'] = row
                    save()
                    code = child.wait()
                row.update(status='complete' if code == 0 else 'failed', returncode=code)
                job['stages'].append(row)
                if code:
                    raise RuntimeError(f'comparison returncode={code}；不重跑')
                regular(root / 'comparison' / 'comparison.json')
                marker = json.loads(regular(root / 'comparison' / 'report.json').read_text())
                if marker.get('status') != 'complete' or marker.get('report_complete') is not True:
                    raise ValueError('聚合报告没有完整完成')
            ensure_unchanged(plan, code_sha, config_sha)
            init = {arm['name']: job['training'][arm['name']]['initialization'] for arm in plan['arms']}
            job['initialization_comparison'] = {component: len({row[component] for row in init.values()}) == 1
                for component in ('C_tensor_sha256', 'Q_tensor_sha256', 'R_tensor_sha256')}
            job.update(status='complete', workflow_complete=True, requires_user_action=False)
        job.update(current=None, ended_at=utc_now())
        save()
        return 1 if failed else 0
    except BaseException as error:
        job.update(status='failed', workflow_complete=False, requires_user_action=True,
                   error=f'{type(error).__name__}: {error}', ended_at=utc_now(),
                   active_children_preserved=[row for _, row in active.values()])
        save()
        traceback.print_exc()
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', type=int, nargs=3, default=[0, 2, 3])
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='独立输出、每臂两步，不生成正式科学报告')
    parser.add_argument('--cached-training', action='store_true')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--device')
    parser.add_argument('--mi-cache-dir', type=Path)
    parser.add_argument('--initialization-receipt', type=Path)
    args = parser.parse_args(argv)
    if args.cached_training:
        if any(value is None for value in (args.config, args.device, args.mi_cache_dir, args.initialization_receipt)):
            raise ValueError('内部训练入口缺少明确配置、GPU、MI目录或初始化回执')
        cached_training(args)
        return 0
    if args.source_run is None:
        raise ValueError('必须指定已完成无监督source-run')
    plan = (build_smoke_plan if args.smoke else build_plan)(args.source_run, args.output, args.gpus)
    if args.plan:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    if args.detach:
        if args.execute:
            raise ValueError('detach和execute不能同时使用')
        logs = Path.cwd() / 'logs'
        logs.mkdir(exist_ok=True)
        receipt, log = (logs / (args.output.name + suffix) for suffix in ('.launch.json', '.log'))
        command = ['nice', '-n', '10', sys.executable, '-u', '-m', 'scripts.run_ising2d_dimg_ablation',
            '--source-run', str(args.source_run.resolve()), '--output', str(args.output.resolve()),
            '--gpus', *map(str, args.gpus), '--execute', *(['--smoke'] if args.smoke else [])]
        save_json(receipt, dict(status='launching', command=command, created_at=utc_now()), exclusive=True)
        with log.open('x') as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True,
                env=dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MPLBACKEND='Agg'))
        result = dict(status='launched', parent_pid=child.pid, output=str(args.output.resolve()),
                      log=str(log), command=command, launched_at=utc_now())
        save_json(receipt, result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0
    return execute(plan)


if __name__ == '__main__':
    raise SystemExit(main())
