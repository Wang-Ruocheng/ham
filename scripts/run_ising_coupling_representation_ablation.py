"""无监督改进序列第二阶段：四臂全新 QCR，GPU2/3 排队，公共报告串行。

Only the new natural arm changes the declared Q decoder/reconstruction target.
Historical files and phase-one code are pinned, not edited. A failure preserves
every already running child and its output; nothing is stopped or retried.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
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
from scripts import run_ising2d_dimg_ablation as baseline


UPDATES = 12000
SMOKE_UPDATES = 2
OVERRIDE_KIND = 'directional_even_gibbs_crossfit_pseudolikelihood_v1'
GPU_MAP = {1: 2, 3: 3}
CHECK_INTERVAL = 10.
MIN_MEMORY_BYTES = 36 * (1 << 30)
MAX_LOAD_PER_CORE = 1.25
MAX_IO_FULL_AVG10 = 20.
NEW_CODE_FILES = (
    'scripts/run_ising_coupling_representation_ablation.py',
    'experiments/ising_coupling_representation.py',
    'recurrent_rg/coupling_decoder.py',
    'scripts/report_ising_coupling_representation.py',
)
ADDITIONAL_CORE_FILES = (
    'scripts/run_ising2d_dimg_ablation.py', 'scripts/run_ising2d_block2_latest.py',
    'experiments/rsmi_separate_qcr.py', 'recurrent_rg/config.py',
    'recurrent_rg/training.py', 'recurrent_rg/learning_rate.py',
    'recurrent_rg/representation.py', 'recurrent_rg/mi_cache.py',
    'recurrent_rg/execution.py',
)
fingerprint, regular = baseline.fingerprint, baseline.regular
save_json, utc_now = baseline.save_json, baseline.utc_now


def canonical_configuration(value):
    """Tuple/list differences in checkpoint-vs-JSON receipts are not recipe changes."""
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _pin(mapping, path, digest=None):
    path = str(regular(path).resolve())
    current = fingerprint(path)
    if digest is not None and current != digest:
        raise ValueError(f'已授权阶段的来源或代码指纹不符：{path}')
    if path in mapping and mapping[path] != current:
        raise ValueError(f'禁止覆盖已固定的代码身份：{path}')
    mapping[path] = current


def phase_one_evidence(sequence_root, *, smoke=False):
    """Read-only authorization/provenance gate; no training tensors are opened."""
    sequence_root = Path(sequence_root).resolve()
    spec_path, job_path = sequence_root / 'plan.json', sequence_root / 'job.json'
    specification = json.loads(regular(spec_path).read_text())
    job = json.loads(regular(job_path).read_text())
    if specification.get('smoke') is not False:
        raise ValueError('第二阶段需要正式序列来源，不接受第一阶段 smoke 冒充正式结果')
    if not smoke and (job.get('phase1_complete') is not True
            or job.get('status') != 'phase1_complete' or job.get('requires_user_action') is True
            or job.get('phase2') is not None or job.get('phase2_complete') is True):
        raise ValueError('正式第一阶段尚未完成/等待审查，或第二阶段已登记；禁止重复启动')
    pinned = {}
    _pin(pinned, spec_path)
    for path, sha in {**specification.get('source_sha256', {}),
                      **specification.get('code_sha256', {})}.items():
        _pin(pinned, path, sha)
    if not specification.get('code_sha256') or not specification.get('source_sha256'):
        raise ValueError('第一阶段 plan 缺少已固定的来源/代码清单')
    if not smoke:
        for dim in (1, 3):
            marker_path = sequence_root / f'phase1/dim{dim}/report/report.json'
            marker = json.loads(regular(marker_path).read_text())
            if marker.get('status') != 'complete' or marker.get('report_complete') is not True:
                raise ValueError('第一阶段正式报告未完成')
            _pin(pinned, marker_path)
            _pin(pinned, sequence_root / f'phase1/dim{dim}/report/comparison.json')
        _pin(pinned, sequence_root / 'phase1/comparison.json')
    return specification, job, pinned


def build_plan(source_run, sequence_root, output, *, smoke=False, smoke_device=None):
    """Only small receipts/configuration and stat identities; no new data audit."""
    for value in (source_run, sequence_root, output):
        if Path(value).is_symlink():
            raise ValueError('来源和输出目录不接受符号链接')
    source, sequence, root = map(lambda value: Path(value).resolve(),
                                 (source_run, sequence_root, output))
    if root.exists() or root.is_symlink():
        raise FileExistsError(f'拒绝覆盖、续跑或重复启动：{root}')
    if root == sequence or root in sequence.parents:
        raise ValueError('只能使用序列下独立第二阶段目录或独立 smoke 根，不能覆盖序列根')
    if smoke_device is not None and (not smoke or smoke_device not in ('cuda:2', 'cuda:3')):
        raise ValueError('smoke-device 仅供独立两步检查，且只允许 GPU2/3')
    sequence_spec, sequence_job, pinned = phase_one_evidence(sequence, smoke=smoke)
    # Reuse the audited baseline-only plan builder as a read-only provenance
    # gate. Its unused dim2 arm is discarded; no GPU0 command enters this plan.
    original = baseline.build_plan(source, root, gpus=(2, 0, 3))
    base = deepcopy(original['arms'][0]['config'])
    for path, sha in original['source_sha256'].items():
        _pin(pinned, path, sha)
    project = Path(__file__).resolve().parents[1]
    for name in (*NEW_CODE_FILES, *ADDITIONAL_CORE_FILES):
        _pin(pinned, project / name)
    arms = []
    for dim in (1, 3):
        for kind in ('control', 'natural'):
            cfg = deepcopy(base)
            cfg['model']['theory_dim'] = dim
            cfg['model']['affine_reconstruction'] = kind == 'control'
            if smoke:
                cfg['training'].update(epochs=SMOKE_UPDATES, log_every=1, checkpoint_every=1)
            if cfg['model'].get('theory_head', 'nonlinear') != 'nonlinear':
                raise ValueError('原 Q head 必须保持 nonlinear；自然耦合 decoder 线性不是 Q head 线性')
            RunConfig.from_dict(cfg).validate()
            name = f'dim{dim}_{kind}'
            gpu = int(smoke_device.split(':')[1]) if smoke_device else GPU_MAP[dim]
            folder, run, config = root / name, root / name / 'run', root / 'configs' / f'{name}.json'
            training = [sys.executable, '-u', '-m', 'experiments.ising_coupling_representation',
                '--arm', kind, '--config', str(config), '--output', str(run), '--device', f'cuda:{gpu}',
                '--mi-cache-dir', original['mi_cache_dir'], '--initialization-receipt', str(folder / 'initialization.json'),
                '--data-sha256', baseline.DATA_SHA]
            report = ['nice', '-n', '10', sys.executable, '-u', '-m', 'scripts.report_ising_coupling_representation',
                '--kind', 'report', '--run', str(run), '--output', str(folder / 'report'),
                '--checkpoint', str(UPDATES), '--threads', '1', '--sampling-repeats', '5']
            response = ['nice', '-n', '10', sys.executable, '-u', '-m', 'scripts.report_ising_coupling_representation',
                '--kind', 'response', '--run', str(run), '--data', original['cache'],
                '--output', str(folder / 'response'), '--c-repeats', '5', '--bootstraps', '1000', '--seed', '12345']
            arms.append(dict(name=name, arm=kind, theory_dim=dim, gpu=gpu, device=f'cuda:{gpu}',
                root=str(folder), run=str(run), config_path=str(config), config=cfg,
                changes=baseline.differences(base, cfg), commands=dict(training=training, **({} if smoke else
                    dict(report=report, response=response)))))
    compare = ['nice', '-n', '10', sys.executable, '-u', '-m', 'scripts.report_ising_coupling_representation',
        '--kind', 'compare', '--arms', *(f"{arm['name']}={arm['root']}/report" for arm in arms),
        '--output', str(root / 'comparison')]
    return dict(schema_version=1, phase=2, source_run=str(source), sequence_root=str(sequence),
        output=str(root), smoke_only=smoke, smoke_device=smoke_device,
        updates=SMOKE_UPDATES if smoke else UPDATES, arms=arms,
        cache=original['cache'], mi_cache_dir=original['mi_cache_dir'],
        mi_cache_keys=baseline.MI_KEYS, immutable_stat_identity=original['immutable_stat_identity'],
        source_sha256=pinned, sequence_plan_sha256=fingerprint(sequence / 'plan.json'),
        sequence_job_sha256_before=fingerprint(sequence / 'job.json'),
        phase1_complete_before=sequence_job.get('phase1_complete'),
        comparison_command=None if smoke else compare,
        protocol=dict(fresh_QCR=True, source_weights_loaded=False, seed=91, labels_used_in_training=False,
            Kc_or_exponent_target=False, automatic_retry=False, resume=False,
            training_updates=SMOKE_UPDATES if smoke else UPDATES, wall_cap=None,
            source_recipe='保留无监督 b2 深度二、108→54→27、Q motif3、独立Q/C/R梯度及原C/R损失',
            contrast='每个dimg内仅Q概率重构对照 vs 七偶方向算符有限Gibbs自然耦合交叉条件似然',
            natural_override=OVERRIDE_KIND, Q_head='原 nonlinear head，不换成线性头',
            reconstruction_units='control为16-bin概率MSE；natural为逐点条件NLL，二者均乘10但单位不相同',
            initialization='每dimg配对C/Qbackbone/R初值严格一致；decoder架构/参数量不同，Qfull不声称相同',
            optimizer='每臂fresh Adam .001；6000/9000完成后乘.3；不恢复源优化器',
            GPU='正式dimg1仅cuda2，dimg3仅cuda3；同GPU control完成训练后才可启动natural；绝不GPU0/1',
            CPU='训练各2线程；正式中文report/response全局串行、CPU1线程、nice10',
            readiness=dict(two_idle_checks=True, minimum_interval_seconds=CHECK_INTERVAL,
                free_gpu_mib=12288, host_memory_bytes=MIN_MEMORY_BYTES,
                max_load_per_core=MAX_LOAD_PER_CORE, max_io_full_avg10=MAX_IO_FULL_AVG10),
            data='原物理缓存只读mmap；不重hash9GB、不重采样、不做新完整审计',
            MI='只读既有b2静态统计；cache miss失败，不构建',
            selection='固定最终12000；不按TEST、Kc或nu调参或选择根/终点',
            stage3='第二阶段报告后暂停科学审查，不自动生成独立尺寸数据或启动第三阶段'))


def io_pressure(path='/proc/pressure/io'):
    result = {}
    for line in Path(path).read_text().splitlines():
        family, *tokens = line.split()
        result[family] = {key: float(value) for key, value in (token.split('=') for token in tokens)}
    if 'full' not in result or 'avg10' not in result['full']:
        raise ValueError('缺少主机I/O压力记录，不盲目启动重负载')
    return result


def host_resources():
    result = baseline.resources()
    result['io_pressure'] = io_pressure()
    return result


def host_ready(resources):
    cores = resources.get('cpu_count')
    return bool(type(cores) is int and cores > 0
        and resources['available_memory_bytes'] >= MIN_MEMORY_BYTES
        and resources['load_average'][0] <= MAX_LOAD_PER_CORE * cores
        and resources['io_pressure']['full']['avg10'] < MAX_IO_FULL_AVG10)


def ensure_unchanged(plan, config_sha=None):
    for path, sha in {**plan['source_sha256'], **(config_sha or {})}.items():
        if fingerprint(regular(path)) != sha:
            raise ValueError(f'已固定来源、第一阶段代码、新代码或配置发生变化：{path}')
    for path, identity in plan['immutable_stat_identity'].items():
        if baseline.stat_identity(path) != identity:
            raise ValueError(f'共享物理数据或MI缓存身份发生变化，不重算：{path}')


def _check_checkpoint(arm, updates):
    """Small final checkpoint validation only; never load physical data here."""
    import torch
    path = regular(Path(arm['run']) / f'training_step_{updates}.pt')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    metadata, state = payload.get('metadata', {}), payload.get('model', {})
    if (payload.get('epoch') != updates
            or canonical_configuration(metadata.get('config')) != canonical_configuration(arm['config'])):
        raise ValueError('最终编号权重更新数或配置不符')
    expected_decoder = ('theory_encoder.distribution_decoder.linear.' if arm['arm'] == 'natural'
                        else 'theory_encoder.distribution_decoder.')
    weight_key, bias_key = expected_decoder + 'weight', expected_decoder + 'bias'
    count = 7 if arm['arm'] == 'natural' else 16
    if (weight_key not in state or bias_key not in state
            or tuple(state[weight_key].shape) != (count, arm['theory_dim'])
            or tuple(state[bias_key].shape) != (count,)
            or any(not torch.isfinite(value).all() or value.dtype != torch.float64 for value in state.values())):
        raise ValueError('最终decoder架构、有限性或float64不符')
    if arm['arm'] == 'natural' and 'theory_encoder.distribution_decoder.weight' in state:
        raise ValueError('自然臂不得保留未使用旧概率decoder参数')
    norms = state[weight_key].norm(dim=0)
    if torch.any((norms > 1e-12) & ((norms - 1).abs() > 1e-8)):
        raise ValueError('decoder列范数投影未保持预声明约定')
    optimizer = payload.get('optimizer', {})
    groups = optimizer.get('param_groups', [])
    names = [name for prefix in ('theory_encoder.', 'rg_kernel.', 'theory_rg.')
             for name in state if name.startswith(prefix) and name.endswith(('.weight', '.bias'))]
    if len(groups) != 1 or len(groups[0].get('params', [])) != len(names):
        raise ValueError('优化器参数组未恰好包含Q含decoder、C和R')
    for parameter, name in zip(groups[0]['params'], names):
        record = optimizer.get('state', {}).get(parameter, {})
        if (not record or float(record.get('step', -1)) != updates
                or tuple(record.get('exp_avg', torch.empty(0)).shape) != tuple(state[name].shape)
                or tuple(record.get('exp_avg_sq', torch.empty(0)).shape) != tuple(state[name].shape)):
            raise ValueError(f'Adam更新数、decoder参与或参数矩形不符：{name}')
    expected_lr = LearningRatePolicy(RunConfig.from_dict(arm['config']).training).rate_for_update(updates)
    if not math.isclose(groups[0].get('lr', -1), expected_lr, rel_tol=1e-12, abs_tol=0):
        raise ValueError('最终Adam学习率不符')
    return dict(checkpoint_sha256=fingerprint(path), optimizer_parameter_count=len(names),
        optimizer_each_parameter_updates=updates, decoder_weight_shape=list(state[weight_key].shape),
        decoder_column_norms=norms.tolist())


def check_training(arm, plan):
    run = Path(arm['run'])
    manifest = json.loads(regular(run / 'manifest.json').read_text())
    summary = json.loads(regular(run / 'summary.json').read_text())
    required = dict(precision='torch.float64', estimator='histogram', environment_mode='full_shell',
        buffer=1, independent_c=True, labels_used_in_training=False, rsmi_weight=2.,
        execution_device=arm['device'], source=None)
    if (any(manifest.get(key) != value for key, value in required.items())
            or manifest.get('q_supervision') is not None or manifest.get('resumed_from') is not None
            or manifest.get('frozen_C') or manifest.get('data_driven_initialization')
            or manifest.get('config') != arm['config'] or summary.get('config') != arm['config']
            or manifest.get('data_sha256') != baseline.DATA_SHA
            or Path(manifest.get('config_source', '')).resolve() != Path(arm['config_path'])):
        raise ValueError('新臂训练配方、精度、初始方式、标签或配置不符')
    natural = arm['arm'] == 'natural'
    override = manifest.get('representation_override')
    if (manifest.get('training_driver') != ('ising_coupling_representation' if natural else 'rsmi_separate_qcr')
            or (natural and (not isinstance(override, dict) or override.get('kind') != OVERRIDE_KIND
                 or manifest.get('underlying_training_driver') != 'rsmi_separate_qcr'
                 or override.get('reconstruction_weight') != 10.
                 or override.get('labels_used_in_training') is not False))
            or (not natural and override is not None)):
        raise ValueError('自然耦合表示覆盖必须明确、真实、且仅用于natural臂')
    records = manifest.get('execution_optimizations', {}).get('mi_statistics_cache', {})
    for split, key in baseline.MI_KEYS.items():
        value = records.get(split, {})
        if (value.get('hit') is not True or value.get('key') != key or value.get('cache_version') != 1
                or Path(value.get('path', '')).resolve() != Path(plan['mi_cache_dir']) / f'mi-v1-{key}.pt'):
            raise ValueError('必须命中预先已有b2 TRAIN/TEST MI缓存')
    updates = plan['updates']
    if (summary.get('actual_epochs') != updates or summary.get('requested_epochs') != updates
            or summary.get('stopped_by_training_limit') or summary.get('stopped_by_wall_limit')
            or summary.get('stopped_by_train_plateau') or summary.get('training_limit_seconds') is not None
            or summary.get('final_report_skipped') is not True):
        raise ValueError('训练必须达到预定真实更新数且无早停/旧内置报告')
    policy = LearningRatePolicy(RunConfig.from_dict(arm['config']).training)
    every = arm['config']['training']['log_every']
    expected = sorted({1, updates, *range(every, updates + 1, every)})
    history = summary.get('history', [])
    if (summary.get('lr_scheduler_state') != policy.checkpoint_state(updates)
            or [value.get('epoch') for value in history] != expected
            or any(not math.isclose(value.get('learning_rate', -1), policy.rate_for_update(value['epoch']),
                                    rel_tol=1e-12, abs_tol=0) for value in history)):
        raise ValueError('记录的真实更新数和学习率策略不符')
    initialization = json.loads(regular(Path(arm['root']) / 'initialization.json').read_text())
    if (initialization.get('status') != 'complete' or initialization.get('fresh') is not True
            or initialization.get('source_checkpoint') is not None or initialization.get('seed') != 91
            or initialization.get('arm') != arm['arm'] or initialization.get('labels_used_in_training') is not False
            or initialization.get('data_reuse', {}).get('data_rehashed') is not False
            or initialization.get('data_reuse', {}).get('data_full_audit_performed') is not False
            or initialization.get('data_reuse', {}).get('coupling_labels_accessed') is not False
            or any(not isinstance(initialization.get(key), str) or len(initialization[key]) != 64 for key in
                ('C_tensor_sha256', 'Q_backbone_tensor_sha256', 'Q_tensor_sha256', 'R_tensor_sha256'))):
        raise ValueError('新随机初始化或只读数据复用回执不符')
    checkpoint = _check_checkpoint(arm, updates)
    return dict(status='complete', actual_epochs=updates, new_updates=updates,
        theory_dim=arm['theory_dim'], arm=arm['arm'], device=arm['device'], initialization=initialization,
        labels_used_in_training=False, mi_cache_hits=records, manifest_sha256=fingerprint(run / 'manifest.json'),
        representation_override=override, timing=summary.get('timing', {}), **checkpoint)


def check_paired_initialization(training):
    result = {}
    for dimension in (1, 3):
        rows = [training[f'dim{dimension}_{kind}']['initialization'] for kind in ('control', 'natural')]
        same = {key: rows[0][key] == rows[1][key] for key in
                ('C_tensor_sha256', 'Q_backbone_tensor_sha256', 'R_tensor_sha256')}
        if not all(same.values()):
            raise ValueError('配对control/natural的C、Qbackbone或R随机初值不相同')
        result[f'dim{dimension}'] = dict(**same,
            full_Q_equal=rows[0]['Q_tensor_sha256'] == rows[1]['Q_tensor_sha256'])
    return result


def check_report(arm, stage):
    return baseline.check_report(arm, stage)


def update_sequence(plan, phase_job, *, failed=False, complete=False):
    if plan['smoke_only']:
        return
    root = Path(plan['sequence_root'])
    path = root / 'job.json'
    job = json.loads(regular(path).read_text())
    if job.get('phase1_complete') is not True:
        raise ValueError('更新序列时发现第一阶段完成身份改变')
    pointer = job.get('phase2')
    if pointer and Path(pointer.get('output', '')).resolve() != Path(plan['output']):
        raise ValueError('序列已有其他第二阶段指针，拒绝覆盖')
    job.setdefault('phase1_parent_pid', job.get('parent_pid'))
    job.update(status='failed' if failed else 'phase2_complete' if complete else 'phase2_running',
        workflow_complete=False, phase2_complete=complete, parent_pid=os.getpid(), phase2_parent_pid=os.getpid(),
        current=None if failed or complete else dict(name='phase2', job=str(Path(plan['output']) / 'job.json')),
        phase2=dict(output=plan['output'], job=str(Path(plan['output']) / 'job.json'),
                    plan=str(Path(plan['output']) / 'plan.json'), status=phase_job['status']),
        pending_experiments=[3], next_experiment=3 if complete else None,
        awaiting_scientific_review=complete, requires_user_action=failed)
    if failed:
        job['error'] = phase_job.get('error')
        job['active_children_preserved'] = phase_job.get('active_children_preserved', [])
    save_json(path, job)


def execute(plan):
    root = Path(plan['output'])
    if root.exists() or root.is_symlink():
        raise FileExistsError('拒绝覆盖、隐式恢复或重复运行')
    if os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('保持cuda:N与nvidia-smi物理索引一致；不接受CUDA_VISIBLE_DEVICES映射')
    ensure_unchanged(plan)
    if not plan['smoke_only']:
        # Recheck before publication, not just when the read-only plan was made.
        phase_one_evidence(plan['sequence_root'], smoke=False)
    root.mkdir(parents=True, exist_ok=False)
    (root / 'logs').mkdir(); (root / 'configs').mkdir()
    config_sha = {}
    for arm in plan['arms']:
        Path(arm['root']).mkdir()
        save_json(arm['config_path'], arm['config'], exclusive=True)
        config_sha[arm['config_path']] = fingerprint(arm['config_path'])
    save_json(root / 'plan.json', plan, exclusive=True)
    job = dict(status='running', workflow_complete=False, phase=2, parent_pid=os.getpid(),
        started_at=utc_now(), current=None, stages=[], training={}, reports={},
        source_sha256=plan['source_sha256'], config_sha256=config_sha, resources=[],
        arms={arm['name']: dict(status='waiting_gpu', current=None) for arm in plan['arms']},
        smoke_only=plan['smoke_only'], requires_user_action=False)
    save = lambda: save_json(root / 'job.json', job)
    save()
    active, complete, last_check, consecutive = {}, set(), {}, {}

    def launch(arm, stage):
        ensure_unchanged(plan, config_sha)
        path = root / 'logs' / f"{arm['name']}_{stage}.log"
        threads = '2' if stage == 'training' else '1'
        environment = dict(os.environ, OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads,
                           OPENBLAS_NUM_THREADS=threads, MPLBACKEND='Agg')
        command = list(arm['commands'][stage])
        if stage == 'response':
            command += ['--expected-checkpoint-sha256', job['training'][arm['name']]['checkpoint_sha256']]
        with path.open('x') as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, env=environment)
        row = dict(name=stage, arm=arm['name'], run=arm['run'], pid=child.pid, log=str(path),
                   command=command, status='running', started_at=utc_now())
        active[(arm['name'], stage)] = child, row
        job['arms'][arm['name']].update(status=stage, current=row)
        save()

    try:
        update_sequence(plan, job)
        while len(complete) < len(plan['arms']):
            for key, (child, row) in list(active.items()):
                code = child.poll()
                if code is None:
                    continue
                del active[key]
                row.update(returncode=code, status='complete' if code == 0 else 'failed', ended_at=utc_now())
                job['stages'].append(row)
                name, stage = key
                arm = next(value for value in plan['arms'] if value['name'] == name)
                job['arms'][name]['current'] = None
                if code:
                    raise RuntimeError(f'{name}/{stage} returncode={code}；保留其他运行子进程，不重试')
                if stage == 'training':
                    job['training'][name] = check_training(arm, plan)
                    if plan['smoke_only']:
                        job['arms'][name]['status'] = 'complete'; complete.add(name)
                    else:
                        job['arms'][name]['status'] = 'awaiting_report'
                else:
                    job['reports'].setdefault(name, {})[stage] = check_report(arm, stage)
                    job['arms'][name]['status'] = 'awaiting_response' if stage == 'report' else 'complete'
                    if stage == 'response':
                        complete.add(name)
                save()
            # A queue never runs two training jobs on one GPU. Natural can start
            # once its paired control TRAINING is verified, without waiting for
            # the control CPU report, while the other dimension remains free.
            for arm in plan['arms']:
                name = arm['name']
                if job['arms'][name]['status'] != 'waiting_gpu':
                    continue
                if arm['arm'] == 'natural' and f"dim{arm['theory_dim']}_control" not in job['training']:
                    continue
                if any(stage == 'training' and next(a for a in plan['arms'] if a['name'] == other)['gpu'] == arm['gpu']
                       for other, stage in active):
                    continue
                if time.monotonic() - last_check.get(name, -math.inf) < CHECK_INTERVAL:
                    continue
                snapshot, host = baseline.gpu_snapshot(arm['gpu']), host_resources()
                last_check[name] = time.monotonic()
                ready = not snapshot['compute_pids'] and snapshot['free_mib'] >= 12288 and host_ready(host)
                consecutive[name] = consecutive.get(name, 0) + 1 if ready else 0
                job['resources'].append(dict(arm=name, checked_at=utc_now(), gpu=snapshot, host=host,
                                            ready=ready, consecutive_ready=consecutive[name]))
                if consecutive[name] >= 2:
                    launch(arm, 'training')
                save()
            if not plan['smoke_only'] and not any(stage != 'training' for _, stage in active):
                for arm in plan['arms']:
                    state = job['arms'][arm['name']]['status']
                    if state in ('awaiting_report', 'awaiting_response'):
                        launch(arm, 'report' if state == 'awaiting_report' else 'response')
                        break
            job['current'] = dict(name='parallel', active=[row for _, row in active.values()]) if active else None
            save()
            if len(complete) < len(plan['arms']):
                time.sleep(2)
        job['paired_initialization'] = check_paired_initialization(job['training'])
        ensure_unchanged(plan, config_sha)
        if not plan['smoke_only']:
            log = root / 'logs' / 'comparison.log'
            with log.open('x') as stream:
                child = subprocess.Popen(plan['comparison_command'], stdout=stream, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                                                      OPENBLAS_NUM_THREADS='1', MPLBACKEND='Agg'))
                row = dict(name='comparison', pid=child.pid, command=plan['comparison_command'],
                           log=str(log), status='running')
                active[('comparison', 'comparison')] = child, row
                job['current'] = row; save()
                code = child.wait()
                del active[('comparison', 'comparison')]
            row.update(status='complete' if code == 0 else 'failed', returncode=code)
            job['stages'].append(row)
            if code:
                raise RuntimeError(f'comparison returncode={code}；不重跑')
            marker = json.loads(regular(root / 'comparison/report.json').read_text())
            if marker.get('status') != 'complete' or marker.get('report_complete') is not True:
                raise ValueError('四臂最终比较完成标志缺失')
            regular(root / 'comparison/comparison.json')
            regular(root / 'comparison/comparison.png')
        ensure_unchanged(plan, config_sha)
        job.update(status='smoke_complete' if plan['smoke_only'] else 'complete', workflow_complete=True,
            current=None, requires_user_action=False, ended_at=utc_now())
        save(); update_sequence(plan, job, complete=True)
        return 0
    except BaseException as error:
        job.update(status='failed', workflow_complete=False, requires_user_action=True,
            error=f'{type(error).__name__}: {error}', ended_at=utc_now(),
            active_children_preserved=[row for _, row in active.values()])
        save()
        update_sequence(plan, job, failed=True)
        traceback.print_exc()
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--sequence-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-device', choices=('cuda:2', 'cuda:3'))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    plan = build_plan(args.source_run, args.sequence_root, args.output, smoke=args.smoke,
                      smoke_device=args.smoke_device)
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2)); return 0
    if args.detach:
        if args.execute:
            raise ValueError('detach 与 execute 不能同时使用')
        logs = Path.cwd() / 'logs'; logs.mkdir(exist_ok=True)
        receipt, log = logs / f'{args.output.name}.launch.json', logs / f'{args.output.name}.log'
        command = ['nice', '-n', '10', sys.executable, '-u', '-m',
            'scripts.run_ising_coupling_representation_ablation', '--source-run', str(args.source_run.resolve()),
            '--sequence-root', str(args.sequence_root.resolve()), '--output', str(args.output.resolve()), '--execute']
        if args.smoke:
            command.append('--smoke')
        if args.smoke_device:
            command += ['--smoke-device', args.smoke_device]
        save_json(receipt, dict(status='launching', created_at=utc_now(), command=command), exclusive=True)
        with log.open('x') as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True,
                env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MPLBACKEND='Agg'))
        result = dict(status='launched', parent_pid=child.pid, command=command, output=str(args.output.resolve()),
                      log=str(log), launched_at=utc_now())
        save_json(receipt, result)
        print(json.dumps(result, ensure_ascii=False, indent=2)); return 0
    return execute(plan)


if __name__ == '__main__':
    raise SystemExit(main())
