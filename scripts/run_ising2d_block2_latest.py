"""同一 L108 数据上的 2×2 C 实验：全新 Q/C/R、seed91、完整 12000 步。

只将已完成的深度二控制配置的 block_factor 从 3 改为 2。
物理数据及其历史采样元数据不变；为 2×2 环境统计建立独立 CPU MI 缓存。
GPU 连续两次空闲检查后训练，任一阶段失败即停止，不续跑、不自动重试。
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
from types import SimpleNamespace

import torch

from experiments.rsmi_separate_qcr import build_parser as training_parser, validate_recipe
from recurrent_rg.config import RunConfig, ScaleConfig
from recurrent_rg.learning_rate import LearningRatePolicy
from recurrent_rg.mi_cache import _cache_identity, lattice_statistics_cache_key
from recurrent_rg.rsmi import region_offsets
from recurrent_rg.runner import file_fingerprint
from scripts.run_ising2d_exponent_ablation import check_data, check_reports, stage
from scripts.run_scalar_lr_gpu_pipeline import save_status, utc_now


DATA_SHA = 'f0e319a771d220e948c70e97fda8bbd373d46fdbb280211f0cbd29ce9d36de54'
SIDECAR_SHA = '0d2d7591b37aa27757bde1d1d964059d38b64ffb6374df1d3d7724993e801e2a'
UPDATES = 12000
CONTROL_FILES = ('manifest.json', 'summary.json', 'full_report/encodings.npz')


def differences(left, right, prefix=''):
    result = []
    for key in left.keys() | right.keys():
        name = f'{prefix}.{key}' if prefix else key
        if isinstance(left.get(key), dict) and isinstance(right.get(key), dict):
            result.extend(differences(left[key], right[key], name))
        elif left.get(key) != right.get(key):
            result.append(name)
    return sorted(result)


def matched_config(manifest):
    """Keep every source setting, changing only the explicitly requested C block."""
    original = manifest['config']
    cfg = deepcopy(original)
    t, m, e = cfg['training'], cfg['model'], cfg['evaluation']
    if (manifest.get('data_sha256') != DATA_SHA or cfg.get('schema_version') != 5
            or t['max_rg_steps'] != 2 or e['max_rg_steps'] != 2
            or t['epochs'] != UPDATES or t['seed'] != 91
            or m['block_factor'] != 3 or m['dimension'] != 2 or m['theory_dim'] != 1
            or t['gradient_mode'] != 'qc_r_separate' or not m['affine_reconstruction']):
        raise ValueError('Unexpected completed depth-2 b3 control configuration')
    cfg['model']['block_factor'] = 2
    parsed = RunConfig.from_dict(cfg)
    parsed.validate()
    args = training_parser().parse_args(['--config', 'unused.json', '--output', 'unused'])
    validate_recipe(args, parsed)
    if differences(original, cfg) != ['model.block_factor']:
        raise ValueError('Only model.block_factor may change')
    return cfg


def build_plan(source, output, gpu=0):
    """Read only the control JSON; never load data/checkpoints or create files."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0:
        raise ValueError('GPU index must be a nonnegative integer')
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('Owned output and historical source must be separate directories')
    manifest = json.loads((source / 'manifest.json').read_text())
    summary = json.loads((source / 'summary.json').read_text())
    cfg = matched_config(manifest)
    if (summary.get('config') != manifest['config']
            or type(summary.get('actual_epochs')) is not int or summary['actual_epochs'] != UPDATES
            or summary.get('requested_epochs') != UPDATES
            or summary.get('stopped_by_training_limit') or summary.get('stopped_by_wall_limit')):
        raise ValueError('Historical control must have completed exactly 12000 updates')
    run, config_path = output / 'run', output / 'configs' / 'block2_latest.json'
    mi_dir = output / 'mi_cache'
    commands = [dict(stage='mi_preprocessing', command=[sys.executable, '-u', '-m',
        'scripts.prepare_qcr_mi_cache', '--config', str(config_path),
        '--output', str(output / 'mi_preprocessing.json'), '--cache-dir', str(mi_dir)])]
    modules = [('training', 'experiments.rsmi_separate_qcr',
                ['--config', str(config_path), '--output', str(run), '--device', f'cuda:{gpu}',
                 '--mi-cache-dir', str(mi_dir)]),
               ('phase_report', 'scripts.report_scalar_joint',
                ['--run', str(run), '--threads', '1', '--sampling-repeats', '5']),
               ('loss_report', 'scripts.report_mi_training', ['--run', str(run), '--threads', '1']),
               ('exponent_report', 'scripts.report_ising_scalar_exponents', ['--run', str(run)]),
               ('local_response', 'scripts.report_ising_local_response',
                ['--run', str(run), '--output', str(output / 'local_response')])]
    for name, module, extra in modules:
        commands.append(dict(stage=name, command=[sys.executable, '-u', '-m',
            'scripts.report_ising_zh', '--module', module, *extra]))
    return dict(source_run=str(source), output=str(output), run=str(run),
        config_path=str(config_path), config=cfg, cache=cfg['data']['cache'],
        mi_cache_dir=str(mi_dir), device=f'cuda:{gpu}', gpu=gpu,
        changes=['model.block_factor'], control_lattice_sizes=[108, 36, 12],
        lattice_sizes=[shape[0] for shape in ScaleConfig((108, 108), 2, block_factor=2).shapes()],
        commands=commands, protocol=dict(
            change='仅 C 的非重叠块边长和步长 3→2；108→54→27，RG 深度保持二',
            initialization='Q、C、R 全新随机初始化；seed91；不读取或续接控制 checkpoint',
            data='复用同一 L108 物理样本；历史侧车保留 b3、深度三的生成元数据',
            mi='CPU 重新计算 2×2、buffer1 完整壳层的静态经验 MI 统计；不生成物理样本',
            objectives='保持最新 Q 重构与子集一致性、C 的 -2MI+2方差下限+颜色NLL、R 闭合目标及独立梯度',
            updates=UPDATES, seed=91, wall_cap=None, automatic_retry=False, resume=False,
            selection='只报告最终实际 12000 步；保留全部有界搜索根及全部逆 K 分支',
            plots='全部新图使用中文；保留数学符号、变量和公式；报告 R(g)、R(g)-g、K→g 及损失曲线',
            comparison='块大小改变会改变 C 架构、MI 统计、尺度尺寸及随机数消耗；相同种子不表示逐步相同轨迹',
            convergence='完整 12000 步不自动证明收敛'))


def control_receipt(source):
    files = {}
    for name in CONTROL_FILES:
        path = Path(source) / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'Historical control file must be a regular existing file: {path}')
        files[str(path)] = file_fingerprint(path)
    return dict(files=files)


def ensure_unchanged(plan, source, config_sha):
    for path, digest in source['files'].items():
        if file_fingerprint(path) != digest:
            raise ValueError('Historical control file changed')
    if file_fingerprint(plan['config_path']) != config_sha:
        raise ValueError('Owned configuration changed')
    cache = Path(plan['cache'])
    if (file_fingerprint(cache) != DATA_SHA
            or file_fingerprint(cache.with_suffix('.json')) != SIDECAR_SHA):
        raise ValueError('Physical data or historical sampling sidecar changed')


def check_preprocessing(plan, data, config_sha):
    receipt = json.loads((Path(plan['output']) / 'mi_preprocessing.json').read_text())
    required = dict(status='complete', data_sha256=DATA_SHA, config_sha256=config_sha,
                    device='cpu', model_initialized=False, optimizer_updates=0,
                    sampling_draws=0, cache_version=1)
    if any(receipt.get(key) != value for key, value in required.items()):
        raise ValueError('MI preprocessing must be static CPU statistics for the owned configuration')
    if Path(receipt.get('config', '')).resolve() != Path(plan['config_path']):
        raise ValueError('MI preprocessing identifies a different configuration')
    cfg = RunConfig.from_dict(plan['config'])
    _, offsets = region_offsets(2, block=2, buffer=1)
    for split in ('train', 'test'):
        row = receipt.get('splits', {}).get(split, {})
        proxy = SimpleNamespace(dtype=torch.float64, shape=data[split + '_shape'])
        identity = lattice_statistics_cache_key(cfg, DATA_SHA, split, proxy, offsets,
            'binary_lattice_mi_statistics', dict(block=2))
        _, key = _cache_identity(identity)
        expected = Path(plan['mi_cache_dir']) / f'mi-v1-{key}.pt'
        record = row.get('cache', {})
        if (record.get('key') != key or Path(record.get('path', '')).resolve() != expected
                or record.get('cache_version') != 1 or not expected.is_file()
                or row.get('data_shape') != data[split + '_shape']):
            raise ValueError('MI preprocessing must retain the new block-2 cache identity in the owned directory')
    return receipt


def check_training(plan, preprocessing):
    run = Path(plan['run'])
    manifest = json.loads((run / 'manifest.json').read_text())
    summary = json.loads((run / 'summary.json').read_text())
    if (manifest.get('config') != plan['config'] or summary.get('config') != plan['config']
            or manifest.get('data_sha256') != DATA_SHA
            or Path(manifest.get('config_source', '')).resolve() != Path(plan['config_path'])):
        raise ValueError('Training differs from the owned configuration or physical data')
    required = dict(training_driver='rsmi_separate_qcr', precision='torch.float64', estimator='histogram',
        environment_mode='full_shell', buffer=1, independent_c=True,
        labels_used_in_training=False, rsmi_weight=2., execution_device=plan['device'], source=None)
    if any(manifest.get(key) != value for key, value in required.items()):
        raise ValueError('Training did not retain the dtype, device, or independent Q/C/R recipe')
    if (manifest.get('resumed_from') is not None or manifest.get('frozen_C')
            or manifest.get('data_driven_initialization')):
        raise ValueError('Q/C/R must start fresh, without checkpoint or data-driven initialization')
    records = manifest.get('execution_optimizations', {}).get('mi_statistics_cache', {})
    for split in ('train', 'test'):
        row, previous = records.get(split, {}), preprocessing['splits'][split]['cache']
        if row.get('hit') is not True or any(row.get(key) != previous.get(key)
                for key in ('path', 'key', 'cache_version')):
            raise ValueError('Training must hit its preprocessed block-2 MI caches')
    # Dense and selected-support empirical MI are both valid execution routes.
    if (type(summary.get('actual_epochs')) is not int or summary['actual_epochs'] != UPDATES
            or summary.get('requested_epochs') != UPDATES
            or summary.get('stopped_by_training_limit') or summary.get('stopped_by_wall_limit')
            or summary.get('training_limit_seconds') is not None):
        raise ValueError('Training must finish exactly 12000 fresh updates without a wall cap')
    policy = LearningRatePolicy(RunConfig.from_dict(plan['config']).training)
    if summary.get('lr_scheduler_state') != policy.checkpoint_state(UPDATES):
        raise ValueError('Completed cumulative learning-rate state differs')
    history = summary.get('history', [])
    log_every = plan['config']['training']['log_every']
    expected_epochs = sorted({1, UPDATES, *range(log_every, UPDATES + 1, log_every)})
    if [row.get('epoch') for row in history] != expected_epochs or any(
            not math.isclose(row.get('learning_rate', -1.), policy.rate_for_update(row['epoch']),
                             rel_tol=1e-12, abs_tol=0.) for row in history):
        raise ValueError('Actual logged update numbers or learning rates differ')
    if summary.get('timing', {}).get('device') != plan['device']:
        raise ValueError('Training timing identifies a different execution device')
    return dict(actual_epochs=UPDATES, requested_epochs=UPDATES, new_updates=UPDATES,
        source=None, dtype=manifest['precision'], device=manifest['execution_device'],
        config_sha256=file_fingerprint(plan['config_path']), data_sha256=DATA_SHA,
        lr_scheduler_state=summary['lr_scheduler_state'], mi_cache_hits=records,
        selected_empirical_mi=manifest.get('execution_optimizations', {}).get('selected_empirical_mi'),
        timing=summary.get('timing', {}), manifest_sha256=file_fingerprint(run / 'manifest.json'))


def gpu_snapshot(gpu):
    query = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=uuid,name,memory.free', '--format=csv,noheader,nounits'], text=True).strip()
    uuid, name, free = [item.strip() for item in query.split(',')]
    processes = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
    return dict(index=gpu, uuid=uuid, name=name, free_mib=int(free),
                compute_pids=processes.splitlines() if processes else [])


def wait_gpu(output, status, gpu):
    started, ready = time.monotonic(), 0
    while True:
        snapshot = gpu_snapshot(gpu)
        ready = ready + 1 if not snapshot['compute_pids'] and snapshot['free_mib'] >= 12288 else 0
        row = dict(stage='wait_gpu', **snapshot, checked_at=utc_now(),
                   elapsed_seconds=time.monotonic() - started, consecutive_ready=ready)
        status.setdefault('gpu_checks', []).append(row)
        status.update(status='waiting_gpu', current=row)
        save_status(output, status)
        if ready >= 2:
            status.update(status='running', current=None, execution_gpu=snapshot,
                          gpu_wait_seconds=time.monotonic() - started)
            save_status(output, status)
            return
        time.sleep(60)


def run(plan):
    output, source, cache = (Path(plan[key]) for key in ('output', 'source_run', 'cache'))
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Refusing overwrite or implicit resume: {output}')
    if os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('Unset CUDA_VISIBLE_DEVICES so nvidia-smi and cuda:N use the same GPU index')
    if cache.is_symlink() or cache.with_suffix('.json').is_symlink():
        raise ValueError('Physical cache and sidecar must be regular files')
    if not cache.is_file() or not cache.with_suffix('.json').is_file():
        raise FileNotFoundError('The immutable L108 cache and historical sidecar must both exist')
    control = control_receipt(source)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'logs').mkdir()
    (output / 'configs').mkdir()
    with Path(plan['config_path']).open('x') as stream:
        stream.write(json.dumps(plan['config'], indent=2, allow_nan=False) + '\n')
    config_sha = file_fingerprint(plan['config_path'])
    status = dict(status='preflight', started_at=utc_now(), pid=os.getpid(),
        hostname=subprocess.check_output(['hostname'], text=True).strip(),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        plan=plan, protocol=plan['protocol'], source=control, config_sha256=config_sha,
        stages=[], current=None, training={}, reports={})
    save_status(output, status)
    started = time.monotonic()
    try:
        torch.set_num_threads(plan['config']['training']['threads'])
        data = check_data(cache, size=108, max_rg_steps=3, expected_digest=DATA_SHA)
        if data['sampling_sidecar_sha256'] != SIDECAR_SHA:
            raise ValueError('Historical data sidecar changed')
        status['data'] = data
        save_status(output, status)
        prep = plan['commands'][0]
        stage(output, status, prep['stage'], prep['command'])
        preprocessing = check_preprocessing(plan, data, config_sha)
        status['mi_preprocessing'] = preprocessing
        save_status(output, status)
        wait_gpu(output, status, plan['gpu'])
        ensure_unchanged(plan, control, config_sha)
        for row in plan['commands'][1:]:
            stage(output, status, row['stage'], row['command'])
            if row['stage'] == 'training':
                status['training'] = check_training(plan, preprocessing)
                save_status(output, status)
        status['reports'] = check_reports(dict(name='block2_latest', run=plan['run']))
        response = json.loads((output / 'local_response' / 'response.json').read_text())
        if (response.get('status') != 'complete' or response.get('actual_epochs') != UPDATES
                or not (output / 'local_response' / 'local_response.png').is_file()):
            raise ValueError('Local-response report did not complete for the final endpoint')
        ensure_unchanged(plan, control, config_sha)
        status.update(status='complete', workflow_complete=True, current=None,
                      ended_at=utc_now(), elapsed_seconds=time.monotonic() - started)
        save_status(output, status)
    except BaseException as error:
        status.update(status='failed', requires_user_action=True,
            error=f'{type(error).__name__}: {error}', ended_at=utc_now(),
            elapsed_seconds=time.monotonic() - started)
        save_status(output, status)
        raise
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args(argv)
    if not args.plan and (args.output.exists() or args.output.is_symlink()):
        raise FileExistsError(f'Refusing overwrite or implicit resume: {args.output}')
    plan = build_plan(args.source_run, args.output, args.gpu)
    if args.plan:
        print(json.dumps(plan, indent=2, ensure_ascii=False, allow_nan=False))
        return plan
    return run(plan)


if __name__ == '__main__':
    main()
