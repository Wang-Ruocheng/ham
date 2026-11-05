"""第三阶段：独立尺寸物理采样与固定模型诊断；不训练、不自动重跑。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone


ARMS = ('dim1_control', 'dim1_natural', 'dim3_control', 'dim3_natural')
K_VALUES = (.43619047619047624, .44068679350977147, .45142857142857146)
NEW_FILES = ('scripts/prepare_ising_matched_sizes.py',
             'scripts/diagnose_ising_matched_sizes.py',
             'scripts/report_ising_matched_sizes.py',
             'scripts/run_ising_matched_sizes.py')
DEPENDENCIES = ('scripts/prepare_ising2d_qcr.py', 'scripts/prepare_ising2d_lr_data.py',
                'scripts/diagnose_ising_mcrg.py', 'scripts/diagnose_ising_dimg_response.py',
                'scripts/report_ising_mcrg.py', 'scripts/report_ising_zh.py',
                'scripts/report_ising_coupling_representation.py',
                'experiments/ising_coupling_representation.py',
                'recurrent_rg/coupling_decoder.py', 'recurrent_rg/checkpoints.py',
                'recurrent_rg/model_nd.py', 'recurrent_rg/binary_pooling.py',
                'recurrent_rg/config.py', 'recurrent_rg/geometry.py',
                'recurrent_rg/adapters.py')


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, payload):
    """Atomic replacement applies only to this workflow's mutable status files."""
    path = Path(path)
    temporary = path.with_name(path.name + '.writing')
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def fingerprint(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'需要非符号链接普通文件：{path}')
    # This workflow never hashes/opens the historical 9 GB physical cache.
    if path.stat().st_size > 32 * (1 << 20):
        raise ValueError(f'来源指纹清单不得包括大物理数据或缓存：{path}')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def resource_snapshot():
    memory = Path('/proc/meminfo').read_text().splitlines()
    available = int(next(line.split()[1] for line in memory if line.startswith('MemAvailable:'))) * 1024
    io = Path('/proc/pressure/io').read_text()
    full = next(line for line in io.splitlines() if line.startswith('full '))
    pressure = float(next(item.split('=')[1] for item in full.split() if item.startswith('avg10=')))
    cores, load = os.cpu_count() or 1, os.getloadavg()[0]
    return dict(checked_at=now(), cpu_count=cores, load1=load,
                available_memory_bytes=available, io_full_avg10=pressure,
                ready=available >= 36 * (1 << 30) and load <= 1.25 * cores and pressure < 20.)


def child_environment():
    return {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
            'OPENBLAS_NUM_THREADS': '1', 'NUMBA_NUM_THREADS': '1', 'MPLBACKEND': 'Agg'}


def ensure_cpu_policy():
    """Only lower this parent's priority; never touch another process or GPU."""
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        os.setpriority(os.PRIO_PROCESS, 0, 10)


def validate_phase2(source, sequence, *, smoke=False):
    source, sequence = Path(source), Path(sequence)
    sj, job = read(sequence/'job.json'), read(source/'job.json')
    if (job.get('status') != 'complete' or job.get('workflow_complete') is not True
            or job.get('smoke_only') is not False or job.get('requires_user_action')):
        raise ValueError('第三阶段必须来自完整正式第二阶段，不能使用smoke或部分结果')
    if (sj.get('phase2_complete') is not True or sj.get('requires_user_action')
            or sj.get('status') != 'phase2_complete'):
        raise ValueError('序列尚未处于第二阶段完成后的科学审查点')
    if not smoke and (sj.get('phase3') is not None or sj.get('phase3_complete') is True
                      or (sequence/'phase3.claim.json').exists()):
        raise ValueError('第三阶段已登记或启动；禁止重复、续跑、覆盖或恢复')
    report = read(source/'comparison/report.json')
    if (report.get('report_complete') is not True or report.get('workflow_complete') is not True
            or report.get('actual_updates_per_arm') != 12000):
        raise ValueError('第二阶段完整比较标志缺失')
    for arm in ARMS:
        tr = job['training'][arm]
        summary = read(source/arm/'run/summary.json')
        if (tr.get('actual_epochs') != 12000 or tr.get('new_updates') != 12000
                or tr.get('labels_used_in_training') is not False
                or summary.get('actual_epochs') != 12000 or summary.get('requested_epochs') != 12000
                or any(v is True for k, v in summary.items() if k.startswith('stopped_by_'))):
            raise ValueError(f'{arm}不是全新、无监督、实际12000终点')
        cfg = summary['config']
        if (cfg.get('schema_version') != 5 or cfg['model'].get('block_factor') != 2
                or cfg['model'].get('kernel_layout') != 'nonoverlapping'
                or cfg['model'].get('fixed_kernel') is not False
                or cfg['model'].get('kernel_initialization') != 'random'
                or cfg['model'].get('width') != 3 or cfg['training'].get('max_rg_steps') != 2):
            raise ValueError('仅延用显式历史b2消融模型，不启动新配方或修改模型')
        completed = read(source/arm/'report/report.json')
        response = read(source/arm/'response/status.json')
        if (completed.get('report_complete') is not True or completed.get('checkpoint_epoch') != 12000
                or response.get('workflow_complete') is not True or response.get('status') != 'complete'):
            raise ValueError(f'{arm}正式报告或物理响应未完成')
        actual_k = read(source/arm/'response/diagnostic.json')['K']
        if actual_k != list(K_VALUES):
            raise ValueError('须固定原报告实际3个K邻点，不使用错误的等间距近似')
    return sj, job


def build_plan(source_root, sequence_root, output, *, seed=20261016, smoke=False):
    raw = [Path(value) for value in (source_root, sequence_root, output)]
    if any(p.is_symlink() for p in raw):
        raise ValueError('来源和输出目录不允许符号链接')
    source, sequence, root = [p.resolve() for p in raw]
    if root.exists():
        raise FileExistsError(f'拒绝覆盖或重复启动：{root}')
    for parent in (source, sequence):
        if root == parent or root in parent.parents or parent in root.parents:
            raise ValueError('第三阶段输出必须为独立根，不得覆盖或嵌套历史源')
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError('采样种子必须为非负整数')
    _, job = validate_phase2(source, sequence, smoke=smoke)
    project = Path(__file__).resolve().parents[1]
    codes = {str(project/p): fingerprint(project/p) for p in (*NEW_FILES, *DEPENDENCIES)}
    for p, sha in job.get('source_sha256', {}).items():
        if p.endswith('.py') and fingerprint(p) != sha:
            raise ValueError(f'已冻结来源代码发生变化：{p}')
    sources = {str(source/p): fingerprint(source/p)
               for p in ('job.json', 'plan.json', 'comparison/report.json', 'comparison/comparison.json')}
    sources[str(sequence/'plan.json')] = fingerprint(sequence/'plan.json')
    for arm in ARMS:
        for p in ('run/manifest.json', 'run/summary.json', 'run/training_step_12000.pt',
                  'report/metrics.json', 'report/fixed_points.json', 'report/loss_comparison.png',
                  'report/loss_history.json', 'response/diagnostic.json', 'response/operator_statistics.npz'):
            path = source/arm/p
            sources[str(path)] = fingerprint(path)
        checkpoint = str(source/arm/'run/training_step_12000.pt')
        if sources[checkpoint] != job['training'][arm]['checkpoint_sha256']:
            raise ValueError(f'{arm}终点checkpoint指纹不符')
    sizes, target = ((8, 16), 4) if smoke else ((54, 108), 27)
    commands = []
    for size in sizes:
        command = [sys.executable, '-u', '-m', 'scripts.prepare_ising_matched_sizes',
                   '--output', str(root/f'data/L{size}'), '--size', str(size), '--seed', str(seed)]
        if smoke:
            command.append('--smoke')
        commands.append(dict(name=f'sample_L{size}', kind='sampling', command=command,
                             marker=str(root/f'data/L{size}/status.json')))
    for arm in ARMS:
        for size in sizes:
            command = [sys.executable, '-u', '-m', 'scripts.diagnose_ising_matched_sizes',
                       '--source-run', str(source/arm/'run'), '--data-dir', str(root/f'data/L{size}'),
                       '--output', str(root/f'arms/{arm}/L{size}/collection'),
                       '--expected-checkpoint-sha256', job['training'][arm]['checkpoint_sha256'],
                       '--seed', str(seed)]
            if smoke:
                command.append('--smoke')
            commands.append(dict(name=f'{arm}_L{size}', kind='collection', command=command,
                                 marker=str(root/f'arms/{arm}/L{size}/collection/status.json')))
    command = [sys.executable, '-u', '-m', 'scripts.report_ising_matched_sizes',
               '--root', str(root), '--phase2-root', str(source), '--output', str(root/'report'),
               '--bootstraps', '100' if smoke else '1000']
    if smoke:
        command.append('--smoke')
    commands.append(dict(name='report', kind='report', command=command, marker=str(root/'report/report.json')))
    return dict(schema_version=1, phase=3, created_at=now(), source_root=str(source),
        sequence_root=str(sequence), output=str(root), seed=seed, smoke_only=smoke,
        initial_sizes=list(sizes), terminal_size=target, block_factor=2,
        K=list(K_VALUES), physical_chains_per_K=2 if smoke else 12, samples_per_chain=8 if smoke else 128,
        production_independent_chains=12 if smoke else 72,
        production_saved_configurations=96 if smoke else 9216,
        c_repeats=5, bootstrap_clusters=100 if smoke else 1000,
        equilibration_cluster_updates=16 if smoke else 4096,
        gap_cluster_updates=2 if smoke else 128, pilot_samples_hot_and_cold=16 if smoke else 512,
        total_wolff_single_cluster_updates=960 if smoke else 2310144,
        new_training_updates=0, MI_estimates=0, source_checkpoint_updates=12000,
        source_models=list(ARMS), source_sha256=sources, code_sha256=codes, commands=commands,
        resources=dict(device='cpu', threads=1, nice=10, gpu_used=False,
            min_available_memory_gib=36, max_load_per_core=1.25, max_io_full_avg10=20,
            sequential_children=True),
        protocol='独立尺寸和独立seed生产新数据；同模型固定参数；末尺度及共有中间尺度比较；标签仅事后评价；预先固定预算不挑nu、不自动扩样。',
        uncertainty='bootstrap按独立物理链成组、C重复先平均；跨尺寸独立重采。单训练seed、仅两个初始尺寸、三K、有限算符截断；不能分离所有有限尺寸和RG历史效应。',
        interpretation='同K及相同末尺度不保证相同完整分布；同时报告6偶算符密度差，不将g差异自动解释为Q错误或物理闭包失败；不认证热模。',
        monitoring='只使用现有ising；重要阶段/异常通知，无重要变化安静；失败保留、不重跑、不停止其他任务。')


def verify_pins(spec):
    for path, expected in {**spec['source_sha256'], **spec['code_sha256']}.items():
        if fingerprint(path) != expected:
            raise ValueError(f'来源/代码指纹变化：{path}')


def update_sequence(sequence, **fields):
    path = Path(sequence)/'job.json'
    current = read(path)
    current.update(fields)
    save(path, current)


def execute(args):
    spec = build_plan(args.source_root, args.sequence_root, args.output, seed=args.seed, smoke=args.smoke)
    root, sequence = Path(spec['output']), Path(spec['sequence_root'])
    root.mkdir(parents=True, exist_ok=False)
    (root/'logs').mkdir()
    save(root/'plan.json', spec)
    job = dict(status='running', workflow_complete=False, phase=3, smoke_only=args.smoke,
               parent_pid=os.getpid(), started_at=now(), current=None, stages=[], requires_user_action=False)
    save(root/'job.json', job)
    claimed = False
    try:
        ensure_cpu_policy()
        if not args.smoke:
            with (sequence/'phase3.claim.json').open('x') as stream:
                json.dump(dict(output=str(root), parent_pid=os.getpid(), created_at=now()), stream)
            claimed = True
            update_sequence(sequence, status='phase3_running', workflow_complete=False,
                phase3=dict(root=str(root), job=str(root/'job.json'), plan=str(root/'plan.json')),
                parent_pid=os.getpid(), awaiting_scientific_review=False, current=None)
        for stage in spec['commands']:
            while True:
                resources = resource_snapshot()
                job['current'] = dict(name='wait_resources', next_stage=stage['name'], resources=resources)
                save(root/'job.json', job)
                if resources['ready']:
                    break
                time.sleep(10)
            verify_pins(spec)
            entry = {**stage, 'status': 'running', 'started_at': now(),
                     'log': str(root/'logs'/(stage['name']+'.log'))}
            with Path(entry['log']).open('x') as stream:
                child = subprocess.Popen(entry['command'], stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT, env=child_environment())
                entry['pid'] = child.pid
                job['current'] = entry
                save(root/'job.json', job)
                returncode = child.wait()
            entry.update(status='complete' if returncode == 0 else 'failed',
                         returncode=returncode, ended_at=now())
            job['stages'].append(entry)
            job['current'] = None
            save(root/'job.json', job)
            if returncode:
                raise RuntimeError(f"{entry['name']}退出码{returncode}；保留结果，禁止自动修复/重跑")
            marker = read(entry['marker'])
            if marker.get('status') != 'complete' or not (marker.get('workflow_complete') is True
                                                          or marker.get('report_complete') is True):
                raise ValueError(f"{entry['name']}缺少完整完成标志")
        verify_pins(spec)
        public = {str(p.relative_to(root)): fingerprint(p) for p in root.rglob('*')
                  if p.is_file() and p.suffix in ('.json', '.npz', '.png')
                  and p.name not in ('job.json',) and 'logs' not in p.parts}
        job.update(status='complete', workflow_complete=True, requires_user_action=False,
                   ended_at=now(), public_files=public, new_training_updates=0, MI_estimates=0)
        save(root/'job.json', job)
        if claimed:
            update_sequence(sequence, status='complete', workflow_complete=True, phase3_complete=True,
                            pending_experiments=[], awaiting_scientific_review=False,
                            current=None, phase3_finished_at=now(), requires_user_action=False)
        print(json.dumps(dict(status='complete', output=str(root), smoke_only=args.smoke)), flush=True)
        return 0
    except BaseException as error:
        job.update(status='failed', workflow_complete=False, requires_user_action=True,
                   error=f'{type(error).__name__}: {error}', ended_at=now(),
                   active_children_preserved=job.get('current'))
        save(root/'job.json', job)
        if claimed:
            update_sequence(sequence, status='phase3_failed', workflow_complete=False,
                            requires_user_action=True, error=job['error'])
        traceback.print_exc()
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('source-root', 'sequence-root', 'output'):
        parser.add_argument('--'+option, type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261016)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args(argv)
    spec = build_plan(args.source_root, args.sequence_root, args.output, seed=args.seed, smoke=args.smoke)
    if args.dry_run:
        print(json.dumps(spec, indent=2, ensure_ascii=False)); return 0
    if args.detach:
        logs = Path.cwd()/'logs'; logs.mkdir(exist_ok=True)
        receipt, log = logs/(args.output.name+'.launch.json'), logs/(args.output.name+'.log')
        command = ['nice', '-n', '10', sys.executable, '-u', '-m', 'scripts.run_ising_matched_sizes',
            '--source-root', str(args.source_root.resolve()), '--sequence-root', str(args.sequence_root.resolve()),
            '--output', str(args.output.resolve()), '--seed', str(args.seed)]
        if args.smoke:
            command.append('--smoke')
        with receipt.open('x') as stream:
            json.dump(dict(status='launching', command=command, created_at=now()), stream)
        environment = child_environment()
        with log.open('x') as stream:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                     start_new_session=True, env=environment)
        result = dict(status='launched', parent_pid=child.pid, output=str(args.output.resolve()),
                      log=str(log), command=command, launched_at=now())
        save(receipt, result)
        print(json.dumps(result, indent=2, ensure_ascii=False)); return 0
    return execute(args)


if __name__ == '__main__':
    raise SystemExit(main())
