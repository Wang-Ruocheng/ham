"""用户授权的固定 Q/C、R 局部响应损失对照；独占输出、顺序训练与中文报告。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def utc_now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def build_plan(args):
    source = args.source_run.resolve()
    report = args.source_report.resolve()
    output = args.output.resolve()
    checkpoint = source / f'training_step_{args.checkpoint}.pt'
    receipt = report.parent / 'checkpoint_summary.json'
    encoding = report / 'encodings.npz'
    for path in (checkpoint, receipt, encoding, source / 'manifest.json'):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists():
        raise FileExistsError(f'拒绝覆盖或重复启动：{output}')
    if args.updates < 1 or args.checkpoint < 1:
        raise ValueError('需要正的源步数及固定 R 更新预算')
    if not (0 < args.learning_rate < float('inf') and 0 < args.alpha < float('inf')):
        raise ValueError('需要有限正学习率及响应权重')
    saved = json.loads(receipt.read_text())
    sha = fingerprint(checkpoint)
    if (saved.get('status') != 'complete' or saved.get('actual_epochs') != args.checkpoint
            or saved.get('checkpoint_sha256') != sha):
        raise ValueError('源模型与已完成正式编码回执不符')
    python = sys.executable
    train = [python, '-u', '-m', 'experiments.ising_response_loss_ablation',
             '--source-run', str(source), '--source-report', str(report),
             '--checkpoint', str(args.checkpoint), '--output', str(output / 'training'),
             '--updates', str(args.updates), '--alpha', str(args.alpha),
             '--learning-rate', str(args.learning_rate), '--threads', '1', '--neighbors', '7']
    report_command = [python, '-u', '-m', 'scripts.report_ising_response_ablation',
                      '--run', str(output / 'training'), '--output', str(output / 'report')]
    project = Path(__file__).resolve().parents[1]
    code_paths = ('scripts/run_ising_response_loss_ablation.py',
                  'experiments/ising_response_loss_ablation.py',
                  'scripts/report_ising_response_ablation.py',
                  'recurrent_rg/losses.py', 'recurrent_rg/checkpoints.py',
                  'recurrent_rg/model_nd.py', 'scripts/report_ising_zh.py',
                  'scripts/report_ising_scalar_exponents.py',
                  'scripts/report_k_supervised_continuation.py')
    return dict(schema_version=1, source_run=str(source), source_report=str(report),
                source_checkpoint=str(checkpoint), source_checkpoint_sha256=sha,
                source_encodings_sha256=fingerprint(encoding),
                source_report_receipt_sha256=fingerprint(receipt),
                source_epoch=args.checkpoint, extra_R_updates=args.updates,
                alpha=args.alpha, learning_rate=args.learning_rate,
                code_sha256={str(project / name): fingerprint(project / name) for name in code_paths},
                output=str(output), commands=[train, report_command],
                execution='CPU 单线程、nice10；不占用 GPU',
                protocol='两臂同源 R，全新 Adam、相同固定更新预算；Q/C 固定；只使用保存编码；不按 TEST 或指数选择',
                prior_experiment='已按用户要求停止进程组32429；日志9400、最新完整编号9000；不恢复旧流程')


def execute(args):
    plan = build_plan(args)
    root = Path(plan['output'])
    root.mkdir(parents=True, exist_ok=False)
    (root / 'logs').mkdir()
    save_json(root / 'plan.json', plan)
    job = dict(status='running', workflow_complete=False, parent_pid=os.getpid(),
               started_at=utc_now(), current=None, stages=[], protocol=plan)
    path = root / 'job.json'
    save_json(path, job)
    try:
        for name, command in zip(('training', 'report'), plan['commands']):
            if (fingerprint(plan['source_checkpoint']) != plan['source_checkpoint_sha256'] or
                    fingerprint(Path(plan['source_report']) / 'encodings.npz') != plan['source_encodings_sha256']):
                raise ValueError('源模型或保存编码在阶段之间发生变化')
            for code_path, digest in plan['code_sha256'].items():
                if fingerprint(code_path) != digest:
                    raise ValueError(f'实验代码在阶段之间发生变化：{code_path}')
            log = root / 'logs' / f'{name}.log'
            start = time.monotonic()
            with log.open('x') as handle:
                child = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
                stage = dict(name=name, command=command, pid=child.pid, log=str(log),
                             status='running', started_at=utc_now())
                job.update(current=stage, updated_at=utc_now())
                save_json(path, job)
                code = child.wait()
            stage.update(status='complete' if code == 0 else 'failed', returncode=code,
                         seconds=time.monotonic() - start, finished_at=utc_now())
            job['stages'].append(stage)
            job.update(current=None, updated_at=utc_now())
            save_json(path, job)
            if code:
                raise RuntimeError(f'{name} returncode={code}；保留结果，不自动修复或重跑')
        if not (root / 'report' / 'comparison.json').is_file():
            raise FileNotFoundError('报告缺少 comparison.json')
        train_job = json.loads((root / 'training' / 'training_job.json').read_text())
        if train_job.get('status') != 'complete':
            raise ValueError('训练子任务未完整完成')
        job.update(status='complete', workflow_complete=True, finished_at=utc_now(),
                   report=str(root / 'report'), requires_user_action=False)
        save_json(path, job)
    except BaseException as exc:
        job.update(status='failed', workflow_complete=False, error=str(exc),
                   requires_user_action=True, finished_at=utc_now())
        save_json(path, job)
        traceback.print_exc()
        return 1
    print(json.dumps(dict(status=job['status'], output=str(root)), ensure_ascii=False), flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--source-report', type=Path, required=True)
    parser.add_argument('--checkpoint', type=int, default=9000)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=6000)
    parser.add_argument('--alpha', type=float, default=1.)
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    plan = build_plan(args)
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    if args.detach:
        if args.execute:
            raise ValueError('不能同时指定 detach 与 execute')
        logs = Path.cwd() / 'logs'
        logs.mkdir(exist_ok=True)
        receipt = logs / (args.output.name + '.launch.json')
        log = logs / (args.output.name + '.log')
        command = ['nice', '-n', '10', sys.executable, '-u', '-m',
                   'scripts.run_ising_response_loss_ablation', '--source-run', str(args.source_run.resolve()),
                   '--source-report', str(args.source_report.resolve()), '--checkpoint', str(args.checkpoint),
                   '--output', str(args.output.resolve()), '--updates', str(args.updates),
                   '--alpha', str(args.alpha), '--learning-rate', str(args.learning_rate), '--execute']
        for module in ('experiments/ising_response_loss_ablation.py',
                       'scripts/report_ising_response_ablation.py'):
            if not Path(module).is_file():
                raise FileNotFoundError(module)
        with receipt.open('x') as handle:
            handle.write(json.dumps(dict(status='launching', created_at=utc_now(), command=command,
                                         source_checkpoint_sha256=plan['source_checkpoint_sha256']),
                                    ensure_ascii=False, indent=2) + '\n')
        environment = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                           OPENBLAS_NUM_THREADS='1', MPLBACKEND='Agg')
        with log.open('x') as handle:
            child = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True, env=environment)
        result = dict(status='launched', parent_pid=child.pid, launched_at=utc_now(),
                      output=plan['output'], log=str(log), command=command,
                      source_checkpoint_sha256=plan['source_checkpoint_sha256'])
        save_json(receipt, result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0
    return execute(args)


if __name__ == '__main__':
    raise SystemExit(main())
