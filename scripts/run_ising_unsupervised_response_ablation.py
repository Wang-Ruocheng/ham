"""严格不使用训练标签的固定 Q/C、R 局部割线消融；独占输出与顺序报告。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from scripts.run_ising_response_loss_ablation import fingerprint, save_json, utc_now


def build_plan(args):
    source, report, output = (p.resolve() for p in (args.source_run, args.source_report, args.output))
    checkpoint = source / f'training_step_{args.checkpoint}.pt'
    encoding, manifest_path = report / 'encodings.npz', source / 'manifest.json'
    receipt_path = source.parent / 'job.json'
    for path in (checkpoint, source / 'checkpoint_last.pt', encoding, manifest_path, receipt_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists():
        raise FileExistsError(f'拒绝覆盖或重复启动：{output}')
    if args.updates < 1 or args.checkpoint != 12000:
        raise ValueError('使用固定的无监督12000源模型及正R更新预算')
    if not (0 < args.learning_rate < float('inf') and 0 < args.alpha < float('inf')):
        raise ValueError('需要有限正学习率和预先固定的响应权重')
    manifest = json.loads(manifest_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    if (manifest.get('labels_used_in_training') is not False or manifest.get('q_supervision')
            or receipt.get('status') != 'complete' or receipt.get('workflow_complete') is not True
            or receipt.get('training', {}).get('actual_epochs') != args.checkpoint):
        raise ValueError('必须使用完整完成且没有K监督的源实验')
    sha = fingerprint(checkpoint)
    encoding_sha = fingerprint(encoding)
    expected = receipt['reports']['files'].get('full_report/encodings.npz')
    if encoding_sha != expected or fingerprint(manifest_path) != receipt['training']['manifest_sha256']:
        raise ValueError('源编码/清单与正式完成回执不符')
    python = sys.executable
    train = [python, '-u', '-m', 'experiments.ising_unsupervised_response_ablation',
             '--source-run', str(source), '--source-report', str(report), '--checkpoint', str(args.checkpoint),
             '--output', str(output / 'training'), '--updates', str(args.updates),
             '--alpha', str(args.alpha), '--learning-rate', str(args.learning_rate), '--threads', '1']
    report_command = [python, '-u', '-m', 'scripts.report_ising_unsupervised_response_ablation',
                      '--run', str(output / 'training'), '--output', str(output / 'report')]
    project = Path(__file__).resolve().parents[1]
    names = ('scripts/run_ising_unsupervised_response_ablation.py',
             'experiments/ising_unsupervised_response_ablation.py',
             'scripts/report_ising_unsupervised_response_ablation.py',
             'scripts/run_ising_response_loss_ablation.py',
             'experiments/ising_response_loss_ablation.py',
             'scripts/report_ising_response_ablation.py',
             'scripts/report_k_supervised_continuation.py',
             'scripts/report_ising_scalar_exponents.py', 'scripts/report_ising_zh.py',
             'recurrent_rg/losses.py', 'recurrent_rg/checkpoints.py', 'recurrent_rg/model_nd.py')
    return dict(schema_version=1, source_run=str(source), source_report=str(report),
                source_checkpoint=str(checkpoint), source_checkpoint_sha256=sha,
                source_encodings_sha256=encoding_sha, source_job_sha256=fingerprint(receipt_path),
                source_epoch=args.checkpoint, extra_R_updates=args.updates,
                alpha=args.alpha, learning_rate=args.learning_rate,
                output=str(output), commands=[train, report_command],
                code_sha256={str(project / n): fingerprint(project / n) for n in names},
                K_label_access_in_training=False, Kc_or_theory_target=False,
                execution='CPU单线程、nice10；不占GPU',
                protocol='源Q/C/R完全无监督；冻结Q/C、相同R初值和fresh Adam、相同固定预算；训练只读取g轨迹，不读取K标签；报告阶段才使用K评价',
                loss_variant='无K局部割线残差一致性；不是此前依赖K的导数loss，也不是独立热响应观测')


def execute(args):
    plan = build_plan(args)
    root = Path(plan['output'])
    root.mkdir(parents=True, exist_ok=False)
    (root / 'logs').mkdir()
    save_json(root / 'plan.json', plan)
    job = dict(status='running', workflow_complete=False, parent_pid=os.getpid(),
               started_at=utc_now(), current=None, stages=[], protocol=plan)
    status_path = root / 'job.json'
    save_json(status_path, job)
    try:
        for name, command in zip(('training', 'report'), plan['commands']):
            if (fingerprint(plan['source_checkpoint']) != plan['source_checkpoint_sha256']
                    or fingerprint(Path(plan['source_report']) / 'encodings.npz') != plan['source_encodings_sha256']):
                raise ValueError('源模型或编码发生变化')
            for path, sha in plan['code_sha256'].items():
                if fingerprint(path) != sha:
                    raise ValueError(f'代码在阶段之间发生变化：{path}')
            log = root / 'logs' / f'{name}.log'
            started = time.monotonic()
            with log.open('x') as handle:
                child = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
                stage = dict(name=name, command=command, pid=child.pid, log=str(log),
                             status='running', started_at=utc_now())
                job.update(current=stage, updated_at=utc_now())
                save_json(status_path, job)
                code = child.wait()
            stage.update(status='complete' if code == 0 else 'failed', returncode=code,
                         seconds=time.monotonic()-started, finished_at=utc_now())
            job['stages'].append(stage)
            job.update(current=None, updated_at=utc_now())
            save_json(status_path, job)
            if code:
                raise RuntimeError(f'{name} returncode={code}；保留结果，不自动重跑')
        train_job = json.loads((root / 'training/training_job.json').read_text())
        comparison = json.loads((root / 'report/comparison.json').read_text())
        if train_job.get('status') != 'complete' or comparison.get('status') != 'complete':
            raise ValueError('子阶段完整性检查失败')
        job.update(status='complete', workflow_complete=True, requires_user_action=False,
                   finished_at=utc_now(), report=str(root / 'report'))
        save_json(status_path, job)
    except BaseException as exc:
        job.update(status='failed', workflow_complete=False, error=str(exc),
                   requires_user_action=True, finished_at=utc_now())
        save_json(status_path, job)
        traceback.print_exc()
        return 1
    print(json.dumps(dict(status=job['status'], output=str(root)), ensure_ascii=False), flush=True)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', type=Path, required=True)
    p.add_argument('--source-report', type=Path, required=True)
    p.add_argument('--checkpoint', type=int, default=12000)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--updates', type=int, default=6000)
    p.add_argument('--alpha', type=float, default=1.)
    p.add_argument('--learning-rate', type=float, default=.001)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--detach', action='store_true')
    p.add_argument('--execute', action='store_true')
    args = p.parse_args(argv)
    plan = build_plan(args)
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    if args.detach:
        if args.execute:
            raise ValueError('不能同时detach和execute')
        logs = Path.cwd() / 'logs'
        logs.mkdir(exist_ok=True)
        receipt, log = (logs / (args.output.name + suffix) for suffix in ('.launch.json', '.log'))
        command = ['nice', '-n', '10', sys.executable, '-u', '-m',
                   'scripts.run_ising_unsupervised_response_ablation',
                   '--source-run', str(args.source_run.resolve()), '--source-report', str(args.source_report.resolve()),
                   '--checkpoint', str(args.checkpoint), '--output', str(args.output.resolve()),
                   '--updates', str(args.updates), '--alpha', str(args.alpha),
                   '--learning-rate', str(args.learning_rate), '--execute']
        with receipt.open('x') as handle:
            handle.write(json.dumps(dict(status='launching', command=command, created_at=utc_now()),
                                    ensure_ascii=False, indent=2)+'\n')
        environment = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MPLBACKEND='Agg')
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
