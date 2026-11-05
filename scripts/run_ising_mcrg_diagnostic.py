"""独占目录执行固定无监督模型的只读采集和报告；不训练、不自动重跑。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from scripts.diagnose_ising_mcrg import sha256_file
from scripts.run_ising_response_loss_ablation import save_json, utc_now


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--k-count', type=int, default=3)
    p.add_argument('--c-repeats', type=int, default=5)
    p.add_argument('--bootstraps', type=int, default=2000)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args(argv)
    root, source, data = a.output.resolve(), a.source_run.resolve(), a.data.resolve()
    if root.exists() or root.is_symlink():
        raise FileExistsError(f'Refusing overwrite or duplicate launch: {root}')
    if source == root or source in root.parents or root in source.parents:
        raise ValueError('Output must not overlap source run')
    project = Path(__file__).resolve().parents[1]
    code_files = ('scripts/diagnose_ising_mcrg.py', 'scripts/report_ising_mcrg.py',
                  'scripts/run_ising_mcrg_diagnostic.py', 'recurrent_rg/model_nd.py',
                  'recurrent_rg/checkpoints.py', 'recurrent_rg/adapters.py',
                  'recurrent_rg/binary_pooling.py', 'recurrent_rg/geometry.py')
    hashes = {str(project / name): sha256_file(project / name) for name in code_files}
    checkpoint = source / 'training_step_12000.pt'
    checkpoint_hash = sha256_file(checkpoint)
    collect = [sys.executable, '-u', '-m', 'scripts.diagnose_ising_mcrg',
               '--source-run', str(source), '--data', str(data), '--output', str(root / 'collection'),
               '--k-count', str(a.k_count), '--c-repeats', str(a.c_repeats), '--seed', '12345']
    if a.smoke:
        collect.append('--smoke')
    report = [sys.executable, '-u', '-m', 'scripts.report_ising_mcrg',
              '--run', str(root / 'collection'), '--output', str(root / 'report'),
              '--bootstraps', str(a.bootstraps)]
    plan = dict(source_run=str(source), source_checkpoint_sha256=checkpoint_hash,
        data=str(data), k_count=a.k_count, c_repeats=a.c_repeats, bootstrap_clusters=a.bootstraps,
        smoke=a.smoke, commands=[collect, report], code_sha256=hashes,
        protocol='固定无监督12000模型；CPU1、nice10；不优化、不估MI、不生成物理样本；只保存公共算符统计。')
    root.mkdir(parents=True, exist_ok=False)
    (root / 'logs').mkdir()
    save_json(root / 'plan.json', plan)
    job = dict(status='running', workflow_complete=False, parent_pid=os.getpid(),
               started_at=utc_now(), current=None, stages=[], plan=plan)
    save_json(root / 'job.json', job)
    try:
        for name, command in zip(('collection', 'report'), (collect, report)):
            for path, expected in hashes.items():
                if sha256_file(path) != expected:
                    raise ValueError(f'Diagnostic code changed: {path}')
            if sha256_file(checkpoint) != checkpoint_hash:
                raise ValueError('Historical checkpoint changed')
            stage = dict(name=name, command=command, status='running', started_at=utc_now())
            with (root / 'logs' / (name + '.log')).open('x') as stream:
                started = time.monotonic()
                child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
                stage['pid'] = child.pid
                job['current'] = stage
                save_json(root / 'job.json', job)
                code = child.wait()
            stage.update(returncode=code, status='complete' if code == 0 else 'failed',
                         elapsed_seconds=time.monotonic() - started, finished_at=utc_now())
            job['stages'].append(stage)
            job['current'] = None
            save_json(root / 'job.json', job)
            if code:
                raise RuntimeError(f'{name} failed with returncode {code}; preserved without retry')
        summary = json.loads((root / 'report/comparison.json').read_text())
        if summary.get('status') != 'complete':
            raise ValueError('Report completion not confirmed')
        files = [path for folder in ('collection', 'report') for path in (root / folder).iterdir()
                 if path.is_file() and path.suffix in ('.json', '.npz', '.png')]
        job.update(status='complete', workflow_complete=True, requires_user_action=False,
                   finished_at=utc_now(), report_files={str(path.relative_to(root)): sha256_file(path)
                                                       for path in files})
        save_json(root / 'job.json', job)
    except BaseException as error:
        job.update(status='failed', workflow_complete=False, error=str(error),
                   requires_user_action=True, finished_at=utc_now())
        save_json(root / 'job.json', job)
        traceback.print_exc()
        return 1
    print(json.dumps(dict(status='complete', output=str(root)), ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
