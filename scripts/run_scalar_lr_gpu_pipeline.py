"""Prepare audited data, then run the existing selection-free GPU LR pilot.

This wrapper provides a durable job status for detached server execution. It
does not change the sampler, QCR recipe, schedules, or final-report selection.
Existing outputs and incomplete cache/sidecar pairs fail without a retry.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.run_scalar_lr_sweep import check_data


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def save_status(output, status):
    target = output / 'job.json'
    temporary = output / 'job.json.tmp'
    temporary.write_text(json.dumps(status, indent=2, allow_nan=False) + '\n')
    temporary.replace(target)


def stage(output, status, name, command):
    record = dict(name=name, command=command, started_at=utc_now(), status='running',
                  log=str(output / 'logs' / (name + '.log')))
    status['current'] = record
    with Path(record['log']).open('x') as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        record['pid'] = process.pid
        save_status(output, status)
        print(json.dumps(record), flush=True)
        code = process.wait()
    record.update(status='complete' if code == 0 else 'failed', returncode=code,
                  ended_at=utc_now())
    status['stages'].append(record)
    status['current'] = None
    save_status(output, status)
    if code:
        raise RuntimeError(f'{name} exited {code}; see {record["log"]}; no automatic retry')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=12000)
    parser.add_argument('--seed', type=int, default=91)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--max-wall-seconds-per-run', type=float, default=2700.)
    args = parser.parse_args(argv)
    # Reuse the same validated plan; do not maintain a second recipe here.
    from scripts.run_scalar_lr_sweep import build_plan
    output, cache = args.output.resolve(), args.cache.resolve()
    plan_args = argparse.Namespace(**(vars(args) | dict(output=output / 'sweep', cache=cache)))
    build_plan(plan_args)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite or resume {output}')
    sidecar = cache.with_suffix('.json')
    if cache.exists() != sidecar.exists():
        raise ValueError('Incomplete existing data/sidecar pair; inspect it, do not regenerate over it')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'logs').mkdir()
    status = dict(status='running', pid=os.getpid(), started_at=utc_now(),
                  git_commit=revision, cache=str(cache), sweep=str(output / 'sweep'),
                  device=args.device, stages=[], current=None,
                  protocol='Existing default b3 scalar LR pilot; labels evaluation only; no automatic retry')
    save_status(output, status)
    try:
        if not cache.exists():
            stage(output, status, 'data_generation', [sys.executable, '-u', '-m',
                  'scripts.prepare_ising2d_lr_data', '--output', str(cache), '--seed', str(args.seed)])
        status['data'] = check_data(cache)
        save_status(output, status)
        stage(output, status, 'lr_sweep', [sys.executable, '-u', '-m',
              'scripts.run_scalar_lr_sweep', '--cache', str(cache), '--output', str(output / 'sweep'),
              '--device', args.device, '--epochs', str(args.epochs), '--seed', str(args.seed),
              '--threads', str(args.threads), '--max-wall-seconds-per-run',
              str(args.max_wall_seconds_per_run)])
    except Exception as error:
        status.update(status='failed', error=f'{type(error).__name__}: {error}', ended_at=utc_now())
        save_status(output, status)
        raise
    status.update(status='complete', ended_at=utc_now())
    save_status(output, status)
    print(json.dumps(dict(status='complete', output=str(output))), flush=True)
    return status


if __name__ == '__main__':
    main()
