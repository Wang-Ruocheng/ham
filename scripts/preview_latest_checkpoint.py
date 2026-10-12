"""Low-priority, read-only checkpoint diagnostics alongside a live trainer.

Uses immutable numbered checkpoints, never the trainer's in-memory model.
All outputs are isolated under monitor_previews; no MI estimation or training.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def latest_saved_checkpoint(run, *, min_age=10., now=None):
    """Avoid files just created by a non-atomic checkpoint writer."""
    now = time.time() if now is None else now
    candidates = []
    for path in Path(run).glob('training_step_*.pt'):
        match = re.fullmatch(r'training_step_(\d+)\.pt', path.name)
        if match:
            stat = path.stat()
            if stat.st_size and now - stat.st_mtime >= min_age:
                candidates.append((int(match[1]), path))
    if not candidates:
        raise FileNotFoundError('No settled numbered checkpoint yet; do not stop or change training')
    return max(candidates, key=lambda item: item[0])


def digest(path):
    # hashlib.file_digest is not available in the server's Python 3.10.
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def preview_stages(run, epoch, out, theory_dim, coupling_names=None, training_driver=None,
                   physical_model='', categorical_states=0):
    """Choose reports without importing Torch or changing trainer state."""
    base = ['--run', str(run), '--checkpoint', str(epoch), '--output', str(out)]
    if theory_dim == 1:
        scalar_ising = (tuple(coupling_names or ()) == ('K',) and not categorical_states
                        and physical_model == 'ising2d')
        stages = [
            ('scalar', [sys.executable, '-m', 'scripts.report_scalar_joint', *base,
                        '--threads', '1', '--sampling-repeats', '0']),
        ]
        # The default single-Ising LR pilot already gets the complete scalar
        # R/drift/K panels from the exponent report; do not route it through
        # a historical dual-Ising phase annotator.
        if not scalar_ising:
            stages.append(('phase', [sys.executable, '-m',
                'scripts.plot_dual_ising_scalar', *base]))
        positive_historical = (tuple(coupling_names or ()) == ('K',) and not categorical_states
                               and 'positive k' in physical_model.lower())
        if positive_historical:
            stages[1][1].append('--positive-ising')
        if scalar_ising or positive_historical:
            stages.append(('exponents', [sys.executable, '-m',
                'scripts.report_ising_scalar_exponents', *base]))
    elif theory_dim == 2:
        report = ('scripts.report_ising2d_KB_vector' if tuple(coupling_names or ()) == ('K', 'B')
                  else 'scripts.report_dual_ising_vector')
        stages = [('phase', [sys.executable, '-m', report,
                             *base, '--threads', '1'])]
    else:
        raise ValueError('Checkpoint previews support dim g = 1 or 2')
    loss_report = ('scripts.report_r_plateau_loss' if training_driver == 'frozen_qc_r_plateau'
                   else 'scripts.report_mi_training')
    return stages + [('loss', [sys.executable, '-m', loss_report,
                               *base, '--threads', '1'])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    epoch, checkpoint = latest_saved_checkpoint(run)
    manifest = json.loads((run / 'manifest.json').read_text())
    config = manifest['config']
    theory_dim = config['model']['theory_dim']
    if theory_dim not in (1, 2):
        raise ValueError('Checkpoint previews support dim g = 1 or 2')
    out = run / 'monitor_previews' / f'epoch_{epoch:06d}'
    status_file = out / 'preview.json'
    checkpoint_sha = digest(checkpoint)
    if status_file.exists():
        saved = json.loads(status_file.read_text())
        if saved.get('phase') == 'completed' and saved.get('checkpoint_sha256') == checkpoint_sha:
            print(json.dumps(dict(reused=True, **saved), indent=2))
            return 0
        raise RuntimeError(f'Existing unfinished/different preview: {status_file}; training is untouched')
    out.mkdir(parents=True, exist_ok=False)
    status = dict(epoch=epoch, checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha,
                  output=str(out), phase='starting', started_at=time.time(),
                  inference_threads=1, extra_sampling_repeats=0,
                  note='Single stochastic-C diagnostic draw (seed12345); not a training or MI calculation')
    status_file.write_text(json.dumps(status, indent=2) + '\n')
    os.nice(10)
    env = dict(os.environ)
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[key] = '1'
    stages = preview_stages(run, epoch, out, theory_dim, config['data']['coupling_names'],
                            manifest.get('training_driver'),config['data'].get('physical_model',''),
                            config['model'].get('categorical_states',0))
    started = time.perf_counter()
    try:
        for name, command in stages:
            status.update(phase=name)
            status_file.write_text(json.dumps(status, indent=2) + '\n')
            with (out / f'{name}.log').open('x') as log:
                subprocess.run(command, cwd=Path(__file__).resolve().parents[1],
                               env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        if digest(checkpoint) != checkpoint_sha:
            raise RuntimeError('Checkpoint changed during diagnostics; do not use this preview')
        status.update(phase='completed', elapsed_seconds=time.perf_counter()-started)
        status_file.write_text(json.dumps(status, indent=2) + '\n')
        print(json.dumps(status, indent=2))
        return 0
    except Exception as error:
        status.update(phase='failed', error=repr(error), elapsed_seconds=time.perf_counter()-started)
        status_file.write_text(json.dumps(status, indent=2) + '\n')
        raise


if __name__ == '__main__':
    sys.exit(main())
