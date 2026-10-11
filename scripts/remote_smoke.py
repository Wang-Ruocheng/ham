"""Run an isolated three-update default QCR smoke; optionally check CUDA separately."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback


REPOSITORY = Path(__file__).resolve().parents[1]
CONFIG_TEMPLATE = REPOSITORY / 'configs' / 'remote_smoke_225.json'


def write_json(path, values):
    path.write_text(json.dumps(values, indent=2, allow_nan=False) + '\n')


def git_result(*arguments):
    result = subprocess.run(
        ['git', *arguments], cwd=REPOSITORY, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def environment_record():
    versions = {}
    for name in ('torch', 'numpy', 'scipy', 'matplotlib', 'numba'):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    porcelain = git_result('status', '--porcelain')
    return {
        'git_sha': git_result('rev-parse', 'HEAD'),
        'git_dirty': None if porcelain is None else bool(porcelain),
        'python': sys.version,
        'python_executable': sys.executable,
        'platform': platform.platform(),
        'package_versions': versions,
        'thread_environment': {
            key: os.environ[key]
            for key in ('KMP_USE_SHM', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')
            if key in os.environ
        },
        'repository': str(REPOSITORY),
    }


def run_command(command, log_path):
    label = command[2] if command[1] == '-m' else Path(command[1]).name
    print(f'Running {label}; log: {log_path}', flush=True)
    with log_path.open('w') as log:
        result = subprocess.run(
            command, cwd=REPOSITORY, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    if result.returncode:
        tail = '\n'.join(log_path.read_text().splitlines()[-25:])
        raise RuntimeError(f'Command exited {result.returncode}: {command}\n{tail}')
    return {'command': command, 'returncode': result.returncode, 'log': str(log_path)}


def check_cuda():
    """Check GPU 0 independently; never move QCR or its training data to CUDA."""
    import torch

    result = {
        'purpose': 'Independent GPU 0 matrix validation; QCR training remains on CPU',
        'torch_version': torch.__version__,
        'torch_cuda_version': torch.version.cuda,
        'cuda_available': torch.cuda.is_available(),
        'requested_device': 'cuda:0',
    }
    try:
        if not result['cuda_available']:
            raise RuntimeError('CUDA is unavailable in this Python environment')
        with torch.cuda.device(0), torch.no_grad():
            x = torch.ones((256, 256), dtype=torch.float64, device='cuda:0')
            y = x @ x
            torch.cuda.synchronize(0)
            error = float((y - 256.).abs().max())
            if error != 0.:
                raise RuntimeError(f'CUDA matrix validation maximum error: {error}')
            result.update(
                status='passed', actual_device=str(y.device),
                device_name=torch.cuda.get_device_name(0),
                device_capability=list(torch.cuda.get_device_capability(0)),
                matrix_shape=[256, 256], dtype=str(y.dtype), max_absolute_error=error,
            )
    except Exception as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
    return result


def generate_cache(output, config):
    import torch
    from recurrent_rg.ising2d_field import sample_field_chain

    train_couplings = [.1, .3, .5, .7]
    test_couplings = [.2, .6]
    sampling = {
        'sampler': 'Swendsen-Wang cluster heat bath; sample_field_chain',
        'size': 36, 'samples': 8, 'equilibration_sweeps': 64, 'sweeps_between': 2,
        'B': 0., 'train_K': train_couplings, 'test_K': test_couplings,
        'train_seeds': [910001 + i for i in range(len(train_couplings))],
        'test_seeds': [920001 + i for i in range(len(test_couplings))],
        'labels_used_in_training': False,
        'interpretation': 'Execution smoke only; short chains and three updates do not '
                          'establish equilibration, physical convergence, or reproduction',
    }

    def sample(couplings, seeds):
        configurations = torch.stack([
            torch.from_numpy(sample_field_chain(
                k, 0., sampling['size'], sampling['samples'],
                sampling['equilibration_sweeps'], sampling['sweeps_between'], seed,
            )).unsqueeze(-1)
            for k, seed in zip(couplings, seeds)
        ])
        return configurations, torch.tensor(couplings, dtype=torch.float64)

    train_x, train_y = sample(train_couplings, sampling['train_seeds'])
    test_x, test_y = sample(test_couplings, sampling['test_seeds'])
    cache = output / 'ensembles.pt'
    torch.save((train_x, train_y, test_x, test_y), cache)
    config['data']['cache'] = str(cache)
    sampling.update(train_shape=list(train_x.shape), test_shape=list(test_x.shape),
                    cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest())
    write_json(output / 'sampling.json', sampling)
    write_json(output / 'config.json', config)
    return sampling


def verify_run(run):
    import torch

    summary = json.loads((run / 'summary.json').read_text())
    manifest = json.loads((run / 'manifest.json').read_text())
    # No map_location: inspect the devices actually serialized by the trainer.
    checkpoint = torch.load(run / 'checkpoint_last.pt', weights_only=True)
    devices = sorted({str(t.device) for t in checkpoint['model'].values()})
    if devices != ['cpu']:
        raise RuntimeError(f'Unexpected QCR checkpoint devices: {devices}')
    if summary['actual_epochs'] != 3 or checkpoint['epoch'] != 3:
        raise RuntimeError('Smoke did not complete exactly three updates')
    for row in summary['history']:
        for key in ('train_total', 'test_total', 'train_rsmi_bits', 'test_rsmi_bits'):
            if not math.isfinite(row[key]):
                raise RuntimeError(f'Nonfinite {key} at update {row["epoch"]}')
    if manifest['labels_used_in_training'] or not manifest['independent_c']:
        raise RuntimeError('Smoke violated label or gradient ownership conventions')
    expected = [
        'checkpoint_last.pt', 'qcr_model.pt', 'summary.json', 'history_progress.json',
        'report/R_vs_g.png', 'report/coupling_vs_g.png', 'report/trajectories.png',
        'report/metrics.json', 'report/trajectories.npz', 'report/loss_comparison.png',
        'report/training_comparison.json',
    ]
    for name in expected:
        if not (run / name).is_file() or (run / name).stat().st_size == 0:
            raise RuntimeError(f'Missing or empty smoke artifact: {name}')
    return {
        'status': 'passed', 'actual_epochs': summary['actual_epochs'],
        'actual_qcr_device': devices[0], 'checkpoint_parameter_devices': devices,
        'estimator': manifest['estimator'], 'environment_mode': manifest['environment_mode'],
        'buffer': manifest['buffer'], 'independent_c': manifest['independent_c'],
        'labels_used_in_training': manifest['labels_used_in_training'],
        'artifacts': [str(run / name) for name in expected],
        'final_train_total': summary['history'][-1]['train_total'],
        'final_test_total': summary['history'][-1]['test_total'],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='New isolated output directory; existing paths are refused')
    parser.add_argument('--check-cuda', action='store_true',
                        help='Also validate a GPU 0 matrix multiplication independently of CPU QCR')
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error(f'Refusing to overwrite existing output: {output}')
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(REPOSITORY))
    started = time.perf_counter()
    status = {
        'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(),
        'environment': environment_record(), 'output': str(output),
        'qcr_training_device': 'cpu', 'cuda_check_requested': args.check_cuda,
        'interpretation': 'Execution smoke only; not a physics result or reproduction',
    }
    status_path = output / 'smoke_status.json'
    write_json(status_path, status)
    try:
        import torch

        status['environment']['torch_runtime_version'] = torch.__version__
        status['environment']['torch_cuda_version'] = torch.version.cuda
        config = json.loads(CONFIG_TEMPLATE.read_text())
        status['config_template'] = str(CONFIG_TEMPLATE)
        status['sampling'] = generate_cache(output, config)
        write_json(status_path, status)
        run = output / 'run'
        status['training'] = run_command(
            [sys.executable, '-m', 'recurrent_rg', 'train', '--config',
             str(output / 'config.json'), '--output', str(run)], output / 'train.log',
        )
        write_json(status_path, status)
        status['loss_report'] = run_command(
            [sys.executable, '-m', 'scripts.report_mi_training',
             '--run', str(run), '--threads', str(config['training']['threads'])],
            output / 'loss_report.log',
        )
        status['qcr_verification'] = verify_run(run)
        if args.check_cuda:
            status['cuda_check'] = check_cuda()
            write_json(output / 'cuda_check.json', status['cuda_check'])
        status['status'] = (
            'failed' if args.check_cuda and status['cuda_check']['status'] != 'passed' else 'passed'
        )
    except Exception as error:
        status.update(status='failed', error=f'{type(error).__name__}: {error}')
        traceback.print_exc()
    status.update(finished_at=datetime.now(timezone.utc).isoformat(),
                  elapsed_seconds=time.perf_counter() - started)
    write_json(status_path, status)
    print(f'Smoke {status["status"]}; CPU QCR status: {status_path}', flush=True)
    return 0 if status['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
