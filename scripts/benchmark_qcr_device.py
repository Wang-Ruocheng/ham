"""Time the real locked QCR CLI on one shared cache, without changing its loss.

Prepare once, transfer that exact cache, then run on local CPU and remote CUDA.
The first warmup updates remain real training updates; only timing excludes them.
Short benchmark figures are execution diagnostics, not converged physics results.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))
from scripts.remote_smoke import environment_record, write_json


CONFIG_TEMPLATE = REPOSITORY / 'configs' / 'remote_smoke_225.json'
INTERPRETATION = (
    'Short same-cache execution benchmark, not a converged physics experiment; '
    'CPU and CUDA use different random streams, so identical losses are not expected. '
    'Speed ratios apply to this float64 workload and these machines, not all runs.'
)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def config_fingerprint(config):
    """Cache paths differ across hosts; compare cache content independently."""
    values = json.loads(json.dumps(config))
    values['data']['cache'] = '<same-content-cache>'
    payload = json.dumps(values, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(payload).hexdigest()


def benchmark_config(cache, *, epochs=60, threads=2):
    config = json.loads(CONFIG_TEMPLATE.read_text())
    config['data']['cache'] = str(Path(cache).resolve())
    config['model']['theory_dim'] = 2
    config['training'].update(
        epochs=epochs, system_batch=8, sample_batch=64, threads=threads,
        checkpoint_every=20, log_every=20,
    )
    config['evaluation'].update(samples=64, systems=8)
    return config


def prepare_cache(output):
    """Generate a small, reusable real SW cache; labels never supervise training."""
    import torch
    from recurrent_rg.ising2d_field import sample_field_chain

    # Interleaved but disjoint complete coupling points, not configuration splits.
    coupling_grid = [.2 + .48 * i / 23 for i in range(24)]
    train_k = [k for i, k in enumerate(coupling_grid) if i % 3 != 1]
    test_k = [k for i, k in enumerate(coupling_grid) if i % 3 == 1]
    sampling = {
        'sampler': 'sample_field_chain: Swendsen-Wang cluster heat bath',
        'physical_model': 'ising2d', 'B': 0., 'size': 36, 'samples': 64,
        'equilibration_sweeps': 128, 'sweeps_between': 4,
        'train_K': train_k, 'test_K': test_k,
        'train_seeds': [910001 + i for i in range(16)],
        'test_seeds': [920001 + i for i in range(8)],
        'split': 'Disjoint complete coupling points',
        'labels_used_in_training': False, 'interpretation': INTERPRETATION,
    }

    def sample(couplings, seeds):
        x = torch.stack([
            torch.from_numpy(sample_field_chain(
                k, 0., 36, 64, 128, 4, seed,
            )).unsqueeze(-1)
            for k, seed in zip(couplings, seeds)
        ])
        return x, torch.tensor(couplings, dtype=torch.float64)

    start = time.perf_counter()
    tx, ty = sample(train_k, sampling['train_seeds'])
    vx, vy = sample(test_k, sampling['test_seeds'])
    cache = output / 'ensembles.pt'
    torch.save((tx, ty, vx, vy), cache)
    sampling.update(
        cache=str(cache), cache_sha256=file_sha256(cache),
        train_shape=list(tx.shape), test_shape=list(vx.shape),
        preparation_seconds=time.perf_counter() - start,
    )
    write_json(output / 'sampling.json', sampling)
    return cache, sampling


def validate_cache(cache):
    import torch

    tx, ty, vx, vy = torch.load(cache, map_location='cpu', weights_only=True)
    expected = ((16, 64, 36, 36, 1), (8, 64, 36, 36, 1))
    if (tuple(tx.shape), tuple(vx.shape)) != expected:
        raise ValueError(f'Benchmark needs TRAIN/TEST shapes {expected}')
    if any(not torch.all((x == -1) | (x == 1)) for x in (tx, vx)):
        raise ValueError('Benchmark expects binary Ising configurations')
    if ty.reshape(len(tx), -1).shape[1] != 1 or vy.reshape(len(vx), -1).shape[1] != 1:
        raise ValueError('Benchmark coupling labels must contain one K per ensemble')
    train_k, test_k = set(ty.flatten().tolist()), set(vy.flatten().tolist())
    if train_k & test_k:
        raise ValueError('Benchmark TRAIN/TEST must use disjoint complete K points')
    return {
        'cache': str(cache), 'cache_sha256': file_sha256(cache),
        'train_shape': list(tx.shape), 'test_shape': list(vx.shape),
        'labels_used_in_training': False,
    }


def timed_command(command, log_path):
    start = time.perf_counter()
    with log_path.open('w') as log:
        result = subprocess.run(
            command, cwd=REPOSITORY, stdout=log, stderr=subprocess.STDOUT, check=False,
        )
    seconds = time.perf_counter() - start
    if result.returncode:
        tail = '\n'.join(log_path.read_text().splitlines()[-25:])
        raise RuntimeError(f'Command exited {result.returncode}: {command}\n{tail}')
    return {'command': command, 'wall_seconds': seconds, 'log': str(log_path)}


def summarize_timing(timing, *, updates, warmup, device):
    if timing['device'] != device or timing['updates'] != updates:
        raise ValueError('Driver timing device/update count differs from requested benchmark')
    if timing['warmup_updates'] != warmup or timing['includes_eval_in_step']:
        raise ValueError('Driver timing did not use the requested update-only warmup convention')
    all_steps = timing['step_seconds_all']
    steady = timing['steady_step_seconds']
    if len(all_steps) != updates or len(steady) != updates - warmup:
        raise ValueError('Missing per-update benchmark timings')
    if timing['measured_updates'] != len(steady):
        raise ValueError('Driver measured update count does not match timing samples')
    if any(not math.isfinite(s) or s <= 0 for s in all_steps + steady):
        raise ValueError('Benchmark update timings must be finite and positive')
    if steady != all_steps[warmup:]:
        raise ValueError('Warmup timings were not excluded by position')
    ordered = sorted(steady)
    p90_index = math.ceil(.9 * len(ordered)) - 1
    return {
        'steady_update_mean_seconds': statistics.mean(steady),
        'steady_update_median_seconds': statistics.median(steady),
        'steady_update_p90_seconds': ordered[p90_index],
        'measured_updates': len(steady), 'warmup_updates': warmup,
        'timing_convention': 'CUDA synchronized around each real optimizer update; '
                             'evaluation, saving and final reports are separate',
        'driver': timing,
    }


def compare_benchmarks(first, second):
    for key in ('cache_sha256', 'config_sha256', 'dtype', 'actual_epochs', 'warmup_updates'):
        if first[key] != second[key]:
            raise ValueError(f'Cannot fairly compare benchmarks: {key} differs')
    records = {first['device'].split(':')[0]: first, second['device'].split(':')[0]: second}
    if set(records) != {'cpu', 'cuda'}:
        raise ValueError('Speed comparison requires one CPU and one CUDA benchmark')
    cpu, gpu = records['cpu'], records['cuda']
    return {
        'cpu_source': cpu.get('output'), 'cuda_source': gpu.get('output'),
        'steady_update_median_speedup': (
            cpu['timing']['steady_update_median_seconds'] /
            gpu['timing']['steady_update_median_seconds']
        ),
        'steady_update_mean_speedup': (
            cpu['timing']['steady_update_mean_seconds'] /
            gpu['timing']['steady_update_mean_seconds']
        ),
        'training_wall_speedup': (
            cpu['timing']['driver']['training_wall_seconds'] /
            gpu['timing']['driver']['training_wall_seconds']
        ),
        'setup_seconds_cpu': cpu['timing']['driver']['setup_seconds'],
        'setup_seconds_cuda': gpu['timing']['driver']['setup_seconds'],
        'ratio_convention': 'CPU seconds / CUDA seconds; >1 means CUDA is faster',
        'limitations': INTERPRETATION,
    }


def verify_training(run, device, epochs):
    import torch

    summary = json.loads((run / 'summary.json').read_text())
    manifest = json.loads((run / 'manifest.json').read_text())
    checkpoint = torch.load(run / 'checkpoint_last.pt', weights_only=True)
    devices = sorted({str(t.device) for t in checkpoint['model'].values()})
    dtypes = sorted({str(t.dtype) for t in checkpoint['model'].values() if t.is_floating_point()})
    if devices != [device] or dtypes != ['torch.float64']:
        raise RuntimeError(f'Unexpected checkpoint device/precision: {devices}, {dtypes}')
    if checkpoint['epoch'] != epochs or summary['actual_epochs'] != epochs:
        raise RuntimeError('Benchmark did not complete its requested updates')
    if manifest['labels_used_in_training'] or not manifest['independent_c']:
        raise RuntimeError('Benchmark violated locked gradient/label conventions')
    for row in summary['history']:
        for key in ('train_total', 'test_total', 'train_rsmi_bits', 'test_rsmi_bits'):
            if not math.isfinite(row[key]):
                raise RuntimeError(f'Nonfinite {key} at update {row["epoch"]}')
    expected = [
        'report/R_vector_field.png', 'report/coupling_vs_g.png',
        'report/trajectories.png', 'report/metrics.json', 'report/fixed_points.json',
        'report/trajectories.npz', 'report/loss_comparison.png',
        'report/training_comparison.json',
    ]
    for name in expected:
        if not (run / name).is_file() or not (run / name).stat().st_size:
            raise RuntimeError(f'Missing benchmark artifact: {name}')
    return summary, {
        'checkpoint_parameter_devices': devices, 'dtype': dtypes[0],
        'independent_c': manifest['independent_c'],
        'labels_used_in_training': manifest['labels_used_in_training'],
        'estimator': manifest['estimator'], 'buffer': manifest['buffer'],
        'environment_mode': manifest['environment_mode'],
        'artifacts': [str(run / name) for name in expected],
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true', help='Generate shared cache only; no training')
    mode.add_argument('--cache', type=Path, help='Reuse the exact shared cache, never regenerate it')
    parser.add_argument('--output', type=Path, required=True, help='New isolated directory')
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cpu')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--compare', type=Path, help='Compare with another completed benchmark.json')
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.threads < 1 or args.epochs < 1 or not 0 <= args.warmup < args.epochs:
        parser.error('Need positive threads/epochs and 0 <= warmup < epochs')
    if args.prepare and args.compare:
        parser.error('--compare requires a training benchmark, not --prepare')
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error(f'Refusing to overwrite existing output: {output}')
    if args.cache and not args.cache.expanduser().resolve().is_file():
        parser.error(f'Missing cache: {args.cache}')
    output.mkdir(parents=True, exist_ok=False)
    status = {
        'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(),
        'output': str(output), 'interpretation': INTERPRETATION,
        'environment': environment_record(), 'device': args.device,
    }
    status_path = output / 'benchmark_status.json'
    write_json(status_path, status)
    started = time.perf_counter()
    try:
        import torch

        torch.set_num_threads(args.threads)
        status['environment'].update(torch_runtime_version=torch.__version__,
                                     torch_cuda_version=torch.version.cuda)
        if args.prepare:
            cache, sampling = prepare_cache(output)
            status.update(status='prepared', cache=str(cache), sampling=sampling)
        else:
            cache = args.cache.expanduser().resolve()
            sampling = validate_cache(cache)
            config = benchmark_config(cache, epochs=args.epochs, threads=args.threads)
            write_json(output / 'config.json', config)
            write_json(output / 'sampling.json', sampling)
            if args.device.startswith('cuda'):
                if not torch.cuda.is_available():
                    raise RuntimeError('CUDA requested but unavailable; refusing CPU fallback')
                status['environment'].update(
                    cuda_device_name=torch.cuda.get_device_name(args.device),
                    cuda_device_capability=list(torch.cuda.get_device_capability(args.device)),
                )
            status.update(sampling=sampling, config_sha256=config_fingerprint(config))
            write_json(status_path, status)
            run = output / 'run'
            status['training_command'] = timed_command([
                sys.executable, '-m', 'recurrent_rg', 'train', '--config',
                str(output / 'config.json'), '--output', str(run),
                '--device', args.device, '--benchmark-warmup', str(args.warmup),
            ], output / 'train.log')
            write_json(status_path, status)
            status['loss_report_command'] = timed_command([
                sys.executable, '-m', 'scripts.report_mi_training', '--run', str(run),
                '--threads', str(args.threads),
            ], output / 'loss_report.log')
            summary, verification = verify_training(run, args.device, args.epochs)
            benchmark = {
                'output': str(output), 'device': args.device,
                'environment': status['environment'],
                'cache_sha256': sampling['cache_sha256'],
                'config_sha256': status['config_sha256'], 'config': config,
                'dtype': verification['dtype'], 'actual_epochs': summary['actual_epochs'],
                'warmup_updates': args.warmup, 'verification': verification,
                'timing': summarize_timing(summary['timing'], updates=args.epochs,
                                           warmup=args.warmup, device=args.device),
                'commands': {
                    'training_and_default_report': status['training_command'],
                    'loss_report': status['loss_report_command'],
                },
                'interpretation': INTERPRETATION,
            }
            if args.compare:
                prior = json.loads(args.compare.expanduser().resolve().read_text())
                benchmark['comparison'] = compare_benchmarks(benchmark, prior)
            write_json(output / 'benchmark.json', benchmark)
            status.update(status='passed', benchmark=str(output / 'benchmark.json'),
                          verification=verification)
    except Exception as error:
        status.update(status='failed', error=f'{type(error).__name__}: {error}')
        traceback.print_exc()
    status.update(finished_at=datetime.now(timezone.utc).isoformat(),
                  elapsed_seconds=time.perf_counter() - started)
    write_json(status_path, status)
    print(f'Benchmark {status["status"]}; status: {status_path}', flush=True)
    return 0 if status['status'] in ('prepared', 'passed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
