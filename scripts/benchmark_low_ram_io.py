"""Small, CPU-only synthetic audit of QCR's storage/selection execution.

This is not physical sampling, training, an MI benchmark, or an estimate of the
minimum RAM of any formal experiment. Each measured path runs in a new process;
the only input is a synthetic file created in a new, exclusive output directory.
The default tensor payload is at most 256 MiB; requests above 1024 MiB are rejected.
"""
from contextlib import contextmanager
import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import subprocess
import sys
import threading
import time


MIB = 1024 ** 2
MAX_SIZE_MIB = 1024


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(MIB), b''):
            digest.update(block)
    return digest.hexdigest()


def peak_rss_bytes(value=None, *, platform=None):
    """ru_maxrss is bytes on macOS, KiB on Linux (not interchangeable)."""
    platform = sys.platform if platform is None else platform
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if value is None else value
    if platform == 'darwin':
        return int(value)
    if platform.startswith('linux'):
        return int(value * 1024)
    raise RuntimeError('Peak RSS units are only defined here for Linux and macOS')


def parse_proc_kib(text):
    result = {}
    for line in text.splitlines():
        key, sep, rest = line.partition(':')
        fields = rest.split()
        if sep and len(fields) == 2 and fields[1] == 'kB':
            result[key] = int(fields[0]) * 1024
    return result


def current_memory():
    """Read our own process only. PSS-file is not mislabeled as file RSS."""
    result = {}
    if sys.platform.startswith('linux'):
        for filename, prefix in (('/proc/self/status', 'status'),
                                 ('/proc/self/smaps_rollup', 'smaps')):
            try:
                values = parse_proc_kib(Path(filename).read_text())
            except (OSError, ValueError):
                continue
            wanted = ('VmRSS', 'RssAnon', 'RssFile', 'RssShmem') if prefix == 'status' else (
                'Rss', 'Pss', 'Anonymous', 'Pss_Anon', 'Pss_File', 'Pss_Shmem')
            result.update({f'{prefix}_{key}_bytes': values[key]
                           for key in wanted if key in values})
        if 'status_VmRSS_bytes' in result:
            result['rss_bytes'] = result['status_VmRSS_bytes']
    else:
        try:
            import psutil
            result['rss_bytes'] = int(psutil.Process().memory_info().rss)
        except ImportError:
            # Lifetime ru_maxrss is still available without this optional tool.
            pass
    return result


class MemoryMonitor:
    """Sampling is approximate; ru_maxrss remains the authoritative RSS peak."""
    def __init__(self, interval=.01):
        self.interval = interval
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.peaks = {}
        self.active = None
        self.active_peaks = None
        self.stages = []

    def sample(self):
        sample = current_memory()
        with self.lock:
            for key, value in sample.items():
                self.peaks[key] = max(value, self.peaks.get(key, 0))
                if self.active_peaks is not None:
                    self.active_peaks[key] = max(value, self.active_peaks.get(key, 0))
        return sample

    def run(self):
        while not self.stop.wait(self.interval):
            self.sample()

    def __enter__(self):
        self.sample()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.sample()

    @contextmanager
    def stage(self, name):
        before = current_memory()
        with self.lock:
            self.active = name
            self.active_peaks = dict(before)
        started = time.perf_counter()
        self.sample()
        try:
            yield
        finally:
            self.sample()
            after = current_memory()
            with self.lock:
                peaks = self.active_peaks
                self.active = self.active_peaks = None
            self.stages.append(dict(name=name, seconds=time.perf_counter()-started,
                current_before=before, current_after=after, sampled_peak=peaks,
                cumulative_ru_maxrss_bytes=peak_rss_bytes()))


def workload_shape(size_mib):
    if not math.isfinite(size_mib) or not 2 <= size_mib <= MAX_SIZE_MIB:
        raise ValueError(f'Synthetic payload must be between 2 and {MAX_SIZE_MIB} MiB')
    samples, side = 32, 36
    ensemble_bytes = samples * side * side * 8
    ensembles = int(size_mib * MIB) // ensemble_bytes
    train = max(2, 2 * ensembles // 3)
    test = ensembles - train
    if test < 2:
        raise ValueError('Synthetic payload must fit at least two ensembles per split')
    return dict(train_shape=[train, samples, side, side, 1],
                test_shape=[test, samples, side, side, 1],
                tensor_payload_bytes=ensembles*ensemble_bytes)


def generate(output, size_mib):
    import torch
    torch.set_num_threads(1)
    output = Path(output)
    metadata = workload_shape(size_mib)
    generator = torch.Generator().manual_seed(91021)
    tensors = []
    for key in ('train_shape', 'test_shape'):
        x = torch.empty(metadata[key], dtype=torch.float64)
        for ensemble in x:
            # Bounded temporary; this is a synthetic random bit fixture, not MC.
            bits = torch.randint(2, ensemble.shape, generator=generator, dtype=torch.int8)
            ensemble.copy_(bits).mul_(2).sub_(1)
        tensors.append(x)
    tx, vx = tensors
    ty = torch.arange(len(tx), dtype=torch.float64)
    vy = torch.arange(len(vx), dtype=torch.float64) + len(tx)
    cache = output/'synthetic.pt'
    torch.save((tx, ty, vx, vy), cache)
    metadata.update(synthetic=True, physical_sampling=False, cache='synthetic.pt',
        requested_size_mib=size_mib, dtype='torch.float64', seed=91021,
        cache_bytes=cache.stat().st_size, cache_sha256=sha256(cache))
    (output/'synthetic_manifest.json').write_text(json.dumps(metadata, indent=2)+'\n')


def _tensor_signature(digest, tensor):
    digest.update(json.dumps([list(tensor.shape), str(tensor.dtype)]).encode())
    # Selected tensors are contiguous; no full source tensor is materialized.
    digest.update(memoryview(tensor.contiguous().numpy()).cast('B'))


def measured_worker(output, mode):
    import torch
    from recurrent_rg.config import DataConfig, EvaluationConfig, ModelConfig, RunConfig, TrainingConfig
    from recurrent_rg.execution import select_training_batch
    from recurrent_rg.runner import prepared_data

    torch.set_num_threads(1)
    torch.manual_seed(91)
    output = Path(output)
    manifest = json.loads((output/'synthetic_manifest.json').read_text())
    if manifest.get('synthetic') is not True or manifest.get('cache') != 'synthetic.pt':
        raise ValueError('Only this benchmark\'s own synthetic cache may be measured')
    cache = output/'synthetic.pt'
    source_sha = sha256(cache)
    if source_sha != manifest['cache_sha256']:
        raise ValueError('Synthetic source fingerprint changed before measurement')
    samples = manifest['train_shape'][1]
    config = RunConfig(DataConfig(str(cache), 'synthetic_low_ram_io', ('K',)),
        model=ModelConfig(dimension=2, block_factor=3),
        training=TrainingConfig(max_rg_steps=2, sample_batch=samples, threads=1),
        evaluation=EvaluationConfig(max_rg_steps=2, samples=samples, systems=24))
    config.validate()
    digest = hashlib.sha256()
    global_rng_before = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
    baseline_memory = current_memory()
    started = time.perf_counter()
    with MemoryMonitor() as monitor:
        with monitor.stage('load_and_complete_validation'):
            if mode == 'reference':
                tx, ty, vx, vy = torch.load(str(cache), map_location='cpu', weights_only=True)
                for x, y in ((tx, ty), (vx, vy)):
                    if (x.dtype != torch.float64 or not torch.isfinite(x).all()
                            or not torch.isfinite(y).all() or not torch.all((x==1)|(x==-1))):
                        raise ValueError('Invalid synthetic payload')
                if set(ty.tolist()) & set(vy.tolist()):
                    raise ValueError('Synthetic split overlap')
            elif mode == 'optimized':
                train, test = prepared_data(config, mmap=True)
                tx, ty, vx, vy = train.configurations, train.couplings, test.configurations, test.couplings
            else:
                raise ValueError('Expected reference or optimized path')
            if list(tx.shape) != manifest['train_shape'] or list(vx.shape) != manifest['test_shape']:
                raise ValueError('Synthetic source geometry changed')
        with monitor.stage('evaluation_panel_setup'):
            datasets = {'train': tx, 'test': vx}
            evaluation_ids = {split: torch.linspace(0, len(x)-1, min(24, len(x))).long()
                              for split, x in datasets.items()}
            panels = ({split: x[evaluation_ids[split]] for split, x in datasets.items()}
                      if mode == 'reference' else None)
        generator = torch.Generator().manual_seed(9021)
        with monitor.stage('same_order_training_batch_gathers'):
            for _ in range(4):
                ids = torch.randint(len(tx), (min(16, len(tx)),), generator=generator)
                indices = torch.randperm(samples, generator=generator)
                selected = select_training_batch(tx, ids, indices, 'cpu', reference=mode=='reference')
                _tensor_signature(digest, ids)
                _tensor_signature(digest, indices)
                _tensor_signature(digest, selected)
                del selected
        with monitor.stage('same_order_one_split_at_a_time_evaluation_panels'):
            for split, x in datasets.items():
                panel = x[evaluation_ids[split]] if panels is None else panels[split]
                _tensor_signature(digest, evaluation_ids[split])
                _tensor_signature(digest, panel)
                del panel
        with monitor.stage('all_source_values_digest'):
            for x in (tx, ty, vx, vy):
                for row in x:
                    _tensor_signature(digest, row)
    wall = time.perf_counter()-started
    global_rng_after = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
    result = dict(status='complete', synthetic=True, physical_sampling=False, training=False,
        mode=mode, process_id=__import__('os').getpid(), device='cpu', dtype='torch.float64',
        tensor_payload_bytes=manifest['tensor_payload_bytes'],
        train_shape=list(tx.shape), test_shape=list(vx.shape),
        source_sha256_before=source_sha, source_sha256_after=sha256(cache),
        source_unchanged=source_sha==sha256(cache), content_and_selection_sha256=digest.hexdigest(),
        global_rng_before=global_rng_before, global_rng_after=global_rng_after,
        batch_generator_state_sha256=hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest(),
        torch_version=torch.__version__, platform=sys.platform, threads=1,
        ru_maxrss_units='bytes' if sys.platform=='darwin' else 'KiB',
        peak_rss_bytes=peak_rss_bytes(), post_import_current_memory=baseline_memory,
        sampled_peak=monitor.peaks, sampling_interval_seconds=monitor.interval,
        stages=monitor.stages, workload_wall_seconds=wall)
    (output/f'{mode}.json').write_text(json.dumps(result, indent=2)+'\n')


def compare(reference, optimized):
    checks = {key: reference[key] == optimized[key] for key in (
        'dtype', 'tensor_payload_bytes', 'train_shape', 'test_shape',
        'source_sha256_before', 'source_sha256_after', 'content_and_selection_sha256',
        'global_rng_before', 'global_rng_after', 'batch_generator_state_sha256')}
    checks['separate_processes'] = reference['process_id'] != optimized['process_id']
    checks['both_sources_unchanged'] = reference['source_unchanged'] and optimized['source_unchanged']
    checks['no_global_rng_draws'] = all(item['global_rng_before'] == item['global_rng_after']
                                      for item in (reference, optimized))
    return dict(passed=all(checks.values()), checks=checks,
        reference_peak_rss_bytes=reference['peak_rss_bytes'],
        optimized_peak_rss_bytes=optimized['peak_rss_bytes'],
        peak_rss_reduction_bytes=reference['peak_rss_bytes']-optimized['peak_rss_bytes'],
        reference_over_optimized_peak_rss=reference['peak_rss_bytes']/optimized['peak_rss_bytes'],
        reference_workload_seconds=reference['workload_wall_seconds'],
        optimized_workload_seconds=optimized['workload_wall_seconds'])


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--size-mib', type=float, default=256)
    parser.add_argument('--_worker', choices=('generate', 'reference', 'optimized'), help=argparse.SUPPRESS)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        workload_shape(args.size_mib)
    except ValueError as error:
        parser.error(str(error))
    if args._worker:
        if args._worker == 'generate':
            generate(args.output, args.size_mib)
        else:
            measured_worker(args.output, args._worker)
        return 0
    if args.output.exists():
        parser.error('Refusing to reuse or overwrite an existing output directory')
    args.output.mkdir(parents=True, exist_ok=False)
    common = [sys.executable, '-m', 'scripts.benchmark_low_ram_io', '--output', str(args.output.resolve()),
              '--size-mib', str(args.size_mib)]
    try:
        for mode in ('generate', 'reference', 'optimized'):
            with (args.output/f'{mode}.log').open('x') as log:
                subprocess.run(common+['--_worker', mode], check=True, stdout=log, stderr=subprocess.STDOUT)
        reference = json.loads((args.output/'reference.json').read_text())
        optimized = json.loads((args.output/'optimized.json').read_text())
        comparison = compare(reference, optimized)
        result = dict(status='complete' if comparison['passed'] else 'failed', synthetic=True,
            physical_sampling=False, model_training=False, MI_cache_loaded=False, CUDA_used=False,
            formal_experiment_RAM_requirement_certified=False, comparison=comparison,
            limitations=[
                'Synthetic CPU storage/validation/indexing only; no physics, QCR model, gradients, MI or CUDA.',
                'Both complete finite/binary validations and all source values are checked; no checks are skipped.',
                'Separate fresh child processes; no cache eviction, host tuning, other-process inspection or intervention.',
                'Checksums and generation warm filesystem pages; timings are not a cold-I/O comparison.',
                'Mapped clean file pages still count in RSS and can be reclaimed; mmap does not mean zero RAM.',
                'Sampling may miss short temporaries; ru_maxrss is the process-lifetime RSS high-water mark.',
                'Allocator, OS/file-cache state and source geometry affect results; no formal RAM gate may be inferred.'])
        (args.output/'benchmark.json').write_text(json.dumps(result, indent=2)+'\n')
        if not comparison['passed']:
            raise RuntimeError('Synthetic execution-equivalence check failed; inspect benchmark.json')
    except Exception as error:
        (args.output/'error.json').write_text(json.dumps(dict(status='failed', synthetic=True,
            error=f'{type(error).__name__}: {error}'), indent=2)+'\n')
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
