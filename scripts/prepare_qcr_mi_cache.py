"""Prepare default binary 2D full-shell MI caches before the training clock.

This CPU-only stage constructs the exact same data-only v1 statistics/key used
by the trainer. It initializes no model or optimizer, draws no random samples,
and never overwrites a corrupt or incomplete existing cache.
"""
import argparse
import json
from pathlib import Path
import time

import torch

from experiments.rsmi_separate_qcr import build_parser as training_parser, validate_recipe
from recurrent_rg.config import RunConfig
from recurrent_rg.empirical_mi import binary_lattice_mi_statistics
from recurrent_rg.mi_cache import lattice_statistics_cache_key, load_or_build_mi_statistics
from recurrent_rg.rsmi import region_offsets
from recurrent_rg.runner import file_fingerprint, prepared_data
from scripts.prepare_ising2d_lr_data import _publish_new_file


def prepare(config_path, output, *, cache_dir=None):
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Refusing to overwrite preprocessing receipt: {output}')
    cfg = RunConfig.load(config_path)
    cfg.validate()
    args = training_parser().parse_args(['--config', str(config_path), '--output', str(output)])
    validate_recipe(args, cfg)
    if (cfg.model.dimension != 2 or cfg.model.categorical_states or
            cfg.model.kernel_layout != 'nonoverlapping' or
            cfg.model.kernel_sampling != 'bernoulli'):
        raise ValueError('Preprocessing supports default binary 2D full-shell probabilistic C only')
    torch.set_num_threads(cfg.training.threads)
    started = time.perf_counter()
    digest = file_fingerprint(cfg.data.cache)
    train, test = prepared_data(cfg)
    _, environment = region_offsets(2, block=cfg.model.block_factor, buffer=1)
    directory = Path(cache_dir) if cache_dir else Path(cfg.data.cache).parent / '.qcr_mi_cache'
    report = dict(status='complete', config=str(config_path),
                  config_sha256=file_fingerprint(config_path), data_sha256=digest,
                  device='cpu', model_initialized=False, optimizer_updates=0,
                  sampling_draws=0, cache_version=1, splits={})
    for split, dataset in [('train', train), ('test', test)]:
        x = dataset.configurations
        kwargs = dict(block=cfg.model.block_factor)
        identity = lattice_statistics_cache_key(cfg, digest, split, x, environment,
                                               binary_lattice_mi_statistics.__name__, kwargs)
        statistics, record = load_or_build_mi_statistics(directory, identity,
            lambda x=x: binary_lattice_mi_statistics(x, flip=cfg.data.spin_flip_augment, **kwargs))
        report['splits'][split] = dict(cache=record, diagnostics=statistics.diagnostics,
                                      data_shape=list(x.shape))
        print(f'MI preprocessing {split}: ' + ('cache hit' if record['hit'] else 'built'), flush=True)
    if file_fingerprint(cfg.data.cache) != digest:
        raise ValueError('Data changed during MI preprocessing; refuse to mix fingerprints')
    report['elapsed_seconds'] = time.perf_counter() - started
    serialized = (json.dumps(report, indent=2, allow_nan=False) + '\n').encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    _publish_new_file(output, lambda stream: stream.write(serialized))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path)
    args = parser.parse_args(argv)
    report = prepare(args.config, args.output, cache_dir=args.cache_dir)
    print(json.dumps(report), flush=True)
    return report


if __name__ == '__main__':
    main()
