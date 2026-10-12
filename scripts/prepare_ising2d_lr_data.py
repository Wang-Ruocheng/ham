"""Audited positive-K, zero-field Ising data for default b=3 QCR runs.

The Wolff sampler and observable/ESS estimates are imported unchanged from the
historical L32 preparation script.  This is a new cache, not a reproduction of
that script's seed stream or geometry.  Coupling labels and the known critical
point are only generation/evaluation metadata, never training supervision.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from recurrent_rg.config import ScaleConfig
from scripts.prepare_ising2d_qcr import chain, stats


KC = float(.5 * np.log(1 + np.sqrt(2)))
K_MIN = .2
K_MAX = .68
GAP_BOUNDARY = .36


@dataclass(frozen=True)
class SamplingOptions:
    size: int = 36
    seed: int = 91
    knots: int = 64
    samples: int = 128
    replicas: int = 3
    equil: int = 4096
    gap_low: int = 512
    gap_high: int = 128
    pilot_samples: int = 1024

    def validate(self):
        minimums = dict(size=1, seed=0, knots=6, samples=8, replicas=1,
                        equil=0, gap_low=1, gap_high=1, pilot_samples=8)
        for name, minimum in minimums.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
        # The default model requires a final side of at least four, not a
        # rounded/cropped lattice.  In particular L32 and L72 with three RG
        # updates must not be mistaken for the 36 -> 12 -> 4 experiment.
        ScaleConfig((self.size, self.size), max_rg_steps=2, block_factor=3).shapes()
        return self


def coupling_knots(count=64):
    if isinstance(count, bool) or not isinstance(count, int) or count < 6:
        raise ValueError('knots must be an integer >= 6')
    values = np.linspace(K_MIN, K_MAX, count, dtype=np.float64)
    values[np.argmin(abs(values - KC))] = KC
    if np.any(np.diff(values) <= 0):
        raise ValueError('Critical-point insertion must leave strictly increasing knots')
    return values


def derived_seed(seed, namespace, index):
    """Separate pilot, production-chain and symmetry-draw seed namespaces."""
    return int(np.random.SeedSequence([seed, namespace, index]).generate_state(
        1, dtype=np.uint32)[0])


def file_sha256(path):
    """Streaming SHA256 also works on the server's Python 3.10 runtime."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _publish_new_file(path, writer):
    """Publish one completed owned temporary file without clobbering a writer."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', dir=path.parent,
                                         prefix=f'.{path.name}.', suffix='.tmp',
                                         delete=False) as stream:
            temporary = Path(stream.name)
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        # Unlike replace(), link() fails if another writer already published.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _checked_chain(k, options, draws, seed, cold=False):
    gap = options.gap_low if k <= GAP_BOUNDARY else options.gap_high
    configurations = np.asarray(chain(k, options.size, draws, options.equil,
                                      gap, seed, cold))
    if configurations.shape != (draws, options.size, options.size):
        raise ValueError('Sampler returned an unexpected configuration shape')
    if not np.all((configurations == -1) | (configurations == 1)):
        raise ValueError('Sampler must return finite binary spins in {-1, +1}')
    return configurations, gap


def prepare_data(output, *, size=36, seed=91, knots=64, samples=128,
                 replicas=3, equil=4096, gap_low=512, gap_high=128,
                 pilot_samples=1024):
    """Write a new float64 tuple cache and return its sampling metadata.

    Explicit smaller sampling options are useful for tests, but all defaults
    retain the requested 64-K/3-chain/128-configuration production experiment.
    The cache and sidecar individually publish atomically and reject existing
    targets; an interrupted publication must be inspected rather than silently
    overwritten or treated as a complete audit.
    """
    options = SamplingOptions(size=size, seed=seed, knots=knots, samples=samples,
                              replicas=replicas, equil=equil, gap_low=gap_low,
                              gap_high=gap_high, pilot_samples=pilot_samples).validate()
    output = Path(output)
    if output.suffix != '.pt':
        raise ValueError('output must be a .pt cache file')
    metadata_path = output.with_suffix('.json')
    for path in (output, metadata_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    values = coupling_knots(options.knots)
    production_count = options.knots * options.replicas
    pilot_seeds = [derived_seed(seed, 0, i) for i in range(6)]
    chain_seeds = [derived_seed(seed, 1, i) for i in range(production_count)]
    if len(set(pilot_seeds + chain_seeds)) != 6 + production_count:
        raise ValueError('Derived independent-chain seed collision; choose another seed')
    started = time.perf_counter()
    report = dict(
        schema_version=1,
        sampler='Wolff fixed cluster count, numba v2',
        sampler_implementation='Unchanged scripts.prepare_ising2d_qcr.chain/stats',
        physical_model='Single-layer square Ising, positive K, B=0',
        coupling_names=['K'], B=0., sites=options.size,
        block_factor=3, max_rg_steps=2,
        lattice_sizes=[shape[0] for shape in ScaleConfig(
            (options.size, options.size), 2, block_factor=3).shapes()],
        seed=seed, sampling_options=asdict(options),
        draws=options.samples, replicas=options.replicas,
        equilibration_cluster_updates=options.equil,
        gap_cluster_updates=dict(boundary=GAP_BOUNDARY, at_or_below=options.gap_low,
                                 above=options.gap_high,
                                 note='Fixed by K, not by the current chain state'),
        couplings=values.tolist(),
        known_critical_K_for_generation_and_evaluation=KC,
        labels_use='Generation and evaluation metadata only; never optimizer supervision',
        split='K index i%3==1 is TEST; all replicas of a K point share the split',
        sampling_spin_flip=dict(enabled=True, scope='Independent global sign per stored configuration',
                                justification='Exact B=0 symmetry; even-observable diagnostics are unchanged',
                                training_spin_flip_augment=False),
        seed_derivation='uint32 SeedSequence([seed, namespace, index]); 0=pilot, 1=chain, 2=global-sign draw',
        historical_reproduction=False,
        historical_note='New L36/default-b3 cache with new namespaced seed streams, not the historical L32 fingerprint',
        uncertainty_note='Autocorrelation, ESS and standard errors are finite-chain estimates, not proof of independent draws or equilibration',
        cache_format='float64 (train_X, train_K, test_X, test_K) tuple',
        pilot=[], ensembles=[])
    for index, k in enumerate((K_MIN, KC, K_MAX)):
        record = dict(K=float(k), samples=options.pilot_samples, seeds={})
        for cold in (False, True):
            label = 'cold' if cold else 'hot'
            pilot_seed = pilot_seeds[2 * index + int(cold)]
            x, gap = _checked_chain(k, options, options.pilot_samples, pilot_seed, cold)
            record[label] = stats(x)
            record['seeds'][label] = pilot_seed
            record['gap_cluster_updates'] = gap
        report['pilot'].append(record)
        print(f'pilot K={k:.7f} hot/cold completed', flush=True)
    train_x, train_k, test_x, test_k = [], [], [], []
    for index, k in enumerate(values):
        for replica in range(options.replicas):
            row = index * options.replicas + replica
            chain_seed = chain_seeds[row]
            flip_seed = derived_seed(seed, 2, row)
            x, gap = _checked_chain(float(k), options, options.samples, chain_seed)
            is_test = index % 3 == 1
            report['ensembles'].append(dict(K=float(k), K_index=index, replica=replica,
                seed=chain_seed, global_flip_seed=flip_seed, test=is_test,
                gap_cluster_updates=gap, **stats(x)))
            rng = np.random.default_rng(flip_seed)
            flipped = x * rng.choice(np.array([-1, 1], dtype=np.int8),
                                     size=(options.samples, 1, 1))
            xs, labels = (test_x, test_k) if is_test else (train_x, train_k)
            xs.append(torch.from_numpy(flipped).to(dtype=torch.float64).unsqueeze(-1))
            labels.append(float(k))
        print(f'data {index + 1}/{options.knots} K={k:.7f}', flush=True)
    payload = (torch.stack(train_x), torch.tensor(train_k, dtype=torch.float64),
               torch.stack(test_x), torch.tensor(test_k, dtype=torch.float64))
    report.update(train_shape=list(payload[0].shape), test_shape=list(payload[2].shape),
                  train_ensembles=len(train_x), test_ensembles=len(test_x),
                  train_K=[float(k) for i, k in enumerate(values) if i % 3 != 1],
                  test_K=[float(k) for i, k in enumerate(values) if i % 3 == 1],
                  elapsed_seconds=time.perf_counter() - started)
    # Reject nonfinite diagnostics before publishing any output.
    json.dumps(report, allow_nan=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    _publish_new_file(output, lambda stream: torch.save(payload, stream))
    report['data_sha256'] = file_sha256(output)
    serialized = (json.dumps(report, indent=2, allow_nan=False) + '\n').encode('utf-8')
    _publish_new_file(metadata_path, lambda stream: stream.write(serialized))
    print(f'saved {output} SHA256={report["data_sha256"]}', flush=True)
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    for name, field in SamplingOptions.__dataclass_fields__.items():
        parser.add_argument('--' + name.replace('_', '-'), type=int, default=field.default)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return prepare_data(**vars(args))
    except ValueError as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
