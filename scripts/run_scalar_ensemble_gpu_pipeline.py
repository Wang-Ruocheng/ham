"""Run one fresh default scalar QCR case on a larger independent-replica cache.

Sampling and data-only MI preprocessing precede the training wall cap.  The
fixed multistep policy uses cumulative updates 6000/9000, without checkpoint
selection or automatic retries.  Existing output and incomplete cache/sidecar
pairs are errors, not implicit resumes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from recurrent_rg.config import RunConfig, ScaleConfig
from recurrent_rg.data.ensembles import load_ensembles
from recurrent_rg.runner import file_fingerprint
from scripts.prepare_ising2d_lr_data import coupling_knots, derived_seed
from scripts.run_scalar_lr_gpu_pipeline import save_status, stage, utc_now
from scripts.run_scalar_lr_sweep import make_configs


def check_data(cache, *, replicas=12, seed=91):
    """Validate this single-case production cache without relaxing LR-sweep rules."""
    if isinstance(replicas, bool) or not isinstance(replicas, int) or replicas < 1:
        raise ValueError('replicas must be a positive integer')
    cache = Path(cache)
    sidecar = cache.with_suffix('.json')
    sidecar_digest = file_fingerprint(sidecar)
    metadata = json.loads(sidecar.read_text())
    expected_sampling = dict(size=36, seed=seed, knots=64, samples=128, replicas=replicas,
                             equil=4096, gap_low=512, gap_high=128, pilot_samples=1024)
    if any(metadata.get('sampling_options', {}).get(key) != value
           for key, value in expected_sampling.items()):
        raise ValueError('Cache sidecar must document the requested production sampling settings')
    if metadata.get('B') != 0. or metadata.get('coupling_names') != ['K']:
        raise ValueError('Cache sidecar must identify the single-layer zero-field K scan')
    if metadata.get('seed') != seed or metadata.get('replicas') != replicas:
        raise ValueError('Cache sidecar seed and independent-replica count do not match the plan')
    if metadata.get('lattice_sizes') != [36, 12, 4]:
        raise ValueError('Cache sidecar must document 36 -> 12 -> 4 at block factor 3')
    digest = file_fingerprint(cache)
    if metadata.get('data_sha256') != digest:
        raise ValueError('Cache bytes differ from the sampling sidecar SHA256')
    values = coupling_knots(64).tolist()
    expected_knots = {split: [k for i, k in enumerate(values)
                              if (i % 3 == 1) == (split == 'test')]
                      for split in ('train', 'test')}
    if metadata.get('couplings') != values:
        raise ValueError('Cache must use the declared 64-K [.2,.68] generation grid')
    records = metadata.get('ensembles', [])
    expected_records = []
    for i, k in enumerate(values):
        for replica in range(replicas):
            row = i * replicas + replica
            expected_records.append((i, replica, k, i % 3 == 1,
                                     derived_seed(seed, 1, row), derived_seed(seed, 2, row)))
    actual_records = [(r.get('K_index'), r.get('replica'), r.get('K'), r.get('test'),
                       r.get('seed'), r.get('global_flip_seed')) for r in records]
    if actual_records != expected_records:
        raise ValueError('All K points require their own documented independent replica/flip seeds')
    if len({r[4] for r in actual_records}) != 64 * replicas:
        raise ValueError('Independent production-chain seeds must be unique')
    train, test = load_ensembles(cache)
    shapes = {}
    for split, ensemble in (('train', train), ('test', test)):
        x, k = ensemble.configurations, ensemble.couplings
        knots = expected_knots[split]
        expected_shape = (len(knots) * replicas, 128, 36, 36, 1)
        if (tuple(x.shape) != expected_shape or k is None or
                tuple(k.shape) not in ((expected_shape[0],), (expected_shape[0], 1))):
            raise ValueError(f'{split}: requires shape {expected_shape} and K-only labels')
        if x.dtype != torch.float64 or k.dtype != torch.float64:
            raise ValueError(f'{split}: production cache configurations and labels must be float64')
        if metadata.get(split + '_K') != knots:
            raise ValueError(f'{split}: sidecar must retain the complete preregistered K-point split')
        if Counter(k.flatten().tolist()) != Counter({value: replicas for value in knots}):
            raise ValueError(f'{split}: K points and {replicas} replicas must match the sampling sidecar')
        if metadata.get(split + '_shape') != list(expected_shape):
            raise ValueError(f'{split}: shape differs from the sampling sidecar')
        shapes[split + '_shape'] = list(expected_shape)
    if set(train.couplings.flatten().tolist()) & set(test.couplings.flatten().tolist()):
        raise ValueError('Complete K points, including all replicas, must not cross the split')
    ScaleConfig((36, 36), max_rg_steps=2, block_factor=3).shapes()
    data = dict(sha256=digest, sampling_sidecar_sha256=sidecar_digest,
                sampling_options=metadata['sampling_options'], **shapes,
                train_K_count=len(expected_knots['train']), test_K_count=len(expected_knots['test']),
                independent_replica_count=replicas, independent_seed_count=len(actual_records))
    ensure_inputs_unchanged(cache, data)
    return data


def ensure_inputs_unchanged(cache, data, config_path=None, config_digest=None):
    if file_fingerprint(cache) != data['sha256']:
        raise ValueError('Shared data changed; do not mix training and report fingerprints')
    if file_fingerprint(Path(cache).with_suffix('.json')) != data['sampling_sidecar_sha256']:
        raise ValueError('Sampling sidecar changed during the pipeline')
    if config_path is not None and file_fingerprint(config_path) != config_digest:
        raise ValueError('Owned configuration changed during the pipeline')


def check_training_manifest(run, config_path, data_digest, *, require_mi_hits=True):
    """Require actual training to use this job's own config and audited data."""
    manifest = json.loads((Path(run) / 'manifest.json').read_text())
    config = json.loads(Path(config_path).read_text())
    if manifest.get('data_sha256') != data_digest:
        raise ValueError('Training manifest data fingerprint differs from the audited cache')
    if manifest.get('config') != config:
        raise ValueError('Training manifest configuration differs from the owned configuration')
    source = manifest.get('config_source')
    if not source or Path(source).resolve() != Path(config_path).resolve():
        raise ValueError('Training manifest must identify this job\'s owned configuration file')
    if (manifest.get('source') is not None or manifest.get('data_driven_initialization') is not None
            or manifest.get('frozen_C') or manifest.get('resumed_from')):
        raise ValueError('This job requires freshly initialized Q/C/R without a source checkpoint')
    required = dict(training_driver='rsmi_separate_qcr', precision='torch.float64',
                    estimator='histogram', environment_mode='full_shell', buffer=1,
                    independent_c=True, labels_used_in_training=False, rsmi_weight=2.)
    if any(manifest.get(key) != value for key, value in required.items()):
        raise ValueError('Training manifest does not match the default independent Q/C/R contract')
    execution = manifest.get('execution_optimizations', {})
    mi_records = execution.get('mi_statistics_cache', {})
    if require_mi_hits and execution.get('selected_empirical_mi') is not True:
        raise ValueError('CUDA training must use selected-ensemble empirical MI execution')
    if require_mi_hits and any(mi_records.get(split, {}).get('hit') is not True
                               for split in ('train', 'test')):
        raise ValueError('Training must hit both MI caches warmed before the training wall cap')
    summary = json.loads((Path(run) / 'summary.json').read_text())
    actual, requested = summary.get('actual_epochs'), summary.get('requested_epochs')
    if (requested != config['training']['epochs'] or isinstance(actual, bool)
            or not isinstance(actual, int) or not 1 <= actual <= requested):
        raise ValueError('Training summary must disclose valid actual and requested update counts')
    return dict(manifest_sha256=file_fingerprint(Path(run) / 'manifest.json'),
                actual_epochs=actual, requested_epochs=requested,
                completed_requested_updates=actual == requested,
                stopped_by_wall_limit=summary.get('stopped_by_wall_limit'),
                stopped_by_training_limit=summary.get('stopped_by_training_limit'),
                timing=summary.get('timing', {}), mi_statistics_cache=mi_records)


def build_plan(args):
    cache, output = args.cache.resolve(), args.output.resolve()
    if cache.suffix != '.pt':
        raise ValueError('cache must be a .pt file')
    if isinstance(args.replicas, bool) or not isinstance(args.replicas, int) or args.replicas < 1:
        raise ValueError('replicas must be a positive integer')
    if args.policy != 'multistep':
        raise ValueError('This single-case experiment has the fixed multistep policy')
    if not math.isfinite(args.training_wall_seconds) or args.training_wall_seconds <= 0:
        raise ValueError('Training wall cap must be finite and positive')
    if args.device != 'cpu' and not (args.device.startswith('cuda:') and args.device[5:].isdigit()):
        raise ValueError('Use an explicit cuda:N device (or CPU for a test)')
    config = make_configs(cache, epochs=args.epochs, seed=args.seed, threads=args.threads)['multistep']
    # Milestones are fixed update numbers even when a pilot/cap ends sooner.
    config['training']['lr_milestones'] = [6000, 9000]
    RunConfig.from_dict(config).validate()
    run, config_path = output / 'run', output / 'config.json'
    mi_cache = args.mi_cache_dir.resolve() if args.mi_cache_dir else output / 'mi_cache'
    training = [sys.executable, '-u', '-m', 'experiments.rsmi_separate_qcr',
                '--config', str(config_path), '--output', str(run), '--device', args.device,
                '--max-training-seconds', str(args.training_wall_seconds)]
    if args.device != 'cpu':
        training += ['--mi-cache-dir', str(mi_cache)]
    commands = [
        dict(stage='data_generation', command=[sys.executable, '-u', '-m',
             'scripts.prepare_ising2d_lr_data', '--output', str(cache),
             '--seed', str(args.seed), '--replicas', str(args.replicas)]),
        dict(stage='mi_preprocessing', command=[sys.executable, '-u', '-m',
             'scripts.prepare_qcr_mi_cache', '--config', str(config_path),
             '--output', str(output / 'mi_preprocessing.json'), '--cache-dir', str(mi_cache)]),
        dict(stage='training', command=training),
        dict(stage='scalar_report', command=[sys.executable, '-u', '-m',
             'scripts.report_scalar_joint', '--run', str(run), '--threads', '1', '--sampling-repeats', '5']),
        dict(stage='loss_report', command=[sys.executable, '-u', '-m',
             'scripts.report_mi_training', '--run', str(run), '--threads', '1']),
        dict(stage='exponent_report', command=[sys.executable, '-u', '-m',
             'scripts.report_ising_scalar_exponents', '--run', str(run)])]
    return dict(cache=str(cache), output=str(output), run=str(run), config_path=str(config_path),
                mi_cache_dir=str(mi_cache), config=config, device=args.device, replicas=args.replicas,
                policy=args.policy, commands=commands,
                protocol=dict(fresh_initialization_seed=args.seed, labels_used_in_training=False,
                    precision='torch.float64', checkpoint_selection='Final actual endpoint; no theory-based selection',
                    root_selection='All bounded-search roots and inverse-K branches retained',
                    sampling_and_mi_preprocessing_outside_training_wall_cap=True,
                    training_wall_cap_seconds=args.training_wall_seconds,
                    wall_cap_scope='Training updates and in-loop logging only; sampling/setup/MI preprocessing/final reports excluded',
                    requested_updates=args.epochs, lr_milestones=[6000, 9000], lr_gamma=.3,
                    stopping='Update limit or wall cap; no automatic retry or continuation',
                    convergence='A completed workflow or capped endpoint is not proof of convergence'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=12000)
    parser.add_argument('--seed', type=int, default=91)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--replicas', type=int, default=12)
    parser.add_argument('--policy', choices=['multistep'], default='multistep')
    parser.add_argument('--training-wall-seconds', type=float, default=900.)
    parser.add_argument('--mi-cache-dir', type=Path)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args(argv)
    plan = build_plan(args)
    if args.plan_only:
        print(json.dumps(plan, indent=2, allow_nan=False))
        return plan
    output, cache = Path(plan['output']), Path(plan['cache'])
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Refusing to overwrite or implicitly resume {output}')
    sidecar = cache.with_suffix('.json')
    if cache.is_symlink() or sidecar.is_symlink():
        raise ValueError('Cache and sidecar must be regular owned files, not symlinks')
    if cache.exists() != sidecar.exists():
        raise ValueError('Incomplete existing data/sidecar pair; inspect it, do not regenerate over it')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'logs').mkdir()
    config_path = Path(plan['config_path'])
    with config_path.open('x') as stream:
        stream.write(json.dumps(plan['config'], indent=2, allow_nan=False) + '\n')
    config_digest = file_fingerprint(config_path)
    status = dict(status='running', pid=os.getpid(), started_at=utc_now(), git_commit=revision,
                  plan=plan, config_sha256=config_digest, stages=[], current=None)
    save_status(output, status)
    started = time.monotonic()
    try:
        generation = plan['commands'][0]
        if not cache.exists():
            stage(output, status, generation['stage'], generation['command'])
        else:
            status['stages'].append(dict(name='data_generation', status='reused_existing_pair',
                                        ended_at=utc_now(), command=generation['command']))
        data = check_data(cache, replicas=args.replicas, seed=args.seed)
        status['data'] = data
        save_status(output, status)
        for command in plan['commands'][1:]:
            ensure_inputs_unchanged(cache, data, config_path, config_digest)
            stage(output, status, command['stage'], command['command'])
            ensure_inputs_unchanged(cache, data, config_path, config_digest)
            if command['stage'] == 'training':
                status['training'] = check_training_manifest(plan['run'], config_path, data['sha256'],
                                                            require_mi_hits=args.device != 'cpu')
                save_status(output, status)
    except Exception as error:
        status.update(status='failed', error=f'{type(error).__name__}: {error}', ended_at=utc_now(),
                      elapsed_seconds=time.monotonic() - started)
        save_status(output, status)
        raise
    needs_action = not status['training']['completed_requested_updates']
    status.update(status='requires_user_action' if needs_action else 'complete',
                  workflow_complete=True, requires_user_action=needs_action,
                  ended_at=utc_now(), elapsed_seconds=time.monotonic() - started)
    if needs_action:
        status['action_required'] = ('Requested updates are incomplete at the saved final endpoint; '
                                     'exact continuation requires user authorization, with no automatic retry')
    save_status(output, status)
    print(json.dumps(dict(status=status['status'], output=str(output), run=plan['run'],
                         training=status['training'])), flush=True)
    return status


if __name__ == '__main__':
    main()
