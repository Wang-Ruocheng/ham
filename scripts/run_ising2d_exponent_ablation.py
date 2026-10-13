"""Three preregistered Ising exponent arms with equal cumulative updates.

The scalar control resumes the completed 9989-update source in a new directory.
The dimg=2 arm starts randomly on the identical L36 cache. The scale arm starts
randomly on a new L108 cache and uses 108 -> 36 -> 12 -> 4. Each arm ends at
12000 updates, retaining the same Q/C/R recipe, batches, and Adam schedule.
Stages stop on errors without automatic retries or endpoint selection.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
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
from scripts.run_scalar_lr_gpu_pipeline import save_status, utc_now
from scripts.run_scalar_lr_sweep import make_configs


SOURCE_DATA_SHA256 = '909bfab8703334b19fc2a3d8dd1a2b64173261e99a86eae726f0fac4dbd54d61'
SOURCE_UPDATES = 9989
FINAL_UPDATES = 12000
ARM_NAMES = ('baseline_control', 'dimg2', 'expanded_scales')


def build_plan(args):
    """Return a side-effect-free plan; remote paths need not exist locally."""
    output = args.output.resolve()
    cache, expanded = args.cache.resolve(), args.expanded_cache.resolve()
    source = args.baseline_run.resolve()
    if cache.suffix != '.pt' or expanded.suffix != '.pt' or cache == expanded:
        raise ValueError('Use distinct .pt files for the L36 and L108 caches')
    if args.epochs != FINAL_UPDATES or args.seed != 91 or args.replicas != 12 or args.threads != 2:
        raise ValueError('This preregistered ablation fixes 12000 updates, seed91, 12 replicas, and 2 threads')
    if args.device != 'cpu' and not (args.device.startswith('cuda:') and args.device[5:].isdigit()):
        raise ValueError('Use an explicit cuda:N device (or CPU for tests)')
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('Owned output and historical baseline source must be separate directories')
    mi_cache = args.mi_cache_dir.resolve() if args.mi_cache_dir else source.parent / 'mi_cache'
    base = make_configs(cache, epochs=FINAL_UPDATES, seed=91, threads=2)['multistep']
    arms = []
    for name in ARM_NAMES:
        config = deepcopy(base)
        size, steps = (108, 3) if name == 'expanded_scales' else (36, 2)
        if name == 'dimg2':
            config['model']['theory_dim'] = 2
        if name == 'expanded_scales':
            config['data']['cache'] = str(expanded)
            config['training']['max_rg_steps'] = steps
            config['evaluation']['max_rg_steps'] = steps
        RunConfig.from_dict(config).validate()
        shapes = ScaleConfig((size, size), max_rg_steps=steps, block_factor=3).shapes()
        run, config_path = output / name / 'run', output / 'configs' / f'{name}.json'
        training = [sys.executable, '-u', '-m', 'experiments.rsmi_separate_qcr',
                    '--config', str(config_path), '--output', str(run), '--device', args.device]
        if args.device != 'cpu':
            training += ['--mi-cache-dir', str(mi_cache)]
        if name == 'baseline_control':
            training += ['--resume-checkpoint', str(source / 'checkpoint_last.pt')]
        preprocessing = [sys.executable, '-u', '-m', 'scripts.prepare_qcr_mi_cache',
                         '--config', str(config_path), '--output', str(output / name / 'mi_preprocessing.json'),
                         '--cache-dir', str(mi_cache)]
        report_module = ('scripts.report_ising2d_K_vector' if name == 'dimg2'
                         else 'scripts.report_scalar_joint')
        commands = [dict(stage=f'{name}_mi_preprocessing', command=preprocessing),
                    dict(stage=f'{name}_training', command=training),
                    dict(stage=f'{name}_phase_report', command=[sys.executable, '-u', '-m', report_module,
                         '--run', str(run), '--threads', '1', '--sampling-repeats', '5']),
                    dict(stage=f'{name}_loss_report', command=[sys.executable, '-u', '-m',
                         'scripts.report_mi_training', '--run', str(run), '--threads', '1'])]
        if name != 'dimg2':
            commands.append(dict(stage=f'{name}_exponent_report', command=[sys.executable, '-u', '-m',
                                 'scripts.report_ising_scalar_exponents', '--run', str(run)]))
        arms.append(dict(name=name, run=str(run), config_path=str(config_path), config=config,
                         size=size, max_rg_steps=steps, lattice_sizes=[shape[0] for shape in shapes],
                         initialization='Exact source checkpoint continuation' if name == 'baseline_control'
                                        else 'Fresh random initialization',
                         start_updates=SOURCE_UPDATES if name == 'baseline_control' else 0,
                         final_updates=FINAL_UPDATES, commands=commands))
    generation = dict(stage='expanded_scales_data_generation', command=[sys.executable, '-u', '-m',
        'scripts.prepare_ising2d_lr_data', '--output', str(expanded), '--size', '108',
        '--max-rg-steps', '3', '--seed', '91', '--replicas', '12'])
    return dict(output=str(output), baseline_run=str(source), cache=str(cache), expanded_cache=str(expanded),
                source_data_sha256=SOURCE_DATA_SHA256, mi_cache_dir=str(mi_cache), device=args.device,
                data_generation=generation, arms=arms,
                protocol=dict(requested_cumulative_updates=FINAL_UPDATES, batch=[16, 128],
                    seed=91, precision='torch.float64', labels_used_in_training=False,
                    training_wall_cap=None, lr_milestones=[6000, 9000], lr_gamma=.3,
                    stopping='Exactly 12000 cumulative updates; fail without retries if interrupted',
                    checkpoint_selection='Final endpoint only; no selection by proximity to theory',
                    root_selection='All bounded-search roots and all inverse-K branches',
                    scale_comparison='Lattice size and RG depth change together; same final side4',
                    magnetic_exponent='Pure B=0 K data do not identify an independent magnetic exponent',
                    convergence='Equal update counts do not certify convergence'))


def check_data(cache, *, size, max_rg_steps, expected_digest=None):
    """Validate actual bytes, sampler metadata, complete K split and replica seeds."""
    cache = Path(cache)
    sidecar = cache.with_suffix('.json')
    metadata = json.loads(sidecar.read_text())
    digest = file_fingerprint(cache)
    if metadata.get('data_sha256') != digest or (expected_digest and digest != expected_digest):
        raise ValueError('Cache bytes differ from the declared or preregistered SHA256')
    expected_options = dict(size=size, max_rg_steps=max_rg_steps, seed=91, knots=64, samples=128,
                            replicas=12, equil=4096, gap_low=512, gap_high=128, pilot_samples=1024)
    options = metadata.get('sampling_options', {})
    # The original source predates explicit depth metadata in SamplingOptions.
    if any(options.get(key, 2 if key == 'max_rg_steps' else None) != value
           for key, value in expected_options.items()):
        raise ValueError('Cache sampling settings differ from the preregistered experiment')
    shapes = [shape[0] for shape in ScaleConfig((size, size), max_rg_steps, block_factor=3).shapes()]
    if (metadata.get('lattice_sizes') != shapes or metadata.get('max_rg_steps') != max_rg_steps
            or metadata.get('block_factor') != 3 or metadata.get('B') != 0.
            or metadata.get('coupling_names') != ['K'] or metadata.get('seed') != 91
            or metadata.get('replicas') != 12):
        raise ValueError('Cache system, depth, independent replicas or seed does not match')
    values = coupling_knots(64).tolist()
    if metadata.get('couplings') != values:
        raise ValueError('Cache must retain the exact positive-K64 grid and critical-point insertion')
    expected_records = [(i, replica, k, i % 3 == 1, derived_seed(91, 1, i * 12 + replica),
                         derived_seed(91, 2, i * 12 + replica))
                        for i, k in enumerate(values) for replica in range(12)]
    actual = [(r.get('K_index'), r.get('replica'), r.get('K'), r.get('test'), r.get('seed'),
               r.get('global_flip_seed')) for r in metadata.get('ensembles', [])]
    if actual != expected_records or len({record[4] for record in actual}) != 768:
        raise ValueError('Each K requires 12 documented independent production and symmetry seeds')
    train, test = load_ensembles(cache)
    results = {}
    for split, dataset in [('train', train), ('test', test)]:
        knots = [k for i, k in enumerate(values) if (i % 3 == 1) == (split == 'test')]
        shape = (len(knots) * 12, 128, size, size, 1)
        x, k = dataset.configurations, dataset.couplings
        if (tuple(x.shape) != shape or k is None or tuple(k.shape) not in ((shape[0],), (shape[0], 1))
                or x.dtype != torch.float64 or k.dtype != torch.float64):
            raise ValueError(f'{split}: actual cache shape/dtype or K metadata differ from {shape}')
        if (metadata.get(split + '_shape') != list(shape) or metadata.get(split + '_K') != knots
                or Counter(k.flatten().tolist()) != Counter({value: 12 for value in knots})):
            raise ValueError(f'{split}: complete K-point split or independent-replica count differs')
        results[split + '_shape'] = list(shape)
    return dict(sha256=digest, sampling_sidecar_sha256=file_fingerprint(sidecar),
                sampling_options=options, lattice_sizes=shapes, **results)


def check_source(source, config, *, device='cuda:0'):
    """Require the actual completed source, including RNG/Adam and full config."""
    source = Path(source)
    manifest = json.loads((source / 'manifest.json').read_text())
    summary = json.loads((source / 'summary.json').read_text())
    checkpoint_path = source / 'checkpoint_last.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if manifest.get('config') != config or summary.get('config') != config:
        raise ValueError('Historical source full configuration differs from baseline control')
    if (summary.get('actual_epochs') != SOURCE_UPDATES or summary.get('requested_epochs') != FINAL_UPDATES
            or checkpoint.get('epoch') != SOURCE_UPDATES):
        raise ValueError('Control must resume the actual 9989-update source into 12000 cumulative updates')
    metadata = checkpoint.get('metadata', {})
    if metadata.get('config') != config or metadata.get('data_sha256') != SOURCE_DATA_SHA256:
        raise ValueError('Source checkpoint metadata or original data fingerprint differs')
    required = ('optimizer', 'batch_generator_state', 'rsmi_generator_state', 'torch_rng_state',
                'history', 'lr_scheduler_state')
    if any(key not in checkpoint for key in required):
        raise ValueError('Source checkpoint must contain exact Adam, minibatch, C sampling and history states')
    if (manifest.get('data_sha256') != SOURCE_DATA_SHA256 or manifest.get('labels_used_in_training') is not False
            or manifest.get('frozen_C') or manifest.get('data_driven_initialization')):
        raise ValueError('Source must be the original unsupervised random trainable QCR run')
    if metadata.get('execution_optimizations', {}).get('selected_empirical_mi') is not True:
        raise ValueError('Source must retain the selected empirical MI execution route')
    if device.startswith('cuda:') and ('cuda_rng_state' not in checkpoint
            or checkpoint.get('cuda_device_index') != int(device[5:])):
        raise ValueError('Control must preserve the source CUDA device and saved CUDA RNG')
    if device == 'cpu' and 'cuda_rng_state' in checkpoint:
        raise ValueError('A CUDA source cannot be resumed exactly on CPU')
    if checkpoint['lr_scheduler_state'].get('completed_updates') != SOURCE_UPDATES:
        raise ValueError('Source cumulative learning-rate state differs from the completed updates')
    return dict(actual_epochs=SOURCE_UPDATES, files={str(source / name): file_fingerprint(source / name)
        for name in ('checkpoint_last.pt', 'manifest.json', 'summary.json')})


def ensure_unchanged(data, source, configs):
    for cache, record in data.items():
        if (file_fingerprint(cache) != record['sha256']
                or file_fingerprint(Path(cache).with_suffix('.json')) != record['sampling_sidecar_sha256']):
            raise ValueError('Audited data or sampling sidecar changed during the ablation')
    for path, digest in {**source['files'], **configs}.items():
        if file_fingerprint(path) != digest:
            raise ValueError('Historical source or owned configuration changed during the ablation')


def check_training(arm, data, *, require_mi_hits):
    run = Path(arm['run'])
    manifest = json.loads((run / 'manifest.json').read_text())
    summary = json.loads((run / 'summary.json').read_text())
    if manifest.get('config') != arm['config'] or manifest.get('data_sha256') != data['sha256']:
        raise ValueError('Training manifest differs from owned full config or audited data')
    if Path(manifest.get('config_source', '')).resolve() != Path(arm['config_path']).resolve():
        raise ValueError('Training must identify the owned arm configuration')
    required = dict(training_driver='rsmi_separate_qcr', precision='torch.float64', estimator='histogram',
                    environment_mode='full_shell', buffer=1, independent_c=True,
                    labels_used_in_training=False, rsmi_weight=2.)
    if any(manifest.get(key) != value for key, value in required.items()):
        raise ValueError('Training did not retain the independent Q/C/R recipe')
    is_resume = arm['name'] == 'baseline_control'
    if bool(manifest.get('resumed_from')) != is_resume or manifest.get('frozen_C') or manifest.get('data_driven_initialization'):
        raise ValueError('Control must resume exactly and both ablation arms must initialize randomly')
    if is_resume and manifest['resumed_from'] != arm['commands'][1]['command'][-1]:
        raise ValueError('Control resumed from an unexpected checkpoint')
    execution = manifest.get('execution_optimizations', {})
    records = execution.get('mi_statistics_cache', {})
    if require_mi_hits and (execution.get('selected_empirical_mi') is not True or
            any(records.get(split, {}).get('hit') is not True for split in ('train', 'test'))):
        raise ValueError('CUDA updates must retain selected empirical MI and hit preprocessed v1 caches')
    if (type(summary.get('actual_epochs')) is not int or summary.get('actual_epochs') != FINAL_UPDATES
            or summary.get('requested_epochs') != FINAL_UPDATES or summary.get('config') != arm['config']
            or summary.get('stopped_by_training_limit') or summary.get('stopped_by_wall_limit')):
        raise ValueError('Training did not complete exactly 12000 cumulative updates; requires user action')
    scheduler = summary.get('lr_scheduler_state', {})
    if scheduler.get('completed_updates') != FINAL_UPDATES or not math.isclose(
            scheduler.get('next_learning_rate', -1.), .001 * .3 ** 2, rel_tol=1e-12):
        raise ValueError('Completed training must retain the cumulative multistep clock and final 9e-5 rate')
    return dict(actual_epochs=FINAL_UPDATES, requested_epochs=FINAL_UPDATES,
                new_updates=FINAL_UPDATES - arm['start_updates'], timing=summary.get('timing', {}),
                lr_scheduler_state=summary.get('lr_scheduler_state'), manifest_sha256=file_fingerprint(run / 'manifest.json'))


def check_reports(arm):
    """Require completed public reports for the preregistered final endpoint."""
    run = Path(arm['run'])
    names = ['full_report/metrics.json', 'full_report/fixed_points.json',
             'full_report/critical_exponents.json', 'full_report/encodings.npz',
             'full_report/coupling_by_scale.png', 'report/loss_comparison.png',
             ('full_report/vector_critical_summary.png' if arm['name'] == 'dimg2'
              else 'full_report/scalar_critical_summary.png')]
    if any(not (run / name).is_file() for name in names):
        raise ValueError('Final report stages did not produce all required public artifacts')
    exponents = json.loads((run / 'full_report/critical_exponents.json').read_text())
    if exponents.get('checkpoint_epoch') != FINAL_UPDATES or not isinstance(exponents.get('roots'), list):
        raise ValueError('Final exponent report must retain all roots at the actual 12000-update endpoint')
    return dict(checkpoint_epoch=FINAL_UPDATES, bounded_root_count=len(exponents['roots']),
                files={name: file_fingerprint(run / name) for name in names})


def stage(output, status, name, command):
    started = time.monotonic()
    record = dict(name=name, command=command, started_at=utc_now(), status='running',
                  log=str(output / 'logs' / f'{name}.log'))
    status['current'] = record
    with Path(record['log']).open('x') as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   env={**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'})
        record['pid'] = process.pid
        save_status(output, status)
        print(json.dumps(record), flush=True)
        code = process.wait()
    record.update(status='failed' if code else 'complete', returncode=code, ended_at=utc_now(),
                  elapsed_seconds=time.monotonic() - started)
    status['stages'].append(record)
    status['current'] = None
    save_status(output, status)
    if code:
        raise RuntimeError(f'{name} exited {code}; see {record["log"]}; no automatic retry')


def _pair_exists(cache):
    cache = Path(cache)
    sidecar = cache.with_suffix('.json')
    if cache.is_symlink() or sidecar.is_symlink():
        raise ValueError('Cache and sidecar must be regular files, not symlinks')
    if cache.exists() != sidecar.exists():
        raise ValueError('Incomplete data/sidecar pair; inspect it rather than regenerating')
    return cache.exists()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-run', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--expanded-cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=FINAL_UPDATES)
    parser.add_argument('--seed', type=int, default=91)
    parser.add_argument('--replicas', type=int, default=12)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--mi-cache-dir', type=Path)
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args(argv)
    plan = build_plan(args)
    if args.plan:
        print(json.dumps(plan, indent=2, allow_nan=False))
        return plan
    output = Path(plan['output'])
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Refusing to overwrite or implicitly resume {output}')
    if not _pair_exists(plan['cache']):
        raise FileNotFoundError('The original L36 data must exist; this ablation never resamples it')
    expanded_exists = _pair_exists(plan['expanded_cache'])
    source = check_source(plan['baseline_run'], plan['arms'][0]['config'], device=args.device)
    data = {plan['cache']: check_data(plan['cache'], size=36, max_rg_steps=2, expected_digest=SOURCE_DATA_SHA256)}
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'logs').mkdir()
    (output / 'configs').mkdir()
    configs = {}
    for arm in plan['arms']:
        (output / arm['name']).mkdir()
        path = Path(arm['config_path'])
        with path.open('x') as stream:
            stream.write(json.dumps(arm['config'], indent=2, allow_nan=False) + '\n')
        configs[str(path)] = file_fingerprint(path)
    status = dict(status='running', started_at=utc_now(), pid=os.getpid(), git_commit=revision,
                  plan=plan, source=source, data=data, config_sha256=configs,
                  stages=[], current=None, training={}, reports={})
    save_status(output, status)
    started = time.monotonic()
    try:
        # Finish the matched control and dimensional arm before the larger sampling stage.
        for arm in plan['arms']:
            if arm['name'] == 'expanded_scales':
                ensure_unchanged(data, source, configs)
                generation = plan['data_generation']
                if expanded_exists:
                    status['stages'].append(dict(name=generation['stage'], status='reused_existing_pair',
                                                ended_at=utc_now(), command=generation['command']))
                else:
                    stage(output, status, generation['stage'], generation['command'])
                data[plan['expanded_cache']] = check_data(plan['expanded_cache'], size=108, max_rg_steps=3)
                save_status(output, status)
            cache = arm['config']['data']['cache']
            for command in arm['commands']:
                ensure_unchanged(data, source, configs)
                stage(output, status, command['stage'], command['command'])
                ensure_unchanged(data, source, configs)
                if command['stage'].endswith('_training'):
                    status['training'][arm['name']] = check_training(arm, data[cache],
                                                                    require_mi_hits=args.device != 'cpu')
                    save_status(output, status)
            status['reports'][arm['name']] = check_reports(arm)
            save_status(output, status)
    except Exception as error:
        status.update(status='failed', requires_user_action=True, error=f'{type(error).__name__}: {error}',
                      ended_at=utc_now(), elapsed_seconds=time.monotonic() - started)
        save_status(output, status)
        raise
    status.update(status='complete', workflow_complete=True, ended_at=utc_now(),
                  elapsed_seconds=time.monotonic() - started)
    save_status(output, status)
    print(json.dumps(dict(status='complete', output=str(output))), flush=True)
    return status


if __name__ == '__main__':
    main()
