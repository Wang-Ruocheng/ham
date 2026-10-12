"""Same-data GPU QCR pilot: only the cumulative Adam learning-rate policy changes.

Prepare data separately with scripts.prepare_ising2d_lr_data. This pipeline
never changes the locked Q/C/R objectives, selects a checkpoint using theory,
or resumes a failed run. Each final endpoint gets a selection-free exponent
audit. A capped run is not automatically a converged run.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from recurrent_rg.config import RunConfig, ScaleConfig
from recurrent_rg.data.ensembles import load_ensembles
from recurrent_rg.runner import file_fingerprint


def make_configs(cache, *, epochs=12000, seed=91, threads=2):
    if epochs < 8 or seed < 0 or threads < 1:
        raise ValueError('Need at least eight updates, a nonnegative seed, and positive threads')
    base = dict(
        schema_version=5,
        data=dict(cache=str(Path(cache).resolve()), physical_model='ising2d', coupling_names=['K'],
                  spin_flip_augment=False, spin_flip_symmetric=True,
                  train_pool_replicas=1, test_pool_replicas=1),
        model=dict(dimension=2, theory_dim=1, hidden=96, width=3,
                   block_factor=3, kernel_layout='nonoverlapping', categorical_states=0,
                   fixed_kernel=False, binary_lookup=True, kernel_initialization='random',
                   kernel_sampling='bernoulli', kernel_temperature=1.,
                   affine_reconstruction=True, reconstruction_hidden=0, theory_head='nonlinear'),
        loss=dict(balanced_reconstruction_weight=0., rollout_weight=5.,
                  isotropy_weight=0., tangent_weight=0., orientation_weight=0.,
                  kernel_sign_weight=1., kernel_neutral_weight=0., kernel_distinct_alpha=0.,
                  closure_metric='euclidean', coordinate_reduction='sum',
                  kernel_noncollapse_mode='variance_floor_ensemble', kernel_variance_ratio=1.,
                  weights=dict(anchor=2., coarse_consistency=5., kernel_noncollapse=2.,
                               separation=10., commutator=5.)),
        training=dict(max_rg_steps=2, epochs=epochs, system_batch=16, sample_batch=128,
                      learning_rate=.001, seed=seed, threads=threads, train_modules=['Q', 'C', 'R'],
                      checkpoint_every=1000, log_every=200, gradient_mode='qc_r_separate'),
        evaluation=dict(max_rg_steps=2, samples=128, systems=24))
    schedules = dict(
        constant=dict(lr_schedule='constant'),
        multistep=dict(lr_schedule='multistep', lr_milestones=[epochs // 2, 3 * epochs // 4], lr_gamma=.3),
        cosine=dict(lr_schedule='cosine', lr_cosine_updates=epochs, lr_min_factor=.05))
    result = {}
    for name, policy in schedules.items():
        raw = deepcopy(base)
        raw['training'].update(policy)
        cfg = RunConfig.from_dict(raw)
        cfg.validate()
        # JSON-normalize tuples once so every written config and plan agrees.
        result[name] = json.loads(json.dumps(cfg.to_dict()))
    return result


def check_data(cache):
    metadata = json.loads(Path(cache).with_suffix('.json').read_text())
    expected_sampling = dict(size=36, knots=64, samples=128, replicas=3,
                             equil=4096, gap_low=512, gap_high=128, pilot_samples=1024)
    if any(metadata.get('sampling_options', {}).get(key) != value
           for key, value in expected_sampling.items()):
        raise ValueError('Cache sidecar must document the preregistered production sampling settings')
    if metadata.get('B') != 0. or metadata.get('coupling_names') != ['K']:
        raise ValueError('Cache sidecar must identify the single-layer zero-field K scan')
    digest = file_fingerprint(cache)
    if metadata.get('data_sha256') != digest:
        raise ValueError('Cache bytes differ from the sampling sidecar SHA256')
    train, test = load_ensembles(cache)
    for split, ensemble in (('train', train), ('test', test)):
        x, k = ensemble.configurations, ensemble.couplings
        expected_shape = (129 if split == 'train' else 63, 128, 36, 36, 1)
        if tuple(x.shape) != expected_shape or k.shape[-1] != 1:
            raise ValueError(f'{split}: this production pilot requires shape {expected_shape} and K-only labels')
        if not ((k >= .2 - 1e-12) & (k <= .68 + 1e-12)).all():
            raise ValueError(f'{split}: K must be within the declared [.2,.68] zero-field scan')
        labels = k.flatten().tolist()
        knots = set(metadata.get(split + '_K', []))
        if set(labels) != knots or any(labels.count(value) != 3 for value in knots):
            raise ValueError(f'{split}: K points and three replicas must match the sampling sidecar')
        ScaleConfig((36, 36), max_rg_steps=2, block_factor=3).shapes()
    if set(train.couplings.flatten().tolist()) & set(test.couplings.flatten().tolist()):
        raise ValueError('Complete coupling points, including all replicas, must not cross the split')
    ensure_data_unchanged(cache, digest)
    return dict(sha256=digest, sampling_sidecar_sha256=file_fingerprint(Path(cache).with_suffix('.json')),
                sampling_options=metadata['sampling_options'],
                train_shape=list(train.configurations.shape), test_shape=list(test.configurations.shape),
                train_K_count=len(set(train.couplings.flatten().tolist())),
                test_K_count=len(set(test.couplings.flatten().tolist())))


def ensure_data_unchanged(cache, expected_digest):
    if file_fingerprint(cache) != expected_digest:
        raise ValueError('Shared data changed during the comparison; do not mix training and report fingerprints')


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def save_status(out, status):
    path = out / 'pipeline.json'
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(status, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def build_plan(args):
    cache, out = args.cache.resolve(), args.output.resolve()
    if not math.isfinite(args.max_wall_seconds_per_run) or args.max_wall_seconds_per_run <= 0:
        raise ValueError('Training wall cap must be finite and positive')
    if args.device != 'cpu' and not (args.device.startswith('cuda:') and args.device[5:].isdigit()):
        raise ValueError('Use an explicit cuda:N device (or CPU for a test)')
    configs = make_configs(cache, epochs=args.epochs, seed=args.seed, threads=args.threads)
    cases = []
    for name, cfg in configs.items():
        run, conf = out / name, out / 'configs' / f'{name}.json'
        commands = [
            ('training', [sys.executable, '-m', 'experiments.rsmi_separate_qcr',
                          '--config', str(conf), '--output', str(run), '--device', args.device,
                          '--max-wall-seconds', str(args.max_wall_seconds_per_run)]),
            ('scalar_report', [sys.executable, '-m', 'scripts.report_scalar_joint',
                               '--run', str(run), '--threads', '1', '--sampling-repeats', '5']),
            ('loss_report', [sys.executable, '-m', 'scripts.report_mi_training',
                             '--run', str(run), '--threads', '1']),
            ('exponent_report', [sys.executable, '-m', 'scripts.report_ising_scalar_exponents',
                                 '--run', str(run)])]
        cases.append(dict(name=name, run=str(run), config_path=str(conf), config=cfg,
                          commands=[dict(stage=stage, command=cmd) for stage, cmd in commands]))
    audit = [sys.executable, '-m', 'scripts.audit_scalar_training_sweep']
    for case in cases:
        audit += ['--run', case['run']]
    audit += ['--output', str(out / 'comparison')]
    return dict(cache=str(cache), output=str(out), device=args.device, cases=cases, audit_command=audit,
                protocol=dict(
                    only_changed_training_setting='Adam cumulative-update learning-rate policy',
                    fresh_identical_seed_per_case=args.seed,
                    thermal_reference=dict(b=3, thermal_lambda=3., nu=1.),
                    checkpoint_selection='Final actual endpoint; no theory-based checkpoint selection',
                    root_selection='All bounded-search roots and inverse-K branches retained',
                    wall_cap_seconds_per_case=args.max_wall_seconds_per_run,
                    wall_cap_scope='Training only, includes setup; data generation and reports are separate',
                    equal_exposure='Compare actual updates; cold MI-cache setup can make equal wall caps unequal, and a capped multistep run may never reach its first decay',
                    convergence='Update/wall limits are not proof of a TRAIN Q/C/R platform',
                    validation='One seed is a pilot; confirm a frozen strategy with independent seeds and finite-size audits',
                    labels_used_in_training=False,
                    no_model_loss_precision_or_gradient_changes=True))


def run_stage(out, status, name, stage, command):
    record = dict(case=name, stage=stage, command=command, started_at=utc_now(), status='running')
    log = out / 'logs' / f'{name}_{stage}.log'
    record['log'] = str(log)
    status['current'] = record
    with log.open('x') as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        record['pid'] = process.pid
        save_status(out, status)
        print(json.dumps(record), flush=True)
        code = process.wait()
    record.update(returncode=code, ended_at=utc_now(), status='complete' if code == 0 else 'failed')
    status['stages'].append(record)
    status['current'] = None
    save_status(out, status)
    if code:
        raise RuntimeError(f'{name}/{stage} failed with exit {code}; see {log}; no automatic retry')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=12000)
    parser.add_argument('--seed', type=int, default=91)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--max-wall-seconds-per-run', type=float, default=2700.)
    parser.add_argument('--plan-only', action='store_true', help='Print plan without files, loading data, or subprocesses')
    args = parser.parse_args(argv)
    plan = build_plan(args)
    if args.plan_only:
        print(json.dumps(plan, indent=2, allow_nan=False))
        return plan
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite or implicitly resume {out}')
    data = check_data(args.cache.resolve())
    status = dict(status='running', started_at=utc_now(), pid=os.getpid(), plan=plan,
                  data=data, stages=[], current=None)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'configs').mkdir()
    (out / 'logs').mkdir()
    for case in plan['cases']:
        Path(case['config_path']).write_text(json.dumps(case['config'], indent=2) + '\n')
    save_status(out, status)
    begin = time.monotonic()
    try:
        for case in plan['cases']:
            for stage in case['commands']:
                ensure_data_unchanged(args.cache, data['sha256'])
                run_stage(out, status, case['name'], stage['stage'], stage['command'])
                ensure_data_unchanged(args.cache, data['sha256'])
        ensure_data_unchanged(args.cache, data['sha256'])
        run_stage(out, status, 'comparison', 'audit', plan['audit_command'])
    except Exception as error:
        status.update(status='failed', error=f'{type(error).__name__}: {error}', ended_at=utc_now())
        save_status(out, status)
        raise
    status.update(status='complete', ended_at=utc_now(), elapsed_seconds=time.monotonic() - begin)
    save_status(out, status)
    print(json.dumps(dict(status=status['status'], output=str(out))), flush=True)
    return status


if __name__ == '__main__':
    main()
