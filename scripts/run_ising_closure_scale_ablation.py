"""Isolated two-arm, label-free closure-scale ablation on completed new data.

No physical sampling, old data, checkpoint initialization, resume or repair.
The 40-update benchmark is execution-only and is never a scientific arm.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys

from scripts import run_ising_future_physics as execution
from scripts.prepare_ising_future_physics import make_config


EXPERIMENT = 'ising_closure_scale_weights_b3_paired_v1'
PROFILES = {'uniform': [1., 1.], 'late': [.2, 1.8]}
GPU_UUID = execution.GPU_UUID
GPU_SECONDS = 3600
CPU_SECONDS = execution.CPU_SECONDS
SOURCE_DATA_SHA = '53afdad1c632b810f7bf00336edc46a75f565bb9d0f4dcb192403cc1358d06de'
file_sha = execution.file_sha
save = execution.save
now = execution.now


def read(path):
    return json.loads(Path(path).read_text())


def subset_equal(raw, expanded):
    return (isinstance(expanded, dict) and all(key in expanded and subset_equal(value, expanded[key])
            for key, value in raw.items())) if isinstance(raw, dict) else raw == expanded


def ordinary_file(path, root):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Input must be a regular file inside the completed new source: {path}')
    return path


def validated_source(source):
    """Validate JSON contracts and hash only the explicit new-data inputs."""
    source = source.resolve()
    job, config = read(source / 'job.json'), read(source / 'config.json')
    receipt, pairs = read(source / 'dataset_receipt.json'), read(source / 'pairs.json')
    manifest, summary = read(source / 'run_control/manifest.json'), read(source / 'run_control/summary.json')
    if (job.get('experiment') != 'ising_future_physics_Q_loss_b3_paired_v1' or
            job.get('status') != 'complete' or job.get('workflow_complete') is not True or
            job.get('current') is not None or job.get('child_still_running', False) or
            not job.get('stages') or
            any(row.get('returncode') != 0 or row.get('status') != 'complete' for row in job.get('stages', []))):
        raise ValueError('Source must be the completed new b3 paired experiment, never a running/failed source')
    data = ordinary_file(source / 'data/ensembles.pt', source)
    if config != make_config(data):
        raise ValueError('Source must retain the exact fresh b3/g1/seed91/6000 original QCR recipe')
    future = manifest.get('q_future_physics', {})
    if (summary.get('actual_epochs') != 6000 or manifest.get('labels_used_in_training') is not False or
            future.get('mode') != 'paired_control' or future.get('weight') != 0 or
            future.get('labels_used') is not False or manifest.get('auxiliary_future_parameters') != 0 or
            not subset_equal(config, manifest.get('config')) or
            summary.get('config') != manifest.get('config') or
            any(summary.get(key, False) for key in ('stopped_by_wall_limit', 'stopped_by_training_limit',
                                                   'stopped_by_train_plateau'))):
        raise ValueError('Completed source control must be unsupervised actual6000 with no future predictor')
    data_sha, pair_sha = file_sha(data), file_sha(source / 'pairs.json')
    if (data_sha != SOURCE_DATA_SHA or data_sha != receipt.get('data_sha256') or data_sha != pairs.get('data_sha256') or
            pair_sha != receipt.get('pair_sha256') or data_sha != manifest.get('data_sha256') or
            data_sha != future.get('data_sha256') or pair_sha != future.get('pairs_sha256') or
            receipt.get('train_shape') != [112, 64, 108, 108, 1] or
            receipt.get('test_shape') != [324, 64, 108, 108, 1] or
            receipt.get('old_9GB_dataset_opened') is not False):
        raise ValueError('Completed new-data fingerprint, shape or paired-sampler identity mismatch')
    cache = manifest.get('execution_optimizations', {}).get('mi_statistics_cache', {})
    if set(cache) != {'train', 'test'}:
        raise ValueError('Require completed TRAIN and TEST new-data MI statistics')
    names = ['data/ensembles.pt', 'pairs.json', 'dataset_receipt.json']
    for split in ('train', 'test'):
        row = cache[split]
        name = f'mi-v1-{row.get("key")}.pt'
        path = ordinary_file(source / 'mi_cache' / name, source)
        if Path(row.get('path', '')).resolve() != path.resolve() or row.get('cache_version') != 1:
            raise ValueError('Source MI cache identity/path mismatch')
        names.append(str(path.relative_to(source)))
    if set(p.name for p in (source / 'mi_cache').glob('mi-v1-*.pt')) != {Path(n).name for n in names[3:]}:
        raise ValueError('Source must contain exactly its two completed data-only MI caches')
    hashes = {name: file_sha(ordinary_file(source / name, source)) for name in names}
    hashes['data/ensembles.pt'] = data_sha
    hashes['pairs.json'] = pair_sha
    return dict(source=str(source), input_sha256=hashes,
        source_json_sha256={name: file_sha(source / name) for name in
            ('job.json', 'config.json', 'run_control/manifest.json', 'run_control/summary.json')},
        data_sha256=data_sha, pair_sha256=pair_sha, warning_count=receipt.get('warning_count'),
        old_9GB_dataset_opened=False, physical_sampling_updates=0,
        reused_model_or_optimizer=False, source_checkpoints_opened=False)


def prepare_inputs(root, source, receipt):
    """Hardlink immutable, completed inputs only; never fall back to a rebuild."""
    (root / 'data').mkdir(); (root / 'mi_cache').mkdir()
    for name, digest in receipt['input_sha256'].items():
        original = ordinary_file(source / name, source)
        if file_sha(original) != digest:
            raise ValueError(f'Completed source changed before immutable reuse: {name}')
        os.link(original, root / name)
    for name, weights in PROFILES.items():
        config = make_config(root / 'data/ensembles.pt')
        config['loss']['closure_scale_weights'] = weights
        save(root / f'config_{name}.json', config)
    # Existing collection/inspection conventions can still find a root config.
    save(root / 'config.json', read(root / 'config_uniform.json'))
    benchmark = read(root / 'config_uniform.json')
    benchmark['training'].update(epochs=40, checkpoint_every=40, log_every=20)
    save(root / 'benchmark_config.json', benchmark)
    save(root / 'input_reuse.json', dict(**receipt, destination=str(root), immutable_hardlinks=True,
        MI_cache_validation='Trainer validates unchanged data-only cache identities; no MI rebuild intended',
        pairing='Retain completed control paired sampler only; no future loss, predictor or normalizer',
        source_immutability='No source bytes or historical checkpoints are modified'))


def assert_frozen(root, job):
    plan = read(root / 'plan.json')
    pin = plan['source_code_sha256']
    if execution.source_hashes(Path(job['code_snapshot'])) != pin:
        raise RuntimeError('Frozen science source fingerprint changed; no next stage started')
    for relative, digest in job['input_reuse']['input_sha256'].items():
        target = root / relative
        if target.is_symlink() or not target.is_file():
            raise RuntimeError(f'Immutable input disappeared or became a symlink: {relative}')
        if file_sha(target) != digest:
            raise RuntimeError(f'Immutable new-data/MI/paired-sampler fingerprint changed: {relative}')
    for name, digest in plan['configuration_sha256'].items():
        if file_sha(root / name) != digest:
            raise RuntimeError(f'Frozen experiment configuration changed: {name}')


def run_stage(root, job, name, command, *, gpu=False):
    assert_frozen(root, job)
    execution.run_stage(root, job, name, command, gpu=gpu)


def trainer_command(root, output, config, *, limit=2400):
    return [sys.executable, '-m', 'experiments.rsmi_separate_qcr', '--config', str(config),
        '--output', str(output), '--device', 'cuda:2', '--q-future-control',
        '--q-future-pairs', str(root / 'pairs.json'), '--mi-cache-dir', str(root / 'mi_cache'),
        '--mi-test-cpu', '--skip-final-report', '--max-training-seconds', str(limit),
        '--benchmark-warmup', '5']


def validate_arm(root, name, expected_steps):
    run_path = root / ('benchmark' if name == 'benchmark' else f'run_{name}')
    manifest, summary = read(run_path / 'manifest.json'), read(run_path / 'summary.json')
    weights = PROFILES['uniform' if name == 'benchmark' else name]
    scale = manifest.get('closure_scale_weighting', {})
    initial = manifest.get('initial_module_sha256', {})
    future = manifest.get('q_future_physics', {})
    if (summary.get('actual_epochs') != expected_steps or summary.get('requested_epochs') != expected_steps or
            any(summary.get(key, False) for key in ('stopped_by_wall_limit', 'stopped_by_training_limit',
                                                   'stopped_by_train_plateau'))):
        raise execution.BudgetHold(f'{name}未完成固定{expected_steps}次更新；不续训或重跑')
    normalized = scale.get('normalized_weights', [])
    if (scale.get('weights') != weights or scale.get('target_levels') != [1, 2] or
            len(normalized) != 2 or any(not math.isclose(a, b, rel_tol=1e-14, abs_tol=1e-15)
                for a, b in zip(normalized, [w / sum(weights) for w in weights])) or
            scale.get('application') != ['one_step_commutator', 'microscopic_start_rollout'] or
            scale.get('labels_used') is not False or
            summary.get('contract', {}).get('closure_scale_weighting') != scale or
            set(initial) != {'Q', 'C', 'R'} or summary.get('contract', {}).get('initial_module_sha256') != initial or
            manifest.get('labels_used_in_training') is not False or
            future.get('mode') != 'paired_control' or future.get('weight') != 0 or
            manifest.get('auxiliary_future_parameters') != 0):
        raise RuntimeError('Completed arm scale-weight, no-label, no-future or initial-state contract mismatch')
    for other in ('benchmark', 'uniform', 'late'):
        if other == name:
            continue
        other_path = root / ('benchmark' if other == 'benchmark' else f'run_{other}') / 'manifest.json'
        if other_path.exists() and read(other_path).get('initial_module_sha256') != initial:
            raise RuntimeError('Paired fresh Q/C/R initial fingerprints differ')
    caches = manifest.get('execution_optimizations', {}).get('mi_statistics_cache', {})
    if set(caches) != {'train', 'test'} or any(not row.get('hit') for row in caches.values()):
        raise RuntimeError('New-data MI cache was not reused exactly; no automatic rebuild accepted')
    return summary


def collect(root, run, device):
    """Five C draws with training-aligned Q and explicit full-Q diagnostics."""
    import numpy as np
    import torch
    from recurrent_rg.config import RunConfig
    from recurrent_rg.checkpoints import load_run, isolated_rng
    from recurrent_rg.data.ensembles import load_ensembles
    from recurrent_rg.execution import configure_cuda_execution
    from scripts.prepare_ising_future_physics import state_sha
    from scripts.ising_qcr_evaluation import collect_saved_encodings
    root, run = root.resolve(), run.resolve()
    if run not in (root / 'run_uniform', root / 'run_late'):
        raise ValueError('Collect only the two completed new ablation arms, never historical checkpoints')
    validate_arm(root, run.name.removeprefix('run_'), 6000)
    out = run / 'report'; out.mkdir(exist_ok=False)
    loaded = load_run(run / 'checkpoint_last.pt')
    cfg = RunConfig.from_dict(loaded.config); torch.set_num_threads(1)
    device = torch.device(device)
    if device.type == 'cuda': torch.cuda.set_device(device)
    model = loaded.model.to(device); model.detach_kernel_representation = True
    configure_cuda_execution(model, device, independent_c=True)
    before = state_sha(model)
    train, test = load_ensembles(cfg.data.cache)
    if file_sha(cfg.data.cache) != loaded.payload['metadata']['data_sha256']:
        raise ValueError('New training cache changed')
    draw_count = 5
    with isolated_rng(device, 2026101723), torch.no_grad():
        arrays, encoding = collect_saved_encodings(model, [('train', train), ('test', test)], device,
                                                   draw_count=draw_count)
    if state_sha(model) != before: raise RuntimeError('Evaluation changed model parameters')
    np.savez_compressed(out / 'encodings.npz', **arrays)
    save(out / 'metrics.json', dict(schema_version=2, status='complete', actual_steps=loaded.payload['epoch'],
        scales=[108, 36, 12], C_draws=draw_count, samples=encoding['samples'], config=cfg.to_dict(),
        encoding_estimator=encoding['estimator'], encoding_protocol=encoding,
        encoding_matches_training=True,
        model_sha256_before=before, model_sha256_after=state_sha(model),
        C_sha256=state_sha(model.rg_kernel), Q_sha256=state_sha(model.theory_encoder),
        encodings_sha256=file_sha(out / 'encodings.npz'), thermal_nu=None,
        qualification='Single fixed training seed and finite physical samples; no root search or thermal-mode certification.'))
    print(json.dumps(dict(status='collected', run=str(run), actual_steps=loaded.payload['epoch'])), flush=True)


def public_files(root):
    paths = [root / name for name in ('job.json', 'plan.json', 'input_reuse.json', 'dataset_receipt.json',
                                     'pairs.json', 'config.json', 'config_uniform.json', 'config_late.json',
                                     'benchmark_config.json', 'launch.json')]
    for name in ('benchmark', 'run_uniform', 'run_late'):
        paths.extend(root / name / item for item in ('manifest.json', 'summary.json'))
        paths.extend((root / name / 'report').glob('*'))
    paths.extend((root / 'report').glob('*'))
    return sorted(str(path.relative_to(root)) for path in paths
                  if path.is_file() and path.suffix.lower() in ('.json', '.npz', '.png'))


def run(root, source):
    root, source = root.resolve(), source.resolve()
    if root.exists():
        # Only the launcher may precreate this exact receipt/log directory.
        if set(p.name for p in root.iterdir()) != {'launch.json', 'logs'} or (root / 'job.json').exists():
            raise FileExistsError('New scale ablation must use a fresh output; no resume/overwrite/restart')
    else:
        root.mkdir(parents=True)
        (root / 'logs').mkdir()
    if socket.gethostname() != 'thinklab-105-225':
        raise RuntimeError('This bounded launch is pinned to thinklab-105-225')
    os.setpriority(os.PRIO_PROCESS, 0, 10)
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    job = dict(schema_version=1, experiment=EXPERIMENT, status='validating_source', workflow_complete=False,
        started_at=now(), parent_pid=os.getpid(), source=str(source), output=str(root),
        code_snapshot=str(Path.cwd().resolve()), device='cuda:2', gpu_uuid=GPU_UUID,
        GPU_card_seconds=GPU_SECONDS, CPU_seconds_completed=0., gpu_seconds_completed=0.,
        gpu_seconds_current_stage=0., peak_own_gpu_mib=0, campaign_peak_own_gpu_mib=0,
        cpu_threads=1, nice=10, current=None, stages=[], error=None, requires_user_action=False,
        authorization='用户明确要求提高高k闭合权重、降低低k闭合权重并试验', thermal_nu=None,
        public_files=[])
    save(root / 'job.json', job)
    plan = dict(schema_version=1, source=str(source),
        training=dict(profiles=PROFILES, updates_per_arm=6000, seed=91, batch_chains=16,
                      samples_per_chain=64, scales=[108, 36, 12], fresh_QCR=True),
        objective=dict(Q='原始20微观subset+5粗尺度subset+10共享仿射重构',
                       C='-2全壳层buffer1经验MI+2逐系综方差floor+1均匀块颜色NLL',
                       R='5单步闭合+5微观起点all-horizon闭合，按目标k加权；R-only梯度',
                       scale_aggregation='sum(w_k * loss_k)/sum(w_k)', labels_used=False,
                       future_loss_weight=0., paired_sampler_retained=True),
        budget=dict(GPU_card_seconds=3600, GPU_peak_mib=8192, CPU_core_hours=24, output_gib=10,
                    projection='benchmark actual +1.5x measured fixed6000 work/setup/logging/checkpoints +120s per collection'),
        benchmark=dict(updates=40, profile='uniform', scientific_result=False, reused_initialization=False),
        source_code_sha256=execution.source_hashes(Path.cwd().resolve()),
        source_experiments_immutable=True, adaptive_hyperparameter_selection=False)
    save(root / 'plan.json', plan)
    try:
        receipt = validated_source(source); job['input_reuse'] = receipt
        prepare_inputs(root, source, receipt)
        plan['configuration_sha256'] = {name: file_sha(root / name) for name in
            ('config.json', 'config_uniform.json', 'config_late.json', 'benchmark_config.json')}
        save(root / 'plan.json', plan)
        save(root / 'job.json', job)
        run_stage(root, job, 'benchmark', trainer_command(root, root / 'benchmark',
                  root / 'benchmark_config.json', limit=600), gpu=True)
        estimate = execution.benchmark_projection(validate_arm(root, 'benchmark', 40))
        for index, name in enumerate(PROFILES):
            if not execution.budget_gate(root, job, estimate, arms=2-index, collections=2-index,
                                         before=f'train_{name}'):
                return
            limit = execution.training_time_limit(job, estimate, later_arms=1-index, collections=2-index)
            run_path = root / f'run_{name}'
            run_stage(root, job, f'train_{name}', trainer_command(root, run_path,
                      root / f'config_{name}.json', limit=limit), gpu=True)
            validate_arm(root, name, 6000)
            if not execution.budget_gate(root, job, estimate, arms=1-index, collections=2-index,
                                         before=f'collect_{name}'):
                return
            run_stage(root, job, f'collect_{name}', [sys.executable, '-m',
                'scripts.run_ising_closure_scale_ablation', 'collect', '--output', str(root),
                '--run', str(run_path), '--device', 'cuda:2'], gpu=True)
        if not execution.budget_gate(root, job, estimate, arms=0, collections=0, before='report'):
            return
        run_stage(root, job, 'report', [sys.executable, '-m', 'scripts.report_ising_closure_scale_ablation',
                                      '--root', str(root)])
        report_status = read(root / 'report/status.json')
        if report_status.get('status') != 'complete' or report_status.get('workflow_complete') is not True:
            raise RuntimeError('Report did not publish a complete workflow status')
        job.update(status='complete', workflow_complete=True, completed_at=now(), current=None)
    except execution.BudgetHold as error:
        job.update(status='budget_hold', workflow_complete=False, requires_user_action=True,
                   error=f'{type(error).__name__}: {error}', ended_at=now())
    except BaseException as error:
        job.update(status='failed', workflow_complete=False, requires_user_action=True,
                   error=f'{type(error).__name__}: {error}', ended_at=now())
        raise
    finally:
        job['public_files'] = public_files(root)
        save(root / 'job.json', job)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('run', 'collect'), nargs='?', default='run')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--device', default='cuda:2', choices=('cuda:2',))
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    if args.stage == 'collect':
        if args.run is None or args.launch: parser.error('collect requires --run and forbids --launch')
        collect(args.output, args.run, args.device); return
    if args.source is None: parser.error('run requires --source')
    if args.launch:
        root = args.output.resolve(); root.mkdir(parents=True, exist_ok=False)
        (root / 'logs').mkdir()
        command = [sys.executable, '-m', 'scripts.run_ising_closure_scale_ablation', '--output', str(root),
                   '--source', str(args.source.resolve())]
        save(root / 'launch.json', dict(status='launching', output=str(root), command=command, started_at=now()))
        with (root / 'logs/parent.log').open('x') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        receipt = dict(status='launched', pid=child.pid, output=str(root), command=command, started_at=now())
        save(root / 'launch.json', receipt); print(json.dumps(receipt), flush=True)
    else:
        run(args.output, args.source)


if __name__ == '__main__': main()
