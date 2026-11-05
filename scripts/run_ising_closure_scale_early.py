"""Fresh [1.8, .2] closure-scale arm with immutable equal/late references.

No physics sampling, benchmark updates, checkpoint initialization, resume or
repair. Carry measured prior campaign usage once; reuse JSON timing only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys

from scripts import run_ising_closure_scale_ablation as pair

execution, read, save, file_sha, now = pair.execution, pair.read, pair.save, pair.file_sha, pair.now
EXPERIMENT = 'ising_closure_scale_early_only_b3_v1'
PRIOR_EXPERIMENT = 'ising_closure_scale_late_only_b3_v2'
WEIGHTS = [1.8, .2]
NEW_FILES = {'scripts/run_ising_closure_scale_early.py', 'scripts/report_ising_closure_scale_early.py'}
REFERENCE_FILES = ('manifest.json', 'summary.json', 'report/metrics.json', 'report/encodings.npz')
STAGES = ('train_early', 'collect_early', 'report')


def prior_processes(prior):
    """Match actual old commands and exact paths, never recycled PID numbers."""
    listing = subprocess.run(['ps', '-eo', 'pid=,ppid=,args='], check=True,
                             capture_output=True, text=True).stdout
    modules = {'scripts.run_ising_closure_scale_late', 'scripts.run_ising_closure_scale_ablation',
               'scripts.report_ising_closure_scale_late', 'experiments.rsmi_separate_qcr'}
    result = []
    for line in listing.splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) != 3:
            continue
        try:
            args = shlex.split(fields[2])
        except ValueError:
            continue
        if not any(args[i:i + 2] == ['-m', name] for name in modules for i in range(len(args) - 1)):
            continue
        targets = [args[i + 1] for i, arg in enumerate(args[:-1]) if arg in ('--output', '--root', '--run')]
        if any(Path(target).resolve() in (prior, prior / 'run_late') for target in targets):
            result.append(dict(pid=int(fields[0]), ppid=int(fields[1]), command=fields[2]))
    return result


def _reference_hashes(run_path, enclosing):
    result = {name: file_sha(pair.ordinary_file(run_path / name, enclosing)) for name in REFERENCE_FILES}
    metrics = read(run_path / 'report/metrics.json')
    if (metrics.get('status') != 'complete' or metrics.get('actual_steps') != 6000 or
            metrics.get('scales') != [108, 36, 12] or metrics.get('C_draws') != 5 or
            metrics.get('samples') != 64 or metrics.get('encodings_sha256') != result['report/encodings.npz'] or
            not metrics.get('model_sha256_before') or
            metrics.get('model_sha256_before') != metrics.get('model_sha256_after')):
        raise ValueError('Historical public collection must be complete6000, five-draw and immutable')
    return result


def validated_prior(prior, source, code):
    """Validate completed v2 without opening any prior model checkpoint."""
    prior, source, code = prior.resolve(), source.resolve(), code.resolve()
    job, plan = read(prior / 'job.json'), read(prior / 'plan.json')
    stages = job.get('stages', [])
    if (job.get('experiment') != PRIOR_EXPERIMENT or job.get('status') != 'complete' or
            job.get('workflow_complete') is not True or job.get('current') is not None or
            job.get('child_still_running', False) or job.get('error') is not None or
            job.get('requires_user_action') is not False or job.get('gpu_seconds_current_stage', 0.) != 0 or
            Path(job.get('source', '')).resolve() != source or
            job.get('device') != 'cuda:2' or job.get('gpu_uuid') != pair.GPU_UUID or
            [row.get('name') for row in stages] != ['train_late', 'collect_late', 'report'] or
            any(row.get('status') != 'complete' or row.get('returncode') != 0 for row in stages)):
        raise ValueError('Prior must be the fully completed late v2, all stages returncode0, no action/error')
    active = prior_processes(prior)
    if active:
        raise ValueError(f'Actual prior processes are still active: {active}')
    without_new = lambda hashes: {key: value for key, value in hashes.items() if key not in NEW_FILES}
    if without_new(execution.source_hashes(code)) != without_new(plan['source_code_sha256']):
        raise ValueError('Prior science SHA differs; only the two new early runner/report files are excluded')
    if set(plan['configuration_sha256']) != {'config.json', 'config_late.json'}:
        raise ValueError('Prior frozen configuration whitelist differs')
    for name, digest in plan['configuration_sha256'].items():
        if file_sha(pair.ordinary_file(prior / name, prior)) != digest:
            raise ValueError('Prior frozen configuration changed')
    late_config = pair.make_config(prior / 'data/ensembles.pt')
    late_config['loss']['closure_scale_weights'] = [.2, 1.8]
    if read(prior / 'config_late.json') != late_config or read(prior / 'config.json') != late_config:
        raise ValueError('Prior config must be the fixed6000 original recipe with [.2,1.8] only')
    manifest, summary = read(prior / 'run_late/manifest.json'), pair.validate_arm(prior, 'late', 6000)
    if (not pair.subset_equal(late_config, manifest.get('config')) or
            summary.get('config') != manifest.get('config')):
        raise ValueError('Prior manifest/summary/config identities differ')
    report_status = read(prior / 'report/status.json')
    if report_status.get('status') != 'complete' or report_status.get('workflow_complete') is not True:
        raise ValueError('Prior formal report is not complete')
    receipt = pair.validated_source(source)
    if receipt['input_sha256'] != job['input_reuse']['input_sha256']:
        raise ValueError('Completed source and prior immutable new-data/MI identities differ')
    for name, digest in receipt['input_sha256'].items():
        if file_sha(pair.ordinary_file(prior / name, prior)) != digest:
            raise ValueError('Prior immutable new-data/MI input changed')
    reference_control = _reference_hashes(prior / 'reference_control', prior)
    original_control = _reference_hashes(source / 'run_control', source)
    if reference_control != original_control:
        raise ValueError('Historical equal-weight public reference differs from completed source')
    reference_late = _reference_hashes(prior / 'run_late', prior)
    benchmark = pair.validate_arm(prior, 'benchmark', 40)
    estimate = execution.benchmark_projection(benchmark)
    initial = manifest['initial_module_sha256']
    if initial != read(prior / 'benchmark/manifest.json')['initial_module_sha256']:
        raise ValueError('Prior late initial Q/C/R does not match its timing reference')
    gpu, cpu = job.get('gpu_seconds_completed'), job.get('CPU_seconds_completed')
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in (gpu, cpu)):
        raise ValueError('Prior measured accounting must be finite and nonnegative')
    return dict(root=str(prior), source=str(source), input_reuse=receipt,
        gpu_seconds=gpu, CPU_seconds=cpu, reference_previous_gpu_seconds=gpu,
        reference_previous_CPU_seconds=cpu,
        accounting='Carry previous cumulative campaign usage exactly once; benchmark already included, never add again',
        previous_peak_own_gpu_mib=job.get('campaign_peak_own_gpu_mib', 0),
        job_sha256=file_sha(prior / 'job.json'), plan_sha256=file_sha(prior / 'plan.json'),
        benchmark_json_sha256={name: file_sha(prior / 'benchmark' / name)
                               for name in ('manifest.json', 'summary.json')},
        historical_reference_sha256={'reference_control': reference_control, 'reference_late': reference_late},
        prior_initial_module_sha256=initial, actual_prior_processes=active,
        benchmark_reference_only=True, benchmark_updates_in_this_run=0,
        reused_model_or_optimizer=False, source_checkpoints_opened=False), estimate


def fresh_initial_sha(config):
    """CPU initialization only; no forward pass, physics data or checkpoint."""
    import torch
    from recurrent_rg.config import RunConfig
    from recurrent_rg.checkpoints import build_model
    from experiments.rsmi_separate_qcr import future_module_sha
    torch.set_num_threads(1)
    cfg = RunConfig.from_dict(config)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg.training.seed)
        model = build_model(cfg.model, cfg.training.max_rg_steps)
        return dict(Q=future_module_sha(model.theory_encoder), C=future_module_sha(model.rg_kernel),
                    R=future_module_sha(model.theory_rg))


def prepare(root, prior, receipt):
    for folder in ('data', 'mi_cache', 'benchmark', 'reference_control/report', 'reference_late/report'):
        (root / folder).mkdir(parents=True)
    for name, digest in receipt['input_reuse']['input_sha256'].items():
        original = pair.ordinary_file(prior / name, prior)
        if file_sha(original) != digest:
            raise ValueError('Input bytes changed before hardlink')
        os.link(original, root / name)
    for name, digest in receipt['benchmark_json_sha256'].items():
        original = pair.ordinary_file(prior / 'benchmark' / name, prior)
        if file_sha(original) != digest:
            raise ValueError('Benchmark JSON changed before hardlink')
        os.link(original, root / 'benchmark' / name)
    config = read(prior / 'config_late.json')
    config['data']['cache'] = str(root / 'data/ensembles.pt')
    config['loss']['closure_scale_weights'] = WEIGHTS.copy()
    save(root / 'config_early.json', config)
    save(root / 'config.json', config)
    initial = fresh_initial_sha(config)
    if initial != receipt['prior_initial_module_sha256']:
        raise ValueError('Fresh Q/C/R initialization differs from prior late; no formal training started')
    for destination, folder in [('reference_control', 'reference_control'), ('reference_late', 'run_late')]:
        for name, digest in receipt['historical_reference_sha256'][destination].items():
            original = pair.ordinary_file(prior / folder / name, prior)
            if file_sha(original) != digest:
                raise ValueError('Historical public reference changed before hardlink')
            os.link(original, root / destination / name)
    save(root / 'preflight_reuse.json', dict(**receipt, fresh_initial_module_sha256=initial,
        CPU_initialization_only_no_forward_pass=True, physical_sampling_updates=0,
        fresh_formal_QCR=True, additional_prepare_calibration_benchmark_updates=0,
        config_changes=['loss.closure_scale_weights: [.2,1.8] -> [1.8,.2]',
                        'data.cache: byte-identical new-data hardlink absolute path'],
        note='Historical JSON timing/public evaluation and data-only caches only; no old model/optimizer loaded.'))


def validate_early(root, expected_steps=6000):
    run_path = root / 'run_early'
    manifest, summary = read(run_path / 'manifest.json'), read(run_path / 'summary.json')
    scale, initial = manifest.get('closure_scale_weighting', {}), manifest.get('initial_module_sha256', {})
    future = manifest.get('q_future_physics', {})
    if (summary.get('actual_epochs') != expected_steps or summary.get('requested_epochs') != expected_steps or
            any(summary.get(key, False) for key in ('stopped_by_wall_limit', 'stopped_by_training_limit',
                                                   'stopped_by_train_plateau'))):
        raise execution.BudgetHold(f'early未完成固定{expected_steps}次更新；不续训或重跑')
    normalized = scale.get('normalized_weights', [])
    if (scale.get('weights') != WEIGHTS or scale.get('target_levels') != [1, 2] or
            len(normalized) != 2 or any(not math.isclose(a, b, rel_tol=1e-14, abs_tol=1e-15)
                for a, b in zip(normalized, [weight / sum(WEIGHTS) for weight in WEIGHTS])) or
            scale.get('application') != ['one_step_commutator', 'microscopic_start_rollout'] or
            scale.get('labels_used') is not False or manifest.get('labels_used_in_training') is not False or
            summary.get('contract', {}).get('closure_scale_weighting') != scale or
            set(initial) != {'Q', 'C', 'R'} or
            summary.get('contract', {}).get('initial_module_sha256') != initial or
            initial != read(root / 'preflight_reuse.json')['fresh_initial_module_sha256'] or
            future.get('mode') != 'paired_control' or future.get('weight') != 0 or
            future.get('labels_used') is not False or manifest.get('auxiliary_future_parameters') != 0):
        raise RuntimeError('Early arm no-label/no-future/weight/initialization contract mismatch')
    config = read(root / 'config_early.json')
    if not pair.subset_equal(config, manifest.get('config')) or summary.get('config') != manifest.get('config'):
        raise RuntimeError('Early manifest/summary/config identities differ')
    caches = manifest.get('execution_optimizations', {}).get('mi_statistics_cache', {})
    if set(caches) != {'train', 'test'} or any(not row.get('hit') for row in caches.values()):
        raise RuntimeError('New-data MI cache was not reused exactly; no automatic rebuild accepted')
    return summary


def run_stage(root, job, name, command, *, gpu=False):
    """Keep all historical public references frozen as well as science/data."""
    receipt = read(root / 'preflight_reuse.json')
    for folder, hashes in receipt['historical_reference_sha256'].items():
        if folder not in ('reference_control', 'reference_late') or set(hashes) != set(REFERENCE_FILES):
            raise RuntimeError('Historical reference whitelist changed')
        for relative, digest in hashes.items():
            if file_sha(pair.ordinary_file(root / folder / relative, root)) != digest:
                raise RuntimeError('Historical reference fingerprint changed; no next stage started')
    try:
        pair.run_stage(root, job, name, command, gpu=gpu)
    finally:
        prior = job.get('prior_execution', {})
        job['gpu_seconds_new_execution'] = job['gpu_seconds_completed'] - prior.get('gpu_seconds', 0.)
        job['CPU_seconds_new_execution'] = job['CPU_seconds_completed'] - prior.get('CPU_seconds', 0.)
        save(root / 'job.json', job)


def collect(root, run_path, device):
    """Training-aligned five-draw collection, restricted to the new early arm."""
    import numpy as np
    import torch
    from recurrent_rg.config import RunConfig
    from recurrent_rg.checkpoints import load_run, isolated_rng
    from recurrent_rg.data.ensembles import load_ensembles
    from recurrent_rg.execution import configure_cuda_execution
    from recurrent_rg.provenance import json_config_equal
    from scripts.prepare_ising_future_physics import state_sha
    from scripts.ising_qcr_evaluation import collect_saved_encodings
    root, run_path = root.resolve(), run_path.resolve()
    if run_path != root / 'run_early' or device != 'cuda:2':
        raise ValueError('Collect only the newly completed early arm on pinned cuda:2')
    validate_early(root)
    out = run_path / 'report'; out.mkdir(exist_ok=False)
    loaded = load_run(run_path / 'checkpoint_last.pt')
    if (loaded.payload.get('epoch') != 6000
            or not json_config_equal(loaded.config, read(run_path / 'manifest.json')['config'])):
        raise ValueError('Early checkpoint/config is not the validated completed6000 arm')
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
    print(json.dumps(dict(status='collected', run=str(run_path), actual_steps=loaded.payload['epoch'])), flush=True)


def public_files(root):
    paths = [root / name for name in ('job.json', 'plan.json', 'preflight_reuse.json', 'dataset_receipt.json',
                                     'pairs.json', 'config.json', 'config_early.json', 'launch.json')]
    for name in ('benchmark', 'run_early', 'reference_control', 'reference_late'):
        paths.extend(root / name / item for item in ('manifest.json', 'summary.json'))
        paths.extend((root / name / 'report').glob('*'))
    paths.extend((root / 'report').glob('*'))
    return sorted(str(path.relative_to(root)) for path in paths
                  if path.is_file() and not path.is_symlink() and path.suffix.lower() in ('.json', '.npz', '.png'))


def run(root, source, prior):
    root, source, prior = root.resolve(), source.resolve(), prior.resolve()
    if root.exists():
        if set(path.name for path in root.iterdir()) != {'launch.json', 'logs'}:
            raise FileExistsError('Fresh output required; no resume/restart/overwrite')
    else:
        root.mkdir(parents=True); (root / 'logs').mkdir()
    if socket.gethostname() != 'thinklab-105-225':
        raise RuntimeError('Pinned host differs')
    os.setpriority(os.PRIO_PROCESS, 0, 10)
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    job = dict(schema_version=1, experiment=EXPERIMENT, status='validating_source', workflow_complete=False,
        started_at=now(), parent_pid=os.getpid(), source=str(source), output=str(root),
        code_snapshot=str(Path.cwd().resolve()), device='cuda:2', gpu_uuid=pair.GPU_UUID,
        GPU_card_seconds=3600, CPU_seconds_completed=0., gpu_seconds_completed=0.,
        gpu_seconds_current_stage=0., peak_own_gpu_mib=0, campaign_peak_own_gpu_mib=0,
        cpu_threads=1, nice=10, current=None, stages=[], error=None, requires_user_action=False,
        authorization='用户明确要求反向[1.8,.2]、其他不变，并与已完成等权/高k结果比较；不重跑旧臂',
        future_loss_weight=0., auxiliary_future_parameters=0, thermal_nu=None, public_files=[])
    save(root / 'job.json', job)
    try:
        receipt, estimate = validated_prior(prior, source, Path.cwd())
        job.update(prior_execution=receipt, input_reuse=receipt['input_reuse'],
            gpu_seconds_completed=receipt['gpu_seconds'], CPU_seconds_completed=receipt['CPU_seconds'],
            campaign_peak_own_gpu_mib=receipt['previous_peak_own_gpu_mib'],
            gpu_seconds_prior_execution=receipt['gpu_seconds'], CPU_seconds_prior_execution=receipt['CPU_seconds'],
            gpu_seconds_new_execution=0., CPU_seconds_new_execution=0.)
        prepare(root, prior, receipt)
        old_plan = read(prior / 'plan.json')
        plan = dict(schema_version=1, source=str(source), prior_reference=str(prior),
            training=dict(profiles={'early': WEIGHTS}, normalized_profiles={'early': [.9, .1]},
                          updates_per_arm=6000, seed=91, fresh_QCR=True,
                          scales=[108, 36, 12], batch_chains=16, samples_per_chain=64),
            objective=old_plan['objective'], budget=old_plan['budget'],
            budget_accounting='Original3600s campaign cap; prior cumulative usage carried once, including its benchmark',
            benchmark=dict(reused_JSON_timing=True, additional_updates=0, scientific_result=False),
            historical_reference=dict(equal='reference_control', late='reference_late', rerun=False,
                qualification='Historical completed runs, same initial seed/data/science; not newly paired executions'),
            source_code_sha256=execution.source_hashes(Path.cwd()),
            configuration_sha256={name: file_sha(root / name) for name in ('config.json', 'config_early.json')},
            source_experiments_immutable=True, adaptive_hyperparameter_selection=False)
        save(root / 'plan.json', plan); save(root / 'job.json', job)
        if not execution.budget_gate(root, job, estimate, arms=1, collections=1, before='train_early'):
            return
        limit = execution.training_time_limit(job, estimate, later_arms=0, collections=1)
        run_stage(root, job, 'train_early', pair.trainer_command(root, root / 'run_early',
            root / 'config_early.json', limit=limit), gpu=True)
        validate_early(root)
        if not execution.budget_gate(root, job, estimate, arms=0, collections=1, before='collect_early'):
            return
        run_stage(root, job, 'collect_early', [sys.executable, '-m', 'scripts.run_ising_closure_scale_early',
            'collect', '--output', str(root), '--run', str(root / 'run_early'), '--device', 'cuda:2'], gpu=True)
        if not execution.budget_gate(root, job, estimate, arms=0, collections=0, before='report'):
            return
        run_stage(root, job, 'report', [sys.executable, '-m', 'scripts.report_ising_closure_scale_early',
                                           '--root', str(root)])
        status = read(root / 'report/status.json')
        if (status.get('status') != 'complete' or status.get('workflow_complete') is not True or
                status.get('qualified_reference_comparison_available') is not True or
                status.get('scientific_result') is not True or
                [row.get('name') for row in job['stages']] != list(STAGES) or
                any(row.get('status') != 'complete' or row.get('returncode') != 0 for row in job['stages'])):
            raise RuntimeError('Formal report/comparison or fixed three-stage workflow incomplete')
        job.update(status='complete', workflow_complete=True, completed_at=now(), current=None)
    except execution.BudgetHold as error:
        job.update(status='budget_hold', requires_user_action=True, error=str(error), ended_at=now())
    except BaseException as error:
        job.update(status='failed', requires_user_action=True, error=f'{type(error).__name__}: {error}', ended_at=now())
        raise
    finally:
        job['public_files'] = public_files(root)
        save(root / 'job.json', job)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('run', 'collect'), nargs='?', default='run')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--prior-execution', type=Path)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--device', choices=('cuda:2',), default='cuda:2')
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    if args.stage == 'collect':
        if args.run is None or args.launch or args.source is not None or args.prior_execution is not None:
            parser.error('collect requires --run only and forbids launch/source/prior-execution')
        collect(args.output, args.run, args.device); return
    if args.source is None or args.prior_execution is None or args.run is not None:
        parser.error('run requires --source and --prior-execution, and forbids --run')
    if not args.launch:
        run(args.output, args.source, args.prior_execution); return
    if socket.gethostname() != 'thinklab-105-225':
        raise RuntimeError('Launch is pinned to thinklab-105-225')
    root = args.output.resolve(); root.mkdir(parents=True, exist_ok=False); (root / 'logs').mkdir()
    command = [sys.executable, '-m', 'scripts.run_ising_closure_scale_early', '--output', str(root),
        '--source', str(args.source.resolve()), '--prior-execution', str(args.prior_execution.resolve())]
    save(root / 'launch.json', dict(status='launching', command=command, started_at=now()))
    with (root / 'logs/parent.log').open('x') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    value = dict(status='launched', pid=child.pid, command=command, output=str(root), started_at=now())
    save(root / 'launch.json', value); print(json.dumps(value), flush=True)


if __name__ == '__main__': main()
