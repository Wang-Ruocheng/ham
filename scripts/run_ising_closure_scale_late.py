"""One fresh late-weight arm; reuse prior execution timing, never its weights."""
from __future__ import annotations
import argparse
import math
import os
from pathlib import Path
import socket
import subprocess
import sys

from scripts import run_ising_closure_scale_ablation as pair

execution, read, save, file_sha, now = pair.execution, pair.read, pair.save, pair.file_sha, pair.now
NEW_FILES = {'scripts/run_ising_closure_scale_late.py', 'scripts/report_ising_closure_scale_late.py'}


def alive(pid):
    if type(pid) is not int or pid <= 0: return False
    try: os.kill(pid, 0)
    except ProcessLookupError: return False
    except PermissionError: return True
    return True


def validated_prior(prior, source, code):
    job, plan = read(prior / 'job.json'), read(prior / 'plan.json')
    stages = job.get('stages', [])
    if (job.get('experiment') != pair.EXPERIMENT or job.get('status') != 'budget_hold' or
            job.get('workflow_complete') is not False or job.get('current') is not None or
            job.get('child_still_running') is not False or job.get('gpu_seconds_current_stage', 0.) != 0 or
            Path(job.get('source', '')).resolve() != source.resolve() or
            job.get('device') != 'cuda:2' or job.get('gpu_uuid') != pair.GPU_UUID or
            [row.get('name') for row in stages] != ['benchmark'] or
            any(row.get('status') != 'complete' or row.get('returncode') != 0 for row in stages) or
            any((prior / name).exists() for name in ('run_uniform', 'run_late')) or
            any(alive(pid) for pid in [job.get('parent_pid')] + [row.get('pid') for row in stages])):
        raise ValueError('Reuse requires only the completed40 benchmark, budget hold, no formal arm or live process')
    without_new = lambda hashes: {k: v for k, v in hashes.items() if k not in NEW_FILES}
    if without_new(execution.source_hashes(code)) != without_new(plan['source_code_sha256']):
        raise ValueError('Prior science SHA differs; only the two new late runner/report files are excluded')
    receipt = pair.validated_source(source)
    if receipt['input_sha256'] != job['input_reuse']['input_sha256']:
        raise ValueError('Prior immutable new-data input identities differ')
    for name, digest in receipt['input_sha256'].items():
        if file_sha(pair.ordinary_file(prior / name, prior)) != digest:
            raise ValueError('Prior immutable new-data/MI bytes changed')
    for name, digest in plan['configuration_sha256'].items():
        if file_sha(prior / name) != digest: raise ValueError('Prior frozen configuration changed')
    late = pair.make_config(prior / 'data/ensembles.pt'); late['loss']['closure_scale_weights'] = [.2, 1.8]
    if read(prior / 'config_late.json') != late:
        raise ValueError('Prior late config is not the fixed6000 fresh original recipe')
    summary = pair.validate_arm(prior, 'benchmark', 40)
    estimate = execution.benchmark_projection(summary)
    gpu, cpu = job.get('gpu_seconds_completed'), job.get('CPU_seconds_completed')
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in (gpu, cpu)):
        raise ValueError('Prior measured CPU/GPU accounting must be finite and nonnegative')
    return dict(root=str(prior), gpu_seconds=gpu, CPU_seconds=cpu,
        previous_peak_own_gpu_mib=job.get('campaign_peak_own_gpu_mib', job.get('peak_own_gpu_mib', 0)),
        input_reuse=receipt, job_sha256=file_sha(prior / 'job.json'),
        benchmark_json_sha256={n: file_sha(prior / 'benchmark' / n) for n in ('manifest.json', 'summary.json')},
        benchmark_reference_only=True, reused_model_or_optimizer=False), estimate


def prepare(root, prior, source, receipt):
    for folder in ('data', 'mi_cache', 'benchmark', 'reference_control/report'):
        (root / folder).mkdir(parents=True)
    for name, digest in receipt['input_reuse']['input_sha256'].items():
        if file_sha(prior / name) != digest: raise ValueError('Input bytes changed before hardlink')
        os.link(prior / name, root / name)
    for name, digest in receipt['benchmark_json_sha256'].items():
        if file_sha(prior / 'benchmark' / name) != digest: raise ValueError('Benchmark JSON changed')
        os.link(prior / 'benchmark' / name, root / 'benchmark' / name)
    config = read(prior / 'config_late.json'); config['data']['cache'] = str(root / 'data/ensembles.pt')
    save(root / 'config_late.json', config); save(root / 'config.json', config)
    # CPU initialization only: no forward pass, data loading, checkpoint or GPU.
    import torch
    from recurrent_rg.config import RunConfig
    from recurrent_rg.checkpoints import build_model
    from experiments.rsmi_separate_qcr import future_module_sha
    torch.set_num_threads(1)
    cfg = RunConfig.from_dict(config)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg.training.seed)
        model = build_model(cfg.model, cfg.training.max_rg_steps)
        initial = dict(Q=future_module_sha(model.theory_encoder), C=future_module_sha(model.rg_kernel),
                       R=future_module_sha(model.theory_rg))
    if initial != read(root / 'benchmark/manifest.json')['initial_module_sha256']:
        raise ValueError('Fresh Q/C/R initialization differs from prior benchmark; no formal training started')
    reference = {}
    for name in ('manifest.json', 'summary.json', 'report/metrics.json', 'report/encodings.npz'):
        path = pair.ordinary_file(source / 'run_control' / name, source)
        reference[name] = file_sha(path); os.link(path, root / 'reference_control' / name)
    save(root / 'preflight_reuse.json', dict(**receipt, historical_reference_sha256=reference,
        fresh_initial_module_sha256=initial, CPU_initialization_only_no_forward_pass=True,
        benchmark_updates_in_this_run=0, physical_sampling_updates=0,
        source_checkpoints_opened=False, fresh_formal_QCR=True,
        note='Only prior JSON timing and immutable data/MI inputs reused. Benchmark JSON is reference, not a new execution stage.'))


def run(root, source, prior):
    root, source, prior = root.resolve(), source.resolve(), prior.resolve()
    if root.exists():
        if set(p.name for p in root.iterdir()) != {'launch.json', 'logs'}:
            raise FileExistsError('Fresh output required; no resume/restart/overwrite')
    else: root.mkdir(parents=True); (root / 'logs').mkdir()
    if socket.gethostname() != 'thinklab-105-225': raise RuntimeError('Pinned host differs')
    os.setpriority(os.PRIO_PROCESS, 0, 10)
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    receipt, estimate = validated_prior(prior, source, Path.cwd())
    prepare(root, prior, source, receipt)
    job = dict(schema_version=1, experiment='ising_closure_scale_late_only_b3_v2', status='running',
        workflow_complete=False, started_at=now(), parent_pid=os.getpid(), source=str(source), output=str(root),
        code_snapshot=str(Path.cwd().resolve()), prior_execution=receipt, input_reuse=receipt['input_reuse'],
        device='cuda:2', gpu_uuid=pair.GPU_UUID, GPU_card_seconds=3600, cpu_threads=1, nice=10,
        gpu_seconds_completed=receipt['gpu_seconds'], gpu_seconds_current_stage=0.,
        CPU_seconds_completed=receipt['CPU_seconds'], peak_own_gpu_mib=0,
        campaign_peak_own_gpu_mib=receipt['previous_peak_own_gpu_mib'], current=None, stages=[],
        error=None, requires_user_action=False, thermal_nu=None, public_files=[],
        authorization='用户明确要求直接进行改权重分布的训练：仅fresh late，不重跑等权对照',
        future_loss_weight=0., auxiliary_future_parameters=0)
    old_plan = read(prior / 'plan.json')
    plan = dict(schema_version=1, source=str(source), prior_execution=str(prior),
        training=dict(profiles={'late': [.2, 1.8]}, updates_per_arm=6000, seed=91, fresh_QCR=True,
                      scales=[108, 36, 12], batch_chains=16, samples_per_chain=64),
        objective=old_plan['objective'], budget=old_plan['budget'],
        benchmark=dict(reused_JSON_timing=True, additional_updates=0, scientific_result=False),
        historical_reference=dict(path='reference_control', original_control=True, rerun=False,
                                  qualification='Historical completed v4 control, not a new paired execution'),
        source_code_sha256=execution.source_hashes(Path.cwd()),
        configuration_sha256={n: file_sha(root / n) for n in ('config.json', 'config_late.json')})
    save(root / 'job.json', job); save(root / 'plan.json', plan)
    try:
        if not execution.budget_gate(root, job, estimate, arms=1, collections=1, before='train_late'): return
        limit = execution.training_time_limit(job, estimate, later_arms=0, collections=1)
        pair.run_stage(root, job, 'train_late', pair.trainer_command(root, root / 'run_late',
            root / 'config_late.json', limit=limit), gpu=True)
        pair.validate_arm(root, 'late', 6000)
        if not execution.budget_gate(root, job, estimate, arms=0, collections=1, before='collect_late'): return
        pair.run_stage(root, job, 'collect_late', [sys.executable, '-m', 'scripts.run_ising_closure_scale_ablation',
            'collect', '--output', str(root), '--run', str(root / 'run_late'), '--device', 'cuda:2'], gpu=True)
        if not execution.budget_gate(root, job, estimate, arms=0, collections=0, before='report'): return
        pair.run_stage(root, job, 'report', [sys.executable, '-m', 'scripts.report_ising_closure_scale_late',
                                           '--root', str(root)])
        status = read(root / 'report/status.json')
        if status.get('status') != 'complete' or status.get('workflow_complete') is not True:
            raise RuntimeError('Report workflow incomplete')
        job.update(status='complete', workflow_complete=True, completed_at=now(), current=None)
    except execution.BudgetHold as error:
        job.update(status='budget_hold', requires_user_action=True, error=str(error), ended_at=now())
    except BaseException as error:
        job.update(status='failed', requires_user_action=True, error=f'{type(error).__name__}: {error}', ended_at=now())
        raise
    finally:
        job['public_files'] = sorted(set(pair.public_files(root) + ['preflight_reuse.json'] +
            [str(p.relative_to(root)) for p in (root / 'reference_control').rglob('*')
             if p.is_file() and p.suffix in ('.json', '.npz')]))
        save(root / 'job.json', job)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--prior-execution', required=True, type=Path)
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    if not args.launch: run(args.output, args.source, args.prior_execution); return
    root = args.output.resolve(); root.mkdir(parents=True, exist_ok=False); (root / 'logs').mkdir()
    command = [sys.executable, '-m', 'scripts.run_ising_closure_scale_late', '--output', str(root),
        '--source', str(args.source.resolve()), '--prior-execution', str(args.prior_execution.resolve())]
    save(root / 'launch.json', dict(status='launching', command=command, started_at=now()))
    with (root / 'logs/parent.log').open('x') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    value = dict(status='launched', pid=child.pid, command=command, output=str(root), started_at=now())
    save(root / 'launch.json', value); print(__import__('json').dumps(value), flush=True)


if __name__ == '__main__': main()
