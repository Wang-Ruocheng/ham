"""Separate, explicitly authorized nonequilibrium-data QCR diagnostic.

The old scientific-stop workflow is immutable. Only finite-chain quality
criteria are nonblocking here; target-law tracking, data/code fingerprints,
label boundaries, execution errors and resource limits remain hard gates.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from scripts import run_ising_j1j2 as base

MODULE = 'scripts.run_ising_j1j2_unthermalized'
DIAGNOSTIC_WARNING = '未通过平衡态采样验证的数据诊断；不得据此认证平衡态相变或临界指数'


def prepare(root, reference, *, cpu_hours=12., gpu_hours=8., directory_gib=10.):
    from experiments.prepare_ising_j1j2 import make_config, protocol_contract
    from scripts.ising_qcr_evaluation import encoding_contract
    for value in (cpu_hours, gpu_hours, directory_gib):
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError('Positive finite resource budgets required')
    if socket.gethostname() != base.HOST or 'CUDA_VISIBLE_DEVICES' in os.environ:
        raise ValueError('Require declared host and unmasked physical ordinals')
    root, reference = Path(root).resolve(), Path(reference).resolve()
    code = Path(__file__).resolve().parents[1]
    if root.exists() or root.is_symlink() or root == reference:
        raise FileExistsError('A diagnostic must have a new, exclusive output root')
    previous = base.read(reference / 'plan.json')
    old_job = base.read(reference / 'job.json')
    if old_job.get('status') != 'scientific_stop' or old_job.get('training_started') is not False:
        raise ValueError('Reference must be the stopped, untrained uniform-K workflow')
    sources = base.source_hashes(code)
    if any(sources.get(name) != value for name, value in previous['source_sha256'].items()):
        raise RuntimeError('The original QCR and sampler implementations must remain unchanged')
    benchmark = reference / 'data/benchmark/receipt.json'
    measurement = base.read(benchmark)
    speed = measurement['measured_worst_cpu_seconds_per_visited_site']
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError('Require a finite positive existing throughput measurement')
    cache = root / 'data/training.pt'
    configs = {'dg5_seed91': make_config(cache, seed=91)}
    profile = make_config(cache, seed=191, updates=20)
    protocol = protocol_contract()
    protocol.update(train_rhos=[0., .35, .7, 1.], heldout_rhos=[],
        train_groups=[0, 1], same_rho_test_groups=[8, 9], training_seeds=[91],
        scales=[108, 36, 12], encoding_protocol=encoding_contract(128),
        diagnostic_only=True, equilibrium_certified=False, thermal_nu=None,
        no_heldout_coupling_claim=True, warning=DIAGNOSTIC_WARNING)
    policy = dict(previous['resource_policy'])
    policy.update(cpu_seconds=cpu_hours * 3600, gpu_seconds=gpu_hours * 3600,
        directory_bytes=int(directory_gib * 1024**3), nice=10,
        sampling_projection_factor=2., MI_cpu_reserve_seconds=7200.,
        profile_cpu_reserve_seconds=1200.,
        max_training_seconds=10800., report_seconds_per_arm=1800.,
        budget_check_granularity='Stage boundaries; no in-flight process is killed; no retry',
        sharing_authorization='Existing user authorization: GPU3 low-utilization sharing only')
    physical = dict(size=108, rhos=[0., .35, .7, 1.],
        train_rhos=[0., .35, .7, 1.], heldout_rhos=[], **base.coupling_grid(),
        groups=4, train_groups=2, group_ids=[0, 1, 8, 9],
        train_group_ids=[0, 1], test_group_ids=[8, 9],
        samples=128, burn_in=4096, spacing=32, master_seed=20261026,
        initials=['hot', 'ferro', 'stripe_x'],
        initial_policy='groups0/8 hot; groups1/9 ferro if rho<.5 else stripe_x',
        convention='H=-sum_NN ss + rho sum_diagonal ss; J1=kB=1',
        symmetry_operations_are_not_independent_samples=True)
    plan = dict(schema_version=5, experiment='J1J2_unthermalized_small_diagnostic',
        host=base.HOST, output=str(root), code_root=str(code), prepared_at=base.now(),
        source_sha256=sources, configs=configs, profile_config=profile,
        physical=physical, pilot=dict(previous['pilot']), protocol=protocol,
        resource_policy=policy,
        diagnostic=dict(diagnostic_only=True, quality_override_authorized=True,
            equilibrium_certified=False, warning=DIAGNOSTIC_WARNING,
            user_authorization='2026-10-26 用户：继续；独立小规模重新采样并将热化门控仅作警告后训练',
            quality_failures_are_nonblocking=True,
            hard_failures=['execution', 'energy_tracking', 'invalid_spins',
                           'fingerprint', 'label_boundary', 'resource_budget'],
            old_pilot_configurations_reused=False, new_sampling=True,
            source_initialization_is_not_thermalization=True),
        reference=dict(root=str(reference), plan_sha256=base.sha(reference / 'plan.json'),
            job_sha256=base.sha(reference / 'job.json'),
            benchmark_path=str(benchmark), benchmark_sha256=base.sha(benchmark),
            worst_cpu_seconds_per_visited_site=speed,
            historical_roots_never_modified=True))
    from scripts.ising_j1j2_unthermalized_data import validate_plan
    validate_plan(plan)
    root.mkdir(parents=True, exist_ok=False)
    for name in ('logs', 'configs', 'data'):
        (root / name).mkdir()
    for name, cfg in [*configs.items(), ('profile', profile)]:
        base.save(root / f'configs/{name}.json', cfg, exclusive=True)
    base.save(root / 'plan.json', plan, exclusive=True)
    base.save(root / 'job.json', dict(status='prepared', workflow_complete=False,
        current=None, stages=[], error=None, requires_user_action=False,
        plan_sha256=base.sha(root / 'plan.json'), gpu_elapsed_seconds=0., training={},
        training_started=False, diagnostic_only=True, equilibrium_certified=False,
        warning=DIAGNOSTIC_WARNING), exclusive=True)
    return plan


def verify(root):
    root = Path(root).resolve()
    plan, job = base.read(root / 'plan.json'), base.read(root / 'job.json')
    if (plan.get('experiment') != 'J1J2_unthermalized_small_diagnostic'
            or socket.gethostname() != base.HOST or plan['host'] != base.HOST
            or plan['output'] != str(root) or 'CUDA_VISIBLE_DEVICES' in os.environ
            or job['plan_sha256'] != base.sha(root / 'plan.json')):
        raise RuntimeError('Host/output/plan/CUDA-mask mismatch')
    from scripts.ising_j1j2_unthermalized_data import validate_plan
    validate_plan(plan)
    if base.source_hashes(Path(plan['code_root'])) != plan['source_sha256']:
        raise RuntimeError('Frozen executable source changed')
    for name, expected in [*plan['configs'].items(), ('profile', plan['profile_config'])]:
        if base.read(root / f'configs/{name}.json') != expected:
            raise RuntimeError('Frozen model/loss/training configuration changed')
    reference = plan['reference']
    for path, expected in ((Path(reference['root']) / 'plan.json', reference['plan_sha256']),
                           (Path(reference['root']) / 'job.json', reference['job_sha256']),
                           (reference['benchmark_path'], reference['benchmark_sha256'])):
        if base.sha(path) != expected:
            raise RuntimeError('Read-only stopped-workflow reference changed')
    return plan, job


def sampling_projection(plan):
    p, policy = plan['physical'], plan['resource_policy']
    visits = (len(p['rhos']) * len(p['group_ids']) * len(p['temperatures'])
              * p['size']**2 * (p['burn_in'] + p['samples'] * p['spacing']))
    sampling = (visits * plan['reference']['worst_cpu_seconds_per_visited_site']
                * policy['sampling_projection_factor'])
    reserve = (policy['MI_cpu_reserve_seconds'] + policy.get('profile_cpu_reserve_seconds', 0.)
               + 2 * policy['max_training_seconds'] + policy['report_seconds_per_arm'])
    return dict(status='complete', diagnostic_only=True, measured_reference=True,
        projected_sampling_cpu_seconds=sampling,
        projected_total_cpu_seconds=sampling + reserve,
        fixed_cpu_seconds=policy['cpu_seconds'],
        budget_passed=sampling + reserve <= policy['cpu_seconds'],
        projection_is_not_a_wall_time_guarantee=True)


def stage_list(root, plan):
    def data(mode, *extra):
        return [sys.executable, '-u', '-m', MODULE, '--data-child', mode,
                '--output', str(root), *extra]
    stages = [(f'sample_r{ri:02d}_g{group:02d}',
               data('sample', '--rho-index', str(ri), '--group', str(group)), False)
              for ri in range(len(plan['physical']['rhos']))
              for group in plan['physical']['group_ids']]
    stages += [('label_free_cache_and_quality_warning', data('assemble'), False),
        ('MI_preprocessing', [sys.executable, '-u', '-m', 'scripts.prepare_qcr_mi_cache',
            '--config', str(root / 'configs/dg5_seed91.json'),
            '--output', str(root / 'data/mi_preprocessing.json'),
            '--cache-dir', str(root / 'data/mi_cache')], False)]
    for arm in ('profile', 'dg5_seed91'):
        stages.append(('training_profile' if arm == 'profile' else f'{arm}_training',
            [sys.executable, '-u', '-m', MODULE, '--train-child', arm,
             '--output', str(root)], True))
    stages.append(('dg5_seed91_report', [sys.executable, '-u', '-m',
        'scripts.report_ising_j1j2_unthermalized',
        '--run', str(root / 'dg5_seed91/run'),
        '--data-metadata', str(root / 'data/metadata.json'),
        '--quality-warning', str(root / 'data/quality_warning.json'),
        '--output', str(root / 'dg5_seed91/report'),
        '--max-seconds', str(plan['resource_policy']['report_seconds_per_arm'])], False))
    return stages


@contextmanager
def diagnostic_training_context():
    """Reuse the unchanged training wrapper with this new plan verifier only."""
    original = base.verify
    base.verify = verify
    try:
        yield
    finally:
        base.verify = original


def train_child(root, arm):
    with diagnostic_training_context():
        return base.train_child(root, arm)


def execute(root):
    root = Path(root).resolve()
    plan, job = verify(root)
    if job['status'] != 'prepared' or job['stages']:
        raise RuntimeError('Only fresh execution; no retry or resume')
    base.save(root / 'execution.json', dict(at=base.now(), parent_pid=os.getpid(),
        plan_sha256=base.sha(root / 'plan.json'), diagnostic_only=True,
        equilibrium_certified=False, quality_override_authorized=True,
        exclusive_execution_lock=True, retry=False), exclusive=True)
    os.nice(plan['resource_policy']['nice'])
    job.update(parent_pid=os.getpid(), actual_nice=os.getpriority(os.PRIO_PROCESS, 0),
               started_at=base.now(), cpu_seconds=0.)
    stages = stage_list(root, plan)
    try:
        decision = sampling_projection(plan)
        base.save(root / 'sampling_budget_receipt.json', decision, exclusive=True)
        if not decision['budget_passed']:
            base.hold(job, 'Small-diagnostic sampling projection exceeds budget')
            return
        for index, (name, command, gpu) in enumerate(stages):
            verify(root)
            policy = plan['resource_policy']
            if (base.cpu_seconds() >= policy['cpu_seconds']
                    or base.directory_bytes(root) >= policy['directory_bytes']
                    or job['gpu_elapsed_seconds'] >= policy['gpu_seconds']):
                base.hold(job, 'Diagnostic resource envelope reached; no automatic continuation')
                break
            if (root / 'profile_receipt.json').is_file():
                projection = base.read(root / 'profile_receipt.json')
                pending = sum(stage[2] for stage in stages[index:])
                if job['gpu_elapsed_seconds'] + pending * projection['projected_gpu_seconds_per_arm'] > policy['gpu_seconds']:
                    base.hold(job, 'Actual GPU time plus pending projection exceeds budget')
                    break
            base.admission(root, plan, job, name, gpu=gpu)
            if name == 'dg5_seed91_training':
                job['training_started'] = True
            started, cpu_before = time.perf_counter(), base.cpu_seconds()
            with (root / f'logs/{name}.log').open('x') as stream:
                child = subprocess.Popen(command, cwd=plan['code_root'],
                    stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                    env=dict(os.environ, OMP_NUM_THREADS=str(2 if gpu else 1),
                        MKL_NUM_THREADS=str(2 if gpu else 1), OPENBLAS_NUM_THREADS='1',
                        NUMBA_NUM_THREADS='1', PYTHONUNBUFFERED='1', CUDA_DEVICE_ORDER='PCI_BUS_ID'))
                row = dict(name=name, command=command, pid=child.pid,
                    started_at=base.now(), status='running', returncode=None, gpu_stage=gpu)
                job['stages'].append(row)
                job.update(status='running', current=name)
                base.save(root / 'job.json', job)
                returncode = child.wait()
            elapsed = time.perf_counter() - started
            row.update(returncode=returncode, status='complete' if returncode == 0 else 'failed',
                ended_at=base.now(), wall_seconds=elapsed,
                cpu_seconds=base.cpu_seconds() - cpu_before)
            job['cpu_seconds'] = base.cpu_seconds()
            if gpu:
                job['gpu_elapsed_seconds'] += elapsed
            base.save(root / 'job.json', job)
            if returncode and name.endswith('_report'):
                status_path = root / 'dg5_seed91/report/status.json'
                if status_path.is_file() and base.read(status_path).get('status') == 'budget_hold':
                    row['status'] = 'analysis_budget_hold'
                    base.hold(job, 'Bounded diagnostic report reached its budget; model preserved')
                    break
            if returncode:
                raise RuntimeError(f'{name} returned {returncode}; preserved output, no retry')
            if name == 'label_free_cache_and_quality_warning':
                quality = base.read(root / 'data/quality_warning.json')
                job['sampling_quality'] = dict(path='data/quality_warning.json',
                    sha256=base.sha(root / 'data/quality_warning.json'),
                    scientific_gate_passed=quality.get('scientific_gate_passed'),
                    quality_failures_nonblocking=True, equilibrium_certified=False)
            elif gpu:
                arm = 'profile' if name == 'training_profile' else 'dg5_seed91'
                result = base.validate_training(root, plan, arm)
                job['training'][arm] = result
                if result['status'] != 'complete':
                    base.hold(job, result['reason'])
                    break
                if arm == 'profile':
                    projection = base.profile_projection(base.read(root / 'profile_run/summary.json'),
                                                        job['gpu_elapsed_seconds'], plan)
                    base.save(root / 'profile_receipt.json', projection, exclusive=True)
                    if not projection['budget_passed'] or not projection['memory_passed']:
                        base.hold(job, 'Execution profile failed budget/memory gate; no formal training')
                        break
            elif name.endswith('_report'):
                base.validate_report(root / 'dg5_seed91/report')
                status = base.read(root / 'dg5_seed91/report/status.json')
                if not status.get('diagnostic_only') or not status.get('equilibrium_not_validated'):
                    raise RuntimeError('Report lost its nonequilibrium qualification')
        else:
            job.update(status='complete', workflow_complete=True, current=None,
                       ended_at=base.now(), thermal_nu=None, equilibrium_certified=False)
    except BaseException as error:
        job.update(status='failed', workflow_complete=False, requires_user_action=True,
                   error=f'{type(error).__name__}: {error}', ended_at=base.now())
        raise
    finally:
        job['public_files'] = base.public_files(root)
        base.save(root / 'job.json', job)


def launch(root):
    root = Path(root).resolve()
    plan, job = verify(root)
    if job['status'] != 'prepared' or job['stages']:
        raise RuntimeError('No duplicate launch')
    with (root / 'launch.json').open('x') as receipt, (root / 'logs/parent.log').open('x') as stream:
        process = subprocess.Popen([sys.executable, '-u', '-m', MODULE,
            '--execute', '--output', str(root)], cwd=plan['code_root'],
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True, env=dict(os.environ, PYTHONUNBUFFERED='1',
                                            CUDA_DEVICE_ORDER='PCI_BUS_ID'))
        result = dict(status='launched', at=base.now(), parent_pid=process.pid,
            host=plan['host'], root=str(root), plan_sha256=base.sha(root / 'plan.json'),
            diagnostic_only=True, quality_override_authorized=True,
            resumed=False, automatic_retry=False)
        receipt.write(__import__('json').dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare', 'launch', 'execute'):
        mode.add_argument('--' + name, action='store_true')
    mode.add_argument('--data-child', choices=('sample', 'assemble'))
    mode.add_argument('--train-child')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--rho-index', type=int)
    parser.add_argument('--group', type=int)
    args = parser.parse_args(argv)
    if args.prepare:
        if args.reference is None:
            parser.error('--prepare requires --reference')
        result = prepare(args.output, args.reference)
    elif args.launch:
        result = launch(args.output)
    elif args.execute:
        return execute(args.output)
    elif args.train_child is not None:
        return train_child(args.output, args.train_child)
    else:
        from scripts import ising_j1j2_unthermalized_data as data
        plan, _ = verify(args.output)
        if args.data_child == 'sample':
            if args.rho_index is None or args.group is None:
                parser.error('sample requires --rho-index and --group')
            result = data.sample_group(plan, args.output, args.rho_index, args.group)
        else:
            result = data.assemble(plan, args.output)
    print(__import__('json').dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return result


if __name__ == '__main__':
    main()
