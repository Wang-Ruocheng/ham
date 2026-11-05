"""One-shot, separately recorded 45-minute continuation of the frozen probe.

Only the execution budget adapter is new. The five original implementation
files, source model, physical seeds, sample sizes and scientific functions stay
byte-identical. Completed physical data and execution prechecks are reused.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

from scripts import run_ising_large_size_probe as base


PRIOR_ROOT = base.BASE / 'results/ising2d_b3_frozen_large_size_probe_seed20261018_vm225_v1'
ROOT = base.BASE / 'results/ising2d_b3_frozen_large_size_probe_seed20261018_vm225_v1_continue45m_v1'
EXPERIMENT = 'ising_b3_frozen_L1080_small_probe_v1_continue45m_v1'
CONTROLLER = 'scripts/continue_ising_large_size_probe.py'
ORIGINAL_FILES = tuple(base.NEW_FILES)
GPU_LIMIT = 2700.
PRIOR_CPU = 215.805779
PRIOR_GPU = 99.14136117696762
NEW_CPU_RESERVE = 30.
C000_SHA = '60cc2bd05524cdc94e252b68dbc3a12d16afcfc0269e5018a8a54db64184de3c'


def finite_nonnegative(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('Invalid nonnegative finite value: ' + name)
    return float(value)


def live_probe_processes(root, *, proc=Path('/proc')):
    """Inspect actual argv/cwd, not merely historical PIDs or a grep process."""
    root = Path(root).resolve()
    found = []
    modules = {'scripts.run_ising_large_size_probe', 'scripts.ising_large_size_probe_sampling',
               'scripts.ising_large_size_probe_evaluation', 'scripts.report_ising_large_size_probe',
               'scripts.continue_ising_large_size_probe'}
    for directory in proc.iterdir():
        if not directory.name.isdecimal() or int(directory.name) == os.getpid():
            continue
        try:
            argv = (directory / 'cmdline').read_bytes().decode(errors='replace').strip('\0').split('\0')
            if not argv or not any(value in modules for value in argv):
                continue
            cwd = (directory / 'cwd').resolve(strict=True)
            roots = [Path(argv[index + 1]).resolve() for index, value in enumerate(argv[:-1]) if value == '--root']
            if root in roots or cwd.is_relative_to(root):
                found.append(dict(pid=int(directory.name), argv=argv, cwd=str(cwd)))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return found


def _pin(path, root, pins):
    value = base.sha(base.ordinary(path, root))
    pins[str(path)] = value
    return value


def validate_prior(prior, deployed_root):
    """Read and fingerprint only this exact completed prefix and public source."""
    prior, deployed_root = Path(prior).resolve(), Path(deployed_root).resolve()
    if prior == deployed_root or prior.is_symlink():
        raise ValueError('Continuation must have a distinct regular root')
    processes = live_probe_processes(prior)
    if processes:
        raise RuntimeError('Prior probe still has actual running processes: ' + json.dumps(processes))
    pins = {}
    for name in ('job.json', 'plan.json', 'gpu_preflight.json', 'benchmark/receipt.json',
                 'benchmark/status.json', 'data/c000/status.json', 'data/c000/sidecar.json',
                 'data/c000/spins.npy'):
        _pin(prior / name, prior, pins)
    job, plan = base.read(prior / 'job.json'), base.read(prior / 'plan.json')
    gate = base.read(prior / 'gpu_preflight.json')
    expected_stages = [('sampling_preflight', 'complete', 0), ('sample_c000', 'complete', 0),
                       ('evaluation_preflight', 'budget_hold', 3)]
    actual_stages = [(row.get('name'), row.get('status'), row.get('returncode')) for row in job.get('stages', [])]
    if (job.get('status') != 'budget_hold' or job.get('workflow_complete') is not False
            or job.get('current') is not None or job.get('child_still_running') is not False
            or job.get('training_updates') != 0 or job.get('MI_computation') is not False
            or actual_stages != expected_stages
            or job.get('budget') != dict(CPU_core_hours=4, GPU_card_seconds=900,
                GPU_peak_mib=8192, output_bytes=2 * 1024**3)):
        raise ValueError('Prior exact stopped prefix/budget contract differs; no continuation')
    cpu = finite_nonnegative(job.get('CPU_seconds_completed'), 'prior CPU')
    gpu = finite_nonnegative(job.get('gpu_seconds_completed'), 'prior GPU')
    if not math.isclose(cpu, PRIOR_CPU, abs_tol=1e-6) or not math.isclose(gpu, PRIOR_GPU, abs_tol=1e-6):
        raise ValueError('Prior cumulative usage differs from authorized stopped receipt')
    if (job.get('gpu_seconds_current_stage') != 0 or (prior / 'data/c001').exists()
            or (prior / 'data/c002').exists() or (prior / 'collection').exists() or (prior / 'report').exists()):
        raise ValueError('Unexpected additional prior stage/output; no implicit retry')
    if set(plan.get('new_code_sha256', {})) != set(ORIGINAL_FILES):
        raise ValueError('Prior five-file implementation manifest mismatch')
    for relative in ORIGINAL_FILES:
        digest = _pin(prior / 'code' / relative, prior / 'code', pins)
        if digest != plan['new_code_sha256'][relative]:
            raise ValueError('Prior scientific implementation changed: ' + relative)
        if base.sha(base.ordinary(deployed_root / 'code' / relative, deployed_root / 'code')) != digest:
            raise ValueError('Continuation must copy original five files byte-for-byte: ' + relative)
    for relative, digest in plan['source']['source_json_sha256'].items():
        if _pin(prior / 'source_record' / relative, prior / 'source_record', pins) != digest:
            raise ValueError('Prior historical source metadata changed')
    base.assert_frozen(prior, plan)
    sampling = dict(K=[.42, .44, .46], size=1080, replicas=2, samples=64, equil=4096, gap=128,
        initial_conditions=['hot', 'cold'], extra_pilots=0, independent_production_chains=True,
        no_equilibrium_certificate=True, seed=20261018)
    if plan.get('sampling') != sampling:
        raise ValueError('Physical protocol differs from unchanged authorized probe')
    benchmark = base.read(prior / 'benchmark/receipt.json')
    benchmark_status = base.read(prior / 'benchmark/status.json')
    if (benchmark.get('status') != 'complete' or benchmark.get('workflow_complete') is not True
            or benchmark.get('budget_gate', {}).get('passed') is not True
            or benchmark_status.get('status') != 'complete'
            or benchmark_status.get('receipt_sha256') != pins[str(prior / 'benchmark/receipt.json')]):
        raise ValueError('Prior same-size CPU precheck is not completed and pinned')
    status, sidecar = base.read(prior / 'data/c000/status.json'), base.read(prior / 'data/c000/sidecar.json')
    if (status.get('status') != 'complete' or sidecar.get('status') != 'complete'
            or status.get('workflow_complete') is not True or sidecar.get('workflow_complete') is not True
            or status.get('sidecar_sha256') != pins[str(prior / 'data/c000/sidecar.json')]
            or status.get('spins_sha256') != C000_SHA or sidecar.get('spins_sha256') != C000_SHA
            or pins[str(prior / 'data/c000/spins.npy')] != C000_SHA
            or sidecar.get('spin_shape') != [2, 64, 1080, 1080]
            or sidecar.get('spin_dtype') != 'int8' or sidecar.get('K0') != .42
            or sidecar.get('target_delta6') != [0.] * 6
            or sidecar.get('sampling_budget') != dict(replicas=2, samples=64, equil=4096,
                gap=128, independent_pilot_chains=0)):
        raise ValueError('Completed first physical condition differs; no reuse')
    if (gate.get('status') != 'budget_hold' or gate.get('workflow_complete') is not False
            or gate.get('benchmark_only') is not True or gate.get('scientific_result') is not False
            or gate.get('physical_predictions_saved') is not False
            or gate.get('checkpoint_sha256') != base.CHECKPOINT_SHA
            or gate.get('physical_input_sha256') != C000_SHA
            or gate.get('gpu_budget_seconds') != 900 or gate.get('training_updates') != 0
            or gate.get('mutual_information_evaluations') != 0 or gate.get('optimizer_created') is not False
            or gate.get('all_parameters_frozen') is not True or gate.get('all_grad_none') is not True
            or gate.get('model_sha256_before') != gate.get('model_sha256_after')
            or gate.get('module_sha256_before') != gate.get('module_sha256_after')
            or gate.get('module_sha256_before') != plan['source']['module_sha256']
            or gate.get('training_depth_unchanged') != 2 or gate.get('evaluation_max_rg_steps') != 3):
        raise ValueError('Prior GPU precheck was not a budget-only frozen execution check')
    remaining = finite_nonnegative(gate.get('projected_remaining_gpu_seconds'), 'remaining GPU')
    finite_nonnegative(gate.get('elapsed_seconds'), 'original GPU precheck elapsed')
    for key in ('peak_gpu_allocated_bytes', 'peak_gpu_reserved_bytes'):
        if finite_nonnegative(gate.get(key), key) > 8 * 1024**3:
            raise ValueError('Prior precheck violated the unchanged8 GiB GPU peak cap')
    model_digest = gate.get('model_sha256_before')
    if not isinstance(model_digest, str) or len(model_digest) != 64:
        raise ValueError('Missing original complete model fingerprint')
    if finite_nonnegative(gate.get('measured_chain_draw_seconds'), 'measured chain-draw GPU') == 0:
        raise ValueError('Missing original positive execution timing')
    if gpu + remaining > GPU_LIMIT:
        raise base.BudgetHold('Unchanged conservative GPU projection still exceeds reauthorized45 minutes')
    return dict(prior_root=str(prior), prior_files_sha256=pins, prior_job=job,
        prior_plan=plan, prior_gpu_preflight=gate, CPU_seconds_completed=cpu,
        gpu_seconds_completed=gpu, no_actual_prior_processes=True,
        CPU_reserve_already_included=True, new_CPU_reserve_seconds=NEW_CPU_RESERVE)


def copy_new(original, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    with Path(original).open('rb') as source, target.open('xb') as destination:
        shutil.copyfileobj(source, destination, length=1024 * 1024)
    if base.sha(original) != base.sha(target):
        raise RuntimeError('Copied immutable prior artifact SHA mismatch')


def reuse_prefix(root, prior_state):
    """Copy immutable outputs only; no scientific kernel, model, or RNG call."""
    prior = Path(prior_state['prior_root'])
    for relative in ('benchmark/receipt.json', 'benchmark/status.json', 'data/c000/status.json',
                     'data/c000/sidecar.json', 'data/c000/spins.npy'):
        copy_new(prior / relative, root / relative)
    for name in prior_state['prior_plan']['source']['source_json_sha256']:
        copy_new(prior / 'source_record' / name, root / 'source_record' / name)
    for name in ('job.json', 'plan.json', 'gpu_preflight.json'):
        copy_new(prior / name, root / 'prior_record' / name)
    derived = copy.deepcopy(prior_state['prior_gpu_preflight'])
    original_elapsed = derived['elapsed_seconds']
    derived.update(status='complete', workflow_complete=True, requires_user_action=False,
        derived_budget_readmission=True, no_new_execution_precheck=True,
        original_receipt_sha256=base.sha(prior / 'gpu_preflight.json'),
        original_receipt_path=str(prior / 'gpu_preflight.json'),
        original_elapsed_seconds=original_elapsed, elapsed_seconds=0.,
        original_prior_gpu_seconds=derived['prior_gpu_seconds'],
        original_projected_total_gpu_seconds=derived['projected_total_gpu_seconds'],
        original_gpu_budget_seconds=derived['gpu_budget_seconds'],
        prior_gpu_seconds=prior_state['gpu_seconds_completed'], gpu_budget_seconds=GPU_LIMIT,
        projected_total_gpu_seconds=prior_state['gpu_seconds_completed'] + derived['projected_remaining_gpu_seconds'],
        authorization='用户明确增加上限并继续；GPU总上限45分钟，保留原样本量/科学计算/已完成数据',
        projection='Already charged prior parent GPU actual + unchanged original remaining conservative projection')
    base.save(root / 'gpu_preflight.json', derived)
    return derived


@contextlib.contextmanager
def budget_adapters(prior_state):
    """Reuse original execution controls with explicit cumulative resource caps."""
    original = {name: getattr(base, name) for name in
        ('GPU_LIMIT', 'PRIOR_CPU_RESERVE', 'NEW_FILES', 'output_bytes', 'assert_frozen')}
    prior = Path(prior_state['prior_root'])
    base.GPU_LIMIT = GPU_LIMIT
    base.PRIOR_CPU_RESERVE = prior_state['CPU_seconds_completed'] + NEW_CPU_RESERVE
    base.NEW_FILES = ORIGINAL_FILES + (CONTROLLER,)
    def cumulative_bytes(root):
        return original['output_bytes'](root) + original['output_bytes'](prior)
    def all_frozen(root, plan):
        original['assert_frozen'](root, plan)
        for path, digest in plan['continuation']['prior_files_sha256'].items():
            if base.sha(base.ordinary(path, prior)) != digest:
                raise RuntimeError('Prior stopped artifacts changed; no next continuation stage')
        for relative, digest in plan['continuation'].get('reused_files_sha256', {}).items():
            if base.sha(base.ordinary(root / relative, root)) != digest:
                raise RuntimeError('Reused immutable prefix changed; no next continuation stage')
        if live_probe_processes(prior):
            raise RuntimeError('A prior probe process reappeared; no duplicate continuation stage')
    base.output_bytes, base.assert_frozen = cumulative_bytes, all_frozen
    try:
        yield
    finally:
        for name, value in original.items():
            setattr(base, name, value)


def continuation_commands(root, prior_gpu):
    return [
        (f'sample_c{index:03d}', base.command(root, 'scripts.ising_large_size_probe_sampling',
            'sample', '--condition-index', index), False) for index in (1, 2)
    ] + [
        ('frozen_collection', base.command(root, 'scripts.continue_ising_large_size_probe',
            '--collect-worker', '--prior-gpu-seconds', prior_gpu), True),
        ('report', base.command(root, 'scripts.report_ising_large_size_probe'), False)]


def collect_worker(root, prior_gpu):
    """Budget-only CLI adapter around the byte-identical scientific collect()."""
    root = Path(root).resolve()
    plan = base.read(root / 'plan.json')
    # Parent publishes the owned child PID immediately after Popen. The budget
    # adapter imports no torch before this check, so allow that short publish race.
    for attempt in range(201):
        job = base.read(root / 'job.json')
        current = job.get('current') or {}
        if current.get('pid') == os.getpid() or attempt == 200:
            break
        time.sleep(.05)
    continuation = plan.get('continuation', {})
    if (root != ROOT or plan.get('experiment') != EXPERIMENT or continuation.get('prior_root') != str(PRIOR_ROOT)
            or plan.get('budget', {}).get('GPU_card_seconds') != GPU_LIMIT
            or job.get('budget', {}).get('GPU_card_seconds') != GPU_LIMIT
            or job.get('status') != 'running' or current.get('name') != 'frozen_collection'
            or current.get('pid') != os.getpid()
            or job.get('training_updates') != 0 or job.get('MI_computation') is not False):
        raise ValueError('Collection adapter requires the authorized new running continuation stage')
    prior_gpu = finite_nonnegative(prior_gpu, 'prior GPU usage passed to collection')
    if not math.isclose(prior_gpu, job['gpu_seconds_completed'], abs_tol=1e-6) or prior_gpu < PRIOR_GPU:
        raise ValueError('Collection adapter must preserve cumulative parent GPU accounting')
    state = dict(prior_root=str(PRIOR_ROOT), CPU_seconds_completed=continuation['prior_CPU_seconds_completed'])
    with budget_adapters(state):
        base.assert_frozen(root, plan)
        from scripts import ising_large_size_probe_evaluation as evaluator
        args = evaluator.build_parser().parse_args(['collect', '--root', str(root),
            '--checkpoint', str(base.CHECKPOINT), '--snapshot-root', str(base.FROZEN_CODE),
            '--baseline-root', str(base.BASELINE_ROOT), '--device', 'cuda:2',
            '--gpu-budget-seconds', str(GPU_LIMIT), '--prior-gpu-seconds', str(prior_gpu)])
        # Original main() enforces its historical900s CLI. This explicit new
        # authorization changes only the argument admission, never collect().
        return evaluator.collect(args)


def run(root):
    root = Path(root).resolve()
    if root != ROOT or socket.gethostname() != base.HOST:
        raise RuntimeError('Exact separately authorized continuation root/host required')
    if (not root.is_dir() or not (root / 'code').is_dir() or (root / 'job.json').exists()
            or any(path.name not in ('code', 'logs', 'launch.json') for path in root.iterdir())):
        raise FileExistsError('Fresh separate continuation root only; no restart/overwrite')
    if Path(__file__).resolve() != root / 'code' / CONTROLLER:
        raise RuntimeError('Execute the separately deployed continuation controller')
    os.chdir(root / 'code')
    os.environ.update(base.stage_environment(root))
    sys.path[:0] = [str(root / 'code'), str(base.FROZEN_CODE)]
    os.setpriority(os.PRIO_PROCESS, 0, 10)
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ['NUMBA_CACHE_DIR'] = str(root / 'runtime_cache/numba')
    os.environ['MPLCONFIGDIR'] = str(root / 'runtime_cache/matplotlib')
    (root / 'logs').mkdir(exist_ok=True)
    job = dict(schema_version=1, experiment=EXPERIMENT, root=str(root), status='validating_prior',
        workflow_complete=False, parent_pid=os.getpid(), code_snapshot=str(root / 'code'),
        frozen_code=str(base.FROZEN_CODE), started_at=base.now(), current=None, stages=[],
        checkpoint=str(base.CHECKPOINT), training_updates=0, MI_computation=False, thermal_nu=None,
        K=[.42, .44, .46], new_size=1080, baseline_size=108, replicas=2, samples=64, C_draws=5,
        CPU_seconds_completed=PRIOR_CPU + NEW_CPU_RESERVE, gpu_seconds_completed=PRIOR_GPU,
        gpu_seconds_current_stage=0., peak_own_gpu_mib=0, child_still_running=False,
        budget=dict(CPU_core_hours=4, GPU_card_seconds=GPU_LIMIT, GPU_peak_mib=8192,
            output_bytes=base.OUTPUT_LIMIT, output_scope='prior plus continuation roots, conservatively counting copied files'),
        prior_root=str(PRIOR_ROOT), error=None, requires_user_action=False,
        authorization='用户明确增加至45分钟GPU总上限并继续；CPU仍4核时，其他协议不变，不重跑已完成阶段')
    base.save(root / 'job.json', job)
    try:
        state = validate_prior(PRIOR_ROOT, root)
        with budget_adapters(state):
            reuse_prefix(root, state)
            continuation = dict(prior_root=str(PRIOR_ROOT), prior_files_sha256=state['prior_files_sha256'],
                prior_CPU_seconds_completed=state['CPU_seconds_completed'], prior_gpu_seconds_completed=state['gpu_seconds_completed'],
                original_CPU_reserve_already_included=True, new_CPU_reserve_seconds=NEW_CPU_RESERVE,
                copied_stage_names=['sampling_preflight', 'sample_c000'],
                derived_readmitted_stage='evaluation_preflight', no_old_stage_reexecution=True,
                reused_files_sha256={str(path.relative_to(root)): base.sha(path) for folder in
                    ('benchmark', 'data/c000', 'source_record', 'prior_record') for path in (root / folder).rglob('*') if path.is_file()})
            continuation['reused_files_sha256']['gpu_preflight.json'] = base.sha(root / 'gpu_preflight.json')
            plan = copy.deepcopy(state['prior_plan'])
            plan.update(experiment=EXPERIMENT, budget=job['budget'], continuation=continuation,
                new_code_sha256={name: base.sha(base.ordinary(root / 'code' / name, root / 'code')) for name in base.NEW_FILES})
            base.save(root / 'plan.json', plan)
            job['prior_completed_prefix'] = [dict(row, execution_origin='prior immutable run') for row in state['prior_job']['stages']]
            job['derived_budget_readmission'] = dict(status='complete', execution_performed=False,
                original_receipt_sha256=state['prior_files_sha256'][str(PRIOR_ROOT / 'gpu_preflight.json')])
            benchmark = base.read(root / 'benchmark/receipt.json')
            for name, cmd, gpu in continuation_commands(root, job['gpu_seconds_completed']):
                if not base.budget_ok(base.cpu_total(), job['gpu_seconds_completed'], base.output_bytes(root)):
                    raise base.BudgetHold('Cumulative prior plus continuation resource cap reached before next stage')
                if name.startswith('sample_c'):
                    index = int(name[-3:])
                    job['cpu_projection'] = base.cpu_projection_gate(benchmark, index, actual_cpu=base.cpu_total())
                    if not job['cpu_projection']['passed']:
                        raise base.BudgetHold('Cumulative CPU plus unchanged remaining stages exceeds4 core-hours')
                base.run_stage(root, job, plan, name, cmd, gpu=gpu)
            status = base.read(root / 'report/status.json')
            if status.get('status') != 'complete' or status.get('workflow_complete') is not True:
                raise RuntimeError('Continuation report is not complete')
            base.assert_frozen(root, plan)
            job.update(status='complete', workflow_complete=True, completed_at=base.now())
    except base.BudgetHold as error:
        job.update(status='budget_hold', requires_user_action=True, error=str(error), checked_at=base.now())
    except BaseException as error:
        job.update(status='failed', requires_user_action=True, error=f'{type(error).__name__}: {error}', checked_at=base.now())
    finally:
        job['public_files'] = base.public_files(root)
        base.save(root / 'job.json', job)
    return job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--collect-worker', action='store_true')
    parser.add_argument('--prior-gpu-seconds', type=float)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.launch and args.collect_worker:
        raise ValueError('Mutually exclusive execution entry points')
    if args.collect_worker:
        result = collect_worker(root, args.prior_gpu_seconds)
    elif args.launch:
        if root != ROOT or not root.is_dir() or (root / 'job.json').exists() or (root / 'launch.json').exists():
            raise FileExistsError('Exact new deployed continuation only; no duplicate launch')
        if socket.gethostname() != base.HOST:
            raise RuntimeError('Pinned continuation host differs')
        (root / 'logs').mkdir(exist_ok=False)
        cmd = [sys.executable, '-m', 'scripts.continue_ising_large_size_probe', '--root', str(root)]
        with (root / 'logs/parent.log').open('x') as log:
            process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                cwd=root / 'code', env=base.stage_environment(root))
        result = dict(status='launched', pid=process.pid, command=cmd, root=str(root), prior_root=str(PRIOR_ROOT),
            started_at=base.now(), GPU_card_seconds=GPU_LIMIT, training_updates=0, pinned_host=base.HOST)
        base.save(root / 'launch.json', result)
    else:
        result = run(root)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
