"""明确授权的 B0 五 seed 等待队列交接；冻结训练代码及完成结果不变。

只允许 seed91 和报告完整结束、seed92 尚未启动时，交接无子进程的
等待父进程。新协调器只运行尚未执行的后缀；不恢复模型、不重跑预检。
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

ROOT = Path('/mnt/nas-new/home/yange/ham/results/ising2d_B0_multiseed_dg5_b3_seed91to95_vm225_20261023')
GPU0_UUID = 'GPU-5bba87f1-7d32-4dde-fe28-caba4468ab42'
GPU1_UUID = 'GPU-18fac97c-11ee-4624-d298-9074e8d7028c'
PREFIX = ['training_profile', 'seed91_training', 'seed91_report']
AUTHORIZATION = '好。以及能不能去掉显存的限制（同意将尚未启动的四个 seed 改用 GPU0）'


def read(path):
    return json.loads(Path(path).read_text())


def remaining_stages(job):
    stages = job.get('stages', [])
    if (job.get('status') != 'waiting_resources' or job.get('current') != 'seed92_training'
            or job.get('workflow_complete') is not False or job.get('error') is not None
            or job.get('requires_user_action') is not False
            or [row.get('name') for row in stages] != PREFIX
            or any(row.get('status') != 'complete' or row.get('returncode') != 0 for row in stages)
            or set(job.get('training', {})) != {'91'}):
        raise RuntimeError('Require exactly the successful prefix and an unstarted seed92; no retry/resume')
    return [name for seed in range(92, 96) for name in (f'seed{seed}_training', f'seed{seed}_report')] + ['comparison']


def runtime_plan(plan):
    if plan.get('device') != 'cuda:1' or plan.get('gpu_uuid') != GPU1_UUID:
        raise RuntimeError('Only the explicitly authorized GPU1 to GPU0 handoff is supported')
    result = deepcopy(plan)
    result.update(device='cuda:0', gpu_uuid=GPU0_UUID)
    result['resource_policy']['gpu_peak_bytes'] = None
    return result


def validate_training(base, root, plan, seed):
    """Delegate every frozen scientific check; only the own-VRAM veto is off.

    Infinity is an in-memory comparison adapter, never serialized into a plan.
    The original validator still records the actual peak and exact endpoint.
    """
    validation = deepcopy(plan)
    disabled = validation['resource_policy']['gpu_peak_bytes'] is None
    if disabled:
        validation['resource_policy']['gpu_peak_bytes'] = math.inf
    result = base.validate_training(root, validation, seed)
    result['gpu_peak_limit_disabled'] = disabled
    return result


def load_runtime(root):
    if root != ROOT:
        raise ValueError('This authorization applies to the exact 20261023 B0 queue only')
    plan = read(root / 'plan.json')
    sys.path.insert(0, plan['code_root'])
    from scripts import run_ising_b0_multiseed as base
    if Path(base.__file__).resolve() != Path(plan['code_root']) / 'scripts/run_ising_b0_multiseed.py':
        raise RuntimeError('Not the exact frozen training runtime')
    plan, job = base.verify(root)
    return base, plan, job


def process_identity(pid):
    try:
        path = Path('/proc') / str(pid)
        fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, state=fields[0], ppid=int(fields[1]),
            process_group=int(fields[2]), session=int(fields[3]), start_ticks=fields[19],
            uid=path.stat().st_uid,
            cpu_seconds=sum(int(fields[i]) for i in (11, 12, 13, 14)) / os.sysconf('SC_CLK_TCK'))
    except (FileNotFoundError, ProcessLookupError):
        return None


def has_children(pid):
    return bool((Path('/proc') / str(pid) / 'task' / str(pid) / 'children').read_text().strip())


def pending_absent(root):
    paths = [root / 'comparison', root / 'logs/comparison.log']
    for seed in range(92, 96):
        paths += [root / f'seed{seed}', root / f'logs/seed{seed}_training.log', root / f'logs/seed{seed}_report.log']
    if any(path.exists() or path.is_symlink() for path in paths):
        raise RuntimeError('An unexecuted stage already has artifacts; no overwrite/retry')


def prepare(root, output):
    base, plan, job = load_runtime(root)
    if output != root / 'execution_gpu0_v1' or output.exists() or output.is_symlink() or (root / 'execution_override.json').exists():
        raise FileExistsError('Require a new exact execution override; never repeat a handoff')
    remaining_stages(job)
    pending_absent(root)
    runtime_plan(plan)  # Reject an unauthorized old device before sending signals.
    base.validate_training(root, plan, None)
    if base.validate_training(root, plan, 91) != job['training']['91']:
        raise RuntimeError('Completed seed91 receipt differs')
    if read(root / 'seed91/report/report.json').get('status') != 'complete':
        raise RuntimeError('Completed formal report is missing')
    public = base.public_files(root)
    checkpoints = {str(root / folder / f'training_step_{step}.pt'): base.stat_identity(root / folder / f'training_step_{step}.pt')
                   for folder, step in [('profile_run', 20), ('seed91/run', 12000)]}
    pid = job['parent_pid']
    identity = process_identity(pid)
    command = (Path('/proc') / str(pid) / 'cmdline').read_bytes().decode().split('\0')[:-1]
    if (pid != 17173 or identity is None or identity['uid'] != os.getuid()
            or identity['state'] == 'Z' or identity['session'] != pid or identity['process_group'] != pid
            or Path(command[0]).resolve() != Path(sys.executable).resolve()
            or command[1:] != ['-u', '-m', base.MODULE, '--execute', '--output', str(root)]
            or has_children(pid)):
        raise RuntimeError('Not the exact own idle scheduling parent; no signal sent')
    stopped = False
    try:
        os.kill(pid, signal.SIGSTOP)
        stopped = True
        for _ in range(100):
            current = process_identity(pid)
            if current and current['state'] in ('T', 't'):
                break
            time.sleep(.01)
        if (current is None or current['start_ticks'] != identity['start_ticks']
                or current['state'] not in ('T', 't') or has_children(pid)):
            raise RuntimeError('Parent advanced or child appeared; release without handoff')
        plan, job = base.verify(root)
        names = remaining_stages(job)
        pending_absent(root)
        if base.public_files(root) != public:
            raise RuntimeError('Completed public artifacts changed before handoff')
        output.mkdir()
        base.save(output / 'original_job.json', job, exclusive=True)
        base.save(output / 'original_plan.json', plan, exclusive=True)
        identity.update(cpu_seconds=current['cpu_seconds'])
        os.kill(pid, signal.SIGTERM)
        os.kill(pid, signal.SIGCONT)
        stopped = False
        for _ in range(100):
            current = process_identity(pid)
            if current is None or (current['start_ticks'] == identity['start_ticks'] and current['state'] == 'Z'):
                break
            if current['start_ticks'] != identity['start_ticks']:
                raise RuntimeError('PID identity changed; no replacement launch')
            time.sleep(.05)
        else:
            raise RuntimeError('Old parent termination not confirmed; no replacement launch')
    finally:
        if stopped:
            current = process_identity(pid)
            if current and current['start_ticks'] == identity['start_ticks']:
                os.kill(pid, signal.SIGCONT)
    policy = dict(status='prepared', at=base.now(), root=str(root), output=str(output),
        authorization=AUTHORIZATION, previous_parent=identity, previous_parent_termination_confirmed=True,
        original_plan_sha256=base.sha(root / 'plan.json'), original_job_sha256=base.sha(root / 'job.json'),
        wrapper_path=str(Path(__file__).resolve()), wrapper_sha256=base.sha(__file__),
        pending_stages=names, retained_stages=PREFIX,
        runtime_plan=runtime_plan(plan), completed_public_files=public, completed_checkpoint_stat=checkpoints,
        carried_cpu_seconds=max(job['cpu_seconds'], identity['cpu_seconds']),
        carried_gpu_seconds=job['gpu_elapsed_seconds'],
        own_gpu_peak_limit_disabled=True, free_gpu_and_no_compute_admission_unchanged=True,
        scientific_configuration_unchanged=True, checkpoint_resume=False, repeated_stages=False, automatic_retry=False)
    base.save(output / 'policy.json', policy, exclusive=True)
    base.save(output / 'policy_fingerprint.json', dict(sha256=base.sha(output / 'policy.json')), exclusive=True)
    new_job = deepcopy(job)
    new_job.update(status='prepared', current=None, parent_pid=None, execution_override='execution_gpu0_v1',
        retained_stage_count=len(PREFIX), cpu_seconds=policy['carried_cpu_seconds'],
        device='cuda:0', gpu_uuid=GPU0_UUID, own_gpu_peak_limit_disabled=True)
    base.save(output / 'job.json', new_job, exclusive=True)
    base.save(root / 'execution_override.json', dict(policy=str(output / 'policy.json'),
        authoritative_job=str(output / 'job.json'), original_parent_terminated=True,
        original_job_preserved=True, scientific_source_unchanged=True), exclusive=True)
    return dict(status='prepared', device='cuda:0', pending_stages=names,
        carried_cpu_seconds=policy['carried_cpu_seconds'], carried_gpu_seconds=policy['carried_gpu_seconds'])


def verify(root, output):
    base, plan, original_job = load_runtime(root)
    if output != root / 'execution_gpu0_v1':
        raise RuntimeError('Wrong override location')
    policy = read(output / 'policy.json')
    if (base.sha(output / 'policy.json') != read(output / 'policy_fingerprint.json')['sha256']
            or policy['wrapper_path'] != str(Path(__file__).resolve())
            or base.sha(__file__) != policy['wrapper_sha256']
            or base.sha(root / 'plan.json') != policy['original_plan_sha256']
            or base.sha(root / 'job.json') != policy['original_job_sha256']
            or read(output / 'original_job.json') != original_job
            or read(output / 'original_plan.json') != plan
            or policy['pending_stages'] != remaining_stages(original_job)
            or policy['runtime_plan'] != runtime_plan(plan)):
        raise RuntimeError('Frozen handoff/source/plan changed')
    # New reports grow the root whitelist; only the original explicit files are immutable.
    for row in policy['completed_public_files']:
        path = root / row['path']
        if path.is_symlink() or not path.is_file() or path.stat().st_size != row['bytes'] or base.sha(path) != row['sha256']:
            raise RuntimeError('A completed public result changed')
    for path, identity in policy['completed_checkpoint_stat'].items():
        if base.stat_identity(path) != identity:
            raise RuntimeError('A completed checkpoint changed')
    old = process_identity(policy['previous_parent']['pid'])
    if old and old['start_ticks'] == policy['previous_parent']['start_ticks'] and old['state'] != 'Z':
        raise RuntimeError('Old scheduling parent is still alive')
    pointer = read(root / 'execution_override.json')
    if pointer['authoritative_job'] != str(output / 'job.json') or pointer['policy'] != str(output / 'policy.json'):
        raise RuntimeError('Execution authority pointer differs')
    return base, policy, read(output / 'job.json')


def child(root, output, seed):
    # Popen can schedule the child before the parent atomically publishes its
    # PID. Wait only for that registration, never for an existing training run.
    for _ in range(100):
        registered = read(output / 'job.json')
        if (registered.get('status') == 'running' and registered.get('current') == f'seed{seed}_training'
                and registered['stages'][-1].get('pid') == os.getpid()):
            break
        time.sleep(.05)
    else:
        raise RuntimeError('Training child PID was not registered by its coordinator')
    base, policy, job = verify(root, output)
    if (seed not in range(92, 96) or job['current'] != f'seed{seed}_training'
            or job['status'] != 'running' or job['stages'][-1]['pid'] != os.getpid()):
        raise RuntimeError('Child is not the single admitted unstarted training stage')
    # Scope the adapter to this child. The frozen trainer, data, config, and RNG
    # implementation remain byte-identical; no global source is edited.
    original_verify = base.verify
    def admitted_child_plan(target):
        if Path(target).resolve() != root:
            raise RuntimeError('Wrong child root')
        return deepcopy(policy['runtime_plan']), deepcopy(job)
    base.verify = admitted_child_plan
    try:
        return base.train_child(root, root / f'configs/seed{seed}.json', root / f'seed{seed}/run')
    finally:
        base.verify = original_verify


def cpu_seconds():
    return sum(row.ru_utime + row.ru_stime for row in
               (resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)))


def execute(root, output):
    base, policy, job = verify(root, output)
    if job['status'] != 'prepared' or [row['name'] for row in job['stages']] != PREFIX:
        raise RuntimeError('Only the one prepared suffix may execute; no retry')
    plan, prior_cpu = policy['runtime_plan'], policy['carried_cpu_seconds']
    os.nice(10)
    job.update(parent_pid=os.getpid(), actual_nice=os.getpriority(os.PRIO_PROCESS, 0), handoff_started_at=base.now())
    base.save(output / 'job.json', job)
    names = policy['pending_stages']
    try:
        for index, name in enumerate(names):
            verify(root, output)
            resource_policy = plan['resource_policy']
            projected_arm = read(root / 'profile_receipt.json')['projected_seconds_per_training_arm']
            remaining_gpu_arms = sum(stage.endswith('_training') for stage in names[index:])
            remaining_reports = sum(stage.endswith('_report') for stage in names[index:])
            job['cpu_seconds'] = prior_cpu + cpu_seconds()
            if (job['gpu_elapsed_seconds'] + remaining_gpu_arms * projected_arm > resource_policy['gpu_seconds']
                    or job['cpu_seconds'] + remaining_gpu_arms * projected_arm * resource_policy['cpu_threads']
                       + remaining_reports * resource_policy['report_reserve_per_seed'] > resource_policy['cpu_seconds']
                    or base.directory_bytes(root) > resource_policy['directory_bytes']):
                job.update(status='budget_hold', requires_user_action=True, error='Carried actual cost plus remaining conservative projection exceeds unchanged budget')
                break
            gpu = name.endswith('_training')
            # Frozen admission still requires two no-compute / >=12GiB checks
            # ten seconds apart, plus the unchanged host RAM/load/I/O gates.
            base.admission(output, plan, job, name, gpu)
            verify(root, output)
            job['cpu_seconds'] = prior_cpu + cpu_seconds()
            if job['cpu_seconds'] >= resource_policy['cpu_seconds']:
                job.update(status='budget_hold', requires_user_action=True, error='CPU budget reached during resource wait')
                break
            if name == 'comparison':
                seed, folder = None, root / 'comparison'
                command = [sys.executable, '-u', '-m', 'scripts.report_ising_b0_multiseed', '--compare',
                    *(f'{s}={root}/seed{s}/report' for s in base.SEEDS), '--output', str(folder)]
            else:
                seed = int(name.split('_')[0][4:])
                folder = root / f'seed{seed}/report'
                if gpu:
                    if (root / f'seed{seed}').exists():
                        raise RuntimeError('Training output already exists; no checkpoint resume')
                    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--train-child',
                        '--root', str(root), '--output', str(output), '--seed', str(seed)]
                else:
                    command = [sys.executable, '-u', '-m', 'scripts.report_ising_b0_multiseed',
                        '--run', str(root / f'seed{seed}/run'), '--output', str(folder),
                        '--threads', '1', '--sampling-repeats', '2']
            start, before_cpu = time.perf_counter(), cpu_seconds()
            with (root / 'logs' / f'{name}.log').open('x') as stream:
                process = subprocess.Popen(command, cwd=plan['code_root'], stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT,
                    env=dict(os.environ, OMP_NUM_THREADS=str(2 if gpu else 1), MKL_NUM_THREADS=str(2 if gpu else 1),
                             PYTHONUNBUFFERED='1', OPENBLAS_NUM_THREADS='1', NUMBA_NUM_THREADS='1', CUDA_DEVICE_ORDER='PCI_BUS_ID'))
                row = dict(name=name, pid=process.pid, command=command, started_at=base.now(), status='running',
                           returncode=None, gpu_stage=gpu, device=plan['device'] if gpu else 'cpu')
                job['stages'].append(row)
                job.update(status='running', current=name)
                base.save(output / 'job.json', job)
                returncode = process.wait()
            elapsed = time.perf_counter() - start
            row.update(status='complete' if returncode == 0 else 'failed', returncode=returncode,
                       ended_at=base.now(), wall_seconds=elapsed, cpu_seconds=cpu_seconds()-before_cpu)
            if gpu:
                job['gpu_elapsed_seconds'] += elapsed
            job['cpu_seconds'] = prior_cpu + cpu_seconds()
            if returncode == 75 and not gpu and read(folder / 'status.json').get('status') == 'analysis_budget_hold':
                row['status'] = 'analysis_budget_hold'
                job.update(status='analysis_budget_hold', requires_user_action=True, error='Frozen report analysis time budget reached; no retry')
                break
            if returncode:
                raise RuntimeError(f'{name} returned {returncode}; outputs preserved, no retry')
            verify(root, output)
            if gpu:
                summary = read(root / f'seed{seed}/run/summary.json')
                if summary.get('stopped_by_training_limit') or summary.get('stopped_by_wall_limit'):
                    job.update(status='budget_hold', requires_user_action=True, error=f'{name} safety clock reached; partial output preserved')
                    break
                job['training'][str(seed)] = validate_training(base, root, plan, seed)
            else:
                if read(folder / 'report.json').get('status') != 'complete':
                    raise RuntimeError('Public report incomplete')
                if seed is not None and read(folder / 'metrics.json').get('encoding_protocol') != plan['protocol']['encoding_protocol']:
                    raise RuntimeError('Training-matched report encoding protocol differs')
            base.save(output / 'job.json', job)
        else:
            job.update(status='complete', workflow_complete=True, current=None, ended_at=base.now(),
                       next_stage='B loss removals not automatically launched')
            job['public_files'] = base.public_files(root)
    except BaseException as error:
        job.update(status='failed', workflow_complete=False, requires_user_action=True,
                   error=f'{type(error).__name__}: {error}', ended_at=base.now())
        raise
    finally:
        job['cpu_seconds'] = prior_cpu + cpu_seconds()
        base.save(output / 'job.json', job)


def launch(root, output):
    base, policy, job = verify(root, output)
    if job['status'] != 'prepared' or (output / 'launch.json').exists():
        raise RuntimeError('No repeated launch')
    with (output / 'launch.json').open('x') as receipt, (output / 'parent.log').open('x') as log:
        process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--execute',
            '--root', str(root), '--output', str(output)], cwd=policy['runtime_plan']['code_root'],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            env=dict(os.environ, PYTHONUNBUFFERED='1', CUDA_DEVICE_ORDER='PCI_BUS_ID'))
        result = dict(status='launched', at=base.now(), parent_pid=process.pid, device='cuda:0',
            policy_sha256=base.sha(output / 'policy.json'), own_gpu_peak_limit_disabled=True,
            fresh_unstarted_seeds=[92, 93, 94, 95], checkpoint_resumed=False)
        receipt.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare', 'launch', 'execute', 'train-child'):
        modes.add_argument('--' + name, action='store_true')
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if args.prepare:
        result = prepare(root, output)
    elif args.launch:
        result = launch(root, output)
    elif args.execute:
        return execute(root, output)
    else:
        return child(root, output, args.seed)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
