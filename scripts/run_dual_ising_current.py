"""Fresh, immutable job: resource admission, standard train, report.

No automatic retries, migration, resampling, or changes to an existing run.
Only nvidia-smi and /proc are read while waiting; torch is not imported.
GPU sharing requires an explicit pinned per-run policy; idle-only is default.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def resource_snapshot():
    def query(args):
        return subprocess.check_output(['nvidia-smi', *args, '--format=csv,noheader,nounits'],
                                       text=True, timeout=15).strip().splitlines()
    rows = query(['--query-gpu=index,uuid,name,memory.free,utilization.gpu'])
    processes = query(['--query-compute-apps=gpu_uuid,pid'])
    busy = {r.split(',')[0].strip() for r in processes if r.strip()}
    memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    full = next(l for l in Path('/proc/pressure/io').read_text().splitlines() if l.startswith('full '))
    pressure = float(dict(p.split('=') for p in full.split()[1:])['avg10'])
    gpus = []
    for row in rows:
        index, uuid, name, free, utilization = [v.strip() for v in row.split(',')]
        gpus.append(dict(index=int(index), uuid=uuid, name=name, free_mib=float(free),
                         utilization_percent=float(utilization) if utilization.isdigit() else None,
                         compute_process_present=uuid in busy))
    return dict(time_unix=time.time(), gpus=gpus, load1=os.getloadavg()[0],
                cpu_count=os.cpu_count(), mem_available_gib=float(memory['MemAvailable'].split()[0])/1024**2,
                io_full_avg10=pressure)


def eligible(snapshot, shared_gpu=None):
    if (snapshot['mem_available_gib'] < 32 or snapshot['load1'] > 1.25*snapshot['cpu_count']
            or snapshot['io_full_avg10'] >= 20):
        return []
    if shared_gpu is not None:
        if (set(shared_gpu) != {'uuid', 'max_utilization_percent', 'min_free_mib'}
                or not shared_gpu['uuid'].startswith('GPU-')
                or not 0 <= shared_gpu['max_utilization_percent'] <= 30
                or shared_gpu['min_free_mib'] < 12288):
            raise ValueError('Invalid explicitly authorized shared-GPU policy')
        return [g for g in snapshot['gpus'] if g['uuid'] == shared_gpu['uuid']
                and g['free_mib'] >= shared_gpu['min_free_mib']
                and g.get('utilization_percent') is not None
                and g['utilization_percent'] <= shared_gpu['max_utilization_percent']]
    return [g for g in snapshot['gpus'] if not g['compute_process_present'] and g['free_mib'] >= 12288]


def verify_plan(plan, snapshot_root):
    for relative, expected in plan['source_sha256'].items():
        path = (snapshot_root / relative).resolve()
        if not path.is_relative_to(snapshot_root.resolve()) or sha256(path) != expected:
            raise ValueError('Source fingerprint mismatch: '+relative)
    if sha256(plan['data_path']) != plan['data_sha256']:
        raise ValueError('Data fingerprint mismatch')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    a = p.parse_args()
    plan = json.loads(a.plan.read_text()); root = Path(plan['output'])
    if os.uname().nodename != 'thinklab-105-225' or os.getpriority(os.PRIO_PROCESS, 0) < 10:
        raise RuntimeError('Wrong host or priority')
    root.mkdir(parents=True, exist_ok=False)
    write_json(root/'plan.json', plan)
    state = dict(status='preflight', workflow_complete=False, parent_pid=os.getpid(),
                 stages=[], error=None, requires_user_action=False, started_unix=time.time())
    write_json(root/'job.json', state)
    try:
        verify_plan(plan, Path.cwd())
        while True:
            first = resource_snapshot()
            state.update(status='waiting_resources', resource=first)
            write_json(root/'job.json', state)
            candidates = eligible(first, plan.get('shared_gpu'))
            if candidates:
                time.sleep(10)
                second = resource_snapshot()
                available = {g['uuid']:g for g in eligible(second, plan.get('shared_gpu'))}
                selected = next((g for g in candidates if g['uuid'] in available), None)
                if selected is not None:
                    state['admission'] = dict(first=first, second=second, selected=selected)
                    break
            time.sleep(60)
        verify_plan(plan, Path.cwd())
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=selected['uuid'], CUDA_DEVICE_ORDER='PCI_BUS_ID',
                   OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   NUMEXPR_NUM_THREADS='1', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
        pipeline = plan.get('pipeline', 'dual_ising_current')
        if pipeline not in ('dual_ising_current', 'ising_KB_ensemble_vae', 'phi4_ru_ensemble_vae'):
            raise ValueError('Unknown pinned pipeline')
        commands = [
            ('joint_and_terminal_R', [sys.executable, '-m', 'experiments.rsmi_separate_qcr',
             '--config', plan['config'], '--output', str(root/'run'), '--device', 'cuda:0',
             '--mi-cache-dir', str(root/'mi_cache'), '--mi-test-cpu', '--benchmark-warmup', '20',
             '--skip-final-report']),
            ('report', [sys.executable, '-m', 'scripts.report_dual_ising_current',
             '--run', str(root/'run'), '--device', 'cpu'])]
        if pipeline == 'phi4_ru_ensemble_vae':
            commands = [
                ('joint_and_terminal_R', [sys.executable, '-m', 'recurrent_rg', 'train',
                 '--config', plan['config'], '--output', str(root/'run'), '--device', 'cuda:0']),
                ('report', [sys.executable, '-m', 'scripts.report_phi4_ru_vae', '--run', str(root/'run')])]
        if pipeline == 'dual_ising_current' and plan.get('comparison_run'):
            commands[-1][1].extend(['--comparison-run', plan['comparison_run']])
        if plan.get('verify_vae_normalization', False):
            if pipeline not in ('phi4_ru_ensemble_vae', 'ising_KB_ensemble_vae'):
                raise ValueError('VAE normalization preflight requires an explicit VAE pipeline')
            commands.insert(0, ('numerical_preflight', [sys.executable, '-m',
                'scripts.verify_vae_normalization', '--device', 'cuda:0',
                '--output', str(root/'normalization_validation.json')]))
        active_start = time.monotonic()
        for name, command in commands:
            stage = dict(name=name, command=command, started_unix=time.time(), returncode=None)
            state['stages'].append(stage)
            state.update(status='running', current=name)
            with (root/(name+'.log')).open('x') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                         env=env, start_new_session=True)
                stage['pid'] = child.pid; write_json(root/'job.json', state)
                try:
                    left = plan['active_wall_limit_seconds']-(time.monotonic()-active_start)
                    code = child.wait(timeout=max(1, left))
                except subprocess.TimeoutExpired:
                    # This is our own fresh child session, never another user's process.
                    os.killpg(child.pid, signal.SIGTERM)
                    try: code = child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL); code = child.wait()
                    stage['returncode'] = code
                    state['status'] = 'budget_hold'
                    raise TimeoutError('Predeclared active wall limit reached; no automatic continuation')
            stage.update(returncode=code, finished_unix=time.time())
            write_json(root/'job.json', state)
            if code != 0:
                progress = root/'run/status.json'
                if progress.exists() and json.loads(progress.read_text()).get('status') == 'budget_hold':
                    state['status'] = 'budget_hold'
                raise RuntimeError(f'{name} exited {code}; no automatic retry')
        report = json.loads((root/'run/report/report.json').read_text())
        if report['status'] != 'complete': raise RuntimeError('Incomplete formal report')
        public = [root/'plan.json']
        if plan.get('verify_vae_normalization', False):
            public.append(root/'normalization_validation.json')
        for base in [root/'run', root/'run/r_only', root/'run/report']:
            public.extend(p for p in base.iterdir() if p.suffix in ('.json', '.png', '.npz') and p.is_file())
        state.update(status='complete', workflow_complete=True, current=None, finished_unix=time.time(),
                     active_wall_seconds=time.monotonic()-active_start,
                     public_files=[dict(path=str(p.relative_to(root)), sha256=sha256(p)) for p in sorted(set(public))])
        write_json(root/'job.json', state)
        return 0
    except Exception as exc:
        if state['status'] != 'budget_hold': state['status'] = 'failed'
        state.update(error=str(exc), requires_user_action=True, finished_unix=time.time())
        write_json(root/'job.json', state)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
