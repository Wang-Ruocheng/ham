"""One immutable, paired current-QCR experiment; no retries or migration."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from scripts.run_dual_ising_current import eligible, resource_snapshot, sha256, verify_plan, write_json


def commands(plan, root):
    stages = []
    for name, weight in [('unsupervised', 0), ('supervised', 10)]:
        stages.append((name, [sys.executable, '-m', 'experiments.rsmi_separate_qcr',
            '--config', plan['config'], '--output', str(root/name), '--device', 'cuda:0',
            '--q-coupling-weight', str(weight), '--mi-cache-dir', plan['mi_cache_dir'],
            '--mi-test-cpu', '--benchmark-warmup', '20', '--skip-final-report']))
    stages.append(('report', [sys.executable, '-m', 'scripts.report_ising_b0_supervision_pair',
                             '--root', str(root)]))
    return stages


def main():
    p = argparse.ArgumentParser(); p.add_argument('--plan', type=Path, required=True)
    plan = json.loads(p.parse_args().plan.read_text())
    if os.uname().nodename != 'thinklab-105-225' or os.getpriority(os.PRIO_PROCESS, 0) < 10:
        raise RuntimeError('Wrong host or priority')
    root = Path(plan['output']); root.mkdir(parents=True, exist_ok=False)
    write_json(root/'plan.json', plan)
    state = dict(status='preflight', workflow_complete=False, parent_pid=os.getpid(),
                 stages=[], error=None, requires_user_action=False, started_unix=time.time())
    write_json(root/'job.json', state)
    active_seconds = 0.; selected_uuid = plan['gpu_uuid']
    try:
        verify_plan(plan, Path.cwd())
        for name, command in commands(plan, root):
            if name != 'report':
                while True:
                    first = resource_snapshot()
                    state.update(status='waiting_resources', current=name, resource=first)
                    write_json(root/'job.json', state)
                    candidates = [g for g in eligible(first)
                                  if selected_uuid is None or g['uuid'] == selected_uuid]
                    if candidates:
                        time.sleep(10)
                        second = resource_snapshot()
                        free = {g['uuid'] for g in eligible(second)}
                        selected = next((g for g in candidates if g['uuid'] in free), None)
                        if selected:
                            selected_uuid = selected['uuid']
                            state.setdefault('admissions', []).append(dict(arm=name, first=first, second=second, selected=selected))
                            break
                    time.sleep(60)
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=selected_uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID',
                OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                NUMEXPR_NUM_THREADS='1', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
            left = plan['active_wall_limit_seconds']-active_seconds
            if left <= 0:
                state['status'] = 'budget_hold'
                raise TimeoutError('Active-wall ceiling exhausted; no automatic continuation')
            stage = dict(name=name, command=command, started_unix=time.time(), returncode=None)
            state['stages'].append(stage); state.update(status='running', current=name)
            started = time.monotonic()
            with (root/(name+'.log')).open('x') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                         env=env, start_new_session=True)
                stage['pid'] = child.pid; write_json(root/'job.json', state)
                try:
                    code = child.wait(timeout=left)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGTERM)
                    try: code = child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL); code = child.wait()
                    stage['returncode'] = code; state['status'] = 'budget_hold'
                    raise TimeoutError('Predeclared active-wall ceiling reached; partial files retained')
            active_seconds += time.monotonic()-started
            stage.update(returncode=code, finished_unix=time.time())
            state['active_wall_seconds'] = active_seconds
            write_json(root/'job.json', state)
            if code != 0: raise RuntimeError(f'{name} exited {code}; no automatic retry')
        report = json.loads((root/'report/report.json').read_text())
        if report['status'] != 'complete': raise RuntimeError('Incomplete comparison')
        public = [root/'plan.json']
        for base in [root/'report', root/'unsupervised', root/'supervised',
                     root/'unsupervised/r_only', root/'supervised/r_only']:
            public.extend(x for x in base.iterdir() if x.is_file() and x.suffix in ('.json','.png','.npz'))
        state.update(status='complete', workflow_complete=True, current=None,
            finished_unix=time.time(), public_files=[dict(path=str(x.relative_to(root)),sha256=sha256(x)) for x in sorted(set(public))])
        write_json(root/'job.json', state)
    except Exception as exc:
        if state['status'] != 'budget_hold': state['status'] = 'failed'
        state.update(error=str(exc), requires_user_action=True, finished_unix=time.time())
        write_json(root/'job.json', state)
        raise


if __name__ == '__main__': main()
