"""Queue and supervise one frozen, fresh nonlinear-decoder joint QCR run.

Resource waiting consumes no GPU and is outside the two-hour active budget.
No resampling, resume, migration or automatic retry. Only the child process
group created here can be stopped if its declared execution budget is exceeded.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import argparse
from pathlib import Path
import resource
import signal
import socket
import subprocess
import sys
import time
import traceback

from experiments import phi4_ru_cnn_nonlinear as spec
from scripts import run_gaussian_scalar_qcr as common
from scripts.run_phi4_ru_qcr import process_snapshot

HOST = 'thinklab-105-225'
DIRECTORY_BYTES = 4*1024**3
CPU_SECONDS = 14400


def build_plan(root, code):
    return dict(schema_version=5,recipe=spec.RECIPE,output=str(Path(root).resolve()),
        code_root=str(Path(code).resolve()),source_sha256=common.source_hashes(code),
        host=HOST,gpu_index=spec.GPU_INDEX,gpu_uuid=spec.GPU_UUID,config=spec.preset(),
        physical_sampling=False,checkpoint_warm_start=False,all_Q_C_R_fresh_trainable=True,
        direction_balancing=False,extra_R_only_stage=False,
        resource_policy=dict(minimum_available_gib=36,load1_max_per_cpu=1.25,
            io_full_avg10_max=20,free_gpu_mib_min=12288,no_compute_process=True,
            admission_checks=2,minimum_check_interval_seconds=10,threads=1,nice=10),
        budgets=dict(active_pipeline_seconds=7200,cpu_seconds=CPU_SECONDS,
            own_cuda_bytes=8*1024**3,directory_bytes=DIRECTORY_BYTES,
            resource_wait_is_outside_active_budget=True),
        scientific_policy=dict(steps=6000,seed=91,labels_used_in_training=False,
            endpoint_selection=False,no_default_change=True,thermal_nu=None,
            retain_existing_sampling_warnings=True,equilibrium_certified=False))


def total_cpu():
    return sum(resource.getrusage(w).ru_utime+resource.getrusage(w).ru_stime
               for w in (resource.RUSAGE_SELF,resource.RUSAGE_CHILDREN))


def execute(root):
    root = Path(root).resolve()
    code = Path(__file__).resolve().parents[1]
    if (root != Path(spec.ROOT) or str(code) != spec.SNAPSHOT
            or socket.gethostname() != HOST or os.getpriority(os.PRIO_PROCESS,0)<10):
        raise RuntimeError('Exact authorized output, frozen snapshot, host and nice>=10 required')
    if root.exists() or root.is_symlink():
        raise FileExistsError('Never resume or overwrite an existing run')
    root.mkdir(parents=True,exist_ok=False)
    (root/'logs').mkdir()
    plan = build_plan(root,code)
    common.save(root/'plan.json',plan,exclusive=True)
    common.save(root/'training_config.json',plan['config'],exclusive=True)
    plan_sha, config_sha = common.sha(root/'plan.json'),common.sha(root/'training_config.json')
    job = dict(status='prepared',workflow_complete=False,current='resource_admission',
               pid=os.getpid(),host=HOST,started_at=common.now(),returncode=None,error=None,
               requires_user_action=False,stages=[],plan_sha256=plan_sha,config_sha256=config_sha)
    def state(**fields):
        job.update(fields)
        common.save(root/'job.json',job)
        common.save(root/'status.json',job)
    state()
    common.save(root/'launch.json',dict(parent_pid=os.getpid(),host=HOST,cwd=str(code),
        command=sys.argv,nice=os.getpriority(os.PRIO_PROCESS,0),gpu_uuid=spec.GPU_UUID,
        plan_sha256=plan_sha,started_at=common.now()),exclusive=True)
    active_start = None
    try:
        if common.sha(Path(plan['config']['data_root'])/'manifest.json') != spec.DATA_SHA:
            raise RuntimeError('Frozen source data manifest mismatch')
        consecutive = 0
        while consecutive < 2:
            row = dict(at=common.now(),**spec.resource_state())
            consecutive = consecutive+1 if row['ready'] else 0
            rows = job.get('resource_checks',[])+[row]
            state(status='waiting_resources',resource_checks=rows[-20:],
                  consecutive_ready_checks=consecutive)
            if consecutive < 2:
                time.sleep(10 if consecutive else 30)
        common.unchanged(plan)
        if common.sha(root/'plan.json')!=plan_sha or common.sha(root/'training_config.json')!=config_sha:
            raise RuntimeError('Frozen plan/config changed during resource wait')
        command=[sys.executable,'-m','recurrent_rg','train','--config',str(root/'training_config.json'),
                 '--output',str(root/'run')]
        env=os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=spec.GPU_UUID,CUDA_DEVICE_ORDER='PCI_BUS_ID',
                   PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
        active_start=time.monotonic()
        gate=None
        with (root/'logs/training_and_report.log').open('x') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            stage=dict(name='training_and_report',pid=child.pid,command=command,
                       started_at=common.now(),status='running',returncode=None)
            job['stages'].append(stage)
            common.save(root/'child_launch.json',dict(stage,gpu_uuid=spec.GPU_UUID,
                         config_sha256=config_sha),exclusive=True)
            state(status='running',current='training_and_report')
            while child.poll() is None:
                live=process_snapshot(child.pid)
                active=time.monotonic()-active_start
                cpu=total_cpu()+live['cpu_seconds']
                if active>plan['budgets']['active_pipeline_seconds']:
                    gate='active_pipeline_seconds'
                elif cpu>CPU_SECONDS:
                    gate='cpu_seconds'
                elif common.directory_bytes(root)>DIRECTORY_BYTES:
                    gate='directory_bytes'
                if gate:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid,signal.SIGKILL);child.wait()
                    break
                state(current_process=live,active_seconds=active,cpu_seconds=cpu)
                time.sleep(5)
            returncode=child.wait()
        stage.update(returncode=returncode,finished_at=common.now(),budget_gate=gate,
                     status='complete' if returncode==0 else 'budget_hold' if gate else 'failed')
        common.save(root/'execution.json',dict(returncode=returncode,budget_gate=gate,
            active_seconds=time.monotonic()-active_start,cpu_seconds=total_cpu(),
            finished_at=common.now()),exclusive=True)
        if returncode or gate:
            outcome=common.read(root/'run/status.json') if (root/'run/status.json').exists() else {}
            hold=bool(gate) or outcome.get('status')=='budget_hold'
            stage['status']='budget_hold' if hold else 'failed'
            state(status=stage['status'],returncode=3 if hold else returncode,
                  error=outcome.get('error') or gate or f'Child exited {returncode}',requires_user_action=True)
            return job['returncode']
        common.unchanged(plan)
        outcome=common.read(root/'run/status.json')
        if (outcome.get('status')!='complete' or not outcome.get('workflow_complete')
                or outcome.get('returncode')!=0 or outcome.get('actual_steps')!=6000):
            raise RuntimeError('Missing complete 6000-step training/report result')
        if common.sha(root/'training_config.json')!=config_sha:
            raise RuntimeError('Frozen config changed')
        public={name:common.sha(root/name) for name in
                ('plan.json','training_config.json','launch.json','child_launch.json','execution.json')}
        for name,digest in outcome['public_files'].items():
            path=Path(name)
            if path.is_absolute() or '..' in path.parts or path.suffix not in ('.json','.png','.npz'):
                raise RuntimeError('Unsafe public whitelist')
            if common.sha(root/'run'/path)!=digest:
                raise RuntimeError('Public result hash mismatch')
            public['run/'+name]=digest
        state(status='complete',workflow_complete=True,current=None,returncode=0,
              public_files=public,finished_at=common.now())
        return 0
    except Exception as exc:
        state(status='failed',returncode=1,error=f'{type(exc).__name__}: {exc}',
              traceback=traceback.format_exc(),requires_user_action=True)
        if not (root/'execution.json').exists():
            common.save(root/'execution.json',dict(returncode=1,error=job['error'],
                active_seconds=None if active_start is None else time.monotonic()-active_start,
                cpu_seconds=total_cpu(),finished_at=common.now()),exclusive=True)
        return 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true',required=True)
    args=parser.parse_args()
    return execute(spec.ROOT)


if __name__=='__main__':
    raise SystemExit(main())
