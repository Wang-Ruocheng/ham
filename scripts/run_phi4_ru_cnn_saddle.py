"""Frozen, fresh range-only workflow: pilot -> sampling -> CNN QCR/report.

No retries, automatic tuning, migration, shortened training or historical writes.
Only this workflow's own child can be terminated at its declared budget gate.
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS',
              'NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import argparse
import json
import math
from pathlib import Path
import resource
import signal
import socket
import subprocess
import sys
import time
import traceback

from experiments import phi4_ru_cnn_saddle as spec
from scripts import run_gaussian_scalar_qcr as common
from scripts import run_phi4_ru_qcr as previous

MODULE = 'scripts.run_phi4_ru_cnn_saddle'
HOST = 'thinklab-105-225'
GPU_UUID = 'GPU-a167decd-ce49-6c8f-6bcf-136fde596683'
CPU_SECONDS = 14400
SAMPLING_SECONDS = 7200
TRAINING_SECONDS = 7200
DIRECTORY_BYTES = 4*1024**3
read, save, sha = common.read, common.save, common.sha


class Hold(RuntimeError):
    pass


def build_plan(root, code):
    from recurrent_rg.data.phi4_ru_pipeline import generation_plan, source_pins
    physical = json.loads(json.dumps(generation_plan(spec.sampling_options())))
    pilot_cells = [0, 4, 40, 44]
    assert set(pilot_cells) <= set(physical['train_cell_indices'])
    pilot_cycles = 12*(128+8*4)
    return dict(schema_version=5, recipe=spec.RECIPE, output=str(Path(root).resolve()),
        code_root=str(Path(code).resolve()), host=HOST, gpu_uuid=GPU_UUID,
        source_sha256=common.source_hashes(code), sampler_source_pins=source_pins(), physical=physical,
        pilot=dict(cells=pilot_cells, initials=['hot','cold_positive','cold_negative'],
                   seed_base=202611030100, samples=8, burn_in=128, spacing=4,
                   expected_chains=12, TRAIN_only=True, discarded=True),
        work_including_pilots=dict(hmc_trajectories=physical['total_hmc_trajectories']+pilot_cycles,
            force_site_evaluations=physical['total_force_site_evaluations']+pilot_cycles*13*36**2),
        training_template=spec.preset('0'*64),
        manifest_binding='Replace only the zero SHA with the fresh complete manifest SHA after validity gate; seal training_config.json before launch',
        budgets=dict(cpu_seconds_total=CPU_SECONDS,sampling_wall_seconds=SAMPLING_SECONDS,
            training_pipeline_wall_seconds=TRAINING_SECONDS,own_cuda_bytes=8*1024**3,
            directory_bytes=DIRECTORY_BYTES,threads=1,nice=10),
        scientific_policy=dict(steps=6000,optimizer_seed=91,labels_used_in_training=False,
            no_pretraining=True,no_R_only_stage=True,no_model_selection=True,
            retained_sampling_warnings=True,equilibrium_certified=False,
            literature_saddle_used_only_to_choose_scan=True,
            reference_saddle_ru=[-1.97,2.13],reference_saddle_is_scheme_dependent=True,
            thermal_nu=None),
        source_reference='https://arxiv.org/html/2510.06508v1#S4.SS2')


def check_host():
    if socket.gethostname()!=HOST or os.getpriority(os.PRIO_PROCESS,0)<10:
        raise RuntimeError('Expected thinklab-105-225 and nice >=10')


def load_plan(root):
    plan, job = read(root/'plan.json'),read(root/'job.json')
    if plan['output']!=str(root.resolve()) or sha(root/'plan.json')!=job['plan_sha256']:
        raise RuntimeError('Frozen plan/output identity changed')
    common.unchanged(plan)
    return plan


def snapshot():
    mem=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines()
                 if x.startswith('MemAvailable:')))*1024
    io=float(next(x.split('avg10=')[1].split()[0] for x in Path('/proc/pressure/io').read_text().splitlines()
                  if x.startswith('full ')))
    gpu=subprocess.check_output(['nvidia-smi','-i','3','--query-gpu=uuid,memory.free',
                                  '--format=csv,noheader,nounits'],text=True,timeout=10).strip().split(',')
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
                                       '--format=csv,noheader'],text=True,timeout=10)
    if gpu[0].strip()!=GPU_UUID:
        raise RuntimeError('Physical GPU3 UUID mismatch; no migration')
    host_ready=mem>=36*1024**3 and os.getloadavg()[0]<=1.25*os.cpu_count() and io<20
    return dict(at=common.now(),mem_available_bytes=mem,load1=os.getloadavg()[0],
        io_full_avg10=io,gpu_uuid=GPU_UUID,free_gpu_mib=int(gpu[1]),
        compute_process_present=GPU_UUID in processes,host_ready=host_ready,
        gpu_ready=host_ready and int(gpu[1])>=12288 and GPU_UUID not in processes)


def write_state(root, job, **fields):
    job.update(fields)
    save(root/'job.json',job)
    save(root/'status.json',job)


def admission(root, job, stage, gpu):
    consecutive=0
    while True:
        row=snapshot()
        job.setdefault('resource_checks',[]).append(row)
        job['resource_checks']=job['resource_checks'][-20:]
        write_state(root,job,status='waiting_resources',current=stage)
        consecutive=consecutive+1 if row['gpu_ready' if gpu else 'host_ready'] else 0
        if consecutive >= (2 if gpu else 1):
            return
        time.sleep(10 if consecutive else 30)


def total_cpu():
    return sum(resource.getrusage(who).ru_utime+resource.getrusage(who).ru_stime
               for who in (resource.RUSAGE_SELF,resource.RUSAGE_CHILDREN))


def run_child(root, job, name, command, wall_budget, gpu=False):
    load_plan(root)
    admission(root,job,name,gpu)
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=GPU_UUID if gpu else '',CUDA_DEVICE_ORDER='PCI_BUS_ID',
               PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
    start=time.monotonic()
    log_path=root/'logs'/(name+'.log')
    with log_path.open('x') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        stage=dict(name=name,command=command,pid=child.pid,started_at=common.now(),
                   status='running',returncode=None,log=str(log_path),wall_budget=wall_budget)
        job['stages'].append(stage)
        write_state(root,job,status='running',current=name)
        exceeded=None
        while child.poll() is None:
            live=previous.process_snapshot(child.pid)
            job['current_process']=live
            if time.monotonic()-start>wall_budget:
                exceeded='stage_wall_budget'
            if total_cpu()+live['cpu_seconds']>CPU_SECONDS:
                exceeded='total_cpu_budget'
            if common.directory_bytes(root)>DIRECTORY_BYTES:
                exceeded='directory_budget'
            if exceeded:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);child.wait()
                break
            write_state(root,job,total_cpu_seconds=total_cpu()+live['cpu_seconds'])
            time.sleep(5)
        code=child.wait()
    stage.update(status='complete' if code==0 else 'failed',returncode=code,
                 wall_seconds=time.monotonic()-start,finished_at=common.now(),budget_gate=exceeded)
    write_state(root,job)
    if exceeded:
        stage['status']='budget_hold'
        raise Hold(exceeded)
    if code:
        # Preserve the child's semantic budget hold rather than calling it a failure.
        child_status=root/('run/status.json' if gpu else 'data/status.json')
        if child_status.exists() and read(child_status).get('status')=='budget_hold':
            raise Hold(read(child_status).get('error','child budget hold'))
        raise RuntimeError(f'{name} exited {code}; see {log_path}')
    load_plan(root)


def preflight(root):
    from recurrent_rg.data.phi4_hybrid import sample_hybrid_chain
    check_host();plan=load_plan(root)
    pilot, options=plan['pilot'],plan['physical']['options']
    started=time.process_time();rows=[]
    for cell in pilot['cells']:
        for initial in pilot['initials']:
            r,u=options['r_values'][cell//5],options['u_values'][cell%5]
            result=sample_hybrid_chain(r,u,size=36,dimension=2,h=0,samples=8,burn_in=128,spacing=4,
                step_size=.08,leapfrog_steps=12,cluster_probability=.5,
                seed=pilot['seed_base']+len(rows),initial=initial)
            row={k:result[k] for k in ('cpu_seconds','wall_seconds','hmc_trajectories',
                                      'force_site_evaluations','production_acceptance','max_absolute_delta_h')}
            if any(not math.isfinite(float(v)) for v in row.values()) or row['production_acceptance']<=0:
                raise FloatingPointError('Nonfinite or zero-acceptance pilot; no automatic adaptation')
            rows.append(dict(row,r=r,u=u,cell=cell,initial=initial,warnings=result['warnings']))
            save(root/'sampler_preflight_progress.json',dict(completed=len(rows),total=12,last=rows[-1]))
    projection=2*max(x['cpu_seconds']/x['hmc_trajectories'] for x in rows)*plan['physical']['total_hmc_trajectories']
    actual=time.process_time()-started
    passed=projection+actual+120<SAMPLING_SECONDS and projection+actual+TRAINING_SECONDS+600<CPU_SECONDS
    receipt=dict(status='complete' if passed else 'budget_hold',budget_passed=passed,rows=rows,
        projected_production_cpu_seconds_conservative=projection,actual_cpu_seconds=actual,
        TRAIN_only=True,production_data_reused=False,execution_only=True,equilibrium_certified=False)
    save(root/'sampler_preflight.json',receipt,exclusive=True)
    return receipt


def execute(root):
    check_host()
    if root!=Path(spec.ROOT):
        raise ValueError('Only the exact fresh authorized root is supported')
    if root.exists() or root.is_symlink():
        raise FileExistsError('Existing output is never resumed, retried or overwritten')
    code=Path(__file__).resolve().parents[1]
    if str(code)!=spec.SNAPSHOT:
        raise RuntimeError('Run only from the separately frozen snapshot')
    plan=build_plan(root,code)
    root.mkdir(parents=True,exist_ok=False);(root/'logs').mkdir()
    save(root/'plan.json',plan,exclusive=True)
    job=dict(status='prepared',workflow_complete=False,current=None,pid=os.getpid(),host=HOST,
        stages=[],returncode=None,error=None,requires_user_action=False,
        plan_sha256=sha(root/'plan.json'),started_at=common.now())
    write_state(root,job)
    save(root/'launch.json',dict(pid=os.getpid(),command=sys.argv,cwd=str(code),host=HOST,
        gpu_uuid=GPU_UUID,nice=os.getpriority(os.PRIO_PROCESS,0),plan_sha256=job['plan_sha256']),exclusive=True)
    start=time.monotonic()
    try:
        run_child(root,job,'sampler_preflight',[sys.executable,'-m',MODULE,'--preflight','--output',str(root)],600)
        if not read(root/'sampler_preflight.json')['budget_passed']:
            raise Hold('Conservative production projection exceeds sampling/CPU budget')
        run_child(root,job,'physical_sampling',previous.sampling_command(root,plan),SAMPLING_SECONDS)
        validity=previous.sampling_validity(root,plan)
        save(root/'sampling_validity.json',validity,exclusive=True)
        if validity['status']!='complete':
            raise Hold('Physical sampling has numerical/zero-acceptance failures; training not started')
        config=spec.preset(sha(root/'data/manifest.json'))
        save(root/'training_config.json',config,exclusive=True)
        job['training_config_sha256']=sha(root/'training_config.json')
        write_state(root,job,data_warning_count=validity['warning_count'])
        run_child(root,job,'training_and_report',[sys.executable,'-m','recurrent_rg','train',
            '--config',str(root/'training_config.json'),'--output',str(root/'run')],TRAINING_SECONDS,gpu=True)
        outcome=read(root/'run/status.json')
        if outcome['status']!='complete' or not outcome['workflow_complete'] or outcome['returncode']!=0:
            raise RuntimeError('Training/report completeness check failed')
        if sha(root/'training_config.json')!=job['training_config_sha256']:
            raise RuntimeError('Frozen training config changed')
        public={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()
                and p.suffix in ('.json','.png') and p.name not in ('job.json','status.json')}
        for relative,digest in outcome['public_files'].items():
            if Path(relative).suffix in ('.json','.npz','.png'):
                public['run/'+relative]=digest
        save(root/'execution.json',dict(returncode=0,wall_seconds=time.monotonic()-start,
             cpu_seconds=total_cpu(),finished_at=common.now()),exclusive=True)
        public['execution.json']=sha(root/'execution.json')
        write_state(root,job,status='complete',workflow_complete=True,current=None,returncode=0,
                    public_files=public,finished_at=common.now())
        return 0
    except Exception as exc:
        hold=isinstance(exc,Hold)
        write_state(root,job,status='budget_hold' if hold else 'failed',returncode=3 if hold else 1,
            requires_user_action=True,error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        save(root/'execution.json',dict(returncode=job['returncode'],error=job['error'],
             wall_seconds=time.monotonic()-start,cpu_seconds=total_cpu(),finished_at=common.now()),exclusive=True)
        return job['returncode']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--plan',action='store_true');mode.add_argument('--execute',action='store_true')
    mode.add_argument('--preflight',action='store_true')
    args=parser.parse_args();root=args.output.resolve()
    if args.plan:
        print(json.dumps(build_plan(root,Path(__file__).resolve().parents[1]),indent=2,ensure_ascii=False))
        return 0
    if args.preflight:
        preflight(root);return 0
    return execute(root)


if __name__=='__main__':
    raise SystemExit(main())
