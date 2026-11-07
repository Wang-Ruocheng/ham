"""Frozen sampling-region ablation; unchanged weak-KL Q/C/R training recipe.

Resource waits do not launch GPU work. No retries, resampling or model selection.
Only this fresh workflow's child process group may be stopped at its budget.
"""
import os
for _n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[_n] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse
import json
from pathlib import Path
import resource
import signal
import socket
import subprocess
import sys
import time
import traceback
from scripts.run_dual_ising_current import sha256, write_json, resource_snapshot, eligible


def options(plan):
    from recurrent_rg.data.phi4_ru_pipeline import Phi4RUOptions
    return Phi4RUOptions(**plan['physical_options'])


def verify(plan):
    frozen = Path(plan["output"])/"plan.json"
    if frozen.exists() and json.loads(frozen.read_text()) != plan:
        raise ValueError("Frozen plan changed")
    for relative, expected in plan['source_sha256'].items():
        p = (Path(plan['snapshot'])/relative).resolve()
        if not p.is_relative_to(Path(plan['snapshot'])) or sha256(p) != expected:
            raise ValueError('Frozen source mismatch: '+relative)


def preflight(plan, root):
    from recurrent_rg.data.phi4_hybrid import sample_hybrid_chain
    from recurrent_rg.data.phi4_ru_pipeline import generation_plan
    import numpy as np
    o = options(plan); work = generation_plan(o); rows=[]
    for cell in plan['pilot_cells']:
        assert cell % o.test_every != o.test_offset
        r,u = o.parameter_cells[cell]
        for initial in ('hot','cold_positive','cold_negative'):
            v = sample_hybrid_chain(r,u,size=36,dimension=2,h=0,samples=8,burn_in=128,spacing=4,
                step_size=o.step_size,leapfrog_steps=o.leapfrog_steps,cluster_probability=.5,
                seed=202611060100+len(rows),initial=initial)
            if (not np.isfinite(v['configurations']).all() or v['production_acceptance'] <= 0
                    or not np.isfinite(v['delta_h']).all()):
                raise FloatingPointError('Invalid pilot; no automatic sampler adaptation')
            rows.append(dict(cell=cell,r=r,u=u,initial=initial,
                cpu_seconds=v['cpu_seconds'],hmc_trajectories=v['hmc_trajectories'],
                production_acceptance=v['production_acceptance'],warnings=v['warnings']))
            write_json(root/'preflight_progress.json',dict(completed=len(rows),total=12,last=rows[-1]))
    projection=2*max(x['cpu_seconds']/x['hmc_trajectories'] for x in rows)*work['total_hmc_trajectories']
    passed=projection < plan['budgets']['sampling_wall_seconds']-120
    write_json(root/'sampling_preflight.json',dict(status='complete' if passed else 'budget_hold',
        passed=passed,rows=rows,conservative_production_cpu_seconds=projection,
        TRAIN_only=True,pilots_discarded=True,equilibrium_certified=False))
    if not passed: raise TimeoutError('Sampling projection exceeds fixed budget')


def sampling(plan, root):
    from recurrent_rg.data.phi4_ru_pipeline import generate_dataset
    generate_dataset(root/'data', options(plan))
    write_json(root/'data/sampling_design.json', plan['design'])


def host_ready(r):
    return r['mem_available_gib']>=32 and r['load1']<=1.25*r['cpu_count'] and r['io_full_avg10']<20


def cpu_used():
    return sum(resource.getrusage(x).ru_utime+resource.getrusage(x).ru_stime
               for x in (resource.RUSAGE_SELF,resource.RUSAGE_CHILDREN))


def child_cpu(pid):
    try:
        v=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return (int(v[11])+int(v[12]))/os.sysconf('SC_CLK_TCK')
    except FileNotFoundError:return 0.


def execute(plan_path):
    plan=json.loads(plan_path.read_text()); root=Path(plan['output'])
    if socket.gethostname()!='thinklab-105-225' or os.getpriority(os.PRIO_PROCESS,0)<10:
        raise RuntimeError('Wrong host or priority')
    if Path.cwd()!=Path(plan['snapshot']):raise RuntimeError('Frozen snapshot cwd required')
    verify(plan); root.mkdir(parents=True,exist_ok=False);(root/'logs').mkdir()
    write_json(root/'plan.json',plan)
    state=dict(status='prepared',workflow_complete=False,parent_pid=os.getpid(),stages=[],
        error=None,requires_user_action=False,started_unix=time.time())
    def status(**kw):
        state.update(kw);write_json(root/'job.json',state)
    def admission(gpu=False,selected=None):
        count=0; chosen=None
        while True:
            r=resource_snapshot();ready=host_ready(r)
            if gpu:
                gs=eligible(r)
                gs.sort(key=lambda x:[2,3,0,1].index(x['index']))
                if selected:gs=[g for g in gs if g['uuid']==selected]
                ready=bool(gs)
                candidate=gs[0]['uuid'] if gs else None
                count=count+1 if ready and candidate==chosen else int(ready)
                chosen=candidate
            else:count=int(ready)
            status(status='waiting_resources',resource=r)
            if count >= (2 if gpu else 1):return chosen
            time.sleep(10 if ready else 30)
    def run(name,command,wall,gpu=None):
        verify(plan)
        stage=dict(name=name,command=command,status='running',returncode=None,started_unix=time.time())
        state['stages'].append(stage)
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu or '',CUDA_DEVICE_ORDER='PCI_BUS_ID')
        start=time.monotonic();reason=None
        with (root/'logs'/(name+'.log')).open('x') as f:
            child=subprocess.Popen(command,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            stage['pid']=child.pid;status(status='running',current=name)
            while child.poll() is None:
                elapsed=time.monotonic()-start;cpu=cpu_used()+child_cpu(child.pid)
                if elapsed>wall:reason='Stage active wall budget'
                if cpu>plan['budgets']['cpu_seconds']:reason='Total CPU budget'
                if sum(p.stat().st_size for p in root.rglob('*') if p.is_file())>plan['budgets']['directory_bytes']:
                    reason='Directory budget'
                if reason:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    break
                status(total_cpu_seconds=cpu,current_elapsed_seconds=elapsed);time.sleep(5)
            code=child.wait()
        stage.update(status='complete' if code==0 else 'failed',returncode=code,
                     wall_seconds=time.monotonic()-start,finished_unix=time.time())
        status()
        if reason:raise TimeoutError(reason)
        if code:
            for p in (root/'run/status.json',root/'sampling_preflight.json'):
                if p.exists() and json.loads(p.read_text()).get('status')=='budget_hold':
                    raise TimeoutError('Child budget_hold: '+name)
            raise RuntimeError(f'{name} exited {code}; preserve outputs, no retry')
        verify(plan)
    try:
        admission()
        run('sampler_preflight',[sys.executable,'-m','scripts.run_phi4_ru_band','--plan',str(plan_path),'--stage','preflight'],600)
        admission()
        run('physical_sampling',[sys.executable,'-m','scripts.run_phi4_ru_band','--plan',str(plan_path),'--stage','sampling'],plan['budgets']['sampling_wall_seconds'])
        quality=json.loads((root/'data/quality_summary.json').read_text())
        if any(min(x['production_acceptance'])<=0 for x in quality['conditions']):
            raise FloatingPointError('Zero production acceptance; training not started')
        run('physical_diagnostics',[sys.executable,'-m','scripts.diagnose_phi4_ru_band','--data',str(root/'data')],600)
        config=json.loads(json.dumps(plan['training_template']));config['data']['manifest_sha256']=sha256(root/'data/manifest.json')
        write_json(root/'training_config.json',config)
        status(training_config_sha256=sha256(root/'training_config.json'),sampling_warning_count=quality['warning_count'])
        chosen=admission(gpu=True);status(selected_gpu_uuid=chosen)
        active=time.monotonic()
        def left():return max(1,plan['budgets']['gpu_pipeline_wall_seconds']-(time.monotonic()-active))
        run('numerical_preflight',[sys.executable,'-m','scripts.verify_vae_normalization','--device','cuda:0',
            '--output',str(root/'normalization_validation.json')],min(180,left()),chosen)
        run('joint_and_terminal_R',[sys.executable,'-m','recurrent_rg','train','--config',str(root/'training_config.json'),
            '--output',str(root/'run'),'--device','cuda:0'],left(),chosen)
        run('report',[sys.executable,'-m','scripts.report_phi4_ru_vae','--run',str(root/'run')],left())
        run('band_comparison',[sys.executable,'-m','scripts.diagnose_phi4_ru_band','--data',str(root/'data'),
            '--run',str(root/'run'),'--reference',plan['reference_run']],left())
        report=json.loads((root/'run/report/report.json').read_text())
        if report['status']!='complete':raise RuntimeError('Incomplete report')
        public=[]
        for p in root.rglob('*'):
            if p.is_file() and p.suffix in ('.json','.png','.npz') and p.name!='data.npz' and p.name!='job.json':
                public.append(dict(path=str(p.relative_to(root)),sha256=sha256(p)))
        status(status='complete',workflow_complete=True,current=None,finished_unix=time.time(),
            returncode=0,total_cpu_seconds=cpu_used(),gpu_pipeline_wall_seconds=time.monotonic()-active,
            public_files=public)
    except Exception as exc:
        status(status='budget_hold' if isinstance(exc,TimeoutError) else 'failed',error=str(exc),
            requires_user_action=True,traceback=traceback.format_exc(),returncode=1)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--stage',choices=['preflight','sampling']);a=p.parse_args()
    if a.stage:
        plan=json.loads(a.plan.read_text());verify(plan)
        (preflight if a.stage=='preflight' else sampling)(plan,Path(plan['output']))
    else:execute(a.plan)
