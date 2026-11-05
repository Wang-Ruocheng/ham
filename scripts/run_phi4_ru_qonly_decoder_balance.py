"""Bounded 20-minute three-arm Q-only diagnostic, followed by paired R readouts.

Uses the completed ru saddle-window experiment read-only. No new physical
sampling, C optimization, default changes, root search or exponent selection.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import argparse
import copy
import json
from pathlib import Path
import subprocess
import signal
import sys
import time
import traceback

import numpy as np
import torch

from experiments import phi4_hopping_qcr as base
from experiments.phi4_qcr import load_dataset
from experiments import phi4_ru_qonly_decoder_balance as core
from recurrent_rg.phi4_cnn_qcr import CNNPhi4QCR
from recurrent_rg.phi4_aligned_qcr import soft_plaquette_features

SOURCE = Path('/mnt/nas-new/home/yange/ham/results/phi4_ru_cnn_saddle_seed91_gpu3_vm225_20261103_v1')
OUTPUT = Path('/mnt/nas-new/home/yange/ham/results/phi4_ru_qonly_decoder_balance_seed91_gpu3_vm225_20261103_v1')
UUID = 'GPU-a167decd-ce49-6c8f-6bcf-136fde596683'
PINS = {'run/training_step_6000.pt':'38f0e374eca493478fb8c793beee7ef6838c4163f66fbd1a1af0ab29c7de7c8c',
        'run/summary.json':'9a1f93ae6e28d3720b9e79b0f9cd8fbdb3f07120f058bc55e9053db7ca8b49cf',
        'data/manifest.json':'586f9865c6afc206bf8c8a712ccde5cf84371775f3b336099e07e9fbe88db205'}


class BudgetHold(RuntimeError):
    pass


def admission():
    mem = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
    io = float(next(x.split('avg10=')[1].split()[0] for x in Path('/proc/pressure/io').read_text().splitlines() if x.startswith('full ')))
    gpu = subprocess.check_output(['nvidia-smi','-i','3','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True,timeout=10).strip().split(',')
    processes = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True,timeout=10)
    if gpu[0].strip()!=UUID:
        raise RuntimeError('GPU identity mismatch')
    receipt = dict(mem_available_bytes=mem,io_full_avg10=io,load1=os.getloadavg()[0],
                   gpu_uuid=UUID,free_gpu_mib=int(gpu[1]),compute_present=UUID in processes)
    if mem<36*1024**3 or io>=20 or os.getloadavg()[0]>1.25*os.cpu_count() or int(gpu[1])<12288 or UUID in processes:
        raise BudgetHold('Resources not admitted; no shared GPU start')
    return receipt


@torch.no_grad()
def evaluate_q(q, levels, per_field, device, check):
    rows = {k:[] for k in ('g','g_full','first','second','decoded')}
    for start in range(0,len(per_field),4):
        ga,gb,gf = [],[],[]
        for x in levels:
            batch = x[start:start+4].to(device)
            ga.append(q(batch[:,:32]));gb.append(q(batch[:,32:]));gf.append(q(batch))
        a,b,f = [torch.stack(x,1) for x in (ga,gb,gf)]
        g = .5*(a+b)
        for key,value in dict(g=g,g_full=f,first=a,second=b,decoded=q.decode(g)).items():
            rows[key].append(value.cpu())
        check()
    return {k:torch.cat(v) for k,v in rows.items()}


def clocks(optimizer, expected):
    steps = [int(v['step']) for v in optimizer.state.values()]
    if not steps or min(steps)!=expected or max(steps)!=expected:
        raise RuntimeError('Adam clock mismatch')
    return dict(min=min(steps),max=max(steps),parameter_tensors=len(steps))


def run(output):
    output = Path(output).resolve()
    if output!=OUTPUT or output.exists():
        raise ValueError('Exact fresh independent output required')
    if os.uname().nodename!='thinklab-105-225' or os.getpriority(os.PRIO_PROCESS,0)<10:
        raise RuntimeError('Wrong host/priority')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=UUID or torch.cuda.is_initialized():
        raise RuntimeError('GPU must be pinned before initialization')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    output.mkdir(parents=True)
    start = time.monotonic()
    job = dict(status='running',workflow_complete=False,current='admission',pid=os.getpid(),
               returncode=None,error=None,requires_user_action=False,actual_Q_steps={},actual_R_steps={},C_updates=0)
    def status(**kw):
        job.update(kw,elapsed_seconds=time.monotonic()-start,cpu_seconds=time.process_time())
        base.write_json(output/'job.json',job);base.write_json(output/'status.json',job)
        print(json.dumps(job,ensure_ascii=False,allow_nan=False),flush=True)
    def check(reserve=0):
        if time.monotonic()-start+reserve>1180:
            raise BudgetHold('20-minute pipeline budget; no automatic shortened endpoint or restart')
        if torch.cuda.is_initialized() and max(torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved())>8*1024**3:
            raise BudgetHold('Own GPU memory exceeds 8GiB')
    status()
    try:
        resources = [admission()];time.sleep(10);resources.append(admission())
        for path,pin in PINS.items():
            if base.sha256_file(SOURCE/path)!=pin:
                raise RuntimeError('Source pin mismatch: '+path)
        code = Path(__file__).resolve().parents[1]
        code_pins = {str(p.relative_to(code)):base.sha256_file(p)
                     for folder in ('experiments','scripts','recurrent_rg') for p in (code/folder).rglob('*.py')}
        device = torch.device('cuda:0')
        torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(device).total_memory)
        torch.cuda.reset_peak_memory_stats()
        status(current='loading_verified_source_and_data')
        checkpoint = torch.load(SOURCE/'run/training_step_6000.pt',map_location='cpu',weights_only=False)
        source = CNNPhi4QCR(checkpoint['manifest']['scale'],seed=91,theory_dim=2)
        source.load_state_dict(checkpoint['model'],strict=True)
        source.requires_grad_(False)
        source_hash = base.state_fingerprint(source)
        summary = json.loads((SOURCE/'run/summary.json').read_text())
        if source_hash!=summary['final_model_sha256']:
            raise RuntimeError('Loaded model fingerprint mismatch')
        dataset = load_dataset(SOURCE/'data')
        if dataset.configurations.shape!=(180,128,36,36,1):
            raise ValueError('Unexpected source geometry')
        train = dataset.train_indices
        plan = dict(schema_version=5,experiment='phi4_ru_frozenC_Q_decoder_balance_v1',
            source=str(SOURCE),source_files=PINS,source_QCR_steps=6000,source_model_sha256=source_hash,
            code_sha256=code_pins,arms=list(core.ARMS),Q_steps=core.Q_STEPS,R_steps=core.R_STEPS,
            Q_learning_rate=.001,R_learning_rate=.003,Q_start='identical source6000; fresh Adam',
            R_start='identical fresh residual 2-96-96-2; diagnostic readout, not source R continuation',
            C_frozen=True,C_updates=0,Q_updates_all_encoder_parameters=True,physical_sampling=False,
            theory_dim=2,sizes=[36,12,4],system_batch=8,sample_batch=32,
            cache_samples=64,cache_indices=list(range(0,128,2)),cache_C_draws_per_field=1,cache_seed_base=202611030400,
            train_chains=len(train),test_chains=len(dataset.test_indices),data_warning_count=dataset.warning_count,
            labels_used_in_training=False,group_ids_only_for_repeated_chain_noise=True,
            metric='TRAIN-only group-mean SVD, variance>5*repeat-chain mean noise; all resolved axes, max gain10; source TRAIN residual normalization separately per scale',
            closure_objective='all_start_anchor',closure_scale_weights=None,Q_weights=[10,20,5],
            no_default_changes=True,no_root_search=True,no_endpoint_selection=True,thermal_nu=None,
            equilibrium_certified=False,resources=resources,budget_seconds=1200,own_gpu_limit_bytes=8*1024**3,
            qualification='Single seed, frozen historical C, source-Q warm start, 64/128 saved fields; not fresh joint QCR or a physical fixed-point certificate')
        base.write_json(output/'plan.json',plan);base.write_json(output/'manifest.json',plan)
        status(current='frozen_C_collection')
        source.to(device)
        levels_all = [[] for _ in range(3)];targets = []
        with torch.no_grad():
            for i in range(180):
                rng = torch.Generator(device=device).manual_seed(202611030400+i)
                levels = source.coarse_grain(dataset.configurations[i:i+1,::2].to(device),generator=rng)
                ps = []
                for k,x in enumerate(levels):
                    levels_all[k].append(x.cpu())
                    side = x.shape[2]
                    per = soft_plaquette_features(x.reshape(64,1,side,side,1),source.scale,chunk_size=262144)
                    ps.append(per.cpu())
                targets.append(torch.stack(ps))
                if i%30==29:status(current='frozen_C_collection',collected_chains=i+1)
                check(120)
        levels = [torch.cat(x) for x in levels_all];targets = torch.stack(targets)
        source.cpu()
        if base.state_fingerprint(source)!=source_hash or any(p.grad is not None for p in source.parameters()):
            raise RuntimeError('Frozen source changed')
        torch.save(dict(levels=levels,per_field_targets=targets),output/'frozen_cache.pt')
        del dataset.configurations
        arms = core.make_arms(source.Q)
        for q in arms.values():q.requires_grad_(True)
        initial = evaluate_q(arms['A_affine'].to(device),levels,targets,device,check)
        arms['A_affine'].cpu()
        p = targets.mean(2)
        metrics = core.fit_metrics(p[train],dataset.cell_ids[train],initial['decoded'][train])
        base.write_json(output/'metric.json',[{k:v.tolist() for k,v in m.items() if k!='axes'} for m in metrics])
        np.savez_compressed(output/'metric_arrays.npz',**{f'{i}_{k}':v.numpy() for i,m in enumerate(metrics) for k,v in m.items()})
        # Disposable pilot; measured before formal optimization, never selected by loss.
        status(current='disposable_throughput_pilot')
        pilot = copy.deepcopy(arms['C_balanced']).to(device)
        opt = torch.optim.Adam(pilot.parameters(),lr=.001)
        batch = [x[train[:8],:32].to(device) for x in levels];pb=targets[train[:8],:,:32].to(device)
        times=[]
        for _ in range(8):
            torch.cuda.synchronize();tick=time.monotonic()
            core.update(pilot,opt,batch,pb,metrics)
            torch.cuda.synchronize();times.append(time.monotonic()-tick);check()
        projection=time.monotonic()-start+1.6*max(times[2:])*core.Q_STEPS*3+180
        timing=dict(step_seconds=times,conservative_total_seconds=projection,budget_seconds=1200,
                    pilot_discarded=True,formal_steps_per_arm=core.Q_STEPS,passed=projection<1140)
        base.write_json(output/'timing.json',timing)
        del pilot,opt,batch,pb;torch.cuda.empty_cache()
        if not timing['passed']:
            raise BudgetHold('Fixed three-arm endpoint does not fit conservative 20-minute projection')
        values = {'source_'+k:v.numpy() for k,v in initial.items()}
        values.update(target=p.numpy(),splits=dataset.splits,cell_ids=dataset.cell_ids,labels=dataset.labels)
        histories={};arm_summaries={}
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(9104);initial_r=core.RReadout()
        for name,q in arms.items():
            q.to(device);optimizer=torch.optim.Adam(q.parameters(),lr=.001)
            rng=torch.Generator().manual_seed(9105);history=[]
            active_metrics=metrics if name=='C_balanced' else None
            status(current='Q_training',arm=name)
            for step in range(1,core.Q_STEPS+1):
                ids=torch.as_tensor(train)[torch.randperm(len(train),generator=rng)[:8]]
                sample=torch.randperm(64,generator=rng)[:32]
                b=[x[ids][:,sample].to(device) for x in levels]
                pt=targets[ids][:,:,sample].to(device)
                row=core.update(q,optimizer,b,pt,active_metrics)
                job['actual_Q_steps'][name]=step
                if step==1 or step%50==0:
                    history.append(dict(step=step,**row));base.write_json(output/(name+'_history.json'),history)
                    status(current='Q_training',arm=name,last_loss=row['q_total'])
                check(90)
            q_clock=clocks(optimizer,core.Q_STEPS)
            evaluation=evaluate_q(q,levels,targets,device,check)
            values.update({name+'_'+k:v.numpy() for k,v in evaluation.items()})
            q.cpu();histories[name]=history
            status(current='R_readout',arm=name)
            r=copy.deepcopy(initial_r).to(device);r_opt=torch.optim.Adam(r.parameters(),lr=.003)
            g=evaluation['g'].to(device);rng=torch.Generator().manual_seed(9106);r_history=[]
            for step in range(1,core.R_STEPS+1):
                ids=torch.as_tensor(train)[torch.randperm(len(train),generator=rng)[:32]].to(device)
                r_opt.zero_grad(set_to_none=True);loss=core.r_objective(r,g[ids])
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite R objective')
                loss.backward()
                if any(x.grad is None or not torch.isfinite(x.grad).all() for x in r.parameters()):
                    raise FloatingPointError('Invalid R gradients')
                r_opt.step();job['actual_R_steps'][name]=step
                if step==1 or step%100==0:r_history.append(dict(step=step,loss=float(loss.detach())))
                if step%100==0:check(45)
            r_clock=clocks(r_opt,core.R_STEPS);r_opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                one=r.advance_theory(g[:,:2]);roll=r.advance_theory(one[:,0])
                inp=g[train,:2].reshape(-1,2)
                lo,hi=inp.amin(0),inp.amax(0);pad=(hi-lo).clamp_min(.01)*.1
                xs=[torch.linspace(float(lo[d]-pad[d]),float(hi[d]+pad[d]),21,device=device,dtype=torch.double) for d in range(2)]
                grid=torch.stack(torch.meshgrid(*xs,indexing='ij'),-1).reshape(-1,2)
                for key,val in dict(one_step=one,rollout=roll,flow_grid=grid,flow_displacement=r.advance_theory(grid)-grid).items():
                    values[name+'_'+key]=val.cpu().numpy()
                values[name+'_one_step_decoded']=q.decode(one.cpu()).numpy()
                values[name+'_rollout_decoded']=q.decode(roll.cpu()).numpy()
            arm_summaries[name]=dict(Q_steps=core.Q_STEPS,R_steps=core.R_STEPS,Q_Adam=q_clock,R_Adam=r_clock,
                decoder_bounds=q.decoder.bounds() if isinstance(q.decoder,core.ResidualDecoder) else dict(fixed_upper_bound=2**.5),
                Q_sha256=base.state_fingerprint(q),R_sha256=base.state_fingerprint(r))
            torch.save(dict(Q=q.state_dict(),R=r.cpu().state_dict(),source_sha256=source_hash,manifest=plan),output/(name+'_final.pt'))
            base.write_json(output/(name+'_R_history.json'),r_history)
            status(current='arm_complete',arm=name)
            del optimizer,r_opt,r,g,b,pt;torch.cuda.empty_cache()
        if base.state_fingerprint(source)!=source_hash or any(x.grad is not None for x in source.parameters()):
            raise RuntimeError('Source was mutated')
        for path,pin in PINS.items():
            if base.sha256_file(SOURCE/path)!=pin:raise RuntimeError('Source file changed')
        for path,pin in code_pins.items():
            if base.sha256_file(code/path)!=pin:raise RuntimeError('Frozen implementation changed')
        np.savez_compressed(output/'evaluation_arrays.npz',**values)
        base.write_json(output/'summary.json',dict(status='complete',arms=arm_summaries,
            source_unchanged=True,C_updates=0,labels_used_in_training=False,thermal_nu=None,
            elapsed_seconds=time.monotonic()-start,peak_cuda_bytes=max(torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved())))
        status(current='report')
        from scripts.report_phi4_ru_qonly_decoder_balance import render
        render(output);check()
        public={str(p.relative_to(output)):base.sha256_file(p) for p in output.rglob('*')
                if p.is_file() and p.suffix in ('.json','.npz','.png') and p.name not in ('job.json','status.json')}
        status(status='complete',workflow_complete=True,current='complete',returncode=0,public_files=public)
    except Exception as exc:
        status(status='budget_hold' if isinstance(exc,BudgetHold) else 'failed',current=job['current'],
               error=str(exc),traceback=traceback.format_exc(),requires_user_action=True,returncode=2)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--supervise',action='store_true')
    args=parser.parse_args()
    if not args.supervise:
        run(args.output)
    else:
        # The hard guard applies only to this newly spawned, owned process group.
        started=time.time();child=subprocess.Popen([sys.executable,'-m','scripts.run_phi4_ru_qonly_decoder_balance','--output',args.output],start_new_session=True)
        deadline=False
        try:rc=child.wait(timeout=1200)
        except subprocess.TimeoutExpired:
            deadline=True;os.killpg(child.pid,signal.SIGTERM)
            try:rc=child.wait(timeout=8)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);rc=child.wait()
        base.write_json(Path('/mnt/nas-new/home/yange/ham/logs')/(OUTPUT.name+'.execution.json'),
            dict(pid=os.getpid(),child_pid=child.pid,returncode=rc,hard_deadline_reached=deadline,
                 started_unix=started,finished_unix=time.time(),elapsed_seconds=time.time()-started))
        sys.exit(rc if rc>=0 else 1)
