"""20-minute paired Q-scale correction and three frozen-Q R readouts.

Reuses only the completed local-window source; no resampling or default edits.
Run --prepare in the fresh frozen snapshot, then --supervise exactly once.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse
import copy
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from experiments import phi4_hopping_qcr as base
from experiments.phi4_qcr import load_dataset
from experiments.phi4_ru_cnn_nonlinear import install_decoder
from experiments import phi4_local_scale_normalization as core
from experiments.phi4_ru_qonly_decoder_balance import RReadout, r_objective
from recurrent_rg.phi4_cnn_qcr import CNNPhi4QCR
from recurrent_rg.phi4_aligned_qcr import soft_plaquette_features
from scripts.run_phi4_ru_qonly_decoder_balance import evaluate_q, clocks

MODULE='scripts.run_phi4_local_scale_normalization'
SOURCE=Path('/mnt/nas-new/home/yange/ham/results/phi4_ru_cnn_local_window_seed91_gpu2_vm225_20261103_v1')
OUTPUT=Path('/mnt/nas-new/home/yange/ham/results/phi4_local_scale_normalization_seed91_gpu2_vm225_20261103_v1')
UUID='GPU-713b07d0-118c-f939-c0a8-565390215e01'
PINS={'run/training_step_6000.pt':'c6672e77b906e243988e05a1ce199194e26a50cc7aff36ecd39e88506e035e0b',
      'run/summary.json':'06dac23dbe0780dc2a1916b1ada3ec25e8349697cbb3c146c0698e72d7e35bf5',
      'run/manifest.json':'e46f0c60132709463a79b7f1e16354d2d82df3d2a73368fb93b6a948c3c23291',
      'data/manifest.json':'75ca5608744ca37dc3234fa0cfa0d5f59223ec0b8aaf90efb90c44df6e66b841'}


class BudgetHold(RuntimeError):pass


def resources():
    mem=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
    io=float(next(x.split('avg10=')[1].split()[0] for x in Path('/proc/pressure/io').read_text().splitlines() if x.startswith('full ')))
    gpu=subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True,timeout=10).strip().split(',')
    compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True,timeout=10)
    if gpu[0].strip()!=UUID:raise RuntimeError('Physical GPU2 identity mismatch')
    row=dict(mem_available_bytes=mem,io_full_avg10=io,load1=os.getloadavg()[0],cpu_count=os.cpu_count(),
             gpu_uuid=UUID,free_gpu_mib=int(gpu[1]),compute_present=UUID in compute)
    row['ready']=(mem>=36*1024**3 and io<20 and row['load1']<=1.25*row['cpu_count']
                  and int(gpu[1])>=12288 and UUID not in compute)
    return row


def verify_source():
    for name,pin in PINS.items():
        if base.sha256_file(SOURCE/name)!=pin:raise RuntimeError('Source changed: '+name)


def code_hashes():
    code=Path(__file__).resolve().parents[1]
    return {str(p.relative_to(code)):base.sha256_file(p) for folder in ('experiments','scripts','recurrent_rg')
            for p in (code/folder).rglob('*.py')}


def prepare():
    if OUTPUT.exists():raise ValueError('Fresh independent output required')
    verify_source()
    plan=dict(schema_version=5,experiment='phi4_local_frozenC_Q_scale_normalization_v1',source=str(SOURCE),
        source_files=PINS,source_QCR_steps=6000,code_root=str(Path(__file__).resolve().parents[1]),
        code_sha256=code_hashes(),arms=list(core.ARMS),R_readouts=list(core.READOUTS),
        Q_steps_per_arm=core.Q_STEPS,R_steps_per_readout=core.R_STEPS,Q_learning_rate=.001,R_learning_rate=.003,
        Q_start='identical source6000 nonlinear CNN Q; fresh Adam',R_start='identical fresh residual network; seed9104',
        C_frozen=True,C_updates=0,physical_sampling=False,theory_dim=2,sizes=[36,12,4],block_size=3,
        cache_samples=64,cache_indices=list(range(0,128,2)),cache_C_draws_per_field=1,cache_seed_base=202611030700,
        Q_system_batch=8,Q_sample_batch=32,R_system_batch=32,Q_batch_seed=9105,R_batch_seed=9106,
        scale_metric='TRAIN ensemble descriptor total variance per scale; inverse variance; one global source-loss matching multiplier',
        scale_metric_frozen_once=True,Q_subset_weights=[20,5],direction_whitening=False,symmetry_augmentation=False,
        closure_objective='all_start_anchor',closure_scale_weights=None,R_combined_weight=10,
        labels_used_in_training=False,TEST_used_for_selection=False,no_root_search=True,thermal_nu=None,
        budget_seconds=1200,own_gpu_limit_bytes=8*1024**3,directory_limit_bytes=2*1024**3,
        threads=1,nice=10,gpu_uuid=UUID,gpu_index=2,default_recipe_changed=False,
        qualification='Single seed, source warm start, frozen learned C, fixed single C trajectory, no fresh physical sampling; not a full joint-QCR or fixed-point certification')
    OUTPUT.mkdir(parents=True)
    base.write_json(OUTPUT/'plan.json',plan)
    base.write_json(OUTPUT/'job.json',dict(status='prepared',workflow_complete=False,current='prepared',
        plan_sha256=base.sha256_file(OUTPUT/'plan.json'),error=None,requires_user_action=False))
    print(json.dumps(dict(output=str(OUTPUT),plan_sha256=base.sha256_file(OUTPUT/'plan.json'))))


def run():
    if os.uname().nodename!='thinklab-105-225' or os.getpriority(os.PRIO_PROCESS,0)<10:
        raise RuntimeError('Unexpected host or priority')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=UUID or torch.cuda.is_initialized():
        raise RuntimeError('GPU must be pinned before initialization')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    job=json.loads((OUTPUT/'job.json').read_text());plan=json.loads((OUTPUT/'plan.json').read_text())
    if job['status']!='prepared' or base.sha256_file(OUTPUT/'plan.json')!=job['plan_sha256']:
        raise RuntimeError('Prepared plan/status mismatch; no resume')
    if code_hashes()!=plan['code_sha256']:raise RuntimeError('Frozen code changed')
    started=time.monotonic()
    job.update(status='running',pid=os.getpid(),returncode=None,actual_Q_steps={},actual_R_steps={},C_updates=0)
    def status(**fields):
        job.update(fields,elapsed_seconds=time.monotonic()-started,cpu_seconds=time.process_time())
        base.write_json(OUTPUT/'job.json',job);base.write_json(OUTPUT/'status.json',job)
        print(json.dumps({k:job.get(k) for k in ('status','current','arm','elapsed_seconds','actual_Q_steps','actual_R_steps','error')},allow_nan=False),flush=True)
    def check(reserve=0):
        if time.monotonic()-started+reserve>1180:raise BudgetHold('20-minute budget; fixed endpoint not shortened')
        if torch.cuda.is_initialized() and max(torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved())>8*1024**3:
            raise BudgetHold('Own CUDA memory exceeded 8GiB')
    try:
        status(current='resource_admission')
        checks=[resources()];time.sleep(10);checks.append(resources())
        base.write_json(OUTPUT/'admission.json',checks)
        if not all(r['ready'] for r in checks):raise BudgetHold('GPU/host not idle enough; no shared launch')
        verify_source()
        device=torch.device('cuda:0')
        torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(device).total_memory)
        torch.cuda.reset_peak_memory_stats()
        status(current='loading_source')
        checkpoint=torch.load(SOURCE/'run/training_step_6000.pt',map_location='cpu',weights_only=False)
        source=install_decoder(CNNPhi4QCR(checkpoint['manifest']['scale'],seed=91,theory_dim=2))
        source.load_state_dict(checkpoint['model'],strict=True);source.requires_grad_(False);source.eval()
        source_hash=base.state_fingerprint(source)
        if source_hash!=json.loads((SOURCE/'run/summary.json').read_text())['final_model_sha256']:
            raise RuntimeError('Loaded source fingerprint mismatch')
        dataset=load_dataset(SOURCE/'data')
        if dataset.configurations.shape!=(180,128,36,36,1):raise ValueError('Unexpected source data')
        train=dataset.train_indices
        manifest=dict(plan,source_model_sha256=source_hash,data_warning_count=dataset.warning_count,
                      train_chains=len(train),test_chains=len(dataset.test_indices),equilibrium_certified=False)
        base.write_json(OUTPUT/'manifest.json',manifest)
        status(current='frozen_C_collection')
        source.to(device);level_rows=[[],[],[]];target_rows=[]
        with torch.no_grad():
            for i in range(180):
                rng=torch.Generator(device=device).manual_seed(plan['cache_seed_base']+i)
                row=source.coarse_grain(dataset.configurations[i:i+1,::2].to(device),generator=rng)
                features=[]
                for k,x in enumerate(row):
                    level_rows[k].append(x.cpu());side=x.shape[2]
                    features.append(soft_plaquette_features(x.reshape(64,1,side,side,1),source.scale,chunk_size=262144).cpu())
                target_rows.append(torch.stack(features))
                if i%30==29:status(collected_chains=i+1)
                check(150)
        levels=[torch.cat(rows) for rows in level_rows];targets=torch.stack(target_rows);p=targets.mean(2)
        source.cpu();del dataset.configurations,level_rows,target_rows,checkpoint
        if base.state_fingerprint(source)!=source_hash or any(x.grad is not None for x in source.parameters()):raise RuntimeError('Frozen source changed')
        torch.save(dict(levels=levels,per_field_targets=targets),OUTPUT/'frozen_cache.pt')
        initial=evaluate_q(source.Q.to(device),levels,targets,device,check);source.Q.cpu()
        metric=core.fit_scale_weights(p[train],initial['decoded'][train])
        base.write_json(OUTPUT/'scale_metric.json',{k:v.tolist() for k,v in metric.items()})
        # No optimizer selected by loss; this copy is discarded after timing.
        status(current='disposable_throughput_pilot')
        pilot=copy.deepcopy(source.Q).requires_grad_(True).to(device)
        optimizer=torch.optim.Adam(pilot.parameters(),lr=.001)
        batch=[x[train[:8],:32].to(device) for x in levels];pt=targets[train[:8],:,:32].to(device);times=[]
        for _ in range(8):
            torch.cuda.synchronize();tick=time.monotonic()
            core.update(pilot,optimizer,batch,pt,metric['weights'])
            torch.cuda.synchronize();times.append(time.monotonic()-tick);check()
        projection=time.monotonic()-started+1.8*max(times[2:])*core.Q_STEPS*2+210
        timing=dict(step_seconds=times,projected_total_seconds=projection,budget_seconds=1200,passed=projection<1140,pilot_discarded=True)
        base.write_json(OUTPUT/'timing.json',timing)
        del pilot,optimizer,batch,pt;torch.cuda.empty_cache()
        if not timing['passed']:raise BudgetHold('Fixed paired endpoints fail conservative runtime projection')
        values={'source_'+k:v.numpy() for k,v in initial.items()}
        values.update(target=p.numpy(),splits=dataset.splits,cell_ids=dataset.cell_ids,labels=dataset.labels)
        evaluations={'source':initial};models={'source':copy.deepcopy(source.Q)};summaries={}
        for name in core.ARMS:
            q=copy.deepcopy(source.Q).requires_grad_(True).to(device)
            initial_hash=base.state_fingerprint(q)
            if initial_hash!=base.state_fingerprint(source.Q):raise RuntimeError('Unpaired Q initialization')
            optimizer=torch.optim.Adam(q.parameters(),lr=.001);rng=torch.Generator().manual_seed(9105)
            weights=torch.ones(3,dtype=torch.double) if name=='A_original' else metric['weights']
            history=[];status(current='Q_training',arm=name)
            for step in range(1,core.Q_STEPS+1):
                ids=torch.as_tensor(train)[torch.randperm(len(train),generator=rng)[:8]]
                samples=torch.randperm(64,generator=rng)[:32]
                b=[x[ids][:,samples].to(device) for x in levels];pt=targets[ids][:,:,samples].to(device)
                row=core.update(q,optimizer,b,pt,weights);job['actual_Q_steps'][name]=step
                if step==1 or step%50==0:
                    history.append(dict(step=step,**row));base.write_json(OUTPUT/(name+'_Q_history.json'),history);status(last_q_loss=row['q_total'])
                check(150)
            q_clock=clocks(optimizer,core.Q_STEPS);q.requires_grad_(False)
            evaluation=evaluate_q(q,levels,targets,device,check)
            values.update({name+'_'+k:v.numpy() for k,v in evaluation.items()})
            q.cpu();models[name]=q;evaluations[name]=evaluation
            summaries[name]=dict(Q_steps=core.Q_STEPS,Q_Adam=q_clock,Q_initial_sha256=initial_hash,Q_sha256=base.state_fingerprint(q))
            del optimizer,b,pt;torch.cuda.empty_cache();status(current='Q_arm_complete',arm=name)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(9104);initial_r=RReadout()
        def save_r_evaluation(name,r,q,g):
            with torch.no_grad():
                one=r.advance_theory(g[:,:2]);roll=r.advance_theory(one[:,0])
                inp=g[train,:2].reshape(-1,2);lo,hi=inp.amin(0),inp.amax(0);pad=(hi-lo).clamp_min(.01)*.1
                axes=[torch.linspace(float(lo[d]-pad[d]),float(hi[d]+pad[d]),21,device=device,dtype=torch.double) for d in range(2)]
                grid=torch.stack(torch.meshgrid(*axes,indexing='ij'),-1).reshape(-1,2)
                for key,val in dict(one_step=one,rollout=roll,flow_grid=grid,flow_displacement=r.advance_theory(grid)-grid).items():values[name+'_'+key]=val.cpu().numpy()
                values[name+'_one_step_decoded']=q.decode(one.cpu()).numpy()
                values[name+'_rollout_decoded']=q.decode(roll.cpu()).numpy()
        original=RReadout().to(device);original.net.load_state_dict(source.R.state_dict())
        save_r_evaluation('original',original,source.Q,initial['g'].to(device));del original
        for name in core.READOUTS:
            q=models[name];q_hash=base.state_fingerprint(q)
            r=copy.deepcopy(initial_r).to(device);r_initial_hash=base.state_fingerprint(r)
            optimizer=torch.optim.Adam(r.parameters(),lr=.003);g=evaluations[name]['g'].to(device)
            rng=torch.Generator().manual_seed(9106);history=[];status(current='R_readout',arm=name)
            for step in range(1,core.R_STEPS+1):
                ids=torch.as_tensor(train)[torch.randperm(len(train),generator=rng)[:32]].to(device)
                optimizer.zero_grad(set_to_none=True);loss=r_objective(r,g[ids])
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite R loss')
                loss.backward()
                if any(x.grad is None or not torch.isfinite(x.grad).all() for x in r.parameters()):raise FloatingPointError('Invalid R gradient')
                optimizer.step();job['actual_R_steps'][name]=step
                if step==1 or step%100==0:
                    history.append(dict(step=step,loss=float(loss.detach())));base.write_json(OUTPUT/(name+'_R_history.json'),history);status(last_r_loss=float(loss.detach()))
                    check(45)
            r_clock=clocks(optimizer,core.R_STEPS);optimizer.zero_grad(set_to_none=True)
            save_r_evaluation(name,r,q,g)
            if base.state_fingerprint(q)!=q_hash or any(x.grad is not None for x in q.parameters()):raise RuntimeError('Frozen Q changed during R readout')
            summaries.setdefault(name,dict(Q_steps=0,Q_sha256=q_hash)).update(R_steps=core.R_STEPS,R_Adam=r_clock,
                R_initial_sha256=r_initial_hash,R_sha256=base.state_fingerprint(r),Q_frozen_verified=True,
                decoder_bounds=q.decoder.bounds())
            torch.save(dict(Q=q.state_dict(),R=r.cpu().state_dict(),source_sha256=source_hash,manifest=manifest),OUTPUT/(name+'_final.pt'))
            del optimizer,r,g;torch.cuda.empty_cache();status(current='arm_complete',arm=name)
        if base.state_fingerprint(source)!=source_hash or any(x.grad is not None for x in source.parameters()):raise RuntimeError('Source changed')
        verify_source()
        if code_hashes()!=plan['code_sha256']:raise RuntimeError('Frozen implementation changed')
        np.savez_compressed(OUTPUT/'evaluation_arrays.npz',**values)
        summary=dict(status='complete',arms=summaries,source_unchanged=True,C_updates=0,labels_used_in_training=False,
            thermal_nu=None,data_warning_count=dataset.warning_count,elapsed_seconds=time.monotonic()-started,
            peak_cuda_bytes=max(torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved()))
        base.write_json(OUTPUT/'summary.json',summary)
        status(current='report')
        from scripts.report_phi4_local_scale_normalization import render
        render(OUTPUT);check()
        size=sum(p.stat().st_size for p in OUTPUT.rglob('*') if p.is_file())
        if size>plan['directory_limit_bytes']:raise BudgetHold('Directory limit exceeded')
        public={str(p.relative_to(OUTPUT)):base.sha256_file(p) for p in OUTPUT.rglob('*')
                if p.is_file() and p.suffix in ('.json','.npz','.png') and p.name not in ('job.json','status.json')}
        status(status='complete',workflow_complete=True,current='complete',returncode=0,public_files=public,directory_bytes=size)
    except Exception as exc:
        status(status='budget_hold' if isinstance(exc,BudgetHold) else 'failed',error=str(exc),
               traceback=traceback.format_exc(),requires_user_action=True,returncode=2)
        raise


def supervise():
    with (OUTPUT/'execution.lock').open('x') as stream:stream.write(str(os.getpid()))
    started=time.time()
    child=subprocess.Popen([sys.executable,'-m',MODULE],start_new_session=True)
    base.write_json(OUTPUT/'launch.json',dict(parent_pid=os.getpid(),child_pid=child.pid,started_unix=started,
        host=os.uname().nodename,gpu_uuid=UUID,plan_sha256=base.sha256_file(OUTPUT/'plan.json'),budget_seconds=1200))
    exceeded=False
    try:rc=child.wait(timeout=1200)
    except subprocess.TimeoutExpired:
        exceeded=True;os.killpg(child.pid,signal.SIGTERM)
        try:rc=child.wait(timeout=8)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);rc=child.wait()
    base.write_json(OUTPUT/'execution.json',dict(parent_pid=os.getpid(),child_pid=child.pid,returncode=rc,
        hard_deadline_reached=exceeded,elapsed_seconds=time.time()-started))
    if exceeded:
        job=json.loads((OUTPUT/'job.json').read_text());job.update(status='budget_hold',workflow_complete=False,
            error='Owned experiment reached 1200-second hard deadline',requires_user_action=True,returncode=rc)
        base.write_json(OUTPUT/'job.json',job);base.write_json(OUTPUT/'status.json',job)
    sys.exit(rc if rc>=0 else 1)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--supervise',action='store_true')
    args=parser.parse_args()
    if args.prepare:prepare()
    elif args.supervise:supervise()
    else:run()
