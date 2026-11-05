"""Frozen-C, four-arm local-descriptor-difference experiment; 20 minute cap."""
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
from experiments import phi4_local_secant_q as core
from experiments.phi4_ru_cnn_nonlinear import install_decoder, JointNonlinearQ
from experiments.phi4_ru_qonly_decoder_balance import RReadout, r_objective
from recurrent_rg.phi4_cnn_qcr import CNNPhi4QCR, ConfigurationCNNQ
from scripts.run_phi4_local_scale_normalization import SOURCE, PINS, UUID, resources, verify_source, code_hashes, BudgetHold
from scripts.run_phi4_ru_qonly_decoder_balance import evaluate_q, clocks

MODULE='scripts.run_phi4_local_secant_q'
OUTPUT=Path('/mnt/nas-new/home/yange/ham/results/phi4_local_secant_q_seed91_gpu2_vm225_20261103_v1')
CACHE=Path('/mnt/nas-new/home/yange/ham/results/phi4_local_scale_normalization_seed91_gpu2_vm225_20261103_v1')
CACHE_PINS={'frozen_cache.pt':'b85cb0ab781fd35d19aa88f612ea2b6c441ff47ae0fc85e5a181a6dec031f361',
    'evaluation_arrays.npz':'ca48ecd7c028251ca074b4e28aa6ec11df02524c7c3de058acb0deb077aade8e',
    'plan.json':'ff49e3d16f272251e9062161d1df475a75254d8e0c1e39faf321cfea781059e5',
    'job.json':'e9ae643b5abf3004711171347a39449f5bc9019ee97ac7ae4025724dbc052f02'}


def verify_cache():
    for name,pin in CACHE_PINS.items():
        if base.sha256_file(CACHE/name)!=pin:raise RuntimeError('Frozen cache changed: '+name)
    job=json.loads((CACHE/'job.json').read_text())
    if not job['workflow_complete'] or job['returncode']!=0:raise RuntimeError('Incomplete cache source')


def prepare():
    if OUTPUT.exists():raise ValueError('Fresh output required; never resume this experiment')
    verify_source();verify_cache()
    plan=dict(schema_version=5,experiment='phi4_local_secant_Q_2x2_v1',source=str(SOURCE),source_files=PINS,
        cache=str(CACHE),cache_files=CACHE_PINS,code_root=str(Path(__file__).resolve().parents[1]),code_sha256=code_hashes(),
        arms=list(core.ARMS),source_QCR_steps=6000,C_updates=0,physical_sampling=False,additional_C_sampling=False,
        sizes=[36,12,4],block_size=3,theory_dim=2,Q_steps=600,R_steps_if_gate_passes=1000,
        Q_learning_rate=.001,R_learning_rate=.003,Q_batch=8,Q_sample_batch=32,local_pairs_per_step=4,
        cache_samples=64,cache_C_draws=1,seed_fresh_Q=9108,seed_batch=9105,seed_validation=9107,seed_R=9104,
        validation_cells=6,validation_from='random subset of original TRAIN condition IDs, seed9107; no coupling values',
        Q_weights=[10,20,5],Q_scale_weights=[1,1,1],Q_decoder='shared nonlinear historical architecture',
        local_metric='six descriptor-nearest TRAIN groups per anchor; local SVD; variance>5*repeat-chain-mean noise; gain cap10; no rank prescription',
        local_target='difference of frozen TRAIN group means; independent chains identify repeated ensembles, not labels',
        local_coefficient='min(1,0.25*source_initial_TRAIN_reconstruction/source_initial_local_loss); frozen same for warm/fresh',
        local_scales='cycle uniformly over k=0,1,2, no cross-scale distance matching',
        gate='internal validation only; >=6 resolved nonleading components; relative weak MSE at least10% lower than paired control; no scale NRMSE >10% worse; mean relative MSE not worse',
        R_policy='if either initialization pair passes, fit all four arms with identical fresh R1000; otherwise zero R updates',
        closure_objective='all_start_anchor',closure_scale_weights=None,R_weight=10,labels_used_in_training=False,
        TEST_used_for_selection=False,default_recipe_changed=False,no_root_search=True,thermal_nu=None,
        budget_seconds=1200,own_gpu_limit_bytes=8*1024**3,directory_limit_bytes=2*1024**3,
        threads=1,nice=10,gpu_uuid=UUID,gpu_index=2,
        qualification='Explicit frozen-C diagnostic; different inner TRAIN split from prior trial; internal validation excluded only from current updates, seen by historical source Q/C; fixed endpoints; not a fresh joint-QCR reproduction or rank certificate')
    OUTPUT.mkdir(parents=True);base.write_json(OUTPUT/'plan.json',plan)
    base.write_json(OUTPUT/'job.json',dict(status='prepared',workflow_complete=False,current='prepared',
        plan_sha256=base.sha256_file(OUTPUT/'plan.json'),error=None,requires_user_action=False))
    print(json.dumps(dict(output=str(OUTPUT),plan_sha256=base.sha256_file(OUTPUT/'plan.json'))))


def run():
    if os.uname().nodename!='thinklab-105-225' or os.getpriority(os.PRIO_PROCESS,0)<10:
        raise RuntimeError('Unexpected host/priority')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=UUID or torch.cuda.is_initialized():raise RuntimeError('GPU not pinned')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    job=json.loads((OUTPUT/'job.json').read_text());plan=json.loads((OUTPUT/'plan.json').read_text())
    if job['status']!='prepared' or base.sha256_file(OUTPUT/'plan.json')!=job['plan_sha256']:raise RuntimeError('No resume or plan changes')
    if code_hashes()!=plan['code_sha256']:raise RuntimeError('Frozen code mismatch')
    start=time.monotonic();job.update(pid=os.getpid(),returncode=None,actual_Q_steps={},actual_R_steps={})
    def status(**fields):
        job.update(fields,elapsed_seconds=time.monotonic()-start,cpu_seconds=time.process_time())
        base.write_json(OUTPUT/'job.json',job);base.write_json(OUTPUT/'status.json',job)
        print(json.dumps({k:job.get(k) for k in ('status','current','arm','elapsed_seconds','actual_Q_steps','actual_R_steps','error')},allow_nan=False),flush=True)
    def check(reserve=0):
        if time.monotonic()-start+reserve>1180:raise BudgetHold('Fixed endpoints exceed 20-minute budget')
        if torch.cuda.is_initialized() and max(torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved())>8*1024**3:
            raise BudgetHold('Own CUDA memory exceeded 8GiB')
    try:
        status(status='running',current='admission')
        checks=[resources()];time.sleep(10);checks.append(resources());base.write_json(OUTPUT/'admission.json',checks)
        if not all(r['ready'] for r in checks):raise BudgetHold('GPU/host resources not admitted; no shared start')
        verify_source();verify_cache()
        device=torch.device('cuda:0');torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(device).total_memory)
        torch.cuda.reset_peak_memory_stats();status(current='load_frozen_cache')
        ckpt=torch.load(SOURCE/'run/training_step_6000.pt',map_location='cpu',weights_only=False)
        source=install_decoder(CNNPhi4QCR(ckpt['manifest']['scale'],seed=91,theory_dim=2))
        source.load_state_dict(ckpt['model'],strict=True);source.requires_grad_(False);source.eval()
        source_hash=base.state_fingerprint(source)
        if source_hash!=json.loads((SOURCE/'run/summary.json').read_text())['final_model_sha256']:raise RuntimeError('Source fingerprint mismatch')
        cache=torch.load(CACHE/'frozen_cache.pt',map_location='cpu',weights_only=True)
        levels,targets=cache['levels'],cache['per_field_targets'];p=targets.mean(2)
        if targets.shape!=(180,3,64,625) or [tuple(x.shape) for x in levels]!=[(180,64,s,s,1) for s in (36,12,4)]:raise ValueError('Cache geometry mismatch')
        with np.load(CACHE/'evaluation_arrays.npz',allow_pickle=False) as z:
            cells=z['cell_ids'].copy();splits=z['splits'].copy()
            initial={key:torch.from_numpy(z['source_'+key].copy()) for key in ('g','g_full','first','second','decoded')}
            if not np.array_equal(p.numpy(),z['target']):raise RuntimeError('Cache descriptors disagree with completed public arrays')
        train_cells=np.unique(cells[splits=='train']);rng=np.random.default_rng(9107);val_cells=rng.permutation(train_cells)[:6]
        inner=splits.astype('<U10');inner[np.isin(cells,val_cells)]='validation'
        fit=np.flatnonzero(inner=='train');val=np.flatnonzero(inner=='validation')
        graph=core.fit_graph(p[fit],cells[fit]);group_rows=[torch.tensor(np.flatnonzero((cells==int(g))&(inner=='train'))) for g in graph['groups']]
        decoded_means=torch.stack([initial['decoded'][ids].mean(0) for ids in group_rows])
        source_local=float(core.initial_local_loss(decoded_means,graph));source_rec=float((initial['decoded'][fit]-p[fit]).square().sum(-1).mean())
        coefficient=min(1.,.25*source_rec/max(source_local,1e-30)) if source_local>0 else 0.
        counts=(graph['gains']>0).sum(-1)
        base.write_json(OUTPUT/'metric.json',dict(coefficient=coefficient,initial_source_local=source_local,
            initial_source_reconstruction=source_rec,resolved_axes_per_anchor=counts.tolist(),group_ids=graph['groups'].tolist(),
            neighbors=graph['neighbors'].tolist(),gains=graph['gains'].tolist(),variances=graph['variance'].tolist(),noise=graph['noise'].tolist()))
        torch.save(graph,OUTPUT/'train_graph.pt')
        manifest=dict(plan,source_model_sha256=source_hash,train_cells=sorted(set(cells[fit].tolist())),
            validation_cells=sorted(val_cells.tolist()),train_chains=len(fit),validation_chains=len(val),test_chains=int((splits=='test').sum()),
            data_warning_count=55,equilibrium_certified=False,coefficient=coefficient)
        base.write_json(OUTPUT/'manifest.json',manifest)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(9108);encoder=ConfigurationCNNQ(float(source.scale),theory_dim=2)
            with torch.no_grad():encoder.decoder.bias.copy_(p[fit].mean((0,1)))
            torch.manual_seed(9103);fresh=JointNonlinearQ(encoder)
        fresh.requires_grad_(False);fresh.eval()
        values={'source_'+key:v.numpy() for key,v in initial.items()}
        values.update(target=p.numpy(),cell_ids=cells,splits=splits,inner_splits=inner)
        fresh_eval=evaluate_q(fresh.to(device),levels,targets,device,check);fresh.cpu()
        values.update({'fresh_initial_'+key:v.numpy() for key,v in fresh_eval.items()})
        def batch(rng,step):
            ids=torch.as_tensor(fit)[torch.randperm(len(fit),generator=rng)[:8]];samples=torch.randperm(64,generator=rng)[:32]
            k=(step-1)%3;anchor=torch.randint(len(group_rows),(4,),generator=rng);neighbor_slot=torch.randint(6,(4,),generator=rng)
            neighbor=graph['neighbors'][k,anchor,neighbor_slot]
            chain_ids=[]
            for i,j in zip(anchor,neighbor):
                chain_ids.extend([group_rows[int(x)][torch.randint(len(group_rows[int(x)]),(1,),generator=rng).item()].item() for x in (i,j)])
            return ([x[ids][:,samples].to(device) for x in levels],targets[ids][:,:,samples].to(device),
                levels[k][chain_ids][:,samples].to(device),graph['means'][anchor,k]-graph['means'][neighbor,k],
                graph['axes'][k,anchor],graph['gains'][k,anchor])
        status(current='disposable_throughput_pilot')
        pilot=copy.deepcopy(source.Q).requires_grad_(True).to(device);opt=torch.optim.Adam(pilot.parameters(),lr=.001)
        times=[];rng=torch.Generator().manual_seed(9105)
        for step in range(1,10):
            b=batch(rng,step);torch.cuda.synchronize();tick=time.monotonic()
            core.update(pilot,opt,*b,coefficient);torch.cuda.synchronize();times.append(time.monotonic()-tick)
        projection=time.monotonic()-start+2*max(times[3:])*600*4+240
        timing=dict(step_seconds=times,conservative_seconds=projection,budget_seconds=1200,passed=projection<1140,pilot_discarded=True)
        base.write_json(OUTPUT/'timing.json',timing);del pilot,opt,b;torch.cuda.empty_cache()
        if not timing['passed']:raise BudgetHold('Fixed four-arm endpoint fails runtime projection')
        evaluations={};models={};summaries={};validation={}
        for name in core.ARMS:
            q=copy.deepcopy(source.Q if name.startswith('warm') else fresh).requires_grad_(True).to(device)
            initial_hash=base.state_fingerprint(q);opt=torch.optim.Adam(q.parameters(),lr=.001);rng=torch.Generator().manual_seed(9105)
            weight=coefficient if name.endswith('local') else 0.;history=[];status(current='Q_training',arm=name)
            for step in range(1,601):
                b=batch(rng,step);row=core.update(q,opt,*b,weight);job['actual_Q_steps'][name]=step
                if step==1 or step%50==0:
                    history.append(dict(step=step,**row));base.write_json(OUTPUT/(name+'_Q_history.json'),history);status(last_q_loss=row['q_total'])
                check(180)
            clock=clocks(opt,600);q.requires_grad_(False)
            evaluation=evaluate_q(q,levels,targets,device,check);q.cpu()
            values.update({name+'_'+key:v.numpy() for key,v in evaluation.items()})
            validation[name]=core.validation_metrics(evaluation['decoded'][val],p[val],cells[val],graph)
            models[name]=q;evaluations[name]=evaluation
            summaries[name]=dict(Q_steps=600,Q_Adam=clock,Q_initial_sha256=initial_hash,Q_sha256=base.state_fingerprint(q),R_steps=0)
            torch.save(dict(Q=q.state_dict(),manifest=manifest,arm=name),OUTPUT/(name+'_Q_final.pt'))
            del opt,b;torch.cuda.empty_cache();status(current='Q_arm_complete',arm=name)
        variance=p[fit].var(0,unbiased=False).mean(-1).tolist()
        gates={kind:core.gate(validation[kind+'_base'],validation[kind+'_local'],variance) for kind in ('warm','fresh')}
        do_r=any(g['passed'] for g in gates.values())
        base.write_json(OUTPUT/'gate.json',dict(validation=validation,pairs=gates,R_authorized_by_gate=do_r,
            TEST_used=False,physical_rank_certified=False))
        status(current='Q_validation_gate',R_gate_passed=do_r)
        # Only after the fixed Q endpoint and internal gate: expose labels for reporting.
        with np.load(CACHE/'evaluation_arrays.npz',allow_pickle=False) as z:
            values['labels']=z['labels'].copy()
            for key in z.files:
                if key.startswith('original_'):values[key]=z[key].copy()
        if do_r:
            with torch.random.fork_rng(devices=[]):torch.manual_seed(9104);initial_r=RReadout()
            for name in core.ARMS:
                q=models[name];qhash=base.state_fingerprint(q);r=copy.deepcopy(initial_r).to(device)
                r_initial=base.state_fingerprint(r);opt=torch.optim.Adam(r.parameters(),lr=.003)
                g=evaluations[name]['g'].to(device);rng=torch.Generator().manual_seed(9106);history=[];status(current='R_training',arm=name)
                for step in range(1,1001):
                    ids=torch.as_tensor(fit)[torch.randperm(len(fit),generator=rng)[:32]].to(device)
                    opt.zero_grad(set_to_none=True);loss=r_objective(r,g[ids])
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite R loss')
                    loss.backward()
                    if any(x.grad is None or not torch.isfinite(x.grad).all() for x in r.parameters()):raise FloatingPointError('Invalid R gradients')
                    opt.step();job['actual_R_steps'][name]=step
                    if step==1 or step%100==0:
                        history.append(dict(step=step,loss=float(loss.detach())));base.write_json(OUTPUT/(name+'_R_history.json'),history);status()
                    check(70)
                r_clock=clocks(opt,1000);opt.zero_grad(set_to_none=True)
                with torch.no_grad():
                    one=r.advance_theory(g[:,:2]);roll=r.advance_theory(one[:,0]);inp=g[fit,:2].reshape(-1,2)
                    lo,hi=inp.amin(0),inp.amax(0);pad=(hi-lo).clamp_min(.01)*.1
                    axes=[torch.linspace(float(lo[d]-pad[d]),float(hi[d]+pad[d]),21,device=device,dtype=torch.double) for d in range(2)]
                    grid=torch.stack(torch.meshgrid(*axes,indexing='ij'),-1).reshape(-1,2)
                    for key,value in dict(one_step=one,rollout=roll,flow_grid=grid,flow_displacement=r.advance_theory(grid)-grid).items():values[name+'_'+key]=value.cpu().numpy()
                    values[name+'_one_step_decoded']=q.decode(one.cpu()).numpy();values[name+'_rollout_decoded']=q.decode(roll.cpu()).numpy()
                if base.state_fingerprint(q)!=qhash or any(x.grad is not None for x in q.parameters()):raise RuntimeError('Q changed in R training')
                summaries[name].update(R_steps=1000,R_Adam=r_clock,R_initial_sha256=r_initial,R_sha256=base.state_fingerprint(r))
                torch.save(dict(R=r.cpu().state_dict(),Q=q.state_dict(),manifest=manifest),OUTPUT/(name+'_QR_final.pt'))
                del opt,r,g;torch.cuda.empty_cache()
        if base.state_fingerprint(source)!=source_hash or any(x.grad is not None for x in source.parameters()):raise RuntimeError('Frozen source changed')
        if code_hashes()!=plan['code_sha256']:raise RuntimeError('Frozen code changed')
        verify_source();verify_cache()
        np.savez_compressed(OUTPUT/'evaluation_arrays.npz',**values)
        base.write_json(OUTPUT/'summary.json',dict(status='complete',arms=summaries,C_updates=0,source_unchanged=True,
            R_gate_passed=do_r,labels_used_in_training=False,thermal_nu=None,elapsed_seconds=time.monotonic()-start,
            peak_cuda_bytes=max(torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved())))
        status(current='report')
        from scripts.report_phi4_local_secant_q import render
        render(OUTPUT);check()
        size=sum(x.stat().st_size for x in OUTPUT.rglob('*') if x.is_file())
        if size>2*1024**3:raise BudgetHold('Directory budget exceeded')
        public={str(x.relative_to(OUTPUT)):base.sha256_file(x) for x in OUTPUT.rglob('*')
                if x.is_file() and x.suffix in ('.json','.npz','.png') and x.name not in ('job.json','status.json')}
        status(status='complete',workflow_complete=True,current='complete',returncode=0,public_files=public,directory_bytes=size)
    except Exception as exc:
        status(status='budget_hold' if isinstance(exc,BudgetHold) else 'failed',error=str(exc),traceback=traceback.format_exc(),
               requires_user_action=True,returncode=2)
        raise


def supervise():
    with (OUTPUT/'execution.lock').open('x') as stream:stream.write(str(os.getpid()))
    started=time.time();child=subprocess.Popen([sys.executable,'-m',MODULE],start_new_session=True)
    base.write_json(OUTPUT/'launch.json',dict(parent_pid=os.getpid(),child_pid=child.pid,started_unix=started,
        host=os.uname().nodename,gpu_uuid=UUID,budget_seconds=1200,plan_sha256=base.sha256_file(OUTPUT/'plan.json')))
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
            error='Owned experiment reached1200-second deadline',requires_user_action=True,returncode=rc)
        base.write_json(OUTPUT/'job.json',job);base.write_json(OUTPUT/'status.json',job)
    sys.exit(rc if rc>=0 else 1)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--supervise',action='store_true')
    args=parser.parse_args()
    if args.prepare:prepare()
    elif args.supervise:supervise()
    else:run()
