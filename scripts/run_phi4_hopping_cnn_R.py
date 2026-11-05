"""Bounded CPU-only fresh R fit on the frozen scalar CNN-Q latent cache.

Explicit R-only diagnostic, not joint QCR continuation. No Q/C inference,
physical field access, sampling, label supervision, or global-default change.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib
import os
import platform
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from scipy.optimize import brentq
from scripts import run_phi4_hopping_macro_R_ablation as common

SOURCE=Path('/mnt/nas-new/home/yange/ham/results/phi4_hopping_b2_L128_qonly_encoders_seed91_gpu3_vm225_20261102_v1')
PUBLIC_SHA='083404754f1e7e70d1b04b2b7bb73c56c3f0b1a7f5a05781967436e950791b7e'
CACHE_SHA='d808de4b7fc24fb40a5f2b8f42e58443f3ec333265186b60844ad74bb9503bb3'
CHECKPOINT_SHA='7f12f5b26cebe4fb13ac58bd044a27802526102336eb062fc8c8a9459ac7df03'
Q_SHA='113efc6dabb5742cba9834284433c87af875945a768628d9c305c0a8cea421de'
C_SHA='b4c0cb400cc64638dac91ea591798947946e68cc91c078219cc77a529ded37af'
PAIRS=common.PAIRS
STEPS=3000
CPU_SECONDS=300.
MAX_BYTES=100*1024**2


def make_r():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(91)
        return torch.nn.Sequential(torch.nn.Linear(1,96),torch.nn.Tanh(),
            torch.nn.Linear(96,96),torch.nn.Tanh(),torch.nn.Linear(96,1)).double()


def update(model,optimizer,train,generator):
    indices=torch.randint(len(train),(32,),generator=generator)
    optimizer.zero_grad(set_to_none=True)
    loss=common.objective(model,train[indices],'A_equal_target')
    if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite loss')
    loss.backward()
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
        raise FloatingPointError('Nonfinite/missing gradient')
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):raise FloatingPointError('Nonfinite R')
    return indices


def host_resources():
    if platform.system()=='Darwin':
        import psutil
        return dict(cpus=os.cpu_count(),load1=os.getloadavg()[0],mem_available_bytes=psutil.virtual_memory().available,
            io_full_avg10=None,platform='Darwin',minimum_available_bytes=4*1024**3,
            io_note='Linux PSI unavailable; tiny CPU latent-cache fit, no raw-field access')
    return dict(cpus=os.cpu_count(),load1=os.getloadavg()[0],
        mem_available_bytes=int(Path('/proc/meminfo').read_text().split('MemAvailable:')[1].split()[0])*1024,
        io_full_avg10=float(Path('/proc/pressure/io').read_text().split('full avg10=')[1].split()[0]),
        platform='Linux',minimum_available_bytes=36*1024**3)


def admitted(host):
    io_ok=host['io_full_avg10'] is None if host.get('platform')=='Darwin' else host['io_full_avg10']<20
    return host['load1']<=1.25*host['cpus'] and host['mem_available_bytes']>=host['minimum_available_bytes'] and io_ok


def bind_source(source=SOURCE,public_only=False,c_summary_path=None):
    source=Path(source).resolve()
    pins={source/'public_files.json':PUBLIC_SHA,source/'evaluation_arrays.npz':CACHE_SHA}
    if not public_only:pins[source/'C_hierarchical_Q_final.pt']=CHECKPOINT_SHA
    for path,digest in pins.items():
        if common.sha(path)!=digest:raise ValueError('Source pin changed: '+str(path))
    inventory=common.read(source/'public_files.json')['files']
    for row in inventory:
        p=Path(row['path'])
        if p.is_absolute() or '..' in p.parts or p.suffix not in ('.json','.npz','.png'):raise ValueError('Unsafe whitelist')
        path=source/p
        if common.sha(path)!=row['sha256'] or path.stat().st_size!=row['bytes']:raise ValueError('Source public evidence changed')
        pins[path]=row['sha256']
    for name in ('job.json','status.json'):common.complete(common.read(source/name),'Source '+name)
    summary=common.read(source/'summary.json');plan=common.read(source/'plan.json')
    if (summary['actual_steps']!=300 or summary['final_Q_sha256']['C_hierarchical']!=Q_SHA
        or not summary['source_Q_C_R_unchanged'] or summary['C_updates']!=0 or summary['R_updates']!=0
        or summary['labels_used_in_training'] or summary['test_used_in_training']):raise ValueError('Unexpected source protocol')
    if not public_only:
        payload=torch.load(source/'C_hierarchical_Q_final.pt',map_location='cpu',weights_only=False)
        if payload['steps']!=300 or payload['theory_dim']!=1 or common.state_sha(payload['Q'])!=Q_SHA:
            raise ValueError('CNN Q checkpoint binding failed')
        for tensor in payload['Q'].values():
            if not bool(torch.isfinite(tensor).all()):raise ValueError('Nonfinite frozen Q checkpoint')
        del payload
    authoritative_csummary=Path(plan['source'])/'training/summary.json'
    csummary=Path(c_summary_path).resolve() if c_summary_path is not None else authoritative_csummary
    expected=plan['source_protected_sha256'].get(str(authoritative_csummary))
    if expected is None or common.sha(csummary)!=expected or common.read(csummary)['final_module_sha256']['C']!=C_SHA:
        raise ValueError('Frozen C provenance changed')
    pins[csummary]=expected
    with np.load(source/'evaluation_arrays.npz',allow_pickle=False) as z:
        arrays={k:z[k].copy() for k in ('labels','splits','cell_ids')}
        arrays.update(direct_g=z['C_hierarchical_direct_g'].copy(),direct_full_g=z['C_hierarchical_direct_full_g'].copy(),all_start_pairs=np.asarray(PAIRS))
        if not np.allclose(arrays['direct_g'],.5*(z['C_hierarchical_g_first']+z['C_hierarchical_g_second']),rtol=1e-10,atol=1e-12):
            raise ValueError('Latent estimator changed')
    common.validate_cache(arrays)
    return arrays,pins


def metrics(direct,predicted,std,mean):
    v=common.calculate_metrics(direct,predicted,std,mean)
    return {k:v[k] for k in ('pairs','equal_target_objective')}


def scalar_derivative(model,g):
    x=torch.tensor([[g]],dtype=torch.float64,requires_grad=True)
    return float(1+torch.autograd.grad(model(x).sum(),x)[0].item())


def find_roots(model,grid,direct,train_mask):
    """Sign-changing roots only on a predeclared TRAIN-derived grid interval.

    No range extension or loss/seed selection based on roots. Tangencies and
    multiple crossings inside a grid cell are not guaranteed to be found.
    """
    x=np.asarray(grid).reshape(-1)
    if len(x)<2 or not (np.diff(x)>0).all():raise ValueError('Ordered root grid required')
    with torch.no_grad():f=model(torch.tensor(x[:,None],dtype=torch.float64)).numpy()[:,0]
    if not np.isfinite(f).all():raise FloatingPointError('Nonfinite flow')
    if np.max(np.abs(f))<1e-12:return dict(roots=[],degenerate_identity=True,method='grid sign changes')
    def residual(v):
        with torch.no_grad():return float(model(torch.tensor([[v]],dtype=torch.float64)).item())
    candidates=[float(v) for v,z in zip(x,f) if z==0]
    for i in range(len(x)-1):
        if np.signbit(f[i])!=np.signbit(f[i+1]):candidates.append(brentq(residual,x[i],x[i+1],xtol=1e-13,rtol=1e-13))
    roots=[];inputs=direct[train_mask,:3,0].ravel();low,high=float(inputs.min()),float(inputs.max())
    for root in sorted(candidates):
        if roots and abs(root-roots[-1]['g'])<1e-8:continue
        slope=scalar_derivative(model,root);kind='attractor' if abs(slope)<1-1e-3 else 'repeller' if abs(slope)>1+1e-3 else 'near_neutral'
        if abs(residual(root))>1e-9:raise RuntimeError('Unresolved numerical root')
        roots.append(dict(g=root,full_R_derivative=slope,residual_abs=abs(residual(root)),numerical_stability=kind,
            inside_train_input_envelope=low<=root<=high,nearest_train_input_distance=float(np.min(np.abs(inputs-root))),
            input_envelope_by_scale=[bool(direct[train_mask,k,0].min()<=root<=direct[train_mask,k,0].max()) for k in range(3)]))
    return dict(roots=roots,degenerate_identity=False,method='4001-point grid + Brent sign-change brackets',
        search_interval=[float(x[0]),float(x[-1])],root_completeness_certified=False,physical_fixed_points_certified=False)


def run(output,public_source=None,source_c_summary=None):
    wall=time.monotonic();output=Path(output).resolve()
    local=public_source is not None;source=Path(public_source).resolve() if local else SOURCE
    if output.exists() or output==source or source in output.parents:raise ValueError('Fresh independent output required')
    if (local and (platform.system()!='Darwin' or source_c_summary is None)) or (not local and os.uname().nodename!='thinklab-105-225') or os.getpriority(os.PRIO_PROCESS,0)<10:raise RuntimeError('Wrong host/priority/public-source policy')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_initialized():raise RuntimeError('CPU-only process required')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);output.mkdir(parents=True)
    job=dict(status='running',workflow_complete=False,returncode=None,error=None,requires_user_action=False,
        pid=os.getpid(),current='resource_admission',actual_steps=0,Q_updates=0,C_updates=0,gpu_seconds=0)
    def status(**kw):
        job.update(kw,cpu_seconds=time.process_time(),wall_seconds=time.monotonic()-wall)
        common.write(output/'job.json',job);common.write(output/'status.json',job);print(job,flush=True)
    def guard(reserve=20):
        if time.process_time()+reserve>CPU_SECONDS or time.monotonic()-wall>1800:raise common.BudgetHold('Bounded CPU/wall budget reached')
        if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>MAX_BYTES:raise common.BudgetHold('Directory budget')
    status()
    try:
        host=host_resources()
        while not admitted(host):
            status(current='waiting_host_resources',resources=host);time.sleep(30);guard();host=host_resources()
        status(current='source_verification',resources=host)
        arrays,pins=bind_source(source,public_only=local,c_summary_path=source_c_summary)
        for path in (Path(__file__),Path(__file__).with_name('report_phi4_hopping_cnn_R.py'),Path(common.__file__),
                     Path(__file__).with_name('report_phi4_hopping_qonly_balance.py')):pins[path.resolve()]=common.sha(path)
        train_mask=arrays['splits']=='train';direct=torch.tensor(arrays['direct_g'],dtype=torch.float64)
        train=direct[train_mask].detach();std=train.numpy().std(0,ddof=0)[:,0];mean=train.numpy().mean(0)[:,0]
        rng_before=torch.random.get_rng_state().clone();model=make_r();initial=copy.deepcopy(model).eval().requires_grad_(False)
        initial_pin=common.state_sha(model.state_dict());optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        plan=dict(schema_version=5,recipe='fixed_scalar_CNN_Q_fresh_R_v1',source_root=str(source),authoritative_remote_source=str(SOURCE),
            execution_host=platform.node(),local_public_cache_only=local,checkpoint_content_rechecked_this_run=not local,
            source_public_sha256=PUBLIC_SHA,source_latent_cache_sha256=CACHE_SHA,CNN_Q_checkpoint_sha256=CHECKPOINT_SHA,
            CNN_Q_state_sha256=Q_SHA,frozen_C_state_sha256=C_SHA,source_Q_steps=300,source_C_QCR_steps=3000,
            R_initialization='fresh random seed91, not old-coordinate R',R_architecture='g + MLP(1,96,96,1), tanh, float64, no coordinate renormalization',
            R_initial_state_sha256=initial_pin,steps=STEPS,optimizer=dict(name='fresh Adam',lr=.001,seed=91),
            batch_physical_chains=32,sampling='TRAIN chains with replacement',train_chains=40,test_chains=20,
            lattice_sizes=[128,64,32,16],block_size=2,theory_dim=1,closure_objective='all_start_anchor',closure_scale_weights=None,
            all_start_pairs=PAIRS,pair_weights=common.pair_weights('A_equal_target').tolist(),commutator=5,rollout_weight=5,
            Q_updates=0,C_updates=0,labels_used_in_training=False,test_used_in_training=False,model_selection=False,
            cached_anchor_definition='mean of frozen CNN Q(first64), Q(second64), not Q(pooled128); fixed original C trajectory',
            no_new_sampling=True,no_Q_C_inference=True,CPU_only=True,cpu_budget_seconds=CPU_SECONDS,directory_budget_bytes=MAX_BYTES,
            warnings_retained=75,equilibrium_certified=False,resource_receipt=host,
            normalization='fixed CNN-Q TRAIN target-scale population std, ddof0',
            flow_domain='all TRAIN g scales min/max plus 10% range; roots searched here only, input support uses k0..2',
            protected_sha256={str(p):s for p,s in pins.items()})
        common.write(output/'plan.json',plan);job['plan_sha256']=common.sha(output/'plan.json')
        disposable=copy.deepcopy(model);opt=torch.optim.Adam(disposable.parameters(),lr=.001);rng=torch.Generator().manual_seed(91)
        before=time.process_time()
        for _ in range(20):update(disposable,opt,train,rng)
        per_step=(time.process_time()-before)/20;del disposable,opt
        common.write(output/'benchmark.json',dict(disposable_updates=20,formal_updates=0,cpu_seconds_per_step=per_step,projected_remaining_cpu_seconds=1.5*per_step*STEPS+40))
        guard(1.5*per_step*STEPS+40)
        history=[]
        def log(step):
            with torch.no_grad():pred=common.predict_pairs(model,train).numpy()
            history.append(dict(step=step,**metrics(train.numpy(),pred,std,mean)))
            common.write(output/'history.json',history);status(current='R_only_training',actual_steps=step);guard()
        log(0);rng=torch.Generator().manual_seed(91);schedule=hashlib.sha256()
        for step in range(1,STEPS+1):
            indices=update(model,optimizer,train,rng);schedule.update(indices.numpy().tobytes())
            if step%100==0:log(step)
        counts=common.optimizer_counts(optimizer,model)
        if counts!=dict(min=3000,max=3000,parameters=6):raise RuntimeError('Incorrect optimizer clocks')
        status(current='final_evaluation',actual_steps=STEPS)
        low,high=float(train.min()),float(train.max());pad=.1*max(high-low,1e-6)
        grid=np.linspace(low-pad,high+pad,4001)[:,None];export=dict(arrays,flow_grid=grid,train_target_std=std,train_target_mean=mean)
        result={}
        for name,r in [('initial',initial),('final',model)]:
            r.eval()
            with torch.no_grad():
                prediction=common.predict_pairs(r,direct).numpy();export[name+'_all_start']=prediction
                export[name+'_one_step']=(direct[:,:3]+r(direct[:,:3])).numpy()
                tg=torch.tensor(grid,dtype=torch.float64);export[name+'_flow_next']=(tg+r(tg)).numpy()
            result[name]={s:metrics(arrays['direct_g'][arrays['splits']==s],prediction[arrays['splits']==s],std,mean) for s in ('train','test')}
        roots=find_roots(model,grid,arrays['direct_g'],train_mask)
        common.write(output/'roots.json',roots);common.write(output/'metrics.json',result)
        np.savez_compressed(output/'evaluation_arrays.npz',**export)
        torch.save(dict(R=model.state_dict(),step=STEPS,optimizer_steps=counts,frozen_Q_sha256=Q_SHA,frozen_C_sha256=C_SHA),output/'R_final.pt')
        def verify():
            if any(common.sha(p)!=s for p,s in pins.items()):raise RuntimeError('Protected source/code changed')
            if common.state_sha(initial.state_dict())!=initial_pin or not torch.equal(rng_before,torch.random.get_rng_state()):raise RuntimeError('Initial state/global RNG changed')
            if torch.cuda.is_initialized() or any(p.grad is not None for p in model.parameters()):raise RuntimeError('CUDA/gradient isolation failed')
        verify()
        common.write(output/'summary.json',dict(status='complete',actual_steps=STEPS,optimizer_steps=counts,source_Q_steps=300,
            Q_updates=0,C_updates=0,source_public_cache_unchanged=True,Q_C_checkpoint_rechecked=not local,Q_C_model_constructions=0,gpu_seconds=0,cuda_initialized=False,
            labels_used_in_training=False,test_used_in_training=False,model_selection=False,thermal_nu=None,critical_kappa=None,
            fixed_point_search_performed=True,physical_fixed_points_certified=False,final_R_state_sha256=common.state_sha(model.state_dict()),
            batch_schedule_sha256=schedule.hexdigest(),sampling_warning_count=75,equilibrium_certified=False))
        status(current='array_only_report');report=importlib.import_module('scripts.report_phi4_hopping_cnn_R')
        report_paths=report.render_report(output);verify();guard(0)
        common.write(output/'execution.json',dict(status='complete',returncode=0,cpu_seconds=time.process_time(),wall_seconds=time.monotonic()-wall,gpu_seconds=0,source_code_unchanged=True))
        status(status='complete',workflow_complete=True,current='complete',returncode=0)
        names=['plan.json','benchmark.json','history.json','roots.json','metrics.json','evaluation_arrays.npz','summary.json','execution.json','job.json','status.json',*report_paths]
        common.write(output/'public_files.json',dict(files=[dict(path=n,bytes=(output/n).stat().st_size,sha256=common.sha(output/n)) for n in names]))
        return 0
    except Exception as exc:
        traceback.print_exc();status(status='budget_hold' if isinstance(exc,common.BudgetHold) else 'failed',workflow_complete=False,
            returncode=2,error=str(exc),requires_user_action=True);return 2


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true')
    p.add_argument('--public-source',type=Path);p.add_argument('--source-c-summary',type=Path);a=p.parse_args()
    if a.execute:raise SystemExit(run(a.output,a.public_source,a.source_c_summary))
    print(dict(execute=False,steps=STEPS,CPU_only=True,output=str(a.output)))
