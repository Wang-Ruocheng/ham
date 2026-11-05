"""Explicit TRAIN-label Q ablation; fresh CNN QCR and a complete 20-minute cap.

Only microscopic Q is supervised. Labels are standardized using TRAIN only;
coarse states have no coupling targets and C/R receive no supervised gradients.
The endpoint is frozen from a disposable throughput pilot, never from losses.
"""
from dataclasses import replace
from itertools import product
import json
import os
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch

from experiments import ising_kb_cnn_qcr as kb
from experiments import ising_cnn_short_trial as binary
from experiments import phi4_hopping_qcr as common
from experiments.rsmi_separate_qcr import microscopic_coupling_loss
from experiments.phi4_hopping_cnn_qcr import resource_snapshot, GPU_UUID
from recurrent_rg.checkpoints import isolated_rng, qcr_state_dict
from recurrent_rg.runner import prepared_data, file_fingerprint
from recurrent_rg.mi_cache import lattice_statistics_cache_key, load_or_build_mi_statistics
from recurrent_rg.rsmi import region_offsets, kernel_probabilities
from recurrent_rg.representation import normalize_affine_decoder_
from recurrent_rg.phi4_ru_flow import bounded_flow
from scripts.run_ising_highdim import source_hashes


RECIPE = 'ising2d_KB_cnn_b3_g2_supervised_short_seed91_v1'


def preset():
    return dict(schema_version=5, recipe=RECIPE, seed=91, theory_dim=2,
        data_cache=kb.DATA_DIR+'/ensembles.pt', data_sha256=kb.DATA_SHA,
        data_audit=kb.DATA_DIR+'/sampling_audit.json', data_audit_sha256=kb.AUDIT_SHA,
        mi_cache_dir=kb.MI_DIR, lattice_sizes=[36,12,4], block_size=3,
        system_batch=32, sample_batch=64, learning_rates=dict(Q=.001,C=.001,R=.003),
        r_updates_per_batch=1, supervision_weight=10.,
        supervision='Q(full microscopic ensemble) -> TRAIN-standardized [K,B]; no coarse labels',
        target_seconds=1080, hard_seconds=1200, min_steps=200, max_steps=3000,
        step_quantum=50, pilot_steps=10, log_every=50, probe_every=250,
        checkpoint_every=500, max_cuda_bytes=8*1024**3, max_output_bytes=2*1024**3)


def validate_config(path):
    value=json.loads(Path(path).read_text())
    if value != preset() or any(type(value[k]) is not int for k in
            ('seed','theory_dim','hard_seconds','min_steps','max_steps','pilot_steps')):
        raise ValueError('Require the exact explicit supervised short-run contract')
    return value


def label_scaler(train_labels):
    y=torch.as_tensor(train_labels).detach().cpu().double()
    if y.ndim!=2 or y.shape[1]!=2 or len(y)<2 or not torch.isfinite(y).all():
        raise ValueError('Expected finite TRAIN [ensemble, K/B] labels')
    center=y.mean(0); scale=y.std(0,unbiased=False)
    if (scale<=1e-12).any(): raise ValueError('Both TRAIN couplings must vary')
    return dict(mean=center.tolist(),std=scale.tolist(),fitting_split='TRAIN only',
                columns=['K','B'],variance_ddof=0)


def standardized(labels, scaler, device):
    y=torch.as_tensor(labels,device=device,dtype=torch.float64).detach()
    return (y-y.new_tensor(scaler['mean']))/y.new_tensor(scaler['std'])


def draw(ds, rng, device, scaler):
    ids=torch.randperm(len(ds.configurations),generator=rng)[:32]
    samples=torch.randperm(ds.configurations.shape[1],generator=rng)[:64]
    x=ds.configurations[ids[:,None],samples[None,:]].to(device=device,dtype=torch.float64)
    return x,standardized(ds.couplings[ids].reshape(-1,2),scaler,device)


def step(model,batch,labels,hist,mi_ids,patterns,cfg,opts,weight=10.):
    losses,terms,info=binary.objectives(model,batch,hist,mi_ids,patterns,cfg)
    supervised=microscopic_coupling_loss(model.theory_encoder,batch,labels)
    q_base=losses['Q']; losses['Q']=q_base+weight*supervised
    gradients={}
    for name,module in binary.modules(model).items():
        if not torch.isfinite(losses[name]): raise FloatingPointError('Nonfinite '+name)
        gradients[name]=torch.autograd.grad(losses[name],list(module.parameters()),retain_graph=True)
    for name,module in binary.modules(model).items():
        opts[name].zero_grad(set_to_none=True)
        for p,g in zip(module.parameters(),gradients[name]):
            if not torch.isfinite(g).all(): raise FloatingPointError('Nonfinite gradient '+name)
            p.grad=g
    for opt in opts.values():opt.step()
    normalize_affine_decoder_(model.theory_encoder)
    return dict(q_total=float(losses['Q']),q_representation=float(q_base),
        q_supervision=float(supervised),q_supervision_weighted=float(weight*supervised),
        c_total=float(losses['C']),r_total=float(losses['R']),
        q_reconstruction=float(terms['separation']),q_subset_micro=float(terms['anchor']/10),
        q_subset_coarse=float(terms['coarse_consistency']),c_information=float(info),
        c_variance=float(terms['kernel_noncollapse']),c_uniform=float(terms['kernel_sign']),
        closure_one_step=float(terms['commutator']),closure_rollout=float(terms['rollout']))


@torch.no_grad()
def probe(model,train,device,scaler):
    values=kb.train_probe(model,train,device)
    ids=np.linspace(0,len(train.configurations)-1,16,dtype=int)
    x=train.configurations[ids].to(device=device,dtype=torch.float64)
    values['supervised_mse']=float(microscopic_coupling_loss(model.theory_encoder,x,
        standardized(train.couplings[ids].reshape(-1,2),scaler,device)))
    return values


def choose_steps(elapsed,durations,probe_seconds,chain_seconds):
    if len(durations)!=10 or not all(np.isfinite(x) and x>0 for x in durations):
        raise ValueError('Require ten finite complete pilot measurements')
    steady=max(durations[2:])*1.35
    reserve=max(180.,2*chain_seconds*714+120.)
    # Charge at least two probes; endpoint/probe count is fixed before formal training.
    count=min(3000,int((1080-elapsed-reserve-2*probe_seconds)/(steady+probe_seconds/250*1.5)//50)*50)
    if count<200:raise common.BudgetHold('Insufficient 20-minute budget; no formal updates started')
    projection=elapsed+steady*count+reserve+probe_seconds*(count//250+2)*1.5
    return dict(formal_steps=count,steady_step_conservative_seconds=steady,
        evaluation_report_reserve_seconds=reserve,conservative_total_seconds=projection,
        target_seconds=1080,hard_seconds=1200,endpoint_selection='throughput only; frozen before formal training')


def supervision_metrics(arrays,scaler):
    result={};mu=np.asarray(scaler['mean']);sigma=np.asarray(scaler['std'])
    y=arrays['labels']
    for split in ('train','test'):
        mask=arrays['splits']==split
        result[split]={}
        for key in ('direct_full_g','direct_g'):
            pred=arrays[key][mask,0]*sigma+mu
            mse=np.mean((pred-y[mask])**2,axis=0)
            variance=np.var(y[mask],axis=0)
            result[split][key]=dict(rmse=np.sqrt(mse).tolist(),
                nrmse_train_std=(np.sqrt(mse)/sigma).tolist(),
                r2=(1-mse/variance).tolist())
    return result


def run(config_path,output):
    contract=validate_config(config_path);cfg=kb.build_config()
    if (os.uname().nodename!='thinklab-105-225' or os.getpriority(os.PRIO_PROCESS,0)<10
            or os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU_UUID or torch.cuda.is_initialized()):
        raise RuntimeError('Wrong host/priority/UUID mask')
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();actual=latest=0
    job=dict(recipe=RECIPE,status='running',workflow_complete=False,pid=os.getpid(),
        returncode=None,error=None,requires_user_action=False)
    def status(**kw):
        job.update(kw,actual_steps=actual,latest_checkpoint_step=latest,Q_updates=actual,
            C_updates=actual,R_updates=actual,elapsed_seconds=time.monotonic()-started)
        common.write_json(out/'status.json',job);common.write_json(out/'job.json',job)
        print(json.dumps(job,ensure_ascii=False),flush=True)
    def check():
        if time.monotonic()-started>1170:raise common.BudgetHold('Complete 20-minute budget; no continuation')
        if torch.cuda.is_initialized() and max(torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved())>8*1024**3:
            raise common.BudgetHold('Own CUDA cap exceeded')
        if sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>2*1024**3:
            raise common.BudgetHold('Output cap exceeded')
    status(current='admission')
    try:
        admission=[resource_snapshot()];time.sleep(10);admission.append(resource_snapshot())
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        if file_fingerprint(contract['data_cache'])!=kb.DATA_SHA or file_fingerprint(contract['data_audit'])!=kb.AUDIT_SHA:
            raise ValueError('Historical data/audit fingerprint mismatch')
        status(current='loading_existing_data_and_MI')
        train,test=prepared_data(cfg,mmap=True)
        physical=kb.check_cells(train,test,json.loads(Path(contract['data_audit']).read_text()))
        scaler=label_scaler(train.couplings.reshape(-1,2))
        device=torch.device('cuda:0');device_receipt=common._device_receipt(device,GPU_UUID)
        torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(device).total_memory)
        torch.cuda.reset_peak_memory_stats()
        _,environment=region_offsets(2,block=3,buffer=1)
        hist={};cache_receipts={}
        def forbidden():raise common.BudgetHold('Existing MI cache required; no automatic rebuild')
        for split,ds in (('train',train),('test',test)):
            key=lattice_statistics_cache_key(cfg,kb.DATA_SHA,split,ds.configurations,environment,
                'binary_lattice_mi_statistics',{'block':3})
            h,cache_receipts[split]=load_or_build_mi_statistics(kb.MI_DIR,key,forbidden,mmap=True)
            if not cache_receipts[split]['hit']:raise RuntimeError('Existing MI cache required')
            h.prepare_selection();hist[split]=h.to(device) if split=='train' else h
        patterns=torch.tensor(list(product((-1.,1.),repeat=9)),dtype=torch.float64,device=device)[None]
        import scripts.report_ising_kb_supervised_short as report
        code=Path(__file__).resolve().parents[1];pins=source_hashes(code);config_sha=file_fingerprint(config_path)
        common.write_json(out/'plan.json',dict(contract=contract,source_sha256=pins,
            config_sha256=config_sha,physical_data=physical,label_scaler=scaler,admission=admission))
        status(current='disposable_TRAIN_timing_pilot')
        pilot=binary.fresh(replace(cfg,training=replace(cfg.training,seed=191)),device)
        opts=kb.optimizers(pilot);rng=torch.Generator().manual_seed(292);mi_rng=torch.Generator().manual_seed(393)
        durations=[]
        with isolated_rng(device,494):
            for _ in range(10):
                torch.cuda.synchronize();tick=time.monotonic()
                batch,labels=draw(train,rng,device,scaler)
                step(pilot,batch,labels,hist['train'],torch.randperm(len(train.configurations),generator=mi_rng)[:32],patterns,cfg,opts)
                torch.cuda.synchronize();durations.append(time.monotonic()-tick);check()
        tick=time.monotonic();probe(pilot,train,device,scaler);torch.cuda.synchronize();probe_seconds=time.monotonic()-tick
        tiny=SimpleNamespace(configurations=train.configurations[:1],couplings=train.couplings[:1])
        empty=SimpleNamespace(configurations=train.configurations[:0],couplings=train.couplings[:0])
        tick=time.monotonic();kb.collect(pilot,tiny,empty,device,check);torch.cuda.synchronize();chain_seconds=time.monotonic()-tick
        timing=choose_steps(time.monotonic()-started,durations,probe_seconds,chain_seconds)
        timing.update(pilot_seed=191,pilot_updates=10,pilot_discarded=True,step_seconds=durations,
            fixed_TRAIN_probe_seconds=probe_seconds,one_TRAIN_chain_eval_seconds=chain_seconds)
        common.write_json(out/'timing_plan.json',timing)
        del pilot,opts;torch.cuda.empty_cache()
        steps=timing['formal_steps'];cfg=replace(cfg,training=replace(cfg.training,epochs=steps))
        model=binary.fresh(cfg,device);initial=common.state_fingerprint(model)
        initialization=kb.initialize_bias(model,train,device)
        manifest=dict(recipe=RECIPE,contract=contract,config=cfg.to_dict(),actual_planned_steps=steps,
            source_sha256=pins,config_sha256=config_sha,data_sha256=kb.DATA_SHA,data_audit_sha256=kb.AUDIT_SHA,
            physical_data=physical,mi_cache=cache_receipts,label_scaler=scaler,
            q_supervision=dict(weight=10.,scale=0,target='TRAIN-standardized [K,B]',
                input='full microscopic batch',gradient='Q only',coarse_labels=False),
            labels_used_in_training=True,test_used_in_training=False,evaluation_label_columns=['K','B'],
            architecture='per-field CNN Q, shared affine motif decoder; binary stochastic b3 C; residual R2-96-96-2',
            Q_objective='original representation objectives +10 microscopic standardized coupling MSE',
            C_objective='-2 empirical buffered full-shell MI +2 per-ensemble variance floor +1 uniform-color NLL',
            R_objective='uniform all-start anchor coefficient10; one Adam update; no labels or future loss',
            gradient_routing='independent simultaneous gradients, supervision only Q',
            all_Q_C_R_fresh_trainable=True,pretraining=False,physical_sampling_updates=0,
            initialization=initialization,initial_random_state_sha256=initial,learning_rates=contract['learning_rates'],
            endpoint_policy='timing only; no loss/TEST/root based selection; no continuation',
            comparison_qualification='Short supervised trial, NOT an equal-budget comparison with historical6000; standardized latent coordinates differ',
            equilibrium_certified=False,thermal_nu=None,device=device_receipt,admission=admission,timing=timing)
        common.write_json(out/'manifest.json',manifest)
        opts=kb.optimizers(model);rng=torch.Generator().manual_seed(192);mi_rng=torch.Generator().manual_seed(293)
        history=[];probes=[dict(step=0,**probe(model,train,device,scaler))]
        common.write_json(out/'train_probes.json',probes);status(current='training',planned_steps=steps)
        with isolated_rng(device,394):
            for actual in range(1,steps+1):
                check();batch,labels=draw(train,rng,device,scaler)
                metrics=step(model,batch,labels,hist['train'],torch.randperm(len(train.configurations),generator=mi_rng)[:32],patterns,cfg,opts)
                if actual%250==0 or actual==steps:
                    probes.append(dict(step=actual,**probe(model,train,device,scaler)))
                    common.write_json(out/'train_probes.json',probes)
                if actual%500==0 or actual==steps:
                    torch.save(dict(recipe=RECIPE,epoch=actual,step=actual,model=qcr_state_dict(model),metadata=manifest,
                        optimizers={k:v.state_dict() for k,v in opts.items()},batch_rng=rng.get_state(),
                        mi_rng=mi_rng.get_state(),cuda_rng=torch.cuda.get_rng_state()),out/f'training_step_{actual}.pt')
                    latest=actual
                if actual==1 or actual%50==0 or actual==steps:
                    history.append(dict(step=actual,elapsed_seconds=time.monotonic()-started,**metrics))
                    common.write_json(out/'history.json',history)
                    status(last_losses={k:metrics[k] for k in ('q_total','q_supervision','c_total','r_total')})
        clocks=common.optimizer_step_counts(opts,steps)
        for p in model.parameters():p.grad=None
        frozen=common.state_fingerprint(model);status(current='frozen_evaluation')
        arrays=kb.collect(model,train,test,device,check)
        roots,flow=bounded_flow(SimpleNamespace(advance=model.advance_theory),arrays,device,budget_check=check)
        mi_final={}
        with torch.no_grad():
            table=kernel_probabilities(model.rg_kernel,patterns)[0].detach()
            for split in ('train','test'):
                values=[];t=table if split=='train' else table.cpu()
                for i in range(0,len(hist[split]),32):
                    values.append(hist[split].information_for(t,torch.arange(i,min(i+32,len(hist[split])))).cpu());check()
                mi_final[split]=float(torch.cat(values).mean())
        if frozen!=common.state_fingerprint(model) or any(p.grad is not None for p in model.parameters()):
            raise RuntimeError('Frozen evaluation changed model/gradients')
        np.savez(out/'evaluation_arrays.npz',**arrays)
        common.write_json(out/'roots.json',dict(roots=roots,flow_analysis=flow,thermal_nu=None))
        summary=dict(status='complete',recipe=RECIPE,actual_steps=steps,optimizer_steps=clocks,
            closure=binary.closure_metrics(arrays),supervision=supervision_metrics(arrays,scaler),
            numerical_fixed_points=roots,flow_analysis=flow,final_MI=mi_final,
            final_state_sha256=frozen,frozen_evaluation_verified=True,labels_used_in_training=True,
            thermal_nu=None,equilibrium_certified=False,peak_cuda_bytes=max(torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved()),
            elapsed_seconds=time.monotonic()-started)
        common.write_json(out/'summary.json',summary);status(current='report');report.render(out);check()
        if source_hashes(code)!=pins or file_fingerprint(config_path)!=config_sha:
            raise RuntimeError('Frozen source/config changed')
        if file_fingerprint(contract['data_cache'])!=kb.DATA_SHA or file_fingerprint(contract['data_audit'])!=kb.AUDIT_SHA:
            raise RuntimeError('Historical data identity changed')
        public={str(p.relative_to(out)):file_fingerprint(p) for p in out.rglob('*') if p.is_file()
            and p.suffix in ('.json','.png','.npz') and p.name not in ('job.json','status.json')}
        status(status='complete',current=None,workflow_complete=True,returncode=0,public_files=public)
        return summary
    except Exception as exc:
        hold=isinstance(exc,common.BudgetHold)
        status(status='budget_hold' if hold else 'failed',error=f'{type(exc).__name__}: {exc}',
            traceback=traceback.format_exc(),requires_user_action=True,returncode=3 if hold else 1)
        if hold:raise SystemExit(3) from exc
        raise
