"""Matched local-motif Q ablation of the scalar supervised CNN run; <=600s.

Q alone sees microscopic TRAIN-standardized K. The explicit free-decoder
gain policy follows the preceding supervised KB experiment. No R-only tail.
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
from experiments import ising_cnn_short_trial as binary
from experiments import ising_kb_cnn_qcr as kb
from experiments import phi4_hopping_qcr as common
from experiments.ising_kb_free_decoder_matched import step
from experiments.ising_kb_supervised_short import standardized,supervision_metrics
from experiments.rsmi_separate_qcr import microscopic_coupling_loss
from experiments.phi4_hopping_cnn_qcr import resource_snapshot,GPU_UUID,root_diagnostics
from recurrent_rg.checkpoints import isolated_rng,qcr_state_dict
from recurrent_rg.config import RunConfig
from recurrent_rg.runner import prepared_data,file_fingerprint
from recurrent_rg.rsmi import region_offsets,kernel_probabilities
from recurrent_rg.mi_cache import lattice_statistics_cache_key,load_or_build_mi_statistics

RECIPE='ising2d_B0_local_motif_g1_supervision10_matched550_seed91_v1'
BASELINE='/mnt/nas-new/home/yange/ham/results/ising2d_B0_cnn_g1_supervised10min_seed91_gpu3_vm225_20261104_v1'
BASELINE_PINS=dict(manifest='5207cbc4e4c27a1bf35211040bc982db5b1830658321e41425364ff0f1e7f913',summary='3c0847489a982aafaf547be576f910884cde4e149476324f56836186df208121',evaluation_arrays='d3c1dcab861e6af0b81309c8b5d71ba7aaa1df87c5ba22dc195485f1aaf3c0a9')


def preset():
    from experiments.ising_b0_supervised_short import preset as cnn_preset
    value=cnn_preset()
    value.update(recipe=RECIPE,q_encoder='local_motif',formal_steps=550,baseline=BASELINE,baseline_pins=BASELINE_PINS)
    for key in ('minimum_steps','maximum_steps'):del value[key]
    return value


def validate_config(path):
    cfg=json.loads(Path(path).read_text())
    if cfg!=preset():raise ValueError('Require the pinned scalar supervised 10-minute recipe')
    return cfg


def build_config():
    cfg=RunConfig.load(Path(__file__).resolve().parents[1]/'configs/ising2d_B0_cnn_b3_g1_short_seed91.json')
    cfg=replace(cfg,model=replace(cfg.model,q_encoder='local_motif'),training=replace(cfg.training,epochs=550))
    cfg.validate()
    return cfg


def label_scaler(labels):
    y=torch.as_tensor(labels).detach().cpu().double().reshape(-1,1)
    if len(y)<2 or not torch.isfinite(y).all() or y.std(unbiased=False)<=1e-12:raise ValueError('Varying finite TRAIN K required')
    return dict(mean=y.mean(0).tolist(),std=y.std(0,unbiased=False).tolist(),fitting_split='TRAIN only',columns=['K'],variance_ddof=0)


def draw(ds,rng,device,scaler):
    ids=torch.randperm(len(ds.configurations),generator=rng)[:32]
    samples=torch.randperm(ds.configurations.shape[1],generator=rng)[:64]
    x=ds.configurations[ids[:,None],samples[None,:]].to(device=device,dtype=torch.float64)
    return x,standardized(ds.couplings[ids].reshape(-1,1),scaler,device)


@torch.no_grad()
def probe(model,train,device,scaler):
    value=kb.train_probe(model,train,device)
    ids=np.linspace(0,len(train.configurations)-1,16,dtype=int)
    x=train.configurations[ids].to(device=device,dtype=torch.float64)
    value['supervised_mse']=float(microscopic_coupling_loss(model.theory_encoder,x,standardized(train.couplings[ids].reshape(-1,1),scaler,device)))
    return value


def choose_steps(elapsed,durations,probe_seconds,chain_seconds):
    if len(durations)!=10 or not all(np.isfinite(x) and x>0 for x in durations):raise ValueError('Ten full update timings required')
    steady=1.35*max(durations[2:]);reserve=max(100.,2*192*chain_seconds+60.)
    projection=elapsed+550*steady+reserve+1.5*probe_seconds*4
    if projection>540:raise common.BudgetHold('Matched 550 steps cannot fit; no shortened or enlarged run')
    return dict(formal_steps=550,steady_step_conservative_seconds=steady,evaluation_report_reserve_seconds=reserve,
        conservative_total_seconds=projection,hard_seconds=600,endpoint_policy='Fixed 550 to match CNN, no outcome selection')


def verify_baseline(cfg):
    root=Path(BASELINE)
    for name,sha in BASELINE_PINS.items():
        if file_fingerprint(root/(name+('.npz' if name=='evaluation_arrays' else '.json')))!=sha:raise ValueError('Baseline changed: '+name)
    manifest=json.loads((root/'manifest.json').read_text())
    old=RunConfig.from_dict(manifest['config'])
    assert replace(old,model=replace(old.model,q_encoder='local_motif')).to_dict()==cfg.to_dict()
    for name,sha in manifest['source_sha256'].items():
        rel=Path(name).relative_to('/mnt/nas-new/home/yange/ham/snapshots/'+Path(BASELINE).name)
        if str(rel)=='recurrent_rg/cli.py':continue
        current=Path(__file__).resolve().parents[1]/rel
        if not current.is_file() or file_fingerprint(current)!=sha:raise ValueError('Baseline numerical source differs: '+str(rel))
    return manifest


def paired_initialization(cfg,model):
    from experiments.ising_b0_supervised_short import build_config as cnn_config
    reference=binary.fresh(cnn_config(),'cpu')
    pairs={}
    for name in ('C','R'):
        a=binary.modules(model)[name].state_dict();b=binary.modules(reference)[name].state_dict()
        pairs[name]=all(torch.equal(a[k].detach().cpu(),b[k]) for k in a)
    pairs['decoder_weight']=torch.equal(model.theory_encoder.distribution_decoder.weight.detach().cpu(),reference.theory_encoder.distribution_decoder.weight)
    if not all(pairs.values()):raise ValueError('Unpaired initial C/R/decoder')
    return pairs


@torch.no_grad()
def iid_size_control(model,device,check):
    gen=torch.Generator().manual_seed(2026110407);rows=[]
    for side in (4,6,8,12,18,36):
        x=(2*torch.randint(2,(4,256,side,side,1),generator=gen)-1).to(device=device,dtype=torch.float64)
        rows.append(dict(L=side,g=model.theory_encoder(x)[:,0].cpu().tolist()))
        check()
    return dict(seed=2026110407,ensembles=4,frames=256,training_updates=0,rows=rows)


def compare_frozen_C(model,arrays):
    saved=torch.load(Path(BASELINE)/'training_step_550.pt',map_location='cpu',weights_only=True)['model']
    current=qcr_state_dict(model);keys=[k for k in saved if k.startswith('rg_kernel.')]
    max_error=max(float((current[k].detach().cpu()-saved[k]).abs().max()) for k in keys)
    with np.load(Path(BASELINE)/'evaluation_arrays.npz',allow_pickle=False) as z:
        if not np.array_equal(arrays['labels'],z['labels']) or not np.array_equal(arrays['splits'],z['splits']):raise ValueError('Evaluation ordering differs')
        target_error=float(np.max(np.abs(arrays['target_features']-z['target_features'])))
    return dict(C_parameters_bitwise_equal=max_error==0,C_max_abs_difference=max_error,
                physical_target_features_max_abs_difference=target_error)


def run(config_path,output):
    contract=validate_config(config_path);cfg=build_config();started=time.monotonic();actual=0
    if os.uname().nodename!='thinklab-105-225' or os.getpriority(os.PRIO_PROCESS,0)<10 or os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU_UUID or torch.cuda.is_initialized():
        raise RuntimeError('Wrong host/priority/GPU mask')
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    job=dict(status='running',workflow_complete=False,pid=os.getpid(),returncode=None,error=None,requires_user_action=False)
    def status(**kw):
        job.update(kw,actual_steps=actual,elapsed_seconds=time.monotonic()-started)
        for name in ('status.json','job.json'):common.write_json(out/name,job)
        print(json.dumps(job),flush=True)
    def check():
        if time.monotonic()-started>580:raise common.BudgetHold('Complete 10-minute deadline; no automatic continuation')
        if torch.cuda.is_initialized() and max(torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved())>8*1024**3:raise common.BudgetHold('Own GPU allocation >8GiB')
    try:
        status(current='admission');admission=[resource_snapshot()];time.sleep(10);admission.append(resource_snapshot())
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        sidecar=Path(binary.DATA).with_suffix('.json');sidecar_sha=file_fingerprint(sidecar);provenance=json.loads(sidecar.read_text())
        if provenance['B']!=0 or provenance['data_sha256']!=binary.DATA_SHA or file_fingerprint(binary.DATA)!=binary.DATA_SHA:raise ValueError('Data fingerprint differs')
        baseline_manifest=verify_baseline(cfg)
        status(current='loading_existing_data_and_MI');train,test=prepared_data(cfg)
        if train.configurations.shape!=(129,128,36,36,1) or test.configurations.shape!=(63,128,36,36,1):raise ValueError('Wrong data layout')
        scaler=label_scaler(train.couplings);device=torch.device('cuda:0');receipt=common._device_receipt(device,GPU_UUID)
        torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(device).total_memory)
        _,env=region_offsets(2,block=3,buffer=1);hist={};cache_receipts={}
        def forbidden():raise common.BudgetHold('Existing MI cache required; no rebuild in this short run')
        for split,ds in [('train',train),('test',test)]:
            key=lattice_statistics_cache_key(cfg,binary.DATA_SHA,split,ds.configurations,env,'binary_lattice_mi_statistics',{'block':3})
            h,cache_receipts[split]=load_or_build_mi_statistics(Path(binary.DATA).parent/'.qcr_mi_cache',key,forbidden,mmap=True)
            if not cache_receipts[split]['hit']:raise ValueError('Existing cache required')
            h.prepare_selection();hist[split]=h.to(device) if split=='train' else h
        patterns=torch.tensor(list(product((-1.,1.),repeat=9)),dtype=torch.float64,device=device)[None]
        import scripts.report_ising_b0_local_motif_supervised as report
        pins={str(p):file_fingerprint(p) for p in binary.loaded_source_files(Path(__file__).resolve().parents[1])}
        pins[str(Path(config_path).resolve())]=file_fingerprint(config_path)
        status(current='disposable_TRAIN_timing_pilot');pilot=binary.fresh(replace(cfg,training=replace(cfg.training,seed=191)),device)
        kb.initialize_bias(pilot,train,device);opts=kb.optimizers(pilot);rng=torch.Generator().manual_seed(292);mi_rng=torch.Generator().manual_seed(393);durations=[]
        with isolated_rng(device,494):
            for _ in range(10):
                torch.cuda.synchronize();tick=time.monotonic();x,y=draw(train,rng,device,scaler)
                step(pilot,x,y,hist['train'],torch.randperm(len(train.configurations),generator=mi_rng)[:32],patterns,cfg,opts)
                torch.cuda.synchronize();durations.append(time.monotonic()-tick);check()
        tick=time.monotonic();probe(pilot,train,device,scaler);torch.cuda.synchronize();probe_seconds=time.monotonic()-tick
        # Measure one frozen chain on both sides; never inspect scientific pilot results.
        tiny=SimpleNamespace(configurations=train.configurations[:1],couplings=train.couplings[:1])
        tick=time.monotonic();binary.collect(pilot,tiny,tiny,device,check);torch.cuda.synchronize();chain_seconds=(time.monotonic()-tick)/2
        timing=choose_steps(time.monotonic()-started,durations,probe_seconds,chain_seconds)
        timing.update(pilot_seed=191,pilot_updates=10,pilot_discarded=True,step_seconds=durations,probe_seconds=probe_seconds,chain_seconds=chain_seconds)
        common.write_json(out/'timing_plan.json',timing);del pilot,opts;torch.cuda.empty_cache()
        steps=timing['formal_steps'];cfg=replace(cfg,training=replace(cfg.training,epochs=steps))
        model=binary.fresh(cfg,device);initial=common.state_fingerprint(model);paired=paired_initialization(cfg,model);init=kb.initialize_bias(model,train,device)
        manifest=dict(recipe=RECIPE,contract=contract,config=cfg.to_dict(),actual_planned_steps=steps,
            baseline=BASELINE,baseline_pins=BASELINE_PINS,paired_initialization=paired,only_architecture_change="Q configuration_cnn -> local_motif",
            source_sha256=pins,data_sha256=binary.DATA_SHA,sidecar_sha256=sidecar_sha,device=receipt,admission=admission,
            mi_cache=cache_receipts,timing=timing,labels_used_in_training=True,test_used_in_training=False,label_scaler=scaler,
            q_supervision=dict(weight=10.,scale=0,input='full microscopic batch',target='TRAIN-standardized K',gradient='Q only',coarse_labels=False),
            decoder_column_constraint='free_after_initialization',all_Q_C_R_fresh_trainable=True,pretraining=False,physical_sampling_updates=0,
            initialization=init,initial_random_state_sha256=initial,learning_rates=contract['learning_rates'],
            C_objective='-2 empirical full-shell buffered MI+2 per-ensemble variance floor+1 color NLL',
            Q_objective='10 affine motif reconstruction+20 micro subset+5 coarse subset+10 micro K supervision',
            R_objective='uniform all_start_anchor coefficient10; R-only gradients; one update per batch',
            gradient_routing='simultaneous independent gradients computed before any optimizer step',
            endpoint_policy='fixed550 matched to CNN; no R-only tail',equilibrium_certified=False,thermal_nu=None)
        common.write_json(out/'manifest.json',manifest);common.write_json(out/'plan.json',manifest)
        opts=kb.optimizers(model);rng=torch.Generator().manual_seed(192);mi_rng=torch.Generator().manual_seed(293)
        history=[];probes=[dict(step=0,**probe(model,train,device,scaler))]
        common.write_json(out/'train_probes.json',probes);status(current='training',planned_steps=steps)
        with isolated_rng(device,394):
            for actual in range(1,steps+1):
                check();x,y=draw(train,rng,device,scaler)
                metrics=step(model,x,y,hist['train'],torch.randperm(len(train.configurations),generator=mi_rng)[:32],patterns,cfg,opts)
                if actual%250==0 or actual==steps:
                    probes.append(dict(step=actual,**probe(model,train,device,scaler)));common.write_json(out/'train_probes.json',probes)
                if actual%500==0 or actual==steps:
                    torch.save(dict(epoch=actual,step=actual,model=qcr_state_dict(model),metadata=manifest,
                        optimizers={k:v.state_dict() for k,v in opts.items()},batch_rng=rng.get_state(),mi_rng=mi_rng.get_state(),cuda_rng=torch.cuda.get_rng_state()),out/f'training_step_{actual}.pt')
                if actual==1 or actual%50==0 or actual==steps:
                    history.append(dict(step=actual,elapsed_seconds=time.monotonic()-started,**metrics));common.write_json(out/'history.json',history)
                    status(current='training',last_losses={k:metrics[k] for k in ('q_total','q_supervision','c_total','r_total')})
        clocks=common.optimizer_step_counts(opts,steps)
        for p in model.parameters():p.grad=None
        frozen=common.state_fingerprint(model);model.eval();status(current='frozen_evaluation')
        arrays=binary.collect(model,train,test,device,check);roots=root_diagnostics(SimpleNamespace(R=model.theory_rg),arrays)
        common.write_json(out/'iid_size_control.json',iid_size_control(model,device,check))
        paired_final=compare_frozen_C(model,arrays)
        mi_final={}
        with torch.no_grad():
            table=kernel_probabilities(model.rg_kernel,patterns)[0].detach()
            for split,ds in [('train',train),('test',test)]:
                t=table if split=='train' else table.cpu()
                vals=[hist[split].information_for(t,torch.arange(i,min(i+32,len(ds.configurations)))).cpu() for i in range(0,len(ds.configurations),32)]
                mi_final[split]=float(torch.cat(vals).mean());check()
        if common.state_fingerprint(model)!=frozen or any(p.grad is not None for p in model.parameters()):raise ValueError('Frozen evaluation changed model')
        np.savez(out/'evaluation_arrays.npz',**arrays);common.write_json(out/'roots.json',roots)
        summary=dict(status='complete',recipe=RECIPE,actual_steps=steps,optimizer_steps=clocks,supervision=supervision_metrics(arrays,scaler),paired_final=paired_final,
            closure=binary.closure_metrics(arrays),final_MI=mi_final,final_state_sha256=frozen,labels_used_in_training=True,
            decoder_column_norms=model.theory_encoder.distribution_decoder.weight.detach().norm(dim=0).cpu().tolist(),
            peak_cuda_bytes=max(torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved()),thermal_nu=None,critical_K=None)
        common.write_json(out/'summary.json',summary);status(current='report');report.render(out);check()
        if file_fingerprint(binary.DATA)!=binary.DATA_SHA or file_fingerprint(sidecar)!=sidecar_sha or any(file_fingerprint(p)!=h for p,h in pins.items()):raise ValueError('Frozen inputs/code changed')
        public=[dict(path=str(p.relative_to(out)),sha256=file_fingerprint(p),bytes=p.stat().st_size) for p in sorted(out.rglob('*'))
            if p.is_file() and p.suffix in ('.json','.png','.npz') and p.name not in ('status.json','job.json')]
        status(status='complete',current='complete',workflow_complete=True,returncode=0,public_files=public)
        return 0
    except Exception as exc:
        status(status='budget_hold' if isinstance(exc,common.BudgetHold) else 'failed',error=str(exc),traceback=traceback.format_exc(),
            returncode=3 if isinstance(exc,common.BudgetHold) else 1,requires_user_action=True)
        raise
