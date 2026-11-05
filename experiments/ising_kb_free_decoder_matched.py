"""Explicit strong-supervision ablation with freely learned affine decoder gains.

Keep the strong weight10, initial state, data, RNG streams and 1250 updates.
Only remove the post-Adam projection of decoder columns to unit norm.
The initial unit columns and TRAIN-only mean bias remain identical.
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
from experiments import ising_kb_supervised_short as supervised
from experiments.phi4_hopping_cnn_qcr import resource_snapshot, GPU_UUID
from recurrent_rg.checkpoints import isolated_rng, qcr_state_dict
from recurrent_rg.runner import prepared_data, file_fingerprint
from recurrent_rg.mi_cache import lattice_statistics_cache_key, load_or_build_mi_statistics
from recurrent_rg.rsmi import region_offsets, kernel_probabilities
from recurrent_rg.phi4_ru_flow import bounded_flow
from scripts.run_ising_highdim import source_hashes

RECIPE = 'ising2d_KB_cnn_b3_g2_supervision10_free_decoder_matched1250_seed91_v1'
BASELINE = '/mnt/nas-new/home/yange/ham/results/ising2d_KB_cnn_supervised_short_seed91_gpu2_vm225_20261104_v1'
BASELINE_PINS = dict(manifest='2e2b28fa3890874b96710c078cdc8f554947842da57a990fae313fd9f1ba30e8',
                     summary='257002fb8856cff5fa123ac8842a0fd1842a4a60a834ba986e2d767062942e37')
STEPS = 1250


def preset():
    value = supervised.preset()
    value.update(recipe=RECIPE, supervision_weight=10., decoder_column_constraint='free_after_initialization',
                 supervision=supervised.preset()['supervision'],
                 steps=STEPS, baseline_root=BASELINE, baseline_json_sha256=BASELINE_PINS)
    for key in ('min_steps', 'max_steps', 'step_quantum'):
        del value[key]
    return value


def validate_config(path):
    value = json.loads(Path(path).read_text())
    if value != preset() or isinstance(value.get('supervision_weight'),bool) or any(type(value[k]) is not int for k in
            ('seed', 'theory_dim', 'hard_seconds', 'steps', 'pilot_steps')):
        raise ValueError('Require exact matched supervision10/free-decoder/1250 contract')
    return value


def matched_config():
    cfg = kb.build_config()
    return replace(cfg, training=replace(cfg.training, epochs=STEPS))


def verify_baseline(code):
    root = Path(BASELINE)
    for name, expected in BASELINE_PINS.items():
        if file_fingerprint(root/(name+'.json')) != expected:
            raise ValueError('Supervised baseline fingerprint mismatch: '+name)
    manifest = json.loads((root/'manifest.json').read_text())
    summary = json.loads((root/'summary.json').read_text())
    if (manifest['recipe'] != supervised.RECIPE or summary['status'] != 'complete'
            or summary['actual_steps'] != STEPS or manifest['config'] != json.loads(json.dumps(matched_config().to_dict()))
            or manifest['q_supervision']['weight'] != 10.):
        raise ValueError('Baseline configuration/endpoint mismatch')
    from scripts.report_ising_kb_weak_supervision_matched import ZERO_BASELINE, ZERO_PINS
    for name,sha in ZERO_PINS.items():
        if file_fingerprint(Path(ZERO_BASELINE)/(name+'.json'))!=sha:
            raise ValueError('Unsupervised baseline fingerprint mismatch')
    # Only CLI dispatch and display preferences changed among original files.
    excluded = {'recurrent_rg/cli.py', 'AGENTS.md'}
    verified = []
    for name, sha in manifest['source_sha256'].items():
        if name not in excluded:
            if file_fingerprint(Path(code)/name) != sha:
                raise ValueError('Numerical source differs from supervised run: '+name)
            verified.append(name)
    return manifest, dict(baseline_root=BASELINE, baseline_json_sha256=BASELINE_PINS,
                          equal_model_data_loss_config=True, numerical_files_verified=len(verified),
                          excluded_display_or_dispatch=sorted(excluded),
                          only_training_objective_change='remove_post_Adam_unit_column_projection_only')


def fixed_budget(elapsed, durations, probe_seconds, chain_seconds):
    if len(durations) != 10 or not all(np.isfinite(t) and t > 0 for t in durations):
        raise ValueError('Require ten complete timing samples')
    steady = 1.35*max(durations[2:])
    reserve = max(180., 2*chain_seconds*714+120.)
    projection = elapsed+steady*STEPS+reserve+1.5*probe_seconds*(STEPS//250+2)
    if projection > 1080:
        raise common.BudgetHold('Matched1250 does not fit time budget; no shortened run')
    return dict(formal_steps=STEPS, steady_step_conservative_seconds=steady,
                evaluation_report_reserve_seconds=reserve, conservative_total_seconds=projection,
                target_seconds=1080, hard_seconds=1200,
                endpoint_selection='fixed1250 to match supervised baseline, never chosen from outcomes')


def step(model,batch,labels,hist,mi_ids,patterns,cfg,opts,weight=10.):
    losses,terms,info=binary.objectives(model,batch,hist,mi_ids,patterns,cfg)
    supervised_loss=supervised.microscopic_coupling_loss(model.theory_encoder,batch,labels)
    q_base=losses['Q']; losses['Q']=q_base+weight*supervised_loss
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
    # Explicit ablation: keep affine weights free after the shared initialization.
    norms=model.theory_encoder.distribution_decoder.weight.detach().norm(dim=0)
    return dict(decoder_column_norms=norms.cpu().tolist(),q_total=float(losses['Q']),q_representation=float(q_base),
        q_supervision=float(supervised_loss),q_supervision_weighted=float(weight*supervised_loss),
        c_total=float(losses['C']),r_total=float(losses['R']),
        q_reconstruction=float(terms['separation']),q_subset_micro=float(terms['anchor']/10),
        q_subset_coarse=float(terms['coarse_consistency']),c_information=float(info),
        c_variance=float(terms['kernel_noncollapse']),c_uniform=float(terms['kernel_sign']),
        closure_one_step=float(terms['commutator']),closure_rollout=float(terms['rollout']))


def run(config_path, output):
    contract=validate_config(config_path);cfg=matched_config()
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
        if time.monotonic()-started>1170:raise common.BudgetHold('Complete20minute cap; no continuation')
        if torch.cuda.is_initialized() and max(torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved())>8*1024**3:
            raise common.BudgetHold('Own CUDA cap exceeded')
        if sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>2*1024**3:
            raise common.BudgetHold('Output cap exceeded')
    status(current='admission')
    try:
        code=Path(__file__).resolve().parents[1]
        baseline,match=verify_baseline(code)
        admission=[resource_snapshot()];time.sleep(10);admission.append(resource_snapshot())
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        if file_fingerprint(contract['data_cache'])!=kb.DATA_SHA or file_fingerprint(contract['data_audit'])!=kb.AUDIT_SHA:
            raise ValueError('Historical data/audit fingerprint mismatch')
        status(current='loading_existing_data_and_MI')
        train,test=prepared_data(cfg,mmap=True)
        physical=kb.check_cells(train,test,json.loads(Path(contract['data_audit']).read_text()))
        if physical!=baseline['physical_data']:raise ValueError('Physical comparison contract mismatch')
        scaler=supervised.label_scaler(train.couplings)
        if scaler!=baseline['label_scaler']:raise ValueError('TRAIN label scaler differs')
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
        import scripts.report_ising_kb_free_decoder_matched as report
        pins=source_hashes(code);config_sha=file_fingerprint(config_path)
        common.write_json(out/'plan.json',dict(contract=contract,source_sha256=pins,
            config_sha256=config_sha,physical_data=physical,baseline_match=match,admission=admission))
        status(current='disposable_TRAIN_timing_pilot')
        pilot=binary.fresh(replace(cfg,training=replace(cfg.training,seed=191)),device)
        opts=kb.optimizers(pilot);rng=torch.Generator().manual_seed(292);mi_rng=torch.Generator().manual_seed(393)
        durations=[]
        with isolated_rng(device,494):
            for _ in range(10):
                torch.cuda.synchronize();tick=time.monotonic()
                batch,labels=supervised.draw(train,rng,device,scaler)
                step(pilot,batch,labels,hist['train'],torch.randperm(len(train.configurations),generator=mi_rng)[:32],patterns,cfg,opts,weight=10.)
                torch.cuda.synchronize();durations.append(time.monotonic()-tick);check()
        tick=time.monotonic();supervised.probe(pilot,train,device,scaler);torch.cuda.synchronize();probe_seconds=time.monotonic()-tick
        tiny=SimpleNamespace(configurations=train.configurations[:1],couplings=train.couplings[:1])
        empty=SimpleNamespace(configurations=train.configurations[:0],couplings=train.couplings[:0])
        tick=time.monotonic();kb.collect(pilot,tiny,empty,device,check);torch.cuda.synchronize();chain_seconds=time.monotonic()-tick
        timing=fixed_budget(time.monotonic()-started,durations,probe_seconds,chain_seconds)
        timing.update(pilot_seed=191,pilot_updates=10,pilot_discarded=True,step_seconds=durations,
                      fixed_TRAIN_probe_seconds=probe_seconds,one_TRAIN_chain_eval_seconds=chain_seconds)
        common.write_json(out/'timing_plan.json',timing)
        del pilot,opts;torch.cuda.empty_cache()
        model=binary.fresh(cfg,device);initial=common.state_fingerprint(model)
        if initial!=baseline['initial_random_state_sha256']:raise ValueError('Initial Q/C/R state differs from supervised')
        initialization=kb.initialize_bias(model,train,device)
        if initialization!=baseline['initialization']:raise ValueError('Mean-bias initialization contract differs')
        manifest=dict(recipe=RECIPE,contract=contract,config=cfg.to_dict(),actual_planned_steps=STEPS,
            source_sha256=pins,config_sha256=config_sha,data_sha256=kb.DATA_SHA,data_audit_sha256=kb.AUDIT_SHA,
            physical_data=physical,mi_cache=cache_receipts,baseline_match=match,
            decoder_column_constraint=contract['decoder_column_constraint'],
            q_supervision=dict(weight=10.,enabled=True,scale=0,input='full microscopic batch',target=baseline['q_supervision']['target'],gradient='Q only',coarse_labels=False),
            label_scaler=scaler,
            labels_used_in_training=True,test_used_in_training=False,evaluation_label_columns=['K','B'],
            architecture=baseline['architecture'],Q_objective='unchanged representation objectives +10*microscopic_coupling_MSE; no decoder column projection',
            C_objective=baseline['C_objective'],R_objective=baseline['R_objective'],
            gradient_routing='independent simultaneous gradients; supervision updates microscopic Q only',
            all_Q_C_R_fresh_trainable=True,pretraining=False,physical_sampling_updates=0,
            initialization=initialization,initial_random_state_sha256=initial,initial_state_matches_supervised=True,
            learning_rates=contract['learning_rates'],batch_seed=192,MI_batch_seed=293,C_training_seed=394,
            endpoint_policy='fixed1250 matched updates; no continuation or outcome-based selection',
            comparison_qualification='matched seed/data/architecture/updates; latent coordinate gauges differ',
            equilibrium_certified=False,thermal_nu=None,device=device_receipt,admission=admission,timing=timing)
        common.write_json(out/'manifest.json',manifest)
        opts=kb.optimizers(model);rng=torch.Generator().manual_seed(192);mi_rng=torch.Generator().manual_seed(293)
        history=[];probes=[dict(step=0,**supervised.probe(model,train,device,scaler))]
        common.write_json(out/'train_probes.json',probes);status(current='training',planned_steps=STEPS)
        with isolated_rng(device,394):
            for actual in range(1,STEPS+1):
                check();batch,labels=supervised.draw(train,rng,device,scaler)
                metrics=step(model,batch,labels,hist['train'],torch.randperm(len(train.configurations),generator=mi_rng)[:32],patterns,cfg,opts,weight=10.)
                if actual%250==0 or actual==STEPS:
                    probes.append(dict(step=actual,**supervised.probe(model,train,device,scaler)))
                    common.write_json(out/'train_probes.json',probes)
                if actual%500==0 or actual==STEPS:
                    torch.save(dict(recipe=RECIPE,epoch=actual,step=actual,model=qcr_state_dict(model),metadata=manifest,
                        optimizers={k:v.state_dict() for k,v in opts.items()},batch_rng=rng.get_state(),
                        mi_rng=mi_rng.get_state(),cuda_rng=torch.cuda.get_rng_state()),out/f'training_step_{actual}.pt')
                    latest=actual
                if actual==1 or actual%50==0 or actual==STEPS:
                    history.append(dict(step=actual,elapsed_seconds=time.monotonic()-started,**metrics))
                    common.write_json(out/'history.json',history)
                    status(last_losses={k:metrics[k] for k in ('q_total','c_total','r_total')})
        clocks=common.optimizer_step_counts(opts,STEPS)
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
        summary=dict(status='complete',recipe=RECIPE,actual_steps=STEPS,optimizer_steps=clocks,
            supervision=supervised.supervision_metrics(arrays,scaler),
            closure=binary.closure_metrics(arrays),numerical_fixed_points=roots,flow_analysis=flow,final_MI=mi_final,
            final_state_sha256=frozen,frozen_evaluation_verified=True,labels_used_in_training=True,
            thermal_nu=None,equilibrium_certified=False,decoder_column_norms=model.theory_encoder.distribution_decoder.weight.detach().norm(dim=0).cpu().tolist(),
            peak_cuda_bytes=max(torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved()),
            elapsed_seconds=time.monotonic()-started)
        common.write_json(out/'summary.json',summary);status(current='report');report.render(out);check()
        if source_hashes(code)!=pins or file_fingerprint(config_path)!=config_sha:raise RuntimeError('Frozen source/config changed')
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
