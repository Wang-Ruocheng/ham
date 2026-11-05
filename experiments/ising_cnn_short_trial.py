"""Explicit, time-bounded B0 CNN experiment via the standard training entry point."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import os
import sys
import time
import traceback
import numpy as np
import torch

from recurrent_rg.config import RunConfig
from recurrent_rg.checkpoints import build_model, isolated_rng, qcr_state_dict
from recurrent_rg.runner import prepared_data, file_fingerprint
from recurrent_rg.data.statistics import plaquette_distribution
from recurrent_rg.losses import qcr_objective
from recurrent_rg.rsmi import region_offsets, kernel_probabilities
from recurrent_rg.mi_cache import lattice_statistics_cache_key, load_or_build_mi_statistics
from recurrent_rg.representation import normalize_affine_decoder_
from recurrent_rg.encoding import half_subset_theory
from recurrent_rg.r_refinement import refine_r_on_targets, refinement_contract
from experiments.phi4_hopping_qcr import write_json, state_fingerprint, optimizer_step_counts, _device_receipt, BudgetHold
from experiments.phi4_hopping_cnn_qcr import resource_snapshot, GPU_UUID, choose_steps, root_diagnostics
from scripts.report_phi4_hopping_qcr import closure_metrics

RECIPE = 'ising2d_B0_configuration_cnn_b3_g1_short_v1'
REFINED_RECIPE = 'ising2d_B0_configuration_cnn_b3_g1_joint_refinedR_v1'
DATA = '/mnt/nas-new/home/yange/ham/data/ising2d_positiveK_L36_b3_lr_seed91.pt'
DATA_SHA = 'ac04dfbf2cea6dfcbd7b0cee062efd80e89cb4cf79cdd420b96c51a6fef2d040'


def loaded_source_files(source_root):
    """Exclude virtual torch modules whose synthetic __file__ is not a file."""
    source_root=Path(source_root).resolve()
    paths=[]
    for name,module in list(sys.modules.items()):
        if not name.startswith(('recurrent_rg.','experiments.','scripts.')):
            continue
        filename=getattr(module,'__file__',None)
        if filename and str(filename).endswith('.py'):
            path=Path(filename).resolve()
            if path.is_relative_to(source_root) and path.is_file(): paths.append(path)
    return paths


def validate(args, cfg):
    from experiments.rsmi_separate_qcr import build_parser, validate_recipe
    # This narrow dispatch cannot silently ignore unrelated CLI interventions.
    default=vars(build_parser().parse_args(['--config',str(args.config),'--output',str(args.output)]))
    allowed={'config','output','device','configuration_cnn_short_trial'}
    changed=[k for k,v in vars(args).items() if k not in allowed and v!=default[k]]
    if changed or args.source or args.device!='cuda:0':
        raise ValueError('CNN short trial requires only config/output/device and its named flag: '+str(changed))
    validate_recipe(args,cfg)
    expected=RunConfig.load(Path(__file__).resolve().parents[1]/'configs/ising2d_B0_cnn_b3_g1_short_seed91.json')
    if cfg.training.r_updates_per_batch == 10:
        expected=replace(expected,training=replace(expected.training,r_updates_per_batch=10))
    cfg.validate()
    if cfg.to_dict()!=expected.to_dict() or cfg.data.cache!=DATA:
        raise ValueError('This bounded trial requires the pinned Ising config')


def modules(model):
    return dict(Q=model.theory_encoder,C=model.rg_kernel,R=model.theory_rg)


def objectives(model,batch,histogram,mi_ids,patterns,cfg,closure_targets=None):
    kwargs={} if closure_targets is None else dict(closure_targets_sink=closure_targets)
    _, terms=qcr_objective(model,batch,pattern_fn=plaquette_distribution,**cfg.loss.objective_kwargs(),**kwargs)
    info=histogram.information_for(kernel_probabilities(model.rg_kernel,patterns)[0],mi_ids).mean()
    q=2*terms['anchor']+5*terms['coarse_consistency']+10*terms['separation']
    c=-2*info+2*terms['kernel_noncollapse']+terms['kernel_sign']
    r=terms['closure_total']
    return dict(Q=q,C=c,R=r),terms,info


def step(model,batch,histogram,mi_ids,patterns,cfg,opts):
    targets=[] if cfg.training.r_updates_per_batch > 1 else None
    losses,terms,info=objectives(model,batch,histogram,mi_ids,patterns,cfg,targets)
    gradients={}
    for name,module in modules(model).items():
        if not torch.isfinite(losses[name]): raise FloatingPointError('Nonfinite '+name)
        # Same selective independent differentiation as rsmi_separate_qcr.
        gradients[name]=torch.autograd.grad(losses[name],list(module.parameters()),retain_graph=True)
    for name,module in modules(model).items():
        opts[name].zero_grad(set_to_none=True)
        for p,g in zip(module.parameters(),gradients[name]):
            if not torch.isfinite(g).all(): raise FloatingPointError('Nonfinite gradient '+name)
            p.grad=g
    for optimizer in opts.values(): optimizer.step()
    normalize_affine_decoder_(model.theory_encoder)
    refinement={}
    if targets is not None:
        if len(targets)!=1:raise RuntimeError('Missing unique online R target snapshot')
        opts['Q'].zero_grad(set_to_none=True);opts['C'].zero_grad(set_to_none=True)
        after=refine_r_on_targets(model,opts['R'],targets[0],cfg.loss,cfg.training.r_updates_per_batch-1)
        refinement=dict(r_batch_after=after,r_updates_per_batch=cfg.training.r_updates_per_batch)
    return dict(q_total=float(losses['Q']),c_total=float(losses['C']),r_total=float(losses['R']),
        q_reconstruction=float(terms['separation']),q_subset_micro=float(terms['anchor']/10),
        q_subset_coarse=float(terms['coarse_consistency']),c_information=float(info),
        c_variance=float(terms['kernel_noncollapse']),c_uniform=float(terms['kernel_sign']),
        closure_one_step=float(terms['commutator']),closure_rollout=float(terms['rollout']),**refinement)


def fresh(cfg,device):
    with isolated_rng('cpu',cfg.training.seed):
        model=build_model(cfg.model,cfg.training.max_rg_steps).to(device)
    model.detach_kernel_representation=True
    return model


def optimizers(model):
    return {n:torch.optim.Adam(m.parameters(),lr=.001) for n,m in modules(model).items()}


def draw(x,generator,device):
    ids=torch.randperm(len(x),generator=generator)[:32]
    samples=torch.randperm(x.shape[1],generator=generator)[:64]
    batch=x[ids[:,None],samples[None,:]].to(device)
    return batch


@torch.no_grad()
def collect(model,train,test,device,check):
    records={k:[] for k in ['direct_g','direct_full_g','one_step_g','rollout_g','all_start_g',
                           'target_features','reconstructed_features']}
    labels=[];splits=[]
    with isolated_rng(device,20261102):
        for split,ds in [('train',train),('test',test)]:
            for ix,chain in enumerate(ds.configurations):
                levels=model.coarse_grain(chain[None].to(device))
                direct=half_subset_theory(model,levels)
                full=torch.stack([model.theory_encoder(x) for x in levels],1)
                one=model.advance_theory(direct[:,:-1])
                roll1=model.advance_theory(direct[:,0]);roll2=model.advance_theory(roll1)
                rollout=torch.stack([direct[:,0],roll1,roll2],1)
                all_start=torch.stack([one[:,0],roll2,one[:,1]],1)
                targets=torch.stack([plaquette_distribution(x) for x in levels],1)
                reconstruction=model.theory_encoder.distribution_decoder(direct)
                values=[direct,full,one,rollout,all_start,targets,reconstruction]
                for key,value in zip(records,values):records[key].append(value.cpu().numpy()[0])
                labels.append(ds.couplings[ix].reshape(-1).cpu().numpy());splits.append(split)
                check()
    arrays={k:np.asarray(v) for k,v in records.items()}
    arrays.update(labels=np.asarray(labels),splits=np.asarray(splits),all_start_pairs=np.array([[0,1],[0,2],[1,2]]))
    arrays['cell_ids']=np.unique(arrays['labels'][:,0],return_inverse=True)[1]
    values=arrays['direct_g'][arrays['splits']=='train',:-1,0]
    lo,hi=float(values.min()),float(values.max());pad=.1*max(hi-lo,1e-6)
    grid=np.linspace(lo-pad,hi+pad,501)[:,None]
    arrays['flow_grid']=grid
    arrays['flow_next']=model.advance_theory(torch.tensor(grid,device=device)).cpu().numpy()
    for k,v in arrays.items():
        if v.dtype.kind=='f' and not np.isfinite(v).all(): raise FloatingPointError(k)
    return arrays


def run(args):
    started=float(os.environ.get('QCR_TASK_STARTED_EPOCH',time.time()))
    cfg=RunConfig.load(args.config);validate(args,cfg)
    if (os.uname().nodename!='thinklab-105-225' or os.getpriority(os.PRIO_PROCESS,0)<10
            or os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU_UUID or torch.cuda.is_initialized()):
        raise RuntimeError('Wrong host/priority/GPU mask')
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    actual=0;job=dict(status='running',workflow_complete=False,pid=os.getpid(),returncode=None,error=None,requires_user_action=False)
    def status(**kw):
        job.update(kw,actual_steps=actual,elapsed_seconds=time.time()-started)
        write_json(out/'status.json',job);print(json.dumps(job),flush=True)
    def check(reserve=0):
        if time.time()-started+reserve>1200:raise BudgetHold('20-minute total budget; no automatic continuation')
        if torch.cuda.is_initialized() and torch.cuda.max_memory_allocated()>8*1024**3:raise BudgetHold('Own GPU allocation above8GiB')
    try:
        status(current='admission');admission=[resource_snapshot()];time.sleep(10);admission.append(resource_snapshot())
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        status(current='existing_data_and_MI_cache')
        sidecar=Path(DATA).with_suffix('.json');sidecar_sha=file_fingerprint(sidecar)
        provenance=json.loads(sidecar.read_text())
        if provenance['B']!=0 or provenance['data_sha256']!=DATA_SHA or file_fingerprint(DATA)!=DATA_SHA:
            raise ValueError('Physical data identity differs')
        train,test=prepared_data(cfg)
        if train.configurations.shape!=(129,128,36,36,1) or test.configurations.shape!=(63,128,36,36,1):
            raise ValueError('Unexpected data geometry')
        device=torch.device(args.device);receipt=_device_receipt(device,GPU_UUID)
        torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(device).total_memory)
        torch.cuda.reset_peak_memory_stats()
        _,environment=region_offsets(2,block=3,buffer=1)
        hist={};cache_receipts={}
        def forbidden():raise BudgetHold('Existing MI cache required; no rebuild in this short trial')
        for split,ds in [('train',train),('test',test)]:
            key=lattice_statistics_cache_key(cfg,DATA_SHA,split,ds.configurations,environment,
                'binary_lattice_mi_statistics',{'block':3})
            h,cache_receipts[split]=load_or_build_mi_statistics(Path(DATA).parent/'.qcr_mi_cache',key,forbidden,mmap=True)
            assert cache_receipts[split]['hit']
            h.prepare_selection();hist[split]=h.to(device) if split=='train' else h
        check()
        from itertools import product
        patterns=torch.tensor(list(product((-1.,1.),repeat=9)),dtype=torch.float64,device=device)[None]
        import recurrent_rg.config as config_module, recurrent_rg.checkpoints as checkpoints_module
        import recurrent_rg.ising_cnn as cnn_module, recurrent_rg.phi4_cnn_qcr as encoder_module
        import recurrent_rg.losses as losses_module, recurrent_rg.empirical_mi as mi_module
        import recurrent_rg.model_nd as models_module, recurrent_rg.encoding as encoding_module
        import scripts.report_ising_cnn_short_trial as report_module
        import experiments.rsmi_separate_qcr as driver_module
        pinned=[Path(m.__file__).resolve() for m in (config_module,checkpoints_module,cnn_module,encoder_module,
                losses_module,mi_module,models_module,encoding_module,report_module,driver_module)]+[Path(__file__).resolve(),Path(args.config).resolve()]
        source_root=Path(__file__).resolve().parents[1]
        pinned+=loaded_source_files(source_root)
        pins={str(p):file_fingerprint(p) for p in pinned}
        status(current='disposable_timing_pilot')
        pilot=fresh(cfg,device);initial=state_fingerprint(pilot);opts=optimizers(pilot)
        batch_rng=torch.Generator().manual_seed(192);mi_rng=torch.Generator().manual_seed(293)
        durations=[]
        with isolated_rng(device,394):
            for i in range(10):
                torch.cuda.synchronize();tick=time.monotonic()
                step(pilot,draw(train.configurations,batch_rng,device),hist['train'],
                     torch.randperm(len(train.configurations),generator=mi_rng)[:32],patterns,cfg,opts)
                torch.cuda.synchronize();durations.append(time.monotonic()-tick);check()
        reserve=150.;steps=choose_steps(time.time()-started,1.3*max(durations[2:]),reserve,target=900)
        timing=dict(pilot_updates=10,step_seconds=durations,formal_updates=steps,reserve_seconds=reserve,
                    endpoint_selected_from='throughput only; never scientific metrics',target_seconds=900,hard_seconds=1200)
        write_json(out/'timing_plan.json',timing)
        del opts,pilot;torch.cuda.empty_cache()
        model=fresh(cfg,device)
        if state_fingerprint(model)!=initial:raise ValueError('Pilot weights leaked into formal run')
        with isolated_rng(device,9091),torch.no_grad():
            levels=model.coarse_grain(train.configurations[:32].to(device))
            center=torch.stack([plaquette_distribution(x).mean(0) for x in levels]).mean(0)
            model.theory_encoder.distribution_decoder.bias.copy_(center)
        cfg=replace(cfg,training=replace(cfg.training,epochs=steps))
        manifest=dict(recipe=RECIPE if cfg.training.r_updates_per_batch==1 else REFINED_RECIPE,
            config=cfg.to_dict(),data_sha256=DATA_SHA,sidecar_sha256=sidecar_sha,
            source_sha256=pins,device=receipt,admission=admission,timing=timing,mi_cache=cache_receipts,
            label_policy='K only generation/evaluation; no K/B/Kc labels in training',labels_used_in_training=False,
            all_Q_C_R_fresh_trainable=True,pretraining=False,initial_random_state_sha256=initial,
            initialization='Random Q/C/R; TRAIN-only affine decoder mean-bias, no PCA or kernel selection',
            C_loss='-2 empirical full-shell buffered MI +2 per-ensemble variance floor(ratio1) +1 uniform-color NLL',
            Q_loss='shared affine16-bin motif reconstruction10 +micro subset20 +coarse subset5',
            R_loss='all-start anchor; uniform target scales; combined coefficient10; R-only gradients',
            gradient_policy='parameter-selective independent gradients computed before the first optimizer step; extra R-only steps reuse detached batch targets',
            R_optimization=refinement_contract(cfg.training),
            scales=[36,12,4],evaluation='128 configurations per physical chain; one fixed stochastic C trajectory; half-subset-mean Q',
            equilibrium_certified=False,physical_fixed_points_certified=False,thermal_nu=None)
        write_json(out/'manifest.json',manifest);write_json(out/'plan.json',manifest)
        opts=optimizers(model);batch_rng=torch.Generator().manual_seed(192);mi_rng=torch.Generator().manual_seed(293)
        history=[];status(current='training',planned_steps=steps)
        with isolated_rng(device,394):
            for actual in range(1,steps+1):
                check(reserve)
                metrics=step(model,draw(train.configurations,batch_rng,device),hist['train'],
                    torch.randperm(len(train.configurations),generator=mi_rng)[:32],patterns,cfg,opts)
                if actual==1 or actual%25==0 or actual==steps:
                    history.append(dict(step=actual,R_updates=actual*cfg.training.r_updates_per_batch,
                                        elapsed_seconds=time.time()-started,**metrics))
                    write_json(out/'history.json',history)
                    status(current='training',planned_steps=steps,R_updates=actual*cfg.training.r_updates_per_batch,
                           last_losses={k:metrics[k] for k in ['q_total','c_total','r_total']})
                if actual%100==0 or actual==steps:
                    torch.save(dict(epoch=actual,model=qcr_state_dict(model),metadata=manifest,
                        optimizers={n:o.state_dict() for n,o in opts.items()},batch_rng=batch_rng.get_state(),
                        mi_rng=mi_rng.get_state(),cuda_rng=torch.cuda.get_rng_state()),out/'latest.pt')
        clocks={}
        for name,opt in opts.items():
            expected=steps*(cfg.training.r_updates_per_batch if name=='R' else 1)
            clocks.update(optimizer_step_counts({name:opt},expected))
        for p in model.parameters():p.grad=None
        frozen=state_fingerprint(model);model.eval();status(current='evaluation')
        arrays=collect(model,train,test,device,check)
        roots=root_diagnostics(SimpleNamespace(R=model.theory_rg),arrays)
        if frozen!=state_fingerprint(model) or any(p.grad is not None for p in model.parameters()):raise ValueError('Evaluation changed model')
        np.savez(out/'evaluation_arrays.npz',**arrays);write_json(out/'roots.json',roots)
        with torch.no_grad():
            table=kernel_probabilities(model.rg_kernel,patterns)[0].detach()
            mi_final=dict(train=float(hist['train'].information_for(table,torch.arange(len(train.configurations))).mean()),
                          test=float(hist['test'].information_for(table.cpu(),torch.arange(len(test.configurations))).mean()))
        summary=dict(status='complete',actual_steps=steps,optimizer_steps=clocks,closure=closure_metrics(arrays),
            final_MI=mi_final,final_state_sha256=frozen,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
            thermal_nu=None,critical_K=None,labels_used_in_training=False)
        write_json(out/'summary.json',summary);status(current='reporting');report_module.render(out);check()
        if file_fingerprint(DATA)!=DATA_SHA or file_fingerprint(sidecar)!=sidecar_sha:raise ValueError('Data changed')
        if any(file_fingerprint(p)!=h for p,h in pins.items()):raise ValueError('Source changed during run')
        files=[dict(path=str(p.relative_to(out)),sha256=file_fingerprint(p),bytes=p.stat().st_size)
               for p in sorted(out.rglob('*')) if p.is_file() and p.suffix in ('.json','.npz','.png') and p.name!='status.json']
        status(status='complete',current='complete',workflow_complete=True,returncode=0,public_files=files)
    except Exception as e:
        status(status='budget_hold' if isinstance(e,BudgetHold) else 'failed',error=str(e),
               traceback=traceback.format_exc(),returncode=3 if isinstance(e,BudgetHold) else 1,requires_user_action=True)
        raise
