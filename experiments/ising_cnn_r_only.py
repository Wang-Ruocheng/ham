"""Bounded, online R-only continuation of the completed scalar Ising CNN run.

No physical resampling, MI, Q/C updates, coordinate normalization, new loss,
learning-rate change, or root/TEST-based selection. Preserve source R Adam.
"""
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import time
import traceback
from types import SimpleNamespace
import numpy as np
import torch

from recurrent_rg.checkpoints import load_run, qcr_state_dict, isolated_rng
from recurrent_rg.config import RunConfig
from recurrent_rg.encoding import half_subset_theory
from recurrent_rg.losses import all_start_anchor_closure_loss
from recurrent_rg.runner import prepared_data, file_fingerprint
from experiments.ising_cnn_short_trial import DATA, DATA_SHA, draw, loaded_source_files
from experiments.phi4_hopping_qcr import write_json, state_fingerprint, optimizer_step_counts, BudgetHold, _device_receipt
from experiments.phi4_hopping_cnn_qcr import GPU_UUID, resource_snapshot, root_diagnostics
from experiments.frozen_qc_r_plateau import plateau_reached
from scripts.report_phi4_hopping_qcr import closure_metrics

SOURCE=Path('/mnt/nas-new/home/yange/ham/results/ising2d_B0_cnn_b3_g1_seed91_gpu3_vm225_20261102_v2')
SOURCE_SHA='c039e0237552318bdf5228f947e62220cc01567488933838864bccc07c6ffd22'
RECIPE='ising2d_B0_cnn_b3_g1_online_R_only_v1'


def freeze_and_continue(model,payload):
    for p in model.parameters(): p.requires_grad_(False);p.grad=None
    for p in model.theory_rg.parameters(): p.requires_grad_(True)
    opt=torch.optim.Adam(model.theory_rg.parameters(),lr=.001)
    opt.load_state_dict(copy.deepcopy(payload['optimizers']['R']))
    if len(opt.param_groups)!=1 or opt.param_groups[0]['lr']!=.001:
        raise ValueError('R Adam learning-rate policy differs')
    optimizer_step_counts({'R':opt},payload['epoch'])
    return opt


def r_loss(model,direct):
    return 10*all_start_anchor_closure_loss(model,direct.detach(),coordinate_reduction='sum')[0]


@torch.no_grad()
def online_targets(model,batch):
    return half_subset_theory(model,model.coarse_grain(batch)).detach()


def update(model,direct,opt):
    opt.zero_grad(set_to_none=True)
    loss=r_loss(model,direct)
    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite R loss')
    loss.backward()
    for p in model.theory_rg.parameters():
        if p.grad is None or not torch.isfinite(p.grad).all(): raise FloatingPointError('Invalid R gradient')
    opt.step()
    return float(loss.detach())


@torch.no_grad()
def train_panel(model,configurations,device,check):
    result=[]
    with isolated_rng(device,91910):
        rng=torch.Generator().manual_seed(91910)
        for _ in range(3):
            indices=torch.randperm(configurations.shape[1],generator=rng)[:64]
            for chunk in configurations.split(32):
                result.append(online_targets(model,chunk[:,indices].to(device)))
                check()
    return torch.cat(result)


@torch.no_grad()
def paired_predictions(model,arrays):
    result={k:v.copy() for k,v in arrays.items()}
    device=next(model.parameters()).device
    direct=torch.tensor(arrays['direct_g'],device=device)
    one=model.advance_theory(direct[:,:-1]);roll1=model.advance_theory(direct[:,0]);roll2=model.advance_theory(roll1)
    result['one_step_g']=one.cpu().numpy()
    result['rollout_g']=torch.stack([direct[:,0],roll1,roll2],1).cpu().numpy()
    result['all_start_g']=torch.stack([one[:,0],roll2,one[:,1]],1).cpu().numpy()
    grid=torch.tensor(arrays['flow_grid'],device=device)
    result['flow_next']=model.advance_theory(grid).cpu().numpy()
    return result


def validate_args(args):
    from experiments.rsmi_separate_qcr import build_parser
    baseline=vars(build_parser().parse_args(['--source',str(args.source),'--output',str(args.output)]))
    allowed={'source','output','device','configuration_cnn_r_only'}
    changed=[k for k,v in vars(args).items() if k not in allowed and v!=baseline[k]]
    if changed or args.source.resolve()!=SOURCE or args.device!='cuda:0':
        raise ValueError('Only the pinned source/output/device and explicit R-only flag are allowed: '+str(changed))


def run(args):
    started=float(os.environ.get('QCR_TASK_STARTED_EPOCH',time.time()));validate_args(args)
    if (os.uname().nodename!='thinklab-105-225' or os.getpriority(os.PRIO_PROCESS,0)<10
            or os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU_UUID or torch.cuda.is_initialized()):
        raise RuntimeError('Wrong host/priority/GPU mask')
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    actual=0;job=dict(status='running',workflow_complete=False,pid=os.getpid(),returncode=None,error=None,requires_user_action=False)
    def status(**kw):
        job.update(kw,extra_R_updates=actual,elapsed_seconds=time.time()-started)
        write_json(out/'status.json',job);print(json.dumps(job),flush=True)
    def check(reserve=0):
        if time.time()-started+reserve>1200: raise BudgetHold('20-minute total budget; no automatic restart')
        if torch.cuda.is_initialized() and torch.cuda.max_memory_allocated()>8*1024**3:
            raise BudgetHold('Own GPU allocation above8GiB')
    try:
        status(current='admission');admission=[resource_snapshot()];time.sleep(10);admission.append(resource_snapshot())
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        source_status=json.loads((SOURCE/'status.json').read_text())
        if source_status['status']!='complete' or not source_status['workflow_complete'] or source_status['returncode']!=0:
            raise ValueError('Source is incomplete')
        if file_fingerprint(SOURCE/'latest.pt')!=SOURCE_SHA: raise ValueError('Source checkpoint differs')
        source_pins={str(SOURCE/p):file_fingerprint(SOURCE/p) for p in ['latest.pt','manifest.json','summary.json','evaluation_arrays.npz','roots.json','status.json']}
        for p in ['evaluation_arrays.npz','manifest.json','summary.json','roots.json']:
            expected=next(x['sha256'] for x in source_status['public_files'] if x['path']==p)
            if source_pins[str(SOURCE/p)]!=expected: raise ValueError('Source public SHA differs: '+p)
        loaded=load_run(SOURCE/'latest.pt');cfg=RunConfig.from_dict(loaded.config);cfg.validate()
        if (loaded.payload['epoch']!=850 or cfg.model.q_encoder!='configuration_cnn' or
                cfg.model.theory_dim!=1 or cfg.model.hidden!=96 or cfg.model.block_factor!=3 or
                cfg.training.max_rg_steps!=2 or cfg.loss.closure_objective!='all_start_anchor' or
                cfg.loss.closure_scale_weights is not None or cfg.loss.weights.commutator!=5 or cfg.loss.rollout_weight!=5):
            raise ValueError('Unexpected source model or objective')
        if cfg.data.cache!=DATA or file_fingerprint(DATA)!=DATA_SHA: raise ValueError('Physical data differs')
        device=torch.device(args.device);device_receipt=_device_receipt(device,GPU_UUID)
        torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(device).total_memory)
        torch.cuda.reset_peak_memory_stats()
        model=loaded.model.to(device).eval();model.detach_kernel_representation=True
        opt=freeze_and_continue(model,loaded.payload)
        qc_before={n:state_fingerprint(m) for n,m in [('Q',model.theory_encoder),('C',model.rg_kernel)]}
        initial_R=copy.deepcopy(model.theory_rg.state_dict());initial_opt=copy.deepcopy(opt.state_dict())
        train,test=prepared_data(cfg)
        if train.configurations.shape!=(129,128,36,36,1) or test.configurations.shape!=(63,128,36,36,1):
            raise ValueError('Unexpected source data layout')
        status(current='TRAIN_panel_and_timing')
        panel=train_panel(model,train.configurations,device,check)
        # Disposable timing only: restore BOTH R weights and original Adam before formal continuation.
        rng=torch.Generator();rng.set_state(loaded.payload['batch_rng'])
        timings=[]
        with isolated_rng(device,1):
            torch.cuda.set_rng_state(loaded.payload['cuda_rng'])
            for _ in range(10):
                torch.cuda.synchronize();tick=time.monotonic()
                update(model,online_targets(model,draw(train.configurations,rng,device)),opt)
                torch.cuda.synchronize();timings.append(time.monotonic()-tick);check()
        model.theory_rg.load_state_dict(initial_R);opt.load_state_dict(initial_opt)
        optimizer_step_counts({'R':opt},850)
        conservative=1.3*max(timings[2:]);reserve=90.
        cap=min(6000,int((900-(time.time()-started)-reserve)/conservative//500)*500)
        if cap<1000:raise BudgetHold('Fewer than1000 R-only updates fit the budget')
        import scripts.report_ising_cnn_r_only as report_module
        pins={str(p):file_fingerprint(p) for p in loaded_source_files(Path(__file__).resolve().parents[1])}
        manifest=dict(recipe=RECIPE,config=replace(cfg,training=replace(cfg.training,train_modules=('R',),epochs=850+cap)).to_dict(),
            source=str(SOURCE),source_checkpoint_sha256=SOURCE_SHA,source_joint_updates=850,
            source_files_sha256=source_pins,data_sha256=DATA_SHA,source_sha256=pins,
            admission=admission,device=device_receipt,planned_extra_R_updates=cap,labels_used_in_training=False,
            optimized_modules=['R'],frozen_modules=['Q','C'],frozen_before_sha256=qc_before,
            R_optimizer='Original Adam lr.001, moments and clocks850 preserved; no scheduler or reset',
            target_policy='Original online32 physical ensembles x64 random configurations; fresh stochastic C; mean of two half-subset Q outputs; no latent-cache training',
            objective='Same all_start_anchor:5 e01+2.5 e02+2.5 e12; no coordinate normalization or loss additions',
            stopping='TRAIN only: after>=3000, <1% best-panel improvement over2000 updates, current within5% of best, three consecutive500-step checks; otherwise fixed cap',
            plateau_panel=dict(split='TRAIN',physical_chains=129,configurations_per_draw=64,C_draws_per_chain=3,seed=91910,independent_physical_chains=129),
            timing=dict(pilot_updates_discarded=10,step_seconds=timings,conservative_step_seconds=conservative,reserve_seconds=reserve,hard_seconds=1200,target_seconds=900),
            model_selection='Final endpoint; no TEST/root/K-based selection',MI_evaluated=False,thermal_nu=None)
        write_json(out/'manifest.json',manifest);write_json(out/'plan.json',manifest)
        history=[];best=float('inf');reason='update_cap'
        def evaluate_train():
            nonlocal best
            with torch.no_grad():value=float(r_loss(model,panel))
            if not np.isfinite(value):raise FloatingPointError('Nonfinite TRAIN panel loss')
            best=min(best,value)
            history.append(dict(extra_R_updates=actual,train={'total':value},best_train_closure=best,elapsed_seconds=time.time()-started))
            write_json(out/'history.json',history)
            status(current='training',planned_extra_R_updates=cap,train_panel_loss=value)
            torch.save(dict(epoch=850+actual,extra_R_updates=actual,model=qcr_state_dict(model),metadata=manifest,
                            optimizers={'R':opt.state_dict()},batch_rng=rng.get_state(),cuda_rng=torch.cuda.get_rng_state()),out/'latest.pt')
        rng=torch.Generator();rng.set_state(loaded.payload['batch_rng'])
        with isolated_rng(device,1):
            torch.cuda.set_rng_state(loaded.payload['cuda_rng']);evaluate_train()
            for actual in range(1,cap+1):
                check(reserve)
                update(model,online_targets(model,draw(train.configurations,rng,device)),opt)
                if actual%500==0 or actual==cap:
                    evaluate_train()
                    if plateau_reached(history,minimum=3000):reason='TRAIN_plateau';break
                elif actual%100==0:status(current='training',planned_extra_R_updates=cap)
        qc_after={n:state_fingerprint(m) for n,m in [('Q',model.theory_encoder),('C',model.rg_kernel)]}
        if qc_before!=qc_after or any(p.grad is not None for mod in [model.theory_encoder,model.rg_kernel] for p in mod.parameters()):
            raise AssertionError('Frozen Q/C changed or received gradients')
        clocks=optimizer_step_counts({'R':opt},850+actual)
        for p in model.parameters():p.grad=None
        frozen_R=state_fingerprint(model.theory_rg)
        status(current='paired_final_evaluation')
        with np.load(SOURCE/'evaluation_arrays.npz',allow_pickle=False) as z:base={k:z[k].copy() for k in z.files}
        # Fixed original128-field,192-chain targets: exact same Q/C coordinates before/after.
        arrays=paired_predictions(model,base)
        roots=root_diagnostics(SimpleNamespace(R=model.theory_rg),arrays)
        final_R=copy.deepcopy(model.theory_rg.state_dict());model.theory_rg.load_state_dict(initial_R)
        before=paired_predictions(model,arrays)
        for key in ['one_step_g','rollout_g','all_start_g']:
            np.testing.assert_allclose(before[key],base[key],rtol=1e-12,atol=1e-12)
        model.theory_rg.load_state_dict(final_R)
        for key in ['flow_next','one_step_g','rollout_g','all_start_g']:arrays['source_'+key]=before[key]
        if state_fingerprint(model.theory_rg)!=frozen_R:raise AssertionError('Evaluation changed R')
        np.savez(out/'evaluation_arrays.npz',**arrays);write_json(out/'roots.json',roots)
        summary=dict(status='complete',reason=reason,source_joint_updates=850,actual_extra_R_updates=actual,
            R_Adam_steps=clocks,additional_Q_updates=0,additional_C_updates=0,
            frozen_before_sha256=qc_before,frozen_after_sha256=qc_after,QC_gradients_none=True,
            before_closure=closure_metrics(before),after_closure=closure_metrics(arrays),
            train_panel_initial=history[0]['train']['total'],train_panel_final=history[-1]['train']['total'],
            train_plateau_reached=(reason=='TRAIN_plateau'),final_R_sha256=frozen_R,
            peak_cuda_bytes=torch.cuda.max_memory_allocated(),thermal_nu=None,
            limitations=['TRAIN plateau is operational, not proof of a global optimum.','Single training seed; fixed final C draws and finite chains do not certify physical criticality.'])
        write_json(out/'summary.json',summary);status(current='reporting');report_module.render(out);check()
        if file_fingerprint(DATA)!=DATA_SHA or any(file_fingerprint(p)!=h for p,h in source_pins.items()):raise ValueError('Source/data changed')
        if any(file_fingerprint(p)!=h for p,h in pins.items()):raise ValueError('Implementation changed')
        public=[dict(path=str(p.relative_to(out)),sha256=file_fingerprint(p),bytes=p.stat().st_size)
                for p in sorted(out.rglob('*')) if p.is_file() and p.suffix in ('.json','.png','.npz') and p.name!='status.json']
        status(status='complete',current='complete',workflow_complete=True,returncode=0,public_files=public)
    except Exception as exc:
        status(status='budget_hold' if isinstance(exc,BudgetHold) else 'failed',error=str(exc),traceback=traceback.format_exc(),
               returncode=3 if isinstance(exc,BudgetHold) else 1,requires_user_action=True)
        raise
