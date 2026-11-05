"""Explicit fixed-trajectory R-only diagnostic of the supervised free-decoder run.

Only TRAIN cached latent trajectories enter optimization. Q/C and their decoder
are frozen; no physical data, MI, label loss, or fresh C draws are evaluated.
This is full-batch fitting of a conditional saved C realization, not an exact
continuation of the online joint stochastic objective.
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
from recurrent_rg.checkpoints import load_run, qcr_state_dict
from recurrent_rg.config import RunConfig
from recurrent_rg.losses import all_start_anchor_closure_loss
from recurrent_rg.runner import file_fingerprint
from recurrent_rg.phi4_ru_flow import bounded_flow
from experiments.ising_cnn_short_trial import loaded_source_files
from experiments.phi4_hopping_qcr import write_json, state_fingerprint, optimizer_step_counts, BudgetHold, _device_receipt
from experiments.phi4_hopping_cnn_qcr import GPU_UUID, resource_snapshot
from experiments.frozen_qc_r_plateau import plateau_reached
from scripts.report_phi4_hopping_qcr import closure_metrics

RECIPE='ising2d_KB_free_decoder_frozen_QC_fullbatch_R_v1'
SOURCE=Path('/mnt/nas-new/home/yange/ham/results/ising2d_KB_cnn_supervision10_free_decoder_matched1250_seed91_gpu3_vm225_20261104_v1')
SOURCE_PINS={
    'training_step_1250.pt':'bffb4a41ff9f5a3d04c000a86f87f2353bafd48d2083579eed9ca70d6b0cd03b',
    'manifest.json':'28d9f7dff47d71885e71e5ee09148d4d16dd668b73207f5faf8bccb46422fffb',
    'summary.json':'5243bf443e4d0b5220d552446b8075f23472ef3f189dedecda104802d5b032bb',
    'evaluation_arrays.npz':'7a524c5c57613a7e782cf86d9195d24909f255d30a46c200c3ba449337fc0506',
    'roots.json':'134cfd4eafcce2a3ad00d7b53b6217cdf10a73e3933aec5a83eb83b9086bd9c2',
    'status.json':'ff1b902ad5c64c496f5d372f3e4ab27e1fa5977b85554a6221bc7bd140f8ffa1'}


def preset():
    return dict(schema_version=5,recipe=RECIPE,source=str(SOURCE),source_sha256=SOURCE_PINS,
        source_joint_updates=1250,learning_rate=.003,max_extra_R_updates=60000,evaluate_every=1000,
        minimum_updates=10000,plateau_window=5000,plateau_rtol=.01,plateau_confirmations=3,
        stability_rtol=.05,hard_seconds=1200,training_seconds=1020,
        train_chains=474,test_chains=240,target_policy='saved_half_subset_direct_g_full_TRAIN',
        labels_used_in_training=False,closure_objective='all_start_anchor',closure_scale_weights=None)


def validate_config(path):
    value=json.loads(Path(path).read_text())
    if value!=preset():raise ValueError('This explicit R-only diagnostic has a pinned protocol')
    return value


def freeze_and_continue(model,payload):
    for p in model.parameters():p.requires_grad_(False);p.grad=None
    for p in model.theory_rg.parameters():p.requires_grad_(True)
    opt=torch.optim.Adam(model.theory_rg.parameters(),lr=.003)
    opt.load_state_dict(copy.deepcopy(payload['optimizers']['R']))
    if len(opt.param_groups)!=1 or opt.param_groups[0]['lr']!=.003:raise ValueError('Wrong inherited R learning rate')
    optimizer_step_counts({'R':opt},payload['epoch'])
    return opt


def validate_loaded_state(model,payload):
    # Historical whole-model hashes include an unused action_network omitted
    # by qcr_state_dict. Compare every saved Q/C/R tensor exactly instead.
    current=qcr_state_dict(model);saved=payload['model']
    if set(current)!=set(saved):raise ValueError('Loaded Q/C/R keys differ')
    for k in saved:
        if current[k].dtype!=saved[k].dtype or not torch.equal(current[k].detach().cpu(),saved[k].detach().cpu()):
            raise ValueError('Loaded Q/C/R tensor differs: '+k)
    return dict(saved_tensor_count=len(saved),all_saved_tensors_bitwise_equal=True,
        qcr_sha256=state_fingerprint(SimpleNamespace(state_dict=lambda:current)),
        omitted_unused_action_network=True)


def r_loss(model,direct):
    return 10*all_start_anchor_closure_loss(model,direct.detach(),coordinate_reduction='sum')[0]


def update(model,direct,opt):
    opt.zero_grad(set_to_none=True);loss=r_loss(model,direct)
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite R loss')
    loss.backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.theory_rg.parameters()):
        raise FloatingPointError('Invalid R gradient')
    opt.step()


def training_targets(arrays,device):
    # Labels are intentionally not read by this helper.
    mask=arrays['splits']=='train';values=arrays['direct_g'][mask]
    if values.shape!=(474,3,2) or not np.isfinite(values).all():raise ValueError('Wrong TRAIN trajectories')
    if not np.array_equal(values,arrays['train_direct']):raise ValueError('TRAIN ordering differs')
    return torch.tensor(values,device=device,dtype=torch.float64).detach()


@torch.no_grad()
def predictions(model,arrays):
    a={k:v.copy() for k,v in arrays.items()};device=next(model.parameters()).device
    g=torch.tensor(a['direct_g'],device=device,dtype=torch.float64)
    one=model.advance_theory(g[:,:-1]);roll2=model.advance_theory(one[:,0])
    a['one_step_g']=one.cpu().numpy()
    a['rollout_g']=torch.stack([g[:,0],one[:,0],roll2],1).cpu().numpy()
    a['all_start_g']=torch.stack([one[:,0],roll2,one[:,1]],1).cpu().numpy()
    if not np.array_equal(a['all_start_pairs'],[[0,1],[0,2],[1,2]]):raise ValueError('Wrong endpoint pairs')
    return a


def run(config,output):
    protocol=validate_config(config);started=time.monotonic();actual=0
    if os.uname().nodename!='thinklab-105-225' or os.getpriority(os.PRIO_PROCESS,0)<10 or os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU_UUID:
        raise RuntimeError('Wrong host, priority, or GPU mask')
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    job=dict(status='running',workflow_complete=False,pid=os.getpid(),returncode=None,error=None,requires_user_action=False)
    def status(**kw):
        job.update(kw,extra_R_updates=actual,elapsed_seconds=time.monotonic()-started)
        for name in ('status.json','job.json'):write_json(out/name,job)
        print(json.dumps(job),flush=True)
    def check():
        if time.monotonic()-started>protocol['hard_seconds']-15:raise BudgetHold('Complete workflow deadline')
        if torch.cuda.is_initialized() and torch.cuda.max_memory_allocated()>8*1024**3:raise BudgetHold('Own GPU memory above 8GiB')
    try:
        status(current='resource_admission');admission=[resource_snapshot()];time.sleep(10);admission.append(resource_snapshot())
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        for p,h in SOURCE_PINS.items():
            if file_fingerprint(SOURCE/p)!=h:raise ValueError('Source fingerprint differs: '+p)
        source_status=json.loads((SOURCE/'status.json').read_text())
        if source_status['status']!='complete' or not source_status['workflow_complete'] or source_status['returncode']!=0:raise ValueError('Source incomplete')
        loaded=load_run(SOURCE/'training_step_1250.pt');cfg=RunConfig.from_dict(loaded.config)
        if loaded.payload['epoch']!=1250 or cfg.model.theory_dim!=2 or cfg.model.block_factor!=3 or cfg.model.hidden!=96 or cfg.model.q_encoder!='configuration_cnn' or cfg.training.max_rg_steps!=2 or cfg.loss.closure_objective!='all_start_anchor' or cfg.loss.closure_scale_weights is not None or cfg.loss.weights.commutator!=5 or cfg.loss.rollout_weight!=5:
            raise ValueError('Source architecture/objective differs')
        device=torch.device('cuda:0');device_receipt=_device_receipt(device,GPU_UUID)
        torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(device).total_memory)
        model=loaded.model.to(device).eval();model.detach_kernel_representation=True
        loaded_state_receipt=validate_loaded_state(model,loaded.payload)
        opt=freeze_and_continue(model,loaded.payload)
        qc_before={n:state_fingerprint(m) for n,m in [('Q',model.theory_encoder),('C',model.rg_kernel)]}
        with np.load(SOURCE/'evaluation_arrays.npz',allow_pickle=False) as z:base={k:z[k].copy() for k in z.files}
        replay=predictions(model,base)
        for key in ('one_step_g','rollout_g','all_start_g'):np.testing.assert_allclose(replay[key],base[key],rtol=1e-12,atol=1e-12)
        panel=training_targets(base,device)
        import scripts.report_ising_kb_frozen_r as report_module
        pins={str(p):file_fingerprint(p) for p in loaded_source_files(Path(__file__).resolve().parents[1])}
        continued_cfg=replace(cfg,training=replace(cfg.training,train_modules=('R',),learning_rate=.003,epochs=1250+protocol['max_extra_R_updates']))
        manifest=dict(recipe=RECIPE,protocol=protocol,config=continued_cfg.to_dict(),source_config=cfg.to_dict(),source=str(SOURCE),
            source_joint_updates=1250,source_sha256=SOURCE_PINS,code_sha256=pins,admission=admission,device=device_receipt,
            loaded_state_verification=loaded_state_receipt,
            labels_used_in_training=False,source_Q_was_supervised=True,test_used_in_training=False,
            optimized_modules=['R'],frozen_modules=['Q','C'],frozen_before_sha256=qc_before,
            optimizer_policy='Inherit R Adam weights, moments and clock1250; lr.003 constant; no reset/scheduler',
            target_policy='474 TRAIN physical-chain latent trajectories, one saved C realization, mean of two half-subset Q outputs; full batch',
            change_from_joint='Fixed conditional targets and full-batch optimizer replace online32-chain stochastic targets',
            objective='5 e01 +2.5 e02 +2.5 e12, each error is mean coordinate-sum squared distance',
            endpoint_policy='Final endpoint; TRAIN plateau or fixed update/time cap, no TEST/root-based selection',
            physical_data_opened=False,MI_evaluated=False,thermal_nu=None)
        write_json(out/'manifest.json',manifest);write_json(out/'plan.json',protocol)
        history=[];best=float('inf');reason='update_cap'
        def evaluate_train():
            nonlocal best
            with torch.no_grad():value=float(r_loss(model,panel))
            if not np.isfinite(value):raise FloatingPointError('Nonfinite TRAIN evaluation')
            best=min(best,value)
            history.append(dict(extra_R_updates=actual,train={'total':value},best_train_closure=best,elapsed_seconds=time.monotonic()-started))
            write_json(out/'history.json',history);status(current='R_training',train_loss=value)
        def save_checkpoint():
            torch.save(dict(epoch=1250+actual,extra_R_updates=actual,model=qcr_state_dict(model),metadata=manifest,
                optimizers={'R':opt.state_dict()}),out/'latest.pt')
        evaluate_train();save_checkpoint()
        for step in range(1,protocol['max_extra_R_updates']+1):
            check()
            if time.monotonic()-started>protocol['training_seconds']:reason='training_time_cap';break
            update(model,panel,opt);actual=step
            if actual%protocol['evaluate_every']==0 or actual==protocol['max_extra_R_updates']:
                evaluate_train();save_checkpoint()
                if plateau_reached(history,minimum=protocol['minimum_updates'],window=protocol['plateau_window'],rtol=protocol['plateau_rtol'],confirmations=protocol['plateau_confirmations'],stability_rtol=protocol['stability_rtol']):
                    reason='TRAIN_plateau';break
        if history[-1]['extra_R_updates']!=actual:evaluate_train()
        save_checkpoint();clocks=optimizer_step_counts({'R':opt},1250+actual)
        qc_after={n:state_fingerprint(m) for n,m in [('Q',model.theory_encoder),('C',model.rg_kernel)]}
        if qc_before!=qc_after or any(p.grad is not None for mod in (model.theory_encoder,model.rg_kernel) for p in mod.parameters()):raise AssertionError('Q/C changed or received gradients')
        for p in model.parameters():p.grad=None
        final_fingerprint=state_fingerprint(model)
        status(current='heldout_evaluation_and_bounded_roots');arrays=predictions(model,base)
        roots,flow=bounded_flow(SimpleNamespace(advance=model.advance_theory),arrays,device,budget_check=check)
        for k in ('one_step_g','rollout_g','all_start_g','flow_g','flow_R','flow_displacement'):arrays['source_'+k]=base[k].copy()
        np.testing.assert_array_equal(arrays['flow_g'],base['flow_g'])
        if state_fingerprint(model)!=final_fingerprint:raise AssertionError('Evaluation mutated model')
        np.savez(out/'evaluation_arrays.npz',**arrays);write_json(out/'roots.json',dict(roots=roots,flow_analysis=flow))
        summary=dict(status='complete',recipe=RECIPE,reason=reason,source_joint_updates=1250,actual_extra_R_updates=actual,
            R_Adam_steps=clocks,additional_Q_updates=0,additional_C_updates=0,frozen_before_sha256=qc_before,frozen_after_sha256=qc_after,
            QC_gradients_none=True,source_predictions_replayed=True,before_closure=closure_metrics(base),after_closure=closure_metrics(arrays),
            source_roots=json.loads((SOURCE/'roots.json').read_text())['roots'],roots=roots,
            train_initial=history[0]['train']['total'],train_final=history[-1]['train']['total'],train_best=best,
            train_plateau_reached=reason=='TRAIN_plateau',final_state_sha256=final_fingerprint,
            peak_cuda_bytes=torch.cuda.max_memory_allocated(),thermal_nu=None,
            limitations=['Fixed saved C realization; no independent C redraw validation.',
                'Operational TRAIN plateau does not prove global optimum.',
                'Single source seed, finite chains/sizes; numerical roots are not certified physical critical points.'])
        write_json(out/'summary.json',summary);status(current='reporting');report_module.render(out);check()
        for p,h in SOURCE_PINS.items():
            if file_fingerprint(SOURCE/p)!=h:raise ValueError('Source modified')
        if any(file_fingerprint(p)!=h for p,h in pins.items()):raise ValueError('Implementation modified')
        public=[dict(path=str(p.relative_to(out)),sha256=file_fingerprint(p),bytes=p.stat().st_size) for p in sorted(out.rglob('*'))
            if p.is_file() and p.suffix in ('.json','.png','.npz') and p.name not in ('status.json','job.json')]
        status(status='complete',current='complete',workflow_complete=True,returncode=0,public_files=public)
        return 0
    except Exception as exc:
        status(status='budget_hold' if isinstance(exc,BudgetHold) else 'failed',error=str(exc),traceback=traceback.format_exc(),
            returncode=3 if isinstance(exc,BudgetHold) else 1,requires_user_action=True)
        raise
