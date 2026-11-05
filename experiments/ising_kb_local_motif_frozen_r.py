"""Explicit fixed-trajectory R-only diagnostic of the supervised local-motif Q run.

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

RECIPE='ising2d_KB_local_motif_frozen_QC_fullbatch_R_v1'
SOURCE=Path('/mnt/nas-new/home/yange/ham/results/ising2d_KB_local_motif_supervised_matched1250_seed91_gpu3_vm225_20261104_v1')
SOURCE_PINS={'training_step_1250.pt': '7d67cd7e992a928c64f928d58ea5de7e6ff1be3ae33f2cc8a63d918deb422fc3', 'manifest.json': 'a03341b8ff9b90bac4088769ac4496bba83426c8eaad1ad8756a6fdaa1b5ec72', 'summary.json': '98be2cd53ba20bba10c90799106837956810774b3f8f63d1013b3c43fd9248be', 'evaluation_arrays.npz': '50dcade2b022e996a6721db4ac65e110e153a45c354a5df1215e902de34bead0', 'roots.json': '111b79ec3ef73d852d6d5021fa81159f065445ad47203eb8e9f818fb14a92a40', 'status.json': '623776e444384f94ac9fb0e3e886a1eed3ca19588126f26bb421e62e75e9cb2f'}


def preset():
    return dict(schema_version=5,recipe=RECIPE,source=str(SOURCE),source_sha256=SOURCE_PINS,
        source_joint_updates=1250,learning_rate=.003,max_extra_R_updates=60000,evaluate_every=1000,
        minimum_updates=10000,plateau_window=5000,plateau_rtol=.01,plateau_confirmations=3,
        stability_rtol=.05,hard_seconds=1200,training_seconds=1020,
        train_chains=474,test_chains=240,target_policy='saved_half_subset_direct_g_full_TRAIN',
        labels_used_in_training=False,closure_objective='all_start_anchor',closure_scale_weights=None,
        final_root_search='original bounded 7x7 starts plus each original root and four fixed offsets; never used to stop training')


def validate_config(path):
    value=json.loads(Path(path).read_text())
    if value!=preset():raise ValueError('This explicit R-only diagnostic has a pinned protocol')
    return value


from experiments.ising_kb_frozen_r import (freeze_and_continue,validate_loaded_state,
    r_loss,update,training_targets,predictions)


def supplement_root_search(model,arrays,roots,source_roots,device,check=lambda:None):
    """Fixed source-root seeds supplement the global search; no root-based selection."""
    from scipy.optimize import least_squares
    from scipy.spatial import ConvexHull
    from recurrent_rg.phi4_ru_flow import full_map_and_jacobian,classify_eigenvalues,numerical_rank
    bounds=np.asarray(arrays['flow_bounds']);center=bounds.mean(1);half=(bounds[:,1]-bounds[:,0])/2
    train=arrays['train_direct'][:,:2].reshape(-1,2);hull=ConvexHull((train-center)/half)
    tol=1e-8*max(1.,float(np.linalg.norm(train.max(0)-train.min(0))))
    attempts=[];offsets=np.array([[0,0],[.05,0],[-.05,0],[0,.05],[0,-.05]])
    def fun(z):
        check();point=center+half*z
        with torch.no_grad():value=model.advance(torch.tensor(point[None],device=device,dtype=torch.float64)).cpu().numpy()[0]
        return (value-point)/half
    def jac(z):
        check();_,j=full_map_and_jacobian(model,(center+half*z)[None],device)
        return (j[0]-np.eye(2))*half[None,:]/half[:,None]
    for index,source in enumerate(source_roots):
        for offset in offsets:
            start=np.clip((np.asarray(source['g'])-center)/half+offset,-1+1e-10,1-1e-10)
            fit=least_squares(fun,start,jac=jac,bounds=(-np.ones(2),np.ones(2)),max_nfev=64,xtol=1e-11,ftol=1e-11,gtol=1e-11)
            point=center+half*fit.x;residual=float(np.linalg.norm(fun(fit.x)*half));accepted=residual<=tol
            attempts.append(dict(source_index=index,source_g=source['g'],normalized_offset=offset.tolist(),
                start=(center+half*start).tolist(),candidate=point.tolist(),residual=residual,accepted=accepted,nfev=int(fit.nfev)))
            if not accepted or any(np.linalg.norm((point-np.asarray(r['g']))/half)<1e-5 for r in roots):continue
            _,j=full_map_and_jacobian(model,point[None],device);eig=np.linalg.eigvals(j[0]);name,nc,ne=classify_eigenvalues(eig)
            distance=np.linalg.norm(train-point,axis=1);near=train[np.argsort(distance,kind='stable')[:16]]
            roots.append(dict(g=point.tolist(),residual=residual,full_DR=j[0].tolist(),
                eigenvalues=[dict(real=float(v.real),imag=float(v.imag),modulus=float(abs(v))) for v in eig],
                classification=name,contracting_directions=nc,expanding_directions=ne,
                within_train_hull=bool(np.all(((point-center)/half)@hull.equations[:,:2].T+hull.equations[:,2]<=1e-9)),
                within_search_bounds=True,nearest_train_distance=float(distance.min()),local_rank=numerical_rank(near),
                local_rank_points=len(near),search_source='fixed_source_root_seeds',physical_critical_point_certified=False,thermal_nu=None))
    roots.sort(key=lambda r:tuple(r['g']))
    return dict(attempts=attempts,maximum_starts=5*len(source_roots),max_nfev=64,
        normalized_offsets=offsets.tolist(),residual_tolerance=tol,
        qualification='A root reached from an old root is not proof of continuous branch identity; failure to find a root is not proof of absence.')


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
        if loaded.payload['epoch']!=1250 or cfg.model.theory_dim!=2 or cfg.model.block_factor!=3 or cfg.model.hidden!=96 or cfg.model.q_encoder!='local_motif' or cfg.training.max_rg_steps!=2 or cfg.loss.closure_objective!='all_start_anchor' or cfg.loss.closure_scale_weights is not None or cfg.loss.weights.commutator!=5 or cfg.loss.rollout_weight!=5:
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
        import scripts.report_ising_kb_local_motif_frozen_r as report_module
        pins={str(p):file_fingerprint(p) for p in loaded_source_files(Path(__file__).resolve().parents[1])}
        source_roots=json.loads((SOURCE/'roots.json').read_text())['roots']
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
        flow_model=SimpleNamespace(advance=model.advance_theory)
        roots,flow=bounded_flow(flow_model,arrays,device,budget_check=check)
        supplemental=supplement_root_search(flow_model,arrays,roots,source_roots,device,check)
        flow['supplemental_source_root_search']=supplemental
        for k in ('one_step_g','rollout_g','all_start_g','flow_g','flow_R','flow_displacement'):arrays['source_'+k]=base[k].copy()
        np.testing.assert_array_equal(arrays['flow_g'],base['flow_g'])
        if state_fingerprint(model)!=final_fingerprint:raise AssertionError('Evaluation mutated model')
        np.savez(out/'evaluation_arrays.npz',**arrays);write_json(out/'roots.json',dict(roots=roots,flow_analysis=flow))
        summary=dict(status='complete',recipe=RECIPE,reason=reason,source_joint_updates=1250,actual_extra_R_updates=actual,
            R_Adam_steps=clocks,additional_Q_updates=0,additional_C_updates=0,frozen_before_sha256=qc_before,frozen_after_sha256=qc_after,
            QC_gradients_none=True,source_predictions_replayed=True,before_closure=closure_metrics(base),after_closure=closure_metrics(arrays),
            source_roots=source_roots,roots=roots,root_search=flow,
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
