"""Continue the completed 60k R-only run to an operational TRAIN plateau.

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

RECIPE='ising2d_KB_local_motif_R_plateau_continue_v1'
SOURCE=Path('/mnt/nas-new/home/yange/ham/results/ising2d_KB_local_motif_frozen_R_seed91_gpu3_vm225_20261104_v1')
SOURCE_PINS={
    'latest.pt':'d25a15ac2eb8dec2cdc389119b0cef6da2d036405f37c89873c7e4f704f339d1',
    'manifest.json':'c0dc3a089873532da5630f241bd39b8ff29745eaae6f318ec931ecceb01ae4f2',
    'summary.json':'c1c2071987e0eb5fc5237482cb4cd6ab35c652ffd4251a10d9f00c9e9b284d6c',
    'evaluation_arrays.npz':'58ba21ce84a6edea6b4a19d29ccf8f0586d789e4e53051dbddf655c9d52fec95',
    'roots.json':'d7f4b73db74c2166b183cfe42ae55520a567729ab4c54dadcf87b6e61e02eae2',
    'history.json':'8f208a78174980ab01e7f8a16098ccbaf5c79b8712e8c58a18962e0446c2ddf5',
    'status.json':'6ce1fbbb5942203d78bc6aeee8fc0e4aef76ec19cfd93615f9e64b5b7432e1d5'}
SOURCE_R_UPDATES=60000
SOURCE_ADAM_STEPS=61250



def preset():
    return dict(schema_version=5,recipe=RECIPE,source=str(SOURCE),source_sha256=SOURCE_PINS,
        source_joint_updates=1250,source_R_updates=SOURCE_R_UPDATES,source_Adam_steps=SOURCE_ADAM_STEPS,
        learning_rate=.003,max_extra_R_updates=600000,evaluate_every=1000,late_checkpoints=6,
        minimum_updates=10000,plateau_window=5000,plateau_rtol=.01,plateau_confirmations=3,
        stability_rtol=.05,hard_seconds=7200,training_seconds=6600,
        train_chains=474,test_chains=240,target_policy='saved_half_subset_direct_g_full_TRAIN',
        labels_used_in_training=False,closure_objective='all_start_anchor',closure_scale_weights=None,
        final_root_search='same bounded 7x7 search and fixed 60k plus original joint root seeds at each of the last six evaluations; never used to stop training')


def validate_config(path):
    value=json.loads(Path(path).read_text())
    if value!=preset():raise ValueError('This explicit R-only diagnostic has a pinned protocol')
    return value


from experiments.ising_kb_frozen_r import (freeze_and_continue,validate_loaded_state,
    r_loss,update,training_targets,predictions)


from experiments.ising_kb_local_motif_frozen_r import supplement_root_search


def inherited_history(path, replayed_loss):
    rows=json.loads(Path(path).read_text())
    if not rows or rows[-1]['extra_R_updates']!=SOURCE_R_UPDATES:
        raise ValueError('Source history endpoint differs')
    if any(b['extra_R_updates']<=a['extra_R_updates'] for a,b in zip(rows,rows[1:])):
        raise ValueError('Non-increasing source history')
    np.testing.assert_allclose(replayed_loss,rows[-1]['train']['total'],rtol=1e-12,atol=1e-12)
    return copy.deepcopy(rows)


def root_stability(snapshots,bounds,source_roots):
    """Location-based matching, with explicit ambiguity flags; not a branch proof."""
    from scipy.optimize import linear_sum_assignment
    width=np.diff(np.asarray(bounds),axis=1).ravel();threshold=.05
    tracks=[[i] for i in range(len(snapshots[-1]['roots']))]
    ambiguities=[False for _ in tracks]
    for at in range(len(snapshots)-1,0,-1):
        previous=snapshots[at-1]['roots'];current=snapshots[at]['roots'];mapping={}
        if previous and current:
            distances=np.linalg.norm((np.array([r['g'] for r in current])[:,None]-np.array([r['g'] for r in previous])[None])/width,axis=2)
            ii,jj=linear_sum_assignment(distances)
            mapping={int(i):(int(j),bool(np.sum(distances[i]<=threshold)>1 or np.sum(distances[:,j]<=threshold)>1)) for i,j in zip(ii,jj) if distances[i,j]<=threshold}
        for n,track in enumerate(tracks):
            found=mapping.get(track[-1]);track.append(found[0] if found else None)
            if found:ambiguities[n]|=found[1]
    rows=[]
    for i,(reverse,ambiguous) in enumerate(zip(tracks,ambiguities)):
        indices=list(reversed(reverse));points=[snapshot['roots'][j] if j is not None else None for snapshot,j in zip(snapshots,indices)]
        present=[r for r in points if r is not None]
        displacement=max((float(np.linalg.norm((np.asarray(r['g'])-np.asarray(present[-1]['g']))/width)) for r in present),default=0.)
        rows.append(dict(final_index=i,g=snapshots[-1]['roots'][i]['g'],matched_indices=indices,
            present_all=all(r is not None for r in points),ambiguous_matching=bool(ambiguous),
            classifications=[r['classification'] if r else None for r in points],
            same_classification=len({r['classification'] for r in present})==1,
            max_normalized_distance_from_final=displacement,points=points))
    reference=min(source_roots,key=lambda r:np.linalg.norm(np.asarray(r['g'])-[.0795104677,-.0773168325]))
    # Anchor at the final central root, then use the same backward assignments as
    # the other roots. A greedy jump from the old 60k position can instead pick
    # a nearby transient saddle at an intermediate checkpoint.
    final=snapshots[-1]['roots']
    anchor=min(range(len(final)),key=lambda i:np.linalg.norm((np.asarray(final[i]['g'])-np.asarray(reference['g']))/width)) if final else None
    central=[]
    for at,snapshot in enumerate(snapshots):
        index=rows[anchor]['matched_indices'][at] if anchor is not None else None
        point=snapshot['roots'][index] if index is not None else None
        central.append(dict(extra_R_updates=snapshot['extra_R_updates'],root_index=index,point=point))
    return dict(snapshots=snapshots,final_root_tracks=rows,central_reference=reference,central_final_index=anchor,central_track=central,
        central_anchor_policy="Final root nearest the source central position, then backward location assignment; no branch identity across the 60k-to-final gap is asserted",
        normalized_matching_threshold=threshold,normalization='full fixed TRAIN search-box width',
        qualification='Nearest-location assignment is diagnostic, not certified branch continuation; bounded starts may miss roots. No root information determines stopping.')


def run(config,output):
    protocol=validate_config(config);started=time.monotonic();actual=0
    if os.uname().nodename!='thinklab-105-225' or os.getpriority(os.PRIO_PROCESS,0)<10 or os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU_UUID:
        raise RuntimeError('Wrong host, priority, or GPU mask')
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    job=dict(status='running',workflow_complete=False,pid=os.getpid(),returncode=None,error=None,requires_user_action=False)
    def status(**kw):
        job.update(kw,new_R_updates=actual,extra_R_updates=SOURCE_R_UPDATES+actual,elapsed_seconds=time.monotonic()-started)
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
        loaded=load_run(SOURCE/'latest.pt');cfg=RunConfig.from_dict(loaded.config)
        if loaded.payload['epoch']!=SOURCE_ADAM_STEPS or loaded.payload['extra_R_updates']!=SOURCE_R_UPDATES or cfg.model.theory_dim!=2 or cfg.model.block_factor!=3 or cfg.model.hidden!=96 or cfg.model.q_encoder!='local_motif' or cfg.training.max_rg_steps!=2 or cfg.loss.closure_objective!='all_start_anchor' or cfg.loss.closure_scale_weights is not None or cfg.loss.weights.commutator!=5 or cfg.loss.rollout_weight!=5:
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
        import scripts.report_ising_kb_r_plateau_continue as report_module
        pins={str(p):file_fingerprint(p) for p in loaded_source_files(Path(__file__).resolve().parents[1])}
        source_roots=json.loads((SOURCE/'roots.json').read_text())['roots']
        original_roots=json.loads((SOURCE/'summary.json').read_text())['source_roots']
        fixed_root_seeds=source_roots+original_roots
        continued_cfg=replace(cfg,training=replace(cfg.training,train_modules=('R',),learning_rate=.003,epochs=SOURCE_ADAM_STEPS+protocol['max_extra_R_updates']))
        manifest=dict(recipe=RECIPE,protocol=protocol,config=continued_cfg.to_dict(),source_config=cfg.to_dict(),source=str(SOURCE),
            source_joint_updates=1250,source_R_updates=SOURCE_R_UPDATES,source_Adam_steps=SOURCE_ADAM_STEPS,source_sha256=SOURCE_PINS,code_sha256=pins,admission=admission,device=device_receipt,
            loaded_state_verification=loaded_state_receipt,
            labels_used_in_training=False,source_Q_was_supervised=True,test_used_in_training=False,
            optimized_modules=['R'],frozen_modules=['Q','C'],frozen_before_sha256=qc_before,
            optimizer_policy='Inherit R Adam weights, moments and clock61250; lr.003 constant; no reset/scheduler',
            target_policy='474 TRAIN physical-chain latent trajectories, one saved C realization, mean of two half-subset Q outputs; full batch',
            change_from_source='None in targets, objective, optimizer or learning rate; extend stopping horizon only',
            objective='5 e01 +2.5 e02 +2.5 e12, each error is mean coordinate-sum squared distance',
            endpoint_policy='Final TRAIN plateau; resource caps explicitly marked inconclusive if reached first; no TEST/root-based selection',
            physical_data_opened=False,MI_evaluated=False,thermal_nu=None)
        write_json(out/'manifest.json',manifest);write_json(out/'plan.json',protocol)
        with torch.no_grad():initial_loss=float(r_loss(model,panel))
        history=inherited_history(SOURCE/'history.json',initial_loss);best=history[-1]['best_train_closure'];reason='update_cap'
        late=[]
        def evaluate_train():
            nonlocal best
            with torch.no_grad():value=float(r_loss(model,panel))
            if not np.isfinite(value):raise FloatingPointError('Nonfinite TRAIN evaluation')
            best=min(best,value)
            history.append(dict(extra_R_updates=SOURCE_R_UPDATES+actual,new_R_updates=actual,train={'total':value},best_train_closure=best,elapsed_seconds=time.monotonic()-started))
            write_json(out/'history.json',history);status(current='R_training',train_loss=value)
            late.append((SOURCE_R_UPDATES+actual,{k:v.detach().cpu().clone() for k,v in model.theory_rg.state_dict().items()}))
            if len(late)>protocol['late_checkpoints']:late.pop(0)
        def save_checkpoint():
            torch.save(dict(epoch=SOURCE_ADAM_STEPS+actual,extra_R_updates=SOURCE_R_UPDATES+actual,new_R_updates=actual,model=qcr_state_dict(model),metadata=manifest,
                optimizers={'R':opt.state_dict()}),out/'latest.pt')
        late.append((SOURCE_R_UPDATES,{k:v.detach().cpu().clone() for k,v in model.theory_rg.state_dict().items()}))
        write_json(out/'history.json',history);save_checkpoint();status(current='R_training',train_loss=initial_loss)
        for step in range(1,protocol['max_extra_R_updates']+1):
            check()
            if time.monotonic()-started>protocol['training_seconds']:reason='training_time_cap';break
            update(model,panel,opt);actual=step
            if actual%protocol['evaluate_every']==0 or actual==protocol['max_extra_R_updates']:
                evaluate_train();save_checkpoint()
                if plateau_reached(history,minimum=protocol['minimum_updates'],window=protocol['plateau_window'],rtol=protocol['plateau_rtol'],confirmations=protocol['plateau_confirmations'],stability_rtol=protocol['stability_rtol']):
                    reason='TRAIN_plateau';break
        if history[-1]['extra_R_updates']!=SOURCE_R_UPDATES+actual:evaluate_train()
        save_checkpoint();clocks=optimizer_step_counts({'R':opt},SOURCE_ADAM_STEPS+actual)
        qc_after={n:state_fingerprint(m) for n,m in [('Q',model.theory_encoder),('C',model.rg_kernel)]}
        if qc_before!=qc_after or any(p.grad is not None for mod in (model.theory_encoder,model.rg_kernel) for p in mod.parameters()):raise AssertionError('Q/C changed or received gradients')
        for p in model.parameters():p.grad=None
        final_fingerprint=state_fingerprint(model)
        status(current='heldout_evaluation_and_bounded_roots');arrays=predictions(model,base)
        snapshots=[]
        for step,state in late:
            model.theory_rg.load_state_dict(state)
            diagnostic=predictions(model,base)
            flow_model=SimpleNamespace(advance=model.advance_theory)
            roots,flow=bounded_flow(flow_model,diagnostic,device,budget_check=check)
            flow['supplemental_source_root_search']=supplement_root_search(flow_model,diagnostic,roots,fixed_root_seeds,device,check)
            snapshots.append(dict(extra_R_updates=step,roots=copy.deepcopy(roots),search=copy.deepcopy(flow)))
            torch.save(dict(cumulative_R_only_updates=step,R=state),out/f'late_R_{step}.pt')
            status(current='late_checkpoint_roots',diagnostic_step=step)
        # Chronological loop ends with the actual final trained R, never best-loss selection.
        arrays=diagnostic
        stability=root_stability(snapshots,arrays['flow_bounds'],source_roots)
        write_json(out/'root_stability.json',stability)
        for k in ('one_step_g','rollout_g','all_start_g','flow_g','flow_R','flow_displacement'):arrays['source_'+k]=base[k].copy()
        np.testing.assert_array_equal(arrays['flow_g'],base['flow_g'])
        if state_fingerprint(model)!=final_fingerprint:raise AssertionError('Evaluation mutated model')
        np.savez(out/'evaluation_arrays.npz',**arrays);write_json(out/'roots.json',dict(roots=roots,flow_analysis=flow))
        summary=dict(status='complete',recipe=RECIPE,reason=reason,source_joint_updates=1250,source_R_updates=SOURCE_R_UPDATES,
            actual_extra_R_updates=SOURCE_R_UPDATES+actual,new_R_updates=actual,
            R_Adam_steps=clocks,additional_Q_updates=0,additional_C_updates=0,frozen_before_sha256=qc_before,frozen_after_sha256=qc_after,
            QC_gradients_none=True,source_predictions_replayed=True,before_closure=closure_metrics(base),after_closure=closure_metrics(arrays),
            source_roots=source_roots,roots=roots,root_search=flow,
            train_initial=initial_loss,original_joint_train_loss=history[0]['train']['total'],train_final=history[-1]['train']['total'],train_best=best,
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
