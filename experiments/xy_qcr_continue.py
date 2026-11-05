"""Joint XY continuation: preserve data/model/Adam, stop on TRAIN closure plateau.

The original checkpoint omitted RNG states. This explicit continuation therefore
uses new pinned sampling streams, and saves both states for future exact resumes.
No TEST, temperature correlation, numerical root, or BKT reference selects an end.
"""
import copy
import json
import os
from pathlib import Path
import time
import traceback
from types import SimpleNamespace
import numpy as np
import torch
from experiments import phi4_hopping_qcr as base
from experiments import xy_qcr as original
from experiments.phi4_hopping_cnn_qcr import resource_snapshot, GPU_UUID, root_diagnostics
from recurrent_rg.xy_qcr import XYQCR, independent_step, circular_information, wrap
from recurrent_rg.phi4_qcr import sample_regions

RECIPE='xy2d_b3_local_g1_joint_train_plateau_continue_v1'
SOURCE=Path('/mnt/nas-new/home/yange/ham/results/xy2d_b3_local_g1_seed91_gpu3_vm225_20261104_v1')
CKPT_SHA='fad87f1216799ddcfbe600f8f923d0486c827869159eb67681ed632482d85362'
DATA_SHA='2ec7177aa8c7e617f2889483841764c96cbe978a2bb459e16f095657b9e7bad1'


def preset():
    return dict(schema_version=5,recipe=RECIPE,source=str(SOURCE),source_step=1250,
        source_checkpoint_sha256=CKPT_SHA,data_sha256=DATA_SHA,
        inherited_training=original.preset(),max_extra_steps=6000,minimum_extra_steps=2000,
        probe_interval=250,plateau_window_probes=4,plateau_rtol=.05,
        plateau_recent_cv=.15,plateau_confirmations=3,
        stopping_metrics=['weighted_closure','normalized_closure'],
        batch_rng_seed=202611041,field_rng_seed=202611042,
        probe_seeds=[91404,91405],probe_samples=32,
        hard_seconds=3600,max_cuda_bytes=8*1024**3,max_output_bytes=2*1024**3)


def validate_config(path):
    c=json.loads(Path(path).read_text())
    if c!=preset():raise ValueError('Exact pinned joint XY continuation required')
    return c


def restore(payload,device):
    if payload.get('recipe')!=original.RECIPE or payload.get('step')!=1250:
        raise ValueError('Wrong source recipe or step')
    model=XYQCR(seed=91).to(device);model.load_state_dict(payload['model'],strict=True)
    opts=original.optimizers(model)
    for n,o in opts.items():o.load_state_dict(payload['optimizers'][n])
    base.optimizer_step_counts(opts,1250)
    for n,o in opts.items():
        if any(g['lr']!=original.preset()['learning_rates'][n] for g in o.param_groups):
            raise ValueError('Changed source learning rate')
    if base.module_fingerprints(model)!=payload['manifest']['expected_final_sha256']:
        raise ValueError('Source state fingerprints differ')
    return model,opts


@torch.no_grad()
def closure_statistics(net,g):
    advance=lambda z:z+net(z)
    preds=[advance(g[...,0,:]),advance(advance(g[...,0,:])),advance(g[...,1,:])]
    targets=[g[...,1,:],g[...,2,:],g[...,2,:]]
    mse=torch.stack([(p-t).square().sum(-1).mean() for p,t in zip(preds,targets)])
    std=g.flatten(0,1).std(0,unbiased=False).square().mean(-1).sqrt()
    variances=torch.stack([std[1]**2,std[2]**2,std[2]**2])
    if float(variances.min())<1e-14:raise FloatingPointError('Collapsed TRAIN probe targets')
    normalized=mse/variances
    weighted=lambda z:.5*(z[0]+.5*(z[1]+z[2]))
    return dict(weighted_closure=float(10*weighted(mse)),normalized_closure=float(weighted(normalized)),
        path_rmse=mse.sqrt().cpu().tolist(),path_nrmse=normalized.sqrt().cpu().tolist(),
        g_std=std.cpu().tolist(),g_mean=g.mean((0,1)).cpu().tolist())


@torch.no_grad()
def fixed_panel(model,data,device,c):
    """TRAIN-only, two fixed C draws; no RNG mutation or evaluation labels."""
    x=data.configurations[data.train_indices,:c['probe_samples']].to(device=device,dtype=torch.double)
    half=x.shape[1]//2;gs=[]
    for seed in c['probe_seeds']:
        levels=model.coarse_grain(x,torch.Generator(device=device).manual_seed(seed))
        gs.append(torch.stack([.5*(model.Q(v[:,:half])+model.Q(v[:,half:])) for v in levels],1))
    blocks,_=sample_regions(x,128,torch.Generator(device=device).manual_seed(91406))
    means=model.C.mean_blocks(blocks)
    return torch.stack(gs),means


def plateau_reached(rows,c):
    """Flat adjacent TRAIN windows, for BOTH absolute and normalized errors."""
    w=c['plateau_window_probes'];confirm=c['plateau_confirmations']
    if len(rows)<2*w+confirm-1:return False
    for offset in range(confirm):
        selected=rows[:len(rows)-offset] if offset else rows
        if selected[-1]['extra_steps']<c['minimum_extra_steps']:return False
        for key in c['stopping_metrics']:
            old=np.array([r[key] for r in selected[-2*w:-w]],float)
            new=np.array([r[key] for r in selected[-w:]],float)
            if not np.isfinite(np.r_[old,new]).all() or old.mean()<=0 or new.mean()<=0:return False
            if abs(new.mean()/old.mean()-1)>c['plateau_rtol']:return False
            if new.std()/new.mean()>c['plateau_recent_cv']:return False
    return True


def plot_progress(rows,out):
    from scripts.report_phi4_hopping_qcr import _setup_plot
    plt=_setup_plot(None);fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    steps=[r['step'] for r in rows]
    for ax,key,title in zip(axes[:2],('weighted_closure','normalized_closure'),('固定 TRAIN 绝对闭环损失','固定 TRAIN 归一化闭环误差')):
        ax.plot(steps,[r[key] for r in rows],'o-');ax.set(xlabel='累计 Q/C/R 更新次数',ylabel=key,title=title);ax.grid(alpha=.2)
    for k in range(3):
        rr=[r for r in rows if 'target_drift_nrmse' in r]
        axes[2].plot([r['step'] for r in rr],[r['target_drift_nrmse'][k] for r in rr],'o-',label=f'k={k}')
    axes[2].set(xlabel='累计 Q/C/R 更新次数',ylabel='相邻评估编码变化 / 上次标准差',title='固定物理输入与 C 随机数的目标漂移')
    axes[2].legend();axes[2].grid(alpha=.2);fig.savefig(out/'continuation_progress.png',dpi=160);plt.close(fig)


def run(config_path,output):
    c=validate_config(config_path);out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();extra=0;job=dict(status='running',workflow_complete=False,current='admission',
        pid=os.getpid(),returncode=None,error=None,requires_user_action=False,source_step=1250)
    def status(**kw):
        job.update(kw,actual_steps=1250+extra,extra_steps=extra,elapsed_seconds=time.monotonic()-started)
        base.write_json(out/'status.json',job);base.write_json(out/'job.json',job)
        print(json.dumps(job,ensure_ascii=False),flush=True)
    def check():
        if torch.cuda.is_initialized() and max(torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved())>c['max_cuda_bytes']:
            raise base.BudgetHold('Own GPU memory budget exceeded')
        if sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>c['max_output_bytes']:raise base.BudgetHold('Output budget exceeded')
    try:
        if os.uname().nodename!='thinklab-105-225' or os.getpriority(os.PRIO_PROCESS,0)<10 or os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU_UUID:
            raise RuntimeError('Wrong host/priority/device')
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        from scripts.run_ising_highdim import source_hashes
        code=Path(__file__).resolve().parents[1];pins=source_hashes(code);config_sha=base.sha256_file(config_path)
        source=Path(c['source']);ckpt=source/'training_step_1250.pt';data_path=source/'data/angles.npz'
        source_job=json.loads((source/'job.json').read_text())
        if source_job['status']!='complete' or source_job['returncode']!=0 or not source_job['workflow_complete']:raise ValueError('Source incomplete')
        if base.sha256_file(ckpt)!=CKPT_SHA or base.sha256_file(data_path)!=DATA_SHA:raise ValueError('Source checkpoint/data changed')
        meta=json.loads((source/'data/metadata.json').read_text())
        if meta['angles_sha256']!=DATA_SHA:raise ValueError('Source metadata data mismatch')
        configurations=np.load(data_path,allow_pickle=False)['configurations']
        data=SimpleNamespace(configurations=torch.from_numpy(configurations),labels=np.asarray(meta['labels']),splits=np.asarray(meta['splits']),
            cell_ids=np.asarray(meta['cell_ids']),train_indices=np.flatnonzero(np.asarray(meta['splits'])=='train'),test_indices=np.flatnonzero(np.asarray(meta['splits'])=='test'))
        (out/'data').mkdir();base.write_json(out/'data/metadata.json',meta)
        base.write_json(out/'config.json',c);base.write_json(out/'plan.json',dict(config=c,source_sha256=pins))
        status(current='gpu_admission');admission=[resource_snapshot()];time.sleep(10);admission.append(resource_snapshot())
        device=torch.device('cuda:0');receipt=base._device_receipt(device,GPU_UUID)
        torch.cuda.set_per_process_memory_fraction(c['max_cuda_bytes']/torch.cuda.get_device_properties(device).total_memory)
        payload=torch.load(ckpt,map_location='cpu',weights_only=False)
        payload['manifest']['expected_final_sha256']=json.loads((source/'summary.json').read_text())['final_module_sha256']
        model,opts=restore(payload,device);del payload
        rng=torch.Generator().manual_seed(c['batch_rng_seed']);fg=torch.Generator(device=device).manual_seed(c['field_rng_seed'])
        manifest=dict(recipe=RECIPE,config=c,source_sha256=pins,config_sha256=config_sha,source=str(source),
            source_checkpoint_sha256=CKPT_SHA,data_sha256=DATA_SHA,admission=admission,device=receipt,
            optimized_modules=['Q','C','R'],gradient_routing='Unchanged simultaneous independent gradients',
            optimizer='All original Adam moments, steps and learning rates restored',
            rng_policy='Source omitted RNG states; new pinned batch/C streams, saved in continuation checkpoints',
            training_recipe_changed=False,stopping_uses='TRAIN fixed physical inputs and two fixed C noise draws only',
            stopping_interpretation='Operational closure plateau; does not prove stationary Q/C, optimum, or BKT structure',
            initial_sha256=base.module_fingerprints(model),labels_used_in_training=False,thermal_nu=None,BKT_certified=False)
        base.write_json(out/'manifest.json',manifest)
        history=list(json.loads((source/'history.json').read_text()));rows=[];previous=None
        def evaluate_probe():
            nonlocal previous
            frozen=base.state_fingerprint(model);rstate=rng.get_state().clone();fstate=fg.get_state().clone()
            g,cm=fixed_panel(model,data,device,c);row=dict(step=1250+extra,extra_steps=extra,elapsed_seconds=time.monotonic()-started,
                **closure_statistics(model.R,g),decoder_norms=model.Q.decoder.weight.norm(dim=0).detach().cpu().tolist())
            if previous is not None:
                pg,pc,pr,ps=previous
                drift=(g-pg).square().mean((0,1,3)).sqrt()/pg.flatten(0,1).std(0,unbiased=False).square().mean(-1).sqrt().clamp_min(1e-12)
                row.update(target_drift_nrmse=drift.cpu().tolist(),C_mean_angle_drift_radians=float(wrap(cm-pc).square().mean().sqrt()),
                    previous_R_on_current_QC=closure_statistics(pr,g),previous_R_on_previous_QC=ps)
            previous=(g.detach(),cm.detach(),copy.deepcopy(model.R).eval(),closure_statistics(model.R,g))
            if frozen!=base.state_fingerprint(model) or not torch.equal(rstate,rng.get_state()) or not torch.equal(fstate,fg.get_state()):raise RuntimeError('Probe changed training state')
            rows.append(row);base.write_json(out/'train_probes.json',rows)
            status(last_probe=row,train_plateau_reached=plateau_reached(rows,c))
        def save():
            torch.save(dict(recipe=RECIPE,step=1250+extra,extra_steps=extra,model=model.state_dict(),manifest=manifest,
                optimizers={k:v.state_dict() for k,v in opts.items()},batch_rng_state=rng.get_state(),field_rng_state=fg.get_state()),out/f'training_step_{1250+extra}.pt')
        evaluate_probe();status(current='joint_training');reason='update_limit'
        for extra in range(1,c['max_extra_steps']+1):
            model.train();metrics=independent_step(model,original.draw(data,rng,device),opts,fg)
            if extra%50==0:
                history.append(dict(step=1250+extra,extra_steps=extra,elapsed_seconds=time.monotonic()-started,**metrics));base.write_json(out/'history.json',history)
                status(last_losses={k:metrics[k] for k in ('q_total','c_total','r_total')});check()
            if extra%c['probe_interval']==0:
                evaluate_probe();save()
                if plateau_reached(rows,c):reason='TRAIN_closure_plateau';break
                if time.monotonic()-started>c['hard_seconds']-240:reason='time_limit';break
        if rows[-1]['extra_steps']!=extra:evaluate_probe();save()
        clocks=base.optimizer_step_counts(opts,1250+extra)
        for p in model.parameters():p.grad=None
        frozen=base.state_fingerprint(model);status(current='frozen_evaluation',stop_reason=reason)
        arrays=base.evaluate_model(model,data,device,budget_check=check);roots=root_diagnostics(model,arrays);roots.pop('critical_kappa',None)
        with torch.no_grad():
            x=data.configurations[data.train_indices[:16],:64].to(device=device,dtype=torch.double)
            b,e=sample_regions(x,128,torch.Generator(device=device).manual_seed(91919));means=model.C.mean_blocks(b)[...,0]
            mi=dict(paired=float(circular_information(means,e)),within_ensemble_mismatch=float(circular_information(means,e,shuffle=True)),estimator=original.preset()['information_estimator'])
        if frozen!=base.state_fingerprint(model) or any(p.grad is not None for p in model.parameters()):raise RuntimeError('Evaluation mutated model')
        np.savez(out/'evaluation_arrays.npz',**arrays);base.write_json(out/'roots.json',roots)
        base.write_json(out/'summary.json',dict(status='complete',actual_steps=1250+extra,extra_steps=extra,optimizer_steps=clocks,
            stop_reason=reason,train_plateau_reached=reason=='TRAIN_closure_plateau',closure=base.closure_metrics(arrays),information_diagnostic=mi,
            final_module_sha256=base.module_fingerprints(model),peak_cuda_bytes=torch.cuda.max_memory_allocated(),labels_used_in_training=False,
            thermal_nu=None,BKT_certified=False,equilibrium_certified=False,elapsed_seconds=time.monotonic()-started))
        status(current='reporting');from scripts.report_xy_qcr import render
        render(out);plot_progress(rows,out/'report');check()
        if source_hashes(code)!=pins or base.sha256_file(config_path)!=config_sha or base.sha256_file(ckpt)!=CKPT_SHA or base.sha256_file(data_path)!=DATA_SHA:raise RuntimeError('Source/config/data fingerprint changed')
        public=[dict(path=str(p.relative_to(out)),sha256=base.sha256_file(p),bytes=p.stat().st_size) for p in sorted(out.rglob('*'))
            if p.is_file() and p.suffix in ('.json','.npz','.png') and p.name not in ('job.json','status.json')]
        status(status='complete',current='complete',workflow_complete=True,returncode=0,public_files=public,
            stop_reason=reason,train_plateau_reached=reason=='TRAIN_closure_plateau')
    except Exception as exc:
        status(status='budget_hold' if isinstance(exc,base.BudgetHold) else 'failed',returncode=3 if isinstance(exc,base.BudgetHold) else 1,
            error=str(exc),requires_user_action=True,traceback=traceback.format_exc());raise
