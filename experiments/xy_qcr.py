"""Fresh, bounded scalar XY trial with a periodic adapter and no labels in loss."""
import json
import os
from pathlib import Path
import time
import traceback
from types import SimpleNamespace
import numpy as np
import torch
from recurrent_rg.data.xy import sample_xy, observables, chain_diagnostics
from recurrent_rg.xy_qcr import XYQCR, independent_step, circular_information
from recurrent_rg.phi4_qcr import sample_regions
from experiments import phi4_hopping_qcr as base
from experiments.phi4_hopping_cnn_qcr import resource_snapshot, GPU_UUID, root_diagnostics

RECIPE='xy2d_b3_local_periodic_freegain_g1_seed91_v1'


def preset():
    return dict(schema_version=5,recipe=RECIPE,seed=91,physical_seed=20261104,
        theory_dim=1,lattice_sizes=[54,18,6],block_size=3,stride=3,
        temperatures=[.50,.60,.70,.78,.82,.86,.89,.92,.95,1.,1.06,1.15,1.25,1.4,1.6],
        physical_chains=4,samples_per_chain=128,burn_sweeps=2048,spacing_sweeps=16,
        steps=1250,system_batch=16,sample_batch=32,information_pairs=128,
        learning_rates=dict(Q=.001,C=.001,R=.003),r_updates_per_batch=1,
        decoder_column_constraint='free_after_initialization',supervision_weight=0.,
        loss=dict(base.LOSS_POLICY),C_noise_sigma=.25,
        information_estimator='circular_full_shell_kernel_smoothed_proxy_not_exact_histogram_MI',
        temperature_labels='generation_and_evaluation_only',
        hard_seconds=3600,max_cuda_bytes=8*1024**3,max_output_bytes=2*1024**3)


def validate_config(path):
    c=json.loads(Path(path).read_text())
    if c!=preset():raise ValueError('Exact explicit XY preset required')
    return c


def optimizers(model):
    return {n:torch.optim.Adam(getattr(model,n).parameters(),lr=lr) for n,lr in preset()['learning_rates'].items()}


def draw(data,rng,device):
    ids=torch.as_tensor(data.train_indices)[torch.randperm(len(data.train_indices),generator=rng)[:16]]
    samples=torch.randperm(data.configurations.shape[1],generator=rng)[:32]
    return data.configurations[ids][:,samples].to(device=device,dtype=torch.double)


def sampling(config,out,status,check):
    data_dir=out/'data';data_dir.mkdir()
    chains=[];labels=[];splits=[];cells=[];rows=[]
    # Compiled disposable timing pilot; no samples enter the formal dataset.
    sample_xy(1.,6,2,404,burn=2,spacing=1)
    tick=time.monotonic();sample_xy(2.,54,8,405,burn=64,spacing=16)
    per_sweep=(time.monotonic()-tick)/192
    projected=per_sweep*(config['burn_sweeps']+128*16)*60*1.5
    base.write_json(out/'sampling_timing.json',dict(disposable=True,estimated_seconds=projected))
    if projected>1500:raise base.BudgetHold('Conservative sampling projection exceeds1500s; no reduced sampling')
    for cell,T in enumerate(config['temperatures']):
        for rep in range(4):
            seed=int(np.random.SeedSequence([config['physical_seed'],54,cell,rep]).generate_state(1)[0])
            status(current='sampling',cell=cell,temperature=T,rep=rep,completed_chains=len(chains))
            angles=sample_xy(1/T,54,128,seed,burn=config['burn_sweeps'],spacing=config['spacing_sweeps'],ordered=bool(rep%2))
            obs=observables(angles,1/T);diag=chain_diagnostics(obs,1/T,54)
            row=dict(cell=cell,temperature=T,K=1/T,rep=rep,seed=seed,initial_state='cold' if rep%2 else 'hot',diagnostics=diag)
            rows.append(row);chains.append(angles[...,None]);labels.append([T,1/T]);cells.append(cell);splits.append('test' if cell%3==1 else 'train')
            base.write_json(data_dir/'sampling_progress.json',dict(completed_chains=len(chains),chains=rows))
            check()
    a=np.stack(chains);np.savez(data_dir/'angles.npz',configurations=a)
    data_sha=base.sha256_file(data_dir/'angles.npz')
    warnings=[]
    for row in rows:
        if row['diagnostics']['warning']:warnings.append(dict(cell=row['cell'],rep=row['rep'],kind='ESS_below32'))
    hc=[]
    for cell,T in enumerate(config['temperatures']):
        block=rows[cell*4:(cell+1)*4];item=dict(temperature=T)
        for name in ('energy','magnetization_squared','vortex_density'):
            values=np.array([r['diagnostics'][name]['mean'] for r in block]);ses=np.array([r['diagnostics'][name]['batch_se'] for r in block])
            diff=float(values[::2].mean()-values[1::2].mean());se=float(np.sqrt(np.sum(ses**2))/2)
            z=abs(diff)/se if se>1e-12 else (0. if abs(diff)<1e-12 else 1e12)
            item[name]=dict(hot_minus_cold=diff,combined_batch_se=se,z=z)
            if z>4:warnings.append(dict(cell=cell,kind='hot_cold_z_above4',observable=name,z=z))
        hc.append(item)
    meta=dict(status='complete',Hamiltonian='H/J=-sum_nn cos(theta_i-theta_j)',dimension=2,
        lattice_size=54,angles='continuous radians in [-pi,pi)',chains=rows,hot_cold=hc,warnings=warnings,
        independent_physical_chains=60,samples_per_chain=128,correlated_samples=True,equilibrium_certified=False,
        angles_sha256=data_sha,labels=labels,splits=splits,cell_ids=cells,labels_used_in_training=False)
    base.write_json(data_dir/'metadata.json',meta)
    # Gross unresolved mixing is a scientific sampling gate, never silently repaired.
    severe=any(r['diagnostics'][n]['ess']<8 for r in rows for n in ('energy','magnetization_squared')) or any(w.get('z',0)>8 for w in warnings)
    if severe:raise base.BudgetHold('Severe sampling diagnostic; retain data, require review before training')
    return SimpleNamespace(configurations=torch.from_numpy(a),labels=np.asarray(labels),splits=np.asarray(splits),cell_ids=np.asarray(cells),
        train_indices=np.flatnonzero(np.asarray(splits)=='train'),test_indices=np.flatnonzero(np.asarray(splits)=='test')),meta


@torch.no_grad()
def probe(model,data,device):
    x=data.configurations[data.train_indices, :16].to(device=device,dtype=torch.double)
    levels=model.coarse_grain(x,torch.Generator(device=device).manual_seed(91404))
    g=torch.stack([.5*(model.Q(f[:,:8])+model.Q(f[:,8:])) for f in levels],1)
    from recurrent_rg.losses import all_start_anchor_closure_loss
    loss,_,_=all_start_anchor_closure_loss(model,g)
    return dict(weighted_closure=float(10*loss),g_mean=g.mean(0).cpu().tolist(),g_std=g.std(0,unbiased=False).cpu().tolist(),decoder_norms=model.Q.decoder.weight.norm(dim=0).cpu().tolist())


def run(config_path,output):
    c=validate_config(config_path);out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();step=0
    job=dict(status='running',workflow_complete=False,current='admission',pid=os.getpid(),returncode=None,error=None,requires_user_action=False)
    def status(**kw):
        job.update(kw,actual_steps=step,elapsed_seconds=time.monotonic()-started)
        base.write_json(out/'status.json',job);base.write_json(out/'job.json',job)
        print(json.dumps(job,ensure_ascii=False),flush=True)
    def check():
        if time.monotonic()-started>c['hard_seconds']-30:raise base.BudgetHold('Whole-pipeline budget exceeded')
        if torch.cuda.is_initialized() and max(torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved())>c['max_cuda_bytes']:
            raise base.BudgetHold('Own GPU memory exceeded8GiB')
        if sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>c['max_output_bytes']:
            raise base.BudgetHold('Output budget exceeded2GiB')
    try:
        if os.uname().nodename!='thinklab-105-225' or os.getpriority(os.PRIO_PROCESS,0)<10 or os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU_UUID:
            raise RuntimeError('Wrong host/priority/device mask')
        mem=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
        io=float(next(x.split('avg10=')[1].split()[0] for x in Path('/proc/pressure/io').read_text().splitlines() if x.startswith('full ')))
        if mem<36*1024**3 or os.getloadavg()[0]>1.25*os.cpu_count() or io>=20:
            raise base.BudgetHold('Host CPU/memory/I/O admission not met')
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        from scripts.run_ising_highdim import source_hashes
        code=Path(__file__).resolve().parents[1];pins=source_hashes(code)
        config_sha=base.sha256_file(config_path)
        base.write_json(out/'config.json',c);base.write_json(out/'plan.json',dict(config=c,source_sha256=pins,config_sha256=config_sha))
        status();data,sampling_meta=sampling(c,out,status,check)
        status(current='gpu_admission',sampling_complete=True)
        admission=[resource_snapshot()];time.sleep(10);admission.append(resource_snapshot())
        device=torch.device('cuda:0');receipt=base._device_receipt(device,GPU_UUID)
        torch.cuda.set_per_process_memory_fraction(c['max_cuda_bytes']/torch.cuda.get_device_properties(device).total_memory)
        torch.cuda.reset_peak_memory_stats()
        status(current='disposable_training_timing_pilot')
        pilot=XYQCR(seed=191).to(device);opts=optimizers(pilot);rng=torch.Generator().manual_seed(292);fg=torch.Generator(device=device).manual_seed(393)
        times=[]
        for _ in range(10):
            torch.cuda.synchronize();tick=time.monotonic();independent_step(pilot,draw(data,rng,device),opts,fg)
            torch.cuda.synchronize();times.append(time.monotonic()-tick);check()
        projected=(time.monotonic()-started)+max(times[2:])*1.35*c['steps']+240
        base.write_json(out/'timing_plan.json',dict(pilot_discarded=True,pilot_updates=10,step_seconds=times,projected_total_seconds=projected,fixed_steps=c['steps']))
        if projected>c['hard_seconds']-60:raise base.BudgetHold('Fixed1250updates do not fit; no truncation or restart')
        del opts,pilot;torch.cuda.empty_cache()
        model=XYQCR(seed=c['seed']).to(device);rng=torch.Generator().manual_seed(192);fg=torch.Generator(device=device).manual_seed(293)
        with torch.no_grad():
            x=data.configurations[data.train_indices[:16],:32].to(device=device,dtype=torch.double)
            levels=model.coarse_grain(x,torch.Generator(device=device).manual_seed(9091))
            target=torch.stack([model.features(v) for v in levels]).mean((0,1))
            g=torch.stack([model.Q(v) for v in levels]).mean((0,1))
            model.Q.decoder.bias.copy_(target-model.Q.decoder.weight@g)
        manifest=dict(recipe=RECIPE,config=c,source_sha256=pins,config_sha256=config_sha,
            data_sha256=sampling_meta['angles_sha256'],labels_used_in_training=False,pretraining=False,
            gradient_routing='simultaneous independent Q/C/R; Q cannot update C; R cannot update Q/C',
            C_adapter='SO(2)-equivariant random MLP angular shift plus wrapped-normal noise; circular variance and uniform NLL',
            Q_adapter='SO(2)-invariant relative sine/cosine local9-site input; joint 8^3 relative plaquette probabilities; free-gain affine decoder',
            information_estimator=c['information_estimator'],admission=admission,device=receipt,
            sampling_warnings=sampling_meta['warnings'],equilibrium_certified=False,thermal_nu=None,
            BKT_certified=False,initial_sha256=base.module_fingerprints(model))
        base.write_json(out/'manifest.json',manifest);opts=optimizers(model);history=[];probes=[dict(step=0,**probe(model,data,device))]
        status(current='training',planned_steps=c['steps'])
        for step in range(1,c['steps']+1):
            check();metrics=independent_step(model,draw(data,rng,device),opts,fg)
            if step==1 or step%50==0:
                history.append(dict(step=step,elapsed_seconds=time.monotonic()-started,**metrics));base.write_json(out/'history.json',history)
                status(last_losses={n:metrics[n] for n in ('q_total','c_total','r_total')})
            if step%250==0:
                probes.append(dict(step=step,**probe(model,data,device)));base.write_json(out/'train_probes.json',probes)
                torch.save(dict(recipe=RECIPE,step=step,model=model.state_dict(),manifest=manifest,optimizers={k:v.state_dict() for k,v in opts.items()}),out/f'training_step_{step}.pt')
        clocks=base.optimizer_step_counts(opts,c['steps'])
        for p in model.parameters():p.grad=None
        frozen=base.state_fingerprint(model);status(current='frozen_evaluation')
        arrays=base.evaluate_model(model,data,device,budget_check=check)
        roots=root_diagnostics(model,arrays);roots.pop('critical_kappa',None)
        with torch.no_grad():
            x=data.configurations[data.train_indices[:16],:64].to(device=device,dtype=torch.double)
            b,e=sample_regions(x,128,torch.Generator(device=device).manual_seed(91919));means=model.C.mean_blocks(b)[...,0]
            mi=dict(paired=float(circular_information(means,e)),within_ensemble_mismatch=float(circular_information(means,e,shuffle=True)),estimator=c['information_estimator'])
        if frozen!=base.state_fingerprint(model) or any(p.grad is not None for p in model.parameters()):raise RuntimeError('Evaluation changed model')
        np.savez(out/'evaluation_arrays.npz',**arrays);base.write_json(out/'roots.json',roots)
        summary=dict(status='complete',actual_steps=step,optimizer_steps=clocks,closure=base.closure_metrics(arrays),
            information_diagnostic=mi,final_module_sha256=base.module_fingerprints(model),
            peak_cuda_bytes=torch.cuda.max_memory_allocated(),labels_used_in_training=False,thermal_nu=None,
            BKT_certified=False,equilibrium_certified=False,elapsed_seconds=time.monotonic()-started)
        base.write_json(out/'summary.json',summary);status(current='reporting')
        from scripts.report_xy_qcr import render
        render(out);check()
        if pins!=source_hashes(code) or config_sha!=base.sha256_file(config_path) or sampling_meta['angles_sha256']!=base.sha256_file(out/'data/angles.npz'):
            raise RuntimeError('Frozen source/config/data changed')
        public=[dict(path=str(p.relative_to(out)),sha256=base.sha256_file(p),bytes=p.stat().st_size) for p in sorted(out.rglob('*'))
                if p.is_file() and p.suffix in ('.json','.npz','.png') and p.name not in ('angles.npz','job.json','status.json','sampling_progress.json')]
        status(status='complete',current='complete',workflow_complete=True,returncode=0,public_files=public)
    except Exception as exc:
        status(status='budget_hold' if isinstance(exc,base.BudgetHold) else 'failed',returncode=3 if isinstance(exc,base.BudgetHold) else 1,
               error=str(exc),requires_user_action=True,traceback=traceback.format_exc())
        raise
