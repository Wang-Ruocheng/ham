"""Fresh ru CNN QCR: fixed 6000 updates, one Adam step each, faster R LR.

Reuses the immutable physical ru grid, not the hopping-kappa dataset. The
disposable TRAIN-only throughput pilot cannot select an endpoint or checkpoint.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from experiments import phi4_hopping_qcr as base
from experiments import phi4_qcr as physical
from experiments.phi4_hopping_cnn_qcr import resource_snapshot, GPU_UUID
from recurrent_rg.phi4_cnn_qcr import CNNPhi4QCR
from recurrent_rg.phi4_ru_flow import bounded_flow
from recurrent_rg.losses import all_start_anchor_closure_loss

RECIPE = 'phi4_ru_b3_cnn_g2_Rlr003_v1'
DATA = '/mnt/nas-new/home/yange/ham/results/phi4_ru_b3_g2_seed91_vm225_20261029/data'
DATA_SHA = '79ebe3fa1a9910e651af101beecd401101b1b12864e32beb13abd4ba09e5c020'


def preset():
    return dict(schema_version=5, recipe=RECIPE, data_root=DATA, data_manifest_sha256=DATA_SHA,
        seed=91, theory_dim=2, lattice_sizes=[36,12,4], block_size=3, stride=3,
        steps=6000, system_batch=32, sample_batch=64, information_pairs=128,
        learning_rates=dict(Q=.001,C=.001,R=.003), r_updates_per_batch=1,
        log_every=50, probe_every=250, checkpoint_every=1000,
        hard_seconds=7200, max_cuda_bytes=8*1024**3, max_output_bytes=2*1024**3,
        loss=copy.deepcopy(base.LOSS_POLICY), early_stopping=False, model_selection=False)


def validate_config(path):
    config = json.loads(Path(path).read_text())
    from experiments import phi4_ru_cnn_saddle as saddle
    from experiments import phi4_ru_cnn_nonlinear as nonlinear
    from experiments import phi4_ru_cnn_local_window as local
    from experiments import phi4_kappa_h_cnn as kh
    expected = (kh.preset(config.get('data_manifest_sha256')) if config.get('recipe') == kh.RECIPE else
                local.preset(config.get('data_manifest_sha256')) if config.get('recipe') == local.RECIPE else
                nonlinear.preset() if config.get('recipe') == nonlinear.RECIPE else
                saddle.preset(config.get('data_manifest_sha256')) if config.get('recipe') == saddle.RECIPE else preset())
    if config != expected or any(type(config.get(k)) is not int for k in
            ('steps','seed','theory_dim','r_updates_per_batch','hard_seconds')):
        raise ValueError('This explicit experiment requires its exact frozen preset')
    return config


def optimizers(model):
    return {name:torch.optim.Adam(getattr(model,name).parameters(),lr=rate)
            for name,rate in preset()['learning_rates'].items()}


def draw_batch(dataset, rng, device):
    indices = torch.as_tensor(dataset.train_indices)
    ids = indices[torch.randperm(len(indices),generator=rng)[:32]]
    samples = torch.randperm(dataset.configurations.shape[1],generator=rng)[:64]
    return dataset.configurations[ids][:,samples].to(device=device,dtype=torch.float64)


def evaluate(model, dataset, device, check):
    arrays = base.evaluate_model(model,dataset,device,budget_check=check)
    for split in ('train','test'):
        mask = arrays['splits']==split
        arrays[split+'_direct'] = arrays['direct_g'][mask]
    return arrays


def train_probe(model, dataset, device):
    """Fixed label-free TRAIN panel, private C RNG; reporting only, no selection."""
    indices = dataset.train_indices[np.linspace(0,len(dataset.train_indices)-1,16,dtype=int)]
    generator = torch.Generator(device=device).manual_seed(91303)
    with torch.no_grad():
        levels = model.coarse_grain(dataset.configurations[indices,:64].to(device),generator=generator)
        gs = torch.stack([.5*(model.Q(x[:,:32])+model.Q(x[:,32:])) for x in levels],dim=1)
        loss,_,_ = all_start_anchor_closure_loss(model,gs,coordinate_reduction='sum')
        result = dict(weighted_closure=float(10*loss),g_mean=gs.mean(0).cpu().tolist(),
                      g_std=gs.std(0,unbiased=False).cpu().tolist())
    return result


def projected_budget(elapsed, timings, probe_seconds, evaluation_seconds, count):
    # First two timing steps may include one-time setup; include their actual
    # time via elapsed, then use a conservative steady update projection.
    return elapsed + max(timings[2:])*1.5*6000 + probe_seconds*1.5*25 + max(300.,evaluation_seconds*count*2+180.)


def run(config_path, output):
    config = validate_config(config_path)
    from experiments import phi4_ru_cnn_nonlinear as nonlinear
    from experiments import phi4_ru_cnn_local_window as local
    from experiments import phi4_kappa_h_cnn as kh
    kappa_h = config['recipe'] == kh.RECIPE
    nonlinear_decoder = config['recipe'] in (nonlinear.RECIPE, local.RECIPE, kh.RECIPE)
    expected_gpu_uuid = nonlinear.GPU_UUID if nonlinear_decoder else GPU_UUID
    admit = nonlinear.resource_snapshot if nonlinear_decoder else resource_snapshot
    data_root, data_sha, recipe = config['data_root'], config['data_manifest_sha256'], config['recipe']
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('A fresh independent output is required')
    if (os.uname().nodename!='thinklab-105-225' or os.getpriority(os.PRIO_PROCESS,0)<10
            or os.environ.get('CUDA_VISIBLE_DEVICES')!=expected_gpu_uuid or torch.cuda.is_initialized()):
        raise RuntimeError('Wrong host/priority/GPU mask')
    output.mkdir(parents=True)
    start = time.monotonic()
    step = latest = 0
    job = dict(status='running',workflow_complete=False,current='admission',pid=os.getpid(),
               returncode=None,error=None,requires_user_action=False,recipe=recipe)
    def status(**kw):
        job.update(kw,actual_steps=step,latest_checkpoint_step=latest,
                   Q_updates=step,C_updates=step,R_updates=step,elapsed_seconds=time.monotonic()-start)
        base.write_json(output/'status.json',job)
        base.write_json(output/'job.json',job)
        print(json.dumps(job,ensure_ascii=False,allow_nan=False),flush=True)
    def check():
        if time.monotonic()-start>config['hard_seconds']-30:
            raise base.BudgetHold('Whole pipeline budget reached; no shortening or restart')
        if torch.cuda.is_initialized() and max(torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved())>config['max_cuda_bytes']:
            raise base.BudgetHold('Own GPU memory budget exceeded')
        if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>config['max_output_bytes']:
            raise base.BudgetHold('Output size budget exceeded')
    status()
    try:
        from scripts.report_ising_zh import select_chinese_font
        font = select_chinese_font()
        admission=[admit()]
        time.sleep(10)
        admission.append(admit())
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        if base.sha256_file(Path(data_root)/'manifest.json')!=data_sha:
            raise RuntimeError('Pinned ru dataset manifest differs')
        if 'physical_options' in config and json.loads((Path(data_root)/'manifest.json').read_text())['options'] != config['physical_options']:
            raise RuntimeError('Fresh saddle-window sampling settings differ')
        status(current='loading_pinned_ru_data')
        if kappa_h:
            from recurrent_rg.data.phi4_kappa_h_pipeline import load_dataset
            dataset=load_dataset(data_root)
        else:
            dataset=physical.load_dataset(data_root)
        if dataset.configurations.shape!=(180,128,36,36,1) or len(dataset.train_indices)!=136 or len(dataset.test_indices)!=44:
            raise ValueError('Historical ru dataset geometry/split differs')
        scale=dataset.scale
        device=torch.device('cuda:0')
        receipt=base._device_receipt(device,expected_gpu_uuid)
        torch.cuda.set_per_process_memory_fraction(config['max_cuda_bytes']/torch.cuda.get_device_properties(device).total_memory)
        torch.cuda.reset_peak_memory_stats()
        # Pin all local implementation modules used by this fresh pipeline.
        import recurrent_rg.phi4_cnn_qcr as cnn
        import recurrent_rg.phi4_ru_flow as flow
        import recurrent_rg.cli as cli
        import experiments.phi4_hopping_cnn_qcr as admission_module
        import scripts.report_phi4_ru_cnn_qcr as report_module
        import scripts.report_ising_zh as font_module
        import scripts.run_phi4_ru_cnn_trial as supervisor_module
        import experiments.phi4_ru_cnn_saddle as saddle_module
        pins=base.source_pins()
        for path in [Path(__file__),Path(config_path),*(Path(m.__file__) for m in
                (cnn,flow,cli,physical,admission_module,report_module,font_module,supervisor_module,saddle_module)),font]:
            pins[str(path.resolve())]=base.sha256_file(path)
        if nonlinear_decoder:
            import experiments.phi4_ru_qonly_decoder_balance as decoder_module
            import scripts.run_phi4_ru_cnn_nonlinear as nonlinear_supervisor
            for module in (nonlinear,decoder_module,nonlinear_supervisor):
                pins[str(Path(module.__file__).resolve())]=base.sha256_file(module.__file__)
        if config['recipe'] == local.RECIPE:
            import scripts.run_phi4_ru_cnn_local_window as local_supervisor
            for module in (local, local_supervisor):
                pins[str(Path(module.__file__).resolve())]=base.sha256_file(module.__file__)
        if kappa_h:
            from recurrent_rg.data import phi4_kappa_h_pipeline as kh_data
            from scripts import run_phi4_kappa_h_cnn as kh_supervisor
            pins.update(kh_data.source_pins())
            for module in (kh,kh_supervisor):
                pins[str(Path(module.__file__).resolve())]=base.sha256_file(module.__file__)
        base.write_json(output/'plan.json',dict(config=config,source_sha256=pins,
            admission=admission,data_warning_count=dataset.warning_count,
            endpoint_policy='fixed 6000; no timing/loss/TEST/root-based endpoint selection'))
        status(current='disposable_train_only_timing_pilot')
        pilot=CNNPhi4QCR(scale,seed=191,theory_dim=2).to(device)
        if nonlinear_decoder:
            nonlinear.install_decoder(pilot)
        opts=optimizers(pilot)
        rng=torch.Generator().manual_seed(292)
        field_rng=torch.Generator(device=device).manual_seed(393)
        timings=[]
        for _ in range(10):
            torch.cuda.synchronize();tick=time.monotonic()
            base.independent_step(pilot,draw_batch(dataset,rng,device),opts,generator=field_rng,information_pairs=128)
            torch.cuda.synchronize();timings.append(time.monotonic()-tick);check()
        tick=time.monotonic();train_probe(pilot,dataset,device);torch.cuda.synchronize()
        probe_seconds=time.monotonic()-tick
        from dataclasses import replace
        ix=dataset.train_indices[:1]
        tiny=replace(dataset,configurations=dataset.configurations[ix],splits=dataset.splits[ix],
                     labels=dataset.labels[ix],cell_ids=dataset.cell_ids[ix])
        tick=time.monotonic();evaluate(pilot,tiny,device,check);torch.cuda.synchronize()
        eval_seconds=time.monotonic()-tick
        projection=projected_budget(time.monotonic()-start,timings,probe_seconds,eval_seconds,len(dataset.splits))
        timing=dict(pilot_seed=191,pilot_updates=10,pilot_discarded=True,step_seconds=timings,
                    train_probe_seconds=probe_seconds,one_train_chain_evaluation_seconds=eval_seconds,
                    conservative_total_seconds=projection,hard_seconds=config['hard_seconds'],
                    formal_steps=6000,passed=projection<config['hard_seconds']-60)
        base.write_json(output/'timing_plan.json',timing)
        if not timing['passed']:
            raise base.BudgetHold('Fixed 6000 updates do not fit conservative total budget; no formal training started')
        del pilot,opts;torch.cuda.empty_cache()
        model=CNNPhi4QCR(scale,seed=91,theory_dim=2).to(device)
        initialization=base.initialize_q_decoder(model,dataset.configurations,dataset.train_indices,seed=91,budget_check=check)
        if nonlinear_decoder:
            nonlinear.install_decoder(model)
            initialization=dict(initialization,
                decoder_upgrade='shared affine plus centered nonlinear residual; zero residual at initialization',
                affine_initialized_state_sha256=initialization['training_start_state_sha256'],
                training_start_state_sha256=base.state_fingerprint(model),
                training_start_module_sha256=base.module_fingerprints(model))
        manifest=dict(config,options=config,source_sha256=pins,scale=scale,device=receipt,
            data_shards=dataset.receipts,data_warning_count=dataset.warning_count,
            sampler_qualification=dataset.qualification,equilibrium_certified=False,
            evaluation_label_columns=(['kappa','h','quartic_lambda'] if kappa_h else ['r','u','h']),
            labels_used_in_training=False,test_used_in_training=False,
            action=(kh_data.kernel.ACTION if kappa_h else
                'S=.5 sum_(x,a)(phi[x+e_a]-phi[x])^2+sum_x(.5*r*phi^2+.25*u*phi^4-h*phi)'),
            initialization=initialization,all_Q_C_R_fresh_trainable=True,pretraining=False,
            architecture=('per-field periodic CNN, 16 frame features; ensemble mean/variance; 32-64-2 head; '
                + ('shared affine 2->625 plus centered 2-32-32-625 tanh decoder; ' if nonlinear_decoder else 'shared affine 2->625 decoder; ')
                + 'residual R 2-96-96-2'),
            c_adaptation='Gaussian C, smoothed buffered full-shell information, variance floor, uniform amplitude penalty; not discrete histogram MI/color NLL',
            gradient_routing='simultaneous independent Q/C/R gradients computed before any optimizer step',
            normalization='one fixed TRAIN microscopic RMS, no coupling labels',
            final_evaluation='all 180 physical chains, all 128 saved fields; half-ensemble encoding and pooled encoding separate',
            train_probe='16 fixed TRAIN chains x64 fields, same private C RNG every 250; never used to stop/select',
            root_policy='bounded TRAIN-input-only 33x33 grid; at most49 starts,64 evaluations each; all roots/spectra, no thermal selection',
            thermal_nu=None,timing_plan=timing,chinese_font=str(font))
        base.write_json(output/'manifest.json',manifest)
        opts=optimizers(model);rng=torch.Generator().manual_seed(192)
        field_rng=torch.Generator(device=device).manual_seed(293)
        history=[];probes=[dict(step=0,**train_probe(model,dataset,device))]
        base.write_json(output/'train_probes.json',probes)
        status(current='training',planned_steps=6000)
        train_start=time.monotonic()
        model.train()
        for step in range(1,6001):
            check()
            metrics=base.independent_step(model,draw_batch(dataset,rng,device),opts,generator=field_rng,information_pairs=128,r_updates_per_batch=1)
            if step%250==0:
                probes.append(dict(step=step,**train_probe(model,dataset,device)))
                base.write_json(output/'train_probes.json',probes)
            if step%1000==0:
                torch.save(dict(recipe=recipe,step=step,model=model.state_dict(),
                    optimizers={k:v.state_dict() for k,v in opts.items()},manifest=manifest,
                    batch_rng=rng.get_state(),field_rng=field_rng.get_state()),output/f'training_step_{step}.pt')
                latest=step
            if step==1 or step%50==0:
                history.append(dict(step=step,elapsed_seconds=time.monotonic()-train_start,**metrics))
                base.write_json(output/'history.json',history)
                status(current='training',last_losses={k:metrics[k] for k in ('q_total','c_total','r_total')})
        clocks=base.optimizer_step_counts(opts,6000)
        for p in model.parameters():p.grad=None
        fingerprint=base.state_fingerprint(model)
        status(current='frozen_evaluation')
        arrays=evaluate(model,dataset,device,check)
        roots,flow=bounded_flow(model,arrays,device,budget_check=check)
        if fingerprint!=base.state_fingerprint(model) or any(p.grad is not None for p in model.parameters()):
            raise RuntimeError('Frozen evaluation changed model or gradients')
        if any(base.sha256_file(p)!=sha for p,sha in pins.items()):
            raise RuntimeError('Frozen source fingerprint changed')
        if base.sha256_file(Path(data_root)/'manifest.json')!=data_sha:
            raise RuntimeError('Physical manifest changed')
        np.savez(output/'evaluation_arrays.npz',**arrays)
        base.write_json(output/'roots.json',dict(roots=roots,flow_analysis=flow,thermal_nu=None))
        summary=dict(status='complete',recipe=recipe,actual_steps=step,optimizer_steps=clocks,
            learning_rates=config['learning_rates'],r_updates_per_batch=1,
            closure=base.closure_metrics(arrays),numerical_fixed_points=roots,flow_analysis=flow,
            final_model_sha256=fingerprint,final_module_sha256=base.module_fingerprints(model),
            labels_used_in_training=False,frozen_evaluation_verified=True,data_warning_count=dataset.warning_count,
            train_chains=136,test_chains=44,thermal_nu=None,equilibrium_certified=False,
            peak_cuda_bytes=max(torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved()),
            elapsed_seconds=time.monotonic()-start)
        if nonlinear_decoder:
            summary['decoder_bounds']=model.Q.decoder.bounds()
            summary['direction_balancing']=False
        base.write_json(output/'summary.json',summary)
        status(current='report')
        report_module.render(output)
        check()
        public={str(p.relative_to(output)):base.sha256_file(p) for p in output.rglob('*')
                if p.is_file() and p.suffix in ('.json','.npz','.png') and p.name not in ('job.json','status.json')}
        status(status='complete',current=None,workflow_complete=True,returncode=0,public_files=public)
        return summary
    except Exception as exc:
        status(status='budget_hold' if isinstance(exc,base.BudgetHold) else 'failed',
               returncode=3 if isinstance(exc,base.BudgetHold) else 1,error=f'{type(exc).__name__}: {exc}',
               traceback=traceback.format_exc(),requires_user_action=True)
        raise
