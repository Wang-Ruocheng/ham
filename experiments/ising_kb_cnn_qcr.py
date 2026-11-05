"""Fresh KB CNN QCR, matched to the current ru run's optimization protocol.

Binary C and empirical buffered full-shell MI retain the locked Ising recipe.
Only configurations enter Q/C/R objectives; K/B are appended after evaluation.
"""
from collections import Counter
from dataclasses import replace
import json
import os
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch

from experiments import ising_cnn_short_trial as binary
from experiments import phi4_hopping_qcr as common
from experiments.phi4_hopping_cnn_qcr import resource_snapshot, GPU_UUID
from recurrent_rg.checkpoints import isolated_rng, qcr_state_dict
from recurrent_rg.config import RunConfig
from recurrent_rg.encoding import half_subset_theory
from recurrent_rg.losses import all_start_anchor_closure_loss
from recurrent_rg.phi4_ru_flow import bounded_flow
from recurrent_rg.runner import prepared_data, file_fingerprint
from recurrent_rg.mi_cache import lattice_statistics_cache_key, load_or_build_mi_statistics
from recurrent_rg.rsmi import region_offsets, kernel_probabilities
from recurrent_rg.data.statistics import plaquette_distribution
from scripts.run_ising_kb_highdim import validate_audit

RECIPE = 'ising2d_KB_cnn_b3_g2_Rlr003_seed91_v1'
DATA_DIR = '/mnt/nas-new/home/yange/ham/data/ising2d_Kfocused_Buniform0p01_L36_n64_seed91'
DATA_SHA = 'ffc776bf6c456610e3176dfe4b328b5a4cbcd6dcb5b90f900399d591d44b5c13'
AUDIT_SHA = 'b84d29686acaba81040325a053c67ad81851d7b421212071991a29a8244a5aaa'
MI_DIR = '/mnt/nas-new/home/yange/ham/results/ising2d_KB_b3_highdim_seed91_vm225_20261022/mi_cache'


def preset():
    return dict(schema_version=5, recipe=RECIPE, data_cache=DATA_DIR+'/ensembles.pt',
        data_sha256=DATA_SHA, data_audit=DATA_DIR+'/sampling_audit.json', data_audit_sha256=AUDIT_SHA,
        mi_cache_dir=MI_DIR, seed=91, theory_dim=2, lattice_sizes=[36, 12, 4], block_size=3,
        steps=6000, system_batch=32, sample_batch=64, learning_rates=dict(Q=.001,C=.001,R=.003),
        r_updates_per_batch=1, log_every=50, checkpoint_every=1000, probe_every=250,
        hard_seconds=7200, max_cuda_bytes=8*1024**3, max_output_bytes=2*1024**3)


def validate_config(path):
    value = json.loads(Path(path).read_text())
    if value != preset() or any(type(value[k]) is not int for k in
            ('steps', 'seed', 'theory_dim', 'r_updates_per_batch', 'hard_seconds')):
        raise ValueError('The KB CNN experiment requires its exact frozen preset')
    return value


def build_config():
    # Explicit default loss, not a historical missing-mode reinterpretation.
    cfg = RunConfig.from_dict(dict(schema_version=5,
        data=dict(cache=DATA_DIR+'/ensembles.pt', physical_model='ising2d', coupling_names=['K','B'],
                  spin_flip_augment=False, spin_flip_symmetric=False, train_pool_replicas=1, test_pool_replicas=1),
        model=dict(q_encoder='configuration_cnn', dimension=2, theory_dim=2, hidden=96,
                   width=3, block_factor=3, binary_lookup=True, kernel_layout='nonoverlapping',
                   fixed_kernel=False, kernel_initialization='random', kernel_sampling='bernoulli',
                   kernel_temperature=1., affine_reconstruction=True, theory_head='nonlinear', local_features=12),
        loss=dict(closure_objective='all_start_anchor', closure_scale_weights=None, rollout_weight=5.,
                  kernel_sign_weight=1., kernel_noncollapse_mode='variance_floor_ensemble', kernel_variance_ratio=1.,
                  coordinate_reduction='sum', closure_metric='euclidean',
                  weights=dict(anchor=2., coarse_consistency=5., kernel_noncollapse=2., separation=10., commutator=5.)),
        training=dict(epochs=6000, max_rg_steps=2, system_batch=32, sample_batch=64,
                      learning_rate=.001, seed=91, threads=1, train_modules=['Q','C','R'],
                      gradient_mode='qc_r_separate', lr_schedule='constant', log_every=50,
                      checkpoint_every=1000, r_updates_per_batch=1),
        evaluation=dict(max_rg_steps=2, samples=64, systems=714)))
    from experiments.rsmi_separate_qcr import build_parser, validate_recipe
    cfg.validate()
    validate_recipe(build_parser().parse_args(['--config','unused','--output','unused']), cfg)
    return cfg


def optimizers(model):
    return {name:torch.optim.Adam(module.parameters(),lr=preset()['learning_rates'][name])
            for name,module in binary.modules(model).items()}


def draw_batch(configurations, generator, device):
    # Historical physical caches store exact binary spins as int8. Convert
    # only the selected batch; retain the original cache and MI identity.
    indices=torch.randperm(len(configurations),generator=generator)[:32]
    samples=torch.randperm(configurations.shape[1],generator=generator)[:64]
    return configurations[indices[:,None],samples[None,:]].to(device=device,dtype=torch.float64)


def check_cells(train, test, audit):
    contract = validate_audit(audit)
    for split, ds, count in ((False, train, 474), (True, test, 240)):
        if list(ds.configurations.shape) != [count, 64, 36, 36, 1]:
            raise ValueError('KB physical cache geometry mismatch')
        expected = Counter((row['K'],row['B']) for row in audit['ensembles'] if row['test']==split)
        observed = Counter(tuple(map(float,row)) for row in ds.couplings.reshape(count,2))
        if expected != observed:
            raise ValueError('Actual cells/replicas differ from the sampling audit')
    return contract


def initialize_bias(model, train, device):
    before = {name:common.state_fingerprint(module) for name,module in binary.modules(model).items()}
    protected = {k:v.detach().clone() for k,v in model.theory_encoder.state_dict().items()
                 if k != 'distribution_decoder.bias'}
    with isolated_rng(device, 9091), torch.no_grad():
        levels = model.coarse_grain(train.configurations[:32].to(device=device,dtype=torch.float64))
        center = torch.stack([plaquette_distribution(level).mean(0) for level in levels]).mean(0)
        model.theory_encoder.distribution_decoder.bias.copy_(center)
    if any(not torch.equal(v,model.theory_encoder.state_dict()[k]) for k,v in protected.items()):
        raise RuntimeError('Mean-bias initialization changed non-bias Q state')
    for name in ('C','R'):
        if common.state_fingerprint(binary.modules(model)[name]) != before[name]:
            raise RuntimeError('Mean-bias initialization changed '+name)
    return dict(mode='first32_TRAIN_mean_affine_bias', samples=64, scale_weighting='uniform',
        private_C_seed=9091, labels_used=False, test_used=False, PCA=False,
        optimizer_updates=0, C_R_unchanged=True)


@torch.no_grad()
def train_probe(model, train, device):
    indices = np.linspace(0,len(train.configurations)-1,16,dtype=int)
    with isolated_rng(device, 91303):
        levels = model.coarse_grain(train.configurations[indices].to(device=device,dtype=torch.float64))
        gs = half_subset_theory(model, levels)
        loss,_,_ = all_start_anchor_closure_loss(model,gs,coordinate_reduction='sum')
    return dict(weighted_closure=float(10*loss), g_mean=gs.mean(0).cpu().tolist(),
                g_std=gs.std(0,unbiased=False).cpu().tolist())


@torch.no_grad()
def collect(model, train, test, device, check):
    records = {k:[] for k in ('direct_g','direct_full_g','one_step_g','rollout_g','all_start_g',
                              'target_features','reconstructed_features')}
    labels, splits = [], []
    model.eval()
    with isolated_rng(device, 20261103):
        for split, ds in (('train',train),('test',test)):
            for index, chain in enumerate(ds.configurations):
                levels = model.coarse_grain(chain[None].to(device=device,dtype=torch.float64))
                direct = half_subset_theory(model,levels)
                full = torch.stack([model.theory_encoder(level) for level in levels],1)
                one = model.advance_theory(direct[:,:-1])
                r1 = model.advance_theory(direct[:,0]); r2 = model.advance_theory(r1)
                roll = torch.stack([direct[:,0],r1,r2],1)
                all_start = torch.stack([one[:,0],r2,one[:,1]],1)
                targets = torch.stack([plaquette_distribution(level) for level in levels],1)
                reconstructed = model.theory_encoder.distribution_decoder(direct)
                for name, value in zip(records,(direct,full,one,roll,all_start,targets,reconstructed)):
                    records[name].append(value.cpu().numpy()[0])
                # Labels only attached AFTER frozen model operations.
                labels.append(ds.couplings[index].reshape(-1).cpu().numpy()); splits.append(split)
                check()
    arrays = {k:np.asarray(v) for k,v in records.items()}
    arrays.update(labels=np.asarray(labels), splits=np.asarray(splits), all_start_pairs=np.array([[0,1],[0,2],[1,2]]))
    arrays['cell_ids'] = np.unique(arrays['labels'],axis=0,return_inverse=True)[1]
    for split in ('train','test'):
        arrays[split+'_direct'] = arrays['direct_g'][arrays['splits']==split]
    for name,value in arrays.items():
        if value.dtype.kind in 'fc' and not np.isfinite(value).all():
            raise FloatingPointError('Nonfinite evaluation '+name)
    return arrays


def projected_seconds(elapsed, timings, probe_seconds, chain_seconds):
    if len(timings) != 10 or not all(np.isfinite(t) and t > 0 for t in timings):
        raise ValueError('Require ten complete finite throughput measurements')
    return elapsed+1.5*max(timings[2:])*6000+1.5*probe_seconds*25+max(300.,2*chain_seconds*714+180.)


def run(config_path, output):
    contract = validate_config(config_path); cfg = build_config()
    if (os.uname().nodename != 'thinklab-105-225' or os.getpriority(os.PRIO_PROCESS,0)<10
            or os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID or torch.cuda.is_initialized()):
        raise RuntimeError('Wrong host/priority/GPU mask')
    out = Path(output).resolve(); out.mkdir(parents=True,exist_ok=False)
    started = time.monotonic(); actual = latest = 0
    job = dict(recipe=RECIPE,status='running',workflow_complete=False,pid=os.getpid(),
               returncode=None,error=None,requires_user_action=False)
    def status(**kw):
        job.update(kw,actual_steps=actual,latest_checkpoint_step=latest,
                   Q_updates=actual,C_updates=actual,R_updates=actual,elapsed_seconds=time.monotonic()-started)
        common.write_json(out/'status.json',job);common.write_json(out/'job.json',job)
        print(json.dumps(job,ensure_ascii=False,allow_nan=False),flush=True)
    def check():
        if time.monotonic()-started>contract['hard_seconds']-30:
            raise common.BudgetHold('Two-hour complete pipeline budget reached; no shortening or retry')
        if torch.cuda.is_initialized() and max(torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved())>contract['max_cuda_bytes']:
            raise common.BudgetHold('Own CUDA memory cap exceeded')
        if sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>contract['max_output_bytes']:
            raise common.BudgetHold('Output size cap exceeded')
    status(current='admission')
    try:
        while True:
            check()
            try:
                admission=[resource_snapshot()];time.sleep(10);admission.append(resource_snapshot())
                break
            except common.BudgetHold as exc:
                status(current='waiting_resources',resource_wait_reason=str(exc))
                time.sleep(30)
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        if file_fingerprint(contract['data_cache'])!=DATA_SHA or file_fingerprint(contract['data_audit'])!=AUDIT_SHA:
            raise ValueError('Historical KB data/audit identity mismatch')
        audit=json.loads(Path(contract['data_audit']).read_text())
        status(current='loading_existing_data_and_MI')
        train,test=prepared_data(cfg,mmap=True);physical=check_cells(train,test,audit)
        device=torch.device('cuda:0');device_receipt=common._device_receipt(device,GPU_UUID)
        torch.cuda.set_per_process_memory_fraction(contract['max_cuda_bytes']/torch.cuda.get_device_properties(device).total_memory)
        torch.cuda.reset_peak_memory_stats()
        _,environment=region_offsets(2,block=3,buffer=1)
        hist={};cache_receipts={}
        def forbidden():raise common.BudgetHold('Pinned existing empirical MI cache missing; do not rebuild historical cache')
        for split,ds in (('train',train),('test',test)):
            key=lattice_statistics_cache_key(cfg,DATA_SHA,split,ds.configurations,environment,
                'binary_lattice_mi_statistics',{'block':3})
            h,cache_receipts[split]=load_or_build_mi_statistics(MI_DIR,key,forbidden,mmap=True)
            if not cache_receipts[split]['hit']:raise RuntimeError('Existing MI cache required')
            h.prepare_selection();hist[split]=h.to(device) if split=='train' else h
        from itertools import product
        patterns=torch.tensor(list(product((-1.,1.),repeat=9)),dtype=torch.float64,device=device)[None]
        import scripts.report_ising_kb_cnn_qcr as report_module
        from scripts.run_ising_highdim import source_hashes
        code=Path(__file__).resolve().parents[1]
        pins=source_hashes(code);config_sha=file_fingerprint(config_path)
        common.write_json(out/'plan.json',dict(contract=contract,config=cfg.to_dict(),source_sha256=pins,
            code_root=str(code),config_sha256=config_sha,admission=admission,physical_data=physical,
            endpoint_policy='fixed6000, no timing/loss/TEST/root-based step reduction'))
        status(current='disposable_TRAIN_timing_pilot')
        pilot_cfg=replace(cfg,training=replace(cfg.training,seed=191))
        pilot=binary.fresh(pilot_cfg,device);opts=optimizers(pilot)
        batch_rng=torch.Generator().manual_seed(292);mi_rng=torch.Generator().manual_seed(393)
        durations=[]
        with isolated_rng(device,494):
            for _ in range(10):
                torch.cuda.synchronize();tick=time.monotonic()
                binary.step(pilot,draw_batch(train.configurations,batch_rng,device),hist['train'],
                    torch.randperm(len(train.configurations),generator=mi_rng)[:32],patterns,cfg,opts)
                torch.cuda.synchronize();durations.append(time.monotonic()-tick);check()
        tick=time.monotonic();train_probe(pilot,train,device);torch.cuda.synchronize();probe_seconds=time.monotonic()-tick
        tick=time.monotonic()
        # One existing TRAIN chain to measure readout; no TEST pilot and no roots.
        tiny=SimpleNamespace(configurations=train.configurations[:1],couplings=train.couplings[:1])
        empty=SimpleNamespace(configurations=train.configurations[:0],couplings=train.couplings[:0])
        collect(pilot,tiny,empty,device,check);torch.cuda.synchronize();chain_seconds=time.monotonic()-tick
        projection=projected_seconds(time.monotonic()-started,durations,probe_seconds,chain_seconds)
        timing=dict(pilot_updates=10,pilot_seed=191,pilot_discarded=True,step_seconds=durations,
            fixed_TRAIN_probe_seconds=probe_seconds,one_TRAIN_chain_eval_seconds=chain_seconds,
            conservative_total_seconds=projection,hard_seconds=7200,formal_steps=6000,passed=projection<7140)
        common.write_json(out/'timing_plan.json',timing)
        if not timing['passed']:raise common.BudgetHold('Fixed6000 updates exceed conservative two-hour projection')
        del pilot,opts;torch.cuda.empty_cache()
        model=binary.fresh(cfg,device);initial=common.state_fingerprint(model)
        initialization=initialize_bias(model,train,device)
        manifest=dict(recipe=RECIPE,config=cfg.to_dict(),contract=contract,learning_rates=contract['learning_rates'],
            physical_data=physical,data_sha256=DATA_SHA,data_audit_sha256=AUDIT_SHA,mi_cache=cache_receipts,
            source_sha256=pins,code_root=str(code),config_sha256=config_sha,admission=admission,device=device_receipt,
            timing=timing,initialization=initialization,initial_random_state_sha256=initial,
            labels_used_in_training=False,test_used_in_training=False,evaluation_label_columns=['K','B'],
            all_Q_C_R_fresh_trainable=True,pretraining=False,physical_sampling_updates=0,
            equilibrium_certified=False,thermal_nu=None,
            architecture='per-field periodic CNN16; mean/variance32; Q head32-64-2; affine2->16; residual R2-96-96-2',
            C_objective='-2 empirical buffered full-shell MI +2 per-ensemble variance floor(ratio1)+1 uniform-block color NLL',
            Q_objective='affine motif reconstruction10 +micro subset20 +coarse subset5',
            R_objective='all-start anchor, equal target weights, coefficient10; one Adam update per batch',
            gradient_routing='independent selective Q/C/R gradients computed before any optimizer step',
            evaluation='all714 physical chains,64 fields,one private stochastic C trajectory; two-half and pooled encodings separate',
            comparison_qualification='Matches ru CNN optimization, not the physical sample count or binary/continuous observation models')
        common.write_json(out/'manifest.json',manifest)
        opts=optimizers(model);batch_rng=torch.Generator().manual_seed(192);mi_rng=torch.Generator().manual_seed(293)
        probes=[dict(step=0,**train_probe(model,train,device))];history=[]
        common.write_json(out/'train_probes.json',probes)
        status(current='training',planned_steps=6000);model.train()
        with isolated_rng(device,394):
            for actual in range(1,6001):
                check()
                metrics=binary.step(model,draw_batch(train.configurations,batch_rng,device),hist['train'],
                    torch.randperm(len(train.configurations),generator=mi_rng)[:32],patterns,cfg,opts)
                if actual%250==0:
                    probes.append(dict(step=actual,**train_probe(model,train,device)))
                    common.write_json(out/'train_probes.json',probes)
                if actual%1000==0:
                    torch.save(dict(recipe=RECIPE,epoch=actual,step=actual,model=qcr_state_dict(model),metadata=manifest,
                        optimizers={n:o.state_dict() for n,o in opts.items()},batch_rng=batch_rng.get_state(),
                        mi_rng=mi_rng.get_state(),cuda_rng=torch.cuda.get_rng_state()),out/f'training_step_{actual}.pt')
                    latest=actual
                if actual==1 or actual%50==0:
                    history.append(dict(step=actual,elapsed_seconds=time.monotonic()-started,**metrics))
                    common.write_json(out/'history.json',history)
                    status(current='training',last_losses={k:metrics[k] for k in ('q_total','c_total','r_total')})
        clocks=common.optimizer_step_counts(opts,6000)
        for p in model.parameters():p.grad=None
        frozen=common.state_fingerprint(model)
        status(current='frozen_evaluation')
        arrays=collect(model,train,test,device,check)
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
        summary=dict(status='complete',recipe=RECIPE,actual_steps=6000,optimizer_steps=clocks,
            learning_rates=contract['learning_rates'],closure=binary.closure_metrics(arrays),final_MI=mi_final,
            numerical_fixed_points=roots,flow_analysis=flow,final_state_sha256=frozen,thermal_nu=None,
            labels_used_in_training=False,frozen_evaluation_verified=True,equilibrium_certified=False,
            peak_cuda_bytes=max(torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved()),elapsed_seconds=time.monotonic()-started)
        common.write_json(out/'summary.json',summary)
        status(current='report');report_module.render(out);check()
        if source_hashes(code)!=pins or file_fingerprint(config_path)!=config_sha:
            raise RuntimeError('Frozen source/config changed')
        if file_fingerprint(contract['data_cache'])!=DATA_SHA or file_fingerprint(contract['data_audit'])!=AUDIT_SHA:
            raise RuntimeError('Historical data identity changed')
        public={str(p.relative_to(out)):file_fingerprint(p) for p in out.rglob('*') if p.is_file()
                and p.suffix in ('.json','.png','.npz') and p.name not in ('job.json','status.json')}
        status(status='complete',current=None,workflow_complete=True,returncode=0,public_files=public)
        return summary
    except Exception as exc:
        status(status='budget_hold' if isinstance(exc,common.BudgetHold) else 'failed',
            error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc(),requires_user_action=True,
            returncode=3 if isinstance(exc,common.BudgetHold) else 1)
        if isinstance(exc,common.BudgetHold):
            raise SystemExit(3) from exc
        raise
