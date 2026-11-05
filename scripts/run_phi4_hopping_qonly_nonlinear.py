"""Explicit dg=1 decoder-structure ablation, reusing a verified frozen C cache.

No physical sampling, new C trajectories, default changes, or R optimization.
The source and cache-producing experiment are read-only and SHA pinned.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch
from torch import nn

from scripts import run_phi4_hopping_qonly_balance as base

CACHE_ROOT = Path('/mnt/nas-new/home/yange/ham/results/phi4_hopping_b2_L128_qonly_scale_balance_seed91_gpu3_vm225_20261102_v1')
CACHE_SHA = '0712203a73292e35fb5d9bf586a4bab2f2f52ab6183af60a9bfffc94957f8747'
CACHE_PUBLIC_SHA = 'ce48a8e6ab0b00e7d715e0bfde3500c033b3aef43acc205dc823009d826b5c87'
ARMS = ('A_affine', 'B_nonlinear')
STEPS = 300
LAYER_NORM_CAP = 2.
GPU_SECONDS, CPU_SECONDS = 1800., 7200.
MAX_BYTES, MAX_CUDA_BYTES = 1024**3, 8 * 1024**3


class ResidualDecoder(nn.Module):
    """D(g)=b+wg+f(g)-f(0); ||w||=1, Lip(f)<=2^3, Lip(D)<=9.

    Three weight matrices have exact spectral-norm projections after updates.
    Tanh derivatives have magnitude <=1. No global injectivity is assumed.
    The zero last layer makes the initial decoder exactly the source affine D.
    """
    def __init__(self, affine, hidden=32):
        super().__init__()
        if affine.in_features != 1:
            raise ValueError('This ablation fixes dg=1')
        self.affine = copy.deepcopy(affine)
        self.residual = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, affine.out_features, bias=False),
        ).to(device=affine.weight.device, dtype=affine.weight.dtype)
        with torch.no_grad():
            self.residual[-1].weight.zero_()
            self.project_residual_()

    def forward(self, g):
        return self.affine(g) + self.residual(g) - self.residual(torch.zeros_like(g))

    @torch.no_grad()
    def project_residual_(self):
        for layer in self.residual:
            if isinstance(layer, nn.Linear):
                norm = torch.linalg.matrix_norm(layer.weight, ord=2)
                layer.weight.div_(torch.clamp(norm / LAYER_NORM_CAP, min=1.))

    @torch.no_grad()
    def normalize_(self):
        self.affine.weight.div_(self.affine.weight.norm(dim=0, keepdim=True).clamp_min(1e-12))
        self.project_residual_()

    def bounds(self):
        norms = [float(torch.linalg.matrix_norm(layer.weight.detach(), ord=2))
                 for layer in self.residual if isinstance(layer, nn.Linear)]
        return dict(residual_matrix_norms=norms,
                    affine_column_norm=float(self.affine.weight.detach().norm()),
                    global_lipschitz_upper_bound=float(self.affine.weight.detach().norm()) + math.prod(norms),
                    fixed_upper_bound=9., injectivity_certified=False)


class NonlinearQ(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.encoder = copy.deepcopy(source)
        affine = self.encoder.decoder
        self.encoder.decoder = nn.Identity()
        self.decoder = ResidualDecoder(affine)

    def forward(self, fields):
        return self.encoder(fields)

    def decode(self, g):
        return self.decoder(g)

    @torch.no_grad()
    def normalize_decoder_(self):
        self.decoder.normalize_()


def make_models(source):
    # Residual initialization is isolated from the paired batch generator.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(192)
        result = {ARMS[0]: copy.deepcopy(source), ARMS[1]: NonlinearQ(source)}
    return {name: model.requires_grad_(True) for name, model in result.items()}


def decoder_diagnostics(q, values):
    """Analytic decoder Jacobians at observed g and a TRAIN-defined grid."""
    device = next(q.parameters()).device
    points = torch.as_tensor(values, dtype=torch.float64, device=device).reshape(-1, 1)
    with torch.no_grad():
        decoded = q.decode(points).cpu().numpy()
        # A forward-mode tangent computes all 625 derivative components at once.
        _, tangent = torch.func.jvp(q.decode, (points,), (torch.ones_like(points),))
        jacobian = tangent.cpu().numpy()
    bounds = q.decoder.bounds() if isinstance(q, NonlinearQ) else dict(
        global_lipschitz_upper_bound=float(q.decoder.weight.detach().norm()),
        fixed_upper_bound=1., injectivity_certified=True)
    norms = np.linalg.norm(jacobian, axis=-1)
    if not np.isfinite(decoded).all() or not np.isfinite(jacobian).all():
        raise FloatingPointError('Nonfinite decoder diagnostic')
    if norms.max() > bounds['global_lipschitz_upper_bound'] + 1e-8:
        raise RuntimeError('Decoder derivative exceeds its verified bound')
    return decoded, jacobian, dict(**bounds, derivative_norm_min=float(norms.min()),
                                  derivative_norm_max=float(norms.max()))


def bind_cache(formal, protected):
    public_path = CACHE_ROOT / 'public_files.json'
    if base.sha(public_path) != CACHE_PUBLIC_SHA:
        raise ValueError('Frozen cache public manifest pin mismatch')
    protected[public_path] = CACHE_PUBLIC_SHA
    inventory={row['path']:row for row in base.read(public_path)['files']}
    for name in ('job.json','plan.json','summary.json','cache_receipt.json'):
        row=inventory[name]
        if base.sha(CACHE_ROOT/name)!=row['sha256'] or (CACHE_ROOT/name).stat().st_size!=row['bytes']:
            raise ValueError('Cache-producing public metadata changed')
    base.complete(base.read(CACHE_ROOT / 'job.json'), 'Cache-producing experiment')
    receipt = base.read(CACHE_ROOT / 'cache_receipt.json')
    if receipt['cache_sha256'] != CACHE_SHA or base.sha(CACHE_ROOT / 'frozen_fields.pt') != CACHE_SHA:
        raise ValueError('Frozen C cache SHA mismatch')
    for name in ('job.json', 'plan.json', 'summary.json', 'cache_receipt.json'):
        protected[CACHE_ROOT / name] = base.sha(CACHE_ROOT / name)
    protected[CACHE_ROOT / 'frozen_fields.pt'] = CACHE_SHA
    cache = torch.load(CACHE_ROOT / 'frozen_fields.pt', map_location='cpu', weights_only=False)
    fields, targets = cache['fields'], cache['target_per_field']
    if len(fields) != 4 or any(x.shape != (60, 128, size, size, 1) or x.dtype != torch.float64
            for x, size in zip(fields, base.SIZES)) or targets.shape != (60, 4, 128, 625):
        raise ValueError('Unexpected frozen cache axes/dtype')
    if not torch.isfinite(targets).all() or any(not torch.isfinite(x).all() for x in fields):
        raise ValueError('Nonfinite frozen cache')
    if np.max(np.abs(targets.mean(2).numpy() - formal['target_features'])) > 1e-10:
        raise ValueError('Frozen cache targets differ from original evaluation')
    return fields, targets, receipt


def run(output, gpu_index, expected_gpu_uuid):
    wall = time.monotonic()
    output = Path(output).resolve()
    if output.exists() or any(root == output or root in output.parents for root in (base.SOURCE, base.SNAPSHOT, CACHE_ROOT)):
        raise ValueError('Fresh independent output required; no overwrite/resume')
    if os.uname().nodename != 'thinklab-105-225' or os.getpriority(os.PRIO_PROCESS, 0) < 10:
        raise RuntimeError('Wrong host or process priority')
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != expected_gpu_uuid
            or os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID' or torch.cuda.is_initialized()):
        raise RuntimeError('Require fixed GPU UUID mask before CUDA initialization')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output.mkdir(parents=True)
    job = dict(status='running', workflow_complete=False, current='resource_admission', actual_steps=0,
        pid=os.getpid(), returncode=None, error=None, requires_user_action=False,
        source_Q_steps=3000, extra_Q_steps_per_arm=STEPS, C_updates=0, R_updates=0)

    def status(**updates):
        job.update(updates, elapsed_seconds=time.monotonic()-wall, cpu_seconds=time.process_time())
        for name in ('job.json', 'status.json'):
            base.write(output/name, job)
        print(json.dumps(job, ensure_ascii=False), flush=True)

    def check(reserve=0., *, directory=False):
        if time.monotonic()-wall+reserve > GPU_SECONDS or time.process_time()+reserve > CPU_SECONDS:
            raise base.BudgetHold('Conservative elapsed reservation/CPU budget exhausted')
        if directory and sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) > MAX_BYTES:
            raise base.BudgetHold('New directory exceeds 1 GiB')
        if torch.cuda.is_initialized() and torch.cuda.max_memory_allocated() > MAX_CUDA_BYTES:
            raise base.BudgetHold('Own CUDA peak exceeds 8 GiB')

    status()
    try:
        admission = []
        while len(admission) < 2:
            try:
                admission.append(base.gpu_snapshot(gpu_index, expected_gpu_uuid))
            except base.BudgetHold:
                admission = []
                status(current='waiting_resources')
                time.sleep(30)
                check()
                continue
            if len(admission) == 1:
                time.sleep(10)
                check()
        status(current='source_and_cache_verification')
        source_state, formal, protected, source_manifest, source_summary = base.bind_sources()
        for name in ('run_phi4_hopping_qonly_nonlinear.py', 'report_phi4_hopping_qonly_nonlinear.py',
                     'run_phi4_hopping_qonly_balance.py', 'report_phi4_hopping_qonly_balance.py'):
            path = Path(__file__).resolve().with_name(name)
            protected[path] = base.sha(path)
        frozen, per_field, old_receipt = bind_cache(formal, protected)
        AlignedPhi4QCR, _, _ = base.snapshot_imports()
        model = AlignedPhi4QCR(float(source_state['scale']), hidden=96, theory_dim=1, local_features=12,
            patch_chunk_size=262144, block_size=2, steps=3)
        model.load_state_dict(source_state, strict=True)
        del source_state
        model.requires_grad_(False)
        device = torch.device('cuda:0')
        if torch.cuda.device_count() != 1:
            raise RuntimeError('Expected exactly one masked GPU')
        torch.cuda.set_per_process_memory_fraction(MAX_CUDA_BYTES / torch.cuda.get_device_properties(device).total_memory, device)
        torch.cuda.reset_peak_memory_stats(device)
        # C/R remain on CPU and are never called. Only Q is evaluated on GPU.
        model.Q.to(device)
        source_eval = base.evaluate_q(model.Q, frozen, budget_check=check)
        replay = {name: float(np.max(np.abs(source_eval[name]-formal[original])))
                  for name, original in (('direct_g','direct_g'),('direct_full_g','direct_full_g'),('decoded','reconstructed_features'))}
        if max(replay.values()) > 1e-10:
            raise RuntimeError(f'Source Q/cache replay mismatch: {replay}')
        train = np.flatnonzero(formal['splits'] == 'train')
        if len(train) != 40:
            raise ValueError('Expected 40 TRAIN chains')
        train_fields = [x[train] for x in frozen]
        train_targets = per_field[train]
        target = per_field.mean(2).numpy()
        mean = target[train].mean(0)
        std = np.sqrt(target[train].var(0).mean(-1))
        weights = np.ones(4)
        plan = dict(schema_version=5, experiment='explicit_dg1_Q_only_residual_decoder_v1',
            source=str(base.SOURCE), source_checkpoint_sha256=base.PINS['training/training_step_3000.pt'],
            frozen_cache=str(CACHE_ROOT/'frozen_fields.pt'), frozen_cache_sha256=CACHE_SHA,
            source_Q_steps=3000, additional_Q_steps_per_arm=STEPS, theory_dim=1, arms=list(ARMS),
            source_geometry=dict(block_size=2, lattice_sizes=list(base.SIZES)),
            decoder=dict(A='shared affine625', B='shared affine + centered tanh residual 1-32-32-625',
                residual_last_layer='zero initialized', spectral_norm_cap_per_residual_layer=2.,
                fixed_decoder_lipschitz_bound=9., no_injectivity_guarantee=True),
            batch=dict(physical_chains=8, fields_per_chain=32, half_fields=16, seed=192, paired=True),
            optimizer=dict(name='fresh Adam', learning_rate=.001),
            loss='10 mean_k SUM625 reconstruction +20 microscopic subset +5 mean coarse subset',
            scale_weights=[1.,1.,1.,1.], labels_used_in_training=False, test_used_in_training=False,
            R_updates=0, C_updates=0, new_physical_samples=0, new_C_trajectories=0,
            changed_sampling='fixed C cache,8x32 batch; source QCR used changing C,32x64; not exact continuation',
            training_started=False, budget=dict(gpu_reservation_seconds=GPU_SECONDS,cpu_seconds=CPU_SECONDS,
                directory_bytes=MAX_BYTES,own_cuda_peak_bytes=MAX_CUDA_BYTES),
            source_protected_sha256={str(p):s for p,s in protected.items()}, admission=admission,
            expected_gpu_uuid=expected_gpu_uuid, physical_gpu_index=gpu_index,
            warnings=source_manifest['data_warning_count'], equilibrium_certified=False, thermal_nu=None)
        base.write(output/'plan.json', plan)
        base.write(output/'cache_receipt.json', dict(cache_sha256=CACHE_SHA,source_reproduction_maxabs=replay,
            cache_reused_read_only=True,new_C_trajectories=0,original_receipt_sha256=base.sha(CACHE_ROOT/'cache_receipt.json'),
            original_target_replay_maxabs=old_receipt['source_reproduction_maxabs']['target_features']))

        def get_batch(indices, samples):
            return ([x[indices][:,samples].to(device) for x in train_fields],
                    train_targets[indices][:,:,samples].to(device))

        # Construct on CPU for deterministic residual initialization, then move.
        source_cpu = copy.deepcopy(model.Q).cpu()
        initial = make_models(source_cpu)
        grid = torch.linspace(-1.,1.,33,dtype=torch.float64).reshape(-1,1)
        if not torch.equal(initial[ARMS[0]].decode(grid),initial[ARMS[1]].decode(grid)):
            raise RuntimeError('Initial decoder functions do not match exactly')
        if base.state_sha(initial[ARMS[1]].encoder.local_mlp.state_dict()) != base.state_sha(source_cpu.local_mlp.state_dict()):
            raise RuntimeError('Initial encoder changed')
        status(current='bounded_benchmark')
        benchmark_models={name:copy.deepcopy(q).to(device) for name,q in initial.items()}
        optim={name:torch.optim.Adam(q.parameters(),lr=.001) for name,q in benchmark_models.items()}
        rng=torch.Generator().manual_seed(192)
        times=[]
        for iteration in range(4):
            torch.cuda.synchronize(); started=time.monotonic()
            levels, targets=get_batch(*base.batch_schedule(40,128,generator=rng))
            for name,q in benchmark_models.items():
                base.update(q,optim[name],levels,targets,weights)
            torch.cuda.synchronize()
            if iteration: times.append(time.monotonic()-started)
            check()
        del benchmark_models,optim,levels,targets
        torch.cuda.empty_cache()
        projection=1.5*max(times)*(STEPS+100)+150.
        base.write(output/'benchmark.json',dict(paired_update_seconds=times,warmup_pairs=1,timed_pairs=3,
            remaining_projection_seconds=projection,elapsed_seconds=time.monotonic()-wall,formal_updates=0))
        check(projection)
        models={name:q.to(device) for name,q in initial.items()}
        optimizers={name:torch.optim.Adam(q.parameters(),lr=.001) for name,q in models.items()}
        initial_pins={name:base.state_sha(q.state_dict()) for name,q in models.items()}
        history=[]

        def log(step):
            row=dict(step=step,arms={name:base.probe(q,train_fields,train_targets,weights,check) for name,q in models.items()})
            row['arms'][ARMS[1]]['decoder_bounds']=models[ARMS[1]].decoder.bounds()
            history.append(row)
            base.write(output/'history.json',history)
            check(directory=True)

        log(0)
        plan['training_started']=True
        base.write(output/'plan.json',plan)
        status(current='paired_Q_training',training_started=True)
        rng=torch.Generator().manual_seed(192); schedule=hashlib.sha256()
        for step in range(1,STEPS+1):
            indices,samples=base.batch_schedule(40,128,generator=rng)
            schedule.update(indices.numpy().tobytes());schedule.update(samples.numpy().tobytes())
            levels,targets=get_batch(indices,samples)
            for name,q in models.items():base.update(q,optimizers[name],levels,targets,weights)
            del levels,targets
            check(150.)
            if step%50==0:log(step);status(actual_steps=step)
        status(current='fixed_endpoint_evaluation',actual_steps=STEPS)
        arrays=dict(target_features=target,train_target_mean=mean,train_target_std=std,
            source_flow_grid=formal['flow_grid'],source_flow_next=formal['flow_next'])
        decoder_info={}
        evaluated={'source':source_eval}
        for name,q in models.items():
            evaluated[name]=base.evaluate_q(q,frozen,budget_check=check)
            torch.save(dict(Q=q.state_dict(),decoder_architecture=plan['decoder'],source_Q_steps=3000,
                additional_Q_steps=STEPS,theory_dim=1),output/(name+'_Q_final.pt'))
        for name,values in evaluated.items():
            for key,value in values.items():arrays[name+'_'+key]=value
            q=model.Q if name=='source' else models[name]
            _, jac, info=decoder_diagnostics(q,values['direct_g'])
            arrays[name+'_decoder_jacobian']=jac.reshape(60,4,625)
            gtrain=values['direct_g'][train].reshape(-1)
            decode_grid=np.linspace(gtrain.min(),gtrain.max(),201).reshape(-1,1)
            dec,der,grid_info=decoder_diagnostics(q,decode_grid)
            arrays[name+'_decoder_grid']=decode_grid
            arrays[name+'_decoder_grid_output']=dec
            arrays[name+'_decoder_grid_jacobian']=der
            decoder_info[name]=dict(observed=info,train_interval_grid=grid_info)
        arrays.update({name:formal[name] for name in ('labels','splits','cell_ids')})
        if any(v.dtype.kind in 'fc' and not np.isfinite(v).all() for v in arrays.values()):
            raise FloatingPointError('Nonfinite endpoint arrays')
        np.savez_compressed(output/'evaluation_arrays.npz',**arrays)
        metrics={split:{name:base.calculate_metrics(target[formal['splits']==split],values['decoded'][formal['splits']==split],std,mean)
            for name,values in evaluated.items()} for split in ('train','test')}
        base.write(output/'metrics.json',metrics)
        counts={name:base.optimizer_counts(optimizers[name],q) for name,q in models.items()}
        if any(c['minimum']!=STEPS or c['maximum']!=STEPS for c in counts.values()):
            raise RuntimeError('Optimizer counters differ from fixed endpoint')

        def verify():
            if any(base.sha(p)!=s for p,s in protected.items()):
                raise RuntimeError('Protected source/cache/code changed')
            if any(base.state_sha(getattr(model,n).state_dict())!=source_summary['final_module_sha256'][n]
                   or any(p.grad is not None for p in getattr(model,n).parameters()) for n in ('Q','C','R')):
                raise RuntimeError('Source Q/C/R changed or acquired gradients')
            if any(p.grad is not None for q in models.values() for p in q.parameters()):
                raise RuntimeError('Final Q gradients not cleared')

        verify()
        summary=dict(status='complete',actual_steps=STEPS,source_Q_steps=3000,additional_Q_steps_per_arm=STEPS,
            theory_dim=1,optimizer_parameter_steps=counts,C_updates=0,R_updates=0,
            source_module_sha256=source_summary['final_module_sha256'],initial_Q_sha256=initial_pins,
            final_Q_sha256={name:base.state_sha(q.state_dict()) for name,q in models.items()},
            source_Q_C_R_unchanged=True,all_final_gradients_none=True,paired_batch_schedule_sha256=schedule.hexdigest(),
            labels_used_in_training=False,test_used_in_training=False,model_selection=False,early_stopping=False,
            warnings=source_manifest['data_warning_count'],equilibrium_certified=False,thermal_nu=None,
            old_R_valid_for_new_Q=False,decoder_diagnostics=decoder_info,
            elapsed_seconds_before_report=time.monotonic()-wall,own_peak_cuda_bytes=torch.cuda.max_memory_allocated(),
            threads=torch.get_num_threads(),nice=os.getpriority(os.PRIO_PROCESS,0))
        base.write(output/'summary.json',summary)
        status(current='array_only_report')
        report=importlib.import_module('scripts.report_phi4_hopping_qonly_nonlinear')
        public_report=report.render_report(output)
        verify();check(directory=True)
        base.write(output/'execution.json',dict(status='complete',returncode=0,
            source_cache_code_unchanged_after_report=True,elapsed_reservation_seconds=time.monotonic()-wall,
            cpu_seconds=time.process_time(),own_peak_cuda_bytes=torch.cuda.max_memory_allocated()))
        status(status='complete',workflow_complete=True,current='complete',returncode=0,error=None,requires_user_action=False)
        public=['plan.json','cache_receipt.json','benchmark.json','history.json','evaluation_arrays.npz',
            'metrics.json','summary.json','execution.json','job.json','status.json',*public_report]
        if any(Path(p).is_absolute() or '..' in Path(p).parts or Path(p).suffix not in ('.json','.png','.npz') for p in public):
            raise ValueError('Unsafe public artifact path')
        base.write(output/'public_files.json',dict(files=[dict(path=p,bytes=(output/p).stat().st_size,sha256=base.sha(output/p)) for p in public],
            forbidden='No PT/checkpoint/raw fields/C cache download'))
        return 0
    except Exception as exc:
        traceback.print_exc()
        status(status='budget_hold' if isinstance(exc,base.BudgetHold) else 'failed',current=job['current'],
            workflow_complete=False,returncode=2,error=str(exc),requires_user_action=True)
        return 2


def main():
    parser=argparse.ArgumentParser(description='dg=1 的 Q-only 非线性解码器配对实验')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu-index',type=int,default=3)
    parser.add_argument('--gpu-uuid',default='GPU-a167decd-ce49-6c8f-6bcf-136fde596683')
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(execution_requested=False,arms=ARMS,theory_dim=1,extra_Q_steps=STEPS,
            gpu_seconds=GPU_SECONDS,cache=str(CACHE_ROOT),output=str(args.output)),indent=2))
        return 0
    return run(args.output,args.gpu_index,args.gpu_uuid)


if __name__=='__main__':
    raise SystemExit(main())
