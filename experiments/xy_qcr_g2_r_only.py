"""Pinned terminal R-only fit of the completed temperature-only dg=2 XY trial."""
import copy
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from experiments import phi4_hopping_qcr as base
from experiments import xy_qcr_g2 as source_trial
from experiments.phi4_hopping_cnn_qcr import GPU_UUID, resource_snapshot
from recurrent_rg.config import LossConfig
from recurrent_rg.terminal_r_fit import collect_train_targets, fit_r, freeze_qc
from recurrent_rg.xy_qcr import XYQCR

RECIPE = 'xy2d_b3_local_g2_terminal_fixed_QC_R_v1'
SOURCE = Path('/mnt/nas-new/home/yange/ham/results/xy2d_b3_local_g2_matched1250_seed91_gpu3_vm225_20261105_v1')
CKPT_SHA = '10282611271d4709b39169479e364af72a360577342f2ff953f57557364cc589'


def preset():
    return dict(schema_version=5, recipe=RECIPE, source=str(SOURCE), source_joint_steps=1250,
        source_checkpoint_sha256=CKPT_SHA, data_sha256=source_trial.DATA_SHA,
        max_extra_R_updates=10000, cache_repeats=8, cache_seed=91915,
        samples=32, systems=16, learning_rate_R=.003,
        closure_objective='all_start_anchor', closure_scale_weights=None,
        stopping='TRAIN only: <1% best-loss improvement over2000 updates; current within5% of best; 3 checks every500 after>=3000',
        hard_seconds=1200, max_cuda_bytes=8*1024**3, max_output_bytes=2*1024**3)


def validate_config(path):
    c = json.loads(Path(path).read_text())
    if c != preset(): raise ValueError('Exact pinned XY terminal R-only recipe required')
    return c


def restore(payload, expected, device):
    if payload.get('recipe') != source_trial.RECIPE or payload.get('step') != 1250:
        raise ValueError('Wrong XY source recipe/update')
    model = XYQCR(seed=91, dim=2).to(device)
    model.load_state_dict(payload['model'], strict=True)
    if base.module_fingerprints(model) != expected: raise ValueError('Source module fingerprints differ')
    freeze_qc(model, model.R)
    opt = torch.optim.Adam(model.R.parameters(), lr=.003)
    opt.load_state_dict(copy.deepcopy(payload['optimizers']['R']))
    if any(group['lr'] != .003 for group in opt.param_groups): raise ValueError('Source R learning rate changed')
    base.optimizer_step_counts({'R':opt},1250)
    return model, opt


@torch.no_grad()
def paired_predictions(model, before):
    a = {k:v.copy() for k,v in before.items() if not k.startswith('flow_')}
    g = torch.as_tensor(a['direct_g'], dtype=torch.float64, device=next(model.parameters()).device)
    one = model.advance(g[:, :-1]); roll1 = model.advance(g[:, 0]); roll2 = model.advance(roll1)
    a.update(one_step_g=one.cpu().numpy(), rollout_g=torch.stack([g[:, 0], roll1, roll2], 1).cpu().numpy(),
        all_start_g=torch.stack([one[:, 0], roll2, one[:, 1]], 1).cpu().numpy())
    return a


def run(config_path, output):
    c = validate_config(config_path); out = Path(output).resolve(); out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); extra = 0
    job = dict(status='running', workflow_complete=False, current='admission', pid=os.getpid(),
        returncode=None, error=None, requires_user_action=False, joint_updates=1250)

    def status(**values):
        job.update(values, extra_R_updates=extra, elapsed_seconds=time.monotonic()-started)
        base.write_json(out/'job.json', job); base.write_json(out/'status.json', job)
        print(json.dumps(job, ensure_ascii=False), flush=True)

    def check():
        if time.monotonic()-started > c['hard_seconds']-60:
            raise base.BudgetHold('20-minute whole-stage budget; no restart or silent truncation')
        if torch.cuda.is_initialized() and max(torch.cuda.max_memory_reserved(), torch.cuda.max_memory_allocated()) > c['max_cuda_bytes']:
            raise base.BudgetHold('Own GPU memory budget exceeded')

    try:
        if os.uname().nodename != 'thinklab-105-225' or os.getpriority(os.PRIO_PROCESS, 0) < 10 or os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID:
            raise RuntimeError('Wrong host, priority or GPU')
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        from scripts.run_ising_highdim import source_hashes
        code = Path(__file__).resolve().parents[1]; pins = source_hashes(code)
        config_sha = base.sha256_file(config_path)
        status(); admission = [resource_snapshot()]; time.sleep(10); admission.append(resource_snapshot())
        source_job = json.loads((SOURCE/'job.json').read_text())
        if not (source_job['status']=='complete' and source_job['workflow_complete'] and source_job['returncode']==0):
            raise ValueError('Incomplete joint source')
        ckpt = SOURCE/'training_step_1250.pt'
        if base.sha256_file(ckpt) != CKPT_SHA: raise ValueError('Pinned checkpoint changed')
        public = {r['path']:r['sha256'] for r in source_job['public_files']}
        names = ['manifest.json','summary.json','evaluation_arrays.npz','roots.json','history.json',
                 'data/metadata.json','reference_scalar.json']
        for name in names:
            if base.sha256_file(SOURCE/name) != public[name]: raise ValueError('Source public file changed: '+name)
        before_summary = json.loads((SOURCE/'summary.json').read_text())
        with np.load(SOURCE/'evaluation_arrays.npz', allow_pickle=False) as z:
            before = {k:z[k].copy() for k in z.files}
        np.savez(out/'before_evaluation_arrays.npz', **before)
        for name, dest in [('summary.json','before_summary.json'),('roots.json','before_roots.json'),
                           ('history.json','history.json'),('reference_scalar.json','reference_scalar.json'),
                           ('data/metadata.json','data/metadata.json')]:
            p = out/dest; p.parent.mkdir(exist_ok=True); p.write_bytes((SOURCE/name).read_bytes())
        status(current='loading_pinned_training_data')
        data, meta, data_pins = source_trial.load_data()
        device = torch.device('cuda:0'); receipt = base._device_receipt(device, GPU_UUID)
        torch.cuda.set_per_process_memory_fraction(c['max_cuda_bytes']/torch.cuda.get_device_properties(device).total_memory)
        torch.cuda.reset_peak_memory_stats()
        payload = torch.load(ckpt, map_location='cpu', weights_only=True)
        model, opt = restore(payload, before_summary['final_module_sha256'], device)
        initial = base.module_fingerprints(model)
        manifest = dict(recipe=RECIPE, config=c, source=str(SOURCE), source_checkpoint_sha256=CKPT_SHA,
            source_public_sha256={name:public[name] for name in names}, source_sha256=pins,
            config_sha256=config_sha, source_joint_updates=1250, initial_sha256=initial,
            optimized_modules=['R'], frozen_modules=['Q','C'], labels_used_in_training=False,
            R_optimizer='Original Adam lr.003, moments and clocks1250 retained',
            target_policy='Fixed TRAIN cache: 40 physical chains x8 subset/C draws, 32 samples split16+16; all rows equally weighted',
            sampling_change='Joint online subsets replaced by one fixed finite cache; no new physical samples',
            closure_objective_unchanged='5 e01 + 2.5 e02 + 2.5 e12; recurrent R gradients retained',
            evaluation_policy='Same stored before/after physical chain encodings; no fresh Q/C evaluation',
            MI_evaluated=False, admission=admission, device=receipt, thermal_nu=None, BKT_certified=False)
        base.write_json(out/'manifest.json', manifest); base.write_json(out/'plan.json', manifest)
        base.write_json(out/'config.json', c); status(current='fixed_TRAIN_encoding_cache')

        @torch.no_grad()
        def encode(x):
            half = x.shape[1]//2
            return torch.stack([.5*(model.Q(v[:, :half])+model.Q(v[:, half:])) for v in model.coarse_grain(x)], 1)

        targets, cache = collect_train_targets(model, data.configurations[data.train_indices], encode,
            samples=c['samples'], systems=c['systems'], repeats=c['cache_repeats'], seed=c['cache_seed'], budget_check=check)
        torch.save(targets, out/'train_targets.pt'); base.write_json(out/'cache.json', cache)
        loss_config = LossConfig()
        status(current='disposable_R_timing')
        pilot = copy.deepcopy(model); pilot_opt = torch.optim.Adam(pilot.R.parameters(), lr=.003)
        pilot_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
        torch.cuda.synchronize(); tick = time.monotonic()
        fit_r(pilot,pilot.R,{'Q':pilot.Q,'C':pilot.C},pilot_opt,targets,loss_config,max_steps=2)
        torch.cuda.synchronize(); warmup_seconds=time.monotonic()-tick; tick=time.monotonic()
        fit_r(pilot,pilot.R,{'Q':pilot.Q,'C':pilot.C},pilot_opt,targets,loss_config,max_steps=8)
        torch.cuda.synchronize(); per_step = (time.monotonic()-tick)/8
        projection = time.monotonic()-started+1.5*per_step*c['max_extra_R_updates']+180
        base.write_json(out/'timing_plan.json',dict(pilot_updates_discarded=10, warmup_updates=2,
            warmup_seconds=warmup_seconds, measured_updates=8, seconds_per_update=per_step,
            projected_total_seconds=projection, max_extra_R_updates=c['max_extra_R_updates']))
        del pilot, pilot_opt; torch.cuda.empty_cache()
        if projection > c['hard_seconds']-30: raise base.BudgetHold('Fixed R update cap/report do not fit budget')

        def save(row, rows):
            nonlocal extra
            extra = row['extra_R_updates']; base.write_json(out/'r_history.json', rows)
            torch.save(dict(recipe=RECIPE, joint_steps=1250, extra_R_updates=extra,
                R_total_updates=1250+extra, model=model.state_dict(), optimizers={'R':opt.state_dict()},
                manifest=manifest, train_cache_sha256=base.sha256_file(out/'train_targets.pt')),
                out/'checkpoint_last.pt')
            status(current='R_only_training', max_extra_R_updates=c['max_extra_R_updates'], train_cache_loss=row['train_loss'])

        result = fit_r(model,model.R,{'Q':model.Q,'C':model.C},opt,targets,loss_config,
            max_steps=c['max_extra_R_updates'],callback=save,budget_check=check)
        final = base.module_fingerprints(model)
        if any(initial[n] != final[n] for n in ('Q','C')): raise AssertionError('Q/C changed')
        status(current='paired_frozen_evaluation')
        a = paired_predictions(model, before)
        roots = source_trial.flow_diagnostics(model, a, device, check)
        if base.module_fingerprints(model) != final or any(p.grad is not None for p in model.parameters()):
            raise AssertionError('Evaluation mutated weights or gradients')
        np.savez(out/'evaluation_arrays.npz', **a); base.write_json(out/'roots.json', roots)
        summary = dict(status='complete', actual_steps=1250+extra, joint_updates=1250,
            extra_R_updates=extra, R_total_updates=1250+extra, R_fit=result,
            closure=base.closure_metrics(a), before_closure=before_summary['closure'],
            information_diagnostic=before_summary['information_diagnostic'], information_diagnostic_reused=True,
            C_final_exactly_matches_scalar=True, final_module_sha256=final, Q_C_unchanged=True,
            optimizer_steps={'Q':{'min':1250,'max':1250},'C':{'min':1250,'max':1250},
                             'R':base.optimizer_step_counts({'R':opt},1250+extra)['R']},
            peak_cuda_bytes=torch.cuda.max_memory_allocated(), elapsed_seconds=time.monotonic()-started,
            BKT_certified=False, thermal_nu=None, labels_used_in_training=False)
        base.write_json(out/'summary.json', summary); status(current='reporting')
        from scripts.report_xy_qcr_g2_r_only import render
        render(out); check()
        if source_hashes(code) != pins or base.sha256_file(config_path) != config_sha or base.sha256_file(ckpt) != CKPT_SHA:
            raise RuntimeError('Code/config/source changed')
        for name in names:
            if base.sha256_file(SOURCE/name) != public[name]: raise RuntimeError('Source public result changed')
        for name,sha in data_pins.items():
            if base.sha256_file(source_trial.SOURCE/name)!=sha: raise RuntimeError('Physical source changed')
        if sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>c['max_output_bytes']:
            raise base.BudgetHold('Output budget exceeded')
        files=[dict(path=str(p.relative_to(out)),sha256=base.sha256_file(p),bytes=p.stat().st_size)
            for p in sorted(out.rglob('*')) if p.is_file() and p.suffix in ('.json','.png','.npz') and p.name not in ('job.json','status.json')]
        status(status='complete',workflow_complete=True,current='complete',returncode=0,
               stop_reason=result['stop_reason'],public_files=files)
    except Exception as exc:
        status(status='budget_hold' if isinstance(exc,base.BudgetHold) else 'failed',
            returncode=3 if isinstance(exc,base.BudgetHold) else 1,error=str(exc),
            requires_user_action=True,traceback=traceback.format_exc())
        raise
