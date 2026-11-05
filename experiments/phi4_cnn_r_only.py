"""Pinned online R-only continuation; frozen Q/C, original R/Adam/targets.

600 new batches x10 R steps, exactly as the source's R inner loop. No latent
coordinate normalization, loss change, MI, labels, or TEST-based selection.
"""
import copy
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from experiments import phi4_hopping_qcr as base
from experiments.phi4_hopping_cnn_qcr import (
    DATA, DATA_SHA, GPU_UUID, REFINED_RECIPE, draw_batch, resource_snapshot, root_diagnostics,
)
from experiments.ising_cnn_r_only import r_loss, paired_predictions
from recurrent_rg.data.phi4_hopping_loader import load_dataset
from recurrent_rg.phi4_cnn_qcr import CNNPhi4QCR

SOURCE = Path('/mnt/nas-new/home/yange/ham/results/phi4_hopping_b3_cnn_refinedR_seed91_gpu3_vm225_20261102_v1')
SOURCE_SHA = '41a173263dcf351af2076da530551824e57fc777b65dc2ede57b31fb3976e464'
RECIPE = 'phi4_hopping_b3_cnn_online_R_only_v1'
EXTRA_STEPS = 6000
INNER_STEPS = 10
SOURCE_CLOCK = 9000


def freeze_and_continue(model, payload, expected=SOURCE_CLOCK):
    for p in model.parameters():
        p.requires_grad_(False)
        p.grad = None
    for p in model.R.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(model.R.parameters(), lr=.001)
    opt.load_state_dict(copy.deepcopy(payload['optimizers']['R']))
    if len(opt.param_groups) != 1 or opt.param_groups[0]['lr'] != .001:
        raise ValueError('Original R Adam lr.001 required')
    base.optimizer_step_counts({'R': opt}, expected)
    return opt


@torch.no_grad()
def online_targets(model, batch, generator):
    levels = model.coarse_grain(batch, generator=generator)
    n = batch.shape[1] // 2
    return torch.stack([.5*(model.Q(x[:, :n])+model.Q(x[:, n:])) for x in levels], 1).detach()


def update(model, direct, opt, repeats=INNER_STEPS):
    value = None
    for _ in range(repeats):
        opt.zero_grad(set_to_none=True)
        loss = r_loss(model, direct)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite R objective')
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.R.parameters()):
            raise FloatingPointError('Invalid R gradient')
        opt.step()
        value = float(loss.detach())
    return value


@torch.no_grad()
def train_panel(model, configurations, device, check):
    # Isolated explicit generators: do not consume either continuation stream.
    rng = torch.Generator().manual_seed(91910)
    field_rng = torch.Generator(device=device).manual_seed(91910)
    rows = []
    for _ in range(3):
        indices = torch.randperm(configurations.shape[1], generator=rng)[:64]
        for chunk in configurations.split(32):
            rows.append(online_targets(model, chunk[:, indices].to(device), field_rng))
            check()
    return torch.cat(rows)


def validate_args(args):
    from experiments.rsmi_separate_qcr import build_parser
    if args.source is None:
        raise ValueError('Pinned --source required')
    defaults = vars(build_parser().parse_args(['--source', str(args.source), '--output', str(args.output)]))
    allowed = {'source', 'output', 'device', 'phi4_cnn_r_only'}
    changed = [k for k, v in vars(args).items() if k not in allowed and v != defaults[k]]
    if changed or args.source.resolve() != SOURCE or args.device != 'cuda:0':
        raise ValueError('Only pinned source/output/device and explicit R-only flag allowed: '+str(changed))


def validate_source(source=SOURCE):
    status = json.loads((source/'status.json').read_text())
    if (status['status'] != 'complete' or status['workflow_complete'] is not True
            or status['returncode'] != 0 or status['error'] is not None):
        raise ValueError('Source is incomplete')
    names = ['latest.pt', 'status.json', 'manifest.json', 'summary.json', 'roots.json', 'evaluation_arrays.npz']
    pins = {str(source/n): base.sha256_file(source/n) for n in names}
    if pins[str(source/'latest.pt')] != SOURCE_SHA:
        raise ValueError('Source checkpoint SHA differs')
    for name in names[2:]:
        expected = next(row['sha256'] for row in status['public_files'] if row['path'] == name)
        if pins[str(source/name)] != expected:
            raise ValueError('Source public file SHA differs: '+name)
    payload = torch.load(source/'latest.pt', map_location='cpu', weights_only=True)
    manifest = json.loads((source/'manifest.json').read_text())
    summary = json.loads((source/'summary.json').read_text())
    if (payload['recipe'] != REFINED_RECIPE or payload['step'] != 900
            or payload['manifest'] != manifest or manifest['loss'] != base.LOSS_POLICY
            or manifest['options']['r_updates_per_batch'] != INNER_STEPS
            or manifest['lattice_sizes'] != [36, 12, 4] or manifest['theory_dim'] != 1
            or manifest['data_root'] != str(DATA) or manifest['data_manifest_sha256'] != DATA_SHA):
        raise ValueError('Source recipe/model/objective differs')
    return payload, manifest, summary, pins


def run(args):
    validate_args(args)
    start = time.monotonic()
    if (os.uname().nodename != 'thinklab-105-225' or os.getpriority(os.PRIO_PROCESS, 0) < 10
            or os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID or torch.cuda.is_initialized()):
        raise RuntimeError('Wrong host/priority/GPU mask')
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    actual = 0
    job = dict(status='running', workflow_complete=False, pid=os.getpid(), returncode=None,
               error=None, requires_user_action=False)
    def status(**kw):
        job.update(kw, extra_R_updates=actual, elapsed_seconds=time.monotonic()-start)
        base.write_json(out/'status.json', job)
        print(json.dumps(job), flush=True)
    def check(reserve=0):
        if time.monotonic()-start+reserve > 1200:
            raise base.BudgetHold('20-minute total budget; no automatic restart')
        if torch.cuda.is_initialized() and max(torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()) > 8*1024**3:
            raise base.BudgetHold('Own GPU peak above8GiB')
    try:
        status(current='admission')
        admission = [resource_snapshot()]
        time.sleep(10)
        admission.append(resource_snapshot())
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        payload, source_manifest, source_summary, source_pins = validate_source()
        if base.sha256_file(DATA/'manifest.json') != DATA_SHA:
            raise ValueError('Dataset differs')
        dataset = load_dataset(DATA)
        if (dataset.configurations.shape != (60, 128, 36, 36, 1) or len(dataset.train_indices) != 40
                or len(dataset.test_indices) != 20 or dataset.scale != source_manifest['scale']):
            raise ValueError('Source data layout/normalization differs')
        device = torch.device(args.device)
        receipt = base._device_receipt(device, GPU_UUID)
        torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(device).total_memory)
        torch.cuda.reset_peak_memory_stats()
        model = CNNPhi4QCR(dataset.scale).to(device).eval()
        model.load_state_dict(payload['model'], strict=True)
        if (base.state_fingerprint(model) != source_summary['final_state_sha256']
                or base.module_fingerprints(model) != source_summary['final_module_sha256']):
            raise ValueError('Model fingerprint differs')
        opt = freeze_and_continue(model, payload)
        qc_before = {k: base.state_fingerprint(getattr(model, k)) for k in ('Q', 'C')}
        initial_R = copy.deepcopy(model.R.state_dict())
        initial_opt = copy.deepcopy(opt.state_dict())
        def streams():
            rng = torch.Generator()
            rng.set_state(payload['batch_rng'])
            field_rng = torch.Generator(device=device)
            field_rng.set_state(payload['field_rng'])
            return rng, field_rng
        status(current='TRAIN_panel_and_disposable_timing')
        panel = train_panel(model, dataset.configurations[dataset.train_indices], device, check)
        rng, field_rng = streams()
        timings = []
        for _ in range(10):
            torch.cuda.synchronize()
            tick = time.monotonic()
            update(model, online_targets(model, draw_batch(dataset, rng, device), field_rng), opt)
            torch.cuda.synchronize()
            timings.append(time.monotonic()-tick)
            check()
        model.R.load_state_dict(initial_R)
        opt.load_state_dict(initial_opt)
        base.optimizer_step_counts({'R': opt}, SOURCE_CLOCK)
        rng, field_rng = streams()
        conservative = 1.3*max(timings[2:])
        projected = time.monotonic()-start + EXTRA_STEPS/INNER_STEPS*conservative + 90
        if projected > 1200:
            raise base.BudgetHold('Fixed6000 R updates do not fit20minutes; no shortened run')
        import scripts.report_ising_cnn_r_only as report_module
        import recurrent_rg.phi4_cnn_qcr as cnn_module
        import experiments.ising_cnn_r_only as shared_module
        import experiments.rsmi_separate_qcr as cli_module
        import experiments.phi4_hopping_cnn_qcr as source_module
        pins = base.source_pins()
        for module in (report_module, cnn_module, shared_module, cli_module, source_module):
            pins[str(Path(module.__file__).resolve())] = base.sha256_file(module.__file__)
        pins[str(Path(__file__).resolve())] = base.sha256_file(__file__)
        manifest = dict(recipe=RECIPE, source=str(SOURCE), source_files_sha256=source_pins,
            source_joint_updates=900, source_R_updates=SOURCE_CLOCK, planned_extra_R_updates=EXTRA_STEPS,
            r_updates_per_batch=INNER_STEPS, data_root=str(DATA), data_manifest_sha256=DATA_SHA,
            data_warning_count=dataset.warning_count, equilibrium_certified=False,
            source_sha256=pins, admission=admission, device=receipt, frozen_before_sha256=qc_before,
            optimized_modules=['R'], frozen_modules=['Q', 'C'], labels_used_in_training=False, MI_evaluated=False,
            R_optimizer='Continue original Adam.001 with moments and clocks9000; no reset or scheduler',
            target_policy='Original TRAIN32 chains x64 random fields; fresh stochastic C per batch; mean of two half-subset Q outputs;10 R updates per batch; no latent-cache training',
            objective='Unchanged uniform-target all_start_anchor:5e01+2.5e02+2.5e12; no coordinate normalization or added loss',
            endpoint='Fixed6000 extra R updates; no TRAIN/TEST/root-based stopping or model selection',
            panel=dict(physical_TRAIN_chains=40, C_draws_per_chain=3, fields_per_draw=64, seed=91910, stopping_used=False),
            timing=dict(discarded_pilot_batches=10, discarded_R_updates=100, batch_seconds=timings,
                conservative_batch_seconds=conservative, projected_total_seconds=projected, hard_seconds=1200),
            thermal_nu=None)
        base.write_json(out/'manifest.json', manifest)
        base.write_json(out/'plan.json', manifest)
        history = []
        def checkpoint():
            with torch.no_grad():
                value = float(r_loss(model, panel))
            if not np.isfinite(value):
                raise FloatingPointError('Nonfinite TRAIN panel loss')
            history.append(dict(extra_R_updates=actual, train={'total':value}, elapsed_seconds=time.monotonic()-start))
            base.write_json(out/'history.json', history)
            torch.save(dict(recipe=RECIPE, source_joint_step=900, R_steps=SOURCE_CLOCK+actual,
                extra_R_updates=actual, model=model.state_dict(), optimizers={'R': opt.state_dict()},
                manifest=manifest, batch_rng=rng.get_state(), field_rng=field_rng.get_state()), out/'latest.pt')
            status(current='training', planned_extra_R_updates=EXTRA_STEPS, train_panel_loss=value)
        checkpoint()
        for batch_index in range(1, EXTRA_STEPS//INNER_STEPS+1):
            check(90)
            update(model, online_targets(model, draw_batch(dataset, rng, device), field_rng), opt)
            actual = batch_index*INNER_STEPS
            if actual % 500 == 0:
                checkpoint()
        qc_after = {k:base.state_fingerprint(getattr(model,k)) for k in ('Q','C')}
        if qc_before != qc_after or any(p.grad is not None for name in ('Q','C') for p in getattr(model,name).parameters()):
            raise AssertionError('Frozen Q/C changed or received gradients')
        clocks = base.optimizer_step_counts({'R':opt}, SOURCE_CLOCK+actual)
        for p in model.parameters():
            p.grad = None
        final_R_sha = base.state_fingerprint(model.R)
        status(current='paired_final_evaluation')
        with np.load(SOURCE/'evaluation_arrays.npz', allow_pickle=False) as z:
            original = {k:z[k].copy() for k in z.files}
        arrays = paired_predictions(model, original)
        roots = root_diagnostics(model, arrays)
        final_R = copy.deepcopy(model.R.state_dict())
        model.R.load_state_dict(initial_R)
        before = paired_predictions(model, arrays)
        for key in ('one_step_g','rollout_g','all_start_g','flow_next'):
            np.testing.assert_allclose(before[key], original[key], rtol=1e-12, atol=1e-12)
            arrays['source_'+key] = before[key]
        model.R.load_state_dict(final_R)
        if base.state_fingerprint(model.R) != final_R_sha:
            raise AssertionError('Evaluation changed R')
        np.savez(out/'evaluation_arrays.npz', **arrays)
        base.write_json(out/'roots.json', roots)
        summary = dict(status='complete', reason='fixed_endpoint', source_joint_updates=900,
            source_R_updates=SOURCE_CLOCK, actual_extra_R_updates=actual, R_Adam_steps=clocks,
            additional_Q_updates=0, additional_C_updates=0, QC_gradients_none=True,
            frozen_before_sha256=qc_before, frozen_after_sha256=qc_after,
            before_closure=base.closure_metrics(before), after_closure=base.closure_metrics(arrays),
            train_panel_initial=history[0]['train']['total'], train_panel_final=history[-1]['train']['total'],
            final_R_sha256=final_R_sha, peak_cuda_bytes=torch.cuda.max_memory_allocated(),
            thermal_nu=None, equilibrium_certified=False, data_warning_count=dataset.warning_count,
            limitations=['Single seed, finite size and correlated chains; fixed final C draws.','No global optimum or physical critical-point certification.'])
        base.write_json(out/'summary.json', summary)
        status(current='reporting')
        report_module.render(out, family='phi4')
        check()
        base.verify_dataset_unchanged(DATA, dataset)
        if any(base.sha256_file(p) != h for p,h in (source_pins | pins).items()):
            raise ValueError('Frozen source/data/code changed')
        public = [dict(path=str(p.relative_to(out)), sha256=base.sha256_file(p), bytes=p.stat().st_size)
            for p in sorted(out.rglob('*')) if p.is_file() and p.suffix in ('.json','.npz','.png') and p.name!='status.json']
        status(status='complete', current='complete', workflow_complete=True, returncode=0, public_files=public)
    except Exception as exc:
        status(status='budget_hold' if isinstance(exc,base.BudgetHold) else 'failed', error=str(exc),
            traceback=traceback.format_exc(), returncode=3 if isinstance(exc,base.BudgetHold) else 1, requires_user_action=True)
        raise
