"""Fresh dimension-only XY control on the immutable scalar trial's data."""
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
from experiments.phi4_hopping_cnn_qcr import GPU_UUID, resource_snapshot
from recurrent_rg.xy_qcr import XYQCR, independent_step, circular_information
from recurrent_rg.phi4_qcr import sample_regions
from recurrent_rg.phi4_ru_flow import bounded_flow, full_map_and_jacobian

RECIPE = 'xy2d_b3_local_periodic_freegain_g2_matched1250_seed91_v1'
SOURCE = Path('/mnt/nas-new/home/yange/ham/results/xy2d_b3_local_g1_seed91_gpu3_vm225_20261104_v1')
DATA_SHA = '2ec7177aa8c7e617f2889483841764c96cbe978a2bb459e16f095657b9e7bad1'
METADATA_SHA = 'd04ef28ec1a1fa52f71b5d2224233bd0a1f9686e4b3bc5473be3a3dd073bd74d'
MANIFEST_SHA = '9a1533ce34482388a4b8ab261f6687c925c4eb9ab3cf54beedec4fc10de680cc'
SUMMARY_SHA = '05b6218a765859fa5c2f7cfb286c581816d5ec09dc50fbc0b0307be65d7de2c4'


def preset():
    c = original.preset()
    c.update(recipe=RECIPE, theory_dim=2, reuse_data_from=str(SOURCE), data_sha256=DATA_SHA,
             data_metadata_sha256=METADATA_SHA, reference_manifest_sha256=MANIFEST_SHA,
             reference_summary_sha256=SUMMARY_SHA, comparison_step=1250,
             batch_rng_seed=192, field_rng_seed=293,
             dimension_ablation='Only latent dimension changes; fresh random Q/C/R and Adam',
             flow_grid_side=33, root_seed_side=7, root_max_nfev=64)
    return c


def validate_config(path):
    c = json.loads(Path(path).read_text())
    if c != preset() or any(type(c[k]) is not int for k in ('theory_dim', 'steps', 'seed')):
        raise ValueError('Exact explicit XY dimension-only control required')
    return c


def load_data():
    pins = {'data/angles.npz': DATA_SHA, 'data/metadata.json': METADATA_SHA,
            'manifest.json': MANIFEST_SHA, 'summary.json': SUMMARY_SHA}
    for name, expected in pins.items():
        if base.sha256_file(SOURCE/name) != expected:
            raise RuntimeError(f'Pinned scalar source changed: {name}')
    job = json.loads((SOURCE/'job.json').read_text())
    if not (job['status'] == 'complete' and job['workflow_complete'] and job['returncode'] == 0):
        raise ValueError('Scalar reference is incomplete')
    meta = json.loads((SOURCE/'data/metadata.json').read_text())
    with np.load(SOURCE/'data/angles.npz', allow_pickle=False) as z:
        a = z['configurations']
    if a.shape != (60, 128, 54, 54, 1) or not np.isfinite(a).all():
        raise ValueError('Wrong physical data geometry or nonfinite angles')
    splits = np.asarray(meta['splits'])
    data = SimpleNamespace(configurations=torch.from_numpy(a), labels=np.asarray(meta['labels']),
        cell_ids=np.asarray(meta['cell_ids']), splits=splits,
        train_indices=np.flatnonzero(splits == 'train'), test_indices=np.flatnonzero(splits == 'test'))
    if len(data.train_indices) != 40 or len(data.test_indices) != 20:
        raise ValueError('Changed whole-temperature split')
    return data, meta, pins


def flow_diagnostics(model, arrays, device, check=None):
    # A numerically identity map must not yield a spurious collection of isolated roots.
    arrays['train_direct'] = arrays['direct_g'][arrays['splits'] == 'train']
    arrays['test_direct'] = arrays['direct_g'][arrays['splits'] == 'test']
    pts = arrays['train_direct'][:, :2].reshape(-1, 2)
    low, high = pts.min(0), pts.max(0)
    width = np.maximum(high-low, 1e-6)
    bounds = np.column_stack((low-.1*width, high+.1*width))
    grid = np.stack(np.meshgrid(*[np.linspace(*b, 33) for b in bounds]), -1).reshape(-1, 2)
    mapped, dr = full_map_and_jacobian(model, grid, device)
    near_identity = bool(np.max(np.linalg.norm(mapped-grid, axis=1)) <= 1e-10*max(1., np.linalg.norm(width)))
    if near_identity:
        eig = np.linalg.eigvals(dr)
        arrays.update(flow_g=grid, flow_R=mapped, flow_displacement=mapped-grid, flow_DR=dr,
                      flow_eigenvalues_real=eig.real, flow_eigenvalues_imag=eig.imag, flow_bounds=bounds)
        roots, receipt = [], dict(near_identity_on_tested_grid=True, actual_starts=0,
            skip_reason='Near identity on fixed grid; no manufactured isolated roots')
    else:
        roots, receipt = bounded_flow(model, arrays, device, budget_check=check)
    receipt.update(application='XY dimension-only control', near_identity_on_tested_grid=near_identity,
                   root_completeness_certified=False, BKT_certified=False,
                   fixed_line_certified=False, thermal_nu=None,
                   interpretation='PCA/hull rank may reflect curvature or noise; roots do not certify BKT')
    return dict(roots=roots, flow_analysis=receipt, thermal_nu=None)


def run(config_path, output):
    c = validate_config(config_path)
    out = Path(output).resolve(); out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); step = 0
    job = dict(status='running', workflow_complete=False, current='admission', pid=os.getpid(),
               returncode=None, error=None, requires_user_action=False)

    def status(**kw):
        job.update(kw, actual_steps=step, elapsed_seconds=time.monotonic()-started)
        base.write_json(out/'status.json', job); base.write_json(out/'job.json', job)
        print(json.dumps(job, ensure_ascii=False), flush=True)

    def check():
        if time.monotonic()-started > c['hard_seconds']-30:
            raise base.BudgetHold('Whole-pipeline deadline; no automatic shortening or restart')
        if torch.cuda.is_initialized() and max(torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()) > c['max_cuda_bytes']:
            raise base.BudgetHold('Own GPU memory exceeded 8GiB')
        if sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) > c['max_output_bytes']:
            raise base.BudgetHold('Output budget exceeded 2GiB')

    try:
        if os.uname().nodename != 'thinklab-105-225' or os.getpriority(os.PRIO_PROCESS, 0) < 10 or os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID:
            raise RuntimeError('Wrong host/priority/device mask')
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        from scripts.run_ising_highdim import source_hashes
        code = Path(__file__).resolve().parents[1]; pins = source_hashes(code)
        config_sha = base.sha256_file(config_path)
        base.write_json(out/'config.json', c)
        base.write_json(out/'plan.json', dict(config=c, source_sha256=pins, config_sha256=config_sha))
        status(current='gpu_admission')
        admission = [resource_snapshot()]; time.sleep(10); admission.append(resource_snapshot())
        status(current='loading_pinned_existing_data')
        data, meta, data_pins = load_data()
        (out/'data').mkdir(); base.write_json(out/'data/metadata.json', meta)
        reference_manifest = json.loads((SOURCE/'manifest.json').read_text())
        reference_summary = json.loads((SOURCE/'summary.json').read_text())
        reference_report = json.loads((SOURCE/'report/report.json').read_text())
        base.write_json(out/'reference_scalar.json', dict(source=str(SOURCE),
            manifest_sha256=MANIFEST_SHA, summary_sha256=SUMMARY_SHA, summary=reference_summary,
            temperature_g_spearman=reference_report['temperature_g_spearman'], comparison_step=1250,
            comparison_caveat='Latent coordinates differ; raw latent RMSE is not a common physical error'))
        device = torch.device('cuda:0'); receipt = base._device_receipt(device, GPU_UUID)
        torch.cuda.set_per_process_memory_fraction(c['max_cuda_bytes']/torch.cuda.get_device_properties(device).total_memory)
        torch.cuda.reset_peak_memory_stats()
        status(current='disposable_training_timing_pilot')
        pilot = XYQCR(seed=191, dim=2).to(device); opts = original.optimizers(pilot)
        rng = torch.Generator().manual_seed(292); fg = torch.Generator(device=device).manual_seed(393)
        timings = []
        for _ in range(10):
            torch.cuda.synchronize(); tick = time.monotonic()
            independent_step(pilot, original.draw(data, rng, device), opts, fg)
            torch.cuda.synchronize(); timings.append(time.monotonic()-tick); check()
        projected = time.monotonic()-started + max(timings[2:])*1.35*c['steps'] + 300
        base.write_json(out/'timing_plan.json', dict(pilot_discarded=True, pilot_updates=10,
            step_seconds=timings, projected_total_seconds=projected, fixed_steps=c['steps']))
        if projected > c['hard_seconds']-60:
            raise base.BudgetHold('Fixed 1250 updates and report do not fit budget')
        del opts, pilot; torch.cuda.empty_cache()
        model = XYQCR(seed=c['seed'], dim=2).to(device)
        rng = torch.Generator().manual_seed(c['batch_rng_seed'])
        fg = torch.Generator(device=device).manual_seed(c['field_rng_seed'])
        with torch.no_grad():
            x = data.configurations[data.train_indices[:16], :32].to(device=device, dtype=torch.double)
            levels = model.coarse_grain(x, torch.Generator(device=device).manual_seed(9091))
            target = torch.stack([model.features(v) for v in levels]).mean((0, 1))
            g = torch.stack([model.Q(v) for v in levels]).mean((0, 1))
            model.Q.decoder.bias.copy_(target-model.Q.decoder.weight@g)
        initial = base.module_fingerprints(model)
        if initial['C'] != reference_manifest['initial_sha256']['C']:
            raise RuntimeError('Initial C differs from scalar control')
        manifest = dict(recipe=RECIPE, config=c, source_sha256=pins, config_sha256=config_sha,
            data_sha256=DATA_SHA, data_metadata_sha256=METADATA_SHA,
            matched_scalar_source=str(SOURCE), matched_scalar_step=1250, model_change='theory_dim 1 -> 2',
            labels_used_in_training=False, pretraining=False, initial_sha256=initial,
            gradient_routing='simultaneous independent Q/C/R; Q cannot update C; R cannot update Q/C',
            information_estimator=c['information_estimator'], decoder_column_constraint=c['decoder_column_constraint'],
            initialization='Same seed and construction order; C identical; dimensional heads/decoder/R freshly initialized',
            batch_and_C_rng_seeds=[192, 293], admission=admission, device=receipt,
            sampling_warnings=meta['warnings'], equilibrium_certified=False, thermal_nu=None, BKT_certified=False)
        base.write_json(out/'manifest.json', manifest)
        opts = original.optimizers(model); history = []; probes = [dict(step=0, **original.probe(model, data, device))]
        base.write_json(out/'train_probes.json', probes); status(current='training', planned_steps=c['steps'])
        for step in range(1, c['steps']+1):
            check(); metrics = independent_step(model, original.draw(data, rng, device), opts, fg)
            if step == 1 or step % 50 == 0:
                history.append(dict(step=step, elapsed_seconds=time.monotonic()-started, **metrics))
                base.write_json(out/'history.json', history)
                status(last_losses={n:metrics[n] for n in ('q_total', 'c_total', 'r_total')})
            if step % 250 == 0:
                probes.append(dict(step=step, **original.probe(model, data, device)))
                base.write_json(out/'train_probes.json', probes)
                torch.save(dict(recipe=RECIPE, step=step, model=model.state_dict(), manifest=manifest,
                    optimizers={k:v.state_dict() for k,v in opts.items()}, batch_rng_state=rng.get_state(),
                    field_rng_state=fg.get_state()), out/f'training_step_{step}.pt')
        clocks = base.optimizer_step_counts(opts, c['steps'])
        for p in model.parameters(): p.grad = None
        frozen = base.state_fingerprint(model); status(current='frozen_evaluation')
        arrays = base.evaluate_model(model, data, device, budget_check=check)
        status(current='bounded_two_dimensional_flow')
        roots = flow_diagnostics(model, arrays, device, check)
        with torch.no_grad():
            # Match the scalar trial exactly, including its diagnostic RNG.
            x = data.configurations[data.train_indices[:16], :64].to(device=device, dtype=torch.double)
            b, e = sample_regions(x, 128, torch.Generator(device=device).manual_seed(91919))
            means = model.C.mean_blocks(b)[..., 0]
            mi = dict(paired=float(circular_information(means, e)),
                within_ensemble_mismatch=float(circular_information(means, e, shuffle=True)),
                estimator=c['information_estimator'])
        if frozen != base.state_fingerprint(model) or any(p.grad is not None for p in model.parameters()):
            raise RuntimeError('Frozen evaluation changed model or parameter gradients')
        np.savez(out/'evaluation_arrays.npz', **arrays); base.write_json(out/'roots.json', roots)
        final = base.module_fingerprints(model)
        base.write_json(out/'summary.json', dict(status='complete', actual_steps=step, optimizer_steps=clocks,
            closure=base.closure_metrics(arrays), information_diagnostic=mi, final_module_sha256=final,
            C_final_exactly_matches_scalar=final['C'] == reference_summary['final_module_sha256']['C'],
            peak_cuda_bytes=torch.cuda.max_memory_allocated(), labels_used_in_training=False,
            thermal_nu=None, BKT_certified=False, equilibrium_certified=False,
            elapsed_seconds=time.monotonic()-started))
        status(current='reporting')
        from scripts.report_xy_qcr_g2 import render
        render(out); check()
        if source_hashes(code) != pins or base.sha256_file(config_path) != config_sha:
            raise RuntimeError('Frozen code/config changed')
        for name, sha in data_pins.items():
            if base.sha256_file(SOURCE/name) != sha: raise RuntimeError('Source changed during run')
        public = [dict(path=str(p.relative_to(out)), sha256=base.sha256_file(p), bytes=p.stat().st_size)
            for p in sorted(out.rglob('*')) if p.is_file() and p.suffix in ('.json', '.npz', '.png')
            and p.name not in ('status.json', 'job.json')]
        status(status='complete', current='complete', workflow_complete=True, returncode=0, public_files=public)
    except Exception as exc:
        status(status='budget_hold' if isinstance(exc, base.BudgetHold) else 'failed',
            returncode=3 if isinstance(exc, base.BudgetHold) else 1, error=str(exc),
            requires_user_action=True, traceback=traceback.format_exc())
        raise
