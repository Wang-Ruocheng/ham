"""Explicit current local-Q/free-gain recipe on the pinned scalar phi4 data.

Fresh, unsupervised, one update per Q/C/R batch; no R-only polishing. The
continuous C adapter is preserved. Package defaults and old runs are untouched.
"""
from dataclasses import replace
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from experiments import phi4_hopping_qcr as base
from experiments.phi4_hopping_cnn_qcr import (
    DATA, DATA_SHA, GPU_UUID, resource_snapshot, draw_batch, root_diagnostics,
)
from recurrent_rg.data.phi4_hopping_loader import load_dataset
from recurrent_rg.phi4_aligned_qcr import AlignedPhi4QCR, RawPatchTheoryEncoder

RECIPE = 'phi4_hopping_b3_local_Q_free_gain_unsupervised1250_seed91_v1'
STEPS = 1250
HARD_SECONDS = 1800


def preset():
    return dict(schema_version=5, recipe=RECIPE, data_root=str(DATA),
        data_manifest_sha256=DATA_SHA, seed=91, theory_dim=1,
        lattice_sizes=[36, 12, 4], block_size=3, stride=3,
        q_encoder='local_motif', decoder_column_constraint='free_after_initialization',
        steps=STEPS, system_batch=32, sample_batch=64, information_pairs=128,
        patch_chunk_size=131072, learning_rates=dict(Q=.001, C=.001, R=.003),
        r_updates_per_batch=1, supervision_weight=0., loss=dict(base.LOSS_POLICY),
        hard_seconds=HARD_SECONDS, max_cuda_bytes=8*1024**3, max_output_bytes=2*1024**3)


def validate_config(path):
    value = json.loads(Path(path).read_text())
    if value != preset():
        raise ValueError('Require this exact explicit fresh unsupervised preset')
    return value


class FreeGainLocalQ(RawPatchTheoryEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.free_after_initialization = True

    @torch.no_grad()
    def normalize_decoder_(self):
        if not getattr(self, 'free_after_initialization', False):
            super().normalize_decoder_()


class CurrentPhi4QCR(AlignedPhi4QCR):
    def __init__(self, scale, seed=91, patch_chunk_size=131072):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            super().__init__(scale, patch_chunk_size=patch_chunk_size)
            old_q = self.Q.state_dict()
            self.Q = FreeGainLocalQ(scale, patch_chunk_size=patch_chunk_size)
            self.Q.load_state_dict(old_q)


def optimizers(model):
    return {name: torch.optim.Adam(getattr(model, name).parameters(), lr=lr)
            for name, lr in preset()['learning_rates'].items()}


@torch.no_grad()
def train_probe(model, dataset, device):
    from recurrent_rg.losses import all_start_anchor_closure_loss
    ids = dataset.train_indices[:16]
    generator = torch.Generator(device=device).manual_seed(91303)
    levels = model.coarse_grain(dataset.configurations[ids, :64].to(device), generator)
    gs = torch.stack([.5*(model.Q(x[:, :32])+model.Q(x[:, 32:])) for x in levels], 1)
    loss, _, _ = all_start_anchor_closure_loss(model, gs, coordinate_reduction='sum')
    return dict(weighted_closure=float(10*loss), g_mean=gs.mean(0).cpu().tolist(),
                g_std=gs.std(0, unbiased=False).cpu().tolist(),
                decoder_column_norms=model.Q.decoder.weight.norm(dim=0).cpu().tolist())


def fixed_budget(elapsed, durations, probe_seconds, evaluation_seconds):
    if len(durations) != 10 or not all(np.isfinite(t) and t > 0 for t in durations):
        raise ValueError('Require ten complete timing observations')
    steady = max(durations[2:])*1.35
    reserve = max(150., evaluation_seconds*30*1.5+90.)
    projected = elapsed + steady*STEPS + 7*probe_seconds*1.5 + reserve
    if projected > HARD_SECONDS-90:
        raise base.BudgetHold('Fixed1250 does not fit 30-minute cap; no shortening or restart')
    return dict(formal_steps=STEPS, steady_conservative_seconds=steady,
                finalization_reserve_seconds=reserve, projected_total_seconds=projected,
                step_seconds=durations, pilot_updates=10, pilot_discarded=True,
                endpoint_policy='Fixed1250; no loss/TEST/root selection')


def run(config_path, output):
    config = validate_config(config_path)
    if (os.uname().nodename != 'thinklab-105-225' or os.getpriority(os.PRIO_PROCESS, 0) < 10
            or os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID or torch.cuda.is_initialized()):
        raise RuntimeError('Wrong host/priority/GPU mask')
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    actual = latest = 0
    job = dict(recipe=RECIPE, status='running', workflow_complete=False, pid=os.getpid(),
               returncode=None, error=None, requires_user_action=False)
    def status(**kwargs):
        job.update(kwargs, actual_steps=actual, latest_checkpoint_step=latest,
                   Q_updates=actual, C_updates=actual, R_updates=actual,
                   elapsed_seconds=time.monotonic()-started)
        base.write_json(out/'status.json', job)
        base.write_json(out/'job.json', job)
        print(json.dumps(job, ensure_ascii=False, allow_nan=False), flush=True)
    def check():
        if time.monotonic()-started > HARD_SECONDS-30:
            raise base.BudgetHold('Complete pipeline time cap; no automatic continuation')
        if torch.cuda.is_initialized() and max(torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()) > config['max_cuda_bytes']:
            raise base.BudgetHold('Own GPU memory cap exceeded')
        if sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) > config['max_output_bytes']:
            raise base.BudgetHold('Output size cap exceeded')
    status(current='admission')
    try:
        admission = [resource_snapshot()]
        time.sleep(10)
        admission.append(resource_snapshot())
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        if base.sha256_file(DATA/'manifest.json') != DATA_SHA:
            raise ValueError('Pinned historical data manifest differs')
        status(current='loading_existing_data')
        dataset = load_dataset(DATA)
        if (dataset.configurations.shape != (60, 128, 36, 36, 1)
                or len(dataset.train_indices) != 40 or len(dataset.test_indices) != 20):
            raise ValueError('Expected historical data geometry and whole-cell split')
        device = torch.device('cuda:0')
        receipt = base._device_receipt(device, GPU_UUID)
        torch.cuda.set_per_process_memory_fraction(config['max_cuda_bytes']/torch.cuda.get_device_properties(device).total_memory)
        torch.cuda.reset_peak_memory_stats()
        from scripts.run_ising_highdim import source_hashes
        from scripts import report_phi4_hopping_local_current as report
        code = Path(__file__).resolve().parents[1]
        pins = source_hashes(code)
        config_sha = base.sha256_file(config_path)
        base.write_json(out/'plan.json', dict(config=config, config_sha256=config_sha,
            source_sha256=pins, admission=admission, data_warning_count=dataset.warning_count,
            labels_used_in_training=False, physical_sampling_updates=0))
        status(current='disposable_TRAIN_timing_pilot')
        pilot = CurrentPhi4QCR(dataset.scale, seed=191).to(device)
        opts = optimizers(pilot)
        rng = torch.Generator().manual_seed(292)
        field_rng = torch.Generator(device=device).manual_seed(393)
        timings = []
        for _ in range(10):
            torch.cuda.synchronize(); tick = time.monotonic()
            base.independent_step(pilot, draw_batch(dataset, rng, device), opts,
                                  generator=field_rng, information_pairs=128)
            torch.cuda.synchronize(); timings.append(time.monotonic()-tick); check()
        tick = time.monotonic(); train_probe(pilot, dataset, device)
        torch.cuda.synchronize(); probe_seconds = time.monotonic()-tick
        ids = dataset.train_indices[:2]
        tiny = replace(dataset, configurations=dataset.configurations[ids], labels=dataset.labels[ids],
                       splits=dataset.splits[ids], cell_ids=dataset.cell_ids[ids])
        tick = time.monotonic(); base.evaluate_model(pilot, tiny, device, budget_check=check)
        torch.cuda.synchronize(); evaluation_seconds = time.monotonic()-tick
        timing = fixed_budget(time.monotonic()-started, timings, probe_seconds, evaluation_seconds)
        base.write_json(out/'timing_plan.json', timing)
        del opts, pilot
        torch.cuda.empty_cache()
        model = CurrentPhi4QCR(dataset.scale).to(device)
        initialization = base.initialize_q_decoder(model, dataset.configurations,
            dataset.train_indices, seed=91, budget_check=check)
        manifest = dict(recipe=RECIPE, config=config, source_sha256=pins, config_sha256=config_sha,
            data_root=str(DATA), data_manifest_sha256=DATA_SHA, data_shards=dataset.receipts,
            data_warning_count=dataset.warning_count, scale=dataset.scale,
            architecture='Local patch9-96-12-12; ensemble mean; head12-96-1; affine1-625; residual R1-96-96-1',
            decoder='unit column at initialization, free gain thereafter; explicit current-recipe ablation',
            C_adaptation='Historical Gaussian C, full-shell smoothed information, variance floor, uniform-amplitude MSE; not binary histogram MI/NLL',
            loss=dict(base.LOSS_POLICY), learning_rates=config['learning_rates'],
            gradient_routing='simultaneous independent Q/C/R gradients; detached fields for Q, detached targets for R',
            all_Q_C_R_fresh_trainable=True, pretraining=False, labels_used_in_training=False,
            test_used_in_training=False, physical_sampling_updates=0, R_only_updates=0,
            initialization=initialization, options=dict(steps=STEPS, r_updates_per_batch=1,
                system_batch=32, sample_batch=64, seed=91, information_pairs=128),
            lattice_sizes=[36,12,4], theory_dim=1, dtype='float64',
            equilibrium_certified=False, thermal_nu=None, admission=admission, device=receipt,
            timing_plan=timing, historical_comparison='Same phi4 data/split/C/loss; Q architecture, decoder gain, R lr/update ratio and length differ from prior CNN run')
        base.write_json(out/'manifest.json', manifest)
        opts = optimizers(model)
        rng = torch.Generator().manual_seed(192)
        field_rng = torch.Generator(device=device).manual_seed(293)
        history = []
        probes = [dict(step=0, **train_probe(model, dataset, device))]
        base.write_json(out/'train_probes.json', probes)
        status(current='training', planned_steps=STEPS)
        for actual in range(1, STEPS+1):
            check()
            metrics = base.independent_step(model, draw_batch(dataset, rng, device), opts,
                                           generator=field_rng, information_pairs=128)
            if actual % 250 == 0:
                probes.append(dict(step=actual, **train_probe(model, dataset, device)))
                base.write_json(out/'train_probes.json', probes)
                torch.save(dict(recipe=RECIPE, step=actual, model=model.state_dict(), manifest=manifest,
                    optimizers={k:v.state_dict() for k,v in opts.items()},
                    batch_rng=rng.get_state(), field_rng=field_rng.get_state()), out/f'training_step_{actual}.pt')
                latest = actual
            if actual == 1 or actual % 50 == 0:
                history.append(dict(step=actual, elapsed_seconds=time.monotonic()-started, **metrics))
                base.write_json(out/'history.json', history)
                status(last_losses={k:metrics[k] for k in ('q_total','c_total','r_total')})
        clocks = base.optimizer_step_counts(opts, STEPS)
        for p in model.parameters(): p.grad = None
        frozen = base.state_fingerprint(model)
        status(current='frozen_evaluation')
        arrays = base.evaluate_model(model, dataset, device, budget_check=check)
        roots = root_diagnostics(model, arrays)
        if frozen != base.state_fingerprint(model) or any(p.grad is not None for p in model.parameters()):
            raise RuntimeError('Evaluation changed model or gradients')
        np.savez(out/'evaluation_arrays.npz', **arrays)
        base.write_json(out/'roots.json', roots)
        base.write_json(out/'summary.json', dict(status='complete', recipe=RECIPE,
            actual_steps=STEPS, optimizer_steps=clocks, closure=base.closure_metrics(arrays),
            final_state_sha256=frozen, final_module_sha256=base.module_fingerprints(model),
            decoder_column_norms=model.Q.decoder.weight.norm(dim=0).detach().cpu().tolist(),
            labels_used_in_training=False, train_chains=40, test_chains=20,
            peak_cuda_bytes=torch.cuda.max_memory_allocated(), data_warning_count=dataset.warning_count,
            thermal_nu=None, equilibrium_certified=False, elapsed_seconds=time.monotonic()-started))
        status(current='reporting')
        report.render(out)
        check()
        base.verify_dataset_unchanged(DATA, dataset)
        if source_hashes(code) != pins or base.sha256_file(config_path) != config_sha:
            raise RuntimeError('Frozen code/config changed')
        public = [dict(path=str(p.relative_to(out)), sha256=base.sha256_file(p), bytes=p.stat().st_size)
            for p in sorted(out.rglob('*')) if p.is_file() and p.suffix in ('.json','.png','.npz')
            and p.name not in ('status.json','job.json')]
        status(status='complete', current='complete', workflow_complete=True, returncode=0, public_files=public)
    except Exception as exc:
        status(status='budget_hold' if isinstance(exc,base.BudgetHold) else 'failed',
               error=str(exc), traceback=traceback.format_exc(), requires_user_action=True,
               returncode=3 if isinstance(exc,base.BudgetHold) else 1)
        raise
