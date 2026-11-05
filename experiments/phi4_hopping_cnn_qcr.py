"""Time-bounded, fresh b3 CNN-Q/C/R trial on the existing hopping dataset.

Timing pilot is disposable. Its throughput alone determines a fixed formal
update count before fresh initialization; no loss/root/TEST-based selection.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import time
import traceback

import numpy as np
import torch

from experiments import phi4_hopping_qcr as base
from recurrent_rg.data.phi4_hopping_loader import load_dataset
from recurrent_rg.phi4_cnn_qcr import CNNPhi4QCR

RECIPE = "phi4_hopping_b3_configuration_cnn_affine_qcr_v1"
REFINED_RECIPE = "phi4_hopping_b3_configuration_cnn_refinedR_v1"
DATA = Path("/mnt/nas-new/home/yange/ham/results/phi4_hopping_b3_g1_kappa032_104_seed91_gpu3_vm225_20261031/data")
DATA_SHA = "f624bbffbb077df7986b971ba74434270945c78d345513807b5f7746cd88ed95"
GPU_INDEX = 3
GPU_UUID = "GPU-a167decd-ce49-6c8f-6bcf-136fde596683"
HARD_SECONDS = 1200.
TARGET_SECONDS = 900.
MAX_CUDA_BYTES = 8 * 1024**3


def refined_config():
    return dict(schema_version=5, recipe=REFINED_RECIPE, data_root=str(DATA),
                data_manifest_sha256=DATA_SHA, seed=91, theory_dim=1,
                lattice_sizes=[36, 12, 4], block_size=3, stride=3,
                system_batch=32, sample_batch=64, information_pairs=128,
                learning_rate=.001, r_updates_per_batch=10,
                target_seconds=TARGET_SECONDS, hard_seconds=HARD_SECONDS,
                loss=base.LOSS_POLICY)


def validate_refined_config(path):
    payload = json.loads(Path(path).read_text())
    if payload != refined_config() or type(payload.get("r_updates_per_batch")) is not int:
        raise ValueError("The explicit refined CNN trial requires its exact frozen preset")
    return payload


def resource_snapshot():
    def command(*args):
        return subprocess.check_output(args, text=True, timeout=10).strip()
    gpu = command("nvidia-smi", "-i", str(GPU_INDEX),
                  "--query-gpu=uuid,memory.free", "--format=csv,noheader,nounits").split(",")
    processes = command("nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader")
    mem = int(next(x.split()[1] for x in Path("/proc/meminfo").read_text().splitlines()
                   if x.startswith("MemAvailable:"))) * 1024
    io = float(next(x.split("avg10=")[1].split()[0] for x in Path("/proc/pressure/io").read_text().splitlines()
                    if x.startswith("full ")))
    receipt = dict(gpu_uuid=gpu[0].strip(), free_gpu_mib=int(gpu[1]),
                   compute_process_present=GPU_UUID in processes, mem_available_bytes=mem,
                   load1=os.getloadavg()[0], cpu_count=os.cpu_count(), io_full_avg10=io)
    if (receipt["gpu_uuid"] != GPU_UUID or receipt["compute_process_present"]
            or receipt["free_gpu_mib"] < 12288 or mem < 36*1024**3
            or receipt["load1"] > 1.25*receipt["cpu_count"] or io >= 20.):
        raise base.BudgetHold("Resource admission not met; no GPU migration or shared launch")
    return receipt


def choose_steps(elapsed, step_seconds, evaluation_reserve, target=TARGET_SECONDS):
    # Fixed endpoint chosen using throughput ONLY, never scientific metrics.
    count = min(3000, int((target-elapsed-evaluation_reserve) / step_seconds // 50)*50)
    if count < 200:
        raise base.BudgetHold("Less than 200 full-batch updates fit the declared time budget")
    return count


def optimizers(model):
    return {name: torch.optim.Adam(getattr(model, name).parameters(), lr=.001)
            for name in ("Q", "C", "R")}


def draw_batch(dataset, rng, device):
    train = torch.as_tensor(dataset.train_indices)
    indices = train[torch.randperm(len(train), generator=rng)[:32]]
    samples = torch.randperm(dataset.configurations.shape[1], generator=rng)[:64]
    return dataset.configurations[indices][:, samples].to(device=device, dtype=torch.float64)


def root_diagnostics(model, arrays):
    from scipy.optimize import brentq
    net = copy.deepcopy(model.R).cpu().eval()
    low, high = arrays["flow_grid"][[0, -1], 0]
    grid = np.linspace(low, high, 4001)
    with torch.no_grad():
        displacement = net(torch.tensor(grid[:, None], dtype=torch.float64)).numpy()[:, 0]
    roots = []
    def function(x):
        with torch.no_grad():
            return float(net(torch.tensor([[x]], dtype=torch.float64))[0, 0])
    inputs = arrays["direct_g"][arrays["splits"] == "train", :-1, 0]
    near_identity = bool(np.max(np.abs(displacement)) < 1e-12)
    for a, b, fa, fb in zip(grid[:-1], grid[1:], displacement[:-1], displacement[1:]):
        if near_identity:
            break  # Do not manufacture thousands of isolated roots for identity.
        if fa == 0 or fa*fb < 0:
            value = float(a if fa == 0 else brentq(function, a, b, xtol=1e-13))
            if any(abs(value-r["g"]) < 1e-9 for r in roots):
                continue
            x = torch.tensor([[value]], dtype=torch.float64, requires_grad=True)
            derivative = float(1 + torch.autograd.grad(net(x).sum(), x)[0])
            roots.append(dict(g=value, full_R_derivative=derivative, residual_abs=abs(function(value)),
                              numerical_stability=("neutral" if abs(abs(derivative)-1)<1e-5
                                                   else "repeller" if abs(derivative)>1 else "attractor"),
                              inside_train_input_envelope=bool(inputs.min()<=value<=inputs.max()),
                              nearest_train_input_distance=float(np.min(np.abs(inputs-value)))))
    arrays["flow_grid"] = grid[:, None]
    arrays["flow_next"] = (grid+displacement)[:, None]
    return dict(roots=roots, search_interval=[float(low), float(high)], near_identity=near_identity,
                root_completeness_certified=False, physical_fixed_points_certified=False,
                thermal_nu=None, critical_kappa=None)


def run(output, *, r_updates_per_batch=1, config_path=None):
    if type(r_updates_per_batch) is not int or r_updates_per_batch not in (1, 10):
        raise ValueError("Only historical 1:1 or explicit current 1:10 trials are supported")
    recipe = REFINED_RECIPE if r_updates_per_batch == 10 else RECIPE
    if r_updates_per_batch == 10:
        validate_refined_config(config_path)
    start = time.monotonic()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Fresh output required")
    if (os.uname().nodename != "thinklab-105-225" or os.getpriority(os.PRIO_PROCESS, 0) < 10
            or os.environ.get("CUDA_VISIBLE_DEVICES") != GPU_UUID or torch.cuda.is_initialized()):
        raise RuntimeError("Wrong host/priority/GPU mask")
    output.mkdir(parents=True)
    step = 0
    job = dict(status="running", workflow_complete=False, current="admission", actual_steps=0,
               pid=os.getpid(), returncode=None, error=None, requires_user_action=False)
    def status(**kw):
        job.update(kw, actual_steps=step, elapsed_seconds=time.monotonic()-start)
        if r_updates_per_batch > 1:
            job.update(Q_updates=step, C_updates=step, R_updates=step*r_updates_per_batch)
        base.write_json(output/"status.json", job)
        print(json.dumps(job, ensure_ascii=False), flush=True)
    def check(reserve=0.):
        if time.monotonic()-start+reserve > HARD_SECONDS:
            raise base.BudgetHold("20-minute total runtime limit; no silent shortening/restart")
        if torch.cuda.is_initialized() and torch.cuda.max_memory_allocated() > MAX_CUDA_BYTES:
            raise base.BudgetHold("Own CUDA allocation exceeded 8 GiB")
    status()
    try:
        admission = [resource_snapshot()]
        time.sleep(10)
        admission.append(resource_snapshot())
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        if base.sha256_file(DATA/"manifest.json") != DATA_SHA:
            raise RuntimeError("Historical b3 data manifest differs")
        status(current="loading_existing_data")
        dataset = load_dataset(DATA)
        if dataset.configurations.shape != (60, 128, 36, 36, 1):
            raise ValueError("Expected historical 60 chains x128 fields, L36")
        device = torch.device("cuda:0")
        device_receipt = base._device_receipt(device, GPU_UUID)
        torch.cuda.set_per_process_memory_fraction(MAX_CUDA_BYTES/torch.cuda.get_device_properties(device).total_memory, device)
        torch.cuda.reset_peak_memory_stats()
        pins = base.source_pins()
        import recurrent_rg.phi4_cnn_qcr as cnn_module
        import scripts.report_phi4_hopping_cnn_qcr as report_module
        import recurrent_rg.r_refinement as refinement_module
        import recurrent_rg.config as config_module
        import recurrent_rg.cli as cli_module
        for path in (Path(__file__), Path(cnn_module.__file__), Path(report_module.__file__),
                     Path(refinement_module.__file__), Path(config_module.__file__), Path(cli_module.__file__)):
            pins[str(path.resolve())] = base.sha256_file(path)
        if config_path is not None:
            pins[str(Path(config_path).resolve())] = base.sha256_file(config_path)
        status(current="disposable_timing_pilot")
        pilot = CNNPhi4QCR(dataset.scale).to(device)
        initial_state = {k:v.detach().cpu().clone() for k,v in pilot.state_dict().items()}
        optimizer = optimizers(pilot)
        rng = torch.Generator().manual_seed(192)
        field_rng = torch.Generator(device=device).manual_seed(293)
        timings = []
        for i in range(10):
            torch.cuda.synchronize()
            tick = time.monotonic()
            base.independent_step(pilot, draw_batch(dataset, rng, device), optimizer,
                                  generator=field_rng, information_pairs=128,
                                  r_updates_per_batch=r_updates_per_batch)
            torch.cuda.synchronize()
            timings.append(time.monotonic()-tick)
            check()
        # Evaluation timing uses only two TRAIN chains; no TEST-based selection.
        ix = dataset.train_indices[:2]
        tiny = replace(dataset, configurations=dataset.configurations[ix], splits=dataset.splits[ix],
                       labels=dataset.labels[ix], cell_ids=dataset.cell_ids[ix])
        tick = time.monotonic()
        base.evaluate_model(pilot, tiny, device, budget_check=check)
        torch.cuda.synchronize()
        evaluation_seconds = time.monotonic()-tick
        reserve = max(90., evaluation_seconds*30*1.5+60.)
        conservative_step = max(timings[2:])*1.25
        steps = choose_steps(time.monotonic()-start, conservative_step, reserve)
        timing = dict(disposable_pilot_updates=10, pilot_R_updates=10*r_updates_per_batch,
                      r_updates_per_batch=r_updates_per_batch, step_seconds=timings,
                      conservative_step_seconds=conservative_step, train_only_evaluation_probe_seconds=evaluation_seconds,
                      finalization_reserve_seconds=reserve, selected_formal_steps=steps,
                      target_seconds=TARGET_SECONDS, hard_seconds=HARD_SECONDS,
                      endpoint_chosen_from="timing only; never loss, TEST or fixed points")
        base.write_json(output/"timing_plan.json", timing)
        del optimizer, pilot
        torch.cuda.empty_cache()
        model = CNNPhi4QCR(dataset.scale).to(device)
        if any(not torch.equal(initial_state[k], v.detach().cpu()) for k,v in model.state_dict().items()):
            raise RuntimeError("Fresh formal initialization differs from pre-pilot state")
        del initial_state
        initialization = base.initialize_q_decoder(model, dataset.configurations, dataset.train_indices,
                                                    seed=91, budget_check=check)
        manifest = dict(schema_version=5, recipe=recipe, data_root=str(DATA), data_manifest_sha256=DATA_SHA,
                        data_shards=dataset.receipts, data_warning_count=dataset.warning_count,
                        equilibrium_certified=False, options=dict(steps=steps, system_batch=32, sample_batch=64,
                        seed=91, information_pairs=128, r_updates_per_batch=r_updates_per_batch),
                        R_optimization=refinement_module.refinement_contract(SimpleNamespace(r_updates_per_batch=r_updates_per_batch)),
                        loss=base.LOSS_POLICY, lattice_sizes=[36,12,4],
                        block_size=3, stride=3, coarse_graining_depth=2, theory_dim=1,
                        architecture="per-field periodic CNN16/depthwise16, spatial mean+second moment, frame MLP32-32-16, ensemble mean+variance, head32-64-1",
                        decoder="shared affine1->625 with post-update unit column norm; NOT the earlier Q-only nonlinear decoder",
                        changed_vs_historical="Q encoder; fixed throughput-budgeted shorter training; implementation chunk sizes only",
                        same_as_historical="data/splits, random C/R, batch32x64, Adam.001, independent gradients, equal-target all-start, all loss weights",
                        c_adaptation="historical Gaussian C + full-shell smoothed information, variance floor, uniform-amplitude penalty; not binary histogram MI/NLL",
                        all_Q_C_R_fresh_trainable=True, pretraining=False, labels_used_in_training=False,
                        test_used_in_training=False, model_selection=False, dtype="float64", scale=dataset.scale,
                        q_initialization_receipt=initialization, source_sha256=pins, timing_plan=timing,
                        device=device_receipt, admission=admission, thermal_nu=None)
        if r_updates_per_batch > 1:
            manifest["changed_vs_previous_CNN_trial"] = "R10 updates per online batch instead of1; endpoint chosen by the same throughput budget; no new loss, Q/C updates, labels or post-training R-only stage"
        base.write_json(output/"manifest.json", manifest)
        base.write_json(output/"plan.json", manifest)
        optimizer = optimizers(model)
        rng = torch.Generator().manual_seed(192)
        field_rng = torch.Generator(device=device).manual_seed(293)
        history = []
        status(current="training", planned_steps=steps)
        training_start = time.monotonic()
        for step in range(1, steps+1):
            check(reserve)
            metrics = base.independent_step(model, draw_batch(dataset, rng, device), optimizer,
                                           generator=field_rng, information_pairs=128,
                                           r_updates_per_batch=r_updates_per_batch)
            if step == 1 or step % 25 == 0 or step == steps:
                history.append(dict(step=step, elapsed_seconds=time.monotonic()-training_start, **metrics))
                base.write_json(output/"history.json", history)
                status(current="training", planned_steps=steps, last_losses={k:metrics[k] for k in ("q_total","c_total","r_total")})
            if step % 100 == 0 or step == steps:
                torch.save(dict(recipe=recipe, step=step, model=model.state_dict(),
                                optimizers={k:v.state_dict() for k,v in optimizer.items()}, manifest=manifest,
                                batch_rng=rng.get_state(), field_rng=field_rng.get_state()), output/"latest.pt")
        counts = base.optimizer_step_counts(optimizer, steps, r_updates_per_batch=r_updates_per_batch)
        for p in model.parameters():
            p.grad = None
        frozen = base.state_fingerprint(model)
        status(current="evaluation")
        arrays = base.evaluate_model(model, dataset, device, budget_check=check)
        roots = root_diagnostics(model, arrays)
        if frozen != base.state_fingerprint(model) or any(p.grad is not None for p in model.parameters()):
            raise RuntimeError("Evaluation changed model/gradients")
        base.verify_dataset_unchanged(DATA, dataset)
        np.savez(output/"evaluation_arrays.npz", **arrays)
        base.write_json(output/"roots.json", roots)
        summary = dict(status="complete", recipe=recipe, actual_steps=steps, optimizer_steps=counts,
                       r_updates_per_batch=r_updates_per_batch,
                       closure=base.closure_metrics(arrays), final_state_sha256=frozen,
                       final_module_sha256=base.module_fingerprints(model), labels_used_in_training=False,
                       train_chains=len(dataset.train_indices), test_chains=len(dataset.test_indices),
                       lattice_sizes=[36,12,4], data_warning_count=dataset.warning_count,
                       peak_cuda_bytes=torch.cuda.max_memory_allocated(), thermal_nu=None,
                       equilibrium_certified=False, physical_fixed_points_certified=False,
                       timing_before_report_seconds=time.monotonic()-start)
        base.write_json(output/"summary.json", summary)
        status(current="reporting")
        report_module.render(output)
        check()
        for path, digest in pins.items():
            if base.sha256_file(path) != digest:
                raise RuntimeError("Frozen source changed during execution")
        public = [dict(path=str(p.relative_to(output)), sha256=base.sha256_file(p), bytes=p.stat().st_size)
                  for p in sorted(output.rglob("*")) if p.is_file() and p.suffix in (".json",".npz",".png")
                  and p.name != "status.json"]
        status(status="complete", current="complete", workflow_complete=True, returncode=0,
               public_files=public, cpu_seconds=time.process_time())
    except Exception as exc:
        status(status="budget_hold" if isinstance(exc, base.BudgetHold) else "failed", returncode=3 if isinstance(exc, base.BudgetHold) else 1,
               error=str(exc), traceback=traceback.format_exc(), requires_user_action=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args(argv)
    if args.config is not None:
        validate_refined_config(args.config)
    if not args.execute:
        print(json.dumps(dict(recipe=REFINED_RECIPE if args.config else RECIPE, data=str(DATA), budget_seconds=HARD_SECONDS,
                              target_seconds=TARGET_SECONDS, steps="200..3000 from disposable timing only",
                              output=args.output), indent=2))
        return
    run(args.output, r_updates_per_batch=10 if args.config else 1, config_path=args.config)


if __name__ == "__main__":
    main()
