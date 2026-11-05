"""Exact, independent-output continuation of the completed u=1 phi4 run.

This entry point restores all three Adam states and both explicit training RNG
streams.  It does not resample data, recalibrate Q, restart Adam, or alter loss.
The short profile restores the same source independently and is discarded.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from experiments import phi4_qcr as original
import recurrent_rg.phi4_qcr as model_module


RECIPE = original.RECIPE
SOURCE_ROOT = "/mnt/nas-new/home/yange/ham/results/phi4_u1_rg_reference_b3_g1_seed91_vm225_20261028"
SOURCE_STEPS = 3000
ADDITIONAL_STEPS = 3000
PROFILE_STEPS = 20
SOURCE_CHECKPOINT_SHA = "8dcccfe68cd3caebb613de153d531c2380d0f9e655011c01c3eccfc0dcd5650f"
SOURCE_DATA_SHA = "d554ac421c9a14ef0b94701ff34741bd39236938de17fbc14d60437f7228085b"
SOURCE_TRAINER_SHA = "1e71119dfd8041269121e5f3a0cbbe1ae6eeb7e7d4126eb06507e827ec28a2bf"
SOURCE_MODEL_SHA = "016dd47a77ee7aa149a1ed65a33e98e35447aeaed904e994030d4eac1c55ed1d"
HIDDEN = 96
LATTICE_SIZES = [36, 12, 4]
GPU_UUID = "GPU-713b07d0-118c-f939-c0a8-565390215e01"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def state_tree_fingerprint(value):
    """Stable optimizer/RNG digest; independent of Torch ZIP serialization."""
    digest = hashlib.sha256()

    def visit(item):
        if torch.is_tensor(item):
            digest.update(b"tensor")
            digest.update(original.tensor_fingerprint(item).encode())
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item, key=lambda key: (str(type(key)), str(key))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode())
            for member in item:
                visit(member)
        else:
            digest.update(type(item).__name__.encode())
            digest.update(repr(item).encode())

    visit(value)
    return digest.hexdigest()


def source_receipt(source):
    """Read-only source identity check, with no model inference or CUDA setup."""
    root = Path(source).resolve()
    if root != Path(SOURCE_ROOT).resolve():
        raise ValueError("Only the explicitly pinned completed u=1 source is authorized")
    run = root / "run"
    files = {name: original.sha256_file(root / name) for name in
             ("status.json", "job.json", "run/status.json", "run/manifest.json",
              "run/summary.json", "run/history_progress.json", "data/manifest.json")}
    parent, job, status = (read(root / name) for name in ("status.json", "job.json", "run/status.json"))
    if any(item.get("status") != "complete" or item.get("workflow_complete") is not True
           or item.get("error") is not None or item.get("requires_user_action", False)
           for item in (parent, job, status)):
        raise ValueError("Source workflow is not successfully complete")
    stages = job.get("stages", [])
    if not stages or any(row.get("status") != "complete" or row.get("returncode") != 0 for row in stages):
        raise ValueError("Source stages do not all have successful execution receipts")
    manifest, summary, history = (read(run / name) for name in
                                  ("manifest.json", "summary.json", "history_progress.json"))
    if (status.get("returncode") != 0 or status.get("summary_sha256") != files["run/summary.json"]
            or status.get("actual_steps") != SOURCE_STEPS or summary.get("actual_steps") != SOURCE_STEPS
            or history.get("latest_step") != SOURCE_STEPS
            or history.get("latest_checkpoint_step") != SOURCE_STEPS
            or status.get("latest_checkpoint_step") != SOURCE_STEPS):
        raise ValueError("Source endpoint/summary/history receipt disagrees")
    expected = dict(schema_version=5, recipe=RECIPE, theory_dim=1, hidden=HIDDEN,
                    lattice_sizes=LATTICE_SIZES, block_size=3, stride=3, dtype="float64",
                    learning_rate=.001, closure_objective="all_start_anchor",
                    closure_scale_weights=None, labels_used_in_training=False)
    if any(manifest.get(name) != value for name, value in expected.items()):
        raise ValueError("Source architecture, schema or objective contract differs")
    options = manifest.get("options", {})
    expected_options = dict(steps=SOURCE_STEPS, seed=91, system_batch=8,
                            sample_batch=64, information_pairs=128, threads=1)
    if any(options.get(name) != value for name, value in expected_options.items()):
        raise ValueError("Source training/batch/seed contract differs")
    if (manifest.get("data_manifest_sha256") != SOURCE_DATA_SHA
            or summary.get("data_manifest_sha256") != SOURCE_DATA_SHA
            or files["data/manifest.json"] != SOURCE_DATA_SHA
            or Path(manifest.get("data_root", "")).resolve() != root / "data"):
        raise ValueError("Source data manifest SHA256/root differs")
    expected_sources = {SOURCE_TRAINER_SHA, SOURCE_MODEL_SHA}
    if (set(manifest.get("source_sha256", {}).values()) != expected_sources
            or original.sha256_file(original.__file__) != SOURCE_TRAINER_SHA
            or original.sha256_file(model_module.__file__) != SOURCE_MODEL_SHA):
        raise ValueError("Original trainer/model source SHA256 differs")
    for name in ("Q", "C", "R"):
        counts = summary.get("optimizer_steps", {}).get(name, {})
        if counts.get("min") != SOURCE_STEPS or counts.get("max") != SOURCE_STEPS or counts.get("parameters", 0) < 1:
            raise ValueError("Source optimizer parameter clocks differ")
    rows = history.get("history", [])
    if (not rows or rows[-1].get("step") != SOURCE_STEPS
            or any(a["step"] >= b["step"] for a, b in zip(rows, rows[1:]))):
        raise ValueError("Source training history is not an ordered complete prefix")
    checkpoint = run / f"training_step_{SOURCE_STEPS}.pt"
    checkpoint_sha = original.sha256_file(checkpoint)
    if checkpoint_sha != SOURCE_CHECKPOINT_SHA:
        raise ValueError("Source checkpoint SHA256 differs")
    return dict(source_root=str(root), source_run=str(run), data_root=str(root / "data"),
        checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha,
        data_manifest_sha256=SOURCE_DATA_SHA, source_steps=SOURCE_STEPS,
        final_state_sha256=summary["final_state_sha256"],
        final_module_sha256=summary["final_module_sha256"], public_files_sha256=files,
        original_trainer_sha256=SOURCE_TRAINER_SHA, original_model_sha256=SOURCE_MODEL_SHA)


def restore_training(checkpoint, manifest, dataset, device, *, expected_step):
    """Restore a checkpoint without mean-bias calibration or global RNG use.

    The historical trainer consumes only batch_rng and field_rng during updates.
    Global RNG states were not in its checkpoint; constructor consumption is
    isolated and restored here, and cannot affect the resumed trajectory.
    """
    if checkpoint.get("recipe") != RECIPE or checkpoint.get("step") != expected_step:
        raise ValueError("Checkpoint recipe/step differs")
    if checkpoint.get("manifest") != manifest:
        raise ValueError("Checkpoint manifest differs from verified public source")
    if dataset.manifest_sha256 != manifest["data_manifest_sha256"] or dataset.scale != manifest["scale"]:
        raise ValueError("Loaded data fingerprint or frozen scale differs")
    state = checkpoint["model"]
    if any(value.is_floating_point() and value.dtype != torch.float64 for value in state.values()):
        raise ValueError("Source state is not float64")
    # Phi4QCR constructs its random parameters on CPU, then .to(device) only
    # copies tensors.  Never query or initialize unrelated devices' CUDA RNGs.
    cpu_rng = torch.get_rng_state().clone()
    try:
        model = original.Phi4QCR(dataset.scale, hidden=manifest["hidden"], theory_dim=1).to(
            device=device, dtype=torch.float64)
    finally:
        torch.set_rng_state(cpu_rng)
    model.load_state_dict(state, strict=True)
    optimizers = {name: torch.optim.Adam(getattr(model, name).parameters(), lr=.001)
                  for name in ("Q", "C", "R")}
    if set(checkpoint.get("optimizers", {})) != set(optimizers):
        raise ValueError("Checkpoint must contain all three Adam states")
    for name, optimizer in optimizers.items():
        optimizer.load_state_dict(checkpoint["optimizers"][name])
        for group in optimizer.param_groups:
            if (group["lr"] != .001 or tuple(group["betas"]) != (.9, .999)
                    or group["eps"] != 1e-8 or group["weight_decay"] != 0
                    or group["amsgrad"] or group.get("maximize", False)):
                raise ValueError("Source Adam hyperparameters differ")
        if state_tree_fingerprint(optimizer.state_dict()) != state_tree_fingerprint(checkpoint["optimizers"][name]):
            raise ValueError("Restored Adam state differs from checkpoint")
    original.optimizer_step_counts(optimizers, expected_step)
    batch_rng = torch.Generator(device="cpu")
    field_rng = torch.Generator(device=device)
    batch_rng.set_state(checkpoint["batch_rng"].cpu())
    field_rng.set_state(checkpoint["field_rng"].cpu())
    for name, generator in (("batch_rng", batch_rng), ("field_rng", field_rng)):
        if not torch.equal(generator.get_state().cpu(), checkpoint[name].cpu()):
            raise ValueError("Restored training RNG differs")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise ValueError("Restored parameters unexpectedly carry gradients")
    return model, optimizers, batch_rng, field_rng


def training_update(model, dataset, optimizers, batch_rng, field_rng, device, options):
    train_indices = torch.as_tensor(dataset.train_indices, dtype=torch.long)
    systems = train_indices[torch.randperm(len(train_indices), generator=batch_rng)[:options["system_batch"]]]
    samples = torch.randperm(dataset.configurations.shape[1], generator=batch_rng)[:options["sample_batch"]]
    batch = dataset.configurations[systems][:, samples].to(device=device, dtype=torch.float64)
    return original.independent_step(model, batch, optimizers, generator=field_rng,
                                     information_pairs=options["information_pairs"])


def checkpoint_payload(model, optimizers, batch_rng, field_rng, step, manifest, receipt):
    return dict(recipe=RECIPE, step=step, model=model.state_dict(),
        optimizers={name: optimizer.state_dict() for name, optimizer in optimizers.items()},
        batch_rng=batch_rng.get_state(), field_rng=field_rng.get_state(), manifest=manifest,
        source_provenance=receipt)


@dataclass
class ContinueOptions:
    source: str
    output: str
    device: str = "cuda:2"
    expected_gpu_uuid: str | None = GPU_UUID
    additional_steps: int = ADDITIONAL_STEPS
    max_training_seconds: float = 3600.
    max_cuda_bytes: int = 8 * 1024 ** 3
    profile_only: bool = False
    profile_steps: int = PROFILE_STEPS

    def validate(self):
        if self.additional_steps != ADDITIONAL_STEPS or self.profile_steps != PROFILE_STEPS:
            raise ValueError("Pinned continuation is exactly 3000 additional updates; profile exactly 20")
        if (not math.isfinite(self.max_training_seconds) or self.max_training_seconds <= 0
                or isinstance(self.max_cuda_bytes, bool) or self.max_cuda_bytes <= 0):
            raise ValueError("Invalid compute budget")
        if self.device not in ("cpu", "cuda:2"):
            raise ValueError("No GPU migration is permitted")
        if self.device == "cuda:2" and self.expected_gpu_uuid != GPU_UUID:
            raise ValueError("Pinned CUDA physical UUID is required")
        if Path(self.output).exists():
            raise FileExistsError("Fresh independent output is required; source cannot be overwritten")
        source, output = Path(self.source).resolve(), Path(self.output).resolve()
        if output == source or source in output.parents or output in source.parents:
            raise ValueError("Output must be independent of the complete source root")


def run_continuation(options):
    options.validate()
    receipt = source_receipt(options.source)
    output = Path(options.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started, cpu_started = time.perf_counter(), time.process_time()
    step, latest_checkpoint = SOURCE_STEPS, SOURCE_STEPS
    write = original.write_json
    base_status = dict(recipe=RECIPE, source_steps=SOURCE_STEPS,
        requested_additional_steps=options.additional_steps, profile_only=options.profile_only)
    write(output / "status.json", dict(**base_status, status="initializing", workflow_complete=False,
        actual_steps=step, additional_steps=0, returncode=None, error=None, requires_user_action=False))
    try:
        torch.set_num_threads(1)
        dataset = original.load_dataset(receipt["data_root"])
        source_manifest = read(Path(receipt["source_run"]) / "manifest.json")
        source_history = read(Path(receipt["source_run"]) / "history_progress.json")["history"]
        device = torch.device(options.device)
        compute_started = time.perf_counter()
        device_receipt = original._device_receipt(device, options.expected_gpu_uuid)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        # Load only the explicitly pinned, trusted local source, never a remote pickle.
        checkpoint = torch.load(receipt["checkpoint"], map_location="cpu", weights_only=False)
        model, optimizers, batch_rng, field_rng = restore_training(
            checkpoint, source_manifest, dataset, device, expected_step=SOURCE_STEPS)
        if (original.state_fingerprint(model) != receipt["final_state_sha256"]
                or original.module_fingerprints(model) != receipt["final_module_sha256"]):
            raise ValueError("Restored model fingerprint differs from completed source")
        target_steps = SOURCE_STEPS + (options.profile_steps if options.profile_only else options.additional_steps)
        manifest = copy.deepcopy(source_manifest)
        manifest["options"].update(output=str(output), device=options.device, steps=target_steps,
            max_training_seconds=options.max_training_seconds, max_cuda_bytes=options.max_cuda_bytes,
            expected_gpu_uuid=options.expected_gpu_uuid)
        manifest.update(initial_state_sha256=receipt["final_state_sha256"],
            initialization="exact complete source checkpoint restoration; Q/C/R and all three Adam states retained",
            source_q_initialization=source_manifest.get("q_initialization"),
            source_q_initialization_receipt=source_manifest.get("q_initialization_receipt"),
            q_initialization="checkpoint_restore_no_calibration",
            q_initialization_receipt=dict(mode="checkpoint_restore_no_calibration",
                calibration_rerun=False, source_steps=SOURCE_STEPS,
                source_checkpoint_sha256=receipt["checkpoint_sha256"]),
            initial_state_sha256_semantics="restored complete source checkpoint; no initialization or fresh Adam",
            training_start_state_sha256=receipt["final_state_sha256"],
            training_start_module_sha256=receipt["final_module_sha256"],
            optimizer="three independent resumed Adam; all moments and parameter counters restored",
            continuation_options=asdict(options), source_provenance=receipt,
            source_steps=SOURCE_STEPS, additional_steps=target_steps-SOURCE_STEPS,
            profile_only=options.profile_only, scientific_result=not options.profile_only,
            q_reinitialized=False, data_resampled=False, loss_changed=False,
            training_rng_restore="exact batch_rng and field_rng; historical updates do not consume global RNG",
            global_rng_restore="CPU constructor consumption isolated; no global CUDA RNG consumed; historical checkpoint has no global RNG",
            resumed_optimizer_state_sha256={name: state_tree_fingerprint(value.state_dict()) for name, value in optimizers.items()},
            resumed_rng_sha256={"batch_rng": original.tensor_fingerprint(batch_rng.get_state()),
                                "field_rng": original.tensor_fingerprint(field_rng.get_state())},
            continuation_source_sha256=original.sha256_file(__file__), device=device_receipt,
            max_training_seconds_semantics="device/model restoration + resumed optimizer updates + frozen final evaluation")
        write(output / "manifest.json", manifest)
        history = copy.deepcopy(source_history)
        elapsed_prefix = float(history[-1].get("elapsed_seconds", 0.))
        training_started = time.perf_counter()
        initialization_seconds = training_started-compute_started
        model.train()
        for update in range(SOURCE_STEPS+1, target_steps+1):
            original._check_budget(options, device, compute_started)
            metrics = training_update(model, dataset, optimizers, batch_rng, field_rng,
                                      device, source_manifest["options"])
            step = update
            peak = original._check_budget(options, device, compute_started)
            extra = step-SOURCE_STEPS
            logged = extra == 1 or step % source_manifest["options"]["log_every"] == 0 or step == target_steps
            if logged:
                elapsed = time.perf_counter()-training_started
                history.append(dict(step=step, additional_step=extra, **metrics,
                    elapsed_seconds=elapsed_prefix+elapsed, continuation_elapsed_seconds=elapsed))
                print(json.dumps(dict(event="phi4_qcr_continue_progress", **history[-1]), allow_nan=False), flush=True)
            if not options.profile_only and (step % 1000 == 0 or step == target_steps):
                torch.save(checkpoint_payload(model, optimizers, batch_rng, field_rng,
                    step, manifest, receipt), output / f"training_step_{step}.pt")
                latest_checkpoint = step
            if logged:
                write(output / "history_progress.json", dict(history=history, latest_step=step,
                    latest_checkpoint_step=latest_checkpoint, source_steps=SOURCE_STEPS,
                    additional_steps=extra, profile_only=options.profile_only, recipe=RECIPE))
                write(output / "status.json", dict(**base_status, status="training", workflow_complete=False,
                    actual_steps=step, additional_steps=extra, latest_checkpoint_step=latest_checkpoint,
                    peak_cuda_bytes=peak, returncode=None, error=None, requires_user_action=False))
        original._sync(device)
        training_seconds = time.perf_counter()-training_started
        optimizer_steps = original.optimizer_step_counts(optimizers, target_steps)
        for parameter in model.parameters():
            parameter.grad = None
        final_sha = original.state_fingerprint(model)
        evaluation_seconds = 0.
        summary = dict(status="complete", recipe=RECIPE, actual_steps=step,
            source_steps=SOURCE_STEPS, additional_steps=step-SOURCE_STEPS,
            requested_additional_steps=options.additional_steps, profile_only=options.profile_only,
            scientific_result=not options.profile_only, optimizer_steps=optimizer_steps,
            latest_checkpoint_step=latest_checkpoint, source_provenance=receipt,
            initial_state_sha256=receipt["final_state_sha256"], final_state_sha256=final_sha,
            final_module_sha256=original.module_fingerprints(model), labels_used_in_training=False,
            data_manifest_sha256=dataset.manifest_sha256, theory_dim=1,
            lattice_sizes=manifest["lattice_sizes"], scale=dataset.scale, device=device_receipt,
            thermal_nu=None, equilibrium_certified=False, data_warning_count=dataset.warning_count,
            sampler_schema=dataset.sampler_schema, sample_semantics=dataset.sample_semantics,
            sampler_qualification=dataset.qualification, q_reinitialized=False,
            fresh_adam=False, data_resampled=False, loss_changed=False)
        if not options.profile_only:
            evaluation_started = time.perf_counter()
            write(output / "status.json", dict(**base_status, status="evaluating", workflow_complete=False,
                actual_steps=step, additional_steps=step-SOURCE_STEPS, latest_checkpoint_step=latest_checkpoint,
                returncode=None, error=None, requires_user_action=False))
            arrays = original.evaluate_model(model, dataset, device,
                budget_check=lambda: original._check_budget(options, device, compute_started))
            original.cell_grid_evaluation(model, arrays, device)
            roots = original.scalar_flow(model, arrays, device)
            evaluation_seconds = time.perf_counter()-evaluation_started
            if final_sha != original.state_fingerprint(model) or any(p.grad is not None for p in model.parameters()):
                raise RuntimeError("Model/gradient changed during frozen evaluation")
            np.savez(output / "evaluation_arrays.npz", **arrays)
            summary.update(frozen_evaluation_verified=True,
                train_chains=int(len(dataset.train_indices)), test_chains=int(len(dataset.test_indices)),
                samples_per_chain=int(dataset.configurations.shape[1]), evaluation_seed=original.EVALUATION_SEED,
                numerical_fixed_points=roots,
                closure={split: original.closure_metrics(arrays[f"{split}_direct"], arrays[f"{split}_one_step"],
                    arrays[f"{split}_rollout"], arrays["train_direct"]) for split in ("train", "test")},
                g_statistics={split: dict(mean=arrays[f"{split}_direct"].mean(axis=0).tolist(),
                    std=arrays[f"{split}_direct"].std(axis=0, ddof=0).tolist(),
                    minimum=arrays[f"{split}_direct"].min(axis=0).tolist(),
                    maximum=arrays[f"{split}_direct"].max(axis=0).tolist()) for split in ("train", "test")})
        if source_receipt(options.source) != receipt or original.sha256_file(__file__) != manifest["continuation_source_sha256"]:
            raise RuntimeError("Pinned source or continuation implementation changed during execution")
        # load_dataset checks actual new-experiment shards before use; confirm
        # those same configurations and sidecars also stayed immutable through
        # continuation.  No historical Ising data or unrelated caches are read.
        data_root = Path(receipt["data_root"]).resolve()
        for shard in dataset.receipts:
            for path_key, sha_key in (("path", "sha256"), ("sidecar", "sidecar_sha256")):
                if original.sha256_file(original._inside(data_root, shard[path_key])) != shard[sha_key]:
                    raise RuntimeError("Source physical shard or sidecar changed during continuation")
        peak = original._check_budget(options, device, compute_started)
        summary.update(peak_cuda_bytes=peak, timing=dict(initialization_seconds=initialization_seconds,
            training_seconds=training_seconds, evaluation_seconds=evaluation_seconds,
            seconds_per_update=training_seconds/(step-SOURCE_STEPS),
            compute_seconds=time.perf_counter()-compute_started, wall_seconds=time.perf_counter()-started,
            cpu_seconds=time.process_time()-cpu_started))
        write(output / "summary.json", summary)
        status = dict(**base_status, status="complete", workflow_complete=not options.profile_only,
            profile_complete=options.profile_only, scientific_result=not options.profile_only,
            actual_steps=step, additional_steps=step-SOURCE_STEPS, latest_checkpoint_step=latest_checkpoint,
            peak_cuda_bytes=peak, returncode=0, error=None, requires_user_action=False,
            summary_sha256=original.sha256_file(output / "summary.json"))
        if not options.profile_only:
            status["evaluation_arrays_sha256"] = original.sha256_file(output / "evaluation_arrays.npz")
        write(output / "status.json", status)
        return summary
    except Exception as exc:
        write(output / "status.json", dict(**base_status,
            status="budget_hold" if isinstance(exc, original.BudgetHold) else "failed",
            workflow_complete=False, actual_steps=step, additional_steps=step-SOURCE_STEPS,
            latest_checkpoint_step=latest_checkpoint, error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(), requires_user_action=True,
            returncode=3 if isinstance(exc, original.BudgetHold) else 1))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--expected-gpu-uuid", default=GPU_UUID)
    parser.add_argument("--additional-steps", type=int, default=ADDITIONAL_STEPS)
    parser.add_argument("--max-training-seconds", type=float, default=3600.)
    parser.add_argument("--max-cuda-bytes", type=int, default=8*1024**3)
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--profile-steps", type=int, default=PROFILE_STEPS)
    options = ContinueOptions(**vars(parser.parse_args(argv)))
    try:
        return run_continuation(options)
    except original.BudgetHold:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
