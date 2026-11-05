"""Isolated sufficient-statistic cache for a frozen-C phi4 Q-only diagnostic.

This explicitly named diagnostic does not sample physical configurations, call
the information objective, update any source parameter, or apply the old R to
new Q coordinates.  It only reads the completed, pinned source run and its
already saved field configurations.  C draws are conditional model randomness,
not additional independent physical ensembles.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from experiments.phi4_qcr import (
    EVALUATION_SEED, _device_receipt, load_dataset, sha256_file,
    state_fingerprint, write_json,
)
from recurrent_rg.phi4_qcr import FEATURE_DIM, Phi4QCR


SCHEMA = "phi4_frozen_C_Q_diagnostic_features_v1"
TRAIN_C_DRAW_SEEDS = (202610270910, 202610270911)
SOURCE_CHECKPOINT_SHA256 = "5df6f4d55d97792be13ecf1811eb44b829ba283210cd46148c8d478e46bc2a7e"
SOURCE_DATA_MANIFEST_SHA256 = "588921b87e3f10be9a8c2ce552d70926758b042481a3800a4efc0fa37430e361"
SOURCE_STATE_SHA256 = "fa1d6983d7c1a1278c1fd5e069f7ef0339b7cbead04320930450049a9080b696"
GPU_UUID = "GPU-18fac97c-11ee-4624-d298-9074e8d7028c"
TARGET_ATOL = 1e-12
MAX_CUDA_BYTES = 8 * 1024 ** 3


class BudgetHold(RuntimeError):
    """The fixed diagnostic cap was reached; no retry or smaller run follows."""


def _frozen(model):
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Feature collection requires an eval-mode, fully frozen Q/C/R model")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise ValueError("Source Q/C/R gradients must all be None")


@torch.no_grad()
def chain_feature_arrays(model, chain, device, *, generator, per_sample=True):
    """One ancestral C path, with no Q/R calls or physical-field generation.

    The batch-axis transpose makes each saved configuration its own temporary
    statistical ensemble.  All configurations have equal lattice area, hence
    averaging their normalized local probabilities equals pooling them.  Half
    features are also computed directly to reproduce source evaluation exactly.
    """
    _frozen(model)
    x = chain.unsqueeze(0).to(device=device, dtype=torch.float64)
    if x.shape[1] < 2 or x.shape[1] % 2:
        raise ValueError("An even number of saved configurations is required")
    levels = model.coarse_grain(x, generator=generator)
    first, second, individual = [], [], []
    maximum_difference = 0.
    for level in levels:
        half = level.shape[1] // 2
        a = model.features(level[:, :half]).cpu().numpy()[0]
        b = model.features(level[:, half:]).cpu().numpy()[0]
        first.append(a)
        second.append(b)
        if per_sample:
            value = model.features(level.transpose(0, 1)).cpu().numpy()
            maximum_difference = max(maximum_difference,
                float(np.max(np.abs(value.mean(axis=0) - .5 * (a + b)))))
            individual.append(value)
    values = np.stack(individual) if per_sample else None
    return values, np.stack(first), np.stack(second), maximum_difference


def collect_feature_arrays(model, dataset, device, *, train_seeds=TRAIN_C_DRAW_SEEDS,
                           evaluation_seed=EVALUATION_SEED, budget_check=None,
                           progress=None):
    """Return statistics only; coupling labels remain explicitly eval metadata."""
    _frozen(model)
    train_indices, test_indices = dataset.train_indices, dataset.test_indices
    draws, maximum_difference = [], 0.
    for draw_index, seed in enumerate(train_seeds):
        generator = torch.Generator(device=device).manual_seed(seed)
        rows = []
        for ordinal, index in enumerate(train_indices):
            if budget_check is not None:
                budget_check()
            value, _, _, difference = chain_feature_arrays(model,
                dataset.configurations[index], device, generator=generator, per_sample=True)
            rows.append(value)
            maximum_difference = max(maximum_difference, difference)
            if progress is not None:
                progress("train", draw_index, ordinal + 1, len(train_indices))
        draws.append(np.stack(rows))
    arrays = dict(train_per_sample=np.stack(draws),
        train_indices=np.asarray(train_indices, dtype=np.int64),
        test_indices=np.asarray(test_indices, dtype=np.int64))
    # Do not skip TEST chains: the source CUDA RNG stream traversed all chains
    # in the original dataset order, so the same stream is reproduced here.
    generator = torch.Generator(device=device).manual_seed(evaluation_seed)
    first, second = [], []
    for ordinal, chain in enumerate(dataset.configurations):
        if budget_check is not None:
            budget_check()
        _, a, b, _ = chain_feature_arrays(model, chain, device,
            generator=generator, per_sample=False)
        first.append(a)
        second.append(b)
        if progress is not None:
            progress("evaluation", 0, ordinal + 1, len(dataset.configurations))
    first, second = np.stack(first), np.stack(second)
    for split, indices in (("train", train_indices), ("test", test_indices)):
        arrays[f"eval_{split}_first"] = first[indices]
        arrays[f"eval_{split}_second"] = second[indices]
        arrays[f"eval_{split}_labels"] = dataset.labels[indices].copy()
        arrays[f"eval_{split}_cell_ids"] = dataset.cell_ids[indices].copy()
        arrays[f"eval_{split}_chain_ids"] = np.asarray(indices, dtype=np.int64)
    if maximum_difference > TARGET_ATOL:
        raise RuntimeError(f"Per-configuration feature pooling failed: {maximum_difference}")
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise RuntimeError("Nonfinite diagnostic statistics")
    _frozen(model)
    return arrays, maximum_difference


def verify_reference_targets(arrays, reference_path, *, tolerance=TARGET_ATOL):
    """Require the same frozen C trajectories as the official source arrays."""
    differences = {}
    with np.load(reference_path, allow_pickle=False) as reference:
        for split in ("train", "test"):
            actual = .5 * (arrays[f"eval_{split}_first"] + arrays[f"eval_{split}_second"])
            expected = reference[f"{split}_feature_target"]
            if actual.shape != expected.shape:
                raise ValueError("Source/diagnostic feature target shape mismatch")
            differences[split] = float(np.max(np.abs(actual - expected)))
            if differences[split] > tolerance:
                raise RuntimeError(f"Source {split} evaluation target mismatch: {differences[split]}")
            for suffix in ("labels", "cell_ids"):
                if not np.array_equal(arrays[f"eval_{split}_{suffix}"], reference[f"{split}_{suffix}"]):
                    raise ValueError(f"Source {split} {suffix} ordering mismatch")
    return differences


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _finite_positive(value, name):
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def collect(source_run, output, *, device="cuda:1", cpu_budget_seconds=180.,
            gpu_budget_seconds=120.):
    """Collect the pinned production-source cache into a fresh diagnostic dir."""
    _finite_positive(cpu_budget_seconds, "CPU budget")
    _finite_positive(gpu_budget_seconds, "GPU budget")
    started, cpu_started = time.perf_counter(), time.process_time()
    source_run, output = Path(source_run).resolve(), Path(output).resolve()
    manifest = _json(source_run / "manifest.json")
    summary, status = _json(source_run / "summary.json"), _json(source_run / "status.json")
    data_root = Path(manifest["data_root"]).resolve()
    source_experiment = data_root.parent
    if output == source_experiment or source_experiment in output.parents:
        raise ValueError("Diagnostic output must not be inside the immutable source experiment")
    if output.exists():
        raise FileExistsError("A fresh cache directory is required; overwrite/resume is disabled")
    if (status.get("status") != "complete" or not status.get("workflow_complete")
            or status.get("returncode") != 0 or summary.get("status") != "complete"
            or summary.get("actual_steps") != 3000):
        raise ValueError("The source formal run must be complete at its fixed 3000-step endpoint")
    if manifest.get("theory_dim") != 1 or manifest.get("lattice_sizes") != [36, 12, 4]:
        raise ValueError("Unexpected source model geometry")
    source_paths = {name: source_run / name for name in
        ("manifest.json", "summary.json", "status.json", "training_step_3000.pt", "evaluation_arrays.npz")}
    source_paths["data_manifest.json"] = data_root / "manifest.json"
    source_hashes = {name: sha256_file(path) for name, path in source_paths.items()}
    if source_hashes["training_step_3000.pt"] != SOURCE_CHECKPOINT_SHA256:
        raise ValueError("Pinned source checkpoint SHA256 mismatch")
    if (source_hashes["data_manifest.json"] != SOURCE_DATA_MANIFEST_SHA256
            or manifest.get("data_manifest_sha256") != SOURCE_DATA_MANIFEST_SHA256):
        raise ValueError("Pinned source physical data manifest SHA256 mismatch")
    if summary.get("final_state_sha256") != SOURCE_STATE_SHA256:
        raise ValueError("Pinned source tensor state SHA256 mismatch")
    if source_hashes["evaluation_arrays.npz"] != status.get("evaluation_arrays_sha256"):
        raise ValueError("Source evaluation archive SHA256 mismatch")
    if source_hashes["summary.json"] != status.get("summary_sha256"):
        raise ValueError("Source summary SHA256 mismatch")
    # The imported model and loader must match the frozen source implementation.
    import experiments.phi4_qcr as training_module
    import recurrent_rg.phi4_qcr as model_module
    for module in (training_module, model_module):
        matching = [digest for path, digest in manifest["source_sha256"].items()
                    if str(path).endswith("/" + module.__name__.replace(".", "/") + ".py")]
        if len(matching) != 1 or sha256_file(module.__file__) != matching[0]:
            raise ValueError(f"Frozen source implementation mismatch: {module.__name__}")
    output.mkdir(parents=True, exist_ok=False)
    gpu_started, gpu_seconds, own_peak = None, 0., 0
    actual_device = torch.device(device)
    if actual_device.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU or explicitly selected CUDA device is supported")
    write_json(output / "status.json", dict(status="initializing", workflow_complete=False,
        returncode=None, error=None, requires_user_action=False, schema=SCHEMA))

    def check_budget():
        nonlocal own_peak
        if gpu_started is not None:
            torch.cuda.synchronize(actual_device)
            own_peak = max(own_peak, int(torch.cuda.max_memory_allocated(actual_device)),
                           int(torch.cuda.max_memory_reserved(actual_device)))
            elapsed_gpu = time.perf_counter() - gpu_started
            if elapsed_gpu > gpu_budget_seconds:
                raise BudgetHold(f"GPU collection cap reached: {elapsed_gpu:.3f}s > {gpu_budget_seconds}s")
            if own_peak > MAX_CUDA_BYTES:
                raise BudgetHold(f"Own CUDA peak exceeded: {own_peak} > {MAX_CUDA_BYTES}")
        elapsed_cpu = time.process_time() - cpu_started
        if elapsed_cpu > cpu_budget_seconds:
            raise BudgetHold(f"CPU collection cap reached: {elapsed_cpu:.3f}s > {cpu_budget_seconds}s")

    def progress(phase, draw, completed, total):
        check_budget()
        write_json(output / "status.json", dict(status="collecting", workflow_complete=False,
            current=phase, draw_index=draw, completed_chains=completed, total_chains=total,
            cpu_seconds=time.process_time() - cpu_started,
            gpu_seconds=time.perf_counter() - gpu_started if gpu_started is not None else 0.,
            peak_cuda_bytes=own_peak, returncode=None, error=None, requires_user_action=False,
            schema=SCHEMA))

    try:
        torch.set_num_threads(1)
        dataset = load_dataset(data_root)
        if dataset.manifest_sha256 != SOURCE_DATA_MANIFEST_SHA256:
            raise ValueError("Physical data manifest changed while loading")
        if (tuple(dataset.configurations.shape) != (48, 128, 36, 36, 1)
                or len(dataset.train_indices) != 32 or len(dataset.test_indices) != 16):
            raise ValueError("Unexpected pinned physical ensemble shape or split")
        if abs(dataset.scale - manifest["scale"]) > 1e-14:
            raise ValueError("TRAIN-only microscopic normalization changed")
        check_budget()
        model = Phi4QCR(manifest["scale"], hidden=manifest["hidden"], theory_dim=1).double()
        checkpoint = torch.load(source_paths["training_step_3000.pt"], map_location="cpu", weights_only=True)
        if checkpoint["step"] != 3000:
            raise ValueError("Source checkpoint step mismatch")
        model.load_state_dict(checkpoint["model"])
        del checkpoint
        model.eval().requires_grad_(False)
        before = state_fingerprint(model)
        if before != SOURCE_STATE_SHA256:
            raise ValueError("Loaded source tensor fingerprint mismatch")
        check_budget()
        if actual_device.type == "cuda":
            gpu_started = time.perf_counter()
        device_receipt = _device_receipt(actual_device, GPU_UUID if actual_device.type == "cuda" else None)
        if actual_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(actual_device)
        model.to(device=actual_device, dtype=torch.float64)
        arrays, pooling_difference = collect_feature_arrays(model, dataset, actual_device,
            budget_check=check_budget, progress=progress)
        target_differences = verify_reference_targets(arrays, source_paths["evaluation_arrays.npz"])
        after = state_fingerprint(model)
        _frozen(model)
        if after != before:
            raise RuntimeError("Source Q/C/R tensor fingerprint changed during collection")
        check_budget()
        if actual_device.type == "cuda":
            # End card-occupancy accounting only after the frozen source tensors
            # and all transient C fields have left the selected device.
            model.cpu()
            torch.cuda.empty_cache()
            torch.cuda.synchronize(actual_device)
            gpu_seconds = time.perf_counter() - gpu_started
            gpu_started = None
            if gpu_seconds > gpu_budget_seconds:
                raise BudgetHold(f"GPU cap exceeded during final release: {gpu_seconds:.3f}s")
        check_budget()
        for name, path in source_paths.items():
            if sha256_file(path) != source_hashes[name]:
                raise RuntimeError(f"Immutable source artifact changed: {name}")
        for row in dataset.receipts:
            if (sha256_file(data_root / row["path"]) != row["sha256"]
                    or sha256_file(data_root / row["sidecar"]) != row["sidecar_sha256"]):
                raise RuntimeError("Source physical shard/sidecar changed during collection")
        check_budget()
        temporary = output / "arrays.npz.partial"
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, output / "arrays.npz")
        check_budget()
        receipt = dict(schema=SCHEMA, status="complete", source_run=str(source_run),
            source_checkpoint=str(source_paths["training_step_3000.pt"]),
            source_checkpoint_sha256=source_hashes["training_step_3000.pt"],
            source_state_sha256=before, source_state_sha256_after=after,
            source_manifest_sha256=source_hashes["manifest.json"],
            source_data_root=str(data_root), source_data_manifest_sha256=dataset.manifest_sha256,
            source_hashes_before=source_hashes, immutable_source_hashes_verified_after=True,
            arrays_sha256=sha256_file(output / "arrays.npz"),
            shapes={key: list(value.shape) for key, value in arrays.items()},
            dtypes={key: str(value.dtype) for key, value in arrays.items()},
            scale=manifest["scale"], feature_dim=FEATURE_DIM,
            train_c_draw_seeds=list(TRAIN_C_DRAW_SEEDS), evaluation_c_draw_seed=EVALUATION_SEED,
            evaluation_target_max_abs_difference=target_differences,
            per_configuration_pooling_max_abs_difference=pooling_difference,
            target_agreement_absolute_tolerance=TARGET_ATOL,
            train_order="Original dataset train_indices order, per draw; each C trajectory is ancestral",
            evaluation_order="Original full dataset order, including TEST, matching source RNG stream",
            chain_id_semantics="Original full-dataset global chain index, not within-cell chain number",
            physical_sampling_added=0, information_objective_calls=0,
            Q_updates=0, C_updates=0, R_updates=0, gradients_all_none=True,
            Q_and_R_forward_calls=0, model_inference="Frozen C conditional draws and fixed local features only",
            training_statistics_keys=["train_per_sample"],
            evaluation_only_metadata_keys=[key for key in arrays if key.endswith(("labels", "cell_ids", "chain_ids"))],
            labels_used_in_training=False, equilibrium_certified=False,
            source_data_warning_count=dataset.warning_count,
            source_sampler_qualification=dataset.qualification,
            budgets=dict(cpu_seconds=cpu_budget_seconds, gpu_seconds=gpu_budget_seconds,
                         own_cuda_peak_bytes=MAX_CUDA_BYTES),
            resources=dict(cpu_seconds=time.process_time() - cpu_started,
                gpu_seconds=gpu_seconds, wall_seconds=time.perf_counter() - started,
                peak_cuda_bytes=own_peak, device=device_receipt, threads=1),
            qualification="Statistics-only cache, not physical configurations. Two C draws do not create independent physical chains. Source sampling warnings remain; this cache certifies neither equilibrium nor a physical RG flow.")
        write_json(output / "receipt.json", receipt)
        write_json(output / "status.json", dict(status="complete", workflow_complete=True,
            returncode=0, error=None, requires_user_action=False, schema=SCHEMA,
            arrays_sha256=receipt["arrays_sha256"], receipt_sha256=sha256_file(output / "receipt.json"),
            **receipt["resources"]))
        return receipt
    except BaseException as error:
        if gpu_started is not None:
            gpu_seconds = time.perf_counter() - gpu_started
        is_budget = isinstance(error, BudgetHold)
        write_json(output / "status.json", dict(status="budget_hold" if is_budget else "failed",
            workflow_complete=False, returncode=2 if is_budget else 1,
            error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(),
            requires_user_action=True, schema=SCHEMA,
            cpu_seconds=time.process_time() - cpu_started, gpu_seconds=gpu_seconds,
            wall_seconds=time.perf_counter() - started, peak_cuda_bytes=own_peak,
            physical_sampling_added=0, parameters_updated=0))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--cpu-budget-seconds", type=float, default=180.)
    parser.add_argument("--gpu-budget-seconds", type=float, default=120.)
    args = parser.parse_args()
    try:
        receipt = collect(args.source_run, args.output, device=args.device,
            cpu_budget_seconds=args.cpu_budget_seconds, gpu_budget_seconds=args.gpu_budget_seconds)
    except BudgetHold as error:
        parser.exit(2, f"{error}\n")
    print(json.dumps(dict(status=receipt["status"], arrays_sha256=receipt["arrays_sha256"],
                         resources=receipt["resources"]), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
