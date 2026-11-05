"""Recover paired 105-feature halves for the fixed-u scalar Q-dimension ablation.

The completed source model and physical fields stay on their host.  This new
cache repeats exactly the subset and C-noise draws in the existing scalar R
cache, then checks their pooled features and old-Q encodings against that cache.
No physical sampler, optimizer or parameter update is invoked.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from experiments import phi4_qcr as original
from experiments import phi4_ru_r_cache as shared
from experiments import phi4_scalar_r_cache as scalar
from recurrent_rg.phi4_qcr import Phi4QCR


RECIPE = "phi4_u1_frozen_C_paired_half_features_v1"
OUTPUT_BUDGET_BYTES = 64 * 1024**2
PREFIXES = (("fit", "fit", "train"), ("eval_train", "eval", "train"),
            ("eval_test", "eval", "test"))


def _verify_cache(cache):
    """Verify old public files without loading physical fields or checkpoint data."""
    cache = Path(cache).resolve()
    inventory_path = cache / "public_files.json"
    inventory = json.loads(inventory_path.read_text())
    if not isinstance(inventory, dict) or set(inventory) != {
            "manifest.json", "receipt.json", "status.json", "cache.npz"}:
        raise ValueError("Require the complete four-member scalar-cache inventory")
    pins = {str(inventory_path): original.sha256_file(inventory_path)}
    for name, digest in inventory.items():
        path = (cache / name).resolve()
        if (path.parent != cache or not isinstance(digest, str) or len(digest) != 64
                or original.sha256_file(path) != digest):
            raise ValueError(f"Invalid or changed scalar-cache public file: {name}")
        pins[str(path)] = digest
    status = json.loads((cache / "status.json").read_text())
    manifest = json.loads((cache / "manifest.json").read_text())
    receipt = json.loads((cache / "receipt.json").read_text())
    if (status.get("status") != "complete" or status.get("workflow_complete") is not True
            or status.get("returncode") != 0 or status.get("error") is not None
            or status.get("requires_user_action") is not False
            or manifest.get("recipe") != "phi4_u1_frozenQC_scalar_R_cache_v1"
            or manifest.get("source_state_sha256") != scalar.SOURCE_STATE_SHA
            or manifest.get("source_data_manifest_sha256") != scalar.DATA_SHA
            or manifest.get("seed_namespace") != scalar.NAMESPACE
            or manifest.get("fit_repeats") != 4 or manifest.get("evaluation_repeats") != 4
            or manifest.get("fields_per_encoding") != 64
            or manifest.get("lattice_sizes") != [36, 12, 4]
            or manifest.get("source_training_steps") != 3000
            or receipt.get("source_unchanged") is not True
            or receipt.get("cache_sha256") != inventory["cache.npz"]):
        raise ValueError("Require the pinned completed fixed-u scalar cache")
    return manifest, pins


def _load_source(source, cache_manifest):
    """Check the same immutable 3000-step source and instantiate it locally."""
    source = Path(source).resolve()
    if Path(cache_manifest["source_root"]).resolve() != source:
        raise ValueError("Source root differs from the old scalar-cache provenance")
    pins = dict(cache_manifest["source_file_sha256"])
    datapins = dict(cache_manifest["source_data_sha256"])
    shared._verify_unchanged(pins | datapins)
    manifest = json.loads((source / "run/manifest.json").read_text())
    summary = json.loads((source / "run/summary.json").read_text())
    status = json.loads((source / "run/status.json").read_text())
    if (status.get("status") != "complete" or status.get("workflow_complete") is not True
            or status.get("returncode") != 0 or status.get("error") is not None
            or summary.get("actual_steps") != 3000 or summary.get("theory_dim") != 1
            or summary.get("final_state_sha256") != scalar.SOURCE_STATE_SHA
            or manifest.get("data_manifest_sha256") != scalar.DATA_SHA
            or manifest.get("lattice_sizes") != [36, 12, 4]
            or manifest.get("hidden") != 96 or manifest.get("closure_objective") != "all_start_anchor"
            or manifest.get("options", {}).get("seed") != 91
            or status.get("summary_sha256") != pins[str(source / "run/summary.json")]
            or status.get("evaluation_arrays_sha256") != pins[str(source / "run/evaluation_arrays.npz")]):
        raise ValueError("Frozen source no longer matches the original scalar experiment")
    codepins = shared._pinned_source_code(manifest)
    codepins.update({str(Path(module.__file__).resolve()): original.sha256_file(module.__file__)
                     for module in (scalar,)})
    dataset = original.load_dataset(source / "data")
    expected_data = {str(source / "data" / entry[key]): entry[hashkey]
                     for entry in dataset.receipts for key, hashkey in
                     (("path", "sha256"), ("sidecar", "sidecar_sha256"))}
    if (dataset.configurations.shape != (52, 128, 36, 36, 1)
            or len(dataset.train_indices) != 32 or len(dataset.test_indices) != 20
            or dataset.manifest_sha256 != scalar.DATA_SHA
            or dataset.receipts != manifest["data_shards"] or datapins != expected_data
            or dataset.scale != summary["scale"]):
        raise ValueError("Original physical chains, fingerprints or amplitude normalization differ")
    checkpoint = torch.load(source / "run/training_step_3000.pt", map_location="cpu", weights_only=False)
    if checkpoint.get("step") != 3000 or checkpoint.get("manifest") != manifest:
        raise ValueError("Checkpoint provenance differs from the original 3000-step model")
    with torch.random.fork_rng(devices=[]):
        model = Phi4QCR(dataset.scale, hidden=96, theory_dim=1).cpu().double()
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    model.eval().requires_grad_(False)
    for parameter in model.parameters():
        parameter.grad = None
    if (original.state_fingerprint(model) != scalar.SOURCE_STATE_SHA
            or original.module_fingerprints(model) != cache_manifest["source_module_sha256"]
            or original.module_fingerprints(model) != summary["final_module_sha256"]):
        raise ValueError("Frozen source Q/C/R module fingerprints differ")
    return model, dataset, pins | datapins | codepins


def encode_half_features(model, configurations, indices, *, c_seed):
    """Return first/second-half features and old-Q checks along one C trajectory."""
    indices = np.asarray(indices)
    if (configurations.device.type != "cpu" or configurations.dtype != torch.float64
            or configurations.ndim != 4 or configurations.shape[0] != 128
            or indices.shape != (64,) or indices.dtype.kind not in "iu"
            or len(np.unique(indices)) != 64 or np.any(indices < 0) or np.any(indices >= 128)):
        raise ValueError("Expected a CPU float64 physical chain and 64 unique saved-field indices")
    if any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()):
        raise ValueError("The source model must remain frozen with all gradients None")
    generator = torch.Generator(device="cpu").manual_seed(int(c_seed))
    x = configurations[torch.from_numpy(indices.astype(np.int64, copy=False))].unsqueeze(0)
    with torch.no_grad():
        levels = model.coarse_grain(x, generator=generator)
        if len(levels) != 3:
            raise ValueError("Require exactly three observed scales")
        features, direct = [], []
        for level in levels:
            first, second = model.features(level[:, :32]), model.features(level[:, 32:])
            features.append(torch.stack((first, second), dim=1)[0])
            direct.append((.5 * (model.Q.from_features(first) + model.Q.from_features(second)))[0])
        values = dict(half_features=torch.stack(features).numpy().copy(),
                      source_direct=torch.stack(direct).numpy().copy())
    if any(not np.isfinite(value).all() for value in values.values()):
        raise FloatingPointError("Nonfinite paired-half feature cache row")
    return values


def compare_reference(values, feature_target, direct):
    """An absolute 1e-12 tolerance checks the identical old draws, not scientific agreement."""
    halves = np.asarray(values["half_features"])
    encoded = np.asarray(values["source_direct"])
    feature_target, direct = np.asarray(feature_target), np.asarray(direct)
    if (halves.shape != (3, 2, 105) or encoded.shape != (3, 1)
            or feature_target.shape != (3, 105) or direct.shape != (3, 1)
            or any(value.dtype != np.float64 or not np.isfinite(value).all()
                   for value in (halves, encoded, feature_target, direct))):
        raise ValueError("Invalid old-cache comparison shapes, dtypes or finite values")
    pooled = .5 * (halves[:, 0] + halves[:, 1])
    feature_error = float(np.max(np.abs(pooled - feature_target)))
    direct_error = float(np.max(np.abs(encoded - direct)))
    if not np.allclose(pooled, feature_target, rtol=0., atol=1e-12):
        raise RuntimeError("Paired feature mean differs from the original cached feature target")
    if not np.allclose(encoded, direct, rtol=0., atol=1e-12):
        raise RuntimeError("Old Q re-encoding differs from the original cached direct g")
    return feature_error, direct_error


def run(source, scalar_cache, output, cpu_budget_seconds=600.):
    source, scalar_cache, output = map(lambda path: Path(path).resolve(), (source, scalar_cache, output))
    if (not math.isfinite(cpu_budget_seconds) or not 0 < cpu_budget_seconds <= 600):
        raise ValueError("Feature collection CPU budget must be positive and <=600 seconds")
    if output == source or source in output.parents or output == scalar_cache or scalar_cache in output.parents:
        raise ValueError("New feature output must be outside both immutable input roots")
    output.mkdir(parents=True, exist_ok=False)
    started, completed = time.process_time(), 0
    write, sha = original.write_json, original.sha256_file

    def check():
        if time.process_time() - started > cpu_budget_seconds:
            raise shared.BudgetHold("Feature collection CPU budget exceeded; draws are not reduced")
        if sum(path.stat().st_size for path in output.rglob("*") if path.is_file()) > OUTPUT_BUDGET_BYTES:
            raise shared.BudgetHold("Public feature cache exceeds64MiB")

    def progress(name, **extra):
        write(output / "status.json", dict(status=name, workflow_complete=False, returncode=None,
            error=None, requires_user_action=False, encoded_rows=completed,
            cpu_seconds=time.process_time() - started, **extra))

    progress("initializing")
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        cache_manifest, cachepins = _verify_cache(scalar_cache)
        model, dataset, sourcepins = _load_source(source, cache_manifest)
        before, modules = original.state_fingerprint(model), original.module_fingerprints(model)
        codepins = {str(Path(__file__).resolve()): sha(__file__)}
        allpins = cachepins | sourcepins | codepins
        arrays, max_features, max_direct = {}, 0., 0.
        # Labels enter only this public provenance copy, never a sampler, optimizer or seed.
        with np.load(scalar_cache / "cache.npz", allow_pickle=False) as old:
            for split, selected in (("train", dataset.train_indices), ("test", dataset.test_indices)):
                ids, labels = old[f"{split}_cell_ids"], old[f"{split}_labels"]
                if (not np.array_equal(ids, dataset.cell_ids[selected])
                        or not np.array_equal(labels, dataset.labels[selected])):
                    raise ValueError("Original physical-chain condition ordering differs")
                arrays[f"{split}_cell_ids"], arrays[f"{split}_labels"] = ids.copy(), labels.copy()
            for prefix, purpose, split in PREFIXES:
                selected = dataset.train_indices if split == "train" else dataset.test_indices
                count = len(selected)
                reference_features = old[f"{prefix}_feature_target"]
                reference_direct = old[f"{prefix}_direct"]
                saved_indices = old[f"{prefix}_subset_indices"]
                saved_seeds = old[f"{prefix}_c_seeds"]
                if (reference_features.shape != (4, count, 3, 105)
                        or reference_direct.shape != (4, count, 3, 1)
                        or saved_indices.shape != (4, count, 64)
                        or saved_seeds.shape != (4, count)):
                    raise ValueError("Old scalar-cache encoding shape differs")
                arrays[f"{prefix}_half_features"] = np.empty((4, count, 3, 2, 105), dtype=np.float64)
                arrays[f"{prefix}_feature_target"] = reference_features.copy()
                arrays[f"{prefix}_source_direct"] = reference_direct.copy()
                arrays[f"{prefix}_subset_indices"] = saved_indices.copy()
                arrays[f"{prefix}_c_seeds"] = saved_seeds.copy()
                for repeat in range(4):
                    for row, chain in enumerate(selected):
                        check()
                        chain = int(chain)
                        indices, cseed = scalar.subset(purpose, repeat, chain), scalar.seed(purpose, repeat, chain, "C")
                        if (not np.array_equal(indices, saved_indices[repeat, row])
                                or int(saved_seeds[repeat, row]) != cseed):
                            raise ValueError("Subset/C seed identity differs from the old cache")
                        values = encode_half_features(model, dataset.configurations[chain], indices, c_seed=cseed)
                        fe, ge = compare_reference(values, reference_features[repeat, row], reference_direct[repeat, row])
                        max_features, max_direct = max(max_features, fe), max(max_direct, ge)
                        arrays[f"{prefix}_half_features"][repeat, row] = values["half_features"]
                        completed += 1
                        if completed % 50 == 0:
                            progress("collecting")
                            print(json.dumps(dict(event="fixed_C_features", rows=completed, total=336)), flush=True)
        if completed != 336 or any(not np.isfinite(value).all() for value in arrays.values()):
            raise RuntimeError("Incomplete or nonfinite feature cache")
        if (original.state_fingerprint(model) != before or original.module_fingerprints(model) != modules
                or any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters())):
            raise RuntimeError("Frozen source parameters or gradients changed")
        shared._verify_unchanged(allpins)
        manifest = dict(schema_version=5, recipe=RECIPE, source_root=str(source),
            scalar_cache_root=str(scalar_cache), input_file_sha256=allpins,
            source_state_sha256=before, source_module_sha256=modules,
            source_training_steps=3000, source_data_manifest_sha256=scalar.DATA_SHA,
            source_QC_has_seen_R_validation_conditions=True,
            lattice_sizes=[36, 12, 4], block_size=3, stride=3, feature_dim=105,
            feature_axes=["repeat", "physical_chain", "scale", "half", "feature"],
            train_chains=32, test_chains=20, training_conditions=8, test_conditions=5,
            fit_repeats=4, evaluation_repeats=4, fields_per_encoding=64, fields_per_half=32,
            seed_namespace=scalar.NAMESPACE, exact_reuse_of_old_subset_and_C_seeds=True,
            encoding="direct g is mean(Q(first_half_features), Q(second_half_features)); never Q(mean features)",
            labels_used_for_encoding_or_seeds=False, labels_used_for_training=False,
            source_Q_invocation="frozen old-Q verification only; no Q optimization",
            old_cache_feature_target_max_abs_difference=max_features,
            old_cache_direct_g_max_abs_difference=max_direct,
            verification_rtol=0., verification_atol=1e-12,
            optimizer_updates=dict(Q=0, C=0, R=0), new_physical_samples=0,
            data_warning_count=cache_manifest["data_warning_count"], equilibrium_certified=False,
            qualification="Four reused encodings per physical chain are finite cache draws, not independent physical samples or the original online-Q training distribution.",
            device="cpu", threads=1, gpu_seconds=0.)
        write(output / "manifest.json", manifest)
        np.savez_compressed(output / "features.npz", **arrays)
        check()
        shared._verify_unchanged(allpins)
        write(output / "receipt.json", dict(status="complete", source_unchanged=True,
            old_scalar_cache_unchanged=True, all_parameter_gradients_none=True,
            encoded_rows=completed, cpu_seconds=time.process_time() - started,
            new_physical_samples=0, optimizer_updates=dict(Q=0, C=0, R=0),
            old_cache_feature_target_max_abs_difference=max_features,
            old_cache_direct_g_max_abs_difference=max_direct,
            features_sha256=sha(output / "features.npz")))
        write(output / "status.json", dict(status="complete", workflow_complete=True, returncode=0,
            error=None, requires_user_action=False, encoded_rows=completed))
        write(output / "public_files.json", {name: sha(output / name) for name in
            ("manifest.json", "features.npz", "receipt.json", "status.json")})
        return manifest
    except Exception as exc:
        write(output / "status.json", dict(status="budget_hold" if isinstance(exc, shared.BudgetHold) else "failed",
            workflow_complete=False, returncode=3 if isinstance(exc, shared.BudgetHold) else 1,
            error=str(exc), traceback=traceback.format_exc(), requires_user_action=True, encoded_rows=completed))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--scalar-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cpu-budget-seconds", type=float, default=600.)
    args = parser.parse_args()
    run(args.source, args.scalar_cache, args.output, args.cpu_budget_seconds)


if __name__ == "__main__":
    main()
