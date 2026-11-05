"""CPU collection of frozen-Q/C encodings for the authorized phi4 R calibration.

The completed source model and its original field shards stay on their host.
Only compressed latent/features arrays and provenance JSON are public outputs.
The four fitting repeats use TRAIN chains only. Four independently seeded
evaluation repeats cover TRAIN and TEST; repeats reuse the original physical
chains and therefore are not independent physical ensembles.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from experiments import phi4_qcr as original
from recurrent_rg import phi4_qcr as model_source
from recurrent_rg.phi4_qcr import Phi4QCR


SCHEMA = "phi4_ru_frozen_QC_64field_repeated_cache_v1"
REPEATS = 4
SUBSET_SIZE = 64
SEED_NAMESPACE = "phi4_ru_R_calibration_seed91_20261029_v1"


class BudgetHold(RuntimeError):
    """A fixed collection budget was exceeded without reducing the plan."""


@dataclass(frozen=True)
class Options:
    source_root: str
    output: str
    cpu_budget_seconds: float = 1200.
    output_budget_bytes: int = 64 * 1024 ** 2

    def validate(self):
        if (not math.isfinite(self.cpu_budget_seconds)
                or not 0 < self.cpu_budget_seconds <= 1200
                or isinstance(self.output_budget_bytes, bool)
                or not isinstance(self.output_budget_bytes, int)
                or not 0 < self.output_budget_bytes <= 64 * 1024 ** 2):
            raise ValueError("Collection budgets must be <=1200 CPU seconds and <=64 MiB")


def derived_seed(purpose: str, repeat: int, chain: int, stream: str) -> int:
    """Stable private seeds depend on identity, never on any coupling label."""
    if (purpose not in ("fit", "eval") or stream not in ("subset", "C")
            or isinstance(repeat, bool) or not isinstance(repeat, int)
            or not 0 <= repeat < REPEATS
            or isinstance(chain, bool) or not isinstance(chain, int) or chain < 0):
        raise ValueError("Invalid cache seed identity")
    token = f"{SEED_NAMESPACE}|{purpose}|{repeat}|{chain}|{stream}".encode("ascii")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "big") & ((1 << 63) - 1)


def subset_indices(purpose: str, repeat: int, chain: int) -> np.ndarray:
    """Ordered sample without replacement; order defines disjoint 32 halves."""
    generator = torch.Generator(device="cpu").manual_seed(
        derived_seed(purpose, repeat, chain, "subset"))
    return torch.randperm(128, generator=generator)[:SUBSET_SIZE].numpy().copy()


def encode_chain(model, configurations, indices, *, c_seed: int):
    """Use one C trajectory and the original mean-of-two-Q encoding at all scales."""
    indices = np.asarray(indices)
    if (configurations.device.type != "cpu" or configurations.dtype != torch.float64
            or configurations.ndim != 4 or configurations.shape[0] != 128
            or indices.shape != (SUBSET_SIZE,) or indices.dtype.kind not in "iu"
            or len(np.unique(indices)) != SUBSET_SIZE
            or np.any(indices < 0) or np.any(indices >= 128)):
        raise ValueError("Expected one CPU float64 chain and 64 unique saved-sample indices")
    if any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise ValueError("Collection requires frozen parameters with no parameter gradients")
    generator = torch.Generator(device="cpu").manual_seed(int(c_seed))
    x = configurations[torch.from_numpy(indices.astype(np.int64, copy=False))].unsqueeze(0)
    with torch.no_grad():
        levels = model.coarse_grain(x, generator=generator)
        if len(levels) != 3:
            raise ValueError("Collection pins three observed scales")
        encoded, targets = [], []
        for level in levels:
            first = model.features(level[:, :32])
            second = model.features(level[:, 32:])
            encoded.append(.5 * (model.Q.from_features(first) + model.Q.from_features(second)))
            targets.append(.5 * (first + second))
        direct = torch.stack(encoded, dim=1)
        one = torch.stack([model.advance(encoded[k]) for k in range(2)], dim=1)
        rollout = [encoded[0]]
        for _ in range(2):
            rollout.append(model.advance(rollout[-1]))
        result = dict(direct=direct[0].numpy().copy(),
            feature_target=torch.stack(targets, dim=1)[0].numpy().copy(),
            source_one_step=one[0].numpy().copy(),
            source_rollout=torch.stack(rollout, dim=1)[0].numpy().copy())
    if any(not np.isfinite(value).all() for value in result.values()):
        raise FloatingPointError("Nonfinite frozen encoding or source prediction")
    return result


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _pinned_source_code(manifest):
    """Match portable repository suffixes, not the original deployment prefix."""
    result = {}
    for suffix, module in (("experiments/phi4_qcr.py", original),
                           ("recurrent_rg/phi4_qcr.py", model_source)):
        matches = [value for path, value in manifest["source_sha256"].items()
                   if Path(path).as_posix().endswith("/" + suffix)]
        if len(matches) != 1:
            raise ValueError(f"Source manifest must uniquely pin {suffix}")
        path = Path(module.__file__).resolve()
        actual = original.sha256_file(path)
        if actual != matches[0]:
            raise RuntimeError(f"Source implementation fingerprint mismatch: {suffix}")
        result[str(path)] = actual
    result[str(Path(__file__).resolve())] = original.sha256_file(__file__)
    return result


def _source_contract(root):
    paths = {"run_manifest": root / "run/manifest.json",
             "run_summary": root / "run/summary.json",
             "run_status": root / "run/status.json",
             "data_manifest": root / "data/manifest.json",
             "checkpoint": root / "run/training_step_3000.pt"}
    fingerprints = {str(path): original.sha256_file(path) for path in paths.values()}
    manifest, summary, status = (_read_json(paths[name])
                                 for name in ("run_manifest", "run_summary", "run_status"))
    if (status.get("status") != "complete" or status.get("workflow_complete") is not True
            or status.get("returncode") != 0 or status.get("error") is not None
            or summary.get("status") != "complete" or summary.get("actual_steps") != 3000
            or summary.get("latest_checkpoint_step") != 3000
            or manifest.get("theory_dim") != 2 or manifest.get("hidden") != 96
            or manifest.get("lattice_sizes") != [36, 12, 4]
            or manifest.get("block_size") != 3 or manifest.get("stride") != 3
            or manifest.get("dtype") != "float64"
            or manifest.get("closure_objective") != "all_start_anchor"
            or manifest.get("options", {}).get("seed") != 91
            or manifest.get("options", {}).get("sample_batch") != 64):
        raise ValueError("Source is not the completed pinned seed91 dg2 b3 3000-step experiment")
    if (status.get("summary_sha256") != fingerprints[str(paths["run_summary"])]
            or manifest.get("data_manifest_sha256") != fingerprints[str(paths["data_manifest"])]
            or summary.get("data_manifest_sha256") != fingerprints[str(paths["data_manifest"])]):
        raise RuntimeError("Source public receipt/manifest fingerprint mismatch")
    sources = _pinned_source_code(manifest)
    return paths, manifest, summary, fingerprints, sources


def _verify_unchanged(fingerprints):
    for path, expected in fingerprints.items():
        if original.sha256_file(path) != expected:
            raise RuntimeError(f"Frozen source changed during collection: {path}")


def _validate_arrays(arrays):
    for prefix, count in (("fit", 136), ("eval_train", 136), ("eval_test", 44)):
        expected = dict(direct=(REPEATS, count, 3, 2),
            feature_target=(REPEATS, count, 3, 105),
            source_one_step=(REPEATS, count, 2, 2),
            source_rollout=(REPEATS, count, 3, 2),
            subset_indices=(REPEATS, count, SUBSET_SIZE),
            subset_seeds=(REPEATS, count), c_seeds=(REPEATS, count))
        for suffix, shape in expected.items():
            value = arrays[f"{prefix}_{suffix}"]
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid collected array {prefix}_{suffix}")
        indices = arrays[f"{prefix}_subset_indices"]
        if (indices.dtype.kind not in "iu" or np.any(indices < 0) or np.any(indices >= 128)
                or np.any(np.diff(np.sort(indices, axis=-1), axis=-1) == 0)):
            raise ValueError("Repeated or invalid saved-sample index in a cache row")
        if not np.array_equal(arrays[f"{prefix}_source_rollout"][:, :, 0],
                              arrays[f"{prefix}_direct"][:, :, 0]):
            raise ValueError("Source rollout must start at the same cached microscopic g")
    for split, count in (("train", 136), ("test", 44)):
        if arrays[f"{split}_cell_ids"].shape != (count,) or arrays[f"{split}_labels"].shape != (count, 3):
            raise ValueError("Invalid evaluation-only cell provenance")
    if set(arrays["train_cell_ids"]) & set(arrays["test_cell_ids"]):
        raise ValueError("TRAIN/TEST physical parameter cells overlap")
    if (arrays["decoder_weight"].shape != (105, 2)
            or arrays["decoder_bias"].shape != (105,)
            or not np.allclose(np.linalg.norm(arrays["decoder_weight"], axis=0), 1., atol=1e-12, rtol=0)):
        raise ValueError("Invalid frozen normalized affine decoder")


def run(options):
    options.validate()
    root, output = Path(options.source_root).resolve(), Path(options.output).resolve()
    if output == root or root in output.parents:
        raise ValueError("New collection output must be outside the frozen source root")
    output.mkdir(parents=True, exist_ok=False)
    started, cpu_started = time.perf_counter(), time.process_time()
    completed = 0

    def check():
        cpu = time.process_time() - cpu_started
        if cpu > options.cpu_budget_seconds:
            raise BudgetHold(f"Collection CPU budget exceeded: {cpu:.3f}s; fixed repeats not reduced")
        used = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
        if used > options.output_budget_bytes:
            raise BudgetHold(f"Collection output budget exceeded: {used} bytes")

    def status(state, **more):
        original.write_json(output / "status.json", dict(status=state, workflow_complete=False,
            returncode=None, error=None, requires_user_action=False, encoded_rows=completed,
            schema=SCHEMA, device="cpu", gpu_seconds=0., **more))

    status("initializing")
    try:
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            if torch.get_num_interop_threads() != 1:
                raise
        paths, manifest, summary, source_files, source_code = _source_contract(root)
        check()
        dataset = original.load_dataset(root / "data")
        check()
        if (dataset.configurations.shape != (180, 128, 36, 36, 1)
                or len(dataset.train_indices) != 136 or len(dataset.test_indices) != 44
                or dataset.manifest_sha256 != manifest["data_manifest_sha256"]
                or dataset.receipts != manifest["data_shards"]
                or summary.get("train_chains") != 136 or summary.get("test_chains") != 44
                or summary.get("samples_per_chain") != 128
                or dataset.scale != summary["scale"] or dataset.scale != manifest["scale"]):
            raise ValueError("Source physical dataset differs from its completed training contract")
        source_data = {}
        for receipt in dataset.receipts:
            source_data[str(root / "data" / receipt["path"])] = receipt["sha256"]
            source_data[str(root / "data" / receipt["sidecar"])] = receipt["sidecar_sha256"]
        # This is a trusted checkpoint produced by our completed source run.
        checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)
        if (checkpoint.get("step") != 3000 or checkpoint.get("manifest") != manifest
                or checkpoint.get("recipe") != manifest["recipe"]):
            raise RuntimeError("Checkpoint provenance differs from the completed source run")
        with torch.random.fork_rng(devices=[]):
            model = Phi4QCR(summary["scale"], hidden=96, theory_dim=2).cpu().double()
        model.load_state_dict(checkpoint["model"], strict=True)
        del checkpoint
        model.requires_grad_(False)
        model.eval()
        for parameter in model.parameters():
            parameter.grad = None
        initial_state = original.state_fingerprint(model)
        initial_modules = original.module_fingerprints(model)
        if (initial_state != summary["final_state_sha256"]
                or initial_modules != summary["final_module_sha256"]):
            raise RuntimeError("Loaded model/module fingerprints differ from the final source state")
        run_manifest = dict(schema_version=5, schema=SCHEMA, options=asdict(options),
            source_root=str(root), source_file_sha256=source_files,
            source_code_sha256=source_code, source_state_sha256=initial_state,
            source_module_sha256=initial_modules, source_data_sha256=source_data,
            source_training_steps=3000, extra_optimizer_updates=dict(Q=0, C=0, R=0),
            device="cpu", threads=1, interop_threads=1, dtype="float64", gpu_seconds=0.,
            fit_repeats=REPEATS, evaluation_repeats=REPEATS,
            source_samples_per_chain=128, selected_samples_per_chain=SUBSET_SIZE,
            subset_policy="64 of 128 ordered indices without replacement per purpose/repeat/chain",
            encoding="mean(Q(features(first32)), Q(features(second32))) along one shared C trajectory",
            feature_target="mean of the same two 32-field feature vectors",
            seed_namespace=SEED_NAMESPACE,
            seed_derivation="SHA256(namespace|purpose|repeat|global_chain_index|stream), first8bytes big-endian masked to63bits",
            rng_streams=["subset", "C"], labels_used_for_seed_or_encoding=False,
            labels_generation_and_evaluation_only=True,
            training_count=136, test_count=44, train_cells=34, test_cells=11,
            lattice_sizes=[36, 12, 4], theory_dim=2, hidden=96,
            independent_physical_chains_added=0, equilibrium_certified=False,
            data_warning_count=dataset.warning_count,
            sampler_qualification=dataset.qualification,
            sampler_schema=dataset.sampler_schema, sample_semantics=dataset.sample_semantics,
            qualification="Repeated stochastic C/subset encodings reuse existing physical chains; no new equilibrium evidence or independent physical samples")
        original.write_json(output / "manifest.json", run_manifest)
        arrays = {}
        for prefix, purpose, selected in (("fit", "fit", dataset.train_indices),
                                          ("eval_train", "eval", dataset.train_indices),
                                          ("eval_test", "eval", dataset.test_indices)):
            count = len(selected)
            arrays.update({f"{prefix}_direct": np.empty((REPEATS, count, 3, 2)),
                f"{prefix}_feature_target": np.empty((REPEATS, count, 3, 105)),
                f"{prefix}_source_one_step": np.empty((REPEATS, count, 2, 2)),
                f"{prefix}_source_rollout": np.empty((REPEATS, count, 3, 2)),
                f"{prefix}_subset_indices": np.empty((REPEATS, count, SUBSET_SIZE), dtype=np.int64),
                f"{prefix}_subset_seeds": np.empty((REPEATS, count), dtype=np.int64),
                f"{prefix}_c_seeds": np.empty((REPEATS, count), dtype=np.int64)})
            for repeat in range(REPEATS):
                for row, chain_value in enumerate(selected):
                    check()
                    chain = int(chain_value)
                    indices = subset_indices(purpose, repeat, chain)
                    c_seed = derived_seed(purpose, repeat, chain, "C")
                    values = encode_chain(model, dataset.configurations[chain], indices, c_seed=c_seed)
                    for name, value in values.items():
                        arrays[f"{prefix}_{name}"][repeat, row] = value
                    arrays[f"{prefix}_subset_indices"][repeat, row] = indices
                    arrays[f"{prefix}_subset_seeds"][repeat, row] = derived_seed(purpose, repeat, chain, "subset")
                    arrays[f"{prefix}_c_seeds"][repeat, row] = c_seed
                    completed += 1
                    if completed % 100 == 0:
                        status("collecting", current=dict(prefix=prefix, repeat=repeat, chain=chain),
                            cpu_seconds=time.process_time() - cpu_started,
                            wall_seconds=time.perf_counter() - started)
                        print(json.dumps(dict(event="phi4_ru_R_cache_progress", encoded_rows=completed,
                            total_rows=REPEATS * (136 + 180), cpu_seconds=time.process_time() - cpu_started)), flush=True)
                    check()
        for split, selected in (("train", dataset.train_indices), ("test", dataset.test_indices)):
            arrays[f"{split}_cell_ids"] = dataset.cell_ids[selected].copy()
            arrays[f"{split}_labels"] = dataset.labels[selected].copy()
            arrays[f"{split}_global_chain_indices"] = selected.copy()
        arrays["decoder_weight"] = model.Q.normalized_decoder_weight().detach().numpy().copy()
        arrays["decoder_bias"] = model.Q.decoder.bias.detach().numpy().copy()
        _validate_arrays(arrays)
        if sum(value.nbytes for value in arrays.values()) > options.output_budget_bytes:
            raise BudgetHold("Uncompressed public cache exceeds output budget")
        if (original.state_fingerprint(model) != initial_state
                or original.module_fingerprints(model) != initial_modules
                or any(p.requires_grad or p.grad is not None for p in model.parameters())):
            raise RuntimeError("Frozen model changed or acquired gradients during collection")
        _verify_unchanged(source_files | source_code | source_data)
        check()
        temporary = output / "cache.npz.partial"
        with temporary.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
        check()
        os.replace(temporary, output / "cache.npz")
        receipt = dict(status="complete", schema=SCHEMA, encoded_rows=completed,
            cache_sha256=original.sha256_file(output / "cache.npz"),
            cache_bytes=(output / "cache.npz").stat().st_size,
            array_shapes={name: list(value.shape) for name, value in arrays.items()},
            source_state_sha256_before=initial_state,
            source_state_sha256_after=original.state_fingerprint(model),
            source_module_sha256_before=initial_modules,
            source_module_sha256_after=original.module_fingerprints(model),
            all_parameters_require_grad_false=True, all_parameter_gradients_none=True,
            source_files_unchanged=True, source_code_unchanged=True, source_data_unchanged=True,
            labels_used_for_seed_or_encoding=False, optimizer_updates=dict(Q=0, C=0, R=0),
            data_warning_count=dataset.warning_count, equilibrium_certified=False,
            cpu_seconds=time.process_time() - cpu_started,
            wall_seconds=time.perf_counter() - started, gpu_seconds=0.)
        original.write_json(output / "receipt.json", receipt)
        check()
        original.write_json(output / "status.json", dict(status="complete", workflow_complete=True,
            returncode=0, error=None, requires_user_action=False, schema=SCHEMA,
            encoded_rows=completed, device="cpu", gpu_seconds=0.,
            cache_sha256=receipt["cache_sha256"], receipt_sha256=original.sha256_file(output / "receipt.json")))
        public = {name: original.sha256_file(output / name)
                  for name in ("cache.npz", "manifest.json", "receipt.json", "status.json")}
        original.write_json(output / "public_files.json", public)
        check()
        print(json.dumps(dict(event="phi4_ru_R_cache_complete", **receipt)), flush=True)
        return receipt
    except Exception as exc:
        original.write_json(output / "status.json", dict(
            status="budget_hold" if isinstance(exc, BudgetHold) else "failed",
            workflow_complete=False, returncode=3 if isinstance(exc, BudgetHold) else 1,
            error=str(exc), traceback=traceback.format_exc(), requires_user_action=True,
            schema=SCHEMA, encoded_rows=completed, device="cpu", gpu_seconds=0.,
            cpu_seconds=time.process_time() - cpu_started,
            wall_seconds=time.perf_counter() - started))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cpu-budget-seconds", type=float, default=1200.)
    parser.add_argument("--output-budget-bytes", type=int, default=64 * 1024 ** 2)
    arguments = parser.parse_args()
    run(Options(**vars(arguments)))


if __name__ == "__main__":
    main()
