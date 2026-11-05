"""Collect independent frozen encodings from the completed fixed-u=1 source."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from experiments import phi4_qcr as original
from experiments import phi4_ru_r_cache as shared
from recurrent_rg.phi4_qcr import Phi4QCR


DATA_SHA = "d554ac421c9a14ef0b94701ff34741bd39236938de17fbc14d60437f7228085b"
SOURCE_STATE_SHA = "86daca6df90dea5ef1937d8b074f1eed52da16e9dfe4e4edd5a0f0265439afe6"
NAMESPACE = "phi4_u1_scalar_R_validation_seed91_20261030_v1"


def seed(purpose, repeat, chain, stream):
    if purpose not in ("fit", "eval") or stream not in ("subset", "C"):
        raise ValueError("Invalid private encoding stream")
    if not 0 <= repeat < 4 or chain < 0:
        raise ValueError("Invalid private encoding identity")
    token = f"{NAMESPACE}|{purpose}|{repeat}|{chain}|{stream}".encode("ascii")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "big") & ((1 << 63) - 1)


def subset(purpose, repeat, chain):
    rng = torch.Generator(device="cpu").manual_seed(seed(purpose, repeat, chain, "subset"))
    return torch.randperm(128, generator=rng)[:64].numpy().copy()


def run(source, output, cpu_budget_seconds=1200.):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not 0 < cpu_budget_seconds <= 1200:
        raise ValueError("Fixed CPU budget must be positive and <=1200 seconds")
    if source == output or source in output.parents:
        raise ValueError("Do not write inside the frozen source experiment")
    output.mkdir(parents=True, exist_ok=False)
    started = time.process_time()
    done = 0
    write, sha = original.write_json, original.sha256_file

    def check():
        if time.process_time() - started > cpu_budget_seconds:
            raise shared.BudgetHold("Encoding CPU budget exceeded; no repeats reduced")
        if sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) > 64 * 1024**2:
            raise shared.BudgetHold("Public encoding cache exceeds64MiB")

    def status(name, **extra):
        write(output / "status.json", dict(status=name, workflow_complete=False,
            returncode=None, error=None, requires_user_action=False, encoded_rows=done,
            cpu_seconds=time.process_time() - started, **extra))

    status("initializing")
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        paths = {key: source / name for key, name in dict(manifest="run/manifest.json",
            summary="run/summary.json", status="run/status.json", data="data/manifest.json",
            checkpoint="run/training_step_3000.pt", arrays="run/evaluation_arrays.npz").items()}
        pins = {str(p): sha(p) for p in paths.values()}
        manifest, summary, state = (json.loads(paths[key].read_text())
                                    for key in ("manifest", "summary", "status"))
        if (state.get("status") != "complete" or state.get("workflow_complete") is not True
                or state.get("returncode") != 0 or state.get("error") is not None
                or state.get("summary_sha256") != pins[str(paths["summary"])]
                or state.get("evaluation_arrays_sha256") != pins[str(paths["arrays"])]
                or summary.get("actual_steps") != 3000 or summary.get("theory_dim") != 1
                or summary.get("final_state_sha256") != SOURCE_STATE_SHA
                or manifest.get("lattice_sizes") != [36, 12, 4]
                or manifest.get("closure_objective") != "all_start_anchor"
                or manifest.get("options", {}).get("seed") != 91
                or manifest.get("data_manifest_sha256") != DATA_SHA
                or pins[str(paths["data"])] != DATA_SHA):
            raise ValueError("Source is not the pinned complete3000-step fixedu1 scalar experiment")
        source_code = shared._pinned_source_code(manifest)
        source_code[str(Path(__file__).resolve())] = sha(__file__)
        source_code[str(Path(shared.__file__).resolve())] = sha(shared.__file__)
        dataset = original.load_dataset(source / "data")
        if (dataset.configurations.shape != (52, 128, 36, 36, 1)
                or len(dataset.train_indices) != 32 or len(dataset.test_indices) != 20
                or dataset.receipts != manifest["data_shards"]
                or dataset.scale != summary["scale"]
                or dataset.manifest_sha256 != DATA_SHA):
            raise ValueError("Physical chains or source normalization changed")
        datapins = {str(source / "data" / entry[key]): entry[hashkey]
                    for entry in dataset.receipts for key, hashkey in
                    (("path", "sha256"), ("sidecar", "sidecar_sha256"))}
        checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)
        if checkpoint.get("step") != 3000 or checkpoint.get("manifest") != manifest:
            raise ValueError("Source checkpoint provenance differs")
        with torch.random.fork_rng(devices=[]):
            model = Phi4QCR(dataset.scale, hidden=96, theory_dim=1).cpu().double()
        model.load_state_dict(checkpoint["model"], strict=True)
        del checkpoint
        model.eval().requires_grad_(False)
        for parameter in model.parameters():
            parameter.grad = None
        before, modules = original.state_fingerprint(model), original.module_fingerprints(model)
        if before != SOURCE_STATE_SHA or modules != summary["final_module_sha256"]:
            raise ValueError("Source model/module state fingerprints differ")
        with np.load(paths["arrays"], allow_pickle=False) as archive:
            original_train = archive["train_direct"].copy()
        if (original_train.shape != (32,3,1) or original_train.dtype != np.float64
                or not np.isfinite(original_train).all()):
            raise ValueError("Original TRAIN evaluation has changed shape/dtype or is nonfinite")
        target_stds = np.std(original_train, axis=0, ddof=0).ravel()
        if np.any(target_stds <= 0):
            raise ValueError("Original TRAIN normalization must be finite and strictly positive")
        low, high = original_train[:, :2].min(), original_train[:, :2].max()
        padding = .1 * max(float(high - low), 1e-6)
        arrays = {"flow_g": np.linspace(low - padding, high + padding, 512)[:, None]}
        flow_input = torch.tensor(arrays["flow_g"], dtype=torch.float64, requires_grad=True)
        flow_value = model.advance(flow_input)
        arrays["source_flow_DR"] = torch.autograd.grad(flow_value.sum(), flow_input)[0].detach().numpy()[:, :, None]
        arrays["source_flow_R"] = flow_value.detach().numpy()
        receipt = dict(schema_version=5, recipe="phi4_u1_frozenQC_scalar_R_cache_v1",
            source_root=str(source), source_file_sha256=pins, source_code_sha256=source_code,
            source_data_sha256=datapins, source_state_sha256=before,
            source_module_sha256=modules, source_training_steps=3000,
            source_data_manifest_sha256=DATA_SHA, theory_dim=1, hidden=96,
            lattice_sizes=[36, 12, 4], target_stds=target_stds.tolist(),
            target_std_source="original immutable128-field TRAIN evaluation; ddof0 per target scale",
            train_input_bounds=[float(low), float(high)], flow_padding_fraction=.1,
            fit_repeats=4, evaluation_repeats=4, fields_per_encoding=64,
            encoding="mean of two disjoint32-field Q encodings along the same C trajectory",
            seed_namespace=NAMESPACE, labels_used_for_seed_or_encoding=False,
            source_training_conditions=8, source_test_conditions=5,
            training_count=32, test_count=20, source_QC_has_seen_R_validation_conditions=True,
            independent_physical_chains_added=0, extra_optimizer_updates=dict(Q=0,C=0,R=0),
            data_warning_count=dataset.warning_count, equilibrium_certified=False,
            sample_semantics=dataset.sample_semantics, sampler_qualification=dataset.qualification,
            device="cpu", gpu_seconds=0., threads=1)
        write(output / "manifest.json", receipt)
        for prefix, purpose, selected in (("fit", "fit", dataset.train_indices),
                ("eval_train", "eval", dataset.train_indices), ("eval_test", "eval", dataset.test_indices)):
            count = len(selected)
            for name, tail in (("direct", (3,1)), ("feature_target", (3,105)),
                               ("source_one_step", (2,1)), ("source_rollout", (3,1))):
                arrays[f"{prefix}_{name}"] = np.empty((4,count) + tail)
            arrays[f"{prefix}_subset_indices"] = np.empty((4,count,64), dtype=np.int64)
            arrays[f"{prefix}_c_seeds"] = np.empty((4,count), dtype=np.int64)
            for repeat in range(4):
                for row, index in enumerate(selected):
                    check()
                    index = int(index)
                    indices, cseed = subset(purpose, repeat, index), seed(purpose, repeat, index, "C")
                    values = shared.encode_chain(model, dataset.configurations[index], indices, c_seed=cseed)
                    for name, value in values.items():
                        arrays[f"{prefix}_{name}"][repeat, row] = value
                    arrays[f"{prefix}_subset_indices"][repeat,row] = indices
                    arrays[f"{prefix}_c_seeds"][repeat,row] = cseed
                    done += 1
                    if done % 50 == 0:
                        status("collecting")
                        print(json.dumps(dict(event="scalar_cache", rows=done, total=336)), flush=True)
        for split, selected in (("train", dataset.train_indices), ("test", dataset.test_indices)):
            arrays[f"{split}_cell_ids"] = dataset.cell_ids[selected].copy()
            arrays[f"{split}_labels"] = dataset.labels[selected].copy()
        arrays["decoder_weight"] = model.Q.normalized_decoder_weight().detach().numpy().copy()
        arrays["decoder_bias"] = model.Q.decoder.bias.detach().numpy().copy()
        if any(not np.isfinite(v).all() for v in arrays.values()):
            raise FloatingPointError("Nonfinite collected arrays")
        if (original.state_fingerprint(model) != before or original.module_fingerprints(model) != modules
                or any(p.requires_grad or p.grad is not None for p in model.parameters())):
            raise RuntimeError("Frozen source model or gradients changed")
        shared._verify_unchanged(pins | source_code | datapins)
        np.savez_compressed(output / "cache.npz", **arrays)
        check()
        write(output / "receipt.json", dict(status="complete", source_unchanged=True,
            all_parameter_gradients_none=True, encoded_rows=done, new_physical_samples=0,
            optimizer_updates=dict(Q=0,C=0,R=0), cpu_seconds=time.process_time()-started,
            cache_sha256=sha(output / "cache.npz")))
        write(output / "status.json", dict(status="complete",workflow_complete=True,returncode=0,
            error=None,requires_user_action=False,encoded_rows=done))
        write(output / "public_files.json", {name:sha(output / name) for name in
            ("manifest.json","cache.npz","receipt.json","status.json")})
        return receipt
    except Exception as exc:
        write(output / "status.json", dict(status="budget_hold" if isinstance(exc, shared.BudgetHold) else "failed",
            workflow_complete=False,returncode=3 if isinstance(exc,shared.BudgetHold) else 1,
            error=str(exc),traceback=traceback.format_exc(),requires_user_action=True,encoded_rows=done))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cpu-budget-seconds", type=float, default=1200.)
    args=parser.parse_args()
    run(args.source,args.output,args.cpu_budget_seconds)


if __name__ == "__main__":
    main()
