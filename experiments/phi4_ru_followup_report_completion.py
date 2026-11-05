"""Complete the interrupted phi4 report from saved measurements, without inference.

The failed physical workflow remains immutable. This separate output only reads
its public cached encodings/predictions and fixed source decoder parameters.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path
import shutil
import socket
import time
import traceback

import numpy as np
import torch

from experiments import phi4_ru_followup_evaluate as evaluator
from scripts import run_phi4_ru_validation_followup as base
from scripts.run_phi4_ru_validation_followup_fine import previous_live_processes


SCHEMA = "phi4_ru_cached_report_completion_v1"
FAILED_ROOT_NAME = "phi4_ru_validation_followup_direct_seed91_vm225_20261030"
PREVIOUS_CPU_SECONDS = 3511.210285067
CPU_CAP_SECONDS = 600.
MODEL_KEYS = tuple(f"{arm}_seed{seed}" for arm in ("adam6000", "selected") for seed in (91, 92, 93))
ERROR = "TypeError: Object of type int64 is not JSON serializable"
SOURCE_TRAIN_EVALUATION_SHA256 = "6b75984089d46f58cbf18e656b605a370d24e05a0fc27ff806902ce1f211dc44"


def verify_failed_root(root):
    root = Path(root).resolve()
    if root.name != FAILED_ROOT_NAME:
        raise ValueError("Completion is restricted to the explicitly failed direct workflow")
    inventory = base.read(root / "public_files.json")
    files = inventory.get("files", inventory)
    required = {"job.json", "status.json", "plan.json", "data_manifest.json",
                "evaluation/manifest.json", "evaluation/status.json",
                "evaluation/groups.json", "evaluation/evaluation_arrays.npz"}
    if not required.issubset(files):
        raise ValueError("Failed workflow public inventory lacks cached report inputs")
    pins = {str(root / "public_files.json"): base.sha(root / "public_files.json")}
    for name, expected in files.items():
        path = base.inside(root, name)
        if path.suffix.lower() not in (".json", ".png", ".npz") or path.name == "data.npz":
            raise ValueError("Forbidden raw physical data or checkpoint in public inventory")
        if base.sha(path) != expected:
            raise RuntimeError(f"Failed workflow public fingerprint mismatch: {name}")
        pins[str(path)] = expected
    job, status, plan = (base.read(root / name) for name in ("job.json", "status.json", "plan.json"))
    stages = job.get("stages", [])
    expected_names = ["fine_sampler_pilot"] + [row["name"] for row in plan["subsets"]] + ["frozen_evaluation"]
    if (job.get("status") != "failed" or job.get("workflow_complete") is not False
            or job.get("returncode") != 1 or job.get("current") != "frozen_evaluation"
            or ERROR not in job.get("error", "") or len(stages) != 15
            or [row.get("name") for row in stages] != expected_names
            or any(row.get("status") != "complete" or row.get("returncode") != 0 for row in stages[:-1])
            or stages[-1].get("status") != "failed" or stages[-1].get("returncode") != 1
            or not math.isclose(job.get("cumulative_cpu_seconds", -1), PREVIOUS_CPU_SECONDS, abs_tol=1e-8)
            or job.get("plan_sha256") != base.sha(root / "plan.json")
            or Path(plan["output"]).resolve() != root
            or plan.get("resource_policy", {}).get("load_gate_user_override") is not True):
        raise RuntimeError("Prior workflow is not the fixed report-only serialization failure")
    for key in ("status", "workflow_complete", "returncode", "current", "error", "cumulative_cpu_seconds"):
        if status.get(key) != job.get(key):
            raise RuntimeError("Failed job and status disagree")
    failed = base.read(root / "evaluation/status.json")
    if (failed.get("status") != "failed" or failed.get("workflow_complete") is not False
            or failed.get("returncode") != 1 or failed.get("error") != ERROR):
        raise RuntimeError("Evaluation failure is not the specific JSON int64 output error")
    data = base.read(root / "data_manifest.json")
    if (data.get("complete") is not True or data.get("independent_physical_chains") != 156
            or data.get("unique_size_parameter_conditions") != 24 or len(data.get("chains", [])) != 156):
        raise RuntimeError("Cached report must follow all 156 completed production chains")
    return pins, job, plan, data


def build_plan(source_root, data_root, validation, output):
    source_root, data_root, validation, output = map(lambda p: Path(p).resolve(),
                                                   (source_root, data_root, validation, output))
    for frozen in (source_root, data_root, validation):
        if output == frozen or frozen in output.parents or output in frozen.parents:
            raise ValueError("Report completion requires a separate fresh output")
    if output.exists():
        raise FileExistsError("Report completion never overwrites or resumes an existing output")
    pins, job, old_plan, data = verify_failed_root(data_root)
    if (source_root != Path(old_plan["source_root"]).resolve()
            or validation != Path(old_plan["validation"]).resolve()):
        raise RuntimeError("Source and fixed R validation differ from failed workflow")
    pins.update(base.verify_public(validation))
    evaluation_manifest = base.read(data_root / "evaluation/manifest.json")
    validation_manifest, validation_summary = (base.read(validation / name) for name in ("manifest.json", "summary.json"))
    expected_r = {key: validation_summary["final_R_sha256"][key] for key in MODEL_KEYS}
    if (evaluation_manifest.get("fixed_R_sha256") != expected_r
            or evaluation_manifest.get("source_state_sha256") != base.SOURCE_STATE_SHA256
            or validation_manifest.get("source_state_sha256") != base.SOURCE_STATE_SHA256
            or any(evaluation_manifest.get(f"{name}_updates") != 0 for name in ("Q", "C", "R"))
            or evaluation_manifest.get("physical_chains") != {"36": 120, "108": 36}):
        raise RuntimeError("Cached fixed model provenance differs")
    for key, value in (("source_root", source_root), ("data_root", data_root), ("validation", validation)):
        if Path(evaluation_manifest.get(key, "")).resolve() != value:
            raise RuntimeError("Cached evaluation input roots differ")
    for relative in ("run/manifest.json", "run/summary.json", "run/status.json", "data/manifest.json",
                     "run/training_step_3000.pt", "run/evaluation_arrays.npz"):
        path = source_root / relative
        pins[str(path)] = base.sha(path)
    checkpoint = source_root / "run/training_step_3000.pt"
    if pins[str(checkpoint)] != base.SOURCE_CHECKPOINT_SHA256:
        raise RuntimeError("Source checkpoint fingerprint differs")
    if pins[str(source_root / "run/evaluation_arrays.npz")] != SOURCE_TRAIN_EVALUATION_SHA256:
        raise RuntimeError("Original TRAIN evaluation fingerprint differs")
    source_summary, source_status, source_manifest = (base.read(source_root / "run" / name)
        for name in ("summary.json", "status.json", "manifest.json"))
    if (source_summary.get("final_state_sha256") != base.SOURCE_STATE_SHA256
            or source_summary.get("actual_steps") != 3000 or source_status.get("status") != "complete"
            or source_status.get("workflow_complete") is not True or source_status.get("returncode") != 0
            or source_status.get("error") is not None
            or source_status.get("summary_sha256") != pins[str(source_root / "run/summary.json")]
            or source_manifest.get("lattice_sizes") != [36, 12, 4]
            or source_manifest.get("data_manifest_sha256") != pins[str(source_root / "data/manifest.json")]):
        raise RuntimeError("Original source contract differs")
    if any(validation_manifest.get("source_module_sha256", {}).get(key)
           != source_summary.get("final_module_sha256", {}).get(key) for key in ("Q", "C")):
        raise RuntimeError("Fixed Q/C fingerprints differ between source and R validation")
    # Sidecars are public metadata; raw data.npz is deliberately never opened or hashed.
    for row in data["chains"]:
        path = base.inside(data_root, row["sidecar_path"])
        if pins.get(str(path)) != row["sidecar_sha256"]:
            raise RuntimeError("Production chain sidecar is not in verified public inventory")
    code_root = Path(__file__).resolve().parents[1]
    for name in ("experiments/phi4_ru_followup_report_completion.py", "experiments/phi4_ru_followup_evaluate.py",
                 "experiments/phi4_qcr.py", "scripts/report_phi4_qcr.py", "recurrent_rg/phi4_qcr.py",
                 "scripts/run_phi4_ru_validation_followup.py", "scripts/run_phi4_ru_validation_followup_fine.py"):
        path = code_root / name
        pins[str(path)] = base.sha(path)
    return dict(schema=SCHEMA, source_root=str(source_root), data_root=str(data_root), validation=str(validation),
        output=str(output), source_state_sha256=base.SOURCE_STATE_SHA256, fixed_R_sha256=expected_r,
        previous_cumulative_cpu_seconds=job["cumulative_cpu_seconds"], cpu_cap_seconds=CPU_CAP_SECONDS,
        total_cpu_budget_seconds=7200., input_sha256=pins, Q_updates=0, C_updates=0, R_updates=0,
        new_physical_samples=0, model_forward_calls=0, stochastic_encoding_calls=0,
        bootstrap_recomputed=False, load_gate_user_override=True)


def tensor_state_sha(state):
    digest = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous().numpy()
        for token in (key, str(value.dtype), str(value.shape)):
            digest.update(token.encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def decoder_parameters(checkpoint, manifest, source_state_sha):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved.get("step") != 3000 or saved.get("manifest") != manifest:
        raise RuntimeError("Source checkpoint step or manifest differs")
    state = saved["model"]
    if tensor_state_sha(state) != source_state_sha:
        raise RuntimeError("Source state tensor fingerprint differs")
    weight, bias = state["Q.decoder.weight"], state["Q.decoder.bias"]
    if weight.shape != (105, 2) or bias.shape != (105,) or not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
        raise ValueError("Invalid frozen decoder parameters")
    return (torch.nn.functional.normalize(weight, p=2, dim=0, eps=1e-15).detach().cpu().numpy(),
            bias.detach().cpu().numpy())


def load_cached(root, data):
    with np.load(Path(root) / "evaluation/evaluation_arrays.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in ("direct", "feature_target", "labels", "moments", "identities")}
        predictions = {key: {kind: archive[f"{key}_{kind}"].copy() for kind in ("one_step", "rollout")}
                       for key in MODEL_KEYS}
        for key in MODEL_KEYS:
            value = archive[f"{key}_at_chain_mean"]
            if value.shape != (156, 2, 2) or not np.isfinite(value).all():
                raise ValueError("Missing saved R(mean g) measurements")
    expected = dict(direct=(4, 156, 3, 2), feature_target=(4, 156, 3, 105), labels=(156, 3),
                    moments=(156, 4), identities=(156,))
    for key, shape in expected.items():
        if arrays[key].shape != shape or (key != "identities" and not np.isfinite(arrays[key]).all()):
            raise ValueError(f"Invalid cached measurement {key}")
    labels = arrays["labels"]
    if {int(size): int(np.sum(labels[:, 0] == size)) for size in np.unique(labels[:, 0])} != {36: 120, 108: 36}:
        raise ValueError("Physical lattice chain counts differ")
    warning_rows, sidecars = [], {}
    for index, row in enumerate(data["chains"]):
        identity = [row["subset"], row["generator_cell_index"], row["generator_chain_index"]]
        if (evaluator.json.loads(str(arrays["identities"][index])) != identity
                or not np.array_equal(labels[index], [row["size"], row["r"], row["u"]])):
            raise ValueError("Cached physical chain identity/order differs from completed production")
        path = base.inside(root, row["sidecar_path"])
        if str(path) not in sidecars:
            sidecars[str(path)] = base.read(path)
        side = sidecars[str(path)]
        chain = side["chains"][row["generator_chain_index"]]
        if (chain["chain_index"] != row["generator_chain_index"]
                or chain["seed"] != row["physical_chain_seed"]
                or not math.isfinite(chain["production_acceptance"]) or chain["production_acceptance"] <= 0):
            raise ValueError("Cached sidecar chain identity or acceptance differs")
        warning_rows.append(dict(identity=identity, warnings=chain["warnings"],
                                 production_acceptance=chain["production_acceptance"]))
    for key, prediction in predictions.items():
        for kind, shape in (("one_step", (4, 156, 2, 2)), ("rollout", (4, 156, 3, 2))):
            if prediction[kind].shape != shape or not np.isfinite(prediction[kind]).all():
                raise ValueError(f"Invalid cached fixed R prediction {key}/{kind}")
        if not np.array_equal(prediction["rollout"][:, :, 0], arrays["direct"][:, :, 0]):
            raise ValueError("Cached R rollout starts from another measurement")
        if not np.array_equal(prediction["rollout"][:, :, 1], prediction["one_step"][:, :, 0]):
            raise ValueError("Cached one-step and rollout measurements disagree")
    groups = base.read(Path(root) / "evaluation/groups.json")
    if (len(groups) != 24 or sum(group["physical_chains"] for group in groups) != 156
            or {tuple((g["size"], g["r"], g["u"])) for g in groups} != set(map(tuple, np.unique(labels, axis=0)))
            or any(set(group["models"]) != set(MODEL_KEYS) for group in groups)):
        raise ValueError("Saved paired bootstrap groups differ from cached measurements")
    return arrays, predictions, groups, warning_rows


def resource_contract():
    if socket.gethostname() != base.HOST:
        raise RuntimeError("Execution is pinned to thinklab-105-225")
    if any(os.environ.get(name) != "1" for name in base.THREAD_VARIABLES):
        raise RuntimeError("All CPU thread variables must be set to 1 before process launch")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        raise RuntimeError("Report completion must not expose GPUs")
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        os.nice(10 - os.getpriority(os.PRIO_PROCESS, 0))
    snapshot = base.resource_snapshot(dict(resource_policy=dict(minimum_host_available_gib=36.,
        load_per_cpu_max=1.25, io_full_avg10_percent_max=20.)))
    if snapshot["mem_available_bytes"] < 36 * 1024**3 or snapshot["io_full_avg10_percent"] >= 20:
        raise RuntimeError("Memory or I/O admission gate is not satisfied")
    return dict(snapshot, load_gate_user_override=True, actual_nice=os.getpriority(os.PRIO_PROCESS, 0))


def run(source_root, data_root, validation, output):
    cpu, wall = time.process_time(), time.perf_counter()
    resources = resource_contract()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    plan = build_plan(source_root, data_root, validation, output)
    root, old = Path(plan["output"]), Path(plan["data_root"])
    if previous_live_processes(old):
        raise RuntimeError("Original workflow still has live related processes; no files changed")
    def guard():
        used = time.process_time() - cpu
        if used > CPU_CAP_SECONDS or PREVIOUS_CPU_SECONDS + used > 7200:
            raise base.BudgetHold("Report-only CPU envelope exceeded; no resampling or model updates allowed")
        if base.directory_bytes(old) + (base.directory_bytes(root) if root.exists() else 0) > 2 * 1024**3:
            raise base.BudgetHold("Original physical data plus separate report exceeds the fixed 2 GiB envelope")
    guard()
    root.mkdir(parents=True, exist_ok=False)
    def status(state, **more):
        used = time.process_time() - cpu
        base.save(root / "status.json", dict(schema=SCHEMA, status=state, workflow_complete=state == "complete",
            returncode=0 if state == "complete" else 3 if state == "budget_hold" else 1 if state == "failed" else None,
            cpu_seconds=used, cumulative_cpu_seconds=PREVIOUS_CPU_SECONDS + used,
            previous_cumulative_cpu_seconds=PREVIOUS_CPU_SECONDS, wall_seconds=time.perf_counter()-wall,
            gpu_seconds=0., original_workflow_status="failed", **more))
    try:
        status("running", error=None, current="cached_report_completion")
        base.save(root / "manifest.json", dict(plan, resources=resources, prepared_at=base.now()))
        (root / "provenance").mkdir()
        for name in ("job.json", "plan.json", "status.json"):
            shutil.copyfile(old / name, root / "provenance" / ("original_" + name))
        shutil.copyfile(old / "evaluation/status.json", root / "provenance/original_evaluation_status.json")
        source_path = Path(plan["source_root"])
        decoder, bias = decoder_parameters(source_path / "run/training_step_3000.pt",
            base.read(source_path / "run/manifest.json"), plan["source_state_sha256"])
        arrays, predictions, groups, warnings = load_cached(old, base.read(old / "data_manifest.json"))
        with np.load(source_path / "run/evaluation_arrays.npz", allow_pickle=False) as z:
            train = z["train_direct"].copy()
        if train.ndim != 3 or train.shape[1:] != (3, 2) or not np.isfinite(train).all():
            raise ValueError("Invalid original TRAIN target-normalization reference")
        stds = {length: float(np.sqrt(np.var(train[:, k], axis=0, ddof=0).mean()))
                for k, length in enumerate((36, 12, 4))}
        if any(value <= 0 for value in stds.values()):
            raise ValueError("Original TRAIN target std must be positive")
        metrics = {key: evaluator.path_metrics(arrays["direct"], predictions[key], arrays["labels"],
            stds, decoder, bias, arrays["feature_target"]) for key in MODEL_KEYS}
        reconstructed = arrays["direct"] @ decoder.T + bias
        qmetrics = [dict(source_size=length, observed_size=length//3**k, scale=k,
            native_feature_rmse=float(np.sqrt(np.mean((reconstructed[:, arrays["labels"][:, 0] == length, k]
                - arrays["feature_target"][:, arrays["labels"][:, 0] == length, k])**2))),
            qualification="固定Q/decoder对本次局部统计的重构，非物理信息丢失比例或不可约误差")
            for length in (36, 108) for k in range(3)]
        guard()
        shutil.copyfile(old / "evaluation/evaluation_arrays.npz", root / "evaluation_arrays.npz")
        shutil.copyfile(old / "evaluation/groups.json", root / "groups.json")
        evaluator.write_plots(root, groups, predictions, arrays)
        guard()
        for path, expected in plan["input_sha256"].items():
            if base.sha(path) != expected:
                raise RuntimeError(f"Report completion input changed: {path}")
        if previous_live_processes(old):
            raise RuntimeError("Original workflow became active during report completion")
        result = dict(status="complete", workflow_complete=True, returncode=0, report_only_completion=True,
            original_workflow_status="failed", original_failure_preserved=True, source_QCR_steps=3000,
            Q_updates=0, C_updates=0, R_updates=0, new_physical_samples=0, model_forward_calls=0,
            stochastic_encoding_calls=0, frozen_source_state_sha256=plan["source_state_sha256"],
            fixed_R_sha256=plan["fixed_R_sha256"], physical_chains_by_size={"36": 120, "108": 36},
            repeats_per_chain=4, conditional_chain_bootstrap=1000, bootstrap_recomputed=False,
            all_models_and_scales_paired_within_size=True, cross_sizes_independently_sampled=True,
            physical_groups=groups, latent_and_native_feature_metrics=metrics,
            target_stds_by_physical_size=stds, Q_native_feature_reconstruction=qmetrics,
            labels_used_for_evaluation_only=True, new_data_used_for_selection=False,
            inherited_sampling_warnings=warnings, equilibrium_certified=False, new_root_search=False,
            critical_r=None, thermal_nu=None, cpu_seconds=time.process_time()-cpu,
            cumulative_cpu_seconds=PREVIOUS_CPU_SECONDS+time.process_time()-cpu,
            previous_cumulative_cpu_seconds=PREVIOUS_CPU_SECONDS, GPU_seconds=0,
            wall_seconds=time.perf_counter()-wall, input_sha256=plan["input_sha256"],
            limitations=["变化带由上一轮发现、本轮预先固定；不构成盲测认证。",
                "不确定性条件于固定Q/C及已选定R，不覆盖选点、热化或有限尺寸偏差。",
                "闭合与箭头改善不认证物理相边界或不动点谱。",
                "原物理流程以JSON输出错误终止；本独立目录仅补全既有测量报告，不改原失败状态。"])
        base.save(root / "summary.json", result)
        guard()
        base.save(root / "completion_receipt.json", dict(schema=SCHEMA, completed_at=base.now(),
            original_root=str(old), original_workflow_status="failed", original_failure_preserved=True,
            summary_sha256=base.sha(root / "summary.json"), inputs_unchanged=True,
            model_forward_calls=0, new_physical_samples=0, stochastic_encoding_calls=0,
            bootstrap_recomputed=False, cpu_seconds=time.process_time()-cpu,
            cumulative_cpu_seconds=PREVIOUS_CPU_SECONDS+time.process_time()-cpu,
            input_sha256=plan["input_sha256"]))
        status("complete", current=None, error=None, summary_sha256=base.sha(root / "summary.json"))
    except BaseException as error:
        status("budget_hold" if isinstance(error, base.BudgetHold) else "failed", current="cached_report_completion",
            error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        base.save(root / "public_files.json", {str(path.relative_to(root)): base.sha(path)
            for path in root.rglob("*") if path.is_file() and path.name != "public_files.json"
            and path.suffix in (".json", ".npz", ".png")})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "data-root", "validation", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = vars(parser.parse_args(argv))
    execute = args.pop("execute")
    result = run(**args) if execute else build_plan(**args)
    print(evaluator.json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
