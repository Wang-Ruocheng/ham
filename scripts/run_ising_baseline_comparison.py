"""安全串行执行同数据对齐的二维 Ising 单 seed 基线；不恢复或重试。

只读既有 seed91 数据、配置、checkpoint 与报告。新输出目录、代码和
预算先冻结，独立执行预算 profile，门控通过后才运行五个固定正式臂。
任何失败或预算门控均保留结果和仍运行子进程，不杀进程或迁移 GPU。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time

from scripts.run_phi4_qcr_large import resource_snapshot, source_hashes


HOST = "thinklab-105-225"
DEVICE = "cuda:0"
GPU_UUID = "GPU-5bba87f1-7d32-4dde-fe28-caba4468ab42"
MODULE = "scripts.run_ising_baseline_comparison"
CHILD_MODULE = "experiments.ising_baseline_comparison"
BASE = Path("/mnt/nas-new/home/yange/ham")
REFERENCE = BASE / "results/ising2d_B0_multiseed_dg5_b3_seed91to95_vm225_20261023"
DATA = BASE / "data/ising2d_positiveK_L108_b3_replicas12_seed91.pt"
DATA_SHA = "f0e319a771d220e948c70e97fda8bbd373d46fdbb280211f0cbd29ce9d36de54"
SIDECAR_SHA = "0d2d7591b37aa27757bde1d1d964059d38b64ffb6374df1d3d7724993e801e2a"
CHECKPOINT_SHA = "423689b75e04f453b329d6fd81e64dd811a6389b2196d77fbdb620a543c31511"
METHODS = ("nmcrg", "rsmine", "decimation", "majority", "qcr_c_pca")
UPDATES = 12000
PROFILE_UPDATES = 40
LABEL_EVALUATION_ERROR = "IndexError: too many indices for tensor of dimension 1"
LABEL_PARENT_ERROR = ("RuntimeError: nmcrg returned 1: " + LABEL_EVALUATION_ERROR
                      + "; no retry or automatic repair")
MANUAL_RETRAINING_REASON = "evaluation_label_shape_compatibility_fix"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value, *, exclusive=False):
    path = Path(path)
    serialized = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x") as stream:
            stream.write(serialized)
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w") as stream:
            stream.write(serialized)
        temporary.replace(path)


def stat_identity(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Existing regular file required: {path}")
    identity = path.stat()
    return dict(size=identity.st_size, inode=identity.st_ino,
                mtime_ns=identity.st_mtime_ns, device=identity.st_dev)


def _finite_nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def policy(*, gpu_hours=2., cpu_hours=4., directory_gib=4., peak_gpu_gib=8.):
    values = dict(gpu_hours=gpu_hours, cpu_hours=cpu_hours,
                  directory_gib=directory_gib, peak_gpu_gib=peak_gpu_gib)
    for name, value in values.items():
        if _finite_nonnegative(value, name) <= 0:
            raise ValueError("Budgets must be positive")
    return dict(cpu_threads=2, nice=10, gpu_seconds=float(gpu_hours * 3600),
        cpu_seconds=float(cpu_hours * 3600), directory_bytes=int(directory_gib * 1024**3),
        gpu_peak_bytes=int(peak_gpu_gib * 1024**3), minimum_host_available_gib=36,
        gpu_free_mib=12288, io_full_avg10_percent_max=20, load_per_cpu_max=1.25,
        two_gpu_checks_seconds_apart=10, occupied_resources="wait without intervention",
        gpu_accounting="all profile and arm child wall time, including collection and reports",
        cpu_accounting="parent and reaped child user+system CPU; live child recorded separately",
        report_cpu_threads=1, automatic_retry=False, resume=False,
        resource_wait_seconds=30, stage_poll_seconds=2)


def _host_guard():
    if socket.gethostname() != HOST or "CUDA_VISIBLE_DEVICES" in os.environ:
        raise ValueError("Requires the declared physical host and unmasked CUDA indices")


def _validate_source(config, sidecar, report, metrics, cache):
    if (config.get("schema_version") != 5
            or config.get("data", {}).get("cache") != str(cache)
            or config.get("data", {}).get("physical_model") != "ising2d"
            or config.get("model", {}).get("block_factor") != 3
            or config.get("model", {}).get("theory_dim") != 5
            or config.get("training", {}).get("seed") != 91
            or config.get("training", {}).get("epochs") != UPDATES
            or sidecar.get("data_sha256") != DATA_SHA
            or sidecar.get("B") != 0. or sidecar.get("coupling_names") != ["K"]
            or report.get("status") != "complete" or report.get("checkpoint_epoch") != UPDATES
            or report.get("checkpoint_sha256_before") != CHECKPOINT_SHA
            or report.get("checkpoint_sha256_after") != CHECKPOINT_SHA
            or report.get("checkpoint_unchanged") is not True
            or metrics.get("data_sha256") != DATA_SHA
            or metrics.get("checkpoint_epoch") != UPDATES
            or metrics.get("labels_used_in_training") is not False):
        raise ValueError("Audited data, reference config or completed source report differs")


def _prior_active_processes(root, pids):
    """Read declared PID command lines; a reused unrelated PID is not a run.

    No signals are sent, and no process state is changed.  A permission error
    is not interpreted as proof that an experiment has exited.
    """
    active = []
    for pid in sorted(set(pids)):
        if type(pid) is not int or pid <= 0:
            raise ValueError("Prior parent/profile PID must be a positive integer")
        try:
            command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        except FileNotFoundError:
            continue
        except OSError as error:
            raise ValueError(f"Cannot establish prior process state for PID {pid}") from error
        arguments = [part.decode(errors="replace") for part in command if part]
        if ((MODULE in arguments or CHILD_MODULE in arguments)
                and (str(root) in arguments or str(root / "plan.json") in arguments)):
            active.append(pid)
    return active


def prior_profile_receipt(prior_root, plan):
    """Accept only an exited, profile-only budget hold under identical limits.

    This is provenance for a NEW independent profiling run, never a restart or
    continuation of the old directory.  No old physical tensor is opened.
    """
    requested = Path(prior_root).absolute()
    if requested.is_symlink() or not requested.is_dir():
        raise ValueError("Prior profile must be an existing regular directory")
    prior = requested.resolve()
    if prior == Path(plan["output"]).resolve() or not prior.is_relative_to((BASE / "results").resolve()):
        raise ValueError("Prior profile requires a distinct declared results root")
    files = {name: prior / name for name in (
        "plan.json", "job.json", "profile/profile.json", "profile_receipt.json",
        "launch.json", "execution.json")}
    records = {name: dict(path=str(path), sha256=sha(path), stat=stat_identity(path))
               for name, path in files.items()}
    old = {name: read(path) for name, path in files.items()}
    old_plan, old_job = old["plan.json"], old["job.json"]
    for key in ("experiment", "host", "device", "gpu_uuid", "reference_root", "source_data",
                "data_sha256", "source_data_stat", "sidecar_sha256", "source_config",
                "source_config_sha256", "source_config_dict", "source_sidecar", "source_sidecar_sha256",
                "source_checkpoint", "source_checkpoint_sha256", "source_report", "source_report_sha256",
                "source_metrics", "source_metrics_sha256", "immutable_stat_identity", "methods",
                "seed", "updates", "C_draws", "resource_policy", "profile", "protocol"):
        if old_plan.get(key) != plan.get(key):
            raise ValueError(f"Prior profile differs in frozen authority or policy: {key}")
    plan_hash = records["plan.json"]["sha256"]
    if old_plan.get("output") != str(prior) or old_job.get("plan_sha256") != plan_hash:
        raise ValueError("Prior plan/output fingerprint differs from its job")
    launch_receipt, execution_receipt = old["launch.json"], old["execution.json"]
    for receipt in (launch_receipt, execution_receipt):
        if (receipt.get("plan_sha256") != plan_hash or receipt.get("device") != DEVICE
                or receipt.get("gpu_uuid") != GPU_UUID):
            raise ValueError("Prior launch/execution fingerprint or physical device differs")
    if (launch_receipt.get("status") != "launched" or launch_receipt.get("output") != str(prior)
            or execution_receipt.get("status") != "started"
            or launch_receipt.get("parent_pid") != execution_receipt.get("parent_pid")
            or execution_receipt.get("parent_pid") != old_job.get("parent_pid")):
        raise ValueError("Prior launch/execution parent identity differs")
    stages = old_job.get("stages", [])
    if (old_job.get("status") != "budget_hold" or old_job.get("workflow_complete") is not False
            or old_job.get("requires_user_action") is not True or old_job.get("child_pid") is not None
            or not old_job.get("ended_at") or len(stages) != 1):
        raise ValueError("Only an ended profile-only budget_hold may carry prior cost")
    stage = stages[0]
    if (stage.get("stage") != "profile" or stage.get("name") != "profile"
            or stage.get("method") is not None or stage.get("status") != "complete"
            or type(stage.get("returncode")) is not int or stage["returncode"] != 0
            or stage.get("gpu_stage") is not True or not stage.get("ended_at")):
        raise ValueError("Prior stage is not a completed zero-returncode execution-only profile")
    for path in (prior / "arms", prior / "comparison"):
        if path.exists() or path.is_symlink():
            raise ValueError("Prior root already contains formal arm/comparison work")
    profile = validate_profile(old["profile/profile.json"])
    if (profile.get("scientific_result") is not False
            or old["profile_receipt.json"].get("budget_passed") is not False):
        raise ValueError("Prior profile is not a non-scientific failed budget projection")
    gpu = _finite_nonnegative(old_job.get("gpu_elapsed_seconds"), "prior GPU seconds")
    cpu = _finite_nonnegative(old_job.get("cpu_seconds"), "prior CPU seconds")
    wall = _finite_nonnegative(stage.get("wall_seconds"), "prior profile wall seconds")
    stage_cpu = _finite_nonnegative(stage.get("cpu_seconds"), "prior profile CPU seconds")
    if gpu <= 0 or cpu <= 0 or wall <= 0 or gpu < wall or cpu < stage_cpu:
        raise ValueError("Prior cumulative costs cannot be below its executed profile costs")
    pids = [old_job.get("parent_pid"), stage.get("pid")]
    active = _prior_active_processes(prior, pids)
    if active:
        raise ValueError(f"Prior profile still has running declared processes: {active}")
    for name, path in files.items():
        if records[name]["stat"] != stat_identity(path) or records[name]["sha256"] != sha(path):
            raise ValueError("Prior public receipts changed while being inspected")
    return dict(root=str(prior), public_json=records, profile_only=True, read_only=True,
                carry_in_gpu_seconds=gpu, carry_in_cpu_seconds=cpu,
                reason="New independent timing-only profile after an exited budget hold; no formal arm was started")


def _verify_prior_profile(plan):
    receipt = plan.get("prior_profile_receipt")
    carry = plan.get("budget_carry_in", dict(gpu_seconds=0., cpu_seconds=0.))
    if receipt is None:
        if plan.get("retrain_from_root") is not None:
            # The separate explicitly authorized fresh-retraining validator
            # checks its own provenance and cumulative costs below.
            if plan.get("prior_profile_root") is not None:
                raise RuntimeError("Profile-only continuation and manual retraining are mutually exclusive")
            return
        if plan.get("prior_profile_root") is not None or carry != dict(gpu_seconds=0., cpu_seconds=0.):
            raise RuntimeError("Prior cost requires a pinned, profile-only receipt")
        return
    for row in receipt["public_json"].values():
        if stat_identity(row["path"]) != row["stat"] or sha(row["path"]) != row["sha256"]:
            raise RuntimeError("Pinned prior profile public JSON changed")
    if (plan.get("prior_profile_root") != receipt["root"]
            or carry != dict(gpu_seconds=receipt["carry_in_gpu_seconds"], cpu_seconds=receipt["carry_in_cpu_seconds"])):
        raise RuntimeError("Prior budget carry-in differs from the pinned receipt")
    if prior_profile_receipt(receipt["root"], plan) != receipt:
        raise RuntimeError("Prior profile eligibility or cost changed")


def retraining_receipt(prior_root, plan):
    """Permit only a new, explicitly authorized all-methods-from-scratch run.

    The eligible old failure is the specific evaluation-only one-dimensional
    coupling-label indexing error, AFTER native C and common R both reached
    12000 updates.  Old model files are fingerprinted, never deserialized or
    used for initialization.  Its cumulative job cost already includes any
    earlier execution-only pilot and is carried exactly once.
    """
    requested = Path(prior_root).absolute()
    if requested.is_symlink() or not requested.is_dir():
        raise ValueError("Retraining source must be an existing regular directory")
    prior = requested.resolve()
    if prior == Path(plan["output"]).resolve() or not prior.is_relative_to((BASE / "results").resolve()):
        raise ValueError("Manual retraining requires a new distinct declared results root")
    json_names = ("plan.json", "job.json", "profile/profile.json", "profile_receipt.json",
        "launch.json", "execution.json", "arms/nmcrg/status.json", "arms/nmcrg/manifest.json",
        "arms/nmcrg/history_progress.json", "arms/nmcrg/native_C_history.json", "arms/nmcrg/readout.json")
    model_names = ("arms/nmcrg/native_C.pt", "arms/nmcrg/common_R.pt")
    records = {}
    for name in (*json_names, *model_names):
        path = prior / name
        for ancestor in (path, *path.parents):
            if ancestor == prior:
                break
            if ancestor.is_symlink():
                raise ValueError("Manual retraining provenance must not follow symlinks")
        identity = stat_identity(path)
        if name in model_names and not 0 < identity["size"] <= 16 * 1024**2:
            raise ValueError("Only small saved native C/common R files may be fingerprinted")
        records[name] = dict(path=str(path), sha256=sha(path), stat=identity)
    old = {name: read(prior / name) for name in json_names}
    old_plan, job = old["plan.json"], old["job.json"]
    for key in ("experiment", "host", "device", "gpu_uuid", "reference_root", "source_data",
                "data_sha256", "source_data_stat", "sidecar_sha256", "source_config",
                "source_config_sha256", "source_config_dict", "source_sidecar", "source_sidecar_sha256",
                "source_checkpoint", "source_checkpoint_sha256", "source_report", "source_report_sha256",
                "source_metrics", "source_metrics_sha256", "immutable_stat_identity", "methods",
                "seed", "updates", "C_draws", "resource_policy", "profile", "protocol"):
        if old_plan.get(key) != plan.get(key):
            raise ValueError(f"Retraining source differs in frozen authority or policy: {key}")
    _verify_prior_profile(old_plan)
    plan_hash = records["plan.json"]["sha256"]
    if old_plan.get("output") != str(prior) or job.get("plan_sha256") != plan_hash:
        raise ValueError("Failed source plan/output fingerprint differs from its job")
    launch_receipt, execution = old["launch.json"], old["execution.json"]
    for receipt in (launch_receipt, execution):
        if (receipt.get("plan_sha256") != plan_hash or receipt.get("device") != DEVICE
                or receipt.get("gpu_uuid") != GPU_UUID or receipt.get("parent_pid") != job.get("parent_pid")):
            raise ValueError("Failed source launch/execution fingerprint or parent differs")
    if (launch_receipt.get("status") != "launched" or launch_receipt.get("output") != str(prior)
            or execution.get("status") != "started"):
        raise ValueError("Failed source launch/execution receipt is incomplete")
    stages = job.get("stages", [])
    if (job.get("status") != "failed" or job.get("workflow_complete") is not False
            or job.get("requires_user_action") is not True or job.get("child_pid") is not None
            or job.get("current") != "nmcrg" or job.get("error") != LABEL_PARENT_ERROR
            or not job.get("ended_at") or len(stages) != 2):
        raise ValueError("Manual retraining only accepts the exited NMCRG evaluation-label IndexError")
    pilot, arm = stages
    if (pilot.get("name") != "profile" or pilot.get("stage") != "profile" or pilot.get("method") is not None
            or pilot.get("status") != "complete" or type(pilot.get("returncode")) is not int
            or pilot["returncode"] != 0 or pilot.get("gpu_stage") is not True or not pilot.get("ended_at")
            or arm.get("name") != "nmcrg" or arm.get("stage") != "arm" or arm.get("method") != "nmcrg"
            or arm.get("status") != "failed" or type(arm.get("returncode")) is not int or arm["returncode"] != 1
            or arm.get("child_status") != "failed" or arm.get("child_error") != LABEL_EVALUATION_ERROR
            or arm.get("gpu_stage") is not True or not arm.get("ended_at")):
        raise ValueError("Only completed profile plus the specific failed first NMCRG arm are eligible")
    arms = prior / "arms"
    if {path.name for path in arms.iterdir()} != {"nmcrg"} or (prior / "comparison").exists():
        raise ValueError("Failed source contains other formal arm or comparison work")
    profile = validate_profile(old["profile/profile.json"])
    if profile.get("scientific_result") is not False or old["profile_receipt.json"].get("budget_passed") is not True:
        raise ValueError("Failed source did not pass its completed execution-only profile")
    status = old["arms/nmcrg/status.json"]
    if (status.get("status") != "failed" or status.get("method") != "nmcrg"
            or status.get("error") != LABEL_EVALUATION_ERROR
            or status.get("current") != "evaluation_and_full5D_roots"
            or status.get("C_step") != UPDATES or status.get("R_step") != UPDATES
            or status.get("requires_user_action") is not True):
        raise ValueError("Failed NMCRG must have completed native C/R before the label-only evaluation error")
    manifest = old["arms/nmcrg/manifest.json"]
    if (manifest.get("method") != "nmcrg" or manifest.get("seed") != plan["seed"]
            or manifest.get("data_sha256") != plan["data_sha256"]
            or manifest.get("source_data") != plan["source_data"]
            or manifest.get("labels_used_in_training") is not False
            or manifest.get("C_draws") != plan["C_draws"] or manifest.get("scales") != [108, 36, 12]
            or manifest.get("native_C", {}).get("updates") != UPDATES
            or manifest.get("common_readout", {}).get("R_updates") != UPDATES):
        raise ValueError("Failed source native/common manifest differs from the fixed scientific protocol")
    for name, expected in {
        "native_C": dict(filter_width=9, stride=3, overlapping=True, batch_configurations=50,
            lr=.001, adam_epsilon=1e-8, updates=UPDATES, dtype="float32"),
        "common_readout": dict(dimension=5, R_updates=UPDATES, R_seed=plan["seed"],
            R_milestones=[6000, 9000], gamma=.3, batch_physical_chains=16,
            one_step_weight=5, rollout_weight=5, target_scale_weights=[1, 1],
            coordinate_reduction="sum", dtype="float64", C_or_PCA_gradients_from_R=False),
    }.items():
        if any(manifest[name].get(key) != value for key, value in expected.items()):
            raise ValueError("Failed source manifest omitted or changed a fixed native/readout setting")
    expected_steps = [1, *range(100, UPDATES + 1, 100)]
    history = old["arms/nmcrg/history_progress.json"]
    native_history = old["arms/nmcrg/native_C_history.json"]
    if (not isinstance(history, list) or not isinstance(native_history, list)
            or [row.get("step") for row in history if row.get("stage") == "C"] != expected_steps
            or [row.get("step") for row in history if row.get("stage") == "R"] != expected_steps
            or [row.get("step") for row in native_history] != expected_steps
            or len(history) != 2 * len(expected_steps) or history[:len(native_history)] != native_history):
        raise ValueError("Failed source does not contain complete fixed-endpoint C/R histories")
    readout = old["arms/nmcrg/readout.json"]
    if (not isinstance(readout, dict) or readout.get("PCA", {}).get("fit_split") != "train"
            or readout.get("PCA", {}).get("dimension") != 5
            or readout.get("affine", {}).get("fit_split") != "train"
            or readout.get("PCA", {}).get("labels_used") is not False
            or readout.get("affine", {}).get("labels_used") is not False):
        raise ValueError("Failed source PCA/affine readout is incomplete or not TRAIN-only")
    pca, affine = readout["PCA"], readout["affine"]
    if (pca.get("kind") != "TRAIN_shared_3x3_PCA_v1"
            or len(pca.get("center", [])) != 512 or len(pca.get("basis", [])) != 512
            or any(len(row) != 5 for row in pca.get("basis", []))
            or len(pca.get("singular_values", [])) < 5
            or affine.get("kind") != "TRAIN_affine_16_motif_readout_v1"
            or len(affine.get("weight", [])) != 5 or len(affine.get("bias", [])) != 16
            or any(len(row) != 16 for row in affine.get("weight", []))
            or len(affine.get("singular_values", [])) != 5):
        raise ValueError("Failed source readout arrays are incomplete")
    gpu = _finite_nonnegative(job.get("gpu_elapsed_seconds"), "failed source cumulative GPU seconds")
    cpu = _finite_nonnegative(job.get("cpu_seconds"), "failed source cumulative CPU seconds")
    old_carry = old_plan.get("budget_carry_in", {})
    gpu_lower = old_carry.get("gpu_seconds", 0.) + sum(
        _finite_nonnegative(row.get("wall_seconds"), "executed stage wall seconds") for row in stages)
    cpu_lower = old_carry.get("cpu_seconds", 0.) + sum(
        _finite_nonnegative(row.get("cpu_seconds"), "executed stage CPU seconds") for row in stages)
    if gpu <= 0 or cpu <= 0 or gpu + 1e-6 < gpu_lower or cpu + 1e-6 < cpu_lower:
        raise ValueError("Failed source cumulative cost omitted executed work or earlier profile cost")
    active = _prior_active_processes(prior, [job.get("parent_pid"), pilot.get("pid"), arm.get("pid")])
    if active:
        raise ValueError(f"Failed source still has running declared processes: {active}")
    for row in records.values():
        if row["stat"] != stat_identity(row["path"]) or row["sha256"] != sha(row["path"]):
            raise ValueError("Failed source receipts changed while being inspected")
    return dict(root=str(prior), public_json={name: records[name] for name in json_names},
        model_file_fingerprints={name: records[name] for name in model_names}, read_only=True,
        carry_in_gpu_seconds=gpu, carry_in_cpu_seconds=cpu,
        carry_is_already_cumulative=True, earlier_cost_added_again=False,
        manual_retraining_authorized=True, reason=MANUAL_RETRAINING_REASON,
        all_methods_from_scratch=True, read_failed_training_weights=False,
        prior_state_used_for_initialization=False, resume=False)


def _verify_retraining(plan):
    receipt = plan.get("retraining_receipt")
    if receipt is None:
        if plan.get("retrain_from_root") is not None or plan.get("manual_retraining_authorized", False):
            raise RuntimeError("Manual fresh retraining requires a pinned eligible failure receipt")
        return
    if (plan.get("prior_profile_root") is not None or plan.get("prior_profile_receipt") is not None
            or plan.get("manual_retraining_authorized") is not True
            or plan.get("manual_retraining_reason") != MANUAL_RETRAINING_REASON
            or plan.get("retrain_from_root") != receipt["root"]):
        raise RuntimeError("Manual retraining authorization/provenance is inconsistent")
    for row in (*receipt["public_json"].values(), *receipt["model_file_fingerprints"].values()):
        if stat_identity(row["path"]) != row["stat"] or sha(row["path"]) != row["sha256"]:
            raise RuntimeError("Pinned failed-run receipt or model-file fingerprint changed")
    if (plan.get("budget_carry_in") != dict(gpu_seconds=receipt["carry_in_gpu_seconds"],
                                            cpu_seconds=receipt["carry_in_cpu_seconds"])
            or retraining_receipt(receipt["root"], plan) != receipt):
        raise RuntimeError("Manual retraining eligibility or cumulative carry-in changed")


def _cumulative_cpu(plan, cpu_origin):
    return plan.get("budget_carry_in", {}).get("cpu_seconds", 0.) + cpu_seconds() - cpu_origin


def prepare(root, *, reference_root=None, gpu_hours=2., cpu_hours=4., directory_gib=4., peak_gpu_gib=8.,
            prior_profile_root=None, retrain_from_root=None):
    """Prepare a new root without loading a model or physical tensor payload.

    The large, already audited physical PT is NOT rehashed.  Its original
    public sidecar SHA/data SHA and regular-file stat identity are pinned.
    The small source checkpoint is SHA-checked against the completed report.
    """
    if prior_profile_root is not None and retrain_from_root is not None:
        raise ValueError("prior_profile_root and retrain_from_root are mutually exclusive")
    _host_guard()
    requested_root = Path(root).absolute()
    if requested_root.is_symlink() or requested_root.exists():
        raise FileExistsError("New output required; never overwrite, resume, or retry")
    root = requested_root.resolve()
    if not root.is_relative_to((BASE / "results").resolve()):
        raise ValueError("New baseline output must be under the declared results directory")
    reference = Path(reference_root or REFERENCE).resolve()
    code = Path(__file__).resolve().parents[1]
    cache = Path(DATA)
    files = dict(source_config=reference / "configs/seed91.json",
        source_sidecar=cache.with_suffix(".json"),
        source_checkpoint=reference / "seed91/run/training_step_12000.pt",
        source_report=reference / "seed91/report/report.json",
        source_metrics=reference / "seed91/report/metrics.json")
    identities = {str(cache): stat_identity(cache)}
    identities.update({str(path): stat_identity(path) for path in files.values()})
    hashes = {name + "_sha256": sha(path) for name, path in files.items()}
    if hashes["source_sidecar_sha256"] != SIDECAR_SHA or hashes["source_checkpoint_sha256"] != CHECKPOINT_SHA:
        raise ValueError("Original public sidecar or source checkpoint SHA differs")
    config, sidecar, report, metrics = (read(files[name]) for name in
        ("source_config", "source_sidecar", "source_report", "source_metrics"))
    _validate_source(config, sidecar, report, metrics, cache)
    frozen_sources = source_hashes(code)
    if not frozen_sources:
        raise ValueError("Executable source snapshot cannot be empty")
    plan = dict(schema_version=5, experiment="ising_B0_single_seed_baseline_same_data_adaptations",
        prepared_at=now(), host=HOST, output=str(root), code_root=str(code), source_code_root=str(code),
        device=DEVICE, gpu_uuid=GPU_UUID, source_sha256=frozen_sources, reference_root=str(reference),
        source_data=str(cache), data_sha256=DATA_SHA, source_data_stat=identities[str(cache)],
        sidecar_sha256=SIDECAR_SHA, immutable_stat_identity=identities,
        **{name: str(path) for name, path in files.items()}, **hashes,
        seed=91, updates=UPDATES, C_draws=3, methods=list(METHODS),
        source_config_dict=config, resource_policy=policy(gpu_hours=gpu_hours, cpu_hours=cpu_hours,
            directory_gib=directory_gib, peak_gpu_gib=peak_gpu_gib),
        profile=dict(updates=PROFILE_UPDATES, collection_physical_chains=4, seed=191,
            scientific_result=False, remaining_projection_includes_all_five_arms=True),
        protocol=dict(physical_sampling_started=False, data_reused=True,
            labels_used_in_training=False, training_seed_count=1,
            original_reference_read_only=True, source_QCR_training_started=False,
            methods_are_same_data_adaptations_not_native_paper_reproductions=True,
            C_spatial_support=dict(nmcrg="overlapping 9x9 stride3", rsmine="declared by arm manifest",
                decimation="fixed stride3", majority="fixed 3x3 stride3", qcr_c_pca="original frozen C"),
            scales=[108, 36, 12], R_updates=UPDATES, automatic_retry=False, resume=False,
            no_hyperparameter_or_seed_search=True, thermal_nu=None))
    plan["prior_profile_root"] = None
    plan["prior_profile_receipt"] = None
    plan["budget_carry_in"] = dict(gpu_seconds=0., cpu_seconds=0.)
    plan.update(retrain_from_root=None, retraining_receipt=None,
                manual_retraining_authorized=False, manual_retraining_reason=None)
    if prior_profile_root is not None:
        prior = prior_profile_receipt(prior_profile_root, plan)
        plan.update(prior_profile_root=prior["root"], prior_profile_receipt=prior,
                    budget_carry_in=dict(gpu_seconds=prior["carry_in_gpu_seconds"],
                                         cpu_seconds=prior["carry_in_cpu_seconds"]))
    elif retrain_from_root is not None:
        receipt = retraining_receipt(retrain_from_root, plan)
        plan.update(retrain_from_root=receipt["root"], retraining_receipt=receipt,
                    manual_retraining_authorized=True, manual_retraining_reason=MANUAL_RETRAINING_REASON,
                    all_methods_from_scratch=True, read_failed_training_weights=False,
                    prior_state_used_for_initialization=False, resume=False,
                    budget_carry_in=dict(gpu_seconds=receipt["carry_in_gpu_seconds"],
                                         cpu_seconds=receipt["carry_in_cpu_seconds"]))
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    save(root / "plan.json", plan, exclusive=True)
    save(root / "job.json", dict(status="prepared", workflow_complete=False, current=None,
        stages=[], error=None, requires_user_action=False, prepared_at=now(),
        gpu_elapsed_seconds=plan["budget_carry_in"]["gpu_seconds"],
        cpu_seconds=plan["budget_carry_in"]["cpu_seconds"],
        budget_carry_in=plan["budget_carry_in"], plan_sha256=sha(root / "plan.json")), exclusive=True)
    return plan


def verify(root):
    _host_guard()
    root = Path(root).resolve()
    plan, job = read(root / "plan.json"), read(root / "job.json")
    if (plan.get("host") != HOST or plan.get("device") != DEVICE or plan.get("gpu_uuid") != GPU_UUID
            or plan.get("output") != str(root) or plan.get("methods") != list(METHODS)
            or plan.get("seed") != 91 or plan.get("updates") != UPDATES or plan.get("C_draws") != 3
            or sha(root / "plan.json") != job.get("plan_sha256")):
        raise RuntimeError("Frozen host/device/output/protocol/plan differs")
    if source_hashes(Path(plan["code_root"])) != plan["source_sha256"]:
        raise RuntimeError("Frozen source changed; no automatic repair or deployment")
    for path, identity in plan["immutable_stat_identity"].items():
        if stat_identity(path) != identity:
            raise RuntimeError("Original data/config/checkpoint/report stat identity changed")
    for name in ("source_config", "source_sidecar", "source_report", "source_metrics"):
        if sha(plan[name]) != plan[name + "_sha256"]:
            raise RuntimeError("Original public provenance changed")
    if read(plan["source_config"]) != plan["source_config_dict"]:
        raise RuntimeError("Original reference config changed")
    _verify_prior_profile(plan)
    _verify_retraining(plan)
    carry = plan.get("budget_carry_in", {})
    if (job.get("gpu_elapsed_seconds", 0.) < carry.get("gpu_seconds", 0.)
            or job.get("cpu_seconds", 0.) < carry.get("cpu_seconds", 0.)):
        raise RuntimeError("Cumulative budget accounting dropped prior execution cost")
    return plan, job


def directory_bytes(root):
    total = 0
    for path in Path(root).rglob("*"):
        if path.is_symlink():
            raise RuntimeError("Experiment output must not contain external symlinks")
        if path.is_file():
            total += path.stat().st_size
    return total


def cpu_seconds():
    return sum(row.ru_utime + row.ru_stime for row in
               (resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)))


def live_child_cpu_seconds(pid):
    try:
        text = Path(f"/proc/{int(pid)}/stat").read_text()
        fields = text[text.rfind(")") + 2:].split()
        return sum(float(fields[index]) for index in (11, 12, 13, 14)) / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError):
        return None


def stage_command(root, stage, method=None):
    if (stage not in ("profile", "arm", "compare")
            or (stage == "arm" and method not in METHODS)
            or (stage != "arm" and method is not None)):
        raise ValueError("Unknown frozen stage/method pair")
    root = Path(root)
    command = [sys.executable, "-u", "-m", CHILD_MODULE,
               "--plan", str(root / "plan.json"), "--stage", stage, "--root", str(root)]
    if method is not None:
        command += ["--method", method]
    return command


def stage_folder(root, stage, method=None):
    root = Path(root)
    return root / ("arms/" + method if stage == "arm" else "profile" if stage == "profile" else "comparison")


def admission(root, plan, job, name, *, gpu):
    ready = 0
    while True:
        snapshot = resource_snapshot(plan)
        if snapshot["gpu_uuid"] != GPU_UUID:
            raise RuntimeError("Physical GPU0 UUID differs; no migration")
        ready = ready + 1 if snapshot["gpu_ready" if gpu else "host_ready"] else 0
        job.update(status="waiting_resources", current=name)
        job.setdefault("resource_checks", []).append(snapshot)
        job["resource_checks"] = job["resource_checks"][-60:]
        save(Path(root) / "job.json", job)
        if ready >= (2 if gpu else 1):
            return
        time.sleep(plan["resource_policy"]["two_gpu_checks_seconds_apart"]
                   if ready else plan["resource_policy"]["resource_wait_seconds"])


def validate_profile(profile):
    if profile.get("status") != "complete":
        raise RuntimeError("Profiling did not complete")
    for name in ("projected_gpu_seconds", "projected_cpu_seconds", "peak_gpu_bytes"):
        _finite_nonnegative(profile.get(name), name)
    for name in ("projected_gpu_seconds_by_method", "projected_cpu_seconds_by_method"):
        if name in profile:
            if set(profile[name]) != set(METHODS):
                raise ValueError("Per-method profile projection must cover exactly the five fixed methods")
            for value in profile[name].values():
                _finite_nonnegative(value, name)
            aggregate = "projected_gpu_seconds" if "gpu" in name else "projected_cpu_seconds"
            if sum(profile[name].values()) > profile[aggregate] + 1e-6:
                raise ValueError("Per-method profile exceeds its aggregate projection")
    if "comparison_cpu_seconds" in profile:
        _finite_nonnegative(profile["comparison_cpu_seconds"], "comparison_cpu_seconds")
    return profile


def budget_projection(plan, job, profile, remaining_methods):
    """Gate actual cumulative accounting plus remaining conservative forecast.

    If per-method timing is unavailable the entire aggregate is retained,
    rather than assuming all five fundamentally different methods cost alike.
    """
    validate_profile(profile)
    remaining = tuple(remaining_methods)
    if any(method not in METHODS for method in remaining) or len(set(remaining)) != len(remaining):
        raise ValueError("Invalid remaining-method list")
    gpu_map = profile.get("projected_gpu_seconds_by_method")
    cpu_map = profile.get("projected_cpu_seconds_by_method")
    gpu = (sum(gpu_map[m] for m in remaining) if gpu_map is not None
           else profile["projected_gpu_seconds"] if remaining else 0.)
    cpu = (sum(cpu_map[m] for m in remaining) + profile.get("comparison_cpu_seconds", 0.)
           if cpu_map is not None else profile["projected_cpu_seconds"])
    policy_value = plan["resource_policy"]
    projected_gpu = job.get("gpu_elapsed_seconds", 0.) + gpu
    projected_cpu = job.get("cpu_seconds", 0.) + cpu
    passed = (projected_gpu <= policy_value["gpu_seconds"]
        and projected_cpu <= policy_value["cpu_seconds"]
        and profile["peak_gpu_bytes"] <= policy_value["gpu_peak_bytes"])
    return dict(profile_is_scientific_result=False, at=now(), remaining_methods=list(remaining),
        projected_total_gpu_seconds=projected_gpu, projected_total_cpu_seconds=projected_cpu,
        actual_gpu_seconds=job.get("gpu_elapsed_seconds", 0.), actual_cpu_seconds=job.get("cpu_seconds", 0.),
        fixed_gpu_budget_seconds=policy_value["gpu_seconds"],
        fixed_cpu_budget_seconds=policy_value["cpu_seconds"],
        peak_gpu_bytes=profile["peak_gpu_bytes"], budget_passed=passed)


def _safe_public_path(folder, name):
    relative = Path(name)
    if (relative.is_absolute() or ".." in relative.parts or not relative.parts
            or relative.suffix not in (".json", ".npz", ".png")):
        raise ValueError("Declared public artifact is outside the JSON/NPZ/PNG namespace")
    folder = Path(folder).resolve()
    target = folder / relative
    for path in (target, *target.parents):
        if path == folder:
            break
        if path.is_symlink():
            raise ValueError("Public artifacts must not follow symlinks")
    if not target.resolve().is_relative_to(folder) or not target.is_file():
        raise ValueError("Declared public artifact is missing or outside its stage")
    return target


def public_files(root):
    root = Path(root).resolve()
    candidates = [root / name for name in ("plan.json", "launch.json", "execution.json", "profile_receipt.json")]
    for stage, method in [("profile", None), *(("arm", m) for m in METHODS), ("compare", None)]:
        folder = stage_folder(root, stage, method)
        status_path = folder / ("profile.json" if stage == "profile" else "status.json")
        if folder.is_symlink() or status_path.is_symlink():
            raise ValueError("Public stage receipts must not follow symlinks")
        if status_path.is_file():
            candidates.append(status_path)
            for name in read(status_path).get("public_files", []):
                candidates.append(_safe_public_path(folder, name))
    result = []
    for path in sorted(set(candidates)):
        if path.is_symlink():
            raise ValueError("Public artifacts must not be symlinks")
        if path.is_file():
            result.append(dict(path=str(path.relative_to(root)), sha256=sha(path), bytes=path.stat().st_size))
    return result


def _held(job, reason):
    job.update(status="budget_hold", workflow_complete=False, requires_user_action=True,
               error=reason, ended_at=now())


def _run_stage(root, plan, job, stage, method, cpu_origin):
    name = method if stage == "arm" else stage
    gpu = stage != "compare"
    admission(root, plan, job, name, gpu=gpu)
    command = stage_command(root, stage, method)
    started, cpu_before = time.perf_counter(), cpu_seconds()
    with (root / "logs" / f"{name}.log").open("x") as stream:
        process = subprocess.Popen(command, cwd=plan["code_root"], stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT,
            env=dict(os.environ, OMP_NUM_THREADS=str(2 if gpu else 1),
                MKL_NUM_THREADS=str(2 if gpu else 1), OPENBLAS_NUM_THREADS="1",
                NUMBA_NUM_THREADS="1", PYTHONUNBUFFERED="1", CUDA_DEVICE_ORDER="PCI_BUS_ID"))
        row = dict(name=name, stage=stage, method=method, pid=process.pid, command=command,
            started_at=now(), status="running", returncode=None, gpu_stage=gpu,
            log=str(root / "logs" / f"{name}.log"))
        job["stages"].append(row)
        job.update(status="running", current=name, child_pid=process.pid)
        save(root / "job.json", job)
        while process.poll() is None:
            row["wall_seconds_so_far"] = time.perf_counter() - started
            row["live_child_cpu_seconds"] = live_child_cpu_seconds(process.pid)
            job["cpu_seconds"] = _cumulative_cpu(plan, cpu_origin)
            job["active_gpu_wall_seconds"] = row["wall_seconds_so_far"] if gpu else 0.
            save(root / "job.json", job)
            time.sleep(plan["resource_policy"]["stage_poll_seconds"])
        returncode = process.wait()
    elapsed = time.perf_counter() - started
    row.update(status="complete" if returncode == 0 else "failed", returncode=returncode,
        ended_at=now(), wall_seconds=elapsed, cpu_seconds=cpu_seconds() - cpu_before)
    job["gpu_elapsed_seconds"] += elapsed if gpu else 0.
    job.update(cpu_seconds=_cumulative_cpu(plan, cpu_origin), child_pid=None, active_gpu_wall_seconds=0.)
    save(root / "job.json", job)
    folder = stage_folder(root, stage, method)
    status_path = folder / ("profile.json" if stage == "profile" else "status.json")
    status = read(status_path) if status_path.is_file() else {}
    row.update(child_status=status.get("status"), child_error=status.get("error"),
               status_path=str(status_path))
    if status.get("status") in ("budget_hold", "analysis_budget_hold"):
        row["status"] = "budget_hold"
        _held(job, f"{name} stopped at its declared budget gate; preserved without retry")
        return None
    if returncode:
        raise RuntimeError(f"{name} returned {returncode}: {status.get('error')}; no retry or automatic repair")
    if status.get("status") != "complete":
        raise RuntimeError(f"{name} returned zero without a complete stage receipt")
    if gpu:
        peak = _finite_nonnegative(status.get("peak_gpu_bytes"), "peak_gpu_bytes")
        row["peak_gpu_bytes"] = peak
        if peak > plan["resource_policy"]["gpu_peak_bytes"]:
            _held(job, f"{name} own GPU peak exceeded the frozen limit; no further stage")
            return None
    return status


def execute(root):
    root = Path(root).resolve()
    plan, job = verify(root)
    if job.get("status") != "prepared" or job.get("stages"):
        raise RuntimeError("Only a newly prepared workflow may execute; no resume or retry")
    # Atomic execution claim also protects against two manual parent launches.
    save(root / "execution.json", dict(status="started", at=now(), parent_pid=os.getpid(),
        plan_sha256=sha(root / "plan.json"), device=DEVICE, gpu_uuid=GPU_UUID,
        budget_carry_in=plan.get("budget_carry_in", {})), exclusive=True)
    cpu_origin = cpu_seconds()
    carry = plan.get("budget_carry_in", {})
    job.update(parent_pid=os.getpid(), started_at=now(),
        gpu_elapsed_seconds=carry.get("gpu_seconds", 0.), cpu_seconds=carry.get("cpu_seconds", 0.))
    stages = [("profile", None), *(("arm", method) for method in METHODS), ("compare", None)]
    profile = None
    try:
        target_nice = plan["resource_policy"]["nice"]
        inherited_nice = os.getpriority(os.PRIO_PROCESS, 0)
        if inherited_nice < target_nice:
            os.nice(target_nice - inherited_nice)
        job.update(inherited_nice=inherited_nice, actual_nice=os.getpriority(os.PRIO_PROCESS, 0))
        for index, (stage, method) in enumerate(stages):
            verify(root)
            job["cpu_seconds"] = _cumulative_cpu(plan, cpu_origin)
            policy_value = plan["resource_policy"]
            if (job["gpu_elapsed_seconds"] >= policy_value["gpu_seconds"]
                    or job["cpu_seconds"] >= policy_value["cpu_seconds"]
                    or directory_bytes(root) >= policy_value["directory_bytes"]):
                _held(job, "Fixed cumulative budget reached; no reduced updates or retry")
                break
            if profile is not None:
                remaining = [m for s, m in stages[index:] if s == "arm"]
                projection = budget_projection(plan, job, profile, remaining)
                job["budget_projection"] = projection
                if not projection["budget_passed"]:
                    _held(job, "Actual usage plus remaining conservative forecast exceeds the fixed budget")
                    break
            status = _run_stage(root, plan, job, stage, method, cpu_origin)
            if status is None:
                break
            verify(root)
            if (job["gpu_elapsed_seconds"] > policy_value["gpu_seconds"]
                    or job["cpu_seconds"] > policy_value["cpu_seconds"]
                    or directory_bytes(root) > policy_value["directory_bytes"]):
                _held(job, "Completed child exceeded a fixed cumulative budget; no further stage")
                break
            if stage == "profile":
                profile = validate_profile(status)
                projection = budget_projection(plan, job, profile, METHODS)
                save(root / "profile_receipt.json", projection, exclusive=True)
                if not projection["budget_passed"]:
                    _held(job, "Measured profile exceeds the frozen budget; no formal arm started")
                    break
        else:
            job.update(status="complete", workflow_complete=True, current=None,
                requires_user_action=False, error=None, ended_at=now())
            job["public_files"] = public_files(root)
    except BaseException as error:
        job.update(status="failed", workflow_complete=False, requires_user_action=True,
            error=f"{type(error).__name__}: {error}", ended_at=now(),
            failure_policy="preserve completed outputs and any still-running child; never kill/retry/migrate")
        raise
    finally:
        job["cpu_seconds"] = _cumulative_cpu(plan, cpu_origin)
        try:
            job["directory_bytes"] = directory_bytes(root)
        except RuntimeError as error:
            job.update(status="failed", workflow_complete=False, requires_user_action=True,
                       error=f"Output audit: {error}")
        save(root / "job.json", job)
    return job


def launch(root):
    root = Path(root).resolve()
    plan, job = verify(root)
    if job.get("status") != "prepared" or job.get("stages") or (root / "execution.json").exists():
        raise RuntimeError("Never relaunch an already started workflow")
    with (root / "launch.json").open("x") as receipt, (root / "logs/parent.log").open("x") as stream:
        process = subprocess.Popen([sys.executable, "-u", "-m", MODULE, "--execute", "--root", str(root)],
            cwd=plan["code_root"], stdin=subprocess.DEVNULL, stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True,
            env=dict(os.environ, PYTHONUNBUFFERED="1", CUDA_DEVICE_ORDER="PCI_BUS_ID"))
        value = dict(status="launched", at=now(), parent_pid=process.pid, output=str(root),
            device=DEVICE, gpu_uuid=GPU_UUID, plan_sha256=sha(root / "plan.json"),
            fresh_experiment=True, resumed=False, budget_carry_in=plan.get("budget_carry_in", {}))
        receipt.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for flag in ("prepare", "launch", "execute"):
        mode.add_argument("--" + flag, action="store_true")
    parser.add_argument("--root", "--output", type=Path, required=True)
    parser.add_argument("--gpu-hours", type=float, default=2.)
    parser.add_argument("--cpu-hours", type=float, default=4.)
    parser.add_argument("--directory-gib", type=float, default=4.)
    parser.add_argument("--peak-gpu-gib", type=float, default=8.)
    prior = parser.add_mutually_exclusive_group()
    prior.add_argument("--prior-profile-root", type=Path)
    prior.add_argument("--retrain-from-root", type=Path)
    args = parser.parse_args(argv)
    if args.prepare:
        result = prepare(args.root, gpu_hours=args.gpu_hours, cpu_hours=args.cpu_hours,
            directory_gib=args.directory_gib, peak_gpu_gib=args.peak_gpu_gib,
            prior_profile_root=args.prior_profile_root, retrain_from_root=args.retrain_from_root)
    elif args.launch:
        result = launch(args.root)
    else:
        result = execute(args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
