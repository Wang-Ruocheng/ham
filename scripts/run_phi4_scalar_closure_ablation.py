"""固定 C、标量 Q/R 的闭合兼容性成对消融；仅复用统计缓存，完全不使用 GPU。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import resource
import socket
import subprocess
import sys
import time

from scripts.run_gaussian_scalar_qcr import directory_bytes, save, sha, source_hashes


HOST = "thinklab-105-225"
SOURCE = Path("/mnt/nas-new/home/yange/ham/results/phi4_Q_mean_bias_full_QCR_seed91_vm225_20261027/run")
CACHE = Path("/mnt/nas-new/home/yange/ham/results/phi4_Q_frozenC_diagnostic_seed91_vm225_20261027/cache")
OLD_SOURCE = Path("/mnt/nas-new/home/yange/ham/results/phi4_u02_hybrid_QCR_b3_g1_L36_seed91_vm225_20261027/exploratory_training_v1/run")
SOURCE_CHECKPOINT_SHA = "2d8a0081cbcfb685111a913f063656b2e1527624dbde1335bcd691945e836054"
OLD_CHECKPOINT_SHA = "5df6f4d55d97792be13ecf1811eb44b829ba283210cd46148c8d478e46bc2a7e"
SOURCE_STATE_SHA = "1ea4c64ab046b1c2edbb289077e4502425b84698ae5520914402108d66e0e185"
FINAL_C_SHA = "c71b1df615990eb737384cbbbce1d1b6902a588fd774d3d036232bcf0aebcfb8"
DATA_SHA = "588921b87e3f10be9a8c2ce552d70926758b042481a3800a4efc0fa37430e361"
CACHE_ARRAYS_SHA = "81489d978d80b00b67801948ac1e95506d0afbfcb290cb8fc5f572413af60899"
STEPS, CPU_SECONDS, TRAINING_CPU_SECONDS = 3000, 900, 800
DIRECTORY_BYTES = 1024 ** 3
RECIPE = "phi4_scalar_frozenC_paired_Q_closure_parent_v1"


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cpu_seconds():
    own, children = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime


def validate_host():
    if socket.gethostname() != HOST:
        raise RuntimeError(f"This independent CPU-only ablation is pinned to {HOST}")


def cpu_environment():
    """Hide every GPU; no device query, CUDA runtime or MI collection is needed."""
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    return env


def commands(root):
    root = Path(root)
    return [
        ("paired_training", [sys.executable, "-u", "-m", "experiments.phi4_scalar_closure_ablation",
            "--source-run", str(SOURCE), "--cache", str(CACHE),
            "--output", str(root / "training"), "--steps", str(STEPS),
            "--cpu-budget-seconds", str(TRAINING_CPU_SECONDS)], False),
        ("report", [sys.executable, "-u", "-m", "scripts.report_phi4_scalar_closure_ablation",
            "--training", str(root / "training"), "--output", str(root / "report")], False)]


def immutable_inputs():
    """Hash explicit endpoints/statistics only, never physical-field shards."""
    paths = [SOURCE.parent / "job.json", SOURCE.parent / "plan.json"]
    for root in (SOURCE, OLD_SOURCE):
        paths += [root / name for name in
                  ("manifest.json", "summary.json", "status.json", "evaluation_arrays.npz",
                   "training_step_3000.pt")]
    paths += [CACHE / name for name in ("receipt.json", "status.json", "arrays.npz")]
    return {str(path): sha(path) for path in paths}


def validate_inputs(pins):
    if pins[str(SOURCE / "training_step_3000.pt")] != SOURCE_CHECKPOINT_SHA:
        raise RuntimeError("Latest source checkpoint SHA does not match the authorized endpoint")
    if pins[str(OLD_SOURCE / "training_step_3000.pt")] != OLD_CHECKPOINT_SHA:
        raise RuntimeError("Cache's historical source endpoint SHA mismatch")
    if pins[str(CACHE / "arrays.npz")] != CACHE_ARRAYS_SHA:
        raise RuntimeError("Frozen sufficient-statistics cache SHA mismatch")
    parent = read(SOURCE.parent / "job.json")
    if (parent.get("status") != "complete" or parent.get("workflow_complete") is not True
            or parent.get("returncode") != 0
            or any(row.get("returncode") != 0 for row in parent.get("stages", []))):
        raise RuntimeError("The latest source workflow is not complete")
    manifest, summary, status = [read(SOURCE / f"{name}.json")
                                 for name in ("manifest", "summary", "status")]
    if (summary.get("status") != "complete" or summary.get("actual_steps") != 3000
            or summary.get("final_state_sha256") != SOURCE_STATE_SHA
            or summary.get("final_module_sha256", {}).get("C") != FINAL_C_SHA
            or manifest.get("theory_dim") != 1 or manifest.get("lattice_sizes") != [36, 12, 4]
            or manifest.get("data_manifest_sha256") != DATA_SHA
            or status.get("status") != "complete" or status.get("workflow_complete") is not True
            or status.get("returncode") != 0):
        raise RuntimeError("Completed scalar source model/data provenance mismatch")
    for key, name in (("summary_sha256", "summary.json"),
                      ("evaluation_arrays_sha256", "evaluation_arrays.npz")):
        if status.get(key) != pins[str(SOURCE / name)]:
            raise RuntimeError(f"Latest source status receipt mismatch: {name}")
    receipt, cache_status = read(CACHE / "receipt.json"), read(CACHE / "status.json")
    if (receipt.get("status") != "complete"
            or Path(receipt.get("source_run", "")).resolve() != OLD_SOURCE
            or Path(receipt.get("source_checkpoint", "")).resolve() != OLD_SOURCE / "training_step_3000.pt"
            or receipt.get("source_checkpoint_sha256") != OLD_CHECKPOINT_SHA
            or receipt.get("source_manifest_sha256") != pins[str(OLD_SOURCE / "manifest.json")]
            or receipt.get("source_data_manifest_sha256") != DATA_SHA
            or receipt.get("source_state_sha256_after") != receipt.get("source_state_sha256")
            or receipt.get("scale") != manifest.get("scale")
            or receipt.get("arrays_sha256") != CACHE_ARRAYS_SHA
            or cache_status.get("status") != "complete"
            or cache_status.get("workflow_complete") is not True or cache_status.get("returncode") != 0
            or cache_status.get("arrays_sha256") != CACHE_ARRAYS_SHA
            or cache_status.get("receipt_sha256") != pins[str(CACHE / "receipt.json")]):
        raise RuntimeError("Frozen cache receipt or historical source identity mismatch")
    # Actual C tensor identity and exact per-scale evaluation targets are checked
    # again by the CPU trainer before its first optimizer update.
    return dict(latest_source_checkpoint_sha256=SOURCE_CHECKPOINT_SHA,
                historical_cache_source_checkpoint_sha256=OLD_CHECKPOINT_SHA,
                source_C_sha256=FINAL_C_SHA, cache_arrays_sha256=CACHE_ARRAYS_SHA,
                data_manifest_sha256=DATA_SHA, cache_C_identity_requires_tensor_check=True)


def prepare(root):
    validate_host()
    root, code = Path(root).resolve(), Path(__file__).resolve().parents[1]
    if root.exists() or root.is_symlink():
        raise FileExistsError("No overwrite, restart or resume is permitted")
    for protected in (SOURCE.parent, CACHE.parent, OLD_SOURCE.parent, code):
        if root == protected or root in protected.parents or protected in root.parents:
            raise ValueError("Independent output must not contain or sit inside a protected source/code root")
    pins = immutable_inputs()
    provenance = validate_inputs(pins)
    plan = dict(schema_version=5, recipe=RECIPE, output=str(root), code_root=str(code),
        host=HOST, device="cpu", GPU_used=False, prepared_at=now(),
        source_sha256=source_hashes(code), immutable_input_sha256=pins, provenance=provenance,
        source_run=str(SOURCE), reused_cache=str(CACHE), cache_original_source_run=str(OLD_SOURCE),
        authorization="用户同意尝试改善一维尺度闭合；固定相同 C 的 Q/R 成对闭合兼容性消融",
        arms={"A": "源 Q/R warm-start，新 Adam；原 Q 目标与公共防塌缩护栏；R 闭合",
              "B": "与 A 完全配对；Q 额外接收参数冻结 R 的闭合输入导数"},
        training=dict(additional_steps_per_arm=STEPS, source_actual_steps=3000,
            source_steps_are_not_additional_updates=True, seed=91, theory_dim=1,
            lattice_sizes=[36, 12, 4], optimizer="fresh Adam .001 per Q/R and arm",
            C_updates=0, labels_used=False, affine_decoder_unchanged=True,
            closure_objective="all_start_anchor", closure_scale_weights=None,
            guard_floor_ratio=.5, guard_reference="source per-scale TRAIN standard deviations",
            common_guard_scale="initial full TRAIN Q-base objective, frozen once",
            B_closure_alpha="initial full TRAIN Q-base / closure, frozen once",
            actual_C_and_cached_targets_verified_before_updates=True),
        resource_policy=dict(cpu_threads=1, nice=10, cpu_seconds=CPU_SECONDS,
            training_cpu_seconds=TRAINING_CPU_SECONDS, report_reserve_cpu_seconds=100,
            directory_bytes=DIRECTORY_BYTES, minimum_host_available_gib=36,
            load_per_cpu_max=1.25, io_full_avg10_percent_max=20,
            GPU_queries=0, CUDA_VISIBLE_DEVICES="", occupied_resources="wait without intervention"),
        forbidden=["物理采样或补样", "MI 计算", "GPU 使用或查询", "C 更新", "改默认训练配方",
            "改源或旧缓存", "重跑或续跑本输出", "TEST 选模", "PCA 初始化", "新非线性 decoder"],
        scientific_limits=["单训练 seed；36→12→4；复用已有固定 C 统计缓存，不是完整联合 QCR 复现",
            "源采样的低 ESS/热冷警告全部保留，不认证平衡",
            "两臂各额外 3000 次 Q/R 更新，与源完整 QCR 的 3000 步分开记录",
            "闭合改善不能单凭数值根认证相变；critical_r 与 thermal_nu 保持 null"],
        commands=[dict(stage=name, command=cmd, gpu=gpu) for name, cmd, gpu in commands(root)])
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    save(root / "plan.json", plan, exclusive=True)
    save(root / "job.json", dict(status="prepared", workflow_complete=False, current=None,
        stages=[], error=None, requires_user_action=False), exclusive=True)
    return plan


def unchanged(plan):
    if source_hashes(plan["code_root"]) != plan["source_sha256"]:
        raise RuntimeError("Frozen ablation source inventory changed")
    if immutable_inputs() != plan["immutable_input_sha256"]:
        raise RuntimeError("Immutable source/cache fingerprints changed")


def snapshot():
    available = int(re.search(r"MemAvailable:\s+(\d+)", Path("/proc/meminfo").read_text())[1]) * 1024
    io = float(re.search(r"full avg10=([\d.]+)", Path("/proc/pressure/io").read_text())[1])
    cpus, load = os.cpu_count() or 1, os.getloadavg()[0]
    return dict(at=now(), mem_available_bytes=available, cpus=cpus, load1=load,
        io_full_avg10_percent=io, host_ready=available >= 36*1024**3 and load <= 1.25*cpus and io < 20)


def admission(root, job, name):
    while True:
        row = snapshot()
        job.update(status="waiting_resources", current=name)
        job.setdefault("resource_checks", []).append(row)
        job["resource_checks"] = job["resource_checks"][-60:]
        save(root / "job.json", job)
        if row["host_ready"]:
            return
        time.sleep(30)


def validate_training(root):
    root = Path(root)
    summary, status = read(root / "summary.json"), read(root / "status.json")
    if (status.get("status") != "complete" or status.get("workflow_complete") is not True
            or status.get("returncode") != 0 or summary.get("status") != "complete"
            or summary.get("actual_steps") != STEPS
            or summary.get("actual_steps_by_arm") != {"A": STEPS, "B": STEPS}
            or summary.get("frozen_source_verified") is not True
            or summary.get("source_state_sha256_before") != SOURCE_STATE_SHA
            or summary.get("source_state_sha256_after") != SOURCE_STATE_SHA
            or summary.get("source_gradients_none") is not True
            or summary.get("C_updates") != 0 or summary.get("physical_sampling_calls") != 0
            or summary.get("MI_calls") != 0 or summary.get("GPU_used") is not False
            or summary.get("labels_used_in_training") is not False
            or summary.get("default_recipe_modified") is not False
            or summary.get("thermal_nu") is not None or summary.get("critical_r") is not None
            or summary.get("equilibrium_certified") is not False):
        raise RuntimeError("Paired training completion/immutability policy mismatch")
    for key, name in (("summary_sha256", "summary.json"),
                      ("evaluation_arrays_sha256", "evaluation_arrays.npz")):
        if status.get(key) != sha(root / name):
            raise RuntimeError(f"Training completion receipt mismatch: {name}")
    arms = summary.get("arms", {})
    if set(arms) != {"source", "A", "B"}:
        raise RuntimeError("Training must preserve source and both paired arms")
    initial = arms["source"].get("final_module_sha256")
    if not initial or initial.get("C") != FINAL_C_SHA:
        raise RuntimeError("Source C module fingerprint mismatch")
    for name in ("A", "B"):
        arm = arms[name]
        counts = arm.get("optimizer_steps", {})
        if (arm.get("actual_steps") != STEPS or arm.get("source_actual_steps") != 3000
                or arm.get("C_updates") != 0 or arm.get("initial_module_sha256") != initial
                or arm.get("final_module_sha256", {}).get("C") != FINAL_C_SHA
                or any(counts.get(key, {}).get("min") != STEPS
                       or counts.get(key, {}).get("max") != STEPS
                       or counts.get(key, {}).get("parameters", 0) < 1 for key in ("Q", "R"))):
            raise RuntimeError(f"Arm {name} paired initialization/update count mismatch")
    return summary


def public_files(root):
    root, public = Path(root), {}
    for folder in ("training", "report"):
        for path in sorted((root / folder).rglob("*")):
            if path.is_file() and not path.is_symlink() and path.suffix in (".json", ".npz", ".png"):
                public[str(path.relative_to(root))] = dict(sha256=sha(path), bytes=path.stat().st_size)
    return public


def execute(root):
    validate_host()
    root = Path(root).resolve()
    plan, job = read(root / "plan.json"), read(root / "job.json")
    if job.get("status") != "prepared" or job.get("stages"):
        raise RuntimeError("No restart, resume or duplicate launch is permitted")
    expected = [dict(stage=name, command=cmd, gpu=gpu) for name, cmd, gpu in commands(root)]
    if plan.get("output") != str(root) or plan.get("commands") != expected or plan.get("recipe") != RECIPE:
        raise RuntimeError("Frozen output/command policy mismatch")
    with (root / "execution.lock").open("x") as stream:
        stream.write(str(os.getpid()) + "\n")
    started, cpu_start = time.perf_counter(), cpu_seconds()
    # Lower priority exactly once if needed; never increase an existing NI>10.
    actual_nice = os.getpriority(os.PRIO_PROCESS, 0)
    if actual_nice < 10:
        os.nice(10 - actual_nice)
    env = cpu_environment()
    job.update(status="running", parent_pid=os.getpid(), started_at=now())
    save(root / "job.json", job)
    save(root / "launch.json", dict(host=HOST, parent_pid=os.getpid(), at=now(),
        plan_sha256=sha(root / "plan.json"), code_root=plan["code_root"],
        actual_nice=os.getpriority(os.PRIO_PROCESS, 0), cpu_threads=1, GPU_used=False,
        child_thread_environment={key: env[key] for key in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS",
            "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
        resource_policy=plan["resource_policy"], duplicate_launch_permitted=False), exclusive=True)
    try:
        unchanged(plan)
        for name, cmd, _ in commands(root):
            admission(root, job, name)
            unchanged(plan)
            row = dict(name=name, status="running", returncode=None, started_at=now(), command=cmd, gpu=False)
            job["stages"].append(row)
            job.update(status="running", current=name)
            save(root / "job.json", job)
            tick = time.perf_counter()
            with (root / "logs" / (name + ".log")).open("x") as stream:
                child = subprocess.Popen(cmd, cwd=plan["code_root"], env=env,
                    stdout=stream, stderr=subprocess.STDOUT)
                row["pid"] = child.pid
                save(root / "job.json", job)
                rc = child.wait()
            row.update(returncode=rc, status="complete" if rc == 0 else "failed",
                finished_at=now(), wall_seconds=time.perf_counter() - tick)
            job.update(cpu_seconds=cpu_seconds() - cpu_start, directory_bytes=directory_bytes(root))
            save(root / "job.json", job)
            unchanged(plan)
            stage_status_path = root / ("training" if name == "paired_training" else "report") / "status.json"
            stage_status = read(stage_status_path) if stage_status_path.exists() else {}
            if stage_status.get("status") == "budget_hold":
                row["status"] = "budget_hold"
                job.update(status="budget_hold", returncode=rc or 3, requires_user_action=True,
                    error=stage_status.get("error", "CPU 预算门控；保留已有结果，不缩短或重跑"))
                save(root / "job.json", job)
                return 3
            if rc:
                raise RuntimeError(f"{name}: nonzero returncode={rc}; status={stage_status}; no automatic retry")
            if job["cpu_seconds"] > CPU_SECONDS or job["directory_bytes"] > DIRECTORY_BYTES:
                job.update(status="budget_hold", returncode=3, requires_user_action=True,
                    error="成对消融总资源预算达到上限；保留已有结果，不续跑或减步数")
                save(root / "job.json", job)
                return 3
            if name == "paired_training":
                validate_training(root / "training")
            elif stage_status.get("status") != "complete" or stage_status.get("workflow_complete") is not True:
                raise RuntimeError("Report did not emit a complete authoritative status")
        job.update(status="complete", workflow_complete=True, current=None, finished_at=now(),
            returncode=0, error=None, requires_user_action=False,
            cpu_seconds=cpu_seconds() - cpu_start, wall_seconds=time.perf_counter() - started,
            source_fingerprints_unchanged=True, actual_steps_by_arm={"A": STEPS, "B": STEPS},
            source_actual_steps=3000, C_updates=0, GPU_used=False, public_files=public_files(root))
        save(root / "job.json", job)
        return 0
    except Exception as exc:
        job.update(status="failed", workflow_complete=False, error=str(exc), requires_user_action=True,
            finished_at=now(), returncode=1, cpu_seconds=cpu_seconds() - cpu_start)
        save(root / "job.json", job)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.prepare == args.execute:
        parser.error("Choose exactly one of --prepare or --execute")
    if args.prepare:
        prepare(args.output)
        return 0
    return execute(args.output)


if __name__ == "__main__":
    raise SystemExit(main())
