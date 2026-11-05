"""b=2 连续场三项单因素检验父程序；复用冻结源数据，禁止重采样与自动重跑。"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from scripts import run_phi4_b2_closure_comparison as support


MODULE = "scripts.run_phi4_b2_validation"
RECIPE = "phi4_b2_single_factor_validation_parent_v1"
HOST, GPU_INDEX, GPU_UUID = support.HOST, support.GPU_INDEX, support.GPU_UUID
SOURCE = Path("/mnt/nas-new/home/yange/ham/results/phi4_b2_closure_comparison_L32_seed91_vm225_20261027")
SOURCE_PLAN_SHA = "994abc7ae83b62bb661de99111ec0fb2712124e06d3d73a9f167543fc55f3c08"
STEPS, PROFILE_STEPS = 3000, 20
CPU_SECONDS, GPU_SECONDS = 7200., 3600.
GPU_PEAK_BYTES, DIRECTORY_BYTES = 8 * 1024 ** 3, 1024 ** 3
REPORT_RESERVE_SECONDS, ADMISSION_RESERVE_SECONDS = 300., 80.
now, read, save, sha = support.now, support.read, support.save, support.sha
source_hashes, directory_bytes, cpu_seconds = support.source_hashes, support.directory_bytes, support.cpu_seconds
resource_snapshot = support.resource_snapshot


def validate_host():
    if socket.gethostname() != HOST or "CUDA_VISIBLE_DEVICES" in os.environ:
        raise RuntimeError("This plan requires vm225 and unmasked physical GPU numbering")


def safe_file(root, name):
    """Reject absolute, parent-traversal and symlink references before reading."""
    root, relative = Path(root).resolve(), Path(name)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise RuntimeError("Unsafe source/public inventory reference")
    path = root
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise RuntimeError("Symlinks are not allowed in scientific input/output inventories")
    if not path.resolve().is_relative_to(root) or not path.is_file():
        raise RuntimeError("Missing or unsafe scientific input/output file")
    return path


def source_inputs(source=SOURCE):
    """Pin public source receipts, raw-data hashes, checkpoint, and old source code.

    This function does not load models, run inference, or open array members.
    The already completed source is immutable; raw fields remain on the host.
    """
    source = Path(source).resolve()
    if source != SOURCE.resolve():
        raise ValueError("Only the explicitly authorized completed b2 source is allowed")
    job = read(source / "job.json")
    if (job.get("status") != "complete" or job.get("workflow_complete") is not True
            or job.get("error") is not None or len(job.get("stages", [])) != 5
            or any(stage.get("returncode") != 0 for stage in job["stages"])):
        raise RuntimeError("Source workflow is not successfully complete")
    if sha(source / "plan.json") != SOURCE_PLAN_SHA:
        raise RuntimeError("Source identity/plan SHA mismatch")
    paths = {source / "job.json", source / "plan.json"}
    public = job.get("public_files")
    if not isinstance(public, dict) or not public:
        raise RuntimeError("Source public file SHA inventory is missing")
    for name, expected in public.items():
        path = safe_file(source, name)
        if not isinstance(expected, str) or sha(path) != expected:
            raise RuntimeError("Source public file SHA mismatch")
        paths.add(path)
    manifest = read(source / "training/manifest.json")
    summary = read(source / "training/summary.json")
    if (manifest.get("recipe") != "phi4_b2_closure_objective_comparison_v1"
            or manifest.get("lattice_sizes") != [32, 16, 8, 4]
            or manifest.get("labels_used_in_training") is not False
            or summary.get("actual_steps_by_arm") != {"legacy": STEPS, "all_start": STEPS}
            or summary.get("C_Q_pairing_verified") is not True):
        raise RuntimeError("Source representation, pairing, labels or endpoint contract changed")
    data_manifest = safe_file(source, "data/manifest.json")
    if sha(data_manifest) != manifest.get("data_manifest_sha256"):
        raise RuntimeError("Source raw-data manifest differs from completed training")
    data = read(data_manifest)
    options, shards = data.get("options", {}), data.get("shards", [])
    if (data.get("status") != "complete" or options.get("sizes") != [32]
            or options.get("chains") != 4 or options.get("samples") != 128
            or len(shards) != 12 or {row.get("cell_index") for row in shards} != set(range(12))):
        raise RuntimeError("The fixed 12-condition, 48-chain physical dataset changed")
    for row in shards:
        for path_field, hash_field in (("path", "sha256"), ("sidecar", "sidecar_sha256")):
            path = safe_file(source / "data", row[path_field])
            if sha(path) != row[hash_field]:
                raise RuntimeError("Source physical data or sidecar SHA mismatch")
            paths.add(path)
    paths.add(data_manifest)
    paths.add(safe_file(source, "training/all_start/training_step_3000.pt"))
    original = read(source / "plan.json")
    actual_code = source_hashes(original["code_root"])
    if actual_code != original.get("source_sha256"):
        raise RuntimeError("Original source code inventory changed")
    for name, expected in actual_code.items():
        path = safe_file(original["code_root"], name)
        if sha(path) != expected:
            raise RuntimeError("Original source code SHA mismatch")
        paths.add(path)
    return {str(path): sha(path) for path in sorted(paths)}


def unchanged(plan):
    if source_hashes(plan["code_root"]) != plan["source_sha256"]:
        raise RuntimeError("Frozen validation source code changed")
    if source_inputs(plan["source"]) != plan["immutable_inputs"]:
        raise RuntimeError("Frozen completed source/data/checkpoint inputs changed")


def prepare(root, *, source=SOURCE, gpu_index=GPU_INDEX, expected_gpu_uuid=GPU_UUID):
    validate_host()
    support.validate_target(gpu_index, expected_gpu_uuid)
    source, root, code = Path(source).resolve(), Path(root).resolve(), Path(__file__).resolve().parents[1]
    if root.exists() or root.is_symlink():
        raise FileExistsError("Existing validation outputs are never overwritten or resumed")
    for protected in (source, code):
        if root == protected or root in protected.parents or protected in root.parents:
            raise ValueError("Validation output must be independent of protected source/code")
    pins = source_inputs(source)
    quality = read(source / "sampling_quality.json")
    sampling_status = read(source / "data/status.json")
    plan = dict(schema_version=5, recipe=RECIPE, prepared_at=now(), host=HOST,
        source=str(source), source_plan_sha256=SOURCE_PLAN_SHA, output=str(root), code_root=str(code),
        source_sha256=source_hashes(code), immutable_inputs=pins,
        gpu_index=GPU_INDEX, gpu_uuid=GPU_UUID, device=f"cuda:{GPU_INDEX}",
        physical_data=dict(reused=True, resampled=False, source=str(source / "data"),
            conditions=12, independent_physical_chains=48, fields_per_chain=128,
            saved_configurations=6144, dataset_not_duplicated=True,
            warning_count=sampling_status.get("warning_count"), quality_issue_count=len(quality.get("issues", [])),
            quality_warnings_preserved=True, equilibrium_certified=False),
        controls=dict(baseline="completed source all_start at 3000; read-only, not retrained",
            joint_arms=["cubic_decoder", "odd_C"], joint_steps_per_arm=STEPS,
            joint_initialization="fresh matched seed91 Q/C/R; new Adam .001; not source-final warm starts",
            odd_C="only C mean made exactly odd; common baseline-C TRAIN mean bias retained for pairing",
            cubic_decoder="shared cubic decoder only; C and R retain source random initialization policy",
            R_tail="source all_start final R; Q/C frozen; stochastic raw batches and fresh conditional C draws",
            r_tail_steps=STEPS, r_tail_source_steps=3000, fixed_encoding_cache_used=False,
            system_batch=8, sample_batch=64, information_pairs=128, learning_rate=.001,
            seed=91, profile_seed=191, profile_steps_per_arm=PROFILE_STEPS,
            closure_objective="all_start_anchor", closure_scale_weights=None,
            Q_closure_feedback=False, independent_gradients=True, labels_used=False,
            profile_checkpoint_reused=False, test_selected_hyperparameters=False,
            default_recipe_modified=False),
        resource_policy=dict(cpu_threads=1, nice=10, cpu_seconds=CPU_SECONDS, gpu_seconds=GPU_SECONDS,
            gpu_peak_bytes=GPU_PEAK_BYTES, directory_bytes=DIRECTORY_BYTES,
            minimum_host_available_gib=36, gpu_free_mib=12288, gpu_compute_processes=0,
            load_per_cpu_max=1.25, io_full_avg10_percent_max=20, two_gpu_checks_seconds_apart=10,
            projection_multiplier=1.25, report_reserve_seconds=REPORT_RESERVE_SECONDS,
            admission_reserve_seconds=ADMISSION_RESERVE_SECONDS,
            occupied_resources="wait; no interference, migration, renice or restart"),
        authorization="用户要求检验非线性解码、随机原始批次 R-only 收尾、C 对称性三项单因素假设",
        forbidden=["新物理采样", "源文件/源checkpoint/默认配方修改", "训练标签", "TEST选模",
            "固定编码缓存冒充随机原始批次R收尾", "失败自动修复/重跑/迁移", "按预算缩减固定步数"],
        scientific_limits=["沿用源数据全部质量警告，不认证平衡", "单 seed、有限尺寸、末尺度 L4",
            "两项联合训练各只有一个结构因素改变，不把两个改动混合",
            "R-only 收尾额外3000步与源联合3000步分开报告", "不同臂的g坐标不可直接混比",
            "信息项仍为连续场核平滑代理；critical_r=null; thermal_nu=null"])
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    save(root / "plan.json", plan, exclusive=True)
    save(root / "job.json", dict(status="prepared", workflow_complete=False, current=None,
        stages=[], error=None, requires_user_action=False, returncode=None, prepared_at=now()), exclusive=True)
    return plan


def admission(root, plan, job, stage, gpu):
    count = 0
    while True:
        row = resource_snapshot(plan)
        if row["gpu_uuid"] != plan["gpu_uuid"]:
            raise RuntimeError("Physical GPU UUID differs from the frozen plan")
        count = count + 1 if row["gpu_ready" if gpu else "host_ready"] else 0
        job.update(status="waiting_resources", current=stage)
        job.setdefault("resource_checks", []).append(row)
        job["resource_checks"] = job["resource_checks"][-60:]
        save(root / "job.json", job)
        if count >= (2 if gpu else 1):
            return
        time.sleep(10 if count else 30)


def training_command(root, plan, stage, *, remaining_cpu_seconds=CPU_SECONDS, remaining_gpu_seconds=GPU_SECONDS):
    if stage not in ("profile", "joint", "r_tail"):
        raise ValueError("Only the fixed three training stages are allowed")
    return [sys.executable, "-u", "-m", "experiments.phi4_b2_validation",
        "--source", plan["source"], "--output", str(Path(root) / stage), "--stage", stage,
        "--device", plan["device"], "--expected-gpu-uuid", plan["gpu_uuid"],
        "--cpu-budget-seconds", str(remaining_cpu_seconds), "--gpu-budget-seconds", str(remaining_gpu_seconds),
        "--max-cuda-bytes", str(GPU_PEAK_BYTES)]


def commands(root, plan):
    root = Path(root)
    return [(stage, training_command(root, plan, stage), True) for stage in ("profile", "joint", "r_tail")] + [
        ("report", [sys.executable, "-u", "-m", "scripts.report_phi4_b2_validation",
            "--source", plan["source"], "--joint", str(root / "joint"),
            "--r-tail", str(root / "r_tail"), "--output", str(root / "report")], False)]


def profile_projection(profile, *, gpu_elapsed):
    if (profile.get("status") != "complete" or profile.get("workflow_complete") is not True
            or profile.get("returncode") != 0 or profile.get("profile_only") is not True
            or profile.get("steps_per_arm") != PROFILE_STEPS
            or profile.get("formal_checkpoint_reused") is not False):
        raise RuntimeError("Three-control execution profile receipt is incomplete")
    counters = [profile.get(name) for name in
        ("initialization_seconds", "joint_update_seconds", "r_tail_update_seconds", "peak_cuda_bytes")]
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or value < 0 for value in counters):
        raise RuntimeError("Nonfinite or missing profile counters")
    initialization, joint, tail, peak = counters
    if joint <= 0 or tail <= 0 or peak > GPU_PEAK_BYTES:
        raise RuntimeError("Profile throughput or own GPU peak violates the fixed contract")
    # The measured setup includes all three arms. Charge it once to each formal
    # stage, conservatively, instead of multiplying one-time setup by 150.
    joint_projection = 1.25 * (initialization + joint * STEPS / PROFILE_STEPS)
    tail_projection = 1.25 * (initialization + tail * STEPS / PROFILE_STEPS)
    formal = joint_projection + tail_projection
    total = gpu_elapsed + formal + REPORT_RESERVE_SECONDS + ADMISSION_RESERVE_SECONDS
    return dict(status="complete", budget_passed=total <= GPU_SECONDS,
        initialization_seconds=initialization, initialization_charged_formal_times=2,
        measured_joint_update_seconds=joint, measured_r_tail_update_seconds=tail,
        projected_joint_gpu_seconds_conservative=joint_projection,
        projected_r_tail_gpu_seconds_conservative=tail_projection,
        projected_formal_gpu_seconds_conservative=formal,
        actual_accumulated_gpu_stage_seconds=gpu_elapsed,
        projected_total_gpu_seconds_conservative=total, gpu_budget_seconds=GPU_SECONDS,
        throughput_safety_factor=1.25, report_reserve_seconds=REPORT_RESERVE_SECONDS,
        admission_reserve_seconds=ADMISSION_RESERVE_SECONDS, peak_cuda_bytes=peak,
        scientific_result=False, profile_parameters_reused_for_formal=False)


def validate_training_receipt(value, plan, stage):
    uuid = str(value.get("gpu_uuid", "")).lower().removeprefix("gpu-")
    if uuid != plan["gpu_uuid"].lower().removeprefix("gpu-"):
        raise RuntimeError("Actual trainer physical GPU differs from the frozen plan")
    peak = value.get("peak_cuda_bytes")
    if not isinstance(peak, (int, float)) or not math.isfinite(peak) or not 0 <= peak <= GPU_PEAK_BYTES:
        raise RuntimeError("Own GPU peak is missing, nonfinite, or exceeds the fixed cap")
    if stage == "profile":
        return
    if (value.get("status") != "complete" or value.get("actual_steps") != STEPS
            or value.get("labels_used_in_training") is not False
            or value.get("gradient_routing_verified") is not True):
        raise RuntimeError("Formal endpoint, label-free or gradient-routing contract mismatch")
    names = ("cubic_decoder", "odd_C") if stage == "joint" else ("r_tail",)
    expected_steps = {"Q": STEPS, "C": STEPS, "R": STEPS} if stage == "joint" else {"Q": 0, "C": 0, "R": STEPS}
    if set(value.get("arms", {})) != set(names):
        raise RuntimeError("Formal control arm inventory differs from the frozen plan")
    for name in names:
        arm = value["arms"][name]
        counters = arm.get("optimizer_steps", {})
        if (arm.get("actual_steps") != STEPS or set(counters) != set(expected_steps)
                or any(not isinstance(counters[key], dict)
                    or counters[key].get("min") != expected
                    or counters[key].get("max") != expected
                    or not isinstance(counters[key].get("parameters"), int)
                    or counters[key]["parameters"] <= 0 for key, expected in expected_steps.items())):
            raise RuntimeError("Fixed optimizer step counts differ from the frozen control")
    if stage == "r_tail" and (value.get("Q_C_frozen_verified") is not True
            or value.get("stochastic_raw_physical_batches") is not True
            or value.get("fixed_encoding_cache_used") is not False
            or value.get("source_steps") != STEPS):
        raise RuntimeError("R-tail must freeze Q/C and use raw stochastic batches, not fixed encodings")


def final_public_files(root):
    """Whitelist receipts and statistics only; no source/raw arrays or checkpoints."""
    root, result = Path(root).resolve(), {}
    for name in ("plan.json", "launch.json", "execution.json", "gpu_preflight.json"):
        if (root / name).exists():
            path = safe_file(root, name)
            result[name] = sha(path)
    for folder in ("profile", "joint", "r_tail", "report"):
        for path in sorted((root / folder).rglob("*")):
            if path.is_symlink():
                raise RuntimeError("Public outputs must not include symlinks")
            if not path.is_file() or path.suffix not in (".json", ".png", ".npz"):
                continue
            if path.suffix == ".npz" and path.name not in ("evaluation.npz", "evaluation_arrays.npz", "comparison_arrays.npz"):
                raise RuntimeError("Only explicitly named statistical NPZ archives are public")
            result[str(path.relative_to(root))] = sha(path)
    return result


def execute(root):
    validate_host()
    root = Path(root).resolve()
    plan, job = read(root / "plan.json"), read(root / "job.json")
    support.validate_target(plan["gpu_index"], plan["gpu_uuid"])
    if plan.get("host") != HOST or plan.get("recipe") != RECIPE or plan.get("device") != "cuda:2":
        raise RuntimeError("Frozen host, recipe or physical target mismatch")
    if job.get("status") != "prepared" or job.get("stages"):
        raise RuntimeError("Already executed; no automatic restart or resume")
    unchanged(plan)
    execution = dict(status="starting", parent_pid=os.getpid(), host=HOST,
        started_at=now(), plan_sha256=sha(root / "plan.json"))
    save(root / "execution.json", execution, exclusive=True)
    initial_nice = os.getpriority(os.PRIO_PROCESS, 0)
    if initial_nice < 10:
        os.nice(10 - initial_nice)
    start, cpu_start, gpu_elapsed, projection = time.monotonic(), cpu_seconds(), 0., None
    job.update(parent_pid=os.getpid(), started_at=now(), actual_nice=os.getpriority(os.PRIO_PROCESS, 0))
    save(root / "job.json", job)
    try:
        for name, command, gpu in commands(root, plan):
            unchanged(plan)
            cpu_elapsed = cpu_seconds() - cpu_start
            if cpu_elapsed >= CPU_SECONDS or gpu_elapsed >= GPU_SECONDS or directory_bytes(root) > DIRECTORY_BYTES:
                raise TimeoutError("Fixed resource budget reached; no retry or work reduction")
            if projection is not None:
                remaining_projection = ((projection["projected_joint_gpu_seconds_conservative"] if name == "joint" else 0)
                    + (projection["projected_r_tail_gpu_seconds_conservative"] if name in ("joint", "r_tail") else 0))
                if gpu_elapsed + remaining_projection + REPORT_RESERVE_SECONDS + ADMISSION_RESERVE_SECONDS > GPU_SECONDS:
                    raise TimeoutError("Actual GPU use plus conservative remaining stages exceeds the fixed budget")
                if cpu_elapsed + remaining_projection + 600 > CPU_SECONDS:
                    raise TimeoutError("Actual CPU use plus conservative remaining stages exceeds the fixed budget")
            admission(root, plan, job, name, gpu)
            unchanged(plan)
            if gpu:
                command[command.index("--cpu-budget-seconds") + 1] = str(CPU_SECONDS - (cpu_seconds()-cpu_start) - REPORT_RESERVE_SECONDS)
                command[command.index("--gpu-budget-seconds") + 1] = str(GPU_SECONDS-gpu_elapsed-REPORT_RESERVE_SECONDS)
                if float(command[command.index("--cpu-budget-seconds")+1]) <= 0 or float(command[command.index("--gpu-budget-seconds")+1]) <= 0:
                    raise TimeoutError("No remaining training budget after resource admission")
            env = dict(os.environ, PYTHONUNBUFFERED="1", CUDA_DEVICE_ORDER="PCI_BUS_ID",
                OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                NUMBA_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", MPLCONFIGDIR=str(root / "matplotlib_cache"))
            if not gpu:
                env["CUDA_VISIBLE_DEVICES"] = ""
            log_path = root / "logs" / (name + ".log")
            stage_start, stage_cpu = time.monotonic(), cpu_seconds()
            with log_path.open("x") as log:
                child = subprocess.Popen(command, cwd=plan["code_root"], env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                stage = dict(name=name, command=command, gpu=gpu, pid=child.pid, status="running",
                    started_at=now(), log=str(log_path))
                job["stages"].append(stage)
                job.update(status="running", current=name)
                save(root / "job.json", job)
                returncode = child.wait()
            elapsed = time.monotonic()-stage_start
            if gpu:
                gpu_elapsed += elapsed
            stage.update(returncode=returncode, status="complete" if returncode == 0 else "failed",
                ended_at=now(), wall_seconds=elapsed, cpu_seconds=cpu_seconds()-stage_cpu)
            save(root / "job.json", job)
            status = read(root / name / "status.json") if (root / name / "status.json").is_file() else {}
            if returncode:
                if status.get("status") == "budget_hold":
                    raise TimeoutError(name + " reached its fixed budget; outputs preserved, no retry")
                raise RuntimeError(f"{name} returned {returncode}; no automatic repair or retry")
            if (status.get("status") != "complete" or status.get("workflow_complete") is not True
                    or status.get("returncode") != 0):
                raise RuntimeError(name + " has no consistent successful completion receipt")
            unchanged(plan)
            if gpu:
                value = read(root / name / ("profile.json" if name == "profile" else "summary.json"))
                validate_training_receipt(value, plan, name)
                if name == "profile":
                    projection = profile_projection(value, gpu_elapsed=gpu_elapsed)
                    save(root / "gpu_preflight.json", projection, exclusive=True)
                    job["gpu_projection"] = projection
                    if not projection["budget_passed"]:
                        raise TimeoutError("Conservative GPU projection exceeds 3600 seconds; fixed steps unchanged")
                else:
                    job.setdefault("actual_steps", {}).update({name: {key: row["actual_steps"] for key, row in value["arms"].items()}})
            if cpu_seconds()-cpu_start > CPU_SECONDS or gpu_elapsed > GPU_SECONDS or directory_bytes(root) > DIRECTORY_BYTES:
                raise TimeoutError("Actual resource budget exceeded; preserve all completed and running work")
        job.update(status="complete", workflow_complete=True, returncode=0, current=None, ended_at=now())
    except BaseException as error:
        held = isinstance(error, TimeoutError)
        job.update(status="budget_hold" if held else "failed", workflow_complete=False, returncode=3 if held else 1,
            error=f"{type(error).__name__}: {error}", requires_user_action=True, ended_at=now())
        if not held:
            raise
    finally:
        job.update(cpu_seconds=cpu_seconds()-cpu_start, gpu_stage_wall_seconds=gpu_elapsed,
            wall_seconds=time.monotonic()-start, directory_bytes=directory_bytes(root))
        execution.update(status=job["status"], workflow_complete=job["workflow_complete"], returncode=job["returncode"],
            ended_at=now(), cpu_seconds=job["cpu_seconds"], gpu_stage_wall_seconds=gpu_elapsed,
            wall_seconds=job["wall_seconds"], directory_bytes=job["directory_bytes"], actual_nice=job["actual_nice"],
            error=job["error"], requires_user_action=job["requires_user_action"])
        save(root / "execution.json", execution)
        if job["status"] == "complete":
            try:
                job["public_files"] = final_public_files(root)
            except BaseException as error:
                job.update(status="failed", workflow_complete=False, returncode=1, requires_user_action=True,
                    error=f"Public inventory: {type(error).__name__}: {error}")
                execution.update(status="failed", workflow_complete=False, returncode=1,
                    requires_user_action=True, error=job["error"])
                save(root / "execution.json", execution)
                save(root / "job.json", job)
                raise
        save(root / "job.json", job)


def launch(root):
    validate_host()
    root = Path(root).resolve()
    plan, job = read(root / "plan.json"), read(root / "job.json")
    support.validate_target(plan["gpu_index"], plan["gpu_uuid"])
    if job.get("status") != "prepared" or job.get("stages") or (root / "execution.json").exists():
        raise RuntimeError("Already executed; no restart or overwrite")
    unchanged(plan)
    with (root / "launch.json").open("x") as receipt, (root / "logs/parent.log").open("x") as log:
        parent = subprocess.Popen([sys.executable, "-u", "-m", MODULE, "--execute", "--output", str(root)],
            cwd=plan["code_root"], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, env=dict(os.environ, PYTHONUNBUFFERED="1", CUDA_DEVICE_ORDER="PCI_BUS_ID",
                OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                NUMBA_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1"))
        value = dict(status="launched", parent_pid=parent.pid, at=now(), output=str(root),
            source=str(SOURCE), plan_sha256=sha(root / "plan.json"), new_experiment=True, resumed=False)
        receipt.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("prepare", "launch", "execute"):
        modes.add_argument("--" + mode, action="store_true")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, default=GPU_INDEX)
    parser.add_argument("--expected-gpu-uuid", default=GPU_UUID)
    args = parser.parse_args(argv)
    if args.prepare:
        plan = prepare(args.output, source=args.source, gpu_index=args.gpu_index, expected_gpu_uuid=args.expected_gpu_uuid)
        value = dict(status="prepared", output=plan["output"], source_files=len(plan["source_sha256"]), input_files=len(plan["immutable_inputs"]))
    elif args.launch:
        value = launch(args.output)
    else:
        execute(args.output)
        value = read(args.output / "job.json")
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if value["status"] in ("prepared", "launched", "complete") else 3


if __name__ == "__main__":
    sys.exit(main())
