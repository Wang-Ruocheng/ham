"""固定 u=1 参考实验的独立续训父程序：保留原结果，精确恢复 3000→6000。"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts import run_gaussian_scalar_qcr as common
from scripts import run_phi4_rg_reference as original


MODULE = "scripts.run_phi4_rg_reference_continue"
SOURCE = "/mnt/nas-new/home/yange/ham/results/phi4_u1_rg_reference_b3_g1_seed91_vm225_20261028"
SOURCE_PLAN_SHA = "e5b446a3e9759e2f83f025df4fb2f302bb13bef313780a7a91bf5fc6d0920ac8"
SOURCE_CHECKPOINT_SHA = "8dcccfe68cd3caebb613de153d531c2380d0f9e655011c01c3eccfc0dcd5650f"
SOURCE_SCIENCE_SHA = {
    "recurrent_rg/phi4_qcr.py": "016dd47a77ee7aa149a1ed65a33e98e35447aeaed904e994030d4eac1c55ed1d",
    "experiments/phi4_qcr.py": "1e71119dfd8041269121e5f3a0cbbe1ae6eeb7e7d4126eb06507e827ec28a2bf",
}
HOST, GPU_INDEX, GPU_UUID = original.HOST, original.GPU_INDEX, original.GPU_UUID
CPU_SECONDS, GPU_SECONDS = 7200, 3600
GPU_PEAK_BYTES, DIRECTORY_BYTES = 8 * 1024**3, 1024**3
SOURCE_STEPS, ADDITIONAL_STEPS, PROFILE_STEPS = 3000, 3000, 20
THREAD_VARIABLES = original.THREAD_VARIABLES
now, read, save, sha = common.now, common.read, common.save, common.sha
source_hashes, unchanged = common.source_hashes, common.unchanged
directory_bytes, cpu_seconds = common.directory_bytes, common.cpu_seconds
resource_snapshot, validate_host = original.resource_snapshot, original.validate_host
process_snapshot = original.process_snapshot


def source_receipt(source):
    """Delayed import keeps planning/CLI help free of CUDA initialization."""
    from experiments.phi4_qcr_continue import source_receipt as verify
    return verify(Path(source).resolve())


def _source_contract(source):
    source = Path(source).resolve()
    receipt = source_receipt(source)
    plan, job = read(source/"plan.json"), read(source/"job.json")
    execution, config = read(source/"execution.json"), read(source/"config.json")
    original._validate_config(config)
    if (sha(source/"plan.json") != SOURCE_PLAN_SHA
            or job.get("plan_sha256") != SOURCE_PLAN_SHA
            or plan.get("config") != config
            or not job.get("workflow_complete") or job.get("status") != "complete"
            or job.get("returncode") != 0 or len(job.get("stages", [])) != 5
            or any(row.get("returncode") != 0 or row.get("status") != "complete"
                   for row in job["stages"])
            or not execution.get("workflow_complete") or execution.get("returncode") != 0):
        raise RuntimeError("Original completed five-stage reference provenance differs")
    if [stage.get("name") for stage in job["stages"]] != ["sampler_preflight", "physical_sampling",
            "training_preflight", "training", "report"]:
        raise RuntimeError("Original five-stage sequence differs")
    public = read(source/"public_files.json")
    if not isinstance(public, dict) or not public:
        raise RuntimeError("Original public whitelist is missing")
    for name, expected in public.items():
        path = (source/name).resolve()
        if (not path.is_relative_to(source) or path.is_symlink() or not path.is_file()
                or path.suffix not in (".json", ".png", ".npz") or sha(path) != expected):
            raise RuntimeError(f"Original public whitelist fingerprint differs: {name}")
    checkpoint = source/"run"/"training_step_3000.pt"
    if sha(checkpoint) != SOURCE_CHECKPOINT_SHA:
        raise RuntimeError("Original step-3000 checkpoint SHA differs")
    old_cpu, old_gpu = execution.get("cpu_seconds"), execution.get("gpu_stage_wall_seconds")
    if any(not isinstance(value, (int, float)) or isinstance(value, bool)
           or not math.isfinite(value) or value < 0 for value in (old_cpu, old_gpu)):
        raise RuntimeError("Original cumulative compute receipt is invalid")
    return dict(receipt=receipt, source=str(source), plan_sha256=SOURCE_PLAN_SHA,
        checkpoint_sha256=SOURCE_CHECKPOINT_SHA, config_sha256=sha(source/"config.json"),
        execution_sha256=sha(source/"execution.json"), job_sha256=sha(source/"job.json"),
        public_files_sha256=sha(source/"public_files.json"), verified_public_files=len(public),
        source_cpu_seconds=float(old_cpu), source_gpu_seconds=float(old_gpu),
        source_directory_bytes=directory_bytes(source), config=config)


def build_plan(root, code_root, source=SOURCE):
    """Read-only source validation and immutable continuation plan; no training."""
    root, code, source = Path(root).resolve(), Path(code_root).resolve(), Path(source).resolve()
    if (root == code or root in code.parents or root == source or root in source.parents
            or source in root.parents):
        raise ValueError("Continuation root must be separate from source and code")
    provenance = _source_contract(source)
    for name, expected in SOURCE_SCIENCE_SHA.items():
        if not (code/name).is_file() or sha(code/name) != expected:
            raise RuntimeError(f"Unchanged original science code differs: {name}")
    if provenance["source_cpu_seconds"] >= CPU_SECONDS or provenance["source_gpu_seconds"] >= GPU_SECONDS:
        raise BudgetHold("Original experiment already exhausted its cumulative budget")
    if provenance["source_directory_bytes"] >= DIRECTORY_BYTES:
        raise BudgetHold("Original plus continuation directory budget already exhausted")
    return dict(schema_version=5, recipe="phi4_u1_rg_reference_exact_continuation_v1",
        prepared_at=now(), code_root=str(code), output=str(root), source=str(source), host=HOST,
        device="cuda:2", gpu_index=GPU_INDEX, gpu_uuid=GPU_UUID,
        source_sha256=source_hashes(code), source_provenance=provenance,
        training=dict(source_steps=SOURCE_STEPS, additional_steps=ADDITIONAL_STEPS,
            total_steps=SOURCE_STEPS+ADDITIONAL_STEPS, profile_steps=PROFILE_STEPS,
            seed=91, learning_rate=.001, system_batch=8, sample_batch=64, information_pairs=128,
            model_and_losses_unchanged=True, restore_model=True, restore_all_three_Adam_states=True,
            restore_training_RNG_states=True,
            global_RNG_semantics="historical checkpoint stores both explicit training streams; constructor global RNG consumption is isolated",
            Q_updates=ADDITIONAL_STEPS, C_updates=ADDITIONAL_STEPS,
            R_updates=ADDITIONAL_STEPS, labels_used=False, resample=False, recalibrate_Q=False,
            profile_is_independent_discarded_execution_check=True, profile_checkpoint_used_for_formal=False,
            closure_objective="all_start_anchor", closure_scale_weights=None),
        resource_policy=dict(cpu_threads=1, nice=10, cpu_seconds=CPU_SECONDS, gpu_seconds=GPU_SECONDS,
            gpu_peak_bytes=GPU_PEAK_BYTES, directory_bytes=DIRECTORY_BYTES,
            count_original_completed_experiment=True, minimum_host_available_gib=36,
            gpu_free_mib=12288, gpu_compute_processes=0, load_per_cpu_max=1.25,
            io_full_avg10_percent_max=20, two_gpu_checks_seconds_apart=10,
            formal_projection_multiplier=1.5, report_reserve_seconds=300,
            gpu_budget_semantics="original GPU ledger plus admission waits plus full profile/formal child wall times",
            occupied_resources="wait for physical GPU2 without sharing, migration, intervention, or automatic retry"),
        scientific_limits=["只续训相同数据、模型和 loss；不重采样、不改参数、不清空 Adam 或随机数状态",
            "原 3000 步独立保留；追加 3000 步，Q/C/R 累计各 6000 步",
            "20 步预检从原状态独立恢复后丢弃，不进入正式参数轨迹",
            "延续连续场专用配方与全部采样警告；不认证平衡、不保证物理不动点或临界指数"])


def prepare(root, *, code_root=None, source=SOURCE):
    validate_host()
    root = Path(root).resolve()
    if root.exists() or root.is_symlink():
        raise FileExistsError("Continuation output is never overwritten or restarted")
    plan = build_plan(root, code_root or Path(__file__).resolve().parents[1], source)
    root.mkdir(parents=True, exist_ok=False)
    (root/"logs").mkdir()
    save(root/"plan.json", plan, exclusive=True)
    save(root/"job.json", dict(status="prepared", workflow_complete=False, current=None,
        stages=[], error=None, requires_user_action=False, returncode=None,
        plan_sha256=sha(root/"plan.json"), public_files=["job.json", "plan.json", "status.json"]), exclusive=True)
    save(root/"status.json", dict(status="prepared", workflow_complete=False, returncode=None), exclusive=True)
    return plan


def load_plan(root, *, verify_source=True):
    root = Path(root).resolve()
    plan, job = read(root/"plan.json"), read(root/"job.json")
    if (sha(root/"plan.json") != job["plan_sha256"] or plan["output"] != str(root)
            or plan["host"] != HOST or plan["device"] != "cuda:2" or plan["gpu_uuid"] != GPU_UUID
            or plan["training"]["additional_steps"] != ADDITIONAL_STEPS
            or plan["training"]["total_steps"] != SOURCE_STEPS+ADDITIONAL_STEPS):
        raise RuntimeError("Frozen continuation plan differs")
    unchanged(plan)
    if verify_source and _source_contract(plan["source"]) != plan["source_provenance"]:
        raise RuntimeError("Original source fingerprints changed; preserve both experiments")
    return plan, job


def cumulative_cpu(plan, elapsed):
    return plan["source_provenance"]["source_cpu_seconds"]+elapsed


def cumulative_gpu(plan, elapsed):
    return plan["source_provenance"]["source_gpu_seconds"]+elapsed


def combined_directory_bytes(root, plan):
    return plan["source_provenance"]["source_directory_bytes"]+directory_bytes(root)


def admission(root, plan, job, stage, gpu):
    started, consecutive = time.monotonic(), 0
    receipt = dict(stage=stage, gpu=gpu, checks=0, status="waiting", started_at=now(),
        minimum_idle_check_separation_seconds=10 if gpu else None)
    job.setdefault("admissions", []).append(receipt)
    while True:
        elapsed = time.monotonic()-started
        if gpu and cumulative_gpu(plan, job.get("gpu_stage_wall_seconds", 0.)+elapsed) >= GPU_SECONDS:
            receipt.update(status="budget_hold", wall_seconds=elapsed, ended_at=now())
            raise BudgetHold("Original plus continuation GPU wait reached cumulative budget")
        row = resource_snapshot(plan)
        receipt.update(checks=receipt["checks"]+1, wall_seconds=elapsed)
        if row["gpu_uuid"] != GPU_UUID:
            raise RuntimeError("Physical GPU2 UUID differs; no migration")
        consecutive = consecutive+1 if row["gpu_ready" if gpu else "host_ready"] else 0
        job.update(status="waiting_resources", current=stage)
        job.setdefault("resource_checks", []).append(row)
        job["resource_checks"] = job["resource_checks"][-60:]
        save(root/"job.json", job)
        save(root/"status.json", dict(status="waiting_resources", workflow_complete=False,
            current=stage, error=None, requires_user_action=False, returncode=None))
        if consecutive >= (2 if gpu else 1):
            receipt.update(status="admitted", wall_seconds=elapsed, admitted_at=now())
            return receipt
        time.sleep(10 if consecutive else 30)


def training_command(root, plan, *, profile, remaining_gpu_seconds=GPU_SECONDS):
    command = [sys.executable, "-u", "-m", "experiments.phi4_qcr_continue",
        "--source", plan["source"], "--output", str(root/("profile_run" if profile else "run")),
        "--device", plan["device"], "--expected-gpu-uuid", GPU_UUID,
        "--additional-steps", str(ADDITIONAL_STEPS), "--max-training-seconds", str(remaining_gpu_seconds),
        "--max-cuda-bytes", str(GPU_PEAK_BYTES)]
    if profile:
        command.extend(["--profile-only", "--profile-steps", str(PROFILE_STEPS)])
    return command


def planned_commands(root, plan):
    return [("training_preflight", training_command(root, plan, profile=True), True),
        ("training", training_command(root, plan, profile=False), True),
        ("report", [sys.executable, "-u", "-m", "scripts.report_phi4_rg_reference_continue",
            "--source", plan["source"], "--run", str(root/"run"), "--output", str(root/"report")], False)]


def validate_training_summary(summary, *, profile):
    expected = SOURCE_STEPS+(PROFILE_STEPS if profile else ADDITIONAL_STEPS)
    uuid = str(summary.get("device", {}).get("uuid", "")).lower().removeprefix("gpu-")
    clocks = summary.get("optimizer_steps", {})
    if (uuid != GPU_UUID.lower().removeprefix("gpu-")
            or summary.get("source_steps") != SOURCE_STEPS
            or summary.get("actual_steps") != expected
            or summary.get("profile_only") is not profile
            or summary.get("labels_used_in_training") is not False
            or summary.get("peak_cuda_bytes", GPU_PEAK_BYTES+1) > GPU_PEAK_BYTES
            or set(clocks) != {"Q", "C", "R"}
            or any(row.get("min") != expected or row.get("max") != expected for row in clocks.values())):
        raise RuntimeError("Exact continuation endpoint/device/optimizer/peak contract differs")
    if not profile and summary.get("frozen_evaluation_verified") is not True:
        raise RuntimeError("Formal frozen evaluation was not verified")


def profile_projection(summary, profile_wall_seconds, gpu_elapsed, cpu_elapsed, plan):
    timing = summary.get("timing", summary)
    initialization = float(timing.get("initialization_seconds", summary.get("initialization_seconds", -1)))
    training = float(timing.get("training_seconds", summary.get("training_seconds", -1)))
    steady = float(summary.get("seconds_per_update", training/PROFILE_STEPS))
    evaluation = float(timing.get("evaluation_seconds", summary.get("evaluation_seconds", 0.)))
    measured = float(timing.get("wall_seconds", initialization+training+evaluation))
    startup = max(0., profile_wall_seconds-measured)
    values = (initialization, training, steady, evaluation, startup, profile_wall_seconds, gpu_elapsed, cpu_elapsed)
    if any(not math.isfinite(value) or value < 0 for value in values) or steady <= 0:
        raise RuntimeError("Invalid exact-state profile timing")
    # The profile deliberately omits scientific evaluation; the original frozen evaluation
    # cost is reserved explicitly and never mistaken for zero-cost final statistics.
    original_summary = read(Path(plan["source"])/"run"/"summary.json")
    original_evaluation = float(original_summary["timing"]["evaluation_seconds"])
    if not math.isfinite(original_evaluation) or original_evaluation < 0:
        raise RuntimeError("Invalid original evaluation timing")
    formal = 1.5*(initialization+startup+ADDITIONAL_STEPS*steady+max(evaluation, original_evaluation))+300.
    projected_gpu = cumulative_gpu(plan, gpu_elapsed+formal+10.)
    projected_cpu = cumulative_cpu(plan, cpu_elapsed+formal+300.)
    return dict(status="complete", profile_is_not_formal_result=True, profile_updates_discarded=PROFILE_STEPS,
        source_steps=SOURCE_STEPS, formal_additional_steps=ADDITIONAL_STEPS, total_formal_steps=6000,
        profile_initialization_seconds=initialization, profile_steady_seconds_per_update=steady,
        profile_process_startup_seconds=startup, original_evaluation_reserve_seconds=original_evaluation,
        conservative_multiplier=1.5, final_writing_reserve_seconds=300,
        remaining_gpu_admission_reserve_seconds=10., projected_formal_gpu_seconds_conservative=formal,
        source_cpu_seconds=plan["source_provenance"]["source_cpu_seconds"],
        source_gpu_seconds=plan["source_provenance"]["source_gpu_seconds"],
        projected_total_cpu_seconds_conservative=projected_cpu,
        projected_total_gpu_seconds_conservative=projected_gpu,
        cpu_budget_seconds=CPU_SECONDS, gpu_budget_seconds=GPU_SECONDS,
        budget_passed=projected_gpu <= GPU_SECONDS and projected_cpu <= CPU_SECONDS,
        projection_is_not_a_wall_time_guarantee=True)


def _wait_child(child, root, job, plan, *, cpu_start, stage_start, gpu_elapsed, gpu):
    while child.poll() is None:
        snapshot = process_snapshot(child.pid)
        job.setdefault("process_resource_checks", []).append(snapshot)
        job["process_resource_checks"] = job["process_resource_checks"][-60:]
        exceeded = (cumulative_cpu(plan, cpu_seconds()-cpu_start+snapshot["cpu_seconds"]) >= CPU_SECONDS
            or combined_directory_bytes(root, plan) > DIRECTORY_BYTES
            or (gpu and cumulative_gpu(plan, gpu_elapsed+time.monotonic()-stage_start) >= GPU_SECONDS))
        save(root/"job.json", job)
        if exceeded:
            # This handle belongs only to the child created by this parent.
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            return child.returncode, True
        time.sleep(1)
    return child.wait(), False


def final_public_files(root):
    files = [root/name for name in ("plan.json", "launch.json", "execution.json", "gpu_preflight.json",
        "source_before.json", "source_after.json", "status.json")]
    files += [p for folder in ("run", "report") for p in (root/folder).rglob("*")
              if p.is_file() and not p.is_symlink() and p.suffix in (".json", ".npz", ".png")]
    # profile_run is execution evidence only; no profile arrays or checkpoints are public.
    files += [root/"profile_run"/name for name in ("manifest.json", "summary.json", "status.json", "history_progress.json")]
    return {str(p.relative_to(root)): sha(p) for p in sorted(set(files)) if p.is_file()}


def execute(root):
    validate_host()
    root = Path(root).resolve()
    plan, job = load_plan(root)
    if job["status"] != "prepared" or job["stages"]:
        raise ValueError("Continuation already executed; no automatic restart or repair")
    save(root/"execution.json", dict(status="starting", parent_pid=os.getpid(), started_at=now(),
        plan_sha256=sha(root/"plan.json")), exclusive=True)
    save(root/"source_before.json", plan["source_provenance"], exclusive=True)
    current_nice = os.getpriority(os.PRIO_PROCESS, 0)
    if current_nice < 10:
        os.nice(10-current_nice)
    started, cpu_start, gpu_elapsed, admission_elapsed = time.monotonic(), cpu_seconds(), 0., 0.
    job.update(parent_pid=os.getpid(), started_at=now(), actual_nice=os.getpriority(os.PRIO_PROCESS, 0))
    try:
        for name, command, gpu in planned_commands(root, plan):
            load_plan(root)
            if (cumulative_cpu(plan, cpu_seconds()-cpu_start) >= CPU_SECONDS
                    or combined_directory_bytes(root, plan) > DIRECTORY_BYTES
                    or (gpu and cumulative_gpu(plan, gpu_elapsed) >= GPU_SECONDS)):
                raise BudgetHold("Original plus continuation resource budget reached")
            tick = time.monotonic()
            try:
                admission(root, plan, job, name, gpu)
            finally:
                waited = time.monotonic()-tick
                if gpu:
                    gpu_elapsed += waited
                    admission_elapsed += waited
                    job.update(gpu_stage_wall_seconds=gpu_elapsed, gpu_admission_wall_seconds=admission_elapsed)
            if gpu:
                remaining = GPU_SECONDS-cumulative_gpu(plan, gpu_elapsed)
                if remaining <= 0:
                    raise BudgetHold("GPU resource wait exhausted cumulative budget")
                command[command.index("--max-training-seconds")+1] = str(remaining)
            if name == "training":
                projection = read(root/"gpu_preflight.json")
                formal_reserve = projection["projected_formal_gpu_seconds_conservative"]
                if (cumulative_gpu(plan, gpu_elapsed+formal_reserve) > GPU_SECONDS
                        or cumulative_cpu(plan, cpu_seconds()-cpu_start+formal_reserve+300.) > CPU_SECONDS):
                    raise BudgetHold("Formal admission wait leaves insufficient conservative continuation budget")
            load_plan(root)
            env = dict(os.environ, PYTHONUNBUFFERED="1", CUDA_DEVICE_ORDER="PCI_BUS_ID",
                MPLCONFIGDIR=str(root/"matplotlib_cache"))
            env.update({key:"1" for key in THREAD_VARIABLES})
            if not gpu:
                env["CUDA_VISIBLE_DEVICES"] = ""
            tick, stage_cpu = time.monotonic(), cpu_seconds()
            with (root/"logs"/f"{name}.log").open("x") as log:
                child = subprocess.Popen(command, cwd=plan["code_root"], env=env,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                stage = dict(name=name, command=command, gpu=gpu, pid=child.pid,
                    status="running", started_at=now(), admission_wall_seconds=waited)
                job["stages"].append(stage)
                job.update(status="running", current=name)
                save(root/"job.json", job)
                save(root/"status.json", dict(status="running", workflow_complete=False,
                    current=name, returncode=None, error=None, requires_user_action=False))
                code, budget_stopped = _wait_child(child, root, job, plan, cpu_start=cpu_start,
                    stage_start=tick, gpu_elapsed=gpu_elapsed, gpu=gpu)
            stage.update(returncode=code, status="complete" if code == 0 else "failed", ended_at=now(),
                wall_seconds=time.monotonic()-tick, cpu_seconds=cpu_seconds()-stage_cpu)
            if gpu:
                gpu_elapsed += stage["wall_seconds"]
            job.update(gpu_stage_wall_seconds=gpu_elapsed, cpu_seconds=cpu_seconds()-cpu_start)
            save(root/"job.json", job)
            if budget_stopped:
                raise BudgetHold("Own continuation child stopped at cumulative budget; outputs preserved")
            if code:
                folder = "profile_run" if name == "training_preflight" else "run" if name == "training" else "report"
                child_status = read(root/folder/"status.json") if (root/folder/"status.json").is_file() else {}
                if child_status.get("status") == "budget_hold":
                    raise BudgetHold(f"{name} reached its fixed remaining budget")
                raise RuntimeError(f"{name} returned {code}; no automatic retry")
            if name in ("training_preflight", "training"):
                profile = name == "training_preflight"
                folder = root/("profile_run" if profile else "run")
                summary = read(folder/"summary.json")
                validate_training_summary(summary, profile=profile)
                if profile:
                    projection = profile_projection(summary, stage["wall_seconds"], gpu_elapsed,
                        cpu_seconds()-cpu_start, plan)
                    save(root/"gpu_preflight.json", projection, exclusive=True)
                    if not projection["budget_passed"]:
                        raise BudgetHold("Exact-state profile projection exceeds original cumulative budget")
            else:
                report_status = read(root/"report"/"status.json")
                if (not report_status.get("workflow_complete")
                        or report_status.get("returncode", 0) != 0):
                    raise RuntimeError("Continuation comparison report incomplete")
            load_plan(root)
        if (cumulative_cpu(plan, cpu_seconds()-cpu_start) > CPU_SECONDS
                or cumulative_gpu(plan, gpu_elapsed) > GPU_SECONDS
                or combined_directory_bytes(root, plan) > DIRECTORY_BYTES):
            raise BudgetHold("Completed continuation exceeded cumulative budget")
        job.update(status="complete", workflow_complete=True, current=None, ended_at=now())
    except Exception as exc:
        job.update(status=exc.status if isinstance(exc, BudgetHold) else "failed", workflow_complete=False,
            error=f"{type(exc).__name__}: {exc}", requires_user_action=True, ended_at=now())
    finally:
        try:
            after = _source_contract(plan["source"])
            save(root/"source_after.json", after, exclusive=True)
            if after != plan["source_provenance"]:
                raise RuntimeError("Immutable original source changed")
        except Exception as exc:
            job.update(status="failed", workflow_complete=False, error=f"{type(exc).__name__}: {exc}",
                requires_user_action=True, ended_at=now())
        job.update(cpu_seconds=cpu_seconds()-cpu_start, gpu_stage_wall_seconds=gpu_elapsed,
            gpu_admission_wall_seconds=admission_elapsed, wall_seconds=time.monotonic()-started,
            directory_bytes=directory_bytes(root),
            cumulative_cpu_seconds=cumulative_cpu(plan, cpu_seconds()-cpu_start),
            cumulative_gpu_seconds=cumulative_gpu(plan, gpu_elapsed),
            cumulative_directory_bytes=combined_directory_bytes(root, plan))
        job["returncode"] = 0 if job["workflow_complete"] else 3 if job["status"].endswith("hold") else 1
        save(root/"status.json", {key:job[key] for key in
            ("status", "workflow_complete", "error", "requires_user_action", "returncode")})
        save(root/"execution.json", dict(status=job["status"], parent_pid=os.getpid(),
            workflow_complete=job["workflow_complete"], returncode=job["returncode"],
            cpu_seconds=job["cpu_seconds"], cumulative_cpu_seconds=job["cumulative_cpu_seconds"],
            gpu_stage_wall_seconds=gpu_elapsed, cumulative_gpu_seconds=job["cumulative_gpu_seconds"],
            gpu_admission_wall_seconds=admission_elapsed, actual_nice=job["actual_nice"], ended_at=now()))
        files = final_public_files(root)
        job["public_files"] = sorted([*files, "job.json", "public_files.json"])
        save(root/"job.json", job)
        files["job.json"] = sha(root/"job.json")
        save(root/"public_files.json", files)
    return job


class BudgetHold(RuntimeError):
    def __init__(self, message, *, status="budget_hold"):
        super().__init__(message)
        self.status = status


def main(argv=None):
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path(SOURCE))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="只读校验续训计划，不训练、不创建目录")
    args = parser.parse_args(argv)
    if args.smoke:
        if args.execute:
            parser.error("--smoke cannot be combined with --execute")
        result = build_plan(args.output, Path(__file__).resolve().parents[1], args.source)
    elif args.execute and args.output.exists():
        result = execute(args.output)
    else:
        result = prepare(args.output, source=args.source)
        if args.execute:
            result = execute(args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    if args.execute and result.get("status") != "complete":
        raise SystemExit(3)
    return result


if __name__ == "__main__":
    main()
