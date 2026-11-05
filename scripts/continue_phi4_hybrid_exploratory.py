"""Explicit-warning-waiver child job using immutable completed hybrid data only.

This is fresh QCR optimization, not a restart/resume of the held sampling job.
No physical sampling, extra configurations, source checkpoint, or loss change.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from scripts import run_phi4_u02_hybrid_qcr as hybrid


MODULE = "scripts.continue_phi4_hybrid_exploratory"
QUALIFICATION = "exploratory_all_sampling_warnings_retained"
USER_AUTHORIZATION = "允许保留质量警告继续探索性训练"
HOST, GPU_UUID, GPU_INDEX = hybrid.HOST, hybrid.GPU_UUID, hybrid.GPU_INDEX
CPU_SECONDS, GPU_SECONDS = 7200, 1800
DIRECTORY_BYTES = 1024 ** 3
common, legacy = hybrid.common, hybrid.legacy
now, read, save, sha = hybrid.now, hybrid.read, hybrid.save, hybrid.sha
cpu_seconds, directory_bytes = hybrid.cpu_seconds, hybrid.directory_bytes
EXPECTED_CRITICAL_SHA = {
    "experiments/phi4_qcr.py": "96dab0de4b3334091a6792e1a22e0b0d9099722c82cfee10a94e6384c039fdb2",
    "recurrent_rg/phi4_qcr.py": "016dd47a77ee7aa149a1ed65a33e98e35447aeaed904e994030d4eac1c55ed1d",
    "scripts/report_phi4_qcr.py": "77e2701a165419c0dd711b729b43b1b041433007a843f87db13b8338fb565598",
    "scripts/report_phi4_hybrid_qcr.py": "558fb0ffc25d7561c076215fead25153bd74547b22a764be3fe95f672163cec2",
}
OLD_MODULES = {"scripts.run_phi4_u02_hybrid_qcr", "scripts.prepare_phi4_hybrid",
               "experiments.phi4_qcr", "scripts.report_phi4_qcr", "scripts.report_phi4_hybrid_qcr"}


def current_code_root():
    return Path(__file__).resolve().parents[1]


def _inside(root, relative):
    relative = Path(relative)
    unresolved = root / relative
    path = unresolved.resolve()
    if relative.is_absolute() or not path.is_relative_to(root) or unresolved.is_symlink():
        raise ValueError("Source manifest reference escapes its immutable dataset")
    return path


def old_processes(source_root):
    """Inspect actual argv identities, not stale PIDs; never signal processes."""
    output = subprocess.check_output(["ps", "-eo", "pid=,ppid=,args="], text=True)
    relevant = []
    for line in output.splitlines():
        pieces = line.strip().split(None, 2)
        if len(pieces) != 3:
            continue
        pid, parent, command = int(pieces[0]), int(pieces[1]), pieces[2]
        if str(source_root) not in command or not any("-m " + module in command for module in OLD_MODULES):
            continue
        try:
            argv = shlex.split(command)
        except ValueError:
            raise RuntimeError("Cannot safely parse process argv while checking the held job")
        if "-m" not in argv or argv.index("-m") + 1 >= len(argv):
            continue
        module = argv[argv.index("-m") + 1]
        if module not in OLD_MODULES:
            continue
        belongs = any(value == str(source_root) or value.startswith(str(source_root) + "/") for value in argv)
        if belongs:
            relevant.append(dict(pid=pid, ppid=parent, module=module, command=command))
    return relevant


def metadata_fingerprints(source_root, child_root):
    result = {}
    for path in sorted(source_root.rglob("*.json")):
        if path.is_relative_to(child_root):
            continue
        if path.is_symlink():
            raise ValueError("Immutable source metadata must not be symlinks")
        if path.is_file():
            result[str(path.relative_to(source_root))] = sha(path)
    return result


def verify_source(source_root, child_root):
    """Full read-only validation, including actual shard hashes (no arrays loaded)."""
    source_root, child_root = Path(source_root).resolve(), Path(child_root).resolve()
    if not source_root.is_dir() or child_root == source_root or not child_root.is_relative_to(source_root):
        raise ValueError("A new child must be a proper descendant of its held source budget root")
    if child_root.is_relative_to(source_root / "data"):
        raise ValueError("Child output cannot be inside immutable source data")
    plan, job = read(source_root / "plan.json"), read(source_root / "job.json")
    execution, launch = read(source_root / "execution.json"), read(source_root / "launch.json")
    if (plan.get("output") != str(source_root) or plan.get("host") != HOST
            or plan.get("gpu_index") != GPU_INDEX or plan.get("gpu_uuid") != GPU_UUID):
        raise ValueError("Held source physical identity differs from the authorized experiment")
    if job.get("status") != "sampling_quality_hold" or job.get("workflow_complete") is not False:
        raise ValueError("Only the completed-data sampling-quality hold can receive this explicit waiver")
    stages = job.get("stages", [])
    if ([stage.get("name") for stage in stages] != ["sampler_preflight", "physical_sampling"]
            or any(stage.get("returncode") != 0 or stage.get("status") != "complete" or stage.get("gpu") is not False for stage in stages)
            or job.get("gpu_stage_wall_seconds") != 0 or job.get("actual_training_steps", 0) != 0):
        raise ValueError("Source must have exactly two successfully completed CPU-only stages")
    if any((source_root / name).exists() or (source_root / name).is_symlink() for name in ("run", "profile_run")):
        raise ValueError("Held source already has a training/profile directory; no restart or warm start")
    if (execution.get("status") != "sampling_quality_hold" or execution.get("workflow_complete") is not False
            or execution.get("plan_sha256") != sha(source_root / "plan.json")
            or launch.get("plan_sha256") != sha(source_root / "plan.json")
            or execution.get("parent_pid") != job.get("parent_pid")
            or launch.get("parent_pid") != job.get("parent_pid")
            or execution.get("gpu_stage_wall_seconds") != 0):
        raise ValueError("Held parent launch/execution receipts disagree")
    spent = job.get("cpu_seconds")
    if (not isinstance(spent, (int, float)) or not math.isfinite(spent) or not 0 < spent < CPU_SECONDS
            or abs(spent - execution.get("cpu_seconds", -1)) > 1e-8):
        raise ValueError("Held source CPU resource receipt is unavailable or inconsistent")
    processes = old_processes(source_root)
    if processes:
        raise RuntimeError(f"Held source still has actual parent/children; no continuation: {processes}")
    common.unchanged(plan)
    code = current_code_root()
    for relative, expected in plan["source_sha256"].items():
        if not (code / relative).is_file() or sha(code / relative) != expected:
            raise RuntimeError(f"New code snapshot differs from an original frozen source: {relative}")
    for relative, expected in EXPECTED_CRITICAL_SHA.items():
        if plan["source_sha256"].get(relative) != expected or sha(code / relative) != expected:
            raise RuntimeError(f"Frozen model/trainer/report differs from the authorized implementation: {relative}")
    data = source_root / "data"
    manifest, data_status = read(data / "manifest.json"), read(data / "status.json")
    manifest_sha = sha(data / "manifest.json")
    if (manifest.get("status") != "complete" or manifest.get("schema") != "euclidean_phi4_hmc_embedded_wolff_v1"
            or data_status.get("status") != "complete" or not data_status.get("workflow_complete")
            or data_status.get("returncode") != 0 or data_status.get("manifest_sha256") != manifest_sha
            or manifest.get("plan_sha256") != sha(data / "plan.json")
            or read(data / "plan.json") != plan["physical"]):
        raise ValueError("Held hybrid dataset is not completely and consistently frozen")
    options = manifest["options"]
    if (options != plan["physical"]["options"] or options.get("sizes") != [36]
            or options.get("dimension") != 2 or options.get("chains") != 4 or options.get("samples") != 128
            or options.get("u_values") != [.2] or options.get("h") != 0
            or options.get("cluster_probability") != .5 or len(options.get("r_values", [])) != 12):
        raise ValueError("Source must be the authorized 6144-configuration hybrid dataset")
    quality, validity = read(source_root / "sampling_quality.json"), read(source_root / "sampling_validity.json")
    if (quality.get("status") != "sampling_quality_hold" or not quality.get("issues")
            or quality.get("data_manifest_sha256") != manifest_sha or quality.get("training_started") is not False
            or quality.get("GPU_profile_started") is not False or validity.get("status") != "complete"
            or validity.get("severe") != [] or len(quality["issues"]) != job.get("sampling_quality_issue_count")):
        raise ValueError("Quality waiver cannot hide severe invalidity or a changed source quality record")
    shards, cells = {}, set()
    for shard in manifest["shards"]:
        cell = shard["cell_index"]
        if cell in cells or not isinstance(cell, int) or not 0 <= cell < 12 or shard["size"] != 36:
            raise ValueError("Unexpected or duplicate hybrid data shard")
        cells.add(cell)
        path, sidecar_path = _inside(data, shard["path"]), _inside(data, shard["sidecar"])
        if sha(path) != shard["sha256"] or sha(sidecar_path) != shard["sidecar_sha256"]:
            raise RuntimeError("Actual physical shard or sidecar SHA differs from the held manifest")
        sidecar = read(sidecar_path)
        split = "test" if cell % options["test_every"] == options["test_offset"] else "train"
        if (sidecar.get("status") != "complete" or sidecar.get("cell_index") != cell
                or sidecar.get("size") != 36 or sidecar.get("dimension") != 2
                or sidecar.get("r") != options["r_values"][cell] or sidecar.get("u") != .2 or sidecar.get("h") != 0
                or sidecar.get("split") != split or shard.get("split") != split
                or sidecar.get("data_sha256") != shard["sha256"] or len(sidecar.get("chains", [])) != 4):
            raise ValueError("Held sidecar identity or physical split differs from the frozen dataset")
        for chain in sidecar["chains"]:
            acceptance, delta = chain.get("production_acceptance"), chain.get("max_absolute_delta_h")
            if (not isinstance(acceptance, (int, float)) or not math.isfinite(acceptance) or not 0 < acceptance <= 1
                    or not isinstance(delta, (int, float)) or not math.isfinite(delta)):
                raise ValueError("Severe zero-acceptance/nonfinite sampling cannot be waived")
        shards[str(path.relative_to(source_root))] = shard["sha256"]
    if cells != set(range(12)):
        raise ValueError("The 12-condition physical grid is incomplete")
    return dict(source_root=str(source_root), data_root=str(data), data_manifest_sha256=manifest_sha,
        source_job_sha256=sha(source_root / "job.json"), source_quality=quality,
        metadata_sha256=metadata_fingerprints(source_root, child_root), shards_sha256=shards,
        source_code_sha256=plan["source_sha256"], source_code_root=plan["code_root"],
        source_cpu_seconds=spent, source_gpu_seconds=0., configurations=6144,
        actual_source_processes=[], physical_data_loaded=False, physical_shards_hashed=True)


def sources_unchanged(plan, *, full=False):
    root, output = Path(plan["source_root"]), Path(plan["output"])
    expected = plan["source_validation"]
    if metadata_fingerprints(root, output) != expected["metadata_sha256"]:
        raise RuntimeError("Immutable source metadata inventory/SHA changed")
    common.unchanged(read(root / "plan.json"))
    if full:
        for relative, expected_sha in expected["shards_sha256"].items():
            if sha(root / relative) != expected_sha:
                raise RuntimeError("Immutable physical source shard changed")


def prepare(source_root, output):
    preparation_cpu_start = cpu_seconds()
    legacy.validate_host()
    source_root, output = Path(source_root).resolve(), Path(output).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError("Exploratory child output exists; no overwrite/restart")
    validation = verify_source(source_root, output)
    if directory_bytes(source_root) >= DIRECTORY_BYTES:
        raise ValueError("Source directory already reaches its shared 1GiB budget")
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    authorization = dict(schema_version=1, at=now(), authorization_type="explicit_user_sampling_quality_waiver",
        user_authorization=USER_AUTHORIZATION, scientific_qualification=QUALIFICATION,
        source_root=str(source_root), source_data_manifest_sha256=validation["data_manifest_sha256"],
        source_job_sha256=validation["source_job_sha256"],
        source_sampling_quality_sha256=validation["metadata_sha256"]["sampling_quality.json"],
        source_quality_issue_count=len(validation["source_quality"]["issues"]),
        source_quality=validation["source_quality"], source_quality_passed=False,
        source_hold_modified=False, severe_sampling_invalidity_waived=False,
        physical_data_resampled=False, loss_or_model_modified=False)
    save(output / "authorization.json", authorization, exclusive=True)
    plan = dict(schema_version=5, recipe="phi4_hybrid_exploratory_training_v1", prepared_at=now(),
        code_root=str(current_code_root()), source_sha256=common.source_hashes(current_code_root()),
        output=str(output), source_root=str(source_root), data_root=validation["data_root"],
        host=HOST, gpu_index=GPU_INDEX, device=f"cuda:{GPU_INDEX}", gpu_uuid=GPU_UUID,
        scientific_qualification=QUALIFICATION, source_quality_status="sampling_quality_hold", source_quality_passed=False,
        source_data_manifest_sha256=validation["data_manifest_sha256"], source_validation=validation,
        authorization_file="authorization.json", authorization_sha256=sha(output / "authorization.json"),
        training=dict(steps=3000, profile_steps=20, seed=91, profile_seed=191, theory_dim=1,
            hidden=96, lattice_sizes=[36, 12, 4], system_batch=8, sample_batch=64,
            learning_rate=.001, optimizer="three independent fresh Adam", closure_objective="all_start_anchor",
            closure_scale_weights=None, labels_used=False, warm_start=False, source_checkpoint_loaded=False),
        resource_policy=dict(cpu_seconds=CPU_SECONDS, source_cpu_seconds=validation["source_cpu_seconds"],
            preparation_cpu_seconds=cpu_seconds()-preparation_cpu_start,
            remaining_cpu_seconds=CPU_SECONDS-validation["source_cpu_seconds"]-(cpu_seconds()-preparation_cpu_start), gpu_seconds=GPU_SECONDS,
            source_gpu_seconds=0., gpu_peak_bytes=8*1024**3, directory_bytes=DIRECTORY_BYTES,
            directory_budget_root=str(source_root), cpu_threads=1, nice=10,
            additional_physical_force_site_evaluations=0, additional_physical_samples=0),
        limitations=["全部采样质量警告保留，原门控仍为 sampling_quality_hold；本次仅获准探索性训练",
            "质量问题条目可能与既有 warning 重复，不代表彼此独立的物理异常",
            "QCR loss/低闭合误差不能消除采样偏差或认证热化；thermal_nu=null",
            "不额外采样、补样、选 seed 或修改模型/loss；profile 不用于正式初始化"])
    save(output / "plan.json", plan, exclusive=True)
    save(output / "job.json", dict(status="prepared", workflow_complete=False, current=None, stages=[],
        scientific_qualification=QUALIFICATION, source_quality_passed=False, source_root=str(source_root),
        source_data_manifest_sha256=validation["data_manifest_sha256"], source_cpu_seconds=validation["source_cpu_seconds"],
        additional_physical_force_site_evaluations=0, error=None, requires_user_action=False), exclusive=True)
    return plan


def _load_plan(source_root, output):
    plan = read(Path(output) / "plan.json")
    if str(Path(source_root).resolve()) != plan["source_root"] or str(Path(output).resolve()) != plan["output"]:
        raise ValueError("Requested held source/child differs from the frozen child plan")
    if plan.get("scientific_qualification") != QUALIFICATION or plan.get("source_quality_passed") is not False:
        raise ValueError("Exploratory quality waiver must not be relabeled as quality passing")
    if sha(Path(output) / plan["authorization_file"]) != plan["authorization_sha256"]:
        raise RuntimeError("Explicit warning-waiver authorization changed")
    return plan


def commands(plan):
    root, data = Path(plan["output"]), plan["data_root"]
    result = []
    for name, profile in (("training_preflight", True), ("training", False)):
        command = hybrid.training_command(root, plan, profile=profile)
        command[command.index("--data")+1] = data
        result.append((name, command, True))
    result.extend([
        ("report", [sys.executable, "-u", "-m", "scripts.report_phi4_qcr", "--run", str(root / "run"), "--output", str(root / "report")], False),
        ("phi4_hybrid_report", [sys.executable, "-u", "-m", "scripts.report_phi4_hybrid_qcr", "--data", data,
            "--run", str(root / "run"), "--output", str(root / "calibration_report")], False),
        ("exploratory_qualification_report", [sys.executable, "-u", "-m", "scripts.report_phi4_hybrid_exploratory",
            "--root", str(root), "--output", str(root / "qualification_report")], False)])
    return result


def execute(source_root, output):
    start, cpu_start = time.monotonic(), cpu_seconds()
    legacy.validate_host()
    output = Path(output).resolve()
    plan = _load_plan(source_root, output)
    job = read(output / "job.json")
    if job["status"] != "prepared" or job["stages"]:
        raise ValueError("Exploratory child already executed; no restart")
    verify_source(Path(plan["source_root"]), output)
    sources_unchanged(plan, full=True)
    common.unchanged(plan)
    execution = dict(status="starting", parent_pid=os.getpid(), host=HOST, started_at=now(),
        plan_sha256=sha(output / "plan.json"), scientific_qualification=QUALIFICATION,
        source_quality_passed=False, source_root=plan["source_root"], additional_physical_force_site_evaluations=0)
    save(output / "execution.json", execution, exclusive=True)
    priority = os.getpriority(os.PRIO_PROCESS, 0)
    if priority < 10:
        os.nice(10-priority)
    gpu = 0.
    launch_path = output / "launch.json"
    launch_cpu = read(launch_path).get("orchestration_cpu_seconds", 0.) if launch_path.is_file() else 0.
    preparation_cpu = plan["resource_policy"].get("preparation_cpu_seconds", 0.)
    prior_cpu = plan["resource_policy"]["source_cpu_seconds"] + preparation_cpu + launch_cpu
    job.update(parent_pid=os.getpid(), host=HOST, started_at=now(), actual_nice=os.getpriority(os.PRIO_PROCESS, 0))
    try:
        for name, command, needs_gpu in commands(plan):
            common.unchanged(plan)
            sources_unchanged(plan)
            if directory_bytes(Path(plan["source_root"])) > DIRECTORY_BYTES or prior_cpu+cpu_seconds()-cpu_start >= CPU_SECONDS:
                legacy.hold(job, "Original source plus exploratory child reached shared directory/CPU budget; no deletion or reduced steps")
                break
            hybrid.admission(output, plan, job, name, needs_gpu)
            if needs_gpu:
                if gpu >= GPU_SECONDS:
                    legacy.hold(job, "Shared GPU budget exhausted; fixed endpoint not reduced")
                    break
                command[command.index("--max-training-seconds")+1] = str(GPU_SECONDS-gpu)
            env = dict(os.environ, PYTHONUNBUFFERED="1", CUDA_DEVICE_ORDER="PCI_BUS_ID", OMP_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", NUMBA_NUM_THREADS="1",
                MPLCONFIGDIR=str(output / "matplotlib_cache"))
            if not needs_gpu:
                env["CUDA_VISIBLE_DEVICES"] = ""
            else:
                env.pop("CUDA_VISIBLE_DEVICES", None)
            stage_start, stage_cpu = time.monotonic(), cpu_seconds()
            with (output / "logs" / f"{name}.log").open("x") as stream:
                child = subprocess.Popen(command, cwd=plan["code_root"], env=env, stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                stage = dict(name=name, status="running", pid=child.pid, command=command, gpu=needs_gpu,
                    started_at=now(), log=str(output / "logs" / f"{name}.log"))
                job["stages"].append(stage)
                job.update(status="running", current=name)
                save(output / "job.json", job)
                returncode = child.wait()
            stage.update(status="complete" if returncode == 0 else "failed", returncode=returncode,
                ended_at=now(), wall_seconds=time.monotonic()-stage_start, cpu_seconds=cpu_seconds()-stage_cpu)
            if needs_gpu:
                gpu += stage["wall_seconds"]
            if returncode:
                folder = "profile_run" if name == "training_preflight" else "run" if name == "training" else \
                    "report" if name == "report" else "calibration_report" if name == "phi4_hybrid_report" else "qualification_report"
                child_status = output / folder / "status.json"
                if child_status.is_file() and read(child_status).get("status") == "budget_hold":
                    legacy.hold(job, f"{name} reached the fixed remaining resource budget; outputs preserved")
                    break
                raise RuntimeError(f"{name} returned {returncode}; no retry/restart")
            common.unchanged(plan)
            sources_unchanged(plan)
            if needs_gpu:
                folder = output / ("profile_run" if name == "training_preflight" else "run")
                summary = read(folder / "summary.json")
                hybrid.validate_training_summary(summary, plan, profile=name == "training_preflight")
                if summary.get("data_manifest_sha256") != plan["source_data_manifest_sha256"]:
                    raise RuntimeError("Training used a different physical source manifest")
                if name == "training_preflight":
                    receipt = common.profile_projection(summary, read(folder / "history_progress.json"), stage["wall_seconds"], gpu)
                    save(output / "gpu_preflight.json", receipt, exclusive=True)
                    job["gpu_projection"] = receipt
                    if not receipt["budget_passed"] or prior_cpu+cpu_seconds()-cpu_start+receipt["projected_formal_gpu_seconds_conservative"]+600 > CPU_SECONDS:
                        legacy.hold(job, "Profile conservative projection exceeds shared remaining CPU/GPU budget; fixed work not reduced")
                        break
                else:
                    job["actual_training_steps"] = 3000
            else:
                folder = "report" if name == "report" else "calibration_report" if name == "phi4_hybrid_report" else "qualification_report"
                status = read(output / folder / "status.json")
                if status.get("status") != "complete" or not status.get("workflow_complete"):
                    raise RuntimeError(f"{name} did not create a complete public report")
            if gpu > GPU_SECONDS or prior_cpu+cpu_seconds()-cpu_start > CPU_SECONDS or directory_bytes(Path(plan["source_root"])) > DIRECTORY_BYTES:
                legacy.hold(job, "Actual resource accounting exceeded the shared fixed budget; artifacts preserved")
                break
        else:
            sources_unchanged(plan, full=True)
            job.update(status="complete", workflow_complete=True, current=None, ended_at=now())
    except BaseException as error:
        job.update(status="failed", workflow_complete=False, error=f"{type(error).__name__}: {error}",
                   requires_user_action=True, ended_at=now())
        raise
    finally:
        job.update(gpu_stage_wall_seconds=gpu, cpu_seconds=cpu_seconds()-cpu_start,
            cumulative_cpu_seconds=prior_cpu+cpu_seconds()-cpu_start, wall_seconds=time.monotonic()-start,
            source_tree_directory_bytes=directory_bytes(Path(plan["source_root"])))
        execution.update(status=job["status"], workflow_complete=job["workflow_complete"], ended_at=now(),
            child_cpu_seconds=job["cpu_seconds"], source_cpu_seconds=plan["resource_policy"]["source_cpu_seconds"],
            preparation_cpu_seconds=preparation_cpu, launch_orchestration_cpu_seconds=launch_cpu,
            cumulative_cpu_seconds=job["cumulative_cpu_seconds"], gpu_stage_wall_seconds=gpu,
            source_tree_directory_bytes=job["source_tree_directory_bytes"], error=job["error"],
            requires_user_action=job["requires_user_action"])
        save(output / "execution.json", execution)
        if job["status"] == "complete":
            files = [path for path in output.rglob("*") if path.is_file() and path.suffix in (".json", ".png", ".npz")
                and not path.is_relative_to(output / "profile_run") and path.name != "job.json"]
            job["public_files"] = {str(path.relative_to(output)): sha(path) for path in sorted(files)}
        save(output / "job.json", job)


def launch(source_root, output):
    launch_cpu_start = cpu_seconds()
    legacy.validate_host()
    output = Path(output).resolve()
    plan = _load_plan(source_root, output)
    job = read(output / "job.json")
    if job["status"] != "prepared" or job["stages"] or (output / "execution.json").exists():
        raise ValueError("Exploratory child already executed; no restart")
    common.unchanged(plan)
    sources_unchanged(plan, full=True)
    if old_processes(Path(plan["source_root"])):
        raise RuntimeError("Actual held-source processes remain; no child launch")
    with (output / "launch.json").open("x") as receipt, (output / "logs" / "parent.log").open("x") as log:
        parent = subprocess.Popen([sys.executable, "-u", "-m", MODULE, "--execute", "--source-root", plan["source_root"],
            "--output", str(output)], cwd=plan["code_root"], stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True,
            env=dict(os.environ, PYTHONUNBUFFERED="1", CUDA_DEVICE_ORDER="PCI_BUS_ID", OMP_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", NUMBA_NUM_THREADS="1"))
        value = dict(status="launched", parent_pid=parent.pid, at=now(), plan_sha256=sha(output / "plan.json"),
            scientific_qualification=QUALIFICATION, source_quality_passed=False,
            source_root=plan["source_root"], fresh_optimization=True, original_parent_restarted=False,
            orchestration_cpu_seconds=cpu_seconds()-launch_cpu_start)
        receipt.write(json.dumps(value, indent=2, ensure_ascii=False)+"\n")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("prepare", "launch", "execute"):
        modes.add_argument("--"+mode, action="store_true")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.prepare:
        plan = prepare(args.source_root, args.output)
        value = dict(status="prepared", output=plan["output"], scientific_qualification=QUALIFICATION)
    elif args.launch:
        value = launch(args.source_root, args.output)
    else:
        execute(args.source_root, args.output)
        value = read(args.output / "job.json")
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)
    return 3 if value["status"] in ("failed", "budget_hold") else 0


if __name__ == "__main__":
    sys.exit(main())
