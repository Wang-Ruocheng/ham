"""Fresh, budget-only continuation of the completed synthetic hopping CUDA check.

The old budget-hold record is immutable.  Its synthetic model is never loaded or
reused, and the continuation performs no further benchmark updates.  Production
sampling and the original fresh 3000-update recipe run in a new exclusive root.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time

from scripts import run_phi4_hopping_qcr as base

CPU_SECONDS = 21600.
GPU_SECONDS = 14400.
PREFLIGHT_SHA256 = "6099be530c3335625a8d624e327842af79872a695a273451d7358ab8f3d549c3"
GPU_UUID = "GPU-a167decd-ce49-6c8f-6bcf-136fde596683"


def _read(path):
    return json.loads(Path(path).read_text())


def _code_root():
    return Path(__file__).resolve().parents[1]


def _real_python_files(code):
    return {str(path.relative_to(code)): path for folder in ("experiments", "recurrent_rg", "scripts")
            for path in (code / folder).rglob("*.py") if not path.name.startswith("._")}


def prerequisites(prior_root, config, gpu_index, gpu_uuid, prior_queue_root=None):
    """Read and bind the terminal preflight evidence without touching PT/data.

    Only a cap change from 1800 to 14400 seconds is permitted in the source JSON.
    All executable computational source files must be byte-identical to the
    original source snapshot. AppleDouble files are metadata, not Python source.
    """
    prior, code = Path(prior_root).resolve(), _code_root()
    job, old_plan = _read(prior / "job.json"), _read(prior / "plan.json")
    receipt = _read(prior / "gpu_preflight.json")
    stages = job.get("stages", [])
    if (job.get("status") != "budget_hold" or job.get("workflow_complete") is not False
            or job.get("returncode") != 3 or job.get("current") is not None
            or job.get("requires_user_action") is not True or len(stages) != 1
            or stages[0].get("name") != "gpu_preflight"
            or stages[0].get("status") != "budget_hold" or stages[0].get("returncode") != 3
            or job.get("plan_sha256") != base.sha(prior / "plan.json")
            or old_plan.get("host") != base.HOST
            or old_plan.get("output") != str(prior)
            or old_plan.get("GPU") != dict(index=gpu_index, uuid=gpu_uuid)
            or gpu_index != 3 or gpu_uuid != GPU_UUID
            or (prior / "data").exists() or (prior / "training").exists()):
        raise ValueError("Prior root is not the exclusive completed GPU3 preflight budget hold")
    if (base.sha(prior / "gpu_preflight.json") != PREFLIGHT_SHA256
            or receipt.get("status") != "budget_hold" or receipt.get("returncode") != 3
            or receipt.get("workflow_complete") is not True or receipt.get("error") is not None
            or receipt.get("synthetic") is not True or receipt.get("physical_result") is not False
            or receipt.get("checkpoint_saved") is not False or receipt.get("formal_model_reused") is not False
            or receipt.get("physical_data_read") is not False or receipt.get("actual_updates") != 7
            or receipt.get("training_batch") != [32, 64, 36, 36, 1]
            or receipt.get("training_cap_seconds") != 1800.
            or receipt.get("peak_cuda_bytes", base.PEAK_BYTES+1) > base.PEAK_BYTES
            or receipt.get("device", {}).get("CUDA_VISIBLE_DEVICES") != gpu_uuid
            or set(receipt.get("optimizer_steps", {})) != {"Q", "C", "R"}
            or any(row.get("min") != 7 or row.get("max") != 7 or row.get("parameters", 0) <= 0
                   for row in receipt["optimizer_steps"].values())):
        raise ValueError("Synthetic receipt does not match the authorized completed check")
    projection = receipt.get("conservative_full_training_projection_seconds", math.inf)
    if not math.isfinite(projection) or projection <= 0:
        raise ValueError("Invalid conservative projection")
    old_config = Path(old_plan["config_source"])
    if base.sha(old_config) != old_plan["config_source_sha256"]:
        raise ValueError("Old source configuration changed")
    expected = _read(old_config)
    if expected.get("training", {}).get("max_training_seconds") != 1800.:
        raise ValueError("Unexpected old training cap")
    expected["training"]["max_training_seconds"] = GPU_SECONDS
    if _read(config) != expected:
        raise ValueError("Only the authorized training budget field may change")

    old_code = Path(old_plan["code_root"]).resolve()
    source_pins = old_plan.get("source_sha256", {})
    old_real, excluded, bindings = {}, 0, {}
    for filename, digest in source_pins.items():
        path = Path(filename)
        if path.name.startswith("._"):
            excluded += 1
            continue
        if base.sha(path) != digest:
            raise ValueError(f"Prior pinned source or configuration changed: {path}")
        bindings[str(path)] = digest
        if old_code in path.parents and path.suffix == ".py":
            relative = str(path.relative_to(old_code))
            current = code / relative
            if not current.is_file() or base.sha(current) != digest:
                raise ValueError(f"Computational source differs from completed preflight: {relative}")
            old_real[relative] = digest
    present = _real_python_files(code)
    allowed = {"scripts/continue_phi4_hopping_budget.py"}
    if not old_real or set(present)-allowed != set(old_real)-allowed:
        raise ValueError("Computational source inventory differs from the frozen preflight snapshot")
    # The old plan itself binds every hash above; pin its terminal job and receipt too.
    for name in ("job.json", "plan.json", "gpu_preflight.json", "training_config.json"):
        bindings[str(prior / name)] = base.sha(prior / name)
    cpu, gpu = float(job.get("cpu_seconds", math.inf)), float(job.get("gpu_seconds", math.inf))
    queue = None
    if prior_queue_root is not None:
        folder = Path(prior_queue_root).resolve()
        queued = _read(folder / "job.json")
        if (queued.get("status") != "cancelled_before_compute" or queued.get("stages") != []
                or queued.get("current") is not None or queued.get("gpu_seconds") != 0
                or queued.get("error") is not None or (folder / "data").exists()
                or (folder / "training").exists()
                or queued.get("plan_sha256") != base.sha(folder / "plan.json")):
            raise ValueError("Prior GPU2 queue is not a cancelled zero-compute record")
        charge = float(queued.get("cpu_seconds", math.inf))
        if not math.isfinite(charge) or charge < 0:
            raise ValueError("Invalid prior queue CPU charge")
        cpu += charge
        queue = dict(root=str(folder), cpu_seconds=charge, gpu_seconds=0., executed_stages=0)
        for name in ("job.json", "plan.json"):
            bindings[str(folder / name)] = base.sha(folder / name)
    if not math.isfinite(cpu) or not math.isfinite(gpu) or cpu < 0 or gpu < 0:
        raise ValueError("Invalid prior resource accounting")
    if gpu+projection+10 > GPU_SECONDS or cpu+projection > CPU_SECONDS:
        raise ValueError("Prior charged use plus projection exceeds the newly authorized budget")
    return dict(prior_root=str(prior), prior_job_sha256=base.sha(prior / "job.json"),
        prior_plan_sha256=base.sha(prior / "plan.json"), preflight_sha256=PREFLIGHT_SHA256,
        original_receipt_status="budget_hold", original_receipt_returncode=3,
        new_budget_gate="pass", projection_seconds=projection, prior_cpu_seconds=cpu,
        prior_gpu_seconds=gpu, prior_queue=queue, executable_source_files=len(old_real),
        excluded_appledouble_metadata_files=excluded, prior_source_bindings=bindings,
        benchmark_reused_as_timing_evidence_only=True, additional_benchmark_updates=0,
        physical_data_read_by_continuation_gate=False, formal_model_reused=False,
        checkpoint_loaded=False, old_root_modified=False)


def plan(prior_root, output, config, gpu_index=3, gpu_uuid=GPU_UUID, prior_queue_root=None):
    from experiments.phi4_hopping_qcr import TrainingOptions, load_config, training_plan
    from recurrent_rg.data.phi4_hopping_pipeline import Phi4HoppingOptions, generation_plan
    root, code, config = Path(output).resolve(), _code_root(), Path(config).resolve()
    if root.exists() or root == code or root in code.parents or code in root.parents:
        raise ValueError("Require a fresh output separate from the frozen code snapshot")
    evidence = prerequisites(prior_root, config, gpu_index, gpu_uuid, prior_queue_root)
    values = load_config(config)
    values.update(data=str(root / "data"), output=str(root / "training"),
                  device="cuda:0", expected_gpu_uuid=gpu_uuid)
    options = TrainingOptions(**values)
    original = _read(Path(prior_root) / "plan.json")
    physical = json.loads(json.dumps(generation_plan(Phi4HoppingOptions())))
    if physical != original["physical"]:
        raise ValueError("Physical generation plan changed from the frozen full experiment")
    command = base.commands(root, config, gpu_uuid)
    del command["gpu_preflight"]
    return dict(schema_version=5, created_at=base.now(), execution_requested=False,
        code_root=str(code), output=str(root), host=base.HOST, config_source=str(config),
        config_source_sha256=base.sha(config), GPU=dict(index=gpu_index, uuid=gpu_uuid),
        physical=physical, training=training_plan(options), commands=command,
        stages=["sampling", "training"], budget_extension=evidence,
        resources=dict(cpu_seconds=CPU_SECONDS, gpu_card_seconds=GPU_SECONDS,
            max_training_seconds=GPU_SECONDS, max_cuda_bytes=base.PEAK_BYTES,
            directory_bytes=base.DIRECTORY_BYTES, cpu_threads=1, nice=10,
            min_memory_bytes=36*1024**3, max_load_per_cpu=1.25, max_io_full_avg10=20.,
            idle_check_spacing_seconds=10., min_gpu_free_bytes=12*1024**3,
            require_no_compute_processes=True),
        no_resume=True, no_retry=True, no_auto_repair=True, no_migration=True,
        qualification="只提高预算的新正式输出；复用已完成的合成计时证据，不复用模型；保留采样质量警告；不认证平衡或临界指数")


def _validate_sampling(root):
    from recurrent_rg.data.phi4_hopping_loader import load_dataset
    dataset = load_dataset(root / "data")
    result = dict(data_manifest_sha256=dataset.manifest_sha256,
                  data_warning_count=dataset.warning_count)
    del dataset
    quality = _read(root / "data/quality_summary.json")
    acceptances = [value for row in quality["conditions"] for value in row["production_acceptance"]]
    if len(acceptances) != 52 or any(not math.isfinite(value) or value <= 0 for value in acceptances):
        raise RuntimeError("Zero/nonfinite HMC production acceptance; data retained, training not started")
    return result


def _public_files(root):
    public = {name: base.sha(root / name) for name in
              ("job.json", "plan.json", "training_config.json", "reused_preflight.json", "budget_extension_receipt.json")}
    for name in _read(root / "data/manifest.json")["public_files"]:
        path = (root / "data" / name).resolve()
        if root / "data" not in path.parents or path.suffix != ".json":
            raise RuntimeError("Invalid public physical metadata path")
        public[f"data/{name}"] = base.sha(path)
    public["training/status.json"] = base.sha(root / "training/status.json")
    for row in _read(root / "training/status.json")["public_files"]:
        path = (root / "training" / row["path"]).resolve()
        if (root / "training" not in path.parents or path.suffix not in (".json", ".png", ".npz")
                or base.sha(path) != row["sha256"]):
            raise RuntimeError("Invalid public training artifact binding")
        public[f"training/{row['path']}"] = row["sha256"]
    base.save(root / "public_files.json", public)


def run(prior_root, output, config, gpu_index=3, gpu_uuid=GPU_UUID, prior_queue_root=None):
    definition = plan(prior_root, output, config, gpu_index, gpu_uuid, prior_queue_root)
    if socket.gethostname() != base.HOST or os.getpriority(os.PRIO_PROCESS, 0) != 10:
        raise ValueError("Execution requires the pinned 225 host and one inherited nice10 layer")
    prior = Path(prior_root)
    old_job = _read(prior / "job.json")
    old_pids = [old_job.get("parent_pid"), *[row.get("pid") for row in old_job["stages"]]]
    if any(isinstance(pid, int) and Path(f"/proc/{pid}").exists() for pid in old_pids):
        raise ValueError("Prior preflight parent/child PID is still present; do not start another workflow")
    root, code = Path(output).resolve(), _code_root()
    pins = {str(path): base.sha(path) for path in _real_python_files(code).values()}
    pins.update(definition["budget_extension"]["prior_source_bindings"])
    pins[str(Path(config).resolve())] = base.sha(config)
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    configuration = copy.deepcopy(_read(config))
    configuration.update(data=str(root / "data"), output=str(root / "training"))
    configuration["training"].update(device="cuda:0", expected_gpu_uuid=gpu_uuid)
    base.save(root / "training_config.json", configuration)
    # Preserve the original JSON bytes/hash and its rc=3; the new gate is separate.
    (root / "reused_preflight.json").write_bytes((prior / "gpu_preflight.json").read_bytes())
    base.save(root / "budget_extension_receipt.json", definition["budget_extension"])
    for name in ("training_config.json", "reused_preflight.json", "budget_extension_receipt.json"):
        pins[str(root / name)] = base.sha(root / name)
    base.save(root / "plan.json", dict(definition, execution_requested=True, source_sha256=pins))
    pins[str(root / "plan.json")] = base.sha(root / "plan.json")
    previous = definition["budget_extension"]
    job = dict(status="prepared", workflow_complete=False, parent_pid=os.getpid(), current=None,
        returncode=None, error=None, requires_user_action=False, started_at=base.now(), stages=[],
        plan_sha256=base.sha(root / "plan.json"), cpu_seconds=previous["prior_cpu_seconds"],
        gpu_seconds=previous["prior_gpu_seconds"], prior_cpu_seconds=previous["prior_cpu_seconds"],
        prior_gpu_seconds=previous["prior_gpu_seconds"], additional_benchmark_updates=0)
    cpu_start = time.process_time()
    child_start = resource.getrusage(resource.RUSAGE_CHILDREN)
    gpu_seconds, active, active_stage = previous["prior_gpu_seconds"], None, None

    def checkpoint():
        children = resource.getrusage(resource.RUSAGE_CHILDREN)
        job["cpu_seconds"] = (previous["prior_cpu_seconds"]+time.process_time()-cpu_start
            + children.ru_utime-child_start.ru_utime+children.ru_stime-child_start.ru_stime)
        if active is not None and active.poll() is None:
            try:
                fields = Path(f"/proc/{active.pid}/stat").read_text().split(")", 1)[1].split()
                job["cpu_seconds"] += (int(fields[11])+int(fields[12]))/os.sysconf("SC_CLK_TCK")
            except (FileNotFoundError, ProcessLookupError):
                pass
        job["gpu_seconds"] = gpu_seconds+(time.monotonic()-active_stage["monotonic_start"]
            if active is not None and active.poll() is None and active_stage["name"] == "training" else 0.)
        job["directory_bytes"] = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
        base.save(root / "job.json", job)

    def verify():
        if any(base.sha(path) != digest for path, digest in pins.items()):
            raise RuntimeError("Pinned computational source/configuration/prior evidence changed; no repair")

    def hold(message):
        job.update(status="budget_hold", current=None, returncode=3, requires_user_action=True,
                   error=message, ended_at=base.now())
        checkpoint()
        return 3

    def wait_resources(name):
        nonlocal gpu_seconds
        gpu = name == "training"
        job.update(status="waiting_resources", current=name)
        while True:
            first = dict(host=base.host_resources())
            if gpu:
                first["gpu"] = base.gpu_resources(gpu_index, gpu_uuid)
            job["resources"] = first
            checkpoint()
            if not base.ready(first, gpu):
                time.sleep(30)
                continue
            if not gpu:
                return
            started = time.monotonic()
            time.sleep(10)
            second = dict(host=base.host_resources(), gpu=base.gpu_resources(gpu_index, gpu_uuid))
            spacing = time.monotonic()-started
            gpu_seconds += spacing
            job["gpu_idle_checks"] = dict(first=first, second=second, spacing_seconds=spacing)
            checkpoint()
            if spacing >= 10 and base.ready(second, True):
                return

    checkpoint()
    try:
        from scripts.report_phi4_hopping_qcr import select_chinese_font
        job["chinese_font"] = str(select_chinese_font(os.environ.get("QCR_ZH_FONT")))
        for name, command in definition["commands"].items():
            verify()
            wait_resources(name)
            verify()
            checkpoint()
            if (job["cpu_seconds"] > CPU_SECONDS or job["gpu_seconds"] > GPU_SECONDS
                    or job["directory_bytes"] > base.DIRECTORY_BYTES):
                return hold("Registered cumulative resource budget exhausted before next stage")
            projection = previous["projection_seconds"]
            if (job["gpu_seconds"]+projection > GPU_SECONDS
                    or job["cpu_seconds"]+projection > CPU_SECONDS):
                return hold("Actual cumulative use plus conservative training projection exceeds new budget")
            stage = dict(name=name, status="running", returncode=None, started_at=base.now(),
                         command=command, monotonic_start=time.monotonic())
            active_stage = stage
            job["stages"].append(stage)
            job.update(status="running", current=name)
            with (root / "logs" / f"{name}.log").open("xb") as log:
                active = subprocess.Popen(command, cwd=code,
                    env=base.child_environment(gpu_uuid if name == "training" else None),
                    stdout=log, stderr=subprocess.STDOUT)
                stage["pid"] = active.pid
                while active.poll() is None:
                    try:
                        stage["child_status"] = _read(root / ("data" if name == "sampling" else "training") / "status.json")
                    except (FileNotFoundError, json.JSONDecodeError):
                        pass
                    checkpoint()
                    time.sleep(5)
                rc = active.returncode
            elapsed = time.monotonic()-stage["monotonic_start"]
            if name == "training":
                gpu_seconds += elapsed
            stage.update(status="complete" if rc == 0 else "budget_hold" if rc == 3 else "failed",
                         returncode=rc, ended_at=base.now(), wall_seconds=elapsed)
            active, active_stage = None, None
            verify()
            if rc:
                job.update(status=stage["status"], current=None, returncode=rc, requires_user_action=True,
                           error=f"{name} exited {rc}; completed outputs preserved", ended_at=base.now())
                checkpoint()
                return rc
            if name == "sampling":
                job.update(_validate_sampling(root))
            else:
                base.validate_training(root, job["data_manifest_sha256"])
            checkpoint()
        if (job["cpu_seconds"] > CPU_SECONDS or job["gpu_seconds"] > GPU_SECONDS
                or job["directory_bytes"] > base.DIRECTORY_BYTES):
            return hold("Completed outputs exceed newly registered cumulative resources")
        job.update(status="complete", workflow_complete=True, current=None, returncode=0,
                   error=None, requires_user_action=False, ended_at=base.now())
        checkpoint()
        _public_files(root)
        return 0
    except BaseException as exc:
        if active is not None and active.poll() is None:
            job.update(status="waiting_running_child_after_parent_error", requires_user_action=True,
                       error=f"{type(exc).__name__}: {exc}")
            checkpoint()
            active.wait()  # Never kill, retry, resume or migrate an active child.
        job.update(status="failed", workflow_complete=False, current=None, returncode=1,
                   requires_user_action=True, error=f"{type(exc).__name__}: {exc}", ended_at=base.now())
        checkpoint()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-root", required=True)
    parser.add_argument("--prior-queue-root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/phi4_hopping_b3_g1_aligned_seed91_budget4h.json")
    parser.add_argument("--gpu-index", type=int, default=3)
    parser.add_argument("--gpu-uuid", default=GPU_UUID)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    values = (args.prior_root, args.output, args.config, args.gpu_index, args.gpu_uuid, args.prior_queue_root)
    if args.execute:
        return run(*values)
    print(json.dumps(plan(*values), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
