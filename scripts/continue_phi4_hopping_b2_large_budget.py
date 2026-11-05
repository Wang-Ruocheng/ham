"""Fresh budget-only continuation of the completed L128/b2 synthetic check.

The prior rc=3 record is immutable and is reused only as seven-step timing
evidence. No benchmark, model, physical dataset, optimizer or RNG is resumed.
Dry planning is the default; only the explicitly pinned GPU3 may be used.
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
from scripts import run_phi4_hopping_b2_large as large

CPU_SECONDS, GPU_SECONDS, DIRECTORY_BYTES = 86400., 57600., 2 * 1024**3
PREFLIGHT_SHA256 = "9511082ecf79091cd9a1a40b0b78c83400b3f9ba9b8e56555fe378560908e5df"
GPU_UUID = "GPU-a167decd-ce49-6c8f-6bcf-136fde596683"
DEFAULT_CONFIG = "configs/phi4_hopping_b2_L128_g1_seed91_budget16h.json"
NEW_SOURCE = "scripts/continue_phi4_hopping_b2_large_budget.py"


def _read(path):
    return json.loads(Path(path).read_text())


def _code_root():
    return Path(__file__).resolve().parents[1]


def budget_ok(cpu_seconds, gpu_seconds, directory_bytes, *, cpu_projection=0., gpu_projection=0.):
    values = (cpu_seconds, gpu_seconds, directory_bytes, cpu_projection, gpu_projection)
    return (all(math.isfinite(x) and x >= 0 for x in values)
        and cpu_seconds + cpu_projection + gpu_projection <= CPU_SECONDS
        and gpu_seconds + gpu_projection <= GPU_SECONDS and directory_bytes <= DIRECTORY_BYTES)


def prerequisites(prior_root, config, gpu_index, gpu_uuid, sampling_plan):
    """Authenticate terminal timing evidence and the one permitted cap change."""
    prior, code = Path(prior_root).resolve(), _code_root()
    job, original = _read(prior / "job.json"), _read(prior / "plan.json")
    stages, receipt = job.get("stages", []), _read(prior / "gpu_preflight.json")
    if (job.get("status") != "budget_hold" or job.get("workflow_complete") is not False
            or job.get("returncode") != 3 or job.get("current") is not None
            or job.get("requires_user_action") is not True or len(stages) != 1
            or stages[0].get("name") != "gpu_preflight"
            or stages[0].get("status") != "budget_hold" or stages[0].get("returncode") != 3
            or job.get("formal_training_started") is not False or job.get("formal_optimizer_updates") != 0
            or job.get("plan_sha256") != base.sha(prior / "plan.json")
            or original.get("host") != base.HOST or original.get("output") != str(prior)
            or original.get("GPU") != dict(index=gpu_index, uuid=gpu_uuid)
            or gpu_index != 3 or gpu_uuid != GPU_UUID
            or (prior / "data").exists() or (prior / "training").exists()
            or (prior / "sampler_pilot.json").exists()):
        raise ValueError("Prior root is not the exclusive zero-production preflight budget hold")
    projection = large.benchmark_projection(receipt.get("initialization_seconds", math.nan),
                                            receipt.get("timed_update_seconds", []))
    clocks = receipt.get("optimizer_steps", {})
    if (base.sha(prior / "gpu_preflight.json") != PREFLIGHT_SHA256
            or receipt.get("status") != "budget_hold" or receipt.get("returncode") != 3
            or receipt.get("workflow_complete") is not False or receipt.get("error") is not None
            or receipt.get("requires_user_action") is not True
            or receipt.get("current") is not None or receipt.get("synthetic") is not True
            or receipt.get("physical_result") is not False or receipt.get("checkpoint_saved") is not False
            or receipt.get("formal_model_reused") is not False or receipt.get("physical_data_read") is not False
            or receipt.get("actual_updates") != 7 or receipt.get("warmup_updates") != 2
            or receipt.get("timed_updates") != 5 or receipt.get("training_batch") != [32, 64, 128, 128, 1]
            or receipt.get("geometry") != dict(block_size=2, depth=3, sizes=[128,64,32,16])
            or receipt.get("training_cap_seconds") != 14400.
            or receipt.get("evaluation_report_reserve_seconds") != large.EVALUATION_REPORT_RESERVE
            or receipt.get("peak_cuda_bytes", base.PEAK_BYTES + 1) > base.PEAK_BYTES
            or receipt.get("device", {}).get("CUDA_VISIBLE_DEVICES") != gpu_uuid
            or receipt.get("conservative_full_training_projection_seconds") != projection
            or set(clocks) != {"Q", "C", "R"}
            or any(row.get("min") != 7 or row.get("max") != 7 or row.get("parameters", 0) <= 0
                   for row in clocks.values())):
        raise ValueError("Synthetic receipt differs from the completed seven-step L128 check")
    old_config = Path(original["config_source"])
    old_sampling_plan = Path(original["sampling_plan_source"])
    if (base.sha(old_config) != original["config_source_sha256"]
            or base.sha(old_sampling_plan) != original["sampling_plan_sha256"]
            or base.sha(sampling_plan) != original["sampling_plan_sha256"]):
        raise ValueError("Pinned source configuration or physical sampling plan changed")
    expected = _read(old_config)
    if expected.get("training", {}).get("max_training_seconds") != 14400.:
        raise ValueError("Unexpected original training cap")
    expected["training"]["max_training_seconds"] = GPU_SECONDS
    if _read(config) != expected:
        raise ValueError("Only the authorized training budget field may change")

    old_code, old_real, bindings, excluded = Path(original["code_root"]).resolve(), {}, {}, 0
    for filename, digest in original.get("source_sha256", {}).items():
        path = Path(filename)
        if path.name.startswith("._"):
            excluded += 1
            continue
        if base.sha(path) != digest:
            raise ValueError(f"Prior pinned source or configuration changed: {path}")
        bindings[str(path)] = digest
        if old_code in path.parents and path.suffix == ".py":
            relative = str(path.relative_to(old_code))
            if not (code / relative).is_file() or base.sha(code / relative) != digest:
                raise ValueError(f"Computational source differs from the frozen check: {relative}")
            old_real[relative] = digest
    present = large._real_python_files(code)
    if not old_real or set(present) - {NEW_SOURCE} != set(old_real) - {NEW_SOURCE}:
        raise ValueError("Computational source inventory differs from the frozen preflight snapshot")
    # Bind every old byte, including terminal logs, without rewriting any of them.
    for path in prior.rglob("*"):
        if path.is_file():
            bindings[str(path)] = base.sha(path)
    cpu, gpu = float(job.get("cpu_seconds", math.inf)), float(job.get("gpu_seconds", math.inf))
    if not budget_ok(cpu, gpu, 0, gpu_projection=projection + 10.):
        raise ValueError("Prior charged use plus projection exceeds the newly authorized budget")
    return dict(prior_root=str(prior), prior_job_sha256=base.sha(prior / "job.json"),
        prior_plan_sha256=base.sha(prior / "plan.json"), preflight_sha256=PREFLIGHT_SHA256,
        original_receipt_status="budget_hold", original_receipt_returncode=3,
        original_receipt_workflow_complete=False, completed_synthetic_updates=7,
        new_budget_gate="pass", projection_seconds=projection, prior_cpu_seconds=cpu,
        prior_gpu_seconds=gpu, prior_directory_bytes=sum(path.stat().st_size for path in prior.rglob("*") if path.is_file()),
        executable_source_files=len(old_real),
        excluded_appledouble_metadata_files=excluded, prior_source_bindings=bindings,
        prior_root_inventory=sorted(str(path.relative_to(prior)) for path in prior.rglob("*")),
        benchmark_reused_as_timing_evidence_only=True, additional_benchmark_updates=0,
        physical_data_read_by_continuation_gate=False, checkpoint_loaded=False,
        formal_model_reused=False, old_root_modified=False)


def prior_live_processes(prior_root, job, proc=Path("/proc")):
    """An extant unrelated/reused PID is not evidence that this workflow lives."""
    prior = str(Path(prior_root).resolve())
    modules = {large.MODULE, "scripts.prepare_phi4_hopping_b2_large", "recurrent_rg"}
    rows = []
    for pid in {job.get("parent_pid"), *[row.get("pid") for row in job.get("stages", [])]}:
        if type(pid) is not int or pid <= 0 or pid == os.getpid():
            continue
        try:
            args = (proc / str(pid) / "cmdline").read_bytes().decode(errors="replace").split("\0")
        except FileNotFoundError:
            continue
        if any(arg == prior or arg.startswith(prior + "/") for arg in args) and modules.intersection(args):
            rows.append(dict(pid=pid, command=args))
    return rows


def plan(prior_root, output, config=DEFAULT_CONFIG, gpu_index=3, gpu_uuid=GPU_UUID,
         sampling_plan=large.DEFAULT_SAMPLING_PLAN):
    from experiments.phi4_hopping_qcr import TrainingOptions, load_config, training_plan
    from recurrent_rg.data.phi4_hopping_b2_large_pipeline import generation_plan
    root, code = Path(output).resolve(), _code_root()
    config, sampling_plan = Path(config).resolve(), Path(sampling_plan).resolve()
    if root.exists() or root == code or root in code.parents or code in root.parents:
        raise ValueError("Require a fresh output separate from the frozen code snapshot")
    evidence = prerequisites(prior_root, config, gpu_index, gpu_uuid, sampling_plan)
    original = _read(Path(prior_root) / "plan.json")
    options = large.physical_options(sampling_plan)
    physical = json.loads(json.dumps(generation_plan(options)))
    pilot_definition = large.pilot_plan(sampling_plan)
    if physical != original["physical"] or pilot_definition != original["pilot"]:
        raise ValueError("Physical production or fifteen-case pilot plan changed")
    values = load_config(config)
    values.update(data=str(root / "data"), output=str(root / "training"), device="cuda:0", expected_gpu_uuid=gpu_uuid)
    training = training_plan(TrainingOptions(**values))
    commands = dict(sampler_pilot=[sys.executable, "-u", "-m", large.MODULE,
            "--pilot-output", str(root / "sampler_pilot.json"), "--sampling-plan", str(sampling_plan)],
        sampling=large.sampling_command(root, options),
        training=[sys.executable, "-u", "-m", "recurrent_rg", "train", "--config",
            str(root / "training_config.json"), "--output", str(root / "training"), "--device", "cuda:0"])
    return dict(schema_version=5, created_at=base.now(), execution_requested=False,
        code_root=str(code), output=str(root), host=base.HOST, config_source=str(config),
        config_source_sha256=base.sha(config), sampling_plan_source=str(sampling_plan),
        sampling_plan_sha256=base.sha(sampling_plan), GPU=dict(index=gpu_index, uuid=gpu_uuid),
        physical=physical, pilot=pilot_definition, training=training, commands=commands,
        stages=list(commands), budget_extension=evidence, work_accounting=original["work_accounting"],
        training_budget_admission=dict(source_config="training_config.json",
            runtime_config="training_runtime_config.json", receipt="training_admission.json",
            only_changed_field="training.max_training_seconds", startup_and_boundary_reserve_seconds=60.,
            runtime_cap="min(remaining cumulative CPU seconds, remaining cumulative GPU seconds)-60",
            projection_must_fit_runtime_cap=True, changes_planned_updates=False),
        resources=dict(cpu_seconds=CPU_SECONDS, gpu_card_seconds=GPU_SECONDS,
            max_training_seconds=GPU_SECONDS, max_cuda_bytes=base.PEAK_BYTES,
            directory_bytes=DIRECTORY_BYTES, directory_scope="new output plus immutable prior output bytes",
            cpu_threads=1, nice=10,
            min_memory_bytes=36*1024**3, max_load_per_cpu=1.25, max_io_full_avg10=20.,
            idle_check_spacing_seconds=10., min_gpu_free_bytes=12*1024**3, require_no_compute_processes=True),
        production_is_fresh=True, historical_data_read=False, historical_model_reused=False,
        no_resume=True, no_retry=True, no_auto_repair=True, no_migration=True,
        qualification="仅提高CPU/GPU及目录预算；复用完成的合成计时证据，不复用模型；新15-case pilot后门控生产；保留质量警告，不认证平衡或临界指数")


def _public_files(root, *, completed=False):
    names = ["job.json", "plan.json", "training_config.json", "reused_preflight.json", "budget_extension_receipt.json"]
    names += [name for name in ("sampler_pilot.json", "training_runtime_config.json", "training_admission.json")
              if (root / name).is_file()]
    public = {name: base.sha(root / name) for name in names}
    if completed:
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


def run(prior_root, output, config=DEFAULT_CONFIG, gpu_index=3, gpu_uuid=GPU_UUID,
        sampling_plan=large.DEFAULT_SAMPLING_PLAN):
    cpu_start, child_start = time.process_time(), resource.getrusage(resource.RUSAGE_CHILDREN)
    definition = plan(prior_root, output, config, gpu_index, gpu_uuid, sampling_plan)
    if socket.gethostname() != base.HOST or os.getpriority(os.PRIO_PROCESS, 0) != 10:
        raise ValueError("Execution requires pinned 225 host and exactly one inherited nice10 layer")
    if prior_live_processes(prior_root, _read(Path(prior_root) / "job.json")):
        raise ValueError("Prior workflow parent/child is still running; no duplicate workflow")
    root, code = Path(output).resolve(), _code_root()
    previous = definition["budget_extension"]
    pins = {str(path): base.sha(path) for path in large._real_python_files(code).values()}
    pins.update(previous["prior_source_bindings"])
    pins.update({str(Path(path).resolve()): base.sha(path) for path in (config, sampling_plan)})
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    configuration = copy.deepcopy(_read(config))
    configuration.update(data=str(root / "data"), output=str(root / "training"))
    configuration["training"].update(device="cuda:0", expected_gpu_uuid=gpu_uuid)
    base.save(root / "training_config.json", configuration)
    (root / "reused_preflight.json").write_bytes((Path(prior_root) / "gpu_preflight.json").read_bytes())
    base.save(root / "budget_extension_receipt.json", previous)
    for name in ("training_config.json", "reused_preflight.json", "budget_extension_receipt.json"):
        pins[str(root / name)] = base.sha(root / name)
    base.save(root / "plan.json", dict(definition, execution_requested=True, source_sha256=pins))
    pins[str(root / "plan.json")] = base.sha(root / "plan.json")
    job = dict(status="prepared", workflow_complete=False, parent_pid=os.getpid(), current=None,
        returncode=None, error=None, requires_user_action=False, started_at=base.now(), stages=[],
        plan_sha256=base.sha(root / "plan.json"), cpu_seconds=previous["prior_cpu_seconds"],
        gpu_seconds=previous["prior_gpu_seconds"], prior_cpu_seconds=previous["prior_cpu_seconds"],
        prior_gpu_seconds=previous["prior_gpu_seconds"], additional_benchmark_updates=0,
        formal_training_started=False, formal_optimizer_updates=0, historical_data_read=False,
        training_projection_seconds=previous["projection_seconds"])
    gpu_seconds, active, active_stage = previous["prior_gpu_seconds"], None, None

    def checkpoint():
        children = resource.getrusage(resource.RUSAGE_CHILDREN)
        job["cpu_seconds"] = (previous["prior_cpu_seconds"] + time.process_time() - cpu_start
            + children.ru_utime - child_start.ru_utime + children.ru_stime - child_start.ru_stime)
        if active is not None and active.poll() is None:
            try:
                fields = Path(f"/proc/{active.pid}/stat").read_text().split(")", 1)[1].split()
                job["cpu_seconds"] += (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
            except (FileNotFoundError, ProcessLookupError):
                pass
        job["gpu_seconds"] = gpu_seconds + (time.monotonic() - active_stage["monotonic_start"]
            if active is not None and active.poll() is None and active_stage["name"] == "training" else 0.)
        job["new_directory_bytes"] = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
        job["prior_directory_bytes"] = previous["prior_directory_bytes"]
        job["directory_bytes"] = job["new_directory_bytes"] + previous["prior_directory_bytes"]
        base.save(root / "job.json", job)

    def verify():
        if any(base.sha(path) != digest for path, digest in pins.items()):
            raise RuntimeError("Pinned source/configuration/prior evidence changed; no repair")
        prior = Path(prior_root).resolve()
        if sorted(str(path.relative_to(prior)) for path in prior.rglob("*")) != previous["prior_root_inventory"]:
            raise RuntimeError("Prior immutable root inventory changed")

    def admitted(name=None):
        return budget_ok(job["cpu_seconds"], job["gpu_seconds"], job["directory_bytes"],
            cpu_projection=job.get("remaining_production_cpu_projection_seconds", 0.) if name == "sampling" else 0.,
            gpu_projection=job["training_projection_seconds"] if name else 0.)

    def hold(message):
        job.update(status="budget_hold", workflow_complete=False, current=None, returncode=3,
            requires_user_action=True, error=message, ended_at=base.now())
        checkpoint()
        _public_files(root)
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
            if not admitted(name):
                return False
            if not base.ready(first, gpu):
                time.sleep(30)
                continue
            if not gpu:
                return True
            started = time.monotonic()
            time.sleep(10)
            second = dict(host=base.host_resources(), gpu=base.gpu_resources(gpu_index, gpu_uuid))
            spacing = time.monotonic() - started
            gpu_seconds += spacing
            job["gpu_idle_checks"] = dict(first=first, second=second, spacing_seconds=spacing)
            checkpoint()
            if not admitted(name):
                return False
            if spacing >= 10 and base.ready(second, True):
                return True

    def child_path(name):
        return root / "sampler_pilot.json" if name == "sampler_pilot" else root / (
            "data" if name == "sampling" else "training") / "status.json"

    def child_status(name, stage):
        try:
            stage["child_status"] = _read(child_path(name))
            if name == "training":
                job["formal_optimizer_updates"] = stage["child_status"].get("actual_steps", 0)
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    checkpoint()
    try:
        from scripts.report_phi4_hopping_qcr import select_chinese_font
        job["chinese_font"] = str(select_chinese_font(os.environ.get("QCR_ZH_FONT")))
        for name, command in definition["commands"].items():
            verify()
            if not wait_resources(name):
                return hold("Cumulative actual use plus remaining projection exceeds budget while waiting; no next stage")
            verify()
            checkpoint()
            if not admitted(name):
                return hold("Cumulative actual use plus remaining conservative CPU/GPU projection exceeds budget")
            if name == "training":
                runtime_cap = min(GPU_SECONDS-job["gpu_seconds"], CPU_SECONDS-job["cpu_seconds"])-60.
                if runtime_cap < job["training_projection_seconds"]:
                    return hold("Remaining cumulative CPU/GPU budget minus boundary reserve cannot admit full projected training")
                runtime_config = copy.deepcopy(configuration)
                runtime_config["training"]["max_training_seconds"] = runtime_cap
                base.save(root / "training_runtime_config.json", runtime_config)
                admission = dict(at=base.now(), source_config_sha256=base.sha(root / "training_config.json"),
                    runtime_config_sha256=base.sha(root / "training_runtime_config.json"),
                    only_changed_field="training.max_training_seconds", configured_cap_seconds=GPU_SECONDS,
                    runtime_cap_seconds=runtime_cap, cumulative_cpu_seconds=job["cpu_seconds"],
                    cumulative_gpu_seconds=job["gpu_seconds"], startup_and_boundary_reserve_seconds=60.,
                    conservative_training_projection_seconds=job["training_projection_seconds"],
                    planned_optimizer_updates=3000, data_model_loss_batch_seed_changed=False)
                base.save(root / "training_admission.json", admission)
                for filename in ("training_runtime_config.json", "training_admission.json"):
                    pins[str(root / filename)] = base.sha(root / filename)
                job["training_admission"] = admission
                command = list(command)
                command[command.index("--config")+1] = str(root / "training_runtime_config.json")
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
                if name == "training":
                    job["formal_training_started"] = True
                while active.poll() is None:
                    child_status(name, stage)
                    checkpoint()
                    if not admitted():
                        job.update(requires_user_action=True, budget_exceeded_with_child_running=True,
                            budget_note="Running child preserved; no kill/restart. No further stage will be admitted.")
                    time.sleep(5)
                rc = active.returncode
            child_status(name, stage)
            elapsed = time.monotonic() - stage["monotonic_start"]
            if name == "training":
                gpu_seconds += elapsed
            state = "complete" if rc == 0 else "budget_hold" if rc == 3 else "scientific_stop" if rc == 4 else "failed"
            stage.update(status=state, returncode=rc, ended_at=base.now(), wall_seconds=elapsed)
            active, active_stage = None, None
            verify()
            if rc:
                job.update(status=state, workflow_complete=(name == "sampler_pilot" and rc == 4),
                    current=None, returncode=rc, requires_user_action=True,
                    error=f"{name} exited {rc}; completed outputs preserved; no retry/adaptation", ended_at=base.now())
                checkpoint()
                _public_files(root)
                return rc
            if name == "sampler_pilot":
                job.update(large.validate_pilot(root, sampling_plan))
                job["actual_endpoint_pilot_force_site_evaluations"] = sum(
                    row["actual_work"]["force_site_evaluations"] for row in _read(root / "sampler_pilot.json")["rows"])
                pins[str(root / "sampler_pilot.json")] = base.sha(root / "sampler_pilot.json")
            elif name == "sampling":
                job.update(large.validate_sampling(root, sampling_plan))
                job["actual_production_force_site_evaluations"] = _read(root / "data/manifest.json")["total_force_site_evaluations"]
                job["actual_total_force_site_evaluations_including_endpoint_pilot"] = (
                    job["actual_production_force_site_evaluations"] + job["actual_endpoint_pilot_force_site_evaluations"])
                job["remaining_production_cpu_projection_seconds"] = 0.
                pins[str(root / "data/manifest.json")] = base.sha(root / "data/manifest.json")
            else:
                base.validate_training(root, job["data_manifest_sha256"])
                job["formal_optimizer_updates"] = 3000
            checkpoint()
        if not admitted():
            return hold("Completed outputs exceed newly registered cumulative resources")
        job.update(status="complete", workflow_complete=True, current=None, returncode=0,
            error=None, requires_user_action=False, ended_at=base.now())
        checkpoint()
        _public_files(root, completed=True)
        return 0
    except BaseException as exc:
        if active is not None and active.poll() is None:
            job.update(status="waiting_running_child_after_parent_error", requires_user_action=True,
                error=f"{type(exc).__name__}: {exc}")
            checkpoint()
            active.wait()  # Never kill, retry, resume or migrate a running child.
        if active is not None and active_stage is not None:
            elapsed = time.monotonic() - active_stage["monotonic_start"]
            if active_stage["name"] == "training":
                gpu_seconds += elapsed
            child_status(active_stage["name"], active_stage)
            active_stage.update(status="child_exited_after_parent_error", returncode=active.returncode,
                ended_at=base.now(), wall_seconds=elapsed)
            active, active_stage = None, None
        job.update(status="failed", workflow_complete=False, current=None, returncode=1,
            requires_user_action=True, error=f"{type(exc).__name__}: {exc}", ended_at=base.now())
        checkpoint()
        _public_files(root)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--sampling-plan", default=large.DEFAULT_SAMPLING_PLAN)
    parser.add_argument("--gpu-index", type=int, default=3)
    parser.add_argument("--gpu-uuid", default=GPU_UUID)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    values = (args.prior_root, args.output, args.config, args.gpu_index, args.gpu_uuid, args.sampling_plan)
    if args.execute:
        return run(*values)
    print(json.dumps(plan(*values), ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
