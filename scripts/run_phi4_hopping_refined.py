"""Fresh fixed-step refinement after the documented HMC overflow; dry by default.

Only the authorized psi integrator .08x12 -> .04x24 and the derived force-work
cap change. Historical evidence, targets, production RNGs, models and losses are
immutable. A CPU-only fixed pilot must pass before production is admitted.
"""
from __future__ import annotations

import argparse
import copy
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

from scripts import run_phi4_hopping_qcr as base
from scripts import continue_phi4_hopping_budget as previous

CPU_SECONDS, GPU_SECONDS = 21600., 14400.
GPU_UUID = previous.GPU_UUID
PREFLIGHT_SHA256 = previous.PREFLIGHT_SHA256
PILOT_BURN_IN, PILOT_SAMPLES, PILOT_SPACING = 128, 32, 1
PILOT_MIN_ACCEPTANCE = .6


def _read(path):
    return json.loads(Path(path).read_text())


def _code_root():
    return Path(__file__).resolve().parents[1]


def physical_options():
    from recurrent_rg.data.phi4_hopping_refined_pipeline import Phi4HoppingRefinedOptions
    return Phi4HoppingRefinedOptions()


def pilot_cases():
    from recurrent_rg.data.phi4_hopping_pipeline import chain_seed
    options = physical_options()
    rows = []
    for index, kappa in enumerate(options.kappa_values):
        for initial in ("hot", "cold_positive"):
            token = json.dumps(["phi4_hopping_fixed_refinement_pilot_v1", options.seed,
                                options.dimension, 36, kappa, options.quartic_lambda, initial])
            seed = int.from_bytes(hashlib.sha256(token.encode()).digest()[:16], "little")
            rows.append(dict(kappa=kappa, initial=initial, seed=seed,
                             shared_initial_stream_regression=False,
                             pilot_not_independent_of_future_chain=False))
    # Test precisely the initial stream which overflowed, with only the NEW
    # integrator. Its data are excluded from production; it is not independent.
    rows.append(dict(kappa=options.kappa_values[0], initial="hot",
        seed=chain_seed(options, 36, 0, 1), shared_initial_stream_regression=True,
        pilot_not_independent_of_future_chain=True))
    return rows


def pilot_plan():
    cases = pilot_cases()
    cycles = (PILOT_BURN_IN+PILOT_SAMPLES*PILOT_SPACING)*len(cases)
    return dict(cases=cases, size=36, dimension=2, quartic_lambda=1., h=0.,
        step_size=.04, leapfrog_steps=24, trajectory_length=.96,
        burn_in=PILOT_BURN_IN, samples=PILOT_SAMPLES, spacing=PILOT_SPACING,
        minimum_production_acceptance=PILOT_MIN_ACCEPTANCE,
        total_hybrid_cycles=cycles, total_force_site_evaluations=cycles*25*36**2,
        seed_purpose_separated_cases=26, shared_initial_stream_regression_cases=1,
        arrays_reused_in_production=False, model_updates=0, cuda_used=False,
        adaptation=False, equilibrium_certified=False,
        qualification="短CPU稳定性/接受率检查，不认证热化；共享失败初态回归链不独立于未来生产链；所有pilot不入训练数据")


def pilot(output):
    """No GPU, training, kernel changes, clipping, adaptation or retry."""
    import numpy as np
    from recurrent_rg.data import phi4_hopping as kernel
    from recurrent_rg.data.phi4_hopping_pipeline import WORK_FIELDS, source_pins
    path = Path(output)
    if path.exists():
        raise FileExistsError("Pilot is fresh-only")
    definition, pins = pilot_plan(), source_pins()
    receipt = dict(status="running", workflow_complete=False, returncode=None,
        error=None, requires_user_action=False, plan=definition, rows=[], source_pins=pins,
        started_at=base.now(), equilibrium_certified=False)
    base.save(path, receipt)
    started = time.process_time()
    try:
        for index, case in enumerate(definition["cases"]):
            result = kernel.sample_chain(case["kappa"], 1., size=36, dimension=2,
                samples=PILOT_SAMPLES, burn_in=PILOT_BURN_IN, spacing=PILOT_SPACING,
                step_size=.04, leapfrog_steps=24, seed=case["seed"],
                initial=case["initial"], h=0., cluster_probability=.5)
            finite = all(np.isfinite(result[name]).all() for name in
                ("configurations", "raw_observables", "delta_h", "trajectory_magnetization"))
            acceptance = float(result["production_acceptance"])
            row = dict(case, finite=finite, production_acceptance=acceptance,
                burn_in_acceptance=result["burn_in_acceptance"],
                max_absolute_delta_h=result["max_absolute_delta_h"],
                accepted=np.asarray(result["accepted"]).tolist(),
                delta_h=np.asarray(result["delta_h"]).tolist(),
                diagnostics=result["diagnostics"], warnings=result["warnings"],
                actual_work={name: int(result[name]) for name in WORK_FIELDS})
            receipt["rows"].append(row)
            receipt.update(completed_cases=index+1, cpu_seconds=time.process_time()-started)
            base.save(path, receipt)
            print(json.dumps(dict(event="sampler_pilot_chain_complete", case=index,
                kappa=case["kappa"], initial=case["initial"], acceptance=acceptance,
                shared_initial_stream_regression=case["shared_initial_stream_regression"])), flush=True)
            if not finite or not math.isfinite(acceptance) or acceptance < PILOT_MIN_ACCEPTANCE:
                receipt.update(status="scientific_stop", workflow_complete=True,
                    returncode=4, requires_user_action=True,
                    error="Fixed sampler pilot failed finite/acceptance gate; no production, no adaptation")
                break
        else:
            if source_pins() != pins:
                raise RuntimeError("Sampler source fingerprint changed during pilot")
            receipt.update(status="complete", workflow_complete=True, returncode=0,
                           requires_user_action=False, error=None)
    except Exception as exc:
        receipt.update(status="failed", workflow_complete=False, returncode=1,
                       requires_user_action=True, error=f"{type(exc).__name__}: {exc}")
    receipt.update(cpu_seconds=time.process_time()-started, ended_at=base.now())
    base.save(path, receipt)
    return receipt["returncode"]


def _validate_pilot(root):
    from recurrent_rg.data.phi4_hopping_pipeline import source_pins
    receipt = _read(Path(root)/"sampler_pilot.json")
    rows = receipt.get("rows", [])
    if (receipt.get("status") != "complete" or receipt.get("workflow_complete") is not True
            or receipt.get("returncode") != 0 or receipt.get("error") is not None
            or receipt.get("requires_user_action") is not False
            or receipt.get("source_pins") != source_pins()
            or receipt.get("plan") != pilot_plan() or len(rows) != 27
            or any(not row.get("finite") or not math.isfinite(row.get("production_acceptance", math.nan))
                   or row["production_acceptance"] < PILOT_MIN_ACCEPTANCE for row in rows)
            or any({key: row.get(key) for key in case} != case
                   for row, case in zip(rows, pilot_cases()))
            or sum(row.get("actual_work",{}).get("hybrid_cycles",-1) for row in rows) != 4320
            or sum(row.get("actual_work",{}).get("force_site_evaluations",-1) for row in rows) != 139968000
            or any(row.get("actual_work",{}).get("hybrid_cycles") != 160
                   or row.get("actual_work",{}).get("hmc_trajectories") != 160
                   or row.get("actual_work",{}).get("force_site_evaluations") != 5184000 for row in rows)):
        raise RuntimeError("Pilot returned zero without its complete fixed 27-case finite/acceptance contract")
    pilot_cpu = receipt.get("cpu_seconds", math.nan)
    if not math.isfinite(pilot_cpu) or pilot_cpu <= 0:
        raise RuntimeError("Pilot has no finite positive measured CPU accounting")
    # CPU pilot time includes the complete fixed HMC/cluster work. The 2x
    # factor is a conservative projection, not a guaranteed wall-clock bound.
    production_cpu = 2*pilot_cpu*106496/4320
    return dict(pilot_sha256=base.sha(Path(root)/"sampler_pilot.json"),
        pilot_min_production_acceptance=min(row["production_acceptance"] for row in rows),
        production_cpu_projection_seconds=production_cpu,
        remaining_production_cpu_projection_seconds=production_cpu,
        production_cpu_projection=dict(pilot_cpu_seconds=pilot_cpu,pilot_hybrid_cycles=4320,
            production_hybrid_cycles=106496,conservative_factor=2.,estimate_not_guarantee=True))


def prerequisites(failed_root, preflight_root, config, gpu_index, gpu_uuid, prior_queue_root=None):
    failed, preflight, code = Path(failed_root).resolve(), Path(preflight_root).resolve(), _code_root()
    job, old_plan = _read(failed/"job.json"), _read(failed/"plan.json")
    data_status = _read(failed/"data/status.json")
    stages = job.get("stages", [])
    if (job.get("status") != "failed" or job.get("workflow_complete") is not False
            or job.get("returncode") != 1 or job.get("current") is not None
            or job.get("requires_user_action") is not True or len(stages) != 1
            or stages[0].get("name") != "sampling" or stages[0].get("returncode") != 1
            or stages[0].get("status") != "failed" or (failed/"training").exists()
            or job.get("plan_sha256") != base.sha(failed/"plan.json")
            or old_plan.get("output") != str(failed) or old_plan.get("host") != base.HOST
            or old_plan.get("GPU") != dict(index=gpu_index, uuid=gpu_uuid)
            or gpu_index != 3 or gpu_uuid != GPU_UUID
            or data_status.get("status") != "failed" or data_status.get("returncode") != 1
            or data_status.get("completed_conditions") != 0
            or data_status.get("error") != "FloatingPointError: overflow encountered in power"):
        raise ValueError("Failed root is not the documented sampling-only overflow with zero complete conditions")
    old_extension = old_plan.get("budget_extension", {})
    if (old_extension.get("prior_root") != str(preflight)
            or old_extension.get("preflight_sha256") != PREFLIGHT_SHA256
            or base.sha(preflight/"gpu_preflight.json") != PREFLIGHT_SHA256
            or base.sha(failed/"reused_preflight.json") != PREFLIGHT_SHA256
            or old_extension.get("prior_plan_sha256") != base.sha(preflight/"plan.json")
            or old_extension.get("prior_job_sha256") != base.sha(preflight/"job.json")
            or _read(config) != _read(old_plan["config_source"])
            or base.sha(config) != old_plan.get("config_source_sha256")):
        raise ValueError("Budget/configuration/synthetic preflight bindings differ from the failed workflow")
    queue = old_extension.get("prior_queue")
    if prior_queue_root is not None and (queue is None or queue.get("root") != str(Path(prior_queue_root).resolve())):
        raise ValueError("Optional prior queue does not match the already charged historical queue")
    # The failed parent already charged both preflight and queue. Never add them again.
    cpu, gpu = float(job.get("cpu_seconds", math.inf)), float(job.get("gpu_seconds", math.inf))
    projection = float(old_extension.get("projection_seconds", math.inf))
    if (not all(math.isfinite(x) and x >= 0 for x in (cpu, gpu, projection))
            or cpu < old_extension.get("prior_cpu_seconds", math.inf)
            or gpu < old_extension.get("prior_gpu_seconds", math.inf)
            or gpu+projection+10 > GPU_SECONDS or cpu+projection > CPU_SECONDS):
        raise ValueError("Cumulative prior use plus unchanged conservative projection exceeds authorized resources")
    bindings, old_real = {}, {}
    old_code = Path(old_plan["code_root"]).resolve()
    for filename, digest in old_plan.get("source_sha256", {}).items():
        path = Path(filename)
        if path.name.startswith("._"):
            continue
        if base.sha(path) != digest:
            raise ValueError(f"Prior pinned source/configuration/evidence changed: {path}")
        bindings[str(path)] = digest
        if old_code in path.parents and path.suffix == ".py":
            relative = str(path.relative_to(old_code))
            if not (code/relative).is_file() or base.sha(code/relative) != digest:
                raise ValueError(f"Computational source changed: {relative}")
            old_real[relative] = digest
    allowed = {"scripts/run_phi4_hopping_refined.py", "scripts/prepare_phi4_hopping_refined.py",
               "recurrent_rg/data/phi4_hopping_refined_pipeline.py"}
    if not old_real or set(previous._real_python_files(code))-allowed != set(old_real)-allowed:
        raise ValueError("Computational source inventory differs beyond the new orchestration wrapper")
    for folder, names in ((failed, ("job.json", "plan.json", "training_config.json",
            "reused_preflight.json", "budget_extension_receipt.json", "data/status.json")),
            (preflight, ("job.json", "plan.json", "gpu_preflight.json"))):
        for name in names:
            bindings[str(folder/name)] = base.sha(folder/name)
    return dict(failed_root=str(failed), preflight_root=str(preflight), prior_queue=queue,
        prior_cpu_seconds=cpu, prior_gpu_seconds=gpu, projection_seconds=projection,
        accounting="failed root cumulative ledger already includes preflight and cancelled queue; no double charge",
        prior_source_bindings=bindings, executable_source_files=len(old_real),
        additional_benchmark_updates=0, checkpoint_loaded=False, formal_model_reused=False,
        failed_data_reused=False, old_root_modified=False, preflight_sha256=PREFLIGHT_SHA256)


def plan(failed_root, preflight_root, output, config, gpu_index=3, gpu_uuid=GPU_UUID, prior_queue_root=None):
    from experiments.phi4_hopping_qcr import TrainingOptions, load_config, training_plan
    from recurrent_rg.data.phi4_hopping_pipeline import Phi4HoppingOptions, generation_plan
    from recurrent_rg.data.phi4_hopping_refined_pipeline import generation_plan as refined_plan
    root, code, config = Path(output).resolve(), _code_root(), Path(config).resolve()
    if root.exists() or root == code or root in code.parents or code in root.parents:
        raise ValueError("Require a fresh output separate from all frozen code/evidence roots")
    for folder in (failed_root, preflight_root, prior_queue_root):
        if folder is not None:
            prior_folder = Path(folder).resolve()
            if root == prior_folder or root in prior_folder.parents or prior_folder in root.parents:
                raise ValueError("Require a fresh output separate from all frozen code/evidence roots")
    evidence = prerequisites(failed_root, preflight_root, config, gpu_index, gpu_uuid, prior_queue_root)
    old_physical = _read(Path(failed_root)/"plan.json")["physical"]
    original = json.loads(json.dumps(generation_plan(Phi4HoppingOptions())))
    if original != old_physical:
        raise ValueError("Original physical target/options/source pins differ from the failed workflow")
    values = load_config(config)
    values.update(data=str(root/"data"), output=str(root/"training"), device="cuda:0", expected_gpu_uuid=gpu_uuid)
    training = training_plan(TrainingOptions(**values))
    command = base.commands(root, config, gpu_uuid)
    del command["gpu_preflight"]
    command["sampling"][3] = "scripts.prepare_phi4_hopping_refined"
    command["sampling"] += ["--step-size", ".04", "--leapfrog-steps", "24",
                            "--max-force-site-evaluations", "4000000000"]
    command = dict(sampler_pilot=[sys.executable, "-u", "-m", "scripts.run_phi4_hopping_refined",
                                  "--pilot-output", str(root/"sampler_pilot.json")], **command)
    return dict(schema_version=5, created_at=base.now(), execution_requested=False, code_root=str(code),
        output=str(root), host=base.HOST, config_source=str(config), config_source_sha256=base.sha(config),
        GPU=dict(index=gpu_index, uuid=gpu_uuid), physical=refined_plan(physical_options()),
        pilot=pilot_plan(), training=training, commands=command,
        stages=["sampler_pilot", "sampling", "training"], refinement=evidence,
        authorized_changes=dict(step_size=[.08,.04], leapfrog_steps=[12,24], trajectory_length=.96,
                                max_force_site_evaluations=[2000000000,4000000000]),
        resources=dict(cpu_seconds=CPU_SECONDS, gpu_card_seconds=GPU_SECONDS,
            max_training_seconds=GPU_SECONDS, max_cuda_bytes=base.PEAK_BYTES,
            directory_bytes=base.DIRECTORY_BYTES, cpu_threads=1, nice=10,
            min_memory_bytes=36*1024**3, max_load_per_cpu=1.25, max_io_full_avg10=20.,
            idle_check_spacing_seconds=10., min_gpu_free_bytes=12*1024**3, require_no_compute_processes=True),
        no_resume=True, no_retry=True, no_auto_repair=True, no_migration=True,
        qualification="仅授权的固定积分细化；短pilot不是热化证明；生产质量警告保留；原失败数据不复用；不认证临界指数")


def _old_process_running(pid, root):
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except (FileNotFoundError, ProcessLookupError):
        return False
    target = str(Path(root).resolve())
    matches_root = any(token == target or token.startswith(target+"/") for token in command.split())
    return matches_root and any(token in command for token in
        ("continue_phi4_hopping_budget", "run_phi4_hopping_qcr", "prepare_phi4_hopping", "recurrent_rg"))


def _public_files(root):
    public = {name: base.sha(root/name) for name in ("job.json", "plan.json", "training_config.json",
        "reused_preflight.json", "refinement_receipt.json", "sampler_pilot.json")}
    for name in _read(root/"data/manifest.json")["public_files"]:
        path = (root/"data"/name).resolve()
        if root/"data" not in path.parents or path.suffix != ".json":
            raise RuntimeError("Invalid public physical metadata path")
        public[f"data/{name}"] = base.sha(path)
    public["training/status.json"] = base.sha(root/"training/status.json")
    for row in _read(root/"training/status.json")["public_files"]:
        path = (root/"training"/row["path"]).resolve()
        if (root/"training" not in path.parents or path.suffix not in (".json", ".png", ".npz")
                or base.sha(path) != row["sha256"]):
            raise RuntimeError("Invalid public training artifact binding")
        public[f"training/{row['path']}"] = row["sha256"]
    base.save(root/"public_files.json", public)


def run(failed_root, preflight_root, output, config, gpu_index=3, gpu_uuid=GPU_UUID, prior_queue_root=None):
    definition = plan(failed_root, preflight_root, output, config, gpu_index, gpu_uuid, prior_queue_root)
    if socket.gethostname() != base.HOST or os.getpriority(os.PRIO_PROCESS, 0) != 10:
        raise ValueError("Execution requires pinned 225 host and one inherited nice10 layer")
    for old_root in (failed_root, preflight_root):
        old_job = _read(Path(old_root)/"job.json")
        if any(_old_process_running(pid, old_root) for pid in
               [old_job.get("parent_pid"), *[row.get("pid") for row in old_job["stages"]]]):
            raise ValueError("An actual prior workflow process remains active; no duplicate execution")
    root, code, prior = Path(output).resolve(), _code_root(), definition["refinement"]
    pins = {str(path): base.sha(path) for path in previous._real_python_files(code).values()}
    pins.update(prior["prior_source_bindings"])
    pins[str(Path(config).resolve())] = base.sha(config)
    root.mkdir(parents=True, exist_ok=False)
    (root/"logs").mkdir()
    configuration = copy.deepcopy(_read(config))
    configuration.update(data=str(root/"data"), output=str(root/"training"))
    configuration["training"].update(device="cuda:0", expected_gpu_uuid=gpu_uuid)
    base.save(root/"training_config.json", configuration)
    (root/"reused_preflight.json").write_bytes((Path(preflight_root)/"gpu_preflight.json").read_bytes())
    base.save(root/"refinement_receipt.json", prior)
    for name in ("training_config.json", "reused_preflight.json", "refinement_receipt.json"):
        pins[str(root/name)] = base.sha(root/name)
    base.save(root/"plan.json", dict(definition, execution_requested=True, source_sha256=pins))
    pins[str(root/"plan.json")] = base.sha(root/"plan.json")
    job = dict(status="prepared", workflow_complete=False, parent_pid=os.getpid(), current=None,
        returncode=None, error=None, requires_user_action=False, started_at=base.now(), stages=[],
        plan_sha256=base.sha(root/"plan.json"), cpu_seconds=prior["prior_cpu_seconds"],
        gpu_seconds=prior["prior_gpu_seconds"], prior_cpu_seconds=prior["prior_cpu_seconds"],
        prior_gpu_seconds=prior["prior_gpu_seconds"], additional_benchmark_updates=0)
    cpu_start, child_start = time.process_time(), resource.getrusage(resource.RUSAGE_CHILDREN)
    gpu_seconds, active, active_stage = prior["prior_gpu_seconds"], None, None

    def checkpoint():
        children = resource.getrusage(resource.RUSAGE_CHILDREN)
        job["cpu_seconds"] = (prior["prior_cpu_seconds"]+time.process_time()-cpu_start
            + children.ru_utime-child_start.ru_utime+children.ru_stime-child_start.ru_stime)
        if active is not None and active.poll() is None:
            try:
                fields = Path(f"/proc/{active.pid}/stat").read_text().split(")",1)[1].split()
                job["cpu_seconds"] += (int(fields[11])+int(fields[12]))/os.sysconf("SC_CLK_TCK")
            except (FileNotFoundError, ProcessLookupError):
                pass
        job["gpu_seconds"] = gpu_seconds+(time.monotonic()-active_stage["monotonic_start"]
            if active is not None and active.poll() is None and active_stage["name"] == "training" else 0.)
        job["directory_bytes"] = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
        base.save(root/"job.json", job)

    def verify():
        if any(base.sha(path) != digest for path, digest in pins.items()):
            raise RuntimeError("Pinned source/configuration/prior evidence changed; no automatic repair")

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
            second = dict(host=base.host_resources(), gpu=base.gpu_resources(gpu_index,gpu_uuid))
            spacing = time.monotonic()-started
            gpu_seconds += spacing
            job["gpu_idle_checks"] = dict(first=first, second=second, spacing_seconds=spacing)
            checkpoint()
            if spacing >= 10 and base.ready(second,True):
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
            projection = prior["projection_seconds"]
            remaining_cpu = job.get("remaining_production_cpu_projection_seconds", 0.)
            if (job["cpu_seconds"]+remaining_cpu+projection > CPU_SECONDS or job["gpu_seconds"]+projection > GPU_SECONDS
                    or job["directory_bytes"] > base.DIRECTORY_BYTES):
                return hold("Cumulative use plus unchanged conservative training projection exceeds budget")
            stage = dict(name=name, status="running", returncode=None, started_at=base.now(),
                         command=command, monotonic_start=time.monotonic())
            active_stage = stage
            job["stages"].append(stage)
            job.update(status="running", current=name)
            with (root/"logs"/f"{name}.log").open("xb") as log:
                active = subprocess.Popen(command, cwd=code,
                    env=base.child_environment(gpu_uuid if name == "training" else None),
                    stdout=log, stderr=subprocess.STDOUT)
                stage["pid"] = active.pid
                while active.poll() is None:
                    child_path = (root/"sampler_pilot.json" if name == "sampler_pilot" else
                                  root/("data" if name == "sampling" else "training")/"status.json")
                    try:
                        stage["child_status"] = _read(child_path)
                    except (FileNotFoundError,json.JSONDecodeError):
                        pass
                    checkpoint()
                    time.sleep(5)
                rc = active.returncode
            elapsed = time.monotonic()-stage["monotonic_start"]
            if name == "training":
                gpu_seconds += elapsed
            stage.update(status="complete" if rc == 0 else "budget_hold" if rc == 3 else
                         "scientific_stop" if rc == 4 else "failed", returncode=rc,
                         ended_at=base.now(), wall_seconds=elapsed)
            active, active_stage = None, None
            verify()
            if rc:
                job.update(status=stage["status"], current=None, returncode=rc, requires_user_action=True,
                    error=f"{name} exited {rc}; completed outputs preserved; no adaptation", ended_at=base.now())
                checkpoint()
                return rc
            if name == "sampler_pilot":
                job.update(_validate_pilot(root))
                pins[str(root/"sampler_pilot.json")] = base.sha(root/"sampler_pilot.json")
            elif name == "sampling":
                job.update(previous._validate_sampling(root))
                job["remaining_production_cpu_projection_seconds"] = 0.
            else:
                base.validate_training(root,job["data_manifest_sha256"])
            checkpoint()
        if (job["cpu_seconds"] > CPU_SECONDS or job["gpu_seconds"] > GPU_SECONDS
                or job["directory_bytes"] > base.DIRECTORY_BYTES):
            return hold("Completed outputs exceed registered cumulative resources")
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
            active.wait()  # Never kill another or our own running child.
        job.update(status="failed", workflow_complete=False, current=None, returncode=1,
            requires_user_action=True, error=f"{type(exc).__name__}: {exc}", ended_at=base.now())
        checkpoint()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-output")
    parser.add_argument("--failed-root")
    parser.add_argument("--preflight-root")
    parser.add_argument("--prior-queue-root")
    parser.add_argument("--output")
    parser.add_argument("--config",default="configs/phi4_hopping_b3_g1_aligned_seed91_budget4h.json")
    parser.add_argument("--gpu-index",type=int,default=3)
    parser.add_argument("--gpu-uuid",default=GPU_UUID)
    parser.add_argument("--execute",action="store_true")
    args = parser.parse_args(argv)
    if args.pilot_output is not None:
        if args.execute or any((args.failed_root,args.preflight_root,args.output,args.prior_queue_root)):
            parser.error("Internal pilot-output mode cannot be combined with parent execution arguments")
        return pilot(args.pilot_output)
    if not all((args.failed_root,args.preflight_root,args.output)):
        parser.error("Require --failed-root, --preflight-root and --output")
    values = (args.failed_root,args.preflight_root,args.output,args.config,
              args.gpu_index,args.gpu_uuid,args.prior_queue_root)
    if args.execute:
        return run(*values)
    print(json.dumps(plan(*values),ensure_ascii=False,indent=2,allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
