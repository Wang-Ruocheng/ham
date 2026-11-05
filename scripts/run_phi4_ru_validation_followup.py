"""固定 Q/C 的新物理留出实验：只读计划或顺序采样、采集与评价。

每个子数据集复用历史四链生成器，独立目录保存；不覆盖、恢复、重试、
修改冻结采样器或终止子进程。所有新增参数点与独立链只用于评价。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone


HOST = "thinklab-105-225"
SCHEMA = "phi4_ru_independent_physical_validation_v1"
CPU_BUDGET_SECONDS = 7200.
OUTPUT_BUDGET_BYTES = 2 * 1024**3
SOURCE_CPU_SECONDS = 982.4789274780001
SOURCE_CHAIN_COUNT = 180
SOURCE_STATE_SHA256 = "6ea4fef61378dd0e33fa6cf86bf53230dbc5303516d7a3832c88ba77c5c58c2d"
SOURCE_CHECKPOINT_SHA256 = "b36dd4a10fa0936075e08fe69426e4bffdc76a88e75a6afc07fde1dce4ffdf7c"
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
BANDS = ((.5, -.7625, -.63125), (.625, -.89375, -.7625),
         (.75, -1.025, -.89375), (.875, -1.15625, -1.025),
         (1., -1.2875, -1.15625))


class BudgetHold(RuntimeError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, value, *, exclusive=False):
    path = Path(path)
    if exclusive:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
    else:
        temporary = path.with_name(path.name + ".partial")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)


def inside(root, relative):
    root = Path(root).resolve()
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("Public path must be relative and remain inside its root")
    path = (root / candidate).resolve()
    if root not in path.parents or path.is_symlink():
        raise ValueError("Public path escapes its root")
    return path


def verify_public(root, *, required=None):
    """Verify the completed upstream stage without interpreting TEST metrics."""
    root = Path(root).resolve()
    status = read(root / "status.json")
    if (status.get("status") != "complete" or status.get("workflow_complete") is not True
            or status.get("returncode") != 0 or status.get("error") is not None):
        raise ValueError("R validation must be complete before any new physical sampling")
    inventory = read(root / "public_files.json")
    files = inventory.get("files", inventory)
    if not isinstance(files, dict):
        raise ValueError("Upstream public inventory must map relative paths to SHA256")
    required = ({"manifest.json", "summary.json", "R_parameters.npz", "status.json"}
                if required is None else set(required))
    # Status is finalized after the public inventory in the existing writers.
    # Validate/pin it directly instead of requiring a circular inventory entry.
    if not (required - {"status.json"}).issubset(files):
        raise ValueError("R validation inventory lacks fixed public model inputs")
    result = {str(root / "public_files.json"): sha(root / "public_files.json"),
              str(root / "status.json"): sha(root / "status.json")}
    for relative, value in files.items():
        path = inside(root, relative)
        if path.suffix.lower() not in (".json", ".npz", ".png"):
            raise ValueError("Only upstream public JSON/NPZ/PNG files are accepted")
        expected = value.get("sha256") if isinstance(value, dict) else value
        if not isinstance(expected, str) or sha(path) != expected:
            raise RuntimeError(f"Upstream public fingerprint mismatch: {relative}")
        result[str(path)] = expected
    return result


def fixed_subsets():
    """13 four-chain datasets, 24 parameter/size conditions, 156 physical chains."""
    rows = []
    for replicate, seed in enumerate((2026103001, 2026103002)):
        for u_index, (u, low, high) in enumerate(BANDS):
            rows.append(dict(name=f"L36_u{u_index:03d}_rep{replicate}", size=36,
                u_index=u_index, u=u, r_values=[low, .5 * (low + high), high],
                seed=seed, replicate=replicate))
    for u_index in (0, 2, 4):
        u, low, high = BANDS[u_index]
        rows.append(dict(name=f"L108_u{u_index:03d}_rep0", size=108,
            u_index=u_index, u=u, r_values=[low, .5 * (low + high), high],
            seed=2026103003, replicate=0))
    # Check the new lattice size early using the already planned chains.
    return sorted(rows, key=lambda row: (-row["size"], row["replicate"], row["u_index"]))


def build_plan(output, source_root, validation, *, code_root=None):
    """No filesystem writes, model loads, RNG draws or resource allocation."""
    from recurrent_rg.data.phi4_ru_pipeline import Phi4RUOptions, generation_plan, source_pins
    code = Path(code_root or Path(__file__).resolve().parents[1]).resolve()
    output, source_root, validation = map(lambda p: Path(p).resolve(),
                                          (output, source_root, validation))
    if output == source_root or source_root in output.parents or output in source_root.parents:
        raise ValueError("New output must be outside and distinct from the frozen source")
    if output == code or output in code.parents:
        raise ValueError("Output must not contain the implementation checkout")
    subsets = []
    for row in fixed_subsets():
        options = Phi4RUOptions(sizes=(row["size"],), r_values=tuple(row["r_values"]),
            u_values=(row["u"],), chains=4, samples=128, burn_in=1024, spacing=8,
            step_size=.08, leapfrog_steps=12, seed=row["seed"], test_every=3, test_offset=1)
        generated = json.loads(json.dumps(generation_plan(options)))
        subsets.append(dict(row, relative_output=f"data/{row['name']}", physical=generated))
    weighted_chains = sum(12 * (row["size"] / 36.)**2 for row in subsets)
    projection = SOURCE_CPU_SECONDS * weighted_chains / SOURCE_CHAIN_COUNT * 1.75
    # Source collection: 1264 rows in 60.657 s. L108 inference includes 9x sites.
    collection = 60.657189751 * (4 * weighted_chains) / 1264 * 2.
    pins = dict(source_pins())
    for relative in ("scripts/run_phi4_ru_validation_followup.py",
                     "scripts/prepare_phi4_ru.py", "experiments/phi4_ru_followup_evaluate.py",
                     "experiments/phi4_ru_r_cache.py", "experiments/phi4_ru_r_calibration.py",
                     "recurrent_rg/phi4_r_calibration.py", "experiments/phi4_qcr.py",
                     "recurrent_rg/phi4_qcr.py", "scripts/report_phi4_qcr.py"):
        path = code / relative
        if path.exists():
            pins[relative] = sha(path)
    return dict(schema_version=5, schema=SCHEMA, output=str(output), source_root=str(source_root),
        validation=str(validation), code_root=str(code), host=HOST, subsets=subsets,
        independent_physical_chains=156, unique_size_parameter_conditions=24,
        microscopic_lattice_sizes=[36, 108], observed_lattice_sizes=[[36, 12, 4], [108, 36, 12]],
        sampler_source_pins=pins, source_checkpoint=str(source_root / "run/training_step_3000.pt"),
        source_state_sha256=SOURCE_STATE_SHA256,
        frozen_R_model_keys=[f"selected_seed{seed}" for seed in (91, 92, 93)],
        fair_baseline_keys=[f"adam6000_seed{seed}" for seed in (91, 92, 93)],
        total_hmc_trajectories=sum(row["physical"]["total_hmc_trajectories"] for row in subsets),
        total_force_site_evaluations=sum(row["physical"]["total_force_site_evaluations"] for row in subsets),
        projected_sampling_output_bytes=sum(row["physical"]["projected_output_bytes"] for row in subsets),
        budget_projection=dict(source_actual_sampling_cpu_seconds=SOURCE_CPU_SECONDS,
            source_independent_chains=SOURCE_CHAIN_COUNT, volume_weighted_equivalent_L36_chains=weighted_chains,
            sampling_safety_multiplier=1.75, sampling_cpu_seconds=projection,
            collection_cpu_seconds=collection, evaluation_and_report_cpu_reserve_seconds=600.,
            total_cpu_seconds=projection + collection + 600.,
            budget_passed=projection + collection + 600. <= CPU_BUDGET_SECONDS,
            projection_is_not_wall_time=True,
            larger_size_throughput_not_measured=True),
        resource_policy=dict(cpu_seconds=CPU_BUDGET_SECONDS, directory_bytes=OUTPUT_BUDGET_BYTES,
            cpu_threads=1, nice=10, gpu_seconds=0., minimum_host_available_gib=36.,
            load_per_cpu_max=1.25, io_full_avg10_percent_max=20.,
            poll_seconds=5., resource_wait_seconds=30., wall_time_is_not_cpu_budget=True,
            cpu_budget_semantics="stage-boundary admission: actual cumulative plus conservative remaining; running child finishes without termination",
            atomic_child_overrun_possible=True),
        stages=[row["name"] for row in subsets] + ["frozen_evaluation"],
        all_new_data_evaluation_only=True, labels_used_in_R_optimization=False,
        upstream_generator_train_test_markers_ignored_for_followup_evaluation=True,
        optimizer_updates=dict(Q=0, C=0, R=0), gpu_used=False,
        warnings_retained_without_resampling=True, equilibrium_certified=False,
        automatic_retry=False, automatic_restart=False, child_termination=False,
        qualification="有限相关新物理链、固定Q/C及预先选定R端点；变化带按旧观察选定，结果为独立链验证而非无条件相变认证。")


def verify_source_and_models(source_root, validation):
    """Resolve source/QC/R identity before spending any physical sampling work."""
    from experiments.phi4_ru_r_cache import _source_contract
    from experiments.phi4_ru_followup_evaluate import restore_maps
    from experiments.phi4_qcr import state_fingerprint
    _, _, summary, pins, code_pins = _source_contract(Path(source_root).resolve())
    checkpoint = Path(source_root).resolve() / "run/training_step_3000.pt"
    upstream_manifest, upstream_summary = read(Path(validation)/"manifest.json"), read(Path(validation)/"summary.json")
    if (summary["final_state_sha256"] != SOURCE_STATE_SHA256
            or sha(checkpoint) != SOURCE_CHECKPOINT_SHA256
            or upstream_manifest.get("source_state_sha256") != SOURCE_STATE_SHA256
            or upstream_manifest.get("source_module_sha256") != summary["final_module_sha256"]):
        raise RuntimeError("Frozen source and R-validation Q/C identities disagree")
    maps = restore_maps(validation)
    for key, model in maps.items():
        if state_fingerprint(model) != upstream_summary["final_R_sha256"][key]:
            raise RuntimeError(f"Fixed validation endpoint identity disagrees: {key}")
    return {**pins, **code_pins}


def stage_commands(plan):
    root = Path(plan["output"])
    result = []
    for row in plan["subsets"]:
        command = [sys.executable, "-u", "-m", "scripts.prepare_phi4_ru", "--execute",
                   "--output", str(root / row["relative_output"])]
        options = row["physical"]["options"]
        for key, value in options.items():
            command.append("--" + key.replace("_", "-"))
            command.extend(str(v) for v in value) if isinstance(value, (list, tuple)) else command.append(str(value))
        result.append((row["name"], command))
    result.append(("frozen_evaluation", [sys.executable, "-u", "-m",
        "experiments.phi4_ru_followup_evaluate", "--source-root", plan["source_root"],
        "--data-root", str(root), "--validation", plan["validation"],
        "--output", str(root / "evaluation")]))
    return result


def directory_bytes(root):
    return sum(path.stat().st_size for path in Path(root).rglob("*") if path.is_file())


def used_cpu_seconds():
    return time.process_time() + resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime


def live_child_cpu(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        # fields[0] is Linux stat field3, utime/stime/cutime/cstime are14..17.
        return sum(float(fields[i]) for i in (11, 12, 13, 14)) / os.sysconf("SC_CLK_TCK")
    except FileNotFoundError:
        return 0.


def resource_snapshot(plan):
    policy = plan["resource_policy"]
    memory = int(re.search(r"MemAvailable:\s+(\d+)", Path("/proc/meminfo").read_text())[1]) * 1024
    io = float(re.search(r"full avg10=([\d.]+)", Path("/proc/pressure/io").read_text())[1])
    cpus, load = os.cpu_count(), os.getloadavg()[0]
    ready = (memory >= policy["minimum_host_available_gib"] * 1024**3
             and load <= policy["load_per_cpu_max"] * cpus and io < policy["io_full_avg10_percent_max"])
    return dict(at=now(), mem_available_bytes=memory, cpu_count=cpus, load1=load,
                io_full_avg10_percent=io, host_ready=ready)


def verify_frozen(plan, input_pins):
    for relative, expected in plan["sampler_source_pins"].items():
        if sha(Path(plan["code_root"]) / relative) != expected:
            raise RuntimeError(f"Frozen implementation changed: {relative}")
    for path, expected in input_pins.items():
        if sha(path) != expected:
            raise RuntimeError(f"Frozen upstream input changed: {path}")


def aggregate_data(plan):
    root, chains, subsets, datasets, condition_ids = Path(plan["output"]), [], [], [], set()
    for row in plan["subsets"]:
        directory = root / row["relative_output"]
        if not (directory / "manifest.json").exists():
            break
        status = read(directory / "status.json")
        if (status.get("status") != "complete" or status.get("workflow_complete") is not True
                or status.get("returncode") != 0 or status.get("error") is not None):
            raise RuntimeError(f"Incomplete physical dataset: {row['name']}")
        manifest = read(directory / "manifest.json")
        if manifest.get("options") != row["physical"]["options"]:
            raise RuntimeError("Actual sampling options differ from the frozen plan")
        if len(manifest["shards"]) != 3:
            raise RuntimeError("Fixed child dataset does not contain exactly three conditions")
        for shard in manifest["shards"]:
            data, sidecar = inside(directory, shard["path"]), inside(directory, shard["sidecar"])
            if sha(data) != shard["sha256"] or sha(sidecar) != shard["sidecar_sha256"]:
                raise RuntimeError("New physical shard fingerprint mismatch")
            cell = int(shard["cell_index"])
            meta = read(sidecar)
            expected_r = row["r_values"][cell]
            if (meta["r"] != expected_r or meta["u"] != row["u"] or meta["h"] != 0.
                    or meta["size"] != row["size"] or meta["independent_chains"] != 4
                    or meta["samples_per_chain"] != 128):
                raise RuntimeError("New physical condition identity differs from the plan")
            for chain_row in meta["chains"]:
                if (not math.isfinite(chain_row["production_acceptance"])
                        or chain_row["production_acceptance"] <= 0):
                    raise RuntimeError("Zero or nonfinite HMC acceptance; preserving all files")
            condition_id = f"L{row['size']}_u{row['u_index']:03d}_r{cell:02d}"
            condition_ids.add(condition_id)
            for local_chain in range(4):
                chains.append(dict(global_chain_index=len(chains), condition_id=condition_id,
                    size=row["size"], r=expected_r, u=row["u"], h=0.,
                    subset=row["name"], generator_cell_index=cell, generator_chain_index=local_chain,
                    physical_seed_namespace=row["seed"], physical_chain_seed=meta["chains"][local_chain]["seed"],
                    data_path=str(data.relative_to(root)), data_sha256=shard["sha256"],
                    sidecar_path=str(sidecar.relative_to(root)), sidecar_sha256=shard["sidecar_sha256"],
                    split="evaluation", generator_split=shard["split"]))
        subsets.append(dict(name=row["name"], relative_output=row["relative_output"],
            manifest_sha256=sha(directory / "manifest.json"), status_sha256=sha(directory / "status.json"),
            warning_count=manifest["warning_count"]))
        datasets.append({key: row[key] for key in ("name", "size", "u", "r_values", "replicate", "seed")})
        datasets[-1].update(root=str(directory), manifest_sha256=sha(directory / "manifest.json"))
    return dict(schema=SCHEMA, complete=len(subsets) == 13, subsets=subsets, datasets=datasets, chains=chains,
        independent_physical_chains=len(chains), unique_size_parameter_conditions=len(condition_ids),
        all_evaluation_only=True, labels_used_in_training=False,
        sampler_warnings_retained=True, equilibrium_certified=False)


def run(output, source_root, validation):
    root = Path(output).resolve()
    plan = build_plan(root, source_root, validation)
    if socket.gethostname() != HOST:
        raise RuntimeError(f"Physical execution is pinned to {HOST}")
    if any(os.environ.get(name) != "1" for name in THREAD_VARIABLES):
        raise RuntimeError("Launch with all recorded CPU-thread variables set to1")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        raise RuntimeError("This CPU-only followup must not expose CUDA devices")
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        os.nice(10 - os.getpriority(os.PRIO_PROCESS, 0))
    input_pins = verify_public(validation)
    input_pins.update(verify_source_and_models(source_root, validation))
    checkpoint = Path(plan["source_checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError("Frozen source checkpoint remains required on its original host")
    input_pins[str(checkpoint)] = sha(checkpoint)
    for name in ("run/manifest.json", "run/summary.json", "run/status.json", "data/manifest.json"):
        path = Path(source_root) / name
        input_pins[str(path.resolve())] = sha(path)
    # Input R validation may already have been copied inside this otherwise fresh root.
    if root.exists():
        allowed = {Path(validation).resolve()} if root == Path(validation).resolve().parent else set()
        if any(path.resolve() not in allowed for path in root.iterdir()):
            raise FileExistsError("Existing experiment output is never overwritten or resumed")
    else:
        root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    (root / "data").mkdir()
    plan.update(prepared_at=now(), frozen_input_sha256=input_pins, actual_nice=os.getpriority(os.PRIO_PROCESS, 0))
    save(root / "plan.json", plan, exclusive=True)
    job = dict(schema=SCHEMA, status="prepared", workflow_complete=False, current=None,
        stages=[], returncode=None, error=None, requires_user_action=False,
        parent_pid=os.getpid(), started_at=now(), actual_nice=plan["actual_nice"],
        plan_sha256=sha(root / "plan.json"), cpu_seconds=0., gpu_seconds=0.)
    start_cpu, start_wall = used_cpu_seconds(), time.monotonic()

    def update(state=None, **more):
        if state is not None:
            job["status"] = state
        job.update(more, cpu_seconds=used_cpu_seconds() - start_cpu,
                   wall_seconds=time.monotonic() - start_wall)
        save(root / "job.json", job)
        save(root / "status.json", {key: job.get(key) for key in (
            "status", "workflow_complete", "current", "returncode", "error",
            "requires_user_action", "cpu_seconds", "wall_seconds", "gpu_seconds")})

    def budget_check(*, before_stage=False):
        actual_cpu = used_cpu_seconds() - start_cpu
        if actual_cpu >= CPU_BUDGET_SECONDS:
            raise BudgetHold("Cumulative CPU budget reached; no stage is retried or reduced")
        if directory_bytes(root) > OUTPUT_BUDGET_BYTES:
            raise BudgetHold("Cumulative output budget reached; all outputs preserved")
        if before_stage:
            completed = {row["name"] for row in job["stages"] if row["status"] == "complete"}
            weighted_remaining = sum(12 * (row["size"] / 36.)**2
                for row in plan["subsets"] if row["name"] not in completed)
            projection = plan["budget_projection"]
            remaining = (projection["sampling_cpu_seconds"] * weighted_remaining /
                         projection["volume_weighted_equivalent_L36_chains"])
            if "frozen_evaluation" not in completed:
                remaining += projection["collection_cpu_seconds"] + projection["evaluation_and_report_cpu_reserve_seconds"]
            job["remaining_conservative_cpu_seconds"] = remaining
            if actual_cpu + remaining > CPU_BUDGET_SECONDS:
                raise BudgetHold("Actual cumulative CPU plus frozen remaining projection exceeds budget")

    update()
    try:
        if not plan["budget_projection"]["budget_passed"]:
            raise BudgetHold("Frozen conservative CPU projection does not fit the cumulative budget")
        for name, command in stage_commands(plan):
            verify_frozen(plan, input_pins)
            budget_check(before_stage=True)
            while True:
                snapshot = resource_snapshot(plan)
                job.setdefault("resource_checks", []).append(snapshot)
                job["resource_checks"] = job["resource_checks"][-30:]
                if snapshot["host_ready"]:
                    break
                update("waiting_resources", current=name)
                time.sleep(plan["resource_policy"]["resource_wait_seconds"])
                budget_check(before_stage=True)
            verify_frozen(plan, input_pins)
            budget_check(before_stage=True)
            update("running", current=name)
            before_cpu, before_wall = used_cpu_seconds(), time.monotonic()
            stage = dict(name=name, command=command, status="running", returncode=None,
                started_at=now(), log=f"logs/{name}.log")
            job["stages"].append(stage)
            with (root / stage["log"]).open("x", encoding="utf-8") as log:
                environment = dict(os.environ, CUDA_VISIBLE_DEVICES="")
                with subprocess.Popen(command, cwd=plan["code_root"], env=environment,
                                      stdout=log, stderr=subprocess.STDOUT) as child:
                    stage["pid"] = child.pid
                    while child.poll() is None:
                        live = live_child_cpu(child.pid)
                        # Budget pressure is recorded; a running physical chain is never killed.
                        job["live_child_cpu_seconds"] = live
                        job["budget_pressure_while_child_running"] = (
                            used_cpu_seconds() - start_cpu + live >= CPU_BUDGET_SECONDS
                            or directory_bytes(root) > OUTPUT_BUDGET_BYTES)
                        update()
                        time.sleep(plan["resource_policy"]["poll_seconds"])
                    code = child.wait()
            stage.update(returncode=code, status="complete" if code == 0 else "failed",
                ended_at=now(), cpu_seconds=used_cpu_seconds() - before_cpu,
                wall_seconds=time.monotonic() - before_wall)
            update()
            if code != 0:
                child_folder = (root / "evaluation" if name == "frozen_evaluation" else
                    root / next(row["relative_output"] for row in plan["subsets"] if row["name"] == name))
                child_status = read(child_folder / "status.json") if (child_folder / "status.json").exists() else {}
                stage["child_status"] = child_status
                message = (f"Stage {name} exited with returncode {code}; "
                           f"error={child_status.get('error')}; see {stage['log']}")
                if child_status.get("status") == "budget_hold":
                    raise BudgetHold(message)
                raise RuntimeError(message)
            if name != "frozen_evaluation":
                save(root / "data_manifest.json", aggregate_data(plan))
            else:
                verify_public(root / "evaluation", required={"manifest.json", "summary.json", "status.json"})
            budget_check()
        verify_frozen(plan, input_pins)
        manifest = aggregate_data(plan)
        if (not manifest["complete"] or manifest["independent_physical_chains"] != 156
                or manifest["unique_size_parameter_conditions"] != 24):
            raise RuntimeError("Final physical-chain inventory differs from the fixed plan")
        save(root / "data_manifest.json", manifest)
        update("complete", workflow_complete=True, current=None, returncode=0, ended_at=now())
    except Exception as exc:
        update("budget_hold" if isinstance(exc, BudgetHold) else "failed",
            workflow_complete=False, returncode=3 if isinstance(exc, BudgetHold) else 1,
            error=f"{type(exc).__name__}: {exc}", requires_user_action=True, ended_at=now())
        raise
    finally:
        # Only public artifacts: raw physical data and checkpoints are never inventoried.
        files = {}
        for path in root.rglob("*"):
            if (not path.is_file() or path.name == "public_files.json"
                    or path.suffix.lower() not in (".json", ".png", ".npz")
                    or path.name == "data.npz"):
                continue
            files[str(path.relative_to(root))] = sha(path)
        save(root / "public_files.json", files)
    return job


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    for variable in THREAD_VARIABLES:
        os.environ[variable] = "1"
    result = (run(args.output, args.source_root, args.validation) if args.execute else
              build_plan(args.output, args.source_root, args.validation))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
