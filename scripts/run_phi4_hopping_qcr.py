"""Exclusive hopping-phi4 sampling and aligned QCR workflow; dry by default.

The synthetic CUDA check is execution evidence only, never physical science.
No child is killed, retried, resumed or migrated after failure or a budget hold.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
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

THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
HOST = "thinklab-105-225"
PEAK_BYTES = 8 * 1024**3
CPU_SECONDS, GPU_SECONDS, DIRECTORY_BYTES = 14400., 3600., 1024**3


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def commands(output, config, gpu_uuid):
    root = Path(output).resolve()
    return {
        "gpu_preflight": [sys.executable, "-u", "-m", "scripts.run_phi4_hopping_qcr",
            "--benchmark-output", str(root / "gpu_preflight.json"), "--gpu-uuid", gpu_uuid],
        "sampling": [sys.executable, "-u", "-m", "scripts.prepare_phi4_hopping",
            "--execute", "--output", str(root / "data")],
        "training": [sys.executable, "-u", "-m", "recurrent_rg", "train",
            "--config", str(root / "training_config.json"), "--output", str(root / "training"),
            "--device", "cuda:0"],
    }


def plan(output, config, gpu_index, gpu_uuid):
    from experiments.phi4_hopping_qcr import TrainingOptions, load_config, training_plan
    from recurrent_rg.data.phi4_hopping_pipeline import Phi4HoppingOptions, generation_plan
    root, code, config = Path(output).resolve(), Path(__file__).resolve().parents[1], Path(config).resolve()
    if root.exists() or root == code or root in code.parents or code in root.parents:
        raise ValueError("Require a fresh output separate from the frozen code snapshot")
    if type(gpu_index) is not int or gpu_index < 0 or not gpu_uuid.startswith("GPU-"):
        raise ValueError("Require an explicit physical GPU index and UUID")
    values = load_config(config)
    values.update(data=str(root / "data"), output=str(root / "training"),
                  device="cuda:0", expected_gpu_uuid=gpu_uuid)
    options = TrainingOptions(**values)
    expected = dict(steps=3000, system_batch=32, sample_batch=64, seed=91, threads=1,
        information_pairs=128, max_training_seconds=1800., max_cuda_bytes=PEAK_BYTES)
    if any(getattr(options, name) != value for name, value in expected.items()):
        raise ValueError("This fresh parent pins the agreed full aligned experiment")
    result = dict(schema_version=5, created_at=now(), execution_requested=False,
        code_root=str(code), output=str(root), host=HOST, config_source=str(config),
        config_source_sha256=sha(config), GPU=dict(index=gpu_index, uuid=gpu_uuid),
        physical=generation_plan(Phi4HoppingOptions()), training=training_plan(options),
        commands=commands(root, config, gpu_uuid), stages=["gpu_preflight", "sampling", "training"],
        resources=dict(cpu_seconds=CPU_SECONDS, gpu_card_seconds=GPU_SECONDS,
            max_training_seconds=1800., max_cuda_bytes=PEAK_BYTES, directory_bytes=DIRECTORY_BYTES,
            cpu_threads=1, nice=10, min_memory_bytes=36*1024**3,
            max_load_per_cpu=1.25, max_io_full_avg10=20., idle_check_spacing_seconds=10.,
            min_gpu_free_bytes=12*1024**3, require_no_compute_processes=True),
        benchmark=dict(synthetic=True, physical_result=False, warmup_updates=2, timed_updates=5,
            projection_factor=1.5, initialization_factor=2., evaluation_report_reserve_seconds=300.,
            initialization_samples_per_chain=128, training_batch=[32, 64, 36, 36, 1],
            no_checkpoint_saved=True, no_physical_sampling=True),
        no_resume=True, no_retry=True, no_auto_repair=True, no_migration=True,
        qualification="连续场独立新实验；保留采样质量警告；CUDA预检不是物理结果；不认证平衡或临界指数")
    return result


def host_resources():
    memory = Path("/proc/meminfo").read_text()
    pressure = Path("/proc/pressure/io").read_text()
    return dict(at=now(), cpus=os.cpu_count(), load1=os.getloadavg()[0],
        mem_available_bytes=int(memory.split("MemAvailable:")[1].split()[0])*1024,
        io_full_avg10=float(pressure.split("full avg10=")[1].split()[0]))


def gpu_resources(index, expected_uuid):
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.free",
        "--format=csv,noheader,nounits"], text=True)
    selected = [line for line in raw.splitlines() if int(line.split(",")[0]) == index]
    if len(selected) != 1:
        raise RuntimeError("Pinned physical GPU is absent")
    _, uuid, free = [value.strip() for value in selected[0].split(",")]
    if uuid.lower() != expected_uuid.lower():
        raise RuntimeError("Pinned physical GPU UUID mismatch; no fallback")
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
        "--format=csv,noheader,nounits"], text=True)
    pids = [int(line.split(",")[1]) for line in active.splitlines()
            if line.split(",")[0].strip().lower() == uuid.lower()]
    return dict(index=index, uuid=uuid, free_bytes=int(free)*1024**2, compute_pids=pids)


def ready(row, gpu=False):
    host = row["host"]
    good = (host["mem_available_bytes"] >= 36*1024**3
            and host["load1"] <= 1.25*host["cpus"] and host["io_full_avg10"] < 20)
    return good and (not gpu or (not row["gpu"]["compute_pids"]
                                and row["gpu"]["free_bytes"] >= 12*1024**3))


def child_environment(uuid=None):
    return {**os.environ, **{name: "1" for name in THREAD_ENV},
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": uuid or ""}


def validate_training(root, manifest_sha):
    from experiments.phi4_hopping_qcr import RECIPE
    folder = Path(root) / "training"
    status, summary, manifest = [json.loads((folder / name).read_text())
        for name in ("status.json", "summary.json", "manifest.json")]
    clocks = summary.get("optimizer_steps", {})
    if (status.get("status") != "complete" or status.get("workflow_complete") is not True
            or status.get("returncode") != 0 or status.get("error") is not None
            or status.get("requires_user_action") is not False or status.get("actual_steps") != 3000
            or status.get("latest_checkpoint_step") != 3000 or summary.get("actual_steps") != 3000
            or summary.get("recipe") != RECIPE or manifest.get("recipe") != RECIPE
            or summary.get("data_manifest_sha256") != manifest_sha
            or manifest.get("data_manifest_sha256") != manifest_sha
            or manifest.get("closure_objective") != "all_start_anchor"
            or manifest.get("closure_scale_weights") is not None
            or summary.get("theory_dim") != 1 or summary.get("thermal_nu") is not None
            or summary.get("peak_cuda_bytes", PEAK_BYTES+1) > PEAK_BYTES
            or status.get("summary_sha256") != sha(folder / "summary.json")
            or status.get("evaluation_arrays_sha256") != sha(folder / "evaluation_arrays.npz")
            or set(clocks) != {"Q", "C", "R"}
            or any(row.get("min") != 3000 or row.get("max") != 3000 or row.get("parameters", 0) <= 0
                   for row in clocks.values())):
        raise RuntimeError("Training subprocess returned zero without the complete frozen 3000-step contract")


def benchmark(output, gpu_uuid):
    """Actual full-batch synthetic gradients, isolated process and RNG streams."""
    import torch
    from experiments.phi4_hopping_qcr import (BudgetHold, _device_receipt, independent_step,
        initialize_q_decoder, module_fingerprints, optimizer_step_counts)
    from recurrent_rg.phi4_aligned_qcr import AlignedPhi4QCR
    path = Path(output)
    if path.exists():
        raise FileExistsError("Synthetic receipt is fresh-only")
    started = time.perf_counter()
    receipt = dict(status="running", synthetic=True, physical_result=False,
        checkpoint_saved=False, formal_model_reused=False, physical_data_read=False,
        training_batch=[32, 64, 36, 36, 1], warmup_updates=2, timed_updates=5,
        error=None, returncode=None)
    try:
        from scripts.report_phi4_hopping_qcr import select_chinese_font
        receipt["chinese_font"] = str(select_chinese_font(os.environ.get("QCR_ZH_FONT")))
        torch.set_num_threads(1)
        torch.manual_seed(91)
        device = torch.device("cuda:0")
        receipt["device"] = _device_receipt(device, gpu_uuid)
        torch.cuda.reset_peak_memory_stats(device)
        fields = torch.randn((32, 128, 36, 36, 1), dtype=torch.float64,
            generator=torch.Generator().manual_seed(20261031))
        model = AlignedPhi4QCR(float(fields.square().mean().sqrt())).to(device)
        constructor = module_fingerprints(model)
        torch.cuda.synchronize(device)
        init_started = time.perf_counter()
        initialize_q_decoder(model, fields, list(range(32)), seed=91)
        torch.cuda.synchronize(device)
        initialization_seconds = time.perf_counter()-init_started
        optimizers = {name: torch.optim.Adam(getattr(model, name).parameters(), lr=.001)
                      for name in ("Q", "C", "R")}
        batch = fields[:, :64].to(device)
        generator = torch.Generator(device=device).manual_seed(293)
        times = []
        for index in range(7):
            torch.cuda.synchronize(device)
            update_started = time.perf_counter()
            metrics = independent_step(model, batch, optimizers, generator=generator,
                                       information_pairs=128)
            torch.cuda.synchronize(device)
            if index >= 2:
                times.append(time.perf_counter()-update_started)
            peak = max(torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device))
            if peak > PEAK_BYTES:
                raise BudgetHold("Synthetic full-batch own CUDA peak exceeds 8 GiB")
        clocks = optimizer_step_counts(optimizers, 7)
        projection = 2*initialization_seconds+1.5*max(times)*3000+300
        passed = projection <= 1800.
        receipt.update(status="complete" if passed else "budget_hold", returncode=0 if passed else 3,
            workflow_complete=True, actual_updates=7, optimizer_steps=clocks,
            constructor_module_sha256=constructor, final_module_sha256=module_fingerprints(model),
            initialization_seconds=initialization_seconds, timed_update_seconds=times,
            conservative_full_training_projection_seconds=projection,
            training_cap_seconds=1800., peak_cuda_bytes=int(peak), last_metrics=metrics,
            requires_user_action=not passed)
    except Exception as exc:
        hold = isinstance(exc, BudgetHold)
        receipt.update(status="budget_hold" if hold else "failed", workflow_complete=False,
                       returncode=3 if hold else 1,
                       requires_user_action=True, error=f"{type(exc).__name__}: {exc}")
    receipt["wall_seconds"] = time.perf_counter()-started
    save(path, receipt)
    return receipt["returncode"]


def run(output, config, gpu_index, gpu_uuid):
    definition = plan(output, config, gpu_index, gpu_uuid)
    if socket.gethostname() != HOST or os.getpriority(os.PRIO_PROCESS, 0) != 10:
        raise ValueError("Execution requires the pinned 225 host and one inherited nice10 layer")
    root, code = Path(output).resolve(), Path(definition["code_root"])
    pins = {str(path): sha(path) for folder in ("experiments", "recurrent_rg", "scripts")
            for path in (code / folder).rglob("*.py")}
    pins[str(Path(config).resolve())] = sha(config)
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    configuration = json.loads(Path(config).read_text())
    configuration.update(data=str(root / "data"), output=str(root / "training"))
    configuration["training"].update(device="cuda:0", expected_gpu_uuid=gpu_uuid)
    save(root / "training_config.json", configuration)
    pins[str(root / "training_config.json")] = sha(root / "training_config.json")
    save(root / "plan.json", dict(definition, execution_requested=True, source_sha256=pins))
    pins[str(root / "plan.json")] = sha(root / "plan.json")
    job = dict(status="prepared", workflow_complete=False, parent_pid=os.getpid(), current=None,
        returncode=None, error=None, requires_user_action=False, started_at=now(), stages=[],
        plan_sha256=sha(root / "plan.json"), cpu_seconds=0., gpu_seconds=0.)
    cpu_start, child_start, gpu_seconds = time.process_time(), resource.getrusage(resource.RUSAGE_CHILDREN), 0.
    active, active_stage = None, None

    def checkpoint():
        child = resource.getrusage(resource.RUSAGE_CHILDREN)
        job["cpu_seconds"] = (time.process_time()-cpu_start+child.ru_utime-child_start.ru_utime
                              + child.ru_stime-child_start.ru_stime)
        if active is not None and active.poll() is None:
            try:
                fields = Path(f"/proc/{active.pid}/stat").read_text().split(")", 1)[1].split()
                job["cpu_seconds"] += (int(fields[11])+int(fields[12]))/os.sysconf("SC_CLK_TCK")
            except (FileNotFoundError, ProcessLookupError):
                pass
        job["gpu_seconds"] = gpu_seconds + (time.monotonic()-active_stage["monotonic_start"]
            if active_stage is not None and active_stage["name"] != "sampling" and active is not None
            and active.poll() is None else 0.)
        job["directory_bytes"] = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
        save(root / "job.json", job)

    def verify():
        if any(sha(path) != digest for path, digest in pins.items()):
            raise RuntimeError("Pinned source or configuration changed; no repair or restart")

    def wait_resources(name):
        nonlocal gpu_seconds
        gpu = name != "sampling"
        job.update(status="waiting_resources", current=name)
        while True:
            first = dict(host=host_resources())
            if gpu:
                first["gpu"] = gpu_resources(gpu_index, gpu_uuid)
            job["resources"] = first
            checkpoint()
            if not ready(first, gpu):
                time.sleep(30)
                continue
            if not gpu:
                return
            started = time.monotonic()
            time.sleep(10)
            second = dict(host=host_resources(), gpu=gpu_resources(gpu_index, gpu_uuid))
            spacing = time.monotonic()-started
            gpu_seconds += spacing
            job["gpu_idle_checks"] = dict(first=first, second=second, spacing_seconds=spacing)
            checkpoint()
            if spacing >= 10 and ready(second, True):
                return

    checkpoint()
    try:
        for name, command in definition["commands"].items():
            verify()
            wait_resources(name)
            verify()
            checkpoint()
            if job["cpu_seconds"] > CPU_SECONDS or job["gpu_seconds"] > GPU_SECONDS or job["directory_bytes"] > DIRECTORY_BYTES:
                job.update(status="budget_hold", current=None, returncode=3, requires_user_action=True,
                           error="Registered resource budget exhausted before next stage")
                checkpoint()
                return 3
            if name == "training":
                projected = json.loads((root / "gpu_preflight.json").read_text())["conservative_full_training_projection_seconds"]
                if job["gpu_seconds"]+projected > GPU_SECONDS or job["cpu_seconds"]+projected > CPU_SECONDS:
                    job.update(status="budget_hold", current=None, returncode=3, requires_user_action=True,
                               error="Actual cumulative use plus conservative training projection exceeds registered budget")
                    checkpoint()
                    return 3
            stage = dict(name=name, status="running", returncode=None, started_at=now(), command=command,
                         monotonic_start=time.monotonic())
            active_stage = stage
            job["stages"].append(stage)
            job.update(status="running", current=name)
            with (root / "logs" / f"{name}.log").open("xb") as log:
                active = subprocess.Popen(command, cwd=code, env=child_environment(gpu_uuid if name != "sampling" else None),
                    stdout=log, stderr=subprocess.STDOUT)
                stage["pid"] = active.pid
                while active.poll() is None:
                    if name != "gpu_preflight":
                        path = root / ("data" if name == "sampling" else "training") / "status.json"
                        try:
                            stage["child_status"] = json.loads(path.read_text())
                        except (FileNotFoundError, json.JSONDecodeError):
                            pass
                    checkpoint()
                    time.sleep(5)
                rc = active.returncode
            elapsed = time.monotonic()-stage["monotonic_start"]
            if name != "sampling":
                gpu_seconds += elapsed
            stage.update(status="complete" if rc == 0 else "budget_hold" if rc == 3 else "failed",
                         returncode=rc, ended_at=now(), wall_seconds=elapsed)
            active, active_stage = None, None
            verify()
            if rc:
                job.update(status=stage["status"], current=None, returncode=rc, requires_user_action=True,
                           error=f"{name} exited {rc}; completed outputs preserved", ended_at=now())
                checkpoint()
                return rc
            if name == "sampling":
                from recurrent_rg.data.phi4_hopping_loader import load_dataset
                dataset = load_dataset(root / "data")
                job["data_manifest_sha256"] = dataset.manifest_sha256
                job["data_warning_count"] = dataset.warning_count
                quality = json.loads((root / "data/quality_summary.json").read_text())
                acceptances = [value for row in quality["conditions"] for value in row["production_acceptance"]]
                if len(acceptances) != 52 or any(not math.isfinite(value) or value <= 0 for value in acceptances):
                    raise RuntimeError("Zero/nonfinite HMC production acceptance; completed data retained, training not started")
                del dataset
            elif name == "training":
                validate_training(root, job["data_manifest_sha256"])
            checkpoint()
        checkpoint()
        if job["cpu_seconds"] > CPU_SECONDS or job["gpu_seconds"] > GPU_SECONDS or job["directory_bytes"] > DIRECTORY_BYTES:
            job.update(status="budget_hold", current=None, returncode=3, requires_user_action=True,
                       error="Completed outputs exceed registered conservative resources", ended_at=now())
            checkpoint()
            return 3
        job.update(status="complete", workflow_complete=True, current=None, returncode=0,
                   error=None, requires_user_action=False, ended_at=now())
        checkpoint()
        public = {"job.json": sha(root / "job.json"), "plan.json": sha(root / "plan.json"),
            "training_config.json": sha(root / "training_config.json"), "gpu_preflight.json": sha(root / "gpu_preflight.json")}
        physical = json.loads((root / "data/manifest.json").read_text())
        for name in physical["public_files"]:
            path = (root / "data" / name).resolve()
            if root / "data" not in path.parents or path.suffix != ".json":
                raise RuntimeError("Invalid public physical metadata path")
            public[f"data/{name}"] = sha(path)
        public["training/status.json"] = sha(root / "training/status.json")
        for row in json.loads((root / "training/status.json").read_text())["public_files"]:
            path = (root / "training" / row["path"]).resolve()
            if (root / "training" not in path.parents or path.suffix not in (".json", ".png", ".npz")
                    or sha(path) != row["sha256"]):
                raise RuntimeError("Invalid public training artifact binding")
            public[f"training/{row['path']}"] = row["sha256"]
        save(root / "public_files.json", public)
        return 0
    except BaseException as exc:
        if active is not None and active.poll() is None:
            job.update(status="waiting_running_child_after_parent_error", requires_user_action=True,
                       error=f"{type(exc).__name__}: {exc}")
            checkpoint()
            active.wait()  # Preserve and await: never kill a running child.
        job.update(status="failed", workflow_complete=False, current=None, returncode=1,
                   requires_user_action=True, error=f"{type(exc).__name__}: {exc}", ended_at=now())
        checkpoint()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    parser.add_argument("--config", default="configs/phi4_hopping_b3_g1_aligned_seed91.json")
    parser.add_argument("--gpu-index", type=int, default=2)
    parser.add_argument("--gpu-uuid", default="GPU-713b07d0-118c-f939-c0a8-565390215e01")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--benchmark-output", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.benchmark_output:
        return benchmark(args.benchmark_output, args.gpu_uuid)
    if not args.output:
        parser.error("--output is required")
    if args.execute:
        return run(args.output, args.config, args.gpu_index, args.gpu_uuid)
    print(json.dumps(plan(args.output, args.config, args.gpu_index, args.gpu_uuid), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
