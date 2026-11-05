"""Fresh b2 Ising output-dimension ablation, with frozen code and idle-GPU gates.

The user explicitly selected 2x2 and a strictly output-only dimension100 arm.
No source weights, physical sampling, MI rebuilding, retry, resume or migration.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import socket
import subprocess
import sys
import time


HOST = "thinklab-105-225"
GPU_INDEX = 2
GPU_UUID = "GPU-713b07d0-118c-f939-c0a8-565390215e01"
DIMENSIONS = (1, 5, 10, 100)
UPDATES = 12000
PACKAGES = ("recurrent_rg", "experiments", "scripts")


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value, *, exclusive=False):
    path = Path(path)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x") as stream:
            stream.write(text)
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(path)


def source_hashes(project):
    project = Path(project)
    result = {str(path.relative_to(project)): sha(path) for package in PACKAGES
              for path in sorted((project / package).rglob("*.py")) if path.is_file()}
    if (project / "AGENTS.md").is_file():
        result["AGENTS.md"] = sha(project / "AGENTS.md")
    return result


def cpu_seconds():
    return sum(getattr(resource.getrusage(who), field) for who in
               (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)
               for field in ("ru_utime", "ru_stime"))


def build_plan(source_run, output, dimensions=DIMENSIONS):
    """JSON/stat only. Reuse the already-validated historical cache identities."""
    from scripts import run_ising2d_dimg_ablation as prior
    from recurrent_rg.config import RunConfig
    from experiments.rsmi_separate_qcr import build_parser, validate_recipe

    dimensions = tuple(dimensions)
    if any(type(dim) is not int for dim in dimensions) or dimensions != DIMENSIONS:
        raise ValueError("The authorized comparison is exactly dimg=1,5,10,100")
    old = prior.build_plan(source_run, output)
    base = RunConfig.from_dict(old["arms"][0]["config"]).to_dict()
    if base["loss"].get("closure_scale_weights") is not None:
        raise ValueError("Require the unchanged uniform target-scale recipe")
    if base["model"]["block_factor"] != 2 or base["model"]["hidden"] != 96:
        raise ValueError("Require explicit b2 and unchanged hidden96")
    root = Path(old["output"])
    arms = []
    for dim in dimensions:
        config = deepcopy(base)
        config["model"]["theory_dim"] = dim
        parsed = RunConfig.from_dict(config)
        parsed.validate()
        validate_recipe(build_parser().parse_args(["--config", "unused", "--output", "unused"]), parsed)
        changes = prior.differences(base, config)
        if changes != ([] if dim == 1 else ["model.theory_dim"]):
            raise ValueError("Only theory_dim may differ among fresh arms")
        arm_root = root / f"dim{dim}"
        config_path = root / "configs" / f"dim{dim}.json"
        smoke_config = root / "configs" / f"smoke_dim{dim}.json"
        command = [sys.executable, "-u", "-m", "scripts.run_ising_highdim", "--cached-training",
                   "--config", str(config_path), "--output", str(arm_root / "run"),
                   "--device", "cuda:2", "--mi-cache-dir", old["mi_cache_dir"],
                   "--initialization-receipt", str(arm_root / "initialization.json")]
        smoke = list(command)
        smoke[smoke.index("--config") + 1] = str(smoke_config)
        smoke[smoke.index("--output") + 1] = str(root / "smoke" / f"dim{dim}" / "run")
        smoke[smoke.index("--initialization-receipt") + 1] = str(root / "smoke" / f"dim{dim}" / "initialization.json")
        report = [sys.executable, "-u", "-m", "scripts.report_ising_highdim", "--run",
                  str(arm_root / "run"), "--output", str(arm_root / "report"),
                  "--checkpoint", str(UPDATES), "--threads", "1", "--sampling-repeats", "5"]
        arms.append(dict(name=f"dim{dim}", theory_dim=dim, root=str(arm_root),
                         config_path=str(config_path), smoke_config_path=str(smoke_config),
                         config=config, changes=changes, training=command, smoke=smoke, report=report))
    compare = [sys.executable, "-u", "-m", "scripts.report_ising_highdim", "--compare",
               *(f"{a['name']}={a['root']}/report" for a in arms), "--output", str(root / "comparison")]
    summary = read(Path(source_run) / "summary.json")
    return dict(schema_version=1, experiment="二维零场Ising，b2，输出维数升维消融",
        output=str(root), source_run=old["source_run"], source_sha256=old["source_sha256"],
        immutable_stat_identity=old["immutable_stat_identity"], cache=old["cache"],
        data_sha256=prior.DATA_SHA, mi_cache_dir=old["mi_cache_dir"], mi_cache_keys=old["mi_cache_keys"],
        arms=arms, compare=compare, protocol=dict(dimensions=list(dimensions), updates=UPDATES,
            scales=[108, 54, 27], block_factor=2, seed=91, threads_per_training_arm=2,
            semantic_change="仅model.theory_dim；四臂同数据、目标、权重、学习率、批大小和预定终点",
            source_usage="只复用小型回执、物理数据和data-only MI缓存；不读取任何历史模型或优化器",
            initialization="每臂全新Q/C/R与Adam；同seed不宣称相同初始化或随机轨迹",
            initialization_receipt_scope="随机构造后、TRAIN仿射偏置校准前；实际首更新状态另见manifest.training_start_module_sha256",
            losses="Q仿射motif重构10+子集一致性20/5；C=-2MI+2逐系综方差下限+颜色NLL；R单步5+微观起点all-horizon5",
            closure_scale_weights="None：两个闭合分支分别对目标尺度等权归一",
            coordinate_reduction="sum；不因升维暗改loss归约；报告逐坐标误差和TRAIN std NRMSE",
            labels_used_in_training=False, extra_response_or_future_loss=False,
            hidden=96, local_Q_features=12, affine_decoder_bins=16,
            dimension100_user_choice="严格只改输出维数，作为过参数化检查；不扩大内部容量",
            dimension100_limitations="Q局部微分秩<=12；R残差输出秩<=96，至少4个单位特征值；偏置正交分量可排除精确根",
            reporting="固定TRAIN-only有界多起点，不用指数网格或高维凸包；保留全部验收根/全部实复负谱；PCA只用于图；thermal_nu=null",
            smoke="各维2步独立CUDA执行检查，正式前全部通过；不用smoke权重或结果",
            parallelism="同一卡串行；报告单线程CPU；不共享他人GPU、不干预其他进程",
            resource_gate=dict(host=HOST, gpu=GPU_INDEX, gpu_uuid=GPU_UUID,
                gpu_free_mib=12288, no_compute_processes=True, checks=2, check_spacing_seconds=10,
                mem_available_gib=36, max_load_per_cpu=1.25, io_full_avg10_below=20, disk_free_gib=20),
            automatic_retry=False, resume=False, checkpoint_selection="预定实际12000，不看理论/测试误差选择终点",
            source_sampling="沿用原有限链数据；不重采、不宣称ESS或热化认证得到改进"),
        timing_reference_only=dict(source_training_seconds=summary.get("timing", {}).get("training_wall_seconds"),
            interpretation="历史另一GPU约50分钟/臂；不是新硬件时间保证或训练预算；未计报告/等待"))


def snapshot():
    if socket.gethostname() != HOST:
        raise RuntimeError("Wrong host; do not migrate")
    query = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.free",
                                    "--format=csv,noheader,nounits"], text=True)
    selected = next(line.split(",") for line in query.splitlines() if int(line.split(",")[0]) == GPU_INDEX)
    if selected[1].strip() != GPU_UUID:
        raise RuntimeError("Physical GPU identity changed")
    listing = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                                      "--format=csv,noheader,nounits"], text=True)
    active = [line for line in listing.splitlines() if line.split(",")[0].strip() == GPU_UUID]
    memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(memory["MemAvailable"].split()[0]) / 1024**2
    full = next(line for line in Path("/proc/pressure/io").read_text().splitlines() if line.startswith("full "))
    io = float(next(item.split("=")[1] for item in full.split() if item.startswith("avg10=")))
    row = dict(at=now(), host=HOST, gpu=GPU_INDEX, gpu_uuid=GPU_UUID,
        gpu_free_mib=int(selected[2]), compute_processes=active, mem_available_gib=available,
        cpu_count=os.cpu_count(), load1=os.getloadavg()[0], io_full_avg10=io,
        disk_free_gib=shutil.disk_usage(Path(__file__).resolve().parents[1]).free / 1024**3)
    row["host_ready"] = (available >= 36 and row["load1"] <= 1.25 * row["cpu_count"]
                         and io < 20 and row["disk_free_gib"] >= 20)
    row["gpu_ready"] = row["host_ready"] and not active and row["gpu_free_mib"] >= 12288
    return row


def admit(root, job, *, gpu):
    ready = 0
    while True:
        row = snapshot()
        job.setdefault("resource_checks", []).append(dict(row, requires_gpu=gpu))
        job["resource_checks"] = job["resource_checks"][-120:]
        ready = ready + 1 if row["gpu_ready" if gpu else "host_ready"] else 0
        save(root / "job.json", job)
        if ready >= (2 if gpu else 1):
            return
        time.sleep(10 if ready else 60)


def ensure_unchanged(plan, project, fingerprints, configs):
    from scripts.run_ising2d_dimg_ablation import ensure_unchanged as previous
    previous(plan, {}, {})
    for name, digest in fingerprints.items():
        if sha(project / name) != digest:
            raise RuntimeError(f"Frozen science code changed: {name}")
    for name, digest in configs.items():
        if sha(name) != digest:
            raise RuntimeError("Frozen experiment configuration changed")


def validate_training(plan, arm, *, smoke):
    from recurrent_rg.learning_rate import LearningRatePolicy
    from recurrent_rg.config import RunConfig
    run = Path(plan["output"]) / "smoke" / arm["name"] / "run" if smoke else Path(arm["root"]) / "run"
    config_path = arm["smoke_config_path"] if smoke else arm["config_path"]
    config = read(config_path)
    manifest, summary = read(run / "manifest.json"), read(run / "summary.json")
    steps = 2 if smoke else UPDATES
    required = dict(training_driver="rsmi_separate_qcr", precision="torch.float64", estimator="histogram",
        environment_mode="full_shell", buffer=1, independent_c=True, labels_used_in_training=False,
        rsmi_weight=2., execution_device="cuda:2", source=None)
    if any(manifest.get(key) != value for key, value in required.items()):
        raise RuntimeError("Training recipe/provenance differs")
    if (manifest.get("config") != config or summary.get("config") != config or
            Path(manifest.get("config_source", "")).resolve() != Path(config_path).resolve() or
            manifest.get("data_sha256") != plan["data_sha256"] or manifest.get("q_supervision") or
            manifest.get("resumed_from") or manifest.get("frozen_C") or manifest.get("data_driven_initialization") or
            summary.get("actual_epochs") != steps or summary.get("requested_epochs") != steps or
            any(summary.get(key) for key in ("stopped_by_training_limit", "stopped_by_wall_limit", "stopped_by_train_plateau")) or
            summary.get("final_report_skipped") is not True):
        raise RuntimeError("Fresh exact endpoint or label-free configuration mismatch")
    records = manifest.get("execution_optimizations", {}).get("mi_statistics_cache", {})
    for split, key in plan["mi_cache_keys"].items():
        record = records.get(split, {})
        if (record.get("hit") is not True or record.get("key") != key or record.get("cache_version") != 1 or
                Path(record.get("path", "")).resolve() != Path(plan["mi_cache_dir"]) / f"mi-v1-{key}.pt"):
            raise RuntimeError("Require exact existing static MI cache hits")
    policy = LearningRatePolicy(RunConfig.from_dict(config).training)
    if summary.get("lr_scheduler_state") != policy.checkpoint_state(steps):
        raise RuntimeError("Learning-rate update clock differs")
    history = summary.get("history", [])
    every = config["training"]["log_every"]
    expected = sorted({1, steps, *range(every, steps + 1, every)})
    if ([row.get("epoch") for row in history] != expected or any(
            not math.isclose(row.get("learning_rate", -1.), policy.rate_for_update(row["epoch"]),
                             rel_tol=1e-12, abs_tol=0.) for row in history)):
        raise RuntimeError("Logged actual update numbers or learning rates differ")
    if not (run / f"training_step_{steps}.pt").is_file():
        raise RuntimeError("Numbered final checkpoint missing")
    init = read(run.parent / "initialization.json")
    if init.get("fresh") is not True or init.get("source_checkpoint") is not None:
        raise RuntimeError("Fresh initialization receipt missing")
    if not smoke:
        smoke_init = read(Path(plan["output"]) / "smoke" / arm["name"] / "initialization.json")
        fields = ("C_tensor_sha256", "Q_tensor_sha256", "R_tensor_sha256", "full_model_tensor_sha256")
        if any(not init.get(key) or init[key] != smoke_init.get(key) for key in fields):
            raise RuntimeError("Formal fresh initialization differs from independent smoke construction")
    return dict(actual_updates=steps, smoke_only=smoke, labels_used_in_training=False,
        initialization=init, timing=summary.get("timing", {}), cache_hits=records,
        manifest_sha256=sha(run / "manifest.json"))


def stage(root, job, name, command, *, gpu, plan, fingerprints, configs, check):
    project = root / "code"
    ensure_unchanged(plan, project, fingerprints, configs)
    job.update(status="waiting_resources", current=name)
    save(root / "job.json", job)
    admit(root, job, gpu=gpu)
    ensure_unchanged(plan, project, fingerprints, configs)
    row = dict(name=name, status="running", command=command, gpu=gpu, started_at=now(),
               log=str(root / "logs" / f"{name}.log"))
    job["stages"].append(row)
    env = dict(os.environ, PYTHONPATH=str(project), OMP_NUM_THREADS="2" if gpu else "1",
        MKL_NUM_THREADS="2" if gpu else "1", OPENBLAS_NUM_THREADS="2" if gpu else "1",
        NUMBA_NUM_THREADS="2" if gpu else "1", MPLCONFIGDIR=str(root / "matplotlib_cache"),
        MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1")
    if gpu:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""
    started = time.monotonic()
    with Path(row["log"]).open("x") as log:
        child = subprocess.Popen(command, cwd=project, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT)
        row["pid"] = child.pid
        job.update(status="running")
        save(root / "job.json", job)
        rc = child.wait()
    row.update(returncode=rc, ended_at=now(), elapsed_seconds=time.monotonic() - started,
               status="complete" if rc == 0 else "failed")
    job["GPU_stage_seconds"] += row["elapsed_seconds"] if gpu else 0.
    save(root / "job.json", job)
    if rc:
        raise RuntimeError(f"{name} returncode={rc}; no retry/repair/restart")
    result = check()
    ensure_unchanged(plan, project, fingerprints, configs)
    job.setdefault("receipts", {})[name] = result
    save(root / "job.json", job)


def report_receipt(path, dim=None):
    path = Path(path)
    marker = read(path / "report.json")
    if marker.get("status") != "complete" or marker.get("report_complete") is not True:
        raise RuntimeError("Formal report not complete")
    if dim is not None and (marker.get("theory_dim") != dim or marker.get("checkpoint_epoch") != UPDATES):
        raise RuntimeError("Report dimension or endpoint differs")
    required = (("metrics.json", "fixed_points.json", "critical_exponents.json", "encodings.npz",
                 "trajectories.npz", "fixed_point_arrays.npz", "R_vs_g.png", "R_displacement.png",
                 "coupling_by_scale.png", "coupling_all_coordinates.png", "loss_comparison.png",
                 "representation_diagnostics.png", "fixed_point_spectrum.png", "status.json")
                if dim is not None else ("comparison.json", "comparison.png"))
    if any(not (path / name).is_file() or (path / name).is_symlink() for name in required):
        raise RuntimeError("Formal report required public files missing")
    if dim is not None:
        metrics = read(path / "metrics.json")
        freeze = metrics.get("freeze_verification", {})
        if (metrics.get("theory_dim") != dim or metrics.get("checkpoint_epoch") != UPDATES or
                metrics.get("labels_used_in_training") is not False or metrics.get("thermal_nu") is not None or
                not all(freeze.get(key) is True for key in
                        ("model_state_unchanged", "model_gradients_none", "checkpoint_unchanged"))):
            raise RuntimeError("Formal report freeze or scientific-provenance check failed")
    return dict(path=str(path), files={p.name: sha(p) for p in sorted(path.iterdir())
        if p.is_file() and not p.is_symlink() and p.suffix in (".json", ".png", ".npz")})


def execute(plan):
    root = Path(plan["output"])
    if root.exists() or root.is_symlink():
        raise FileExistsError("Refuse duplicate launch, overwrite or resume")
    if socket.gethostname() != HOST or os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Require host225 and native physical GPU indexing")
    project = Path(__file__).resolve().parents[1]
    fingerprints = source_hashes(project)
    root.mkdir(parents=True, exist_ok=False)
    for folder in ("logs", "configs", "smoke", "code"):
        (root / folder).mkdir()
    for name, digest in fingerprints.items():
        target = root / "code" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project / name, target)
        if sha(target) != digest:
            raise RuntimeError("Code snapshot copy mismatch")
    save(root / "code/fingerprints.json", fingerprints, exclusive=True)
    configs = {}
    for arm in plan["arms"]:
        Path(arm["root"]).mkdir()
        (root / "smoke" / arm["name"]).mkdir()
        save(arm["config_path"], arm["config"], exclusive=True)
        smoke_config = deepcopy(arm["config"])
        smoke_config["training"].update(epochs=2, checkpoint_every=1, log_every=1)
        save(arm["smoke_config_path"], smoke_config, exclusive=True)
        for key in ("config_path", "smoke_config_path"):
            configs[arm[key]] = sha(arm[key])
    plan.update(source_code_sha256=fingerprints, configuration_sha256=configs)
    save(root / "plan.json", plan, exclusive=True)
    job = dict(status="running", workflow_complete=False, started_at=now(), parent_pid=os.getpid(),
        host=HOST, device="cuda:2", gpu_uuid=GPU_UUID, error=None, requires_user_action=False,
        actual_nice=os.getpriority(os.PRIO_PROCESS, 0), current=None, stages=[], GPU_stage_seconds=0.,
        physical_sampling_updates=0, labels_used_in_training=False, completed_arms=[], public_files=[])
    save(root / "job.json", job, exclusive=True)
    started, cpu_start = time.monotonic(), cpu_seconds()
    try:
        for arm in plan["arms"]:
            stage(root, job, "smoke_" + arm["name"], arm["smoke"], gpu=True, plan=plan,
                fingerprints=fingerprints, configs=configs,
                check=lambda a=arm: validate_training(plan, a, smoke=True))
        job["all_smoke_checks_passed"] = True
        save(root / "job.json", job)
        for arm in plan["arms"]:
            stage(root, job, "train_" + arm["name"], arm["training"], gpu=True, plan=plan,
                fingerprints=fingerprints, configs=configs,
                check=lambda a=arm: validate_training(plan, a, smoke=False))
            stage(root, job, "report_" + arm["name"], arm["report"], gpu=False, plan=plan,
                fingerprints=fingerprints, configs=configs,
                check=lambda a=arm: report_receipt(Path(a["root"]) / "report", a["theory_dim"]))
            job["completed_arms"].append(arm["name"])
            save(root / "job.json", job)
        stage(root, job, "comparison", plan["compare"], gpu=False, plan=plan,
            fingerprints=fingerprints, configs=configs,
            check=lambda: report_receipt(root / "comparison"))
        public = ["job.json", "plan.json", *[str(Path(name).relative_to(root)) for name in configs]]
        for arm in plan["arms"]:
            public += [f"{arm['name']}/{name}" for name in
                       ("initialization.json", "run/manifest.json", "run/summary.json", "run/history_progress.json")]
            public += [str(path.relative_to(root)) for path in sorted((Path(arm["root"]) / "report").iterdir())
                       if path.is_file() and not path.is_symlink() and path.suffix in (".json", ".png", ".npz")]
        public += [str(path.relative_to(root)) for path in sorted((root / "comparison").iterdir())
                   if path.is_file() and not path.is_symlink() and path.suffix in (".json", ".png", ".npz")]
        job.update(status="complete", workflow_complete=True, current=None, ended_at=now(),
                   public_files=public)
    except Exception as error:
        job.update(status="failed", workflow_complete=False, requires_user_action=True,
                   error=f"{type(error).__name__}: {error}", ended_at=now())
        raise
    finally:
        job.update(elapsed_seconds=time.monotonic() - started, CPU_core_seconds=cpu_seconds() - cpu_start)
        save(root / "job.json", job)
    return job


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--cached-training", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--mi-cache-dir", type=Path)
    parser.add_argument("--initialization-receipt", type=Path)
    args = parser.parse_args(argv)
    if args.cached_training:
        if args.device != "cuda:2" or any(value is None for value in
                (args.config, args.mi_cache_dir, args.initialization_receipt)):
            parser.error("Cache-only training requires explicit config/cache/receipt on cuda:2")
        from scripts.run_ising2d_dimg_ablation import cached_training
        return cached_training(args)
    if args.source_run is None:
        parser.error("--source-run required")
    plan = build_plan(args.source_run, args.output)
    if args.plan:
        print(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False))
        return plan
    return execute(plan)


if __name__ == "__main__":
    main()
