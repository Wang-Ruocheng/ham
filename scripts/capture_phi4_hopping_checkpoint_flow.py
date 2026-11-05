"""显式授权的独立CPU中期预览；仅向stdout输出小型JSON，不写正式实验。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def binding(path):
    path = Path(path)
    before = path.stat()
    value = (before.st_size, before.st_mtime_ns, sha(path))
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("Checkpoint changed while fingerprinting")
    return value


def choose_checkpoint(root, step=None, expected_sha256=None):
    root = Path(root)
    status = read(root / "training/status.json")
    latest = status.get("latest_checkpoint_step", 0)
    selected = latest if step is None else step
    if (type(selected) is not int or selected <= 0 or type(latest) is not int
            or selected > latest or status.get("actual_steps", 0) < selected):
        raise ValueError("Requested checkpoint is not confirmed completely saved by training status")
    path = root / "training" / f"training_step_{selected}.pt"
    first = binding(path)
    time.sleep(1.)
    if first[0] <= 0 or binding(path) != first:
        raise RuntimeError("Checkpoint is not stable; no read or automatic retry")
    if expected_sha256 is not None and first[2] != expected_sha256:
        raise RuntimeError("Checkpoint differs from the requested immutable fingerprint")
    return path, selected, first


def verify_sources(pins, code):
    if not isinstance(pins, dict) or not pins:
        raise ValueError("Require frozen source bindings")
    for name, digest in pins.items():
        path = Path(name)
        path = path if path.is_absolute() else code / path
        if sha(path) != digest:
            raise RuntimeError(f"Frozen source/evidence fingerprint differs: {path}")


def fixed_cell_indices(cell_ids):
    cell_ids = np.asarray(cell_ids)
    cells = sorted(set(cell_ids.tolist()))
    if cells != list(range(15)):
        raise ValueError("Expected all fifteen fixed physical parameter cells")
    return [int(np.flatnonzero(cell_ids == cell)[0]) for cell in cells]


def crossing_candidates(grid, drift):
    grid, drift = np.asarray(grid), np.asarray(drift)
    values = []
    for i in range(len(grid) - 1):
        if drift[i] == 0:
            z = float(grid[i])
        elif np.signbit(drift[i]) != np.signbit(drift[i + 1]):
            z = float(grid[i] - drift[i] * (grid[i + 1] - grid[i]) / (drift[i + 1] - drift[i]))
        else:
            continue
        if not values or not math.isclose(z, values[-1], rel_tol=0., abs_tol=1e-12):
            values.append(z)
    if drift[-1] == 0 and (not values or grid[-1] != values[-1]):
        values.append(float(grid[-1]))
    return values


def capture(root, *, step=None, expected_sha256=None, samples=8, max_cpu_seconds=300.):
    started, cpu_started = time.perf_counter(), time.process_time()
    if samples != 8 or not 0 < max_cpu_seconds <= 300:
        raise ValueError("This bounded preview fixes eight saved configurations and <=300 CPU seconds")
    memory = Path("/proc/meminfo").read_text()
    pressure = Path("/proc/pressure/io").read_text()
    host = dict(cpus=os.cpu_count(), load1=os.getloadavg()[0],
                mem_available_bytes=int(memory.split("MemAvailable:")[1].split()[0]) * 1024,
                io_full_avg10=float(pressure.split("full avg10=")[1].split()[0]))
    if (host["mem_available_bytes"] < 36 * 1024**3 or host["load1"] > 1.25 * host["cpus"]
            or host["io_full_avg10"] >= 20 or os.getpriority(os.PRIO_PROCESS, 0) < 10):
        raise RuntimeError("CPU preview host resources/priority gate failed; no fallback")

    import torch
    from experiments import phi4_hopping_qcr as training
    from recurrent_rg import phi4_aligned_qcr as core
    from recurrent_rg.data import phi4_hopping_loader as loader

    torch.set_num_threads(1)
    if torch.cuda.is_initialized():
        raise RuntimeError("Preview must not initialize CUDA")
    root = Path(root).resolve()
    plan_path, manifest_path = root / "plan.json", root / "training/manifest.json"
    plan, job, manifest = read(plan_path), read(root / "job.json"), read(manifest_path)
    plan_sha, manifest_sha = sha(plan_path), sha(manifest_path)
    if job.get("plan_sha256") != plan_sha:
        raise RuntimeError("Parent plan fingerprint differs")
    code = Path(plan["code_root"]).resolve()
    for module in (training, core, loader):
        if code not in Path(module.__file__).resolve().parents:
            raise RuntimeError("Imported diagnostic model/loader is outside the frozen code snapshot")
    verify_sources(plan["source_sha256"], code)
    verify_sources(manifest["source_sha256"], code)
    checkpoint, selected_step, checkpoint_binding = choose_checkpoint(root, step, expected_sha256)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (saved.get("step") != selected_step or saved.get("recipe") != training.RECIPE
            or saved.get("manifest") != manifest or manifest.get("theory_dim") != 1
            or manifest.get("model") != training.model_policy(2, 3)
            or manifest.get("lattice_sizes") != [128, 64, 32, 16]
            or manifest.get("labels_used_in_training") is not False):
        raise ValueError("Checkpoint recipe/step/manifest/geometry is incompatible")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(202611011000)
        model = core.AlignedPhi4QCR(manifest["scale"], hidden=96, theory_dim=1,
            local_features=12, patch_chunk_size=262144, block_size=2, steps=3)
    model.load_state_dict(saved["model"], strict=True)
    del saved  # Optimizer/RNG payloads are never restored or used.
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model_sha, modules_sha = training.state_fingerprint(model), training.module_fingerprints(model)
    cpu_rng = torch.get_rng_state().clone()

    def guard():
        if time.process_time() - cpu_started > max_cpu_seconds:
            raise RuntimeError("Bounded CPU preview budget exceeded; no automatic retry")

    dataset = loader.load_dataset(root / "data", block_size=2, depth=3)
    if dataset.manifest_sha256 != manifest["data_manifest_sha256"]:
        raise RuntimeError("Preview physical dataset differs from the checkpoint")
    selected = fixed_cell_indices(dataset.cell_ids)
    generator = torch.Generator(device="cpu").manual_seed(202611011000)
    direct, next_values = [], []
    guard()
    with torch.no_grad():
        for index in selected:
            levels = model.coarse_grain(dataset.configurations[index, :samples].unsqueeze(0),
                                       generator=generator)
            gs = [.5 * (model.Q(level[:, :samples // 2]) + model.Q(level[:, samples // 2:]))
                  for level in levels]
            direct.append([float(g.item()) for g in gs])
            next_values.append([float(model.advance(g).item()) for g in gs[:-1]])
            guard()
    direct, next_values = np.asarray(direct), np.asarray(next_values)
    ranges = [[float(direct[:, k].min()), float(direct[:, k].max())] for k in range(3)]
    low, high = float(direct[:, :3].min()), float(direct[:, :3].max())
    pad = .1 * max(high - low, 1e-6)
    grid = np.linspace(low - pad, high + pad, 501)
    x = torch.tensor(grid[:, None], dtype=torch.float64, requires_grad=True)
    mapped = model.advance(x)
    slopes = torch.autograd.grad(mapped.sum(), x)[0].detach().numpy()[:, 0]
    mapped = mapped.detach().numpy()[:, 0]
    drift = mapped - grid
    zeros = []
    for value in crossing_candidates(grid, drift):
        z = torch.tensor([[value]], dtype=torch.float64, requires_grad=True)
        rz = model.advance(z)
        slope = float(torch.autograd.grad(rz.sum(), z)[0].item())
        zeros.append(dict(g=value, full_DR=slope, residual=abs(float(rz.item()) - value),
            classification=("attracting" if abs(slope) < 1 - 1e-6 else
                            "repelling" if abs(slope) > 1 + 1e-6 else "neutral"),
            inside_sampled_input_range=bool(low <= value <= high),
            per_input_scale_range_membership=[bool(a <= value <= b) for a, b in ranges],
            nearest_preview_input_distance=float(np.abs(direct[:, :3] - value).min()),
            method="501-point grid sign crossing with linear interpolation; no iterative root search",
            physical_critical_point_certified=False, thermal_nu=None))
    guard()
    if (binding(checkpoint) != checkpoint_binding or sha(plan_path) != plan_sha
            or sha(manifest_path) != manifest_sha or training.state_fingerprint(model) != model_sha
            or training.module_fingerprints(model) != modules_sha
            or any(p.grad is not None for p in model.parameters())
            or not torch.equal(cpu_rng, torch.get_rng_state()) or torch.cuda.is_initialized()):
        raise RuntimeError("Read-only CPU checkpoint/model/RNG identity guard failed")
    verify_sources(plan["source_sha256"], code)
    verify_sources(manifest["source_sha256"], code)
    guard()
    if not all(np.isfinite(a).all() for a in (direct, next_values, grid, mapped, drift, slopes)):
        raise FloatingPointError("Nonfinite preview output")
    latest_status = read(root / "training/status.json")
    return dict(kind="explicit_user_requested_interim_CPU_preview", final_report=False,
        checkpoint_step=selected_step, checkpoint_sha256=checkpoint_binding[2],
        checkpoint_bytes=checkpoint_binding[0], frozen_plan_sha256=plan_sha,
        training_manifest_sha256=manifest_sha, model_state_sha256=model_sha,
        module_sha256=modules_sha, captured_at_utc=datetime.now(timezone.utc).isoformat(),
        training_status_at_finish=latest_status, device="cpu", cuda_initialized=False,
        checkpoint_unchanged=True, model_unchanged=True, source_unchanged=True,
        new_optimizer_updates=0, new_physical_sampling=0, optimizer_state_restored=False,
        training_rng_restored=False, preview_cpu_rng_unchanged=True,
        samples_per_chain=samples, sample_indices=list(range(samples)),
        physical_chains=len(selected), C_draws_per_chain=1, preview_seed=202611011000,
        selected_ensemble_indices=selected, chain_selection="first existing physical chain in each fixed cell",
        lattice_sizes=manifest["lattice_sizes"], TRAIN_microscopic_scale=manifest["scale"],
        direct=direct.tolist(), one_step=next_values.tolist(),
        labels=dataset.labels[selected].tolist(), cell_ids=dataset.cell_ids[selected].tolist(),
        splits=dataset.splits[selected].tolist(), per_input_scale_sampled_ranges=ranges,
        sampled_input_range=[low, high], flow_g=grid.tolist(), flow_R=mapped.tolist(),
        flow_displacement=drift.tolist(), full_DR=slopes.tolist(), interpolated_zeros=zeros,
        thermal_nu=None, data_warning_count=dataset.warning_count, host_resources=host,
        preview_wall_seconds=time.perf_counter() - started, preview_cpu_seconds=time.process_time() - cpu_started,
        preview_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1024 if sys.platform != "darwin" else 1),
        limitations=["中期保存权重，不是最终3000步结果或完整正式评价。",
            "每个κ仅一条物理链、前8个已保存构型、一次私有随机C轨迹；没有独立链置信区间。",
            "输入范围只是少量预览点的包络，不保证间隙内有数据或真实局部支撑。",
            "插值零点及神经映射导数不是已认证物理临界点；不计算热临界指数。",
            "CPU随机C轨迹不与正式GPU评价逐比特配对；原75条采样警告与热化不确定性保留。"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--step", type=int)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--max-cpu-seconds", type=float, default=300.)
    args = parser.parse_args(argv)
    print(json.dumps(capture(args.root, step=args.step, expected_sha256=args.expected_sha256,
                             samples=args.samples, max_cpu_seconds=args.max_cpu_seconds),
                     ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
