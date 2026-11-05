"""Bounded, CPU-only C configuration diagnostic; export only PNG and scalar JSON.

Execute the saved source over SSH stdin from the immutable model snapshot.
No physical sampling, Q/R construction, optimization, or checkpoint export.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np


BASE = Path("/mnt/nas-new/home/yange/ham")
FORMAL_ROOT = BASE / "results/phi4_hopping_b2_L128_g1_seed91_gpu3_vm225_20261101_integrator_refined_v1"
EARLY_ROOT = BASE / "results/phi4_hopping_b2_L128_g1_seed91_earlystop100_200_gpu3_vm225_20261102"
SNAPSHOT = BASE / "run_code/phi4_hopping_b2_L128_seed91_20261101_integrator_refined_v1"
EXPECTED_DATA = "3d2023b401b15e20a6dca6c6b602959c237a030ff0e2f8f89676a7ca9ed30fa5"
CORE_SHA = "155bdf2e32b33f5ad812fddfffedbf1c9e37f0ab3cce1e3c4b16e8f2663bbdba"
CHECKPOINTS = (
    ("learned120", EARLY_ROOT, 120, "18133a562398e1e22d9fdd04f92264835395d60e1a36e56b0adce6b45d15ec8c",
     "b755bf95c6dbb9aa5bc8b319eb6b20431697766fa089793ee45a2eb9b8983f42"),
    ("learned3000", FORMAL_ROOT, 3000, "1aec98b9d9c9d562ca32b6bef972cb79fa46ccc5f06819bb752129709dcf50d7",
     "b4c0cb400cc64638dac91ea591798947946e68cc91c078219cc77a529ded37af"),
)
CELL_KAPPAS = ((1, .50), (5, .62), (7, .68), (9, .74), (13, .86))
SAVED_INDICES = (32, 64, 96)
DISPLAY_CHAINS = (0, 2)
DISPLAY_INDEX = 64
METHODS = ("block_mean", "learned120", "learned3000")
NOISE_SEED = 202611020000


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def state_fingerprint(module):
    result = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous().numpy()
        digest = hashlib.sha256()
        for part in (str(value.dtype).encode(), str(value.shape).encode(), value.tobytes()):
            digest.update(part)
        result.update(name.encode())
        result.update(digest.hexdigest().encode())
    return result.hexdigest()


def block_mean(field):
    """Nonoverlapping 2 x 2 arithmetic average, origin zero, stride two."""
    x = np.asarray(field)
    if x.ndim != 2 or x.shape[0] != x.shape[1] or x.shape[0] % 2:
        raise ValueError("Expected a square even-sided field")
    length = x.shape[0]
    result = x.reshape(length // 2, 2, length // 2, 2).mean(axis=(1, 3))
    if not np.isclose(result.mean(), x.mean(), rtol=1e-12, atol=1e-12):
        raise AssertionError("Arithmetic block average must preserve the spatial mean")
    return result


def select_fields(values):
    """Preserve chain/sample/site axes; never squeeze away a physical axis."""
    if values.shape != (4, 128, 128, 128, 1) or values.dtype != np.float64:
        raise ValueError("Expected four chains, 128 saved 128 x 128 float64 fields")
    selected = np.take(values[..., 0], SAVED_INDICES, axis=1).copy()
    if selected.shape != (4, 3, 128, 128) or not np.isfinite(selected).all():
        raise ValueError("Unexpected selected field axes or nonfinite values")
    return selected


def moments(field, scale, original_mean=None):
    value = np.asarray(field, dtype=np.float64)
    if value.ndim != 2 or not np.isfinite(value).all() or not scale > 0:
        raise ValueError("Finite two-dimensional field and positive scale required")
    mean = float(value.mean())
    rms = float(np.sqrt(np.mean(value ** 2)))
    positive, negative, zero = np.mean(value > 0), np.mean(value < 0), np.mean(value == 0)
    source = mean if original_mean is None else original_mean
    sign_retention = .5 if source == 0 or mean == 0 else float(source * mean > 0)
    return dict(size=int(value.shape[0]), mean=mean, mean_over_s=mean / scale,
                rms=rms, rms_over_s=rms / scale,
                q_abs_mean_over_rms=abs(mean) / rms if rms else 0.,
                majority_fraction=float(max(positive, negative) + .5 * zero),
                sign_retention=sign_retention)


def oddness_ratio(positive_mean, negative_mean):
    """RMS of mu(x)+mu(-x), relative to the symmetric conditional-mean RMS."""
    plus, minus = np.asarray(positive_mean), np.asarray(negative_mean)
    if plus.shape != minus.shape or not np.isfinite(plus).all() or not np.isfinite(minus).all():
        raise ValueError("Paired finite conditional-mean fields required")
    denominator = float(np.sqrt(np.mean(.5 * (plus**2 + minus**2))))
    numerator = float(np.sqrt(np.mean((plus + minus)**2)))
    return numerator / denominator if denominator else 0.


def aggregate(rows):
    """Average saved configurations within chains, then equally over four chains."""
    result = []
    for cell, kappa in CELL_KAPPAS:
        for method in METHODS:
            for level in range(4):
                chain_rows = []
                for chain in range(4):
                    selected = [r["moments"][level] for r in rows
                                if r["cell_index"] == cell and r["method"] == method
                                and r["chain"] == chain]
                    if len(selected) != 3:
                        raise ValueError("Every chain must contribute exactly three saved fields")
                    metrics = {key: float(np.mean([r[key] for r in selected]))
                               for key in selected[0] if key != "size"}
                    chain_rows.append(dict(chain=chain, **metrics))
                means = {key: float(np.mean([r[key] for r in chain_rows]))
                         for key in chain_rows[0] if key != "chain"}
                result.append(dict(cell_index=cell, kappa=kappa, method=method, level=level,
                    size=128 // 2**level, chains=chain_rows, four_chain_mean=means))
    return result


def _complete(value, name):
    if (value.get("status") != "complete" or value.get("workflow_complete") is not True
            or value.get("returncode") != 0 or value.get("error")
            or value.get("requires_user_action") is not False):
        raise ValueError(f"{name} is not a clean completed workflow")


def render_grid(plt, fields, method, chain, scale, limit, target):
    names = {"block_mean": "二乘二算术块平均", "learned120": "学习 C：早停选择的 120 步",
             "learned3000": "学习 C：最终 3000 步"}
    fig, axes = plt.subplots(5, 4, figsize=(10.6, 12.6))
    for row, (cell, kappa) in enumerate(CELL_KAPPAS):
        levels = fields[(cell, chain, method)]
        for level, field in enumerate(levels):
            ax = axes[row, level]
            picture = ax.imshow(field / scale, cmap="RdBu_r", vmin=-limit, vmax=limit,
                                interpolation="nearest", origin="upper")
            metric = moments(field, scale)
            ax.set_title(f"尺度 {level}：{field.shape[0]}乘{field.shape[0]}\n"
                         + rf"$|m|/\mathrm{{RMS}}={metric['q_abs_mean_over_rms']:.2f}$", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            if level == 0:
                ax.set_ylabel(rf"$\kappa={kappa:.2f}$", fontsize=12)
    fig.suptitle(names[method] + "\n" + f"预先固定物理链 {chain}，保存序号 64；不翻转整体符号", fontsize=15)
    fig.subplots_adjust(left=.10, right=.87, top=.91, bottom=.075, hspace=.31, wspace=.06)
    cax = fig.add_axes((.90, .17, .024, .65))
    fig.colorbar(picture, cax=cax, label=r"场幅度 $\phi/s$（所有方法与尺度共用色标）")
    fig.text(.5, .040, "块平均无额外幅度重标度；学习 C 使用实际高斯随机输出。两套 C 配对使用相同私有噪声。", ha="center", fontsize=9)
    fig.text(.5, .018, "有限构型展示不能认证热化、临界点或精确 RG；原有采样质量警告全部保留。", ha="center", fontsize=9)
    fig.savefig(target, dpi=140)
    plt.close(fig)


def render_metrics(plt, summaries, target):
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.2), sharex=True, sharey="row")
    styles = (("block_mean", "二乘二块平均", "black"),
              ("learned120", "学习 C：120 步", "#1976b5"),
              ("learned3000", "学习 C：3000 步", "#bd442c"))
    for level in range(4):
        for method, label, color in styles:
            rows = [r for r in summaries if r["method"] == method and r["level"] == level]
            for ri, metric in enumerate(("q_abs_mean_over_rms", "majority_fraction")):
                ax = axes[ri, level]
                ax.plot([r["kappa"] for r in rows], [r["four_chain_mean"][metric] for r in rows],
                        "o-", color=color, label=label, markersize=4)
                for chain in range(4):
                    ax.scatter([r["kappa"] for r in rows], [r["chains"][chain][metric] for r in rows],
                               s=12, alpha=.25, color=color)
        axes[0, level].set_title(f"尺度 {level}：边长 {128 // 2**level}")
        axes[1, level].set_xlabel(r"耦合 $\kappa$")
        for ax in axes[:, level]:
            ax.grid(alpha=.2)
    axes[0, 0].set_ylabel(r"符号无关有序程度 $|m|/\mathrm{RMS}$")
    axes[1, 0].set_ylabel("同号多数占比")
    axes[0, 0].set_ylim(0, 1.03)
    axes[1, 0].set_ylim(.48, 1.02)
    axes[0, -1].legend(fontsize=9)
    fig.suptitle("少量既有构型的跨尺度符号聚集：每条件四条物理链", fontsize=16)
    fig.tight_layout(rect=(0, .09, 1, .94))
    fig.text(.5, .055, "每链三个固定保存构型先平均，再对四链等权平均；浅色点为各链均值，不是置信区间。", ha="center", fontsize=10)
    fig.text(.5, .021, "块平均可因局部噪声消减而提高该比例；这不认证幅度流、平衡或临界位置。", ha="center", fontsize=10)
    fig.savefig(target, dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-cpu-seconds", type=float, default=300.)
    args = parser.parse_args(argv)
    started = time.process_time()
    if not 0 < args.max_cpu_seconds <= 300:
        raise ValueError("CPU limit must be in (0,300]")
    output = args.output.resolve()
    if output.exists() or any(root == output or root in output.parents for root in (FORMAL_ROOT, EARLY_ROOT, SNAPSHOT)):
        raise ValueError("Require a new independent output directory outside frozen roots")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if os.uname().nodename != "thinklab-105-225":
        raise RuntimeError("Unexpected host; no fallback or migration")
    script_digest = os.environ.get("DIAGNOSTIC_SCRIPT_SHA256", "")
    if len(script_digest) != 64 or any(c not in "0123456789abcdef" for c in script_digest):
        raise ValueError("Require the saved diagnostic script SHA256")
    host = dict(hostname=os.uname().nodename, cpus=os.cpu_count(), load1=os.getloadavg()[0],
                mem_available_bytes=int(Path("/proc/meminfo").read_text().split("MemAvailable:")[1].split()[0]) * 1024,
                io_full_avg10=float(Path("/proc/pressure/io").read_text().split("full avg10=")[1].split()[0]))
    if (host["mem_available_bytes"] < 36 * 1024**3 or host["load1"] > 1.25 * host["cpus"]
            or host["io_full_avg10"] >= 20 or os.getpriority(os.PRIO_PROCESS, 0) < 10):
        raise RuntimeError("CPU resource/priority gate failed; no fallback")
    def guard():
        if time.process_time() - started > args.max_cpu_seconds:
            raise RuntimeError("CPU diagnostic budget exhausted; no retry")

    sys.path.insert(0, str(SNAPSHOT))
    import torch
    from recurrent_rg.phi4_aligned_qcr import _Block2GaussianCoarseGrainer
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA must remain uninitialized")
    if (Path(sys.modules[_Block2GaussianCoarseGrainer.__module__].__file__).resolve()
            != SNAPSHOT / "recurrent_rg/phi4_aligned_qcr.py"):
        raise ValueError("C implementation imported outside the frozen snapshot")
    if sha256_file(SNAPSHOT / "recurrent_rg/phi4_aligned_qcr.py") != CORE_SHA:
        raise ValueError("Frozen C source differs")
    protected, models, model_receipts, scale = {}, {}, [], None
    for method, root, step, expected_pt, expected_c in CHECKPOINTS:
        job, status, summary, manifest, plan = [read(root / name) for name in
            ("job.json", "training/status.json", "training/summary.json", "training/manifest.json", "plan.json")]
        _complete(job, "Parent")
        _complete(status, "Training")
        if job.get("plan_sha256") != sha256_file(root / "plan.json"):
            raise ValueError("Parent plan binding differs")
        if (summary.get("status") != "complete" or summary.get("evaluated_checkpoint_step", summary.get("actual_steps")) != step
                or manifest.get("data_manifest_sha256") != EXPECTED_DATA
                or manifest.get("lattice_sizes") != [128, 64, 32, 16]
                or manifest.get("block_size") != 2 or manifest.get("labels_used_in_training") is not False
                or Path(manifest["data_root"]) != FORMAL_ROOT / "data"
                or summary.get("final_module_sha256", {}).get("C") != expected_c):
            raise ValueError("Frozen model endpoint/geometry/data contract differs")
        for pins in (manifest["source_sha256"], plan["source_sha256"]):
            for name, digest in pins.items():
                path = Path(name) if Path(name).is_absolute() else Path(plan["code_root"]) / name
                if sha256_file(path) != digest:
                    raise ValueError(f"Frozen source/evidence differs: {path}")
                protected[path] = digest
        checkpoint_path = root / "training" / f"training_step_{step}.pt"
        if sha256_file(checkpoint_path) != expected_pt:
            raise ValueError("Requested saved checkpoint SHA differs")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("step") != step or checkpoint.get("manifest") != manifest:
            raise ValueError("Checkpoint step/manifest differs")
        c_state = {name[2:]: value for name, value in checkpoint["model"].items() if name.startswith("C.")}
        candidate_scale = float(c_state["scale"])
        if candidate_scale != manifest["scale"] or (scale is not None and candidate_scale != scale):
            raise ValueError("Model amplitude normalization differs")
        scale = candidate_scale
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(NOISE_SEED)
            model = _Block2GaussianCoarseGrainer(scale, hidden=96).eval()
        model.load_state_dict(c_state, strict=True)
        model.requires_grad_(False)
        if state_fingerprint(model) != expected_c:
            raise ValueError("Loaded C state fingerprint differs")
        del checkpoint, c_state  # Never restore optimizer or training RNG payloads.
        models[method] = model
        protected[checkpoint_path] = expected_pt
        for name in ("job.json", "training/status.json", "training/summary.json", "training/manifest.json", "plan.json"):
            protected[root / name] = sha256_file(root / name)
        model_receipts.append(dict(method=method, checkpoint_step=step, checkpoint_sha256=expected_pt,
                                   C_sha256=expected_c, root=str(root)))
        guard()
    cpu_rng = torch.get_rng_state().clone()
    data_root = FORMAL_ROOT / "data"
    if sha256_file(data_root / "manifest.json") != EXPECTED_DATA:
        raise ValueError("Physical manifest differs")
    physical = read(data_root / "manifest.json")
    if any(physical["options"].get(key) != value for key, value in
           (("sizes", [128]), ("quartic_lambda", 1.), ("h", 0.), ("chains", 4), ("samples", 128))):
        raise ValueError("Physical data geometry differs")
    protected[data_root / "manifest.json"] = EXPECTED_DATA
    rows, display = [], {}
    for cell, kappa in CELL_KAPPAS:
        shard = next(row for row in physical["shards"] if row["cell_index"] == cell)
        for name, digest in ((shard["path"], shard["sha256"]), (shard["sidecar"], shard["sidecar_sha256"])):
            path = (data_root / name).resolve()
            if data_root not in path.parents or sha256_file(path) != digest:
                raise ValueError("Selected shard/sidecar identity differs")
            protected[path] = digest
        sidecar = read(data_root / shard["sidecar"])
        if not math.isclose(sidecar["kappa"], kappa, rel_tol=0, abs_tol=1e-12):
            raise ValueError("Predeclared physical condition differs")
        with np.load(data_root / shard["path"], allow_pickle=False) as archive:
            selected = select_fields(archive["configurations"])
        for chain in range(4):
            for si, saved_index in enumerate(SAVED_INDICES):
                original = selected[chain, si]
                levels = [original]
                for _ in range(3):
                    levels.append(block_mean(levels[-1]))
                paths, oddness = {"block_mean": levels}, {"block_mean": [0., 0., 0.]}
                noise_seed = NOISE_SEED + cell * 10000 + chain * 1000 + saved_index
                for method, model in models.items():
                    generator = torch.Generator(device="cpu").manual_seed(noise_seed)
                    x = torch.from_numpy(original.copy())[None, None, :, :, None]
                    levels = [original]
                    ratios = []
                    with torch.no_grad():
                        for _ in range(3):
                            mu_plus, mu_minus = model.mean_field(x), model.mean_field(-x)
                            ratios.append(oddness_ratio(mu_plus.numpy(), mu_minus.numpy()))
                            x = model(x, generator=generator)
                            levels.append(x[0, 0, :, :, 0].numpy().copy())
                    paths[method] = levels
                    oddness[method] = ratios
                for method, levels in paths.items():
                    if [f.shape for f in levels] != [(size, size) for size in (128, 64, 32, 16)]:
                        raise ValueError("C trajectory geometry differs")
                    rows.append(dict(cell_index=cell, kappa=kappa, split=shard["split"], method=method,
                        chain=chain, chain_initial=sidecar["chains"][chain]["initial"], saved_index=saved_index,
                        noise_seed=noise_seed if method != "block_mean" else None,
                        conditional_mean_oddness_relative_RMS_by_input_level=oddness[method],
                        moments=[moments(f, scale, float(original.mean())) for f in levels]))
                    if chain in DISPLAY_CHAINS and saved_index == DISPLAY_INDEX:
                        display[(cell, chain, method)] = levels
                guard()
        del selected
    summaries = aggregate(rows)
    if not torch.equal(cpu_rng, torch.get_rng_state()) or torch.cuda.is_initialized():
        raise RuntimeError("Global RNG/CUDA read-only guard failed")
    for receipt in model_receipts:
        model = models[receipt["method"]]
        if state_fingerprint(model) != receipt["C_sha256"] or any(p.grad is not None for p in model.parameters()):
            raise RuntimeError("Read-only C state/gradient guard failed")
    for path, digest in protected.items():
        if sha256_file(path) != digest:
            raise RuntimeError(f"Frozen file changed during preview: {path}")
    guard()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if not font.exists():
        raise FileNotFoundError("Chinese plotting font unavailable")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(font)).get_name(),
                         "axes.unicode_minus": False, "font.size": 10})
    limit = math.ceil(max(float(np.max(np.abs(f))) / scale for levels in display.values() for f in levels) * 2) / 2
    output.mkdir(parents=True, exist_ok=False)
    public = []
    for method in METHODS:
        for chain in DISPLAY_CHAINS:
            target = output / f"{method}_chain{chain}.png"
            render_grid(plt, display, method, chain, scale, limit, target)
            public.append(dict(path=target.name, sha256=sha256_file(target), bytes=target.stat().st_size))
            guard()
    target = output / "phase_metrics.png"
    render_metrics(plt, summaries, target)
    public.append(dict(path=target.name, sha256=sha256_file(target), bytes=target.stat().st_size))
    guard()
    receipt = dict(diagnostic="phi4_b2_C_phase_configuration_check_v1", device="cpu", cpu_threads=1,
        host_resources=host, diagnostic_script_sha256=script_digest,
        models=model_receipts, C_state_unchanged=True, sources_and_checkpoints_unchanged=True,
        selected_physical_shards_verified_before_and_after=5, frozen_file_bindings={str(p): d for p, d in protected.items()},
        global_cpu_rng_unchanged=True, cuda_initialized=False, all_parameter_gradients_none=True,
        Q_or_R_constructed=False, new_optimizer_updates=0, new_physical_sampling=0,
        optimizer_or_training_rng_restored=False, fields_or_checkpoints_exported=False,
        physical_field_count=60, data_manifest_sha256=EXPECTED_DATA, data_warning_count=75,
        selection=dict(cells=list(CELL_KAPPAS), chains=[0, 1, 2, 3], saved_indices=list(SAVED_INDICES),
                       display_chains=list(DISPLAY_CHAINS), display_index=DISPLAY_INDEX,
                       policy="predeclared; no visual, sector, or quality selection"),
        shared_color_scale_over_s=[-limit, limit], shared_TRAIN_RMS=scale,
        global_sign_flip=False, color_clipping=False, private_noise_seed_base=NOISE_SEED,
        conditional_oddness_definition="RMS(mu(x)+mu(-x)) / sqrt(mean((mu(x)^2+mu(-x)^2)/2)); evaluated on actual stochastic path inputs, no extra trajectory or noise draw",
        scalar_rows=rows, chain_summaries=summaries, public_files=public,
        limitations=["少量固定保存构型，三次随机 C；无新物理采样，不认证热化或临界位置。",
          "先平均每链三个构型，再对四链等权平均；无链或随机 C 置信区间。",
          "不将正负扇区的原始磁化直接混合判序；以每构型 |m|/RMS 和同号多数占比作描述。",
          "符号保留率以原始空间均值符号为参照；零均值计 0.5，仅作描述。",
          "二乘二块平均保持空间均值，但滤除局部噪声可提高 |m|/RMS；并非精确 RG 或幅度流认证。",
          "学习 C 未强制奇对称；更均匀或单色不等于保留原物理相。",
          "原有 75 条采样质量警告及有限尺寸不确定性全部保留。"],
        cpu_seconds=time.process_time() - started)
    target = output / "receipt.json"
    with target.open("x") as stream:
        json.dump(receipt, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    print(json.dumps(dict(status="complete", output=str(output), cpu_seconds=receipt["cpu_seconds"],
        public_files=public, receipt_sha256=sha256_file(target)), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
