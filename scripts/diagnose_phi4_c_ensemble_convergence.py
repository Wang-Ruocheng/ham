"""Read-only final-C diagnostics on saved current-run fields, CPU only.

Export scalar diagnostics and rendered PNGs, never fields/checkpoints. Run on
the original host via SSH stdin. No Q/R inference, physical sampling or updates.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np


FORMAL_ROOT = Path("/mnt/nas-new/home/yange/ham/results/phi4_hopping_b3_g1_aligned_seed91_hmc004_budget4h_gpu3_vm225_20261031")
SNAPSHOT = Path("/mnt/nas-new/home/yange/ham/run_code/phi4_hopping_aligned_seed91_refined_hmc_20261031")
EXPECTED_C = "bfcece7c7f45249a616a5762700fec6d8aaba60e4a7ea3a58e466bb17b484332"
EXPECTED_DATA = "61892f0306ad0c2596f621dd4cb0fb94fd67ce54ee4372ebeb54ee8c2f82c683"
CELLS = (0, 6, 7, 8, 9, 10, 11, 12)
INDICES = tuple(range(0, 128, 8))
NOISE_BASE_SEED = 20314232
METRICS = ("m_over_s", "rms_over_s", "std_over_s", "q", "majority_fraction",
           "relative_spatial_fluctuation", "original_sign_retention", "unanimous")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def block_mean(fields):
    fields = np.asarray(fields)
    side = fields.shape[-1]
    if fields.shape[-2] != side or side % 3:
        raise ValueError("Expected square fields divisible by three")
    result = fields.reshape(*fields.shape[:-2], side // 3, 3, side // 3, 3).mean(axis=(-3, -1))
    np.testing.assert_allclose(result.mean(axis=(-2, -1)), fields.mean(axis=(-2, -1)), atol=1e-12, rtol=1e-12)
    return result


def select_saved_fields(value, indices=INDICES):
    """Keep physical-chain axis first; avoid mixed advanced-index reordering."""
    if value.shape != (4, 128, 36, 36, 1):
        raise ValueError("Expected the current experiment's chain/sample/space/channel axes")
    selected = np.ascontiguousarray(np.take(value[..., 0], indices, axis=1))
    if selected.shape != (4, len(indices), 36, 36):
        raise AssertionError("Physical-chain/sample axes changed during selection")
    return selected


def field_metrics(fields, scale, original_sign):
    flat = np.asarray(fields, dtype=np.float64).reshape(*fields.shape[:2], -1) / scale
    if flat.shape[:2] != original_sign.shape or not np.isfinite(flat).all():
        raise ValueError("Invalid physical chain/sample axes")
    mean = flat.mean(axis=-1)
    rms = np.sqrt(np.mean(flat ** 2, axis=-1))
    if np.any(rms <= 0):
        raise ValueError("Zero RMS cannot define relative field metrics")
    std = flat.std(axis=-1)
    majority = np.maximum((flat > 0).mean(axis=-1), (flat < 0).mean(axis=-1))
    return dict(m_over_s=mean, rms_over_s=rms, std_over_s=std, q=np.abs(mean) / rms,
                majority_fraction=majority, relative_spatial_fluctuation=std / rms,
                original_sign_retention=(np.sign(mean) == original_sign).astype(float),
                unanimous=(majority == 1.).astype(float))


def cross_chain_distances(fields, scale, signs=None):
    """Per-site distances; only pairs from different physical chains.

    Pool pair MSE and pair energy first, then take sqrt; never subtract vectors
    from different lattice sizes. Chain-pair results are descriptive, not CIs.
    """
    flat = np.asarray(fields, dtype=np.float64).reshape(*fields.shape[:2], -1) / scale
    if signs is not None:
        flat = flat * signs[..., None]
    records = []
    for first, second in itertools.combinations(range(flat.shape[0]), 2):
        a, b = flat[first], flat[second]
        qa, qb = np.mean(a ** 2, axis=-1), np.mean(b ** 2, axis=-1)
        mse = qa[:, None] + qb[None, :] - 2 * a.dot(b.T) / a.shape[-1]
        if mse.min() < -1e-10:
            raise AssertionError("Negative squared distance beyond rounding")
        mse = np.maximum(mse, 0.)
        energy = .5 * (qa[:, None] + qb[None, :])
        amplitude = (a.mean(axis=-1)[:, None] - b.mean(axis=-1)[None, :]) ** 2
        records.append(dict(chains=[first, second], mse_over_s2=float(mse.mean()),
                            pair_energy_over_s2=float(energy.mean()),
                            amplitude_mse_over_s2=float(amplitude.mean()),
                            texture_mse_over_s2=float(mse.mean() - amplitude.mean())))
    mse = float(np.mean([r["mse_over_s2"] for r in records]))
    energy = float(np.mean([r["pair_energy_over_s2"] for r in records]))
    return dict(relative_rms=math.sqrt(mse / energy), absolute_rms_over_s=math.sqrt(mse),
                mse_over_s2=mse, pair_energy_over_s2=energy,
                amplitude_mse_over_s2=float(np.mean([r["amplitude_mse_over_s2"] for r in records])),
                texture_mse_over_s2=float(np.mean([r["texture_mse_over_s2"] for r in records])),
                chain_pairs=records)


def aggregate(fields_by_draw, scale, original_sign):
    per_field = [field_metrics(fields, scale, original_sign) for fields in fields_by_draw]
    result = {}
    for key in METRICS:
        means = np.mean([row[key] for row in per_field], axis=0).mean(axis=1)
        result[key] = dict(mean=float(means.mean()), chain_means=means.tolist(),
                           chain_range=[float(means.min()), float(means.max())])
    for prefix, signs in (("unaligned", None), ("aligned", original_sign)):
        values = [cross_chain_distances(fields, scale, signs) for fields in fields_by_draw]
        for key, source in (("relative_rms_distance", "relative_rms"),
                            ("absolute_rms_distance_over_s", "absolute_rms_over_s")):
            chain_values = []
            for pair in range(len(values[0]["chain_pairs"])):
                estimates = [row["chain_pairs"][pair] for row in values]
                chain_values.append(float(np.mean([
                    math.sqrt(item["mse_over_s2"] / item["pair_energy_over_s2"])
                    if source == "relative_rms" else math.sqrt(item["mse_over_s2"])
                    for item in estimates])))
            result[prefix + "_" + key] = dict(mean=float(np.mean([row[source] for row in values])),
                chain_pair_means=chain_values, chain_pair_range=[min(chain_values), max(chain_values)])
        result[prefix + "_distance_decomposition"] = {
            key: float(np.mean([row[key] for row in values]))
            for key in ("mse_over_s2", "pair_energy_over_s2", "amplitude_mse_over_s2", "texture_mse_over_s2")}
    return result


def render_samples(plt, levels, kappa, scale, limit, signs, output):
    """Eight predeclared examples: all chains, saved positions 0 and 64."""
    positions = (0, 8)  # positions in INDICES, not saved sample numbers
    examples = [(chain, pos) for chain in range(4) for pos in positions]
    fig, axes = plt.subplots(3, 8, figsize=(19, 8))
    for col, (chain, pos) in enumerate(examples):
        orientation = 1 if signs is None else signs[chain, pos]
        for row, field in enumerate(levels):
            ax = axes[row, col]
            picture = ax.imshow(orientation * field[chain, pos] / scale, cmap="RdBu_r",
                                vmin=-limit, vmax=limit, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"物理链 {chain}／序号 {INDICES[pos]}", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{field.shape[-1]}乘{field.shape[-1]}", fontsize=12)
    title = "原始输出：保留正负对称扇区" if signs is None else "辅助展示：按原始磁化符号对齐"
    fig.suptitle(rf"$\kappa={kappa:.2f}$：多组已保存构型的尺度变化" + "\n" + title, fontsize=16)
    fig.subplots_adjust(left=.055, right=.93, top=.86, bottom=.12, hspace=.10, wspace=.035)
    cax = fig.add_axes((.945, .17, .014, .60))
    fig.colorbar(picture, cax=cax, label=r"$\phi/s$（所有构型图共用色标）")
    note = "仅为显示作全局正负翻转；C 输入和输出均未因此改变。" if signs is not None else "未按正负扇区选样或翻转；每条链展示预先固定的保存序号 0、64。"
    fig.text(.5, .065, note, ha="center", fontsize=11)
    fig.text(.5, .035, "只有两次粗粒化；图像趋同不等于相变或 RG 正确性认证。", ha="center", fontsize=11)
    fig.savefig(output, dpi=145)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.process_time()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if args.output.exists() or FORMAL_ROOT in args.output.resolve().parents:
        raise FileExistsError("Refuse overwrite or writes inside frozen formal run")
    for name in ("job.json", "training/status.json"):
        status = json.loads((FORMAL_ROOT / name).read_text())
        if status.get("status") != "complete" or status.get("workflow_complete") is not True:
            raise ValueError("Formal workflow is not complete")
    manifest = json.loads((FORMAL_ROOT / "training/manifest.json").read_text())
    summary = json.loads((FORMAL_ROOT / "training/summary.json").read_text())
    for name, expected in manifest["source_sha256"].items():
        if SNAPSHOT not in Path(name).parents or sha256_file(name) != expected:
            raise ValueError("Frozen source mismatch")
    data_root = Path(manifest["data_root"])
    if data_root != FORMAL_ROOT / "data" or sha256_file(data_root / "manifest.json") != EXPECTED_DATA:
        raise ValueError("Current data manifest mismatch")
    physical = json.loads((data_root / "manifest.json").read_text())
    quality = json.loads((data_root / "quality_summary.json").read_text())
    sys.path.insert(0, str(SNAPSHOT))
    import torch
    from recurrent_rg.phi4_qcr import GaussianCoarseGrainer
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    path = FORMAL_ROOT / "training/training_step_3000.pt"
    checkpoint_sha = sha256_file(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["step"] != 3000:
        raise ValueError("Unexpected final step")
    c_state = {name[2:]: value for name, value in checkpoint["model"].items() if name.startswith("C.")}
    scale = float(c_state["scale"])
    model = GaussianCoarseGrainer(scale, 96, .1).eval().requires_grad_(False)
    model.load_state_dict(c_state, strict=True)
    if state_fingerprint(model) != EXPECTED_C or summary["final_module_sha256"]["C"] != EXPECTED_C:
        raise ValueError("Final C fingerprint mismatch")
    del checkpoint, c_state
    results, condition_receipts, noise, display, arrays = [], [], [], {}, {}
    for cell in CELLS:
        shard = next(row for row in physical["shards"] if row["cell_index"] == cell)
        source, sidecar_path = data_root / shard["path"], data_root / shard["sidecar"]
        if sha256_file(source) != shard["sha256"] or sha256_file(sidecar_path) != shard["sidecar_sha256"]:
            raise ValueError("Physical shard fingerprint mismatch")
        sidecar = json.loads(sidecar_path.read_text())
        with np.load(source, allow_pickle=False) as archive:
            value = archive["configurations"]
        if value.shape != (4, 128, 36, 36, 1) or value.dtype != np.float64 or not np.isfinite(value).all():
            raise ValueError("Unexpected current-run physical fields")
        original = select_saved_fields(value)
        signs = np.sign(original.mean(axis=(-2, -1)))
        if np.any(signs == 0):
            raise ValueError("Exact zero microscopic m cannot define diagnostic orientation")
        del value
        baseline = [original, block_mean(original)]
        baseline.append(block_mean(baseline[-1]))
        x = torch.from_numpy(original[..., None].copy())
        random_levels = []
        seeds = []
        with torch.inference_mode():
            for draw in range(2):
                seed = NOISE_BASE_SEED + 100 * cell + draw
                seeds.append(seed)
                generator = torch.Generator(device="cpu").manual_seed(seed)
                first = model(x, generator=generator)
                second = model(first, generator=generator)
                random_levels.append([original, first[..., 0].numpy().copy(), second[..., 0].numpy().copy()])
            first = model.mean_field(x)
            second = model.mean_field(first)
            means = [original, first[..., 0].numpy().copy(), second[..., 0].numpy().copy()]
        methods = dict(block_mean=[baseline], learned_sample=random_levels, learned_mean_chain=[means])
        for draws in methods.values():
            for levels in draws:
                if [row.shape for row in levels] != [(4, 16, size, size) for size in (36, 12, 4)]:
                    raise AssertionError("C diagnostic changed physical-chain/sample axes")
        kappa = sidecar["kappa"]
        results.append(dict(kappa=kappa, cell_index=cell, split=shard["split"], method="original",
                            scale=36, level=0, metrics=aggregate([original], scale, signs)))
        for method, draws in methods.items():
            for level in range(3):
                fields = [row[level] for row in draws]
                row_metrics = [field_metrics(field, scale, signs) for field in fields]
                # Export only scalar field summaries, not any site-level array.
                for key in METRICS:
                    arrays[f"c{cell:04d}_{method}_level{level}_{key}"] = np.stack([row[key] for row in row_metrics])
                if level:
                    results.append(dict(kappa=kappa, cell_index=cell, split=shard["split"], method=method,
                                        scale=fields[0].shape[-1], level=level,
                                        metrics=aggregate(fields, scale, signs)))
        for level in (1, 2):
            difference = (random_levels[0][level] - random_levels[1][level]) / scale
            per_chain = np.mean(difference ** 2, axis=(1, 2, 3))
            noise.append(dict(kappa=kappa, level=level, scale=random_levels[0][level].shape[-1],
                              same_input_two_draw_rms_over_s=float(np.sqrt(per_chain.mean())),
                              chain_rms_over_s=np.sqrt(per_chain).tolist(),
                              interpretation="two full independent C trajectories per same saved physical field; not two physical samples"))
        condition_receipts.append(dict(cell_index=cell, kappa=kappa, split=shard["split"],
            source_path=shard["path"], source_sha256=shard["sha256"], sidecar_sha256=shard["sidecar_sha256"],
            noise_seeds=seeds, selected_positive_original_m=int(np.sum(signs > 0)),
            selected_negative_original_m=int(np.sum(signs < 0)),
            physical_quality=next(row for row in quality["conditions"] if row["cell_index"] == cell)))
        if cell in (0, 7, 12):
            display[cell] = dict(kappa=kappa, learned_sample=random_levels[0], block_mean=baseline, signs=signs)
        print(json.dumps(dict(event="condition_complete", cell=cell, kappa=kappa)), flush=True)

    if state_fingerprint(model) != EXPECTED_C or any(p.grad is not None for p in model.parameters()):
        raise AssertionError("Model changed")
    if sha256_file(path) != checkpoint_sha:
        raise AssertionError("Checkpoint changed")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if not font.exists():
        raise FileNotFoundError("Chinese font unavailable")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(font)).get_name(), "axes.unicode_minus": False})
    maximum = max(np.max(np.abs(field)) / scale for row in display.values()
                  for method in ("learned_sample", "block_mean") for field in row[method])
    limit = math.ceil(2 * maximum) / 2
    args.output.mkdir(parents=True, exist_ok=False)
    files = []
    for cell, record in display.items():
        for aligned in ((False,) if cell == 0 else (False, True)):
            target = args.output / f"learned_{'aligned' if aligned else 'raw'}_c{cell:04d}.png"
            render_samples(plt, record["learned_sample"], record["kappa"], scale, limit,
                           record["signs"] if aligned else None, target)
            files.append(dict(path=target.name, sha256=sha256_file(target), bytes=target.stat().st_size))
    target = args.output / "block_mean_aligned_c0012.png"
    render_samples(plt, display[12]["block_mean"], display[12]["kappa"], scale, limit, display[12]["signs"], target)
    files.append(dict(path=target.name, sha256=sha256_file(target), bytes=target.stat().st_size))
    target = args.output / "per_field_metrics.npz"
    np.savez_compressed(target, **arrays)
    files.append(dict(path=target.name, sha256=sha256_file(target), bytes=target.stat().st_size))
    output = dict(metadata=dict(diagnostic="final_C_saved_ensemble_convergence_v2", kappas=[r["kappa"] for r in condition_receipts],
        scales=[36, 12, 4], field_scale_s=scale, n_physical_per_condition=64, n_physical_total=512,
        n_physical_chains_per_condition=4, n_c_draws=2, sample_indices=list(INDICES), reference_kappa=.680601,
        noise_std=float(model.noise_std), noise_seed_base=NOISE_BASE_SEED, checkpoint_step=3000,
        final_C_sha256=EXPECTED_C, checkpoint_sha256=checkpoint_sha, data_manifest_sha256=EXPECTED_DATA,
        C_state_unchanged=True, no_parameter_gradients=True, physical_sampling_or_training=False,
        learned_output_sampling=True, Q_or_R_inference=False, device="cpu", cpu_threads=1,
        raw_fields_or_checkpoint_exported=False, cpu_seconds=time.process_time() - started,
        method_notes={"block_mean":"3x3 arithmetic average, no additional field renormalization or noise",
                      "learned_sample":"actual Gaussian C; each metric computed for each of two draws then averaged",
                      "learned_mean_chain":"composition of conditional means, not expectation of stochastic two-step C"},
        source_pins_verified=manifest["source_sha256"], color_scale_over_s=[-limit, limit], color_clipping=False),
        metric_definitions={
            "q":"mean over physical fields of abs(spatial mean)/sqrt(spatial mean(field^2)); 1 means uniform nonzero field",
            "relative_spatial_fluctuation":"spatial std/sqrt(spatial mean(field^2)); 0 means uniform",
            "majority_fraction":"mean over fields of max(fraction positive sites, fraction negative sites)",
            "alignment":"auxiliary multiplication of every output by sign(original microscopic spatial mean); no actual C input/output changed; same sign across all levels",
            "distances":"only different physical chains, all16x16 saved-sample pairs within each of6 chain pairs; equal chain-pair weights; per-site MSE and pair-average energy pooled first then sqrt; independent C draws computed separately then averaged",
            "aligned_relative_rms_distance":"sqrt(mean_cross_pairs spatial_mean((sign(m0_i)*y_i-sign(m0_j)*y_j)^2) / mean_cross_pairs[(spatial_mean(y_i^2)+spatial_mean(y_j^2))/2])",
            "aligned_absolute_rms_distance_over_s":"sqrt(mean_cross_pairs spatial_mean((sign(m0_i)*y_i-sign(m0_j)*y_j)^2))/s",
            "unaligned_distances":"same formulas without sign multiplication",
            "distance_decomposition":"MSE=global-mean-difference squared + centered-texture MSE; all in s^2 units",
            "ranges":"descriptive chain or chain-pair ranges; no independence assertion or confidence interval"},
        results=results, conditions=condition_receipts, noise_repeats=noise, public_files=files,
        qualifications=["Same phase is not same global sign or identical configurations",
                        "Only two RG steps; not a convergence-to-infinite-depth test",
                        "Four physical chains per condition; saved configurations may remain correlated; C draws are not physical replicas",
                        "All physical quality warnings retained; no equilibrium certification",
                        "Some scanned conditions were TRAIN, some TEST; split is reported, not all heldout",
                        "No root search, Q/R evaluation, training, physical resampling, parameter update or data repair",
                        "These diagnostics alone cannot establish the causal source of Q/R closure or fixed-point error"])
    (args.output / "convergence.json").write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status="complete", output=str(args.output), cpu_seconds=output["metadata"]["cpu_seconds"],
                         public_files=files, convergence_sha256=sha256_file(args.output / "convergence.json")), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
