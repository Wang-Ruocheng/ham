"""Saved-array local charts of the phi4 r/u family, with chain uncertainty.

Coupling labels enter evaluation finite differences only. This script uses no
network, physical field, optimization, root search, or PCA. Finite-grid rank is
descriptive: chords of a curved one-dimensional image can also be noncollinear.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from scripts.report_phi4_qcr import _plotting, sha256


SCALES = ("k=0，L=36", "k=1，L=12", "k=2，L=4")
BOOTSTRAPS = 1000


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def stencil(index, count, stride):
    """Centered where available; forward/backward at the stencil boundary."""
    if count < 2 or not 1 <= stride < count or not 0 <= index < count:
        raise ValueError("Invalid finite-difference stencil")
    if index >= stride and index + stride < count:
        return index - stride, index + stride, True
    if index + stride < count:
        return index, index + stride, False
    return index - stride, index, False


def local_chart(g, r_values, u_values, *, stride=1):
    """Return dg/d(r/domain width), dg/d(u/domain width) in final columns.

    g has [..., r, u, scale, g_coordinate]. Leading bootstrap axes are retained.
    """
    g = np.asarray(g, dtype=np.float64)
    r, u = map(lambda x: np.asarray(x, dtype=np.float64), (r_values, u_values))
    if (g.shape[-4:-2] != (len(r), len(u)) or g.shape[-1] != 2
            or not np.isfinite(g).all() or np.any(np.diff(r) <= 0)
            or np.any(np.diff(u) <= 0)):
        raise ValueError("Invalid ordered two-parameter latent grid")
    rn, un = (r - r[0]) / np.ptp(r), (u - u[0]) / np.ptp(u)
    result = np.empty(g.shape + (2,), dtype=np.float64)
    interior = np.zeros((len(r), len(u)), dtype=bool)
    endpoints = np.empty((len(r), len(u), 2, 2), dtype=np.int64)
    for i in range(len(r)):
        ra, rb, rc = stencil(i, len(r), stride)
        for j in range(len(u)):
            ua, ub, uc = stencil(j, len(u), stride)
            result[..., i, j, :, :, 0] = (g[..., rb, j, :, :] - g[..., ra, j, :, :]) / (rn[rb] - rn[ra])
            result[..., i, j, :, :, 1] = (g[..., i, ub, :, :] - g[..., i, ua, :, :]) / (un[ub] - un[ua])
            interior[i, j] = rc and uc
            endpoints[i, j] = ((ra, rb), (ua, ub))
    return result, interior, endpoints


def bootstrap_chains(grouped, *, count=BOOTSTRAPS, seed=20261101):
    """Grouped [r,u,physical_chain,scale,g]; keep scales/coordinates paired."""
    grouped = np.asarray(grouped, dtype=np.float64)
    if grouped.ndim != 5 or grouped.shape[-1] != 2 or grouped.shape[2] < 2:
        raise ValueError("Expected a parameter grid with independent physical chains")
    nr, nu, chains = grouped.shape[:3]
    indices = np.random.default_rng(seed).integers(chains, size=(count, nr, nu, chains))
    sampled = grouped[np.arange(nr)[None, :, None, None],
                      np.arange(nu)[None, None, :, None], indices]
    return sampled.mean(axis=3), indices


def chart_statistics(chart, boot_chart):
    sigma = np.linalg.svd(chart, compute_uv=False)
    deviation = np.linalg.svd(boot_chart - chart, compute_uv=False)[..., 0]
    noise95 = np.quantile(deviation, .95, axis=0)
    ratio = np.divide(sigma[..., 1], sigma[..., 0], out=np.zeros_like(noise95),
                      where=sigma[..., 0] > 0)
    resolved = (ratio > .01) & (sigma[..., 1] > 2 * noise95)
    return sigma, noise95, ratio, resolved


def counts(resolved, interior, train_grid):
    result = {}
    for name, mask in (("all", np.ones(interior.shape, bool)), ("interior", interior),
                       ("one_sided", ~interior), ("train", train_grid), ("test", ~train_grid)):
        result[name] = dict(conditions=int(mask.sum()),
                            resolved_rank2_by_scale=resolved[mask].sum(axis=0).tolist())
    return result


def run(cache, output):
    started = time.process_time()
    cache, output = Path(cache).resolve(), Path(output).resolve()
    if output == cache or cache in output.parents:
        raise ValueError("Output must be separate from immutable cache")
    output.mkdir(parents=True, exist_ok=False)
    paths = [cache / name for name in ("cache.npz", "manifest.json", "receipt.json",
                                       "status.json", "public_files.json")]
    paths += [Path(__file__).resolve(), Path(__file__).resolve().with_name("report_phi4_qcr.py")]
    hashes = {str(p): sha256(p) for p in paths}
    state = json.loads((cache / "status.json").read_text())
    if state.get("status") != "complete" or not state.get("workflow_complete"):
        raise ValueError("Frozen cache is not complete")
    inventory = json.loads((cache / "public_files.json").read_text())
    for name, digest in inventory.items():
        if (cache / name).parent != cache or sha256(cache / name) != digest:
            raise ValueError(f"Public cache fingerprint mismatch: {name}")
    with np.load(cache / "cache.npz", allow_pickle=False) as z:
        fit = z["fit_direct"].copy()
        direct = np.concatenate((z["eval_train_direct"], z["eval_test_direct"]), axis=1)
        labels = np.concatenate((z["train_labels"], z["test_labels"]))
        cell_ids = np.concatenate((z["train_cell_ids"], z["test_cell_ids"]))
        train_ids = z["train_cell_ids"].copy()
    if direct.shape != (4, 180, 3, 2) or not np.isfinite(direct).all():
        raise ValueError("Unexpected repeated evaluation encodings")
    if set(train_ids) & set(cell_ids[len(train_ids):]):
        raise ValueError("Condition split overlaps")
    r, u = np.unique(labels[:, 0]), np.unique(labels[:, 1])
    if len(r) != 9 or len(u) != 5 or not np.all(labels[:, 2] == 0):
        raise ValueError("Expected the frozen 9 by 5 h=0 parameter grid")
    grouped = np.empty((len(r), len(u), 4, 3, 2))
    grid_ids = np.empty((len(r), len(u)), dtype=np.int64)
    train_grid = np.empty(grid_ids.shape, dtype=bool)
    chain_mean = direct.mean(axis=0)
    for i, rv in enumerate(r):
        for j, uv in enumerate(u):
            mask = (labels[:, 0] == rv) & (labels[:, 1] == uv)
            if mask.sum() != 4 or len(np.unique(cell_ids[mask])) != 1:
                raise ValueError("Each parameter grid point must have four physical chains")
            grouped[i, j] = chain_mean[mask]
            grid_ids[i, j] = cell_ids[mask][0]
            train_grid[i, j] = grid_ids[i, j] in train_ids
    center = fit[..., :2, :].reshape(-1, 2).mean(axis=0)
    scale = fit[..., :2, :].reshape(-1, 2).std(axis=0, ddof=0)
    if np.any(scale <= 0):
        raise ValueError("Degenerate FIT TRAIN input coordinate scale")
    means = grouped.mean(axis=2)
    boot, boot_indices = bootstrap_chains(grouped)
    payload = dict(r_values=r, u_values=u, cell_ids=grid_ids, train_grid=train_grid,
                   cell_mean_g=means, chain_mean_g=grouped, fit_input_center=center,
                   fit_input_std=scale, bootstrap_cell_mean_g=boot,
                   bootstrap_chain_indices=boot_indices)
    analyses = {}
    for stride in (1, 2):
        chart, interior, endpoints = local_chart((means - center) / scale, r, u, stride=stride)
        bchart, _, _ = local_chart((boot - center) / scale, r, u, stride=stride)
        sigma, noise, ratio, resolved = chart_statistics(chart, bchart)
        sensitivity = []
        for ratio_cut in (.01, .03, .05, .1):
            for noise_factor in (1., 2., 3.):
                flag = (ratio > ratio_cut) & (sigma[..., 1] > noise_factor * noise)
                sensitivity.append(dict(ratio_cut=ratio_cut, noise_factor=noise_factor,
                                         counts=counts(flag, interior, train_grid)))
        analyses[f"stride{stride}"] = dict(counts=counts(resolved, interior, train_grid),
            sigma1_median_by_scale=np.median(sigma[..., 0], axis=(0, 1)).tolist(),
            sigma2_median_by_scale=np.median(sigma[..., 1], axis=(0, 1)).tolist(),
            sigma_ratio_median_by_scale=np.median(ratio, axis=(0, 1)).tolist(),
            noise95_median_by_scale=np.median(noise, axis=(0, 1)).tolist(),
            sensitivity=sensitivity)
        payload.update({f"stride{stride}_{key}": value for key, value in (
            ("chart", chart), ("singular_values", sigma), ("bootstrap_noise95", noise),
            ("sigma_ratio", ratio), ("resolved_rank2", resolved),
            ("centered_interior", interior), ("stencil_endpoints", endpoints))})
    common = payload["stride2_centered_interior"]
    agreement = payload["stride1_resolved_rank2"] & payload["stride2_resolved_rank2"]
    delta_chart = payload["stride2_chart"] - payload["stride1_chart"]
    sensitivity_between_stencils = dict(common_centered_conditions=int(common.sum()),
        resolved_in_both_by_scale=agreement[common].sum(axis=0).tolist(),
        all_condition_flag_agreement_by_scale=np.mean(
            payload["stride1_resolved_rank2"] == payload["stride2_resolved_rank2"], axis=(0, 1)).tolist(),
        chart_change_operatornorm_median_by_scale=np.median(
            np.linalg.svd(delta_chart, compute_uv=False)[..., 0], axis=(0, 1)).tolist())
    plt = _plotting(None)
    fig, axes = plt.subplots(1, 3, figsize=(12., 4.2))
    for k, ax in enumerate(axes):
        x = 2 * payload["stride1_bootstrap_noise95"][..., k]
        y = payload["stride1_singular_values"][..., k, 1]
        for name, mask, color in (("训练条件", train_grid, "#518db0"),
                                   ("留出条件", ~train_grid, "#d49154")):
            ax.scatter(x[mask], y[mask], s=24, color=color, alpha=.85, label=name)
        ceiling = max(float(x.max()), float(y.max())) * 1.05
        ax.plot([0, ceiling], [0, ceiling], color="#777777", linestyle="--", linewidth=.8)
        ax.set(title=SCALES[k], xlabel="2 × bootstrap 噪声95%范数", ylabel="第二奇异值 σ₂",
               xlim=(0, ceiling), ylim=(0, ceiling))
        ax.set_aspect("equal", adjustable="box")
        ax.text(.97, .04, f"描述性二维可分辨 {int(payload['stride1_resolved_rank2'][..., k].sum())}/45",
                ha="right", va="bottom", transform=ax.transAxes, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("局部参数映射的二维可分辨性", fontsize=14)
    fig.text(.5, .015, "散点为参数条件；r、u 按扫描宽度缩放，g 按 FIT TRAIN 标准差缩放。\n"
             "有限网格奇异值反映可分辨的局部非共线性，不能单独认证内禀维度。",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .09, 1, .95))
    fig.savefig(output / "local_geometry.png", dpi=190, bbox_inches="tight")
    plt.close(fig)
    np.savez_compressed(output / "geometry_arrays.npz", **payload)
    summary = dict(status="complete", workflow_complete=True, returncode=0,
        method="finite-difference local charts of repeated-encoding, physical-chain means",
        parameter_grid=[len(r), len(u)], parameter_ranges=dict(r=[float(r.min()), float(r.max())],
        u=[float(u.min()), float(u.max())]), scales=[36, 12, 4], physical_chains_per_condition=4,
        encoding_repeats_per_chain=4, bootstrap_replicates=BOOTSTRAPS,
        whitening="g diagonal standard deviation, FIT TRAIN input scales 0/1 only; no PCA",
        input_chart_coordinates="r divided by full r scan width; u divided by full u scan width",
        bootstrap="Within each parameter condition resample four physical chains after averaging subset/C repeats; scales/coordinates paired",
        noise_definition="95th percentile of spectral norm (bootstrap chart minus original mean chart)",
        descriptive_rank2_rule="sigma2/sigma1 > 0.01 and sigma2 > 2*noise95",
        analyses=analyses, stencil_sensitivity=sensitivity_between_stencils,
        labels_used_for_training=False, labels_used_for_evaluation_chart=True,
        physical_sampling=False, model_inference=False, new_training=False, root_search=False,
        gpu_used=False, input_sha256=hashes, cpu_seconds=time.process_time() - started,
        limitations=[
            "Finite-difference chord directions of a curved one-dimensional image may be non-collinear; rank2 flags do not certify intrinsic dimension.",
            "One-sided derivatives at scan/stencil boundaries and two-grid-step sensitivity are reported separately.",
            "Rank criteria depend on the chosen dimensionless parameter and latent coordinate scaling.",
            "Four physical chains and a finite correlated source dataset limit bootstrap resolution; sampling warnings persist and no equilibrium certification is made.",
            "Bootstrap covers between-chain variation conditional on frozen Q/C and the cached repeat means; it does not include Q/C training-seed or discretization bias.",
            "Failure to resolve a second direction is not evidence that the underlying physical family is one-dimensional or that Q needs a larger output dimension.",
            "Successive observed scales arise from one 36→12→4 trajectory, not independent physical systems of three sizes.",
            "No phase boundary, fixed point, thermal eigenvalue, or missing-state claim follows from this geometry diagnostic alone."])
    if any(sha256(path) != digest for path, digest in hashes.items()):
        raise RuntimeError("Immutable input or script changed during analysis")
    write_json(output / "summary.json", summary)
    write_json(output / "public_files.json", {p.name: sha256(p) for p in output.iterdir() if p.is_file()})
    print(json.dumps(dict(output=str(output), analyses={name: value["counts"] for name, value in analyses.items()},
                          cpu_seconds=summary["cpu_seconds"])), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.cache, args.output)


if __name__ == "__main__":
    main()
