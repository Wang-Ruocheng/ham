"""Compare saved paired coarse-graining displacements with frozen R outputs.

This diagnostic reads public NPZ/JSON only. It does not load a network, search
for roots, sample fields, or optimize parameters. Subset/C repeats are averaged
inside each physical chain before between-chain summaries and bootstrapping.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from scripts.report_phi4_qcr import _plotting, sha256


SELECTED = "scaled256_seed93"
COMPARISON = "scaled96_seed93"
PATH_NAMES = ("36→12", "12→4")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def chain_and_cell_means(repeated, cell_ids):
    """Return C/subset-averaged chains and equally weighted condition means."""
    repeated, cell_ids = np.asarray(repeated), np.asarray(cell_ids)
    if repeated.ndim < 3 or repeated.shape[1] != len(cell_ids):
        raise ValueError("Expected [repeat, physical chain, ...]")
    if not np.isfinite(repeated).all():
        raise ValueError("Nonfinite repeated measurements")
    chains = repeated.mean(axis=0)
    cells = np.unique(cell_ids)
    counts = [int(np.sum(cell_ids == cell)) for cell in cells]
    if len(set(counts)) != 1 or counts[0] < 2:
        raise ValueError("This diagnostic requires equally many physical chains per condition")
    grouped = np.stack([chains[cell_ids == cell] for cell in cells])
    return chains, grouped, grouped.mean(axis=1), cells


def bootstrap_cell_means(grouped, *, count=1000, seed=20261031):
    """Keep all scales/coordinates of a physical chain paired in a resample."""
    grouped = np.asarray(grouped)
    cells, chains = grouped.shape[:2]
    rng = np.random.default_rng(seed)
    indices = rng.integers(chains, size=(count, cells, chains))
    sampled = grouped[np.arange(cells)[None, :, None], indices]
    return sampled.mean(axis=2), indices


def vector_metrics(truth, prediction):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if truth.shape != prediction.shape or truth.shape[-1] != 2:
        raise ValueError("Paired two-component vectors required")
    error = prediction - truth
    norms = np.linalg.norm(truth, axis=-1)
    pnorms = np.linalg.norm(prediction, axis=-1)
    valid = (norms > 1e-14) & (pnorms > 1e-14)
    cosine = np.full(norms.shape, np.nan)
    cosine[valid] = np.sum(truth[valid] * prediction[valid], axis=-1) / (norms[valid] * pnorms[valid])
    angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    rms = float(np.sqrt(np.mean(error ** 2)))
    magnitude = float(np.sqrt(np.mean(truth ** 2)))
    return dict(rmse=rms, empirical_displacement_coordinate_rms=magnitude,
                relative_displacement_error=rms / magnitude if magnitude > 0 else None,
                median_angle_degrees=float(np.nanmedian(angle)) if valid.any() else None,
                p90_angle_degrees=float(np.nanquantile(angle, .9)) if valid.any() else None,
                opposite_fraction=float(np.mean(cosine[valid] < 0)) if valid.any() else None,
                valid_direction_count=int(valid.sum()), vector_count=int(norms.size)), angle


def _radius95(samples, mean):
    return np.quantile(np.linalg.norm(samples - mean, axis=-1), .95, axis=0)


def paired_neighborhoods(cell_g, boot_g, predictions, boot_predictions,
                         cell_ids, cell_splits, scale, *, selection_g=None):
    """Describe close-input/different-successor pairs; do not claim nonclosure.

    Proximity is selected from input coordinates alone. A steep continuous map
    and finite input measurement noise can both generate such pairs. Comparing
    observed and predicted successor differences shows model mismatch, not a
    proof that no autonomous map exists.
    """
    x = cell_g[:, :2] / scale
    selection_x = x if selection_g is None else np.asarray(selection_g)[:, :2] / scale
    bx = boot_g[:, :, :2] / scale
    y = cell_g[:, 1:] / scale
    by = boot_g[:, :, 1:] / scale
    p, bp = predictions / scale, boot_predictions / scale
    nodes = [(i, k) for i in range(len(cell_ids)) for k in (0, 1)]
    pairs = set()
    for i, k in nodes:
        for category in ("same_scale", "cross_scale"):
            candidates = [(j, l) for j, l in nodes
                          if (l == k) == (category == "same_scale")
                          and not (j == i and l == k)]
            distances = [np.linalg.norm(selection_x[j, l] - selection_x[i, k]) for j, l in candidates]
            for index in np.argsort(distances)[:2]:
                a, b = sorted(((i, k), candidates[index]))
                pairs.add((a, b))
    rows = []
    for (i, k), (j, l) in sorted(pairs):
        dx, dy, dp = x[j, l] - x[i, k], y[j, l] - y[i, k], p[j, l] - p[i, k]
        bdx, bdy = bx[:, j, l] - bx[:, i, k], by[:, j, l] - by[:, i, k]
        bdp = bp[:, j, l] - bp[:, i, k]
        input_radius = float(_radius95(bdx, dx))
        successor_radius = float(_radius95(bdy, dy))
        mismatch_radius = float(_radius95(bdy - bdp, dy - dp))
        in_norm, out_norm = float(np.linalg.norm(dx)), float(np.linalg.norm(dy))
        mismatch = float(np.linalg.norm(dy - dp))
        rows.append(dict(cell_a=int(cell_ids[i]), scale_a=k, split_a=cell_splits[i],
            cell_b=int(cell_ids[j]), scale_b=l, split_b=cell_splits[j],
            category="same_scale" if k == l else "cross_scale",
            same_physical_condition=bool(i == j),
            neighbor_selection_input_distance=float(np.linalg.norm(selection_x[j, l] - selection_x[i, k])),
            normalized_input_distance=in_norm,
            normalized_successor_distance=out_norm,
            predicted_successor_difference_norm=float(np.linalg.norm(dp)),
            successor_contrast_error_norm=mismatch,
            input_bootstrap_radius95=input_radius,
            successor_bootstrap_radius95=successor_radius,
            contrast_error_bootstrap_radius95=mismatch_radius,
            input_unresolved_descriptive=in_norm <= input_radius,
            successor_resolved_descriptive=out_norm > successor_radius,
            contrast_error_resolved_descriptive=mismatch > mismatch_radius,
            required_gain_lower_descriptive=max(0., out_norm - successor_radius) /
                (in_norm + input_radius) if in_norm + input_radius > 0 else None))
    return rows


def draw_flow(output, direct, cell_g, predictions, *, model, plt):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.5), sharex=True, sharey=True)
    empirical = cell_g[:, 1:] - cell_g[:, :2]
    predicted = predictions - cell_g[:, :2]
    extent = np.concatenate((direct.reshape(-1, 2), predictions.reshape(-1, 2)))
    lower, upper = extent.min(0), extent.max(0)
    padding = .07 * np.maximum(upper - lower, 1e-8)
    for k in (0, 1):
        for j, (title, delta) in enumerate((("配对数据位移", empirical[:, k]),
                                          ("R 的预测位移", predicted[:, k]))):
            ax = axes[k, j]
            for scale_index, color, label in ((k, "#5594bd", "输入重复编码"),
                                               (k + 1, "#dfaa60", "目标重复编码")):
                points = direct[..., scale_index, :].reshape(-1, 2)
                ax.scatter(*points.T, s=8, color=color, alpha=.20, linewidths=0, label=label)
            start = cell_g[:, k]
            ax.quiver(*start.T, *delta.T, angles="xy", scale_units="xy", scale=1,
                      color="#253f4f", alpha=.85, width=.0036,
                      headwidth=3.5, headlength=4.5, pivot="tail")
            ax.set(title=f"{PATH_NAMES[k]}：{title}", xlabel="g₁", ylabel="g₂",
                   xlim=(lower[0] - padding[0], upper[0] + padding[0]),
                   ylim=(lower[1] - padding[1], upper[1] + padding[1]))
            ax.set_aspect("equal", adjustable="box")
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(fontsize=9, frameon=False)
    name = "归一化256，种子93" if model == SELECTED else "归一化96，种子93"
    fig.suptitle(f"真实粗粒化与预测流场：{name}", fontsize=14)
    fig.text(.5, .025, "每个箭头为一个参数条件的链与编码重复均值；箭长为原始位移。\n"
             "预测端点为 mean R(g)，并非 R(mean g)；散点为保存的实际编码。",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .075, 1, .95))
    fig.savefig(output / f"empirical_vs_{model}.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def draw_alignment(output, model_rows, reliability, plt):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    colors = {"source_seed0": "#8a8a8a", COMPARISON: "#4c9a80", SELECTED: "#ad709d"}
    names = {"source_seed0": "原联合训练 R", COMPARISON: "归一化96，种子93", SELECTED: "归一化256，种子93"}
    for k, ax in enumerate(axes):
        reliable = reliability[:, k]
        for key in ("source_seed0", COMPARISON, SELECTED):
            values = np.asarray(model_rows[key][k]["cell_angles"])
            ax.hist(values[reliable], bins=np.arange(0, 181, 15), histtype="step",
                    linewidth=1.7, color=colors[key], label=names[key])
        ax.set(title=f"{PATH_NAMES[k]}：方向夹角", xlabel="与配对数据位移的夹角（度）",
               ylabel="参数条件数", xlim=(0, 180))
        ax.text(.98, .97, f"可分辨位移 {int(reliable.sum())}/45", ha="right", va="top",
                transform=ax.transAxes, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False)
    fig.text(.5, .015, "仅纳入位移模长超过配对链 bootstrap 描述性95%误差半径的条件；非联合显著性检验。",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .045, 1, 1))
    fig.savefig(output / "direction_alignment.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def run(root, cache, output):
    started = time.process_time()
    root, cache, output = map(lambda p: Path(p).resolve(), (root, cache, output))
    output.mkdir(parents=True, exist_ok=False)
    sources = [root / x for x in ("evaluation_arrays.npz", "summary.json", "manifest.json", "public_files.json")]
    sources += [cache / x for x in ("cache.npz", "public_files.json")]
    hashes = {str(p): sha256(p) for p in sources}
    for folder in (root, cache):
        inventory = json.loads((folder / "public_files.json").read_text())
        for path in sources:
            if path.parent == folder and path.name in inventory and sha256(path) != inventory[path.name]:
                raise ValueError(f"Public fingerprint mismatch: {path}")
    with np.load(root / "evaluation_arrays.npz", allow_pickle=False) as data:
        a = {key: data[key].copy() for key in data.files}
    with np.load(cache / "cache.npz", allow_pickle=False) as data:
        fit = data["fit_direct"].copy()
        for split in ("train", "test"):
            if not np.array_equal(a[f"{split}_direct"], data[f"eval_{split}_direct"]):
                raise ValueError("Frozen cache and endpoint evaluation encodings differ")
    if set(a["train_cell_ids"]) & set(a["test_cell_ids"]):
        raise ValueError("Whole-condition split violated")
    direct = np.concatenate((a["train_direct"], a["test_direct"]), axis=1)
    ids = np.concatenate((a["train_cell_ids"], a["test_cell_ids"]))
    labels = np.concatenate((a["train_labels"], a["test_labels"]))
    chains, grouped, means, cells = chain_and_cell_means(direct, ids)
    boot, bootstrap_indices = bootstrap_cell_means(grouped)
    splits = ["train" if cell in a["train_cell_ids"] else "test" for cell in cells]
    cell_labels = np.stack([labels[ids == cell][0] for cell in cells])
    delta = means[:, 1:] - means[:, :2]
    bdelta = boot[:, :, 1:] - boot[:, :, :2]
    radii = _radius95(bdelta, delta)
    reliable = np.linalg.norm(delta, axis=-1) > radii
    scale = fit[..., :2, :].reshape(-1, 2).std(0, ddof=0)
    if np.any(scale <= 0):
        raise ValueError("Degenerate TRAIN input scale")
    keys = ["source_seed0"] + [f"{arm}_seed{seed}" for seed in (91, 92, 93)
                                for arm in ("raw96", "scaled96", "scaled256")]
    predictions, boot_predictions, all_rows, repeat_bias = {}, {}, {}, {}
    payload = dict(cell_ids=cells, cell_labels=cell_labels, cell_direct=means,
                   chain_direct=chains, empirical_displacement=delta,
                   empirical_radius95=radii, displacement_resolved_descriptive=reliable,
                   bootstrap_cell_direct=boot, train_input_scale=scale)
    split_chain = np.array([cell in a["train_cell_ids"] for cell in ids])
    split_cell = np.array([split == "train" for split in splits])
    model_metrics = {}
    for key in keys:
        repeated = np.concatenate([a[f"eval_{key}_{split}_one_step"]
                                   for split in ("train", "test")], axis=1)
        pchains, pgrouped, pmean, _ = chain_and_cell_means(repeated, ids)
        bp = pgrouped[np.arange(len(cells))[None, :, None], bootstrap_indices].mean(axis=2)
        predictions[key], boot_predictions[key] = pmean, bp
        residual = repeated - direct[:, :, 1:]
        repeat_variance = residual.var(axis=0, ddof=1).mean(axis=(0, 2))
        mean_residual_mse = (residual.mean(axis=0) ** 2).mean(axis=(0, 2))
        repeat_bias[key] = dict(total_coordinate_mse=(residual ** 2).mean(axis=(0, 1, 3)).tolist(),
            repeat_variance_coordinate_mean=repeat_variance.tolist(),
            chain_mean_residual_mse=mean_residual_mse.tolist(),
            conditional_bias_mse_mc_corrected=(mean_residual_mse - repeat_variance / direct.shape[0]).tolist(),
            qualification="Conditional on existing saved physical configurations and fixed Q/C; four repeats only. Corrected estimates may be negative and are retained. This is not irreducible error or equilibrium error.")
        rows, per_scale = [], []
        for k in (0, 1):
            _, cell_angles = vector_metrics(delta[:, k], pmean[:, k] - means[:, k])
            per_scale.append(dict(cell_angles=cell_angles.tolist()))
            for split, chain_mask, cell_mask in (("train", split_chain, split_cell),
                                                 ("test", ~split_chain, ~split_cell)):
                truth = chains[chain_mask, k + 1] - chains[chain_mask, k]
                prediction = pchains[chain_mask, k] - chains[chain_mask, k]
                m, _ = vector_metrics(truth, prediction)
                cm, _ = vector_metrics(delta[cell_mask, k], pmean[cell_mask, k] - means[cell_mask, k])
                mask = cell_mask & reliable[:, k]
                rm, _ = vector_metrics(delta[mask, k], pmean[mask, k] - means[mask, k]) if mask.any() else ({}, None)
                rows.append(dict(split=split, path=PATH_NAMES[k], physical_chains=int(chain_mask.sum()),
                                 conditions=int(cell_mask.sum()), chain_mean_metrics=m,
                                 condition_mean_metrics=cm, resolved_condition_metrics=rm))
        all_rows[key], model_metrics[key] = per_scale, rows
        payload[f"{key}_cell_prediction"] = pmean
    _, _, selection_means, selection_cells = chain_and_cell_means(fit, a["train_cell_ids"])
    if not np.array_equal(selection_cells, cells[split_cell]):
        raise ValueError("FIT and EVAL TRAIN condition pairing differs")
    neighborhoods = paired_neighborhoods(means[split_cell], boot[:, split_cell],
        predictions[SELECTED][split_cell], boot_predictions[SELECTED][:, split_cell],
        cells[split_cell], ["train"] * int(split_cell.sum()), scale,
        selection_g=selection_means)
    pair_summary = {}
    for category in ("same_scale", "cross_scale"):
        rows = [r for r in neighborhoods if r["category"] == category]
        close = [r for r in rows if r["input_unresolved_descriptive"]]
        collisions = [r for r in close if r["successor_resolved_descriptive"]]
        pair_summary[category] = dict(selected_pairs=len(rows),
            unresolved_input_pairs=len(close), resolved_successor_among_unresolved_inputs=len(collisions),
            resolved_contrast_error_among_these=sum(r["contrast_error_resolved_descriptive"] for r in collisions),
            maximum_required_gain_lower=max(r["required_gain_lower_descriptive"] or 0 for r in rows))
    plt = _plotting(None)
    for key in (SELECTED, COMPARISON):
        draw_flow(output, direct, means, predictions[key], model=key, plt=plt)
    draw_alignment(output, all_rows, reliable, plt)
    np.savez_compressed(output / "diagnostic_arrays.npz", **payload)
    write_json(output / "neighbor_pairs.json", dict(rows=neighborhoods,
        selection="two nearest distinct TRAIN condition/scale nodes chosen from FIT input coordinates; confirmed on independent EVAL subset/C repeats of the same physical chains",
        statistical_qualification="Descriptive bootstrap contrasts, no multiple-comparison correction. Unresolved input and resolved successor do not prove nonclosure: steep maps and input measurement noise remain possible."))
    summary = dict(status="complete", workflow_complete=True, returncode=0,
        method="paired saved measurements; mean C/subset repeats inside physical chain; paired physical-chain bootstrap within each parameter condition",
        physical_conditions=len(cells), physical_chains=len(chains), C_subset_repeats=direct.shape[0],
        bootstrap_replicates=len(boot), physical_chains_per_condition=grouped.shape[1],
        bootstrap_keeps_scales_and_predictions_paired=True,
        selected_model=SELECTED, selected_before_this_analysis=True,
        resolved_empirical_displacements_by_scale=reliable.sum(0).tolist(),
        model_metrics=model_metrics, repeat_error_decomposition=repeat_bias,
        neighborhood_summary=pair_summary,
        labels_used_for_training=False, new_training=False, model_inference=False,
        physical_sampling=False, root_search=False, gpu_used=False,
        inputs_sha256=hashes, script_sha256=sha256(__file__), cpu_seconds=time.process_time()-started,
        limitations=["All results condition on one trained Q/C, one size chain and existing sampling warnings.",
                     "Four physical chains per condition give limited uncertainty resolution; bootstrap is descriptive and no multiplicity correction is applied.",
                     "Subset/C repeats share the same saved physical configurations and are not independent physical samples.",
                     "Empirical g trajectories are the frozen Q/C coarse-graining outputs, not a certified exact physical RG.",
                     "Close inputs with different successors can arise from a steep map or measurement noise; they alone do not prove missing state or nonclosure.",
                     "Condition-level errors and mean R(g) are distinct from R(mean g). Existing TEST is exploratory, already inspected.",
                     "A boundary or fixed point is not inferred from a coarse grid of empirical arrows."])
    if any(sha256(path) != expected for path, expected in hashes.items()):
        raise RuntimeError("Immutable input changed")
    write_json(output / "summary.json", summary)
    write_json(output / "public_files.json", {p.name: sha256(p) for p in output.iterdir() if p.is_file()})
    print(json.dumps(dict(output=str(output), resolved=reliable.sum(0).tolist(),
                          neighborhood_summary=pair_summary), ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.run, args.cache, args.output)


if __name__ == "__main__":
    main()
