"""Minimal directions and observed coverage, using public cached arrays only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from scripts.report_phi4_qcr import _plotting


SEEDS = (91, 92, 93)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_cached(root):
    root = Path(root).resolve()
    inventory = json.loads((root / "public_files.json").read_text())
    pins = {str(root / "public_files.json"): sha(root / "public_files.json")}
    for name in ("status.json", "summary.json", "groups.json", "evaluation_arrays.npz"):
        path = root / name
        if sha(path) != inventory.get(name):
            raise ValueError(f"公共数组指纹不一致：{name}")
        pins[str(path)] = inventory[name]
    status = json.loads((root / "status.json").read_text())
    if (status.get("status") != "complete" or status.get("workflow_complete") is not True
            or status.get("returncode") != 0 or status.get("error") is not None
            or status.get("summary_sha256") != pins[str(root / "summary.json")]):
        raise ValueError("需要已完成、校验一致的公开报告")
    summary = json.loads((root / "summary.json").read_text())
    groups = json.loads((root / "groups.json").read_text())
    with np.load(root / "evaluation_arrays.npz", allow_pickle=False) as z:
        direct, labels = z["direct"].copy(), z["labels"].copy()
        predictions = {seed: z[f"selected_seed{seed}_one_step"].copy() for seed in SEEDS}
    if direct.shape != (4, 156, 3, 2) or labels.shape != (156, 3):
        raise ValueError("需要固定的156条物理链、四次编码、三个尺度")
    if {int(size): int(np.sum(labels[:, 0] == size)) for size in np.unique(labels[:, 0])} != {36: 120, 108: 36}:
        raise ValueError("物理链尺寸计数不一致")
    values = [direct, labels, *predictions.values()]
    if any(not np.isfinite(value).all() for value in values):
        raise ValueError("缓存包含非有限值")
    if any(value.shape != (4, 156, 2, 2) for value in predictions.values()):
        raise ValueError("缓存预测形状不一致")
    return direct, labels, predictions, groups, summary, pins


def directions(raw_delta, length):
    raw_delta = np.asarray(raw_delta, dtype=float)
    norms = np.linalg.norm(raw_delta, axis=1)
    keep = norms > 1e-12
    result = np.zeros_like(raw_delta)
    result[keep] = raw_delta[keep] / norms[keep, None] * length
    return result, keep


def geometry(direct, labels, predictions, groups):
    clouds, origins, deltas, identities = [], [], {seed: [] for seed in SEEDS}, []
    for group in groups:
        mask = np.all(labels == [group["size"], group["r"], group["u"]], axis=1)
        if mask.sum() != group["physical_chains"]:
            raise ValueError("条件分组与缓存不一致")
        for k in (0, 1):
            cloud = direct[:, mask, k].reshape(-1, 2)
            origin = cloud.mean(0)
            if not np.allclose(origin, group["g_mean"][k], rtol=0, atol=1e-13):
                raise ValueError("已保存条件均值不一致")
            clouds.append(cloud)
            origins.append(origin)
            identities.append(dict(size=group["size"], r=group["r"], u=group["u"], input_scale=k))
            for seed in SEEDS:
                delta = (predictions[seed][:, mask, k] - direct[:, mask, k]).mean((0, 1))
                expected = group["models"][f"selected_seed{seed}"]["mean_R_minus_g"][k]
                if not np.allclose(delta, expected, rtol=0, atol=1e-13):
                    raise ValueError("缓存平均位移与分组回执不一致")
                deltas[seed].append(delta)
    if len(clouds) != 48:
        raise ValueError("必须保留24个尺寸/参数条件的两个输入尺度")
    return clouds, np.asarray(origins), {key: np.asarray(value) for key, value in deltas.items()}, identities


def run(root, output):
    started = time.process_time()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("使用新目录，不覆盖原报告或既有图")
    direct, labels, predictions, groups, summary, pins = read_cached(root)
    pins[str(Path(__file__).resolve())] = sha(__file__)
    clouds, origins, raw, identities = geometry(direct, labels, predictions, groups)
    all_points = np.concatenate(clouds)
    arrow_length = float(.055 * np.ptp(all_points, axis=0).min())
    if arrow_length <= 0:
        raise ValueError("数据覆盖退化，无法生成方向图")
    scaled = {seed: directions(raw[seed], arrow_length) for seed in SEEDS}
    shown = np.concatenate([all_points, origins, *[origins + value[0] for value in scaled.values()]])
    limits = np.column_stack((shown.min(0), shown.max(0)))
    padding = .10 * np.ptp(shown, axis=0)
    limits += np.column_stack((-padding, padding))
    output.mkdir(parents=True, exist_ok=False)
    plt = _plotting(None)
    from matplotlib.patches import Patch, Polygon
    from matplotlib.lines import Line2D
    receipts = {}
    for seed in SEEDS:
        fig, ax = plt.subplots(figsize=(7.3, 8.5))
        for cloud in clouds:
            try:
                hull = ConvexHull(cloud)
                ax.add_patch(Polygon(cloud[hull.vertices], closed=True,
                    facecolor="#a9afb5", edgecolor="none", alpha=.24, zorder=1))
            except QhullError:
                pass  # A collinear cloud remains visible as actual points.
            ax.scatter(cloud[:, 0], cloud[:, 1], s=5, color="#8e959d", alpha=.28,
                       edgecolors="none", zorder=2)
        delta, keep = scaled[seed]
        ax.quiver(origins[keep, 0], origins[keep, 1], delta[keep, 0], delta[keep, 1],
                  angles="xy", scale_units="xy", scale=1, pivot="mid", color="#202c36",
                  width=.0032, headwidth=4.1, headlength=5.0, headaxislength=4.5, zorder=3)
        ax.set(xlim=limits[0], ylim=limits[1], xlabel=r"潜变量 $g_1$", ylabel=r"潜变量 $g_2$",
               title=f"重整化流向与数据覆盖：种子{seed}")
        ax.set_aspect("equal", adjustable="box")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(handles=[Patch(facecolor="#a9afb5", alpha=.4, label="评价数据覆盖"),
                           Line2D([], [], color="#202c36", marker=">", markersize=7,
                                  label="条件平均流向")], loc="upper right", frameon=False)
        fig.text(.5, .060, r"箭头：平均 $R(g)-g$ 的方向；等长，不表示位移大小。",
                 ha="center", fontsize=10)
        fig.text(.5, .032, r"灰色含 $L=36/108$、输入尺度 $k=0/1$ 的固定重复编码；空白区未插值。",
                 ha="center", fontsize=9)
        fig.text(.5, .007, "局部包络仅描述样本范围，不是置信域或已认证支持域。",
                 ha="center", fontsize=9, color="#666666")
        fig.tight_layout(rect=(0, .085, 1, 1))
        fig.savefig(output / f"direction_flow_seed{seed}.png", dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        receipts[str(seed)] = dict(model_key=f"selected_seed{seed}",
            fixed_R_sha256=summary["fixed_R_sha256"][f"selected_seed{seed}"],
            origins=origins.tolist(), raw_mean_delta=raw[seed].tolist(),
            plotted_delta=delta.tolist(), shown_arrows=int(keep.sum()),
            skipped_near_zero=int((~keep).sum()))
    if any(sha(path) != value for path, value in pins.items()):
        raise RuntimeError("绘图期间不可变输入发生变化")
    receipt = dict(schema_version=5, status="complete", workflow_complete=True, returncode=0,
        plots="仅方向箭头和观测点云/局部凸包；无实际位移对照、根、轨迹或热图",
        arrow_averaging="mean over encoding repeats and physical chains of R(g)-g; not R(mean g)-mean g",
        arrow_length_data_units=arrow_length, pointwise_direction_normalization=True,
        bounds=limits.tolist(), coverage_clouds=len(clouds), cloud_identities=identities,
        model_forward_calls=0, new_physical_samples=0, training_updates=0, root_search=False,
        input_sha256=pins, models=receipts, cpu_seconds=time.process_time()-started,
        output_sha256={path.name: sha(path) for path in output.glob("*.png")})
    (output / "plot_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    receipt = run(args.run, args.output)
    print(json.dumps(dict(status=receipt["status"], output=str(args.output)), ensure_ascii=False))


if __name__ == "__main__":
    main()
