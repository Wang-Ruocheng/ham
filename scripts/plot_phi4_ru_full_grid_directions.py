"""Frozen R forward on a uniform plotting grid; no training or root search."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial import ConvexHull, QhullError
import torch

from recurrent_rg.phi4_r_calibration import FrozenResidualMap
from scripts import plot_phi4_ru_cached_directions as cached
from scripts.report_phi4_qcr import _plotting


def fingerprint(model):
    import hashlib
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        for token in (key, str(array.dtype), str(array.shape)):
            digest.update(token.encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def restore_model(parameters, key, expected):
    with np.load(parameters, allow_pickle=False) as z:
        state = {name[len(key) + 1:]: torch.from_numpy(z[name].copy())
                 for name in z.files if name.startswith(key + ".")}
    required = {"center", "scale", "net.0.weight", "net.0.bias", "net.2.weight",
                "net.2.bias", "net.4.weight", "net.4.bias"}
    if set(state) != required:
        raise ValueError("冻结R参数不完整")
    with torch.random.fork_rng(devices=[]):
        model = FrozenResidualMap(2, int(state["net.0.weight"].shape[0]),
                                  center=state["center"], scale=state["scale"])
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    if fingerprint(model) != expected:
        raise ValueError("冻结R参数指纹不一致")
    return model


@torch.no_grad()
def mapped(model, points):
    points = np.asarray(points, dtype=np.float64)
    values = [model.advance(torch.from_numpy(points[start:start + 512].copy())).numpy()
              for start in range(0, len(points), 512)]
    result = np.concatenate(values)
    if result.shape != points.shape or not np.isfinite(result).all():
        raise ValueError("网格前向结果非有限或形状不一致")
    return result


def make_grid(bounds, nx=19, ny=29):
    bounds = np.asarray(bounds, dtype=float)
    if (bounds.shape != (2, 2) or not np.isfinite(bounds).all()
            or np.any(bounds[:, 1] <= bounds[:, 0]) or nx < 2 or ny < 2):
        raise ValueError("无效的绘图范围或网格尺寸")
    dx, dy = np.diff(bounds, axis=1).ravel() / [nx, ny]
    x = bounds[0, 0] + (np.arange(nx) + .5) * dx
    y = bounds[1, 0] + (np.arange(ny) + .5) * dy
    return np.stack(np.meshgrid(x, y), axis=-1).reshape(-1, 2), float(.72 * min(dx, dy))


def coverage_mask(points, clouds):
    mask = np.zeros(len(points), dtype=bool)
    for cloud in clouds:
        try:
            hull = ConvexHull(cloud)
        except QhullError:
            continue
        mask |= np.all(points @ hull.equations[:, :2].T + hull.equations[:, 2] <= 1e-12, axis=1)
    return mask


def run(root, validation, previous_plot, output):
    started = time.process_time()
    root, validation, previous_plot, output = map(lambda p: Path(p).resolve(),
                                                  (root, validation, previous_plot, output))
    if output.exists():
        raise FileExistsError("使用新目录，不覆盖原报告或先前图像")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    direct, labels, predictions, groups, summary, pins = cached.read_cached(root)
    clouds, _, _, _ = cached.geometry(direct, labels, predictions, groups)
    old = json.loads(previous_plot.read_text())
    if old.get("status") != "complete" or old.get("returncode") != 0:
        raise ValueError("需要上一幅已完成方向图的坐标范围")
    for path, expected in old["input_sha256"].items():
        if cached.sha(path) != expected:
            raise ValueError("上一幅图的输入或实现已变化")
    inventory = json.loads((validation / "public_files.json").read_text())
    for name in ("summary.json", "R_parameters.npz"):
        if cached.sha(validation / name) != inventory.get(name):
            raise ValueError("R验证参数的公开指纹不一致")
        pins[str(validation / name)] = inventory[name]
    expected_maps = json.loads((validation / "summary.json").read_text())["final_R_sha256"]
    for seed in cached.SEEDS:
        key = f"selected_seed{seed}"
        if (expected_maps[key] != summary["fixed_R_sha256"][key]
                or old["models"][str(seed)]["fixed_R_sha256"] != expected_maps[key]):
            raise ValueError("绘图模型与正式评价端点不一致")
    for path in (previous_plot, validation / "public_files.json", Path(__file__).resolve(),
                 Path(cached.__file__).resolve()):
        pins[str(path)] = cached.sha(path)
    bounds = np.asarray(old["bounds"], dtype=float)
    grid, length = make_grid(bounds)
    covered = coverage_mask(grid, clouds)
    output.mkdir(parents=True, exist_ok=False)
    plt = _plotting(None)
    from matplotlib.patches import Patch, Polygon
    from matplotlib.lines import Line2D
    receipts = {}
    for seed in cached.SEEDS:
        key = f"selected_seed{seed}"
        model = restore_model(validation / "R_parameters.npz", key, expected_maps[key])
        # Reproduce saved forward values to verify the local implementation,
        # including internal center/scale, before evaluating any off-data point.
        check_points = direct[:, :, :2].reshape(-1, 2)
        check_target = predictions[seed].reshape(-1, 2)
        check = mapped(model, check_points)
        if not np.allclose(check, check_target, rtol=1e-10, atol=1e-12):
            raise RuntimeError("本地冻结R前向与已保存预测不一致")
        raw = mapped(model, grid) - grid
        delta, keep = cached.directions(raw, length)
        fig, ax = plt.subplots(figsize=(7.3, 8.5))
        for cloud in clouds:
            try:
                hull = ConvexHull(cloud)
                ax.add_patch(Polygon(cloud[hull.vertices], closed=True, facecolor="#a4abb2",
                                     edgecolor="none", alpha=.32, zorder=1))
            except QhullError:
                pass
            ax.scatter(cloud[:, 0], cloud[:, 1], s=5, color="#8e959d", alpha=.30,
                       edgecolors="none", zorder=2)
        for mask, opacity in ((keep & ~covered, .58), (keep & covered, .95)):
            ax.quiver(grid[mask, 0], grid[mask, 1], delta[mask, 0], delta[mask, 1],
                      angles="xy", scale_units="xy", scale=1, pivot="mid", color="#26323d",
                      alpha=opacity, width=.0021, headwidth=4.1, headlength=5., zorder=3)
        ax.set(xlim=bounds[0], ylim=bounds[1], xlabel=r"潜变量 $g_1$", ylabel=r"潜变量 $g_2$")
        ax.set_aspect("equal", adjustable="box")
        ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(f"全区域重整化流向：种子{seed}", y=.98)
        fig.legend(handles=[Patch(facecolor="#a4abb2", alpha=.5, label="评价数据覆盖"),
                            Line2D([], [], color="#26323d", marker=">", markersize=6,
                                   label="冻结模型流向")], loc="upper center", ncol=2,
                   bbox_to_anchor=(.53, .945), frameon=False)
        fig.text(.5, .060, r"箭头：网格处 $R(g)-g$ 的方向；等长，不表示位移大小。",
                 ha="center", fontsize=10)
        fig.text(.5, .032, "整个画框均计算冻结模型；灰色覆盖外的箭头属于模型外推。",
                 ha="center", fontsize=9)
        fig.text(.5, .007, "灰色仅为样本点云及局部包络，不是置信域或已认证支持域。",
                 ha="center", fontsize=9, color="#666666")
        fig.tight_layout(rect=(0, .085, 1, .92))
        fig.savefig(output / f"full_grid_flow_seed{seed}.png", dpi=200,
                    bbox_inches="tight", facecolor="white")
        plt.close(fig)
        if fingerprint(model) != expected_maps[key] or any(p.grad is not None for p in model.parameters()):
            raise RuntimeError("绘图期间模型参数或梯度状态变化")
        receipts[str(seed)] = dict(model_key=key, fixed_R_sha256=expected_maps[key],
            cached_prediction_max_abs_difference=float(np.max(np.abs(check - check_target))),
            grid_raw_delta=raw.tolist(), shown_arrows=int(keep.sum()),
            skipped_near_zero=int((~keep).sum()), model_unchanged=True, gradients_none=True)
    if any(cached.sha(path) != value for path, value in pins.items()):
        raise RuntimeError("绘图期间不可变输入变化")
    receipt = dict(schema_version=5, status="complete", workflow_complete=True, returncode=0,
        bounds=bounds.tolist(), grid_shape=[29, 19], grid_points=grid.tolist(),
        grid_points_inside_local_sample_envelopes=int(covered.sum()),
        grid_points_outside_local_sample_envelopes=int((~covered).sum()),
        arrow_length_data_units=length, pointwise_direction_normalization=True,
        model_forward_performed=True, training_updates=0, new_physical_samples=0,
        Q_or_C_loaded=False, root_search=False, coverage_outside_is_model_extrapolation=True,
        input_sha256=pins, models=receipts, cpu_seconds=time.process_time()-started,
        output_sha256={p.name: cached.sha(p) for p in output.glob("*.png")})
    (output / "plot_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--previous-plot", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    receipt = run(args.run, args.validation, args.previous_plot, args.output)
    print(json.dumps(dict(status=receipt["status"], output=str(args.output)), ensure_ascii=False))


if __name__ == "__main__":
    main()
