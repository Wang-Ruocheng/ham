"""从已校验公共数组绘制 KB 前两主成分相图；不载入模型、不推理、不求根。"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


SOURCE_NAMES = ("metrics.json", "encodings.npz", "grid_encodings.npz", "fixed_points.json")
LIMITATIONS = [
    "PCA 沿用正式报告中仅由 TRAIN 的 k=0、1 输入尺度拟合的中心和基底；未重新选轴。",
    "两组模型各有独立坐标系；横纵轴符号、方向和坐标值不能跨模型直接比较。",
    "箭头为完整 R(平均 g)-平均 g 的正交投影，不是平均 R(g)-平均 g；两者单独保存。",
    "只在已有 KB 参数格点画箭头，未对主成分平面任意位置推理、插值或填充流场。",
    "只画训练所覆盖的两种输入尺度 k=0、1；k=2 仅作为直接粗粒化轨迹终点。",
    "投影交叉、近零箭头或高方差解释率均不认证二维自治流、物理相边界或不动点。",
    "K、B 仅用于事后着色和连接参数格点；不作为 PCA 或模型的训练目标。",
    "未获物理支持的数值根不标成相变点，完整已存根及其投影均保留在 JSON。",
]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class ReportArrays:
    dimension: int
    metrics: dict
    grid: dict
    encoded: dict
    roots: dict
    source_hashes: dict


def _require_close(actual, expected, message, *, atol=2e-12):
    if np.shape(actual) != np.shape(expected) or not np.allclose(actual, expected, atol=atol, rtol=2e-11):
        raise ValueError(message)


def project_pc12(points, displacement, center, components):
    """Center points, never vectors; retain original axis order and signs."""
    points, displacement = np.asarray(points), np.asarray(displacement)
    center, components = np.asarray(center), np.asarray(components)
    dimension = center.size
    if (center.shape != (dimension,) or components.shape != (dimension, dimension)
            or points.shape != displacement.shape or points.shape[-1] != dimension
            or dimension < 2):
        raise ValueError("Invalid full-coordinate projection geometry")
    if not all(np.isfinite(value).all() for value in (points, displacement, center, components)):
        raise ValueError("Nonfinite projection inputs")
    _require_close(components @ components.T, np.eye(dimension), "PCA basis is not orthonormal")
    basis = components[:2]
    return (points - center) @ basis.T, displacement @ basis.T


def drift_diagnostics(displacement, components):
    """RMS norm and off-plane *energy*, not a physical irrelevant-mode test."""
    displacement = np.asarray(displacement, dtype=np.float64)
    basis = np.asarray(components, dtype=np.float64)[:2]
    projected = displacement @ basis.T
    normal = displacement - projected @ basis
    full_energy = float(np.mean(np.sum(displacement ** 2, axis=-1)))
    normal_energy = float(np.mean(np.sum(normal ** 2, axis=-1)))
    return dict(full_displacement_rms_norm=float(np.sqrt(full_energy)),
        pc12_displacement_rms_norm=float(np.sqrt(np.mean(np.sum(projected ** 2, axis=-1)))),
        outside_pc12_displacement_rms_norm=float(np.sqrt(normal_energy)),
        outside_pc12_energy_fraction=normal_energy / full_energy if full_energy > 0 else None,
        outside_pc12_rms_fraction=float(np.sqrt(normal_energy / full_energy)) if full_energy > 0 else None,
        zero_flow=full_energy == 0,
        definition="在已采样参数格点平均位移范数平方；平面外能量 / 完整位移能量")


def load_report(source, dimension):
    """Verify download receipt, all source SHA, PCA and grid semantics before plotting."""
    source = Path(source).resolve()
    receipt = json.loads((source / "download_receipt.json").read_text())
    if receipt.get("status") != "verified" or receipt.get("all_sha256_verified") is not True:
        raise ValueError("Require the verified public-file download receipt")
    paths, hashes = {}, {}
    for name in SOURCE_NAMES:
        relative = f"dim{dimension}/report/{name}"
        path = source / relative
        if not path.resolve().is_relative_to(source):
            raise ValueError("Public report escapes source directory")
        actual = sha256(path)
        if receipt.get("files", {}).get(relative) != actual:
            raise ValueError(f"Public report fingerprint mismatch: {relative}")
        paths[name], hashes[relative] = path, actual
    metrics = json.loads(paths["metrics.json"].read_text())
    roots = json.loads(paths["fixed_points.json"].read_text())
    with np.load(paths["grid_encodings.npz"], allow_pickle=False) as values:
        grid = {name: values[name].copy() for name in values.files}
    with np.load(paths["encodings.npz"], allow_pickle=False) as values:
        encoded = {name: values[name].copy() for name in values.files}
    if (metrics.get("theory_dim") != dimension or metrics.get("checkpoint_epoch") != 12000
            or metrics.get("labels_used_in_training") is not False
            or metrics.get("coupling_names") != ["K", "B"]
            or metrics.get("lattice_sizes") != [36, 12, 4]):
        raise ValueError("Require the completed 12000-update b3 KB report")
    if not all(np.issubdtype(value.dtype, np.number) and np.isfinite(value).all()
               for value in (*grid.values(), *encoded.values())):
        raise ValueError("Public arrays contain nonnumeric or nonfinite values")
    K, B, means = grid["K"], grid["B"], grid["mean_by_scale"]
    if (K.ndim != 1 or B.ndim != 1 or min(len(K), len(B)) < 2
            or np.any(np.diff(K) <= 0) or np.any(np.diff(B) <= 0)
            or means.shape != (len(K), len(B), 3, dimension)):
        raise ValueError("Invalid complete KB grid geometry")
    if np.count_nonzero(np.isclose(B, 0., atol=1e-14, rtol=0)) != 1:
        raise ValueError("Require one explicit zero-field column")
    if not B[0] < 0 < B[-1]:
        raise ValueError("Require both field signs")
    center, components = grid["pca_center"], grid["pca_components"]
    _require_close(center, metrics["pca"]["center"], "PCA center differs between JSON and NPZ")
    _require_close(components, metrics["pca"]["components"], "PCA basis differs between JSON and NPZ")
    train = encoded["train_direct"]
    if train.ndim != 3 or train.shape[1:] != (3, dimension):
        raise ValueError("Invalid TRAIN encoding geometry")
    inputs = train[:, :-1].reshape(-1, dimension)
    _require_close(inputs.mean(0), center, "Saved PCA center is not TRAIN-input-only")
    covariance = (inputs - center).T @ (inputs - center)
    singular = np.asarray(metrics["pca"]["singular_values"])
    _require_close(components @ covariance @ components.T, np.diag(singular ** 2),
                   "Saved PCA basis does not diagonalize TRAIN input covariance", atol=1e-10)
    _require_close(np.asarray(metrics["pca"]["explained_variance_fraction"]),
                   singular ** 2 / np.sum(singular ** 2), "Invalid PCA variance fractions")
    point, vector = project_pc12(means, grid["displacement"], center, components)
    _require_close(grid["R_of_mean"] - means, grid["displacement"], "Stored displacement is not R(mean g)-mean g")
    _require_close(point, grid["projected_mean_by_scale"][..., :2], "Saved point projection mismatch")
    _require_close(vector, grid["projected_true_displacement"][..., :2], "Saved displacement projection mismatch")
    _require_close(grid["mean_R"] - grid["R_of_mean"], grid["jensen_difference"], "Jensen difference mismatch")
    # Reconstruct both grid means from saved ensemble arrays, without a model.
    labels = np.concatenate((encoded["train_labels"], encoded["test_labels"]))
    direct = np.concatenate((encoded["train_direct"], encoded["test_direct"]))
    one_step = np.concatenate((encoded["train_one_step"], encoded["test_one_step"]))
    if labels.shape != (len(direct), 2):
        raise ValueError("KB evaluation labels do not match saved ensembles")
    if one_step.shape != (len(direct), 2, dimension):
        raise ValueError("Invalid saved one-step prediction geometry")
    train_cells = set(map(tuple, encoded["train_labels"].tolist()))
    test_cells = set(map(tuple, encoded["test_labels"].tolist()))
    if train_cells & test_cells:
        raise ValueError("TRAIN/TEST physical parameter cells overlap")
    for i, k in enumerate(K):
        for j, b in enumerate(B):
            selected = (labels[:, 0] == k) & (labels[:, 1] == b)
            if not selected.any():
                raise ValueError("Missing parameter cell")
            _require_close(direct[selected].mean(0), means[i, j], "Grid mean differs from saved ensemble encodings")
            _require_close(one_step[selected].mean(0), grid["mean_R"][i, j, :2],
                           "Mean R differs from saved per-ensemble predictions")
    return ReportArrays(dimension, metrics, grid, encoded, roots, hashes)


def coupling_associations(points, K, B):
    """Evaluation-only correlation/parity, not RG eigenoperators or phase labels."""
    points, K, B = np.asarray(points), np.asarray(K), np.asarray(B)
    if points.shape != (len(K), len(B), 2):
        raise ValueError("Expected a two-PC KB grid at one scale")
    axes = np.broadcast_arrays(K[:, None], B[None, :])
    result = []
    symmetric_B = np.allclose(B, -B[::-1], rtol=0, atol=1e-14)
    for coordinate in range(2):
        values = points[..., coordinate]
        centered = values - values.mean()
        energy = float(np.mean(centered ** 2))
        correlations = []
        for axis in axes:
            a = axis - axis.mean()
            denominator = float(np.sqrt(np.mean(a ** 2) * energy))
            correlations.append(float(np.mean(a * centered) / denominator) if denominator > 0 else None)
        odd = (values - values[:, ::-1]) / 2 if symmetric_B else None
        result.append(dict(principal_component=coordinate + 1,
            correlation_with_K=correlations[0], correlation_with_B=correlations[1],
            B_odd_centered_variance_fraction=(float(np.mean(odd ** 2) / energy)
                if symmetric_B and energy > 0 else None),
            interpretation="仅样本域的参数关联；不是热/磁 RG 本征方向认证"))
    return result


def summarize(report):
    grid, components = report.grid, report.grid["pca_components"]
    rows = []
    direct = np.concatenate((report.encoded["train_direct"], report.encoded["test_direct"]))
    one_step = np.concatenate((report.encoded["train_one_step"], report.encoded["test_one_step"]))
    for scale in (0, 1):
        displacement = grid["displacement"][:, :, scale]
        mean_displacement = (grid["mean_R"] - grid["mean_by_scale"])[:, :, scale]
        jensen = grid["jensen_difference"][:, :, scale]
        full_rms = np.sqrt(np.mean(np.sum(displacement ** 2, axis=-1)))
        rows.append(dict(input_scale=scale, input_size=36 // 3 ** scale,
            target_size=36 // 3 ** (scale + 1),
            R_of_mean=drift_diagnostics(displacement, components),
            mean_R=drift_diagnostics(mean_displacement, components),
            per_ensemble_R=drift_diagnostics(one_step[:, scale] - direct[:, scale], components),
            jensen_difference_rms_norm=float(np.sqrt(np.mean(np.sum(jensen ** 2, axis=-1)))),
            jensen_rms_over_R_of_mean_displacement_rms=(
                float(np.sqrt(np.mean(np.sum(jensen ** 2, axis=-1))) / full_rms) if full_rms > 0 else None)))
    root_rows = []
    for root in report.roots["roots"]:
        point = np.asarray(root["g"])
        projected = (point - grid["pca_center"]) @ components[:2].T
        root_rows.append(dict(root_id=root["root_id"], full_g=point.tolist(),
            pc12=projected.tolist(), residual_norm=root.get("residual_norm"),
            geometric_support_candidate=root.get("geometric_support_candidate", False),
            physical_support_certified=root.get("physical_support_certified", False),
            interpretation="仅已存数值根；不标注为相变点"))
    fractions = report.metrics["pca"]["explained_variance_fraction"]
    return dict(dimension=report.dimension, pca_variance_fractions=fractions[:2],
        pca_variance_fraction_first_two=float(sum(fractions[:2])),
        microscopic_coupling_associations=coupling_associations(
            grid["projected_mean_by_scale"][:, :, 0, :2], grid["K"], grid["B"]),
        displacement_by_input_scale=rows, saved_roots=root_rows,
        accepted_numerical_roots=report.roots["accepted_unique_root_count"],
        certified_physical_roots=sum(bool(row["physical_support_certified"]) for row in root_rows),
        limitations=LIMITATIONS)


def _font_setup():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    from scripts.report_ising_zh import select_chinese_font
    path = select_chinese_font()
    font_manager.fontManager.addfont(str(path))
    name = font_manager.FontProperties(fname=str(path)).get_name()
    matplotlib.rcParams.update({"font.family": name, "font.size": 10,
        "axes.unicode_minus": False, "savefig.dpi": 180})
    return path


def _frame(ax, report):
    fractions = report.metrics["pca"]["explained_variance_fraction"]
    ax.set_xlabel(f"主成分1（训练输入方差 {100 * fractions[0]:.2f}%）")
    ax.set_ylabel(f"主成分2（训练输入方差 {100 * fractions[1]:.2f}%）")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=.16)


def _grid_lines(ax, xy, *, alpha=.22):
    for j in np.unique(np.linspace(0, xy.shape[1] - 1, 5).round().astype(int)):
        ax.plot(*xy[:, j].T, color=".45", lw=.55, alpha=alpha, zorder=1)
    for i in np.unique(np.linspace(0, xy.shape[0] - 1, 5).round().astype(int)):
        ax.plot(*xy[i].T, color=".45", lw=.55, alpha=alpha, zorder=1)


def _zero_field(ax, xy, K, B, *, annotate=False):
    column = int(np.flatnonzero(np.isclose(B, 0., atol=1e-14, rtol=0))[0])
    line = xy[:, column]
    ax.plot(*line.T, "o-", color="#253238", markerfacecolor="white", ms=3,
            lw=1.1, zorder=5, label="零场 B=0 的参数轨迹")
    if annotate:
        for target in (.2, .35, .44, .53, .68):
            index = int(np.argmin(abs(K - target)))
            ax.annotate(f"K={K[index]:.2f}", line[index], xytext=(5, 5),
                        textcoords="offset points", fontsize=8, color="#253238")


def _plot_phase_portrait(reports, output):
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    fig, axs = plt.subplots(len(reports), 2, figsize=(13.5, 6.0 * len(reports)), squeeze=False)
    fig.subplots_adjust(left=.075, right=.89, bottom=.12, top=.91, hspace=.28, wspace=.28)
    for row, report in enumerate(reports):
        grid = report.grid
        xy, drift = project_pc12(grid["mean_by_scale"], grid["displacement"],
                                grid["pca_center"], grid["pca_components"])
        K, B = grid["K"], grid["B"]
        color_values = np.broadcast_to(B[None], (len(K), len(B)))
        cap = float(np.max(abs(B)))
        norm = TwoSlopeNorm(vmin=-cap, vcenter=0., vmax=cap)
        choose_i = np.unique(np.linspace(0, len(K) - 1, min(9, len(K))).round().astype(int))
        choose_j = np.unique(np.linspace(0, len(B) - 1, min(5, len(B))).round().astype(int))
        selected = np.zeros((len(K), len(B)), dtype=bool)
        selected[np.ix_(choose_i, choose_j)] = True
        points_for_limits = [xy[:, :, :2].reshape(-1, 2)]
        for scale in (0, 1):
            points_for_limits.append((xy[:, :, scale] + drift[:, :, scale])[selected])
        limits = np.concatenate(points_for_limits)
        low, high = limits.min(0), limits.max(0)
        margin = np.maximum((high - low) * .08, 1e-5)
        for scale, ax in enumerate(axs[row]):
            points, vectors = xy[:, :, scale], drift[:, :, scale]
            _grid_lines(ax, points)
            color = ax.scatter(*points.reshape(-1, 2).T, c=color_values.ravel(),
                               cmap="coolwarm", norm=norm, s=13, alpha=.8, zorder=2)
            ax.quiver(*points[selected].T, *vectors[selected].T, angles="xy",
                      scale_units="xy", scale=1., width=.0034, color="#14232b", zorder=4)
            _zero_field(ax, points, K, B)
            _frame(ax, report)
            ax.set(xlim=(low[0] - margin[0], high[0] + margin[0]),
                   ylim=(low[1] - margin[1], high[1] + margin[1]),
                   title=f"{report.dimension}维；k={scale}，{36 // 3 ** scale}→{36 // 3 ** (scale + 1)}")
            if scale == 0:
                ax.legend(loc="best", fontsize=8)
        cax = fig.add_axes([.915, .13 + .405 * (len(reports) - 1 - row), .015, .30])
        fig.colorbar(color, cax=cax, label="微观磁场 B（仅评价着色）")
    fig.suptitle("KB 扫描：前两主成分上的流场相图", fontsize=17, y=.975)
    fig.text(.5, .025, "箭头为 R(平均 g)−平均 g 的真实投影，长度未归一化；仅已有数据格点，无平面补点或流线插值。\n"
             "两模型分别沿用自身训练主成分；零场折线不是临界线，未获数据支持的数值根未标为相变点。",
             ha="center", fontsize=10)
    fig.savefig(output / "PC12_phase_portrait.png")
    plt.close(fig)


def _plot_coupling_maps(reports, output):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, TwoSlopeNorm
    fig, axs = plt.subplots(len(reports), 2, figsize=(13.5, 5.8 * len(reports)),
                            squeeze=False, layout="constrained")
    for row, report in enumerate(reports):
        grid = report.grid
        xy, _ = project_pc12(grid["mean_by_scale"], grid["displacement"],
                             grid["pca_center"], grid["pca_components"])
        points, K, B = xy[:, :, 0], grid["K"], grid["B"]
        for column, (values, title, cmap, norm) in enumerate((
                (np.broadcast_to(K[:, None], points.shape[:2]), "微观 K", "viridis", Normalize(K[0], K[-1])),
                (np.broadcast_to(B[None], points.shape[:2]), "微观 B", "coolwarm",
                 TwoSlopeNorm(vmin=B[0], vcenter=0., vmax=B[-1])))):
            ax = axs[row, column]
            _grid_lines(ax, points)
            color = ax.scatter(*points.reshape(-1, 2).T, c=values.ravel(), cmap=cmap,
                               norm=norm, s=23, alpha=.85)
            _zero_field(ax, points, K, B, annotate=column == 0)
            _frame(ax, report)
            ax.set_title(f"{report.dimension}维微观编码；按{title}着色")
            fig.colorbar(color, ax=ax, label=f"{title}（仅评价）", fraction=.045)
    fig.suptitle("KB 参数网格如何嵌入前两主成分平面\n同一系综条件的两条物理链先平均；未进行相标签分类", fontsize=15)
    fig.savefig(output / "PC12_coupling_map.png")
    plt.close(fig)


def _plot_scale_trajectories(reports, output):
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    fig, axs = plt.subplots(1, len(reports), figsize=(7.1 * len(reports), 6.5),
                            squeeze=False, layout="constrained")
    for ax, report in zip(axs[0], reports):
        grid = report.grid
        xy, _ = project_pc12(grid["mean_by_scale"], grid["displacement"],
                             grid["pca_center"], grid["pca_components"])
        K, B = grid["K"], grid["B"]
        values = np.broadcast_to(B[None], xy.shape[:2]).ravel()
        cap = float(np.max(abs(B)))
        norm = TwoSlopeNorm(vmin=-cap, vcenter=0., vmax=cap)
        for scale, (marker, size) in enumerate((("o", 16), ("^", 14), ("s", 11))):
            color = ax.scatter(*xy[:, :, scale].reshape(-1, 2).T, c=values,
                cmap="coolwarm", norm=norm, marker=marker, s=size, alpha=.55,
                label=f"k={scale}，L={36 // 3 ** scale}")
        for i in np.unique(np.linspace(0, len(K) - 1, 5).round().astype(int)):
            for j in (0, len(B) // 2, len(B) - 1):
                path = xy[i, j]
                ax.plot(*path.T, color=".3", ls="--", lw=.7, alpha=.7)
                for scale in (0, 1):
                    ax.annotate("", xy=path[scale + 1], xytext=path[scale],
                                arrowprops=dict(arrowstyle="->", lw=.7, color=".3", alpha=.7))
        _frame(ax, report)
        ax.set_title(f"{report.dimension}维；同一参数的直接粗粒化轨迹")
        ax.legend(fontsize=9)
        fig.colorbar(color, ax=ax, label="微观磁场 B（仅评价）", fraction=.045)
    fig.suptitle("直接编码的跨尺度轨迹：36→12→4\n虚线连接同一 KB 条件；不是 R 的额外迭代或二维流线", fontsize=15)
    fig.savefig(output / "PC12_scale_trajectories.png")
    plt.close(fig)


def _plot_diagnostics(reports, summaries, output):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    labels = [f"{report.dimension}维" for report in reports]
    variance = [100 * row["pca_variance_fraction_first_two"] for row in summaries]
    bars = axs[0].bar(labels, variance, color=("#3a7ca5", "#ad5c7b"))
    axs[0].set(ylim=(0, 105), ylabel="训练输入方差比例（%）", title="前两主成分保留的静态方差")
    for bar, value in zip(bars, variance):
        axs[0].text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.2f}%", ha="center")
    for scale, offset in ((0, -.18), (1, .18)):
        values = [row["displacement_by_input_scale"][scale]["per_ensemble_R"]["outside_pc12_energy_fraction"]
                  for row in summaries]
        heights = [100 * value if value is not None else np.nan for value in values]
        bars = axs[1].bar(np.arange(len(labels)) + offset, heights, width=.32,
                         label=f"输入尺度 k={scale}")
        for bar, value in zip(bars, heights):
            if np.isfinite(value):
                axs[1].text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.1f}%", ha="center")
    axs[1].set(xticks=np.arange(len(labels)), xticklabels=labels, ylim=(0, 105),
               ylabel="逐系综完整位移的平面外能量比例（%）", title="高方差解释率不等于二维流闭合")
    axs[1].legend()
    fig.suptitle("投影充分性的检查：统计几何与动力学分别检验", fontsize=15)
    fig.savefig(output / "PC12_projection_diagnostics.png")
    plt.close(fig)


def write_plots(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    # Never place new files inside either frozen original report directory.
    if output.exists() or output.is_symlink():
        raise FileExistsError("Refuse to overwrite an existing plot directory")
    if any(output.is_relative_to(source / f"dim{dimension}" / "report") for dimension in (5, 10)):
        raise ValueError("New plots must not be placed inside an original report directory")
    reports = [load_report(source, dimension) for dimension in (5, 10)]
    summaries = [summarize(report) for report in reports]
    font = _font_setup()
    output.mkdir(parents=True, exist_ok=False)
    _plot_phase_portrait(reports, output)
    _plot_coupling_maps(reports, output)
    _plot_scale_trajectories(reports, output)
    _plot_diagnostics(reports, summaries, output)
    arrays = {}
    for report in reports:
        grid = report.grid
        points, drift = project_pc12(grid["mean_by_scale"], grid["displacement"],
                                    grid["pca_center"], grid["pca_components"])
        prefix = f"dim{report.dimension}_"
        arrays.update({prefix + key: value for key, value in dict(K=grid["K"], B=grid["B"],
            pc12_points=points, pc12_R_of_mean_displacement=drift,
            pc12_mean_R_displacement=(grid["mean_R"] - grid["mean_by_scale"]) @ grid["pca_components"][:2].T,
            full_R_of_mean_displacement=grid["displacement"],
            pca_center=grid["pca_center"], pca_components=grid["pca_components"]).items()})
    np.savez_compressed(output / "PC12_arrays.npz", **arrays)
    source_hashes = {key: value for report in reports for key, value in report.source_hashes.items()}
    if any(sha256(source / key) != value for key, value in source_hashes.items()):
        raise ValueError("Source reports changed during plotting")
    result = dict(status="complete", source=str(source), source_sha256=source_hashes,
        font_path=str(font), models=summaries, limitations=LIMITATIONS,
        additional_model_inference=False, physical_sampling=False, training_updates=0,
        new_root_search=False, interpolation_or_off_data_grid=False,
        script_sha256=sha256(Path(__file__)),
        output_sha256={path.name: sha256(path) for path in sorted(output.iterdir()) if path.is_file()})
    (output / "PC12_report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = write_plots(args.source, args.output)
    print(json.dumps(dict(status=result["status"], models=result["models"]), ensure_ascii=False, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
