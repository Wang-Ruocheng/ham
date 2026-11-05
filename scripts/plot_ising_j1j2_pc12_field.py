"""只读已导出的共享 R 网格，绘制一张 PC1/PC2 切片流场。

This module imports no model, runs no inference, and never searches for roots.
The colored mask describes projected TRAIN geometry, not equilibrium validity
or support of the full five-dimensional slice.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np


REQUIRED_KEYS = frozenset((
    "grid_x", "grid_y", "projected_delta", "full_delta", "support_mask",
    "projected_test", "projected_train", "rho_test",
    "observed_projected_delta_test", "pca_ratio",
))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_field(field, metadata):
    """Validate a small public array export; no PT/model loading is allowed."""
    field, metadata = Path(field).resolve(), Path(metadata).resolve()
    if field.suffix != ".npz" or metadata.suffix != ".json":
        raise ValueError("仅接受公开 NPZ 网格与 JSON 导出回执")
    details = json.loads(metadata.read_text(encoding="utf-8"))
    if (details.get("status") != "complete"
            or details.get("diagnostic_only") is not True
            or details.get("equilibrium_not_validated") is not True):
        raise ValueError("需要已完成、明确标注未热化诊断的导出回执")
    expected = details.get("field_sha256", details.get("field_npz_sha256", details.get("npz_sha256")))
    if not isinstance(expected, str) or expected != sha256(field):
        raise ValueError("网格 NPZ 的 SHA256 与回执不一致")
    for key in ("support_description", "slice_description"):
        if not isinstance(details.get(key), str) or not details[key].strip():
            raise ValueError(f"缺少几何解释：{key}")
    with np.load(field, allow_pickle=False) as archive:
        if not REQUIRED_KEYS <= set(archive.files):
            raise ValueError(f"网格缺少字段：{sorted(REQUIRED_KEYS - set(archive.files))}")
        values = {name: archive[name].copy() for name in REQUIRED_KEYS}
    x, y = values["grid_x"], values["grid_y"]
    for coordinate in (x, y):
        if (coordinate.ndim != 1 or not 3 <= len(coordinate) <= 500
                or not np.isfinite(coordinate).all()
                or not np.all(np.diff(coordinate) > 0)
                or not np.allclose(np.diff(coordinate), np.diff(coordinate)[0],
                                   rtol=1e-7, atol=1e-12)):
            raise ValueError("流场网格坐标必须有限、严格递增且等间距")
    shape = (len(y), len(x))
    if (values["projected_delta"].shape != (*shape, 2)
            or values["full_delta"].shape != (*shape, 5)
            or values["support_mask"].shape != shape
            or values["support_mask"].dtype != np.bool_):
        raise ValueError("位移或几何掩码的维度不符合二维网格、五维潜变量约定")
    for key in ("projected_train", "projected_test"):
        array = values[key]
        if array.ndim != 3 or array.shape[1:] != (3, 2) or not len(array):
            raise ValueError("真实编码投影应为 [系综数, 三个尺度, 两个主成分]")
    test_shape = values["projected_test"].shape
    if (values["observed_projected_delta_test"].shape != test_shape
            or values["rho_test"].shape != (test_shape[0],)
            or values["pca_ratio"].shape != (5,)):
        raise ValueError("测试集标签、真实位移或主成分占比维度不一致")
    if any(not np.isfinite(array).all() for key, array in values.items()
           if key != "support_mask"):
        raise ValueError("导出数组含非有限数值")
    ratio = values["pca_ratio"]
    if (np.any(ratio < 0) or np.any(np.diff(ratio) > 1e-10)
            or not np.isclose(ratio.sum(), 1., rtol=0, atol=1e-6)):
        raise ValueError("五维 PCA 占比应非负、降序且和为一")
    if len(np.unique(values["rho_test"])) > 8:
        raise ValueError("此单图最多支持八个耦合比，避免图例拥挤")
    return values, details, {"field_npz": sha256(field), "metadata_json": sha256(metadata)}


def _apply_chinese_font(figure, font_path):
    """Check all Chinese glyphs, including newly added labels, before save."""
    from matplotlib.ft2font import FT2Font
    from matplotlib.text import Text

    available = FT2Font(str(font_path)).get_charmap()
    texts = []
    for artist in figure.findobj(Text):
        value = artist.get_text()
        missing = {character for character in value
                   if re.fullmatch(r"[\u3400-\u9fff]", character)
                   and ord(character) not in available}
        if missing:
            raise RuntimeError(f"中文字体缺少字形：{''.join(sorted(missing))}")
        properties = artist.get_fontproperties().copy()
        properties.set_file(str(font_path))
        artist.set_fontproperties(properties)
        if value:
            texts.append(value)
    return texts


def write_plot(field, metadata, output):
    """Create an exclusive new directory with the figure and its provenance."""
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("使用新目录，不覆盖既有图或正式报告")
    output = output.resolve()
    arrays, details, sources = load_field(field, metadata)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    from scripts.report_ising_zh import select_chinese_font

    font = select_chinese_font()
    x, y = arrays["grid_x"], arrays["grid_y"]
    delta, mask = arrays["projected_delta"], arrays["support_mask"]
    speed = np.linalg.norm(delta, axis=-1)
    supported_speed = speed[mask]
    vmax = float(np.max(supported_speed if supported_speed.size else speed))
    norm = Normalize(0., max(vmax, np.finfo(float).eps))
    cumulative = float(arrays["pca_ratio"][:2].sum())

    with plt.rc_context({"font.size": 11, "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False}):
        figure = plt.figure(figsize=(12.4, 8.5))
        axis = figure.add_axes((.09, .285, .72, .53))
        color_axis = figure.add_axes((.845, .285, .022, .53))
        figure.text(.085, .955, "NN–NNN Ising：前两主成分上的二维流场",
                    fontsize=18, weight="bold", color="#16324a")
        figure.text(.085, .91, "未热化诊断：不是已认证的平衡态相图",
                    fontsize=13, weight="bold", color="#b42332")
        figure.text(.085, .865,
                    f"共享映射 $R$ 的五维均值切片；前两主成分累计方差 {100 * cumulative:.2f}%",
                    color="#42566a", fontsize=12)
        axis.set_facecolor("#fafbfc")
        background = axis.streamplot(x, y, delta[..., 0], delta[..., 1],
                                     density=1.2, color="#c6cbd0", linewidth=.7,
                                     arrowsize=.75, zorder=1)
        background.lines.set_alpha(.65)
        if mask.any():
            masked_x = np.ma.masked_where(~mask, delta[..., 0])
            masked_y = np.ma.masked_where(~mask, delta[..., 1])
            axis.streamplot(x, y, masked_x, masked_y, density=1.8,
                            color=np.ma.masked_where(~mask, speed), cmap="viridis",
                            norm=norm, linewidth=1.25, arrowsize=1., zorder=2)
        if mask.any() and not mask.all():
            axis.contour(x, y, mask.astype(float), levels=[.5], colors=["#aab3bc"],
                         linewidths=.65, linestyles="dotted", zorder=2)
        # Preserve the vector direction and relative length with one common
        # quiver scale. The scale is disclosed, never normalized pointwise.
        normalized_speed = np.linalg.norm(delta / [np.ptp(x), np.ptp(y)], axis=-1)
        largest = float(np.max(normalized_speed[mask])) if mask.any() else 0.
        quiver_scale = max(largest / .055, np.finfo(float).eps)
        sy, sx = max(1, len(y) // 15), max(1, len(x) // 17)
        xx, yy = np.meshgrid(x, y)
        sparse = np.zeros_like(mask)
        sparse[::sy, ::sx] = True
        selected = sparse & mask & (speed > 0)
        axis.quiver(xx[selected], yy[selected], delta[..., 0][selected],
                    delta[..., 1][selected], angles="xy", scale_units="xy",
                    scale=quiver_scale, width=.0021, color="#223a50", alpha=.5,
                    headwidth=3.5, zorder=3)

        test = arrays["projected_test"]
        rho = arrays["rho_test"]
        palette = ("#cb493c", "#246bad", "#a364b0", "#da8c24",
                   "#309079", "#8c564b", "#b6507b", "#777777")
        rho_handles = []
        for value, color in zip(np.unique(rho), palette):
            keep = rho == value
            for scale, marker in ((0, "o"), (1, "^")):
                axis.scatter(test[keep, scale, 0], test[keep, scale, 1], marker=marker,
                             s=24, facecolor=color, edgecolor="white", linewidth=.45,
                             alpha=.85, zorder=4)
            rho_handles.append(Line2D([], [], marker="o", linestyle="", markersize=6,
                                      markerfacecolor=color, markeredgecolor="white",
                                      label=rf"耦合比 $\rho={value:g}$"))
        scale_handles = [Line2D([], [], marker=marker, linestyle="", markersize=6,
                               color="#536373", label=label)
                         for marker, label in (("o", r"真实测试系综：$k=0$"),
                                               ("^", r"真实测试系综：$k=1$"))]
        figure.legend(handles=rho_handles, loc="lower left", bbox_to_anchor=(.085, .162),
                      frameon=False, ncol=min(4, len(rho_handles)), handletextpad=.3,
                      columnspacing=1.5)
        figure.legend(handles=scale_handles, loc="lower left", bbox_to_anchor=(.085, .119),
                      frameon=False, ncol=2, handletextpad=.3, columnspacing=2.)
        axis.set(xlabel="第一主成分坐标（PC1）", ylabel="第二主成分坐标（PC2）",
                 xlim=(x[0], x[-1]), ylim=(y[0], y[-1]))
        axis.grid(alpha=.12, linewidth=.5)
        colorbar = figure.colorbar(ScalarMappable(norm=norm, cmap="viridis"), cax=color_axis)
        colorbar.set_label(r"投影位移大小 $\|[R(g)-g]_{\mathrm{PC1,PC2}}\|$", labelpad=13)
        figure.text(.085, .086,
                    "其余主成分固定训练均值；彩色区域仅投影数据邻域，灰色为外推。",
                    fontsize=10, color="#465766")
        figure.text(.085, .052,
                    r"流线为 $R(g)-g$ 方向示意，不是离散迭代轨迹；方向箭头按统一比例缩放。",
                    fontsize=10, color="#465766")
        figure.text(.085, .021,
                    "点为真实五维编码的投影，不保证位于该切片；未搜索或标注不动点。",
                    fontsize=9.5, color="#677583")
        try:
            _apply_chinese_font(figure, font)
            figure.canvas.draw()
            texts = _apply_chinese_font(figure, font)
            output.mkdir(parents=True, exist_ok=False)
            image_path = output / "flow2d.png"
            with image_path.open("xb") as image:
                figure.savefig(image, format="png", dpi=180, facecolor="white")
        finally:
            plt.close(figure)

    receipt = dict(
        schema_version=1, status="complete", diagnostic_only=True,
        equilibrium_not_validated=True, model_inference_performed=False,
        root_search_performed=False, panels=1,
        vector_definition="projected R(g)-g; PC3–PC5 fixed at TRAIN mean",
        streamlines_are_discrete_rg_trajectories=False,
        plotted_observed_input_scales=[0, 1], final_scale_as_R_input=False,
        points_are_projections_not_slice_members=True,
        support_is_projected_geometry_only=True,
        support_description=details["support_description"],
        slice_description=details["slice_description"],
        source_paths=dict(field_npz=str(Path(field).resolve()),
                          metadata_json=str(Path(metadata).resolve())),
        source_sha256=sources,
        source_export_hashes=details.get("source_hashes", details.get("source_sha256")),
        output_sha256={"flow2d.png": sha256(image_path)},
        pca_ratio=arrays["pca_ratio"].tolist(), pc12_variance_fraction=cumulative,
        projected_support_grid_points=int(mask.sum()), grid_points=int(mask.size),
        train_ensembles=len(arrays["projected_train"]),
        test_ensembles=len(test), rho_values=np.unique(rho).tolist(),
        color_range=[0., norm.vmax], quiver_common_scale=quiver_scale,
        font_path=str(font), rendered_text=texts,
    )
    for name in ("slice_discrepancy_to_observed_displacement_RMS_ratio",
                 "supported_normal_to_full_displacement_RMS_ratio"):
        if name in details:
            receipt[name] = details[name]
    with (output / "plot.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    if sources != {"field_npz": sha256(field), "metadata_json": sha256(metadata)}:
        raise RuntimeError("绘图期间输入文件发生变化")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    receipt = write_plot(args.field, args.metadata, args.output)
    print(json.dumps(dict(status=receipt["status"], output=str(args.output.resolve()),
                          pc12_variance_fraction=receipt["pc12_variance_fraction"]),
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
