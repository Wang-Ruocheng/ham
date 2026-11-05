"""数据域内的 PC12 连续流线及已验收高维根投影；原报告保持只读。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay, cKDTree

from scripts.plot_ising_kb_pc12 import _font_setup, load_report, sha256


class InterpolatedFlow:
    """Piecewise-linear full displacement, with explicit unsupported-domain masks.

    Interpolation is done separately for each input scale. A zero in the first
    two components is not a zero of all components. The masks are diagnostic
    plotting criteria, not a statistical certification of manifold support.
    """
    def __init__(self, points, full_displacements, basis, *, states=None,
                 max_edge_ratio=8., max_hidden_slope=5.):
        self.points = np.asarray(points, dtype=np.float64)
        self.full_displacements = np.asarray(full_displacements, dtype=np.float64)
        self.basis = np.asarray(basis, dtype=np.float64)
        if (self.points.ndim != 2 or self.points.shape[1] != 2
                or self.full_displacements.ndim != 2
                or len(self.points) != len(self.full_displacements)
                or self.basis.shape != (2, self.full_displacements.shape[1])
                or len(self.points) < 3
                or not all(np.isfinite(x).all() for x in
                           (self.points, self.full_displacements, self.basis))):
            raise ValueError("Require finite points[N,2], displacements[N,d], basis[2,d]")
        if not np.allclose(self.basis @ self.basis.T, np.eye(2), atol=1e-12, rtol=0):
            raise ValueError("Projection basis must be orthonormal")
        if not np.isfinite(max_edge_ratio) or max_edge_ratio <= 0 or max_hidden_slope <= 0:
            raise ValueError("Mask thresholds must be positive")
        self.states = None if states is None else np.asarray(states, dtype=np.float64)
        if self.states is not None and (self.states.shape != self.full_displacements.shape
                                        or not np.isfinite(self.states).all()):
            raise ValueError("States must match full displacements")
        self.vectors = self.full_displacements @ self.basis.T
        self.triangulation = Delaunay(self.points)
        indices = self.triangulation.simplices
        vertices = self.points[indices]
        lengths = np.linalg.norm(vertices - np.roll(vertices, 1, axis=1), axis=-1)
        local_spacing = cKDTree(self.points).query(self.points, k=min(5, len(self.points)))[0][:, -1]
        floor = max(float(np.ptp(self.points, axis=0).max()) * 1e-10, 1e-14)
        self.edge_ratio = lengths.max(1) / np.maximum(np.median(local_spacing[indices], axis=1), floor)
        doubled_area = np.abs(np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]))
        self.quality = doubled_area / np.maximum(lengths.max(1) ** 2, floor ** 2)
        self.hidden_slope = np.zeros(len(indices))
        if self.states is not None:
            hidden = self.states - (self.states @ self.basis.T) @ self.basis
            delta = hidden[indices] - np.roll(hidden[indices], 1, axis=1)
            self.hidden_slope = np.max(np.linalg.norm(delta, axis=-1) / np.maximum(lengths, floor), axis=1)
        self.valid_triangles = ((self.edge_ratio <= max_edge_ratio)
                                & (self.quality > 1e-7)
                                & (self.hidden_slope <= max_hidden_slope))
        self.max_edge_ratio, self.max_hidden_slope = max_edge_ratio, max_hidden_slope

    def weights(self, query):
        query = np.asarray(query, dtype=np.float64)
        if query.shape[-1] != 2 or not np.isfinite(query).all():
            raise ValueError("Require finite two-coordinate queries")
        flat = query.reshape(-1, 2)
        simplex = self.triangulation.find_simplex(flat)
        valid = simplex >= 0
        valid[valid] &= self.valid_triangles[simplex[valid]]
        weights = np.full((len(flat), 3), np.nan)
        if valid.any():
            transform = self.triangulation.transform[simplex[valid]]
            first = np.einsum("nij,nj->ni", transform[:, :2], flat[valid] - transform[:, 2])
            weights[valid] = np.column_stack((first, 1 - first.sum(1)))
        return simplex, weights, valid

    def interpolate(self, query, values):
        query = np.asarray(query)
        values = np.asarray(values)
        if values.ndim != 2 or len(values) != len(self.points):
            raise ValueError("Interpolation values must match field points")
        simplex, weights, valid = self.weights(query)
        result = np.full((len(simplex), values.shape[1]), np.nan)
        if valid.any():
            indices = self.triangulation.simplices[simplex[valid]]
            result[valid] = np.einsum("ni,nij->nj", weights[valid], values[indices])
        return result.reshape(*query.shape[:-1], values.shape[-1])

    def evaluate(self, query):
        return self.interpolate(query, self.full_displacements)

    def zeros(self):
        records = []
        tolerance = max(float(np.ptp(self.points, axis=0).max()) * 1e-7, 1e-12)
        for simplex, indices in enumerate(self.triangulation.simplices):
            if not self.valid_triangles[simplex]:
                continue
            matrix = np.vstack((self.vectors[indices].T, np.ones(3)))
            if np.linalg.cond(matrix) > 1e12:
                continue
            weights = np.linalg.solve(matrix, np.array([0., 0., 1.]))
            if weights.min() < -1e-10 or weights.max() > 1 + 1e-10:
                continue
            point = weights @ self.points[indices]
            full = weights @ self.full_displacements[indices]
            projected = full @ self.basis.T
            if np.linalg.norm(projected) > 1e-9:
                continue
            if any(np.linalg.norm(point - np.asarray(row["xy"])) < tolerance for row in records):
                continue
            affine = np.linalg.solve(np.column_stack((self.points[indices], np.ones(3))),
                                     self.vectors[indices])
            records.append(dict(xy=point.tolist(), simplex_index=int(simplex),
                projected_residual=float(np.linalg.norm(projected)),
                interpolated_full_residual=float(np.linalg.norm(full)),
                full_root_certified=False, kind="two_dimensional_interpolation_zero",
                interpolated_vector_jacobian=affine[:2].T.tolist(),
                barycentric_weights=weights.tolist()))
        return records


def find_interpolated_zeros(points, vectors):
    return InterpolatedFlow(points, vectors, np.eye(2)).zeros()


def saved_root_markers(report):
    basis, center = report.grid["pca_components"][:2], report.grid["pca_center"]
    return [dict(root_id=root["root_id"], full_g=list(root["g"]),
        xy=((np.asarray(root["g"]) - center) @ basis.T).tolist(),
        residual_norm=root.get("residual_norm"),
        geometric_support_candidate=root.get("geometric_support_candidate", False),
        physical_support_certified=root.get("physical_support_certified", False),
        kind="full_coordinate_root_projection") for root in report.roots["roots"]]


def make_grid(field, bounds, resolution=180):
    if not isinstance(resolution, int) or not 20 <= resolution <= 400:
        raise ValueError("Grid resolution must be between 20 and 400")
    lower, upper = np.asarray(bounds[0]), np.asarray(bounds[1])
    if lower.shape != (2,) or upper.shape != (2,) or not np.all(upper > lower):
        raise ValueError("Invalid plot bounds")
    x, y = (np.linspace(lower[i], upper[i], resolution) for i in (0, 1))
    queries = np.stack(np.meshgrid(x, y), axis=-1)
    vectors = field.evaluate(queries) @ field.basis.T
    return x, y, np.ma.masked_invalid(vectors)


class FrozenR:
    def __init__(self, values, dimension):
        self.dimension = dimension
        prefix = f"dim{dimension}_"
        self.W = [values[prefix + f"W{i}"] for i in range(3)]
        self.b = [values[prefix + f"b{i}"] for i in range(3)]
        if any(value.dtype != np.float64 or not np.isfinite(value).all() for value in (*self.W, *self.b)):
            raise ValueError("Require finite float64 R parameters")
        if (self.W[0].shape != (96, dimension) or self.W[1].shape != (96, 96)
                or self.W[2].shape != (dimension, 96)
                or [x.shape for x in self.b] != [(96,), (96,), (dimension,)]):
            raise ValueError("Unexpected frozen R architecture")
        inputs, outputs = values[prefix + "validation_input"], values[prefix + "validation_output"]
        self.validation_error = float(np.max(abs(inputs + self.displacement(inputs) - outputs)))
        if self.validation_error > 1e-12:
            raise ValueError("NumPy R disagrees with original frozen module")

    def displacement(self, points):
        value = np.tanh(np.asarray(points) @ self.W[0].T + self.b[0])
        value = np.tanh(value @ self.W[1].T + self.b[1])
        return value @ self.W[2].T + self.b[2]


def _load_frozen_R(export, report):
    export = Path(export)
    metadata = json.loads((export / "metadata.json").read_text())
    if sha256(export / "R_parameters.npz") != metadata["npz_sha256"]:
        raise ValueError("R parameter export SHA mismatch")
    receipt = metadata["models"][f"dim{report.dimension}"]
    expected = report.metrics["freeze_verification"]["checkpoint_sha256_before"]
    if (receipt["checkpoint_sha256_before"] != expected
            or receipt["checkpoint_sha256_after"] != expected
            or receipt["checkpoint_epoch"] != 12000 or not receipt["source_unchanged"]
            or metadata["activation"] != "tanh" or metadata["device"] != "cpu"):
        raise ValueError("Frozen R export does not match the formal report")
    with np.load(export / "R_parameters.npz", allow_pickle=False) as arrays:
        model = FrozenR(arrays, report.dimension)
    expected_drift = report.grid["displacement"]
    error = float(np.max(abs(model.displacement(report.grid["mean_by_scale"]) - expected_drift)))
    if error > 1e-11:
        raise ValueError("R parameters do not reproduce saved report displacements")
    return model, dict(parameter_export_sha256=metadata["npz_sha256"],
        metadata_sha256=sha256(export / "metadata.json"), validation_error=model.validation_error,
        saved_grid_displacement_max_abs_error=error, checkpoint_sha256=expected)


def write_streamplots(source, export, output, *, resolution=180):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError("Refuse to overwrite an earlier figure directory")
    if any(output.is_relative_to(source / f"dim{d}/report") for d in (5, 10)):
        raise ValueError("New figures must be outside original reports")
    reports = [load_report(source, dimension) for dimension in (5, 10)]
    _font_setup()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.colors import Normalize

    output.mkdir(parents=True, exist_ok=False)
    fig, axs = plt.subplots(2, 2, figsize=(13.5, 11.6))
    fig.subplots_adjust(left=.075, right=.89, top=.90, bottom=.14, hspace=.28, wspace=.25)
    rows, arrays, source_hashes = [], {}, {}
    with np.errstate(over="raise", invalid="raise"):
        for row, report in enumerate(reports):
            source_hashes.update(report.source_hashes)
            model, provenance = _load_frozen_R(export, report)
            grid = report.grid
            basis, center = grid["pca_components"][:2], grid["pca_center"]
            projected = (grid["mean_by_scale"] - center) @ basis.T
            run_speed_max = float(np.max(np.linalg.norm(
                grid["displacement"][:, :, :2] @ basis.T, axis=-1)))
            source_inputs = projected[:, :, :2].reshape(-1, 2)
            span = np.maximum(np.ptp(source_inputs, axis=0), 1e-8)
            lower, upper = source_inputs.min(0) - .12 * span, source_inputs.max(0) + .12 * span
            markers = saved_root_markers(report)
            for root in markers:
                residual = float(np.linalg.norm(model.displacement(np.asarray(root["full_g"]))))
                if residual > 1e-8:
                    raise ValueError("A saved full-coordinate root failed R-only reverification")
                root["reverified_full_residual"] = residual
            panels = []
            for scale, ax in enumerate(axs[row]):
                points = projected[:, :, scale].reshape(-1, 2)
                states = grid["mean_by_scale"][:, :, scale].reshape(-1, report.dimension)
                displacement = grid["displacement"][:, :, scale].reshape(-1, report.dimension)
                field = InterpolatedFlow(points, displacement, basis, states=states)
                x, y, vectors = make_grid(field, (lower, upper), resolution)
                speed = np.sqrt(np.sum(vectors ** 2, axis=-1))
                norm = Normalize(0., run_speed_max)
                ax.set_facecolor("#f0f0f0")
                ax.contourf(x, y, np.where(np.ma.getmaskarray(speed), np.nan, 1.),
                            levels=[.5, 1.5], colors=["white"], zorder=0)
                streams = ax.streamplot(x, y, vectors[..., 0], vectors[..., 1], color=speed,
                    cmap="viridis", norm=norm, density=1.35, linewidth=1.05, arrowsize=1.05,
                    minlength=.035, maxlength=3.5, zorder=3)
                ax.scatter(*points.T, c="#9dabb0", s=5, alpha=.23, zorder=2)
                zeros = field.zeros()
                for index, zero in enumerate(zeros):
                    zero["zero_id"] = f"P{index + 1}"
                    xy = np.asarray(zero["xy"])
                    indices = field.triangulation.simplices[zero["simplex_index"]]
                    lifted = np.asarray(zero["barycentric_weights"]) @ states[indices]
                    actual = model.displacement(lifted)
                    zero.update(lifted_full_g=lifted.tolist(),
                        actual_R_at_lift_projected_residual=float(np.linalg.norm(actual @ basis.T)),
                        actual_R_at_lift_full_residual=float(np.linalg.norm(actual)),
                        actual_R_at_lift_normal_residual=float(np.linalg.norm(actual - (actual @ basis.T) @ basis)))
                    ax.scatter(*xy, s=53, marker="o", facecolors="white", edgecolors="#b63679", lw=1.5, zorder=8)
                    ax.annotate(zero["zero_id"], xy, xytext=(-21, 9), textcoords="offset points",
                                color="#a5296a", fontsize=8, zorder=9)
                for marker in markers:
                    xy = np.asarray(marker["xy"])
                    ax.scatter(*xy, marker="*", s=180, facecolors="#ffcf5b", edgecolors="#9a4900", lw=1., zorder=10)
                    ax.annotate(marker["root_id"], xy,
                                xytext=(7, -13) if marker["root_id"] == "F1" else (5, 7),
                                textcoords="offset points",
                                color="#8c4100", fontsize=9, weight="bold", zorder=11)
                ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]),
                       xlabel="训练主成分1", ylabel="训练主成分2",
                       title=f"{report.dimension}维；k={scale}，{36 // 3 ** scale}→{36 // 3 ** (scale + 1)}")
                ax.set_aspect("equal", adjustable="box")
                ax.grid(alpha=.09)
                if not markers:
                    ax.text(.03, .95, "本次有限搜索未找到完整十维不动点", transform=ax.transAxes,
                            va="top", fontsize=8, color="#825014")
                cax = fig.add_axes([.915, .555 if row == 0 else .17, .014, .25]) if scale == 1 else None
                if cax is not None:
                    # Same run/scales share the norm to make color comparable.
                    fig.colorbar(streams.lines, cax=cax, label="投影位移大小")
                panels.append(dict(input_scale=scale, projected_interpolation_zeros=zeros,
                    simplices=len(field.valid_triangles), retained_simplices=int(field.valid_triangles.sum()),
                    max_edge_ratio=field.max_edge_ratio, max_hidden_slope=field.max_hidden_slope,
                    masked_grid_fraction=float(np.mean(np.ma.getmaskarray(speed)))))
                prefix = f"dim{report.dimension}_k{scale}_"
                arrays.update({prefix + name: value for name, value in dict(x=x, y=y,
                    projected_displacement=np.ma.filled(vectors, np.nan),
                    valid_triangles=field.valid_triangles, simplices=field.triangulation.simplices).items()})
            rows.append(dict(dimension=report.dimension, R_provenance=provenance,
                             full_coordinate_roots=markers, panels=panels))
    fig.suptitle("前两主成分的连续流场与不动点投影", fontsize=18, y=.975)
    legend = [Line2D([], [], marker="*", linestyle="none", markersize=13,
                     markerfacecolor="#ffcf5b", markeredgecolor="#9a4900", label="F：完整高维数值不动点的投影"),
              Line2D([], [], marker="o", linestyle="none", markersize=8,
                     markerfacecolor="white", markeredgecolor="#b63679", label="P：仅二维投影插值场的零点")]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.49, .065), ncol=2, fontsize=10)
    fig.text(.49, .022, "流线是离散位移方向的连续示意；仅数据域内分片线性插值，灰区不绘制。\n"
             "F 的其余主成分不同于插值数据面，故投影未必落在流线零点；五维六个数值根均未通过物理支持认证。",
             ha="center", fontsize=9)
    fig.savefig(output / "PC12_streamlines_fixed_points.png", dpi=190)
    plt.close(fig)
    np.savez_compressed(output / "stream_field_arrays.npz", **arrays)
    if any(sha256(source / path) != digest for path, digest in source_hashes.items()):
        raise ValueError("Original reports changed during plotting")
    result = dict(status="complete", source_sha256=source_hashes, models=rows,
        method="各输入尺度独立的分片线性完整位移插值，再投影；不训练或采样。",
        two_dimensional_zeros_are_not_full_roots=True, new_full_root_search=False,
        interpolation_masks_are_not_support_certificates=True,
        source_unchanged=True, script_sha256=sha256(Path(__file__)),
        output_sha256={path.name: sha256(path) for path in output.iterdir() if path.is_file()})
    (output / "stream_report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--r-export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = write_streamplots(args.source, args.r_export, args.output)
    print(json.dumps(dict(status=result["status"], models=[dict(dimension=m["dimension"],
        full_roots=len(m["full_coordinate_roots"]),
        projection_zeros=[len(p["projected_interpolation_zeros"]) for p in m["panels"]])
        for m in result["models"]]), ensure_ascii=False))


if __name__ == "__main__":
    main()
