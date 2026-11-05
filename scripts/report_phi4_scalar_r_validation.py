"""Public-JSON/NPZ scalar phi4 R report, without Q/C or checkpoint loading.

The report evaluates frozen public residual-map parameters in NumPy.  Numerical
roots are searched only in the original TRAIN input interval plus ten percent;
neither coupling labels nor TEST observations define the search interval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import brentq, minimize_scalar


SEEDS = (91, 92, 93)
MODEL_KEYS = tuple(f"{endpoint}_seed{seed}" for seed in SEEDS
                   for endpoint in ("adam6000", "selected"))
STATE_KEYS = {"center", "scale", "net.0.weight", "net.0.bias", "net.2.weight",
              "net.2.bias", "net.4.weight", "net.4.bias"}
CLASS_NAMES = {"attracting": "吸引点", "repelling": "排斥点",
               "neutral_or_unresolved": "近中性／未定"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def state_fingerprint(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        value = np.ascontiguousarray(value)
        for token in (key, str(value.dtype), str(value.shape)):
            digest.update(token.encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


class FrozenScalarMap:
    """Exact scalar counterpart of FrozenResidualMap; buffers stay immutable."""

    def __init__(self, state, expected_sha256):
        self.state = {name: np.array(value, copy=True) for name, value in state.items()}
        if set(self.state) != STATE_KEYS:
            raise ValueError("冻结 R 参数不完整")
        width = self.state["net.0.weight"].shape[0]
        shapes = {"center": (1,), "scale": (1,), "net.0.weight": (width, 1),
                  "net.0.bias": (width,), "net.2.weight": (width, width),
                  "net.2.bias": (width,), "net.4.weight": (1, width),
                  "net.4.bias": (1,)}
        if width < 1 or any(value.dtype != np.dtype("float64")
                or value.shape != shapes[name] or not np.isfinite(value).all()
                for name, value in self.state.items()) or self.state["scale"][0] <= 0:
            raise ValueError("冻结 R 必须是一维、有限、float64 且归一化尺度为正")
        if state_fingerprint(self.state) != expected_sha256:
            raise ValueError("冻结 R 参数指纹不一致")
        self.expected_sha256 = expected_sha256
        for value in self.state.values():
            value.setflags(write=False)

    @classmethod
    def restore(cls, path, key, expected_sha256):
        with np.load(path, allow_pickle=False) as archive:
            state = {name[len(key) + 1:]: archive[name].copy()
                     for name in archive.files if name.startswith(key + ".")}
        return cls(state, expected_sha256)

    def forward_and_derivative(self, points):
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 1 or not np.isfinite(points).all():
            raise ValueError("R 输入必须为有限的 [点数,1] 数组")
        s = self.state
        z = (points - s["center"]) / s["scale"]
        # optimize=False intentionally uses NumPy's scalar contractions rather
        # than a host-dependent multithreaded BLAS GEMM for this tiny network.
        def linear(value, weight):
            return np.einsum("ni,ji->nj", value, weight, optimize=False)
        h1 = np.tanh(linear(z, s["net.0.weight"]) + s["net.0.bias"])
        h2 = np.tanh(linear(h1, s["net.2.weight"]) + s["net.2.bias"])
        result = points + s["scale"] * (linear(h2, s["net.4.weight"]) + s["net.4.bias"])
        # Scalar input/output normalization cancels in the derivative.  The
        # leading identity is part of DR, not a continuous-drift Jacobian.
        d1 = (1 - h1 ** 2) * s["net.0.weight"][:, 0]
        d2 = (1 - h2 ** 2) * linear(d1, s["net.2.weight"])
        derivative = 1 + linear(d2, s["net.4.weight"])
        if not np.isfinite(result).all() or not np.isfinite(derivative).all():
            raise FloatingPointError("冻结 R 或完整 DR 非有限")
        return result, derivative[:, :, None]

    def assert_unchanged(self):
        if (state_fingerprint(self.state) != self.expected_sha256
                or any(value.flags.writeable for value in self.state.values())):
            raise RuntimeError("报告期间冻结 R 参数发生变化")


def train_bounds(direct):
    direct = np.asarray(direct, dtype=np.float64)
    if direct.ndim not in (3, 4) or direct.shape[-2:] != (3, 1) or not direct.size:
        raise ValueError("需要原 TRAIN 的三尺度一维编码")
    values = direct[..., :2, :].ravel()
    if not np.isfinite(values).all() or np.ptp(values) <= 0:
        raise ValueError("TRAIN 输入区间必须有限且非退化")
    low, high = float(values.min()), float(values.max())
    width = high - low
    return [low - .1 * width, high + .1 * width], [low, high]


def find_roots(model, bounds, *, grid_points=1025):
    """Strict bounded scalar roots; finite search is not an exhaustivity proof."""
    low, high = map(float, bounds)
    width = high - low
    if not np.isfinite([low, high]).all() or width <= 0 or grid_points < 65:
        raise ValueError("无效的有界标量搜索范围")
    grid = np.linspace(low, high, grid_points)
    mapped, derivative = model.forward_and_derivative(grid[:, None])
    drift = mapped[:, 0] - grid
    scaled = drift / width
    tolerance, deduplication = 1e-10, 1e-7
    receipt = dict(bounds=[low, high], grid_points=grid_points,
        scaled_residual_tolerance=tolerance, normalized_deduplication_tolerance=deduplication,
        full_DR_includes_identity=True, exhaustive_root_count_certified=False,
        stability_rule="离散映射完整 DR 的绝对值；不是 DR-1 的符号")
    if np.max(np.abs(scaled)) <= tolerance and np.max(np.abs(derivative.ravel() - 1)) <= 1e-7:
        receipt.update(nonisolated_or_flat_identity=True,
                       note="整个探测网格近似恒等，不能枚举孤立不动点")
        return [], receipt

    def function(value):
        result, _ = model.forward_and_derivative(np.array([[value]], dtype=np.float64))
        return float((result[0, 0] - value) / width)

    candidates = [float(grid[index]) for index in (0, len(grid) - 1)
                  if abs(scaled[index]) <= tolerance]
    brackets = np.flatnonzero(np.signbit(scaled[:-1]) != np.signbit(scaled[1:]))
    for index in brackets:
        candidates.append(float(brentq(function, grid[index], grid[index + 1],
            xtol=max(width * 1e-13, np.finfo(float).tiny), rtol=1e-14)))
    candidates.extend(float(grid[index]) for index in np.flatnonzero(scaled == 0))
    minima = np.flatnonzero((np.abs(scaled[1:-1]) <= np.abs(scaled[:-2]))
                            & (np.abs(scaled[1:-1]) <= np.abs(scaled[2:]))) + 1
    # Local drift minima additionally probe tangential roots.  The cap avoids
    # near-flat neural maps causing unbounded reporting work.
    minima = sorted(minima, key=lambda index: abs(scaled[index]))[:32]
    for index in minima:
        result = minimize_scalar(lambda value: function(value) ** 2,
            bounds=(grid[index - 1], grid[index + 1]), method="bounded",
            options={"xatol": width * 1e-12, "maxiter": 100})
        if abs(function(result.x)) <= tolerance:
            candidates.append(float(result.x))
    roots = []
    for value in sorted(candidates):
        residual = abs(function(value))
        if residual > tolerance or any(abs(value - row["g"]) / width <= deduplication for row in roots):
            continue
        mapped, full_dr = model.forward_and_derivative(np.array([[value]]))
        lam = float(full_dr[0, 0, 0])
        h = width * 1e-5
        sides, _ = model.forward_and_derivative(np.array([[value - h], [value + h]]))
        fd = float((sides[1, 0] - sides[0, 0]) / (2 * h))
        if not np.isclose(fd, lam, rtol=1e-5, atol=1e-6):
            raise RuntimeError("不动点完整 DR 的中心差分核对失败")
        classification = ("attracting" if abs(lam) < 1 - 1e-5 else
                          "repelling" if abs(lam) > 1 + 1e-5 else "neutral_or_unresolved")
        roots.append(dict(g=float(value), residual=float(abs(mapped[0, 0] - value)),
            scaled_residual=float(residual), full_DR=lam, eigenvalue=lam,
            eigenvalue_modulus=abs(lam), classification=classification,
            finite_difference_DR=fd, locally_nonsingular=bool(abs(lam - 1) > 1e-7),
            physical_critical_point_certified=False, critical_r=None, thermal_nu=None))
    receipt.update(sign_change_brackets=int(len(brackets)), local_minima_probed=int(len(minima)),
                   nonisolated_or_flat_identity=False, accepted_roots=len(roots))
    return roots, receipt


def verified_public(root, required, pins):
    root = Path(root).resolve()
    status_path, inventory_path = root / "status.json", root / "public_files.json"
    status = json.loads(status_path.read_text())
    if (status.get("status") != "complete" or status.get("workflow_complete") is not True
            or status.get("returncode", 0) != 0 or status.get("error") is not None):
        raise ValueError("仅允许读取成功完成的公开输入")
    inventory = json.loads(inventory_path.read_text())
    for name in required:
        path = root / name
        if path.suffix not in (".json", ".npz") or sha256(path) != inventory.get(name):
            raise ValueError(f"公开输入 SHA 不一致：{name}")
        pins[str(path)] = inventory[name]
    for path in (status_path, inventory_path):
        pins[str(path)] = sha256(path)


def _direct(value):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim == 3:
        value = value[None]
    if value.ndim != 4 or value.shape[-2:] != (3, 1) or not value.size or not np.isfinite(value).all():
        raise ValueError("公开编码必须为有限的 [重复,物理链,3,1]")
    return value


def check_cached_predictions(model, arrays, key):
    difference = 0.
    for split in ("fit", "validation", "test"):
        direct = _direct(arrays[f"{split}_direct"])
        expected = np.asarray(arrays[f"eval_{key}_{split}_one_step"])
        actual, _ = model.forward_and_derivative(direct[..., :2, :].reshape(-1, 1))
        actual = actual.reshape(direct.shape[:-2] + (2, 1))
        if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=1e-10, atol=1e-12):
            raise ValueError("NumPy 冻结 R 与已保存的一步预测不一致")
        difference = max(difference, float(np.max(np.abs(actual - expected))))
    return difference


def draw(output, arrays, cache_arrays, records, models, roots, bounds):
    from scripts.report_phi4_qcr import _plotting
    plt = _plotting(None)
    grid = np.linspace(*bounds, 512)[:, None]
    source_g = np.asarray(cache_arrays["flow_g"]).reshape(-1)
    source_r = np.asarray(cache_arrays["source_flow_R"]).reshape(-1)
    order = np.argsort(source_g)
    source_keep = (source_g[order] >= bounds[0]) & (source_g[order] <= bounds[1])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharex=True, sharey=True,
                              constrained_layout=True)
    for ax, seed in zip(axes, SEEDS):
        ax.plot(grid[:, 0], grid[:, 0], ":", color=".3", lw=1, label="恒等映射")
        ax.plot(source_g[order][source_keep], source_r[order][source_keep], color=".65",
                lw=1.5, label="原联合训练 R")
        for endpoint, color, label in (("adam6000", "#4a87a6", "新 R：Adam 6000步"),
                                        ("selected", "#bd6458", "新 R：验证选定端点")):
            key = f"{endpoint}_seed{seed}"
            value, _ = models[key].forward_and_derivative(grid)
            ax.plot(grid[:, 0], value[:, 0], color=color, lw=1.5, label=label)
            for root in roots[key]:
                marker = {"attracting": "o", "repelling": "^", "neutral_or_unresolved": "D"}[root["classification"]]
                ax.scatter(root["g"], root["g"], color=color, marker=marker, s=42, zorder=5)
        for k, color in ((0, "#478e69"), (1, "#947bb0")):
            values = np.concatenate([_direct(arrays[f"{split}_direct"])[..., k, 0].ravel()
                                     for split in ("fit", "validation", "test")])
            ax.scatter(values, np.full_like(values, .025 + .025 * k), s=5, alpha=.35,
                       color=color, transform=ax.get_xaxis_transform(),
                       label=f"输入数据 k={k}" if seed == 91 else None)
        ax.set(title=f"冻结 Q/C；R 种子 {seed}", xlabel="潜变量 $g$", ylabel="$R(g)$", xlim=bounds)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("一维潜变量重整化映射：圆为吸引点，三角为排斥点，菱形为近中性点")
    fig.savefig(output / "R_vs_g.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for k, ax in enumerate(axes):
        for split, label, color, marker in (("fit", "R 拟合条件", "#497fa0", "o"),
                ("validation", "R 验证条件", "#bd765e", "s"), ("test", "原留出条件", "#669e7c", "^")):
            direct = _direct(arrays[f"{split}_direct"])[..., k, 0].mean(0)
            ids, labels = arrays[f"{split}_cell_ids"], np.asarray(arrays[f"{split}_labels"])
            if labels.ndim != 2 or labels.shape[0] != len(direct) or labels.shape[1] < 2:
                raise ValueError("耦合标签必须为每条物理链的 [r,u]")
            if not np.allclose(labels[:, 1], 1., rtol=0, atol=1e-12):
                raise ValueError("本报告只适用固定 u=1 的参数族")
            points = [(float(labels[ids == cell, 0][0]), float(direct[ids == cell].mean()))
                      for cell in np.unique(ids)]
            points = np.array(sorted(points))
            ax.scatter(points[:, 0], points[:, 1], marker=marker, s=28, color=color, label=label)
        ax.set(title=f"直接编码尺度 k={k}", xlabel="裸参数 $r$（仅事后评价）", ylabel="潜变量 $g$")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=9, frameon=False)
    fig.suptitle("固定 u=1：先平均编码重复，再平均同参数物理链；不是新物理采样")
    fig.savefig(output / "coupling_by_scale.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
    for ax, seed in zip(axes, SEEDS):
        for stage, style, offset in (("adam", "-", 0), ("lbfgs", "--", 6000)):
            rows = records[str(seed)]["history"][stage]
            x = [offset + row["step"] for row in rows]
            ax.plot(x, [row["fit_loss"] for row in rows], style, color="#497fa0",
                    label="拟合：逐编码重复目标" if stage == "adam" else None)
            ax.plot(x, [row["validation"]["chain_mean_loss"] for row in rows], style,
                    color="#bd765e", label="验证：先平均重复残差" if stage == "adam" else None)
        ax.axvline(6000, color=".65", lw=.8)
        ax.set(title=f"新 R 种子 {seed}", xlabel="Adam 更新数＋L-BFGS 累计迭代数",
               ylabel="原 $g$ 坐标闭合目标", yscale="log")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("R-only 收敛检查；虚线为 L-BFGS，验证不接收梯度")
    fig.savefig(output / "loss.png", dpi=180)
    plt.close(fig)


def run(validation, cache, output):
    started = time.process_time()
    validation, cache, output = map(lambda value: Path(value).resolve(), (validation, cache, output))
    if output.exists() or any(output == source or source in output.parents for source in (validation, cache)):
        raise FileExistsError("报告必须写入独立新目录，不覆盖训练或缓存")
    pins = {}
    verified_public(validation, ("manifest.json", "summary.json", "history.json", "evaluation_arrays.npz", "R_parameters.npz"), pins)
    verified_public(cache, ("cache.npz", "manifest.json"), pins)
    from scripts import report_phi4_qcr as plotting
    from scripts import report_ising_zh as fonts
    for path in (Path(__file__).resolve(), Path(plotting.__file__).resolve(), Path(fonts.__file__).resolve()):
        pins[str(path)] = sha256(path)
    summary = json.loads((validation / "summary.json").read_text())
    validation_manifest = json.loads((validation / "manifest.json").read_text())
    cache_manifest = json.loads((cache / "manifest.json").read_text())
    if (validation_manifest.get("cache_manifest") != cache_manifest
            or validation_manifest.get("source_state_sha256") != cache_manifest.get("source_state_sha256")
            or validation_manifest.get("source_module_sha256") != cache_manifest.get("source_module_sha256")
            or validation_manifest.get("theory_dim") != 1
            or summary.get("all_seeds_selected_before_TEST_and_labels_loaded") is not True
            or summary.get("selected_parameter_keys") != [f"selected_seed{s}" for s in SEEDS]):
        raise ValueError("R 验证端点与冻结缓存来源或选点顺序不一致")
    records = json.loads((validation / "history.json").read_text())
    with np.load(validation / "evaluation_arrays.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    with np.load(cache / "cache.npz", allow_pickle=False) as archive:
        cache_arrays = {name: archive[name].copy() for name in
            ("eval_train_direct", "flow_g", "source_flow_R", "source_flow_DR")}
    # The new repeated 64-field encodings have sampling noise of their own.
    # Keep the immutable source 128-field TRAIN range, not their sample extrema.
    observed = np.asarray(cache_manifest.get("train_input_bounds"), dtype=np.float64)
    if (observed.shape != (2,) or not np.isfinite(observed).all() or observed[1] <= observed[0]
            or cache_manifest.get("flow_padding_fraction") != .1):
        raise ValueError("需要原128构型 TRAIN 输入区间及预定10%绘图边界")
    padding = .1 * max(float(observed[1] - observed[0]), 1e-6)
    bounds = [float(observed[0] - padding), float(observed[1] + padding)]
    observed = observed.tolist()
    source_g, source_r, source_dr = [np.asarray(cache_arrays[name]) for name in
                                    ("flow_g", "source_flow_R", "source_flow_DR")]
    if (source_g.shape != source_r.shape or source_g.ndim != 2 or source_g.shape[1] != 1
            or source_dr.shape != (len(source_g), 1, 1)
            or any(not np.isfinite(value).all() for value in (source_g, source_r, source_dr))):
        raise ValueError("原 R 的公开标量网格缓存不完整")
    if (not np.all(np.diff(source_g[:, 0]) > 0)
            or not np.allclose(source_g[[0, -1], 0], bounds, rtol=1e-12, atol=1e-14)):
        raise ValueError("原 R 网格边界与固定 TRAIN 区间不一致")
    models, roots, searches, checks = {}, {}, {}, {}
    for key in MODEL_KEYS:
        if time.process_time() - started > 120:
            raise RuntimeError("报告 CPU 预算 120 秒已用尽")
        model = FrozenScalarMap.restore(validation / "R_parameters.npz", key,
                                         summary["final_R_sha256"][key])
        checks[key] = check_cached_predictions(model, arrays, key)
        roots[key], searches[key] = find_roots(model, bounds)
        for root in roots[key]:
            root["inside_original_TRAIN_input_interval"] = bool(observed[0] <= root["g"] <= observed[1])
        models[key] = model
    output.mkdir(parents=True, exist_ok=False)
    draw(output, arrays, cache_arrays, records, models, roots, bounds)
    if time.process_time() - started > 120:
        raise RuntimeError("报告 CPU 预算 120 秒已用尽")
    for model in models.values():
        model.assert_unchanged()
    if any(sha256(path) != digest for path, digest in pins.items()):
        raise RuntimeError("报告期间不可变公开输入发生变化")
    result = dict(schema_version=5, status="complete", workflow_complete=True, returncode=0,
        fixed_u=1., physical_dimension=2, theory_dim=1, R_seeds=list(SEEDS),
        original_TRAIN_input_interval=observed, search_interval=bounds,
        bounds_source="原联合训练保存的128构型 TRAIN 直接编码输入尺度0/1；不是新64构型重复编码极值",
        bounds_use_labels_or_TEST=False, final_R_sha256={key: summary["final_R_sha256"][key] for key in MODEL_KEYS},
        numerical_fixed_points=roots, searches=searches, cached_prediction_max_abs_difference=checks,
        latent_metrics=summary.get("latent_metrics"), closure_scores=summary.get("closure_scores"),
        input_sha256=pins, frozen_parameters_unchanged=True, parameter_gradients_created=False,
        public_NPZ_only=True, Q_or_C_loaded=False, checkpoint_loaded=False,
        new_training_updates=0, new_physical_samples=0, CPU_threads=1, GPU_seconds=0,
        cpu_seconds=time.process_time() - started, critical_r=None, thermal_nu=None,
        physical_fixed_point_certified=False,
        limitations=["固定 u=1 的一维潜变量不保证在粗粒化下闭合。",
            "R 验证独立性仅相对 R 拟合；Q/C 见过原 TRAIN 条件。",
            "三个 R 种子共享一个旧 Q/C 与旧物理样本，不是三个独立 QCR 训练。",
            "数值不动点及完整 DR 稳定性不认证物理临界点、热方向或临界指数。",
            "有限网格和局部搜索不能保证枚举所有不动点；原 R 仅画已缓存曲线。",
            "同链编码重复不是独立物理系综；原采样质量警告继续保留。"])
    write_json(output / "analysis.json", result)
    write_json(output / "status.json", dict(status="complete", workflow_complete=True, returncode=0,
        error=None, requires_user_action=False, analysis_sha256=sha256(output / "analysis.json")))
    write_json(output / "public_files.json", {path.name: sha256(path) for path in output.iterdir() if path.is_file()})
    return result


def main():
    parser = argparse.ArgumentParser(description="固定 u=1、d_g=1 的冻结 R 验证报告")
    parser.add_argument("--validation", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    result = run(arguments.validation, arguments.cache, arguments.output)
    print(json.dumps(dict(status=result["status"], cpu_seconds=result["cpu_seconds"])))


if __name__ == "__main__":
    main()
