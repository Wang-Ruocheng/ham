"""二维 r-u 扫描的离线公共数组报告；不读取模型或物理构型。

Only explicitly named JSON sidecars and the frozen evaluation NPZ are read.
The report does not infer R, interpolate zero crossings, search roots, fit a
phase boundary, or estimate critical exponents. Saved full two-dimensional
roots and Jacobians are retained, including complex and negative spectra.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts import report_phi4_qcr as generic
from scripts.report_gaussian_scalar_qcr import (
    _safe_public_json, _optional_feature_arrays, prediction_metrics,
    reconstruction_metrics,
)
from scripts.report_phi4_rg_reference import ACTION, DATA_SCHEMA, SAMPLE_SEMANTICS, OBSERVABLES
from scripts.report_phi4_u02_qcr import sampling_summary


R_VALUES = np.linspace(-1.55, -.5, 9)
U_VALUES = np.linspace(.5, 1., 5)
REFERENCES = [dict(u=.5, r_critical=-.7212248), dict(u=.7, r_critical=-.9515697),
              dict(u=1., r_critical=-1.2724682)]
REFERENCE_URL = "https://arxiv.org/html/2512.16536v1"
PUBLIC_FILES = ("R_vs_g.png", "R_displacement.png", "coupling_by_scale.png",
                "component_variance.png", "loss_comparison.png", "physics_ru.png",
                "metrics.json", "physics_summary.json", "analysis_arrays.npz",
                "report.json", "status.json")
LIMITATIONS = [
    "二维周期欧氏 φ⁴ 的正测度路径积分；不涉及实时路径积分或符号问题。",
    "r、u 仅用于数据生成和事后评价；整参数条件留出，训练不使用耦合标签。",
    "单微观尺寸 L=36、粗尺度 12、4、单训练种子；有限尺寸变化不等于无限体积临界边界。",
    "文献点是相同作用量的无限体积蒙特卡罗参考，仅绘于裸参数图，不作为标签、拟合边界或编码空间不动点。",
    "临界分界线一般横向不稳定而可沿线流动；临界线上的每个裸参数点不是固定点，二维鞍点也不保证存在于本次窗口。",
    "箭头严格为保存的完整离散 R(g)−g；只展示训练末次评价保存的根，不在报告中求根或做零点插值。",
    "训练凸包仅是弱支持条件，不能排除包内稀疏空洞；局部秩和近邻距离也不认证物理支撑。",
    "主成分、有效秩是表示的样本几何诊断，不是原系统独立耦合数；低有效秩时不能宣称二维物理鞍点。",
    "Q 与 C 同时学习；原生软局部统计重构或闭合不等于完整分布、长程关联、尺寸外泛化或普适指数恢复。",
    "所有尺度使用同一冻结 TRAIN 微观 RMS；NRMSE 使用各目标尺度原 TRAIN 总体标准差，非测试集拟合。",
    "保留全部 ESS、自相关、热冷差异与接受率警告；本报告不认证平衡，也不自动补样。",
    "连续高斯 C 使用全壳平滑信息代理，非离散精确经验 MI 或 InfoNCE；其有限样本与带宽偏差仍在。",
    "R(mean g) 与 mean R(g) 分别保存；统计均值箭头不能替代真实逐链点的 R(g)−g。",
    "完整 DR 的全部实、复、负特征值均保留，不挑选热方向；thermal_nu 与物理临界位置始终为 null。",
]


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _snapshot(root, relative, fingerprints):
    path = _safe_public_json(root, relative)
    digest = generic.sha256(path)
    result = generic._json(path)
    if generic.sha256(path) != digest:
        raise ValueError("Public JSON changed while being read")
    fingerprints[str(path.relative_to(root))] = digest
    return result


def _finite(value, name, nullable=False):
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{name} must be finite numeric data")
    return float(value)


def load_public_data(data):
    """Check the exact 45-condition plan using JSON, never raw field archives."""
    root, fingerprints = Path(data).resolve(), {}
    manifest = _snapshot(root, "manifest.json", fingerprints)
    status = _snapshot(root, "status.json", fingerprints)
    plan = _snapshot(root, "plan.json", fingerprints)
    if (manifest.get("schema") != DATA_SCHEMA or manifest.get("status") != "complete"
            or manifest.get("sample_semantics") != SAMPLE_SEMANTICS
            or status.get("schema") != DATA_SCHEMA or status.get("status") != "complete"
            or status.get("sample_semantics") != SAMPLE_SEMANTICS
            or status.get("workflow_complete") is not True or status.get("returncode") != 0
            or status.get("error") is not None or status.get("requires_user_action", False)
            or status.get("manifest_sha256") != fingerprints["manifest.json"]):
        raise ValueError("Completed hybrid data receipt or manifest SHA256 disagrees")
    if (manifest.get("plan_sha256") != fingerprints["plan.json"]
            or plan.get("action") != ACTION or plan.get("boundary") != "periodic"
            or plan.get("options") != manifest.get("options")):
        raise ValueError("Generation plan, same-action identity, or SHA256 disagrees")
    options = manifest.get("options", {})
    r = generic._numeric(options.get("r_values", []), "r_values")
    u = generic._numeric(options.get("u_values", []), "u_values")
    if (r.shape != (9,) or u.shape != (5,)
            or not np.allclose(r, R_VALUES, rtol=0, atol=1e-13)
            or not np.allclose(u, U_VALUES, rtol=0, atol=1e-13)
            or options.get("dimension") != 2 or options.get("sizes") != [36]
            or options.get("h") != 0. or options.get("chains") != 4
            or options.get("samples") != 128 or options.get("test_every") != 4
            or options.get("test_offset") != 1 or options.get("cluster_probability") != .5):
        raise ValueError("Expected fixed 9x5 r-u grid, L36, four chains, 128 samples and cell%4==1 TEST")
    pins = manifest.get("source_pins")
    if (not isinstance(pins, dict) or not pins or manifest.get("source_pins_match") is not True
            or plan.get("source_pins") != pins or manifest.get("cluster_probability") != .5
            or manifest.get("symmetry_augmentation") is not False
            or manifest.get("artificial_global_sign_flips") != 0
            or manifest.get("production_adaptation") is not False):
        raise ValueError("Fixed sampler schedule or frozen source pins disagree")
    sidecars, seen = [], set()
    for shard in manifest.get("shards", []):
        sidecar = _snapshot(root, shard.get("sidecar"), fingerprints)
        key = str(_safe_public_json(root, shard["sidecar"]).relative_to(root))
        if shard.get("sidecar_sha256") != fingerprints[key]:
            raise ValueError("Sidecar SHA256 disagrees")
        cell = shard.get("cell_index")
        if type(cell) is not int or cell not in range(45) or cell in seen:
            raise ValueError("Duplicate or unexpected parameter cell")
        seen.add(cell)
        split = "test" if cell % 4 == 1 else "train"
        expected = dict(schema=DATA_SCHEMA, sample_semantics=SAMPLE_SEMANTICS, status="complete",
            cell_index=cell, size=36, dimension=2, r=float(r[cell // 5]), u=float(u[cell % 5]),
            h=0., split=split, independent_chains=4, samples_per_chain=128,
            cluster_probability=.5, source_pins=pins, symmetry_augmentation=False,
            artificial_global_sign_flips=0, production_adaptation=False)
        if (any(sidecar.get(name) != value for name, value in expected.items())
                or shard.get("size") != 36 or shard.get("split") != split
                or sidecar.get("data_sha256") != shard.get("sha256")):
            raise ValueError("Physical sidecar identity, grid order, or split disagrees")
        rows = sidecar.get("chains", [])
        if (len(rows) != 4 or sorted(row.get("chain_index", -1) for row in rows) != list(range(4))):
            raise ValueError("Independent chain inventory disagrees")
        for row in rows:
            rate = _finite(row.get("production_acceptance"), "production_acceptance")
            if not 0 <= rate <= 1:
                raise ValueError("Acceptance must lie in [0,1]")
            if set(row.get("diagnostics", {})) != set(OBSERVABLES):
                raise ValueError("Observable inventory disagrees")
            for name, diagnostic in row["diagnostics"].items():
                if diagnostic.get("samples") != 128:
                    raise ValueError("Diagnostic saved sample count disagrees")
                _finite(diagnostic.get("mean"), name)
                _finite(diagnostic.get("ess"), f"{name} ESS", nullable=True)
        sidecars.append(sidecar)
    if seen != set(range(45)) or status.get("completed_conditions") != 45:
        raise ValueError("Incomplete 45-condition grid")
    quality = _snapshot(root, manifest.get("quality_summary"), fingerprints)
    key = str(_safe_public_json(root, manifest["quality_summary"]).relative_to(root))
    if (manifest.get("quality_summary_sha256") != fingerprints[key]
            or quality.get("schema") != DATA_SCHEMA or quality.get("status") != "complete"
            or quality.get("sample_semantics") != SAMPLE_SEMANTICS):
        raise ValueError("Quality summary identity or SHA256 disagrees")
    wrapper = _snapshot(root, "wrapper_receipt.json", fingerprints)
    if (wrapper.get("schema_version") != 5 or wrapper.get("status") != "complete"
            or wrapper.get("workflow_complete") is not True or wrapper.get("returncode") != 0
            or wrapper.get("historical_kernel_unchanged") is not True
            or wrapper.get("historical_work_cap_unchanged") is not True
            or wrapper.get("source_pins_before") != wrapper.get("source_pins_after")
            or not isinstance(wrapper.get("source_pins_before"), dict)
            or len(wrapper["source_pins_before"]) != 4
            or any(wrapper["source_pins_before"].get(name) != digest for name, digest in pins.items())
            or wrapper.get("labels_used_in_training", False)):
        raise ValueError("Authorized r-u wrapper frozen source receipt disagrees")
    if manifest.get("labels_used_in_training", False) or any(row.get("labels_used_in_training", False) for row in sidecars):
        raise ValueError("Coupling-label supervision is forbidden")
    if fingerprints != {name: generic.sha256(root / name) for name in fingerprints}:
        raise ValueError("Public data changed during reading")
    return root, manifest, status, sorted(sidecars, key=lambda row: row["cell_index"]), fingerprints, quality


def _shape(arrays, key, shape):
    if key not in arrays:
        raise ValueError(f"Missing saved public array {key}")
    value = generic._numeric(arrays[key], key)
    if value.shape != shape:
        raise ValueError(f"{key} has wrong shape: {value.shape}, expected {shape}")
    arrays[key] = value
    return value


def load_public_run(run):
    """Strict dg=2 endpoint validation, independent of the scalar validator."""
    root = Path(run).resolve()
    fingerprints = {name: generic.sha256(root / name) for name in generic.INPUT_FILES}
    manifest, summary, history, status = [generic._json(root / name) for name in generic.INPUT_FILES[:4]]
    if (manifest.get("schema_version") != 5 or any(obj.get("recipe") != generic.RECIPE for obj in (manifest, summary, status))
            or summary.get("status") != "complete" or status.get("status") != "complete"
            or status.get("workflow_complete") is not True or status.get("returncode") != 0
            or status.get("error") is not None or status.get("requires_user_action", False)):
        raise ValueError("Expected successfully complete schema5 continuous dg2 training")
    for key, name in (("summary_sha256", "summary.json"), ("evaluation_arrays_sha256", "evaluation_arrays.npz")):
        if status.get(key) != fingerprints[name]:
            raise ValueError(f"Completed {key} disagrees with source SHA256")
    if (manifest.get("options", {}).get("steps") != 3000
            or any(obj.get("actual_steps") != 3000 for obj in (summary, status))
            or any(obj.get("latest_checkpoint_step") != 3000 for obj in (summary, status, history))
            or history.get("latest_step") != 3000):
        raise ValueError("Expected actual/checkpoint/history endpoint 3000")
    rows = generic.history_rows(history)
    if not rows or rows[-1]["step"] != 3000:
        raise ValueError("History does not reach endpoint")
    for name in ("Q", "C", "R"):
        receipt = summary.get("optimizer_steps", {}).get(name, {})
        if receipt.get("min") != 3000 or receipt.get("max") != 3000 or receipt.get("parameters", 0) <= 0:
            raise ValueError(f"{name} Adam clocks disagree")
    if (manifest.get("theory_dim") != 2 or summary.get("theory_dim") != 2
            or manifest.get("lattice_sizes") != [36, 12, 4] or summary.get("lattice_sizes") != [36, 12, 4]
            or manifest.get("hidden") != 96 or manifest.get("dtype") != "float64"
            or manifest.get("block_size") != 3 or manifest.get("stride") != 3
            or manifest.get("closure_objective") != "all_start_anchor"
            or manifest.get("closure_scale_weights") is not None
            or manifest.get("learning_rate") != .001
            or manifest.get("learning_rate_policy") != "constant"):
        raise ValueError("Expected fixed dg2 b3 hidden96 float64 equal all-start recipe")
    scale = _finite(manifest.get("scale"), "frozen TRAIN RMS")
    if scale <= 0 or summary.get("scale") != scale:
        raise ValueError("One common frozen TRAIN microscopic RMS must match")
    if (any(obj.get("labels_used_in_training", False) for obj in (manifest, summary))
            or manifest.get("model_input_members") != ["configurations"]
            or summary.get("thermal_nu") is not None
            or summary.get("physical_critical_point_certified", False)
            or summary.get("frozen_evaluation_verified") is not True):
        raise ValueError("Unsupervised frozen evaluation identity disagrees")
    with np.load(root / "evaluation_arrays.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    permitted = {f"{split}_{key}" for split in ("train", "test") for key in (
        "direct", "direct_full", "one_step", "rollout", "labels", "cell_ids", "feature_target",
        "feature_reconstruction", "one_step_feature_prediction", "rollout_feature_prediction")}
    permitted.update(("grid_cell_ids", "grid_labels", "grid_mean_direct", "grid_R_of_mean", "grid_mean_R",
        "grid_jensen_difference", "grid_displacement", "flow_g", "flow_R", "flow_displacement", "flow_DR",
        "flow_eigenvalues_real", "flow_eigenvalues_imag", "flow_in_train_hull", "flow_bounds", "flow_train_mean",
        "flow_pca_components", "flow_pca_explained_variance_ratio"))
    if set(arrays) - permitted:
        raise ValueError("Unexpected array outside the public evaluation contract")
    for split, count, cells in (("train", 136, 34), ("test", 44, 11)):
        for key, shape in (("direct", (count, 3, 2)), ("direct_full", (count, 3, 2)),
                ("one_step", (count, 2, 2)), ("rollout", (count, 3, 2)), ("labels", (count, 3))):
            _shape(arrays, f"{split}_{key}", shape)
        ids = np.asarray(arrays.get(f"{split}_cell_ids"))
        if ids.shape != (count,) or ids.dtype.kind not in "iu" or len(np.unique(ids)) != cells:
            raise ValueError("Whole-cell physical chain inventory disagrees")
        arrays[f"{split}_cell_ids"] = ids
        if not np.allclose(arrays[f"{split}_rollout"][:, 0], arrays[f"{split}_direct"][:, 0], rtol=1e-10, atol=1e-12):
            raise ValueError("Rollout must start from the same microscopic encoding")
    available = _optional_feature_arrays(arrays)
    if not all(available.values()) or arrays["train_feature_target"].shape[-1] != 105:
        raise ValueError("All 105 native-statistic targets and predictions are required")
    for key in ("flow_g", "flow_R", "flow_displacement"):
        _shape(arrays, key, (1089, 2))
    _shape(arrays, "flow_DR", (1089, 2, 2))
    for key in ("flow_eigenvalues_real", "flow_eigenvalues_imag"):
        _shape(arrays, key, (1089, 2))
    mask = np.asarray(arrays.get("flow_in_train_hull"))
    if mask.shape != (1089,) or mask.dtype.kind != "b":
        raise ValueError("Saved hull mask must be a Boolean grid vector")
    for key, shape in (("flow_bounds", (2, 2)), ("flow_train_mean", (2,)),
            ("flow_pca_components", (2, 2)), ("flow_pca_explained_variance_ratio", (2,))):
        _shape(arrays, key, shape)
    if not np.allclose(arrays["flow_R"] - arrays["flow_g"], arrays["flow_displacement"], rtol=1e-10, atol=1e-12):
        raise ValueError("Flow arrows must equal full R(g)-g")
    for key, shape in (("grid_cell_ids", (45,)), ("grid_labels", (45, 3)),
            ("grid_mean_direct", (45, 3, 2)), ("grid_R_of_mean", (45, 2, 2)),
            ("grid_mean_R", (45, 2, 2)), ("grid_jensen_difference", (45, 2, 2)),
            ("grid_displacement", (45, 2, 2))):
        _shape(arrays, key, shape)
    if (not np.allclose(arrays["grid_mean_R"] - arrays["grid_R_of_mean"], arrays["grid_jensen_difference"])
            or not np.allclose(arrays["grid_R_of_mean"] - arrays["grid_mean_direct"][:, :2], arrays["grid_displacement"])):
        raise ValueError("R(mean g), mean R(g), or saved displacement semantics disagree")
    inputs = arrays["train_direct"][:, :2].reshape(-1, 2)
    if not np.allclose(inputs.mean(axis=0), arrays["flow_train_mean"], atol=1e-12, rtol=1e-10):
        raise ValueError("Flow PCA/hull training support must use TRAIN inputs only")
    _, singular, components = np.linalg.svd(inputs - inputs.mean(axis=0), full_matrices=False)
    variance = singular ** 2
    ratios = variance / variance.sum() if variance.sum() else np.zeros(2)
    basis = arrays["flow_pca_components"]
    covariance = (inputs - inputs.mean(axis=0)).T @ (inputs - inputs.mean(axis=0)) / len(inputs)
    if (not np.allclose(basis @ basis.T, np.eye(2), atol=1e-10)
            or not np.allclose(basis @ covariance @ basis.T, np.diag(variance / len(inputs)), atol=1e-10)
            or not np.allclose(arrays["flow_pca_explained_variance_ratio"], ratios, atol=1e-10)):
        raise ValueError("Saved PCA is not the TRAIN-input covariance eigensystem")
    bounds = arrays["flow_bounds"]
    expected_grid = np.stack(np.meshgrid(*[np.linspace(*bound, 33) for bound in bounds], indexing="xy"), axis=-1).reshape(-1, 2)
    if np.any(bounds[:, 1] <= bounds[:, 0]) or not np.allclose(arrays["flow_g"], expected_grid, atol=1e-12, rtol=1e-10):
        raise ValueError("Saved flow coordinates must be the exact rectangular 33x33 grid")
    grid_eigenvalues = np.linalg.eigvals(arrays["flow_DR"])
    saved_eigenvalues = arrays["flow_eigenvalues_real"] + 1j * arrays["flow_eigenvalues_imag"]
    spectral_error = np.minimum(np.max(np.abs(grid_eigenvalues - saved_eigenvalues), axis=1),
                                np.max(np.abs(grid_eigenvalues - saved_eigenvalues[:, ::-1]), axis=1))
    if np.any(spectral_error > 1e-8):
        raise ValueError("Saved grid eigenvalues disagree with the full DR")
    for root_row in summary.get("numerical_fixed_points", []):
        point = generic._numeric(root_row.get("g"), "saved root g")
        jacobian = generic._numeric(root_row.get("full_DR"), "saved full DR")
        spectra = root_row.get("eigenvalues", [])
        if (point.shape != (2,) or jacobian.shape != (2, 2) or len(spectra) != 2
                or root_row.get("thermal_nu") is not None or root_row.get("physical_critical_point_certified", False)):
            raise ValueError("Saved full two-dimensional root contract disagrees")
        values = []
        for eig in spectra:
            real, imag, modulus = [_finite(eig.get(name), f"eigenvalue {name}") for name in ("real", "imag", "modulus")]
            if not np.isclose(modulus, np.hypot(real, imag)):
                raise ValueError("Saved eigenvalue modulus disagrees")
            values.append(complex(real, imag))
        expected = np.linalg.eigvals(jacobian)
        if not min(np.max(np.abs(expected - np.asarray(values))), np.max(np.abs(expected - np.asarray(values)[::-1]))) < 1e-8:
            raise ValueError("Saved eigenvalues must belong to full DR, including the identity")
    if fingerprints != {name: generic.sha256(root / name) for name in generic.INPUT_FILES}:
        raise ValueError("Public run changed during reading")
    return generic.PublicRun(root, manifest, summary, history, status, arrays, fingerprints)


def check_data_run_consistency(manifest, sidecars, public, fingerprints):
    digest = fingerprints["manifest.json"]
    if public.manifest.get("data_manifest_sha256") != digest or public.summary.get("data_manifest_sha256") != digest:
        raise ValueError("Run/data manifest SHA256 disagrees")
    a = public.arrays
    for split in ("train", "test"):
        expected = [row for row in sidecars if row["split"] == split]
        ids, labels = a[f"{split}_cell_ids"], a[f"{split}_labels"]
        if set(ids.tolist()) != {row["cell_index"] for row in expected}:
            raise ValueError("Run/data whole-cell split disagrees")
        for row in expected:
            selected = labels[ids == row["cell_index"]]
            if selected.shape != (4, 3) or not np.all(selected == [row["r"], row["u"], 0.]):
                raise ValueError("Run labels changed or duplicated a physical chain condition")
    direct = np.concatenate((a["train_direct"], a["test_direct"]))
    ids = np.concatenate((a["train_cell_ids"], a["test_cell_ids"]))
    predicted = np.concatenate((a["train_one_step"], a["test_one_step"]))
    if not np.array_equal(a["grid_cell_ids"], np.arange(45)):
        raise ValueError("Saved condition grid must retain all 45 conditions")
    for cell in range(45):
        if (not np.allclose(a["grid_mean_direct"][cell], direct[ids == cell].mean(axis=0))
                or not np.allclose(a["grid_mean_R"][cell], predicted[ids == cell].mean(axis=0))
                or not np.allclose(a["grid_labels"][cell], [sidecars[cell]["r"], sidecars[cell]["u"], 0.])):
            raise ValueError("Saved cell means or mean R(g) disagree with actual chain arrays")


def pca_diagnostics(arrays):
    """TRAIN-only SVD; project TEST without refitting, never infer coupling count."""
    rows, saved = [], {}
    for label, source in [(f"k{scale}", arrays["train_direct"][:, scale]) for scale in range(3)] + [
            ("all_inputs", arrays["train_direct"][:, :2].reshape(-1, 2))]:
        mean = source.mean(axis=0)
        _, singular, components = np.linalg.svd(source - mean, full_matrices=False)
        variance = singular ** 2 / len(source)
        total = float(variance.sum())
        ratio = variance / total if total > 0 else np.zeros(2)
        effective = float(np.exp(-np.sum(ratio[ratio > 0] * np.log(ratio[ratio > 0])))) if total else 0.
        rank = int(np.sum(singular > (singular[0] * .01 if singular[0] else 0)))
        rows.append(dict(support=label, fit_split="train", mean=mean.tolist(), components=components.tolist(),
            variance=variance.tolist(), explained_variance_ratio=ratio.tolist(), entropy_effective_rank=effective,
            relative_1percent_rank=rank, numerical_matrix_rank=int(np.linalg.matrix_rank(source - mean)),
            coupling_dimension_certified=False))
        saved[f"pca_{label}_mean"] = mean
        saved[f"pca_{label}_components"] = components
        saved[f"pca_{label}_variance_ratio"] = ratio
        for split in ("train", "test"):
            values = arrays[f"{split}_direct"][:, int(label[1:])] if label.startswith("k") else arrays[f"{split}_direct"][:, :2].reshape(-1, 2)
            saved[f"pca_{label}_{split}_projection"] = (values - mean) @ components.T
    return rows, saved


def dense_support_diagnostics(public):
    """Geometric density context only; no model, labels or new flow roots."""
    inputs = public.arrays["train_direct"][:, :2].reshape(-1, 2)
    center, scale = inputs.mean(axis=0), inputs.std(axis=0, ddof=0)
    scale = np.where(scale > 0, scale, 1.)
    normalized = (inputs - center) / scale
    pairwise = np.linalg.norm(normalized[:, None] - normalized[None], axis=-1)
    np.fill_diagonal(pairwise, np.inf)
    fifth = np.partition(pairwise, 4, axis=1)[:, 4]
    reference = float(np.median(fifth))
    rows = []
    for index, root in enumerate(public.summary.get("numerical_fixed_points", []), 1):
        distance = np.linalg.norm(normalized - (np.asarray(root["g"]) - center) / scale, axis=1)
        closest = np.argsort(distance, kind="stable")[:16]
        local = normalized[closest]
        singular = np.linalg.svd(local - local.mean(axis=0), compute_uv=False)
        rank = int(np.sum(singular > (singular[0] * .01 if singular[0] else 0)))
        rows.append(dict(saved_root_index=index, nearest_standardized_distance=float(distance.min()),
            fifth_neighbor_standardized_distance=float(np.sort(distance)[4]),
            fifth_neighbor_to_train_typical_ratio=float(np.sort(distance)[4] / reference) if reference > 0 else None,
            local_16_relative_1percent_rank=rank,
            local_singular_ratio=float(singular[-1] / singular[0]) if singular[0] else 0.,
            physical_support_certified=False))
    return dict(fit_split="train", coordinate_scale=scale.tolist(),
        median_train_fifth_neighbor_distance=reference,
        train_fifth_neighbor_distance_quantiles=np.quantile(fifth, [.05, .5, .95]).tolist(),
        roots=rows, automatic_density_threshold_or_root_selection=False,
        qualification="Density diagnostics are descriptive, not a calibrated support test; no roots are discarded.")


def physics_summary(manifest, sidecars, quality):
    rows = []
    for sidecar in sidecars:
        means = {name: float(np.mean([chain["diagnostics"][name]["mean"] for chain in sidecar["chains"]])) for name in OBSERVABLES}
        supplied = sidecar.get("summary", {})
        ess = [chain["diagnostics"][name]["ess"] for chain in sidecar["chains"] for name in OBSERVABLES]
        valid = [value for value in ess if value is not None]
        rows.append(dict(cell_index=sidecar["cell_index"], r=sidecar["r"], u=sidecar["u"], split=sidecar["split"],
            observable_means=means, binder_1_minus_m4_over_3m2sq=_finite(supplied.get("binder_1_minus_m4_over_3m2sq"), "Binder", nullable=True),
            susceptibility_connected=_finite(supplied.get("susceptibility_connected"), "connected susceptibility", nullable=True),
            susceptibility_absolute_subtraction_per_site=36 ** 2 * (means["magnetization_squared"] - means["absolute_magnetization"] ** 2),
            minimum_saved_ess=min(valid) if valid else None,
            minimum_production_acceptance=min(chain["production_acceptance"] for chain in sidecar["chains"]),
            hot_cold_comparison=supplied.get("hot_cold_comparison", {})))
    return dict(rows=rows, reference_points=REFERENCES, reference_url=REFERENCE_URL,
        reference_description="同作用量无限体积蒙特卡罗参考，仅用于事后裸参数图",
        equilibrium_certified=False, fitted_transition_location=None, thermal_nu=None,
        absolute_subtraction_is_h0_linear_response=False, error_bars_available=False,
        sampling_diagnostics=sampling_summary(manifest, sidecars, sample_semantics=SAMPLE_SEMANTICS),
        sampling_quality_summary=quality, sampling_sidecars=sidecars, limitations=LIMITATIONS)


def _hull_lines(points):
    """Deterministic geometric convex hull, not a phase boundary or root search."""
    values = sorted(set(map(tuple, np.asarray(points).tolist())))
    if len(values) < 3:
        return np.asarray(values)
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for point in values:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    for point in reversed(values):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1] + [lower[0]])


def _roots(ax, public):
    input_rank = pca_diagnostics(public.arrays)[0][-1]["relative_1percent_rank"]
    for index, root in enumerate(public.summary.get("numerical_fixed_points", []), 1):
        point = root["g"]
        supported = root.get("within_train_hull", False) and root.get("local_rank", 0) >= 2 and input_rank >= 2
        label = f"保存数值根 {index}（{'包内局部二维' if supported else '支撑不足'}）"
        ax.scatter(*point, marker="*" if supported else "x", s=110, color="#b02060", zorder=8, label=label)
        kind = {"saddle": "鞍型线性化", "attracting": "吸引型线性化", "repelling": "排斥型线性化"}.get(root.get("classification"), "数值型态见报告")
        if input_rank < 2:
            kind = "表示支撑近一维，未认证二维型态"
        center = public.arrays["flow_bounds"].mean(axis=1)
        right, above = point[0] > center[0], point[1] > center[1]
        ax.annotate(f"{index}：{kind}", point, xytext=(-5 if right else 5, -10 if above else 7),
            textcoords="offset points", ha="right" if right else "left", va="top" if above else "bottom", fontsize=8)


def plot_flow(public, output, plt):
    a = public.arrays
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.7), constrained_layout=True)
    for ax in axes:
        hull = _hull_lines(a["train_direct"][:, :2].reshape(-1, 2))
        if len(hull) > 1:
            ax.fill(hull[:, 0], hull[:, 1], color="#b9d5e9", alpha=.17, label="训练输入凸包（弱支持）")
        for split, marker, color, label in (("train", "o", "#234f9c", "训练物理链"), ("test", "^", "#e47e30", "留出物理链")):
            for scale in range(2):
                points = a[f"{split}_direct"][:, scale]
                ax.scatter(points[:, 0], points[:, 1], marker=marker, s=8, alpha=.25, color=color, label=label if scale == 0 else None)
        _roots(ax, public)
        ax.set(xlabel=r"$g_1$", ylabel=r"$g_2$")
        ax.grid(alpha=.2)
    grid = a["flow_g"].reshape(33, 33, 2)[::2, ::2].reshape(-1, 2)
    drift = a["flow_displacement"].reshape(33, 33, 2)[::2, ::2].reshape(-1, 2)
    mask = a["flow_in_train_hull"].reshape(33, 33)[::2, ::2].reshape(-1)
    for selected, color, label in ((mask, "#234f9c", "凸包内保存网格"), (~mask, ".7", "凸包外仅数值延拓")):
        axes[0].quiver(grid[selected, 0], grid[selected, 1], drift[selected, 0], drift[selected, 1],
            angles="xy", scale_units="xy", scale=1, width=.003, color=color, alpha=.75, label=label)
    axes[0].set_title("保存二维网格：箭头为真实 R(g)−g")
    for scale, color in ((0, "#23629c"), (1, "#b35430")):
        for split, label in (("train", "训练"), ("test", "留出")):
            g = a[f"{split}_direct"][:, scale]
            displacement = a[f"{split}_one_step"][:, scale] - g
            axes[1].quiver(g[:, 0], g[:, 1], displacement[:, 0], displacement[:, 1],
                angles="xy", scale_units="xy", scale=1, width=.003, color=color, alpha=.5,
                label=f"{label}逐链 k={scale} 的 R(g)−g")
    axes[1].set_title("真实逐链输入：未以条件均值替代")
    for ax in axes:
        ax.legend(fontsize=7, loc="best")
    fig.suptitle("二维 φ⁴：共享离散流场；保存数值根不认证物理临界点")
    fig.savefig(output / "R_vs_g.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
    for scale, ax in enumerate(axes):
        g = a["grid_mean_direct"][:, scale]
        for prediction, label, color in ((a["grid_R_of_mean"][:, scale], "R(mean g)−mean g", "#234f9c"),
                (a["grid_mean_R"][:, scale], "mean R(g)−mean g", "#c46a30")):
            drift = prediction - g
            ax.quiver(g[:, 0], g[:, 1], drift[:, 0], drift[:, 1], angles="xy", scale_units="xy", scale=1,
                width=.004, color=color, alpha=.75, label=label)
        ax.scatter(g[:, 0], g[:, 1], c=a["grid_labels"][:, 1], cmap="viridis", s=18)
        ax.set(xlabel=r"$g_1$", ylabel=r"$g_2$", title=f"条件均值诊断：输入尺度 k={scale}")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("两个不同的非线性均值映射：不混入逐链闭合误差")
    fig.savefig(output / "R_displacement.png", dpi=180)
    plt.close(fig)


def plot_couplings(public, output, plt):
    a = public.arrays
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8), constrained_layout=True)
    for component in range(2):
        for scale in range(3):
            ax = axes[component, scale]
            points = ax.scatter(a["grid_labels"][:, 0], a["grid_labels"][:, 1],
                c=a["grid_mean_direct"][:, scale, component], cmap="coolwarm", s=85, marker="s")
            held = a["grid_cell_ids"].astype(int) % 4 == 1
            ax.scatter(a["grid_labels"][held, 0], a["grid_labels"][held, 1], s=120,
                facecolors="none", edgecolors="black", label="整条件留出")
            fig.colorbar(points, ax=ax, label=rf"条件均值 $g_{component + 1}$")
            ax.set(xlabel="生成参数 r（仅评价）", ylabel="生成参数 u（仅评价）",
                title=f"尺度 k={scale}，L={[36, 12, 4][scale]}；成分 {component + 1}")
            ax.legend(fontsize=7)
    fig.suptitle("裸参数到表示的映射：共同冻结 TRAIN 微观 RMS，无标签监督")
    fig.savefig(output / "coupling_by_scale.png", dpi=180)
    plt.close(fig)


def plot_pca(rows, arrays, output, plt):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    names = ["微观 k=0", "粗尺度 k=1", "粗尺度 k=2", "全部训练输入"]
    values = np.array([row["explained_variance_ratio"] for row in rows])
    for component, offset in ((0, -.18), (1, .18)):
        axes[0].bar(np.arange(4) + offset, values[:, component], .36, label=f"主成分 {component + 1}")
    axes[0].set(xticks=np.arange(4), xticklabels=names, ylabel="训练表示方差占比", title="仅在 TRAIN 上拟合主成分")
    axes[0].legend()
    for split, label, color in (("train", "训练物理链", "#234f9c"), ("test", "留出物理链", "#e47e30")):
        points = arrays[f"pca_all_inputs_{split}_projection"]
        axes[1].scatter(points[:, 0], points[:, 1], color=color, alpha=.35, s=10, label=label)
    axes[1].set(xlabel="训练主成分 1 投影", ylabel="训练主成分 2 投影", title="同一训练基底投影：不重新拟合 TEST")
    axes[1].legend()
    fig.suptitle("表示几何而非耦合数认证；没有将投影零点认作全维不动点")
    fig.savefig(output / "component_variance.png", dpi=180)
    plt.close(fig)


def plot_physics(physics, output, plt):
    rows = physics["rows"]
    fig, axes = plt.subplots(1, 3, figsize=(14.1, 4.8), constrained_layout=True)
    panels = [("平均绝对磁化", lambda row: row["observable_means"]["absolute_magnetization"]),
              ("有限尺寸宾德比", lambda row: row["binder_1_minus_m4_over_3m2sq"]),
              ("绝对磁化扣除量（非零场线性响应）", lambda row: row["susceptibility_absolute_subtraction_per_site"])]
    for ax, (title, getter) in zip(axes, panels):
        points = ax.scatter([row["r"] for row in rows], [row["u"] for row in rows],
            c=[getter(row) if getter(row) is not None else np.nan for row in rows], marker="s", s=105, cmap="viridis")
        fig.colorbar(points, ax=ax, label=title)
        ax.scatter([row["r_critical"] for row in REFERENCES], [row["u"] for row in REFERENCES],
            marker="*", s=120, facecolors="none", edgecolors="#df3b31", label="文献无限体积蒙特卡罗参考")
        ax.set(xlabel="生成参数 r", ylabel="生成参数 u", title=title)
        ax.legend(fontsize=7)
    fig.suptitle("原始微观 L=36 物理描述：仅侧文件点估计，未拟合相边界")
    fig.savefig(output / "physics_ru.png", dpi=180)
    plt.close(fig)


def return_files(output):
    root = Path(output).resolve()
    names = sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file())
    if any(Path(name).suffix not in (".json", ".npz", ".png") for name in names):
        raise ValueError("Report contains nonpublic output")
    return names


def write_report(data, run, output, *, font=None):
    data_root, manifest, _, sidecars, data_fingerprints, quality = load_public_data(data)
    public = load_public_run(run)
    check_data_run_consistency(manifest, sidecars, public, data_fingerprints)
    output = Path(output).resolve()
    if any(output == root or root.is_relative_to(output) or output.is_relative_to(root) for root in (data_root, public.root)):
        raise ValueError("Report output must not overlap either input")
    if output.exists():
        raise FileExistsError("Report requires a fresh independent directory")
    pca, pca_arrays = pca_diagnostics(public.arrays)
    available = _optional_feature_arrays(public.arrays)
    physics = physics_summary(manifest, sidecars, quality)
    metrics = dict(schema_version=1, experiment="phi4_ru_qcr", theory_dim=2, actual_steps=3000,
        lattice_sizes=[36, 12, 4], labels_used_in_training=False, labels_used_in_report=True,
        scale=public.manifest["scale"], normalization="same frozen TRAIN microscopic RMS across all cells/scales; NRMSE denominator is original TRAIN target population std, ddof=0",
        latent_closure=prediction_metrics(public.arrays),
        native_statistic_prediction=prediction_metrics(public.arrays, features=True, available=available),
        q_reconstruction=reconstruction_metrics(public.arrays, available), pca=pca,
        saved_numerical_roots=public.summary.get("numerical_fixed_points", []),
        saved_flow_analysis=public.summary.get("flow_analysis", {}),
        input_relative_1percent_rank=pca[-1]["relative_1percent_rank"],
        two_dimensional_physical_saddle_certified=False,
        dense_support_diagnostics=dense_support_diagnostics(public),
        dense_support_qualification="TRAIN hull is necessary only; saved local ranks/distances must also be read. A hull may bridge empty regions.",
        condition_mean_jensen_rms=float(np.sqrt(np.mean(public.arrays["grid_jensen_difference"] ** 2))),
        critical_r=None, thermal_nu=None, physical_critical_point_certified=False,
        root_search_performed=False, saved_grid_zero_interpolation_performed=False, limitations=LIMITATIONS)
    plt = generic._plotting(font)
    output.mkdir(parents=True, exist_ok=False)
    plot_flow(public, output, plt)
    plot_couplings(public, output, plt)
    plot_pca(pca, pca_arrays, output, plt)
    generic.plot_losses(public, output, plt)
    plot_physics(physics, output, plt)
    _write(output / "metrics.json", metrics)
    _write(output / "physics_summary.json", physics)
    np.savez_compressed(output / "analysis_arrays.npz", **pca_arrays,
        cell_ids=public.arrays["grid_cell_ids"], labels_ru=public.arrays["grid_labels"][:, :2],
        g_mean_by_scale=public.arrays["grid_mean_direct"], R_of_mean=public.arrays["grid_R_of_mean"],
        mean_R=public.arrays["grid_mean_R"], jensen_difference=public.arrays["grid_jensen_difference"],
        absolute_magnetization=np.asarray([row["observable_means"]["absolute_magnetization"] for row in physics["rows"]]),
        binder=np.asarray([np.nan if row["binder_1_minus_m4_over_3m2sq"] is None else row["binder_1_minus_m4_over_3m2sq"] for row in physics["rows"]]),
        susceptibility_absolute_subtraction_per_site=np.asarray([row["susceptibility_absolute_subtraction_per_site"] for row in physics["rows"]]))
    after_data = {name: generic.sha256(data_root / name) for name in data_fingerprints}
    after_run = {name: generic.sha256(public.root / name) for name in public.fingerprints}
    if after_data != data_fingerprints or after_run != public.fingerprints:
        raise ValueError("Public inputs changed during reporting")
    outputs = {name: generic.sha256(output / name) for name in return_files(output)}
    provenance = dict(schema_version=1, status="complete", source_data=str(data_root), source_run=str(public.root),
        sources_sha256_before=dict(data=data_fingerprints, run=public.fingerprints),
        sources_sha256_after=dict(data=after_data, run=after_run), public_sources_unchanged=True,
        model_loaded=False, physical_data_loaded=False, GPU_used=False, inference_performed=False,
        root_search_performed=False, saved_grid_zero_interpolation_performed=False,
        outputs_sha256=outputs, public_files=sorted(PUBLIC_FILES), return_files=sorted(PUBLIC_FILES),
        thermal_nu=None, limitations=LIMITATIONS)
    _write(output / "report.json", provenance)
    _write(output / "status.json", dict(status="complete", workflow_complete=True, returncode=0, error=None,
        requires_user_action=False, report_sha256=generic.sha256(output / "report.json"), thermal_nu=None))
    if return_files(output) != sorted(PUBLIC_FILES):
        raise ValueError("Public report inventory disagrees")
    return provenance, metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--font", type=Path)
    args = parser.parse_args(argv)
    report, _ = write_report(args.data, args.run, args.output, font=args.font)
    print(json.dumps(dict(status=report["status"], output=str(args.output.resolve())), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
