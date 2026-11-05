"""Read-only, public-array calibration of the independent Gaussian QCR pilot.

python -m scripts.report_gaussian_scalar_qcr --data DATA --run RUN --output NEW_DIR

This report never loads a model/checkpoint or a physical configuration archive.
It does not infer a critical point, exponent, or new flow root. Couplings are
evaluation labels only. Existing phi4 reports provide flow, coupling, and loss
plots; this supplemental report tests reconstruction and closure baselines.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.report_phi4_qcr import _json, _numeric, _plotting, load_public_run, sha256


DATA_SCHEMA = "euclidean_gaussian_fft_v1"
SAMPLE_SEMANTICS = "independent_iid_groups_not_markov_chains"
LIMITATIONS = [
    "u=0、h=0、r>0 的周期有限格点自由高斯场；r=0 的零模不可归一化，r<0 不稳定。",
    "独立高斯组不是马尔可夫链；解析协方差和有限样本波动检验不等于 QCR 物理闭合证明。",
    "r 仅用于数据生成与事后评价，不监督 Q、C 或 R；TRAIN/TEST 按整个参数条件分离。",
    "特征重构及预测仅涉及原连续场软局部统计，不代表完整构型分布、长程关联或生成能力。",
    "连续场 C 的信息代理保留已知有限样本与带宽偏差；本次没有修复或认证该代理。",
    "有限正 r 的单侧数据不认证双侧相变、临界点或有限 g 中的排斥点；不搜索新数值根。",
    "自由场关联长度的理论指数 ν=1/2 仅为背景，不是本次拟合结果；thermal_nu 始终为 null。",
    "一次训练种子、有限尺寸与有限样本的校准结果不能替代多种子和更大尺度外推。",
]


def _safe_public_json(root, relative):
    """Resolve an explicitly referenced JSON, rejecting path escape/binary data."""
    if not isinstance(relative, str) or not relative:
        raise ValueError("Public JSON reference must be a nonempty relative path")
    rel = Path(relative)
    path = (root / rel).resolve()
    if rel.is_absolute() or not path.is_relative_to(root) or path.suffix != ".json":
        raise ValueError("Only in-root public JSON references may be read")
    return path


def load_public_data(data):
    """Read manifest and explicitly public diagnostic JSON, never shard payloads."""
    root = Path(data).resolve()
    manifest = _json(root / "manifest.json")
    fingerprints = {"manifest.json": sha256(root / "manifest.json")}
    if manifest.get("schema") != DATA_SCHEMA or manifest.get("status") != "complete":
        raise ValueError("Expected a completed exact-Gaussian data manifest")
    if manifest.get("sample_semantics") != SAMPLE_SEMANTICS:
        raise ValueError("Gaussian groups must explicitly be independent IID groups")
    options = manifest.get("options", {})
    r = _numeric(options.get("r_values", []), "data r_values")
    if r.ndim != 1 or not len(r) or np.any(r <= 0) or len(np.unique(r)) != len(r):
        raise ValueError("Gaussian data requires distinct positive r values")
    if options.get("u_values", [options.get("u", 0.)]) != [0.] or options.get("h", 0.) != 0.:
        raise ValueError("Gaussian calibration requires u=0 and h=0")
    sidecars = []
    condition_diagnostics = []
    for shard in manifest.get("shards", []):
        path = _safe_public_json(root, shard.get("sidecar"))
        key = str(path.relative_to(root))
        digest = sha256(path)
        if shard.get("sidecar_sha256") != digest:
            raise ValueError("Data sidecar SHA256 disagrees with manifest")
        fingerprints[key] = digest
        sidecar = _json(path)
        if sidecar.get("status") != "complete":
            raise ValueError("An exact-Gaussian sidecar is not complete")
        sidecars.append(sidecar)
        if "diagnostics" in sidecar:
            diagnostic_file = _safe_public_json(path.parent, sidecar["diagnostics"])
            diagnostic_key = str(diagnostic_file.relative_to(root))
            diagnostic_hash = sha256(diagnostic_file)
            if sidecar.get("diagnostics_sha256") != diagnostic_hash:
                raise ValueError("Gaussian condition diagnostics SHA256 disagrees with sidecar")
            fingerprints[diagnostic_key] = diagnostic_hash
            values = _json(diagnostic_file)
            summary = {key: values[key] for key in ("point_variance", "mean_field", "twice_action_chi_squared", "warnings") if key in values}
            for name in ("covariance_by_displacement", "structure_factor"):
                expected_key, observed_key = f"exact_{name}", f"empirical_{name}"
                if expected_key in values and observed_key in values:
                    expected_value = _numeric(values[expected_key], expected_key)
                    observed_value = _numeric(values[observed_key], observed_key)
                    if observed_value.shape != expected_value.shape:
                        raise ValueError("Analytic and empirical Gaussian statistic shapes disagree")
                    difference = observed_value - expected_value
                    norm = float(np.linalg.norm(expected_value))
                    summary[name] = dict(absolute_rmse=float(np.sqrt(np.mean(difference * difference))),
                        maximum_absolute_error=float(np.max(np.abs(difference))),
                        relative_frobenius_error=float(np.linalg.norm(difference)) / norm if norm > 0 else None)
            condition_diagnostics.append(dict(
                **{key: sidecar[key] for key in ("r", "u", "h", "size", "cell_index", "split") if key in sidecar},
                source=diagnostic_key, diagnostics=summary))
    if not sidecars:
        raise ValueError("Gaussian manifest must reference public sidecars")
    # The exact sampler emits one small root diagnostic JSON. It contains
    # analytic comparisons, not physical configurations or unapproved archives.
    diagnostic_path = root / "diagnostics.json"
    diagnostics = dict(conditions=condition_diagnostics,
                      note="Analytic finite-lattice IID Gaussian comparisons, not model certification or a mixing diagnostic")
    if diagnostic_path.is_file():
        fingerprints["diagnostics.json"] = sha256(diagnostic_path)
        diagnostics["root"] = _json(diagnostic_path)
        expected = manifest.get("diagnostics_sha256")
        if expected is not None and expected != fingerprints["diagnostics.json"]:
            raise ValueError("Gaussian diagnostics SHA256 disagrees with manifest")
    if fingerprints != {key: sha256(root / key) for key in fingerprints}:
        raise ValueError("Public Gaussian data sources changed during reading")
    return root, manifest, sidecars, diagnostics, fingerprints


def _optional_feature_arrays(arrays):
    available = {}
    count = arrays["train_direct"].shape[1]
    dimensions = None
    for suffix, scales in (("feature_target", count), ("feature_reconstruction", count),
                           ("one_step_feature_prediction", count - 1),
                           ("rollout_feature_prediction", count)):
        presence = [f"{split}_{suffix}" in arrays for split in ("train", "test")]
        if any(presence) and not all(presence):
            raise ValueError(f"Both splits must save {suffix}, or neither")
        available[suffix] = all(presence)
        if not all(presence):
            continue
        for split in ("train", "test"):
            key = f"{split}_{suffix}"
            values = _numeric(arrays[key], key)
            if (values.ndim != 3 or values.shape[:2] != (len(arrays[f"{split}_direct"]), scales)
                    or values.shape[-1] <= 0):
                raise ValueError(f"{key} must be [physical group, scale, feature]")
            if dimensions is not None and values.shape[-1] != dimensions:
                raise ValueError("All native-statistic feature dimensions must agree")
            dimensions = values.shape[-1]
            arrays[key] = values
    if any(available.values()) and not available["feature_target"]:
        raise ValueError("Saved feature predictions require saved feature targets")
    return available


def _errors(prediction, target, train_target):
    error = prediction - target
    rmse = float(np.sqrt(np.mean(error * error)))
    denominator = float(np.sqrt(np.mean(np.var(train_target, axis=0, ddof=0))))
    return dict(absolute_rmse=rmse, mean_squared_l2=float(np.mean(np.sum(error * error, axis=-1))),
                train_target_population_std=denominator,
                fixed_train_nrmse=rmse / denominator if denominator > 0 else None,
                number_of_independent_physical_groups=len(target))


def reconstruction_metrics(arrays, available):
    if not available["feature_reconstruction"]:
        return dict(available=False, reason="Public arrays do not contain feature targets and reconstructions", rows=[])
    rows = []
    training = arrays["train_feature_target"]
    for split in ("train", "test"):
        target, prediction = arrays[f"{split}_feature_target"], arrays[f"{split}_feature_reconstruction"]
        for scale in range(target.shape[1]):
            reference = training[:, scale]
            constant = np.broadcast_to(reference.mean(axis=0), target[:, scale].shape)
            model = _errors(prediction[:, scale], target[:, scale], reference)
            baseline = _errors(constant, target[:, scale], reference)
            ratio = (model["absolute_rmse"] / baseline["absolute_rmse"]
                     if baseline["absolute_rmse"] > 0 else None)
            rows.append(dict(split=split, scale=scale, reconstruction=model,
                             same_scale_train_mean_constant=baseline,
                             rmse_ratio_to_constant=ratio,
                             better_than_constant=model["absolute_rmse"] < baseline["absolute_rmse"]))
    return dict(available=True, statistic="native continuous soft local feature vector", rows=rows,
                estimator="target=mean of two half-ensemble features; reconstruction=shared affine decode(mean-half Q)",
                baseline="same-scale TRAIN feature target mean, never fitted on TEST")


def prediction_metrics(arrays, *, features=False, available=None):
    """Evaluate saved predictions only; original TRAIN target std is immutable."""
    if features and not available["feature_target"]:
        return dict(available=False, reason="Public native-statistic targets not saved", rows=[])
    target_suffix = "feature_target" if features else "direct"
    train = arrays[f"train_{target_suffix}"]
    rows = []
    unavailable = []
    for split in ("train", "test"):
        target = arrays[f"{split}_{target_suffix}"]
        for branch in ("one_step", "rollout"):
            prediction_suffix = f"{branch}_feature_prediction" if features else branch
            if features and not available[prediction_suffix]:
                unavailable.append(prediction_suffix)
                continue
            prediction = arrays[f"{split}_{prediction_suffix}"]
            for target_scale in range(1, target.shape[1]):
                source = target_scale - 1 if branch == "one_step" else 0
                predicted = prediction[:, target_scale - 1 if branch == "one_step" else target_scale]
                reference = train[:, target_scale]
                truth = target[:, target_scale]
                baseline = np.broadcast_to(reference.mean(axis=0), truth.shape)
                model = _errors(predicted, truth, reference)
                identity = _errors(target[:, source], truth, reference)
                constant = _errors(baseline, truth, reference)
                rows.append(dict(split=split, branch="one_step" if branch == "one_step" else "microscopic_rollout",
                    source_scale=source, target_scale=target_scale, model=model,
                    identity_source_statistic=identity, same_scale_train_mean_constant=constant,
                    better_than_identity=model["absolute_rmse"] < identity["absolute_rmse"],
                    better_than_constant=model["absolute_rmse"] < constant["absolute_rmse"]))
    return dict(available=bool(rows), rows=rows, unavailable_saved_predictions=sorted(set(unavailable)),
                target="native continuous soft local statistics" if features else "direct g encoding",
                complete_distribution_or_long_range_accuracy_certified=False)


def _ranks(values):
    """Average ranks for exact ties; no stochastic jitter or external dependency."""
    values = np.asarray(values)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = .5 * (start + end - 1)
        start = end
    return ranks


def rank_correlations(arrays):
    """Evaluate r information using cell means; never train/calibrate using r."""
    rows = []
    for split in ("train", "test"):
        cell_ids = arrays[f"{split}_cell_ids"]
        labels = arrays[f"{split}_labels"]
        direct = arrays[f"{split}_direct"]
        cells = np.unique(cell_ids)
        r = np.array([labels[cell_ids == cell, 0][0] for cell in cells])
        for scale in range(direct.shape[1]):
            g = np.array([direct[cell_ids == cell, scale, 0].mean() for cell in cells])
            a, b = _ranks(r), _ranks(g)
            rho = (float(np.corrcoef(a, b)[0, 1]) if len(cells) >= 2 and np.std(a) > 0 and np.std(b) > 0 else None)
            rows.append(dict(split=split, scale=scale, number_of_parameter_cells=len(cells),
                cell_mean_r_g_spearman=rho, absolute_spearman=abs(rho) if rho is not None else None,
                parameter_r=r.tolist(), cell_mean_g=g.tolist()))
    return dict(rows=rows, labels_used_in_training=False,
                note="Signed Spearman of r and per-cell mean g; sign is coordinate-dependent. Monotonicity is not thermal-mode certification.")


def _check_data_run_consistency(manifest, sidecars, public):
    labels = np.concatenate([public.arrays[f"{split}_labels"] for split in ("train", "test")])
    expected = np.sort(np.asarray(manifest["options"]["r_values"], dtype=float))
    if not np.array_equal(np.sort(np.unique(labels[:, 0])), expected) or not np.all(labels[:, 1:] == 0.):
        raise ValueError("Run evaluation labels disagree with Gaussian r>0, u=h=0 data")
    sidecar_rows = {}
    for sidecar in sidecars:
        if any(key not in sidecar for key in ("r", "u", "h", "split")):
            raise ValueError("Gaussian sidecars must identify physical parameters and split")
        key = (float(sidecar["r"]), float(sidecar["u"]), float(sidecar["h"]))
        if key in sidecar_rows and sidecar_rows[key] != sidecar["split"]:
            raise ValueError("Gaussian physical condition appears in both splits")
        sidecar_rows[key] = sidecar["split"]
    for split in ("train", "test"):
        for label in public.arrays[f"{split}_labels"]:
            if sidecar_rows.get(tuple(label)) != split:
                raise ValueError("Run split disagrees with whole-cell Gaussian data split")


def _plot_diagnostics(calibration, output, plt):
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.2), constrained_layout=True)
    q = calibration["q_reconstruction"]
    ax = axes[0, 0]
    if q["available"]:
        rows = [row for row in q["rows"] if row["split"] == "test"]
        x = np.arange(len(rows))
        ax.bar(x - .18, [row["reconstruction"]["absolute_rmse"] for row in rows], .36, label="Q 重构")
        ax.bar(x + .18, [row["same_scale_train_mean_constant"]["absolute_rmse"] for row in rows], .36,
               label="原训练集同尺度均值")
        ax.set_xticks(x, [f"尺度 k={row['scale']}" for row in rows])
        ax.legend(fontsize=9)
    else:
        ax.text(.5, .5, "未保存特征重构数组，不能评价", transform=ax.transAxes, ha="center")
    ax.set(title="测试集 Q 局部统计重构", ylabel="绝对 RMSE")
    for ax, key, title in ((axes[0, 1], "latent_closure", "测试集 R 潜坐标闭合"),
                            (axes[1, 0], "native_statistic_prediction", "测试集原生软局部统计预测")):
        payload = calibration[key]
        if payload["available"]:
            rows = [row for row in payload["rows"] if row["split"] == "test"]
            x = np.arange(len(rows))
            for offset, metric, label in ((-.24, "model", "R 预测"),
                (0, "identity_source_statistic", "源尺度恒等预测"),
                (.24, "same_scale_train_mean_constant", "原训练集目标均值")):
                y = [row[metric]["fixed_train_nrmse"] for row in rows]
                ax.bar(x + offset, [np.nan if v is None else v for v in y], .24, label=label)
            ax.set_xticks(x, [("单步 " if row["branch"] == "one_step" else "微观起点 ")
                             + f"{row['source_scale']}→{row['target_scale']}" for row in rows],
                          rotation=10, fontsize=9)
            ax.legend(fontsize=8)
        else:
            ax.text(.5, .5, "未保存预测数组，不能评价", transform=ax.transAxes, ha="center")
        ax.set(title=title, ylabel="原训练集目标标准差 NRMSE")
    ax = axes[1, 1]
    for split, label, offset in (("train", "训练条件", -.17), ("test", "留出条件", .17)):
        rows = [row for row in calibration["r_information"]["rows"] if row["split"] == split]
        ax.bar(np.arange(len(rows)) + offset,
               [np.nan if row["absolute_spearman"] is None else row["absolute_spearman"] for row in rows],
               .34, label=label)
        ax.set_xticks(np.arange(len(rows)), [f"尺度 k={row['scale']}" for row in rows])
    ax.set(title="条件平均 g 与 r 的单调关联（仅评价）", ylabel="Spearman 相关系数绝对值", ylim=(0, 1.08))
    ax.legend(fontsize=9)
    for ax in axes.ravel():
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("自由高斯场 QCR 校准：独立样本不保证表示与闭合学好\n仅检验已保存数组；不认证相变、不拟合临界指数", fontsize=14)
    fig.savefig(output / "diagnostics.png", dpi=180)
    plt.close(fig)


def write_report(data, run, output, *, font=None):
    public = load_public_run(run)
    root, manifest, sidecars, diagnostics, fingerprints = load_public_data(data)
    data_manifest_sha = fingerprints["manifest.json"]
    if (public.manifest.get("data_manifest_sha256") != data_manifest_sha
            or public.summary.get("data_manifest_sha256") != data_manifest_sha):
        raise ValueError("Training manifest/summary data manifest SHA256 disagrees with the supplied Gaussian data")
    _check_data_run_consistency(manifest, sidecars, public)
    available = _optional_feature_arrays(public.arrays)
    output = Path(output).resolve()
    for source in (root, public.root):
        if output == source or source.is_relative_to(output):
            raise ValueError("Output must not contain or replace a source directory")
    if output.exists():
        raise FileExistsError(f"Calibration directory already exists: {output}")
    plt = _plotting(font)
    output.mkdir(parents=True, exist_ok=False)
    q = reconstruction_metrics(public.arrays, available)
    latent = prediction_metrics(public.arrays)
    physical = prediction_metrics(public.arrays, features=True, available=available)
    calibration = dict(schema_version=1, experiment="gaussian_scalar_qcr_calibration", theory_dim=1,
        status="complete", actual_steps=public.summary.get("actual_steps"),
        lattice_sizes=public.summary.get("lattice_sizes"), data_schema=DATA_SCHEMA,
        sample_semantics=SAMPLE_SEMANTICS, labels_used_in_training=False, labels_used_in_report=True,
        normalization="RMSE averages physical groups and coordinates; denominator is same-scale original TRAIN target population std (ddof=0)",
        sampling_diagnostics=diagnostics, sampling_sidecars=sidecars,
        q_reconstruction=q, latent_closure=latent, native_statistic_prediction=physical,
        r_information=rank_correlations(public.arrays), theoretical_gaussian_nu_background=.5,
        thermal_nu=None, thermal_mode_certified=False, physical_critical_point_certified=False,
        critical_r=None, root_search_performed=False, limitations=LIMITATIONS)
    _plot_diagnostics(calibration, output, plt)
    (output / "calibration.json").write_text(json.dumps(calibration, ensure_ascii=False, indent=2,
        allow_nan=False) + "\n", encoding="utf-8")
    after_data = {name: sha256(root / name) for name in fingerprints}
    after_run = {name: sha256(public.root / name) for name in public.fingerprints}
    if after_data != fingerprints or after_run != public.fingerprints:
        raise ValueError("Public source artifacts changed during Gaussian report generation")
    outputs = {name: sha256(output / name) for name in ("calibration.json", "diagnostics.png")}
    report = dict(schema_version=1, status="complete", source_data=str(root), source_run=str(public.root),
        sources_sha256_before=dict(data=fingerprints, run=public.fingerprints),
        sources_sha256_after=dict(data=after_data, run=after_run), public_sources_unchanged=True,
        model_loaded=False, physical_data_loaded=False, GPU_used=False, inference_performed=False,
        root_search_performed=False, thermal_nu=None, outputs_sha256=outputs,
        public_files=["calibration.json", "diagnostics.png", "report.json", "status.json"], limitations=LIMITATIONS)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "status.json").write_text(json.dumps(dict(status="complete", workflow_complete=True,
        error=None, requires_user_action=False, thermal_nu=None), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report, calibration


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
