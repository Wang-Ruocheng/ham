"""Offline public-artifact report for the independent u=0.2 scalar QCR run.

python -m scripts.report_phi4_u02_qcr --data DATA --run RUN --output NEW_DIR

No physical NPZ, checkpoint, model, GPU, sampler, or new root search is used.
HMC warning evidence is preserved, not converted into a training failure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.report_phi4_qcr import _json, _numeric, _plotting, load_public_run, sha256
from scripts.report_gaussian_scalar_qcr import (
    _optional_feature_arrays, _safe_public_json, prediction_metrics,
    rank_correlations, reconstruction_metrics,
)


DATA_SCHEMA = "euclidean_phi4_hmc_v1"
SAMPLE_SEMANTICS = "independent_hmc_chains_with_saved_autocorrelated_configurations"
LIMITATIONS = [
    "二维周期格点欧氏 φ⁴，固定 u=0.2、h=0，r∈[−0.5,0.3]；不是实时路径积分或符号问题。",
    "HMC 构型可能相关；保留接受率、ESS、自相关、符号变化和热冷初态差异，不认证热化。",
    "高接受率不保证跨势阱混合；有限链热冷比较及 ESS 点估计也不是平衡证明。",
    "r 仅用于生成和事后评价；完整参数条件划分 TRAIN/TEST，耦合值不监督 Q、C 或 R。",
    "特征重构及预测只检验原生连续场软局部统计，不代表完整场分布或长程关联恢复。",
    "连续场信息代理的有限样本、带宽及平滑偏差沿用原分支，未在本轮修复或认证。",
    "共享同尺度 TRAIN 目标均值基线不使用 TEST 拟合；NRMSE 使用原 TRAIN 目标总体标准差。",
    "单参数采样族与一维 g 不保证物理 RG 闭合；不强求不动点，也不搜索新数值根。",
    "r 扫描范围不保证覆盖真实临界位置；有限尺寸数值根不能认证物理临界点或热方向。",
    "单 seed 与有限尺寸不足以认证普适指数；critical_r、thermal_nu 始终为 null。",
]


def _snapshot_json(root, relative, fingerprints):
    path = _safe_public_json(root, relative)
    digest = sha256(path)
    value = _json(path)
    if sha256(path) != digest:
        raise ValueError("Public JSON changed while being read")
    fingerprints[str(path.relative_to(root))] = digest
    return value


def load_public_data(data, *, expected_schema=DATA_SCHEMA, sample_semantics=SAMPLE_SEMANTICS):
    """Validate completed HMC identities using public JSON only."""
    root = Path(data).resolve()
    fingerprints = {}
    manifest = _snapshot_json(root, "manifest.json", fingerprints)
    status = _snapshot_json(root, "status.json", fingerprints)
    if manifest.get("schema") != expected_schema or manifest.get("status") != "complete":
        raise ValueError("Expected a completed phi4 HMC data manifest")
    if expected_schema != DATA_SCHEMA and manifest.get("sample_semantics") != sample_semantics:
        raise ValueError("Data sample semantics disagree with the explicitly requested sampler")
    if (status.get("schema") != expected_schema or status.get("status") != "complete"
            or status.get("workflow_complete") is not True or status.get("returncode") != 0
            or status.get("error") is not None or status.get("requires_user_action", False)):
        raise ValueError("HMC data execution receipt is not successfully complete")
    if status.get("manifest_sha256") != fingerprints["manifest.json"]:
        raise ValueError("HMC data status manifest SHA256 disagrees")
    options = manifest.get("options", {})
    r = _numeric(options.get("r_values", []), "data r_values")
    if (r.ndim != 1 or len(r) < 2 or len(np.unique(r)) != len(r)
            or not np.all(np.diff(r) > 0) or not np.isclose(r[0], -.5, atol=1e-12, rtol=0)
            or not np.isclose(r[-1], .3, atol=1e-12, rtol=0)):
        raise ValueError("Expected distinct increasing r values spanning [-0.5, 0.3]")
    if (options.get("u_values") != [.2] or options.get("h") != 0.
            or options.get("dimension") != 2 or options.get("sizes") != [36]):
        raise ValueError("Expected fixed u=0.2, h=0, dimension=2, L=36")
    every, offset = options.get("test_every"), options.get("test_offset")
    if (not isinstance(every, int) or every < 2 or not isinstance(offset, int)
            or offset < 0 or offset >= every):
        raise ValueError("Invalid whole-parameter-cell split")
    sidecars, cells = [], set()
    for shard in manifest.get("shards", []):
        relative = shard.get("sidecar")
        sidecar = _snapshot_json(root, relative, fingerprints)
        key = str(_safe_public_json(root, relative).relative_to(root))
        if shard.get("sidecar_sha256") != fingerprints[key]:
            raise ValueError("HMC sidecar SHA256 disagrees with manifest")
        cell = shard.get("cell_index")
        if not isinstance(cell, int) or not 0 <= cell < len(r) or cell in cells:
            raise ValueError("Duplicate or unexpected HMC parameter cell")
        cells.add(cell)
        split = "test" if cell % every == offset else "train"
        if expected_schema != DATA_SCHEMA and sidecar.get("sample_semantics") != sample_semantics:
            raise ValueError("Sidecar sample semantics disagree with the explicitly requested sampler")
        if (sidecar.get("schema") != expected_schema or sidecar.get("status") != "complete"
                or sidecar.get("cell_index") != cell or sidecar.get("size") != 36
                or sidecar.get("dimension") != 2 or sidecar.get("r") != r[cell]
                or sidecar.get("u") != .2 or sidecar.get("h") != 0.
                or sidecar.get("split") != split or shard.get("split") != split
                or shard.get("size") != 36 or sidecar.get("data_sha256") != shard.get("sha256")):
            raise ValueError("HMC sidecar physical identity or whole-cell split disagrees")
        chains = sidecar.get("chains")
        if (not isinstance(chains, list) or len(chains) != options.get("chains")
                or sidecar.get("independent_chains") != options.get("chains")
                or sidecar.get("samples_per_chain") != options.get("samples")):
            raise ValueError("HMC chain count or saved sample metadata disagrees")
        sidecars.append(sidecar)
    if cells != set(range(len(r))) or {sidecar["split"] for sidecar in sidecars} != {"train", "test"}:
        raise ValueError("Incomplete HMC parameter grid or missing TRAIN/TEST conditions")
    if status.get("completed_conditions") != len(sidecars):
        raise ValueError("Completed HMC condition count disagrees")
    if manifest.get("labels_used_in_training", False) or any(row.get("labels_used_in_training", False) for row in sidecars):
        raise ValueError("Expected HMC labels for generation/evaluation only")
    if fingerprints != {name: sha256(root / name) for name in fingerprints}:
        raise ValueError("Public HMC data sources changed during reading")
    return root, manifest, status, sidecars, fingerprints


def sampling_summary(manifest, sidecars, *, sample_semantics=SAMPLE_SEMANTICS):
    """Keep all supplied warnings; quality concerns do not abort this report."""
    acceptances, ess, lag1, iat, sign_changes = [], [], [], [], []
    warnings, hot_cold = [], []
    for sidecar in sidecars:
        identity = {key: sidecar[key] for key in ("cell_index", "r", "u", "h", "split", "size")}
        for chain in sidecar["chains"]:
            for name, accumulator in (("production_acceptance", acceptances),
                                      ("saved_magnetization_sign_changes", sign_changes)):
                value = chain.get(name)
                if isinstance(value, (float, int)) and np.isfinite(value):
                    accumulator.append(float(value))
            for diagnostic in chain.get("diagnostics", {}).values():
                for name, accumulator in (("ess", ess), ("lag1", lag1), ("iat_saved_intervals", iat)):
                    value = diagnostic.get(name)
                    if isinstance(value, (float, int)) and np.isfinite(value):
                        accumulator.append(float(value))
            for warning in chain.get("warnings", []):
                warnings.append(dict(**identity, chain_index=chain.get("chain_index"),
                    initial=chain.get("initial"), warning=warning, source="saved_chain_warning"))
        comparison = sidecar.get("summary", {}).get("hot_cold_comparison", {})
        hot_cold.append(dict(**identity, comparison=comparison))
        for observable, value in comparison.items():
            if value.get("descriptive_three_se_warning", False):
                warnings.append(dict(**identity, observable=observable,
                    source="saved_hot_cold_three_se_warning",
                    warning="Saved hot/cold mean difference exceeds its descriptive three-SE threshold"))
    def bounds(values):
        return [min(values), max(values)] if values else None
    return dict(equilibrium_certified=False, sample_semantics=sample_semantics,
        production_acceptance_range=bounds(acceptances), ess_range=bounds(ess),
        lag1_range=bounds(lag1), iat_saved_intervals_range=bounds(iat),
        saved_magnetization_sign_changes_range=bounds(sign_changes),
        chains_without_saved_magnetization_sign_change=sum(value == 0 for value in sign_changes),
        total_chains=sum(len(row["chains"]) for row in sidecars),
        declared_warning_count=manifest.get("warning_count"), preserved_warning_count=len(warnings),
        warning_count_matches_manifest=manifest.get("warning_count") == len(warnings),
        warnings=warnings, hot_cold_comparison=hot_cold,
        raw_chain_diagnostics_preserved_in="sampling_sidecars",
        warning_policy="Saved sampling concerns are scientific limitations, not report-generation failures; no automatic resampling or change of seed")


def check_data_run_consistency(manifest, sidecars, public, fingerprints):
    digest = fingerprints["manifest.json"]
    if (public.manifest.get("data_manifest_sha256") != digest
            or public.summary.get("data_manifest_sha256") != digest):
        raise ValueError("Training manifest/summary data manifest SHA256 disagrees with supplied HMC data")
    if public.summary.get("lattice_sizes") != [36, 12, 4]:
        raise ValueError("Expected QCR geometry 36→12→4")
    expected = np.asarray(manifest["options"]["r_values"], dtype=np.float64)
    labels = np.concatenate([public.arrays[f"{split}_labels"] for split in ("train", "test")])
    if (not np.array_equal(np.sort(np.unique(labels[:, 0])), expected)
            or not np.all(labels[:, 1] == .2) or not np.all(labels[:, 2] == 0.)):
        raise ValueError("Run labels disagree with the fixed-u HMC parameter grid")
    rows = {row["r"]: row for row in sidecars}
    for split in ("train", "test"):
        for r in np.unique(public.arrays[f"{split}_labels"][:, 0]):
            if rows[r]["split"] != split:
                raise ValueError("Run split disagrees with whole-cell HMC data split")
            mask = public.arrays[f"{split}_labels"][:, 0] == r
            if int(mask.sum()) != rows[r]["independent_chains"]:
                raise ValueError("Run evaluation omits or duplicates an HMC physical chain")
            if not np.all(public.arrays[f"{split}_cell_ids"][mask] == rows[r]["cell_index"]):
                raise ValueError("Run cell IDs disagree with the whole HMC parameter condition")


def plot_diagnostics(calibration, output, plt, *, title=None):
    """A distinct Chinese plot; the frozen Gaussian report stays unchanged."""
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.2), constrained_layout=True)
    q, ax = calibration["q_reconstruction"], axes[0, 0]
    if q["available"]:
        rows = [row for row in q["rows"] if row["split"] == "test"]
        x = np.arange(len(rows))
        for offset, key, label in ((-.18, "reconstruction", "Q 重构"),
                (.18, "same_scale_train_mean_constant", "原训练集同尺度均值")):
            ax.bar(x + offset, [row[key]["absolute_rmse"] for row in rows], .36, label=label)
        ax.set_xticks(x, [f"尺度 k={row['scale']}" for row in rows])
        ax.legend(fontsize=9)
    else:
        ax.text(.5, .5, "未保存重构数组，不能评价", transform=ax.transAxes, ha="center")
    ax.set(title="测试集 Q 局部统计重构", ylabel="绝对 RMSE")
    for ax, key, panel_title in ((axes[0, 1], "latent_closure", "测试集 R 潜坐标闭合"),
                           (axes[1, 0], "native_statistic_prediction", "测试集原生软局部统计预测")):
        payload = calibration[key]
        if payload["available"]:
            rows = [row for row in payload["rows"] if row["split"] == "test"]
            x = np.arange(len(rows))
            for offset, metric, label in ((-.24, "model", "R 预测"),
                    (0, "identity_source_statistic", "源尺度恒等预测"),
                    (.24, "same_scale_train_mean_constant", "原训练集目标均值")):
                values = [row[metric]["fixed_train_nrmse"] for row in rows]
                ax.bar(x + offset, [np.nan if value is None else value for value in values], .24, label=label)
            ax.set_xticks(x, [("单步 " if row["branch"] == "one_step" else "微观起点 ")
                + f"{row['source_scale']}→{row['target_scale']}" for row in rows], rotation=10, fontsize=9)
            ax.legend(fontsize=8)
        else:
            ax.text(.5, .5, "未保存预测数组，不能评价", transform=ax.transAxes, ha="center")
        ax.set(title=panel_title, ylabel="原训练集目标标准差 NRMSE")
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
    fig.suptitle(title or "二维 φ⁴ QCR 校准：固定 u=0.2、h=0\n保留 HMC 混合警告；不认证热化、相变或临界指数", fontsize=14)
    fig.savefig(output / "diagnostics.png", dpi=180)
    plt.close(fig)


def write_report(data, run, output, *, font=None):
    public = load_public_run(run)
    root, manifest, status, sidecars, fingerprints = load_public_data(data)
    check_data_run_consistency(manifest, sidecars, public, fingerprints)
    available = _optional_feature_arrays(public.arrays)
    output = Path(output).resolve()
    for source in (root, public.root):
        if output == source or source.is_relative_to(output):
            raise ValueError("Output must not contain or replace a source directory")
    if output.exists():
        raise FileExistsError(f"Calibration directory already exists: {output}")
    plt = _plotting(font)
    output.mkdir(parents=True, exist_ok=False)
    calibration = dict(schema_version=1, experiment="phi4_u02_scalar_qcr_calibration", theory_dim=1,
        status="complete", actual_steps=public.summary.get("actual_steps"), lattice_sizes=[36, 12, 4],
        data_schema=DATA_SCHEMA, sample_semantics=SAMPLE_SEMANTICS,
        labels_used_in_training=False, labels_used_in_report=True,
        normalization="RMSE averages physical chains and coordinates; denominator is same-scale original TRAIN target population std (ddof=0)",
        sampling_execution=status, sampling_diagnostics=sampling_summary(manifest, sidecars),
        sampling_sidecars=sidecars, equilibrium_certified=False,
        q_reconstruction=reconstruction_metrics(public.arrays, available),
        latent_closure=prediction_metrics(public.arrays),
        native_statistic_prediction=prediction_metrics(public.arrays, features=True, available=available),
        r_information=rank_correlations(public.arrays), critical_r=None, thermal_nu=None,
        physical_critical_point_certified=False, thermal_mode_certified=False,
        root_search_performed=False, limitations=LIMITATIONS)
    plot_diagnostics(calibration, output, plt)
    (output / "calibration.json").write_text(json.dumps(calibration, ensure_ascii=False, indent=2,
        allow_nan=False) + "\n", encoding="utf-8")
    after_data = {name: sha256(root / name) for name in fingerprints}
    after_run = {name: sha256(public.root / name) for name in public.fingerprints}
    if after_data != fingerprints or after_run != public.fingerprints:
        raise ValueError("Public source artifacts changed during u=0.2 report generation")
    outputs = {name: sha256(output / name) for name in ("calibration.json", "diagnostics.png")}
    report = dict(schema_version=1, status="complete", source_data=str(root), source_run=str(public.root),
        sources_sha256_before=dict(data=fingerprints, run=public.fingerprints),
        sources_sha256_after=dict(data=after_data, run=after_run), public_sources_unchanged=True,
        model_loaded=False, physical_data_loaded=False, GPU_used=False, inference_performed=False,
        root_search_performed=False, equilibrium_certified=False, thermal_nu=None,
        outputs_sha256=outputs, public_files=["calibration.json", "diagnostics.png", "report.json", "status.json"],
        limitations=LIMITATIONS)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "status.json").write_text(json.dumps(dict(status="complete", workflow_complete=True,
        error=None, requires_user_action=False, equilibrium_certified=False, thermal_nu=None),
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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
