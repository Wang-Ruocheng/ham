"""Offline public-array report of the fixed lazy-Wolff/HMC scalar QCR run.

python -m scripts.report_phi4_hybrid_qcr --data DATA --run RUN --output NEW_DIR

Read public JSON and saved evaluation arrays only: no physical NPZ/PT, model,
sampling, training, inference, or new root search. The separate standard phi4
report retains R(g)-g, coupling-by-scale, and loss plots from the same arrays.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts import report_phi4_u02_qcr as base
from scripts.report_gaussian_scalar_qcr import _optional_feature_arrays, _safe_public_json
from scripts.report_phi4_qcr import _json, _plotting, load_public_run, sha256


DATA_SCHEMA = "euclidean_phi4_hmc_embedded_wolff_v1"
SAMPLE_SEMANTICS = "independent_correlated_hybrid_markov_chains"
WORK_FIELDS = (
    "hmc_trajectories", "hybrid_cycles", "force_evaluations", "force_site_evaluations",
    "wolff_cluster_updates", "lazy_cluster_skips", "total_cluster_sites",
    "total_cluster_changed_sites", "total_cluster_neighbor_visits",
    "total_cluster_bond_tests", "total_cluster_activated_bonds",
)
PINNED_SOURCES = {
    "recurrent_rg/data/phi4.py", "recurrent_rg/data/phi4_hybrid.py",
    "recurrent_rg/data/phi4_hybrid_pipeline.py",
}
LIMITATIONS = [
    "二维周期格点欧氏 φ⁴，固定 u=0.2、h=0；每个固定 HMC-Metropolis 周期后以 1/2 概率执行嵌入 Ising Wolff。",
    "惰性簇概率事先固定，生产中不调参；跳过的簇更新仍属于真实 Markov 周期，不删去 HMC 拒绝态。",
    "簇反射是真实采样转移，不是保存后随机全局 Z₂ 增强；人工全局符号翻转为零。",
    "独立的是四条物理链而非链内保存构型；簇更新不保证偶观测量混合或临界去相关。",
    "保留全部 ESS、自相关、接受率、符号变化和热冷初态警告；不因警告重采、换种子或删链，不认证平衡。",
    "r 只用于生成和评价，不监督 Q/C/R；TRAIN/TEST 按完整参数条件分离。",
    "模型与训练目标沿用连续场分支；没有修复其信息代理的有限样本、带宽或平滑偏差。",
    "软局部统计重构和预测不等于完整构型分布或长程物理统计恢复。",
    "误差同时给出绝对 RMSE 和原 TRAIN 同尺度目标总体标准差 NRMSE，不将小绝对值解释成高物理准确率。",
    "R(g)-g、耦合映射和 loss 由同一 run 的标准只读报告保留；本补充报告不搜索新根或挑选热模。",
    "一维 g、单 seed、有限尺寸不保证物理闭合；数值根不认证相变，critical_r 和 thermal_nu 始终为 null。",
]


def _counter(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Actual sampler work {name} must be a nonnegative integer")
    return value


def load_public_data(data):
    root, manifest, status, sidecars, fingerprints = base.load_public_data(
        data, expected_schema=DATA_SCHEMA, sample_semantics=SAMPLE_SEMANTICS)
    options = manifest["options"]
    if (len(options["r_values"]) != 12 or not np.allclose(options["r_values"], np.linspace(-.5, .3, 12), atol=1e-14, rtol=0)
            or options.get("cluster_probability") != .5 or manifest.get("cluster_probability") != .5
            or any(options.get(key) != value for key, value in
                   (("chains", 4), ("samples", 128), ("burn_in", 1024), ("spacing", 8),
                    ("step_size", .08), ("leapfrog_steps", 12), ("seed", 91)))):
        raise ValueError("Hybrid data settings disagree with the fixed u=.2 twelve-r lazy-cluster experiment")
    if (manifest.get("symmetry_augmentation") is not False or manifest.get("artificial_global_sign_flips") != 0
            or manifest.get("production_adaptation") is not False or manifest.get("source_pins_match") is not True
            or set(manifest.get("source_pins", {})) != PINNED_SOURCES):
        raise ValueError("Hybrid provenance must preserve source pins and prohibit adaptation/post-Z2 augmentation")
    if status.get("sample_semantics") != SAMPLE_SEMANTICS:
        raise ValueError("Hybrid execution receipt sample semantics disagree")
    totals = {name: 0 for name in WORK_FIELDS}
    per_chain = []
    cycles = options["burn_in"] + options["samples"] * options["spacing"]
    for sidecar in sidecars:
        if (sidecar.get("cluster_probability") != .5 or sidecar.get("source_pins") != manifest["source_pins"]
                or sidecar.get("symmetry_augmentation") is not False
                or sidecar.get("artificial_global_sign_flips") != 0
                or sidecar.get("production_adaptation") is not False):
            raise ValueError("Hybrid sidecar schedule/source/augmentation provenance disagrees")
        condition_totals = {name: 0 for name in WORK_FIELDS}
        for chain in sidecar["chains"]:
            if (chain.get("schema") != DATA_SCHEMA or chain.get("cluster_probability") != .5
                    or chain.get("cluster_schedule") != "fixed_lazy_(1-p)I+pW_after_each_HMC"
                    or chain.get("hmc_only_control") is not False
                    or chain.get("production_adaptation") is not False
                    or chain.get("symmetry_augmentation") is not False
                    or chain.get("artificial_global_sign_flips") != 0):
                raise ValueError("Hybrid chain must use the fixed lazy kernel without post-Z2 augmentation")
            work = {name: _counter(chain.get(name), name) for name in WORK_FIELDS}
            if (work["hmc_trajectories"] != cycles or work["hybrid_cycles"] != cycles
                    or work["force_evaluations"] != cycles * 13
                    or work["force_site_evaluations"] != cycles * 13 * 36**2
                    or work["wolff_cluster_updates"] + work["lazy_cluster_skips"] != cycles
                    or work["total_cluster_changed_sites"] > work["total_cluster_sites"]
                    or work["total_cluster_neighbor_visits"] != 4 * work["total_cluster_sites"]
                    or work["total_cluster_sites"] > work["wolff_cluster_updates"] * 36**2
                    or work["total_cluster_activated_bonds"] != work["total_cluster_sites"] - work["wolff_cluster_updates"]):
                raise ValueError("Actual HMC/cluster work disagrees with the fixed cycle counters")
            for name in WORK_FIELDS:
                totals[name] += work[name]
                condition_totals[name] += work[name]
            per_chain.append(dict(cell_index=sidecar["cell_index"], chain_index=chain.get("chain_index"),
                initial=chain.get("initial"), actual_work=work,
                mean_production_cluster_fraction=chain.get("mean_production_cluster_fraction"),
                mean_attempted_cluster_fraction=chain.get("mean_attempted_cluster_fraction")))
        if sidecar.get("summary", {}).get("cluster_work") != condition_totals:
            raise ValueError("Saved condition cluster-work totals disagree with chain counters")
    if (manifest.get("actual_work") != totals or status.get("actual_work") != totals
            or manifest.get("total_hmc_trajectories") != totals["hmc_trajectories"]
            or manifest.get("total_force_site_evaluations") != totals["force_site_evaluations"]):
        raise ValueError("Hybrid root actual-work receipts disagree with saved chain counters")
    quality_path = _safe_public_json(root, manifest.get("quality_summary"))
    digest = sha256(quality_path)
    if manifest.get("quality_summary_sha256") != digest:
        raise ValueError("Hybrid quality-summary SHA256 disagrees with manifest")
    quality = _json(quality_path)
    fingerprints[str(quality_path.relative_to(root))] = digest
    if (quality.get("schema") != DATA_SCHEMA or quality.get("sample_semantics") != SAMPLE_SEMANTICS
            or quality.get("status") != "complete" or quality.get("actual_work") != totals):
        raise ValueError("Hybrid public quality-summary identity or work counters disagree")
    if fingerprints != {name: sha256(root / name) for name in fingerprints}:
        raise ValueError("Public hybrid sampler sources changed during reading")
    work = dict(actual_work=totals, per_chain=per_chain, cluster_probability=.5,
        cluster_schedule="fixed_lazy_(1-p)I+pW_after_each_HMC", saved_sample_spacing_unit="hybrid cycles",
        artificial_global_sign_flips=0, production_adaptation=False, symmetry_augmentation=False,
        source_pins=manifest["source_pins"], source_pins_receipts_match=True,
        physical_npz_loaded_for_work_validation=False)
    return root, manifest, status, sidecars, fingerprints, quality, work


def write_report(data, run, output, *, font=None):
    public = load_public_run(run)
    root, manifest, status, sidecars, fingerprints, quality, work = load_public_data(data)
    base.check_data_run_consistency(manifest, sidecars, public, fingerprints)
    if (public.summary.get("actual_steps") != 3000 or public.manifest.get("options", {}).get("seed") != 91
            or public.manifest.get("options", {}).get("steps") != 3000):
        raise ValueError("Expected the independent seed91, 3000-update QCR endpoint")
    available = _optional_feature_arrays(public.arrays)
    output = Path(output).resolve()
    for source in (root, public.root):
        if output == source or source.is_relative_to(output):
            raise ValueError("Output must not contain or replace a source directory")
    if output.exists():
        raise FileExistsError(f"Calibration directory already exists: {output}")
    plt = _plotting(font)
    output.mkdir(parents=True, exist_ok=False)
    calibration = dict(schema_version=1, experiment="phi4_u02_hybrid_scalar_qcr_calibration", theory_dim=1,
        status="complete", actual_steps=3000, lattice_sizes=[36, 12, 4], data_schema=DATA_SCHEMA,
        sample_semantics=SAMPLE_SEMANTICS, labels_used_in_training=False, labels_used_in_report=True,
        normalization="RMSE averages independent physical chains and coordinates; fixed original same-scale TRAIN target population std (ddof=0)",
        sampling_execution=status, sampling_diagnostics=base.sampling_summary(manifest, sidecars, sample_semantics=SAMPLE_SEMANTICS),
        sampling_sidecars=sidecars, sampler_actual_work=work, sampling_quality_summary=quality,
        optimizer_steps=public.summary.get("optimizer_steps"),
        training_source_sha256=public.manifest.get("source_sha256"),
        q_reconstruction=base.reconstruction_metrics(public.arrays, available),
        latent_closure=base.prediction_metrics(public.arrays),
        native_statistic_prediction=base.prediction_metrics(public.arrays, features=True, available=available),
        r_information=base.rank_correlations(public.arrays), equilibrium_certified=False,
        critical_r=None, thermal_nu=None, physical_critical_point_certified=False, thermal_mode_certified=False,
        root_search_performed=False, standard_report_companion="../report: R_vs_g.png, coupling_by_scale.png, loss_comparison.png",
        limitations=LIMITATIONS)
    base.plot_diagnostics(calibration, output, plt, title=
        "二维 φ⁴ QCR：固定 HMC＋惰性嵌入簇采样，u=0.2\n每周期以 1/2 概率簇更新；保留全部警告，不认证平衡或临界指数")
    (output / "calibration.json").write_text(json.dumps(calibration, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    after_data = {name: sha256(root / name) for name in fingerprints}
    after_run = {name: sha256(public.root / name) for name in public.fingerprints}
    if after_data != fingerprints or after_run != public.fingerprints:
        raise ValueError("Public hybrid sources changed during report generation")
    outputs = {name: sha256(output / name) for name in ("calibration.json", "diagnostics.png")}
    provenance = dict(schema_version=1, status="complete", source_data=str(root), source_run=str(public.root),
        sources_sha256_before=dict(data=fingerprints, run=public.fingerprints),
        sources_sha256_after=dict(data=after_data, run=after_run), public_sources_unchanged=True,
        outputs_sha256=outputs, public_files=[*outputs, "report.json", "status.json"],
        model_loaded=False, physical_data_loaded=False, GPU_used=False, inference_performed=False,
        root_search_performed=False, symmetry_augmentation=False, equilibrium_certified=False, thermal_nu=None,
        limitations=LIMITATIONS)
    (output / "report.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "status.json").write_text(json.dumps(dict(status="complete", workflow_complete=True,
        error=None, requires_user_action=False, equilibrium_certified=False, thermal_nu=None), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return provenance, calibration


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--font", type=Path)
    args = parser.parse_args(argv)
    provenance, _ = write_report(args.data, args.run, args.output, font=args.font)
    print(json.dumps(dict(status=provenance["status"], output=str(args.output.resolve())), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
