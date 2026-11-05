"""同作用量 φ⁴ 参考基准的只读报告，不加载模型或物理构型归档。

python -m scripts.report_phi4_rg_reference --data DATA --run RUN --output NEW_DIR

Only public JSON sidecars and the saved QCR evaluation NPZ are consumed. The
external Monte Carlo critical mass is a reference, not an RG prediction, and
no new root search, interpolation of flow zeros, or exponent fit is performed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts import report_phi4_qcr as generic
from scripts.report_gaussian_scalar_qcr import _safe_public_json
from scripts.report_phi4_u02_qcr import sampling_summary


DATA_SCHEMA = "euclidean_phi4_hmc_embedded_wolff_v1"
SAMPLE_SEMANTICS = "independent_correlated_hybrid_markov_chains"
R_CRITICAL = -1.2724682
R_UNCERTAINTY = 4.8e-6
R_VALUES = R_CRITICAL + np.linspace(-.4, .4, 13)
ACTION = "S=.5*sum_(x,a)(phi[x+e_a]-phi[x])^2+sum_x(.5*r*phi^2+.25*u*phi^4)"
OBSERVABLES = (
    "action_density", "magnetization", "absolute_magnetization",
    "magnetization_squared", "magnetization_fourth", "mean_field_squared",
    "structure_factor_zero", "structure_factor_minimum_momentum",
)
REFERENCE = dict(
    title="Critical coupling in phi_2^4 theory",
    authors=["Stephan Dürr", "Tolga S. H. Kiel"],
    url="https://arxiv.org/html/2512.16536v1", arxiv="2512.16536v1",
    action_location="Eq. (3)", critical_mass_location="Table 2, lambda_hat=1.00",
    r_critical=R_CRITICAL, r_uncertainty=R_UNCERTAINTY,
    published_value="-1.2724682(48)", method="Monte Carlo, infinite-volume extrapolation",
    parameter_identification="r=mu_0_hat^2; u=lambda_hat; h=0",
    lattice_action=ACTION, finite_size_reference=False, RG_bare_mass_prediction=False,
    used_as_training_supervision=False,
)
LIMITATIONS = [
    "文献 r_c 是同一格点作用量在 u=1 的无限体积蒙特卡罗外推参考，不是解析或传统 RG 给出的裸质量预测。",
    "本次只有微观 L=36，粗尺度为 12、4；有限尺寸观测量的变化或峰值不等于无限体积相变位置。",
    "中心参数条件整体留出，r、u、h 仅用于生成和事后评价，不监督 Q、C、R。",
    "侧文件均值、宾德比和磁化率均为相关有限链的描述性点估计，无置信区间或有限尺寸外推。",
    "绝对磁化扣除量 L²(⟨m²⟩−⟨|m|⟩²) 对应文献式 (8) 除以体积，是有限尺寸描述量，不是 h=0 线性响应；它与侧文件的 L²(⟨m²⟩−⟨m⟩²) 分开保存。",
    "保留全部 ESS、自相关、接受率、符号变化和热冷初态差异；高接受率或频繁变号不认证平衡。",
    "连续概率 C 使用全壳高斯平滑信息代理，非有限字母精确经验 MI；其平滑与有限样本偏差仍在。",
    "一维 g、单训练种子、有限尺度闭合不保证完整物理 RG 闭合或临界指数恢复。",
    "仅绘制已保存 R(g) 与 R(g)-g；不运行模型、不搜索新根、不拟合临界参数或指数。",
]
GENERIC_FILES = ("R_vs_g.png", "coupling_by_scale.png", "loss_comparison.png",
                 "metrics.json", "report.json", "status.json")
OWN_FILES = ("physics_vs_r.png", "physics_arrays.npz", "physics_summary.json",
             "report.json", "status.json")
PUBLIC_FILES = tuple(sorted([*OWN_FILES, *(f"generic/{name}" for name in GENERIC_FILES)]))


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def _snapshot(root, relative, fingerprints):
    path = _safe_public_json(root, relative)
    digest = generic.sha256(path)
    value = generic._json(path)
    if generic.sha256(path) != digest:
        raise ValueError("Public JSON changed while being read")
    fingerprints[str(path.relative_to(root))] = digest
    return value


def _finite(value, name, *, nullable=False):
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{name} must be finite numeric data")
    return float(value)


def load_public_data(data):
    """Validate the fixed benchmark using JSON; never open a physical NPZ."""
    root, fingerprints = Path(data).resolve(), {}
    manifest = _snapshot(root, "manifest.json", fingerprints)
    status = _snapshot(root, "status.json", fingerprints)
    plan = _snapshot(root, "plan.json", fingerprints)
    if (manifest.get("schema") != DATA_SCHEMA or manifest.get("status") != "complete"
            or manifest.get("sample_semantics") != SAMPLE_SEMANTICS):
        raise ValueError("Expected completed hybrid phi4 benchmark data")
    if (status.get("status") != "complete" or status.get("schema") != DATA_SCHEMA
            or status.get("sample_semantics") != SAMPLE_SEMANTICS
            or status.get("workflow_complete") is not True or status.get("returncode") != 0
            or status.get("error") is not None or status.get("requires_user_action", False)
            or status.get("manifest_sha256") != fingerprints["manifest.json"]):
        raise ValueError("Data execution receipt or manifest SHA256 disagrees")
    if (manifest.get("plan_sha256") != fingerprints["plan.json"]
            or plan.get("action") != ACTION or plan.get("boundary") != "periodic"
            or plan.get("options") != manifest.get("options")):
        raise ValueError("Same-action generation plan or SHA256 disagrees")
    options = manifest["options"]
    r = generic._numeric(options.get("r_values", []), "data r_values")
    if (r.shape != (13,) or not np.allclose(r, R_VALUES, rtol=0, atol=1e-13)
            or options.get("u_values") != [1.] or options.get("h") != 0.
            or options.get("dimension") != 2 or options.get("sizes") != [36]):
        raise ValueError("Expected 13 uniformly spaced reference-centered r values, u=1, h=0, L=36")
    every, offset = options.get("test_every"), options.get("test_offset")
    if (type(every) is not int or every < 2 or type(offset) is not int or not 0 <= offset < every):
        raise ValueError("Invalid whole-parameter-cell split")
    if 6 % every != offset:
        raise ValueError("Critical-reference condition cell 6 must be held out")
    chains, samples = options.get("chains"), options.get("samples")
    if (type(chains) is not int or chains < 4 or chains % 4 or type(samples) is not int
            or samples < 4 or samples % 2):
        raise ValueError("Invalid independent-chain or saved-sample counts")
    pins = manifest.get("source_pins")
    if (not isinstance(pins, dict) or not pins or manifest.get("source_pins_match") is not True
            or plan.get("source_pins") != pins or options.get("cluster_probability") != .5
            or manifest.get("cluster_probability") != .5
            or manifest.get("symmetry_augmentation") is not False
            or manifest.get("artificial_global_sign_flips") != 0
            or manifest.get("production_adaptation") is not False):
        raise ValueError("Hybrid sampler source or fixed schedule provenance disagrees")
    sidecars, seen = [], set()
    for shard in manifest.get("shards", []):
        relative = shard.get("sidecar")
        sidecar = _snapshot(root, relative, fingerprints)
        key = str(_safe_public_json(root, relative).relative_to(root))
        if shard.get("sidecar_sha256") != fingerprints[key]:
            raise ValueError("Sidecar SHA256 disagrees with manifest")
        cell = shard.get("cell_index")
        if type(cell) is not int or cell not in range(13) or cell in seen:
            raise ValueError("Duplicate or unexpected parameter cell")
        seen.add(cell)
        split = "test" if cell % every == offset else "train"
        expected = dict(schema=DATA_SCHEMA, sample_semantics=SAMPLE_SEMANTICS, status="complete",
                        cell_index=cell, size=36, dimension=2, r=float(r[cell]), u=1., h=0.,
                        split=split, independent_chains=chains, samples_per_chain=samples,
                        cluster_probability=.5, source_pins=pins, symmetry_augmentation=False,
                        artificial_global_sign_flips=0, production_adaptation=False)
        if (any(sidecar.get(name) != value for name, value in expected.items())
                or shard.get("size") != 36 or shard.get("split") != split
                or sidecar.get("data_sha256") != shard.get("sha256")):
            raise ValueError("Sidecar physical identity, whole-cell split, or provenance disagrees")
        rows = sidecar.get("chains")
        if (not isinstance(rows, list) or len(rows) != chains
                or sorted(row.get("chain_index", -1) for row in rows) != list(range(chains))):
            raise ValueError("Independent chain inventory disagrees")
        for row in rows:
            rate = _finite(row.get("production_acceptance"), "production_acceptance")
            if not 0 <= rate <= 1:
                raise ValueError("production_acceptance must be in [0,1]")
            diagnostics = row.get("diagnostics", {})
            if set(diagnostics) != set(OBSERVABLES):
                raise ValueError("Saved observable diagnostic inventory disagrees")
            for name, diagnostic in diagnostics.items():
                if diagnostic.get("samples") != samples:
                    raise ValueError("Diagnostic saved-sample count disagrees")
                _finite(diagnostic.get("mean"), f"{name} mean")
                _finite(diagnostic.get("ess"), f"{name} ESS", nullable=True)
        sidecars.append(sidecar)
    if (seen != set(range(13)) or {row["split"] for row in sidecars} != {"train", "test"}
            or status.get("completed_conditions") != 13):
        raise ValueError("Incomplete parameter grid or data execution condition count")
    if manifest.get("labels_used_in_training", False) or any(row.get("labels_used_in_training", False) for row in sidecars):
        raise ValueError("Coupling labels are for generation/evaluation only")
    quality = _snapshot(root, manifest.get("quality_summary"), fingerprints)
    quality_key = str(_safe_public_json(root, manifest["quality_summary"]).relative_to(root))
    if (manifest.get("quality_summary_sha256") != fingerprints[quality_key]
            or quality.get("schema") != DATA_SCHEMA or quality.get("status") != "complete"
            or quality.get("sample_semantics") != SAMPLE_SEMANTICS):
        raise ValueError("Quality-summary identity or SHA256 disagrees")
    if fingerprints != {name: generic.sha256(root / name) for name in fingerprints}:
        raise ValueError("Public data sources changed while being read")
    return root, manifest, status, sorted(sidecars, key=lambda row: row["cell_index"]), fingerprints, quality


def check_data_run_consistency(manifest, sidecars, public, fingerprints):
    digest = fingerprints["manifest.json"]
    if (public.manifest.get("data_manifest_sha256") != digest
            or public.summary.get("data_manifest_sha256") != digest):
        raise ValueError("Run/data manifest SHA256 disagrees")
    if (public.manifest.get("schema_version") != 5 or public.manifest.get("theory_dim") != 1
            or public.summary.get("lattice_sizes") != [36, 12, 4]
            or public.manifest.get("block_size") != 3 or public.manifest.get("stride") != 3
            or public.manifest.get("options", {}).get("steps") != 3000
            or public.summary.get("actual_steps") != 3000
            or public.manifest.get("closure_objective") != "all_start_anchor"
            or public.manifest.get("closure_scale_weights") is not None
            or public.manifest.get("q_initialization") != "train_mean_bias"):
        raise ValueError("Expected completed b3, dg1, mean-bias, equal all-start, 3000-step benchmark")
    for split in ("train", "test"):
        labels, ids = public.arrays[f"{split}_labels"], public.arrays[f"{split}_cell_ids"]
        expected = [row for row in sidecars if row["split"] == split]
        if set(ids.tolist()) != {row["cell_index"] for row in expected}:
            raise ValueError("Run whole-cell split disagrees with data")
        for row in expected:
            selected = labels[ids == row["cell_index"]]
            if (selected.shape != (manifest["options"]["chains"], 3)
                    or not np.all(selected == np.array([row["r"], 1., 0.]))):
                raise ValueError("Run labels omit, duplicate, or change physical-chain conditions")


def physics_summary(manifest, sidecars, public, quality):
    """Pool equal-length chain means and retain the supplied nonlinear summaries."""
    rows = []
    for sidecar in sidecars:
        chain_means = [{name: chain["diagnostics"][name]["mean"] for name in OBSERVABLES}
                       for chain in sidecar["chains"]]
        means = {name: float(np.mean([row[name] for row in chain_means])) for name in OBSERVABLES}
        supplied = sidecar.get("summary", {})
        scalars = {name: _finite(supplied.get(name), name, nullable=True) for name in (
            "binder_1_minus_m4_over_3m2sq", "susceptibility_connected",
            "structure_factor_zero", "structure_factor_minimum_momentum")}
        ess = [chain["diagnostics"][name]["ess"] for chain in sidecar["chains"] for name in OBSERVABLES]
        valid_ess = [value for value in ess if value is not None]
        split = sidecar["split"]
        g = public.arrays[f"{split}_direct"][public.arrays[f"{split}_cell_ids"] == sidecar["cell_index"], :, 0]
        rows.append(dict(cell_index=sidecar["cell_index"], r=sidecar["r"], u=1., h=0., split=split,
            size=36, critical_reference_condition=sidecar["cell_index"] == 6,
            observable_means=means, chain_observable_means=chain_means, **scalars,
            susceptibility_absolute_subtraction_per_site=36 ** 2 * (
                means["magnetization_squared"] - means["absolute_magnetization"] ** 2),
            minimum_saved_ess=min(valid_ess) if valid_ess else None,
            unavailable_ess_count=sum(value is None for value in ess),
            minimum_production_acceptance=min(chain["production_acceptance"] for chain in sidecar["chains"]),
            g_mean_by_scale=g.mean(axis=0).tolist(), g_chain_population_std_by_scale=g.std(axis=0).tolist(),
            hot_cold_comparison=supplied.get("hot_cold_comparison", {})))
    return dict(schema_version=1, experiment="phi4_same_action_rg_reference", actual_steps=3000,
        lattice_sizes=[36, 12, 4], theory_dim=1, reference=REFERENCE,
        critical_reference_condition_held_out=True, critical_reference_cell=6, rows=rows,
        aggregation="equal-weight means of equal-length independent chains; nonlinear condition summaries copied from sidecars",
        susceptibility_definitions=dict(
            susceptibility_connected="L^2*(mean(m^2)-mean(m)^2); copied from the saved sidecar",
            susceptibility_absolute_subtraction_per_site="L^2*(mean(m^2)-mean(abs(m))^2); paper Eq. (8) divided by lattice volume",
            absolute_subtraction_is_h0_linear_response=False, pseudocritical_fit_performed=False),
        error_bars_available=False, fitted_transition_location=None,
        sampling_diagnostics=sampling_summary(manifest, sidecars, sample_semantics=SAMPLE_SEMANTICS),
        sampling_quality_summary=quality, sampling_sidecars=sidecars,
        labels_used_in_training=False, labels_used_in_report=True,
        equilibrium_certified=False, critical_r=None, thermal_nu=None,
        physical_critical_point_certified=False, root_search_performed=False,
        saved_grid_zero_interpolation_performed=False, limitations=LIMITATIONS)


def _plot_flow(public, output, plt):
    """Plot saved arrays only, without calling generic zero interpolation."""
    g, mapped, displacement = [public.arrays[name] for name in ("flow_g", "flow_R", "flow_displacement")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].plot(g, mapped, label="已保存共享映射 R(g)", color="#234f9c")
    axes[0].plot(g, g, "--", color=".5", label="恒等映射")
    axes[0].set(xlabel="输入 g", ylabel="输出 R(g)", title="完整离散映射")
    axes[1].plot(g, displacement, color="#234f9c", label="已保存位移 R(g)−g")
    indices = np.unique(np.linspace(0, len(g) - 1, min(22, len(g))).astype(int))
    axes[1].quiver(g[indices], np.zeros(len(indices)), displacement[indices], np.zeros(len(indices)),
                   angles="xy", scale_units="xy", scale=1, width=.004, color="#e47e30")
    axes[1].axhline(0, color=".5", linestyle="--", linewidth=1)
    axes[1].set(xlabel="输入 g", ylabel="位移 R(g)−g", title="真实长度的离散流向箭头")
    inputs = public.arrays["train_direct"][:, :-1, 0]
    for ax in axes:
        ax.axvspan(float(inputs.min()), float(inputs.max()), alpha=.12, color="#234f9c", label="训练输入范围")
        ax.legend(fontsize=9)
        ax.grid(alpha=.2)
    fig.suptitle("二维 φ⁴ 同作用量基准：仅使用已保存数组，未搜索或插值新根")
    fig.savefig(output / "R_vs_g.png", dpi=180)
    plt.close(fig)


def plot_physics(summary, output, plt):
    rows = summary["rows"]
    panels = [
        ("平均绝对磁化", r"$\langle|m|\rangle$", lambda row: row["observable_means"]["absolute_magnetization"]),
        ("有限尺寸宾德比", r"$1-\langle m^4\rangle/(3\langle m^2\rangle^2)$", lambda row: row["binder_1_minus_m4_over_3m2sq"]),
        ("有限尺寸连通磁化率", r"$L^2(\langle m^2\rangle-\langle m\rangle^2)$", lambda row: row["susceptibility_connected"]),
        ("绝对磁化扣除量（非 h=0 线性响应）", r"$L^2(\langle m^2\rangle-\langle|m|\rangle^2)$", lambda row: row["susceptibility_absolute_subtraction_per_site"]),
        ("八种观测量的最小链有效样本量", "保存样本的有效样本量", lambda row: row["minimum_saved_ess"]),
        ("最低生产接受率", "混合蒙特卡罗接受比例", lambda row: row["minimum_production_acceptance"]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for ax, (title, ylabel, value) in zip(axes.flat, panels):
        x = [row["r"] for row in rows]
        y = [np.nan if value(row) is None else value(row) for row in rows]
        ax.plot(x, y, color=".75", linewidth=1, zorder=1)
        for split, label, marker, color in (("train", "训练条件", "o", "#234f9c"),
                                            ("test", "留出条件", "s", "#e47e30")):
            selected = [row for row in rows if row["split"] == split]
            ax.scatter([row["r"] for row in selected], [np.nan if value(row) is None else value(row) for row in selected],
                       label=label, marker=marker, color=color, s=32, zorder=3)
        ax.axvspan(R_CRITICAL - R_UNCERTAINTY, R_CRITICAL + R_UNCERTAINTY, color="#ac2663", alpha=.2)
        ax.axvline(R_CRITICAL, color="#ac2663", linestyle="--", linewidth=1,
                   label="文献无限体积蒙特卡罗参考")
        ax.set(title=title, xlabel="生成参数 r（仅用于评价）", ylabel=ylabel)
        ax.grid(alpha=.2)
    axes[1, 1].axhline(32, color=".5", linestyle=":", label="既有有效样本量 32 诊断阈值")
    axes[1, 2].axhline(.6, color=".5", linestyle=":", label="既有接受率 0.6 诊断阈值")
    axes[1, 2].set_ylim(0, 1.05)
    for ax in axes.flat:
        ax.legend(fontsize=7.5)
    fig.suptitle("二维 φ⁴：u=1、h=0、L=36；中心参数条件完整留出\n"
                 "文献参考 r_c=−1.2724682±0.0000048；有限链点估计无误差条，不认证平衡或临界指数", fontsize=13)
    fig.savefig(output / "physics_vs_r.png", dpi=180)
    plt.close(fig)


def return_files(output=None):
    """List own and companion files relative to the report root, never PT/logs."""
    if output is not None:
        root = Path(output).resolve()
        for relative in PUBLIC_FILES:
            path = (root / relative).resolve()
            if (not path.is_relative_to(root) or path.suffix not in (".json", ".npz", ".png")
                    or not path.is_file()):
                raise ValueError("Unsafe or missing public report file")
    return list(PUBLIC_FILES)


def write_report(data, run, output, *, font=None):
    public = generic.load_public_run(run)
    root, manifest, status, sidecars, fingerprints, quality = load_public_data(data)
    check_data_run_consistency(manifest, sidecars, public, fingerprints)
    summary = physics_summary(manifest, sidecars, public, quality)
    output = Path(output).resolve()
    for source in (root, public.root):
        if source.is_relative_to(output) or output.is_relative_to(source):
            raise ValueError("Report output must not overlap a source directory")
    if output.exists():
        raise FileExistsError("A fresh report directory is required")
    plt = generic._plotting(font)
    output.mkdir(parents=True, exist_ok=False)
    companion = output / "generic"
    companion.mkdir()
    _plot_flow(public, companion, plt)
    generic.plot_coupling(public, companion, plt)
    generic.plot_losses(public, companion, plt)
    generic_metrics = dict(schema_version=1, closure=generic.closure_metrics(public.arrays),
        saved_numerical_roots=public.summary.get("numerical_fixed_points", public.summary.get("roots", [])),
        root_search_performed=False, saved_grid_zero_interpolation_performed=False,
        critical_r=None, thermal_nu=None, limitations=LIMITATIONS)
    _write(companion / "metrics.json", generic_metrics)
    _write(companion / "report.json", dict(status="complete", public_files=list(GENERIC_FILES),
        generic_functions_reused=["load_public_run", "plot_coupling", "plot_losses", "closure_metrics"],
        flow_plot_semantics="saved R and displacement arrays only; no new zero interpolation",
        model_loaded=False, inference_performed=False, root_search_performed=False))
    completed = dict(status="complete", workflow_complete=True, returncode=0, error=None,
                     requires_user_action=False, equilibrium_certified=False, thermal_nu=None)
    _write(companion / "status.json", completed)
    plot_physics(summary, output, plt)
    _write(output / "physics_summary.json", summary)
    rows = summary["rows"]
    arrays = {name: np.asarray([row[name] for row in rows]) for name in (
        "cell_index", "r", "u", "h", "split", "critical_reference_condition",
        "g_mean_by_scale", "g_chain_population_std_by_scale", "unavailable_ess_count")}
    for name in OBSERVABLES:
        arrays[name] = np.asarray([row["observable_means"][name] for row in rows], dtype=np.float64)
    for name in ("binder_1_minus_m4_over_3m2sq", "susceptibility_connected", "susceptibility_absolute_subtraction_per_site", "minimum_saved_ess",
                 "minimum_production_acceptance"):
        arrays[name] = np.asarray([np.nan if row[name] is None else row[name] for row in rows], dtype=np.float64)
    np.savez_compressed(output / "physics_arrays.npz", **arrays)
    after_data = {name: generic.sha256(root / name) for name in fingerprints}
    after_run = {name: generic.sha256(public.root / name) for name in public.fingerprints}
    if after_data != fingerprints or after_run != public.fingerprints:
        raise ValueError("Public sources changed during report generation")
    files = return_files()
    provenance = dict(schema_version=1, status="complete", source_data=str(root), source_run=str(public.root),
        reference=REFERENCE, sources_sha256_before=dict(data=fingerprints, run=public.fingerprints),
        sources_sha256_after=dict(data=after_data, run=after_run), public_sources_unchanged=True,
        outputs_sha256={name: generic.sha256(output / name) for name in files if name not in ("report.json", "status.json")},
        public_files=files, return_files=files, model_loaded=False, physical_data_loaded=False,
        GPU_used=False, inference_performed=False, root_search_performed=False,
        saved_grid_zero_interpolation_performed=False, critical_reference_condition_held_out=True,
        equilibrium_certified=False, critical_r=None, thermal_nu=None, sampling_execution=status,
        limitations=LIMITATIONS)
    _write(output / "report.json", provenance)
    _write(output / "status.json", completed)
    return_files(output)
    return provenance, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "run", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--font", type=Path)
    args = parser.parse_args(argv)
    provenance, _ = write_report(args.data, args.run, args.output, font=args.font)
    print(json.dumps(dict(status="complete", output=str(args.output.resolve()),
                          return_files=provenance["return_files"]), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
