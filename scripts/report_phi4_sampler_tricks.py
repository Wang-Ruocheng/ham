"""Read-only Chinese plots of the frozen finite-length sampler comparison.

python -m scripts.report_phi4_sampler_tricks --root ROOT [--output NEW_DIR]

Only plan/status/report/chains JSON and observations.npz are read. No physical
configurations, sampler, model, training, new MCMC, or equilibrium certificate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.report_phi4_qcr import _plotting, sha256


INPUTS = ("plan.json", "status.json", "report.json", "chains.json", "observations.npz")
METHOD_LABELS = {
    "hmc_long_warmup": "HMC：延长预热",
    "hmc_fine_step": "HMC：细化积分",
    "hmc_wolff": "HMC＋嵌入 Wolff",
}
OBSERVABLE_LABELS = {"action_density": "作用量密度", "absolute_magnetization": "绝对磁化 |m|"}
LIMITATIONS = [
    "仅对已完成的固定长度采样诊断做离线展示，不是生产数据或 QCR 训练结果。",
    "逐独立链显示磁化轨迹，不把四链平均当成扇区混合证明；符号变化不能代替偶观测量检查。",
    "秩归一化分链 R̂ 使用原报告保存的 rank 与 folded 两者最大值，不重估或筛选轨迹。",
    "单链 ESS 以及 R̂ 都受有限长度限制；低警告值不认证平衡，未删除任何链或警告。",
    "ESS/CPU秒使用各链采样 CPU 时间，包含预热与生产，未把不同方法当作配对轨迹。",
    "簇更新不生成独立构型；本报告没有对数据做后处理随机 Z₂ 翻转或重复计数。",
    "没有延长链、追加采样、修改积分参数、挑选种子或认证临界位置、临界指数。",
]


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _finite_numeric(value, label):
    array = np.asarray(value)
    if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite numeric public observations")
    return array.astype(np.float64, copy=False)


def load_public(root):
    root = Path(root).resolve()
    fingerprints = {name: sha256(root / name) for name in INPUTS}
    plan, status, report, chains = [_read_json(root / name) for name in INPUTS[:4]]
    if (not isinstance(plan, dict) or plan.get("schema") != "phi4_sampler_tricks_diagnostic_v1"
            or not isinstance(status, dict) or status.get("status") != "complete"
            or status.get("workflow_complete") is not True or status.get("returncode") != 0
            or status.get("error") is not None or status.get("requires_user_action", False)
            or not isinstance(report, dict) or report.get("status") != "complete"):
        raise ValueError("Only a successfully completed frozen sampler diagnostic can be plotted")
    for name in INPUTS:
        if name != "status.json" and status.get("artifacts_sha256", {}).get(name) != fingerprints[name]:
            raise ValueError(f"Completed status SHA256 disagrees with {name}")
    for payload in (plan, report, status):
        if payload.get("training_updates") != 0 or payload.get("formal_sampling_started") is not False:
            raise ValueError("Expected a sampling diagnostic with zero training and no formal production")
    if report.get("equilibrium_certified", False) or status.get("equilibrium_certified", False):
        raise ValueError("A finite-chain diagnostic must not claim certified equilibrium")
    methods = plan.get("methods", [])
    method_names = [row.get("name") for row in methods]
    if len(method_names) != 3 or set(method_names) != set(METHOD_LABELS):
        raise ValueError("Expected the three predeclared HMC/HMC-fine/HMC-Wolff methods")
    r = _finite_numeric(plan.get("r_values", []), "r_values")
    starts, samples, spacing = plan.get("starts", []), plan.get("samples"), plan.get("spacing")
    if (r.shape != (3,) or len(np.unique(r)) != 3 or starts != ["hot", "hot", "cold_positive", "cold_negative"]
            or not isinstance(samples, int) or samples < 8 or not isinstance(spacing, int) or spacing <= 0
            or report.get("spacing") != spacing):
        raise ValueError("Predeclared three-r/four-chain sampling geometry is inconsistent")
    if not isinstance(chains, list) or len(chains) != 36 or status.get("completed_chains") != 36:
        raise ValueError("All 36 independent-chain receipts are required")
    names = chains[0].get("raw_observable_names")
    if (not isinstance(names, list) or len(set(names)) != len(names)
            or not set([*OBSERVABLE_LABELS, "magnetization"]).issubset(names)):
        raise ValueError("Public observable name inventory is missing required scalar statistics")
    groups = report.get("groups", [])
    if not isinstance(groups, list) or len(groups) != 9:
        raise ValueError("All nine method/parameter group summaries are required")
    group_map, chain_map = {}, {}
    for row in groups:
        identity = (row.get("method"), row.get("r"))
        if identity in group_map or identity[0] not in METHOD_LABELS or identity[1] not in r:
            raise ValueError("Duplicate or unexpected saved method/parameter group")
        group_map[identity] = row
    for row in chains:
        identity = (row.get("method"), row.get("r"), row.get("chain_index"))
        if (identity in chain_map or identity[:2] not in group_map or identity[2] not in range(4)
                or row.get("initial") != starts[identity[2]] or row.get("raw_observable_names") != names):
            raise ValueError("Chain identities or independent initial states disagree with the plan")
        chain_map[identity] = row
    arrays = {}
    with np.load(root / "observations.npz", allow_pickle=False) as archive:
        for column, value in enumerate(r):
            for method in method_names:
                key = f"{method}_r{column}"
                raw = _finite_numeric(archive[key], key)
                if raw.shape != (4, samples, len(names)):
                    raise ValueError("Public observations must be [four independent chains, saved draw, observable]")
                arrays[key] = raw.copy()
                group = group_map[(method, value)]
                for observable in (*OBSERVABLE_LABELS, "magnetization"):
                    values = raw[:, :, names.index(observable)]
                    saved = group.get("observables", {}).get(observable, {})
                    if not np.allclose(saved.get("chain_means", []), values.mean(axis=1), atol=1e-12, rtol=1e-10):
                        raise ValueError("Saved chain means disagree with public scalar observations")
                    ess = saved.get("ess_per_chain")
                    if not isinstance(ess, list) or len(ess) != 4:
                        raise ValueError("Each saved group must preserve all four chain ESS values")
                    for chain in range(4):
                        diagnostic = chain_map[(method, value, chain)].get("diagnostics", {}).get(observable, {})
                        if diagnostic.get("ess") != ess[chain]:
                            raise ValueError("Group and single-chain ESS receipts disagree")
    if fingerprints != {name: sha256(root / name) for name in INPUTS}:
        raise ValueError("Public sampler artifacts changed during reading")
    return dict(root=root, plan=plan, status=status, report=report, chains=chains,
                names=names, arrays=arrays, group_map=group_map, chain_map=chain_map,
                fingerprints=fingerprints)


def comparison_table(public):
    """Summarize saved diagnostics only, preserving every chain and warning."""
    plan, rows = public["plan"], []
    for method in [item["name"] for item in plan["methods"]]:
        for r in plan["r_values"]:
            group = public["group_map"][(method, r)]
            chains = [public["chain_map"][(method, r, chain)] for chain in range(4)]
            diagnostics = {}
            for observable in OBSERVABLE_LABELS:
                saved = group["observables"][observable]
                per_chain = []
                for index, chain in enumerate(chains):
                    value, cpu = saved["ess_per_chain"][index], chain.get("cpu_seconds")
                    rate = (float(value) / float(cpu) if isinstance(value, (int, float))
                            and isinstance(cpu, (int, float)) and cpu > 0 else None)
                    per_chain.append(dict(chain_index=index, initial=chain["initial"], ess=value,
                        sampling_cpu_seconds=cpu, ess_per_cpu_second=rate,
                        saved_series_diagnostics=chain["diagnostics"][observable],
                        saved_chain_warnings=chain.get("warnings", [])))
                diagnostics[observable] = dict(saved_group_diagnostics=saved, per_chain=per_chain,
                    ess_per_cpu_second=group.get("energy_ess_per_cpu_second" if observable == "action_density"
                                                else "absolute_m_ess_per_cpu_second"))
            rows.append(dict(method=method, method_label=METHOD_LABELS[method], r=r,
                hmc_acceptance_range=group.get("hmc_acceptance_range"),
                sign_changes_per_chain=group.get("sign_changes_per_chain"),
                total_sampling_cpu_seconds=group.get("total_sampling_cpu_seconds"),
                any_even_observable_warning=group.get("any_even_observable_warning"), observables=diagnostics))
    return dict(status="complete", samples_per_chain=plan["samples"], burn_in=plan.get("burn_in"),
        spacing=plan["spacing"], group_count=9, chain_count=36,
        warning_thresholds=dict(rank_split_rhat=plan.get("diagnostic_rhat_warning_threshold"),
            single_chain_ess=plan.get("minimum_chain_ess_warning"), acceptance=plan.get("minimum_acceptance_warning")),
        rows=rows, saved_group_summaries=public["report"]["groups"],
        saved_chain_receipts=public["chains"], saved_limitations=public["report"].get("limitations", []),
        equilibrium_certified=False, training_updates=0, formal_sampling_started=False,
        metrics_reestimated=False, sampler_executed=False, model_loaded=False,
        limits=LIMITATIONS)


def plot_chain_traces(public, output, plt):
    plan = public["plan"]
    fig, axes = plt.subplots(3, 3, figsize=(15, 10.5), sharex=True, sharey="col", constrained_layout=True)
    colors = ["#2878b5", "#46a99a", "#e98b2a", "#b3437d"]
    labels = ["链1：热初态", "链2：热初态", "链3：正冷初态", "链4：负冷初态"]
    x = (np.arange(plan["samples"]) + 1) * plan["spacing"]
    for row, item in enumerate(plan["methods"]):
        for column, r in enumerate(plan["r_values"]):
            ax = axes[row, column]
            values = public["arrays"][f"{item['name']}_r{column}"][:, :, public["names"].index("magnetization")]
            for chain in range(4):
                ax.plot(x, values[chain], color=colors[chain], linewidth=.9, alpha=.9, label=labels[chain])
            ax.axhline(0, color=".5", linewidth=.7, linestyle="--")
            ax.set(title=f"{METHOD_LABELS[item['name']]}；r={r:g}",
                   xlabel="预热后 HMC 轨迹数", ylabel="单链磁化 m")
            ax.grid(alpha=.15)
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle("φ⁴ 采样对照：四条独立链分别展示，未作链间平均\n簇更新可改变符号，但不能仅凭符号变化认证平衡", fontsize=14)
    fig.savefig(output / "m_traces.png", dpi=180)
    plt.close(fig)


def plot_mixing(public, output, plt):
    plan = public["plan"]
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.9), constrained_layout=True)
    colors = ["#2878b5", "#e98b2a", "#b3437d"]
    x = np.arange(3)
    for column, observable in enumerate(OBSERVABLE_LABELS):
        for method_index, method in enumerate(plan["methods"]):
            groups = [public["group_map"][(method["name"], r)] for r in plan["r_values"]]
            diagnostic = [group["observables"][observable] for group in groups]
            y = [item.get("rank_split_rhat", {}).get("maximum") for item in diagnostic]
            axes[0, column].plot(x, [np.nan if value is None else value for value in y],
                "o-", color=colors[method_index], label=METHOD_LABELS[method["name"]])
            for chain in range(4):
                y = [item["ess_per_chain"][chain] for item in diagnostic]
                axes[1, column].scatter(x + (method_index - 1) * .2 + (chain - 1.5) * .025,
                    [np.nan if value is None else value for value in y], s=24,
                    color=colors[method_index], alpha=.7,
                    label=METHOD_LABELS[method["name"]] if chain == 0 else None)
        rhat = plan.get("diagnostic_rhat_warning_threshold")
        if isinstance(rhat, (int, float)):
            axes[0, column].axhline(rhat, color=".4", linestyle="--", label=f"预设警告参考 {rhat:g}")
        ess = plan.get("minimum_chain_ess_warning")
        if isinstance(ess, (int, float)):
            axes[1, column].axhline(ess, color=".4", linestyle="--", label=f"预设警告参考 {ess:g}")
        axes[0, column].set(title=f"{OBSERVABLE_LABELS[observable]}：秩归一化分链 R̂", ylabel="R̂（含折叠版本的最大值）")
        axes[1, column].set(title=f"{OBSERVABLE_LABELS[observable]}：每点为一条独立链", ylabel="保存间隔单位的单链 ESS")
        for row in range(2):
            axes[row, column].set_xticks(x, [f"r={r:g}" for r in plan["r_values"]])
            axes[row, column].set_xlabel("事先固定的参数条件")
            axes[row, column].grid(alpha=.2)
            axes[row, column].legend(fontsize=8)
    fig.suptitle(f"有限链混合诊断：每链仅 {plan['samples']} 次保存；阈值不是平衡认证\nESS／CPU秒及全部单链警告见随附指标表", fontsize=14)
    fig.savefig(output / "mixing_diagnostics.png", dpi=180)
    plt.close(fig)


def write_report(root, output=None, *, font=None):
    public = load_public(root)
    output = (public["root"] / "analysis" if output is None else Path(output)).resolve()
    if output == public["root"] or public["root"].is_relative_to(output):
        raise ValueError("Analysis output must not contain or replace the diagnostic source")
    if output.exists():
        raise FileExistsError(f"Analysis output already exists: {output}")
    comparison = comparison_table(public)
    plt = _plotting(font)
    output.mkdir(parents=True, exist_ok=False)
    plot_chain_traces(public, output, plt)
    plot_mixing(public, output, plt)
    (output / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2,
        allow_nan=False) + "\n", encoding="utf-8")
    after = {name: sha256(public["root"] / name) for name in INPUTS}
    if after != public["fingerprints"]:
        raise ValueError("Frozen sampler source artifacts changed during plotting")
    outputs = {name: sha256(output / name) for name in ("comparison.json", "m_traces.png", "mixing_diagnostics.png")}
    provenance = dict(status="complete", source_root=str(public["root"]),
        sources_sha256_before=public["fingerprints"], sources_sha256_after=after,
        public_sources_unchanged=True, outputs_sha256=outputs,
        public_files=[*outputs, "report.json", "status.json"], sampler_executed=False,
        physical_configurations_loaded=False, model_loaded=False, GPU_used=False,
        training_updates=0, equilibrium_certified=False, limitations=LIMITATIONS)
    (output / "report.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2,
        allow_nan=False) + "\n", encoding="utf-8")
    (output / "status.json").write_text(json.dumps(dict(status="complete", workflow_complete=True,
        error=None, requires_user_action=False, equilibrium_certified=False), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return provenance, comparison


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--font", type=Path)
    args = parser.parse_args(argv)
    provenance, _ = write_report(args.root, args.output, font=args.font)
    print(json.dumps(dict(status=provenance["status"], output=str((args.output or args.root / "analysis").resolve())), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
