"""独立、有限预算的 φ⁴ 采样器对照；不生成训练数据、不启动 QCR。

原失败记录不变。固定比较较长预热 HMC、细步长 HMC 与 HMC+嵌入 Wolff。
参数与停止规则在采样前写入不可覆盖的 plan.json；不按结果加样或换种子。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import time

import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata

from recurrent_rg.data.phi4 import OBSERVABLES, sample_hmc_chain


METHODS = (
    dict(name="hmc_long_warmup", label="HMC：延长预热", step_size=.08, leapfrog_steps=12),
    dict(name="hmc_fine_step", label="HMC：细化积分", step_size=.04, leapfrog_steps=24),
    dict(name="hmc_wolff", label="HMC＋嵌入簇更新", step_size=.08, leapfrog_steps=12),
)
STARTS = ("hot", "hot", "cold_positive", "cold_negative")
R_VALUES = (-.5, .3, -.33158)
LIMITATIONS = [
    "独立采样诊断，不是生产构型或 QCR 训练；所有参数、种子和长度事先固定。",
    "只有 r=-.5、.3 是原 TRAIN 端点；r=-.33158 只作预定压力验证，不拟合临界位置。",
    "接受率、ESS、热冷初态比较及秩归一化分链 R̂ 是有限链诊断，不认证平衡。",
    "簇更新不是独立样本生成器；符号变化不能替代能量和 |m| 等偶观测量的混合检查。",
    "不同方法使用相同初态种子，但后续随机数消费不同，不假定整条轨迹随机数配对。",
    "没有选择结果最好的 seed、删链、丢弃拒绝态或依据结果自动调整参数。",
    "本诊断不修改已冻结实验、Q/C/R 架构、loss、训练步数或数据范围。",
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value, *, exclusive=False):
    with Path(path).open("x" if exclusive else "w") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def seed_for(r, chain):
    message = json.dumps(["phi4_sampler_tricks_diagnostic_v1", 91, float(r), int(chain)])
    return int.from_bytes(hashlib.sha256(message.encode()).digest()[:8], "little")


def source_fingerprints():
    base = Path(__file__).resolve().parents[1]
    names = ("scripts/diagnose_phi4_sampler_tricks.py", "recurrent_rg/data/phi4.py",
             "recurrent_rg/data/phi4_hybrid.py", "recurrent_rg/data/spin_glass_pipeline.py")
    return {str(base / name): sha(base / name) for name in names}


def make_plan():
    # Keep the independent diagnostic below 3.2e8 force-sites. With the held
    # original pilot and unchanged proposed production this remains <2e9;
    # it does not authorize production or a finer-step production budget.
    samples, burn_in, spacing = 256, 128, 1
    per_chain = burn_in + samples * spacing
    original_trajectories = 9 * (24 + 8 * 4)
    comparison_trajectories = len(METHODS) * len(R_VALUES) * len(STARTS) * per_chain
    comparison_forces = sum((item["leapfrog_steps"] + 1) * len(R_VALUES)
                            * len(STARTS) * per_chain for item in METHODS)
    force_sites = (original_trajectories * 13 + comparison_forces) * 36**2
    if force_sites > 320_000_000 or force_sites + 8_491_392 + 1_656_225_792 > 2_000_000_000:
        raise ValueError("Diagnostic plus original held pilot and proposed unchanged production exceeds force budget")
    return dict(schema="phi4_sampler_tricks_diagnostic_v1", size=36, dimension=2, u=.2, h=0.,
        r_values=list(R_VALUES), r_roles=["TRAIN端点", "TRAIN端点", "预定压力验证"],
        starts=list(STARTS), samples=samples, burn_in=burn_in, spacing=spacing,
        methods=list(METHODS), seeds={str(r): [seed_for(r, c) for c in range(4)] for r in R_VALUES},
        original_replay=dict(r_values=[-.5, -.33158, .3],
            starts=["hot", "cold_positive", "cold_negative"], seed_base=202610270200,
            samples=8, burn_in=24, spacing=4, step_size=.08, leapfrog_steps=12),
        total_hmc_trajectories=original_trajectories + comparison_trajectories,
        total_force_site_evaluations=force_sites,
        maximum_diagnostic_force_site_evaluations=320_000_000,
        original_held_preflight_force_site_evaluations=8_491_392,
        unchanged_production_force_site_evaluations=1_656_225_792,
        original_experiment_force_site_cap=2_000_000_000,
        production_authorized_by_this_diagnostic=False,
        cluster_probability=.5,
        maximum_cluster_updates=len(R_VALUES) * len(STARTS) * per_chain,
        expected_cluster_updates=.5 * len(R_VALUES) * len(STARTS) * per_chain,
        lazy_cluster_reason="事先固定 (I+W)/2，避免有序区近全簇逐步翻转与偶数保存间隔混叠",
        cpu_budget_seconds=600, wall_budget_seconds=900, threads=1,
        diagnostic_rhat_warning_threshold=1.05, minimum_chain_ess_warning=32,
        minimum_acceptance_warning=.6, sources=source_fingerprints(),
        references=["https://doi.org/10.1103/PhysRevLett.62.1087",
                    "https://github.com/daschaich/2d_phi4",
                    "https://doi.org/10.1214/20-BA1221"],
        limitations=LIMITATIONS, formal_sampling_started=False, training_updates=0)


def _rhat(x):
    n = x.shape[1]
    within = float(np.mean(np.var(x, axis=1, ddof=1)))
    between = float(n * np.var(x.mean(axis=1), ddof=1))
    if within <= 0:
        return None
    return float(math.sqrt(((n - 1.) / n * within + between / n) / within))


def rank_split_rhat(values):
    """Vehtari et al.: max(rank-normalized split, folded rank split Rhat)."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 8 or not np.isfinite(values).all():
        raise ValueError("Expected at least two finite chains with eight draws each")
    half = values.shape[1] // 2
    split = np.concatenate((values[:, :half], values[:, -half:]))
    def normalized(array):
        ranks = rankdata(array.reshape(-1), method="average").reshape(array.shape)
        return ndtri((ranks - 3./8) / (array.size + 1./4))
    rank = _rhat(normalized(split))
    folded = _rhat(normalized(np.abs(split - np.median(split))))
    available = [number for number in (rank, folded) if number is not None]
    return dict(rank_normalized_split=rank, folded_rank_normalized_split=folded,
                maximum=max(available) if available else None)


def group_summary(rows, observations):
    """Independent-chain summaries; no concatenation into a fake iid ensemble."""
    result = {}
    for index, name in enumerate(OBSERVABLES):
        diagnostics = [row["diagnostics"][name] for row in rows]
        ess = [row["ess"] for row in diagnostics]
        rhat = rank_split_rhat(observations[:, :, index])
        hot, cold = observations[:2, :, index].mean(1), observations[2:, :, index].mean(1)
        se = math.sqrt(float(np.var(hot, ddof=1) / 2 + np.var(cold, ddof=1) / 2))
        difference = float(hot.mean() - cold.mean())
        result[name] = dict(chain_means=[row["mean"] for row in diagnostics],
            mean=float(observations[:, :, index].mean()), ess_per_chain=ess,
            minimum_ess=min(ess) if all(value is not None for value in ess) else None,
            median_ess=float(np.median(ess)) if all(value is not None for value in ess) else None,
            rank_split_rhat=rhat, hot_minus_cold=difference,
            hot_cold_between_chain_se=se,
            hot_cold_three_se_warning=abs(difference) > max(3 * se, 1e-14),
            warning=any(row.get("warning") for row in diagnostics)
                or any(value is None or value < 32 for value in ess)
                or rhat["maximum"] is None or rhat["maximum"] > 1.05
                or abs(difference) > max(3 * se, 1e-14))
    cpu = sum(row["cpu_seconds"] for row in rows)
    return dict(observables=result, equilibrium_certified=False,
        hmc_acceptance_range=[min(row["production_acceptance"] for row in rows),
                              max(row["production_acceptance"] for row in rows)],
        sign_changes_per_chain=[row["saved_magnetization_sign_changes"] for row in rows],
        total_sampling_cpu_seconds=cpu,
        energy_ess_per_cpu_second=(sum(result["action_density"]["ess_per_chain"]) / cpu
            if cpu and all(v is not None for v in result["action_density"]["ess_per_chain"]) else None),
        absolute_m_ess_per_cpu_second=(sum(result["absolute_magnetization"]["ess_per_chain"]) / cpu
            if cpu and all(v is not None for v in result["absolute_magnetization"]["ess_per_chain"]) else None),
        any_even_observable_warning=any(result[name]["warning"] for name in OBSERVABLES if name != "magnetization"))


def plot_diagnostic(root, report, *, font=None):
    from scripts.report_phi4_qcr import _plotting
    plt = _plotting(font)
    with np.load(root / "observations.npz", allow_pickle=False) as arrays:
        fig, axes = plt.subplots(3, 3, figsize=(15, 10), constrained_layout=True)
        for column, r in enumerate(R_VALUES):
            for method in METHODS:
                key = f"{method['name']}_r{column}"
                raw = arrays[key]
                for row, (index, label) in enumerate(((0, "作用量密度"), (1, "磁化 m"), (2, "绝对磁化 |m|"))):
                    axes[row, column].plot(np.arange(raw.shape[1]) * report["spacing"],
                        raw[:, :, index].mean(0), linewidth=1., alpha=.85, label=method["label"])
                    axes[row, column].set_ylabel(label)
                    axes[row, column].set_xlabel("预热后 HMC 轨迹数")
            axes[0, column].set_title(f"r={r:g}；四条独立链平均")
            axes[0, column].legend(fontsize=8)
        fig.suptitle("φ⁴ 采样方法对照：均值轨迹不能替代单链诊断；不认证平衡")
        fig.savefig(root / "sampling_comparison.png", dpi=160)
        plt.close(fig)


def execute(root, *, font=None):
    from recurrent_rg.data.phi4_hybrid import sample_hybrid_chain
    root = Path(root).resolve()
    plan = make_plan()
    root.mkdir(parents=True, exist_ok=False)
    save(root / "plan.json", plan, exclusive=True)
    cpu_start, wall_start = time.process_time(), time.monotonic()
    rows, replay, arrays, groups = [], [], {}, []
    status = dict(status="running", workflow_complete=False, host=socket.gethostname(),
                  pid=os.getpid(), training_updates=0, formal_sampling_started=False)
    save(root / "status.json", status)
    def checkpoint():
        if source_fingerprints() != plan["sources"]:
            raise RuntimeError("Diagnostic source fingerprint changed during execution")
        if time.process_time() - cpu_start > plan["cpu_budget_seconds"] or time.monotonic() - wall_start > plan["wall_budget_seconds"]:
            raise TimeoutError("Fixed diagnostic budget reached; no automatic continuation")
    try:
        spec = plan["original_replay"]
        for r in spec["r_values"]:
            for initial in spec["starts"]:
                checkpoint()
                row = sample_hmc_chain(r, .2, size=36, dimension=2, samples=8, burn_in=24,
                    spacing=4, step_size=.08, leapfrog_steps=12,
                    seed=spec["seed_base"] + len(replay), initial=initial)
                replay.append(dict(r=r, initial=initial, seed=row["seed"],
                    production_acceptance=row["production_acceptance"], burn_in_acceptance=row["burn_in_acceptance"],
                    hmc_trajectories=row["hmc_trajectories"], cpu_seconds=row["cpu_seconds"]))
        save(root / "original_replay.json", replay, exclusive=True)
        for r_index, r in enumerate(R_VALUES):
            for method in METHODS:
                group_rows, raw = [], []
                for chain, initial in enumerate(STARTS):
                    checkpoint()
                    sampler = sample_hybrid_chain if method["name"] == "hmc_wolff" else sample_hmc_chain
                    row = sampler(r, .2, h=0., size=36, dimension=2,
                        samples=plan["samples"], burn_in=plan["burn_in"], spacing=plan["spacing"],
                        step_size=method["step_size"], leapfrog_steps=method["leapfrog_steps"],
                        seed=seed_for(r, chain), initial=initial)
                    raw.append(row.pop("raw_observables"))
                    row.pop("configurations")
                    trace_key = f"{method['name']}_r{r_index}_chain{chain}"
                    for name in tuple(row):
                        if isinstance(row[name], np.ndarray):
                            arrays[f"{trace_key}_{name}"] = row.pop(name)
                    row.update(r=r, u=.2, method=method["name"], chain_index=chain)
                    rows.append(row)
                    group_rows.append(row)
                    save(root / "chains.json", rows)
                    status.update(completed_chains=len(rows), current=dict(r=r, method=method["name"], chain=chain))
                    save(root / "status.json", status)
                    print(json.dumps(dict(event="diagnostic_chain_complete", r=r, method=method["name"],
                        chain=chain, acceptance=row["production_acceptance"], cpu_seconds=row["cpu_seconds"])), flush=True)
                raw = np.stack(raw)
                arrays[f"{method['name']}_r{r_index}"] = raw
                groups.append(dict(r=r, method=method["name"], **group_summary(group_rows, raw)))
                save(root / "group_progress.json", groups)
        checkpoint()
        np.savez(root / "observations.npz", **arrays)
        report = dict(status="complete", groups=groups, original_replay=replay,
            spacing=plan["spacing"], equilibrium_certified=False,
            all_parameters_frozen_before_execution=True, formal_sampling_started=False,
            training_updates=0, cpu_seconds=time.process_time() - cpu_start,
            wall_seconds=time.monotonic() - wall_start, limitations=LIMITATIONS)
        save(root / "report.json", report, exclusive=True)
        plot_diagnostic(root, report, font=font)
        checkpoint()
        artifacts = {path.name: sha(path) for path in root.iterdir() if path.is_file() and path.name != "status.json"}
        status.update(status="complete", workflow_complete=True, current=None, returncode=0,
            error=None, requires_user_action=False, completed_chains=len(rows),
            cpu_seconds=time.process_time() - cpu_start, wall_seconds=time.monotonic() - wall_start,
            artifacts_sha256=artifacts, equilibrium_certified=False)
        save(root / "status.json", status)
        return report
    except Exception as error:
        if arrays:
            np.savez(root / "partial_observations.npz", **arrays)
        status.update(status="budget_hold" if isinstance(error, TimeoutError) else "failed",
            workflow_complete=False, current=status.get("current"), returncode=1,
            error=f"{type(error).__name__}: {error}", requires_user_action=True,
            cpu_seconds=time.process_time() - cpu_start, wall_seconds=time.monotonic() - wall_start)
        save(root / "status.json", status)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--font", type=Path)
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(make_plan(), indent=2, ensure_ascii=False))
        return
    if args.output is None:
        parser.error("--execute requires a new --output")
    execute(args.output, font=args.font)


if __name__ == "__main__":
    main()
