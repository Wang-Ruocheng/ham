"""诊断保存的 Q/R 跨尺度分支与闭合误差；无模型调用、拟合或根搜索。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.plot_phi4_saved_q_scales import sha256, grouped_statistics
from scripts.report_phi4_hopping_qcr import load_public_run, _setup_plot


def closure_comparison(arrays):
    direct = arrays["direct_g"][..., 0]
    one = arrays["one_step_g"][..., 0]
    rollout = arrays["rollout_g"][..., 0]
    result = {}
    for split in ("train", "test"):
        mask = arrays["splits"] == split
        result[split] = []
        for start, target, prediction, weight in (
            (0, 1, one[:, 0], 5.), (1, 2, one[:, 1], 2.5),
            (0, 2, rollout[:, 2], 2.5),
        ):
            mse = float(np.mean((prediction[mask] - direct[mask, target])**2))
            identity_mse = float(np.mean((direct[mask, start] - direct[mask, target])**2))
            result[split].append(dict(start=start, target=target,
                rmse=float(np.sqrt(mse)), identity_rmse=float(np.sqrt(identity_mse)),
                mse_ratio_to_identity=mse / identity_mse if identity_mse else None,
                objective_weight=weight, weighted_mse=weight*mse))
    return result


def saved_r_query(arrays, value=.64):
    x, y = arrays["flow_grid"][:, 0], arrays["flow_next"][:, 0]
    if not x[0] <= value <= x[-1]:
        raise ValueError("Saved R query must stay inside the existing grid")
    return dict(g=value, R_g=float(np.interp(value, x, y)),
        drift=float(np.interp(value, x, y-x)),
        saved_grid_finite_difference_derivative=float(np.interp(value, x, np.gradient(y, x))),
        derivative_is_not_autograd=True, root_search_performed=False)


def branch_rows(arrays):
    g = arrays["direct_g"][..., 0]
    rows = []
    for cell in np.unique(arrays["cell_ids"]):
        mask = arrays["cell_ids"] == cell
        for start in (0, 1):
            x, y = g[mask, start], g[mask, start+1]
            delta = y-x
            rows.append(dict(cell=int(cell), kappa=float(arrays["labels"][mask, 0][0]),
                split=str(arrays["splits"][mask][0]), start=start, target=start+1,
                input_mean=float(x.mean()), input_chain_sd=float(x.std(ddof=1)),
                target_mean=float(y.mean()), delta_mean=float(delta.mean()),
                delta_chain_sd=float(delta.std(ddof=1)),
                positive_chains=int(np.count_nonzero(delta > 0)),
                negative_chains=int(np.count_nonzero(delta < 0))))
    return rows


def history_statistics(history):
    keys = [("closure_start_0_target_1", 5.),
            ("closure_start_1_target_2", 2.5),
            ("closure_start_0_target_2", 2.5)]
    rows = [r for r in history if 2000 <= r["step"] <= 3000]
    return dict(saved_log_points=len(rows), step_range=[2000, 3000],
        weighted_pair_means={key: float(weight*np.mean([r[key] for r in rows])) for key, weight in keys},
        weighted_total_mean=float(np.mean([r["r_total"] for r in rows])),
        qualification="11 saved minibatch log points, not all last 1000 updates or fixed-target convergence")


def draw_diagnostic(plt, arrays, branches, comparison, output):
    grid, mapped = arrays["flow_grid"][:, 0], arrays["flow_next"][:, 0]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.9))
    ax = axes[0]
    ax.plot(grid, mapped-grid, color="black", lw=2, label="保存的 R(g)−g")
    for start, color in ((0, "#0072B2"), (1, "#D55E00")):
        selected = sorted((r for r in branches if r["start"] == start), key=lambda r: r["input_mean"])
        ax.errorbar([r["input_mean"] for r in selected], [r["delta_mean"] for r in selected],
            xerr=[r["input_chain_sd"] for r in selected], yerr=[r["delta_chain_sd"] for r in selected],
            color=color, marker="o", ms=4, lw=1.2, capsize=2,
            label=f"Q 实测位移：k={start} → {start+1}")
    ax.axhline(0, color=".55", ls="--", lw=1)
    ax.axvline(.64, color=".55", ls=":", lw=1)
    ax.text(.641, -.024, "g=0.64", color=".4", fontsize=9)
    ax.set(xlabel="输入 g", ylabel="下一尺度位移", xlim=(.59, .77), ylim=(-.027, .071),
        title="同一输入附近，不同尺度的目标位移有分歧")
    ax.legend(fontsize=9, loc="upper left")
    ax = axes[1]
    rows = comparison["test"]
    x = np.arange(3)
    for offset, key, color, label in ((-.18, "rmse", "#7b4a96", "当前 R"),
                                     (.18, "identity_rmse", "#999999", "恒等预测：不改变 g")):
        values = [r[key] for r in rows]
        ax.bar(x+offset, values, width=.34, color=color, label=label)
        for pos, value in zip(x+offset, values):
            ax.text(pos, value+.0007, f"{value:.4f}", ha="center", fontsize=9)
    ax.set(xticks=x, xticklabels=["一步 0→1", "一步 1→2", "两步 0→2"],
        ylabel="留出物理链上的潜变量 RMSE", title="1→2 的误差明显高于恒等基线", ylim=(0, .05))
    ax.legend(fontsize=9, loc="upper left")
    for ax in axes:
        ax.grid(alpha=.16)
        ax.set_axisbelow(True)
    fig.suptitle("最终 R 为什么没有在 g≈0.64 停住？", fontsize=16)
    fig.subplots_adjust(left=.08, right=.985, top=.84, bottom=.20, wspace=.27)
    fig.text(.5, .105, "左图点为每条件 4 链均值；误差棒为链间标准差，连线仅辅助观察，不定义连续 RG 或新不动点。", ha="center", fontsize=9.5)
    fig.text(.5, .055, "正式闭环口径：两个 64 张子系综分别编码后取均值；单次固定 C 随机流，原采样警告保留。", ha="center", fontsize=9.5)
    fig.savefig(output / "R_empirical_drift_and_error.png", dpi=175)
    plt.close(fig)


def draw_history(plt, history, output):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0))
    for ax, minimum, title in ((axes[0], 0, "完整训练日志"), (axes[1], 1000, "后期日志：不是固定 Q/C 的收敛检验")):
        selected = [r for r in history if r["step"] >= minimum]
        steps = [r["step"] for r in selected]
        ax.plot(steps, [r["r_total"] for r in selected], color="black", label="R 总目标")
        for key, weight, color, label in (
            ("closure_start_0_target_1", 5., "#0072B2", "0→1，权重 5"),
            ("closure_start_1_target_2", 2.5, "#D55E00", "1→2，权重 2.5"),
            ("closure_start_0_target_2", 2.5, "#009E73", "0→2，权重 2.5"),
        ):
            ax.plot(steps, [weight*r[key] for r in selected], color=color, label=label)
        ax.set(title=title, xlabel="训练更新次数", ylabel="加权闭环损失")
        ax.set_yscale("log")
        ax.grid(alpha=.2)
        ax.legend(fontsize=9)
    fig.suptitle("R 三条闭环分支的实际加权贡献", fontsize=15)
    fig.subplots_adjust(left=.08, right=.985, top=.83, bottom=.17, wspace=.23)
    fig.text(.5, .045, "仅使用每 100 步保存的 minibatch 日志；Q/C 同时变化，不能用早期小 loss 证明最终尺度动力学已收敛。", ha="center", fontsize=9.5)
    fig.savefig(output / "R_weighted_loss_history.png", dpi=175)
    plt.close(fig)


def render(run, output, font=None):
    run, output = Path(run).resolve(), Path(output).resolve()
    manifest, summary, history, arrays, pins = load_public_run(run)
    status = json.loads((run / "status.json").read_text())
    if status.get("evaluation_arrays_sha256") != pins["evaluation_arrays.npz"] or status.get("returncode") != 0:
        raise ValueError("Public endpoint fingerprint/status mismatch")
    for item in status["public_files"]:
        if item["path"] in pins and item["sha256"] != pins[item["path"]]:
            raise ValueError("Public input differs from endpoint whitelist")
    comparison = closure_comparison(arrays)
    branches = branch_rows(arrays)
    source_check = {}
    for local in ("recurrent_rg/phi4_aligned_qcr.py", "recurrent_rg/losses.py", "experiments/phi4_hopping_qcr.py"):
        source = Path(__file__).resolve().parents[1] / local
        expected = [digest for name, digest in manifest["source_sha256"].items() if name.endswith("/"+local)]
        actual = sha256(source)
        if expected != [actual]:
            raise ValueError("Current source differs from frozen implementation: "+local)
        source_check[local] = actual
    result = dict(source_run=str(run), public_source_sha256=pins,
        implementation_verified_sha256=source_check, script_sha256=sha256(__file__),
        actual_training_steps=summary["actual_steps"], model_fingerprints=summary["final_module_sha256"],
        query_at_g064=saved_r_query(arrays), closure_comparison=comparison,
        empirical_branches=branches, pooled_Q_statistics=grouped_statistics(arrays),
        last_saved_logs=history_statistics(history),
        formal_objective="5*MSE(R(g0),g1)+2.5*MSE(R(g1),g2)+2.5*MSE(R(R(g0)),g2)",
        Q_C_receive_R_gradients=False, no_new_model_calls=True, no_training=True,
        no_model_fitting=True, no_sampling=True, root_search_performed=False,
        critical_kappa=None, thermal_nu=None, equilibrium_certified=False,
        original_warning_count=summary["data_warning_count"],
        limitations=["Branch proximity is not exact equality or a proof that no deterministic interpolant exists.",
            "Fixed C/Q R-only convergence has not been tested; representation, finite-sample and optimization effects are not fully separated.",
            "Only one C random evaluation stream; chain SD mixes physical and kernel variation."])
    plt = _setup_plot(font)
    output.mkdir(parents=True, exist_ok=False)
    draw_diagnostic(plt, arrays, branches, comparison, output)
    draw_history(plt, history, output)
    if pins != {name: sha256(run / name) for name in pins}:
        raise RuntimeError("Public input changed during audit")
    result["plot_sha256"] = {p.name: sha256(p) for p in sorted(output.glob("*.png"))}
    (output / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--font")
    args = parser.parse_args()
    result = render(args.run, args.output, args.font)
    print(json.dumps(dict(query=result["query_at_g064"], comparison=result["closure_comparison"]), indent=2))


if __name__ == "__main__":
    main()
