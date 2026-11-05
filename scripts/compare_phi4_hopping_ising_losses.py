"""Offline loss-history comparison; never loads models, checkpoints or fields."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PHI_DIR = ROOT / "results/phi4_hopping_b3_g1_aligned_seed91_vm225_20261031_monitor/hmc004_budget4h_gpu3/complete/training"
ISING_DIR = ROOT / "results/ising2d_lr_gpu225_monitor_seed91/constant/full_report"
COMPONENTS = (
    ("q_reconstruction", "Q：局部联合概率重建", 10.0),
    ("q_subset_micro", "Q：微观子集一致性", 20.0),
    ("q_subset_coarse", "Q：粗尺度子集一致性", 5.0),
    ("c_information", "C：信息奖励（保留负号）", -2.0),
    ("c_variance", "C：方差防塌缩", 2.0),
    ("c_uniform", "C：均匀块保持（NLL／MSE）", 1.0),
    ("e01", r"R：一步闭合 $e_{01}$", 5.0),
    ("e12", r"R：一步闭合 $e_{12}$", 2.5),
    ("e02", r"R：两步闭合 $e_{02}$", 2.5),
)
CAVEATS = [
    "此为历史运行日志的描述性对照，不是同数据、同统计口径的公平基准，不能由曲线差异作因果归因。",
    "均为二维、36→12→4、b=3、d_g=1、隐藏层96、共享仿射Q重建；均有TRAIN均值偏置初始化。",
    "Ising取12000步恒定学习率运行的前3000步，不代表其完整训练终点；不插值、不平滑、不外推。",
    "Ising重建目标为16类离散联合概率；φ⁴为625类软联合概率，绝对MSE与潜坐标尺度不可直接比较。",
    "Ising的C采用经验直方图MI与保色NLL；φ⁴采用核平滑信息代理与均匀块MSE，绝对值不可直接比较。",
    "Ising更新小批量16×128，日志在optimizer.step后以固定TRAIN集合24×128评价并隔离C随机数；φ⁴日志为更新前当前随机小批量32×64。",
    "两转移时，旧5×一步均值+5×微观起点多步均值，恰等于all-start的5e01+2.5e12+2.5e02；不是闭合公式差异。",
    "相对200步曲线仅描述各运行自身的变化，不能消除目标、估计器、坐标尺度及日志口径差异。",
    "归一化信息面板绘制正信息量I(t)/I(200)，不是负奖励的下降率；信息估计增大不等于物理学习改善。C总目标只记录相对200步的有符号差，不报告下降倍数。",
    "短期loss下降不等于正确不动点或临界指数恢复；本脚本不训练、不加载模型、不重新评价物理数据。",
]


def _number(row: dict, key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite history value: {key}")
    return value


def _equal(a: float, b: float, name: str) -> None:
    if not math.isclose(a, b, rel_tol=2e-9, abs_tol=2e-12):
        raise ValueError(f"Inconsistent objective {name}: {a} != {b}")


def _nonnegative(value: float, name: str) -> tuple[float, bool]:
    if value < -2e-12:
        raise ValueError(f"Negative squared-error component {name}: {value}")
    return max(0.0, value), value < 0


def canonical_history(history: list[dict], kind: str, max_step: int = 3000) -> dict:
    if kind not in {"ising", "phi4"}:
        raise ValueError("Unknown history kind")
    points, clipped = [], []
    previous = 0
    for source in history:
        step = int(_number(source, "epoch" if kind == "ising" else "step"))
        if step <= previous:
            raise ValueError("History steps must be strictly increasing")
        previous = step
        if step > max_step:
            continue
        if step < 1:
            raise ValueError("History steps must be positive")
        if kind == "ising":
            e01 = _number(source, "train_rollout_level_1")
            e02 = _number(source, "train_rollout_level_2")
            e12 = 2 * _number(source, "train_commutator") - e01
            raw = {
                "q_reconstruction": _number(source, "train_separation"),
                "q_subset_micro": _number(source, "train_anchor") / 10,
                "q_subset_coarse": _number(source, "train_coarse_consistency"),
                "c_information": -_number(source, "train_rsmi"),
                "c_variance": _number(source, "train_kernel_noncollapse"),
                "c_uniform": _number(source, "train_kernel_sign"),
                "e01": e01, "e12": e12, "e02": e02,
            }
            _equal((e01 + e02) / 2, _number(source, "train_rollout"), "Ising rollout")
        else:
            raw = {key: _number(source, key) for key, _, _ in COMPONENTS[:6]}
            raw.update(e01=_number(source, "closure_start_0_target_1"),
                       e12=_number(source, "closure_start_1_target_2"),
                       e02=_number(source, "closure_start_0_target_2"))
            _equal((raw["e01"] + raw["e12"]) / 2, _number(source, "closure_one_step"), "phi4 one step")
            _equal((raw["e01"] + raw["e02"]) / 2, _number(source, "closure_rollout"), "phi4 rollout")
        for key in raw:
            raw[key], was_clipped = _nonnegative(raw[key], key)
            if was_clipped:
                clipped.append({"step": step, "component": key, "reason": "negative rounding only, |x| ≤ 2e-12"})
        weighted = {key: weight * raw[key] for key, _, weight in COMPONENTS}
        totals = {group: sum(weighted[key] for key, _, _ in COMPONENTS if key.startswith(group.lower() + "_") or (group == "R" and key.startswith("e"))) for group in ("Q", "C", "R")}
        closure_one_step = (raw["e01"] + raw["e12"]) / 2
        closure_rollout = (raw["e01"] + raw["e02"]) / 2
        _equal(totals["R"], 5 * closure_one_step + 5 * closure_rollout, "S=2 closure identity")
        if kind == "ising":
            _equal(sum(totals.values()), _number(source, "train_total"), "Ising train_total")
            _equal(totals["Q"] + totals["R"] + 2 * raw["c_variance"] + raw["c_uniform"], _number(source, "train_qcr_total"), "Ising train_qcr_total")
        else:
            for group in ("Q", "C", "R"):
                _equal(totals[group], _number(source, group.lower() + "_total"), f"phi4 {group}_total")
        level_keys = [f"train_reconstruction_mse_level_{level}" if kind == "ising" else f"reconstruction_level_{level}" for level in range(3)]
        levels = [_number(source, key) for key in level_keys] if all(key in source for key in level_keys) else None
        points.append({"step": step, "unweighted": raw, "weighted": weighted, "totals": totals,
                       "reconstruction_by_level": levels,
                       "closure_one_step": closure_one_step, "closure_rollout": closure_rollout})
    if not points:
        raise ValueError("No history points in requested range")
    return {"points": points, "rounding_clips": clipped}


def _ratio(value: float, denominator: float):
    return value / denominator if denominator != 0 else (0.0 if value == 0 else None)


def metrics(series: dict) -> dict:
    points = series["points"]
    by_step = {point["step"]: point for point in points}
    anchor = by_step.get(200)
    if anchor is None:
        raise ValueError("Exact 200-step log point required; interpolation is forbidden")
    selected = {}
    for step in (200, 1000, 2000, 3000):
        point = by_step.get(step)
        if point is None:
            selected[str(step)] = None
            continue
        selected[str(step)] = {
            **point,
            "weighted_ratio_to_200": {key: _ratio(value, anchor["weighted"][key]) for key, value in point["weighted"].items()},
            "total_ratio_to_200": {key: None if key == "C" else _ratio(value, anchor["totals"][key]) for key, value in point["totals"].items()},
            "signed_C_total_delta_from_200": point["totals"]["C"] - anchor["totals"]["C"],
            "closure_one_step_ratio_to_200": _ratio(point["closure_one_step"], anchor["closure_one_step"]),
            "closure_rollout_ratio_to_200": _ratio(point["closure_rollout"], anchor["closure_rollout"]),
        }
    best = min(points, key=lambda point: point["unweighted"]["q_reconstruction"])
    final = points[-1]
    at1000 = by_step.get(1000)
    level_ratios = ([_ratio(value, reference) for value, reference in zip(final["reconstruction_by_level"], at1000["reconstruction_by_level"])]
                    if at1000 and at1000["reconstruction_by_level"] is not None and final["reconstruction_by_level"] is not None else None)
    return {"exact_logged_steps": selected,
            "reconstruction_by_level_final_over_1000": level_ratios,
            "q_reconstruction_logged_minimum": {"step": best["step"], "unweighted": best["unweighted"]["q_reconstruction"],
                "final_step": final["step"], "final_unweighted": final["unweighted"]["q_reconstruction"],
                "final_over_logged_minimum": _ratio(final["unweighted"]["q_reconstruction"], best["unweighted"]["q_reconstruction"])} }


def _font():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    candidates = [Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"), Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")]
    for path in candidates:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=str(path)).get_name()
            break
    else:
        raise RuntimeError("Chinese font unavailable")
    matplotlib.rcParams["axes.unicode_minus"] = False


def _scale(ax, values, signed=False):
    all_values = np.concatenate([np.asarray(v) for v in values])
    if signed or np.any(all_values <= 0):
        positive = np.abs(all_values[all_values != 0])
        threshold = float(positive.min() / 10) if positive.size else 1e-6
        ax.set_yscale("symlog", linthresh=threshold)
        if np.any(all_values == 0) and np.all(all_values >= 0):
            ax.set_ylim(bottom=0)
            top = ax.get_ylim()[1]
            ticks = [value for value in ax.get_yticks() if 0 < value <= top]
            ax.set_yticks([0., *ticks])
    else:
        ax.set_yscale("log")


def plot_comparison(runs: dict, output: Path) -> list[str]:
    _font()
    import matplotlib.pyplot as plt
    colors = {"ising": "#2263aa", "phi4": "#d16a21"}
    names = {"ising": "二维 Ising（前3000步）", "phi4": r"二维 $\phi^4$（3000步）"}
    files = []
    caption = "仅绘原始日志点，无插值／平滑。目标、C估计器与日志口径不同，绝对值不是公平优劣判据。\nIsing：更新后固定训练评价面板24×128；φ⁴：更新前当前随机小批量32×64。"
    for normalized in (False, True):
        fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True)
        for ax, (key, title, _) in zip(axes.flat, COMPONENTS):
            values = []
            for kind, run in runs.items():
                points = run["points"]
                anchor = next(point for point in points if point["step"] == 200)
                y = [point["weighted"][key] for point in points]
                x = [point["step"] for point in points]
                if normalized:
                    base = anchor["weighted"][key]
                    if base == 0 and any(value != 0 for value in y):
                        raise ValueError(f"Undefined 200-step normalization: {kind} {key}")
                    y = [(_ratio(value, base) or 0.0) for value in y]
                values.append(y)
                ax.plot(x, y, marker="o", markersize=3, linewidth=1.1, label=names[kind], color=colors[kind])
            if normalized:
                ax.axhline(1, color="0.5", linewidth=.7, linestyle="--")
            if normalized and key == "c_information":
                title = "C：信息量相对200步（上升＝奖励增强）"
            ax.set_title(title, fontsize=11)
            _scale(ax, values, signed=(key == "c_information" and not normalized))
            ax.grid(alpha=.2)
            ax.set_xlim(0, 3000)
            ax.set_xlabel("实际更新步数")
            ax.set_ylabel("相对200步比值" if normalized else "带权目标项（原坐标）")
        axes.flat[0].legend(fontsize=9)
        fig.suptitle("各项目标的运行内变化" if normalized else "二维 Ising 与连续 φ⁴：逐项目标轨迹", fontsize=17)
        fig.text(.5, .025, caption, ha="center", fontsize=10)
        fig.tight_layout(rect=(0, .07, 1, .96))
        filename = "loss_components_relative200.png" if normalized else "loss_components_weighted.png"
        fig.savefig(output / filename, dpi=170)
        plt.close(fig)
        files.append(filename)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True)
    for ax, group in zip(axes, ("Q", "C", "R")):
        values = []
        for kind, run in runs.items():
            x = [p["step"] for p in run["points"]]
            y = [p["totals"][group] for p in run["points"]]
            values.append(y)
            ax.plot(x, y, marker="o", markersize=3, linewidth=1.1, color=colors[kind], label=names[kind])
        _scale(ax, values, signed=(group == "C"))
        ax.set_title(f"{group}：带权分项之和")
        ax.set_xlabel("实际更新步数")
        ax.set_ylabel("带权总目标（原坐标）")
        ax.set_xlim(0, 3000)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=9)
    fig.suptitle("Q／C／R 总目标轨迹（不同物理目标，不直接比较绝对高低）", fontsize=15)
    fig.text(.5, .025, caption, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .12, 1, .92))
    fig.savefig(output / "loss_totals_weighted.png", dpi=170)
    plt.close(fig)
    files.append("loss_totals_weighted.png")
    if all(point["reconstruction_by_level"] is not None for run in runs.values() for point in run["points"]):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True)
        for level, ax in enumerate(axes):
            values = []
            for kind, run in runs.items():
                x = [p["step"] for p in run["points"]]
                y = [p["reconstruction_by_level"][level] for p in run["points"]]
                values.append(y)
                ax.plot(x, y, marker="o", markersize=3, linewidth=1.1, color=colors[kind], label=names[kind])
            _scale(ax, values)
            ax.set_title(f"Q：第{level}层重建（L={36 // (3 ** level)}）")
            ax.set_xlabel("实际更新步数")
            ax.set_ylabel("重构平方误差和（再平均系综）")
            ax.set_xlim(0, 3000)
            ax.grid(alpha=.2)
        axes[0].legend(fontsize=9)
        fig.suptitle("Q 重建的分尺度轨迹（16类与625类目标，绝对值不直接可比）", fontsize=15)
        fig.text(.5, .025, caption, ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .12, 1, .92))
        fig.savefig(output / "q_reconstruction_by_scale.png", dpi=170)
        plt.close(fig)
        files.append("q_reconstruction_by_scale.png")
    return files


def run(phi_history: Path, phi_manifest: Path, ising_summary: Path, ising_manifest: Path, output: Path) -> dict:
    inputs = {"phi4_history": phi_history, "phi4_manifest": phi_manifest, "ising_summary": ising_summary, "ising_manifest": ising_manifest}
    sources = {key: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for key, path in inputs.items()}
    phi = json.loads(phi_history.read_text())
    ising = json.loads(ising_summary.read_text())["history"]
    # Read manifests as source evidence without loading any model or physical arrays.
    manifests = {"phi4": json.loads(phi_manifest.read_text()), "ising": json.loads(ising_manifest.read_text())}
    runs = {"ising": canonical_history(ising, "ising"), "phi4": canonical_history(phi, "phi4")}
    report = {"status": "complete", "offline_only": True, "max_actual_update": 3000, "sources": sources,
              "caveats_zh": CAVEATS, "components": [{"key": key, "title_zh": title, "weight": weight} for key, title, weight in COMPONENTS],
              "objective_identity": "5 mean(e01,e12) + 5 mean(e01,e02) = 5e01 + 2.5e12 + 2.5e02",
              "mapping": {"ising_q_subset_micro": "train_anchor / 10 (logged anchor contains its internal factor 10)",
                          "ising_e12": "2 * train_commutator - train_rollout_level_1", "ising_c_information": "-train_rsmi (natural logarithm, not bits)"},
              "metrics": {kind: metrics(series) for kind, series in runs.items()}, "series": runs,
              "manifest_recipe": {kind: value.get("recipe", value.get("training_recipe")) for kind, value in manifests.items()}}
    output.mkdir(parents=True, exist_ok=False)
    report["figures"] = plot_comparison(runs, output)
    (output / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    for key, path in inputs.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != sources[key]["sha256"]:
            raise RuntimeError("Input changed during offline comparison")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="仅从公开JSON日志比较二维Ising和连续φ⁴各项loss")
    parser.add_argument("--phi-history", type=Path, default=PHI_DIR / "history.json")
    parser.add_argument("--phi-manifest", type=Path, default=PHI_DIR / "manifest.json")
    parser.add_argument("--ising-summary", type=Path, default=ISING_DIR / "summary.json")
    parser.add_argument("--ising-manifest", type=Path, default=ISING_DIR / "manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run(args.phi_history, args.phi_manifest, args.ising_summary, args.ising_manifest, args.output)
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve()), "figures": report["figures"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
