"""连续场 QCR 的离线公共数组报告；不加载模型、物理构型或 GPU。

python -m scripts.report_phi4_qcr --run RUN --output NEW_REPORT_DIRECTORY

Only the five explicitly named JSON/NPZ artifacts are read. Couplings are used
for evaluation plots, never training. No root search, model inference, physical
sampling, MI re-estimation, or exponent estimate occurs in this report.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.report_ising_zh import select_chinese_font


INPUT_FILES = ("manifest.json", "summary.json", "history_progress.json",
               "status.json", "evaluation_arrays.npz")
RECIPE = "phi4_continuous_gaussian_soft_statistics_v1"
LIMITATIONS = [
    "二维欧氏 φ⁴ 的正测度路径积分；并非实时路径积分或符号问题的求解。",
    "连续场软分箱、概率高斯 C 与全壳平滑信息代理是授权新配方；不等同于二值精确经验直方图 MI，也不是 InfoNCE。",
    "固定 u、h 的单 r 参数族及一维 g 不保证在粗粒化下闭合。",
    "有限样本闭合误差与数值零点不认证物理相变或热方向；不估计 r 临界值，thermal_nu 始终为 null。",
    "训练与测试按完整参数条件分离；同条件多链是物理副本，不是不同参数条件。",
    "每个尺度的直接 Q 编码包含随机 C 的有限抽样误差；低误差不证明热化或完整物理统计正确。",
]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _numeric(value, name):
    result = np.asarray(value)
    if result.dtype.kind not in "fiu" or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite numeric values")
    return result.astype(np.float64, copy=False)


@dataclass
class PublicRun:
    root: Path
    manifest: dict
    summary: dict
    history: dict
    status: dict
    arrays: dict
    fingerprints: dict


def load_public_run(run):
    """Validate public arrays, identities and split semantics without model I/O."""
    root = Path(run).resolve()
    fingerprints = {name: sha256(root / name) for name in INPUT_FILES}
    manifest, summary, history, status = [_json(root / name) for name in INPUT_FILES[:4]]
    if status.get("status") != "complete":
        raise ValueError("Only a completed training endpoint can receive a final report")
    for key, name in (("summary_sha256", "summary.json"),
                      ("evaluation_arrays_sha256", "evaluation_arrays.npz")):
        if key in status and status[key] != fingerprints[name]:
            raise ValueError(f"Completed status {key} disagrees with the public source")
    # Tiny synthetic fixtures may omit execution receipts. A real versioned
    # training manifest must have the complete, mutually consistent endpoint.
    if "recipe" in manifest or "schema_version" in manifest:
        if manifest.get("schema_version") != 5:
            raise ValueError("Expected continuous-field schema_version 5")
        if any(value.get("recipe") != RECIPE for value in (manifest, summary, status)):
            raise ValueError("Continuous-field recipe identities disagree")
        if (status.get("workflow_complete") is not True or status.get("returncode") != 0
                or status.get("error") is not None or status.get("requires_user_action", False)
                or summary.get("status") != "complete"):
            raise ValueError("Final execution receipt is not successfully complete")
        for key in ("summary_sha256", "evaluation_arrays_sha256"):
            if key not in status:
                raise ValueError(f"Completed execution receipt is missing {key}")
        steps = manifest.get("options", {}).get("steps")
        if (not isinstance(steps, int) or steps <= 0 or summary.get("actual_steps") != steps
                or status.get("actual_steps") != steps or history.get("latest_step") != steps
                or history.get("latest_checkpoint_step") != steps
                or summary.get("latest_checkpoint_step") != steps
                or status.get("latest_checkpoint_step") != steps):
            raise ValueError("Requested steps and actual training endpoint disagree")
        rows = history_rows(history)
        if not rows or rows[-1]["step"] != steps:
            raise ValueError("History does not reach the requested training endpoint")
        for name in ("Q", "C", "R"):
            receipt = summary.get("optimizer_steps", {}).get(name, {})
            if receipt.get("min") != steps or receipt.get("max") != steps or receipt.get("parameters", 0) <= 0:
                raise ValueError(f"{name} Adam parameter counters disagree with the training endpoint")
        if (manifest.get("lattice_sizes") != summary.get("lattice_sizes")
                or manifest.get("theory_dim") != 1 or summary.get("theory_dim") != 1):
            raise ValueError("Saved model geometry disagrees between manifest and summary")
    if summary.get("thermal_nu") is not None:
        raise ValueError("A scalar phi4 pilot cannot certify thermal_nu")
    with np.load(root / "evaluation_arrays.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    if not arrays:
        raise ValueError("Empty public evaluation arrays")
    scales = None
    for split in ("train", "test"):
        direct = _numeric(arrays[f"{split}_direct"], f"{split}_direct")
        if direct.ndim != 3 or direct.shape[-1] != 1 or len(direct) == 0 or direct.shape[1] < 2:
            raise ValueError("Direct encodings must be nonempty [ensemble, scale, 1] arrays")
        if scales is not None and direct.shape[1] != scales:
            raise ValueError("TRAIN and TEST scales disagree")
        scales = direct.shape[1]
        arrays[f"{split}_direct"] = direct
        if f"{split}_direct_full" in arrays:
            pooled = _numeric(arrays[f"{split}_direct_full"], f"{split}_direct_full")
            if pooled.shape != direct.shape:
                raise ValueError("Pooled-feature direct_full must match direct encoding shape")
            arrays[f"{split}_direct_full"] = pooled
        for kind, count in (("one_step", scales - 1), ("rollout", scales)):
            values = _numeric(arrays[f"{split}_{kind}"], f"{split}_{kind}")
            if values.shape != (len(direct), count, 1):
                raise ValueError(f"{split}_{kind} shape disagrees with direct encodings")
            arrays[f"{split}_{kind}"] = values
        if not np.allclose(arrays[f"{split}_rollout"][:, 0], direct[:, 0], atol=1e-12, rtol=1e-10):
            raise ValueError("Microscopic-start rollout must begin at the direct microscopic encoding")
        labels = _numeric(arrays[f"{split}_labels"], f"{split}_labels")
        if labels.shape != (len(direct), 3):
            raise ValueError("Evaluation labels must be [ensemble, 3] in r, u, h order")
        arrays[f"{split}_labels"] = labels
        ids = np.asarray(arrays[f"{split}_cell_ids"])
        if ids.shape != (len(direct),) or ids.dtype.kind not in "UiS":
            raise ValueError("Cell IDs must be an integer or Unicode vector")
        for cell in np.unique(ids):
            if not np.all(labels[ids == cell] == labels[ids == cell][0]):
                raise ValueError("A cell ID cannot refer to different physical parameters")
    train_ids = set(arrays["train_cell_ids"].tolist())
    test_ids = set(arrays["test_cell_ids"].tolist())
    if train_ids & test_ids:
        raise ValueError("TRAIN/TEST cell IDs overlap")
    if set(map(tuple, arrays["train_labels"].tolist())) & set(map(tuple, arrays["test_labels"].tolist())):
        raise ValueError("TRAIN/TEST physical parameter conditions overlap")
    labels = np.concatenate((arrays["train_labels"], arrays["test_labels"]))
    if not np.all(labels[:, 1:] == labels[0, 1:]):
        raise ValueError("The scalar-r pilot report requires fixed u and h")
    for key in ("flow_g", "flow_R", "flow_displacement"):
        values = _numeric(arrays[key], key)
        if values.ndim == 2 and values.shape[-1] == 1:
            values = values[:, 0]
        if values.ndim != 1 or len(values) < 2:
            raise ValueError("Flow arrays must be vectors with at least two samples")
        arrays[key] = values
    if not (arrays["flow_g"].shape == arrays["flow_R"].shape == arrays["flow_displacement"].shape):
        raise ValueError("Flow vector shapes disagree")
    if not np.all(np.diff(arrays["flow_g"]) > 0):
        raise ValueError("Flow g grid must be strictly increasing")
    if not np.allclose(arrays["flow_R"] - arrays["flow_g"], arrays["flow_displacement"], atol=1e-12, rtol=1e-10):
        raise ValueError("Flow arrows must equal R(g)-g, including the identity contribution")
    if manifest.get("labels_used_in_training", False) or summary.get("labels_used_in_training", False):
        raise ValueError("Expected an unsupervised run, not coupling-label training")
    for numerical_root in summary.get("numerical_fixed_points", summary.get("roots", [])):
        if (not isinstance(numerical_root, dict) or not np.isfinite(numerical_root.get("g", np.nan))
                or numerical_root.get("thermal_nu") is not None
                or numerical_root.get("physical_critical_point_certified", False)):
            raise ValueError("Saved scalar numerical roots must not claim physical critical certification")
    if fingerprints != {name: sha256(root / name) for name in INPUT_FILES}:
        raise ValueError("Public source changed during reading")
    return PublicRun(root, manifest, summary, history, status, arrays, fingerprints)


def closure_metrics(arrays):
    """RMSE and fixed original TRAIN-target population standard deviation."""
    training = arrays["train_direct"]
    rows = []
    for split in ("train", "test"):
        target = arrays[f"{split}_direct"]
        for target_scale in range(1, target.shape[1]):
            denominator = float(np.sqrt(np.mean(np.var(training[:, target_scale], axis=0, ddof=0))))
            for branch, prediction in (("one_step", arrays[f"{split}_one_step"][:, target_scale - 1]),
                                       ("microscopic_rollout", arrays[f"{split}_rollout"][:, target_scale])):
                error = prediction - target[:, target_scale]
                rmse = float(np.sqrt(np.mean(error * error)))
                rows.append(dict(split=split, branch=branch, target_scale=target_scale,
                                 source_scale=target_scale - 1 if branch == "one_step" else 0,
                                 absolute_rmse=rmse, train_target_population_std=denominator,
                                 fixed_train_nrmse=rmse / denominator if denominator > 0 else None,
                                 number_of_physical_ensembles=len(error)))
    return rows


def interpolated_flow_zeros(g, displacement):
    """Report all sampled sign changes; never search or certify neural roots."""
    g, displacement = _numeric(g, "g"), _numeric(displacement, "displacement")
    if g.ndim != 1 or g.shape != displacement.shape or not np.all(np.diff(g) > 0):
        raise ValueError("Expected a strictly increasing scalar flow grid")
    roots = []
    for index in range(len(g)):
        if displacement[index] == 0:
            roots.append(float(g[index]))
        elif index + 1 < len(g) and displacement[index] * displacement[index + 1] < 0:
            fraction = -displacement[index] / (displacement[index + 1] - displacement[index])
            roots.append(float(g[index] + fraction * (g[index + 1] - g[index])))
    return [dict(g=value, kind="saved_grid_linear_interpolation_zero", full_root_certified=False,
                 physical_critical_point_certified=False, thermal_nu=None) for value in roots]


def history_rows(history):
    rows = history.get("history", [])
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("history_progress.history must be a list of row objects")
    steps = [row.get("step") for row in rows]
    if any(not isinstance(step, (int, float)) or not np.isfinite(step) for step in steps):
        raise ValueError("Every history row must have a finite numeric step")
    if any(right <= left for left, right in zip(steps, steps[1:])):
        raise ValueError("History steps must be strictly increasing")
    return rows


def _plotting(font):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    path = select_chinese_font(font)
    font_manager.fontManager.addfont(str(path))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(path)).get_name(),
                         "axes.unicode_minus": False, "font.size": 10})
    return plt


def plot_flow(run, output, plt):
    a = run.arrays
    g, mapped, drift = (a[key] for key in ("flow_g", "flow_R", "flow_displacement"))
    saved_roots = run.summary.get("numerical_fixed_points", run.summary.get("roots", []))
    roots = saved_roots or interpolated_flow_zeros(g, drift)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].plot(g, mapped, color="#234f9c", label="共享映射 R(g)")
    axes[0].plot(g, g, "--", color=".45", label="恒等映射")
    axes[0].set(xlabel="输入 g", ylabel="输出 R(g)", title="一维共享重整化映射")
    axes[1].plot(g, drift, color="#234f9c", label="离散位移 R(g)−g")
    axes[1].axhline(0, color=".45", linestyle="--", linewidth=1)
    indices = np.unique(np.linspace(0, len(g) - 1, min(22, len(g))).astype(int))
    axes[1].quiver(g[indices], np.zeros(len(indices)), drift[indices], np.zeros(len(indices)),
                   angles="xy", scale_units="xy", scale=1, width=.004, color="#e47e30")
    axes[1].set(xlabel="输入 g", ylabel="位移 R(g)−g", title="真实长度的离散流向箭头")
    train_inputs = a["train_direct"][:, :-1, 0].ravel()
    for ax in axes:
        ax.axvspan(float(train_inputs.min()), float(train_inputs.max()), color="#b8d7ec", alpha=.14,
                   label="训练输入范围")
        ax.grid(alpha=.2)
    for index, root in enumerate(roots, 1):
        value = root["g"]
        marker = "*" if saved_roots else "o"
        label = "已保存数值根（未认证物理）" if saved_roots else "保存网格的插值零点"
        axes[0].scatter([value], [value], marker=marker, s=80, facecolor="white", edgecolor="#ac2663",
                        zorder=5, label=label if index == 1 else None)
        axes[1].scatter([value], [0], marker=marker, s=80, facecolor="white", edgecolor="#ac2663", zorder=5)
        axes[1].annotate(f"数值根 {index}" if saved_roots else f"插值零点 {index}", (value, 0),
                         xytext=(5, 9), textcoords="offset points", color="#ac2663")
    for ax in axes:
        ax.legend(fontsize=9)
    note = "星号为已保存数值根，非认证物理临界点" if saved_roots else "圆点仅为保存网格的线性插值零点"
    fig.suptitle(f"二维欧氏 φ⁴：标量 g；{note}")
    fig.savefig(output / "R_vs_g.png", dpi=180)
    plt.close(fig)


def plot_coupling(run, output, plt):
    count = run.arrays["train_direct"].shape[1]
    sizes = run.summary.get("lattice_sizes", run.manifest.get("lattice_sizes", []))
    fig, axes = plt.subplots(1, count, figsize=(4.25 * count, 4.7), squeeze=False, constrained_layout=True)
    colors = {"train": "#234f9c", "test": "#e47e30"}
    for scale, ax in enumerate(axes[0]):
        for split, label in (("train", "训练集"), ("test", "测试集")):
            a, color = run.arrays, colors[split]
            r, values = a[f"{split}_labels"][:, 0], a[f"{split}_direct"][:, scale, 0]
            ax.scatter(r, values, s=15, alpha=.34, color=color, label=f"{label}物理链")
            cells = np.unique(a[f"{split}_cell_ids"])
            means = [(float(r[a[f"{split}_cell_ids"] == cell][0]),
                      float(values[a[f"{split}_cell_ids"] == cell].mean())) for cell in cells]
            means.sort()
            ax.plot([row[0] for row in means], [row[1] for row in means], "o-", markersize=4,
                    color=color, label=f"{label}条件均值")
            if scale:
                rollout = a[f"{split}_rollout"][:, scale, 0]
                rolled = [(float(r[a[f"{split}_cell_ids"] == cell][0]),
                           float(rollout[a[f"{split}_cell_ids"] == cell].mean())) for cell in cells]
                rolled.sort()
                ax.plot([row[0] for row in rolled], [row[1] for row in rolled], "--", color=color,
                        linewidth=1.3, label=f"{label}从微观 R 迭代")
        size = f"，L={sizes[scale]}" if len(sizes) == count else ""
        ax.set(xlabel="生成参数 r（仅用于评价）", ylabel="直接编码 g", title=f"尺度 k={scale}{size}")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("耦合到表示的关系：散点为物理链，实线为条件均值，虚线为 R 迭代")
    fig.savefig(output / "coupling_by_scale.png", dpi=180)
    plt.close(fig)


def plot_losses(run, output, plt):
    rows = history_rows(run.history)
    groups = [
        ("Q 的表示目标（未加权）", [("reconstruction", "q_reconstruction", "共享统计重构"),
            ("subset_micro", "q_subset_micro", "微观子集一致性"), ("subset_coarse", "q_subset_coarse", "粗尺度子集一致性")]),
        ("C 的连续场目标（未加权）", [("information_proxy", "c_information", "平滑全壳信息代理"),
            ("variance_floor", "c_variance", "系综方差下限"), ("uniform_amplitude", "c_uniform", "均匀块保持")]),
        ("R 的闭合目标（尺度等权）", [("closure_one_step", "r_one_step", "单步闭合"),
            ("closure_rollout", "r_rollout", "微观起点多步闭合")]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.7), constrained_layout=True)
    for ax, (title, entries) in zip(axes, groups):
        plotted = False
        for key, alias, label in entries:
            pairs = [(float(row["step"]), float(row.get(key, row.get(alias)))) for row in rows
                     if isinstance(row.get(key, row.get(alias)), (float, int))
                     and np.isfinite(row.get(key, row.get(alias)))]
            if pairs:
                ax.plot([pair[0] for pair in pairs], [pair[1] for pair in pairs], label=label)
                plotted = True
        if not plotted:
            ax.text(.5, .5, "未保存相应损失分项", transform=ax.transAxes, ha="center", va="center")
        else:
            ax.legend(fontsize=8)
        ax.set(xlabel="训练更新次数", ylabel="损失值", title=title)
        ax.grid(alpha=.2)
    fig.suptitle("独立梯度的连续场 QCR：图中仅展示已保存损失，不重算 MI 或模型")
    fig.savefig(output / "loss_comparison.png", dpi=180)
    plt.close(fig)


def write_report(run, output, *, font=None):
    """Write a new independent report directory, never overwrite prior evidence."""
    public = load_public_run(run)
    history_rows(public.history)
    output = Path(output).resolve()
    if output == public.root or public.root.is_relative_to(output):
        raise ValueError("Output must not contain or replace the source run")
    if output.exists():
        raise FileExistsError(f"Report directory already exists: {output}")
    plt = _plotting(font)
    output.mkdir(parents=True, exist_ok=False)
    plot_flow(public, output, plt)
    plot_coupling(public, output, plt)
    plot_losses(public, output, plt)
    metrics = dict(schema_version=1, experiment="continuous_phi4_qcr", theory_dim=1,
                   labels_used_in_training=False, labels_used_in_report=True,
                   label_order=["r", "u", "h"], lattice_sizes=public.summary.get("lattice_sizes"),
                   actual_steps=public.summary.get("actual_steps"), closure=closure_metrics(public.arrays),
                   normalization="same-scale original TRAIN target population standard deviation; ddof=0",
                   interpolated_flow_zeros=interpolated_flow_zeros(public.arrays["flow_g"], public.arrays["flow_displacement"]),
                   saved_numerical_roots=public.summary.get("numerical_fixed_points", public.summary.get("roots", [])),
                   evaluation_direct_semantics=public.summary.get("evaluation_direct_semantics"),
                   encoding_jensen_difference=public.summary.get("encoding_jensen_difference"),
                   pooled_feature_encoding_note="direct 为两半系综 Q 编码的平均，与预测输入匹配；direct_full 为整系综特征的单次 Q 编码，二者的差单独作为 Jensen 诊断，不混入闭合指标。",
                   thermal_mode_certified=False, physical_critical_point_certified=False,
                   critical_r=None, thermal_nu=None, limitations=LIMITATIONS)
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    after = {name: sha256(public.root / name) for name in INPUT_FILES}
    if after != public.fingerprints:
        raise ValueError("Source public artifacts changed during report generation")
    outputs = {path.name: sha256(path) for path in output.iterdir() if path.is_file()}
    report = dict(schema_version=1, status="complete", source_run=str(public.root),
                  sources_sha256_before=public.fingerprints, sources_sha256_after=after,
                  public_sources_unchanged=True, model_loaded=False, physical_data_loaded=False,
                  GPU_used=False, inference_performed=False, root_search_performed=False,
                  outputs_sha256=outputs, public_files=sorted([*outputs, "report.json", "status.json"]),
                  thermal_nu=None, limitations=LIMITATIONS)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "status.json").write_text(json.dumps(dict(status="complete", workflow_complete=True,
            error=None, requires_user_action=False, thermal_nu=None), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report, metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--font", type=Path)
    args = parser.parse_args(argv)
    report, _ = write_report(args.run, args.output, font=args.font)
    print(json.dumps(dict(status=report["status"], output=str(args.output.resolve())), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
