"""显式请求的中期流场诊断；CPU只读检查点，不修改正式实验。

Remote capture can execute this source on stdin from an immutable code snapshot.
Only the resulting small JSON/PNG is retained locally; never copy checkpoints
or configuration arrays. This is not the prescribed final evaluation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np


def capture(root, step=1000, samples=16):
    import torch
    import experiments.phi4_qcr as training_module
    import recurrent_rg.phi4_qcr as continuous_module
    from experiments.phi4_qcr import load_dataset, sha256_file, state_fingerprint
    from recurrent_rg.phi4_qcr import Phi4QCR

    if step < 1 or samples < 4 or samples % 2:
        raise ValueError("Require a saved step and an even sample count >=4")
    torch.set_num_threads(1)
    root = Path(root).resolve()
    plan_path = root / "plan.json"
    plan_sha = sha256_file(plan_path)
    plan = json.loads(plan_path.read_text())
    job = json.loads((root / "job.json").read_text())
    if plan_sha != job["plan_sha256"]:
        raise RuntimeError("Frozen plan fingerprint mismatch")
    code = Path(plan["code_root"]).resolve()
    for module in (training_module, continuous_module):
        if code not in Path(module.__file__).resolve().parents:
            raise RuntimeError("Imported model/training code is outside the frozen code_root")
    for name, digest in plan["source_sha256"].items():
        if sha256_file(code / name) != digest:
            raise RuntimeError(f"Frozen source changed: {name}")
    checkpoint = root / "run" / f"training_step_{step}.pt"
    checkpoint_sha = sha256_file(checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    manifest = saved["manifest"]
    if saved["step"] != step or manifest["theory_dim"] != 1:
        raise ValueError("Checkpoint step or scalar geometry mismatch")
    if manifest["recipe"] != "phi4_continuous_gaussian_soft_statistics_v1":
        raise ValueError("Unexpected recipe")
    model = Phi4QCR(manifest["scale"], hidden=96, theory_dim=1)
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    state_sha = state_fingerprint(model)
    dataset = load_dataset(root / "data")
    if dataset.manifest_sha256 != manifest["data_manifest_sha256"]:
        raise RuntimeError("Dataset identity differs from the checkpoint")
    if samples > dataset.configurations.shape[1]:
        raise ValueError("Preview must not request unavailable configurations")
    rng = torch.Generator(device="cpu").manual_seed(202610221000)
    direct, one_step = [], []
    started = time.perf_counter()
    with torch.no_grad():
        for chain in dataset.configurations:
            x = chain[:samples].unsqueeze(0)
            values = []
            for level in model.coarse_grain(x, generator=rng):
                half = samples // 2
                first = model.Q.from_features(model.features(level[:, :half]))
                second = model.Q.from_features(model.features(level[:, half:]))
                values.append(.5 * (first + second))
            direct.append(torch.stack(values, dim=1).numpy()[0, :, 0])
            one_step.append(torch.stack([model.advance(g) for g in values[:2]], dim=1).numpy()[0, :, 0])
            if time.perf_counter() - started > 180:
                raise RuntimeError("CPU preview time guard exceeded; no automatic retry")
    direct, one_step = np.asarray(direct), np.asarray(one_step)
    low, high = float(direct[:, :2].min()), float(direct[:, :2].max())
    padding = .1 * max(high - low, 1e-6)
    grid = np.linspace(low - padding, high + padding, 401)
    x = torch.tensor(grid[:, None], dtype=torch.float64, requires_grad=True)
    mapped = model.advance(x)
    derivative = torch.autograd.grad(mapped.sum(), x)[0].detach().numpy()[:, 0]
    mapped = mapped.detach().numpy()[:, 0]
    drift = mapped - grid
    zeros = []
    for index in range(len(grid) - 1):
        if drift[index] == 0:
            value = float(grid[index])
        elif drift[index] * drift[index + 1] < 0:
            value = float(grid[index] - drift[index] * (grid[index + 1] - grid[index])
                          / (drift[index + 1] - drift[index]))
        else:
            continue
        z = torch.tensor([[value]], dtype=torch.float64, requires_grad=True)
        rz = model.advance(z)
        slope = float(torch.autograd.grad(rz.sum(), z)[0])
        zeros.append(dict(g=value, residual=float(abs(rz.detach().item() - value)),
            full_DR=slope, classification="attracting" if abs(slope) < 1 - 1e-6 else "repelling" if abs(slope) > 1 + 1e-6 else "neutral",
            method="saved_401_point_grid_linear_interpolation; not an exhaustive root search",
            inside_sampled_input_range=bool(low <= value <= high),
            physical_critical_point_certified=False, thermal_nu=None))
    cells = []
    for cell in sorted(set(dataset.cell_ids.tolist())):
        mask = dataset.cell_ids == cell
        mean = direct[mask].mean(axis=0)
        with torch.no_grad():
            r_mean = model.advance(torch.tensor(mean[:2, None], dtype=torch.float64)).numpy()[:, 0]
        cells.append(dict(cell=cell, labels_r_u_h=dataset.labels[mask][0].tolist(),
            split=str(dataset.splits[mask][0]), physical_chains=int(mask.sum()),
            mean_direct=mean.tolist(), std_direct=direct[mask].std(axis=0, ddof=0).tolist(),
            mean_R=one_step[mask].mean(axis=0).tolist(), R_of_mean=r_mean.tolist()))
    if (checkpoint_sha != sha256_file(checkpoint) or plan_sha != sha256_file(plan_path)
            or state_sha != state_fingerprint(model) or any(p.grad is not None for p in model.parameters())
            or torch.cuda.is_initialized()):
        raise RuntimeError("Read-only CPU/model identity check failed")
    for name, digest in plan["source_sha256"].items():
        if sha256_file(code / name) != digest:
            raise RuntimeError(f"Frozen source changed during preview: {name}")
    return dict(kind="explicit_user_requested_interim_CPU_preview", checkpoint_step=step,
        checkpoint_sha256=checkpoint_sha, model_state_sha256=state_sha,
        frozen_plan_sha256=plan_sha, checked_at_utc=datetime.now(timezone.utc).isoformat(),
        final_report=False, new_optimizer_updates=0, new_physical_sampling=0,
        checkpoint_unchanged=True, model_unchanged=True, source_unchanged=True, cuda_initialized=False,
        samples_per_chain=samples, physical_chains=len(direct), C_draws_per_chain=1,
        preview_seed=202610221000, device="cpu", lattice_sizes=manifest["lattice_sizes"],
        direct=direct.tolist(), one_step=one_step.tolist(), labels=dataset.labels.tolist(),
        cell_ids=dataset.cell_ids.tolist(), splits=dataset.splits.tolist(), cells=cells,
        flow_g=grid.tolist(), flow_R=mapped.tolist(), flow_displacement=drift.tolist(),
        full_DR=derivative.tolist(), interpolated_zeros=zeros,
        sampled_input_range=[low, high], thermal_nu=None, data_warning_count=dataset.warning_count,
        preview_inference_seconds=time.perf_counter() - started,
        limitations=[f"{step}步中期检查点，不是3000步终点，也不是完整正式评价。",
            f"每条链仅使用前{samples}个已保存构型、一次CPU随机C轨迹；不含抽样置信区间。",
            "CPU随机C轨迹不与正式GPU评价逐比特配对；均值点不等于全部支持域。",
            f"{dataset.warning_count}项原始采样警告与已知信息代理偏差仍保留；不能认证相变或热指数。"])


def plot(source, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from scripts.report_ising_zh import select_chinese_font

    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError("Never overwrite an existing preview image")
    data = json.loads(source.read_text())
    font = select_chinese_font(None)
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(font)).get_name(),
                         "axes.unicode_minus": False})
    g, rg, displacement = [np.asarray(data[key]) for key in ("flow_g", "flow_R", "flow_displacement")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    axes[0].plot(g, rg, label="学习映射 $R(g)$")
    axes[0].plot(g, g, "--", color="gray", label="恒等线")
    for k, name in enumerate(("微观起点", "第一粗尺度起点")):
        points = np.asarray([cell["mean_direct"][k] for cell in data["cells"]])
        values = np.asarray([cell["R_of_mean"][k] for cell in data["cells"]])
        axes[0].scatter(points, values, s=18, label=name)
        targets = np.asarray([cell["mean_direct"][k + 1] for cell in data["cells"]])
        axes[0].scatter(points, targets, s=24, marker="x", label=f"{name}的实际下一尺度")
    axes[0].set(xlabel="$g$", ylabel="$R(g)$", title="映射与恒等线")
    axes[0].legend(fontsize=8)
    axes[1].plot(g, displacement, color="tab:blue")
    axes[1].axhline(0, linestyle="--", color="gray")
    selected = np.arange(0, len(g), 30)
    axes[1].quiver(g[selected], np.zeros(len(selected)), displacement[selected], np.zeros(len(selected)),
                   angles="xy", scale_units="xy", scale=1, width=.005, color="tab:orange")
    arrow_end = g[selected] + displacement[selected]
    left = min(float(g.min()), float(arrow_end.min()))
    right = max(float(g.max()), float(arrow_end.max()))
    axes[1].set_xlim(left - .03 * (right - left), right + .03 * (right - left))
    if not data["interpolated_zeros"]:
        axes[1].text(.03, .95, "所查范围内未见插值零点", transform=axes[1].transAxes,
                     va="top", fontsize=9)
    for zero in data["interpolated_zeros"]:
        axes[0].plot(zero["g"], zero["g"], "*", color="crimson", markersize=11)
        axes[1].plot(zero["g"], 0, "*", color="crimson", markersize=11)
    axes[1].set(xlabel="$g$", ylabel="$R(g)-g$", title="真实位移与方向（箭头未放大）")
    labels = np.asarray([cell["labels_r_u_h"][0] for cell in data["cells"]])
    means = np.asarray([cell["mean_direct"] for cell in data["cells"]])
    chain_direct, chain_labels = np.asarray(data["direct"]), np.asarray(data["labels"])
    for k, length in enumerate(data["lattice_sizes"]):
        line, = axes[2].plot(labels, means[:, k], "o-", markersize=4, label=f"直接编码均值，边长{length}")
        axes[2].scatter(chain_labels[:, 0], chain_direct[:, k], color=line.get_color(), alpha=.25, s=10)
    axes[2].set(xlabel="作用量参数 $r$（仅评价）", ylabel="链均值 $g$", title="参数到坐标的尺度映射")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.ticklabel_format(axis="both", style="plain", useOffset=False)
        ax.grid(alpha=.2)
    fig.suptitle(f"大尺寸连续场：第{data['checkpoint_step']}步中期预览，非最终结果", fontsize=13)
    fig.text(.5, .01, f"每链{data['samples_per_chain']}构型、一次随机粗粒化；采样热化未认证；不认证物理临界点或临界指数。",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .045, 1, .93))
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root")
    parser.add_argument("--step", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--input")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.root and not args.input and not args.output:
        print(json.dumps(capture(args.root, args.step, args.samples), ensure_ascii=False, allow_nan=False))
    elif args.input and args.output and not args.root:
        plot(args.input, args.output)
    else:
        parser.error("Use either --root (stdout-only CPU capture) or --input JSON --output NEW_PNG")


if __name__ == "__main__":
    main()
