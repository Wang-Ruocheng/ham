"""Plot a saved C-only diagnostic; no model, checkpoint, or field loading."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


EXPECTED_C_SHA = "bfcece7c7f45249a616a5762700fec6d8aaba60e4a7ea3a58e466bb17b484332"


def main() -> None:
    parser = argparse.ArgumentParser(description="最终 C 的离线响应图")
    parser.add_argument("diagnostic", type=Path)
    args = parser.parse_args()
    data = json.loads(args.diagnostic.read_text())
    if not data["state_unchanged"] or not data["source_files_match_training_manifest"]:
        raise ValueError("Diagnostic integrity flags are not satisfied")
    if data["C_state_sha256"] != EXPECTED_C_SHA or data["checkpoint_step"] != 3000:
        raise ValueError("Unexpected final C fingerprint or training step")

    font = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
    plt.rcParams.update({"font.size": 12, "axes.unicode_minus": False})

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8))
    level = data["levels"]["0"]
    weights = np.asarray(level["linear_surrogate"]["weights"])
    vmax = np.max(np.abs(weights))
    heat = axes[0].imshow(weights, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    for i in range(3):
        for j in range(3):
            axes[0].text(j, i, f"{weights[i, j]:.3f}", ha="center", va="center",
                         color="white" if abs(weights[i, j]) > 0.58 * vmax else "black")
    axes[0].set_xticks(range(3), ["左", "中", "右"])
    axes[0].set_yticks(range(3), ["上", "中", "下"])
    axes[0].set_title("九点线性近似：位置系数\n不是网络的固定卷积核")
    fig.colorbar(heat, ax=axes[0], fraction=0.046, pad=0.04, label="近似系数")
    axes[0].set_xlabel(r"$\mu(x)\approx c+\sum_{ij}w_{ij}x_{ij}$")

    x = np.asarray(data["uniform_probe"]["input_over_scale"])
    y = np.asarray(data["uniform_probe"]["mean_over_scale"])
    axes[1].plot(x, y, color="#2166ac", lw=2.3, label="已学条件均值")
    axes[1].plot(x, x, "--", color="#666666", lw=1.5, label="恒等响应")
    anchor_idx = [int(np.argmin(abs(x - value))) for value in (-2, -1, 0, 1, 2)]
    axes[1].scatter(x[anchor_idx], y[anchor_idx], s=45, color="#d95f02", zorder=4,
                    label="保均匀项的五个训练探针")
    axes[1].axhline(0, color="gray", alpha=0.3, lw=0.7)
    axes[1].axvline(0, color="gray", alpha=0.3, lw=0.7)
    axes[1].set(xlabel=r"均匀块输入 $a/s$", ylabel=r"条件均值 $\mu(a,\ldots,a)/s$",
                title="均匀块响应并非处处恒等\n零附近的局部增益约为 3.78")
    axes[1].legend(fontsize=10, loc="upper left")
    axes[1].grid(alpha=0.18)

    names = ["均匀平均", "平均＋增益\n与偏置", "最佳单点\n＋增益与偏置", "九点线性\n＋偏置"]
    r2 = [level["block_average"]["test"]["R2"],
          level["block_average"]["test_affine_fit"]["R2"],
          level["decimation"]["test_affine_fit"]["R2"],
          level["linear_surrogate"]["test"]["R2"]]
    bars = axes[2].bar(range(4), r2, color=["#92c5de", "#4393c3", "#d6604d", "#2166ac"], width=0.65)
    for bar, value in zip(bars, r2):
        axes[2].text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center")
    axes[2].set_xticks(range(4), names, fontsize=10)
    axes[2].set(ylim=(0, 1), ylabel=r"对已学条件均值的测试集 $R^2$",
                title="接近哪种简单规则？\n仅在训练集拟合，在留出条件上评价")
    axes[2].grid(axis="y", alpha=0.18)
    axes[2].set_axisbelow(True)

    fig.suptitle(r"最终 $\phi^4$ 模型的 $3\times3$ 随机粗粒化 $C$：3000 步", fontsize=17)
    fig.text(0.5, 0.035,
             "208 个已保存物理场的描述性诊断；右图拟合对象是已学 C，不是物理真值。固定高斯噪声标准差为 0.0921。",
             ha="center", fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 0.91), w_pad=2)
    output = args.diagnostic.with_name("final_C_response.png")
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
