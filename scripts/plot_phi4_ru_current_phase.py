"""Literature phase reference on a measured dataset's bare (u,r) grid.

No model, raw configurations, or learned RG field is loaded. The MC-fit domain
and its unvalidated continuation are kept separate in both the plot and receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.plot_phi4_ur_reference import C0, HTML_URL, configure_fonts, critical_r_reference


def extrapolated_rc(u):
    """Unvalidated continuation, NOT an independently known critical boundary."""
    u = np.asarray(u, dtype=float)
    if not np.isfinite(u).all() or np.any(u < 2):
        raise ValueError("This explicitly extrapolated branch requires finite u>=2.")
    return u / C0 + 3 * u / (4 * np.pi) * np.log(u / (32 * C0))


def load_grid(path):
    manifest = json.loads(Path(path).read_text())
    opt = manifest["options"]
    if opt["dimension"] != 2 or opt["h"] != 0 or len(opt["sizes"]) != 1:
        raise ValueError("Expected one zero-field two-dimensional lattice size.")
    # Phi4Options.conditions: size, then r, then u.
    ru = np.array([(r, u) for r in opt["r_values"] for u in opt["u_values"]], dtype=float)
    if not np.isfinite(ru).all() or np.any(ru[:, 1] <= 0):
        raise ValueError("Expected finite couplings and positive quartic coupling.")
    shards = manifest["shards"]
    if sorted(s["cell_index"] for s in shards) != list(range(len(ru))):
        raise ValueError("Grid and shard indices disagree.")
    split = {s["cell_index"]: s["split"] for s in shards}
    if not set(split.values()) <= {"train", "test"}:
        raise ValueError("Unknown split.")
    return opt, ru, np.array([split[i] == "test" for i in range(len(ru))])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    options, ru, test = load_grid(args.manifest)
    rmin, umin = ru.min(axis=0)
    rmax, umax = ru.max(axis=0)
    if not umin < 2 < umax:
        raise ValueError("This layout is for a scan straddling the u=2 fit limit.")
    configure_fonts()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    us = np.linspace(umin, np.nextafter(2., 0.), 501)
    ue = np.linspace(2., umax, 301)
    rs, re = critical_r_reference(us), extrapolated_rc(ue)
    fig, ax = plt.subplots(figsize=(11.8, 9.0))
    fig.subplots_adjust(left=.10, right=.96, bottom=.30, top=.80)
    fig.suptitle(r"当前二维 $\phi^4$ 数据的 $u$–$r$ 相图参考", fontsize=21, y=.975)
    fig.text(.5, .895,
             r"$S=\frac{1}{2}\sum_{\langle ij\rangle}(\phi_i-\phi_j)^2"
             r"+\sum_i(\frac{r}{2}\phi_i^2+\frac{u}{4}\phi_i^4)$，$h=0$",
             fontsize=16, ha="center")
    fig.text(.5, .84,
             f"实际采样：{len(options['r_values'])}×{len(options['u_values'])} 个参数点；"
             f"初始尺寸 L={options['sizes'][0]}；底色表示热力学极限相区参考",
             fontsize=12, ha="center", color="#536170")
    ax.fill_between(us, np.clip(rs, rmin, rmax), rmax, color="#e8f2fc")
    ax.fill_between(us, rmin, np.clip(rs, rmin, rmax), color="#fff0e1")
    ax.axvspan(2., umax, facecolor="#f0f1f3", edgecolor="#dadde2", hatch="//", lw=0)
    ax.axvline(2., color="#969da8", lw=1., ls=":")
    ax.plot(us, rs, color="#236e63", lw=2.8, zorder=4)
    ax.plot(ue, re, color="#68727f", lw=2., ls="--", zorder=4)
    ax.scatter(ru[~test, 1], ru[~test, 0], s=43, facecolors="white",
               edgecolors="#344354", linewidths=1.2, zorder=6, clip_on=False)
    ax.scatter(ru[test, 1], ru[test, 0], s=46, marker="x", color="#bd4879",
               linewidths=1.7, zorder=7, clip_on=False)
    ax.text(1.37, -.65, "对称／无序相", ha="center", color="#285d86", fontsize=17,
            bbox=dict(fc="#e8f2fc", ec="none", alpha=.9, pad=4))
    ax.text(.98, -2.35, "有序相：$Z_2$ 自发破缺", ha="center", color="#9a5428", fontsize=16,
            bbox=dict(fc="#fff0e1", ec="none", alpha=.95, pad=4))
    ax.text(2.5, -.69, "拟合范围外\n虚线仅作外推示意", ha="center", va="center", fontsize=13,
            color="#5b626d", bbox=dict(fc="#f0f1f3", ec="none", alpha=.95, pad=5))
    # These coordinates are an approximate projected RG fixed point in the
    # cited paper, not a measured point on the thermodynamic critical boundary.
    ax.errorbar(2.13, -1.97, xerr=.13, yerr=.12, fmt="*", ms=17,
                mfc="#8246a3", mec="white", ecolor="#8246a3", lw=1.5,
                capsize=4, zorder=8)
    ax.annotate("文献投影 RG 鞍点\n不是精确物理相界点", xy=(2.13, -1.97),
                xytext=(2.55, -1.41), ha="center", va="center", color="#743c95", fontsize=12,
                bbox=dict(boxstyle="round,pad=.4", fc="white", ec="#cab8d6"),
                arrowprops=dict(arrowstyle="-", color="#8246a3", lw=1.3), zorder=9)
    ax.set(xlim=(umin, umax), ylim=(rmin, rmax), xlabel=r"四次耦合 $u$", ylabel=r"二次耦合 $r$")
    ax.set_xticks(options["u_values"])
    ax.set_xticklabels([f"{v:g}" for v in options["u_values"]])
    ax.set_yticks([-2.8, -2.4, -2., -1.6, -1.2, -.8, -.5])
    ax.grid(alpha=.14, lw=.65)
    handles = [
        Line2D([], [], color="#236e63", lw=2.8, label="文献蒙卡参考拟合（$0<u<2$）"),
        Line2D([], [], color="#68727f", lw=2, ls="--", label="同一公式外推（未经验证）"),
        Line2D([], [], marker="o", mfc="white", mec="#344354", ls="none", label=f"训练参数点（{sum(~test)}）"),
        Line2D([], [], marker="x", color="#bd4879", ls="none", label=f"留出参数点（{sum(test)}）"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.53, .173),
               ncol=2, frameon=False, fontsize=11.5)
    fig.text(.10, .137, r"相界参考：$r_c(u)\simeq u/9.9+(3u/4\pi)\ln[u/(32\times9.9)]$；不是精确解析解。", fontsize=11.5)
    fig.text(.10, .094, "星号来自 Zhao 等的 b=2 投影 RG：坐标依赖粗粒化、场归一化与截断，并非 QCR 必须复现的真值。", fontsize=10.5, color="#743c95")
    fig.text(.10, .052, "当前有限尺寸仅能呈现平滑交叉；本图未分析构型、未运行模型，也未计算 RG 箭头或新的相变点。", fontsize=10.5, color="#536170")
    args.output.mkdir(parents=True)
    for extension in ("png", "svg"):
        fig.savefig(args.output / f"current_ur_phase.{extension}", dpi=190)
    plt.close(fig)
    np.savez_compressed(args.output / "phase_reference_arrays.npz", scan_ru=ru, test_mask=test,
                        fitted_u=us, fitted_rc=rs, extrapolated_u=ue, extrapolated_rc=re)
    receipt = {
        "kind": "literature_phase_reference_not_measured_QCR_result",
        "action": "0.5*sum_forward_edges(phi_i-phi_j)^2+sum_i(r*phi_i^2/2+u*phi_i^4/4)",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "options": options, "scan_cells": len(ru), "train_cells": int(sum(~test)), "test_cells": int(sum(test)),
        "fit": {"formula": "u/9.9+3*u/(4*pi)*log(u/(32*9.9))", "empirical_domain": "0<u<2",
                "strict_formula_asymptotic_regime": "u much less than 1",
                "rc_at_in_domain_scan_u": {str(u): float(critical_r_reference(u)) for u in options["u_values"] if u < 2},
                "extrapolated_range": [2., float(umax)], "extrapolation_validated": False,
                "phase_classification_above_u2": "not assigned"},
        "literature_projected_fixed_point": {"u": 2.13, "r": -1.97, "u_reported_error": .13,
                "r_reported_error": .12, "b": 2, "is_exact_thermodynamic_boundary_point": False,
                "is_QCR_prediction_or_target": False},
        "sources": [HTML_URL], "raw_data_loaded": False, "models_loaded": False,
        "new_sampling": False, "new_training": False, "RG_arrows_generated": False,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "output_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.output.iterdir())},
    }
    (args.output / "reference.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "fit": receipt["fit"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
