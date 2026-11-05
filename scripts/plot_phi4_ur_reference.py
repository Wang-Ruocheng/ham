"""Plot a literature-reference phi4 phase boundary, not a computed RG flow."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np


PAPER_URL = "https://doi.org/10.1088/2632-2153/ae493b"
HTML_URL = "https://arxiv.org/html/2510.06508v1#S4.SS1"
C0 = 9.9


def critical_r_reference(u: np.ndarray | float) -> np.ndarray:
    """Eq. (27), using the paper's finite-interval MC-fit choice c0=9.9."""
    u = np.asarray(u, dtype=float)
    if np.any(u <= 0) or np.any(u >= 2):
        raise ValueError("The plotted reference is restricted to 0 < u < 2.")
    return u / C0 + 3 * u / (4 * np.pi) * np.log(u / (32 * C0))


def configure_fonts() -> None:
    names = {f.name for f in font_manager.fontManager.ttflist}
    preferred = ["Heiti TC", "PingFang SC", "Arial Unicode MS", "Noto Sans CJK SC"]
    selected = next((name for name in preferred if name in names), None)
    if selected is None:
        raise RuntimeError("A Chinese font is required for this figure.")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [selected, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 13,
            "axes.labelsize": 15,
            "mathtext.fontset": "dejavusans",
            "svg.fonttype": "path",
            "savefig.facecolor": "white",
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing output: {output}")
    configure_fonts()

    u = np.linspace(0.001, 1.999, 1600)
    rc = critical_r_reference(u)
    rc_one = float(critical_r_reference(1.0))
    scan_r = np.linspace(-2.0, 0.5, 11)
    assert np.isfinite(rc).all() and np.all(np.diff(rc) < 0)
    assert -1.5 < rc_one < -1.25
    assert np.count_nonzero(scan_r < rc_one) == 3
    assert np.count_nonzero(scan_r > rc_one) == 8

    fig, ax = plt.subplots(figsize=(11.8, 8.5))
    fig.subplots_adjust(left=0.105, right=0.975, bottom=0.30, top=0.84)
    fig.suptitle(r"二维 $\phi^4$ 模型的 $u$–$r$ 相图参考（$h=0$）", y=0.97, fontsize=21)
    fig.text(
        0.5,
        0.902,
        r"$S=\frac{1}{2}\sum_{\langle ij\rangle}(\phi_i-\phi_j)^2"
        r"+\sum_i\left(\frac{r}{2}\phi_i^2+\frac{u}{4}\phi_i^4\right)$",
        ha="center",
        fontsize=17,
    )
    ax.fill_between(u, rc, 0.68, color="#e9f3fc", zorder=0)
    ax.fill_between(u, -2.52, rc, color="#fff0e2", zorder=0)
    ax.plot(u, rc, color="#283b4d", lw=2.8, zorder=3)
    ax.axvline(1, color="#8b4caa", lw=1.5, ls=(0, (5, 4)), alpha=0.8, zorder=2)
    ax.plot(
        np.ones_like(scan_r), scan_r, linestyle="none", marker="o", ms=6.8,
        markerfacecolor="white", markeredgecolor="#8b4caa", markeredgewidth=1.8,
        zorder=5,
    )
    ax.plot(1, rc_one, marker="D", ms=9.5, color="#9c2d75", zorder=6)
    ax.annotate(
        "$u=1$ 的相界交点\n" + rf"$r_c\approx {rc_one:.3f}$",
        xy=(1, rc_one), xytext=(1.24, -0.85), fontsize=13.5,
        color="#852560", va="center",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": "#dec8d8"},
        arrowprops={"arrowstyle": "-", "lw": 1.4, "color": "#9c2d75"},
        zorder=7,
    )
    ax.text(1.46, 0.17, "对称／无序相", fontsize=20, color="#225c87", ha="center")
    ax.text(1.46, -0.10, r"$m=0$", fontsize=17, color="#225c87", ha="center")
    ax.text(0.43, -1.80, "有序相", fontsize=20, color="#a35e29", ha="center")
    ax.text(0.43, -2.08, r"$Z_2$ 自发破缺，$m=\pm m_0$", fontsize=15, color="#a35e29", ha="center")
    ax.text(
        0.055, 0.95, "热力学极限；有序相的 $m$ 指选定纯相后的磁化", transform=ax.transAxes,
        fontsize=10.8, color="#52606d", va="top",
    )
    ax.text(1.0, 0.57, "此前的固定耦合切片", color="#80469c", ha="center", fontsize=11)
    ax.set(xlim=(0, 2), ylim=(-2.5, 0.68), xlabel=r"四次耦合 $u$", ylabel=r"二次耦合 $r$")
    ax.set_xticks(np.arange(0, 2.01, 0.25))
    ax.set_yticks(np.arange(-2.5, 0.51, 0.5))
    ax.grid(alpha=0.18, color="#718096", lw=0.7)
    for spine in ax.spines.values():
        spine.set_color("#788592")
    handles = [
        Line2D([], [], color="#283b4d", lw=2.8, label="文献所用的蒙特卡洛参考拟合相界"),
        Line2D([], [], marker="o", color="#8b4caa", ls="--", markerfacecolor="white", label="此前 QCR 的 11 个初始采样参数点"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.145), ncol=1, frameon=False, fontsize=12)
    fig.text(
        0.105, 0.109,
        r"相界参考：$r_c(u)\approx u/c_0+(3u/4\pi)\ln[u/(32c_0)]$，$c_0=9.9$。",
        fontsize=12,
    )
    fig.text(0.105, 0.070, "依据 Zhao 等（2026）式（27），采用其在 0<u<2 内的经验拟合；不是精确解析解。", fontsize=10.8, color="#52606d")
    fig.text(0.105, 0.035, "不是本轮模拟或 QCR 输出；未重建 RG 箭头；菱形为临界参数交点，并非完整 RG 不动点。", fontsize=10.8, color="#52606d")

    output.mkdir(parents=True)
    for extension in ["png", "svg"]:
        fig.savefig(output / f"phi4_ur_phase_reference.{extension}", dpi=200)
    plt.close(fig)

    receipt = {
        "kind": "literature_reference_phase_boundary_not_QCR_or_RG_flow",
        "action": "0.5*sum_nn_edges(phi_i-phi_j)^2 + sum_sites(r*phi_i^2/2 + u*phi_i^4/4)",
        "dimension": 2,
        "external_field": 0,
        "boundary": {
            "equation": "r_c(u) = u/c0 + (3*u/(4*pi))*log(u/(32*c0))",
            "c0": C0,
            "source_equation": 27,
            "domain_plotted": [float(u.min()), float(u.max())],
            "status": "finite_interval_MC_reference_fit_not_exact",
            "strict_asymptotic_regime_of_formula": "u much smaller than 1; c0=9.9 is used by the cited paper as a fit over 0<u<2, not the continuum-limit value",
            "r_c_at_u1_reference": rc_one,
            "above_boundary": "symmetric_disordered",
            "below_boundary": "Z2_broken_ordered_thermodynamic_pure_phases",
        },
        "historical_QCR_initial_slice": {"u": 1, "r_values": scan_r.tolist(), "source": "docs/phi4_qcr.md"},
        "sources": [PAPER_URL, HTML_URL],
        "new_physical_sampling": False,
        "new_training": False,
        "model_loaded": False,
        "RG_arrows_generated": False,
        "QCR_performance_claim": False,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "outputs_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(output.iterdir()) if path.is_file()
        },
    }
    (output / "reference.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "r_c_u1_reference": rc_one}, ensure_ascii=False))


if __name__ == "__main__":
    main()
