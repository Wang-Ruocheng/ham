"""Locate the frozen sampling grid on a qualified literature phase reference."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from scripts.plot_phi4_ur_reference import configure_fonts, critical_r_reference, HTML_URL


def main(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    job = json.loads((root / "job.json").read_text())
    plan = json.loads((root / "plan.json").read_text())
    plan_sha = hashlib.sha256((root / "plan.json").read_bytes()).hexdigest()
    assert plan_sha == job["plan_sha256"]
    options = plan["physical"]["options"]
    assert options["dimension"] == 2 and options["h"] == 0
    rs, us = np.array(options["r_values"]), np.array(options["u_values"])
    source = root / "run/evaluation_arrays.npz"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == job["public_files"]["run/evaluation_arrays.npz"]
    with np.load(source, allow_pickle=False) as archive:
        cells, labels, splits = (archive[k].copy() for k in ("cell_ids", "labels", "splits"))
    first = [np.flatnonzero(cells == cell)[0] for cell in np.unique(cells)]
    grid, test = labels[first, :2], splits[first] == "test"
    assert len(grid) == len(rs) * len(us) == 45
    assert set(map(tuple, grid)) == {(r, u) for r in rs for u in us}
    configure_fonts()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10.4, 7.7), layout="constrained")
    u = np.linspace(.001, 1.99999, 500)
    rc = critical_r_reference(u)
    ue = np.linspace(2, 3.1, 180)
    re = ue / 9.9 + 3 * ue / (4 * np.pi) * np.log(ue / (32 * 9.9))
    ax.fill_between(u, rc, .3, color="#edf5fc")
    ax.fill_between(u, -3.25, rc, color="#fff2e4")
    ax.axvspan(2, 3.1, facecolor="#f1f2f4", edgecolor="#e1e3e7", hatch="//", lw=0)
    ax.plot(u, rc, color="#39424c", lw=2.1)
    ax.plot(ue, re, color="#777f89", lw=1.8, ls="--")
    ax.axvline(2, color="#b3b8c0", ls=":", lw=1)
    ax.add_patch(Rectangle((.5, -2.8), 2.5, 2.3, fill=False,
                           edgecolor="#777777", lw=1.6, ls="--", zorder=3))
    ax.add_patch(Rectangle((us.min(), rs.min()), np.ptp(us), np.ptp(rs), fill=False,
                           edgecolor="#b82737", lw=2.8, zorder=6))
    ax.scatter(grid[~test, 1], grid[~test, 0], s=27, color="#b82737", zorder=7)
    ax.scatter(grid[test, 1], grid[test, 0], s=37, marker="x", color="#2f5b97", zorder=7)
    ax.errorbar(2.13, -1.97, xerr=.13, yerr=.12, fmt="*", ms=17,
                color="#813fa0", mec="white", capsize=3, lw=1.1, zorder=9)
    ax.annotate("文献投影鞍点\n(u,r)≈(2.13,−1.97)", xy=(2.13, -1.97),
                xytext=(2.43, -.89), fontsize=10.5, ha="center", color="#71348f",
                bbox=dict(fc="white", ec="#c8b5d2", alpha=.96, boxstyle="round,pad=.3"),
                arrowprops=dict(arrowstyle="->", color="#813fa0"), zorder=10)
    ax.scatter([0], [0], color="black", s=45, zorder=8)
    ax.annotate("高斯参考点 (0,0)\n不在采样窗口内", (0, 0), xytext=(.25, -.26),
                fontsize=10, arrowprops=dict(arrowstyle="-", color="#555555"))
    ax.text(.65, -.38, "对称／无序相参考", color="#285d86", fontsize=11)
    ax.text(.38, -2.05, "有序相参考\n$Z_2$ 自发破缺", color="#98602e", fontsize=11)
    ax.text(2.57, .05, "u≥2：相界公式仅作外推\n不据此确定精确相区", ha="center", fontsize=10,
            bbox=dict(fc="white", ec="none", alpha=.9))
    ax.text(1.64, -2.72, "当前窗口：u∈[1.63,2.63]，r∈[−2.57,−1.37]",
            color="#a12230", fontsize=10)
    ax.set(xlim=(-.08, 3.1), ylim=(-3.2, .3), xlabel="四次耦合 $u$", ylabel="二次耦合 $r$",
           title="当前数据窗口在原始 u–r 参数平面中的位置\n45 个采样条件，h=0，初始尺寸 L=36")
    handles = [
        Line2D([], [], color="#b82737", lw=2.8, label="当前局部窗口"),
        Line2D([], [], color="#777777", lw=1.6, ls="--", label="上一轮宽窗口"),
        Line2D([], [], color="#39424c", lw=2.1, label="蒙卡相界参考拟合（0<u<2）"),
        Line2D([], [], color="#777f89", lw=1.8, ls="--", label="同一公式外推（未验证）"),
        Line2D([], [], color="#b82737", marker="o", ls="none", label="34 个训练参数点"),
        Line2D([], [], color="#2f5b97", marker="x", ls="none", label="11 个留出参数点"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=9, ncol=2)
    ax.grid(alpha=.12)
    fig.supxlabel("相界参考来自文献公式 (27)，$c_0=9.9$；星号来自其 b=2、小尺寸投影 RG。\n"
                  "本图不是精确 RG 箭头场；星号不是我们 b=3 模型必须复现的物理坐标。", fontsize=10)
    output.mkdir(parents=True, exist_ok=False)
    fig.savefig(output / "sampling_window_ur.png", dpi=180)
    plt.close(fig)
    receipt = dict(plan_sha256=plan_sha, source=str(root), sources=[HTML_URL],
                   r_values=rs.tolist(), u_values=us.tolist(), reference_ru=[-1.97, 2.13],
                   phase_fit_validated_domain="0<u<2, empirical reference, not exact",
                   action="0.5*sum_edges(phi_i-phi_j)^2 + sum_i(r*phi_i^2/2+u*phi_i^4/4)",
                   models_loaded=False, sampling_performed=False, RG_arrows_generated=False)
    (output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(output / "sampling_window_ur.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args.root, args.output)
