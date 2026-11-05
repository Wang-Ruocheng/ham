"""IR one-loop illustration and independent MC phase reference for the actual scan.

No sampling, model loading, training, or measured-field analysis is performed.
The continuum sharp circular cutoff is a declared approximation, not an exact
bare-lattice matching prescription or the learned b=3 QCR transformation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.plot_phi4_ur_reference import configure_fonts, critical_r_reference


RG_SOURCE = "https://www.damtp.cam.ac.uk/user/tong/sft/sfthtml/S3.html"
PHASE_SOURCE = "https://arxiv.org/html/2510.06508v1#S4"


def beta_ru(state: np.ndarray, cutoff: float = np.pi) -> np.ndarray:
    """[dr/dell, du/dell], d=2, ell=ln b, interaction u*phi^4/4.

    Tong uses g*phi^4, so g=u/4. The 2D shell angular factor is 1/(2*pi).
    The cutoff is restored after each infinitesimal IR RG step. No eta term
    is included at this loop order. Negative/zero fast-mode denominators are
    outside this expansion's domain; they are never silently clipped.
    """
    state = np.asarray(state, dtype=float)
    if state.shape[-1:] != (2,) or not np.isfinite(state).all():
        raise ValueError("Expected finite states with trailing coordinates [r,u].")
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("A positive finite cutoff is required.")
    r, u = state[..., 0], state[..., 1]
    denominator = cutoff**2 + r
    if np.any(denominator <= 0):
        raise ValueError("The symmetric fast-mode expansion requires Lambda^2+r>0.")
    factor = cutoff**2 / (2 * np.pi)
    return np.stack([2*r + 3*u*factor/denominator,
                     2*u - 9*u**2*factor/denominator**2], axis=-1)


def truncated_fixed_points(cutoff: float = np.pi) -> np.ndarray:
    """Analytic zeros of these truncated equations; NOT physical exact coordinates."""
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("A positive finite cutoff is required.")
    return np.array([[0., 0.], [-cutoff**2/4, np.pi*cutoff**2/4]])


def load_scan(manifest: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    data = json.loads(manifest.read_text())
    options = data["options"]
    if options["dimension"] != 2 or options["h"] != 0:
        raise ValueError("This figure requires the zero-field two-dimensional scan.")
    states = np.array([(r, u) for r in options["r_values"] for u in options["u_values"]])
    if np.any(states[:, 1] <= 0) or np.any(states[:, 1] >= 2):
        raise ValueError("The MC reference is only used over 0<u<2.")
    shards = data["shards"]
    if len(shards) != len(states) or sorted(s["cell_index"] for s in shards) != list(range(len(states))):
        raise ValueError("Expected one manifest shard per scanned parameter cell.")
    splits = {s["cell_index"]: s["split"] for s in shards}
    if not set(splits.values()) <= {"train", "test"}:
        raise ValueError("Unexpected data split.")
    test = np.array([splits[i] == "test" for i in range(len(states))])
    return options, states, test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    options, states, test = load_scan(args.manifest)
    rmin, umin = states.min(axis=0)
    rmax, umax = states.max(axis=0)
    cutoff = np.pi
    qr, qu = np.meshgrid(np.linspace(rmin+.035, rmax-.035, 15),
                         np.linspace(umin+.02, umax-.02, 13), indexing="ij")
    field = beta_ru(np.stack([qr, qu], axis=-1), cutoff)
    # Normalize by one common scalar at each point, preserving vector direction.
    norm = np.linalg.norm(field, axis=-1)
    normalized = field / np.maximum(norm[..., None], 1e-15)
    ucurve = np.linspace(umin, umax, 401)
    rcurve = critical_r_reference(ucurve)
    fps = truncated_fixed_points(cutoff)
    assert np.allclose(beta_ru(fps, cutoff), 0, atol=1e-13)

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    configure_fonts()
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 8.9), sharex=True, sharey=True)
    fig.subplots_adjust(left=.08, right=.97, bottom=.33, top=.80, wspace=.18)
    fig.suptitle(r"二维 $\phi^4$：实际扫描窗口内的理论参照", fontsize=22, y=.965)
    fig.text(.5, .909, r"$r\in[-1.55,-0.50]$，$u\in[0.50,1.00]$，$h=0$；散点为实际的 45 个采样参数点",
             ha="center", fontsize=13)
    fig.text(.5, .862, "两幅图来源不同：一圈近似箭头不作为蒙特卡洛相界的精确流场", ha="center",
             fontsize=12, color="#9a444c")
    axes[0].set_title("一圈动量壳 RG：方向场", fontsize=15, pad=13)
    axes[1].set_title("独立的文献相界参考", fontsize=15, pad=13)
    axes[0].quiver(qu, qr, normalized[..., 1], normalized[..., 0],
                   angles="xy", scale_units="xy", scale=27, pivot="mid",
                   width=.004, headwidth=3.4, headlength=4.5,
                   color="#507792", alpha=.9, zorder=2)
    axes[1].fill_between(ucurve, rcurve, rmax, color="#eaf3fc", zorder=0)
    axes[1].fill_between(ucurve, rmin, rcurve, color="#fff0e2", zorder=0)
    axes[1].plot(ucurve, rcurve, color="#27816b", lw=2.5, zorder=3)
    axes[1].text(.77, -.64, "对称／无序相", ha="center", color="#285f86", fontsize=15)
    axes[1].text(.71, -1.42, "有序相", ha="center", color="#a3632e", fontsize=15)
    for ax in axes:
        ax.scatter(states[~test, 1], states[~test, 0], s=37, facecolors="white",
                   edgecolors="#343e4a", linewidths=1.1, zorder=5, clip_on=False)
        ax.scatter(states[test, 1], states[test, 0], s=39, marker="x",
                   color="#9c467a", linewidths=1.5, zorder=6, clip_on=False)
        ax.set(xlim=(umin, umax), ylim=(rmin, rmax), xlabel=r"四次耦合 $u$")
        ax.set_xticks(options["u_values"])
        ax.set_xticklabels([f"{v:g}" for v in options["u_values"]])
        ax.grid(alpha=.13, lw=.6, zorder=0)
        for spine in ax.spines.values():
            spine.set_color("#8b969e")
    axes[0].set_ylabel(r"二次耦合 $r$")
    handles = [Line2D([], [], marker="o", mfc="white", mec="#343e4a", ls="none", label=f"训练参数点（{np.count_nonzero(~test)}）"),
               Line2D([], [], marker="x", color="#9c467a", ls="none", label=f"留出参数点（{np.count_nonzero(test)}）"),
               Line2D([], [], color="#27816b", lw=2.5, label="蒙特卡洛参考拟合相界")]
    fig.legend(handles=handles, ncol=3, loc="lower center", bbox_to_anchor=(.5, .215),
               frameon=False, fontsize=11)
    fig.text(.08, .181, r"左图：$\dot r=2r+\frac{3u}{2\pi}\frac{\Lambda^2}{\Lambda^2+r}$，"
             r"$\dot u=2u-\frac{9u^2}{2\pi}\frac{\Lambda^2}{(\Lambda^2+r)^2}$；"
             r"$\ell=\ln b$ 增大为红外方向。", fontsize=12)
    fig.text(.08, .133, r"取 $a=1,\ \Lambda=\pi$ 的圆形连续截止；箭头归一化、只示方向，不是离散 $b=3$ 的一步位移。",
             fontsize=11, color="#52606d")
    fig.text(.08, .091, r"右图：$r_c(u)\simeq u/9.9+(3u/4\pi)\ln[u/(32\times9.9)]$，采用文献的有限区间经验拟合。",
             fontsize=11, color="#52606d")
    fig.text(.08, .045, "一圈两耦合截断在二维不提供定量保证；未做精确格点匹配。两图均不是学习到的潜空间流场。",
             fontsize=11, color="#9a444c")

    output.mkdir(parents=True)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"scan_theory.{suffix}", dpi=190)
    plt.close(fig)
    np.savez_compressed(output / "flow_arrays.npz", r=qr, u=qu, beta_r=field[..., 0],
                        beta_u=field[..., 1], scan_ru=states, test_mask=test,
                        reference_u=ucurve, reference_rc=rcurve, truncated_fixed_points_ru=fps)
    receipt = {
        "kind": "one_loop_continuum_illustration_plus_independent_MC_phase_reference",
        "action": "0.5*sum_forward_edges(phi_i-phi_j)^2+sum_sites(r*phi^2/2+u*phi^4/4)",
        "dimension": 2, "h": 0, "cutoff": float(cutoff), "lattice_spacing": 1,
        "cutoff_scheme": "sharp circular continuum cutoff, not exact square-lattice Brillouin-zone matching",
        "beta_r": "2*r+3*u/(2*pi)*Lambda^2/(Lambda^2+r)",
        "beta_u": "2*u-9*u^2/(2*pi)*Lambda^2/(Lambda^2+r)^2",
        "source_conversion": "Tong g*phi^4: g=u/4; shell factor Omega_1/(2*pi)^2=1/(2*pi)",
        "flow_parameter": "ell=ln(b), increasing towards IR; infinitesimal, not discrete b=3",
        "arrow_normalization": "one scalar per vector; raw beta components saved separately",
        "truncated_equation_fixed_points_ru": fps.tolist(),
        "fixed_point_in_scan": bool(np.any(np.all((fps >= [rmin, umin]) & (fps <= [rmax, umax]), axis=1))),
        "phase_reference": {"formula": "u/9.9+3*u/(4*pi)*log(u/(32*9.9))",
                            "rc_at_scan_u": critical_r_reference(options["u_values"]).tolist(),
                            "not_a_separatrix_of_left_panel": True,
                            "status": "finite_interval_MC_reference_fit_not_exact"},
        "scan_options": options,
        "scan_manifest": str(args.manifest.resolve()),
        "scan_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "limitations": ["No controlled critical quantitative accuracy for this one-loop truncation in d=2",
                        "Higher operators, anomalous dimension and nonperturbative effects omitted",
                        "Matching lattice bare parameters to continuum couplings is only naive here",
                        "Fixed-point coordinates and arrows depend on RG scheme and coordinates",
                        "No accuracy or fixed-point conclusion about the learned QCR follows from this plot"],
        "sources": [RG_SOURCE, PHASE_SOURCE],
        "new_physical_sampling": False, "new_training": False, "QCR_model_loaded": False,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "outputs_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir())},
    }
    (output / "reference.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output), "truncated_fixed_points_ru": fps.tolist(),
                      "reference_rc_at_scan_u": receipt["phase_reference"]["rc_at_scan_u"]}))


if __name__ == "__main__":
    main()
