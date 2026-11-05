"""One-loop phi4 RG topology illustration; not an exact two-dimensional flow."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SOURCE = "https://www.damtp.cam.ac.uk/user/tong/sft/sfthtml/S3.html"


def beta(state: np.ndarray, epsilon: float = 1.0) -> np.ndarray:
    """IR beta functions; state order is [dimensionless r, rescaled u]."""
    state = np.asarray(state, dtype=float)
    r, u = state[..., 0], state[..., 1]
    return np.stack(
        [2 * r + u / (1 + r), epsilon * u - 3 * u**2 / (1 + r) ** 2],
        axis=-1,
    )


def jacobian(state: np.ndarray, epsilon: float = 1.0) -> np.ndarray:
    r, u = np.asarray(state, dtype=float)
    return np.array(
        [[2 - u / (1 + r) ** 2, 1 / (1 + r)],
         [6 * u**2 / (1 + r) ** 3, epsilon - 6 * u / (1 + r) ** 2]]
    )


def fixed_points(epsilon: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    if epsilon <= 0:
        raise ValueError("This illustration requires epsilon > 0.")
    return np.zeros(2), np.array([-epsilon / (6 + epsilon), 12 * epsilon / (6 + epsilon) ** 2])


def integrate(
    initial: np.ndarray,
    *,
    epsilon: float = 1.0,
    direction: float = 1.0,
    duration: float = 45.0,
    step: float = 0.003,
) -> np.ndarray:
    """Small deterministic RK4 integration, avoiding the r=-1 pole."""
    state = np.array(initial, dtype=float)
    points = [state.copy()]
    for _ in range(int(duration / step)):
        k1 = direction * beta(state, epsilon)
        k2 = direction * beta(state + step * k1 / 2, epsilon)
        k3 = direction * beta(state + step * k2 / 2, epsilon)
        k4 = direction * beta(state + step * k3, epsilon)
        state = state + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        if not np.isfinite(state).all():
            raise FloatingPointError("Nonfinite RG trajectory.")
        points.append(state.copy())
        r, u = state
        if not (-0.76 < r < 0.39 and -1e-8 < u < 0.84):
            break
        if direction < 0 and np.linalg.norm(state) < 1e-11:
            break
    return np.asarray(points)


def critical_branches(epsilon: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Backward integrate both branches of the WF stable manifold."""
    _, wf = fixed_points(epsilon)
    values, vectors = np.linalg.eig(jacobian(wf, epsilon))
    stable = vectors[:, int(np.argmin(values))].real
    stable *= np.sign(stable[1])
    delta = 1e-6
    left = integrate(wf - delta * stable, epsilon=epsilon, direction=-1)
    right = integrate(wf + delta * stable, epsilon=epsilon, direction=-1)
    return left, right


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from plot_phi4_ur_reference import configure_fonts

    configure_fonts()
    epsilon = 1.0
    gaussian, wf = fixed_points(epsilon)
    left, right = critical_branches(epsilon)
    assert np.linalg.norm(beta(wf, epsilon)) < 1e-12
    assert np.all(np.linalg.eigvals(jacobian(gaussian, epsilon)) > 0)
    assert np.prod(np.linalg.eigvals(jacobian(wf, epsilon))) < 0
    assert np.linalg.norm(left[-1]) < 1e-9

    curve = np.concatenate([left[::-1], wf[None], right])
    curve = curve[np.argsort(curve[:, 1])]
    unique = np.concatenate([[True], np.diff(curve[:, 1]) > 1e-14])
    curve = curve[unique]
    u = np.linspace(0, 0.8, 281)
    r = np.linspace(-0.72, 0.35, 281)
    U, R = np.meshgrid(u, r)
    field = beta(np.stack([R, U], axis=-1), epsilon)
    dr, du = field[..., 0], field[..., 1]
    speed = np.hypot(du, dr)
    scale = np.maximum(speed, 1e-12)
    critical_r = np.interp(u, curve[:, 1], curve[:, 0])

    fig, ax = plt.subplots(figsize=(12.4, 9.0))
    fig.subplots_adjust(left=0.105, right=0.97, bottom=0.275, top=0.835)
    fig.suptitle("高斯排斥点与 WF 鞍点：理论 RG 流场示意", fontsize=22, y=0.97)
    fig.text(0.5, 0.909, r"一圈两耦合截断：$d=4-\epsilon$，此处取 $\epsilon=1$（$d=3$），外场为零", ha="center", fontsize=14)
    fig.text(0.5, 0.867, "用于解释流场拓扑；不是二维格点模型的精确流场，也不是 QCR 结果", ha="center", color="#9b4450", fontsize=12)
    ax.fill_between(u, critical_r, r[-1], color="#eaf3fc", zorder=0)
    ax.fill_between(u, r[0], critical_r, color="#fff0e1", zorder=0)
    ax.streamplot(u, r, du / scale, dr / scale, color="#71869c", density=1.20,
                  linewidth=0.85, arrowsize=1.25, minlength=0.14, zorder=1)
    ax.plot(curve[:, 1], curve[:, 0], color="#16805b", lw=3.3, zorder=4)

    def arrow_on_curve(x0: float, x1: float) -> None:
        y0, y1 = np.interp([x0, x1], curve[:, 1], curve[:, 0])
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops={"arrowstyle": "-|>", "color": "#16805b", "lw": 2.8, "mutation_scale": 18}, zorder=5)

    for pair in [(0.045, 0.085), (0.14, 0.18), (0.39, 0.34), (0.68, 0.62)]:
        arrow_on_curve(*pair)

    values, vectors = np.linalg.eig(jacobian(wf, epsilon))
    unstable = vectors[:, int(np.argmax(values))].real
    unstable *= np.sign(unstable[0])
    unstable_paths = []
    for sign in [-1, 1]:
        path = integrate(wf + sign * 1e-4 * unstable, epsilon=epsilon, duration=8)
        unstable_paths.append(path)
        ax.plot(path[:, 1], path[:, 0], color="#b74349", lw=2.2, zorder=4)
        distances = np.linalg.norm(path - wf, axis=1)
        indices = [int(np.argmin(abs(distances - d))) for d in (0.12, 0.19)]
        p0, p1 = path[indices[0]], path[indices[1]]
        ax.annotate("", xy=(p1[1], p1[0]), xytext=(p0[1], p0[0]),
                    arrowprops={"arrowstyle": "-|>", "color": "#b74349", "lw": 2.6, "mutation_scale": 18}, zorder=5)

    ax.plot(0, 0, "o", ms=10, color="#32475e", clip_on=False, zorder=7)
    ax.plot(wf[1], wf[0], marker="*", ms=19, markeredgewidth=0.8,
            markeredgecolor="white", color="#a52678", zorder=7)
    ax.annotate("高斯固定点：排斥点", xy=(0, 0), xytext=(0.035, 0.105), fontsize=13,
                color="#32475e", arrowprops={"arrowstyle": "-", "color": "#32475e"}, zorder=8)
    ax.annotate("WF 固定点：鞍点", xy=(wf[1], wf[0]), xytext=(0.38, -0.035), fontsize=14,
                color="#9c246f", arrowprops={"arrowstyle": "-", "color": "#9c246f"},
                bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#ead5e2"}, zorder=8)
    ax.text(0.58, 0.23, "流向无序相", color="#285f86", fontsize=19, ha="center", zorder=8,
            bbox={"fc": "#eaf3fc", "ec": "none", "alpha": 0.9, "pad": 3})
    ax.text(0.59, -0.59, "流向有序相", color="#a3632e", fontsize=19, ha="center", zorder=8,
            bbox={"fc": "#fff0e1", "ec": "none", "alpha": 0.9, "pad": 3})
    ax.annotate("沿临界轨道向 WF 靠近", xy=(0.12, float(np.interp(0.12, curve[:, 1], curve[:, 0]))),
                xytext=(0.035, -0.34), fontsize=12, color="#0e7954",
                arrowprops={"arrowstyle": "-", "color": "#16805b"},
                bbox={"fc": "white", "ec": "#c5e1d5", "boxstyle": "round,pad=0.3"}, zorder=8)
    ax.set(xlim=(0, 0.8), ylim=(-0.72, 0.35), xlabel=r"重标定四次耦合 $\bar u$", ylabel=r"无量纲质量参数 $\bar r$")
    ax.grid(color="#6b7d90", alpha=0.13, lw=0.6)
    handles = [Line2D([], [], color="#16805b", lw=3, label="临界分离轨道：两侧相的边界"),
               Line2D([], [], color="#b74349", lw=2.3, label="热扰动轨道：远离 WF")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.145), frameon=False, fontsize=12)
    fig.text(0.105, 0.112, r"所用方程：$\dot{\bar r}=2\bar r+\bar u/(1+\bar r)$，$\dot{\bar u}=\epsilon\bar u-3\bar u^2/(1+\bar r)^2$。", fontsize=12)
    fig.text(0.105, 0.075, "箭头指向更大尺度（红外）；背景流线速度归一化，不以箭头长度表示流速。", fontsize=11, color="#52606d")
    fig.text(0.105, 0.039, "坐标经过重标定，不是此前裸参数相图；分离轨道数值积分自 WF 的稳定方向，并非手连直线。", fontsize=10.8, color="#52606d")

    output.mkdir(parents=True)
    for suffix in ["png", "svg"]:
        fig.savefig(output / f"phi4_theoretical_rg.{suffix}", dpi=200)
    plt.close(fig)
    np.savez_compressed(output / "flow_arrays.npz", u=u, r=r, beta_r=dr, beta_u=du,
                        critical_left=left, critical_right=right, fixed_points=np.stack([gaussian, wf]),
                        thermal_negative=unstable_paths[0], thermal_positive=unstable_paths[1])
    report = {
        "kind": "one_loop_RG_truncation_topology_illustration",
        "not_exact_2d_flow": True, "epsilon": epsilon, "dimension": 4 - epsilon,
        "source": SOURCE, "source_section": "3.5",
        "coordinate_order": ["r_bar", "u_bar"],
        "coordinate_definition": "r_bar=mu^2/Lambda^2; u_bar=12*Omega_(d-1)/(2*pi)^d * g_tilde in Tong's g*phi^4 convention",
        "beta_r": "2*r_bar+u_bar/(1+r_bar)",
        "beta_u": "epsilon*u_bar-3*u_bar^2/(1+r_bar)^2",
        "RG_direction": "increasing coarse-graining length / infrared",
        "fixed_points": [
            {"name": "Gaussian", "r_u": gaussian.tolist(), "beta_jacobian_eigenvalues": np.linalg.eigvals(jacobian(gaussian)).tolist(), "classification": "repeller"},
            {"name": "WF_of_truncated_equations", "r_u": wf.tolist(), "beta_jacobian_eigenvalues": values.tolist(), "classification": "saddle"},
        ],
        "eigenvalue_caveat": "These are eigenvalues of the unexpanded truncated equations, not strict-order epsilon results or certified physical exponents.",
        "critical_left_endpoint_norm": float(np.linalg.norm(left[-1])),
        "critical_construction": "backward RK4 integration from +/-1e-6 along the WF contracting eigenvector; step=0.003",
        "new_training": False, "new_physical_sampling": False, "QCR_model_loaded": False,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "outputs_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir())},
    }
    (output / "reference.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output), "WF_r_u": wf.tolist(), "gaussian_connection_residual": report["critical_left_endpoint_norm"]}))


if __name__ == "__main__":
    main()
