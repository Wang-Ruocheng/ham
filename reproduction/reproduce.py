#!/usr/bin/env python3
"""Mac-compatible validation of the released FP action and PRL AIC analysis."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MY_NUMBA_TARGET", "python")
os.environ.setdefault("GAUGE_GROUP", "su3")
os.environ.setdefault("PRECISION", "single")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "third_party" / "lge-cnn"
RESULTS = ROOT / "results" / "reproduction"
CHECKPOINT = UPSTREAM / "fpaction" / "finetuned_model_v01.ckpt"
TABLE = ROOT / "reproduction" / "data" / "fp_observables.csv"
sys.path.insert(0, str(UPSTREAM))

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import norm


def traceless_hermitian(shape: tuple[int, ...], scale: float) -> torch.Tensor:
    x = torch.randn(*shape, 3, 3) + 1j * torch.randn(*shape, 3, 3)
    h = 0.5 * (x + x.mH)
    identity = torch.eye(3, dtype=h.dtype)
    trace = h.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None]
    return scale * (h - trace * identity / 3.0)


def random_su3(shape: tuple[int, ...], scale: float) -> torch.Tensor:
    return torch.matrix_exp(1j * traceless_hermitian(shape, scale)).to(torch.complex64)


def gauge_transform(u: torch.Tensor, omega: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    """U_mu(x) -> Omega(x) U_mu(x) Omega(x+mu)^dagger."""
    batch, _, directions, nc, _ = u.shape
    grid = u.reshape(batch, *dims, directions, nc, nc)
    transformed = torch.empty_like(grid)
    for mu in range(directions):
        omega_plus = torch.roll(omega, shifts=-1, dims=mu + 1)
        transformed[..., mu, :, :] = omega @ grid[..., mu, :, :] @ omega_plus.mH
    return transformed.reshape_as(u)


def load_released_model():
    from fpaction.models import RegularFPActionModelLocal3

    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing released checkpoint: {CHECKPOINT}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        model = RegularFPActionModelLocal3.load_from_checkpoint(
            str(CHECKPOINT),
            map_location="cpu",
            layer_implementation="wilsonline",
        )
    # Upstream creates this attribute only in its CUDA branch but reads it in
    # forward() for every implementation. This compatibility assignment does
    # not alter the architecture or its weights.
    if not hasattr(model, "summed_clover"):
        model.summed_clover = None
    return model.cpu().eval()


def action(model, u: torch.Tensor) -> torch.Tensor:
    return model(torch.view_as_real(u))


def force(model, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    value, derivative_ri = model.group_derivative(torch.view_as_real(u), use_fast=True)
    return value, torch.view_as_complex(derivative_ri)


def rk3_flow_step(model, u: torch.Tensor, dt: float, prefactor: float = 2.0) -> torch.Tensor:
    """Third-order Lie-group RK step matching upstream hmc.flow_step_rk3."""
    _, d0 = force(model, u)
    z0 = -1j * prefactor * dt * d0
    y = 0.25 * z0
    u1 = torch.matrix_exp(y) @ u

    _, d1 = force(model, u1)
    z1 = -1j * prefactor * dt * d1
    y = (8.0 / 9.0) * z1 - (17.0 / 36.0) * z0
    u2 = torch.matrix_exp(y) @ u1

    _, d2 = force(model, u2)
    z2 = -1j * prefactor * dt * d2
    y = 0.75 * z2 - y
    return torch.matrix_exp(y) @ u2


def run_pretrained_validation(seed: int = 20260911, flow_steps: int = 20, dt: float = 0.01) -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    model = load_released_model()
    dims = tuple(int(x) for x in model.hparams.dims)
    n_sites = int(np.prod(dims))
    identity = torch.eye(3, dtype=torch.complex64)

    vacuum = identity.reshape(1, 1, 1, 3, 3).expand(1, n_sites, 4, 3, 3).clone()
    with torch.no_grad():
        vacuum_action = float(action(model, vacuum).item())

    u = random_su3((1, n_sites, 4), scale=0.20)
    omega = random_su3((1, *dims), scale=1.0)
    u_gauge = gauge_transform(u, omega, dims)
    with torch.no_grad():
        a = float(action(model, u).item())
        a_gauge = float(action(model, u_gauge).item())
    gauge_abs = abs(a_gauge - a)
    gauge_rel = gauge_abs / max(abs(a), 1e-12)

    # Check the group derivative along exp(i eps H) U at one link.
    _, d_action = force(model, u)
    h = traceless_hermitian((1,), scale=1.0)[0].to(torch.complex64)
    epsilon = 0.03
    fd_values = []
    for sign in (-1.0, 1.0):
        perturbed = u.clone()
        perturbed[:, 0, 0] = torch.matrix_exp(1j * sign * epsilon * h) @ perturbed[:, 0, 0]
        with torch.no_grad():
            fd_values.append(float(action(model, perturbed).item()))
    finite_difference = (fd_values[1] - fd_values[0]) / (2.0 * epsilon)
    force_contraction = float((2.0 * torch.trace(h @ d_action[0, 0, 0]).real).item())
    force_rel = abs(finite_difference - force_contraction) / max(abs(finite_difference), 1e-12)

    # A short classically-perfect gradient flow using the released action.
    flow_u = random_su3((1, n_sites, 4), scale=0.25)
    flow_actions = []
    flow_times = []
    start = time.perf_counter()
    for step in range(flow_steps + 1):
        with torch.no_grad():
            flow_actions.append(float(action(model, flow_u).item()))
        flow_times.append(step * dt)
        if step < flow_steps:
            flow_u = rk3_flow_step(model, flow_u, dt)
    runtime = time.perf_counter() - start

    unit_error = float((flow_u.mH @ flow_u - identity).abs().max().item())
    determinant_error = float((torch.linalg.det(flow_u) - 1.0).abs().max().item())
    action_increases = int(np.sum(np.diff(flow_actions) > 2e-4))

    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    ax.plot(flow_times, flow_actions, "o-", ms=3.5, lw=1.5, color="#315da8")
    ax.set(xlabel="flow time t", ylabel=r"learned action $A_{FP}[U(t)]$",
           title="Released fixed-point action: short RK3 gradient flow")
    ax.grid(alpha=0.25)
    fig.savefig(RESULTS / "fp_gradient_flow.png", dpi=180)
    plt.close(fig)

    report = {
        "seed": seed,
        "device": "cpu",
        "reason_for_cpu": "upstream CUDA kernels and unsupported complex compiled operators on MPS",
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "architecture": {
            "lattice_dims": list(dims),
            "conv_kernel_size": list(model.hparams.conv_kernel_size),
            "conv_channels": list(model.hparams.conv_channels),
            "layer_implementation": "wilsonline (pure PyTorch equivalent of checkpoint CUDA path)",
            "parameters": int(sum(p.numel() for p in model.parameters())),
            "checkpoint_epoch": 99,
            "checkpoint_global_step": 32000,
        },
        "checks": {
            "vacuum_action": vacuum_action,
            "gauge_action_before": a,
            "gauge_action_after": a_gauge,
            "gauge_absolute_error": gauge_abs,
            "gauge_relative_error": gauge_rel,
            "finite_difference_derivative": finite_difference,
            "network_force_contraction": force_contraction,
            "force_relative_error": force_rel,
            "flow_initial_action": flow_actions[0],
            "flow_final_action": flow_actions[-1],
            "flow_action_increase_count": action_increases,
            "flow_unitarity_max_error": unit_error,
            "flow_determinant_max_error": determinant_error,
            "flow_runtime_seconds": runtime,
        },
        "pass": {
            "vacuum": abs(vacuum_action) < 1e-6,
            "gauge_invariance": gauge_rel < 2e-5,
            "force": force_rel < 5e-3,
            "gradient_flow": flow_actions[-1] < flow_actions[0] and action_increases == 0,
            "su3_manifold": unit_error < 1e-4 and determinant_error < 1e-4,
        },
    }
    (RESULTS / "pretrained_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


@dataclass
class Fit:
    degree: int
    n_points: int
    intercept: float
    sigma: float
    chi2: float
    aic: float
    weight: float = 0.0
    coefficients: tuple[float, ...] = ()


def aic_fits(x: np.ndarray, y: np.ndarray, error: np.ndarray) -> list[Fit]:
    order = np.argsort(x)
    x, y, error = x[order], y[order], error[order]
    n_total = len(x)
    fits: list[Fit] = []
    for degree in (0, 1, 2):
        k = degree + 1
        for n_points in range(max(k + 1, 3), n_total + 1):
            xx, yy, ee = x[:n_points], y[:n_points], error[:n_points]
            design = np.vander(xx, N=k, increasing=True)
            weighted_design = design / ee[:, None]
            normal = weighted_design.T @ weighted_design
            covariance = np.linalg.inv(normal)
            coefficients = covariance @ (design.T @ (yy / ee**2))
            residual = (yy - design @ coefficients) / ee
            chi2 = float(residual @ residual)
            aic = chi2 + 2 * k + 2 * (n_total - n_points)
            fits.append(Fit(degree, n_points, float(coefficients[0]),
                            float(np.sqrt(covariance[0, 0])), chi2, aic,
                            coefficients=tuple(float(v) for v in coefficients)))
    raw_weights = np.exp(-0.5 * (np.array([f.aic for f in fits]) - min(f.aic for f in fits)))
    raw_weights /= raw_weights.sum()
    for fit, weight in zip(fits, raw_weights):
        fit.weight = float(weight)
    return fits


def mixture_quantiles(fits: list[Fit]) -> tuple[float, float, float]:
    means = np.array([f.intercept for f in fits])
    sigmas = np.array([f.sigma for f in fits])
    weights = np.array([f.weight for f in fits])
    lo = float(np.min(means - 8 * sigmas))
    hi = float(np.max(means + 8 * sigmas))
    grid = np.linspace(lo, hi, 100_000)
    cdf = np.sum(weights[:, None] * norm.cdf((grid[None] - means[:, None]) / sigmas[:, None]), axis=0)
    return tuple(float(v) for v in np.interp((0.16, 0.50, 0.84), cdf, grid))


def read_table() -> dict[str, np.ndarray]:
    with TABLE.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {key: np.array([float(row[key]) for row in rows]) for key in rows[0]}


def run_continuum_analysis() -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    data = read_table()
    x = 1.0 / data["t03_a2"]
    specs = [
        ("t03_w03", r"$t_{0.3}/w_{0.3}^2$", 0.9553),
        ("t04_w04", r"$t_{0.4}/w_{0.4}^2$", 1.0305),
        ("t05_w05", r"$t_{0.5}/w_{0.5}^2$", 1.0726),
        ("t04_t03", r"$t_{0.4}/t_{0.3}$", 1.3356),
        ("t05_t03", r"$t_{0.5}/t_{0.3}$", 1.6443),
        ("beta_fn", r"$\beta(g^2=15.79)$", 14.853),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 11.2), constrained_layout=True)
    report: dict[str, object] = {
        "method": "weighted polynomial fits of degree 0-2 with AIC = chi2 + 2k + 2(n_total-n)",
        "note": "deterministic Gaussian-mixture analogue of the paper's 2000-bootstrap analysis",
        "observables": {},
    }
    x_line = np.linspace(0.0, max(x) * 1.03, 250)
    for ax, (key, label, paper_value) in zip(axes.flat, specs):
        y, error = data[key], data[f"{key}_err"]
        fits = aic_fits(x, y, error)
        q16, median, q84 = mixture_quantiles(fits)
        top_fits = sorted(fits, key=lambda item: item.weight, reverse=True)[:20]
        for fit in top_fits:
            coefficients = np.array(fit.coefficients)
            curve = np.vander(x_line, N=fit.degree + 1, increasing=True) @ coefficients
            ax.plot(x_line, curve, color="#315da8", alpha=min(0.65, 0.08 + 2.5 * fit.weight), lw=1.0)
        ax.errorbar(x, y, yerr=error, fmt="o", ms=3.8, capsize=2,
                    color="#174a96", ecolor="#7695c2", label="FP data")
        ax.errorbar([0], [median], yerr=[[median - q16], [q84 - median]],
                    fmt="s", ms=5, capsize=3, color="#c44e52", label="AIC limit")
        ax.axhline(paper_value, color="#dd8452", ls="--", lw=1.2, label="paper Table IV")
        ax.set(xlabel=r"$a^2/t_{0.3}$", ylabel=label)
        ax.grid(alpha=0.22)
        ax.legend(fontsize=8)
        report["observables"][key] = {
            "reproduced_median": median,
            "reproduced_minus": median - q16,
            "reproduced_plus": q84 - median,
            "paper_table_iv": paper_value,
            "absolute_difference": abs(median - paper_value),
            "highest_aic_weight": top_fits[0].weight,
            "highest_weight_degree": top_fits[0].degree,
            "highest_weight_n_points": top_fits[0].n_points,
        }
    fig.suptitle("FP continuum extrapolations from PRL Tables I-II", fontsize=14)
    fig.savefig(RESULTS / "fp_continuum_aic.png", dpi=180)
    plt.close(fig)
    (RESULTS / "continuum_aic.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("smoke", "continuum", "all"), nargs="?", default="all")
    parser.add_argument("--flow-steps", type=int, default=20)
    args = parser.parse_args()
    if args.stage in ("smoke", "all"):
        report = run_pretrained_validation(flow_steps=args.flow_steps)
        if not all(report["pass"].values()):
            raise SystemExit("One or more released-model validation checks failed")
    if args.stage in ("continuum", "all"):
        run_continuum_analysis()


if __name__ == "__main__":
    main()
