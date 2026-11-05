"""Zero-field soft-spin hopping model, using an exact field-coordinate map.

The physical field is ``phi`` and the once-per-forward-periodic-edge action is

    S = -kappa sum_(x,a) phi[x] phi[x+e_a]
        + sum_x (phi[x]**2 + quartic_lambda * (phi[x]**2 - 1)**2).

For kappa>0, psi=sqrt(kappa)*phi maps this action to the unchanged gradient-form
``phi4_hybrid`` kernel with r=2*(1-2*lambda)/kappa-2*d and
u=4*lambda/kappa**2, up to the constant lambda*volume.  HMC is integrated in
psi with fresh unit-mass Gaussian momenta; in physical phi coordinates this is
a mass kappa, not a unit-mass phi integrator.  The fixed step size is therefore
a psi-coordinate step size.  The Jacobian is field independent and cancels in
Metropolis ratios.  Embedded bonds have probability
1-exp(-2*kappa*abs(phi[x]*phi[y])) for aligned signs.

No physical sampling implementation is duplicated or modified here.  The
mapping changes only coordinates, units and receipts.  Samples, raw series and
their diagnostics are returned in physical phi units, with no global Z2
augmentation, parameter adaptation, retry, extra samples or changed seeds.
"""
from __future__ import annotations

import math
import time

import numpy as np

from . import phi4 as base
from . import phi4_hybrid as hybrid


SCHEMA = "euclidean_phi4_hopping_hmc_embedded_wolff_v1"
OBSERVABLES = base.OBSERVABLES
ACTION = (
    "S=-kappa*sum_(x,a)phi[x]*phi[x+e_a]"
    "+sum_x(phi[x]^2+quartic_lambda*(phi[x]^2-1)^2); "
    "each forward periodic edge once; h=0"
)
QUALIFICATION = (
    "周期实标量软自旋h=0；物理作用量为固定局域四次势加最近邻hopping项。"
    "以psi=sqrt(kappa)*phi精确映射到固定HMC-Metropolis与lazy embedded-Ising Wolff核，"
    "积分步长属于psi坐标；物理phi坐标的HMC有效质量为kappa。"
    "构型、原始统计及诊断使用物理phi单位，作用量密度包含quartic_lambda常数项。"
    "无额外全局随机Z2增强；不丢拒绝态，不自适应，不自动补样、换seed或重跑。"
    "有限链ESS、符号变化及热冷比较均不认证平衡；生成参数不是训练监督标签。"
    "kappa=0的独立格点极限不由此奇异坐标映射实现；非零外场未实现。"
)


def gradient_parameters(kappa, quartic_lambda, dimension):
    """Return JSON-safe gradient-kernel parameters and the exact coordinate map.

    Only dimensions 1..3, positive finite kappa and positive finite lambda are
    implemented.  kappa=0 is a valid separate physical model but this coordinate
    map is singular there and is explicitly rejected rather than approximated.
    """
    kappa = base._finite(kappa, "kappa")
    quartic_lambda = base._finite(quartic_lambda, "quartic_lambda")
    dimension = base._integer(dimension, "dimension", 1)
    if dimension > 3:
        raise ValueError("dimension must be 1, 2, or 3")
    if kappa <= 0.:
        raise ValueError("kappa must be positive; the kappa=0 mapping is singular")
    if quartic_lambda <= 0.:
        raise ValueError("quartic_lambda must be positive")
    try:
        # Sequential division avoids needless overflow of kappa**2.
        r = 2. * (1. - 2. * quartic_lambda) / kappa - 2. * dimension
        u = (4. * quartic_lambda / kappa) / kappa
        field_scale = math.sqrt(kappa)
    except (OverflowError, ZeroDivisionError) as exc:
        raise ValueError("Mapped gradient parameters are outside finite float64 range") from exc
    if not all(math.isfinite(value) for value in (r, u, field_scale)) or u <= 0.:
        raise ValueError("Mapped gradient parameters require finite r and positive finite u")
    return dict(kappa=kappa, quartic_lambda=quartic_lambda, dimension=dimension,
        r=float(r), u=float(u), h=0., field_scale=float(field_scale),
        coordinate_relation="psi=sqrt(kappa)*phi",
        action_relation="S_gradient(psi)=S_hopping(phi)-quartic_lambda*volume",
        physical_action_constant_density=quartic_lambda,
        field_independent_log_measure_jacobian_per_site=-.5 * math.log(kappa),
        integrator_coordinate="psi", integrator_mass_psi=1.,
        physical_effective_mass_phi=kappa,
        embedded_bond_probability="aligned signs: -expm1(-2*kappa*abs(phi_i*phi_j))")


def action_and_gradient(phi, kappa, quartic_lambda):
    """Compute the physical hopping action/gradient independently of the map.

    ``phi`` has 1..3 spatial axes, no channel axis and side lengths >=3.  Neither
    the field nor a caller-owned array is mutated.  The additive lambda*volume
    term is retained so action_density has the stated physical convention.
    """
    phi = base._field(phi)
    mapping = gradient_parameters(kappa, quartic_lambda, phi.ndim)
    kappa, quartic_lambda = mapping["kappa"], mapping["quartic_lambda"]
    with np.errstate(over="raise", invalid="raise"):
        square = phi * phi
        action = np.sum(square + quartic_lambda * (square - 1.) ** 2)
        gradient = 2. * phi + 4. * quartic_lambda * phi * (square - 1.)
        for axis in range(phi.ndim):
            forward = np.roll(phi, -1, axis=axis)
            backward = np.roll(phi, 1, axis=axis)
            action -= kappa * np.sum(phi * forward)
            gradient = gradient - kappa * (forward + backward)
    if not np.isfinite(action) or not np.isfinite(gradient).all():
        raise FloatingPointError("Nonfinite hopping action or force; parameters were not changed")
    return float(action), gradient


def _physical_observations(raw_psi, mapping):
    """Transform every named raw series; do not mutate the kernel's arrays."""
    raw = np.array(raw_psi, dtype=np.float64, copy=True)
    if raw.ndim != 2 or raw.shape[1] != len(OBSERVABLES):
        raise ValueError("Gradient kernel returned an incompatible raw-observable array")
    scale, kappa = mapping["field_scale"], mapping["kappa"]
    raw[:, 0] += mapping["quartic_lambda"]
    raw[:, 1:3] /= scale
    raw[:, (3, 5, 6, 7)] /= kappa
    # Sequential divisions avoid overflow when kappa itself is representable.
    raw[:, 4] /= kappa
    raw[:, 4] /= kappa
    if not np.isfinite(raw).all():
        raise FloatingPointError("Nonfinite physical observables after exact field conversion")
    return raw


def sample_chain(kappa, quartic_lambda, *, size, dimension=2, samples, burn_in,
                 spacing, step_size, leapfrog_steps, seed, initial="hot", h=0.,
                 cluster_probability=.5):
    """Run the unchanged fixed hybrid kernel and return physical-phi arrays.

    Work counts, HMC accept/reject decisions and delta_h are those of the
    underlying psi-coordinate trajectory, not newly generated proposals.
    Initial states likewise follow the unchanged psi-kernel convention; the
    receipt explicitly records their physical scaling.  Diagnostics are
    recomputed from physical saved series, before any postprocessing.
    """
    wall, cpu = time.perf_counter(), time.process_time()
    mapping = gradient_parameters(kappa, quartic_lambda, dimension)
    h = base._finite(h, "h")
    if h != 0.:
        raise ValueError("Hopping embedded sign-cluster sampler requires h=0")
    kernel = hybrid.sample_hybrid_chain(mapping["r"], mapping["u"], size=size,
        dimension=mapping["dimension"], samples=samples, burn_in=burn_in,
        spacing=spacing, step_size=step_size, leapfrog_steps=leapfrog_steps,
        seed=seed, initial=initial, h=0., cluster_probability=cluster_probability)
    if kernel["raw_observable_names"] != list(OBSERVABLES):
        raise ValueError("Gradient kernel observable order changed")
    configurations = kernel["configurations"] / mapping["field_scale"]
    trajectory = kernel["trajectory_magnetization"] / mapping["field_scale"]
    if not np.isfinite(configurations).all() or not np.isfinite(trajectory).all():
        raise FloatingPointError("Nonfinite physical field after exact field conversion")
    raw = _physical_observations(kernel["raw_observables"], mapping)
    diagnostics = {name: base.series_diagnostics(raw[:, index])
                   for index, name in enumerate(OBSERVABLES)}
    warnings = list(kernel["warnings"])
    for name, row in diagnostics.items():
        if row.get("warning") or row.get("ess") is None or row["ess"] < 32:
            warning = f"{name}: {row.get('warning') or 'ESS below 32 saved draws'}"
            if warning not in warnings:
                warnings.append(warning)
    result = dict(kernel)
    result.update(schema=SCHEMA, kernel_schema=hybrid.SCHEMA, action=ACTION,
        configurations=configurations, raw_observables=raw,
        trajectory_magnetization=trajectory, diagnostics=diagnostics,
        warnings=warnings, qualification=QUALIFICATION, mapping=mapping,
        physical_field_coordinate="phi", raw_observable_coordinate="physical_phi",
        integrator_coordinate="psi", integrator_mass_psi=1.,
        physical_effective_mass_phi=mapping["kappa"],
        step_size_coordinate="psi=sqrt(kappa)*phi",
        initial_state_coordinate="unchanged gradient kernel psi; phi=psi/sqrt(kappa)",
        default_lazy_cluster_probability=.5,
        lazy_cluster_skips=int(kernel["hybrid_cycles"] - kernel["wolff_cluster_updates"]),
        kernel_wall_seconds=kernel["wall_seconds"],
        kernel_cpu_seconds=kernel["cpu_seconds"],
        wall_seconds=time.perf_counter() - wall, cpu_seconds=time.process_time() - cpu)
    return result
