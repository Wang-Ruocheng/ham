"""Bounded, label-free support analysis of an already trained two-coordinate R.

This module neither trains models nor uses r/u labels to choose bounds, starts,
roots or eigenmodes.  Numerical fixed points are not physical certifications.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull, QhullError


FLOW_SCHEMA = "phi4_ru_bounded_flow_v1"


def numerical_rank(points, relative_tolerance=1e-8):
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return 0
    singular = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    return int(np.sum(singular > max(float(singular[0]) * relative_tolerance,
                                     np.finfo(np.float64).eps * 32)))


def classify_eigenvalues(eigenvalues, tolerance=1e-5):
    modulus = np.abs(np.asarray(eigenvalues))
    contractions = int(np.sum(modulus < 1 - tolerance))
    expansions = int(np.sum(modulus > 1 + tolerance))
    if contractions == 2:
        name = "attracting"
    elif expansions == 2:
        name = "repelling"
    elif contractions == 1 and expansions == 1:
        name = "saddle"
    else:
        name = "neutral_or_unresolved"
    return name, contractions, expansions


def full_map_and_jacobian(model, points, device):
    """Compute DR, including the identity in residual advance(g)=g+R_net(g)."""
    x = torch.as_tensor(points, device=device, dtype=torch.float64).detach().requires_grad_(True)
    mapped = model.advance(x)
    if mapped.shape != x.shape or x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("Two-coordinate pointwise advance is required")
    # Each output row depends only on its own input row in this MLP.  Summing
    # rows differentiates all pointwise Jacobians without quadratic batch work.
    jacobian = torch.stack([torch.autograd.grad(mapped[:, axis].sum(), x,
        retain_graph=axis == 0)[0] for axis in range(2)], dim=1)
    result = mapped.detach().cpu().numpy(), jacobian.detach().cpu().numpy()
    if not all(np.isfinite(value).all() for value in result):
        raise FloatingPointError("Nonfinite frozen R or full DR")
    return result


def bounded_flow(model, arrays, device, *, budget_check=None, grid_side=33,
                 seed_side=7, max_nfev=64):
    """Populate public flow arrays and return roots plus bounded-search receipt.

    Fixed starts are a 7-by-7 grid restricted to the TRAIN-input convex hull;
    at most 49 starts are attempted.  TEST and labels enter only post-hoc root
    support descriptions.  Rank-deficient TRAIN support is reported, never
    repaired by pseudoinverse or by adding roots outside the bounded search.
    """
    if grid_side != 33 or seed_side != 7 or max_nfev != 64:
        raise ValueError("The frozen ru flow plan requires 33/7/64 bounds")
    train_direct = np.asarray(arrays["train_direct"], dtype=np.float64)
    if train_direct.ndim != 3 or train_direct.shape[1:] != (3, 2):
        raise ValueError("TRAIN direct encodings must have shape [chains,3,2]")
    train = train_direct[:, :2].reshape(-1, 2)
    if not len(train) or not np.isfinite(train).all():
        raise ValueError("Nonempty finite TRAIN flow inputs required")
    mean = train.mean(axis=0)
    centered = train - mean
    _, singular, components = np.linalg.svd(centered, full_matrices=False)
    variances = singular ** 2 / len(train)
    ratios = variances / variances.sum() if variances.sum() > 0 else np.zeros(2)
    rank = numerical_rank(train)
    low, high = train.min(axis=0), train.max(axis=0)
    width = np.maximum(high - low, 1e-6)
    bounds = np.column_stack([low - .1 * width, high + .1 * width])
    center, halfwidth = bounds.mean(axis=1), np.diff(bounds, axis=1)[:, 0] / 2
    normalized_train = (train - center) / halfwidth
    hull, hull_error = None, None
    if rank == 2 and len(np.unique(train, axis=0)) >= 3:
        try:
            hull = ConvexHull(normalized_train)
        except QhullError as exc:
            hull_error = str(exc).splitlines()[0]

    def in_hull(points):
        points = np.atleast_2d(points)
        if hull is None:
            return np.zeros(len(points), dtype=bool)
        normalized = (points - center) / halfwidth
        return np.all(normalized @ hull.equations[:, :2].T + hull.equations[:, 2] <= 1e-9, axis=1)

    grid = np.stack(np.meshgrid(*[np.linspace(*bound, grid_side) for bound in bounds],
                                indexing="xy"), axis=-1).reshape(-1, 2)
    mapped_parts, derivative_parts = [], []
    for offset in range(0, len(grid), 128):
        if budget_check is not None:
            budget_check()
        mapped, derivative = full_map_and_jacobian(model, grid[offset:offset + 128], device)
        mapped_parts.append(mapped)
        derivative_parts.append(derivative)
    mapped, derivative = np.concatenate(mapped_parts), np.concatenate(derivative_parts)
    grid_eigenvalues = np.linalg.eigvals(derivative)
    arrays.update(flow_g=grid, flow_R=mapped, flow_displacement=mapped - grid,
        flow_DR=derivative, flow_eigenvalues_real=grid_eigenvalues.real,
        flow_eigenvalues_imag=grid_eigenvalues.imag, flow_in_train_hull=in_hull(grid),
        flow_bounds=bounds, flow_train_mean=mean, flow_pca_components=components,
        flow_pca_explained_variance_ratio=ratios)
    starts = np.stack(np.meshgrid(*[np.linspace(low[axis], high[axis], seed_side)
        for axis in range(2)], indexing="xy"), axis=-1).reshape(-1, 2)
    starts = starts[in_hull(starts)]
    roots, attempts = [], []
    residual_tolerance = 1e-8 * max(1., float(np.linalg.norm(width)))

    def function(z):
        if budget_check is not None:
            budget_check()
        point = center + halfwidth * z
        with torch.no_grad():
            x = torch.as_tensor(point[None], device=device, dtype=torch.float64)
            value = model.advance(x).detach().cpu().numpy()[0] - point
        if not np.isfinite(value).all():
            raise FloatingPointError("Nonfinite bounded root residual")
        return value / halfwidth

    def jacobian(z):
        if budget_check is not None:
            budget_check()
        _, dr = full_map_and_jacobian(model, (center + halfwidth * z)[None], device)
        return (dr[0] - np.eye(2)) * halfwidth[None, :] / halfwidth[:, None]

    for start in starts:
        result = least_squares(function, (start - center) / halfwidth, jac=jacobian,
            bounds=(-np.ones(2), np.ones(2)), max_nfev=max_nfev,
            xtol=1e-11, ftol=1e-11, gtol=1e-11)
        point = center + halfwidth * result.x
        residual = float(np.linalg.norm(function(result.x) * halfwidth))
        accepted = bool(residual <= residual_tolerance)
        attempts.append(dict(start=start.tolist(), nfev=int(result.nfev),
            njev=int(result.njev or 0), success=bool(result.success),
            status=int(result.status), residual=residual, accepted=accepted))
        if not accepted or any(np.linalg.norm((point - np.asarray(root["g"])) / halfwidth) < 1e-5
                               for root in roots):
            continue
        _, dr = full_map_and_jacobian(model, point[None], device)
        eigenvalues = np.linalg.eigvals(dr[0])
        classification, contractions, expansions = classify_eigenvalues(eigenvalues)
        distance = np.linalg.norm(train - point, axis=1)
        nearby = train[np.argsort(distance, kind="stable")[:min(16, len(train))]]
        support = []
        for split in ("train", "test"):
            if f"{split}_direct" not in arrays:
                continue
            for scale in range(3):
                sample = arrays[f"{split}_direct"][:, scale]
                support.append(dict(split=split, scale=scale,
                    within_coordinate_ranges=bool(np.all(point >= sample.min(axis=0))
                                                   and np.all(point <= sample.max(axis=0))),
                    nearest_distance=float(np.min(np.linalg.norm(sample - point, axis=1)))))
        roots.append(dict(g=point.tolist(), residual=residual, full_DR=dr[0].tolist(),
            eigenvalues=[dict(real=float(value.real), imag=float(value.imag),
                             modulus=float(abs(value))) for value in eigenvalues],
            classification=classification, contracting_directions=contractions,
            expanding_directions=expansions, within_train_hull=bool(in_hull(point)[0]),
            within_search_bounds=bool(np.all(point >= bounds[:, 0]) and np.all(point <= bounds[:, 1])),
            nearest_train_distance=float(distance.min()), local_rank=numerical_rank(nearby),
            local_rank_points=int(len(nearby)), data_nearby=support,
            physical_critical_point_certified=False, thermal_nu=None))
    roots.sort(key=lambda root: tuple(root["g"]))
    receipt = dict(schema=FLOW_SCHEMA, train_inputs_only_for_bounds_hull_and_pca=True,
        input_scales=[0, 1], train_input_count=int(len(train)), train_input_rank=rank,
        rank_relative_tolerance=1e-8, pca_mean=mean.tolist(),
        pca_components=components.tolist(), pca_variances=variances.tolist(),
        pca_explained_variance_ratio=ratios.tolist(), bounds=bounds.tolist(),
        bound_padding_fraction=.1, coordinate_width_floor=1e-6,
        grid_side=grid_side, grid_points=int(len(grid)),
        hull_available=hull is not None, hull_error=hull_error,
        support_qualification=("Full numerical rank does not certify sufficient physical support"
            if hull is not None else "Insufficient two-dimensional TRAIN support; no root search"),
        starts_rule="fixed 7x7 TRAIN bounding-box grid, restricted to TRAIN convex hull",
        maximum_starts=49, actual_starts=int(len(starts)), max_nfev_per_start=max_nfev,
        residual_tolerance=residual_tolerance, deduplication_normalized_tolerance=1e-5,
        root_search_attempts=attempts, full_DR_includes_identity=True,
        all_spectra_preserved=True, eigenmode_selection="none", thermal_nu=None,
        physical_critical_point_certified=False)
    return roots, receipt
