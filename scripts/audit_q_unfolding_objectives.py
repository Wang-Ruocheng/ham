"""Posthoc, label-free audit of the paired Q-unfolding experiment.

This does not train networks or optimize any test-set quantity. It verifies
decoder scales, compares linear reconstruction with the train-only PCA
reference, measures the subset Jensen gap, and fits a single positive latent
scale on TRAIN to evaluate the original Q objective in common fixed units.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
import torch

from experiments.q_only_candidates import OPTIONS, validate_frozen_levels
from recurrent_rg.data.statistics import categorical_plaquette_distribution
from recurrent_rg.losses import distribution_geometry_loss
from recurrent_rg.runner import file_fingerprint


ARMS = (
    "original",
    "graph_coordinate",
    "neural_linear_reconstruction",
    "histogram_linear_reconstruction",
)


@torch.no_grad()
def audit(run: Path, cache: Path, output: Path):
    torch.set_num_threads(2)
    manifest = json.loads((run / "manifest.json").read_text())
    cache_manifest = json.loads((cache / "manifest.json").read_text())
    cache_file = cache / "frozen_levels.pt"
    if file_fingerprint(cache_file) != manifest["cache_sha256"]:
        raise ValueError("Experiment and diagnostic cache fingerprints differ")
    if manifest["data_sha256"] != cache_manifest["data_sha256"]:
        raise ValueError("Experiment and diagnostic data fingerprints differ")
    levels = torch.load(cache_file, weights_only=True)
    descriptors = {}
    half_descriptors = {}
    for split, values in levels.items():
        validate_frozen_levels(values)
        if any(v.shape[1] != 64 for v in values):
            raise ValueError("This paired audit requires the recorded 64-sample ensembles")
        descriptors[split] = torch.stack(
            [categorical_plaquette_distribution(v.double()) for v in values], dim=1
        )
        half_descriptors[split] = [
            (
                categorical_plaquette_distribution(v[:, :32].double()),
                categorical_plaquette_distribution(v[:, 32:].double()),
            )
            for v in values
        ]
    del levels

    encodings_file = run / "encodings.npz"
    encodings = np.load(encodings_file)
    chart_file = run / "data_chart.pt"
    chart = torch.load(chart_file, weights_only=True)
    center, axis = chart["PCA_center"], chart["PCA_axis"]
    actual_center = descriptors["train"].reshape(-1, len(center)).mean(0)
    torch.testing.assert_close(actual_center, center, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(axis.norm(), torch.ones((), dtype=axis.dtype))

    pca = {}
    for split, p in descriptors.items():
        coordinate = (p - center) @ axis
        reconstructed = center + coordinate[..., None] * axis
        errors = (p - reconstructed).square().sum(-1)
        constant = (p - center).square().sum(-1)
        pca[split] = dict(
            sum_bin_MSE=float(errors.mean()),
            per_scale_MSE=errors.mean(0).tolist(),
            constant_reference_MSE=float(constant.mean()),
            normalized_RMSE=float((errors.mean() / constant.mean()).sqrt()),
        )

    jensen, reconstruction, calibration = {}, {}, {}
    for name in ARMS:
        halves = {
            split: torch.from_numpy(encodings[f"{name}_{split}_halves"])
            for split in descriptors
        }
        full = {
            split: torch.from_numpy(encodings[f"{name}_{split}"])
            for split in descriptors
        }
        jensen[name] = {}
        for split in descriptors:
            delta = full[split] - halves[split].mean(-1)
            rms = delta.square().mean(0).sqrt()
            std = full[split].std(0, correction=0)
            jensen[name][split] = dict(
                per_scale_RMS=rms.tolist(),
                per_scale_relative_RMS=(rms / std.clamp_min(1e-12)).tolist(),
            )

        def original_terms(split, scale):
            h = halves[split] * scale
            g = h.mean(-1)[..., None]
            anchor = (h[:, 0, 0] - h[:, 0, 1]).square().mean()
            coarse = (h[:, 1:, 0] - h[:, 1:, 1]).square().mean()
            geometry = torch.stack(
                [
                    distribution_geometry_loss(g[:, k], pa, pb, **OPTIONS)[0]
                    for k, (pa, pb) in enumerate(half_descriptors[split])
                ]
            ).mean()
            return dict(
                total=float(10 * geometry + 20 * anchor + 5 * coarse),
                geometry_contribution=float(10 * geometry),
                micro_subset_contribution=float(20 * anchor),
                coarse_subset_contribution=float(5 * coarse),
            )

        # One positive scalar shared by every scale. No coupling labels and
        # no test objective are available to the calibration optimizer.
        optimum = minimize_scalar(
            lambda logscale: original_terms("train", math.exp(logscale))["total"],
            bounds=(-8.0, 8.0),
            method="bounded",
            options={"xatol": 1e-10},
        )
        if not optimum.success or abs(optimum.x) > 7.99:
            raise ValueError("Train-only scalar calibration did not find an interior optimum")
        scale = math.exp(optimum.x)
        calibration[name] = dict(
            scale_fitted_on="train",
            positive_scale=scale,
            raw={split: original_terms(split, 1.0) for split in descriptors},
            calibrated={split: original_terms(split, scale) for split in descriptors},
            optimizer_success=bool(optimum.success),
        )

        if "reconstruction" not in name:
            continue
        payload = torch.load(run / f"{name}.pt", weights_only=True)
        weight = payload["decoder"]["weight"]
        bias = payload["decoder"]["bias"]
        if weight.shape != (len(center), 1):
            raise ValueError("This audit expects a scalar affine reconstruction decoder")
        reconstruction[name] = dict(
            decoder_column_norm=weight.norm(dim=0).tolist(),
            decoder_PCA_axis_absolute_cosine=float(
                abs(weight[:, 0] @ axis) / weight[:, 0].norm()
            ),
            splits={},
        )
        for split, p in descriptors.items():
            reconstruction[name]["splits"][split] = {}
            for kind, g in (("full", full[split]), ("mean_half", halves[split].mean(-1))):
                prediction = g[..., None] * weight.T + bias
                error = (prediction - p).square().sum(-1)
                reconstruction[name]["splits"][split][kind] = dict(
                    sum_bin_MSE=float(error.mean()),
                    per_scale_MSE=error.mean(0).tolist(),
                    PCA_residual_ratio=float(error.mean()) / pca[split]["sum_bin_MSE"],
                    normalized_RMSE=math.sqrt(
                        float(error.mean()) / pca[split]["constant_reference_MSE"]
                    ),
                )

    # The experiment source Q changed before this experiment, but C must
    # still match the checkpoint that produced the shared trajectory cache.
    source_payload = torch.load(manifest["source"], weights_only=True)
    cache_source_payload = torch.load(cache_manifest["source_checkpoint"], weights_only=True)
    source_state = source_payload.get("model", source_payload)
    cache_source_state = cache_source_payload.get("model", cache_source_payload)
    c_difference = max(
        float((source_state[k] - cache_source_state[k]).abs().max())
        for k in source_state if k.startswith("rg_kernel.")
    )
    if c_difference != 0:
        raise ValueError("Source C differs from the checkpoint producing the cache")

    result = dict(
        run=str(run),
        cache=str(cache),
        fingerprints=dict(
            manifest=file_fingerprint(run / "manifest.json"),
            encodings=file_fingerprint(encodings_file),
            cache=file_fingerprint(cache_file),
            chart=file_fingerprint(chart_file),
        ),
        protocol=dict(
            optimization="Posthoc positive scalar fit on TRAIN only; no network updates",
            objective="10 relative_local geometry +20 microscopic subset MSE +5 coarse subset MSE",
            geometry_options=OPTIONS,
            calibration_log_scale_bounds=[-8.0, 8.0],
            labels_used=False,
            test_used_in_calibration=False,
            source_C_max_difference=c_difference,
        ),
        PCA_reference=pca,
        reconstruction=reconstruction,
        subset_Jensen_gap=jensen,
        original_objective_calibration=calibration,
        interpretation=(
            "After train-only scale calibration, unfolded representations have lower ORIGINAL "
            "Q objective than the trained folded baseline. This supports an optimization "
            "barrier or attraction-basin explanation; it does not establish a global optimum "
            "of the neural-network objective or RG closure."
        ),
    )
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(output), "original_objective_calibration": calibration}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    audit(args.run, args.cache, args.output or args.run / "objective_audit.json")
