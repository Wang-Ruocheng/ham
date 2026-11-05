"""Bounded, frozen CPU diagnostics; no optimizer, HMC, root search or updates."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from experiments.phi4_qcr import load_dataset, sha256_file, state_fingerprint
from recurrent_rg.phi4_qcr import Phi4QCR, sample_regions, gaussian_bin_probabilities, smoothed_information


def corr(a, b):
    a, b = a.flatten(), b.flatten()
    a, b = a - a.mean(), b - b.mean()
    denominator = a.square().mean().sqrt() * b.square().mean().sqrt()
    return float((a * b).mean() / denominator) if denominator > 0 else None


def diagnostic(run, *, max_seconds=180.):
    started = time.perf_counter()
    run = Path(run).resolve()
    torch.set_num_threads(1)
    torch.manual_seed(2026102207)
    manifest = json.loads((run / "manifest.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    checkpoint = run / "training_step_3000.pt"
    checkpoint_sha = sha256_file(checkpoint)
    dataset = load_dataset(manifest["data_root"])
    assert dataset.manifest_sha256 == manifest["data_manifest_sha256"]
    model = Phi4QCR(manifest["scale"]).double()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert state["step"] == 3000
    model.load_state_dict(state["model"])
    del state
    model.eval().requires_grad_(False)
    before = state_fingerprint(model)
    assert before == summary["final_state_sha256"]
    generator = torch.Generator().manual_seed(2026102207)
    features, predictions, encodings, c_rows, mi_rows = [], [], [], [], []
    with torch.no_grad():
        for index, chain in enumerate(dataset.configurations):
            if time.perf_counter() - started > max_seconds:
                raise RuntimeError("Read-only audit budget exceeded; no partial conclusion")
            x = chain[None]
            levels = model.coarse_grain(x, generator=generator)
            chain_f, chain_p, chain_g = [], [], []
            for k, field in enumerate(levels):
                half = field.shape[1] // 2
                fa, fb = model.features(field[:, :half]), model.features(field[:, half:])
                target = .5 * (fa + fb)
                g = .5 * (model.Q.from_features(fa) + model.Q.from_features(fb))
                chain_f.append(target.numpy()[0]); chain_p.append(model.Q.decode(g).numpy()[0])
                chain_g.append(g.numpy()[0])
                centered = field - field.mean()
                variance = field.var(unbiased=False)
                radii = sorted({1, max(1, field.shape[2] // 4), field.shape[2] // 2})
                spatial = {}
                for radius in radii:
                    value = .5 * sum((centered * torch.roll(centered, radius, axis)).mean() for axis in (2, 3))
                    spatial[str(radius)] = float(value / variance) if variance > 0 else None
                row = dict(chain=index, cell=int(dataset.cell_ids[index]), k=k,
                    split=str(dataset.splits[index]), mean=float(field.mean()), std=float(variance.sqrt()),
                    connected_spatial_correlation_over_site_variance=spatial)
                if k < 2:
                    means = model.C.mean_field(field)
                    blocks = (field[..., 0].reshape(1, field.shape[1], field.shape[2] // 3, 3,
                        field.shape[3] // 3, 3).permute(0, 1, 2, 4, 3, 5))
                    block_mean = blocks.mean((-2, -1))
                    first_site = blocks[..., 0, 0]
                    row.update(mean_C_std_over_input_std=float(means.std(unbiased=False) / variance.sqrt()),
                        correlation_C_mean_with_block_mean=corr(means[..., 0], block_mean),
                        correlation_C_mean_with_first_site=corr(means[..., 0], first_site))
                c_rows.append(row)
            features.append(chain_f); predictions.append(chain_p); encodings.append(chain_g)

            visible, environment = sample_regions(x, pairs=128, generator=generator)
            means = model.C.mean_blocks(visible)[..., 0]
            actual = float(smoothed_information(means, environment, model.scale))
            shuffled = [float(smoothed_information(means[:, torch.randperm(128, generator=generator)],
                environment, model.scale)) for _ in range(5)]
            env = environment / model.scale
            distance = (env[:, :, None] - env[:, None, :]).square().mean(-1)
            logits = -.5 * distance
            logits[:, torch.arange(128), torch.arange(128)] = -torch.inf
            weights = logits.softmax(-1)
            entropy = -(weights * weights.clamp_min(1e-300).log()).sum(-1)
            probabilities = gaussian_bin_probabilities(means, model.scale)
            marginal = probabilities.mean(1)
            coarse_entropy = -(marginal * marginal.clamp_min(1e-300).log()).sum(-1)
            mi_rows.append(dict(chain=index, cell=int(dataset.cell_ids[index]), split=str(dataset.splits[index]),
                proxy_nats=actual, shuffled_proxy_nats=shuffled,
                effective_environment_neighbors=float(entropy.exp().mean()), maximum_neighbors=127,
                coarse_binned_entropy_nats=float(coarse_entropy.mean())))
    features, predictions, encodings = map(np.asarray, (features, predictions, encodings))
    train = dataset.splits == "train"
    pooled_train = features[train].reshape(-1, features.shape[-1])
    shared_mean = pooled_train.mean(0)
    _, singular, vectors = np.linalg.svd(pooled_train - shared_mean, full_matrices=False)
    pca_prediction = shared_mean + ((features - shared_mean) @ vectors[0])[..., None] * vectors[0]
    q_rows = []
    for split in ("train", "test"):
        mask = dataset.splits == split
        for k in range(3):
            mean = features[train, k].mean(0)
            constant_mse = float(np.mean(np.sum((features[mask, k] - mean) ** 2, axis=-1)))
            error = float(np.mean(np.sum((predictions[mask, k] - features[mask, k]) ** 2, axis=-1)))
            q_rows.append(dict(split=split, k=k, reconstruction_l2_squared=error,
                TRAIN_same_scale_mean_baseline_l2_squared=constant_mse,
                reconstruction_over_constant_baseline=error / constant_mse if constant_mse else None,
                explained_vs_same_scale_constant=1 - error / constant_mse if constant_mse else None,
                TRAIN_shared_mean_baseline_l2_squared=float(np.mean(np.sum((features[mask, k] - shared_mean) ** 2, -1))),
                TRAIN_shared_PCA1_reconstruction_l2_squared=float(np.mean(np.sum((pca_prediction[mask, k] - features[mask, k]) ** 2, -1)))))
    model_after = state_fingerprint(model)
    assert before == model_after and all(p.grad is None for p in model.parameters())
    assert checkpoint_sha == sha256_file(checkpoint)
    assert dataset.manifest_sha256 == sha256_file(Path(manifest["data_root"]) / "manifest.json")
    return dict(status="complete", checkpoint_sha256_before=checkpoint_sha,
        checkpoint_sha256_after=sha256_file(checkpoint), model_state_sha256_before=before,
        model_state_sha256_after=model_after, parameters_updated=0, optimizer_created=False,
        physical_sampling=False, GPU_used=False, root_search=False,
        source_manifest_sha256=sha256_file(run / "manifest.json"), max_seconds=max_seconds,
        elapsed_seconds=time.perf_counter() - started,
        C_draws="Fresh fixed-seed CPU Gaussian model draws; not the original CUDA draws and not physical resampling",
        PCA="Post-hoc TRAIN-only shared rank-one affine reconstruction benchmark; not Q initialization or training",
        q_reconstruction=q_rows, q_feature_variance_PCA1_fraction=float(singular[0] ** 2 / np.sum(singular ** 2)),
        c_field_diagnostics=c_rows, information_proxy_diagnostics=mi_rows,
        labels="Cell IDs/splits for grouping only; no r/u/h training or model input",
        uncertainty="Finite correlated chains, one new C draw, five conditional shuffles per chain; no equilibrium or significance certification")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--max-seconds", type=float, default=180.)
    args = parser.parse_args()
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        raise ValueError("Invalid diagnostic budget")
    print(json.dumps(diagnostic(args.run, max_seconds=args.max_seconds), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
