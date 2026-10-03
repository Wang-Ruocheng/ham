#!/usr/bin/env python3
"""Unified QCR training for 1D and 2D Ising models.

Jointly learns Q, C, R without RSMI or action model.
Default: joint training (no Q pretraining).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Compatibility exports for historical imports.
from recurrent_rg.config import QCRWeights
from recurrent_rg.data.views import balanced_views, ensemble_views
from recurrent_rg.data.statistics import two_spin_distribution, plaquette_distribution
from recurrent_rg.losses import (
    binary_kernel_noncollapse, distribution_geometry_loss, latent_squared_error,
    multistep_closure_loss, binary_kernel_variance_floor, data_geometry_objective,
    data_geometry_statistics_objective, qcr_objective,
)
from recurrent_rg.training import train, evaluate
from recurrent_rg.evaluation import trajectory_data, summarize_representation, find_fixed_points_1d
from recurrent_rg.ising import DTYPE, sample_periodic_ensembles
from recurrent_rg.ising2d import sample_wolff_ensembles
from recurrent_rg.model_nd import RecurrentRGNetwork1D, RecurrentRGNetwork2D

ISING_2D_CRITICAL_COUPLING = 0.5 * np.log(1.0 + np.sqrt(2.0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, choices=[1, 2], required=True)
    parser.add_argument("--sites", type=int, default=32)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--theory-dim", type=int, default=1)
    parser.add_argument("--replicas", type=int, default=3)
    parser.add_argument("--ensemble-size", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--system-batch", type=int, default=32)
    parser.add_argument("--sample-batch", type=int, default=20)
    parser.add_argument("--eval-samples", type=int, default=48)
    parser.add_argument("--eval-systems", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=24,
                        help="Hidden dimension of Q/R networks")
    parser.add_argument("--data-cache", type=Path, help="Explicit prevalidated dataset cache")
    parser.add_argument("--checkpoint-chunks", type=int, default=0)
    parser.add_argument('--objective-mode', choices=('data_geometry',), default='data_geometry')
    parser.add_argument('--geometry-scale', type=float, default=4.)
    parser.add_argument('--coordinate-reduction', choices=('mean', 'sum'), default='sum',
                        help='sum is the current convention; mean reproduces historical vector runs')
    parser.add_argument('--noise-multiplier', type=float, default=2.)
    parser.add_argument('--rollout-weight', type=float, default=5.)
    parser.add_argument("--binary-lookup", action="store_true",
                        help="Equivalent binary-pattern local-MLP evaluation, preserving input gradients")
    parser.add_argument("--output-dir", type=Path, default=Path("results/joint_qcr_ising"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    if args.dimension == 1:
        model = RecurrentRGNetwork1D(levels=args.levels, theory_dim=args.theory_dim, hidden=args.hidden).to(dtype=DTYPE)
        pattern_fn = two_spin_distribution
        coupling_grid = np.linspace(0.01, 2.0, 64)  # extended range
    else:
        model = RecurrentRGNetwork2D(levels=args.levels, theory_dim=args.theory_dim, hidden=args.hidden).to(dtype=DTYPE)
        pattern_fn = plaquette_distribution
        coupling_grid = np.linspace(0.20, 0.68, 37)

    train_mask = np.arange(coupling_grid.size) % 3 != 1

    cache_dir = Path("data_cache")
    cache_dir.mkdir(exist_ok=True)
    cache_name = (
        f"ising{args.dimension}d_sites{args.sites}_replicas{args.replicas}_"
        f"ensemble{args.ensemble_size}_seed{args.seed}_nJ{len(coupling_grid)}.pt"
    )
    cache_path = args.data_cache or (cache_dir / cache_name)
    if args.data_cache and not cache_path.exists():
        raise FileNotFoundError(cache_path)
    model.theory_encoder.checkpoint_chunks = args.checkpoint_chunks
    model.theory_encoder.binary_lookup = args.binary_lookup
    model.rg_kernel.binary_lookup = args.binary_lookup

    if cache_path.exists():
        print(f"loading cached data from {cache_path}")
        cached = torch.load(cache_path, weights_only=True)
        train_x, train_j, test_x, test_j = cached
    else:
        if args.dimension == 1:
            print("sampling 1D train data...")
            train_x, train_j = sample_periodic_ensembles(
                coupling_grid[train_mask], args.replicas, args.ensemble_size, args.sites, args.seed
            )
            print("sampling 1D test data...")
            test_x, test_j = sample_periodic_ensembles(
                coupling_grid[~train_mask], args.replicas, args.ensemble_size, args.sites, args.seed + 1
            )
        else:
            print("sampling 2D train data...")
            train_x, train_j = sample_wolff_ensembles(
                coupling_grid[train_mask], args.replicas, args.ensemble_size, args.sites, args.seed
            )
            print("sampling 2D test data...")
            test_x, test_j = sample_wolff_ensembles(
                coupling_grid[~train_mask], args.replicas, args.ensemble_size, args.sites, args.seed + 1
            )
        print(f"saving data to {cache_path}")
        torch.save((train_x, train_j, test_x, test_j), cache_path)

    print(f"train: {train_x.shape}, test: {test_x.shape}")

    history = train(
        model, train_x, test_x, pattern_fn=pattern_fn, epochs=args.epochs,
        system_batch=min(args.system_batch, train_x.shape[0]), sample_batch=args.sample_batch,
        eval_samples=args.eval_samples, eval_systems=args.eval_systems,
        learning_rate=args.learning_rate, seed=args.seed,
        progress_dir=args.output_dir,
        objective_kwargs=dict(objective_mode=args.objective_mode, geometry_scale=args.geometry_scale,
                              noise_multiplier=args.noise_multiplier, rollout_weight=args.rollout_weight,
                              coordinate_reduction=args.coordinate_reduction),
    )
    print(f"finished joint QCR: test objective={history[-1]['test_total']:.6g}", flush=True)

    _, train_direct, train_rollout, train_split = trajectory_data(model, train_x, args.eval_samples)
    _, test_direct, test_rollout, test_split = trajectory_data(model, test_x, args.eval_samples)

    if args.theory_dim == 1:
        support = [
            float(min(train_direct.min(), test_direct.min())),
            float(max(train_direct.max(), test_direct.max())),
        ]
        roots = find_fixed_points_1d(model, support[0], support[1])
    else:
        support = [
            [float(min(train_direct[..., 0].min(), test_direct[..., 0].min())),
             float(min(train_direct[..., 1].min(), test_direct[..., 1].min()))],
            [float(max(train_direct[..., 0].max(), test_direct[..., 0].max())),
             float(max(train_direct[..., 1].max(), test_direct[..., 1].max()))],
        ]
        roots = []

    summary = {
        "config": vars(args) | {"output_dir": str(args.output_dir), "data_cache": str(cache_path), "kernel_layout": "nonoverlapping"},
        "training_contract": {
            "optimized_modules": ["Q", "C", "R"],
            "rsmi_used": False,
            "action_used": False,
            "couplings_used_in_loss": False,
            "joint_training_only": True,
            "kernel_noncollapse_used": True,
            "kernel_noncollapse_kind": "binary marginal balance across each coarse level",
            "forced_separation_gate": False,
            "fixed_separation_target": False,
            "objective_mode": args.objective_mode,
            "geometry_scale": args.geometry_scale,
            "noise_multiplier": args.noise_multiplier,
            "rollout_weight": args.rollout_weight,
            "rollout_horizons": list(range(1,args.levels)),
            "coordinate_reduction": args.coordinate_reduction,
        },
        "data": {"train_shape": list(train_x.shape), "test_shape": list(test_x.shape)},
        "train_metrics": summarize_representation(train_direct, train_rollout, train_split, train_j.numpy()),
        "test_metrics": summarize_representation(test_direct, test_rollout, test_split, test_j.numpy()),
        "learned_R": {"trained_support": support, "fixed_points_in_support": roots},
        "history": history,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    torch.save(
        {name: value for name, value in model.state_dict().items()
         if name.startswith(("theory_encoder.", "rg_kernel.", "theory_rg."))},
        args.output_dir / "qcr_model.pt",
    )
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "test_metrics": summary["test_metrics"],
        "learned_R": summary["learned_R"],
    }, indent=2))


    if args.theory_dim == 1:
        from experiments.qcr_scalar_report import scalar_report
        scalar_report(args.output_dir, cache_path, args.dimension)


if __name__ == "__main__":
    main()
