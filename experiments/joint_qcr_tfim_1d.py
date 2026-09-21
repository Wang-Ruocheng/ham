#!/usr/bin/env python3
"""Joint QCR training on the 1D transverse-field Ising model.

Learns Q, C, R without RSMI or action model, using finite-temperature
Suzuki-Trotter path-integral Monte Carlo ensembles of sigma^z configurations. The physical coupling is the
transverse-field ratio h/J. Theory dimension is configurable; extra dimensions
do not by themselves guarantee closure or preservation of phase information.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.joint_qcr_ising import (
    find_fixed_points_1d,
    qcr_objective,
    summarize_representation,
    train,
    trajectory_data,
    two_spin_distribution,
)
from recurrent_rg.model_nd import RecurrentRGNetwork1D, FixedDecimationCoarseGrainer
from recurrent_rg.tfim import DTYPE, sample_tfim_ensembles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", type=int, default=128)
    parser.add_argument("--levels", type=int, default=6)
    parser.add_argument("--theory-dim", type=int, default=2)
    parser.add_argument("--fixed-kernel", action="store_true",
                        help="Use non-learned decimation for C; only train Q and R")
    parser.add_argument("--replicas", type=int, default=3)
    parser.add_argument("--ensemble-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--system-batch", type=int, default=32)
    parser.add_argument("--sample-batch", type=int, default=128)
    parser.add_argument("--eval-samples", type=int, default=48)
    parser.add_argument("--eval-systems", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument('--objective-mode', choices=('data_geometry',), default='data_geometry')
    parser.add_argument('--geometry-scale', type=float, default=4.0)
    parser.add_argument('--coordinate-reduction', choices=('mean','sum'), default='sum',
                        help='sum removes latent-dimension normalization in four consistency/closure terms')
    parser.add_argument('--noise-multiplier', type=float, default=2.0)
    parser.add_argument('--rollout-weight', type=float, default=5.0,
                        help='Weight of equal-horizon autonomous multistep closure')
    parser.add_argument('--binary-lookup', action='store_true', help='Exact binary lookup acceleration for Q and C')
    parser.add_argument("--hidden", type=int, default=60,
                        help="Hidden dimension of Q/R networks")
    parser.add_argument("--time-slices", type=int, default=32)
    parser.add_argument("--beta-over-slices", type=float, default=0.5)
    parser.add_argument("--equilibration-sweeps", type=int, default=32,
                        help="Wolff single-cluster updates discarded before sampling")
    parser.add_argument("--decorrelation-sweeps", type=int, default=4,
                        help="Wolff single-cluster updates between saved samples")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/joint_qcr_tfim_1d"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    model = RecurrentRGNetwork1D(
        levels=args.levels,
        theory_dim=args.theory_dim,
        hidden=args.hidden,
        learnable_kernel=True,
    ).to(dtype=DTYPE)
    if args.fixed_kernel:
        # Preserve Q/R initialization relative to the learned-C paired-seed run.
        model.rg_kernel = FixedDecimationCoarseGrainer(model.geometry)
    model.theory_encoder.binary_lookup = args.binary_lookup
    model.rg_kernel.binary_lookup = args.binary_lookup
    pattern_fn = two_spin_distribution
    # Same coupling grid as the 1D classical Ising baseline, now interpreted
    # as the TFIM ratio x = h / J.
    coupling_grid = np.linspace(0.01, 2.0, 64)
    train_mask = np.arange(coupling_grid.size) % 3 != 1

    cache_dir = Path("data_cache")
    cache_dir.mkdir(exist_ok=True)
    delta_tau = format(args.beta_over_slices, ".8g").replace(".", "p")
    cache_name = (
        f"tfim1d_wolff_v2_sites{args.sites}_M{args.time_slices}_dtau{delta_tau}_"
        f"eq{args.equilibration_sweeps}_skip{args.decorrelation_sweeps}_"
        f"replicas{args.replicas}_ensemble{args.ensemble_size}_"
        f"seed{args.seed}_nJ{len(coupling_grid)}.pt"
    )
    cache_path = cache_dir / cache_name

    if cache_path.exists():
        print(f"loading cached data from {cache_path}")
        cached = torch.load(cache_path, weights_only=True)
        train_x, train_j, test_x, test_j = cached
    else:
        print("sampling TFIM train data...")
        train_x, train_j = sample_tfim_ensembles(
            coupling_grid[train_mask],
            args.replicas,
            args.ensemble_size,
            args.sites,
            args.seed,
            time_slices=args.time_slices,
            beta_over_slices=args.beta_over_slices,
            equilibration_sweeps=args.equilibration_sweeps,
            decorrelation_sweeps=args.decorrelation_sweeps,
        )
        print("sampling TFIM test data...")
        test_x, test_j = sample_tfim_ensembles(
            coupling_grid[~train_mask],
            args.replicas,
            args.ensemble_size,
            args.sites,
            args.seed + 1,
            time_slices=args.time_slices,
            beta_over_slices=args.beta_over_slices,
            equilibration_sweeps=args.equilibration_sweeps,
            decorrelation_sweeps=args.decorrelation_sweeps,
        )
        print(f"saving data to {cache_path}")
        torch.save((train_x, train_j, test_x, test_j), cache_path)

    print(f"train: {train_x.shape}, test: {test_x.shape}")

    history = train(
        model, train_x, test_x, pattern_fn=pattern_fn, epochs=args.epochs,
        system_batch=min(args.system_batch, train_x.shape[0]),
        sample_batch=args.sample_batch,
        eval_samples=args.eval_samples, eval_systems=args.eval_systems,
        learning_rate=args.learning_rate, seed=args.seed,
        progress_dir=args.output_dir,
        objective_kwargs={'objective_mode': args.objective_mode,
                          'coordinate_reduction': args.coordinate_reduction,
                          'geometry_scale': args.geometry_scale,
                          'noise_multiplier': args.noise_multiplier,
                          'rollout_weight': args.rollout_weight},
    )
    print(f"finished joint QCR: test objective={history[-1]['test_total']:.6g}",
          flush=True)

    _, train_direct, train_rollout, train_split = trajectory_data(
        model, train_x, args.eval_samples
    )
    _, test_direct, test_rollout, test_split = trajectory_data(
        model, test_x, args.eval_samples
    )

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
        "config": vars(args) | {"output_dir": str(args.output_dir), "kernel_layout": "nonoverlapping"},
        "training_contract": {
            "optimized_modules": ["Q", "R"] if args.fixed_kernel else ["Q", "C", "R"],
            "fixed_kernel_rule": "positive decimation X[..., ::2, :]" if args.fixed_kernel else None,
            "initialization": "Construct learned-C baseline then replace C if fixed, preserving Q/R initialization",
            "rsmi_used": False,
            "action_used": False,
            "couplings_used_in_loss": False,
            "joint_training_only": True,
            "kernel_noncollapse_used": True,
            "kernel_noncollapse_kind": "binary marginal balance across each coarse level",
            "forced_separation_gate": False,
            "fixed_separation_target": False,
            "objective_mode": args.objective_mode,
            "coordinate_reduction": args.coordinate_reduction,
            "geometry_target": 'two-spin pattern distances at every scale with detached split-noise intervals' if args.objective_mode == 'data_geometry' else 'legacy',
            "geometry_scale": args.geometry_scale,
            "noise_multiplier": args.noise_multiplier,
            "rollout_weight": args.rollout_weight,
            "rollout_horizons": list(range(1, args.levels)),
            "scale_decay": None if args.objective_mode == 'data_geometry' else 0.5,
            "pattern_scope": 'local two-spin marginals; not complete configuration distributions',
            "physical_model": "1D transverse-field Ising",
            "coupling_meaning": "h / J",
            "sampler": "finite-temperature Suzuki-Trotter Wolff v2",
            "inverse_temperature": args.time_slices * args.beta_over_slices,
            "trotter_slices": args.time_slices,
            "beta_over_slices": args.beta_over_slices,
        },
        "data": {"train_shape": list(train_x.shape),
                 "test_shape": list(test_x.shape)},
        "train_metrics": summarize_representation(
            train_direct, train_rollout, train_split, train_j.numpy()
        ),
        "test_metrics": summarize_representation(
            test_direct, test_rollout, test_split, test_j.numpy()
        ),
        "learned_R": {"trained_support": support,
                      "fixed_points_in_support": roots},
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
        scalar_report(args.output_dir, cache_path, 1)


if __name__ == "__main__":
    main()
