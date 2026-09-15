#!/usr/bin/env python3
"""Joint label-free recurrent-RG training for the periodic 2D Ising model."""

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

from recurrent_rg.ising import DTYPE
from recurrent_rg.ising2d import bond_sum_2d, sample_wolff_ensembles
from recurrent_rg.metrics import centered_action_relative_rmse, rank_correlation
from recurrent_rg.model_nd import RecurrentRGNetwork2D
from recurrent_rg.unsupervised2d import (
    pseudolikelihood_2d,
    unsupervised_objective_2d,
)
from recurrent_rg.unsupervised_nd import UnsupervisedLossWeights
from experiments._utils import fixed_eval_subset


def train(
    model: RecurrentRGNetwork2D,
    configurations: torch.Tensor,
    validation_configurations: torch.Tensor,
    *,
    epochs: int,
    system_batch: int,
    sample_batch: int,
    eval_system_batch: int,
    eval_sample_batch: int,
    learning_rate: float,
    seed: int,
    buffer: int,
) -> list[dict[str, float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    weights = UnsupervisedLossWeights()
    record_every = max(1, epochs // 50)
    history = []
    train_eval = fixed_eval_subset(configurations, eval_system_batch, eval_sample_batch)
    test_eval = fixed_eval_subset(
        validation_configurations, eval_system_batch, eval_sample_batch
    )

    def loss_record(prefix: str, batch: torch.Tensor) -> dict[str, float]:
        model.eval()
        with torch.no_grad():
            total, terms = unsupervised_objective_2d(
                model,
                batch,
                buffer=buffer,
                weights=weights,
            )
        return {f"{prefix}_total": float(total)} | {
            f"{prefix}_{name}": float(value) for name, value in terms.items()
        }
    for epoch in range(epochs):
        model.train()
        systems = torch.randperm(configurations.shape[0], generator=generator)[:system_batch]
        samples = torch.randperm(configurations.shape[1], generator=generator)[:sample_batch]
        batch = configurations[systems][:, samples].clone()
        # This is data augmentation only: the model itself is not constrained to
        # be invariant or equivariant under a global spin flip.
        flip = torch.rand(batch.shape[:2], generator=generator) < 0.5
        batch *= torch.where(flip, -1.0, 1.0)[:, :, None, None, None]
        optimizer.zero_grad(set_to_none=True)
        loss, terms = unsupervised_objective_2d(
            model,
            batch,
            buffer=buffer,
            weights=weights,
        )
        loss.backward()
        optimizer.step()
        if epoch == 0 or (epoch + 1) % record_every == 0 or epoch + 1 == epochs:
            history.append(
                {"epoch": epoch + 1, "optimization_batch_total": float(loss.detach())}
                | loss_record("train", train_eval)
                | loss_record("test", test_eval)
            )
    return history


def evaluate(
    model: RecurrentRGNetwork2D,
    train_x: torch.Tensor,
    train_j: torch.Tensor,
    test_x: torch.Tensor,
    test_j: torch.Tensor,
) -> dict[str, object]:
    model.eval()
    with torch.no_grad():
        train_x = train_x[:, :128]
        test_x = test_x[:, :128]
        train_trajectory = model(train_x)
        trajectory = model(test_x)
        error = (trajectory.rollout_theory[:, 1:] - trajectory.direct_theory[:, 1:]).square()
        variance = (
            trajectory.direct_theory[:, 1:]
            - trajectory.direct_theory[:, 1:].mean(dim=0, keepdim=True)
        ).square()
        flow_at_1 = torch.sqrt(error[:, 0].mean() / variance[:, 0].mean().clamp_min(1e-12))
        flow_at_2 = torch.sqrt(error.sum(dim=1).mean() / variance.sum(dim=1).mean().clamp_min(1e-12))

        learned_pl_levels = torch.stack(
            [
                pseudolikelihood_2d(model, level, trajectory.direct_theory[:, k, :])
                for k, level in enumerate(trajectory.configurations)
            ]
        )
        spins = test_x.squeeze(-1)
        flipped = spins.clone()
        flipped[:, :, 0, 0] *= -1
        action = -test_j[:, None] * bond_sum_2d(spins)
        flipped_action = -test_j[:, None] * bond_sum_2d(flipped)
        exact_microscopic_pl = F.softplus(action - flipped_action).mean()
        train_prediction = model.action_network(
            train_x, train_trajectory.direct_theory[:, 0, :]
        )
        test_prediction = model.action_network(
            test_x, trajectory.direct_theory[:, 0, :]
        )
        train_exact = -train_j[:, None] * bond_sum_2d(train_x.squeeze(-1))
        test_exact = -test_j[:, None] * bond_sum_2d(test_x.squeeze(-1))
        train_action_relative = centered_action_relative_rmse(
            train_prediction, train_exact
        )
        test_action_relative = centered_action_relative_rmse(
            test_prediction, test_exact
        )

    latent = trajectory.direct_theory[:, 0, 0].detach().numpy()
    return {
        "flow": {
            "epsilon_flow_at_1": float(flow_at_1),
            "epsilon_flow_at_2": float(flow_at_2),
        },
        "action": {
            "learned_pseudolikelihood_by_level": learned_pl_levels.tolist(),
            "exact_microscopic_pseudolikelihood": float(exact_microscopic_pl),
            "microscopic_excess_pseudolikelihood": float(
                learned_pl_levels[0] - exact_microscopic_pl
            ),
            "train_microscopic_action_relative_rmse": float(train_action_relative),
            "test_microscopic_action_relative_rmse": float(test_action_relative),
        },
        "posthoc": {
            "latent_j_pearson": float(np.corrcoef(latent, test_j.numpy())[0, 1]),
            "latent_j_spearman": rank_correlation(latent, test_j.numpy()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--theory-dim", type=int, default=1)
    parser.add_argument("--ensemble-size", type=int, default=192)
    parser.add_argument("--replicas", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--system-batch", type=int, default=8)
    parser.add_argument("--sample-batch", type=int, default=32)
    parser.add_argument("--eval-system-batch", type=int, default=8)
    parser.add_argument("--eval-sample-batch", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--buffer", type=int, default=1)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/recurrent_rg/unsupervised_joint_2d"),
    )
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    coupling_grid = np.linspace(0.20, 0.68, 25)
    train_mask = np.arange(len(coupling_grid)) % 3 != 1
    train_x, train_j = sample_wolff_ensembles(
        coupling_grid[train_mask], args.replicas, args.ensemble_size, args.size, args.seed
    )
    test_x, test_j = sample_wolff_ensembles(
        coupling_grid[~train_mask], args.replicas, args.ensemble_size, args.size, args.seed + 1
    )
    model = RecurrentRGNetwork2D(args.levels, theory_dim=args.theory_dim).to(dtype=DTYPE)
    initial_kernel_state = {
        name: value.detach().tolist()
        for name, value in model.rg_kernel.state_dict().items()
    }
    history = train(
        model,
        train_x,
        test_x,
        epochs=args.epochs,
        system_batch=args.system_batch,
        sample_batch=args.sample_batch,
        eval_system_batch=args.eval_system_batch,
        eval_sample_batch=args.eval_sample_batch,
        learning_rate=args.learning_rate,
        seed=args.seed + 2,
        buffer=args.buffer,
    )
    metrics = evaluate(model, train_x, train_j, test_x, test_j)
    summary = {
        "config": vars(args) | {"output_dir": str(args.output_dir)},
        "training_contract": {
            "configuration_only": True,
            "couplings_used_in_training_loss": False,
            "couplings_used_posthoc_for_evaluation": True,
            "all_modules_updated_by_one_joint_objective": True,
            "symmetry_built_into_model": False,
            "global_spin_flip_used_as_data_augmentation": True,
            "wolff_used_only_for_sampling": True,
            "geometry_adapter": "periodic_square",
            "field_head": "binary_straight_through",
            "local_conditional": "binary_single_site_flip",
        },
        "data": {
            "train_shape": list(train_x.shape),
            "test_shape": list(test_x.shape),
            "train_energy_per_site": float(bond_sum_2d(train_x.squeeze(-1)).mean() / args.size**2),
            "test_energy_per_site": float(bond_sum_2d(test_x.squeeze(-1)).mean() / args.size**2),
        },
        "initial_kernel_state": initial_kernel_state,
        "metrics": metrics,
        "history": history,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    torch.save(
        {"state_dict": model.state_dict(), "config": summary["config"]},
        args.output_dir / "model.pt",
    )
    print(json.dumps({"output_dir": str(args.output_dir), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
