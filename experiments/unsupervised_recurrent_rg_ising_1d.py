#!/usr/bin/env python3
"""Jointly train recurrent RG modules from unlabeled 1D Ising configurations."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recurrent_rg.ising import DTYPE, bond_sum, exact_decimation_flow, sample_periodic_ensembles
from recurrent_rg.metrics import centered_action_relative_rmse, rank_correlation
from recurrent_rg.model_nd import RecurrentRGNetwork1D
from recurrent_rg.unsupervised1d import pseudolikelihood_1d, unsupervised_objective_1d
from recurrent_rg.unsupervised_nd import UnsupervisedLossWeights
from experiments._utils import fixed_eval_subset


def train(
    model: RecurrentRGNetwork1D,
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
    weights: UnsupervisedLossWeights,
) -> list[dict[str, float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    history: list[dict[str, float]] = []
    record_every = max(1, epochs // 50)
    train_eval = fixed_eval_subset(configurations, eval_system_batch, eval_sample_batch)
    test_eval = fixed_eval_subset(
        validation_configurations, eval_system_batch, eval_sample_batch
    )

    def loss_record(prefix: str, batch: torch.Tensor) -> dict[str, float]:
        model.eval()
        with torch.no_grad():
            total, terms = unsupervised_objective_1d(
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
        system_index = torch.randperm(
            configurations.shape[0], generator=generator
        )[:system_batch]
        sample_index = torch.randperm(
            configurations.shape[1], generator=generator
        )[:sample_batch]
        batch = configurations[system_index][:, sample_index]

        optimizer.zero_grad(set_to_none=True)
        loss, terms = unsupervised_objective_1d(
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
    model: RecurrentRGNetwork1D,
    train_x: torch.Tensor,
    train_j: torch.Tensor,
    test_x: torch.Tensor,
    test_j: torch.Tensor,
    levels: int,
) -> dict[str, object]:
    model.eval()
    with torch.no_grad():
        train_trajectory = model(train_x[:, :128])
        test_trajectory = model(test_x[:, :128])
        test_target = exact_decimation_flow(test_j, levels)
        test_action_pl = torch.stack(
            [
                pseudolikelihood_1d(
                    model, level, test_trajectory.direct_theory[:, k, :]
                )
                for k, level in enumerate(test_trajectory.configurations)
            ]
        ).mean()

        train_target_flow = exact_decimation_flow(train_j, levels)
        train_action_relative = []
        test_action_relative = []
        for level_index, (train_level, test_level) in enumerate(
            zip(train_trajectory.configurations, test_trajectory.configurations)
        ):
            train_prediction = model.action_network(
                train_level, train_trajectory.direct_theory[:, level_index, :]
            )
            test_prediction = model.action_network(
                test_level, test_trajectory.direct_theory[:, level_index, :]
            )
            train_exact = -train_target_flow[:, level_index, None] * bond_sum(
                train_level.squeeze(-1)
            )
            test_exact = -test_target[:, level_index, None] * bond_sum(
                test_level.squeeze(-1)
            )
            train_action_relative.append(
                float(centered_action_relative_rmse(train_prediction, train_exact))
            )
            test_action_relative.append(
                float(centered_action_relative_rmse(test_prediction, test_exact))
            )

    train_g = train_trajectory.direct_theory[..., 0].detach().numpy()
    test_g = test_trajectory.direct_theory[..., 0].detach().numpy()
    train_target = exact_decimation_flow(train_j, levels).numpy()
    coefficients = np.polyfit(train_g.reshape(-1), train_target.reshape(-1), deg=3)
    calibrated_direct = np.polyval(coefficients, test_g)
    calibrated_rollout = np.polyval(
        coefficients, test_trajectory.rollout_theory[..., 0].detach().numpy()
    )

    visible_states = torch.tensor(
        list(itertools.product((-1.0, 1.0), repeat=model.geometry.patch_volume)),
        dtype=DTYPE,
    )
    with torch.no_grad():
        hard_outputs = torch.where(
            model.rg_kernel.field_head.relaxed(
                model.rg_kernel.local_encoder(visible_states)
            ).squeeze(-1)
            >= 0,
            1.0,
            -1.0,
        )
    return {
        "kernel_visible_states": visible_states.tolist(),
        "kernel_hard_outputs": hard_outputs.tolist(),
        "microscopic_latent_pearson": float(np.corrcoef(test_g[:, 0], test_j.numpy())[0, 1]),
        "microscopic_latent_spearman": rank_correlation(test_g[:, 0], test_j.numpy()),
        "calibration_polynomial_g_to_j": coefficients.tolist(),
        "calibrated_direct_rmse": float(np.sqrt(np.mean((calibrated_direct - test_target.numpy()) ** 2))),
        "calibrated_rollout_rmse": float(np.sqrt(np.mean((calibrated_rollout - test_target.numpy()) ** 2))),
        "test_action_pseudolikelihood": float(test_action_pl),
        "train_action_relative_rmse_by_level": train_action_relative,
        "test_action_relative_rmse_by_level": test_action_relative,
        "train_microscopic_action_relative_rmse": train_action_relative[0],
        "test_microscopic_action_relative_rmse": test_action_relative[0],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", type=int, default=64)
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--theory-dim", type=int, default=1)
    parser.add_argument("--ensemble-size", type=int, default=512)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--system-batch", type=int, default=16)
    parser.add_argument("--sample-batch", type=int, default=64)
    parser.add_argument("--eval-system-batch", type=int, default=8)
    parser.add_argument("--eval-sample-batch", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--buffer", type=int, default=1)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/recurrent_rg/unsupervised_joint"),
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    coupling_grid = np.linspace(0.12, 1.20, 49)
    train_mask = np.arange(len(coupling_grid)) % 3 != 1
    train_x, train_j = sample_periodic_ensembles(
        coupling_grid[train_mask], args.replicas, args.ensemble_size, args.sites, args.seed
    )
    test_x, test_j = sample_periodic_ensembles(
        coupling_grid[~train_mask], args.replicas, args.ensemble_size, args.sites, args.seed + 1
    )

    model = RecurrentRGNetwork1D(
        args.levels, theory_dim=args.theory_dim
    ).to(dtype=DTYPE)
    loss_weights = UnsupervisedLossWeights()
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
        weights=loss_weights,
    )
    metrics = evaluate(model, train_x, train_j, test_x, test_j, args.levels)
    summary = {
        "config": vars(args) | {"output_dir": str(args.output_dir)},
        "training_contract": {
            "configuration_only": True,
            "couplings_used_in_training_loss": False,
            "couplings_used_posthoc_for_evaluation": True,
            "all_modules_updated_by_one_joint_objective": True,
            "symmetry_built_into_model": False,
            "geometry_adapter": "periodic_chain",
            "field_head": "binary_straight_through",
            "rsmi_estimator": "contrastive_infonce",
            "local_conditional": "binary_single_site_flip",
        },
        "loss_weights": loss_weights.__dict__,
        "initial_kernel_state": initial_kernel_state,
        "metrics": metrics,
        "history": history,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    torch.save(
        {"state_dict": model.state_dict(), "config": summary["config"]},
        args.output_dir / "model.pt",
    )
    print(json.dumps({"output_dir": str(args.output_dir), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
