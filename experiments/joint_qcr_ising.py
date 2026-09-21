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

from experiments._utils import QCRWeights, balanced_views
from recurrent_rg.ising import DTYPE, sample_periodic_ensembles
from recurrent_rg.ising2d import sample_wolff_ensembles
from recurrent_rg.metrics import rank_correlation, mean_pairwise_square
from recurrent_rg.model_nd import RecurrentRGNetwork1D, RecurrentRGNetwork2D


ISING_2D_CRITICAL_COUPLING = 0.5 * np.log(1.0 + np.sqrt(2.0))


def two_spin_distribution(configurations: torch.Tensor) -> torch.Tensor:
    """1D: empirical probabilities of 4 two-spin patterns."""
    spins = configurations.squeeze(-1)
    bits = (spins > 0).long()
    pattern = 2 * bits + torch.roll(bits, -1, dims=2)
    return F.one_hot(pattern, num_classes=4).to(configurations.dtype).mean(dim=(1, 2))


def plaquette_distribution(configurations: torch.Tensor) -> torch.Tensor:
    """2D: empirical probabilities of 16 oriented 2x2 spin plaquettes."""
    spins = configurations.squeeze(-1)
    bits = (spins > 0).long()
    pattern = (
        8 * bits
        + 4 * torch.roll(bits, -1, dims=2)
        + 2 * torch.roll(bits, -1, dims=3)
        + torch.roll(torch.roll(bits, -1, dims=2), -1, dims=3)
    )
    return F.one_hot(pattern, num_classes=16).to(configurations.dtype).mean(dim=(1, 2, 3))


def binary_kernel_noncollapse(levels: list[torch.Tensor]) -> torch.Tensor:
    """Keep both binary coarse states populated across a balanced batch.

    Every coarse tensor has a straight-through hard forward value, so a
    constant +1/-1 kernel gives loss one while gradients still flow through
    the relaxed binary head. The microscopic level is intentionally excluded.
    """
    coarse_means = torch.stack([level.mean() for level in levels[1:]])
    return coarse_means.square().mean()


def distribution_geometry_loss(theory, patterns_a, patterns_b, *, scale=4.0, noise_multiplier=2.0):
    """Match distances to fixed-unit, split-noise intervals in pattern space.

    Targets are detached (including from C). A half-split difference estimates
    the noise of the mean distribution: E||pa-pb||^2/4. This is a heuristic
    noise allowance, not a confidence interval or a distribution-equality test.
    No batch standardization, fixed spread, labels, or trainable target scale.
    """
    if scale <= 0 or noise_multiplier < 0:
        raise ValueError('Need positive geometry scale and nonnegative noise multiplier')
    if theory.shape[0] < 2:
        raise ValueError('Need at least two ensembles')
    with torch.no_grad():
        mean = .5 * (patterns_a + patterns_b)
        noise = .25 * (patterns_a - patterns_b).square().sum(dim=-1)
        i, j = torch.triu_indices(len(theory), len(theory), offset=1, device=theory.device)
        distance = (mean[i] - mean[j]).norm(dim=-1)
        allowance = noise_multiplier * (noise[i] + noise[j]).sqrt()
        lower = scale * (distance - allowance).clamp_min(0)
        upper = scale * (distance + allowance)
    latent = torch.pdist(theory)
    loss = (F.relu(lower-latent).square() + F.relu(latent-upper).square()).mean()
    return loss, {
        'resolved_fraction': (lower > 0).to(theory.dtype).mean(),
        'target_distance': (scale * distance).mean(),
        'noise_allowance': (scale * allowance).mean(),
        'latent_distance': latent.mean(),
    }


def latent_squared_error(pred, target, coordinate_reduction='sum'):
    if coordinate_reduction == 'sum':
        return (pred-target).square().sum(dim=-1).mean()
    if coordinate_reduction != 'mean':
        raise ValueError('coordinate_reduction must be mean or sum')
    return F.mse_loss(pred, target)


def multistep_closure_loss(model, direct, coordinate_reduction='sum'):
    """Autonomous rollout from g0; retain gradients through every R call."""
    predicted = direct[:, 0]
    errors = []
    for k in range(1, direct.shape[1]):
        predicted = model.advance_theory(predicted)
        errors.append(latent_squared_error(predicted, direct[:, k], coordinate_reduction))
    return torch.stack(errors).mean(), errors


def binary_kernel_variance_floor(levels, ratio=0.25):
    """Allow biased binary marginals; penalize near-constant batch outputs.

    Require 1-m_out^2 >= ratio*(1-m_in^2), with a detached input
    reference. A hinge on |m_out| retains an STE gradient at constant outputs.
    This is a soft anti-collapse heuristic, not a magnetization constraint.
    """
    if not 0 < ratio <= 1:
        raise ValueError('variance floor ratio must be in (0, 1]')
    terms = []
    for source, target in zip(levels[:-1], levels[1:]):
        variance = (1-source.detach().mean().square()).clamp(0, 1)
        bound = (1-ratio*variance).sqrt()
        terms.append(F.relu(target.mean().abs()-bound).square())
    return torch.stack(terms).mean()


def data_geometry_objective(model, levels, first, second, pattern_fn, weights,
                            geometry_scale, noise_multiplier, rollout_weight=5.0,
                            kernel_noncollapse_mode='balance', kernel_variance_ratio=0.25,
                            coordinate_reduction='sum'):
    """Equal-scale distribution geometry replaces anchor spread + hinge floor."""
    midpoint = levels[0].shape[1] // 2
    pa = [pattern_fn(level[:, :midpoint]) for level in levels]
    pb = [pattern_fn(level[:, midpoint:]) for level in levels]
    kernel = (binary_kernel_variance_floor(levels, kernel_variance_ratio)
              if kernel_noncollapse_mode == 'variance_floor'
              else binary_kernel_noncollapse(levels))
    return data_geometry_statistics_objective(model, first, second, pa, pb, kernel,
                  weights, geometry_scale, noise_multiplier, rollout_weight, coordinate_reduction)


def data_geometry_statistics_objective(model, first, second, pattern_first, pattern_second,
                                       kernel, weights, geometry_scale, noise_multiplier,
                                       rollout_weight=5.0, coordinate_reduction='sum'):
    """Shared objective algebra, also usable with exact fixed-C sufficient statistics."""
    direct = .5 * (first + second)
    geometry, extra = [], {}
    for k, (pa, pb) in enumerate(zip(pattern_first, pattern_second)):
        term, metrics = distribution_geometry_loss(direct[:, k], pa, pb,
                        scale=geometry_scale, noise_multiplier=noise_multiplier)
        geometry.append(term)
        extra[f'data_geometry_level_{k}'] = term
        for name, value in metrics.items():
            extra[f'{name}_level_{k}'] = value
        extra[f'pairwise_level_{k}'] = mean_pairwise_square(direct[:, k])
    # Preserve old subset-consistency and C/R weights, removing only the
    # old unit-variance/covariance/correlation anchor and fixed separation.
    anchor = 10. * latent_squared_error(first[:, 0], second[:, 0], coordinate_reduction)
    consistency = latent_squared_error(first[:, 1:], second[:, 1:], coordinate_reduction)
    separation = torch.stack(geometry).mean()
    predicted = model.advance_theory(direct[:, :-1])
    commutator = latent_squared_error(predicted, direct[:, 1:], coordinate_reduction)
    rollout, rollout_levels = multistep_closure_loss(model, direct, coordinate_reduction)
    extra['rollout'] = rollout
    for k, term in enumerate(rollout_levels, start=1):
        extra[f'rollout_level_{k}'] = term
    total = (weights.anchor * anchor + weights.coarse_consistency * consistency
             + weights.kernel_noncollapse * kernel + weights.separation * separation
             + weights.commutator * commutator + rollout_weight * rollout)
    return total, {'anchor': anchor, 'coarse_consistency': consistency,
                   'kernel_noncollapse': kernel, 'separation': separation,
                   'commutator': commutator, **extra}


def qcr_objective(
    model,
    microscopic: torch.Tensor,
    *,
    pattern_fn,
    weights: QCRWeights = QCRWeights(anchor=2.0, separation=10.0),
    objective_mode: str = 'data_geometry',
    geometry_scale: float = 4.0,
    noise_multiplier: float = 2.0,
    rollout_weight: float = 5.0,
    kernel_noncollapse_mode: str = 'balance',
    kernel_variance_ratio: float = 0.25,
    coordinate_reduction: str = 'sum',
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Joint all-scale distribution geometry and single/multistep closure."""
    if objective_mode != "data_geometry":
        raise ValueError("Legacy loss moved to archive.qcr_ablations.legacy_objective")
    if coordinate_reduction not in ('mean', 'sum'):
        raise ValueError('coordinate_reduction must be mean or sum')
    if rollout_weight < 0:
        raise ValueError('rollout_weight must be nonnegative')
    if kernel_noncollapse_mode not in ('balance', 'variance_floor'):
        raise ValueError('Unknown kernel noncollapse mode')
    samples = microscopic.shape[1]
    midpoint = samples // 2
    if midpoint < 1 or samples % 2:
        raise ValueError("QCR batches need an even number of samples")

    levels = model.coarse_grain(microscopic)
    first = torch.stack(
        [model.theory_encoder(level[:, :midpoint]) for level in levels], dim=1
    )
    second = torch.stack(
        [model.theory_encoder(level[:, midpoint:]) for level in levels], dim=1
    )
    return data_geometry_objective(model, levels, first, second, pattern_fn,
                                   weights, geometry_scale, noise_multiplier, rollout_weight,
                                   kernel_noncollapse_mode, kernel_variance_ratio, coordinate_reduction)
def ensemble_views(configurations, samples, *, spin_flip_augment=True):
    """Nonzero-field ensembles must never be balanced by artificial spin flips."""
    if samples < 2 or samples % 2 or samples > configurations.shape[1]:
        raise ValueError('Need an even sample count available in every ensemble')
    subset = (balanced_views(configurations, samples) if spin_flip_augment
              else configurations[:, :samples])
    return subset.to(dtype=DTYPE)


def evaluate(model, configurations, pattern_fn, samples: int, *, spin_flip_augment=True,
             objective_kwargs=None) -> dict[str, float]:
    """Evaluate QCR objective on a subset."""
    model.eval()
    subset = ensemble_views(configurations, samples, spin_flip_augment=spin_flip_augment)
    with torch.no_grad():
        total, terms = qcr_objective(model, subset, pattern_fn=pattern_fn, **(objective_kwargs or {}))
    return {"total": float(total)} | {key: float(value) for key, value in terms.items()}


def train(model, train_x, test_x, *, pattern_fn, epochs: int, system_batch: int,
          sample_batch: int, eval_samples: int, eval_systems: int,
          learning_rate: float, seed: int, progress_dir=None,
          spin_flip_augment=True, objective_kwargs=None, resume_state=None,
          initial_history=None) -> list[dict[str, float]]:
    """Joint QCR training."""
    parameters = (
        parameter
        for module in (model.theory_encoder, model.rg_kernel, model.theory_rg)
        for parameter in module.parameters()
    )
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    history = list(initial_history or [])
    start_epoch = 0
    if resume_state is not None:
        model.load_state_dict(resume_state['model'], strict=False)
        optimizer.load_state_dict(resume_state['optimizer'])
        generator.set_state(resume_state['batch_generator_state'])
        torch.set_rng_state(resume_state['torch_rng_state'])
        start_epoch = int(resume_state['epoch'])
        if epochs <= start_epoch:
            raise ValueError('Total epochs must exceed the resumed checkpoint epoch')
    record_every = max(1, epochs // 50)
    train_indices = torch.linspace(0, train_x.shape[0] - 1, min(eval_systems, train_x.shape[0])).long()
    test_indices = torch.linspace(0, test_x.shape[0] - 1, min(eval_systems, test_x.shape[0])).long()
    train_eval = train_x[train_indices]
    test_eval = test_x[test_indices]

    for epoch in range(start_epoch, epochs):
        model.train()
        systems = torch.randperm(train_x.shape[0], generator=generator)[:system_batch]
        half = sample_batch // 2
        samples = torch.randperm(train_x.shape[1], generator=generator)[:2 * half]
        selected = train_x[systems][:, samples].to(dtype=DTYPE)
        first_base, second_base = selected[:, :half], selected[:, half:]
        batch = (torch.cat((first_base, -first_base, second_base, -second_base), dim=1)
                 if spin_flip_augment else selected)

        optimizer.zero_grad(set_to_none=True)
        loss, _ = qcr_objective(model, batch, pattern_fn=pattern_fn, **(objective_kwargs or {}))
        loss.backward()
        optimizer.step()

        if progress_dir is not None and (epoch + 1) % 500 == 0:
            progress_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model': {k: v for k, v in model.state_dict().items()
                          if not k.startswith('action_network.')},
                'optimizer': optimizer.state_dict(),
                'batch_generator_state': generator.get_state(),
                'torch_rng_state': torch.get_rng_state(),
            }, progress_dir / f'training_step_{epoch + 1}.pt')

        if epoch == 0 or (epoch + 1) % record_every == 0 or epoch + 1 == epochs:
            train_metrics = evaluate(model, train_eval, pattern_fn, eval_samples, spin_flip_augment=spin_flip_augment, objective_kwargs=objective_kwargs)
            test_metrics = evaluate(model, test_eval, pattern_fn, eval_samples, spin_flip_augment=spin_flip_augment, objective_kwargs=objective_kwargs)
            history.append(
                {"epoch": epoch + 1}
                | {f"train_{k}": v for k, v in train_metrics.items()}
                | {f"test_{k}": v for k, v in test_metrics.items()}
            )
            print(
                f"epoch {epoch + 1}/{epochs}: "
                f"train={train_metrics['total']:.6g} test={test_metrics['total']:.6g}",
                flush=True,
            )
            if progress_dir is not None:
                progress_dir.mkdir(parents=True, exist_ok=True)
                (progress_dir / "history_progress.json").write_text(json.dumps(history, indent=2)+"\n")
                torch.save({k: v for k, v in model.state_dict().items()
                            if not k.startswith("action_network.")}, progress_dir / "qcr_latest.pt")
    return history


def trajectory_data(model, configurations, samples: int):
    """Collect direct and rollout trajectories."""
    model.eval()
    subset = balanced_views(configurations, samples)
    with torch.no_grad():
        levels = model.coarse_grain(subset)
        direct = torch.stack([model.theory_encoder(level) for level in levels], dim=1)
        rollout = [direct[:, 0]]
        for _ in range(model.levels - 1):
            rollout.append(model.advance_theory(rollout[-1]))
        rollout = torch.stack(rollout, dim=1)
        split = torch.stack([
            model.theory_encoder(level[:, : level.shape[1] // 2])
            for level in levels
        ], dim=1)
    return levels, direct.numpy(), rollout.numpy(), split.numpy()


def summarize_representation(direct, rollout, split, couplings):
    """Unified metrics for 1D/2D Ising."""
    levels = direct.shape[1]
    theory_dim = direct.shape[2]

    if theory_dim == 1:
        g = direct[:, :, 0]
        spearman = [rank_correlation(g[:, k], couplings) for k in range(levels)]
        std = [g[:, k].std() for k in range(levels)]
    else:
        g_norm = np.linalg.norm(direct, axis=2)
        spearman = [rank_correlation(g_norm[:, k], couplings) for k in range(levels)]
        std = [g_norm[:, k].std() for k in range(levels)]

    rmse = np.sqrt(np.mean((direct - rollout) ** 2, axis=(0, 2)))
    return {
        "absolute_spearman_g_j_by_level": [abs(float(s)) for s in spearman],
        "between_system_std_by_level": [float(s) for s in std],
        "rollout_direct_rmse_by_level": [float(r) for r in rmse],
    }


def find_fixed_points_1d(model, support_min, support_max):
    """Find fixed points of 1D scalar flow."""
    g = torch.linspace(support_min, support_max, 1000, dtype=DTYPE).unsqueeze(-1)
    with torch.no_grad():
        Rg = model.advance_theory(g).squeeze(-1).numpy()
    g_np = g.squeeze(-1).numpy()
    residual = Rg - g_np
    sign_changes = np.where(np.diff(np.sign(residual)))[0]
    roots = []
    for idx in sign_changes:
        g0, g1 = g_np[idx], g_np[idx + 1]
        r0, r1 = residual[idx], residual[idx + 1]
        g_star = g0 - r0 * (g1 - g0) / (r1 - r0)
        eps = 1e-4
        g_plus = torch.tensor([[g_star + eps]], dtype=DTYPE)
        g_minus = torch.tensor([[g_star - eps]], dtype=DTYPE)
        with torch.no_grad():
            R_plus = model.advance_theory(g_plus).item()
            R_minus = model.advance_theory(g_minus).item()
        derivative = (R_plus - R_minus) / (2 * eps)
        stability = "stable" if abs(derivative) < 1 else "unstable"
        roots.append({"g": float(g_star), "derivative": float(derivative), "stability": stability})
    return roots


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
