"""Historical fixed-spread loss; not used by current QCR training."""
import torch
import torch.nn.functional as F
from experiments._utils import QCRWeights
from experiments.ensemble_theory_encoder_ising_1d import encoder_objective
from recurrent_rg.metrics import mean_pairwise_square

def theory_diameter(theory: torch.Tensor) -> torch.Tensor:
    """Translation-invariant ensemble diameter; scalar case equals max-min."""
    if theory.shape[0] < 2:
        return theory.sum() * 0.0
    return torch.pdist(theory).max()



def legacy_qcr_objective(model, microscopic, *, pattern_fn,
                         weights=QCRWeights(anchor=2., separation=10.),
                         separation_fraction=.05, separation_decay=.5):
    from experiments.joint_qcr_ising import binary_kernel_noncollapse
    midpoint = microscopic.shape[1] // 2
    if midpoint < 1 or microscopic.shape[1] % 2:
        raise ValueError("Need even sample count")
    levels = model.coarse_grain(microscopic)
    first = torch.stack([model.theory_encoder(v[:, :midpoint]) for v in levels], 1)
    second = torch.stack([model.theory_encoder(v[:, midpoint:]) for v in levels], 1)
    direct = .5 * (first + second)
    anchor, _ = encoder_objective(
        model.theory_encoder,
        microscopic[:, :midpoint],
        microscopic[:, midpoint:],
        pattern_fn=pattern_fn,
    )
    coarse_consistency = F.mse_loss(first[:, 1:], second[:, 1:])
    kernel_noncollapse = binary_kernel_noncollapse(levels)

    pairwise = torch.stack([mean_pairwise_square(direct[:, k]) for k in range(model.levels)])
    split_noise = torch.stack(
        [F.mse_loss(first[:, k], second[:, k]) for k in range(model.levels)]
    )
    spread = direct.std(dim=0, unbiased=False).square().mean(dim=-1).sqrt()
    snr = spread / split_noise.sqrt().clamp_min(1e-12)

    # Absolute separation: require pairwise distance >= absolute floor
    absolute_floor = separation_fraction  # e.g. 0.05, treated as absolute distance
    range_target = 0.5  # require g to span at least this range per level

    separation_terms = []
    for level in range(1, model.levels):
        # Absolute pairwise distance floor
        penalty = F.relu(absolute_floor - pairwise[level])
        # Ensemble diameter, not a range mixing different coordinate axes.
        g_range = theory_diameter(direct[:, level])
        range_penalty = F.relu(range_target - g_range)
        separation_terms.append(separation_decay ** (level - 1) * (penalty + range_penalty))
    separation = torch.stack(separation_terms).sum()

    predicted = torch.stack(
        [model.advance_theory(direct[:, k]) for k in range(model.levels - 1)], dim=1
    )
    commutator = F.mse_loss(predicted, direct[:, 1:])
    total = (
        weights.anchor * anchor
        + weights.coarse_consistency * coarse_consistency
        + weights.kernel_noncollapse * kernel_noncollapse
        + weights.separation * separation
        + weights.commutator * commutator
    )
    terms = {
        "anchor": anchor,
        "coarse_consistency": coarse_consistency,
        "kernel_noncollapse": kernel_noncollapse,
        "separation": separation,
        "commutator": commutator,
    }
    for level in range(model.levels):
        terms[f"snr_level_{level}"] = snr[level]
        terms[f"pairwise_level_{level}"] = pairwise[level]
        if level > 0:
            terms[f"kernel_imbalance_level_{level}"] = levels[level].mean().abs()
    return total, terms
