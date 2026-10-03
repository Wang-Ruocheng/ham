"""Sample selection and explicit symmetry augmentation for ensembles."""
import torch

def fixed_eval_subset(
    configurations: torch.Tensor, systems: int, samples: int
) -> torch.Tensor:
    """Deterministic grid of systems × samples for loss-curve evaluation."""
    count = min(systems, configurations.shape[0])
    indices = torch.linspace(0, configurations.shape[0] - 1, count).long()
    return configurations[indices, : min(samples, configurations.shape[1])]


def balanced_views(configurations: torch.Tensor, samples: int) -> torch.Tensor:
    """Build two exactly spin-flip-balanced ensemble views of ``samples`` each.

    Returns ``2 * samples`` configurations where the first half are the two
    views (each of size ``samples``).  Every base configuration is paired with
    its global spin flip, preventing the encoder from exploiting spontaneous
    magnetisation to distinguish theories.
    """
    if samples % 2:
        raise ValueError("sample count must be even for exact spin-flip pairing")
    half = samples // 2
    if 2 * half > configurations.shape[1]:
        raise ValueError("not enough configurations for two balanced views")
    first = configurations[:, :half]
    second = configurations[:, half : 2 * half]
    return torch.cat((first, -first, second, -second), dim=1)

def ensemble_views(configurations, samples, *, spin_flip_augment=True):
    """Nonzero-field ensembles must never be balanced by artificial spin flips."""
    if samples < 2 or samples % 2 or samples > configurations.shape[1]:
        raise ValueError('Need an even sample count available in every ensemble')
    subset = (balanced_views(configurations, samples) if spin_flip_augment
              else configurations[:, :samples])
    return subset.to(dtype=torch.float64)
