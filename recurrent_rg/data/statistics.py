"""Binary local distribution descriptors, separate from the QCR objective."""
import torch
import torch.nn.functional as F


def categorical_plaquette_distribution(configurations):
    """Empirical probabilities of q**4 oriented plaquettes; no labels."""
    q = configurations.shape[-1]
    state = configurations.argmax(-1)
    pattern = (q**3*state + q**2*torch.roll(state,-1,2) + q*torch.roll(state,-1,3)
               + torch.roll(torch.roll(state,-1,2),-1,3))
    flat = pattern.reshape(len(pattern),-1)
    offsets = torch.arange(len(flat),device=flat.device)[:,None]*q**4
    counts = torch.bincount((flat+offsets).flatten(),minlength=len(flat)*q**4)
    return counts.reshape(len(flat),q**4).to(configurations.dtype)/flat.shape[1]

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
