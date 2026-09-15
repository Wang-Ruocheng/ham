"""Replaceable field and local-conditional adapters for recurrent RG."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import nn


class FieldHead(nn.Module, ABC):
    """Map local features to a coarse variable in a chosen field space."""

    output_channels: int

    @abstractmethod
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def relaxed(self, features: torch.Tensor) -> torch.Tensor:
        """Differentiable (relaxed) representation used for straight-through gradient estimation."""
        raise NotImplementedError


class BinaryFieldHead(FieldHead):
    """Straight-through binary variable; no Z2 symmetry is imposed."""

    def __init__(self, features: int) -> None:
        super().__init__()
        self.output_channels = 1
        self.projection = nn.Linear(features, 1)

    def logits(self, features: torch.Tensor) -> torch.Tensor:
        return self.projection(features)

    def relaxed(self, features: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.logits(features))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        relaxed = self.relaxed(features)
        hard = torch.where(relaxed >= 0, 1.0, -1.0)
        return hard.detach() + relaxed - relaxed.detach()


class ContinuousFieldHead(FieldHead):
    """Unconstrained real-valued coarse field."""

    def __init__(self, features: int, channels: int) -> None:
        super().__init__()
        self.output_channels = channels
        self.projection = nn.Linear(features, channels)

    def relaxed(self, features: torch.Tensor) -> torch.Tensor:
        return self.projection(features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.relaxed(features)


class CategoricalFieldHead(FieldHead):
    """Straight-through one-hot coarse variable for Potts-like fields."""

    def __init__(self, features: int, states: int, temperature: float = 1.0) -> None:
        super().__init__()
        if states < 2 or temperature <= 0:
            raise ValueError("states must be >=2 and temperature must be positive")
        self.output_channels = states
        self.temperature = temperature
        self.projection = nn.Linear(features, states)

    def relaxed(self, features: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.projection(features) / self.temperature, dim=-1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        relaxed = self.relaxed(features)
        hard = F.one_hot(relaxed.argmax(dim=-1), self.output_channels).to(relaxed.dtype)
        return hard.detach() + relaxed - relaxed.detach()


class LocalConditional(nn.Module, ABC):
    """Field-specific local conditional loss for an unnormalized action."""

    @abstractmethod
    def forward(
        self, action: nn.Module, configurations: torch.Tensor, theory: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError


class BinaryFlipConditional(LocalConditional):
    """Single-site pseudolikelihood for binary site variables, in any dimension."""

    def forward(
        self, action: nn.Module, configurations: torch.Tensor, theory: torch.Tensor
    ) -> torch.Tensor:
        if configurations.shape[-1] != 1:
            raise ValueError("BinaryFlipConditional expects one field channel")
        if not hasattr(action, "geometry") or not hasattr(action, "local_energy"):
            flipped = configurations.clone()
            origin = (slice(None), slice(None)) + (0,) * (configurations.ndim - 3) + (0,)
            flipped[origin] *= -1.0
            return F.softplus(action(configurations, theory) - action(flipped, theory)).mean()

        # For a local action, flipping the origin affects only the patches whose
        # centers lie inside one patch radius. This keeps the adapter dimension
        # independent without reevaluating the full lattice action twice.
        geometry = action.geometry
        patches = geometry.extract_patches(configurations)
        offsets = geometry.patch_offsets()
        spatial_shape = configurations.shape[2:-1]
        original = torch.stack(
            [
                patches[(slice(None), slice(None)) + tuple(
                    offset % size for offset, size in zip(center, spatial_shape)
                ) + (slice(None),)]
                for center in offsets
            ],
            dim=2,
        )
        flipped = original.clone()
        offset_to_index = {offset: index for index, offset in enumerate(offsets)}
        for center_index, center in enumerate(offsets):
            origin_in_patch = tuple(-coordinate for coordinate in center)
            patch_index = offset_to_index[origin_in_patch]
            flipped[:, :, center_index, patch_index] *= -1.0

        condition = theory[:, None, None, :].expand(
            configurations.shape[0], configurations.shape[1], len(offsets), theory.shape[-1]
        )
        original_density = action.local_energy(
            torch.cat((original, condition), dim=-1)
        ).squeeze(-1)
        flipped_density = action.local_energy(
            torch.cat((flipped, condition), dim=-1)
        ).squeeze(-1)
        return F.softplus((original_density - flipped_density).sum(dim=2)).mean()


class CategoricalSiteConditional(LocalConditional):
    """Exact single-site pseudolikelihood for one-hot categorical fields."""

    def __init__(self, states: int) -> None:
        super().__init__()
        if states < 2:
            raise ValueError("states must be at least two")
        self.states = states

    def forward(
        self, action: nn.Module, configurations: torch.Tensor, theory: torch.Tensor
    ) -> torch.Tensor:
        if configurations.shape[-1] != self.states:
            raise ValueError("the field channel count must equal the number of states")
        origin = (slice(None), slice(None)) + (0,) * (configurations.ndim - 3)
        target = configurations[origin].argmax(dim=-1)
        energies = []
        for state in range(self.states):
            alternative = configurations.clone()
            alternative[origin] = F.one_hot(
                torch.full_like(target, state), self.states
            ).to(configurations.dtype)
            energies.append(action(alternative, theory))
        logits = -torch.stack(energies, dim=-1)
        return F.cross_entropy(logits.flatten(0, 1), target.flatten())


class DenoisingScoreConditional(LocalConditional):
    """Denoising score-matching loss for unconstrained continuous fields."""

    def __init__(self, noise_std: float = 0.1) -> None:
        super().__init__()
        if noise_std <= 0:
            raise ValueError("noise_std must be positive")
        self.noise_std = noise_std

    def forward(
        self, action: nn.Module, configurations: torch.Tensor, theory: torch.Tensor
    ) -> torch.Tensor:
        noise = torch.randn_like(configurations)
        noisy = (configurations + self.noise_std * noise).requires_grad_(True)
        energy = action(noisy, theory).sum()
        predicted_score = -torch.autograd.grad(energy, noisy, create_graph=True)[0]
        target_score = -noise / self.noise_std
        return self.noise_std**2 * (predicted_score - target_score).square().mean()
