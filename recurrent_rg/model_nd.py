"""Dimension- and field-adaptable recurrent real-space RG network."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .adapters import BinaryFieldHead, FieldHead
from .geometry import LatticeGeometry, chain_geometry, square_geometry


@dataclass
class RGTrajectory:
    configurations: list[torch.Tensor]
    direct_theory: torch.Tensor
    rollout_theory: torch.Tensor


class LocalCoarseGrainer(nn.Module):
    def __init__(
        self,
        geometry: LatticeGeometry,
        input_channels: int,
        hidden: int,
        field_head: FieldHead,
        kernel_layout: str = "nonoverlapping",
    ) -> None:
        super().__init__()
        if field_head.output_channels != input_channels:
            raise ValueError(
                "a recurrent kernel currently requires equal microscopic and coarse channels"
            )
        self.geometry = geometry
        self.input_channels = input_channels
        if kernel_layout not in ("nonoverlapping", "overlapping"):
            raise ValueError("kernel_layout must be nonoverlapping or overlapping")
        self.kernel_layout = kernel_layout
        volume = geometry.block_volume if kernel_layout == "nonoverlapping" else geometry.patch_volume
        self.local_encoder = nn.Sequential(
            nn.Linear(volume * input_channels, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.field_head = field_head

    @torch.no_grad()
    def initialize_positive_decimation(self) -> None:
        """Initialize the binary map to the first block spin, without freezing C.

        A unit-weight path carries s0 through the two tanh layers and output
        head. The remaining hidden rows keep their random initialization;
        their output weights start at zero but remain trainable. No RNG draws
        or architectural changes, so paired Q/R initialization is preserved.
        """
        if self.kernel_layout != 'nonoverlapping' or self.input_channels != 1 or not isinstance(self.field_head, BinaryFieldHead):
            raise ValueError('Requires a nonoverlapping single-channel binary kernel')
        first, second = self.local_encoder[0], self.local_encoder[2]
        first.weight[0].zero_()
        first.weight[0, 0] = 1.
        first.bias[0] = 0.
        second.weight[0].zero_()
        second.weight[0, 0] = 1.
        second.bias[0] = 0.
        head = self.field_head.projection
        head.weight.zero_()
        head.weight[0, 0] = 1.
        head.bias.zero_()

    def local_features(self, configurations: torch.Tensor) -> torch.Tensor:
        extract = (self.geometry.extract_blocks if self.kernel_layout == "nonoverlapping"
                   else self.geometry.extract_patches)
        return self.local_encoder(extract(configurations))

    def information_field(self, configurations: torch.Tensor) -> torch.Tensor:
        # One hard field per block by default; the legacy layout produces one
        # field per microscopic site before downsampling. Both retain STE gradients.
        if getattr(self, 'categorical_states', 0):
            from .categorical_pooling import CategoricalLocal
            extract = self.geometry.extract_blocks if self.kernel_layout == 'nonoverlapping' else self.geometry.extract_patches
            logits = CategoricalLocal.apply(extract(configurations), self.categorical_states,
                lambda p: self.field_head.logits(self.local_encoder(p)), *self.parameters())
            return self.field_head.sample_logits(logits)
        if getattr(self, "binary_lookup", False):
            from .binary_pooling import BinaryLocal
            extract = (self.geometry.extract_blocks if self.kernel_layout == "nonoverlapping"
                       else self.geometry.extract_patches)
            if getattr(self.field_head, 'sampling_mode', 'deterministic') == 'bernoulli':
                logits = BinaryLocal.apply(extract(configurations),
                    lambda p: self.field_head.logits(self.local_encoder(p)), *self.parameters())
                return self.field_head.sample_logits(logits)
            return BinaryLocal.apply(extract(configurations),
                                     lambda p: self.field_head(self.local_encoder(p)),
                                     *self.parameters())
        return self.field_head(self.local_features(configurations))

    def forward(self, configurations: torch.Tensor) -> torch.Tensor:
        fields = self.information_field(configurations)
        if self.kernel_layout == "nonoverlapping":
            return fields
        return self.geometry.downsample(fields)


class FixedDecimationCoarseGrainer(nn.Module):
    """Non-learned coarse grainer: plain decimation (take every block_factor-th site).

    This retains sites without bond moving or a Hamiltonian truncation; it is
    not a Migdal--Kadanoff coupling recursion. All input channels are retained.
    It has no trainable parameters and is differentiable on the kept sites.
    """

    def __init__(self, geometry: LatticeGeometry) -> None:
        super().__init__()
        self.geometry = geometry

    def information_field(self, configurations: torch.Tensor) -> torch.Tensor:
        return configurations

    def forward(self, configurations: torch.Tensor) -> torch.Tensor:
        return self.geometry.downsample(configurations)


class EnsembleTheoryEncoder(nn.Module):
    def __init__(
        self,
        geometry: LatticeGeometry,
        channels: int,
        theory_dim: int,
        hidden: int,
        features: int,
    ) -> None:
        super().__init__()
        self.geometry = geometry
        self.local_mlp = nn.Sequential(
            nn.Linear(geometry.patch_volume * channels, hidden),
            nn.Tanh(),
            nn.Linear(hidden, features),
            nn.Tanh(),
            nn.Linear(features, features),
            nn.Tanh(),
        )
        self.head = nn.Sequential(
            nn.Linear(features, hidden), nn.Tanh(), nn.Linear(hidden, theory_dim)
        )

    def forward(self, configurations: torch.Tensor) -> torch.Tensor:
        patches = self.geometry.extract_patches(configurations)
        spatial_dims = tuple(range(1, patches.ndim - 1))
        # Keep the system axis and average over samples and lattice translations.
        if getattr(self, 'categorical_states', 0):
            from .categorical_pooling import CategoricalMean
            return self.head(CategoricalMean.apply(patches, self.categorical_states, self.local_mlp, *self.local_mlp.parameters()))
        if getattr(self, "binary_lookup", False):
            from .binary_pooling import binary_mean
            return self.head(binary_mean(patches, self.local_mlp))
        if getattr(self, "checkpoint_chunks", 0) and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            def pooled(chunk):
                return self.local_mlp(chunk).mean(dim=spatial_dims)
            features = torch.cat([
                checkpoint(pooled, chunk, use_reentrant=False)
                for chunk in patches.split(self.checkpoint_chunks)
            ])
            return self.head(features)
        return self.head(self.local_mlp(patches).mean(dim=spatial_dims))


class LocalNeuralAction(nn.Module):
    def __init__(
        self,
        geometry: LatticeGeometry,
        channels: int,
        theory_dim: int,
        hidden: int,
    ) -> None:
        super().__init__()
        self.geometry = geometry
        self.theory_dim = theory_dim
        self.local_energy = nn.Sequential(
            nn.Linear(geometry.patch_volume * channels + theory_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, configurations: torch.Tensor, theory: torch.Tensor) -> torch.Tensor:
        patches = self.geometry.extract_patches(configurations)
        spatial_shape = patches.shape[2:-1]
        condition = theory[(slice(None),) + (None,) * (1 + len(spatial_shape)) + (slice(None),)]
        condition = condition.expand(
            configurations.shape[0], configurations.shape[1], *spatial_shape, self.theory_dim
        )
        density = self.local_energy(torch.cat((patches, condition), dim=-1)).squeeze(-1)
        return density.sum(dim=tuple(range(2, 2 + self.geometry.dimensions)))


class RecurrentRGNetwork(nn.Module):
    def __init__(
        self,
        geometry: LatticeGeometry,
        field_head: FieldHead,
        *,
        input_channels: int,
        levels: int = 3,
        theory_dim: int = 1,
        hidden: int = 24,
        local_features: int = 12,
        learnable_kernel: bool = True,
        kernel_layout: str = "nonoverlapping",
    ) -> None:
        super().__init__()
        if levels < 2 or theory_dim < 1:
            raise ValueError("levels must be >=2 and theory_dim must be positive")
        self.geometry = geometry
        self.levels = levels
        self.theory_dim = theory_dim
        if learnable_kernel:
            self.rg_kernel = LocalCoarseGrainer(
                geometry, input_channels, hidden, field_head, kernel_layout
            )
        else:
            self.rg_kernel = FixedDecimationCoarseGrainer(geometry)
        self.theory_encoder = EnsembleTheoryEncoder(
            geometry, input_channels, theory_dim, hidden, local_features
        )
        self.theory_rg = nn.Sequential(
            nn.Linear(theory_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, theory_dim),
        )
        self.action_network = LocalNeuralAction(
            geometry, input_channels, theory_dim, hidden
        )

    def advance_theory(self, theory: torch.Tensor) -> torch.Tensor:
        return theory + self.theory_rg(theory)

    def coarse_grain(self, microscopic: torch.Tensor, *, max_rg_steps: int | None = None) -> list[torch.Tensor]:
        """Optional evaluation depth does not mutate the training depth."""
        steps = self.levels - 1 if max_rg_steps is None else max_rg_steps
        if not isinstance(steps, int) or steps < 0:
            raise ValueError('max_rg_steps must be a nonnegative integer')
        configurations = [microscopic]
        for _ in range(steps):
            configurations.append(self.rg_kernel(configurations[-1]))
        return configurations

    def forward(self, microscopic: torch.Tensor) -> RGTrajectory:
        configurations = self.coarse_grain(microscopic)
        direct = torch.stack(
            [self.theory_encoder(level) for level in configurations], dim=1
        )
        rollout = [direct[:, 0]]
        for _ in range(self.levels - 1):
            rollout.append(self.advance_theory(rollout[-1]))
        return RGTrajectory(
            configurations=configurations,
            direct_theory=direct,
            rollout_theory=torch.stack(rollout, dim=1),
        )





class RecurrentRGNetwork1D(RecurrentRGNetwork):
    """Periodic chain binary adapter — convenience constructor for 1D."""

    def __init__(
        self,
        levels: int = 3,
        width: int = 3,
        hidden: int = 48,
        theory_dim: int = 1,
        block_factor: int = 2,
        learnable_kernel: bool = True,
        kernel_layout: str = "nonoverlapping",
    ) -> None:
        geometry = chain_geometry(width, block_factor)
        super().__init__(
            geometry,
            BinaryFieldHead(hidden),
            input_channels=1,
            levels=levels,
            theory_dim=theory_dim,
            hidden=hidden,
            learnable_kernel=learnable_kernel,
            kernel_layout=kernel_layout,
        )


class RecurrentRGNetwork2D(RecurrentRGNetwork):
    """Periodic square-lattice binary model — convenience constructor for 2D."""

    def __init__(
        self,
        levels: int = 3,
        width: int = 3,
        hidden: int = 24,
        theory_dim: int = 1,
        block_factor: int = 2,
        kernel_layout: str = "nonoverlapping",
        learnable_kernel: bool = True,
    ) -> None:
        geometry = square_geometry(width, block_factor)
        super().__init__(
            geometry,
            BinaryFieldHead(hidden),
            input_channels=1,
            levels=levels,
            theory_dim=theory_dim,
            hidden=hidden,
            kernel_layout=kernel_layout,
            learnable_kernel=learnable_kernel,
        )
