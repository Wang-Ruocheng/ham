"""Dimension-independent periodic lattice geometry utilities."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import prod

import torch


def _as_tuple(value: int | tuple[int, ...], dimensions: int) -> tuple[int, ...]:
    if isinstance(value, int):
        return (value,) * dimensions
    if len(value) != dimensions:
        raise ValueError(f"expected {dimensions} entries, got {len(value)}")
    return tuple(value)


@dataclass(frozen=True)
class LatticeGeometry:
    """Local neighborhoods and blocking on a periodic hypercubic lattice.

    Configurations always have shape ``[systems, samples, *spatial, channels]``.
    A different lattice only needs to provide the same geometry methods; the RG
    model and objectives do not otherwise assume two spatial dimensions.
    """

    dimensions: int
    patch_width: int | tuple[int, ...] = 3
    block_factor: int | tuple[int, ...] = 2
    periodic: bool = True

    def __post_init__(self) -> None:
        widths = _as_tuple(self.patch_width, self.dimensions)
        factors = _as_tuple(self.block_factor, self.dimensions)
        if self.dimensions < 1:
            raise ValueError("dimensions must be positive")
        if any(width < 1 or width % 2 == 0 for width in widths):
            raise ValueError("centered patch widths must be positive odd integers")
        if any(factor < 1 for factor in factors):
            raise ValueError("block factors must be positive")
        if not self.periodic:
            raise NotImplementedError("only periodic boundaries are implemented")

    @property
    def widths(self) -> tuple[int, ...]:
        return _as_tuple(self.patch_width, self.dimensions)

    @property
    def factors(self) -> tuple[int, ...]:
        return _as_tuple(self.block_factor, self.dimensions)

    @property
    def patch_volume(self) -> int:
        return prod(self.widths)

    @property
    def spatial_dims(self) -> tuple[int, ...]:
        return tuple(range(2, 2 + self.dimensions))

    def validate(self, configurations: torch.Tensor) -> None:
        expected = self.dimensions + 3
        if configurations.ndim != expected:
            raise ValueError(
                f"expected [systems,samples,*spatial,channels] with {expected} axes"
            )

    def patch_offsets(self) -> tuple[tuple[int, ...], ...]:
        ranges = [range(-(width // 2), width // 2 + 1) for width in self.widths]
        return tuple(product(*ranges))

    def extract_patches(self, configurations: torch.Tensor) -> torch.Tensor:
        """Return a centered local patch at every lattice site."""
        self.validate(configurations)
        shifted = [
            torch.roll(
                configurations,
                shifts=tuple(-offset for offset in offsets),
                dims=self.spatial_dims,
            )
            for offsets in self.patch_offsets()
        ]
        patches = torch.stack(shifted, dim=-2)
        return patches.flatten(start_dim=-2)

    def downsample(self, fields: torch.Tensor) -> torch.Tensor:
        """Select one coarse variable per blocking cell."""
        self.validate(fields)
        spatial_shape = fields.shape[2 : 2 + self.dimensions]
        if any(length % factor for length, factor in zip(spatial_shape, self.factors)):
            raise ValueError("every spatial size must be divisible by its block factor")
        selection = (slice(None), slice(None)) + tuple(
            slice(None, None, factor) for factor in self.factors
        ) + (slice(None),)
        return fields[selection]

    @property
    def block_volume(self) -> int:
        return prod(self.factors)

    def extract_blocks(self, configurations: torch.Tensor) -> torch.Tensor:
        """Tile the lattice from index zero, with no overlap or dropped sites.

        Returns [systems, samples, *coarse_shape, block_volume * channels].
        Sites within a block are ordered lexicographically, channels last.
        This is independent of the centered patch width used by Q and A.
        """
        self.validate(configurations)
        spatial_shape = configurations.shape[2 : 2 + self.dimensions]
        if any(length % factor for length, factor in zip(spatial_shape, self.factors)):
            raise ValueError("every spatial size must be divisible by its block factor")
        blocks = []
        for offsets in product(*(range(factor) for factor in self.factors)):
            selection = (slice(None), slice(None)) + tuple(
                slice(offset, None, factor) for offset, factor in zip(offsets, self.factors)
            ) + (slice(None),)
            blocks.append(configurations[selection])
        return torch.stack(blocks, dim=-2).flatten(start_dim=-2)

    def environment_offsets(self, buffer: int = 1) -> tuple[tuple[int, ...], ...]:
        """Probe the corners/faces of a shell outside the visible patch."""
        if buffer < 0:
            raise ValueError("buffer must be non-negative")
        radii = tuple(width // 2 + buffer + 1 for width in self.widths)
        offsets = product(*[(-radius, 0, radius) for radius in radii])
        zero = (0,) * self.dimensions
        return tuple(offset for offset in offsets if offset != zero)

    def extract_environment(self, configurations: torch.Tensor, buffer: int = 1) -> torch.Tensor:
        """Return environment probes without discretizing or histogramming them."""
        self.validate(configurations)
        probes = [
            torch.roll(
                configurations,
                shifts=tuple(-offset for offset in offsets),
                dims=self.spatial_dims,
            )
            for offsets in self.environment_offsets(buffer)
        ]
        return torch.stack(probes, dim=-2).flatten(start_dim=-2)


def square_geometry(width: int = 3, block_factor: int = 2) -> LatticeGeometry:
    return LatticeGeometry(2, width, block_factor)


def chain_geometry(width: int = 3, block_factor: int = 2) -> LatticeGeometry:
    return LatticeGeometry(1, width, block_factor)


def cubic_geometry(width: int = 3, block_factor: int = 2) -> LatticeGeometry:
    return LatticeGeometry(3, width, block_factor)
