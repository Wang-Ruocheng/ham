"""A minimal recurrent renormalization-group network."""

from .geometry import LatticeGeometry, chain_geometry, cubic_geometry, square_geometry
from .model_nd import (
    FixedMajorityCoarseGrainer2D,
    RecurrentRGNetwork,
    RecurrentRGNetwork1D,
    RecurrentRGNetwork2D,
    RGTrajectory,
)

__all__ = [
    "FixedMajorityCoarseGrainer2D",
    "LatticeGeometry",
    "RecurrentRGNetwork",
    "RecurrentRGNetwork1D",
    "RecurrentRGNetwork2D",
    "RGTrajectory",
    "chain_geometry",
    "cubic_geometry",
    "square_geometry",
]
