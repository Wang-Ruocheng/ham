"""A minimal recurrent renormalization-group network."""

from .geometry import LatticeGeometry, chain_geometry, cubic_geometry, square_geometry
from .model_nd import (
    RecurrentRGNetwork,
    RecurrentRGNetwork1D,
    RecurrentRGNetwork2D,
    RGTrajectory,
)

__all__ = [
    "LatticeGeometry",
    "RecurrentRGNetwork",
    "RecurrentRGNetwork1D",
    "RecurrentRGNetwork2D",
    "RGTrajectory",
    "chain_geometry",
    "cubic_geometry",
    "square_geometry",
]
