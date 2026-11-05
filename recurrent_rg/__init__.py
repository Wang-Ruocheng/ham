"""A minimal recurrent renormalization-group network."""

from .geometry import LatticeGeometry, anchored_cubic_geometry, chain_geometry, cubic_geometry, square_geometry
from .model_nd import (
    RecurrentRGNetwork,
    RecurrentRGNetwork1D,
    RecurrentRGNetwork2D,
    RecurrentRGNetwork3D,
    RGTrajectory,
)

__all__ = [
    "LatticeGeometry",
    "RecurrentRGNetwork",
    "RecurrentRGNetwork1D",
    "RecurrentRGNetwork2D",
    "RecurrentRGNetwork3D",
    "RGTrajectory",
    "chain_geometry",
    "anchored_cubic_geometry",
    "cubic_geometry",
    "square_geometry",
]
