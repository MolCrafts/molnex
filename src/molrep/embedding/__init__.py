"""molrep embedding components.

Provides embedding and feature extraction modules:
- JointEmbedding: Combined discrete + continuous embedding
- SphericalHarmonics: Equivariant angular basis functions
- BesselRBF: Radial basis functions
- CosineCutoff: Cosine-based cutoff envelope
- PolynomialCutoff: Polynomial-based cutoff envelope
"""

from .node import JointEmbedding
from .cutoff import CosineCutoff, PolynomialCutoff
from .radial import BesselRBF
from .angular import SphericalHarmonics

__all__ = [
    "JointEmbedding",
    "SphericalHarmonics",
    "BesselRBF",
    "CosineCutoff",
    "PolynomialCutoff",
]

