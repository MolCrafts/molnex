"""molrep embedding components.

Provides embedding and feature extraction modules:
- JointEmbedding: Combined discrete + continuous embedding
- SphericalHarmonics: Equivariant angular basis functions
- BesselRBF: Radial basis functions
- CosineCutoff: Cosine-based cutoff envelope
- PolynomialCutoff: Polynomial-based cutoff envelope
"""

from .angular import SphericalHarmonics
from .cutoff import CosineCutoff, PolynomialCutoff
from .node import JointEmbedding
from .radial import BesselRBF

__all__ = [
    "JointEmbedding",
    "SphericalHarmonics",
    "BesselRBF",
    "CosineCutoff",
    "PolynomialCutoff",
]
