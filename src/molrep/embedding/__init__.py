"""molrep embedding components.

Provides embedding and feature extraction modules:
- JointEmbedding: Combined discrete + continuous embedding
- SphericalHarmonics: Equivariant angular basis functions
- BesselRBF / GaussianBasis / PolynomialBasis: Radial basis functions
- CosineCutoff / TanhCutoff / HalfCosineCutoff / PolynomialCutoff: Cutoff envelopes
"""

from .angular import SphericalHarmonics
from .cutoff import CosineCutoff, HalfCosineCutoff, PolynomialCutoff, TanhCutoff
from .node import JointEmbedding
from .radial import BesselRBF, GaussianBasis, PolynomialBasis

__all__ = [
    "BesselRBF",
    "CosineCutoff",
    "GaussianBasis",
    "HalfCosineCutoff",
    "JointEmbedding",
    "PolynomialBasis",
    "PolynomialCutoff",
    "SphericalHarmonics",
    "TanhCutoff",
]
