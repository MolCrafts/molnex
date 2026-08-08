"""molrep embedding components.

Provides embedding and feature extraction modules:
- JointEmbedding: Combined discrete + continuous embedding
- SphericalHarmonics: Equivariant angular basis functions
- BesselRBF / GaussianBasis / PolynomialBasis: Radial basis functions
- CosineCutoff / TanhCutoff / HalfCosineCutoff / PolynomialCutoff: Cutoff envelopes
"""

from .angular import SphericalHarmonics
from .covalent import covalent_radii
from .cutoff import CosineCutoff, HalfCosineCutoff, PolynomialCutoff, TanhCutoff
from .mlp import MomentNormalizedMLP, normalize2mom
from .node import JointEmbedding, JointFeatureEmbedding, JointFeatureSpec
from .radial import (
    AgnesiTransform,
    BesselRBF,
    GaussianBasis,
    PolynomialBasis,
)

__all__ = [
    "AgnesiTransform",
    "BesselRBF",
    "covalent_radii",
    "MomentNormalizedMLP",
    "normalize2mom",
    "CosineCutoff",
    "GaussianBasis",
    "HalfCosineCutoff",
    "JointEmbedding",
    "JointFeatureEmbedding",
    "JointFeatureSpec",
    "PolynomialBasis",
    "PolynomialCutoff",
    "SphericalHarmonics",
    "TanhCutoff",
]
