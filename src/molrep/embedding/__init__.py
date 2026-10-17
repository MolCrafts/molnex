"""molrep embedding components.

Provides embedding and feature extraction modules:
- JointEmbedding: Combined discrete + continuous embedding
- SphericalHarmonics: Equivariant angular basis functions
- BesselRBF / GaussianBasis / PolynomialBasis: Radial basis functions
- CosineCutoff / TanhCutoff / HalfCosineCutoff / PolynomialCutoff: Cutoff envelopes
- ChemicalSupportIndex: L2 kNN chemical-support bank (provenance surfaces)
"""

from .angular import SphericalHarmonics
from .covalent import covalent_radii
from .cutoff import CosineCutoff, HalfCosineCutoff, PolynomialCutoff, TanhCutoff
from .geometry import edge_lengths, edge_vectors
from .mlp import FFLayer, MomentNormalizedMLP, ScalarMLPFunction, normalize2mom
from .node import JointEmbedding
from .radial import (
    AgnesiTransform,
    BesselRBF,
    GaussianBasis,
    PolynomialBasis,
    SincBesselRBF,
)
from .support import ChemicalSupportIndex

__all__ = [
    "AgnesiTransform",
    "BesselRBF",
    "ChemicalSupportIndex",
    "covalent_radii",
    "edge_lengths",
    "edge_vectors",
    "FFLayer",
    "MomentNormalizedMLP",
    "normalize2mom",
    "CosineCutoff",
    "GaussianBasis",
    "HalfCosineCutoff",
    "JointEmbedding",
    "PolynomialBasis",
    "PolynomialCutoff",
    "ScalarMLPFunction",
    "SincBesselRBF",
    "SphericalHarmonics",
    "TanhCutoff",
]
