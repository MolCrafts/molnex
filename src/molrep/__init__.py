"""molrep: Molecular representation learning.

Provides components for building molecular representation models:

Embeddings:
  - JointEmbedding: Combined discrete + continuous embedding
  - SphericalHarmonics: Equivariant angular basis functions
  - BesselRBF: Radial basis functions with Bessel functions
  - CosineCutoff: Cosine-based cutoff envelope
  - PolynomialCutoff: Polynomial-based cutoff envelope

Interactions:
  - ConvTP: Tensor product convolution
  - RadialWeightMLP: Radial-to-TP-weight MLP
  - SymmetricContraction: Multi-body basis construction
  - EquivariantLinear: SO(3)-equivariant linear
  - ElementUpdate: Element-specific residual
  - MessageAggregation: Scatter-sum aggregation

Readout:
  - ProductHead: Multi-body basis → scalar features
  - ScalarHead: Pooling + MLP for scalar prediction
  - masked_sum_pooling / masked_mean_pooling
"""

from molrep.embedding import (
    JointEmbedding,
    SphericalHarmonics,
    BesselRBF,
    CosineCutoff,
    PolynomialCutoff,
)
from molrep.head.scalar_head import ScalarHead
from molrep.readout.pooling import masked_sum_pooling, masked_mean_pooling
from molrep.readout.product_head import ProductHead

__all__ = [
    "JointEmbedding",
    "SphericalHarmonics",
    "BesselRBF",
    "CosineCutoff",
    "PolynomialCutoff",
    "ScalarHead",
    "ProductHead",
    "masked_sum_pooling",
    "masked_mean_pooling",
]
