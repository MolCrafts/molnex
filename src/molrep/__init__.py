"""molrep: Molecular representation learning.

Provides components for building molecular representation models:

Embeddings:
  - JointEmbedding: Combined discrete + continuous embedding
  - SphericalHarmonics: Equivariant angular basis functions
  - BesselRBF: Radial basis functions with Bessel functions
  - CosineCutoff: Cosine-based cutoff envelope
  - PolynomialCutoff: Polynomial-based cutoff envelope

Heads and Pooling:
  - ScalarHead: Pooling + MLP for scalar prediction
  - EnergyHead: Energy prediction head with scatter pooling
  - ForceHead: Compute forces via autograd -dE/dpos
  - StressHead: Compute stress tensor via autograd
  - masked_sum_pooling: Sum pooling with mask support
  - masked_mean_pooling: Mean pooling with mask support
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
from molrep.readout.heads import EnergyHead, ForceHead, StressHead

__all__ = [
    "JointEmbedding",
    "SphericalHarmonics",
    "BesselRBF",
    "CosineCutoff",
    "PolynomialCutoff",
    "ScalarHead",
    "masked_sum_pooling",
    "masked_mean_pooling",
    "EnergyHead",
    "ForceHead",
    "StressHead",
]
