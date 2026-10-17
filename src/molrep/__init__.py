"""molrep: Molecular representation learning.

Provides components for building molecular representation models:

Embeddings:
  - JointEmbedding: Combined discrete + continuous embedding
  - SphericalHarmonics: Equivariant angular basis functions
  - BesselRBF: Radial basis functions with Bessel functions
  - CosineCutoff: Cosine-based cutoff envelope
  - PolynomialCutoff: Polynomial-based cutoff envelope

Interactions (``molrep.interaction``):
  - irreps_from_l_max / sh_irreps_from_l_max: irreps-string helpers; the
    message-passing blocks live with their model (``molzoo.mace.layers``)

Heads (``molrep.heads``):
  - ScalarHead: Pooling + MLP for scalar prediction
"""

from molrep.embedding import (
    BesselRBF,
    CosineCutoff,
    JointEmbedding,
    PolynomialCutoff,
    SphericalHarmonics,
)
from molrep.heads.scalar import ScalarHead

__all__ = [
    "JointEmbedding",
    "SphericalHarmonics",
    "BesselRBF",
    "CosineCutoff",
    "PolynomialCutoff",
    "ScalarHead",
]
