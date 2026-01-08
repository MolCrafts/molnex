"""molrep: Molecular representation learning.

Provides components for building molecular representation models:
- AtomTypeEmbedding: Embed atomic numbers to learned embeddings
- TransformerBlock: Single SDPA-based transformer block
- TransformerEncoder: Stack of transformer blocks
- EdgeBiasMLP: Convert RBF features to attention bias
- ScalarHead: Pooling + MLP for scalar prediction
- EnergyHead: Energy prediction head with masked pooling
- EquivariantPotentialNet: Equivariant potential network with cuEquivariance
"""

from molrep.encoder.embedding import AtomTypeEmbedding
from molrep.encoder.transformer import TransformerBlock, TransformerEncoder
from molrep.encoder.edge_bias import EdgeBiasMLP, densify_edge_bias
from molrep.head.scalar_head import ScalarHead
from molrep.readout.pooling import masked_sum_pooling, masked_mean_pooling
from molrep.readout.heads import EnergyHead
from molrep.models.equivariant_potential import EquivariantPotentialNet

__all__ = [
    "AtomTypeEmbedding",
    "TransformerBlock",
    "TransformerEncoder",
    "EdgeBiasMLP",
    "densify_edge_bias",
    "ScalarHead",
    "masked_sum_pooling",
    "masked_mean_pooling",
    "EnergyHead",
    "EquivariantPotentialNet",
]
