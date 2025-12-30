"""molrep: Molecular representation learning with SDPA + NestedTensor.

Provides components for building molecular representation models:
- AtomEmbedding: Embed atomic numbers and positions
- TransformerBlock: Single SDPA-based transformer block
- TransformerEncoder: Stack of transformer blocks
- ScalarHead: Pooling + MLP for scalar prediction
"""

from molrep.encoder.embedding import AtomEmbedding
from molrep.encoder.transformer import TransformerBlock, TransformerEncoder
from molrep.head.scalar_head import ScalarHead

# Functional API (PyTorch-style)
from molrep import F

__all__ = [
    "AtomEmbedding",
    "TransformerBlock",
    "TransformerEncoder",
    "ScalarHead",
    "F",  # Functional API
]
