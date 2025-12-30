"""molrep encoder components.

Provides SDPA-based transformer encoder with NestedTensor support.
"""

from molrep.encoder.embedding import AtomEmbedding
from molrep.encoder.transformer import TransformerBlock, TransformerEncoder

__all__ = [
    "AtomEmbedding",
    "TransformerBlock",
    "TransformerEncoder",
]
