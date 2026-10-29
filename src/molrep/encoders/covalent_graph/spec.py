"""Validated configuration of the covalent-graph encoder.

Torch-free: imports only ``pydantic``.

Reference:
    Zheng, T. et al. "Data-driven parametrization of molecular mechanics force
    fields for expansive chemical space coverage." Chem. Sci. 2025.
    https://arxiv.org/abs/2408.12817
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["CovalentGraphEncoderSpec"]


class CovalentGraphEncoderSpec(BaseModel):
    """Configuration of :class:`~molrep.encoders.covalent_graph.encoder.CovalentGraphEncoder`.

    The defaults are the published model's sizes (the reference
    ``fftrainer_config_in_use.yaml``).

    Attributes:
        atom_embedding_dim: Width of the element embedding.
        channel_dim: Width of the degree / charge / ring embeddings.
        node_mlp: ``(hidden, out, layers)`` of the atom input MLP.
        edge_mlp: ``(hidden, out, layers)`` of the bond input MLP.
        width: Node and edge width of the graph transformer, and of the atom
            features handed to the heads.
        n_layers: Transformer layers; the receptive field in covalent hops.
        heads: Attention heads per layer.
        head_dim: Width of one head.
        ffn: ``(hidden, layers)`` of the feed-forward MLPs.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    atom_embedding_dim: int = Field(default=32, gt=0)
    channel_dim: int = Field(default=8, gt=0)
    node_mlp: tuple[int, int, int] = (64, 256, 2)
    edge_mlp: tuple[int, int, int] = (64, 256, 2)
    width: int = Field(default=256, gt=0)
    n_layers: int = Field(default=4, ge=1)
    heads: int = Field(default=16, gt=0)
    head_dim: int = Field(default=16, gt=0)
    ffn: tuple[int, int] = (256, 2)
