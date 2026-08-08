"""MACE-only embedding block.

:class:`EmbeddingBlock` composes the embedding-layer primitives the MACE
encoders need — ``JointEmbedding`` for node attributes, ``BesselRBF`` for the
radial basis, ``SphericalHarmonics`` for edge directions and ``CosineCutoff``
for the envelope — into the single ``(node_feats, edge_attrs, edge_feats)``
producer that feeds the interaction stack.

Relocated verbatim from ``molzoo/mace.py``, which continues to re-export both
names for backwards compatibility. ``molrep.embedding.__init__`` deliberately
does *not* re-export them: ``EmbeddingBlock`` / ``EmbeddingSpec`` are too
generic a name at package level, so consumers import from this module path.

Reference:
    Batatia et al. "MACE: Higher Order Equivariant Message Passing Neural
    Networks for Fast and Accurate Force Fields" NeurIPS 2022.
    https://arxiv.org/abs/2206.07697
"""

from __future__ import annotations

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field

from molrep.embedding.angular import SphericalHarmonics
from molrep.embedding.cutoff import CosineCutoff
from molrep.embedding.node import (
    ContinuousEmbeddingSpec,
    DiscreteEmbeddingSpec,
    JointEmbedding,
)
from molrep.embedding.radial import BesselRBF


class EmbeddingSpec(BaseModel):
    """Configuration for the embedding block.

    Attributes:
        node_attr_specs: Embedding specifications for node attributes
            (e.g. atomic number Z, charge).
        num_features: Number of feature channels (scalar multiplicity at l=0).
        r_max: Radial cutoff distance in Angstroms.
        num_bessel: Number of Bessel radial basis functions.
        l_max: Maximum angular momentum order.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    node_attr_specs: list[DiscreteEmbeddingSpec | ContinuousEmbeddingSpec] = Field(
        ..., min_length=1
    )
    num_features: int = Field(..., gt=0)
    r_max: float = Field(..., gt=0.0)
    num_bessel: int = Field(8, gt=0)
    l_max: int = Field(2, ge=0)


class EmbeddingBlock(nn.Module):
    """Node and edge embedding block.

    Computes initial node features via ``JointEmbedding`` and edge features
    via Bessel radial basis, spherical harmonics, and a cosine cutoff envelope.

    Attributes:
        node_embedding: Joint embedding for node attributes.
        radial_embedding: Bessel radial basis functions.
        spherical_harmonics: Spherical harmonics for edge directions.
        cutoff_fn: Cosine cutoff envelope.
    """

    def __init__(
        self,
        *,
        node_attr_specs: list[DiscreteEmbeddingSpec | ContinuousEmbeddingSpec],
        num_features: int,
        r_max: float,
        num_bessel: int = 8,
        l_max: int = 2,
    ):
        """Initialize embedding block.

        Args:
            node_attr_specs: Embedding specs for node attributes.
            num_features: Scalar channel multiplicity (l=0 count).
            r_max: Radial cutoff in Angstroms.
            num_bessel: Number of Bessel basis functions.
            l_max: Maximum angular momentum order.
        """
        super().__init__()

        self.config = EmbeddingSpec(
            node_attr_specs=node_attr_specs,
            num_features=num_features,
            r_max=r_max,
            num_bessel=num_bessel,
            l_max=l_max,
        )

        # Node embedding
        self.node_embedding = JointEmbedding(
            embedding_specs=node_attr_specs,
            out_dim=num_features,
        )

        # Edge radial basis
        self.radial_embedding = BesselRBF(
            r_cut=r_max,
            num_radial=num_bessel,
        )

        # Spherical harmonics
        self.spherical_harmonics = SphericalHarmonics(
            l_max=l_max,
        )

        # Cutoff envelope
        self.cutoff_fn = CosineCutoff(
            r_cut=r_max,
        )

    def forward(
        self,
        Z: torch.Tensor,
        edge_dist: torch.Tensor,
        edge_diff: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute initial node and edge features.

        Args:
            Z: Atomic numbers (n_nodes,).
            edge_dist: Bond distances (n_edges,).
            edge_diff: Bond vectors (target - source) (n_edges, 3).

        Returns:
            tuple of:
                - node_feats: Node features (n_nodes, num_features).
                - edge_attrs: Spherical harmonics (n_edges, sh_dim).
                - edge_feats: Radial basis features (n_edges, num_bessel).
        """
        # Node features
        node_feats = self.node_embedding(Z=Z)

        # Edge direction
        edge_dir = edge_diff / (edge_dist.unsqueeze(-1) + 1e-8)

        # Spherical harmonics
        edge_attrs = self.spherical_harmonics(edge_dir)

        # Radial basis * cutoff → edge_feats
        edge_radial = self.radial_embedding(edge_dist)
        edge_cutoff = self.cutoff_fn(edge_dist)
        edge_feats = edge_radial * edge_cutoff.unsqueeze(-1)

        return node_feats, edge_attrs, edge_feats
