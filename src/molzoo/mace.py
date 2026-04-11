"""MACE: Multi-Atomic Cluster Expansion encoder.

Equivariant message-passing encoder that produces per-layer node features.
Downstream readout, classical potential terms, and force derivation are
handled outside this module.

Example:
    >>> from molzoo import MACE
    >>> from molrep.embedding.node import DiscreteEmbeddingSpec
    >>> encoder = MACE(
    ...     node_attr_specs=[DiscreteEmbeddingSpec(
    ...         input_key="Z", num_classes=119, emb_dim=64)],
    ...     num_elements=118,
    ...     num_features=128,
    ...     r_max=5.0,
    ... )
    >>> features = encoder(
    ...     Z=Z,
    ...     bond_dist=bond_dist,
    ...     bond_diff=bond_diff,
    ...     edge_index=edge_index,
    ... )
    >>> print(features.shape)  # (n_nodes, num_layers, num_features)

Reference:
    Batatia et al. "MACE: Higher Order Equivariant Message Passing Neural
    Networks for Fast and Accurate Force Fields" NeurIPS 2022
    https://arxiv.org/abs/2206.07697
"""

from __future__ import annotations

import cuequivariance as cue
import cuequivariance_torch as cuet
import torch
import torch.nn as nn
from cuequivariance import O3, Irreps
from pydantic import BaseModel, ConfigDict, Field
from tensordict.nn import TensorDictModuleBase

from molix import config
from molix.data.types import GraphBatch
from molrep.embedding.angular import SphericalHarmonics
from molrep.embedding.cutoff import CosineCutoff
from molrep.embedding.node import (
    ContinuousEmbeddingSpec,
    DiscreteEmbeddingSpec,
    JointEmbedding,
)
from molrep.embedding.radial import BesselRBF
from molrep.interaction.element_update import ElementUpdate
from molrep.interaction.radial_mlp import RadialWeightMLP
from molrep.interaction.tensor_product import (
    ConvTP,
    irreps_from_l_max,
    sh_irreps_from_l_max,
)
from molrep.readout.product_head import ProductHead

# ===========================================================================
# Embedding Block
# ===========================================================================


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
        bond_dist: torch.Tensor,
        bond_diff: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute initial node and edge features.

        Args:
            Z: Atomic numbers (n_nodes,).
            bond_dist: Bond distances (n_edges,).
            bond_diff: Bond vectors (target - source) (n_edges, 3).

        Returns:
            tuple of:
                - node_feats: Node features (n_nodes, num_features).
                - edge_attrs: Spherical harmonics (n_edges, sh_dim).
                - edge_feats: Radial basis features (n_edges, num_bessel).
        """
        # Node features
        node_feats = self.node_embedding(Z=Z)

        # Edge direction
        edge_dir = bond_diff / (bond_dist.unsqueeze(-1) + 1e-8)

        # Spherical harmonics
        edge_attrs = self.spherical_harmonics(edge_dir)

        # Radial basis * cutoff → edge_feats
        edge_radial = self.radial_embedding(bond_dist)
        edge_cutoff = self.cutoff_fn(bond_dist)
        edge_feats = edge_radial * edge_cutoff.unsqueeze(-1)

        return node_feats, edge_attrs, edge_feats


# ===========================================================================
# Interaction Block
# ===========================================================================


class InteractionSpec(BaseModel):
    """Configuration for a single interaction block.

    Attributes:
        num_features: Scalar channel multiplicity.
        num_bessel: Number of Bessel radial basis functions.
        l_max: Maximum angular momentum order.
        avg_num_neighbors: Average number of neighbors for normalization.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    num_features: int = Field(..., gt=0)
    num_bessel: int = Field(8, gt=0)
    l_max: int = Field(2, ge=0)
    avg_num_neighbors: float = Field(1.0, gt=0.0)


class InteractionBlock(nn.Module):
    """Equivariant message passing with tensor product convolution.

    Performs geometric message passing via cuEquivariance-accelerated tensor products,
    returning updated node features and skip connection for residual updates.

    Architecture:
        node_feats → node_linear → tensor_product(edge_attrs, tp_weights)
        → aggregate → linear → (node_feats_out, skip_connection)

    Attributes:
        conv_tp: Tensor product convolution (cuEquivariance ChannelWiseTensorProduct).
        node_linear: Pre-convolution equivariant linear transformation.
        radial_mlp: MLP generating tensor product weights from edge features.
        linear: Post-convolution equivariant linear projection.
        avg_num_neighbors: Message normalization constant.

    Reference:
        https://docs.nvidia.com/cuda/cuequivariance/tutorials/pytorch/MACE.html
    """

    def __init__(
        self,
        *,
        num_features: int,
        num_bessel: int = 8,
        l_max: int = 2,
        avg_num_neighbors: float = 1.0,
    ):
        """Initialize interaction block.

        Args:
            num_features: Scalar channel multiplicity.
            num_bessel: Number of Bessel basis functions.
            l_max: Maximum angular momentum order.
            avg_num_neighbors: Average neighbor count for message normalization.
        """
        super().__init__()

        self.config = InteractionSpec(
            num_features=num_features,
            num_bessel=num_bessel,
            l_max=l_max,
            avg_num_neighbors=avg_num_neighbors,
        )

        irreps_str = irreps_from_l_max(l_max, num_features)
        sh_irreps_str = sh_irreps_from_l_max(l_max)

        # 1. Tensor product convolution (define first to get weight_numel)
        self.conv_tp = ConvTP(
            in_irreps=irreps_str,
            out_irreps=irreps_str,
            sh_irreps=sh_irreps_str,
        )

        # Actual TP output irreps (may differ from requested out_irreps)
        tp_out_irreps = str(self.conv_tp.cue_tp.irreps_out)

        # 2. Pre-convolution equivariant linear
        self.node_linear = cuet.Linear(
            irreps_in=cue.Irreps("O3", irreps_str),
            irreps_out=cue.Irreps("O3", irreps_str),
            layout=cue.ir_mul,
            dtype=config.ftype,
        )

        # 3. Radial MLP for TP weights
        self.radial_mlp = RadialWeightMLP(
            in_dim=num_bessel,
            hidden_dim=num_features,
            out_dim=self.conv_tp.weight_numel,
            num_layers=2,
        )

        # 4. Post-convolution equivariant linear
        self.linear = cuet.Linear(
            irreps_in=cue.Irreps("O3", tp_out_irreps),
            irreps_out=cue.Irreps("O3", irreps_str),
            layout=cue.ir_mul,
            dtype=config.ftype,
        )

        self.avg_num_neighbors = avg_num_neighbors

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one interaction layer.

        Args:
            node_feats: Node features ``(n_nodes, irreps_dim)``.
            edge_attrs: Spherical harmonics ``(n_edges, sh_dim)``.
            edge_feats: Radial basis features ``(n_edges, num_bessel)``.
            edge_index: Edge indices ``(n_edges, 2)``.

        Returns:
            tuple of:
                - ``node_feats``: Updated node features ``(n_nodes, irreps_dim)``.
                - ``sc``: Skip connection (original input) ``(n_nodes, irreps_dim)``.
        """
        sc = node_feats  # skip connection for EquivariantProductBasisBlock

        # Pre-convolution linear
        node_feats_up = self.node_linear(node_feats)

        # TP weights from radial basis
        tp_weights = self.radial_mlp(edge_feats)

        # Tensor product convolution with neighbor aggregation
        messages = self.conv_tp(
            node_features=node_feats_up,
            edge_angular=edge_attrs,
            edge_index=edge_index,
            tp_weights=tp_weights,
        )

        # Normalize by average number of neighbors
        messages = messages / self.avg_num_neighbors

        # Post-convolution linear
        node_feats = self.linear(messages)

        return node_feats, sc


# ===========================================================================
# MACE Encoder (Feature Extractor)
# ===========================================================================


class MACE(TensorDictModuleBase):
    """MACE equivariant feature encoder.

    Accepts a ``GraphBatch`` TensorDict and writes ``node_features``
    into the ``atoms`` sub-dict, returning a ``GraphBatch`` with
    ``NodeRepAtoms`` atoms.

    Architecture::

        GraphBatch(atoms, edges)
          → [Embedding] → node_feats, edge_attrs, edge_feats
          → [Interaction₁] → [ProductHead₁] → [ElementUpdate₁]
          → ...
          → [Interactionₙ] → [ProductHeadₙ]
          → atoms.node_features (n_nodes, num_interactions, num_features)

    Reference:
        Batatia et al. "MACE: Higher Order Equivariant Message Passing Neural
        Networks for Fast and Accurate Force Fields" NeurIPS 2022
        https://arxiv.org/abs/2206.07697
    """

    in_keys = [
        ("atoms", "Z"),
        ("atoms", "pos"),
        ("edges", "edge_index"),
        ("edges", "bond_diff"),
        ("edges", "bond_dist"),
    ]
    out_keys = [("atoms", "node_features")]

    def __init__(
        self,
        *,
        node_attr_specs: list[DiscreteEmbeddingSpec | ContinuousEmbeddingSpec],
        num_elements: int,
        num_features: int,
        r_max: float,
        num_bessel: int = 8,
        l_max: int = 2,
        num_interactions: int = 2,
        correlation: int = 2,
        avg_num_neighbors: float = 1.0,
        layer_norm: bool = False,
    ):
        """Initialize MACE feature extractor.

        Args:
            node_attr_specs: Embedding specs for node attributes (e.g. Z).
            num_elements: Number of atomic element types.
            num_features: Scalar channel multiplicity at l=0.
            r_max: Radial cutoff in Angstroms.
            num_bessel: Number of Bessel radial basis functions.
            l_max: Maximum angular momentum order.
            num_interactions: Number of interaction-product-update layers.
            correlation: Body-order correlation for symmetric contraction.
            avg_num_neighbors: Average neighbor count for message normalization.
            layer_norm: Whether to apply layer normalization between layers.
        """
        super().__init__()

        self.config = MACESpec(
            node_attr_specs=node_attr_specs,
            num_elements=num_elements,
            num_features=num_features,
            r_max=r_max,
            num_bessel=num_bessel,
            l_max=l_max,
            num_interactions=num_interactions,
            correlation=correlation,
            avg_num_neighbors=avg_num_neighbors,
            layer_norm=layer_norm,
        )

        # Embedding
        self.embedding = EmbeddingBlock(
            node_attr_specs=node_attr_specs,
            num_features=num_features,
            r_max=r_max,
            num_bessel=num_bessel,
            l_max=l_max,
        )
        # Hidden irreps dimension for message passing paths
        irreps_str = irreps_from_l_max(l_max, num_features)
        with cue.assume(O3):
            irreps_dim = Irreps(irreps_str).dim

        # Initial projection: scalar embeddings -> hidden irreps
        self.initial_projection = nn.Linear(num_features, irreps_dim, dtype=config.ftype)

        # Interaction blocks
        self.interactions = nn.ModuleList(
            [
                InteractionBlock(
                    num_features=num_features,
                    num_bessel=num_bessel,
                    l_max=l_max,
                    avg_num_neighbors=avg_num_neighbors,
                )
                for _ in range(num_interactions)
            ]
        )

        # Product heads (from molrep, replaces former ProductBlock)
        self.products = nn.ModuleList(
            [
                ProductHead(
                    hidden_dim=irreps_dim,
                    out_dim=num_features,
                    num_radial=num_bessel,
                    l_max=l_max,
                    max_body_order=correlation,
                    num_species=num_elements,
                )
                for _ in range(num_interactions)
            ]
        )

        # Projection: num_features → irreps_dim (for residual path)
        self.projections = nn.ModuleList(
            [
                nn.Linear(num_features, irreps_dim, dtype=config.ftype)
                for _ in range(num_interactions)
            ]
        )

        # Element-specific residual updates (all layers except last)
        self.element_updates = nn.ModuleList(
            [
                ElementUpdate(hidden_dim=irreps_dim, num_species=num_elements)
                for _ in range(max(num_interactions - 1, 0))
            ]
        )

        # Layer normalization (all layers except last)
        self.layer_norms = nn.ModuleList(
            [
                nn.LayerNorm(irreps_dim) if layer_norm else nn.Identity()
                for _ in range(max(num_interactions - 1, 0))
            ]
        )

    def forward(self, td: GraphBatch) -> GraphBatch:
        """Extract per-layer geometric features.

        Args:
            td: ``GraphBatch`` with ``atoms`` and ``edges`` sub-dicts.

        Returns:
            Same ``GraphBatch`` with ``atoms.node_features``
            ``(n_nodes, num_interactions, num_features)`` added.
        """
        Z = td["atoms", "Z"]
        bond_dist = td["edges", "bond_dist"]
        bond_diff = td["edges", "bond_diff"]
        edge_index = td["edges", "edge_index"]

        # ---- Embedding ----
        node_feats_init, edge_attrs, edge_feats = self.embedding(
            Z=Z,
            bond_dist=bond_dist,
            bond_diff=bond_diff,
        )

        # ---- Initial projection: scalar embeddings -> hidden irreps ----
        node_feats = self.initial_projection(node_feats_init)

        # ---- Interaction-Product-Update loop ----
        per_layer_features: list[torch.Tensor] = []

        for i in range(self.config.num_interactions):
            node_feats_msg, sc = self.interactions[i](
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=edge_index,
            )

            h_product = self.products[i](
                node_features=node_feats_msg,
                atom_types=Z,
            )

            per_layer_features.append(h_product)

            h_proj = self.projections[i](h_product)

            is_last = i == (self.config.num_interactions - 1)
            if not is_last:
                node_feats = self.element_updates[i](
                    h_prev=sc,
                    m_curr=h_proj,
                    atom_types=Z,
                )
                node_feats = self.layer_norms[i](node_feats)
            else:
                node_feats = h_proj

        td["atoms", "node_features"] = torch.stack(per_layer_features, dim=1)
        return td


class MACESpec(BaseModel):
    """Configuration for the MACE feature extractor.

    Attributes:
        node_attr_specs: Embedding specs for node attributes.
        num_elements: Number of atomic element types.
        num_features: Scalar channel multiplicity.
        r_max: Radial cutoff in Angstroms.
        num_bessel: Number of Bessel basis functions.
        l_max: Maximum angular momentum order.
        num_interactions: Number of interaction-product layers.
        correlation: Body-order correlation for symmetric contraction.
        avg_num_neighbors: Average neighbor count for normalization.
        layer_norm: Whether to apply layer normalization.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    node_attr_specs: list[DiscreteEmbeddingSpec | ContinuousEmbeddingSpec] = Field(
        ..., min_length=1
    )
    num_elements: int = Field(..., gt=0)
    num_features: int = Field(..., gt=0)
    r_max: float = Field(..., gt=0.0)
    num_bessel: int = Field(8, gt=0)
    l_max: int = Field(2, ge=0)
    num_interactions: int = Field(2, gt=0)
    correlation: int = Field(2, ge=1, le=3)
    avg_num_neighbors: float = Field(1.0, gt=0.0)
    layer_norm: bool = False
