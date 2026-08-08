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
    ...     edge_dist=edge_dist,
    ...     edge_diff=edge_diff,
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
import torch
import torch.nn as nn
from cuequivariance import O3, Irreps
from pydantic import BaseModel, ConfigDict, Field
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase

from molix import config
from molrep.embedding.mace import EmbeddingBlock, EmbeddingSpec
from molrep.embedding.node import (
    ContinuousEmbeddingSpec,
    DiscreteEmbeddingSpec,
)
from molrep.interaction.element import ElementUpdate
from molrep.interaction.mace.block import InteractionBlock, InteractionSpec
from molrep.interaction.product import irreps_from_l_max
from molrep.readout.product import ProductHead

#: Blocks promoted to ``molrep`` by mace-subpackage-restructure-01 and re-exported
#: here so ``from molzoo.mace import EmbeddingBlock`` keeps resolving. Removed in
#: 06-wire, once the consumers import from their new homes.
__all__ = [
    "EmbeddingBlock",
    "EmbeddingSpec",
    "InteractionBlock",
    "InteractionSpec",
    "MACE",
    "MACESpec",
]


# ===========================================================================
# MACE Encoder (Feature Extractor)
# ===========================================================================


class MACE(TensorDictModuleBase):
    """MACE equivariant feature encoder.

    Accepts a ``TensorDict`` TensorDict and writes ``node_features``
    into the ``atoms`` sub-dict in place, returning the same
    ``TensorDict`` with the new key added.

    Architecture::

        TensorDict(atoms, edges)
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
        ("edges", "edge_diff"),
        ("edges", "edge_dist"),
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
        use_fallback: bool = True,
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
            use_fallback: Pure-torch cuEq path (default ``True``, functorch-safe
                for ``ForceDerivation(method="functorch")``). Set ``False`` for
                the fused kernels when forces use the autograd backend — the
                tensor product and symmetric contraction are the encoder's two
                hottest blocks.
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
            use_fallback=use_fallback,
        )

        # Embedding
        self.embedding = EmbeddingBlock(
            node_attr_specs=node_attr_specs,
            num_features=num_features,
            r_max=r_max,
            num_bessel=num_bessel,
            l_max=l_max,
        )
        # Mixed-l message dimension (transient TP output consumed by ProductHead)
        irreps_str = irreps_from_l_max(l_max, num_features)
        with cue.assume(O3):
            irreps_dim = Irreps(irreps_str).dim

        # The node *state* carried between layers is pure scalar (num_features);
        # only the per-edge messages are mixed-l. This keeps every node-state
        # op equivariant. Initial projection is therefore scalar -> scalar.
        self.initial_projection = nn.Linear(num_features, num_features, dtype=config.ftype)

        # Interaction blocks
        self.interactions = nn.ModuleList(
            [
                InteractionBlock(
                    num_features=num_features,
                    num_bessel=num_bessel,
                    l_max=l_max,
                    avg_num_neighbors=avg_num_neighbors,
                    use_fallback=use_fallback,
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
                    use_fallback=use_fallback,
                )
                for _ in range(num_interactions)
            ]
        )

        # Projection of the (scalar) product readout back into the scalar node
        # state for the residual path. Scalar -> scalar keeps it equivariant.
        self.projections = nn.ModuleList(
            [
                nn.Linear(num_features, num_features, dtype=config.ftype)
                for _ in range(num_interactions)
            ]
        )

        # Element-specific residual updates (all layers except last). Operates on
        # the scalar node state, so ElementUpdate's scalar (l=0) treatment is now
        # correct rather than silently mixing l>0 components.
        self.element_updates = nn.ModuleList(
            [
                ElementUpdate(hidden_dim=num_features, num_species=num_elements)
                for _ in range(max(num_interactions - 1, 0))
            ]
        )

        # Layer normalization (all layers except last) over the scalar state.
        self.layer_norms = nn.ModuleList(
            [
                nn.LayerNorm(num_features) if layer_norm else nn.Identity()
                for _ in range(max(num_interactions - 1, 0))
            ]
        )

    def forward(self, td: TensorDict) -> TensorDict:
        """Extract per-layer geometric features.

        Args:
            td: ``TensorDict`` with ``atoms`` and ``edges`` sub-dicts.

        Returns:
            Same ``TensorDict`` with ``atoms.node_features``
            ``(n_nodes, num_interactions, num_features)`` added.
        """
        Z = td["atoms", "Z"]
        edge_dist = td["edges", "edge_dist"]
        edge_diff = td["edges", "edge_diff"]
        edge_index = td["edges", "edge_index"]

        # ---- Embedding ----
        node_feats_init, edge_attrs, edge_feats = self.embedding(
            Z=Z,
            edge_dist=edge_dist,
            edge_diff=edge_diff,
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
    use_fallback: bool = True
