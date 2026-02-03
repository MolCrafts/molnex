"""MACE: Multi-Atomic Cluster Expansion neural network.

Equivariant message passing neural network for molecular property prediction.
Naming conventions follow the upstream ACEsuit/mace project where applicable.

Example:
    >>> from molzoo import MACE, ScaleShiftMACE
    >>> from molrep.embedding.node import DiscreteEmbeddingSpec
    >>>
    >>> # Feature extractor only
    >>> encoder = MACE(
    ...     node_attr_specs=[DiscreteEmbeddingSpec(
    ...         input_key="Z", num_classes=119, emb_dim=64)],
    ...     num_elements=118,
    ...     num_features=128,
    ...     r_max=5.0,
    ... )
    >>>
    >>> # Complete energy/force model
    >>> model = ScaleShiftMACE(
    ...     node_attr_specs=[DiscreteEmbeddingSpec(
    ...         input_key="Z", num_classes=119, emb_dim=64)],
    ...     num_elements=118,
    ...     num_features=128,
    ...     r_max=5.0,
    ...     compute_forces=True,
    ... )

Reference:
    Batatia et al. "MACE: Higher Order Equivariant Message Passing Neural
    Networks for Fast and Accurate Force Fields" NeurIPS 2022
    https://arxiv.org/abs/2206.07697
"""

from __future__ import annotations

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field

import cuequivariance as cue
import cuequivariance_torch as cuet
from cuequivariance import Irreps, O3

from molix import config
from molrep.embedding.angular import SphericalHarmonics
from molrep.embedding.cutoff import CosineCutoff
from molrep.embedding.node import (
    ContinuousEmbeddingSpec,
    DiscreteEmbeddingSpec,
    JointEmbedding,
)
from molrep.embedding.radial import BesselRBF
from molrep.interaction.element_update import ElementUpdate
from molrep.interaction.symmetric_contraction import SymmetricContraction
from molrep.interaction.tensor_product import (
    ConvTP,
    irreps_from_l_max,
    sh_irreps_from_l_max,
)
from molrep.readout import EnergyHead, ForceHead
from molrep.readout.basis_projection import BasisProjection

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Type alias for TensorDict keys
Key = str | tuple[str]


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
        self.radial_mlp = nn.Sequential(
            nn.Linear(num_bessel, num_features),
            nn.SiLU(),
            nn.Linear(num_features, num_features),
            nn.SiLU(),
            nn.Linear(num_features, self.conv_tp.weight_numel),
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
# Equivariant Product Basis Block
# ===========================================================================


class ProductBlock(nn.Module):
    """Multi-body feature construction via symmetric tensor products.

    Builds higher-order geometric features through symmetric contraction
    of node representations, enabling the model to capture multi-body interactions.

    Architecture:
        node_feats → SymmetricContraction(atom_types)
        → BasisProjection → Linear → product_features

    Attributes:
        symmetric_contraction: Multi-body basis construction.
        basis_projection: Basis feature projection (currently passthrough).
        linear: Output projection to scalar features.
    """

    def __init__(
        self,
        *,
        irreps_dim: int,
        num_features: int,
        num_bessel: int = 8,
        l_max: int = 2,
        correlation: int = 2,
        num_elements: int = 118,
    ):
        """Initialize product basis block.

        Args:
            irreps_dim: Dimension of the full irreps representation.
            num_features: Scalar channel count (output dimension).
            num_bessel: Number of radial basis functions.
            l_max: Maximum angular momentum order.
            correlation: Correlation order (body order) for symmetric contraction.
            num_elements: Number of atomic element types.
        """
        super().__init__()

        self.symmetric_contraction = SymmetricContraction(
            hidden_dim=irreps_dim,
            num_species=num_elements,
            max_body_order=correlation,
        )

        self.basis_projection = BasisProjection(
            hidden_dim=irreps_dim,
            num_radial=num_bessel,
            l_max=l_max,
            max_body_order=correlation,
        )

        self.linear = nn.Linear(irreps_dim, num_features, dtype=config.ftype)

    def forward(
        self,
        node_feats: torch.Tensor,
        node_attrs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute product basis features.

        Args:
            node_feats: Interaction output ``(n_nodes, irreps_dim)``.
            node_attrs: Atomic numbers ``(n_nodes,)`` for species-specific
                contraction.

        Returns:
            Product features ``(n_nodes, num_features)``.
        """
        basis = self.symmetric_contraction(node_feats, node_attrs)
        features = self.basis_projection(basis)
        return self.linear(features)


# ===========================================================================
# MACE Encoder (Feature Extractor)
# ===========================================================================


class MACE(nn.Module):
    """MACE equivariant feature encoder.

    Pure geometric feature extractor combining three core blocks:
    Embedding, Interaction, and Product. Returns per-layer scalar features
    for downstream readout heads.

    Architecture::

        Input (Z, bond_dist, bond_diff, edge_index)
          → [Embedding] → node_feats, edge_attrs, edge_feats
          → [Interaction₁] → (node_feats, sc) → [Product₁] → [ElementUpdate₁]
          → [Interaction₂] → (node_feats, sc) → [Product₂] → [ElementUpdate₂]
          → ...
          → [Interactionₙ] → (node_feats, sc) → [Productₙ]
          → per_layer_features (n_nodes, num_interactions, num_features)

    Attributes:
        embedding: EmbeddingBlock for initial node/edge features.
        interactions: InteractionBlock modules for message passing.
        products: ProductBlock modules for multi-body features.
        projections: Linear layers projecting product features back to irreps.
        element_updates: Element-specific residual connections (N-1 layers).
        layer_norms: Optional layer normalization between layers.
    """

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

        # Product blocks
        self.products = nn.ModuleList(
            [
                ProductBlock(
                    irreps_dim=irreps_dim,
                    num_features=num_features,
                    num_bessel=num_bessel,
                    l_max=l_max,
                    correlation=correlation,
                    num_elements=num_elements,
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

    def forward(
        self,
        Z: torch.Tensor,
        bond_dist: torch.Tensor,
        bond_diff: torch.Tensor,
        edge_index: torch.Tensor,
        **_kwargs,
    ) -> torch.Tensor:
        """Extract per-layer geometric features.

        Args:
            Z: Atomic numbers (n_nodes,).
            bond_dist: Bond distances (n_edges,).
            bond_diff: Bond vectors (n_edges, 3).
            edge_index: Edge indices (n_edges, 2).

        Returns:
            Per-layer product features (n_nodes, num_interactions, num_features).
        """
        # ---- Embedding ----
        node_feats_init, edge_attrs, edge_feats = self.embedding(
            Z=Z,
            bond_dist=bond_dist,
            bond_diff=bond_diff,
        )

        # ---- Initial projection: scalar embeddings -> hidden irreps ----
        node_feats = self.initial_projection(node_feats_init)

        # Primary species attribute for contraction/updates
        z = Z

        # ---- Interaction-Product-Update loop ----
        per_layer_features: list[torch.Tensor] = []

        for i in range(self.config.num_interactions):
            # Message passing → (node_feats_out, sc)
            node_feats_msg, sc = self.interactions[i](
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=edge_index,
            )

            # Product basis → scalar features
            h_product = self.products[i](
                node_feats=node_feats_msg,
                node_attrs=z,
            )

            # Store per-layer product features
            per_layer_features.append(h_product)

            # Project back to irreps_dim for next layer
            h_proj = self.projections[i](h_product)

            # Element-specific residual update (except last layer)
            is_last = i == (self.config.num_interactions - 1)
            if not is_last:
                node_feats = self.element_updates[i](
                    h_prev=sc,
                    m_curr=h_proj,
                    atom_types=z,
                )
                node_feats = self.layer_norms[i](node_feats)
            else:
                node_feats = h_proj

        # Stack per-layer features: (n_nodes, num_interactions, num_features)
        return torch.stack(per_layer_features, dim=1)


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


# ===========================================================================
# ScaleShiftMACE (Complete Energy/Force Model)
# ===========================================================================


class ScaleShiftMACESpec(BaseModel):
    """Configuration for the complete ScaleShiftMACE model.

    Attributes:
        encoder_spec: Configuration for the MACE feature extractor.
        compute_forces: Whether to compute forces via autograd.
        atomic_inter_scale: Scale factor for interaction energies.
        atomic_inter_shift: Shift for interaction energies.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    encoder_spec: MACESpec
    compute_forces: bool = False
    atomic_inter_scale: float = 1.0
    atomic_inter_shift: float = 0.0


class ScaleShiftMACE(nn.Module):
    """Complete MACE model: Encoder + Readout + Prediction Heads.

    Combines three components:
    1. **MACE encoder**: Geometric feature extraction (Embedding + Interaction + Product).
    2. **Readout layers**: Per-layer linear/MLP projections to atomic energies.
    3. **Prediction heads**: Energy pooling and force computation.

    Architecture::

        Input (Z, pos, bond_dist, bond_diff, edge_index, batch)
          → [MACE] → per_layer_features (n_nodes, num_layers, num_features)
          → [Readouts] → per_layer_energies (n_nodes, num_layers)
          → sum → atomic_energies → scale_shift → [EnergyHead] → energy
          → [ForceHead] → forces (optional)

    Attributes:
        encoder: MACE feature encoder.
        readouts: Per-layer readout modules (Linear for intermediate, MLP for last).
        energy_head: Energy pooling head.
        force_head: Force computation head (optional).
    """

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
        compute_forces: bool = False,
        atomic_inter_scale: float = 1.0,
        atomic_inter_shift: float = 0.0,
    ):
        """Initialize ScaleShiftMACE.

        Args:
            node_attr_specs: Embedding specs for node attributes.
            num_elements: Number of atomic element types.
            num_features: Scalar channel multiplicity at l=0.
            r_max: Radial cutoff in Angstroms.
            num_bessel: Number of Bessel radial basis functions.
            l_max: Maximum angular momentum order.
            num_interactions: Number of interaction-product layers.
            correlation: Body-order correlation for symmetric contraction.
            avg_num_neighbors: Average neighbor count for message normalization.
            layer_norm: Whether to apply layer normalization.
            compute_forces: Whether to compute forces via autograd.
            atomic_inter_scale: Scale factor for interaction energies.
            atomic_inter_shift: Shift for interaction energies.
        """
        super().__init__()

        self.compute_forces = compute_forces

        # 1. MACE Encoder
        self.encoder = MACE(
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

        # 2. Readout layers (Linear for intermediate, MLP for last)
        self.readouts = nn.ModuleList()
        for i in range(num_interactions):
            if i < num_interactions - 1:
                # Linear readout for intermediate layers
                self.readouts.append(
                    nn.Linear(num_features, 1, dtype=config.ftype)
                )
            else:
                # MLP readout for last layer
                self.readouts.append(
                    nn.Sequential(
                        nn.Linear(num_features, 16, dtype=config.ftype),
                        nn.SiLU(),
                        nn.Linear(16, 1, dtype=config.ftype),
                    )
                )

        # Scale and shift
        self.register_buffer(
            "scale",
            torch.tensor(atomic_inter_scale, dtype=config.ftype),
        )
        self.register_buffer(
            "shift",
            torch.tensor(atomic_inter_shift, dtype=config.ftype),
        )

        # 3. Prediction heads
        self.energy_head = EnergyHead(pooling="sum")
        if compute_forces:
            self.force_head = ForceHead()

        # Configuration
        self.config = ScaleShiftMACESpec(
            encoder_spec=self.encoder.config,
            compute_forces=compute_forces,
            atomic_inter_scale=atomic_inter_scale,
            atomic_inter_shift=atomic_inter_shift,
        )

    def forward(
        self,
        Z: torch.Tensor,
        pos: torch.Tensor,
        bond_dist: torch.Tensor,
        bond_diff: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
        **_kwargs,
    ) -> dict[str, torch.Tensor]:
        """Predict energy and optionally forces.

        Args:
            Z: Atomic numbers (n_nodes,).
            pos: Atomic positions (requires_grad=True if compute_forces=True).
            bond_dist: Bond distances (n_edges,).
            bond_diff: Bond vectors (n_edges, 3).
            edge_index: Edge indices (n_edges, 2).
            batch: Batch indices (n_nodes,).
            num_graphs: Number of graphs in the batch.

        Returns:
            Dictionary with "energy" and optionally "forces".
        """
        # 1. Encode: per-layer geometric features
        per_layer_features = self.encoder(
            Z=Z,
            bond_dist=bond_dist,
            bond_diff=bond_diff,
            edge_index=edge_index,
        )
        # Shape: (n_nodes, num_interactions, num_features)

        # 2. Readout: per-layer features → per-layer atomic energies
        per_layer_energies = []
        for i, readout in enumerate(self.readouts):
            layer_energy = readout(per_layer_features[:, i, :])  # (n_nodes, 1)
            per_layer_energies.append(layer_energy.squeeze(-1))  # (n_nodes,)

        # Sum per-layer contributions → total atomic energy
        atomic_energy = torch.stack(per_layer_energies, dim=-1).sum(dim=-1)  # (n_nodes,)

        # Scale and shift
        atomic_energy = atomic_energy * self.scale + self.shift

        # 3. Energy head: pool to molecular energies
        energy = self.energy_head(atomic_energy, batch, num_graphs)

        results = {"energy": energy}

        # 4. Force head: compute forces via autograd (optional)
        if self.compute_forces:
            forces = self.force_head(energy, pos)
            results["forces"] = forces

        return results
