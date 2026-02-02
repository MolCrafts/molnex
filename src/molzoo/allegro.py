"""Allegro: Strictly local equivariant interatomic potential.

Pair-level equivariant neural network using iterated tensor products
without message passing. Features live on edges (pairs) rather than nodes.

Example:
    >>> from molzoo.allegro import Allegro, ScaleShiftAllegro
    >>>
    >>> # Feature extractor only
    >>> encoder = Allegro(
    ...     num_elements=118,
    ...     num_scalar_features=64,
    ...     num_tensor_features=16,
    ...     r_max=5.0,
    ... )
    >>>
    >>> # Complete energy/force model
    >>> model = ScaleShiftAllegro(
    ...     num_elements=118,
    ...     num_scalar_features=64,
    ...     num_tensor_features=16,
    ...     r_max=5.0,
    ...     compute_forces=True,
    ... )

Reference:
    Musaelian et al. "Learning Local Equivariant Representations for
    Large-Scale Atomistic Dynamics" Nature Communications 2023
    https://arxiv.org/abs/2204.05249
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
from molrep.embedding.cutoff import PolynomialCutoff
from molrep.embedding.radial import BesselRBF
from molrep.interaction.tensor_product import irreps_from_l_max, sh_irreps_from_l_max
from molrep.readout import EnergyHead, ForceHead


# ===========================================================================
# PairEmbedding Block
# ===========================================================================


class PairEmbedding(nn.Module):
    """Two-body radial-chemical embedding on edges.

    Computes initial scalar and tensor features for each atom pair (edge)
    from radial distances, spherical harmonics, and atom type embeddings.

    Architecture::

        bond_dist → BesselRBF * PolynomialCutoff → edge_radial
        edge_dir  → SphericalHarmonics → edge_angular
        Z_i, Z_j  → Embed(Z_i) * Embed(Z_j) → type_embed
        (edge_radial ⊕ type_embed) → MLP → scalar_features
        scalar_features → Linear → tensor_env_weights

    Attributes:
        radial_basis: Bessel radial basis functions.
        cutoff_fn: Polynomial cutoff envelope.
        spherical_harmonics: Spherical harmonics for angular features.
        type_embedding: Atom type embedding layer.
        scalar_mlp: MLP producing initial scalar features.
    """

    def __init__(
        self,
        *,
        num_elements: int,
        num_scalar_features: int,
        num_tensor_features: int,
        r_max: float,
        num_bessel: int = 8,
        l_max: int = 2,
        type_emb_dim: int = 16,
    ):
        super().__init__()

        self.num_scalar_features = num_scalar_features
        self.num_tensor_features = num_tensor_features
        self.l_max = l_max

        # Radial basis + cutoff
        self.radial_basis = BesselRBF(r_cut=r_max, num_radial=num_bessel)
        self.cutoff_fn = PolynomialCutoff(r_cut=r_max)

        # Angular basis
        self.spherical_harmonics = SphericalHarmonics(l_max=l_max)

        # Atom type embedding: product of source and target embeddings
        self.type_embedding = nn.Embedding(num_elements, type_emb_dim, dtype=config.ftype)

        # Scalar MLP: (edge_radial + type_embed) → scalar features
        scalar_in_dim = num_bessel + type_emb_dim
        self.scalar_mlp = nn.Sequential(
            nn.Linear(scalar_in_dim, num_scalar_features, dtype=config.ftype),
            nn.SiLU(),
            nn.Linear(num_scalar_features, num_scalar_features, dtype=config.ftype),
            nn.SiLU(),
        )

        # Tensor track: scalar → environment weights for initial tensor features
        # tensor_dim = num_tensor_features per irrep channel
        with cue.assume(O3):
            self.irreps_dim = Irreps(irreps_from_l_max(l_max, num_tensor_features)).dim
        self.tensor_env = nn.Linear(
            num_scalar_features, num_tensor_features, dtype=config.ftype
        )

    def forward(
        self,
        Z: torch.Tensor,
        bond_dist: torch.Tensor,
        bond_diff: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute initial pair embeddings.

        Args:
            Z: Atomic numbers (n_nodes,).
            bond_dist: Bond distances (n_edges,).
            bond_diff: Bond vectors (n_edges, 3).
            edge_index: Edge indices (n_edges, 2).

        Returns:
            Tuple of:
                - scalar_features: (n_edges, num_scalar_features)
                - tensor_features: (n_edges, irreps_dim)
                - edge_angular: (n_edges, sh_dim)
                - edge_cutoff: (n_edges,)
        """
        src, dst = edge_index[:, 0], edge_index[:, 1]

        # Radial features with cutoff envelope
        edge_radial = self.radial_basis(bond_dist)
        edge_cutoff = self.cutoff_fn(bond_dist)
        edge_radial = edge_radial * edge_cutoff.unsqueeze(-1)

        # Angular features
        edge_dir = bond_diff / (bond_dist.unsqueeze(-1) + 1e-8)
        edge_angular = self.spherical_harmonics(edge_dir)

        # Type embedding: element product of source and target
        type_src = self.type_embedding(Z[src])
        type_dst = self.type_embedding(Z[dst])
        type_embed = type_src * type_dst

        # Scalar MLP
        scalar_in = torch.cat([edge_radial, type_embed], dim=-1)
        scalar_features = self.scalar_mlp(scalar_in)

        # Initial tensor features: scalar env weights ⊗ spherical harmonics
        # env_weights: (n_edges, num_tensor_features)
        env_weights = self.tensor_env(scalar_features)
        # Expand env_weights across angular components:
        # (n_edges, num_tensor) × (n_edges, sh_dim) → (n_edges, num_tensor * sh_dim)
        # Using ir_mul layout: for each (l,m), repeat num_tensor times
        sh_dim = edge_angular.shape[-1]
        # ir_mul layout: group by irrep, then channels
        # edge_angular: (n_edges, sh_dim) where sh_dim = sum(2l+1)
        # We want: for each l, multiply env_weights[:, :num_tensor] with Y_l^m
        tensor_features = torch.zeros(
            edge_angular.shape[0], self.irreps_dim,
            dtype=edge_angular.dtype, device=edge_angular.device,
        )
        offset_sh = 0
        offset_tp = 0
        for l in range(self.l_max + 1):
            deg = 2 * l + 1
            # Y_l^m components: (n_edges, deg)
            ylm = edge_angular[:, offset_sh : offset_sh + deg]
            # env_weights: (n_edges, num_tensor)
            # outer product → (n_edges, deg, num_tensor) → reshape to (n_edges, deg * num_tensor)
            # ir_mul layout: (n_edges, deg * num_tensor) where angular index is outer
            block = ylm.unsqueeze(-1) * env_weights.unsqueeze(-2)  # (n_edges, deg, num_tensor)
            block = block.reshape(block.shape[0], -1)  # (n_edges, deg * num_tensor)
            tensor_features[:, offset_tp : offset_tp + deg * self.num_tensor_features] = block
            offset_sh += deg
            offset_tp += deg * self.num_tensor_features

        return scalar_features, tensor_features, edge_angular, edge_cutoff


# ===========================================================================
# AllegroLayer
# ===========================================================================


class AllegroLayer(nn.Module):
    """Pair-level tensor product layer with dual-track (scalar + tensor) features.

    Updates edge features via:
    1. Tensor product of tensor_features with edge_angular → new equivariant features.
    2. Equivariant linear projection on tensor product output.
    3. Extract L=0 scalar invariants from tensor features.
    4. Concatenate invariants with existing scalar features.
    5. Latent MLP on combined scalars → updated scalar features.
    6. Linear projection from scalars → tensor environment weights.

    Attributes:
        tp: cuEquivariance ChannelWiseTensorProduct for pair features.
        tp_linear: Equivariant linear after tensor product.
        latent_mlp: MLP processing scalar track.
        tensor_env: Linear projection for tensor mixing weights.
    """

    def __init__(
        self,
        *,
        num_scalar_features: int,
        num_tensor_features: int,
        l_max: int = 2,
        mlp_depth: int = 1,
    ):
        super().__init__()

        self.num_scalar_features = num_scalar_features
        self.num_tensor_features = num_tensor_features
        self.l_max = l_max

        irreps_str = irreps_from_l_max(l_max, num_tensor_features)
        sh_irreps_str = sh_irreps_from_l_max(l_max)

        cue_irreps_in = cue.Irreps("O3", irreps_str)
        cue_irreps_sh = cue.Irreps("O3", sh_irreps_str)

        # Tensor product: tensor_features ⊗ edge_angular → tp_out
        self.tp = cuet.ChannelWiseTensorProduct(
            cue_irreps_in,
            cue_irreps_sh,
            layout=cue.ir_mul,
            shared_weights=True,
            internal_weights=True,
            dtype=config.ftype,
        )

        # Equivariant linear after TP
        tp_out_irreps = self.tp.irreps_out
        self.tp_linear = cuet.Linear(
            irreps_in=tp_out_irreps,
            irreps_out=cue_irreps_in,
            layout=cue.ir_mul,
            dtype=config.ftype,
        )

        # Scalar invariants: num_tensor_features from L=0 channel
        latent_in_dim = num_scalar_features + num_tensor_features

        # Latent MLP for scalar track
        mlp_layers: list[nn.Module] = []
        in_dim = latent_in_dim
        for _ in range(mlp_depth):
            mlp_layers.append(nn.Linear(in_dim, num_scalar_features, dtype=config.ftype))
            mlp_layers.append(nn.SiLU())
            in_dim = num_scalar_features
        mlp_layers.append(nn.Linear(in_dim, num_scalar_features, dtype=config.ftype))
        self.latent_mlp = nn.Sequential(*mlp_layers)

        # Tensor environment: scalar → mixing weights for tensor features
        self.tensor_env = nn.Linear(
            num_scalar_features, num_tensor_features, dtype=config.ftype
        )

        with cue.assume(O3):
            self.irreps_dim = Irreps(irreps_str).dim

    def forward(
        self,
        scalar_features: torch.Tensor,
        tensor_features: torch.Tensor,
        edge_angular: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update pair features via tensor product and MLP.

        Args:
            scalar_features: Scalar track (n_edges, num_scalar_features).
            tensor_features: Tensor track (n_edges, irreps_dim).
            edge_angular: Spherical harmonics (n_edges, sh_dim).

        Returns:
            Tuple of updated (scalar_features, tensor_features).
        """
        # 1. Tensor product: tensor_features ⊗ edge_angular
        tp_out = self.tp(tensor_features, edge_angular)

        # 2. Equivariant linear projection
        new_tensor = self.tp_linear(tp_out)

        # 3. Extract L=0 invariants (first num_tensor_features components in ir_mul layout)
        invariants = new_tensor[:, : self.num_tensor_features]

        # 4. Concatenate invariants with scalar features
        combined = torch.cat([scalar_features, invariants], dim=-1)

        # 5. Latent MLP → updated scalars
        updated_scalars = self.latent_mlp(combined)

        # 6. Tensor environment weights from scalars
        env_weights = self.tensor_env(updated_scalars)

        # Scale tensor features by environment weights per irrep
        # env_weights: (n_edges, num_tensor)
        # new_tensor: (n_edges, irreps_dim) in ir_mul layout
        # Apply channel-wise scaling: for each (l, m), scale by env_weights
        offset = 0
        updated_tensor = torch.zeros_like(new_tensor)
        for l in range(self.l_max + 1):
            deg = 2 * l + 1
            block_size = deg * self.num_tensor_features
            # block: (n_edges, deg * num_tensor) → reshape to (n_edges, deg, num_tensor)
            block = new_tensor[:, offset : offset + block_size]
            block = block.reshape(-1, deg, self.num_tensor_features)
            scaled = block * env_weights.unsqueeze(-2)
            updated_tensor[:, offset : offset + block_size] = scaled.reshape(-1, block_size)
            offset += block_size

        return updated_scalars, updated_tensor


# ===========================================================================
# Allegro Encoder
# ===========================================================================


class AllegroSpec(BaseModel):
    """Configuration for the Allegro encoder."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    num_elements: int = Field(..., gt=0)
    num_scalar_features: int = Field(64, gt=0)
    num_tensor_features: int = Field(16, gt=0)
    r_max: float = Field(..., gt=0.0)
    num_bessel: int = Field(8, gt=0)
    l_max: int = Field(2, ge=0)
    num_layers: int = Field(2, gt=0)
    mlp_depth: int = Field(1, ge=0)


class Allegro(nn.Module):
    """Allegro equivariant feature encoder.

    Pair-level feature extractor using iterated tensor products without
    message passing. Returns per-layer scalar edge features for downstream
    readout.

    Architecture::

        Input (Z, bond_dist, bond_diff, edge_index)
          → [PairEmbedding] → scalar_features, tensor_features, edge_angular
          → [AllegroLayer₁] → (scalars₁, tensors₁)
          → [AllegroLayer₂] → (scalars₂, tensors₂)
          → ...
          → per_layer_scalars (n_edges, num_layers, num_scalar)

    Attributes:
        embedding: PairEmbedding for initial edge features.
        layers: AllegroLayer modules for pair-level tensor products.
    """

    def __init__(
        self,
        *,
        num_elements: int,
        num_scalar_features: int = 64,
        num_tensor_features: int = 16,
        r_max: float,
        num_bessel: int = 8,
        l_max: int = 2,
        num_layers: int = 2,
        mlp_depth: int = 1,
    ):
        super().__init__()

        self.config = AllegroSpec(
            num_elements=num_elements,
            num_scalar_features=num_scalar_features,
            num_tensor_features=num_tensor_features,
            r_max=r_max,
            num_bessel=num_bessel,
            l_max=l_max,
            num_layers=num_layers,
            mlp_depth=mlp_depth,
        )

        # Pair embedding
        self.embedding = PairEmbedding(
            num_elements=num_elements,
            num_scalar_features=num_scalar_features,
            num_tensor_features=num_tensor_features,
            r_max=r_max,
            num_bessel=num_bessel,
            l_max=l_max,
        )

        # Allegro layers
        self.layers = nn.ModuleList(
            [
                AllegroLayer(
                    num_scalar_features=num_scalar_features,
                    num_tensor_features=num_tensor_features,
                    l_max=l_max,
                    mlp_depth=mlp_depth,
                )
                for _ in range(num_layers)
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
        """Extract per-layer pair scalar features.

        Args:
            Z: Atomic numbers (n_nodes,).
            bond_dist: Bond distances (n_edges,).
            bond_diff: Bond vectors (n_edges, 3).
            edge_index: Edge indices (n_edges, 2).

        Returns:
            Per-layer scalar features (n_edges, num_layers, num_scalar).
        """
        scalar_features, tensor_features, edge_angular, _ = self.embedding(
            Z=Z,
            bond_dist=bond_dist,
            bond_diff=bond_diff,
            edge_index=edge_index,
        )

        per_layer_scalars: list[torch.Tensor] = []

        for layer in self.layers:
            scalar_features, tensor_features = layer(
                scalar_features=scalar_features,
                tensor_features=tensor_features,
                edge_angular=edge_angular,
            )
            per_layer_scalars.append(scalar_features)

        return torch.stack(per_layer_scalars, dim=1)


# ===========================================================================
# ScaleShiftAllegro (Complete Model)
# ===========================================================================


class ScaleShiftAllegroSpec(BaseModel):
    """Configuration for the complete ScaleShiftAllegro model."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    encoder_spec: AllegroSpec
    compute_forces: bool = False
    atomic_inter_scale: float = 1.0
    atomic_inter_shift: float = 0.0


class ScaleShiftAllegro(nn.Module):
    """Complete Allegro model: Encoder + Readout + Prediction Heads.

    Combines three components:
    1. **Allegro encoder**: Pair-level geometric feature extraction.
    2. **Readout layers**: Per-layer MLP projections from edge scalars to pair energies.
    3. **Prediction heads**: Edge→node scatter, energy pooling, force computation.

    Architecture::

        Input (Z, pos, bond_dist, bond_diff, edge_index, batch)
          → [Allegro] → per_layer_scalars (n_edges, num_layers, num_scalar)
          → [Readouts] → per_layer_pair_energies (n_edges, num_layers)
          → sum over layers → pair_energies (n_edges,)
          → scatter_sum to target nodes → atomic_energies (n_nodes,)
          → scale_shift → [EnergyHead] → energy
          → [ForceHead] → forces (optional)

    Attributes:
        encoder: Allegro feature encoder.
        readouts: Per-layer readout modules.
        energy_head: Energy pooling head.
        force_head: Force computation head (optional).
    """

    def __init__(
        self,
        *,
        num_elements: int,
        num_scalar_features: int = 64,
        num_tensor_features: int = 16,
        r_max: float,
        num_bessel: int = 8,
        l_max: int = 2,
        num_layers: int = 2,
        mlp_depth: int = 1,
        compute_forces: bool = False,
        atomic_inter_scale: float = 1.0,
        atomic_inter_shift: float = 0.0,
    ):
        super().__init__()

        self.compute_forces = compute_forces

        # 1. Allegro Encoder
        self.encoder = Allegro(
            num_elements=num_elements,
            num_scalar_features=num_scalar_features,
            num_tensor_features=num_tensor_features,
            r_max=r_max,
            num_bessel=num_bessel,
            l_max=l_max,
            num_layers=num_layers,
            mlp_depth=mlp_depth,
        )

        # 2. Readout layers (Linear for intermediate, MLP for last)
        self.readouts = nn.ModuleList()
        for i in range(num_layers):
            if i < num_layers - 1:
                self.readouts.append(
                    nn.Linear(num_scalar_features, 1, dtype=config.ftype)
                )
            else:
                self.readouts.append(
                    nn.Sequential(
                        nn.Linear(num_scalar_features, 16, dtype=config.ftype),
                        nn.SiLU(),
                        nn.Linear(16, 1, dtype=config.ftype),
                    )
                )

        # Scale and shift
        self.register_buffer(
            "scale", torch.tensor(atomic_inter_scale, dtype=config.ftype)
        )
        self.register_buffer(
            "shift", torch.tensor(atomic_inter_shift, dtype=config.ftype)
        )

        # 3. Prediction heads
        self.energy_head = EnergyHead(pooling="sum")
        if compute_forces:
            self.force_head = ForceHead()

        self.config = ScaleShiftAllegroSpec(
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
        **_kwargs,
    ) -> dict[str, torch.Tensor]:
        """Predict energy and optionally forces.

        Args:
            Z: Atomic numbers (n_nodes,).
            pos: Atomic positions (n_nodes, 3).
            bond_dist: Bond distances (n_edges,).
            bond_diff: Bond vectors (n_edges, 3).
            edge_index: Edge indices (n_edges, 2).
            batch: Batch indices (n_nodes,).

        Returns:
            Dictionary with "energy" and optionally "forces".
        """
        # 1. Encode: per-layer pair scalar features
        per_layer_scalars = self.encoder(
            Z=Z,
            bond_dist=bond_dist,
            bond_diff=bond_diff,
            edge_index=edge_index,
        )
        # Shape: (n_edges, num_layers, num_scalar)

        # 2. Readout: per-layer scalars → per-layer pair energies
        per_layer_energies = []
        for i, readout in enumerate(self.readouts):
            pair_e = readout(per_layer_scalars[:, i, :]).squeeze(-1)  # (n_edges,)
            per_layer_energies.append(pair_e)

        # Sum per-layer pair energies
        pair_energies = torch.stack(per_layer_energies, dim=-1).sum(dim=-1)  # (n_edges,)

        # 3. Scatter pair energies to target nodes
        dst = edge_index[:, 1]
        n_nodes = int(batch.shape[0])
        atomic_energies = torch.zeros(
            n_nodes, dtype=pair_energies.dtype, device=pair_energies.device
        )
        atomic_energies.index_add_(0, dst, pair_energies)

        # Scale and shift
        atomic_energies = atomic_energies * self.scale + self.shift

        # 4. Energy head: pool to molecular energies
        energy = self.energy_head(atomic_energies, batch)

        results: dict[str, torch.Tensor] = {"energy": energy}

        # 5. Force head (optional)
        if self.compute_forces:
            forces = self.force_head(energy, pos)
            results["forces"] = forces

        return results
