"""Allegro: local equivariant edge encoder.

Pair-level equivariant encoder using iterated tensor products with neighborhood
aggregation. It returns edge features only; downstream parameterization, potential
composition, and force derivation live outside this module.

Example:
    >>> from molzoo.allegro import Allegro
    >>> encoder = Allegro(
    ...     num_elements=118,
    ...     num_scalar_features=64,
    ...     num_tensor_features=16,
    ...     r_max=5.0,
    ... )
    >>> features = encoder(
    ...     Z=Z,
    ...     bond_dist=bond_dist,
    ...     bond_diff=bond_diff,
    ...     edge_index=edge_index,
    ... )
    >>> print(features.shape)  # (n_edges, num_layers, num_scalar)

Reference:
    Musaelian et al. "Learning Local Equivariant Representations for
    Large-Scale Atomistic Dynamics" Nature Communications 2023
    https://arxiv.org/abs/2204.05249
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
from molrep.embedding.cutoff import PolynomialCutoff
from molrep.embedding.radial import BesselRBF
from molrep.interaction.tensor_product import irreps_from_l_max, sh_irreps_from_l_max


# ===========================================================================
# Shared helper
# ===========================================================================


def _env_weight_harmonics(
    edge_angular: torch.Tensor,
    env_weights: torch.Tensor,
    l_max: int,
    num_tensor_features: int,
    irreps_dim: int,
) -> torch.Tensor:
    """Weight spherical harmonics by per-channel environment weights.

    For each angular momentum l, computes the outer product of the (2l+1)
    spherical harmonic components with the ``num_tensor_features`` channel
    weights, then lays them out in ``ir_mul`` order.

    Args:
        edge_angular: Spherical harmonics ``(n_edges, sh_dim)``.
        env_weights: Per-channel weights ``(n_edges, num_tensor_features)``.
        l_max: Maximum angular momentum.
        num_tensor_features: Number of tensor feature channels.
        irreps_dim: Total dimension of the ir_mul tensor representation.

    Returns:
        Weighted tensor features ``(n_edges, irreps_dim)`` in ir_mul layout.
    """
    result = torch.zeros(
        edge_angular.shape[0],
        irreps_dim,
        dtype=edge_angular.dtype,
        device=edge_angular.device,
    )
    offset_sh = 0
    offset_tp = 0
    for l in range(l_max + 1):
        deg = 2 * l + 1
        ylm = edge_angular[:, offset_sh : offset_sh + deg]
        block = (ylm.unsqueeze(-1) * env_weights.unsqueeze(-2)).reshape(
            ylm.shape[0], -1
        )
        result[:, offset_tp : offset_tp + deg * num_tensor_features] = block
        offset_sh += deg
        offset_tp += deg * num_tensor_features
    return result


def _scale_by_channel(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    l_max: int,
    num_tensor_features: int,
) -> torch.Tensor:
    """Scale an ir_mul tensor feature by per-channel scalar weights.

    For each angular momentum l and each magnetic quantum number m, multiplies
    each of the ``num_tensor_features`` channels by the corresponding element
    of ``scale``.  This is an O(3)-equivariant operation (scalar × equivariant
    = equivariant).

    Args:
        tensor: Equivariant tensor in ir_mul layout
            ``(n_edges, irreps_dim)``.
        scale: Per-channel scale factors ``(n_edges, num_tensor_features)``.
        l_max: Maximum angular momentum.
        num_tensor_features: Number of tensor feature channels.

    Returns:
        Scaled tensor ``(n_edges, irreps_dim)`` in ir_mul layout.
    """
    result = torch.empty_like(tensor)
    offset = 0
    for l in range(l_max + 1):
        deg = 2 * l + 1
        bsz = deg * num_tensor_features
        block = tensor[:, offset : offset + bsz].reshape(-1, deg, num_tensor_features)
        result[:, offset : offset + bsz] = (block * scale.unsqueeze(-2)).reshape(
            -1, bsz
        )
        offset += bsz
    return result


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
        Z_i, Z_j  → [Embed(Z_i) ⊕ Embed(Z_j)] → type_embed  (concatenation)
        (edge_radial ⊕ type_embed) → MLP → scalar_features
        scalar_features → Linear → tensor_env_weights
        tensor_env_weights ⊗ edge_angular → tensor_features (ir_mul layout)

    Attributes:
        radial_basis: Bessel radial basis functions.
        cutoff_fn: Polynomial cutoff envelope.
        spherical_harmonics: Spherical harmonics for angular features.
        type_embedding: Atom type embedding layer.
        scalar_mlp: MLP producing initial scalar features.
        tensor_env: Linear projection for initial tensor env weights.
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
        scalar_mlp_hiddens: list[int] | None = None,
        poly_p: int = 6,
    ):
        super().__init__()

        self.num_scalar_features = num_scalar_features
        self.num_tensor_features = num_tensor_features
        self.l_max = l_max

        # Radial basis + cutoff
        self.radial_basis = BesselRBF(r_cut=r_max, num_radial=num_bessel)
        self.cutoff_fn = PolynomialCutoff(r_cut=r_max, exponent=poly_p)

        # Angular basis
        self.spherical_harmonics = SphericalHarmonics(l_max=l_max)

        # Atom type embedding: separate embeddings for source and destination,
        # concatenated to preserve centre-neighbour directionality.
        self.type_embedding = nn.Embedding(num_elements, type_emb_dim, dtype=config.ftype)

        # Two-body scalar MLP: (edge_radial ⊕ type_src ⊕ type_dst) →
        # hidden_1 → hidden_2 → … → num_scalar_features.
        scalar_in_dim = num_bessel + 2 * type_emb_dim
        if scalar_mlp_hiddens is None:
            scalar_mlp_hiddens = [num_scalar_features, num_scalar_features]
        hiddens = list(scalar_mlp_hiddens) + [num_scalar_features]
        layers: list[nn.Module] = []
        prev = scalar_in_dim
        for h in hiddens:
            layers.append(nn.Linear(prev, h, dtype=config.ftype))
            layers.append(nn.SiLU())
            prev = h
        self.scalar_mlp = nn.Sequential(*layers)

        # Tensor track: scalar → environment weights for initial tensor features
        with cue.assume(O3):
            self.irreps_dim = Irreps(irreps_from_l_max(l_max, num_tensor_features)).dim
        self.tensor_env = nn.Linear(num_scalar_features, num_tensor_features, dtype=config.ftype)

    def forward(
        self,
        Z: torch.Tensor,
        bond_dist: torch.Tensor,
        bond_diff: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute initial pair embeddings.

        Args:
            Z: Atomic numbers ``(n_nodes,)``.
            bond_dist: Bond distances ``(n_edges,)``.
            bond_diff: Bond vectors ``(n_edges, 3)``.
            edge_index: Edge indices ``(n_edges, 2)``.

        Returns:
            Tuple of:
                - scalar_features: ``(n_edges, num_scalar_features)``
                - tensor_features: ``(n_edges, irreps_dim)``
                - edge_angular: ``(n_edges, sh_dim)``
                - edge_cutoff: ``(n_edges,)``
        """
        src, dst = edge_index[:, 0], edge_index[:, 1]

        # Radial features with cutoff envelope
        edge_radial = self.radial_basis(bond_dist)
        edge_cutoff = self.cutoff_fn(bond_dist)
        edge_radial = edge_radial * edge_cutoff.unsqueeze(-1)

        # Angular features
        edge_dir = bond_diff / (bond_dist.unsqueeze(-1) + 1e-8)
        edge_angular = self.spherical_harmonics(edge_dir)

        # Type embedding: concatenate source and destination embeddings
        # to preserve centre-neighbour directionality (vs element-wise product).
        type_src = self.type_embedding(Z[src])
        type_dst = self.type_embedding(Z[dst])
        type_embed = torch.cat([type_src, type_dst], dim=-1)

        # Scalar MLP
        scalar_in = torch.cat([edge_radial, type_embed], dim=-1)
        scalar_features = self.scalar_mlp(scalar_in)

        # Initial tensor features via shared helper
        env_weights = self.tensor_env(scalar_features)
        tensor_features = _env_weight_harmonics(
            edge_angular, env_weights, self.l_max, self.num_tensor_features, self.irreps_dim
        )

        return scalar_features, tensor_features, edge_angular, edge_cutoff


# ===========================================================================
# AllegroLayer
# ===========================================================================


class AllegroLayer(nn.Module):
    """Pair-level tensor product layer with neighbourhood aggregation.

    Each layer:

    1. Computes per-edge env weights from the most recent scalar features.
    2. Pre-scales ``tensor_features`` channel-wise by those env weights.
    3. Aggregates *unweighted* spherical harmonics to source nodes, giving a
       single-channel neighbourhood embedding ``node_Y`` (many-body context).
    4. Tensor-products the scaled edge features with the neighbourhood Y,
       using ``ChannelWiseTensorProduct(irreps_in, sh_irreps)`` — the same
       fast kernel path as in the original Allegro reference.
    5. Projects the TP output back to ``irreps_in`` via an equivariant linear.
    6. Extracts L=0 scalar invariants from the new tensor features.
    7. Runs a latent MLP on all accumulated scalars (DenseNet) plus the fresh
       invariants, producing updated scalar features.
    8. Scales the new tensor features by per-channel weights derived from the
       updated scalars.

    Why this TP decomposition is fast and correct
    ---------------------------------------------
    The original Allegro computes ``TP(V_{ij}, sum_k env_w_k * Y_k)`` where
    both sides carry ``num_tensor_features`` channels, causing cuEquivariance
    to fall back to a slow naive path.

    Instead we factor the product as::

        TP(env_w_{ij} · V_{ij},  Y_agg_i)
        where  Y_agg_i = (1/|N(i)|) * sum_{k in N(i)} Y_k   (single-channel)

    By the bilinearity of the CG tensor product:

        TP(c · A, B) = c · TP(A, B)   for scalar c

    so scaling the *left* input by ``env_w`` is equivalent to scaling the TP
    output by ``env_w``.  The right-hand side ``Y_agg`` stays single-channel
    (``sh_irreps``), preserving the fast ``ChannelWiseTensorProduct(irreps_in,
    sh_irreps)`` execution path while still encoding the full neighbourhood
    geometry in each layer.

    Attributes:
        tp: cuEquivariance ChannelWiseTensorProduct (irreps_in × sh_irreps).
        tp_linear: Equivariant linear projecting TP output back to irreps_in.
        env_embed: Linear mapping the most-recent scalars → pre-TP env weights.
        latent_mlp: MLP processing the DenseNet-accumulated scalar track.
        tensor_env: Linear mapping updated scalars → post-TP tensor env weights.
    """

    def __init__(
        self,
        *,
        num_scalar_features: int,
        num_tensor_features: int,
        l_max: int = 2,
        mlp_depth: int = 1,
        latent_in_dim: int | None = None,
        latent_mlp_hiddens: list[int] | None = None,
    ):
        super().__init__()

        self.num_scalar_features = num_scalar_features
        self.num_tensor_features = num_tensor_features
        self.l_max = l_max

        irreps_str = irreps_from_l_max(l_max, num_tensor_features)
        sh_irreps_str = sh_irreps_from_l_max(l_max)

        cue_irreps_in = cue.Irreps("O3", irreps_str)
        cue_irreps_sh = cue.Irreps("O3", sh_irreps_str)

        # Tensor product: pre-scaled edge features ⊗ aggregated neighbourhood Y.
        # Right-hand side uses sh_irreps (single-channel) → fast kernel path,
        # same as the original Allegro implementation.
        self.tp = cuet.ChannelWiseTensorProduct(
            cue_irreps_in,
            cue_irreps_sh,
            layout=cue.ir_mul,
            shared_weights=True,
            internal_weights=True,
            dtype=config.ftype,
        )

        # Equivariant linear: project TP output back to input irreps
        tp_out_irreps = self.tp.irreps_out
        self.tp_linear = cuet.Linear(
            irreps_in=tp_out_irreps,
            irreps_out=cue_irreps_in,
            layout=cue.ir_mul,
            dtype=config.ftype,
        )

        # Pre-TP env embed: maps the most recent scalars → per-channel scale
        # factors applied to tensor_features *before* the tensor product.
        # Encodes the chemical context of the current edge into the geometric
        # query without requiring multi-channel Y aggregation.
        self.env_embed = nn.Linear(num_scalar_features, num_tensor_features, dtype=config.ftype)

        # DenseNet latent MLP:
        # Input = all accumulated scalars (embedding + all previous layers)
        #         + L=0 invariants from the new tensor features.
        # latent_in_dim for layer i (0-indexed) =
        #     (i + 1) * num_scalar_features + num_tensor_features
        actual_latent_in_dim = (
            latent_in_dim
            if latent_in_dim is not None
            else num_scalar_features + num_tensor_features
        )
        # Per-layer hidden schedule: if ``latent_mlp_hiddens`` is given, build
        # an MLP whose hidden widths follow the list and whose final output is
        # ``num_scalar_features``. Each hidden layer is followed by SiLU; the
        # final projection is linear (no activation), matching Allegro's spec.
        # ``mlp_depth=0`` with no hiddens → a single linear projection.
        if latent_mlp_hiddens is None:
            hiddens = [num_scalar_features] * mlp_depth
        else:
            hiddens = list(latent_mlp_hiddens)
        mlp_layers: list[nn.Module] = []
        in_dim = actual_latent_in_dim
        for h in hiddens:
            mlp_layers.append(nn.Linear(in_dim, h, dtype=config.ftype))
            mlp_layers.append(nn.SiLU())
            in_dim = h
        mlp_layers.append(nn.Linear(in_dim, num_scalar_features, dtype=config.ftype))
        self.latent_mlp = nn.Sequential(*mlp_layers)

        # Post-TP tensor env: maps updated scalars → per-channel output scaling
        self.tensor_env = nn.Linear(num_scalar_features, num_tensor_features, dtype=config.ftype)

        with cue.assume(O3):
            self.irreps_dim = Irreps(irreps_str).dim

        # sh_dim needed for node-Y aggregation buffer
        with cue.assume(O3):
            self._sh_dim = Irreps(sh_irreps_str).dim

    def forward(
        self,
        accumulated_scalars: torch.Tensor,
        tensor_features: torch.Tensor,
        edge_angular: torch.Tensor,
        edge_index: torch.Tensor,
        n_nodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update pair features via neighbourhood-aggregated tensor product.

        Args:
            accumulated_scalars: DenseNet-concatenated scalar track from all
                previous layers including the initial embedding
                ``(n_edges, k * num_scalar_features)`` where k = number of
                layers seen so far (including embedding).
            tensor_features: Equivariant tensor track ``(n_edges, irreps_dim)``.
            edge_angular: Spherical harmonics reused from embedding
                ``(n_edges, sh_dim)``.
            edge_index: Edge connectivity ``(n_edges, 2)``.
            n_nodes: Number of nodes in the graph.

        Returns:
            Tuple of updated
                ``(scalar_features (n_edges, num_scalar_features),
                   tensor_features (n_edges, irreps_dim))``.
        """
        src = edge_index[:, 0]

        # --- Pre-scale tensor_features by env weights (fast, left-side only) ---
        # env_w is derived from the most-recent scalar features of the current
        # edge, encoding its chemical context into the TP query.
        current_scalars = accumulated_scalars[:, -self.num_scalar_features :]
        env_w = self.env_embed(current_scalars)  # (n_edges, num_tensor)
        scaled_V = _scale_by_channel(
            tensor_features, env_w, self.l_max, self.num_tensor_features
        )  # (n_edges, irreps_dim)

        # --- Many-body neighbourhood aggregation (single-channel Y) ---
        # Aggregate *unweighted* spherical harmonics to source nodes.
        # Result: Y_agg[i] = mean_k Y(r_{ik}), same shape as sh_irreps.
        # This stays single-channel, preserving the fast TP kernel path.
        node_Y = torch.zeros(
            n_nodes,
            self._sh_dim,
            dtype=tensor_features.dtype,
            device=tensor_features.device,
        )
        node_Y.scatter_add_(0, src.unsqueeze(-1).expand_as(edge_angular), edge_angular)
        src_count = torch.zeros(
            n_nodes, dtype=tensor_features.dtype, device=tensor_features.device
        )
        src_count.scatter_add_(
            0, src, torch.ones(src.shape[0], dtype=tensor_features.dtype, device=tensor_features.device)
        )
        node_Y = node_Y / src_count.clamp(min=1.0).unsqueeze(-1)

        # Pull aggregated neighbourhood Y back to each edge via its source atom
        edge_node_Y = node_Y[src]  # (n_edges, sh_dim)

        # --- Tensor product: (env_w · V) ⊗ Y_agg  [FAST: sh_irreps right side] ---
        tp_out = self.tp(scaled_V, edge_node_Y)
        new_tensor = self.tp_linear(tp_out)

        # --- Scalar track update (DenseNet) ---
        # Extract L=0 scalar invariants (first block in ir_mul layout)
        invariants = new_tensor[:, : self.num_tensor_features]

        # Latent MLP on all accumulated scalars + fresh invariants
        combined = torch.cat([accumulated_scalars, invariants], dim=-1)
        updated_scalars = self.latent_mlp(combined)

        # --- Tensor track update ---
        # Post-TP channel-wise scaling by weights from updated scalars
        env_weights_out = self.tensor_env(updated_scalars)
        updated_tensor = _scale_by_channel(
            new_tensor, env_weights_out, self.l_max, self.num_tensor_features
        )

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
    poly_p: int = Field(6, ge=1)
    scalar_mlp_hiddens: list[int] | None = None
    latent_mlp_hiddens: list[int] | None = None


class Allegro(TensorDictModuleBase):
    """Allegro equivariant feature encoder.

    Accepts a ``GraphBatch`` TensorDict and writes ``edge_features``
    into the ``edges`` sub-dict.

    Architecture::

        GraphBatch(atoms, edges)
          → [PairEmbedding]  → scalar₀, tensor₀, edge_angular
          → [AllegroLayer₁]  → scalar₁, tensor₁  (accumulated=[scalar₀, scalar₁])
          → ...
          → edges.edge_features (n_edges, num_layers, num_scalar)

    Key algorithmic properties vs. a naive pair-only model:

    * **Many-body interactions**: each layer aggregates env-weighted spherical
      harmonics from all neighbours of the source atom, so the tensor product
      captures the local chemical environment rather than just the bond vector.
    * **DenseNet scalar accumulation**: each layer's latent MLP receives the
      concatenation of ALL previous layer scalar outputs, preserving gradient
      pathways to early-layer features.

    Reference:
        Musaelian et al. "Learning Local Equivariant Representations for
        Large-Scale Atomistic Dynamics" Nature Communications 2023
        https://arxiv.org/abs/2204.05249
    """

    in_keys = [
        ("atoms", "Z"),
        ("edges", "edge_index"),
        ("edges", "bond_diff"),
        ("edges", "bond_dist"),
    ]
    out_keys = [("edges", "edge_features")]

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
        poly_p: int = 6,
        scalar_mlp_hiddens: list[int] | None = None,
        latent_mlp_hiddens: list[int] | None = None,
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
            poly_p=poly_p,
            scalar_mlp_hiddens=scalar_mlp_hiddens,
            latent_mlp_hiddens=latent_mlp_hiddens,
        )

        # Pair embedding (two-body)
        self.embedding = PairEmbedding(
            num_elements=num_elements,
            num_scalar_features=num_scalar_features,
            num_tensor_features=num_tensor_features,
            r_max=r_max,
            num_bessel=num_bessel,
            l_max=l_max,
            scalar_mlp_hiddens=scalar_mlp_hiddens,
            poly_p=poly_p,
        )

        # Allegro layers with increasing DenseNet input dimension.
        # Layer i receives accumulated scalars of dim (i+1)*num_scalar_features
        # (embedding + outputs of layers 0..i-1) plus num_tensor_features
        # invariants from the current TP.
        self.layers = nn.ModuleList(
            [
                AllegroLayer(
                    num_scalar_features=num_scalar_features,
                    num_tensor_features=num_tensor_features,
                    l_max=l_max,
                    mlp_depth=mlp_depth,
                    latent_in_dim=(i + 1) * num_scalar_features + num_tensor_features,
                    latent_mlp_hiddens=latent_mlp_hiddens,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, td: GraphBatch) -> GraphBatch:
        """Extract per-layer pair scalar features.

        Args:
            td: ``GraphBatch`` with ``atoms`` and ``edges`` sub-dicts.

        Returns:
            Same ``GraphBatch`` with ``edges.edge_features``
            ``(n_edges, num_layers, num_scalar)`` added.
        """
        Z = td["atoms", "Z"]
        bond_dist = td["edges", "bond_dist"]
        bond_diff = td["edges", "bond_diff"]
        edge_index = td["edges", "edge_index"]
        n_nodes: int = int(Z.shape[0])

        scalar_features, tensor_features, edge_angular, _ = self.embedding(
            Z=Z,
            bond_dist=bond_dist,
            bond_diff=bond_diff,
            edge_index=edge_index,
        )

        # DenseNet accumulation: start with the embedding scalars.
        accumulated: list[torch.Tensor] = [scalar_features]
        per_layer_scalars: list[torch.Tensor] = []

        for layer in self.layers:
            accumulated_cat = torch.cat(accumulated, dim=-1)
            scalar_features, tensor_features = layer(
                accumulated_scalars=accumulated_cat,
                tensor_features=tensor_features,
                edge_angular=edge_angular,
                edge_index=edge_index,
                n_nodes=n_nodes,
            )
            accumulated.append(scalar_features)
            per_layer_scalars.append(scalar_features)

        td["edges", "edge_features"] = torch.stack(per_layer_scalars, dim=1)
        return td
