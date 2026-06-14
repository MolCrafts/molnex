"""Allegro: faithful port of mir-group/allegro architecture.

This module is a direct port of the reference implementation at
``mir-group/allegro`` (commit pinned in ``specs/allegro.md`` §9). The
architecture, data flow, weight initialisation, residual structure and
readout shape all mirror the reference verbatim. Any deviation belongs in
the §4 adaptation ledger of the spec, not here.

Reference files reproduced:

* ``allegro/nn/_allegro.py::Allegro_Module`` — layer loop with DenseNet
  scalar accumulation; env weights for layer ``ℓ+1`` are sliced from
  layer ``ℓ`` 's latent MLP output, not produced by a separate ``Linear``.
* ``allegro/nn/scalarembed.py::TwoBodyBesselScalarEmbed`` /
  ``allegro/nn/_edgeembed.py::ProductTypeEmbedding`` — Bessel × cutoff,
  projected to type-embedding dim, then **multiplied** with a
  concat-of-center-and-neighbor type embedding (Hadamard, not concat).
* ``allegro/nn/tensorembed.py::TwoBodySphericalHarmonicTensorEmbed`` —
  initial tensor track ``V_0 = MakeWeightedChannels(SH, weights)`` with
  per-(irrep, channel) weights from a 1-layer ``ScalarMLPFunction``.
* ``allegro/nn/_strided/_channels.py::MakeWeightedChannels`` — per-
  (irrep, channel) weighting of a single-channel SH basis.
* ``nequip/nn/mlp.py::ScalarMLPFunction`` / ``ScalarLinearLayer`` —
  variance-preserving init: ``weight ~ U(-√3, √3)`` then scaled by
  ``α = gain / √fan_in`` with ``gain = √2`` (relu) for hidden layers and
  ``gain = 1`` for the first layer (``forward_weight_init=True``); no bias.
* ``nequip/nn/embedding/_edge.py::BesselEdgeLengthEncoding`` —
  ``bessel = sinc(n·r/r_max) · n`` then multiplied by the polynomial
  cutoff envelope.

Reference:
    Musaelian et al. "Learning Local Equivariant Representations for
    Large-Scale Atomistic Dynamics" Nature Communications 14, 579 (2023)
    https://arxiv.org/abs/2204.05249
"""

from __future__ import annotations

import itertools
import math
from typing import Optional

import cuequivariance as cue
import torch
import torch.nn as nn
from cuequivariance import O3
from pydantic import BaseModel, ConfigDict, Field
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase

from molix import config
from molrep.embedding.angular import SphericalHarmonics
from molrep.embedding.cutoff import PolynomialCutoff
from molrep.embedding.mlp import ScalarMLPFunction
from molrep.interaction.product import (
    EquivariantPolynomialTP,
    sh_irreps_from_l_max,
)

__all__ = ["Allegro", "AllegroSpec"]


# ===========================================================================
# Per-(irrep, channel) weighting of a single-channel SH basis
# ===========================================================================


def _make_weighted_channels(
    sh: torch.Tensor,
    weights: torch.Tensor,
    mul: int,
    sh_irrep_dims: list[int],
) -> torch.Tensor:
    """Port of ``MakeWeightedChannels`` (``weight_individual_irreps=True``).

    For each irrep ``r`` of dim ``d_r`` and each channel ``u``, applies
    ``out[e, m_in_r, u] = sh[e, r, m_in_r] * weights[e, r, u]``. The
    output is laid out in cuequivariance ``ir_mul`` order: per irrep,
    the ``m`` axis is outer and the channel axis is inner-fast, so the
    flat dim is ``Σ_r d_r · mul = irreps_dim · mul``.

    Args:
        sh: ``(E, irreps_dim)`` single-channel SH features (mul=1).
        weights: ``(E, num_irreps · mul)`` flat per-(irrep, channel) weights,
            grouped by irrep (channel index runs fast inside each irrep).
        mul: Channel multiplicity ``u``.
        sh_irrep_dims: ``[d_r for each irrep r]`` of the SH basis.

    Returns:
        ``(E, irreps_dim · mul)`` weighted features in ir_mul layout.
    """
    chunks: list[torch.Tensor] = []
    sh_pos = 0
    for r, d_r in enumerate(sh_irrep_dims):
        sh_block = sh[..., sh_pos : sh_pos + d_r]  # (E, d_r)
        w_block = weights[..., r * mul : (r + 1) * mul]  # (E, mul)
        # ir_mul: m axis outer, channel axis inner → (E, d_r, mul) → flatten
        chunks.append(
            (sh_block.unsqueeze(-1) * w_block.unsqueeze(-2)).reshape(*sh.shape[:-1], d_r * mul)
        )
        sh_pos += d_r
    return torch.cat(chunks, dim=-1)


# ===========================================================================
# Per-channel Allegro TP descriptor (subscripts "u,iu,ju,ku+ijk")
# ===========================================================================


def _allegro_uuu_descriptor(
    irreps_in: cue.Irreps,
    irreps_sh: cue.Irreps,
    irreps_out: cue.Irreps,
) -> cue.EquivariantPolynomial:
    r"""Per-channel Allegro TP descriptor with paths fused into shared output segments.

    Mirrors :class:`mir-group/allegro::Contracter` with
    ``path_channel_coupling=True`` and ``num_paths > 1`` — every CG path
    leading to the same output ``ir`` accumulates into the **same**
    output segment, so the trainable weight count per output ``ir`` is
    ``u · num_paths_to_ir`` (matching the reference's ``(u, num_paths)``
    weight tensor). There is no extra equivariant ``Linear`` after the
    TP; the descriptor's output is already in the target ``irreps_out``.

    Subscripts: ``weights[u], lhs[iu], rhs[ju], output[ku]``.

    .. math::
       \text{out}_{(\ell_3,m_3),u} = \sum_{p\in\mathrm{paths}\to\ell_3}
       w_{u,p}\, \mathrm{CG}_{(\ell_1^p,m_1)(\ell_2^p,m_2)\to(\ell_3,m_3)}\,
       V_{(\ell_1^p,m_1),u}\, v_{(\ell_2^p,m_2),u}

    Args:
        irreps_in: Irreps of the LHS (uniform multiplicity ``u``).
        irreps_sh: Irreps of the SH basis (single channel).
        irreps_out: Target output irreps (uniform multiplicity ``u``).
            Must already be pre-pruned to irs reachable from
            ``irreps_in × irreps_sh``; the caller in :class:`Allegro`
            does this via the same forward + backward pruning as the
            reference (``allegro/nn/_allegro.py:115-160``).

    Returns:
        :class:`cue.EquivariantPolynomial` with inputs
        ``(weights, lhs, rhs)`` and output of irreps ``irreps_out``.
    """
    if len(set(irreps_in.muls)) != 1:
        raise ValueError(
            "_allegro_uuu_descriptor requires uniform multiplicity in irreps_in "
            f"(got muls={irreps_in.muls})"
        )
    if len(set(irreps_out.muls)) != 1 or irreps_out.muls[0] != irreps_in.muls[0]:
        raise ValueError(
            "irreps_out must have the same uniform multiplicity as irreps_in "
            f"(got irreps_in.muls={irreps_in.muls}, irreps_out.muls={irreps_out.muls})"
        )
    u = irreps_in.muls[0]
    G = irreps_in.irrep_class
    d = cue.SegmentedTensorProduct.from_subscripts("u,iu,ju,ku+ijk")

    for _mul, ir in irreps_in:
        d.add_segment(1, (ir.dim, u))
    for _mul, ir in irreps_sh:
        d.add_segment(2, (ir.dim, u))

    # Pre-create one output segment per ir in irreps_out, in the order the
    # caller declared them. Fusing paths into a shared segment per ir is
    # what makes the trainable weight count match the reference Contracter.
    out_seg_idx_by_ir: dict[cue.Irrep, int] = {}
    for seg_idx, (_mul, ir) in enumerate(irreps_out):
        out_seg_idx_by_ir[ir] = seg_idx
        d.add_segment(3, (ir.dim, u))

    # Add CG paths — only for combinations leading to a target ir.
    # Each path contributes a fresh ``(u,)`` weight segment to operand 0,
    # so the total weight count is ``u · num_paths_total``, distributed
    # per-ir as ``u · num_paths_to_ir`` (== reference's ``(u, num_paths)``).
    for (i1, (_m1, ir1)), (i2, (_m2, ir2)) in itertools.product(
        enumerate(irreps_in), enumerate(irreps_sh)
    ):
        for ir3 in ir1 * ir2:
            if ir3 not in out_seg_idx_by_ir:
                continue
            seg_idx = out_seg_idx_by_ir[ir3]
            for cg in G.clebsch_gordan(ir1, ir2, ir3):
                d.add_path(None, i1, i2, seg_idx, c=cg, dims={"u": u})

    return cue.EquivariantPolynomial(
        [
            cue.IrrepsAndLayout(irreps_in.new_scalars(d.operands[0].size), cue.ir_mul),
            cue.IrrepsAndLayout(irreps_in, cue.ir_mul),
            cue.IrrepsAndLayout(cue.Irreps(G, [(u, ir) for _, ir in irreps_sh]), cue.ir_mul),
        ],
        [cue.IrrepsAndLayout(irreps_out, cue.ir_mul)],
        cue.SegmentedPolynomial.eval_last_operand(d),
    )


def _tp_path_exists(irreps_in1: cue.Irreps, irreps_in2: cue.Irreps, ir_out: cue.Irrep) -> bool:
    """Port of ``nequip.nn.tp_path_exists`` — does any CG path land in ``ir_out``?"""
    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False


def _build_layer_irreps(
    sh_irreps: cue.Irreps,
    num_layers: int,
    G: type,
    *,
    last_layer_keep_tensors: bool = False,
) -> list[cue.Irreps]:
    """Two-pass irreps build matching ``mir-group/allegro::Allegro_Module``.

    Forward pass (``allegro/nn/_allegro.py:115-141``) — for each layer,
    start from the ``tensor_track_allowed_irreps`` (= the SH irreps), and
    drop those that no path from the previous layer's argument can reach.
    Last layer is forced to ``0e`` only **unless** ``last_layer_keep_tensors``
    is True (in which case the last layer keeps the full SH irreps stack
    just like the middle layers — required when an equivariant downstream
    head consumes ``("edges","edge_tensor_features")``).

    Backward pass (``allegro/nn/_allegro.py:143-161``) — walking from the
    last (scalar) output back to layer 0, drop any ``arg_ir`` that cannot
    eventually reach the final output via subsequent ``arg × env`` TPs.

    Returns ``L+1`` irreps lists with ``mul=1``: ``[arg_layer_0,
    arg_layer_1, ..., arg_layer_L, out_layer_L_pruned]``. The first
    entry is the LHS irreps for layer 0's TP, the last is the encoder's
    final tensor output (scalars-only by default; full irreps when
    ``last_layer_keep_tensors``). ``arg_layer_0`` should equal
    ``sh_irreps`` for typical configs (no backward pruning at depth 0).
    """
    env_embed_irreps = sh_irreps  # mul=1
    tensor_track_allowed_irreps = sh_irreps  # mul=1, full SH
    SCALAR_IRREPS = cue.Irreps(G, [(1, "0e")])

    # === forward pass ===
    tps_irreps: list[cue.Irreps] = [env_embed_irreps]
    for layer_idx in range(num_layers):
        if layer_idx == num_layers - 1 and not last_layer_keep_tensors:
            allowed = SCALAR_IRREPS
        else:
            allowed = tensor_track_allowed_irreps
        arg = tps_irreps[-1]
        pruned = cue.Irreps(
            G,
            [(1, ir) for _, ir in allowed if _tp_path_exists(arg, env_embed_irreps, ir)],
        )
        tps_irreps.append(pruned)

    # === backward pass ===
    out_irreps_iter = tps_irreps[-1]
    new_tps_irreps: list[cue.Irreps] = [out_irreps_iter]
    for arg_irreps in reversed(tps_irreps[:-1]):
        kept: list[tuple[int, cue.Irrep]] = []
        for _, arg_ir in arg_irreps:
            for _, env_ir in env_embed_irreps:
                if any(i in out_irreps_iter for i in arg_ir * env_ir):
                    kept.append((1, arg_ir))
                    break
        kept_irreps = cue.Irreps(G, kept)
        new_tps_irreps.append(kept_irreps)
        out_irreps_iter = kept_irreps

    if len(new_tps_irreps) != len(tps_irreps):
        raise RuntimeError("internal: backward pruning produced wrong number of layers")
    return list(reversed(new_tps_irreps))


# ===========================================================================
# Spec
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
    poly_p: int = Field(6, ge=1)
    # Two-body scalar embed dim. Reference name: ``module_output_dim``
    # in :class:`TwoBodyBesselScalarEmbed` / :class:`ProductTypeEmbedding`.
    # The two-body scalar embedding equals ``type_embed × basis_linear(bessel)``
    # of dim ``type_embed_dim``; this is what feeds directly into the layer
    # loop's ``first_layer_env_embed_projection`` and ``env_embed_linear``.
    type_embed_dim: int = Field(64, gt=0)
    # Per-layer Allegro latent MLP. Reference: ``latent_kwargs.hidden_layers_*``.
    latent_mlp_depth: int = Field(2, ge=0)
    latent_mlp_width: int = Field(128, gt=0)
    avg_num_neighbors: float = Field(..., gt=0.0)
    # When True, ``forward`` additionally writes the final layer's tensor
    # features (full irreps stack, mul=num_tensor_features, ir_mul layout)
    # to ``("edges", "edge_tensor_features")`` so equivariant downstream
    # heads (e.g. ``molpot.heads.PermMultipoleHead`` with ``dipole=True``
    # or ``quadrupole=True``) can consume the encoder's l ≥ 1 channels.
    # Off by default — the scalar DenseNet
    # output is unchanged either way; the extra tensor only adds an
    # in-place TensorDict write of size ``E · irreps_dim``.
    expose_tensor_track: bool = False


# ===========================================================================
# Allegro encoder
# ===========================================================================


class Allegro(TensorDictModuleBase):
    """Faithful port of mir-group/allegro encoder.

    Inputs (read from ``TensorDict``):

    * ``("atoms","Z")`` — atomic numbers ``(N,)``.
    * ``("edges","edge_index")`` — ``(E, 2)``, ``[:,0]=src/center``.
    * ``("edges","bond_diff")`` — ``(E, 3)``, ``pos[dst] - pos[src]``.
    * ``("edges","bond_dist")`` — ``(E,)``.

    Output:

    * ``("edges","edge_features")`` — ``(E, num_scalar_features · (num_layers + 1))``,
      the concatenation of the two-body scalar features and every layer's
      newly-produced scalar features (DenseNet output, mirroring
      ``allegro/nn/_allegro.py:323``).

    Pipeline (mirrors reference verbatim):

    1. ``r/r_max → bessel · u(r)`` (BesselEdgeLengthEncoding).
    2. ``twobody = type_embed × basis_linear(bessel)`` (ProductTypeEmbedding).
       This *is* the entire two-body scalar embedding — no extra MLP.
    3. ``env_embed_linear(twobody) → V_0 = MakeWeightedChannels(SH, w_V0)``
       (TwoBodySphericalHarmonicTensorEmbed).
    4. ``first_layer_env_embed_projection(twobody) → [scalar_features, env_w]``.
    5. For each Allegro layer:

       a. ``MakeWeightedChannels(SH, env_w) → env_w_edges``.
       b. ``scatter_add → / √⟨|N|⟩``.
       c. ``Contracter(V, env_w_scatter[src]) → V'`` (per-channel uuu TP
          with paths fused into per-ir output segments — the descriptor
          targets the layer's pruned ``ir_out`` directly, no extra Linear).
       d. ``scalars = V'[:, :u]`` (L=0 invariants in ir_mul).
       e. ``latents = latent_mlp(cat(accumulated + [scalars]))``.
       f. ``new_scalar = latents[:, :F]`` appended to ``accumulated``;
          ``env_w = latents[:, F:F+weight_numel]`` (skipped on last layer).

    6. ``edge_features = cat(accumulated)``.
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
        poly_p: int = 6,
        type_embed_dim: int = 64,
        latent_mlp_depth: int = 2,
        latent_mlp_width: int = 128,
        latent_activation: Optional[type[nn.Module]] = nn.SiLU,
        avg_num_neighbors: float,
        expose_tensor_track: bool = False,
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
            poly_p=poly_p,
            type_embed_dim=type_embed_dim,
            latent_mlp_depth=latent_mlp_depth,
            latent_mlp_width=latent_mlp_width,
            avg_num_neighbors=avg_num_neighbors,
            expose_tensor_track=expose_tensor_track,
        )
        self.expose_tensor_track = bool(expose_tensor_track)

        # Reference requires even type_embed_dim (split between center and neighbor).
        if type_embed_dim % 2 != 0:
            raise ValueError(
                f"type_embed_dim must be even (split center/neighbor); got {type_embed_dim}"
            )

        self.num_layers = int(num_layers)
        self.num_scalar_features = int(num_scalar_features)
        self.num_tensor_features = int(num_tensor_features)
        self.l_max = int(l_max)
        self.r_max = float(r_max)
        self.avg_num_neighbors_inv_sqrt = 1.0 / math.sqrt(float(avg_num_neighbors))

        # === bessel × cutoff (BesselEdgeLengthEncoding) ===
        # Reference: bessel = sinc(n·r/r_max) · n; weights are non-trainable.
        bessel_n = torch.linspace(1.0, num_bessel, steps=num_bessel, dtype=config.ftype).unsqueeze(
            0
        )  # (1, num_bessel)
        self.register_buffer("bessel_n", bessel_n, persistent=False)
        self.cutoff_fn = PolynomialCutoff(r_cut=r_max, exponent=poly_p)

        # === atomic type embedding (ProductTypeEmbedding) ===
        # `type_embed × basis_linear(bessel)` is the ENTIRE two-body scalar
        # embedding; reference's `TwoBodyBesselScalarEmbed` does not insert
        # any extra MLP between this and the layer loop.
        half = type_embed_dim // 2
        self.center_embed = nn.Embedding(num_elements, half, dtype=config.ftype)
        self.neighbor_embed = nn.Embedding(num_elements, half, dtype=config.ftype)
        self.basis_linear = ScalarMLPFunction(
            input_dim=num_bessel,
            output_dim=type_embed_dim,
            hidden_layers_depth=0,
            nonlinearity=None,
            bias=False,
        )

        # === spherical harmonics (single channel) ===
        self.spherical_harmonics = SphericalHarmonics(l_max=l_max)
        # Per-irrep dims of the SH basis: [1, 3, 5, ...] up to l_max.
        self._sh_irrep_dims: list[int] = [(2 * l + 1) for l in range(l_max + 1)]
        # Number of irreps in the SH basis (= num env-w slots per channel).
        self._num_sh_irreps: int = l_max + 1

        # weight_numel = num_irreps · num_tensor_features (per-(irrep, channel))
        self._env_weight_numel: int = self._num_sh_irreps * num_tensor_features

        # === V_0 weights: env_embed_linear (in TwoBodySphericalHarmonicTensorEmbed) ===
        # Reference: input dim = `module_output_dim` = `type_embed_dim`.
        self.env_embed_linear = ScalarMLPFunction(
            input_dim=type_embed_dim,
            output_dim=self._env_weight_numel,
            hidden_layers_depth=0,
            nonlinearity=None,
            bias=False,
        )

        # === first-layer env-embed projection ===
        # Outputs [scalar_features, env_w_for_layer_0] jointly. Reference
        # input dim = `scalar_input_dim` = `type_embed_dim`.
        self.first_layer_env_embed_projection = ScalarMLPFunction(
            input_dim=type_embed_dim,
            output_dim=num_scalar_features + self._env_weight_numel,
            hidden_layers_depth=0,
            nonlinearity=None,
            bias=False,
        )

        # === Per-layer TPs and latent MLPs ===
        # Two-pass irreps construction, mirroring `mir-group/allegro::
        # Allegro_Module.__init__:115-161`. Forward pass starts from the
        # full SH irreps and prunes irs that no path can reach; the last
        # layer is forced to scalars-only. Backward pass then drops irs
        # in earlier layers that cannot eventually feed the final scalar.
        sh_irreps = cue.Irreps(O3, sh_irreps_from_l_max(l_max))  # mul=1
        tps_irreps_mul1 = _build_layer_irreps(
            sh_irreps,
            num_layers,
            O3,
            last_layer_keep_tensors=self.expose_tensor_track,
        )
        # Inflate to mul=u for the actual TP construction.
        tps_irreps_mu: list[cue.Irreps] = [
            cue.Irreps(O3, [(num_tensor_features, ir) for _, ir in irr]) for irr in tps_irreps_mul1
        ]
        # Layer 0's LHS = V_0, which has the full SH irreps. Backward
        # pruning is a no-op at depth 0 for typical configs (env spans
        # all parities up to ``l_max``); guard against that being false
        # so we don't silently drop V_0 channels.
        if tps_irreps_mul1[0] != sh_irreps:
            raise NotImplementedError(
                "Backward pruning trimmed layer 0's input irreps "
                f"(got {tps_irreps_mul1[0]}, V_0 has {sh_irreps}) — "
                "this configuration is exotic enough that the V_0 → "
                "layer-0-TP slicing path is not implemented yet."
            )

        self.tps = nn.ModuleList()
        # ``per_layer_n_scalar_outs[ℓ]`` is the multiplicity of 0e in
        # ``tps_irreps_mu[ℓ+1]`` (= layer ℓ's TP output). Used to size
        # the latent MLP input. ``tps_irreps_mu[ℓ+1][0]`` is always the
        # 0e block (irreps are sorted by l with 0e first), so this is
        # ``num_tensor_features`` for every layer that emits scalars.
        n_scalar_outs: list[int] = []
        for layer_idx in range(num_layers):
            arg_irreps = tps_irreps_mu[layer_idx]
            ir_out = tps_irreps_mu[layer_idx + 1]
            if len(ir_out) == 0 or ir_out[0].ir.l != 0:
                raise RuntimeError(
                    f"layer {layer_idx} output irreps {ir_out} must start "
                    "with 0e — pruning should preserve scalars first."
                )
            n_scalar_outs.append(int(ir_out[0].mul))

            poly = _allegro_uuu_descriptor(arg_irreps, sh_irreps, ir_out)
            tp = EquivariantPolynomialTP(
                poly,
                shared_weights=True,
                internal_weights=True,
                method="uniform_1d",
                dtype=config.ftype,
            )
            self.tps.append(tp)
        self._n_scalar_outs: list[int] = n_scalar_outs
        # Final tensor-track output irreps (mul=num_tensor_features, ir_mul
        # layout). Exposed for equivariant downstream heads via
        # ``("edges", "edge_tensor_features")`` when ``expose_tensor_track``.
        # ``out_keys`` is intentionally left at the class default — the extra
        # write is opt-in and visible to consumers via direct TD access.
        self.tensor_track_irreps: cue.Irreps = tps_irreps_mu[num_layers]

        # === Per-layer latent MLPs (DenseNet + env-w slice for next layer) ===
        # Each layer's latent input is cat([all previous accumulated scalars,
        # this layer's L=0 invariants from the TP]); each output is
        # cat([new scalar feature, env_w for next layer]) (env_w omitted on
        # the last layer).
        self.latents = nn.ModuleList()
        for layer_idx in range(num_layers):
            latent_input_dim = num_scalar_features * (layer_idx + 1) + n_scalar_outs[layer_idx]
            latent_output_dim = num_scalar_features + (
                self._env_weight_numel if layer_idx < num_layers - 1 else 0
            )
            self.latents.append(
                ScalarMLPFunction(
                    input_dim=latent_input_dim,
                    output_dim=latent_output_dim,
                    hidden_layers_depth=latent_mlp_depth,
                    hidden_layers_width=latent_mlp_width,
                    nonlinearity=latent_activation,
                    bias=False,
                )
            )

        # Final output dim (DenseNet stack of all scalar layers).
        self.output_dim: int = num_scalar_features * (num_layers + 1)

    def forward(self, td: TensorDict) -> TensorDict:
        """Run the encoder and write ``("edges","edge_features")`` in place."""
        Z = td["atoms", "Z"]
        bond_dist = td["edges", "bond_dist"]
        bond_diff = td["edges", "bond_diff"]
        edge_index = td["edges", "edge_index"]
        n_nodes: int = int(Z.shape[0])
        src = edge_index[:, 0]
        dst = edge_index[:, 1]

        # === 1. Bessel × polynomial cutoff (BesselEdgeLengthEncoding) ===
        x_norm = (bond_dist / self.r_max).unsqueeze(-1)  # (E, 1)
        # torch.sinc(z) = sin(πz)/(πz); sinc(n·r/r_max) · n.
        bessel = torch.sinc(x_norm * self.bessel_n) * self.bessel_n  # (E, num_bessel)
        edge_cutoff = self.cutoff_fn(bond_dist)  # (E,)
        edge_radial = bessel * edge_cutoff.unsqueeze(-1)  # (E, num_bessel)

        # === 2. ProductTypeEmbedding: type_embed × basis_linear(bessel) ===
        # This is the ENTIRE two-body scalar embedding (= reference's
        # ``EDGE_EMBEDDING_KEY`` after ``TwoBodyBesselScalarEmbed``).
        type_embed = torch.cat(
            [self.center_embed(Z[src]), self.neighbor_embed(Z[dst])], dim=-1
        )  # (E, type_embed_dim)
        twobody_scalar_embed = type_embed * self.basis_linear(edge_radial)
        # twobody_scalar_embed: (E, type_embed_dim)

        # === 3. Spherical harmonics + initial tensor track V_0 ===
        # ``SphericalHarmonics(normalize=True)`` normalises ``bond_diff`` internally
        # (cuEquivariance kernel). NeighborList guarantees ``bond_dist > 0``
        # (self-edges excluded by ``get_neighbor_pairs``), so no ``+ε`` shim
        # is needed; passing ``bond_diff`` directly saves one division per edge.
        tensor_basis = self.spherical_harmonics(bond_diff)  # (E, irreps_sh_dim)
        v0_weights = self.env_embed_linear(twobody_scalar_embed)  # (E, weight_numel)
        tensor_features = _make_weighted_channels(
            tensor_basis,
            v0_weights,
            mul=self.num_tensor_features,
            sh_irrep_dims=self._sh_irrep_dims,
        )  # (E, irreps_dim · u) ir_mul

        # === 4. First-layer env-embed projection ===
        projection = self.first_layer_env_embed_projection(twobody_scalar_embed)
        twobody_scalar_features = projection[..., : self.num_scalar_features]
        env_w = projection[..., self.num_scalar_features :]
        accumulated: list[torch.Tensor] = [twobody_scalar_features]

        # === 5. Allegro layers ===
        for layer_idx in range(self.num_layers):
            tp = self.tps[layer_idx]
            latent = self.latents[layer_idx]
            n_scalar = self._n_scalar_outs[layer_idx]

            # (a) per-edge weighted SH using current env_w
            env_w_edges = _make_weighted_channels(
                tensor_basis,
                env_w,
                mul=self.num_tensor_features,
                sh_irrep_dims=self._sh_irrep_dims,
            )  # (E, irreps_sh_dim · u)

            # (b) scatter to nodes, then /√⟨|N|⟩
            env_w_scatter = torch.zeros(
                n_nodes,
                env_w_edges.shape[-1],
                dtype=env_w_edges.dtype,
                device=env_w_edges.device,
            )
            env_w_scatter.scatter_add_(0, src.unsqueeze(-1).expand_as(env_w_edges), env_w_edges)
            env_w_scatter = env_w_scatter * self.avg_num_neighbors_inv_sqrt

            # (c) per-channel uuu TP, gathering env_w_scatter per-edge by source.
            # The descriptor outputs directly into the layer's pruned ``ir_out``
            # (no extra equivariant linear — see ``_allegro_uuu_descriptor``).
            new_tensor = tp(tensor_features, env_w_scatter, indices_2=src)

            # (d) extract L=0 invariants (first ``u`` components in ir_mul,
            # since ``ir_out``'s first irrep is always 0e).
            scalars = new_tensor[..., :n_scalar]  # (E, u)

            # (e) latent MLP over DenseNet input
            latents_out = latent(
                torch.cat(accumulated + [scalars], dim=-1)
            )  # (E, F + weight_numel?)

            # (f) split into new scalar feature and (optionally) next env_w
            new_scalar = latents_out[..., : self.num_scalar_features]
            accumulated.append(new_scalar)
            if layer_idx < self.num_layers - 1:
                env_w = latents_out[..., self.num_scalar_features :]

            # Update the tensor track for the next layer.
            tensor_features = new_tensor

        # === 6. Final scalar output: cat all accumulated layer features ===
        td["edges", "edge_features"] = torch.cat(accumulated, dim=-1)
        if self.expose_tensor_track:
            # Final layer's TP output, shape (E, tensor_track_irreps.dim) ir_mul.
            td["edges", "edge_tensor_features"] = tensor_features
        return td
