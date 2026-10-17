"""MACE building blocks on cuEquivariance: convolution, interactions, product basis.

Every block here serves the MACE family only (:mod:`molzoo.mace.encoder`,
:mod:`molzoo.mace.potential`, :mod:`molzoo.mace.polar`); the generic pieces
they are built from — irreps-string helpers, radial bases, the moment-normalised
MLP — stay in :mod:`molrep`. Sub-layer names mirror upstream MACE throughout so
the official weights transfer by direct copy.

**Channel-wise convolution.** :class:`_CueqChannelwiseBlockMixin` owns the
radial heads and the fused indexed convolution of the density and residual
interactions. The path is the official cuEquivariance indexed convolution:
``indices_1`` gathers source nodes and ``indices_out`` scatters directly into
target nodes, so no ``(E, target.dim)`` message is ever materialised.

**Radial MLP.** :class:`RadialMLP` maps radial basis features (e.g. Bessel RBF
output) to tensor-product weight vectors, ``Linear -> LayerNorm -> SiLU`` blocks
in the ESEN/FairChem style; the residual interaction and the Polar field blocks
use it for both the weight generator and the edge-density MLP.

**Gated nonlinearity.** :class:`GatedNonlinearity` is a graph-break-free gated
nonlinearity with MACE's numerics, expressed with ``cuequivariance`` irreps so
one irreps backend serves the whole stack. It has **no learnable parameters** —
only activation functions and their ``normalize2mom`` constants. The input is
the sorted direct sum of ``(scalars, gates, gated)`` irreps, the same convention
as MACE's gate and ``GatedEquivariantBlock``: scalars are activated directly;
each gated (l>0) irrep is multiplied by a sigmoid-activated scalar gate.
Supports ``ir_mul`` (cuequivariance) and ``mul_ir`` (multiplicity-major)
layouts.

**Element index.** Every per-element map reads one :class:`ElementIndex`,
built once per energy call by ``MACEEncoder.element_index`` from the atomic
numbers: the element row of each atom, the stable order that groups atoms by
element and the fixed-size per-element counts. Nothing downstream recovers the
element from a one-hot (``argmax``) or re-sorts per layer.

**One-hot linear.** :class:`OneHotLinear` is ``cuet.Linear(Cx0e → Fx0e)`` fed a
one-hot: against a one-hot, the linear is the row lookup
``y_n = c · W[e_n, :]`` with ``c`` the descriptor's path coefficient
(``1/√C``). The weight is the linear's own, same key and layout, so a
checkpoint loads as before; the lookup replaces an ``(N, C) @ (C, F)`` product
(MACE's ``node_embedding`` and the residual interaction's source / target
embeddings).

**Per-element linear (``skip_tp``).** MACE's ``skip_tp`` is
``FullyConnectedTensorProduct(x, one_hot(element))`` with ``C`` scalar
attributes. Against a one-hot, that product is a linear map whose weights depend
only on the element: path ``p`` (an input irrep block to the output block of the
same irrep) contributes

.. math:: y_k = c_p \\, x_i \\, W_p[:, e, :],

with ``W_p`` the ``(u, C, w)`` block of the product's weight, ``e`` the atom's
element and ``c_p = 1/\\sqrt{u C}`` the descriptor's coefficient (identity in the
irrep components). :class:`ElementLinear` keeps the product's weight unchanged —
same parameter, same layout, so a checkpoint loads as before — and evaluates the
map per atom on every device: the element's ``(u, w)`` block of each path is
gathered (``index_select`` on the :class:`ElementIndex`) and contracted with
the atom's block. The gather holds ``u·w`` floats per atom and path (MACE-MP's
first ``skip_tp``, 4096 atoms, 89 elements, fp32: 0.79 GiB peak against 0.45
GiB for the kernel below), never syncs with the host, and records into a CUDA
graph. The alternatives, measured on GH200: the dense product over the one-hot
runs on a fallback kernel of cue's ``fused_tp`` (791 ms against 3.7 ms for the
same ``skip_tp`` on 4600 atoms, forward plus input gradient);
``cuequivariance::indexed_linear_B`` (cue 0.12.0; atoms sorted by element, one
call per path, the layout of cue's MACE tutorial) is 1.1-1.3x faster forward
but reads the counts on the host — one sync per call — and invalidates a
CUDA-graph capture (``cudaErrorStreamCaptureInvalidated``, job 3972523), so
``Compiler(cuda_graphs=True)`` cannot run it; its forward plus input gradient
is 1-2x slower than the gather's.

**"Real agnostic" interactions.** Faithful ports of MACE's
``RealAgnosticInteractionBlock``, ``RealAgnosticResidualInteractionBlock``
(MACE-OFF, MACE-MP-0) and their density-normalised twins
``RealAgnosticDensityInteractionBlock`` /
``RealAgnosticDensityResidualInteractionBlock`` (MatPES). One edge→node
message-passing layer with:

* an official cuEquivariance channel-wise convolution ``node ⊗ Y_l(r̂)``;
* a normalisation of the aggregated message — either the constant
  ``1 / avg_num_neighbors`` (:class:`AgnosticInteraction`,
  :class:`AgnosticResidualInteraction`) or the **learned density**
  ``1 / (ρ + 1)`` with ``ρ_i = Σ_j tanh(MLP(edge_feats_ij)²)·u(r_ij)``
  (:class:`DensityInteraction`, :class:`DensityResidualInteraction`), which
  lets one model span the density range of the periodic table;
* an element-selecting ``skip_tp`` (fully-connected tensor product against the
  one-hot node attributes, evaluated as :class:`ElementLinear`).

Normalisation and skip placement are independent: the first-layer blocks apply
``skip_tp`` to the outgoing message (there is no residual to carry), the
residual blocks apply it to the incoming node features and hand the result to
the downstream product block as the skip connection.

**Residual non-linear interaction.** :class:`ResidualInteraction` is a faithful
port of MACE's ``RealAgnosticResidualNonLinearInteractionBlock``: per-element
source/target embeddings concatenated onto the radial edge features that drive
the tensor-product weights, the channel-wise convolution, a learnable density
normalisation ``message / (ρ·β + α)``, a residual (``linear_res``) path and an
equivariant gated nonlinearity, and a separate ``skip_tp`` (Linear) carried out
for the downstream product block.

**Product basis.** :class:`EquivariantProductBasis` is a faithful port of MACE's
``EquivariantProductBasisBlock``: a per-element (or element-agnostic) symmetric
contraction that raises the body order of the node features, followed by an
equivariant linear and an optional residual skip add. Sub-layer names
(``symmetric_contractions``, ``linear``) mirror MACE.

All blocks are built from ``cuequivariance`` primitives in the ``cue.ir_mul``
layout.

Reference:
    Batatia et al. "MACE: Higher Order Equivariant Message Passing Neural
    Networks for Fast and Accurate Force Fields" NeurIPS 2022, Eq. (10)-(11).
    https://arxiv.org/abs/2206.07697
    Batatia et al. "A foundation model for atomistic materials chemistry"
    (MACE-MP-0). https://arxiv.org/abs/2401.00096
    MACE ``mace.modules.gate``, ``mace.modules.radial.RadialMLP``;
    FairChem/ESEN radial MLP.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import NamedTuple

import cuequivariance as cue
import cuequivariance_torch as cuet
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from molix import config
from molix.F.scatter import scatter_sum_compile_safe as _scatter_sum
from molrep import cueq
from molrep.embedding.mlp import MomentNormalizedMLP, normalize2mom


class _CueqChannelwiseBlockMixin:
    """Radial heads and the fused cuEq convolution shared by the density and residual blocks."""

    conv_tp: cuet.ChannelWiseTensorProduct
    conv_tp_weights: nn.Module
    #: The learned edge density head, or ``None`` for a block normalised by a
    #: constant (``avg_num_neighbors``); frozen at construction.
    density_fn: nn.Module | None

    def _radial_heads(
        self,
        features: torch.Tensor,
        source_index: torch.Tensor,
        target_index: torch.Tensor,
        projected: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Tensor-product weights and edge density (``None`` without a density head).

        With node embeddings, the radial MLPs read
        ``cat([edge_feats, source_emb[src], target_emb[tgt]])``. That
        concatenation is never built: ``projected`` carries layer 0 split by
        those three blocks, with the two node halves already multiplied
        per *node*, so only ``(E, hidden)`` is gathered per edge instead of
        ``(E, edge_dim + 2·n_scalar)``.

        Beyond the memory — 1056 channels per edge at Polar-1-M widths — the
        wide form is what made Inductor emit a ``1056 * edge`` index constant
        that leaves int32 above ~2·10⁶ edges and fails to compile.
        """
        if not projected:
            density = (
                None if self.density_fn is None else torch.tanh(self.density_fn(features) ** 2)
            )
            return self.conv_tp_weights(features), density
        w_conv, src_conv, tgt_conv, b_conv, w_dens, src_dens, tgt_dens, b_dens = projected
        hidden_conv = features @ w_conv + src_conv[source_index] + tgt_conv[target_index]
        hidden_dens = features @ w_dens + src_dens[source_index] + tgt_dens[target_index]
        # The first layer's bias belongs to the sum as a whole, once.
        if b_conv.numel() != 0:
            hidden_conv = hidden_conv + b_conv
        if b_dens.numel() != 0:
            hidden_dens = hidden_dens + b_dens
        return (
            self.conv_tp_weights.forward_from_first(hidden_conv),
            torch.tanh(self.density_fn.forward_from_first(hidden_dens) ** 2),
        )

    def _project_node_features(
        self,
        edge_dim: int,
        source_features: torch.Tensor | None,
        target_features: torch.Tensor | None,
    ) -> tuple[torch.Tensor, ...]:
        """Split each radial MLP's layer 0 and pre-multiply the node halves."""
        if source_features is None or target_features is None:
            return ()
        n_scalar = source_features.shape[-1]
        widths = [edge_dim, n_scalar, n_scalar]
        empty = source_features.new_empty(0)
        w_conv, ws_conv, wt_conv = self.conv_tp_weights.first_layer_parts(widths)
        w_dens, ws_dens, wt_dens = self.density_fn.first_layer_parts(widths)
        b_conv = self.conv_tp_weights.first_layer_bias()
        b_dens = self.density_fn.first_layer_bias()
        return (
            w_conv,
            source_features @ ws_conv,
            target_features @ wt_conv,
            empty if b_conv is None else b_conv,
            w_dens,
            source_features @ ws_dens,
            target_features @ wt_dens,
            empty if b_dens is None else b_dens,
        )

    def _cueq_channelwise_conv(
        self,
        node_features: torch.Tensor,
        edge_angular: torch.Tensor,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: torch.Tensor | None,
        *,
        source_features: torch.Tensor | None = None,
        target_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Evaluate the official indexed cuEq convolution and its scalar density.

        The density is ``None`` for a block without a density head.

        cuEq is the sole owner of CG, contraction, scatter and autograd: the
        radial weights are handed to the indexed tensor product together with
        ``indices_1`` / ``indices_out``, so nothing of edge width is built
        outside the kernel.
        """
        if (source_features is None) != (target_features is None):
            raise ValueError("source_features and target_features must be supplied together")

        num_nodes = node_features.shape[0]
        source = edge_index[:, 0]
        target = edge_index[:, 1]

        projected = self._project_node_features(
            edge_features.shape[-1], source_features, target_features
        )

        weights, edge_density = self._radial_heads(edge_features, source, target, projected)
        if cutoff is not None:
            weights = weights * cutoff
            if edge_density is not None:
                edge_density = edge_density * cutoff
        message = self.conv_tp(
            node_features,
            edge_angular,
            weights,
            indices_1=source,
            indices_out=target,
            size_out=num_nodes,
        )
        if edge_density is None:
            return message, None
        return message, _scatter_sum(edge_density, target, num_nodes)


class RadialMLP(nn.Module):
    """Radial MLP with ``Linear -> LayerNorm -> SiLU`` blocks (ESEN/FairChem style).

    Faithful to ``mace.modules.radial.RadialMLP``: given ``channels_list``
    ``[c0, c1, ..., cN]`` it builds, for each ``ci`` (i>=1), a ``Linear`` and —
    for every layer except the last — a ``LayerNorm`` and ``SiLU``. The final
    layer is a bare ``Linear``. Layers live under ``self.net`` so the official
    weights (``...net.0``, ``...net.1``, ``...net.3`` …) transfer by direct copy.

    Used by the nonlinear residual interaction for both the tensor-product
    weight generator and the edge-density MLP.

    Reference:
        Batatia et al. NeurIPS 2022; FairChem/ESEN radial MLP.
        https://arxiv.org/abs/2206.07697
    """

    def __init__(self, channels_list: list[int], *, dtype: torch.dtype | None = None) -> None:
        """Initialize the radial MLP.

        Args:
            channels_list: Channel sizes ``[in, hidden..., out]`` (>= 2 entries).
            dtype: Floating dtype of every layer (the owning module's).
        """
        super().__init__()
        if len(channels_list) < 2:
            raise ValueError("channels_list must have at least [in, out].")
        modules: list[nn.Module] = []
        in_channels = channels_list[0]
        n = len(channels_list)
        ftype = (
            config.ftype if dtype is None else dtype
        )  # transitional: goes with molix.config (2c)
        for idx, out_channels in enumerate(channels_list[1:], start=1):
            modules.append(nn.Linear(in_channels, out_channels, bias=True, dtype=ftype))
            in_channels = out_channels
            if idx < n - 1:
                modules.append(nn.LayerNorm(out_channels, dtype=ftype))
                modules.append(nn.SiLU())
        self.net = nn.Sequential(*modules)
        self.hs = list(channels_list)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the radial MLP.

        Args:
            inputs: Edge features ``(n_edges, channels_list[0])``.

        Returns:
            ``(n_edges, channels_list[-1])``.
        """
        return self.net(inputs)

    def first_layer_parts(self, widths: Sequence[int]) -> tuple[torch.Tensor, ...]:
        """The first ``Linear``'s weight, split by a partition of its inputs.

        A caller that would otherwise concatenate several per-edge blocks into
        one wide input can instead multiply each block by its own slice and sum
        — the same value, without ever building the concatenation. The bias is
        :meth:`first_layer_bias` and must be added exactly once.

        Args:
            widths: Input-channel widths, summing to ``channels_list[0]``.

        Returns:
            One ``(width_i, channels_list[1])`` weight per part, transposed
            from ``nn.Linear``'s ``(out, in)`` so ``part_i @ w_i`` is that
            part's contribution.
        """
        if sum(widths) != self.hs[0]:
            raise ValueError(f"widths {list(widths)} do not partition {self.hs[0]} inputs")
        weight = self.net[0].weight
        return tuple(chunk.t() for chunk in weight.split(list(widths), dim=1))

    def first_layer_bias(self) -> torch.Tensor | None:
        """The first ``Linear``'s bias, to be added once to the split sum."""
        return self.net[0].bias

    def forward_from_first(self, hidden: torch.Tensor) -> torch.Tensor:
        """Continue after the first ``Linear``, given its output.

        Args:
            hidden: First-layer output ``(n_edges, channels_list[1])``.

        Returns:
            ``(n_edges, channels_list[-1])`` — identical to :meth:`forward` on
            the input that produced ``hidden``.
        """
        for module in list(self.net)[1:]:
            hidden = module(hidden)
        return hidden


def _irreps(spec) -> cue.Irreps:
    return spec if isinstance(spec, cue.Irreps) else cue.Irreps("O3", spec)


class GatedNonlinearity(torch.nn.Module):
    """Parameter-free gated equivariant nonlinearity (MACE's gate numerics).

    Args:
        irreps_scalars: Scalar (l=0) irreps activated directly.
        act_scalars: One activation per scalar group (e.g. ``[silu]``).
        irreps_gates: Scalar (l=0) irreps used as gates; ``num_irreps`` must
            equal that of ``irreps_gated``.
        act_gates: One activation per gate group (e.g. ``[sigmoid]``).
        irreps_gated: l>0 irreps gated by ``irreps_gates``.
        layout: ``"ir_mul"`` (cuequivariance) or ``"mul_ir"`` (multiplicity-major).
    """

    def __init__(
        self,
        irreps_scalars,
        act_scalars: Sequence[Callable | None],
        irreps_gates,
        act_gates: Sequence[Callable | None],
        irreps_gated,
        layout: str = "ir_mul",
    ) -> None:
        super().__init__()
        irreps_scalars = _irreps(irreps_scalars)
        irreps_gates = _irreps(irreps_gates)
        irreps_gated = _irreps(irreps_gated)

        def _lmax(irr: cue.Irreps) -> int:
            return max((mi.ir.l for mi in irr), default=0)

        if _lmax(irreps_gates) > 0:
            raise ValueError(f"gates must be scalars, got {irreps_gates}")
        if _lmax(irreps_scalars) > 0:
            raise ValueError(f"scalars must be scalars, got {irreps_scalars}")
        if irreps_gates.num_irreps != irreps_gated.num_irreps:
            raise ValueError(
                f"irreps_gated has {irreps_gated.num_irreps} irreps but "
                f"irreps_gates has {irreps_gates.num_irreps}"
            )
        if layout not in ("mul_ir", "ir_mul"):
            raise ValueError(f"layout must be 'mul_ir' or 'ir_mul', got '{layout}'")
        self._layout_is_ir_mul = layout == "ir_mul"

        scalars = list(irreps_scalars)
        gates = list(irreps_gates)
        gated = list(irreps_gated)
        unsorted = scalars + gates + gated

        # sort the concatenated irreps (stable), as MACE does
        sort = cue.Irreps("O3", [mi for mi in unsorted]).sort()
        sorted_mis = list(sort.irreps)
        perm = list(sort.perm)  # sorted[i] == unsorted[perm[i]]
        self._irreps_in = sort.irreps.simplify()

        n_scalar = len(scalars)
        n_gate = len(gates)

        scalar_sorted, gate_sorted, gated_sorted = [], [], []
        for si in range(len(sorted_mis)):
            orig = int(perm[si])
            if orig < n_scalar:
                scalar_sorted.append(si)
            elif orig < n_scalar + n_gate:
                gate_sorted.append(si)
            else:
                gated_sorted.append(si)

        def _d(mi) -> int:
            return mi.mul * mi.ir.dim

        offsets, off = [], 0
        for mi in sorted_mis:
            offsets.append(off)
            off += _d(mi)

        if scalar_sorted:
            self._s_start = offsets[scalar_sorted[0]]
            self._s_len = sum(_d(sorted_mis[i]) for i in scalar_sorted)
        else:
            self._s_start, self._s_len = 0, 0
        if gate_sorted:
            self._g_start = offsets[gate_sorted[0]]
            self._g_len = sum(_d(sorted_mis[i]) for i in gate_sorted)
        else:
            self._g_start, self._g_len = 0, 0

        gate_off_by_gated, cum = [], 0
        for mi in gated:
            gate_off_by_gated.append(cum)
            cum += mi.mul

        gated_info = []
        for si in gated_sorted:
            mi = sorted_mis[si]
            gated_orig = int(perm[si]) - n_scalar - n_gate
            gated_info.append(
                (offsets[si], _d(mi), mi.ir.dim, mi.mul, gate_off_by_gated[gated_orig])
            )
        self._gated_info = gated_info

        acts = list(act_scalars)
        if len(acts) == 1 and len(scalars) > 1:
            acts = acts * len(scalars)
        self._act_scalar = acts[0] if acts and acts[0] is not None else None
        self._scalar_cst = normalize2mom(self._act_scalar) if self._act_scalar is not None else 1.0

        actg = list(act_gates)
        if len(actg) == 1 and len(gates) > 1:
            actg = actg * len(gates)
        self._act_gate = actg[0] if actg and actg[0] is not None else None
        self._gate_cst = normalize2mom(self._act_gate) if self._act_gate is not None else 1.0

    @property
    def irreps_in(self) -> cue.Irreps:
        """Sorted+simplified input irreps the block expects."""
        return self._irreps_in

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply the gated nonlinearity.

        Args:
            features: ``(..., irreps_in.dim)`` in the configured layout.

        Returns:
            ``(..., irreps_out.dim)`` with scalars activated and l>0 irreps gated.
        """
        scalars = features.narrow(-1, self._s_start, self._s_len)
        if self._act_scalar is not None:
            scalars = self._act_scalar(scalars) * self._scalar_cst

        if not self._gated_info:
            return scalars

        gates = features.narrow(-1, self._g_start, self._g_len)
        if self._act_gate is not None:
            gates = self._act_gate(gates) * self._gate_cst

        ir_mul = self._layout_is_ir_mul
        batch_shape = features.shape[:-1]
        parts = [scalars]
        for gd_start, gd_len, ir_dim, mul, g_off in self._gated_info:
            gated_chunk = features.narrow(-1, gd_start, gd_len)
            gate_chunk = gates.narrow(-1, g_off, mul)
            if ir_mul:
                gated_3d = gated_chunk.reshape(*batch_shape, ir_dim, mul)
                result = (gated_3d * gate_chunk.unsqueeze(-2)).reshape(*batch_shape, ir_dim * mul)
            else:
                gated_3d = gated_chunk.reshape(*batch_shape, mul, ir_dim)
                result = (gated_3d * gate_chunk.unsqueeze(-1)).reshape(*batch_shape, mul * ir_dim)
            parts.append(result)
        return torch.cat(parts, dim=-1)


class ElementIndex(NamedTuple):
    """The element of every atom, in the forms the per-element maps consume.

    Built once per energy call (``MACEEncoder.element_index``) and handed down
    the layer stack. Every field is a fixed-shape tensor: no host sync, no
    data-dependent size.

    Attributes:
        index: Element row per atom ``(N,)`` int64 in ``[0, C)``.
        index32: ``index`` as int32, the dtype cuEquivariance's symmetric
            contraction indexes its weights with.
    """

    index: torch.Tensor
    index32: torch.Tensor


class OneHotLinear(nn.Module):
    """``y = Linear(one_hot(element))`` as a row lookup of the linear's weight.

    Args:
        irreps_in: The one-hot's irreps, ``Cx0e``.
        irreps_out: Output irreps, ``Fx0e``.
        dtype: Weight dtype.

    Raises:
        ValueError: If either side is not a single block of even scalars — a
            one-hot only ever selects a row, so anything else is not this map.
    """

    def __init__(
        self, irreps_in: cue.Irreps, irreps_out: cue.Irreps, *, dtype: torch.dtype
    ) -> None:
        super().__init__()
        if str(irreps_in) != f"{irreps_in.dim}x0e":
            raise ValueError(f"OneHotLinear input must be one-hot scalars Cx0e, got {irreps_in}")
        if str(irreps_out) != f"{irreps_out.dim}x0e":
            raise ValueError(f"OneHotLinear output must be scalars Fx0e, got {irreps_out}")
        polynomial = cue.descriptors.linear(irreps_in, irreps_out).polynomial
        ((_, product),) = polynomial.operations
        (weight_operand, _) = polynomial.inputs
        (path,) = product.paths
        self.num_elements = irreps_in.dim
        self.out_features = irreps_out.dim
        self.coefficient = float(np.asarray(path.coefficients).reshape(-1)[0])
        self.weight = nn.Parameter(torch.randn(1, weight_operand.size, dtype=dtype))

    def forward(self, element: torch.Tensor) -> torch.Tensor:
        """Look up the rows.

        Args:
            element: Element index ``(N,)`` int64 in ``[0, C)``.

        Returns:
            ``(N, F)`` — the linear's output on ``one_hot(element)``.
        """
        rows = self.weight.view(self.num_elements, self.out_features)
        return F.embedding(element, rows) * self.coefficient


class _Path(NamedTuple):
    """One input irrep block to the output block of the same irrep."""

    x_start: int
    out_index: int
    ir_dim: int
    u: int
    w: int
    weight_start: int
    coefficient: float


class ElementLinear(nn.Module):
    """``y = skip_tp(x, one_hot(element))``, evaluated as a per-element linear.

    Args:
        irreps_in: Input irreps (``ir_mul`` layout).
        attrs_irreps: The one-hot's irreps, ``Cx0e``.
        irreps_out: Output irreps (``ir_mul`` layout).
        dtype: Weight dtype.

    Raises:
        ValueError: If ``attrs_irreps`` is not ``C`` scalars, or a path of the
            fully connected product is not the identity in its irrep
            components — either way the product is not a per-element linear.
    """

    def __init__(
        self,
        irreps_in: cue.Irreps,
        attrs_irreps: cue.Irreps,
        irreps_out: cue.Irreps,
        *,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if str(attrs_irreps) != f"{attrs_irreps.dim}x0e":
            raise ValueError(f"skip_tp attributes must be one-hot scalars Cx0e, got {attrs_irreps}")
        num_elements = attrs_irreps.dim
        polynomial = cue.descriptors.fully_connected_tensor_product(
            irreps_in, attrs_irreps, irreps_out
        ).polynomial
        ((_, product),) = polynomial.operations
        (
            weight_operand,
            x_operand,
            _,
        ) = polynomial.inputs
        (out_operand,) = polynomial.outputs
        weight_starts = np.cumsum([0] + [int(np.prod(s)) for s in weight_operand.segments])
        x_starts = np.cumsum([0] + [int(np.prod(s)) for s in x_operand.segments])
        paths = []
        for path in product.paths:
            weight_index, x_index, _, out_index = path.indices
            coefficients = np.asarray(path.coefficients)[:, 0, :]
            scale = float(coefficients[0, 0])
            if not np.allclose(coefficients, scale * np.eye(coefficients.shape[0])):
                raise ValueError("a skip_tp path is not a per-element linear map")
            u, _, w = weight_operand.segments[weight_index]
            paths.append(
                _Path(
                    x_start=int(x_starts[x_index]),
                    out_index=out_index,
                    ir_dim=coefficients.shape[0],
                    u=u,
                    w=w,
                    weight_start=int(weight_starts[weight_index]),
                    coefficient=scale,
                )
            )
        self.num_elements = num_elements
        self._paths = tuple(paths)
        self._out_segments = tuple(int(np.prod(s)) for s in out_operand.segments)
        self.weight = nn.Parameter(torch.randn(1, weight_operand.size, dtype=dtype))

    def forward(self, x: torch.Tensor, elements: ElementIndex) -> torch.Tensor:
        """Apply the map.

        Args:
            x: Features ``(N, irreps_in.dim)``, ``ir_mul``.
            elements: The atoms' elements over this map's ``C`` rows.

        Returns:
            ``(N, irreps_out.dim)``, ``ir_mul``.
        """
        n = x.shape[0]
        outputs: dict[int, torch.Tensor] = {}
        for path in self._paths:
            d = path.ir_dim
            a = x[:, path.x_start : path.x_start + d * path.u].reshape(n, d, path.u)
            weights = self._blocks(path).index_select(1, elements.index)  # (u, N, w)
            y = path.coefficient * torch.einsum("ndu,unw->ndw", a, weights)
            outputs[path.out_index] = y.reshape(n, d * path.w) + outputs.get(path.out_index, 0)
        return self._assemble(outputs, x)

    def _blocks(self, path: _Path) -> torch.Tensor:
        size = path.u * self.num_elements * path.w
        block = self.weight[0, path.weight_start : path.weight_start + size]
        return block.view(path.u, self.num_elements, path.w)

    def _assemble(self, outputs: dict[int, torch.Tensor], like: torch.Tensor) -> torch.Tensor:
        n = like.shape[0]
        parts = [
            outputs[k] if k in outputs else like.new_zeros(n, size)
            for k, size in enumerate(self._out_segments)
        ]
        return torch.cat(parts, dim=1)


class _ChannelwiseInteractionBase(_CueqChannelwiseBlockMixin, nn.Module):
    """Shared wiring of the four interaction blocks.

    Subclasses choose the normalisation (:meth:`_normalize`, plus the density
    head when there is one) and, through the skip mixins, ``skip_tp``'s irreps
    and placement.

    Args:
        node_attrs_irreps: One-hot atomic-number irreps, e.g. ``"89x0e"``.
        node_feats_irreps: Incoming node feature irreps.
        edge_attrs_irreps: Spherical-harmonics irreps, e.g. ``"1x0e+1x1o+1x2e+1x3o"``.
        edge_feats_irreps: Radial basis irreps, e.g. ``"10x0e"``.
        edge_irreps: ``linear_up`` output irreps.
        target_irreps: Tensor-product / block output irreps.
        radial_mlp: Hidden widths of the radial weight MLP, e.g. ``[64, 64, 64]``.
        dtype: Floating dtype every parameter is built at (the owning module's).
    """

    def __init__(
        self,
        *,
        node_attrs_irreps: str,
        node_feats_irreps: str,
        edge_attrs_irreps: str,
        edge_feats_irreps: str,
        edge_irreps: str,
        target_irreps: str,
        radial_mlp: list[int],
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        ftype = (
            config.ftype if dtype is None else dtype
        )  # transitional: goes with molix.config (2c)
        #: The build dtype, read by the skip mixins and the density head.
        self.dtype = ftype

        node_feats = cue.Irreps("O3", node_feats_irreps)
        edge_attrs = cue.Irreps("O3", edge_attrs_irreps)
        edge_feats = cue.Irreps("O3", edge_feats_irreps)
        edge_ir = cue.Irreps("O3", edge_irreps)
        target = cue.Irreps("O3", target_irreps)
        self.node_attrs_irreps = cue.Irreps("O3", node_attrs_irreps)
        self.irreps_out = target
        self._skip_tp_in = node_feats

        self.linear_up = cueq.Linear(node_feats, edge_ir, layout=cue.ir_mul, dtype=ftype)

        self.conv_tp = cueq.ChannelWiseTensorProduct(
            edge_ir,
            edge_attrs,
            target,
            layout=cue.ir_mul,
            shared_weights=False,
            internal_weights=False,
            dtype=ftype,
        )

        self._num_radial = edge_feats.dim
        self.conv_tp_weights = MomentNormalizedMLP(
            [self._num_radial] + list(radial_mlp) + [self.conv_tp.weight_numel], dtype=ftype
        )
        # Built right after the weight MLP: parameter creation order is what a
        # seeded initialisation reproduces.
        self.density_fn = self._build_density_head()

        self.linear = cueq.Linear(self.conv_tp.irreps_out, target, layout=cue.ir_mul, dtype=ftype)

    def _build_density_head(self) -> nn.Module | None:
        """The learned edge-density MLP, or ``None`` for a constant normalisation."""
        raise NotImplementedError

    def _normalize(self, message: torch.Tensor, density: torch.Tensor | None) -> torch.Tensor:
        """Normalise the linearly mixed aggregated message."""
        raise NotImplementedError

    def _message(
        self,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: torch.Tensor | None,
    ) -> torch.Tensor:
        """Normalised aggregated message ``(N, target.dim)``."""
        node_up = self.linear_up(node_feats)
        message, density = self._cueq_channelwise_conv(
            node_up,
            edge_attrs,
            edge_feats,
            edge_index,
            cutoff,
        )
        return self._normalize(self.linear(message), density)


class _DensityNormalised(_ChannelwiseInteractionBase):
    """``message / (ρ + 1)`` with the learned edge density ``density_fn``."""

    def _build_density_head(self) -> nn.Module:
        return MomentNormalizedMLP([self._num_radial, 1], dtype=self.dtype)

    def _normalize(self, message: torch.Tensor, density: torch.Tensor | None) -> torch.Tensor:
        return message / (density + 1.0)


class _AverageNormalised(_ChannelwiseInteractionBase):
    """``message / avg_num_neighbors`` — a dataset constant, no density head.

    Args:
        avg_num_neighbors: Mean neighbour count of the training data.
    """

    def __init__(self, *, avg_num_neighbors: float, **kwargs) -> None:
        self.avg_num_neighbors = float(avg_num_neighbors)
        super().__init__(**kwargs)

    def _build_density_head(self) -> None:
        return None

    def _normalize(self, message: torch.Tensor, density: torch.Tensor | None) -> torch.Tensor:
        return message / self.avg_num_neighbors


class _FirstLayerSkip:
    """``skip_tp`` on the outgoing message; no residual to pass on."""

    irreps_out: cue.Irreps
    node_attrs_irreps: cue.Irreps

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # message ⊗ one_hot(Z) → message: element-selective output mixing.
        self.skip_tp = ElementLinear(
            self.irreps_out, self.node_attrs_irreps, self.irreps_out, dtype=self.dtype
        )

    def forward(
        self,
        elements: ElementIndex,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        """Run one first-layer interaction.

        Args:
            elements: The atoms' elements (:class:`ElementIndex`).
            node_feats: Node features ``(N, node_feats_irreps.dim)``.
            edge_attrs: Spherical harmonics ``(E, edge_attrs_irreps.dim)``.
            edge_feats: Radial basis features ``(E, num_radial)``.
            edge_index: ``(E, 2)`` with ``[:, 0]`` = source, ``[:, 1]`` = target
                (the repo-wide edge convention).
            cutoff: Optional per-edge cutoff envelope ``(E, 1)``.

        Returns:
            ``(message (N, target_irreps.dim), None)``.
        """
        message = self._message(node_feats, edge_attrs, edge_feats, edge_index, cutoff)
        return self.skip_tp(message, elements), None


class _ResidualSkip:
    """``skip_tp`` maps the incoming node features onto ``hidden_irreps``: the skip.

    Args:
        hidden_irreps: ``skip_tp`` output irreps (consumed by the product block).
    """

    node_attrs_irreps: cue.Irreps
    _skip_tp_in: cue.Irreps

    def __init__(self, *, hidden_irreps: str, **kwargs) -> None:
        super().__init__(**kwargs)
        # node_feats ⊗ one_hot(Z) → hidden: the residual the product adds back.
        self.skip_tp = ElementLinear(
            self._skip_tp_in,
            self.node_attrs_irreps,
            cue.Irreps("O3", hidden_irreps),
            dtype=self.dtype,
        )

    def forward(
        self,
        elements: ElementIndex,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one residual interaction.

        Args:
            elements: The atoms' elements (:class:`ElementIndex`).
            node_feats: Node features ``(N, node_feats_irreps.dim)``.
            edge_attrs: Spherical harmonics ``(E, edge_attrs_irreps.dim)``.
            edge_feats: Radial basis features ``(E, num_radial)``.
            edge_index: ``(E, 2)`` with ``[:, 0]`` = source, ``[:, 1]`` = target
                (the repo-wide edge convention).
            cutoff: Optional per-edge cutoff envelope ``(E, 1)``.

        Returns:
            ``(message (N, target_irreps.dim), skip (N, hidden_irreps.dim))``.
        """
        sc = self.skip_tp(node_feats, elements)
        message = self._message(node_feats, edge_attrs, edge_feats, edge_index, cutoff)
        return message, sc


class DensityInteraction(_FirstLayerSkip, _DensityNormalised):
    """First-layer density-normalised interaction (MACE ``RealAgnosticDensity``).

    Args:
        See :class:`_ChannelwiseInteractionBase`.
    """


class DensityResidualInteraction(_ResidualSkip, _DensityNormalised):
    """Residual density-normalised interaction (MACE ``RealAgnosticDensityResidual``).

    Args:
        hidden_irreps: ``skip_tp`` output irreps (consumed by the product block).
        Other arguments: see :class:`_ChannelwiseInteractionBase`.
    """


class AgnosticInteraction(_FirstLayerSkip, _AverageNormalised):
    """First-layer interaction normalised by ``avg_num_neighbors`` (MACE ``RealAgnostic``).

    Args:
        avg_num_neighbors: Mean neighbour count of the training data.
        Other arguments: see :class:`_ChannelwiseInteractionBase`.
    """


class AgnosticResidualInteraction(_ResidualSkip, _AverageNormalised):
    """Residual interaction normalised by ``avg_num_neighbors`` (MACE ``RealAgnosticResidual``).

    Args:
        avg_num_neighbors: Mean neighbour count of the training data.
        hidden_irreps: ``skip_tp`` output irreps (consumed by the product block).
        Other arguments: see :class:`_ChannelwiseInteractionBase`.
    """


def _partition_gate(irreps_out: cue.Irreps) -> tuple[cue.Irreps, cue.Irreps, cue.Irreps]:
    """Split output irreps into (scalars, gates, gated) for the gated nonlinearity."""
    scalars, gated = [], []
    for mi in irreps_out:
        (scalars if mi.ir.l == 0 else gated).append(mi)
    gates = [(mi.mul, "0e") for mi in gated]
    return (
        cue.Irreps("O3", scalars),
        cue.Irreps("O3", " + ".join(f"{m}x{ir}" for m, ir in gates) if gates else ""),
        cue.Irreps("O3", gated),
    )


class ResidualInteraction(_CueqChannelwiseBlockMixin, nn.Module):
    """MACE residual non-linear interaction layer (cuEquivariance).

    Args:
        node_attrs_irreps: One-hot atomic-number irreps, e.g. ``"83x0e"``.
        node_feats_irreps: Incoming node feature irreps.
        edge_attrs_irreps: Spherical-harmonics irreps, e.g. ``"1x0e+1x1o+1x2e+1x3o"``.
        edge_feats_irreps: Radial basis irreps, e.g. ``"8x0e"`` (scalar count = num_bessel).
        edge_irreps: ``linear_up`` output irreps (per-layer reduced channels).
        target_irreps: Tensor-product / output irreps (also the gated nonlin output).
        hidden_irreps: ``skip_tp`` output irreps (consumed by the product block).
        radial_mlp: Hidden sizes of the radial MLPs, e.g. ``[128, 128, 128]``.
        dtype: Floating dtype every parameter is built at (the owning module's).
    """

    def __init__(
        self,
        *,
        node_attrs_irreps: str,
        node_feats_irreps: str,
        edge_attrs_irreps: str,
        edge_feats_irreps: str,
        edge_irreps: str,
        target_irreps: str,
        hidden_irreps: str,
        radial_mlp: list[int],
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        ftype = (
            config.ftype if dtype is None else dtype
        )  # transitional: goes with molix.config (2c)

        node_feats = cue.Irreps("O3", node_feats_irreps)
        edge_attrs = cue.Irreps("O3", edge_attrs_irreps)
        edge_feats = cue.Irreps("O3", edge_feats_irreps)
        edge_ir = cue.Irreps("O3", edge_irreps)
        target = cue.Irreps("O3", target_irreps)
        hidden = cue.Irreps("O3", hidden_irreps)
        node_attrs = cue.Irreps("O3", node_attrs_irreps)

        n_scalar = sum(mi.mul for mi in node_feats if mi.ir.l == 0)
        node_scalar = cue.Irreps("O3", f"{n_scalar}x0e")

        def lin(i_in, i_out):
            return cueq.Linear(i_in, i_out, layout=cue.ir_mul, dtype=ftype)

        self.source_embedding = OneHotLinear(node_attrs, node_scalar, dtype=ftype)
        self.target_embedding = OneHotLinear(node_attrs, node_scalar, dtype=ftype)
        self.linear_up = lin(node_feats, edge_ir)

        self.conv_tp = cueq.ChannelWiseTensorProduct(
            edge_ir,
            edge_attrs,
            target,
            layout=cue.ir_mul,
            shared_weights=False,
            internal_weights=False,
            dtype=ftype,
        )
        irreps_mid = self.conv_tp.irreps_out

        input_dim = edge_feats.dim + 2 * node_scalar.dim
        self.conv_tp_weights = RadialMLP(
            [input_dim] + list(radial_mlp) + [self.conv_tp.weight_numel], dtype=ftype
        )

        self.skip_tp = lin(node_feats, hidden)

        irreps_scalars, irreps_gates, irreps_gated = _partition_gate(target)
        self.equivariant_nonlin = GatedNonlinearity(
            irreps_scalars,
            [F.silu],
            irreps_gates,
            [torch.sigmoid] * len(list(irreps_gated)),
            irreps_gated,
            layout="ir_mul",
        )
        irreps_nonlin = self.equivariant_nonlin.irreps_in

        self.linear_res = lin(edge_ir, irreps_nonlin)
        self.linear_1 = lin(irreps_mid, irreps_nonlin)
        self.linear_2 = lin(target, target)

        self.density_fn = RadialMLP([input_dim, 64, 1], dtype=ftype)
        self.alpha = nn.Parameter(torch.tensor(20.0, dtype=ftype))
        self.beta = nn.Parameter(torch.tensor(0.0, dtype=ftype))

    def forward(
        self,
        elements: ElementIndex,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one residual non-linear interaction layer.

        Args:
            elements: The atoms' elements (:class:`ElementIndex`).
            node_feats: Node features ``(N, node_feats_irreps.dim)``.
            edge_attrs: Spherical harmonics ``(E, edge_attrs_irreps.dim)``.
            edge_feats: Radial basis features ``(E, num_bessel)``.
            edge_index: ``(E, 2)`` with ``[:, 0]`` = source, ``[:, 1]`` = target
                (the repo-wide edge convention).
            cutoff: Optional per-edge cutoff envelope ``(E, 1)``.

        Returns:
            ``(message (N, target_irreps.dim), skip (N, hidden_dim))``.
        """
        sc = self.skip_tp(node_feats)
        node_up = self.linear_up(node_feats)
        node_feats_res = self.linear_res(node_up)

        source_emb = self.source_embedding(elements.index)
        target_emb = self.target_embedding(elements.index)
        message, density = self._cueq_channelwise_conv(
            node_up,
            edge_attrs,
            edge_feats,
            edge_index,
            cutoff,
            source_features=source_emb,
            target_features=target_emb,
        )
        message = self.linear_1(message) / (density * self.beta + self.alpha)
        message = message + node_feats_res
        message = self.equivariant_nonlin(message)
        message = self.linear_2(message)
        return message, sc


class EquivariantProductBasis(nn.Module):
    """Symmetric-contraction product basis + linear (MACE product block).

    Args:
        node_feats_irreps: Incoming node feature irreps (interaction output),
            e.g. ``"1024x0e+1024x1o+1024x2e+1024x3o"``.
        target_irreps: Output irreps, e.g. ``"1024x0e+1024x1o+1024x2e"``.
        correlation: Body-order correlation (symmetric-contraction degree).
        num_elements: Element table size; ``1`` for an element-agnostic product
            (OMOL), in which case all atoms share one weight set.
        use_sc: Whether to add the skip connection ``sc`` from the interaction.
        group: The O(3) group class whose Clebsch-Gordan phases define the
            contraction basis — ``cue.O3``, or a checkpoint's own convention
            (``molzoo.mace.coupling.MaceO3``). Weights are only meaningful in
            the convention they were trained in.
        dtype: Floating dtype every parameter is built at (the owning module's).
    """

    def __init__(
        self,
        *,
        node_feats_irreps: str,
        target_irreps: str,
        correlation: int,
        num_elements: int = 1,
        use_sc: bool = True,
        group: type = cue.O3,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        ftype = (
            config.ftype if dtype is None else dtype
        )  # transitional: goes with molix.config (2c)
        self.use_sc = use_sc
        self.num_elements = int(num_elements)
        #: Element-agnostic product: every atom reads weight row 0.
        self.shared = self.num_elements == 1

        irreps_in = cue.Irreps(group, node_feats_irreps)
        irreps_out = cue.Irreps(group, target_irreps)

        self.symmetric_contractions = cueq.SymmetricContraction(
            irreps_in,
            irreps_out,
            contraction_degree=correlation,
            num_elements=self.num_elements,
            layout_in=cue.ir_mul,
            layout_out=cue.ir_mul,
            original_mace=True,
            dtype=ftype,
            math_dtype=ftype,
        )
        # A linear map acts per irrep; it has no coupling to take a phase from.
        self.linear = cueq.Linear(
            cue.Irreps("O3", target_irreps),
            cue.Irreps("O3", target_irreps),
            layout=cue.ir_mul,
            dtype=ftype,
        )

    def forward(
        self,
        node_feats: torch.Tensor,
        sc: torch.Tensor | None,
        elements: ElementIndex,
    ) -> torch.Tensor:
        """Raise body order and project.

        Args:
            node_feats: ``(N, node_feats_irreps.dim)`` interaction message
                (``cue.ir_mul``).
            sc: Skip connection ``(N, target_irreps.dim)`` or ``None``.
            elements: The atoms' elements (:class:`ElementIndex`); only read
                by the per-element product.

        Returns:
            ``(N, target_irreps.dim)`` node features.
        """
        # ``shared`` is a plain bool fixed in ``__init__``: dynamo folds the
        # branch, so either way this is one indexed kernel.
        index = torch.zeros_like(elements.index32) if self.shared else elements.index32
        out = self.symmetric_contractions(node_feats, index)
        out = self.linear(out)
        if self.use_sc and sc is not None:
            out = out + sc
        return out
