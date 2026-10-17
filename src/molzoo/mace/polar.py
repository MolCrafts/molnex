"""MACE-POLAR energy (local + Fukui + field + Hartree), isolated or periodic.

:class:`MacePolar` is the polarisable :class:`~molzoo.mace.potential.MacePotential`.
Per atom and per graph, in eV::

    E_i = E0(Z_i) + scale·ε_i + shift + E_electron,i        (atoms.energy)
    E_g = Σ_{i∈g} E_i + E_Hartree,g                          (graphs.energy)

The Hartree energy of the equilibrated GTO density is a graph-level term with
no per-atom attribution, added after the per-atom scatter. Isolated molecules
take the real-space electrostatics; a batch carrying ``graphs.cell`` switches
the two GTO (Gaussian-type-orbital) submodules to the official periodic
reciprocal-space sum, which needs one eager :meth:`MacePolar.bind_cell`
before compiling or exporting (see that method).

``total_spin`` is the spin multiplicity ``M = 2S+1`` (``S`` = total spin, so a
closed-shell singlet is ``M = 1``) — a multiplicity, not a count of unpaired
electrons.

The representation blocks the product is assembled from live here too, above
the product, in native ``cue.ir_mul`` (no energy; GTO electrostatics are
:mod:`molpot.potentials.gto`). Submodule names match the official remap.

- :class:`SparseUvuTensorProduct` — the sparse uvu product of the field path,
  matching ACEsuit ``SparseUvuTensorProduct``: only ``ℓ × ℓ → 0`` (invariant
  contraction) and ``ℓ × 0 → ℓ`` (scalar modulation). Each path is divided by
  ``√(2ℓ+1)`` and then scaled by the upstream ``path_normalization='element'``
  coefficient so a Polar-1-M overlay matches the official tensor product. No
  runtime ``mul_ir`` transpose.
- :class:`MultiLayerFeatureMixer` — sum of per-layer equivariant linears
  (official polar-1-M: two layers of ``512x0e+512x1o``, dim 2048).
- :class:`EnvironmentDependentSpinSourceBlock` — a single ``cuet.Linear`` from
  node features onto two concatenated copies of ``Y_{l_max}``; Polar-1-M uses
  ``l_max=1`` so the flat output is ``(N, 8)`` and ``view (N, 2, 4)`` is one
  spin channel of monopole + dipole. The ``cuet.Linear`` is registered bare,
  never inside a wrapper module — a wrapper would prefix the checkpoint key as
  ``linear.linear.weight``.
- :class:`AgnosticEmbeddedOneBodyVariableUpdate` /
  :class:`OneBodyMLPFieldReadout` — the one-body field update and the
  post-induction field readout.

Reference:
    Batatia et al. "MACE-POLAR-1: A Polarisable Electrostatic Foundation Model
    for Molecular Chemistry" arXiv:2602.19411.
    Batatia et al. "MACE: Higher Order Equivariant Message Passing Neural
    Networks for Fast and Accurate Force Fields" NeurIPS 2022.
    https://arxiv.org/abs/2206.07697
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Self

import cuequivariance as cue
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDictBase

from molix import config
from molix.F.scatter import scatter_sum_compile_safe
from molix.schema import ATOMIC_ENERGY_KEY, ENERGY_KEY
from molpot.potentials.gto.features import GTOFieldFeatures
from molpot.potentials.gto.fukui import FukuiEquilibration
from molpot.potentials.gto.hartree import GTOHartreeEnergy
from molrep import cueq
from molrep.embedding.geometry import edge_lengths, edge_vectors
from molrep.interaction.product import irreps_from_l_max, sh_irreps_from_l_max
from molzoo.mace.checkpoint import polar_keys
from molzoo.mace.heads import NonLinearBiasReadout
from molzoo.mace.layers import GatedNonlinearity, RadialMLP
from molzoo.mace.potential import MacePotential, Wrt
from molzoo.mace.spec import MACEPolarSpec


def _slices(irreps: cue.Irreps) -> list[tuple[int, int, int, int, int, int]]:
    """``(start, stop, mul, l, p, dim)`` per irrep in ``ir_mul`` storage order."""
    out: list[tuple[int, int, int, int, int, int]] = []
    start = 0
    for mi in irreps:
        dim = mi.ir.dim
        stop = start + mi.mul * dim
        out.append((start, stop, mi.mul, mi.ir.l, int(mi.ir.p), dim))
        start = stop
    return out


class SparseUvuTensorProduct(nn.Module):
    """Shared-weight uvu product on flattened ``(N, irreps.dim)`` features.

    Args:
        irreps_in1: First operand, e.g. ``"8x0e+8x1o"``.
        irreps_in2: Second operand.
        irreps_out: Output irreps. For Polar dots this is ``n_0e x 0e``;
            for ``tp_out`` it matches ``irreps_in1``.
    """

    def __init__(
        self, irreps_in1: str, irreps_in2: str, irreps_out: str, *, dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        dtype = (
            config.ftype if dtype is None else dtype
        )  # transitional: goes with molix.config (2c)
        in1 = cue.Irreps("O3", irreps_in1)
        in2 = cue.Irreps("O3", irreps_in2)
        out = cue.Irreps("O3", irreps_out)
        self.irreps_in1 = in1
        self.irreps_in2 = in2
        self.irreps_out = out
        slices1 = _slices(in1)
        slices2 = _slices(in2)
        slices_out = _slices(out)
        if not slices_out:
            raise ValueError("irreps_out is empty")

        pending: list[tuple[int, int, int, int, int, int, int, int, int, int, int, int, int]] = []
        x_out: dict[tuple[int, int], float] = {}
        w_off = 0
        for s1, e1, mul1, l1, p1, d1 in slices1:
            for s2, e2, mul2, l2, p2, d2 in slices2:
                for so, eo, mulo, lo, po, do in slices_out:
                    if mulo != mul1:
                        continue
                    if do == 1 and d1 == d2 and l1 == l2 and p1 * p2 == po:
                        mode = 0
                    elif d2 == 1 and do == d1 and lo == l1 and p1 * p2 == po:
                        mode = 1
                    else:
                        continue
                    w_size = mul1 * mul2
                    pending.append(
                        (s1, e1, s2, e2, so, eo, w_off, w_off + w_size, mode, mul1, mul2, d1, do)
                    )
                    x_out[(so, eo)] = x_out.get((so, eo), 0.0) + float(mul2)
                    w_off += w_size
        if not pending:
            raise NotImplementedError(
                "SparseUvuTensorProduct only supports (l×l→0) and (l×0→l) uvu paths "
                f"for {irreps_in1} ⊗ {irreps_in2} → {irreps_out}"
            )
        # upstream element path_normalization for uvu: x = Σ mul_in2 over paths
        # that share this output irrep; weight = √(dim_out / x).
        self._paths = [
            (
                s1,
                e1,
                s2,
                e2,
                so,
                eo,
                ws,
                we,
                mode,
                mul1,
                mul2,
                d1,
                math.sqrt(float(do) / x_out[(so, eo)]),
            )
            for s1, e1, s2, e2, so, eo, ws, we, mode, mul1, mul2, d1, do in pending
        ]
        self.weight = nn.Parameter(torch.randn(w_off, dtype=dtype))
        self.register_buffer("output_mask", torch.ones(out.dim, dtype=dtype))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Contract two flattened irreps tensors.

        Args:
            x1: ``(N, irreps_in1.dim)``.
            x2: ``(N, irreps_in2.dim)``.

        Returns:
            ``(N, irreps_out.dim)``.
        """
        if x1.ndim != 2 or x2.ndim != 2:
            raise ValueError(
                "SparseUvuTensorProduct expects 2-D tensors (N, irreps.dim), "
                f"got {tuple(x1.shape)} and {tuple(x2.shape)}"
            )
        batch = x1.shape[0]
        out = x1.new_zeros((batch, self.irreps_out.dim))
        weight = self.weight
        # ``einsum`` promoted a mixed-precision pair for us; ``F.linear`` and
        # ``matmul`` below refuse one. Promote once per call, not per path.
        common = torch.promote_types(x2.dtype, weight.dtype)
        if weight.dtype != common:
            weight = weight.to(common)
        if x2.dtype != common:
            x2 = x2.to(common)
        for s1, e1, s2, e2, so, eo, ws, we, mode, mul1, mul2, d1, path_weight in self._paths:
            scale = path_weight / math.sqrt(float(d1))
            w = weight[ws:we].reshape(mul1, mul2)
            if mode == 0:
                a = x1[:, s1:e1].reshape(batch, d1, mul1).transpose(1, 2)
                b = x2[:, s2:e2].reshape(batch, d1, mul2).transpose(1, 2)
                # Reassociate the same uvu contraction before materialising
                # it: sum_v b[b,v,d] * w[u,v] first, then dot with a over d.
                # The literal pair tensor is (N, mul1, mul2); Polar-1-M would
                # allocate (N, 512, 512), or 34.57 GiB at N=35400 in fp32.
                # F.linear uses the shared (u,v) weights as one GEMM and keeps
                # the transient at (N, d, mul1), where d <= 2*l_max+1.
                projected = F.linear(b.transpose(1, 2), w).transpose(1, 2)
                mixed = (a * projected).sum(dim=-1) * scale
                out[:, so:eo] = out[:, so:eo] + mixed
            else:
                a = x1[:, s1:e1].reshape(batch, d1, mul1).transpose(1, 2)
                scalars = x2[:, s2:e2].reshape(batch, mul2)
                mixed = scalars @ w.transpose(0, 1)
                contrib = (a * mixed.unsqueeze(-1) * scale).transpose(1, 2).reshape(batch, -1)
                out[:, so:eo] = out[:, so:eo] + contrib
        return out * self.output_mask


class MultiLayerFeatureMixer(nn.Module):
    """``tilde h = sum_t W^{(t)} h^{(t)}`` with one ``cuet.Linear`` per layer.

    Args:
        irreps: Node-feature irreps of every layer, e.g. ``"512x0e+512x1o"``.
        num_layers: Number of interaction layers ``T``. Must be >= 1.

    Forward:
        ``all_node_feats`` of shape ``(T, N, irreps.dim)``. ``T`` must equal
        ``num_layers``.
    """

    def __init__(self, *, irreps: str, num_layers: int, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        dtype = (
            config.ftype if dtype is None else dtype
        )  # transitional: goes with molix.config (2c)
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.num_layers = num_layers
        ir = cue.Irreps("O3", irreps)
        self.irreps_dim = ir.dim
        self.linears = nn.ModuleList(
            [cueq.Linear(ir, ir, layout=cue.ir_mul, dtype=dtype) for _ in range(num_layers)]
        )

    def forward(self, all_node_feats: torch.Tensor) -> torch.Tensor:
        """Mix stacked per-layer features.

        Args:
            all_node_feats: ``(T, N, irreps.dim)``.

        Returns:
            Mixed features ``(N, irreps.dim)``.

        Raises:
            ValueError: If ``T != num_layers``.
        """
        n_stack = all_node_feats.shape[0]
        if n_stack != self.num_layers:
            raise ValueError(
                f"all_node_feats has {n_stack} layers but mixer was built with "
                f"num_layers={self.num_layers}"
            )
        mixed = self.linears[0](all_node_feats[0])
        for i in range(1, self.num_layers):
            mixed = mixed + self.linears[i](all_node_feats[i])
        return mixed


class EnvironmentDependentSpinSourceBlock(nn.Module):
    """Linear map from hidden irreps to two spin channels of GTO multipoles.

    Args:
        irreps_in: Node-feature irreps, e.g. ``"512x0e+512x1o"``.
        l_max: Highest multipole order (polar-1-M dump: 1).
        zero_charges: If True, zero the monopole (0e) of each spin channel.

    Returns:
        ``(N, 2, (l_max+1)**2)`` — channel 0/1 are spin-up/down; last axis is
        one copy of ``Y_{l_max}`` (monopole then dipole for ``l_max=1``).
        Physical units of the coefficients are assigned by the GTO energy (e / e·Å).
    """

    def __init__(
        self,
        *,
        irreps_in: str,
        l_max: int,
        zero_charges: bool = False,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        dtype = (
            config.ftype if dtype is None else dtype
        )  # transitional: goes with molix.config (2c)
        if l_max < 0:
            raise ValueError(f"l_max must be >= 0, got {l_max}")
        self.l_max = l_max
        self.zero_charges = zero_charges
        self.y_dim = (l_max + 1) ** 2
        # Two concatenated Y copies, not simplify(2x0e+2x1o), so view (N, 2, y_dim)
        # lands monopoles at [:, 0, 0] and [:, 1, 0].
        y = sh_irreps_from_l_max(l_max).replace(" ", "")
        irreps_out = f"{y}+{y}"
        self.linear = cueq.Linear(
            cue.Irreps("O3", irreps_in),
            cue.Irreps("O3", irreps_out),
            layout=cue.ir_mul,
            dtype=dtype,
        )

    def forward(self, node_feats: torch.Tensor) -> torch.Tensor:
        """Map node features to spin-resolved multipole coefficients.

        Args:
            node_feats: ``(N, irreps_in.dim)`` in ``ir_mul``.

        Returns:
            ``(N, 2, (l_max+1)**2)``.
        """
        flat = self.linear(node_feats)
        sources = flat.view(flat.shape[0], 2, self.y_dim)
        if self.zero_charges:
            sources = sources.clone()
            sources[..., 0] = 0
        return sources


def _count_0e(irreps: cue.Irreps) -> int:
    return sum(mi.mul for mi in irreps if mi.ir.l == 0 and int(mi.ir.p) == 1)


def _lmax(irreps: cue.Irreps) -> int:
    return max((mi.ir.l for mi in irreps), default=0)


def _mul_sh(l_max: int, mul: int) -> str:
    y = cue.Irreps("O3", sh_irreps_from_l_max(l_max))
    return "+".join(f"{mul * mi.mul}x{mi.ir}" for mi in y)


def _join(mis: list[cue.MulIrrep]) -> str:
    return "+".join(f"{mi.mul}x{mi.ir}" for mi in mis) if mis else "0x0e"


class _LinearWithScalarBias(nn.Module):
    """Equivariant linear plus even-scalar (0e) bias, matching upstream ``biases=True``.

    Bias is applied to every 0e slice in irrep order so unsimplified
    ``2x0e+2x1o+2x0e`` keeps Fukui on the last two channels.
    """

    def __init__(
        self, irreps_in: str, irreps_out: str, *, dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        dtype = (
            config.ftype if dtype is None else dtype
        )  # transitional: goes with molix.config (2c)
        out = cue.Irreps("O3", irreps_out)
        self.linear = cueq.Linear(
            cue.Irreps("O3", irreps_in),
            out,
            layout=cue.ir_mul,
            dtype=dtype,
        )
        slices: list[tuple[int, int, int]] = []
        offset = 0
        n_0e = 0
        for mi in out:
            width = mi.mul * mi.ir.dim
            if mi.ir.l == 0 and int(mi.ir.p) == 1:
                slices.append((offset, offset + width, mi.mul))
                n_0e += mi.mul
            offset += width
        self._bias_slices = slices
        self.bias = nn.Parameter(torch.zeros(n_0e, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear(x)
        if not self._bias_slices:
            return y
        extra = y.new_zeros(y.shape)
        b_off = 0
        for start, stop, mul in self._bias_slices:
            extra[..., start:stop] = self.bias[b_off : b_off + mul]
            b_off += mul
        return y + extra


class GeneralNonLinearBiasReadout(nn.Module):
    """Gated readout: ``gate(linear_1) → gate(linear_mid) → linear_2``.

    Official Polar uses ``MLP = 32 × Y_{lmax}`` and output ``charges ⊕ 2x0e``.
    ``linear_mid`` / ``linear_2`` carry 0e bias; ``linear_1`` does not.
    """

    def __init__(
        self,
        *,
        irreps_in: str,
        mlp_irreps: str,
        irreps_out: str,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        dtype = (
            config.ftype if dtype is None else dtype
        )  # transitional: goes with molix.config (2c)
        mlp = cue.Irreps("O3", mlp_irreps)
        out = cue.Irreps("O3", irreps_out)
        out_ls = {(mi.ir.l, int(mi.ir.p)) for mi in out}
        scalars = [mi for mi in mlp if mi.ir.l == 0 and (mi.ir.l, int(mi.ir.p)) in out_ls]
        gated = [mi for mi in mlp if mi.ir.l > 0 and (mi.ir.l, int(mi.ir.p)) in out_ls]
        irreps_scalars = cue.Irreps("O3", _join(scalars))
        irreps_gated = cue.Irreps("O3", _join(gated))
        irreps_gates = cue.Irreps("O3", "+".join(f"{mi.mul}x0e" for mi in gated) or "0x0e")
        self.equivariant_nonlin = GatedNonlinearity(
            irreps_scalars,
            [F.silu],
            irreps_gates,
            [F.sigmoid],
            irreps_gated,
            layout="ir_mul",
        )
        nonlin = self.equivariant_nonlin.irreps_in
        self.linear_1 = cueq.Linear(
            cue.Irreps("O3", irreps_in),
            nonlin,
            layout=cue.ir_mul,
            dtype=dtype,
        )
        self.linear_mid = _LinearWithScalarBias(str(mlp), str(nonlin), dtype=dtype)
        self.linear_2 = _LinearWithScalarBias(str(mlp), str(out), dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.equivariant_nonlin(self.linear_1(x))
        x = self.equivariant_nonlin(self.linear_mid(x))
        return self.linear_2(x)


class _AgnosticChargeBiasedLinearPotentialEmbedding(nn.Module):
    """Sum of three ``cuet.Linear`` maps: potential + node feats + local charges."""

    def __init__(
        self,
        *,
        potential_irreps: str,
        node_feats_irreps: str,
        charges_irreps: str,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        dtype = (
            config.ftype if dtype is None else dtype
        )  # transitional: goes with molix.config (2c)
        feats = cue.Irreps("O3", node_feats_irreps)
        self.potential_linear = cueq.Linear(
            cue.Irreps("O3", potential_irreps),
            feats,
            layout=cue.ir_mul,
            dtype=dtype,
        )
        self.node_feats_linear = cueq.Linear(
            feats,
            feats,
            layout=cue.ir_mul,
            dtype=dtype,
        )
        self.charge_embedding = cueq.Linear(
            cue.Irreps("O3", charges_irreps),
            feats,
            layout=cue.ir_mul,
            dtype=dtype,
        )

    def forward(
        self,
        potential_feats: torch.Tensor,
        node_feats: torch.Tensor,
        local_charges: torch.Tensor,
    ) -> torch.Tensor:
        return (
            self.potential_linear(potential_feats)
            + self.node_feats_linear(node_feats)
            + self.charge_embedding(local_charges)
        )


class AgnosticEmbeddedOneBodyVariableUpdate(nn.Module):
    """One non-SCF field-update step: embed potential, uvu-dot, RadialMLP, modulate, readout.

    Output is ``charges ⊕ 2x0e`` in ``ir_mul`` (last two channels are Fukui scalars).
    Physical units of the coefficients are assigned by the GTO energy (e / e·Å).

    Args:
        node_attrs_irreps: Species one-hot, e.g. ``"83x0e"``.
        node_feats_irreps: Mixed node features, e.g. ``"512x0e+512x1o"``.
        potential_irreps: Electrostatic-potential features at each atom.
        charges_irreps: Spin-resolved multipoles, e.g. ``"2x0e+2x1o"``.
    """

    def __init__(
        self,
        *,
        node_attrs_irreps: str,
        node_feats_irreps: str,
        potential_irreps: str,
        charges_irreps: str,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        dtype = (
            config.ftype if dtype is None else dtype
        )  # transitional: goes with molix.config (2c)
        feats = cue.Irreps("O3", node_feats_irreps)
        charges = cue.Irreps("O3", charges_irreps)
        n_0e = _count_0e(feats)
        invar = f"{n_0e}x0e"
        self.potential_embedding = _AgnosticChargeBiasedLinearPotentialEmbedding(
            potential_irreps=potential_irreps,
            node_feats_irreps=node_feats_irreps,
            charges_irreps=charges_irreps,
            dtype=dtype,
        )
        self.source_embedding = cueq.Linear(
            cue.Irreps("O3", node_attrs_irreps),
            cue.Irreps("O3", invar),
            layout=cue.ir_mul,
            dtype=dtype,
        )
        self.dot_products = SparseUvuTensorProduct(
            node_feats_irreps, node_feats_irreps, invar, dtype=dtype
        )
        self.nonlinearity = RadialMLP([2 * n_0e, 64, 64, 64, n_0e], dtype=dtype)
        self.tp_out = SparseUvuTensorProduct(
            node_feats_irreps, invar, node_feats_irreps, dtype=dtype
        )
        mlp_irreps = _mul_sh(_lmax(charges), 32)
        irreps_out = f"{charges_irreps}+2x0e"
        self.readout = GeneralNonLinearBiasReadout(
            irreps_in=node_feats_irreps,
            mlp_irreps=mlp_irreps,
            irreps_out=irreps_out,
            dtype=dtype,
        )

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        potential_features: torch.Tensor,
        local_charges: torch.Tensor,
    ) -> torch.Tensor:
        """Run one field-update step.

        Args:
            node_attrs: Species one-hot ``(N, n_elements)``.
            node_feats: Node features ``(N, irreps.dim)`` in ``ir_mul``.
            potential_features: Local electrostatic potential ``(N, potential.dim)``.
            local_charges: Current multipoles ``(N, charges.dim)``.

        Returns:
            Updated ``charges ⊕ 2x0e`` of shape ``(N, charges.dim + 2)``.
        """
        mixed = self.potential_embedding(potential_features, node_feats, local_charges)
        invariant = self.dot_products(node_feats, mixed)
        source = self.source_embedding(node_attrs)
        nonlin = self.nonlinearity(torch.cat([invariant, source], dim=-1))
        new_feats = self.tp_out(node_feats, nonlin)
        return self.readout(new_feats)


class OneBodyMLPFieldReadout(nn.Module):
    """Post-induction scalar energy from charges and the electrostatic potential.

    ``q_in = charges_induced + charges_0``, then two SparseUvu dots against
    node features and ``RadialMLP([2 n_0e, 128, 128, 128, 1])``.

    Args:
        node_feats_irreps: Mixed node features.
        potential_irreps: Field / potential irreps.
        charges_irreps: Multipole irreps.

    Returns:
        Per-atom energy ``(N,)`` in eV.
    """

    def __init__(
        self,
        *,
        node_feats_irreps: str,
        potential_irreps: str,
        charges_irreps: str,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        dtype = (
            config.ftype if dtype is None else dtype
        )  # transitional: goes with molix.config (2c)
        n_0e = _count_0e(cue.Irreps("O3", node_feats_irreps))
        invar = f"{n_0e}x0e"
        self.linear_up_q = _LinearWithScalarBias(charges_irreps, node_feats_irreps, dtype=dtype)
        self.linear_up_v = _LinearWithScalarBias(potential_irreps, node_feats_irreps, dtype=dtype)
        self.dot_products_q = SparseUvuTensorProduct(
            node_feats_irreps, node_feats_irreps, invar, dtype=dtype
        )
        self.dot_products_v = SparseUvuTensorProduct(
            node_feats_irreps, node_feats_irreps, invar, dtype=dtype
        )
        self.mlp = RadialMLP([2 * n_0e, 128, 128, 128, 1], dtype=dtype)

    def forward(
        self,
        node_feats: torch.Tensor,
        field_feats: torch.Tensor,
        charges_0: torch.Tensor,
        charges_induced: torch.Tensor,
    ) -> torch.Tensor:
        """Read a per-atom non-local energy.

        Args:
            node_feats: ``(N, irreps.dim)``.
            field_feats: Potential features ``(N, potential.dim)``.
            charges_0: Reference multipoles ``(N, charges.dim)``.
            charges_induced: Induced multipoles ``(N, charges.dim)``.

        Returns:
            ``(N,)`` eV/atom.
        """
        q_up = self.linear_up_q(charges_induced + charges_0)
        v_up = self.linear_up_v(field_feats)
        invar = torch.cat(
            [
                self.dot_products_q(node_feats, q_up),
                self.dot_products_v(node_feats, v_up),
            ],
            dim=-1,
        )
        return self.mlp(invar).squeeze(-1)


def _expand_field_norms(norms: list[float], n_widths: int, l_max: int) -> list[float]:
    expanded: list[float] = []
    for ell in range(l_max + 1):
        for j in range(n_widths):
            expanded.extend([norms[ell * n_widths + j]] * (2 * ell + 1))
    return expanded


class MacePolar(MacePotential):
    """MACE-POLAR-1 on the potential contract.

    **Periodic use.** Two submodules hold reciprocal-space grids —
    ``coulomb_energy`` (:class:`~molpot.potentials.gto.hartree.GTOHartreeEnergy`)
    and ``electric_potential_descriptor``
    (:class:`~molpot.potentials.gto.features.GTOFieldFeatures`). Each sizes its
    grid on its first *eager* call with a cell, so the lifecycle is
    ``construct → .to(device) → bind_cell(cell) → compile / export →
    forward(frame with graphs.cell) each step``. See :meth:`bind_cell`.

    ``MacePolar.load(path)`` loads the official dumps (their auxiliary heads
    are returned, not raised, by the remap); the serialised ``kspace_cutoff``
    of a dump wins over the heuristic ``kspace_cutoff_factor``.

    Args:
        spec: The Polar body and electrostatic head configuration.
        wrt: As :class:`~molzoo.mace.potential.MacePotential`.
        energy_dtype: As :class:`~molzoo.mace.potential.MacePotential`;
            covers the Hartree term too.
        input_dtype: As :class:`~molzoo.mace.potential.MacePotential`; the
            per-graph charge and spin follow the geometry's cast, as the
            Fukui equilibration is float arithmetic on them.
        dtype: As :class:`~molzoo.mace.potential.MacePotential`.
    """

    def __init__(
        self,
        spec: MACEPolarSpec,
        *,
        wrt: Wrt = (),
        energy_dtype: torch.dtype | None = torch.float64,
        input_dtype: torch.dtype | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            spec,
            wrt=wrt,
            energy_dtype=energy_dtype,
            input_dtype=input_dtype,
            dtype=dtype,
        )
        hidden = irreps_from_l_max(spec.max_hidden_l, spec.num_features)
        n_el = len(spec.atomic_numbers)
        y = sh_irreps_from_l_max(spec.multipole_lmax)
        charges = f"{y}+{y}"
        n_w = len(spec.field_widths)
        # Official Polar keeps two unsimplified spin copies
        # (``2x0e+2x1o+2x0e+2x1o``), not ``4x0e+4x1o``. ``cat(feat_α, feat_β)``
        # of two ``ir_mul`` field features is exactly that layout.
        field = irreps_from_l_max(spec.field_lmax, n_w)
        potential = f"{field}+{field}"
        node_attrs = f"{n_el}x0e"

        self.lr_source_maps = nn.ModuleList(
            [
                EnvironmentDependentSpinSourceBlock(
                    irreps_in=hidden, l_max=spec.multipole_lmax, dtype=self.dtype
                )
                for _ in range(spec.num_interactions)
            ]
        )
        self.layer_feature_mixer = MultiLayerFeatureMixer(
            irreps=hidden, num_layers=spec.num_interactions, dtype=self.dtype
        )
        self.fukui_source_map = NonLinearBiasReadout(
            irreps_in=hidden, mlp_dim=spec.mlp_dim, out_mul=2, dtype=self.dtype
        )
        self.fukui = FukuiEquilibration()
        self.field_dependent_charges_maps = nn.ModuleList(
            [
                AgnosticEmbeddedOneBodyVariableUpdate(
                    node_attrs_irreps=node_attrs,
                    node_feats_irreps=hidden,
                    potential_irreps=potential,
                    charges_irreps=charges,
                    dtype=self.dtype,
                )
                for _ in range(spec.num_recursion_steps)
            ]
        )
        self.local_electron_energy = OneBodyMLPFieldReadout(
            node_feats_irreps=hidden,
            potential_irreps=potential,
            charges_irreps=charges,
            dtype=self.dtype,
        )
        self.electric_potential_descriptor = GTOFieldFeatures(
            density_sigma=spec.multipole_sigma,
            density_lmax=spec.multipole_lmax,
            feature_lmax=spec.field_lmax,
            feature_widths=list(spec.field_widths),
            kspace_cutoff_factor=spec.kspace_cutoff_factor,
            k_cutoff=spec.kspace_cutoff,
            include_self_interaction=spec.field_self_interaction,
        )
        self.coulomb_energy = GTOHartreeEnergy(
            sigma=spec.multipole_sigma,
            l_max=spec.multipole_lmax,
            include_self_interaction=spec.electrostatic_self_interaction,
            kspace_cutoff_factor=spec.kspace_cutoff_factor,
            k_cutoff=spec.kspace_cutoff,
        )
        expanded = _expand_field_norms(list(spec.field_norms), n_w, spec.field_lmax)
        self.register_buffer(
            "field_feature_norms",
            torch.tensor(expanded, dtype=torch.float64),
            persistent=True,
        )
        self.field_feature_norms: torch.Tensor
        # The reciprocal over both spin copies, at the compute dtype: one
        # multiply per recursion step. Derived, so not a checkpoint key.
        self.register_buffer(
            "_field_norms_inv",
            (1.0 / self.field_feature_norms).repeat(2).to(dtype=self.dtype),
            persistent=False,
        )
        self._field_norms_inv: torch.Tensor
        self.add_local_electron_energy = spec.add_local_electron_energy
        self.num_recursion_steps = spec.num_recursion_steps
        self._y_dim = (spec.multipole_lmax + 1) ** 2

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        config: str | Path | None = None,
        wrt: Wrt = (),
        energy_dtype: torch.dtype | None = torch.float64,
        input_dtype: torch.dtype | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        """Load an official Polar dump (:meth:`MacePotential.load`)."""
        return cls._load(
            path,
            config,
            MACEPolarSpec.from_official,
            polar_keys,
            on_unexpected="return",
            wrt=wrt,
            energy_dtype=energy_dtype,
            input_dtype=input_dtype,
            dtype=dtype,
        )

    def bind_cell(self, cell: torch.Tensor) -> None:
        """Size (first call) or capacity-check (later calls) both k-space grids.

        A construction-time precondition of the periodic path, like the
        element-table check: each GTO block sizes one fixed-capacity reciprocal
        grid by reading the three lattice-vector norms back to the host, which
        a traced call cannot do. So this call precedes compiling the potential
        (``Compiler(...)(model)``), together with one eager forward.

        Idempotent and O(1): the first call fixes each grid's integer box, and
        every later call only re-checks that the current lattice still fits
        inside the capacity that first call bought — so MD may call it after
        every barostat update. It never truncates a grid silently
        (:class:`molix.schema.CellBound`).

        Args:
            cell: Lattice vectors ``(3, 3)`` or ``(1, 3, 3)`` in Å, rows being
                the lattice vectors. Both shapes reach the grids unchanged.

        Raises:
            RuntimeError: Propagated from
                :class:`~molpot.potentials.gto.kspace.KSpaceGrid` — on the
                first call when the mesh would exceed ``max_axis_index``, on a
                later call when the cell has outgrown the capacity fixed by the
                first (raise ``kspace_capacity_factor`` or build a new model).
            ValueError: Propagated likewise, if ``cell`` is neither ``(3, 3)``
                nor ``(1, 3, 3)``.
        """
        self.electric_potential_descriptor.bind_cell(cell)
        self.coulomb_energy.bind_cell(cell)

    def energy(self, frame: TensorDictBase) -> TensorDictBase:
        """Write the Polar site energies and the graph energy with its Hartree term.

        Reads what :meth:`~molzoo.mace.potential.MacePotential.energy` reads,
        plus ``graphs.cell`` ``(3, 3)`` / ``(1, 3, 3)`` in Å when present —
        the batch must then hold exactly one system, the reciprocal sum knows
        one cell — and ``graphs.{total_charge,total_spin}`` (defaulting to the
        neutral singlet) for the Fukui equilibration, which spreads the
        residual between the predicted and the requested totals over the atoms
        of each graph; the spin multiplicity sets the spin-channel targets
        ``(Q ± (M-1))/2``.

        A ghost atom (``atoms.mask`` false) carries no charge: beyond its
        per-atom energies, its density and Fukui source are multiplied to zero
        before every equilibration, so the field descriptor and the Hartree
        sum never see it. Multiplies only, never a slice.

        Raises:
            RuntimeError: On the eager periodic path, if the cell has outgrown
                a k-grid's capacity; under ``torch.compile``, if a grid was
                never bound.
            ValueError: If ``graphs.cell`` is given with more than one graph,
                or is neither ``(3, 3)`` nor ``(1, 3, 3)``.
        """
        Z = frame["atoms", "Z"]
        self._validate_once(Z)
        keys = frame.keys(include_nested=True)
        positions = frame["atoms", "pos"]
        edge_diff = frame["pairs", "pair_diff"] if ("pairs", "pair_diff") in keys else None
        cell = frame["graphs", "cell"] if ("graphs", "cell") in keys else None
        total_charge = (
            frame["graphs", "total_charge"] if ("graphs", "total_charge") in keys else None
        )
        total_spin = frame["graphs", "total_spin"] if ("graphs", "total_spin") in keys else None
        dtype = self._input_dtype
        if dtype is not None:
            positions = positions.to(dtype=dtype)
            edge_diff = None if edge_diff is None else edge_diff.to(dtype=dtype)
            cell = None if cell is None else cell.to(dtype=dtype)
            total_charge = None if total_charge is None else total_charge.to(dtype=dtype)
            total_spin = None if total_spin is None else total_spin.to(dtype=dtype)
        batch = frame["atoms", "batch"]
        num_graphs = frame["graphs"].batch_size[0]
        edge_index = frame["pairs", "pair_index"].contiguous()
        atom_mask = frame["atoms", "mask"] if ("atoms", "mask") in keys else None

        vectors = edge_vectors(positions, edge_index, edge_diff=edge_diff)
        lengths = edge_lengths(vectors)
        elements = self.element_index(Z)
        node_attrs = self.node_attrs(Z, positions.dtype)
        edge_feats, cutoff = self.radial_features(lengths, Z, edge_index)
        # Official PolarMACE permutes the displacement to yzx before its
        # spherical harmonics. The scalar local energy is invariant; the GTO
        # dipoles are not.
        edge_attrs = self.angular_features(vectors.roll(-1, dims=1))
        per_layer = self.layer_features(
            node_feats=self.initial_node_features(elements),
            elements=elements,
            edge_attrs=edge_attrs,
            edge_feats=edge_feats,
            edge_index=edge_index,
            cutoff=cutoff,
        )
        site = self._accumulate(self.atomic_energies(Z)) + self._accumulate(
            self.scale_shift(self._sum_readouts(self.readouts, per_layer))
        )

        density = per_layer[0].new_zeros(per_layer[0].shape[0], 2, self._y_dim)
        for source, layer in zip(self.lr_source_maps, per_layer, strict=True):
            density = density + source(layer)
        mixed = self.layer_feature_mixer(torch.stack(per_layer, dim=0))

        charge_mask = None if atom_mask is None else atom_mask.to(density.dtype).unsqueeze(-1)
        if charge_mask is not None:
            density = density * charge_mask.unsqueeze(-1)
        fukui_source = self.fukui_source_map(per_layer[-1])
        if charge_mask is not None:
            fukui_source = fukui_source * charge_mask

        if total_charge is None:
            total_charge = positions.new_zeros(num_graphs)
        if total_spin is None:
            total_spin = positions.new_ones(num_graphs)
        # Every later step rebinds ``density`` (the equilibration returns a
        # new tensor), so the field-independent charges need no copy.
        density = self.fukui(density, fukui_source, batch, total_charge, total_spin)
        field_independent = density
        for update in self.field_dependent_charges_maps:
            feat_a = self.electric_potential_descriptor(
                density[:, 0, :], positions, batch, cell=cell, num_graphs=num_graphs
            )
            feat_b = self.electric_potential_descriptor(
                density[:, 1, :], positions, batch, cell=cell, num_graphs=num_graphs
            )
            potential_features = torch.cat([feat_a, feat_b], dim=-1) * self._field_norms_inv
            out = update(
                node_attrs,
                mixed,
                potential_features,
                density.reshape(density.shape[0], -1),
            )
            if charge_mask is not None:
                out = out * charge_mask
            density = density + out[:, :-2].reshape(density.shape)
            density = self.fukui(density, out[:, -2:], batch, total_charge, total_spin)

        if self.add_local_electron_energy:
            electron = self.local_electron_energy(
                per_layer[-1],
                potential_features,
                field_independent.reshape(density.shape[0], -1),
                density.reshape(density.shape[0], -1),
            )
            site = site + self._accumulate(electron)
        if atom_mask is not None:
            site = site * atom_mask.to(dtype=site.dtype)
        hartree = self.coulomb_energy(
            density.sum(dim=1), positions, batch, cell=cell, num_graphs=num_graphs
        )
        energy = scatter_sum_compile_safe(site, batch, num_graphs)
        frame[ATOMIC_ENERGY_KEY] = site
        frame[ENERGY_KEY] = energy + hartree.to(dtype=energy.dtype)
        return frame
