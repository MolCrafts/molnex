"""Polar field-update and field-readout blocks (no energy, no molpot).

Native ``cue.ir_mul`` ports of ACEsuit ``AgnosticEmbeddedOneBodyVariableUpdate``
and ``OneBodyMLPFieldReadout``. Submodule names match the official remap.

Reference:
    Batatia et al. "MACE-POLAR-1: A Polarisable Electrostatic Foundation Model
    for Molecular Chemistry" arXiv:2602.19411.
"""

from __future__ import annotations

import cuequivariance as cue
import cuequivariance_torch as cuet
import torch
import torch.nn as nn
import torch.nn.functional as F

from molix import config
from molrep.interaction.gate import GatedNonlinearity
from molrep.interaction.polar.uvu import SparseUvuTensorProduct
from molrep.interaction.product import sh_irreps_from_l_max
from molrep.interaction.radial import RadialMLP


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
    """Equivariant linear plus even-scalar (0e) bias, matching e3nn ``biases=True``.

    Bias is applied to every 0e slice in irrep order so unsimplified
    ``2x0e+2x1o+2x0e`` keeps Fukui on the last two channels.
    """

    def __init__(self, irreps_in: str, irreps_out: str) -> None:
        super().__init__()
        out = cue.Irreps("O3", irreps_out)
        self.linear = cuet.Linear(
            cue.Irreps("O3", irreps_in),
            out,
            layout=cue.ir_mul,
            dtype=config.ftype,
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
        self.bias = nn.Parameter(torch.zeros(n_0e, dtype=config.ftype))

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

    def __init__(self, *, irreps_in: str, mlp_irreps: str, irreps_out: str) -> None:
        super().__init__()
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
        self.linear_1 = cuet.Linear(
            cue.Irreps("O3", irreps_in),
            nonlin,
            layout=cue.ir_mul,
            dtype=config.ftype,
        )
        self.linear_mid = _LinearWithScalarBias(str(mlp), str(nonlin))
        self.linear_2 = _LinearWithScalarBias(str(mlp), str(out))

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
    ) -> None:
        super().__init__()
        feats = cue.Irreps("O3", node_feats_irreps)
        self.potential_linear = cuet.Linear(
            cue.Irreps("O3", potential_irreps),
            feats,
            layout=cue.ir_mul,
            dtype=config.ftype,
        )
        self.node_feats_linear = cuet.Linear(
            feats,
            feats,
            layout=cue.ir_mul,
            dtype=config.ftype,
        )
        self.charge_embedding = cuet.Linear(
            cue.Irreps("O3", charges_irreps),
            feats,
            layout=cue.ir_mul,
            dtype=config.ftype,
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
    ) -> None:
        super().__init__()
        feats = cue.Irreps("O3", node_feats_irreps)
        charges = cue.Irreps("O3", charges_irreps)
        n_0e = _count_0e(feats)
        invar = f"{n_0e}x0e"
        self.potential_embedding = _AgnosticChargeBiasedLinearPotentialEmbedding(
            potential_irreps=potential_irreps,
            node_feats_irreps=node_feats_irreps,
            charges_irreps=charges_irreps,
        )
        self.source_embedding = cuet.Linear(
            cue.Irreps("O3", node_attrs_irreps),
            cue.Irreps("O3", invar),
            layout=cue.ir_mul,
            dtype=config.ftype,
        )
        self.dot_products = SparseUvuTensorProduct(node_feats_irreps, node_feats_irreps, invar)
        self.nonlinearity = RadialMLP([2 * n_0e, 64, 64, 64, n_0e])
        self.tp_out = SparseUvuTensorProduct(node_feats_irreps, invar, node_feats_irreps)
        mlp_irreps = _mul_sh(_lmax(charges), 32)
        irreps_out = f"{charges_irreps}+2x0e"
        self.readout = GeneralNonLinearBiasReadout(
            irreps_in=node_feats_irreps,
            mlp_irreps=mlp_irreps,
            irreps_out=irreps_out,
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
    ) -> None:
        super().__init__()
        n_0e = _count_0e(cue.Irreps("O3", node_feats_irreps))
        invar = f"{n_0e}x0e"
        self.linear_up_q = _LinearWithScalarBias(charges_irreps, node_feats_irreps)
        self.linear_up_v = _LinearWithScalarBias(potential_irreps, node_feats_irreps)
        self.dot_products_q = SparseUvuTensorProduct(node_feats_irreps, node_feats_irreps, invar)
        self.dot_products_v = SparseUvuTensorProduct(node_feats_irreps, node_feats_irreps, invar)
        self.mlp = RadialMLP([2 * n_0e, 128, 128, 128, 1])

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
