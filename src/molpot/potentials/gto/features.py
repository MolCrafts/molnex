"""GTO electrostatic *field* features (Polar field-update path).

Isolated molecules: real-space finite-difference port of ACEsuit
``RealSpaceFiniteDifferenceElectrostaticFeatures``. The pair kernel is
``erf(r / (2 σ_comb)) / r`` with
``σ_comb = sqrt((σ_density² + σ_feature²) / 2)``.

Periodic systems: pass ``cell`` for the official k-space potential and
analytic ``∇φ`` (same GTO Fourier convention as Hartree).

Output is native ``cue.ir_mul``. Units: charges e, positions Å, features in
eV (potential) / eV·Å⁻¹ (field).

Reference:
    Batatia et al. "MACE-POLAR-1" arXiv:2602.19411, §II.2.6b and SI V.6.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from molix.F.scatter import scatter_sum
from molpot.potentials.elec.prefactors import eV_A
from molpot.potentials.gto.hartree import cached_complete_directed_pairs
from molpot.potentials.gto.kspace import cartesian_dipole, electrostatic_potential, kspace_cutoff


def _cl_sigma(ell: int, sigma: float, normalize: str) -> float:
    """GTO radial normalisation ``C_ℓ(σ)`` (official ``get_Cl_sigma``)."""
    if normalize == "multipoles":
        l_dep = math.sqrt(4.0 * math.pi / (2 * ell + 1)) * 2 ** ((2 * ell + 1) / 2)
        l_dep *= math.gamma((2 * ell + 3) / 2)
        return 1.0 / (l_dep * sigma ** (2 * ell + 3))
    if normalize == "receiver":
        l_dep = 2 ** ((ell + 1) / 2) * math.gamma((ell + 3) / 2)
        return 1.0 / (l_dep * sigma ** (ell + 3))
    raise ValueError(f"normalize must be 'multipoles' or 'receiver', got {normalize!r}")


class GTOFieldFeatures(nn.Module):
    """Per-atom electrostatic potential / field features from GTO multipoles.

    Args:
        density_sigma: Source GTO width in Å (Polar-1-M dump: 1.5).
        density_lmax: Source multipole order (dump: 1).
        feature_lmax: Feature multipole order (dump: 1).
        feature_widths: Projection GTO widths in Å (dump: ``[1.5, 3.0]``).
        offset: Finite-difference displacement in Å (official features: 0.1).
        kspace_cutoff_factor: Polar ``κ`` for the PBC path. Training default
            ``1.0``.
    """

    def __init__(
        self,
        *,
        density_sigma: float,
        density_lmax: int,
        feature_lmax: int,
        feature_widths: list[float],
        offset: float = 0.1,
        kspace_cutoff_factor: float = 1.0,
    ) -> None:
        super().__init__()
        if density_lmax not in (0, 1) or feature_lmax not in (0, 1):
            raise NotImplementedError("GTOFieldFeatures supports l_max=0 or 1")
        if not feature_widths:
            raise ValueError("feature_widths must be non-empty")
        self.density_lmax = density_lmax
        self.feature_lmax = feature_lmax
        self.offset = float(offset)
        self.num_radial = len(feature_widths)
        comb = [math.sqrt((density_sigma**2 + float(w) ** 2) / 2.0) for w in feature_widths]
        self.register_buffer(
            "combined_sigma", torch.tensor(comb, dtype=torch.float64), persistent=True
        )
        l0 = [
            _cl_sigma(0, float(w), "receiver") / _cl_sigma(0, float(w), "multipoles")
            for w in feature_widths
        ]
        self.register_buffer("l0_factors", torch.tensor(l0, dtype=torch.float64))
        l1 = [
            math.sqrt(3.0)
            * float(w) ** 2
            * (_cl_sigma(1, float(w), "receiver") / _cl_sigma(0, float(w), "multipoles"))
            / self.offset
            for w in feature_widths
        ]
        self.register_buffer("l1_factors", torch.tensor(l1, dtype=torch.float64))
        self.density_sigma = float(density_sigma)
        self.kspace_cutoff_factor = float(kspace_cutoff_factor)

    @property
    def out_dim(self) -> int:
        """Flattened ``ir_mul`` width ``n_radial * (feature_lmax + 1)**2``."""
        return self.num_radial * (self.feature_lmax + 1) ** 2

    def _pair_features(
        self,
        charges: torch.Tensor,
        positions: torch.Tensor,
        batch: torch.Tensor,
        orig: torch.Tensor | None,
        n_out: int,
    ) -> torch.Tensor:
        src, dst = cached_complete_directed_pairs(self, batch, orig)
        out = charges.new_zeros(n_out, self.num_radial)
        if src.numel() == 0:
            return out
        r = (positions[src] - positions[dst]).norm(dim=-1, keepdim=True)
        sigma = self.combined_sigma.to(dtype=charges.dtype, device=charges.device)
        kernel = torch.erf(r / (2.0 * sigma)) / r.clamp(min=1e-6)
        contrib = charges[src].unsqueeze(-1) * kernel * eV_A
        return scatter_sum(contrib, dst, dim=0, dim_size=n_out)

    def forward(
        self,
        source_feats: torch.Tensor,
        positions: torch.Tensor,
        batch: torch.Tensor,
        cell: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate field features.

        Args:
            source_feats: Multipoles ``(N, (density_lmax+1)**2)`` in e / e·Å.
            positions: ``(N, 3)`` Å.
            batch: Graph id ``(N,)``.
            cell: Optional lattice ``(3, 3)`` Å. When set, the official
                k-space potential / field is used (single system).

        Returns:
            ``(N, out_dim)`` in ``ir_mul`` of ``n_radial x Y_{lmax}``.
        """
        feats = source_feats.reshape(source_feats.shape[0], -1)
        n = positions.shape[0]
        if cell is not None:
            return self._kspace_features(feats, positions, batch, cell)
        need_fd = self.density_lmax == 1 or self.feature_lmax == 1
        if not need_fd:
            scalars = self._pair_features(feats[:, 0], positions, batch, None, n)
            return scalars * self.l0_factors.to(dtype=feats.dtype, device=feats.device)

        offset = self.offset
        ext_pos = positions.repeat_interleave(4, dim=0)
        ext_pos[1::4, 0] = ext_pos[1::4, 0] + offset
        ext_pos[2::4, 1] = ext_pos[2::4, 1] + offset
        ext_pos[3::4, 2] = ext_pos[3::4, 2] + offset
        charges = feats.new_zeros(n * 4)
        if self.density_lmax == 1 and feats.shape[-1] >= 4:
            charges[1::4] = feats[:, 3] / offset
            charges[2::4] = feats[:, 1] / offset
            charges[3::4] = feats[:, 2] / offset
            charges[0::4] = feats[:, 0] - (charges[1::4] + charges[2::4] + charges[3::4])
        else:
            charges[0::4] = feats[:, 0]
        ext_batch = batch.repeat_interleave(4)
        orig = torch.arange(n, device=batch.device).repeat_interleave(4)
        scalar = self._pair_features(charges, ext_pos, ext_batch, orig, n * 4)
        l0 = self.l0_factors.to(dtype=feats.dtype, device=feats.device)
        if self.feature_lmax == 0:
            return scalar[0::4] * l0
        l1 = self.l1_factors.to(dtype=feats.dtype, device=feats.device)
        # mul_ir  n x 0e + n x 1o (y, z, x per radial) → ir_mul
        scalars = scalar[0::4] * l0
        vec_y = l1 * (scalar[2::4] - scalar[0::4])
        vec_z = l1 * (scalar[3::4] - scalar[0::4])
        vec_x = l1 * (scalar[1::4] - scalar[0::4])
        vec_ir_mul = torch.stack([vec_y, vec_z, vec_x], dim=1).reshape(n, -1)
        return torch.cat([scalars, vec_ir_mul], dim=-1)

    def _kspace_features(
        self,
        feats: torch.Tensor,
        positions: torch.Tensor,
        batch: torch.Tensor,
        cell: torch.Tensor,
    ) -> torch.Tensor:
        n_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
        if n_graphs > 1:
            raise ValueError(
                "GTO k-space field features are single-system (one cell); "
                f"got {n_graphs} graphs. Drop cell for isolated molecules."
            )
        dipoles = (
            cartesian_dipole(feats) if self.density_lmax == 1 and feats.shape[-1] >= 4 else None
        )
        k_cut = kspace_cutoff(
            min(self.density_sigma, float(self.combined_sigma.min())),
            max(self.density_lmax, self.feature_lmax),
            self.kspace_cutoff_factor,
        )
        phi, grad = electrostatic_potential(
            positions,
            feats[:, 0],
            cell,
            sigma_comb=self.combined_sigma,
            k_cutoff=k_cut,
            dipoles=dipoles,
        )
        l0 = self.l0_factors.to(dtype=feats.dtype, device=feats.device)
        scalars = phi * l0
        if self.feature_lmax == 0:
            return scalars
        # Real-space l1 multiplies the FD difference ``φ(r+δ)-φ(r) ≈ δ ∂φ``.
        l1 = self.l1_factors.to(dtype=feats.dtype, device=feats.device) * self.offset
        vec_y = l1 * grad[:, 1, :]
        vec_z = l1 * grad[:, 2, :]
        vec_x = l1 * grad[:, 0, :]
        vec_ir_mul = torch.stack([vec_y, vec_z, vec_x], dim=1).reshape(positions.shape[0], -1)
        return torch.cat([scalars, vec_ir_mul], dim=-1)
