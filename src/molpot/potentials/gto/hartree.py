"""GTO Hartree energy: isolated real-space pair sum or Polar k-space.

Isolated molecules use a directed complete graph excluding self-pairs; each
directed edge carries ``½ k_e q_i q_j erf(r/(2σ))/r``. Periodic systems pass
``cell`` and switch to the official reciprocal sum
(:mod:`molpot.potentials.gto.kspace`) — not PME.

Units: charges e, positions Å, energy eV. Prefactor
:data:`molpot.potentials.elec.prefactors.eV_A`.

Reference:
    Batatia et al. "MACE-POLAR-1" arXiv:2602.19411, §II.2.6b and SI V.6.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from molix.F.scatter import scatter_sum
from molpot.potentials.elec.prefactors import eV_A
from molpot.potentials.gto.kernel import GTOCoulombKernel
from molpot.potentials.gto.kspace import cartesian_dipole, hartree_energy, kspace_cutoff


def _cl_multipoles(ell: int, sigma: float) -> float:
    """GTO radial normalisation ``C_ℓ(σ)`` with ``normalize='multipoles'``."""
    l_dep = math.sqrt(4.0 * math.pi / (2 * ell + 1)) * 2 ** ((2 * ell + 1) / 2)
    l_dep *= math.gamma((2 * ell + 3) / 2)
    return 1.0 / (l_dep * sigma ** (2 * ell + 3))


def gto_self_overlap(sigma: float, l_max: int) -> torch.Tensor:
    """On-site GTO Coulomb overlap (official ``GTOSelfInteractionBlock``).

    Layout is one copy of ``Y_{l_max}``: monopole then dipole ``(y, z, x)``.
    Isolated Hartree adds ``½ Σ_i Σ_b ω_b q_{i b}²``. The dump at ``σ=1.5 Å``,
    ``l_max=1`` is ``ω = [eV_A/(σ√π), 0.40119, 0.40119, 0.40119]``.

    Args:
        sigma: GTO width in Å.
        l_max: Multipole order, 0 or 1.

    Returns:
        ``((l_max+1)²,)`` overlap in eV per (e or e·Å)².
    """
    n = (l_max + 1) ** 2
    out = torch.zeros(n, dtype=torch.float64)
    field_constant = eV_A * 4.0 * math.pi
    grid = torch.linspace(1.0e-4, 10.0 * float(sigma), 10000, dtype=torch.float64)
    r = grid
    sigma_f = float(sigma)
    for ell in range(l_max + 1):
        a = (2 * ell + 3) / 2
        p = torch.special.gammainc(torch.tensor(a, dtype=torch.float64), 0.5 * r * r / sigma_f**2)
        f1 = (2 ** (ell + 0.5)) * (sigma_f ** (2 * ell + 3)) * p * math.gamma(a) / r.pow(ell + 1)
        f2 = (sigma_f**2) * r.pow(ell) * torch.exp(-0.5 * r * r / sigma_f**2)
        integrand = r.pow(ell + 2) * torch.exp(-0.5 * r * r / sigma_f**2) * (f1 + f2)
        value = float(torch.trapezoid(integrand, r))
        cl = _cl_multipoles(ell, sigma_f)
        coeff = cl * cl * field_constant / (2 * ell + 1) * value
        start = ell**2
        out[start : start + 2 * ell + 1] = coeff
    return out


def _skip_diagonal(n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """All directed pairs ``(i, j)`` with ``i != j``, no ``N×N`` mask."""
    if n < 2:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty
    idx = torch.arange(n * (n - 1), device=device)
    src = torch.div(idx, n - 1, rounding_mode="floor")
    dst = idx.remainder(n - 1)
    return src, dst + (dst >= src).to(dtype=dst.dtype)


def _skip_equal_orig_regular(
    n_atoms: int, copies: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Directed pairs of ``n_atoms * copies`` sites whose original atom differs.

    Layout is ``orig = arange(n_atoms).repeat_interleave(copies)``.
    """
    atom_src, atom_dst = _skip_diagonal(n_atoms, device)
    copy = torch.arange(copies, device=device)
    src, dst = torch.broadcast_tensors(
        atom_src[:, None, None] * copies + copy[None, :, None],
        atom_dst[:, None, None] * copies + copy[None, None, :],
    )
    return src.reshape(-1), dst.reshape(-1)


def _complete_directed_pairs(
    batch: torch.Tensor,
    orig: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Directed pairs that share a graph and come from different original atoms.

    ``orig`` is the pre-duplication atom id (identity for ``l_max=0``; repeated
    4× for the dipole finite-difference copies). Copies of the same atom never
    interact — that is the optional self-interaction term.
    """
    n = int(batch.shape[0])
    device = batch.device
    empty = torch.empty(0, dtype=torch.long, device=device)
    if n == 0:
        return empty, empty
    single = bool((batch == batch[0]).all().item())
    if orig is None:
        src, dst = _skip_diagonal(n, device)
        if single:
            return src, dst
        keep = batch[src] == batch[dst]
        return src[keep], dst[keep]

    if single:
        first = orig[0]
        copies = int((orig == first).sum().item())
        n_atoms = n // copies if copies else 0
        if copies > 1 and n_atoms * copies == n:
            expected = torch.arange(n_atoms, device=device, dtype=orig.dtype).repeat_interleave(
                copies
            )
            if torch.equal(orig, expected):
                return _skip_equal_orig_regular(n_atoms, copies, device)
        src, dst = _skip_diagonal(n, device)
        keep = orig[src] != orig[dst]
        return src[keep], dst[keep]

    src, dst = _skip_diagonal(n, device)
    keep = (batch[src] == batch[dst]) & (orig[src] != orig[dst])
    return src[keep], dst[keep]


def cached_complete_directed_pairs(
    owner: object,
    batch: torch.Tensor,
    orig: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reuse pair indices while ``N``, graph membership and orig layout stay put."""
    osum = 0 if orig is None else int(orig.sum().item()) + 1
    key = (int(batch.shape[0]), int(batch.sum().item()), osum, batch.device)
    cached = getattr(owner, "_pair_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1], cached[2]
    src, dst = _complete_directed_pairs(batch, orig)
    owner._pair_cache = (key, src, dst)
    return src, dst


class GTOHartreeEnergy(nn.Module):
    """GTO Hartree: isolated pair sum, or Polar SI k-space when ``cell`` is set.

    Isolated ``l_max=1`` expands each dipole as three displaced scalar charges
    (official offset 0.02 Å). Periodic ``l_max=1`` uses the analytic dipole
    structure factor (SI V.6), not the finite-difference copies.

    Args:
        sigma: GTO width in Å (Polar-1-M dump: 1.5).
        l_max: Multipole order, 0 or 1.
        include_self_interaction: Polar dump default is True. Pair path still
            skips ``i=i``; this flag adds the on-site GTO overlap
            ``½ Σ_i Σ_b ω_b q_{ib}²``.
        offset: Finite-difference displacement in Å for isolated ``l_max=1``.
        kspace_cutoff_factor: Polar ``κ``. Training default ``1.0``.
    """

    def __init__(
        self,
        *,
        sigma: float,
        l_max: int = 0,
        include_self_interaction: bool = False,
        offset: float = 0.02,
        kspace_cutoff_factor: float = 1.0,
    ) -> None:
        super().__init__()
        if l_max not in (0, 1):
            raise NotImplementedError("GTOHartreeEnergy supports l_max=0 or 1")
        self.l_max = l_max
        self.include_self_interaction = include_self_interaction
        self.offset = float(offset)
        self.sigma = float(sigma)
        self.kspace_cutoff_factor = float(kspace_cutoff_factor)
        self.kernel = GTOCoulombKernel(sigma=sigma)
        self.register_buffer("self_overlap", gto_self_overlap(sigma, l_max), persistent=True)

    @property
    def k_cutoff(self) -> float:
        """Reciprocal cutoff in Å⁻¹ from Polar SI V.6."""
        return kspace_cutoff(self.sigma, self.l_max, self.kspace_cutoff_factor)

    def _pair_energy(
        self,
        charges: torch.Tensor,
        positions: torch.Tensor,
        batch: torch.Tensor,
        orig: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
        src, dst = cached_complete_directed_pairs(self, batch, orig)
        if src.numel() == 0:
            return charges.new_zeros(n_graphs)
        r = (positions[dst] - positions[src]).norm(dim=-1)
        edge = 0.5 * eV_A * self.kernel(r) * charges[src] * charges[dst]
        node = scatter_sum(edge, dst, dim=0, dim_size=charges.shape[0])
        return scatter_sum(node, batch, dim=0, dim_size=n_graphs)

    def _self_energy(self, feats: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        n_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
        overlap = self.self_overlap.to(dtype=feats.dtype, device=feats.device)
        width = min(int(overlap.shape[0]), int(feats.shape[-1]))
        node = 0.5 * (feats[:, :width] * overlap[:width] * feats[:, :width]).sum(dim=-1)
        return scatter_sum(node, batch, dim=0, dim_size=n_graphs)

    def forward(
        self,
        source_feats: torch.Tensor,
        positions: torch.Tensor,
        batch: torch.Tensor,
        cell: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Hartree energy per graph.

        Args:
            source_feats: Multipoles ``(N, (l_max+1)**2)`` in e / e·Å.
            positions: Cartesian positions ``(N, 3)`` in Å.
            batch: Graph id ``(N,)``.
            cell: Optional lattice ``(3, 3)`` or ``(1, 3, 3)`` Å. When set,
                the official k-space sum runs (single system). When omitted,
                the isolated real-space pair sum runs.

        Returns:
            Per-graph energy ``(B,)`` in eV.
        """
        feats = source_feats.reshape(source_feats.shape[0], -1)
        if cell is not None:
            energy = self._kspace_energy(feats, positions, batch, cell)
        elif self.l_max == 0:
            energy = self._pair_energy(feats[:, 0], positions, batch)
        else:
            offset = self.offset
            n = positions.shape[0]
            extended_pos = positions.repeat_interleave(4, dim=0)
            extended_pos[1::4, 0] = extended_pos[1::4, 0] + offset
            extended_pos[2::4, 1] = extended_pos[2::4, 1] + offset
            extended_pos[3::4, 2] = extended_pos[3::4, 2] + offset
            charges = feats.new_zeros(n * 4)
            charges[1::4] = feats[:, 3] / offset
            charges[2::4] = feats[:, 1] / offset
            charges[3::4] = feats[:, 2] / offset
            charges[0::4] = feats[:, 0] - (charges[1::4] + charges[2::4] + charges[3::4])
            extended_batch = batch.repeat_interleave(4)
            orig = torch.arange(n, device=batch.device).repeat_interleave(4)
            energy = self._pair_energy(charges, extended_pos, extended_batch, orig)
        if self.include_self_interaction:
            energy = energy + self._self_energy(feats, batch)
        return energy

    def _kspace_energy(
        self,
        feats: torch.Tensor,
        positions: torch.Tensor,
        batch: torch.Tensor,
        cell: torch.Tensor,
    ) -> torch.Tensor:
        n_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
        if n_graphs > 1:
            raise ValueError(
                "GTO k-space Hartree is single-system (one cell); "
                f"got {n_graphs} graphs. Drop cell for isolated molecules."
            )
        dipoles = cartesian_dipole(feats) if self.l_max == 1 and feats.shape[-1] >= 4 else None
        energy = hartree_energy(
            positions,
            feats[:, 0],
            cell,
            sigma=self.sigma,
            k_cutoff=self.k_cutoff,
            dipoles=dipoles,
        )
        out = feats.new_zeros(max(n_graphs, 1))
        out[0] = energy
        return out
