"""Latent Ewald Summation (LES): the long-range energy of latent atomic charges.

LES assigns every atom a latent charge ``q_i`` (in e, predicted by a readout of
the short-range model) and adds the electrostatic energy of those charges
smeared as Gaussians of width ``σ`` — ``erf(r / (√2 σ)) / r`` between a pair.
In eV::

    isolated:  E = ½ Σ_{i≠j} q_i q_j · erf(r_ij / (√2 σ)) / r_ij · c / 2π
    periodic:  E = (c / Ω) Σ_{k≠0, |k| < 2π/Δ} exp(-σ² k² / 2) / k² |S(k)|²
                   − c Σ_i q_i² / (σ (2π)^{3/2})
    S(k) = Σ_i q_i exp(i k · r_i)

``c`` is the reference implementation's ``norm_factor`` (``90.0474``: the
Coulomb constant ``1/(4πε₀)`` in eV·Å/e² times 2π, to the precision LES
hard-codes it), ``Ω`` the cell volume in Å³ and ``Δ`` the reciprocal-grid
spacing ``dl``, so the cutoff is ``|k| < 2π / Δ``. The periodic sum runs over
half of a fixed-capacity Miller box (:class:`~molpot.potentials.gto.kspace.KSpaceGrid`,
weight 2 per ``±k`` pair) and subtracts each Gaussian's self-energy; no host
sync, no data-dependent shape. Positions, charges and the cell are ordinary
autograd inputs, so forces and the virial come from
:class:`molix.autograd.Derivative`.

Two deviations from LES's legacy ``Ewald`` module, both measure-zero: the
cutoff is strict (``|k| < 2π/Δ``, LES keeps ``≤``), and LES's per-axis box
``|n_i| ≤ ⌊‖a_i‖/Δ⌋`` is implied by the ball and therefore not applied again.

Reference:
    Cheng, B. "Latent Ewald summation for machine learning of long-range
    interactions." npj Comput. Mater. 11, 80 (2025).
    https://doi.org/10.1038/s41524-025-01577-7 ; code ``ChengUCB/les``
    (``les/module/ewald.py``).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from molix.F.scatter import scatter_sum_compile_safe
from molpot.potentials.gto.kspace import KSpaceGrid, reciprocal_basis, structure_factor

_TWOPI = 2.0 * math.pi


class LatentEwaldEnergy(nn.Module):
    """Gaussian-charge Ewald energy of latent charges, isolated or periodic.

    Lifecycle on the periodic path is that of :class:`KSpaceGrid`: one eager
    :meth:`bind_cell` sizes the reciprocal box, then :meth:`forward` traces.
    The isolated path is a dense same-graph pair sum, ``O(N²)`` in memory — for
    molecules, not condensed phases.

    Args:
        sigma: Gaussian width ``σ`` in Å.
        dl: Reciprocal-grid spacing ``Δ`` in Å; the cutoff is ``2π / Δ`` Å⁻¹.
        norm_factor: ``c`` in eV·Å/e² (LES's ``norm_factor``).
        remove_self_interaction: Subtract the Gaussian self-energy on the
            periodic path (LES default ``True``).
        capacity_factor: Head-room of the reciprocal box under a barostat; see
            :class:`KSpaceGrid`.
    """

    def __init__(
        self,
        *,
        sigma: float = 1.0,
        dl: float = 2.0,
        norm_factor: float = 90.0474,
        remove_self_interaction: bool = True,
        capacity_factor: float = 1.2,
    ) -> None:
        super().__init__()
        self.sigma = float(sigma)
        self.dl = float(dl)
        self.norm_factor = float(norm_factor)
        self.remove_self_interaction = bool(remove_self_interaction)
        self.grid = KSpaceGrid(_TWOPI / self.dl, capacity_factor=capacity_factor)
        #: Empty device marker for :meth:`_apply`; never saved.
        self.register_buffer("_anchor", torch.empty(0), persistent=False)

    def _apply(self, fn, recurse: bool = True):
        """Carry the reciprocal box along with ``.to()`` / ``.cuda()``.

        The box is integer index metadata on a plain :class:`KSpaceGrid`, not a
        buffer; it follows the device marker ``_anchor``.
        """
        super()._apply(fn, recurse)
        self.grid.to(self._anchor.device)
        return self

    def bind_cell(self, cell: torch.Tensor) -> None:
        """Size (first call) or capacity-check (later calls) the reciprocal box.

        Eager only; must precede compiling a periodic forward.

        Args:
            cell: Lattice vectors ``(3, 3)`` or ``(1, 3, 3)`` in Å, one per row.
        """
        self.grid.bind(cell)

    def forward(
        self,
        positions: torch.Tensor,
        charges: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
        cell: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Long-range energy per graph.

        Args:
            positions: Positions ``(N, 3)`` in Å — unwrapped or wrapped; the
                periodic sum is invariant to lattice translations.
            charges: Latent charges ``(N,)`` in e.
            batch: Graph id ``(N,)``.
            num_graphs: Graph count ``B``.
            cell: Lattice vectors ``(1, 3, 3)`` or ``(3, 3)`` in Å for a single
                periodic system; ``None`` for isolated graphs.

        Returns:
            Energy ``(B,)`` in eV.
        """
        if cell is None:
            return self._isolated(positions, charges, batch, num_graphs)
        if num_graphs != 1:
            raise ValueError(f"periodic LES is single-system, got num_graphs={num_graphs}")
        return self._periodic(positions, charges, cell).reshape(1)

    def _isolated(
        self,
        positions: torch.Tensor,
        charges: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        diff = positions.unsqueeze(0) - positions.unsqueeze(1)
        same = (batch.unsqueeze(0) == batch.unsqueeze(1)) & ~torch.eye(
            positions.shape[0], dtype=torch.bool, device=positions.device
        )
        # Guard the diagonal before the square root: d‖x‖/dx at 0 is NaN even
        # where the pair is masked out afterwards.
        dist_sq = torch.where(same, (diff * diff).sum(-1), torch.ones_like(diff[..., 0]))
        dist = dist_sq.sqrt()
        kernel = torch.erf(dist / (math.sqrt(2.0) * self.sigma)) / dist
        kernel = torch.where(same, kernel, torch.zeros_like(kernel))
        phi = kernel @ charges
        per_atom = 0.5 * charges * phi * (self.norm_factor / _TWOPI)
        return scatter_sum_compile_safe(per_atom, batch, num_graphs)

    def _periodic(
        self, positions: torch.Tensor, charges: torch.Tensor, cell: torch.Tensor
    ) -> torch.Tensor:
        kvec, k_sq, weights = self.grid.vectors(cell)
        del kvec
        recip = reciprocal_basis(cell)
        s_real, s_imag = structure_factor(
            positions, charges, self.grid.n, recip, self.grid.capacity
        )
        volume = torch.linalg.det(cell.reshape(3, 3))
        kfac = weights * torch.exp(-0.5 * self.sigma**2 * k_sq) / k_sq
        energy = (kfac * (s_real * s_real + s_imag * s_imag)).sum() / volume * self.norm_factor
        if self.remove_self_interaction:
            energy = (
                energy - (charges * charges).sum() / (self.sigma * _TWOPI**1.5) * self.norm_factor
            )
        return energy
