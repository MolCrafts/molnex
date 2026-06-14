"""Smoothed Coulomb potential :math:`1/r` with range separation."""

from typing import Optional

import torch

from molpot.potentials.elec.potentials.potential import Potential


def _pbc_correction(
    periodic: Optional[torch.Tensor],
    positions: torch.Tensor,
    cell: torch.Tensor,
    charges: torch.Tensor,
) -> torch.Tensor:
    """2D-periodicity correction for the :math:`1/r` potential.

    Args:
        periodic: Boolean mask ``(3,)`` indicating periodic directions.
        positions: Cartesian coordinates ``(n_atoms, 3)``.
        cell: Unit cell matrix ``(3, 3)``.
        charges: Atomic charges ``(n_atoms, n_channels)``.

    Returns:
        Slab correction ``(n_atoms, n_channels)``.
    """
    if periodic is None:
        periodic = torch.tensor([True, True, True], device=cell.device)
    n_periodic = torch.sum(periodic)
    is_2d = n_periodic == 2
    axis = torch.argmax(
        torch.where(
            is_2d.unsqueeze(-1),
            (~periodic).to(torch.int64),
            torch.zeros_like(periodic, dtype=torch.int64),
        ),
        dim=-1,
    )
    E_slab = torch.zeros_like(charges)
    z_i = torch.gather(positions, 1, axis.expand(positions.shape[0]).unsqueeze(-1))
    basis_len = torch.gather(torch.linalg.norm(cell, dim=-1), 0, axis)
    V = torch.abs(torch.linalg.det(cell))
    charge_tot = torch.sum(charges, dim=0)
    M_axis = torch.sum(charges * z_i, dim=0)
    M_axis_sq = torch.sum(charges * z_i**2, dim=0)
    E_slab_2d = (4.0 * torch.pi / V) * (
        z_i * M_axis - 0.5 * (M_axis_sq + charge_tot * z_i**2) - charge_tot / 12.0 * basis_len**2
    )

    return torch.where(is_2d.unsqueeze(-1), E_slab_2d, E_slab)


class CoulombPotential(Potential):
    """Smoothed electrostatic Coulomb potential :math:`1/r`.

    Supports full, short-range, long-range, and Fourier-domain evaluation
    with Gaussian range separation controlled by ``smearing``.

    Args:
        smearing: Gaussian width for SR/LR splitting.
        exclusion_radius: Distance within which the potential is smoothly zeroed.
        exclusion_degree: Sharpness of the exclusion cutoff.
        prefactor: Electrostatics prefactor (see :mod:`molpot.potentials.elec.prefactors`).
    """

    def __init__(
        self,
        smearing: Optional[float] = None,
        exclusion_radius: Optional[float] = None,
        exclusion_degree: int = 1,
        prefactor: float = 1.0,
    ):
        super().__init__(smearing, exclusion_radius, exclusion_degree, prefactor)

    def from_dist(
        self, dist: torch.Tensor, pair_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Full :math:`1/r` potential.

        Args:
            dist: Interatomic distances ``(n_edges,)``.
            pair_mask: Optional boolean mask ``(n_edges,)``.

        Returns:
            Potential values ``(n_edges,)``.
        """
        result = 1.0 / dist.clamp(min=1e-15)

        if pair_mask is not None:
            result = result * pair_mask

        return self.prefactor * result

    def lr_from_dist(
        self, dist: torch.Tensor, pair_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Long-range part of the range-separated :math:`1/r` potential.

        Args:
            dist: Interatomic distances ``(n_edges,)``.
            pair_mask: Optional boolean mask ``(n_edges,)``.

        Returns:
            Long-range potential values ``(n_edges,)``.
        """
        if self.smearing is None:
            raise ValueError(
                "Cannot compute long-range contribution without specifying `smearing`."
            )
        result = torch.erf(dist / self.smearing / 2.0**0.5) / dist.clamp(min=1e-12)
        if pair_mask is not None:
            result = result * pair_mask

        return self.prefactor * result

    def lr_from_k_sq(self, k_sq: torch.Tensor) -> torch.Tensor:
        """Fourier transform of the LR part in terms of :math:`k^2`.

        Args:
            k_sq: Squared k-vector norms ``(...)``.

        Returns:
            Fourier-domain potential values matching ``k_sq`` shape.
        """
        if self.smearing is None:
            raise ValueError("Cannot compute long-range kernel without specifying `smearing`.")

        masked = torch.where(k_sq == 0, 1.0, k_sq)
        return self.prefactor * torch.where(
            k_sq == 0,
            0.0,
            4 * torch.pi * torch.exp(-0.5 * self.smearing**2 * masked) / masked,
        )

    def self_contribution(self) -> torch.Tensor:
        """Self-correction for the :math:`1/r` potential.

        Returns:
            Scalar correction.
        """
        if self.smearing is None:
            raise ValueError("Cannot compute self contribution without specifying `smearing`.")
        return self.prefactor * (2 / torch.pi) ** 0.5 / self.smearing

    def background_correction(self) -> torch.Tensor:
        """Charge-neutrality correction for the :math:`1/r` potential.

        Returns:
            Scalar correction.
        """
        if self.smearing is None:
            raise ValueError("Cannot compute background correction without specifying `smearing`.")
        return self.prefactor * torch.pi * self.smearing**2

    def pbc_correction(
        self,
        periodic: Optional[torch.Tensor],
        positions: torch.Tensor,
        cell: torch.Tensor,
        charges: torch.Tensor,
    ) -> torch.Tensor:
        """2D slab correction for Coulomb potential.

        Args:
            periodic: Boolean mask ``(3,)``.
            positions: Cartesian coordinates ``(n_atoms, 3)``.
            cell: Unit cell matrix ``(3, 3)``.
            charges: Atomic charges ``(n_atoms, n_channels)``.

        Returns:
            Correction ``(n_atoms, n_channels)``.
        """
        return self.prefactor * _pbc_correction(periodic, positions, cell, charges)
