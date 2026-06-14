"""Inverse power-law potentials of the form :math:`1/r^p`."""

from typing import Optional

import torch
from torch.special import gammainc

from molpot.potentials.elec.lib.math import gamma, gammaincc_over_powerlaw
from molpot.potentials.elec.potentials.coulomb import _pbc_correction
from molpot.potentials.elec.potentials.potential import Potential


class InversePowerLawPotential(Potential):
    """Inverse power-law potential :math:`1/r^p`.

    Supports full, short-range, long-range, and Fourier-domain evaluation
    with Gaussian range separation.

    Args:
        exponent: The exponent :math:`p` in :math:`1/r^p`.
        smearing: Gaussian width for SR/LR splitting.
        exclusion_radius: Distance within which the potential is smoothly zeroed.
        exclusion_degree: Sharpness of the exclusion cutoff.
        prefactor: Potential prefactor.
    """

    def __init__(
        self,
        exponent: int,
        smearing: Optional[float] = None,
        exclusion_radius: Optional[float] = None,
        exclusion_degree: int = 1,
        prefactor: float = 1.0,
    ):
        super().__init__(smearing, exclusion_radius, exclusion_degree, prefactor)

        # Validate exponent via helper call
        gammaincc_over_powerlaw(exponent, torch.tensor(1.0))
        self.register_buffer("exponent", torch.tensor(exponent, dtype=torch.float64))

    def from_dist(
        self, dist: torch.Tensor, pair_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Full :math:`1/r^p` potential.

        Args:
            dist: Interatomic distances ``(n_edges,)``.
            pair_mask: Optional boolean mask ``(n_edges,)``.

        Returns:
            Potential values ``(n_edges,)``.
        """
        result = torch.pow(dist.clamp(min=1e-15), -self.exponent)
        if pair_mask is not None:
            result = result * pair_mask
        return self.prefactor * result

    def lr_from_dist(
        self, dist: torch.Tensor, pair_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Long-range part of the range-separated :math:`1/r^p` potential.

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

        x = 0.5 * dist**2 / self.smearing**2
        peff = self.exponent / 2
        prefac = 1.0 / (2 * self.smearing**2) ** peff
        result = prefac * gammainc(peff, x.clamp(min=1e-15)) / (x.clamp(min=1e-15) ** peff)
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

        peff = (3 - self.exponent) / 2
        prefac = torch.pi**1.5 / gamma(self.exponent / 2) * (2 * self.smearing**2) ** peff
        x = 0.5 * self.smearing**2 * k_sq

        masked = torch.where(x == 0, 1.0, x)
        k0_limit = 0.0

        if self.exponent > 3:
            k0_limit = -prefac / peff

        return self.prefactor * torch.where(
            k_sq == 0, k0_limit, prefac * gammaincc_over_powerlaw(self.exponent, masked)
        )

    def self_contribution(self) -> torch.Tensor:
        """Self-correction for :math:`1/r^p` potential.

        Returns:
            Scalar correction.
        """
        if self.smearing is None:
            raise ValueError("Cannot compute self contribution without specifying `smearing`.")
        phalf = self.exponent / 2
        return self.prefactor / gamma(phalf + 1) / (2 * self.smearing**2) ** phalf

    def background_correction(self) -> torch.Tensor:
        """Charge-neutrality correction for :math:`1/r^p`.

        Returns:
            Scalar correction.
        """
        if self.smearing is None:
            raise ValueError("Cannot compute background correction without specifying `smearing`.")
        if self.exponent >= 3:
            return torch.zeros_like(self.smearing)
        prefac = torch.pi**1.5 * (2 * self.smearing**2) ** ((3 - self.exponent) / 2)
        prefac /= (3 - self.exponent) * gamma(self.exponent / 2)
        return self.prefactor * prefac

    def pbc_correction(self, periodic, positions, cell, charges):
        """Periodic-boundary correction for the inverse-power-law potential.

        For the Coulomb case (``exponent == 1``) applies the dedicated
        prefactor-scaled correction; otherwise defers to the base class.

        Args:
            periodic: Per-system periodicity flags.
            positions: Atom positions ``(N, 3)``.
            cell: Box vectors ``(3, 3)``.
            charges: Per-atom charges ``(N,)``.

        Returns:
            The PBC energy correction term.
        """
        if self.exponent == 1:
            return self.prefactor * _pbc_correction(periodic, positions, cell, charges)
        return super().pbc_correction(periodic, positions, cell, charges)
