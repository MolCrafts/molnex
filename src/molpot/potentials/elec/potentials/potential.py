"""Base class for pair potential energy functions between monopoles.

Provides the interface to compute short-range and long-range pair potentials
in real space, and the Fourier-domain version of the long-range component.
"""

from typing import Optional

import torch


class Potential(torch.nn.Module):
    """Base interface for a pair potential energy function between monopoles.

    The potential :math:`V(r) = V_{SR}(r) + V_{LR}(r)` is split into
    short-range and long-range parts. Derived classes implement subsets
    of ``from_dist``, ``lr_from_dist``, and ``lr_from_k_sq``.

    Args:
        smearing: Length scale for range separation between SR and LR parts.
        exclusion_radius: Distance within which the potential is smoothly zeroed.
        exclusion_degree: Sharpness of the exclusion cutoff (raised cosine exponent).
        prefactor: Potential prefactor (see :mod:`molpot.potentials.elec.prefactors`).
    """

    def __init__(
        self,
        smearing: Optional[float] = None,
        exclusion_radius: Optional[float] = None,
        exclusion_degree: int = 1,
        prefactor: float = 1.0,
    ):
        super().__init__()

        if smearing is not None:
            self.register_buffer("smearing", torch.tensor(smearing, dtype=torch.float64))
        else:
            self.smearing = None

        self.exclusion_radius = exclusion_radius
        self.exclusion_degree = exclusion_degree
        self.register_buffer("prefactor", torch.tensor(prefactor, dtype=torch.float64))

    def f_cutoff(
        self, dist: torch.Tensor, pair_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Smooth cutoff for excluding the local region.

        Uses a shifted cosine:
        :math:`1 - ((1 - \\cos \\pi r / r_{cut}) / 2)^n`.

        Args:
            dist: Interatomic distances ``(n_edges,)``.
            pair_mask: Optional boolean mask ``(n_edges,)``.

        Returns:
            Cutoff values ``(n_edges,)``.
        """
        if self.exclusion_radius is None:
            raise ValueError("Cannot compute cutoff function when `exclusion_radius` is not set")

        result = torch.where(
            dist < self.exclusion_radius,
            1
            - ((1 - torch.cos(torch.pi * (dist / self.exclusion_radius))) * 0.5)
            ** self.exclusion_degree,
            0.0,
        )
        if pair_mask is not None:
            result = result * pair_mask

        return result

    def from_dist(
        self, dist: torch.Tensor, pair_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Full pair potential given interatomic distances.

        Args:
            dist: Distances ``(n_edges,)``.
            pair_mask: Optional boolean mask ``(n_edges,)``.

        Returns:
            Potential values ``(n_edges,)``.
        """
        raise NotImplementedError(f"from_dist is not implemented for {self.__class__.__name__}")

    def sr_from_dist(
        self, dist: torch.Tensor, pair_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Short-range part of the pair potential in real space.

        Computed as :math:`V_{SR}(r) = V(r) - V_{LR}(r)`, or
        :math:`-V_{LR}(r) * f_{cut}(r)` when ``exclusion_radius`` is set.

        Args:
            dist: Distances ``(n_edges,)``.
            pair_mask: Optional boolean mask ``(n_edges,)``.

        Returns:
            Short-range potential values ``(n_edges,)``.
        """
        if self.smearing is None:
            raise ValueError(
                "Cannot compute range-separated potential when `smearing` is not specified."
            )

        if self.exclusion_radius is None:
            return self.from_dist(dist, pair_mask=pair_mask) - self.lr_from_dist(
                dist, pair_mask=pair_mask
            )
        return -self.lr_from_dist(dist, pair_mask=pair_mask) * self.f_cutoff(
            dist, pair_mask=pair_mask
        )

    def lr_from_dist(
        self, dist: torch.Tensor, pair_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Long-range part of the pair potential in real space.

        Args:
            dist: Distances ``(n_edges,)``.
            pair_mask: Optional boolean mask ``(n_edges,)``.

        Returns:
            Long-range potential values ``(n_edges,)``.
        """
        raise NotImplementedError(f"lr_from_dist is not implemented for {self.__class__.__name__}")

    def lr_from_k_sq(self, k_sq: torch.Tensor) -> torch.Tensor:
        """Fourier-domain long-range potential :math:`\\hat{V}_{LR}(k)`.

        Expressed in terms of :math:`k^2` to avoid unnecessary sqrt.

        Args:
            k_sq: Squared k-vector norms ``(...)``.

        Returns:
            Fourier-domain potential values matching ``k_sq`` shape.
        """
        raise NotImplementedError(f"lr_from_k_sq is not implemented for {self.__class__.__name__}")

    def kernel_from_k_sq(self, k_sq: torch.Tensor) -> torch.Tensor:
        """Compatibility with :class:`KSpaceKernel` interface.

        Delegates to :meth:`lr_from_k_sq`.
        """
        return self.lr_from_k_sq(k_sq)

    def self_contribution(self) -> torch.Tensor:
        """Self-interaction correction from the range-split smearing charge.

        Returns:
            Scalar correction tensor.
        """
        raise NotImplementedError(
            f"self_contribution is not implemented for {self.__class__.__name__}"
        )

    def background_correction(self) -> torch.Tensor:
        """Correction for net-charge neutrality in periodic systems.

        Returns:
            Scalar correction tensor.
        """
        raise NotImplementedError(
            f"background_correction is not implemented for {self.__class__.__name__}"
        )

    def pbc_correction(
        self,
        periodic: Optional[torch.Tensor],
        positions: torch.Tensor,
        cell: torch.Tensor,
        charges: torch.Tensor,
    ) -> torch.Tensor:
        """Correction for 2D-periodic systems (slab geometry).

        Args:
            periodic: Boolean mask ``(3,)`` indicating periodic directions.
            positions: Cartesian coordinates ``(n_atoms, 3)``.
            cell: Unit cell matrix ``(3, 3)``.
            charges: Atomic charges ``(n_atoms, n_channels)``.

        Returns:
            Correction tensor ``(n_atoms, n_channels)``.
        """
        return self.prefactor * torch.zeros_like(charges)
