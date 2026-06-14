"""Pair potential energy function between point dipoles."""

from typing import Optional

import torch


class PotentialDipole(torch.nn.Module):
    """Pair potential between point dipoles.

    The interaction is:

    .. math::

        V(\\vec{r}) = \\frac{(\\vec{\\mu}_i \\cdot \\vec{\\mu}_j)}{r^3}
        - \\frac{3 (\\vec{\\mu}_i \\cdot \\vec{r})(\\vec{\\mu}_j \\cdot \\vec{r})}{r^5}

    Args:
        smearing: Gaussian width for SR/LR splitting.
        exclusion_radius: Distance within which the potential is smoothly zeroed.
        exclusion_degree: Sharpness of the exclusion cutoff.
        epsilon: Dielectric constant of the medium.
        prefactor: Potential prefactor.
    """

    def __init__(
        self,
        smearing: Optional[float] = None,
        exclusion_radius: Optional[float] = None,
        exclusion_degree: int = 1,
        epsilon: float = 0.0,
        prefactor: float = 1.0,
    ):
        super().__init__()

        self.exclusion_degree = exclusion_degree
        if smearing is not None:
            self.register_buffer("smearing", torch.tensor(smearing, dtype=torch.float64))
        else:
            self.smearing = None
        if exclusion_radius is not None:
            self.register_buffer(
                "exclusion_radius",
                torch.tensor(exclusion_radius, dtype=torch.float64),
            )
        else:
            self.exclusion_radius = None
        self.register_buffer("epsilon", torch.tensor(epsilon, dtype=torch.float64))
        self.register_buffer("prefactor", torch.tensor(prefactor, dtype=torch.float64))

    def f_cutoff(self, vector: torch.Tensor) -> torch.Tensor:
        """Smooth cutoff for excluding the local region.

        Uses a shifted cosine:
        :math:`1 - ((1 - \\cos \\pi r / r_{cut}) / 2)^n`.

        Args:
            vector: Pair vectors ``(n_edges, 3)``.

        Returns:
            Cutoff values ``(n_edges, 1)``.
        """
        r_mag = torch.norm(vector, dim=1, keepdim=True)
        if self.exclusion_radius is None:
            raise ValueError("Cannot compute cutoff function when `exclusion_radius` is not set")

        return torch.where(
            r_mag < self.exclusion_radius,
            1
            - ((1 - torch.cos(torch.pi * (r_mag / self.exclusion_radius))) * 0.5)
            ** self.exclusion_degree,
            0.0,
        )

    def from_dist(self, vector: torch.Tensor) -> torch.Tensor:
        """Full dipolar potential as a function of pair vectors.

        Args:
            vector: Pair vectors ``(n_edges, 3)``.

        Returns:
            Potential tensor ``(n_edges, 3, 3)``.
        """
        r_mag = torch.norm(vector, dim=1, keepdim=True)
        scalar_potential = 1.0 / (r_mag**3)
        r_outer = torch.bmm(vector.unsqueeze(2), vector.unsqueeze(1))
        return self.prefactor * (
            scalar_potential.unsqueeze(-1) * torch.eye(3).to(r_outer).unsqueeze(0)
            - 3.0 * r_outer / (r_mag**5).unsqueeze(-1)
        )

    def sr_from_dist(self, dist: torch.Tensor) -> torch.Tensor:
        """Short-range part of the dipolar potential.

        Args:
            dist: Pair vectors ``(n_edges, 3)``.

        Returns:
            Short-range potential tensor ``(n_edges, 3, 3)``.
        """
        if self.smearing is None:
            raise ValueError(
                "Cannot compute range-separated potential when `smearing` is not specified."
            )
        if self.exclusion_radius is None:
            result = self.from_dist(dist) - self.lr_from_dist(dist)
        else:
            result = -self.lr_from_dist(dist) * self.f_cutoff(dist).unsqueeze(-1)

        return result

    def lr_from_dist(self, dist: torch.Tensor) -> torch.Tensor:
        """Long-range part of the range-separated dipolar potential.

        Args:
            dist: Pair vectors ``(n_edges, 3)``.

        Returns:
            Long-range potential tensor ``(n_edges, 3, 3)``.
        """
        if self.smearing is None:
            raise ValueError(
                "Cannot compute long-range contribution without specifying `smearing`."
            )
        alpha = 1 / (2 * self.smearing**2)
        r_mag = torch.norm(dist, dim=1, keepdim=True)
        r_outer = torch.bmm(dist.unsqueeze(2), dist.unsqueeze(1))
        B1 = torch.erfc(torch.sqrt(alpha) * r_mag) / r_mag**3
        B2 = 2 * torch.sqrt(alpha / torch.pi) * torch.exp(-alpha * r_mag**2) / r_mag**2
        B = 1.0 / (r_mag**3) - B1 - B2
        C1 = 3.0 * torch.erfc(torch.sqrt(alpha) * r_mag) / r_mag**5
        C2 = (
            2
            * torch.sqrt(alpha / torch.pi)
            * (2 * alpha + 3 / r_mag**2)
            * torch.exp(-alpha * r_mag**2)
            / r_mag**2
        )
        C = 3.0 / (r_mag**5) - C1 - C2
        return self.prefactor * (
            B.unsqueeze(-1) * torch.eye(3).to(r_outer).unsqueeze(0) - r_outer * C.unsqueeze(-1)
        )

    def lr_from_k_sq(self, k_sq: torch.Tensor) -> torch.Tensor:
        """Fourier transform of the long-range dipolar potential.

        Args:
            k_sq: Squared k-vector norms ``(...)``.

        Returns:
            Fourier-domain potential values.
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
        """Self-interaction correction for dipoles.

        Returns:
            Scalar correction.
        """
        if self.smearing is None:
            raise ValueError(
                "Cannot compute long-range contribution without specifying `smearing`."
            )
        alpha = 1 / (2 * self.smearing**2)
        return self.prefactor * 4 * torch.pi / 3 * torch.sqrt((alpha / torch.pi) ** 3)

    def background_correction(self, volume) -> torch.Tensor:
        """Background correction for dipolar systems.

        Args:
            volume: Cell volume (scalar).

        Returns:
            Scalar correction.
        """
        if self.epsilon == 0.0:
            return self.epsilon
        return self.prefactor * 4 * torch.pi / (2 * self.epsilon + 1) / volume
