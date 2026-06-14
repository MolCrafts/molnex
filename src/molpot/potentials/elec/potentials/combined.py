"""Linear combination of multiple potentials with learnable weights."""

from typing import Optional

import torch

from molpot.potentials.elec.potentials.potential import Potential


class CombinedPotential(Potential):
    """Potential that is a weighted linear combination of multiple potentials.

    Weights can be fixed or trainable, enabling learned composition of
    interaction terms.

    Args:
        potentials: List of :class:`Potential` instances to combine.
        initial_weights: Initial combination weights ``(n_potentials,)``.
            Defaults to ones.
        learnable_weights: If True, weights are ``nn.Parameter``.
        smearing: Global smearing (required if any child has smearing).
        exclusion_radius: Distance within which the potential is smoothly zeroed.
        exclusion_degree: Sharpness of the exclusion cutoff.
    """

    def __init__(
        self,
        potentials: list[Potential],
        initial_weights: Optional[torch.Tensor] = None,
        learnable_weights: Optional[bool] = True,
        smearing: Optional[float] = None,
        exclusion_radius: Optional[float] = None,
        exclusion_degree: int = 1,
    ):
        super().__init__(
            smearing=smearing,
            exclusion_radius=exclusion_radius,
            exclusion_degree=exclusion_degree,
        )

        smearings = [pot.smearing for pot in potentials]
        if not all(smearings) and any(smearings):
            raise ValueError(
                "Cannot combine direct (`smearing=None`) and range-separated "
                "(`smearing=float`) potentials."
            )

        if all(smearings) and not self.smearing:
            raise ValueError(
                "You should specify a `smearing` when combining range-separated "
                "(`smearing=float`) potentials."
            )
        if not any(smearings) and self.smearing:
            raise ValueError(
                "Cannot specify `smearing` when combining direct (`smearing=None`) potentials."
            )

        if initial_weights is not None:
            if len(initial_weights) != len(potentials):
                raise ValueError(
                    "The number of initial weights must match the number of "
                    "potentials being combined"
                )
        else:
            initial_weights = torch.ones(len(potentials))

        self.potentials = torch.nn.ModuleList(potentials)
        if learnable_weights:
            self.weights = torch.nn.Parameter(initial_weights)
        else:
            self.register_buffer("weights", initial_weights)

    def from_dist(
        self, dist: torch.Tensor, pair_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Weighted combination of full potentials.

        Args:
            dist: Interatomic distances ``(n_edges,)``.
            pair_mask: Optional boolean mask ``(n_edges,)``.

        Returns:
            Combined potential values ``(n_edges,)``.
        """
        pots = [pot.from_dist(dist, pair_mask) for pot in self.potentials]
        pots = torch.stack(pots, dim=-1)
        return torch.inner(self.weights, pots)

    def sr_from_dist(
        self, dist: torch.Tensor, pair_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Weighted combination of short-range parts.

        Args:
            dist: Interatomic distances ``(n_edges,)``.
            pair_mask: Optional boolean mask ``(n_edges,)``.

        Returns:
            Combined short-range values ``(n_edges,)``.
        """
        pots = [pot.sr_from_dist(dist, pair_mask) for pot in self.potentials]
        pots = torch.stack(pots, dim=-1)
        return torch.inner(self.weights, pots)

    def lr_from_dist(
        self, dist: torch.Tensor, pair_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Weighted combination of long-range real-space parts.

        Args:
            dist: Interatomic distances ``(n_edges,)``.
            pair_mask: Optional boolean mask ``(n_edges,)``.

        Returns:
            Combined long-range values ``(n_edges,)``.
        """
        pots = [pot.lr_from_dist(dist, pair_mask) for pot in self.potentials]
        pots = torch.stack(pots, dim=-1)
        return torch.inner(self.weights, pots)

    def lr_from_k_sq(self, k_sq: torch.Tensor) -> torch.Tensor:
        """Weighted combination of Fourier-domain kernels.

        Args:
            k_sq: Squared k-vector norms ``(...)``.

        Returns:
            Combined Fourier-domain values.
        """
        pots = [pot.lr_from_k_sq(k_sq) for pot in self.potentials]
        pots = torch.stack(pots, dim=-1)
        return torch.inner(self.weights, pots)

    def self_contribution(self) -> torch.Tensor:
        """Weighted combination of self corrections.

        Returns:
            Combined self correction.
        """
        pots = [pot.self_contribution() for pot in self.potentials]
        pots = torch.stack(pots, dim=-1)
        return torch.inner(self.weights, pots)

    def background_correction(self) -> torch.Tensor:
        """Weighted combination of background corrections.

        Returns:
            Combined background correction.
        """
        pots = [pot.background_correction() for pot in self.potentials]
        pots = torch.stack(pots, dim=-1)
        return torch.inner(self.weights, pots)
