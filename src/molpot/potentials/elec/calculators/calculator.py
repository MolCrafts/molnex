"""Base calculator for pair potentials with real-space neighbor summation."""

from typing import Optional

import torch
from torch import profiler

from molpot.potentials.elec._utils import _validate_parameters
from molpot.potentials.elec.potentials import Potential


class Calculator(torch.nn.Module):
    """Base calculator combining real-space and reciprocal-space potential evaluation.

    Computes :math:`V_i = \\frac{1}{2} \\sum_j q_j v(r_{ij})` via neighbor list
    summation, optionally adding a Fourier-domain long-range part.

    Args:
        potential: :class:`Potential` providing ``from_dist``, ``sr_from_dist``,
            ``lr_from_dist``, ``lr_from_k_sq``, ``self_contribution``, and
            ``background_correction``.
        full_neighbor_list: If True, neighbor list contains both (i,j) and (j,i).
            Default False (half list).
    """

    def __init__(
        self,
        potential: Potential,
        full_neighbor_list: bool = False,
    ):
        super().__init__()

        if not isinstance(potential, Potential):
            raise TypeError(f"Potential must be an instance of Potential, got {type(potential)}")

        self.potential = potential
        self.full_neighbor_list = full_neighbor_list

    def _compute_rspace(
        self,
        charges: torch.Tensor,
        neighbor_indices: torch.Tensor,
        neighbor_distances: torch.Tensor,
        pair_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the real-space potential via neighbor summation.

        Args:
            charges: Atomic charges ``(n_atoms, n_channels)``.
            neighbor_indices: Edge index pairs ``(n_edges, 2)``.
            neighbor_distances: Edge distances ``(n_edges,)``.
            pair_mask: Optional boolean mask ``(n_edges,)``.

        Returns:
            Real-space potential ``(n_atoms, n_channels)``.
        """
        with profiler.record_function("compute bare potential"):
            if self.potential.smearing is None:
                if self.potential.exclusion_radius is None:
                    potentials_bare = self.potential.from_dist(neighbor_distances, pair_mask)
                else:
                    potentials_bare = self.potential.from_dist(neighbor_distances, pair_mask) * (
                        1 - self.potential.f_cutoff(neighbor_distances, pair_mask)
                    )
            else:
                potentials_bare = self.potential.sr_from_dist(neighbor_distances, pair_mask)

        atom_is = neighbor_indices[:, 0]
        atom_js = neighbor_indices[:, 1]
        with profiler.record_function("compute real potential"):
            contributions_is = charges[atom_js] * potentials_bare.unsqueeze(-1)

        with profiler.record_function("assign potential"):
            potential = torch.zeros_like(charges)
            potential.index_add_(0, atom_is, contributions_is)
            if not self.full_neighbor_list:
                contributions_js = charges[atom_is] * potentials_bare.unsqueeze(-1)
                potential.index_add_(0, atom_js, contributions_js)

        return potential / 2

    def _compute_kspace(
        self,
        charges: torch.Tensor,
        cell: torch.Tensor,
        positions: torch.Tensor,
        periodic: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
        kvectors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the Fourier-domain long-range contribution.

        Args:
            charges: Atomic charges ``(n_atoms, n_channels)``.
            cell: Unit cell matrix ``(3, 3)``.
            positions: Cartesian coordinates ``(n_atoms, 3)``.
            periodic: Boolean mask ``(3,)`` for periodic directions.
            kvectors: Optional precomputed k-vectors ``(n_kvecs, 3)``.
            node_mask: Optional boolean mask ``(n_atoms,)``.

        Returns:
            K-space potential ``(n_atoms, n_channels)``.
        """
        raise NotImplementedError(f"`compute_kspace` not implemented for {self.__class__.__name__}")

    def forward(
        self,
        charges: torch.Tensor,
        cell: torch.Tensor,
        positions: torch.Tensor,
        neighbor_indices: torch.Tensor,
        neighbor_distances: torch.Tensor,
        periodic: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
        pair_mask: Optional[torch.Tensor] = None,
        kvectors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the total potential (real + k-space).

        Args:
            charges: Atomic charges ``(n_atoms, n_channels)``.
            cell: Unit cell matrix ``(3, 3)``.
            positions: Cartesian coordinates ``(n_atoms, 3)``.
            neighbor_indices: Edge index pairs ``(n_edges, 2)``.
            neighbor_distances: Edge distances ``(n_edges,)``.
            periodic: Boolean mask ``(3,)`` for periodic directions.
            node_mask: Optional boolean mask ``(n_atoms,)``.
            pair_mask: Optional boolean mask ``(n_edges,)``.
            kvectors: Optional precomputed k-vectors ``(n_kvecs, 3)``.

        Returns:
            Total potential ``(n_atoms, n_channels)``.
        """
        _validate_parameters(
            charges=charges,
            cell=cell,
            positions=positions,
            neighbor_indices=neighbor_indices,
            neighbor_distances=neighbor_distances,
            periodic=periodic,
            pair_mask=pair_mask,
            node_mask=node_mask,
            kvectors=kvectors,
        )

        potential_sr = self._compute_rspace(
            charges=charges,
            neighbor_indices=neighbor_indices,
            neighbor_distances=neighbor_distances,
            pair_mask=pair_mask,
        )

        if self.potential.smearing is None:
            return potential_sr
        potential_lr = self._compute_kspace(
            charges=charges,
            cell=cell,
            positions=positions,
            periodic=periodic,
            kvectors=kvectors,
            node_mask=node_mask,
        )

        return potential_sr + potential_lr
