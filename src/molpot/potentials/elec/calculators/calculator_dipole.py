"""Calculator for point dipole interactions with Ewald summation."""

from typing import Optional

import torch
from torch import profiler

from molpot.potentials.elec._utils import _validate_parameters
from molpot.potentials.elec.lib.kvectors import generate_kvectors_for_ewald
from molpot.potentials.elec.potentials import PotentialDipole


class CalculatorDipole(torch.nn.Module):
    """Base calculator for interacting point dipoles.

    Computes dipole-dipole interaction energies via real-space neighbor
    summation and optional reciprocal-space Ewald summation.

    Args:
        potential: :class:`PotentialDipole` providing dipole pair potentials.
        full_neighbor_list: If True, full neighbor list expected.
        lr_wavelength: Spatial resolution for the reciprocal-space sum.
    """

    def __init__(
        self,
        potential: PotentialDipole,
        full_neighbor_list: bool = False,
        lr_wavelength: Optional[float] = None,
    ):
        super().__init__()

        if not isinstance(potential, PotentialDipole):
            raise TypeError(
                f"Potential must be an instance of PotentialDipole, got {type(potential)}"
            )

        self.potential = potential
        self.lr_wavelength = lr_wavelength

        assert (
            self.lr_wavelength is not None
            and self.potential.smearing is not None
            or (self.lr_wavelength is None and self.potential.smearing is None)
        ), "Either both `lr_wavelength` and `smearing` must be set or both must be None"

        self.full_neighbor_list = full_neighbor_list

    def _compute_rspace(
        self,
        dipoles: torch.Tensor,
        neighbor_indices: torch.Tensor,
        neighbor_vectors: torch.Tensor,
    ) -> torch.Tensor:
        """Compute real-space dipole-dipole potential.

        Args:
            dipoles: Atomic dipoles ``(n_atoms, 3)``.
            neighbor_indices: Edge index pairs ``(n_edges, 2)``.
            neighbor_vectors: Edge vectors ``(n_edges, 3)``.

        Returns:
            Real-space potential ``(n_atoms, 3)``.
        """
        with profiler.record_function("compute bare potential"):
            if self.potential.smearing is None:
                potentials_bare = self.potential.from_dist(neighbor_vectors)
            else:
                potentials_bare = self.potential.sr_from_dist(neighbor_vectors)

        atom_is = neighbor_indices[:, 0]
        atom_js = neighbor_indices[:, 1]
        with profiler.record_function("compute real potential"):
            contributions_is = torch.bmm(potentials_bare, dipoles[atom_js].unsqueeze(-1)).squeeze(
                -1
            )

        with profiler.record_function("assign potential"):
            potential = torch.zeros_like(dipoles)
            potential.index_add_(0, atom_is, contributions_is)
            if not self.full_neighbor_list:
                contributions_js = torch.bmm(
                    potentials_bare, dipoles[atom_is].unsqueeze(-1)
                ).squeeze(-1)
                potential.index_add_(0, atom_js, contributions_js)

        return potential / 2

    def _compute_kspace(
        self,
        dipoles: torch.Tensor,
        cell: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute reciprocal-space dipole Ewald contribution.

        Args:
            dipoles: Atomic dipoles ``(n_atoms, 3)``.
            cell: Unit cell matrix ``(3, 3)``.
            positions: Cartesian coordinates ``(n_atoms, 3)``.

        Returns:
            K-space potential ``(n_atoms, 3)``.
        """
        k_cutoff = 2 * torch.pi / self.lr_wavelength

        basis_norms = torch.linalg.norm(cell, dim=1)
        ns_float = k_cutoff * basis_norms / 2 / torch.pi
        ns = torch.ceil(ns_float).long()

        kvectors = generate_kvectors_for_ewald(ns=ns, cell=cell)
        knorm_sq = torch.sum(kvectors**2, dim=1)
        G = self.potential.lr_from_k_sq(knorm_sq)

        trig_args = kvectors @ (positions.T)  # [k, i]
        c = torch.cos(trig_args)  # [k, i]
        s = torch.sin(trig_args)  # [k, i]
        sc = torch.stack([c, s], dim=0)  # [2 "f", k, i]
        mu_k = dipoles @ kvectors.T  # [i, k]
        sc_summed_G = torch.einsum("fki, ik, k->fk", sc, mu_k, G)
        energy = torch.einsum("fk, fki, kc->ic", sc_summed_G, sc, kvectors)
        energy /= torch.abs(cell.det())
        energy -= dipoles * self.potential.self_contribution()
        energy += self.potential.background_correction(torch.abs(cell.det())) * dipoles.sum(dim=0)
        return energy / 2

    def forward(
        self,
        dipoles: torch.Tensor,
        cell: torch.Tensor,
        positions: torch.Tensor,
        neighbor_indices: torch.Tensor,
        neighbor_vectors: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the total dipole interaction potential.

        Args:
            dipoles: Atomic dipoles ``(n_atoms, 3)``.
            cell: Unit cell matrix ``(3, 3)``.
            positions: Cartesian coordinates ``(n_atoms, 3)``.
            neighbor_indices: Edge index pairs ``(n_edges, 2)``.
            neighbor_vectors: Edge vectors ``(n_edges, 3)``.

        Returns:
            Total potential ``(n_atoms, 3)``.
        """
        _validate_parameters(
            charges=dipoles,
            cell=cell,
            positions=positions,
            neighbor_indices=neighbor_indices,
            neighbor_distances=neighbor_vectors.norm(dim=-1),
        )

        potential_sr = self._compute_rspace(
            dipoles=dipoles,
            neighbor_indices=neighbor_indices,
            neighbor_vectors=neighbor_vectors,
        )

        if self.potential.smearing is None:
            return potential_sr
        potential_lr = self._compute_kspace(
            dipoles=dipoles,
            cell=cell,
            positions=positions,
        )

        return potential_sr + potential_lr
