"""Ewald summation calculator — O(N²) exact reciprocal-space sum."""

from typing import Optional

import torch

from molpot.potentials.elec.calculators.calculator import Calculator
from molpot.potentials.elec.lib.kvectors import generate_kvectors_for_ewald
from molpot.potentials.elec.potentials import Potential


class EwaldCalculator(Calculator):
    """Potential computed via the Ewald summation method.

    Scaling: :math:`\\mathcal{O}(N^2)` with particle count. Uses explicit
    trigonometric summation over all reciprocal-space vectors.

    Args:
        potential: :class:`Potential` with smearing set (controls SR/LR split).
        lr_wavelength: Spatial resolution for the reciprocal-space sum.
            All k-vectors with wavelength >= this value are kept.
        full_neighbor_list: If True, neighbor list contains both (i,j) and (j,i).
    """

    def __init__(
        self,
        potential: Potential,
        lr_wavelength: float,
        full_neighbor_list: bool = False,
    ):
        super().__init__(potential=potential, full_neighbor_list=full_neighbor_list)
        if potential.smearing is None:
            raise ValueError("Must specify range radius to use a potential with EwaldCalculator")
        if potential.smearing <= 0:
            raise ValueError(f"`smearing` is {potential.smearing} but must be positive")

        if lr_wavelength <= 0:
            raise ValueError(f"`lr_wavelength` is {lr_wavelength} but must be positive")
        self.lr_wavelength: float = lr_wavelength

    def _compute_kspace(
        self,
        charges: torch.Tensor,
        cell: torch.Tensor,
        positions: torch.Tensor,
        periodic: Optional[torch.Tensor] = None,
        kvectors: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the reciprocal-space Ewald contribution.

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
        if kvectors is None:
            k_cutoff = 2 * torch.pi / self.lr_wavelength

            basis_norms = torch.linalg.norm(cell, dim=1)
            ns_float = k_cutoff * basis_norms / 2 / torch.pi
            ns = torch.ceil(ns_float).long()

            kvectors = generate_kvectors_for_ewald(ns=ns, cell=cell)

        knorm_sq = torch.sum(kvectors**2, dim=-1)

        G = self.potential.lr_from_k_sq(knorm_sq)

        trig_args = kvectors @ (positions.T)  # [k, i]

        c = torch.cos(trig_args)  # [k, i]
        s = torch.sin(trig_args)  # [k, i]
        sc = torch.stack([c, s], dim=0)  # [2 "f", k, i]
        sc_summed_G = torch.einsum("fki,ic, k->fkc", sc, charges, G)
        energy = torch.einsum("fkc,fki->ic", sc_summed_G, sc)
        energy /= torch.abs(cell.det())

        energy -= charges * self.potential.self_contribution()

        ivolume = torch.abs(cell.det()).pow(-1)
        charge_tot = torch.sum(charges, dim=0)
        prefac = self.potential.background_correction()
        energy -= 2 * prefac * charge_tot * ivolume
        energy += self.potential.pbc_correction(periodic, positions, cell, charges)
        if node_mask is not None:
            energy = energy * node_mask.unsqueeze(-1)
        return energy / 2
