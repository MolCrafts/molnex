"""Pair Potential terms owned by :class:`Potentials` assembly.

Cutoff and kernel parameters live **on the Potential**, not on Potentials.
"""

from __future__ import annotations

from typing import Literal

import torch

from molpot.potentials.elec.calculators import EwaldCalculator, P3MCalculator, PMECalculator
from molpot.potentials.elec.calculators.calculator import Calculator
from molpot.potentials.elec.potentials import CoulombPotential
from molpot.potentials.elec.prefactors import kcalmol_A
from molpot.potentials.vdw.lj126 import lj126_pair_energy, lorentz_berthelot

__all__ = [
    "LJCutPotential",
    "CoulombOpenPotential",
    "CoulombPeriodicPotential",
]


class LJCutPotential:
    """Lennard-Jones 12-6 with explicit real-space cutoff.

    Wraps :func:`lj126_pair_energy` + Lorentz–Berthelot. Pairs with
    ``r >= cutoff`` are zeroed (caller may also pre-filter the list).
    """

    def __init__(self, cutoff: float = 10.0) -> None:
        if cutoff <= 0:
            raise ValueError(f"LJ cutoff must be positive, got {cutoff}")
        self.cutoff = float(cutoff)

    def energy(
        self,
        *,
        edge_index: torch.Tensor,
        distances: torch.Tensor,
        epsilon: torch.Tensor,
        sigma: torch.Tensor,
        scales: torch.Tensor,
        edge_batch: torch.Tensor | None = None,
        num_graphs: int | None = None,
    ) -> torch.Tensor:
        """Scalar LJ energy for a half neighbor list."""
        within = distances < self.cutoff
        ei = edge_index[within]
        dist = distances[within]
        sc = scales[within]
        mixed = lorentz_berthelot({"epsilon": epsilon, "sigma": sigma}, ei)
        e_pair = lj126_pair_energy(dist, mixed["epsilon_ij"], mixed["sigma_ij"])
        pair_energy = e_pair * sc
        if edge_batch is None:
            return pair_energy.sum()
        energy = pair_energy.new_zeros(int(num_graphs))
        energy.index_add_(0, edge_batch[within].long(), pair_energy)
        return energy


class CoulombOpenPotential:
    """Open-boundary Coulomb using :class:`CoulombPotential`.

    Optional real-space ``cutoff`` (None = all intramolecular pairs).
    Prefactor defaults to Class-I real units (``kcalmol_A``).
    """

    def __init__(
        self,
        *,
        cutoff: float | None = None,
        prefactor: float = kcalmol_A,
    ) -> None:
        if cutoff is not None and cutoff <= 0:
            raise ValueError(f"Coulomb cutoff must be positive, got {cutoff}")
        self.cutoff = cutoff
        self.potential = CoulombPotential(prefactor=prefactor)

    def energy(
        self,
        *,
        edge_index: torch.Tensor,
        distances: torch.Tensor,
        charges: torch.Tensor,
        scales: torch.Tensor,
        edge_batch: torch.Tensor | None = None,
        num_graphs: int | None = None,
    ) -> torch.Tensor:
        if self.cutoff is not None:
            within = distances < self.cutoff
            edge_index = edge_index[within]
            distances = distances[within]
            scales = scales[within]
            if edge_batch is not None:
                edge_batch = edge_batch[within]
        v = self.potential.from_dist(distances)
        qi = charges[edge_index[:, 0]]
        qj = charges[edge_index[:, 1]]
        pair_energy = scales * qi * qj * v
        if edge_batch is None:
            return pair_energy.sum()
        energy = pair_energy.new_zeros(int(num_graphs))
        energy.index_add_(0, edge_batch.long(), pair_energy)
        return energy


class CoulombPeriodicPotential:
    """Periodic Coulomb: :class:`CoulombPotential` + Ewald/PME/P3M calculator.

    The pair potential owns smearing / mesh parameters; the calculator is
    selected by algorithm name (not a product brand).
    """

    def __init__(
        self,
        *,
        smearing: float = 5.0,
        calculator: Literal["ewald", "pme", "p3m"] = "ewald",
        lr_wavelength: float = 1.0,
        mesh_spacing: float = 1.0,
        prefactor: float = kcalmol_A,
        real_space_cutoff: float | None = None,
    ) -> None:
        if smearing <= 0:
            raise ValueError("smearing must be positive")
        self.smearing = float(smearing)
        self.calculator_name = calculator
        self.lr_wavelength = float(lr_wavelength)
        self.mesh_spacing = float(mesh_spacing)
        # real-space neighbor cutoff; default ~3*smearing when unset
        self.real_space_cutoff = (
            float(real_space_cutoff) if real_space_cutoff is not None else 3.0 * self.smearing
        )
        self.potential = CoulombPotential(smearing=self.smearing, prefactor=prefactor)
        self._calc: Calculator = self._build_calculator()

    def _build_calculator(self) -> Calculator:
        name = self.calculator_name
        if name == "ewald":
            return EwaldCalculator(self.potential, lr_wavelength=self.lr_wavelength)
        if name == "pme":
            return PMECalculator(self.potential, mesh_spacing=self.mesh_spacing)
        if name == "p3m":
            return P3MCalculator(self.potential, mesh_spacing=self.mesh_spacing)
        raise ValueError(f"unknown electrostatics calculator {name!r}; expected ewald|pme|p3m")

    def energy(
        self,
        *,
        positions: torch.Tensor,
        charges: torch.Tensor,
        cell: torch.Tensor,
        edge_index: torch.Tensor,
        distances: torch.Tensor,
        scales: torch.Tensor,
    ) -> torch.Tensor:
        """Total electrostatic energy (real + reciprocal)."""
        # Drop fully excluded pairs from real-space list
        keep = scales > 0
        edge_index = edge_index[keep]
        distances = distances[keep]
        scales = scales[keep]

        # Restrict to real-space cutoff on the Potential
        within = distances < self.real_space_cutoff
        edge_index = edge_index[within]
        distances = distances[within]
        scales = scales[within]

        q = charges
        charges_c = q.unsqueeze(-1)
        cell = cell.to(dtype=positions.dtype, device=positions.device)
        periodic = torch.tensor([True, True, True], device=positions.device)

        if edge_index.numel() == 0:
            # k-space only path: pass empty neighbors
            empty_idx = torch.zeros(0, 2, dtype=torch.long, device=positions.device)
            empty_d = torch.zeros(0, dtype=positions.dtype, device=positions.device)
            phi = self._calc(
                charges=charges_c,
                cell=cell,
                positions=positions,
                neighbor_indices=empty_idx,
                neighbor_distances=empty_d,
                periodic=periodic,
            )
            return (charges_c * phi).sum()

        phi = self._calc(
            charges=charges_c,
            cell=cell,
            positions=positions,
            neighbor_indices=edge_index,
            neighbor_distances=distances,
            periodic=periodic,
        )
        e_full = (charges_c * phi).sum()
        if torch.allclose(scales, torch.ones_like(scales)):
            return e_full
        # Adjust SR for 1–4 scales (k-space remains unscaled)
        v_sr = self.potential.sr_from_dist(distances)
        e_sr_calc = (q[edge_index[:, 0]] * q[edge_index[:, 1]] * v_sr).sum()
        e_sr_scaled = (scales * q[edge_index[:, 0]] * q[edge_index[:, 1]] * v_sr).sum()
        return e_full - e_sr_calc + e_sr_scaled
