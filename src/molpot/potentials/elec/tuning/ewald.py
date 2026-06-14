"""Parameter tuning for EwaldCalculator."""

import math
from typing import Any
from warnings import warn

import torch

from molpot.potentials.elec.calculators import EwaldCalculator
from molpot.potentials.elec.tuning.tuner import GridSearchTuner, TuningErrorBounds


def tune_ewald(
    charges: torch.Tensor,
    cell: torch.Tensor,
    positions: torch.Tensor,
    cutoff: float,
    neighbor_indices: torch.Tensor,
    neighbor_distances: torch.Tensor,
    full_neighbor_list: bool = False,
    prefactor: float = 1.0,
    exponent: int = 1,
    ns_lo: int = 1,
    ns_hi: int = 14,
    accuracy: float = 1e-3,
) -> tuple[float, dict[str, Any], float]:
    """Find optimal parameters for :class:`EwaldCalculator`.

    Args:
        charges: Atomic charges ``(n_atoms, 1)``.
        cell: Unit cell matrix ``(3, 3)``.
        positions: Cartesian coordinates ``(n_atoms, 3)``.
        cutoff: Real-space cutoff distance.
        neighbor_indices: Edge index pairs ``(n_edges, 2)``.
        neighbor_distances: Edge distances ``(n_edges,)``.
        full_neighbor_list: If True, full neighbor list expected.
        prefactor: Electrostatics prefactor.
        exponent: Exponent for potential (only 1 supported).
        ns_lo: Minimum k-space resolution along each axis.
        ns_hi: Maximum k-space resolution along each axis.
        accuracy: Desired accuracy (1e-3 balanced, 1e-6 accurate).

    Returns:
        Tuple of ``(smearing, best_params, best_timing)``.
    """
    min_dimension = float(torch.min(torch.linalg.norm(cell, dim=1)))
    params = [{"lr_wavelength": min_dimension / ns} for ns in range(ns_lo, ns_hi + 1)]

    tuner = GridSearchTuner(
        charges=charges,
        cell=cell,
        positions=positions,
        cutoff=cutoff,
        exponent=exponent,
        neighbor_indices=neighbor_indices,
        neighbor_distances=neighbor_distances,
        full_neighbor_list=full_neighbor_list,
        prefactor=prefactor,
        calculator=EwaldCalculator,
        error_bounds=EwaldErrorBounds(charges=charges, cell=cell, positions=positions),
        params=params,
    )
    smearing = tuner.estimate_smearing(accuracy)
    errs, timings = tuner.tune(accuracy)

    if any(err < accuracy for err in errs):
        return smearing, params[timings.index(min(timings))], min(timings)
    warn(
        f"No parameter meets the accuracy requirement.\n"
        f"Returning the parameter with the smallest error, which is {min(errs)}.\n",
        stacklevel=1,
    )
    return smearing, params[errs.index(min(errs))], timings[errs.index(min(errs))]


class EwaldErrorBounds(TuningErrorBounds):
    """Error bounds for :class:`EwaldCalculator`.

    Args:
        charges: Atomic charges ``(n_atoms, 1)``.
        cell: Unit cell matrix ``(3, 3)``.
        positions: Cartesian coordinates ``(n_atoms, 3)``.
    """

    def __init__(
        self,
        charges: torch.Tensor,
        cell: torch.Tensor,
        positions: torch.Tensor,
    ):
        super().__init__(charges, cell, positions)

        self.volume = torch.abs(torch.det(cell))
        self.sum_squared_charges = (charges**2).sum()
        self.prefac = 2 * self.sum_squared_charges / math.sqrt(len(positions))
        self.cell = cell
        self.positions = positions

    def err_kspace(self, smearing: torch.Tensor, lr_wavelength: torch.Tensor) -> torch.Tensor:
        """Fourier-space error for Ewald.

        Args:
            smearing: Gaussian smearing parameter.
            lr_wavelength: Long-range wavelength cutoff.

        Returns:
            Estimated k-space error.
        """
        return (
            self.prefac**0.5
            / smearing
            / torch.pi
            / torch.sqrt(self.volume / lr_wavelength)
            * torch.exp(-2 * (torch.pi * smearing / lr_wavelength) ** 2)
        )

    def err_rspace(self, smearing: torch.Tensor, cutoff: torch.Tensor) -> torch.Tensor:
        """Real-space error for Ewald.

        Args:
            smearing: Gaussian smearing parameter.
            cutoff: Real-space cutoff distance.

        Returns:
            Estimated real-space error.
        """
        return (
            self.prefac
            / torch.sqrt(cutoff * self.volume)
            * torch.exp(-(cutoff**2) / 2 / smearing**2)
        )

    def forward(self, smearing: float, lr_wavelength: float, cutoff: float) -> torch.Tensor:
        """Total error bound for Ewald.

        Args:
            smearing: Gaussian smearing parameter.
            lr_wavelength: Long-range wavelength cutoff.
            cutoff: Real-space cutoff distance.

        Returns:
            Total error estimate.
        """
        smearing = torch.tensor(smearing)
        lr_wavelength = torch.tensor(lr_wavelength)
        cutoff = torch.tensor(cutoff)
        return torch.sqrt(
            self.err_kspace(smearing, lr_wavelength) ** 2 + self.err_rspace(smearing, cutoff) ** 2
        )
