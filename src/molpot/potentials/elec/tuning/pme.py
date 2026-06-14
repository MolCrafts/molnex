"""Parameter tuning for PMECalculator."""

import math
from itertools import product
from typing import Any
from warnings import warn

import torch

from molpot.potentials.elec.calculators import PMECalculator
from molpot.potentials.elec.tuning.tuner import GridSearchTuner, TuningErrorBounds


def tune_pme(
    charges: torch.Tensor,
    cell: torch.Tensor,
    positions: torch.Tensor,
    cutoff: float,
    neighbor_indices: torch.Tensor,
    neighbor_distances: torch.Tensor,
    full_neighbor_list: bool = False,
    prefactor: float = 1.0,
    exponent: int = 1,
    nodes_lo: int = 3,
    nodes_hi: int = 7,
    mesh_lo: int = 2,
    mesh_hi: int = 7,
    accuracy: float = 1e-3,
) -> tuple[float, dict[str, Any], float]:
    r"""Find optimal parameters for :class:`PMECalculator`.

    Args:
        charges: Atomic charges ``(n_atoms, 1)``.
        cell: Unit cell matrix ``(3, 3)``.
        positions: Cartesian coordinates ``(n_atoms, 3)``.
        cutoff: Real-space cutoff distance.
        neighbor_indices: Edge index pairs ``(n_edges, 2)``.
        neighbor_distances: Edge distances ``(n_edges,)``.
        full_neighbor_list: If True, full neighbor list expected.
        prefactor: Electrostatics prefactor.
        exponent: Exponent (only 1 supported).
        nodes_lo: Minimum interpolation nodes.
        nodes_hi: Maximum interpolation nodes.
        mesh_lo: Minimum mesh points along shortest axis as :math:`2^{mesh\_lo}`.
        mesh_hi: Maximum mesh points along shortest axis as :math:`2^{mesh\_hi}`.
        accuracy: Desired accuracy.

    Returns:
        Tuple of ``(smearing, best_params, best_timing)``.
    """
    min_dimension = float(torch.min(torch.linalg.norm(cell, dim=1)))
    params = [
        {
            "interpolation_nodes": interpolation_nodes,
            "mesh_spacing": 2 * min_dimension / (2**ns - 1),
        }
        for interpolation_nodes, ns in product(
            range(nodes_lo, nodes_hi + 1), range(mesh_lo, mesh_hi + 1)
        )
    ]

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
        calculator=PMECalculator,
        error_bounds=PMEErrorBounds(charges=charges, cell=cell, positions=positions),
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


class PMEErrorBounds(TuningErrorBounds):
    """Error bounds for :class:`PMECalculator`.

    Reference: Darden, T. et al. J. Chem. Phys. 98, 10089–10092 (1993)

    Args:
        charges: Atomic charges ``(n_atoms, 1)``.
        cell: Unit cell matrix ``(3, 3)``.
        positions: Cartesian coordinates ``(n_atoms, 3)``.
    """

    def __init__(self, charges: torch.Tensor, cell: torch.Tensor, positions: torch.Tensor):
        super().__init__(charges, cell, positions)

        self.volume = torch.abs(torch.det(cell))
        self.sum_squared_charges = (charges**2).sum()
        self.prefac = 2 * self.sum_squared_charges / math.sqrt(len(positions))
        self.cell_dimensions = torch.linalg.norm(cell, dim=1)

    def err_kspace(
        self,
        smearing: torch.Tensor,
        mesh_spacing: torch.Tensor,
        interpolation_nodes: torch.Tensor,
    ) -> torch.Tensor:
        """Fourier-space error of PME.

        Args:
            smearing: Gaussian smearing parameter.
            mesh_spacing: Mesh spacing.
            interpolation_nodes: Number of interpolation nodes.

        Returns:
            Estimated k-space error.
        """
        actual_spacing = self.cell_dimensions / (2 * self.cell_dimensions / mesh_spacing + 1)
        h = torch.prod(actual_spacing) ** (1 / 3)
        i_n_factorial = torch.exp(torch.lgamma(interpolation_nodes + 1))
        RMS_phi = [None, None, 0.246, 0.404, 0.950, 2.51, 8.42]

        return (
            self.prefac
            * torch.pi**0.25
            * (6 * (1 / 2**0.5 / smearing) / (2 * interpolation_nodes + 1)) ** 0.5
            / self.volume ** (2 / 3)
            * (2**0.5 / smearing * h) ** interpolation_nodes
            / i_n_factorial
            * torch.exp(interpolation_nodes * (torch.log(interpolation_nodes / 2) - 1) / 2)
            * RMS_phi[interpolation_nodes - 1]
        )

    def err_rspace(self, smearing: torch.Tensor, cutoff: torch.Tensor) -> torch.Tensor:
        """Real-space error of PME.

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

    def error(
        self,
        cutoff: float,
        smearing: float,
        mesh_spacing: float,
        interpolation_nodes: float,
    ) -> torch.Tensor:
        """Total error bound for PME.

        Args:
            cutoff: Real-space cutoff distance.
            smearing: Gaussian smearing parameter.
            mesh_spacing: Mesh spacing.
            interpolation_nodes: Number of interpolation nodes.

        Returns:
            Total error estimate.
        """
        smearing = torch.tensor(smearing)
        mesh_spacing = torch.tensor(mesh_spacing)
        cutoff = torch.tensor(cutoff)
        interpolation_nodes = torch.tensor(interpolation_nodes)
        return torch.sqrt(
            self.err_rspace(smearing, cutoff) ** 2
            + self.err_kspace(smearing, mesh_spacing, interpolation_nodes) ** 2
        )
