"""Parameter tuning for P3MCalculator."""

import math
from itertools import product
from typing import Any
from warnings import warn

import torch

from molpot.potentials.elec.calculators import P3MCalculator
from molpot.potentials.elec.tuning.tuner import GridSearchTuner, TuningErrorBounds

# Coefficients for the P3M Fourier error, Table II of
# http://dx.doi.org/10.1063/1.477415
A_COEF = [
    [None, 2 / 3, 1 / 50, 1 / 588, 1 / 4320, 1 / 23_232, 691 / 68_140_800, 1 / 345_600],
    [
        None,
        None,
        5 / 294,
        7 / 1440,
        3 / 1936,
        7601 / 13_628_160,
        13 / 57_600,
        3617 / 35_512_320,
    ],
    [
        None,
        None,
        None,
        21 / 3872,
        7601 / 2_271_360,
        143 / 69_120,
        47_021 / 35_512_320,
        745_739 / 838_397_952,
    ],
    [
        None,
        None,
        None,
        None,
        143 / 28_800,
        517_231 / 106_536_960,
        9_694_607 / 2_095_994_880,
        56_399_353 / 12_773_376_000,
    ],
    [
        None,
        None,
        None,
        None,
        None,
        106_640_677 / 11_737_571_328,
        733_191_589 / 59_609_088_000,
        25_091_609 / 1_560_084_480,
    ],
    [
        None,
        None,
        None,
        None,
        None,
        None,
        326_190_917 / 11_700_633_600,
        1_755_948_832_039 / 36_229_939_200_000,
    ],
    [None, None, None, None, None, None, None, 4_887_769_399 / 37_838_389_248],
]


def tune_p3m(
    charges: torch.Tensor,
    cell: torch.Tensor,
    positions: torch.Tensor,
    cutoff: float,
    neighbor_indices: torch.Tensor,
    neighbor_distances: torch.Tensor,
    full_neighbor_list: bool = False,
    prefactor: float = 1.0,
    exponent: int = 1,
    nodes_lo: int = 2,
    nodes_hi: int = 5,
    mesh_lo: int = 2,
    mesh_hi: int = 7,
    accuracy: float = 1e-3,
) -> tuple[float, dict[str, Any], float]:
    r"""Find optimal parameters for :class:`P3MCalculator`.

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
        nodes_lo: Minimum interpolation nodes (2-5 for P3M).
        nodes_hi: Maximum interpolation nodes (2-5 for P3M).
        mesh_lo: Minimum mesh points as :math:`2^{mesh\_lo}`.
        mesh_hi: Maximum mesh points as :math:`2^{mesh\_hi}`.
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
        calculator=P3MCalculator,
        error_bounds=P3MErrorBounds(charges=charges, cell=cell, positions=positions),
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


class P3MErrorBounds(TuningErrorBounds):
    """Error bounds for :class:`P3MCalculator`.

    Reference: Deserno, M. & Holm, C. J. Chem. Phys. 109, 7678–7693 (1998)

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
        self.cell = cell
        self.positions = positions

    def err_kspace(
        self,
        smearing: torch.Tensor,
        mesh_spacing: torch.Tensor,
        interpolation_nodes: torch.Tensor,
    ) -> torch.Tensor:
        """Fourier-space error of P3M.

        Args:
            smearing: Gaussian smearing parameter.
            mesh_spacing: Mesh spacing.
            interpolation_nodes: Number of interpolation nodes.

        Returns:
            Estimated k-space error.
        """
        actual_spacing = self.cell_dimensions / (2 * self.cell_dimensions / mesh_spacing + 1)
        h = torch.prod(actual_spacing) ** (1 / 3)

        return (
            self.prefac
            / self.volume ** (2 / 3)
            * (h * (1 / 2**0.5 / smearing)) ** interpolation_nodes
            * torch.sqrt(
                (1 / 2**0.5 / smearing)
                * self.volume ** (1 / 3)
                * math.sqrt(2 * torch.pi)
                * sum(
                    A_COEF[m][interpolation_nodes] * (h * (1 / 2**0.5 / smearing)) ** (2 * m)
                    for m in range(interpolation_nodes)
                )
            )
        )

    def err_rspace(self, smearing: torch.Tensor, cutoff: torch.Tensor) -> torch.Tensor:
        """Real-space error of P3M.

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

    def forward(
        self,
        smearing: float,
        mesh_spacing: float,
        cutoff: float,
        interpolation_nodes: int,
    ) -> torch.Tensor:
        """Total error bound for P3M.

        Args:
            smearing: Gaussian smearing parameter.
            mesh_spacing: Mesh spacing.
            cutoff: Real-space cutoff distance.
            interpolation_nodes: Number of interpolation nodes.

        Returns:
            Total error estimate.
        """
        smearing = torch.tensor(smearing)
        mesh_spacing = torch.tensor(mesh_spacing)
        cutoff = torch.tensor(cutoff)
        interpolation_nodes = torch.tensor(interpolation_nodes)
        return torch.sqrt(
            self.err_kspace(smearing, mesh_spacing, interpolation_nodes) ** 2
            + self.err_rspace(smearing, cutoff) ** 2
        )
