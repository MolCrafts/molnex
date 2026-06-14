"""Parameter tuning framework for Ewald/PME/P3M calculators."""

import math
import time
from typing import Optional

import torch

from molpot.potentials.elec._utils import _validate_parameters
from molpot.potentials.elec.calculators import Calculator
from molpot.potentials.elec.potentials import InversePowerLawPotential


class TuningErrorBounds(torch.nn.Module):
    """Base class for error bound estimation.

    Calculates real-space and Fourier-space errors for parameter tuning.

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
        super().__init__()
        self._charges = charges
        self._cell = cell
        self._positions = positions

    def forward(self, *args, **kwargs):
        """Alias for :meth:`error` so the bound can be called as a module."""
        return self.error(*args, **kwargs)

    def error(self, *args, **kwargs):
        """Return the total estimated error bound; implemented by subclasses.

        Raises:
            NotImplementedError: This base class defines no error formula.
        """
        raise NotImplementedError


class TunerBase:
    """Base class for parameter tuning.

    Estimates smearing from the real-space cutoff and provides
    the tuning interface for subclasses.

    Args:
        charges: Atomic charges ``(n_atoms, 1)``.
        cell: Unit cell matrix ``(3, 3)``.
        positions: Cartesian coordinates ``(n_atoms, 3)``.
        cutoff: Real-space cutoff distance.
        calculator: Calculator class to tune.
        exponent: Exponent for :math:`1/r^p` potential (only 1 supported).
        full_neighbor_list: If True, full neighbor list expected.
        prefactor: Electrostatics prefactor.
    """

    def __init__(
        self,
        charges: torch.Tensor,
        cell: torch.Tensor,
        positions: torch.Tensor,
        cutoff: float,
        calculator: type[Calculator],
        exponent: int = 1,
        full_neighbor_list: bool = False,
        prefactor: float = 1.0,
    ):
        if exponent != 1:
            raise NotImplementedError(f"Only exponent = 1 is supported but got {exponent}.")

        _validate_parameters(
            charges=charges,
            cell=cell,
            positions=positions,
            neighbor_indices=torch.tensor([[0, 1]], device=positions.device),
            neighbor_distances=torch.tensor([1.0], device=positions.device, dtype=positions.dtype),
        )
        self.charges = charges
        self.cell = cell
        self.positions = positions
        self.cutoff = cutoff
        self.calculator = calculator
        self.exponent = exponent
        self.full_neighbor_list = full_neighbor_list
        self.prefactor = prefactor

        self._smearing_esti_prefac = 2 * float((charges**2).sum()) / math.sqrt(len(positions))

    def tune(self, accuracy: float = 1e-3):
        """Search for calculator parameters meeting ``accuracy``; subclass-defined.

        Args:
            accuracy: Target error bound the tuned parameters should satisfy.

        Raises:
            NotImplementedError: This base class defines no tuning strategy.
        """
        raise NotImplementedError

    def estimate_smearing(self, accuracy: float) -> float:
        """Estimate smearing from the real-space error formula.

        Sets smearing so that the real-space error is approximately
        ``accuracy / 4``.

        Args:
            accuracy: Desired total accuracy.

        Returns:
            Estimated smearing value.
        """
        if not isinstance(accuracy, float):
            raise ValueError(f"'{accuracy}' is not a float.")
        ratio = math.sqrt(
            -2
            * math.log(
                accuracy
                / 2
                / self._smearing_esti_prefac
                * math.sqrt(self.cutoff * float(torch.abs(self.cell.det())))
            )
        )
        smearing = self.cutoff / ratio

        return float(smearing)

    @staticmethod
    def filter_neighbors(
        cutoff: float, neighbor_indices: torch.Tensor, neighbor_distances: torch.Tensor
    ):
        """Filter neighbor list to include only pairs within the cutoff.

        Args:
            cutoff: Real-space cutoff distance.
            neighbor_indices: Edge index pairs ``(n_edges, 2)``.
            neighbor_distances: Edge distances ``(n_edges,)``.

        Returns:
            Tuple of filtered ``(neighbor_indices, neighbor_distances)``.
        """
        filter_idx = torch.where(neighbor_distances < cutoff)
        return neighbor_indices[filter_idx], neighbor_distances[filter_idx]


class GridSearchTuner(TunerBase):
    """Parameter tuner using grid search over Fourier-space parameters.

    Args:
        charges: Atomic charges ``(n_atoms, 1)``.
        cell: Unit cell matrix ``(3, 3)``.
        positions: Cartesian coordinates ``(n_atoms, 3)``.
        cutoff: Real-space cutoff distance.
        calculator: Calculator class to tune.
        error_bounds: Error bound estimator class.
        params: List of parameter dicts to search over.
        neighbor_indices: Edge index pairs ``(n_edges, 2)``.
        neighbor_distances: Edge distances ``(n_edges,)``.
        full_neighbor_list: If True, full neighbor list expected.
        prefactor: Electrostatics prefactor.
        exponent: Exponent for potential (only 1 supported).
    """

    def __init__(
        self,
        charges: torch.Tensor,
        cell: torch.Tensor,
        positions: torch.Tensor,
        cutoff: float,
        calculator: type[Calculator],
        error_bounds: TuningErrorBounds,
        params: list[dict],
        neighbor_indices: torch.Tensor,
        neighbor_distances: torch.Tensor,
        full_neighbor_list: bool = False,
        prefactor: float = 1.0,
        exponent: int = 1,
    ):
        super().__init__(
            charges=charges,
            cell=cell,
            positions=positions,
            cutoff=cutoff,
            calculator=calculator,
            exponent=exponent,
            full_neighbor_list=full_neighbor_list,
            prefactor=prefactor,
        )
        self.error_bounds = error_bounds
        self.params = params
        neighbor_indices, neighbor_distances = self.filter_neighbors(
            cutoff, neighbor_indices, neighbor_distances
        )
        self.time_func = TuningTimings(
            charges,
            cell,
            positions,
            neighbor_indices,
            neighbor_distances,
            True,
        )

    def tune(self, accuracy: float = 1e-3) -> tuple[list[float], list[float]]:
        """Estimate error and timing for each parameter set.

        Args:
            accuracy: Desired accuracy threshold.

        Returns:
            Tuple of ``(param_errors, param_timings)``.
        """
        if not isinstance(accuracy, float):
            raise ValueError(f"'{accuracy}' is not a float.")
        smearing = self.estimate_smearing(accuracy)
        param_errors = []
        param_timings = []
        for param in self.params:
            error = self.error_bounds(smearing=smearing, cutoff=self.cutoff, **param)
            param_errors.append(float(error))
            param_timings.append(
                self._timing(smearing, param) if error <= accuracy else float("inf")
            )

        return param_errors, param_timings

    def _timing(self, smearing: float, k_space_params: dict):
        calculator = self.calculator(
            potential=InversePowerLawPotential(
                exponent=self.exponent,
                smearing=smearing,
                prefactor=self.prefactor,
            ),
            full_neighbor_list=self.full_neighbor_list,
            **k_space_params,
        )
        calculator.to(device=self.positions.device, dtype=self.positions.dtype)
        return self.time_func(calculator)


class TuningTimings(torch.nn.Module):
    """Benchmark timer for calculator execution.

    Args:
        charges: Atomic charges ``(n_atoms, 1)``.
        cell: Unit cell matrix ``(3, 3)``.
        positions: Cartesian coordinates ``(n_atoms, 3)``.
        neighbor_indices: Edge index pairs ``(n_edges, 2)``.
        neighbor_distances: Edge distances ``(n_edges,)``.
        n_repeat: Number of timing repetitions.
        n_warmup: Number of warmup runs.
        run_backward: If True, include backward pass in timing.
    """

    def __init__(
        self,
        charges: torch.Tensor,
        cell: torch.Tensor,
        positions: torch.Tensor,
        neighbor_indices: torch.Tensor,
        neighbor_distances: torch.Tensor,
        n_repeat: int = 4,
        n_warmup: int = 4,
        run_backward: Optional[bool] = True,
    ):
        super().__init__()

        _validate_parameters(
            charges=charges,
            cell=cell,
            positions=positions,
            neighbor_indices=neighbor_indices,
            neighbor_distances=neighbor_distances,
        )

        self.charges = charges
        self.cell = cell
        self.positions = positions
        self.n_repeat = n_repeat
        self.n_warmup = n_warmup
        self.run_backward = run_backward
        self.neighbor_indices = neighbor_indices
        self.neighbor_distances = neighbor_distances

    def forward(self, calculator: torch.nn.Module) -> float:
        """Estimate average execution time.

        Args:
            calculator: The calculator module to benchmark.

        Returns:
            Average execution time in seconds.
        """
        execution_time = 0.0

        for iteration in range(self.n_repeat + self.n_warmup):
            if iteration == self.n_warmup:
                execution_time = 0.0
            positions = self.positions.clone()
            cell = self.cell.clone()
            charges = self.charges.clone()
            if self.run_backward:
                positions.requires_grad_(True)
                cell.requires_grad_(True)
                charges.requires_grad_(True)
            execution_time -= time.monotonic()
            result = calculator.forward(
                positions=positions,
                charges=charges,
                cell=cell,
                neighbor_indices=self.neighbor_indices,
                neighbor_distances=self.neighbor_distances,
            )
            value = result.sum()
            if self.run_backward:
                value.backward(retain_graph=True)

            execution_time += time.monotonic()

        return execution_time / self.n_repeat
