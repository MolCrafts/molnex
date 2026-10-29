"""A published elemental dress, taken as given and never refitted.

Numbers someone else computed — isolated-atom energies copied from a paper, a
table MolHub recorded beside the dataset, a previous run's fit — enter as a
:class:`SuppliedDress`. It has no ``fit``, so nothing can overwrite it the way
PiNN's ``init_params`` refits a supplied ``e_dress``.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from molix.data.dress import AtomicDress


class SuppliedDress:
    """``c[Z]`` from a mapping of atomic number to energy.

    Args:
        coefficients: ``{Z: c[Z]}`` in the unit of ``graphs.energy`` (eV).
            An explicit ``0.0`` is a coefficient; an absent ``Z`` is
            unsupported. No dress is no :class:`~molix.data.EnergyResidual`,
            not an empty mapping.

    Attributes:
        table: The :class:`~molix.data.AtomicDress`, length ``max(Z) + 1``.

    Raises:
        ValueError: If *coefficients* is empty or has a negative ``Z``.
    """

    def __init__(self, coefficients: Mapping[int, float]) -> None:
        if not coefficients:
            raise ValueError("SuppliedDress needs at least one coefficient")
        numbers = [int(z) for z in coefficients]
        if min(numbers) < 0:
            raise ValueError(f"SuppliedDress atomic numbers must be >= 0 (got {numbers})")
        e0 = torch.zeros(max(numbers) + 1, dtype=torch.float64)
        support = torch.zeros(max(numbers) + 1, dtype=torch.bool)
        for z, value in coefficients.items():
            e0[int(z)] = float(value)
            support[int(z)] = True
        self.table = AtomicDress(e0=e0, support=support)
