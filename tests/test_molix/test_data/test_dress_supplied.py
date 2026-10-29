"""Unit tests for ``molix.data.dress_supplied.SuppliedDress`` — a published table."""

from __future__ import annotations

import pytest
import torch

from molix.data import EnergyResidual
from molix.data.dress_least_squares import LeastSquaresDress
from molix.data.dress_supplied import SuppliedDress

# eV.
_C1 = -13.5
_C8 = -2043.75


class TestSuppliedDress:
    def test_the_numbers_of_a_fit_share_its_task_id(self) -> None:
        """Identity is the table's bytes, not the algorithm that produced them."""
        fitted = LeastSquaresDress([1, 8]).fit(
            torch.tensor([[2, 0], [2, 1]]),
            torch.tensor([2 * _C1, 2 * _C1 + _C8], dtype=torch.float64),
        )
        supplied = SuppliedDress({1: float(fitted.e0[1]), 8: float(fitted.e0[8])}).table
        assert supplied.digest == fitted.digest
        assert EnergyResidual(supplied).task_id == EnergyResidual(fitted).task_id

    def test_an_empty_mapping_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            SuppliedDress({})

    def test_an_explicit_zero_is_supported(self) -> None:
        table = SuppliedDress({1: 0.0, 8: _C8}).table
        assert table.support.nonzero().flatten().tolist() == [1, 8]
        assert float(table.e0[1]) == 0.0
