"""Unit tests for molrep.chem.bond_table."""

from __future__ import annotations

import pytest
import torch

from molrep.chem.bond_table import BondTable


class TestBondTable:
    def test_a_pair_finds_its_row_in_either_order(self):
        table = BondTable(torch.tensor([0, 1, 3]), torch.tensor([1, 2, 1]), n_atoms=4)
        row, found = table.rows(torch.tensor([2, 1, 1]), torch.tensor([1, 2, 3]))
        assert row.tolist() == [1, 1, 2]
        assert found.tolist() == [True, True, True]

    def test_a_pair_that_is_not_a_bond_gathers_zero(self):
        table = BondTable(torch.tensor([0, 1]), torch.tensor([1, 2]), n_atoms=3)
        h_bond = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        got = table.gather(h_bond, torch.tensor([0, 0]), torch.tensor([2, 1]))
        torch.testing.assert_close(got, torch.tensor([[0.0, 0.0], [1.0, 2.0]]))

    def test_an_empty_table_finds_nothing(self):
        table = BondTable(torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long), 2)
        got = table.gather(torch.zeros(0, 3), torch.tensor([0]), torch.tensor([1]))
        assert got.shape == (1, 3)
        assert len(table) == 0

    def test_refuses_mismatched_columns(self):
        with pytest.raises(ValueError, match="matching"):
            BondTable(torch.tensor([0, 1]), torch.tensor([1]), n_atoms=2)

    def test_refuses_features_of_another_bond_list(self):
        table = BondTable(torch.tensor([0]), torch.tensor([1]), n_atoms=2)
        with pytest.raises(ValueError, match="rows"):
            table.gather(torch.zeros(2, 3), torch.tensor([0]), torch.tensor([1]))
