"""Unit tests for molrep.chem.equivalence."""

from __future__ import annotations

import torch

from molrep.chem.equivalence import Equivalence


class TestEquivalence:
    def test_rows_of_one_class_become_their_mean(self):
        features = torch.tensor([[1.0], [3.0], [5.0], [7.0]], dtype=torch.float64)
        out = Equivalence()(features, torch.tensor([0, 0, 2, 3]))
        assert out[:, 0].tolist() == [2.0, 2.0, 5.0, 7.0]

    def test_a_bond_class_is_the_unordered_pair_of_its_endpoint_classes(self):
        # Water: O(0) with two equivalent H (1, 2); a third, distinct atom 3 bonded to O.
        atom_class = torch.tensor([0, 1, 1, 3])
        i, j = torch.tensor([0, 2, 0]), torch.tensor([1, 0, 3])
        bonds = Equivalence().bonds(i, j, atom_class)
        assert bonds[0] == bonds[1] and bonds[0] != bonds[2]
