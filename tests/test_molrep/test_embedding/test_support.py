"""Tests for :class:`molrep.embedding.support.ChemicalSupportIndex`.

Spec: learnable-classical-ff-09-provenance.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from molrep.embedding.support import ChemicalSupportIndex


class TestChemicalSupportIndex:
    """kNN L2 bank membership and coverage."""

    def test_contains_exact_bank_members(self):
        """Bank points lie inside the support radius (distance 0)."""
        bank = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        index = ChemicalSupportIndex(bank, radius=0.1, k=1)
        query = bank.clone()
        assert torch.equal(index.contains(query), torch.tensor([True, True, True]))

    def test_contains_outside_radius(self):
        """Points farther than radius are out of support."""
        bank = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
        index = ChemicalSupportIndex(bank, radius=0.5, k=1)
        query = torch.tensor(
            [
                [0.0, 0.0],  # dist 0
                [0.4, 0.0],  # dist 0.4 <= 0.5
                [1.0, 0.0],  # dist 1.0 > 0.5
            ],
            dtype=torch.float64,
        )
        assert torch.equal(
            index.contains(query),
            torch.tensor([True, True, False]),
        )

    def test_knn_distances_and_indices(self):
        """Nearest neighbour is the expected bank row under L2."""
        bank = torch.tensor(
            [
                [0.0, 0.0],
                [10.0, 0.0],
                [0.0, 10.0],
            ],
            dtype=torch.float64,
        )
        index = ChemicalSupportIndex(bank, radius=1.0, k=1)
        query = torch.tensor([[0.1, 0.0]], dtype=torch.float64)
        dist, nn_idx = index.knn(query)
        assert dist.shape == (1, 1)
        assert nn_idx.shape == (1, 1)
        assert int(nn_idx[0, 0]) == 0
        assert float(dist[0, 0]) == pytest.approx(0.1, abs=1e-8)

    def test_min_distance(self):
        bank = torch.tensor([[0.0, 0.0], [3.0, 0.0]], dtype=torch.float64)
        index = ChemicalSupportIndex(bank, radius=1.0, k=1)
        query = torch.tensor([[1.0, 0.0], [3.0, 4.0]], dtype=torch.float64)
        d = index.min_distance(query)
        assert d.shape == (2,)
        assert float(d[0]) == pytest.approx(1.0, abs=1e-8)
        assert float(d[1]) == pytest.approx(4.0, abs=1e-8)

    def test_coverage_fraction(self):
        """coverage = (# queries in support) / (# queries)."""
        bank = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float64)
        index = ChemicalSupportIndex(bank, radius=0.0, k=1)
        # type-id style 1-D bank: exact match only
        query = torch.tensor([[0.0], [1.0], [99.0], [2.0]], dtype=torch.float64)
        assert index.coverage_fraction(query) == pytest.approx(0.75)

    def test_from_type_ids_exact_membership(self):
        """Discrete type_id sets map to a 1-D L2 bank with radius 0."""
        index = ChemicalSupportIndex.from_type_ids({0, 2, 5}, radius=0.0)
        ids = torch.tensor([0, 1, 2, 5, 7], dtype=torch.long)
        assert torch.equal(
            index.contains_type_ids(ids),
            torch.tensor([True, False, True, True, False]),
        )
        assert index.coverage_fraction_type_ids(ids) == pytest.approx(3 / 5)

    def test_empty_query_coverage_is_one(self):
        """Empty prediction set is fully covered by convention."""
        bank = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
        index = ChemicalSupportIndex(bank, radius=1.0, k=1)
        empty = torch.empty(0, 2, dtype=torch.float64)
        assert index.coverage_fraction(empty) == 1.0

    def test_k_greater_than_one(self):
        bank = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        index = ChemicalSupportIndex(bank, radius=10.0, k=2)
        query = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
        dist, nn_idx = index.knn(query)
        assert dist.shape == (1, 2)
        assert set(nn_idx[0].tolist()) == {0, 1} or set(nn_idx[0].tolist()) == {0, 2}
        assert float(dist[0, 0]) == pytest.approx(0.0, abs=1e-8)

    def test_rejects_bad_bank_rank(self):
        with pytest.raises(ValueError, match="2-D"):
            ChemicalSupportIndex(torch.tensor([1.0, 2.0]), radius=1.0)

    def test_no_molpot_import_in_support_module(self):
        """molrep embedding support must not depend on molpot (ac-006 spirit)."""
        path = (
            Path(__file__).resolve().parents[3]
            / "src"
            / "molrep"
            / "embedding"
            / "support.py"
        )
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("molpot"), alias.name
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("molpot"), node.module
