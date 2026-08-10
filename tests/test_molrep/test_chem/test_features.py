"""Tests for ChemEmbeddings first-class container."""

from __future__ import annotations

import torch

from molrep.chem.features import ChemEmbeddings


class TestChemEmbeddings:
    """ChemEmbeddings holds the five feature tensors."""

    def test_fields_and_shapes(self):
        emb = ChemEmbeddings(
            atom=torch.zeros(4, 8),
            bond=torch.zeros(3, 8),
            angle=torch.zeros(2, 8),
            proper=torch.zeros(1, 8),
            improper=torch.zeros(1, 8),
        )
        assert emb.atom.shape == (4, 8)
        assert emb.bond.shape == (3, 8)
        assert emb.angle.shape == (2, 8)
        assert emb.proper.shape == (1, 8)
        assert emb.improper.shape == (1, 8)

    def test_empty_optional_impropers(self):
        """Missing impropers use empty (0, D) rather than absent."""
        emb = ChemEmbeddings(
            atom=torch.zeros(2, 4),
            bond=torch.zeros(1, 4),
            angle=torch.zeros(0, 4),
            proper=torch.zeros(0, 4),
            improper=torch.zeros(0, 4),
        )
        assert emb.improper.shape == (0, 4)
        assert emb.angle.shape == (0, 4)

    def test_as_dict_keys(self):
        emb = ChemEmbeddings(
            atom=torch.zeros(1, 2),
            bond=torch.zeros(1, 2),
            angle=torch.zeros(0, 2),
            proper=torch.zeros(0, 2),
            improper=torch.zeros(0, 2),
        )
        d = emb.as_dict()
        assert set(d) == {"atom", "bond", "angle", "proper", "improper"}
