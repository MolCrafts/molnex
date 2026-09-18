"""Unit tests for :func:`molix.data.bond_adapter.bond_index_from_columns`.

Source mirror for ``src/molix/data/bond_adapter.py``. The adapter is the
molpy-column -> canonical COO ``bond_index`` ``[2, N]`` boundary; the ``(2, N)``
shape (as opposed to ``pair_index``'s ``(E, 2)``) is the type-level anti-alias
guard that stops a cutoff graph being fed to ``BondHarmonic``.

The collate-side scenario (offset rebase, packed roundtrip) stays in
``test_bond_collate.py`` — these are single-function tests of the adapter only.
"""

import pytest
import torch

from molix.data.bond_adapter import bond_index_from_columns


class TestBondIndexFromColumns:
    def test_stacks_columns_into_coo_rows(self):
        """Row 0 = source (``atomi``), row 1 = target (``atomj``), shape ``(2, N)``."""
        atomi = torch.tensor([0, 1, 3])
        atomj = torch.tensor([1, 2, 4])
        bond_types = torch.tensor([0, 1, 0])

        bond_index, out_types = bond_index_from_columns(atomi, atomj, bond_types)

        assert bond_index.shape == (2, 3)
        assert bond_index.tolist() == [[0, 1, 3], [1, 2, 4]]
        assert out_types.tolist() == [0, 1, 0]

    def test_casts_to_long_and_flattens_column_vectors(self):
        """int32 / ``(N, 1)`` molpy columns are normalised to flat long tensors.

        molpy frames hand back narrow integer dtypes; downstream indexing
        (``pos[bond_index[0]]``) requires ``torch.long`` and rank-1 rows.
        """
        atomi = torch.tensor([[0], [2]], dtype=torch.int32)
        atomj = torch.tensor([[1], [3]], dtype=torch.int32)
        bond_types = torch.tensor([0, 1], dtype=torch.int32)

        bond_index, out_types = bond_index_from_columns(atomi, atomj, bond_types)

        assert bond_index.dtype == torch.long
        assert out_types.dtype == torch.long
        assert bond_index.shape == (2, 2)
        assert bond_index.tolist() == [[0, 2], [1, 3]]

    def test_ragged_columns_raise_value_error(self):
        """Mismatched column lengths raise instead of silently truncating.

        A short ``bond_types`` column would otherwise mis-assign force-field
        parameters to bonds, which is a wrong-energy bug rather than a crash.
        """
        atomi = torch.tensor([0, 1, 2])
        atomj = torch.tensor([1, 2])
        bond_types = torch.tensor([0, 1, 0])

        with pytest.raises(ValueError, match=r"atomi/atomj/bond_types must share length"):
            bond_index_from_columns(atomi, atomj, bond_types)

    def test_empty_columns_give_empty_coo_index(self):
        """A bond-free molecule yields ``(2, 0)`` — still COO, still long."""
        empty = torch.zeros(0, dtype=torch.long)

        bond_index, out_types = bond_index_from_columns(empty, empty, empty)

        assert bond_index.shape == (2, 0)
        assert bond_index.dtype == torch.long
        assert out_types.shape == (0,)
