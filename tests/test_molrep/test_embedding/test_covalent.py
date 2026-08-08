"""Tests for molrep.embedding.covalent module."""

import pytest
import torch

from molrep.embedding.covalent import covalent_radii


class TestCovalentRadii:
    """Test the Z-indexed covalent-radius table."""

    def test_length_is_max_z_plus_one(self):
        """Table is indexable by Z directly, so it needs a row for Z=0."""
        assert covalent_radii(max_z=10).shape == (11,)

    def test_known_values(self):
        """Spot-check against Cordero et al. (2008), the table MACE also uses."""
        table = covalent_radii(max_z=8)
        assert table[1] == pytest.approx(0.31, abs=1e-6)  # H
        assert table[6] == pytest.approx(0.76, abs=1e-6)  # C
        assert table[8] == pytest.approx(0.66, abs=1e-6)  # O

    def test_dummy_slot_zero(self):
        """Index 0 is a dummy-atom placeholder, matching the reference table."""
        assert covalent_radii(max_z=4)[0] == pytest.approx(0.2, abs=1e-9)

    def test_all_positive(self):
        """Every real element has a positive radius."""
        assert bool((covalent_radii(max_z=96)[1:] > 0).all())

    def test_rejects_empty_table(self):
        """A table with no elements is a caller error, not an empty tensor."""
        with pytest.raises(ValueError, match="max_z"):
            covalent_radii(max_z=0)

    def test_dtype_follows_config(self):
        """The table is built in the project dtype so it can be a module buffer."""
        from molix import config

        assert covalent_radii(max_z=4).dtype == config.ftype

    def test_is_a_plain_tensor(self):
        """Callers register it as a buffer; it must not carry grad."""
        assert not covalent_radii(max_z=4).requires_grad
        assert isinstance(covalent_radii(max_z=4), torch.Tensor)
