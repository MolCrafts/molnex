"""Tests for molix.md.types — typed pytree contracts."""

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

from molix.md import MDObservables, MDState


class TestMDState:
    """MDState must behave as a transparent pytree for torch.compile/func."""

    def test_is_a_pytree(self):
        s = MDState(torch.randn(4, 3), torch.randn(4, 3), torch.randn(4, 3), torch.tensor(1.0))
        leaves, spec = tree_flatten(s)
        rebuilt = tree_unflatten(leaves, spec)
        assert isinstance(rebuilt, MDState)
        assert torch.equal(rebuilt.pos, s.pos) and torch.equal(rebuilt.energy, s.energy)

    def test_optional_cell_and_virial_default_to_none(self):
        s = MDState(torch.randn(2, 3), torch.randn(2, 3), torch.randn(2, 3), torch.tensor(1.0))
        assert s.cell is None and s.virial is None

    def test_forces_field_is_plural(self):
        """The per-atom force tensor is named ``forces`` everywhere — the
        singular ``force`` / plural ``forces`` split was a naming-drift bug."""
        assert "forces" in MDState._fields
        assert "force" not in MDState._fields


class TestMDObservables:
    def test_fields_cover_the_hook_contract(self):
        assert MDObservables._fields == (
            "pos",
            "vel",
            "forces",
            "potential",
            "kinetic",
            "total",
            "temperature",
            "cell",
            "virial",
            "volume",
            "pressure",
        )
