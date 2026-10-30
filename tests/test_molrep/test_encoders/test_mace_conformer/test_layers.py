"""Unit tests for molrep.encoders.mace_conformer.layers."""

from __future__ import annotations

import torch

from molrep.encoders.mace_conformer.layers import AtomReadout, BondReadout

DTYPE = torch.float64


class TestAtomReadout:
    def test_charges_beyond_two_read_the_end_rows(self):
        torch.manual_seed(0)
        readout = AtomReadout(3, width=4, hidden=5, charge_dim=2, dtype=DTYPE)
        scalars = torch.randn(1, 3, dtype=DTYPE).expand(4, 3)
        out = readout(scalars, torch.tensor([-5, -2, 2, 7]))
        torch.testing.assert_close(out[0], out[1])
        torch.testing.assert_close(out[2], out[3])


class TestBondReadout:
    def _readout(self) -> BondReadout:
        torch.manual_seed(0)
        return BondReadout(3, width=4, hidden=5, lengths=(0.8, 2.4, 4), dtype=DTYPE)

    def test_a_bond_reads_the_same_in_both_directions(self):
        readout = self._readout()
        a, b = torch.randn(2, 3, dtype=DTYPE), torch.randn(2, 3, dtype=DTYPE)
        length = torch.tensor([1.1, 1.5], dtype=DTYPE)
        order, ring = torch.tensor([0, 3]), torch.tensor([0, 1])
        torch.testing.assert_close(
            readout(a, b, length, order, ring), readout(b, a, length, order, ring)
        )

    def test_a_dead_row_of_zero_length_is_finite(self):
        readout = self._readout()
        zeros = torch.zeros(1, 3, dtype=DTYPE)
        length = torch.zeros(1, dtype=DTYPE)
        out = readout(zeros, zeros, length, torch.tensor([0]), torch.tensor([0]))
        assert torch.isfinite(out).all()
