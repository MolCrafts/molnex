"""Unit tests for molzoo.pinet.layers (the readout)."""

from __future__ import annotations

import torch

from molzoo.pinet.layers import OutLayer
from tests.test_molrep.test_encoders.test_pinet.conftest import DTYPE


class TestOutLayer:
    def test_adds_onto_the_running_output(self):
        head = OutLayer([8], in_dim=4, dtype=DTYPE)
        px = torch.randn(3, 4, dtype=DTYPE)
        prev = torch.randn(3, 1, dtype=DTYPE)
        torch.testing.assert_close(head(px, prev) - head(px, torch.zeros_like(prev)), prev)

    def test_final_projection_has_no_bias(self):
        assert OutLayer([8], in_dim=4, dtype=DTYPE).out_units.bias is None
