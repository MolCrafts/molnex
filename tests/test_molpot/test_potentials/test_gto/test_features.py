"""Tests for GTOFieldFeatures."""

from __future__ import annotations

import pytest
import torch

from molix import config
from molpot.potentials.gto.features import GTOFieldFeatures


@pytest.fixture(autouse=True)
def fp64():
    previous = config["ftype"]
    config.set_precision("fp64")
    yield
    config.set_precision("fp64" if previous == torch.float64 else "fp32")


class TestGTOFieldFeatures:
    """Isolated-molecule field features for Polar-1-M dump widths."""

    def test_dump_widths_emit_ir_mul_dim_8(self):
        """Two widths, lmax=1 → 2x0e+2x1o dim 8."""
        block = GTOFieldFeatures(
            density_sigma=1.5,
            density_lmax=1,
            feature_lmax=1,
            feature_widths=[1.5, 3.0],
        )
        assert block.out_dim == 8
        n = 3
        out = block(
            torch.randn(n, 4, dtype=torch.float64),
            torch.randn(n, 3, dtype=torch.float64),
            torch.zeros(n, dtype=torch.long),
        )
        assert out.shape == (n, 8)

    def test_translation_invariance(self):
        """Relative geometry: a rigid translation must not change features."""
        block = GTOFieldFeatures(
            density_sigma=1.5,
            density_lmax=1,
            feature_lmax=1,
            feature_widths=[1.5, 3.0],
        )
        torch.manual_seed(0)
        q = torch.randn(4, 4, dtype=torch.float64)
        pos = torch.randn(4, 3, dtype=torch.float64)
        batch = torch.zeros(4, dtype=torch.long)
        a = block(q, pos, batch)
        b = block(q, pos + 1.7, batch)
        assert torch.allclose(a, b, atol=1e-8)
