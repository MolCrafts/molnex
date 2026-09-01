"""Tests for SparseUvuTensorProduct."""

from __future__ import annotations

import math

import pytest
import torch

from molix import config
from molrep.interaction.polar.uvu import SparseUvuTensorProduct
from molrep.utils.equivariance import rotate_irreps_features_simple


@pytest.fixture(autouse=True)
def fp64():
    previous = config["ftype"]
    config.set_precision("fp64")
    yield
    config.set_precision("fp64" if previous == torch.float64 else "fp32")


class TestSparseUvuTensorProduct:
    """Test the Polar sparse uvu product (l×l→0 and l×0→l)."""

    def test_l0_contraction_is_product_times_weight(self):
        """W=1, x1=2, x2=3 → 6. path_weight=1 and √(2·0+1)=1."""
        tp = SparseUvuTensorProduct("1x0e", "1x0e", "1x0e")
        with torch.no_grad():
            tp.weight.fill_(1.0)
        out = tp(
            torch.tensor([[2.0]], dtype=torch.float64),
            torch.tensor([[3.0]], dtype=torch.float64),
        )
        assert out.shape == (1, 1)
        assert out.item() == pytest.approx(6.0)

    def test_two_independent_scalar_copies(self):
        """Identity W: channel u contracts only with channel u.

        e3nn element path_weight is ``1/√mul_in2 = 1/√2``.
        """
        tp = SparseUvuTensorProduct("2x0e", "2x0e", "2x0e")
        with torch.no_grad():
            tp.weight.copy_(torch.eye(2, dtype=torch.float64).reshape(-1))
        out = tp(
            torch.tensor([[2.0, 5.0]], dtype=torch.float64),
            torch.tensor([[3.0, 7.0]], dtype=torch.float64),
        )
        scale = 1.0 / math.sqrt(2.0)
        assert torch.allclose(out, torch.tensor([[6.0 * scale, 35.0 * scale]], dtype=torch.float64))

    def test_vector_dot_is_z_rotation_invariant(self):
        """(1o × 1o → 0e) equals a·b / √3 and survives a z-rotation."""
        tp = SparseUvuTensorProduct("1x1o", "1x1o", "1x0e")
        with torch.no_grad():
            tp.weight.fill_(1.0)
        torch.manual_seed(0)
        x1 = torch.randn(5, 3, dtype=torch.float64)
        x2 = torch.randn(5, 3, dtype=torch.float64)
        out = tp(x1, x2)
        expect = (x1 * x2).sum(dim=-1, keepdim=True) / math.sqrt(3.0)
        assert torch.allclose(out, expect, atol=1e-12)
        theta = 0.7
        c, s = math.cos(theta), math.sin(theta)
        rot = torch.tensor(
            [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float64,
        )
        out_rot = tp(
            rotate_irreps_features_simple(x1, rot, "1x1o"),
            rotate_irreps_features_simple(x2, rot, "1x1o"),
        )
        assert torch.allclose(out, out_rot, atol=1e-10)

    def test_scalar_modulation_scales_vector(self):
        """(1o × 0e → 1o): path_weight √3 cancels the √(2ℓ+1) kernel factor."""
        tp = SparseUvuTensorProduct("1x1o", "1x0e", "1x1o")
        with torch.no_grad():
            tp.weight.fill_(1.0)
        vec = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
        scalar = torch.tensor([[4.0]], dtype=torch.float64)
        out = tp(vec, scalar)
        assert torch.allclose(out, vec * 4.0, atol=1e-12)

    def test_rejects_non_uvu_paths(self):
        """Only (ℓ×ℓ→0) and (ℓ×0→ℓ); a leftover vector path is refused."""
        with pytest.raises(NotImplementedError, match="uvu"):
            SparseUvuTensorProduct("1x1o", "1x1o", "1x1o")

    def test_rejects_non_2d_inputs(self):
        """Flattened (N, irreps.dim) only — a 3-D view would mix ir_mul axes."""
        tp = SparseUvuTensorProduct("1x0e", "1x0e", "1x0e")
        with pytest.raises(ValueError, match="2-D"):
            tp(torch.zeros(2, 1, 1, dtype=torch.float64), torch.zeros(2, 1, dtype=torch.float64))
