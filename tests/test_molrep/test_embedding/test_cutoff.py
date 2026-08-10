"""Tests for molrep.embedding.cutoff module."""

import math

import pytest
import torch

from molrep.embedding.cutoff import (
    CosineCutoff,
    CosineCutoffSpec,
    PolynomialCutoff,
    PolynomialCutoffSpec,
)

#: A radius with no exact fp32 representation — ``float32(5.1)`` is
#: ``5.099999904632568``, off by ~9.5e-8. Any buffer that silently lands in
#: fp32 shifts the whole envelope by that much, far above the 1e-12 position
#: tolerance an fp64 run is asking for.
R_CUT_NOT_FP32_EXACT = 5.1


class TestCosineCutoffSpec:
    """Test CosineCutoffSpec configuration."""

    def test_valid_config(self):
        """Test creation with valid parameters."""
        spec = CosineCutoffSpec(r_cut=5.0)
        assert spec.r_cut == 5.0

    def test_invalid_r_cut(self):
        """Test validation for r_cut."""
        with pytest.raises(ValueError):
            CosineCutoffSpec(r_cut=0.0)

        with pytest.raises(ValueError):
            CosineCutoffSpec(r_cut=-1.0)


class TestCosineCutoff:
    """Test CosineCutoff envelope function."""

    def test_initialization(self):
        """Test CosineCutoff initialization."""
        cutoff = CosineCutoff(r_cut=5.0)
        assert cutoff.config.r_cut == 5.0

    def test_forward_shape(self):
        """Test output shape matches input."""
        cutoff = CosineCutoff(r_cut=5.0)
        distances = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

        output = cutoff(distances)
        assert output.shape == distances.shape

    def test_forward_batch(self):
        """Test with batched input."""
        cutoff = CosineCutoff(r_cut=10.0)
        distances = torch.randn(5, 20).abs()

        output = cutoff(distances)
        assert output.shape == (5, 20)

    def test_cutoff_values(self):
        """Test cutoff behavior at specific distances."""
        cutoff = CosineCutoff(r_cut=5.0)

        # At r=0, cutoff should be 1.0
        out0 = cutoff(torch.tensor([0.0]))
        assert torch.allclose(out0, torch.tensor([1.0]), atol=1e-5)

        # At r=r_cut, cutoff should be 0.0
        out1 = cutoff(torch.tensor([5.0]))
        assert torch.allclose(out1, torch.tensor([0.0]), atol=1e-5)

        # Beyond r_cut, cutoff should be 0.0
        out2 = cutoff(torch.tensor([6.0]))
        assert torch.allclose(out2, torch.tensor([0.0]), atol=1e-5)

    def test_smoothness(self):
        """Test that cutoff is smooth between 0 and r_cut."""
        cutoff = CosineCutoff(r_cut=5.0)

        # Sample points between 0 and r_cut
        distances = torch.linspace(0, 5.0, 100)
        output = cutoff(distances)

        # Should be monotonically decreasing
        assert (output[:-1] >= output[1:]).all()

        # Should be in range [0, 1]
        assert (output >= 0).all()
        assert (output <= 1).all()

    def test_differentiable(self):
        """Test that gradients flow through cutoff."""
        cutoff = CosineCutoff(r_cut=5.0)
        distances = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

        output = cutoff(distances)
        loss = output.sum()
        loss.backward()

        assert distances.grad is not None
        assert not torch.isnan(distances.grad).any()

    def test_gradient_at_cutoff(self):
        """Test that gradient is zero at cutoff (smooth boundary)."""
        cutoff = CosineCutoff(r_cut=5.0)
        distances = torch.tensor([5.0], requires_grad=True)

        output = cutoff(distances)
        output.backward()

        # Gradient should be zero or very small at cutoff
        assert abs(distances.grad.item()) < 1e-5

    def test_different_r_cut_values(self):
        """Test with different cutoff radii."""
        for r_cut in [1.0, 3.0, 5.0, 10.0]:
            cutoff = CosineCutoff(r_cut=r_cut)

            # Test at 0, mid, and beyond cutoff
            distances = torch.tensor([0.0, r_cut / 2, r_cut, r_cut + 1.0])
            output = cutoff(distances)

            assert torch.allclose(output[0], torch.tensor(1.0), atol=1e-5)
            assert output[1] > 0.4  # Should be significant at midpoint
            assert output[1] < 0.6
            assert torch.allclose(output[2], torch.tensor(0.0), atol=1e-5)
            assert torch.allclose(output[3], torch.tensor(0.0), atol=1e-5)

    def test_dtype_consistency(self):
        """Test that output dtype matches input."""
        cutoff = CosineCutoff(r_cut=5.0)

        # Float32
        dist_f32 = torch.tensor([1.0, 2.0], dtype=torch.float32)
        out_f32 = cutoff(dist_f32)
        assert out_f32.dtype == torch.float32

        # Float64
        dist_f64 = torch.tensor([1.0, 2.0], dtype=torch.float64)
        out_f64 = cutoff(dist_f64)
        assert out_f64.dtype == torch.float64

    def test_r_cut_buffer_honours_the_fp64_precision(self, fp64):
        """The ``r_cut`` buffer is fp64 when the module is built under fp64.

        ``config["ftype"]`` is the single source of truth for the working
        precision; an fp32 ``r_cut`` inside an otherwise-fp64 model silently
        demotes every distance ratio it participates in.
        """
        cutoff = CosineCutoff(r_cut=5.0)

        assert cutoff.r_cut.dtype == torch.float64

    def test_r_cut_keeps_full_precision_for_a_non_fp32_representable_radius(self, fp64):
        """An fp32 ``r_cut`` truncates the requested radius by ~1e-7 Å."""
        cutoff = CosineCutoff(r_cut=R_CUT_NOT_FP32_EXACT)

        assert float(cutoff.r_cut) == pytest.approx(R_CUT_NOT_FP32_EXACT, abs=1e-12)

    def test_forward_matches_the_fp64_envelope(self, fp64):
        """c(r) matches the double-precision formula at the fp64 tolerance.

        The tolerance sits well above fp64 round-off (~1e-16) and well below
        the ~1e-8 error an fp32-truncated ``r_cut`` introduces.
        """
        cutoff = CosineCutoff(r_cut=R_CUT_NOT_FP32_EXACT)
        r = torch.tensor([0.5, 2.55, 4.0], dtype=torch.float64)

        got = cutoff(r)

        expected = 0.5 * (torch.cos(math.pi * r / R_CUT_NOT_FP32_EXACT) + 1.0)
        torch.testing.assert_close(got, expected, rtol=0.0, atol=1e-10)

    def test_broadcasting(self):
        """Test broadcasting behavior."""
        cutoff = CosineCutoff(r_cut=5.0)

        # Different shapes
        dist1d = torch.tensor([1.0, 2.0, 3.0])
        dist2d = torch.randn(4, 5).abs()
        dist3d = torch.randn(2, 3, 4).abs()

        out1d = cutoff(dist1d)
        out2d = cutoff(dist2d)
        out3d = cutoff(dist3d)

        assert out1d.shape == dist1d.shape
        assert out2d.shape == dist2d.shape
        assert out3d.shape == dist3d.shape


class TestPolynomialCutoff:
    """Test PolynomialCutoff follows the NequIP/DimeNet envelope formula."""

    def test_boundary_values(self):
        """u(0) == 1 and u(r >= r_cut) == 0 for all supported exponents."""
        for p in (2, 6, 48):
            cutoff = PolynomialCutoff(r_cut=5.0, exponent=p)
            r = torch.tensor([0.0, 5.0, 6.0, 10.0])
            out = cutoff(r)
            assert torch.isclose(out[0], torch.tensor(1.0))
            assert out[1].item() == 0.0
            assert out[2].item() == 0.0
            assert out[3].item() == 0.0

    @pytest.mark.parametrize("p", [2, 6, 48])
    def test_matches_paper_formula(self, p):
        """u(x) == 1 - (p+1)(p+2)/2·x^p + p(p+2)·x^(p+1) - p(p+1)/2·x^(p+2)."""
        cutoff = PolynomialCutoff(r_cut=1.0, exponent=p)
        r = torch.tensor([0.25, 0.5, 0.75, 0.9])
        got = cutoff(r)
        expected = (
            1.0
            - ((p + 1.0) * (p + 2.0) / 2.0) * r**p
            + p * (p + 2.0) * r ** (p + 1)
            - (p * (p + 1.0) / 2.0) * r ** (p + 2)
        )
        assert torch.allclose(got, expected, rtol=1e-5, atol=1e-5)

    def test_smooth_gradient_at_cutoff(self):
        """Derivative vanishes at r = r_cut (critical for autograd forces)."""
        cutoff = PolynomialCutoff(r_cut=5.0, exponent=6)
        r = torch.tensor([5.0 - 1e-4], requires_grad=True)
        out = cutoff(r).sum()
        out.backward()
        assert r.grad.abs().item() < 1e-2

    def test_invalid_exponent(self):
        with pytest.raises(ValueError):
            PolynomialCutoffSpec(r_cut=5.0, exponent=0)
        with pytest.raises(ValueError):
            PolynomialCutoffSpec(r_cut=5.0, exponent=-1)

    def test_r_cut_buffer_honours_the_fp64_precision(self, fp64):
        """The ``r_cut`` buffer is fp64 when the module is built under fp64.

        Same construction-time contract as :class:`CosineCutoff` — this
        envelope is the NequIP/Allegro default, so an fp32 ``r_cut`` here
        demotes the radial ratio on the main representation path.
        """
        cutoff = PolynomialCutoff(r_cut=5.0, exponent=6)

        assert cutoff.r_cut.dtype == torch.float64

    def test_r_cut_keeps_full_precision_for_a_non_fp32_representable_radius(self, fp64):
        """An fp32 ``r_cut`` truncates the requested radius by ~1e-7 Å."""
        cutoff = PolynomialCutoff(r_cut=R_CUT_NOT_FP32_EXACT, exponent=6)

        assert float(cutoff.r_cut) == pytest.approx(R_CUT_NOT_FP32_EXACT, abs=1e-12)


class TestPolynomialCutoffEnvelope:
    """Test the static envelope used with a per-edge cutoff radius."""

    def test_matches_forward_for_the_module_radius(self):
        """The module's forward is the static envelope at its own r_cut."""
        cutoff = PolynomialCutoff(r_cut=5.0, exponent=5)
        r = torch.linspace(0.0, 6.0, 25)
        assert torch.allclose(cutoff(r), PolynomialCutoff.envelope(r, 5.0, 5))

    def test_accepts_a_per_element_radius(self):
        """ZBL needs one cutoff per edge, from the pair's covalent radii."""
        r = torch.tensor([1.0, 1.0, 1.0])
        per_edge = torch.tensor([0.8, 2.0, 5.0])
        out = PolynomialCutoff.envelope(r, per_edge, 5)
        assert float(out[0]) == 0.0  # beyond its own cutoff
        assert float(out[1]) > 0.0
        assert float(out[2]) > float(out[1])  # further inside a wider cutoff

    def test_is_one_at_zero_and_zero_beyond(self):
        """Envelope endpoints: 1 at contact, exactly 0 past the radius."""
        assert float(PolynomialCutoff.envelope(torch.zeros(1), 3.0, 5)) == pytest.approx(1.0)
        assert float(PolynomialCutoff.envelope(torch.tensor([3.5]), 3.0, 5)) == 0.0

    def test_derivative_vanishes_at_the_cutoff(self):
        """Smooth shutoff is what keeps autograd forces continuous."""
        r = torch.tensor([2.9999], requires_grad=True)
        PolynomialCutoff.envelope(r, 3.0, 5).backward()
        assert abs(float(r.grad)) < 1e-6
