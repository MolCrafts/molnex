"""Class-I heads: the symmetric bonded head and its tanh window, Rondo's D2 window around GAFF2."""

from __future__ import annotations

import math

import pytest
import torch

from molpot.classical import ExpAround, LJHead, SymmetricHead, TanhWindow

#: Floating dtype every module under test is built at.
DTYPE = torch.float32

FEATURES = torch.randn(4, 8, generator=torch.Generator().manual_seed(0))


class TestExpAround:
    def test_a_unit_of_raw_is_one_factor_e_times_the_gain(self):
        ref = torch.tensor([450.0])
        torch.testing.assert_close(ExpAround(1.0)(torch.tensor([1.0]), ref), ref * math.e)
        torch.testing.assert_close(ExpAround(0.1)(torch.tensor([1.0]), ref), ref * math.exp(0.1))

    def test_the_window_is_approached_by_saturation_and_the_gradient_fades(self):
        # A straight-through clamp held the value and passed the whole gradient,
        # so a row the loss wanted *at* a bound pulled on its features forever
        # (GH200 3723816: improper stream RMS 5.9). Saturation stops the pull.
        raw = torch.tensor([5.0, -5.0, 50.0, -50.0], requires_grad=True)
        out = ExpAround(1.0, lo=1 / 8, hi=8.0)(raw, torch.full((4,), 450.0))
        assert torch.all(out <= 450.0 * 8.0) and torch.all(out >= 450.0 / 8.0)
        assert out[0] > 0.9 * 450.0 * 8.0 and out[1] < 1.1 * 450.0 / 8.0
        torch.testing.assert_close(out[2:], torch.tensor([3600.0, 56.25]))
        out.sum().backward()
        assert 0.0 < raw.grad[0] < 0.05 * out[0] and raw.grad[2] < 1e-6 * out[2]

    def test_a_unit_of_raw_is_the_gain_at_the_reference_inside_a_window(self):
        raw = torch.zeros(1, requires_grad=True)
        out = ExpAround(1.0, lo=1 / 8, hi=8.0)(raw, torch.tensor([450.0]))
        out.backward()
        torch.testing.assert_close(raw.grad, torch.tensor([450.0]))

    def test_refuses_an_empty_window(self):
        with pytest.raises(ValueError, match="hi > lo"):
            ExpAround(1.0, lo=2.0, hi=1.0)
        with pytest.raises(ValueError, match="positive"):
            ExpAround(1.0, lo=0.0, hi=1.0)
        with pytest.raises(ValueError, match="contain the reference"):
            ExpAround(1.0, lo=2.0, hi=3.0)


class TestTanhWindow:
    def test_a_zero_raw_is_the_mean(self):
        window = TanhWindow(200.0, 700.0, 80.0, 4000.0, tanh_output=20.0)
        torch.testing.assert_close(window(torch.zeros(2)), torch.full((2,), 700.0))

    def test_the_clamp_holds_the_value_and_passes_the_gradient(self):
        # Straight-through clamp: forward is the clamped value exactly, backward the identity.
        window = TanhWindow(1.0, 0.0, -1.0, 1.0, tanh_output=20.0)
        raw = torch.tensor([0.5, -0.5], requires_grad=True)
        out = window(raw)
        assert out.tolist() == [1.0, -1.0]
        out.sum().backward()
        expected = 20.0 * (1.0 - torch.tanh(raw.detach()) ** 2)
        torch.testing.assert_close(raw.grad, expected)


class TestSymmetricHead:
    def test_the_output_does_not_depend_on_the_order_of_the_copies(self):
        head = SymmetricHead(
            8,
            {"k": 1, "x0": 2},
            hidden_dim=8,
            width=8,
            layers=3,
            out_hidden_dim=8,
            out_layers=2,
            dtype=DTYPE,
        )
        a, b, c = FEATURES[:2], FEATURES[2:], FEATURES[[1, 3]]
        one, two = head.raw((a, b, c)), head.raw((c, a, b))
        for name in ("k", "x0"):
            torch.testing.assert_close(one[name], two[name])
        assert one["k"].shape == (2, 1) and one["x0"].shape == (2, 2)


class TestLJHead:
    def test_starts_exactly_on_the_reference_well(self):
        sigma = torch.tensor([3.4, 3.4, 2.6, 3.0])
        epsilon = torch.tensor([0.1, 0.1, 0.02, 0.17])
        out = LJHead(8, 8, dtype=DTYPE)(FEATURES, sigma_ref=sigma, epsilon_ref=epsilon)
        torch.testing.assert_close(out["sigma"], sigma)
        torch.testing.assert_close(out["epsilon"], epsilon)

    def test_a_nan_reference_makes_a_nan_parameter(self):
        # An atom outside the reference table must not price silently.
        sigma = torch.tensor([3.4, float("nan"), 2.6, 3.0])
        out = LJHead(8, 8, dtype=DTYPE)(FEATURES, sigma_ref=sigma, epsilon_ref=torch.ones(4))
        assert torch.isnan(out["sigma"][1]) and not torch.isnan(out["sigma"][0])


class TestHeadProtocol:
    def test_every_lj_output_layer_starts_at_zero(self):
        for mlp in LJHead(8, 8, dtype=DTYPE).mlps.values():
            assert mlp[-1].weight.abs().sum().item() == 0.0
            assert mlp[-1].bias.abs().sum().item() == 0.0
