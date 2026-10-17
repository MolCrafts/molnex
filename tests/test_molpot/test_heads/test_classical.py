"""Class-I heads: dimensionless deviations around a per-row reference, zero at init."""

from __future__ import annotations

import math

import pytest
import torch

from molpot.heads.classical import (
    IMPROPER_K_BOUNDS,
    K_BOUNDS,
    LENGTH_BOUNDS,
    AngleHead,
    BondHead,
    CovalentRadiusSum,
    CvffHead,
    ExpAround,
    LinearAround,
    LJHead,
    PeriodicHead,
)

FEATURES = torch.randn(4, 8, generator=torch.Generator().manual_seed(0))


def _set_raw(head, name: str, value: float) -> None:
    with torch.no_grad():
        head.mlps[name][-1].bias.fill_(value)


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


class TestBounds:
    def test_every_bond_parameter_stays_in_its_window(self):
        head = BondHead(4, 4)
        with torch.no_grad():
            for mlp in head.mlps.values():
                mlp[-1].bias.fill_(50.0)
        out = head(torch.zeros(3, 4), r0_ref=torch.full((3,), 1.5))
        torch.testing.assert_close(out["k"], torch.full((3,), 450.0 * K_BOUNDS[1]))
        torch.testing.assert_close(out["r0"], torch.full((3,), 1.5 * LENGTH_BOUNDS[1]))

    def test_an_improper_may_go_lower_than_a_force_constant(self):
        assert IMPROPER_K_BOUNDS[0] < K_BOUNDS[0]
        head = CvffHead(4, 4, k_ref=1.5)
        with torch.no_grad():
            head.mlps["k"][-1].bias.fill_(-50.0)
        out = head(torch.zeros(2, 4))
        torch.testing.assert_close(out["k"], torch.full((2,), 1.5 * IMPROPER_K_BOUNDS[0]))


class TestLinearAround:
    def test_the_slope_at_the_reference_is_the_gain(self):
        raw = torch.zeros(1, requires_grad=True)
        out = LinearAround(20.0, 0.5, 179.5)(raw, torch.tensor([109.47]))
        torch.testing.assert_close(out, torch.tensor([109.47]))
        out.backward()
        torch.testing.assert_close(raw.grad, torch.tensor([20.0]))

    def test_a_unit_of_raw_is_nearly_the_gain_far_from_the_bounds(self):
        got = LinearAround(20.0, 0.5, 179.5)(torch.tensor([2.0]), torch.tensor([109.47]))
        assert 145.0 < float(got) < 149.47  # 40° asked, 36° given on a 70° side

    def test_the_bound_is_approached_smoothly_and_never_crossed(self):
        raw = torch.tensor([1.0, 10.0], requires_grad=True)
        out = LinearAround(20.0, 0.5, 179.5)(raw, torch.full((2,), 177.5))
        assert torch.all(out <= 179.5)
        torch.testing.assert_close(out, torch.tensor([179.5, 179.5]))
        out.sum().backward()
        assert torch.all(raw.grad < 1e-3)

    def test_a_reference_on_the_bound_is_held_there(self):
        raw = torch.tensor([3.0, -1.0], requires_grad=True)
        out = LinearAround(20.0, 0.5, 180.0)(raw, torch.full((2,), 180.0))
        assert float(out[0]) == 180.0
        assert 160.0 < float(out[1]) < 180.0  # the open side still moves
        out.sum().backward()
        assert float(raw.grad[0]) == 0.0


class TestCovalentRadiusSum:
    def test_sums_molpy_covalent_radii(self):
        from molpy import Element

        got = CovalentRadiusSum()(torch.tensor([6]), torch.tensor([1]))
        want = float(Element(6).covalent) + float(Element(1).covalent)
        assert got.item() == pytest.approx(want, rel=1e-6)


class TestBondHead:
    def test_starts_at_the_reference(self):
        head = BondHead(8, 8, k_ref=450.0)
        r0_ref = torch.tensor([1.52, 1.07, 1.52, 1.07])
        out = head(FEATURES, r0_ref=r0_ref)
        torch.testing.assert_close(out["k"], torch.full((4,), 450.0))
        torch.testing.assert_close(out["r0"], r0_ref)
        assert set(out) == {"k", "r0", "k_ref", "r0_ref"}

    def test_r0_moves_ten_percent_per_unit_of_raw(self):
        head = BondHead(8, 8)
        _set_raw(head, "r0", 1.0)
        out = head(FEATURES, r0_ref=torch.ones(4))
        # 10 % asked; a little less given, the ×1.4 window bending it
        assert torch.all(out["r0"] > 1.095) and torch.all(out["r0"] < math.exp(0.1))
        torch.testing.assert_close(out["k"], torch.full((4,), 450.0))


class TestAngleHead:
    def test_each_centre_starts_at_its_own_reference(self):
        head = AngleHead(8, 8, k_ref=40.0)
        ref = torch.tensor([109.47, 120.0, 177.5, 109.47])
        out = head(FEATURES, theta0_ref=ref)
        torch.testing.assert_close(out["theta0"], ref)
        torch.testing.assert_close(out["k"], torch.full((4,), 40.0))

    def test_theta0_moves_about_the_gain_per_unit_on_every_range(self):
        head = AngleHead(8, 8, theta0_gain_deg=20.0)
        ref = torch.tensor([60.0, 109.47, 150.0, 177.5])
        _set_raw(head, "theta0", -1.0)
        out = head(FEATURES, theta0_ref=ref)
        # 20° asked on every range; within a degree given (the 0.5° floor bends it)
        torch.testing.assert_close(out["theta0"], ref - 20.0, atol=1.0, rtol=0.0)

    def test_theta0_reaches_180_degrees_and_no_further(self):
        """A linear centre may sit exactly at 180°, where the harmonic bend is smooth."""
        head = AngleHead(8, 8)
        _set_raw(head, "theta0", 3.0)
        out = head(FEATURES, theta0_ref=torch.full((4,), 177.5))
        torch.testing.assert_close(out["theta0"], torch.full((4,), 180.0))


class TestPeriodicHead:
    def test_starts_at_zero_torsion(self):
        out = PeriodicHead(8, 8)(FEATURES)["k"]
        assert out.shape == (4, 6)
        torch.testing.assert_close(out, torch.zeros(4, 6))

    def test_a_unit_of_raw_is_one_over_n(self):
        head = PeriodicHead(8, 8, periodicity=(1, 2, 3))
        _set_raw(head, "k", 1.0)
        torch.testing.assert_close(head(FEATURES)["k"][0], torch.tensor([1.0, 0.5, 1 / 3]))

    def test_phases_alternate_zero_and_180_with_the_periodicity(self):
        head = PeriodicHead(8, 8, periodicity=(1, 2, 3, 4))
        torch.testing.assert_close(head.phase, torch.tensor([0.0, 180.0, 0.0, 180.0]))

    def test_rejects_a_nonpositive_periodicity(self):
        with pytest.raises(ValueError, match="periodicity"):
            PeriodicHead(8, 8, periodicity=(0, 1))


class TestCvffHead:
    def test_starts_on_the_reference(self):
        head = CvffHead(8, 8, k_ref=1.5, periodicity=2)
        out = head(FEATURES)
        torch.testing.assert_close(out["k"], torch.full((4,), 1.5))
        torch.testing.assert_close(out["k_ref"], torch.full((4,), 1.5))
        assert int(head.periodicity) == 2

    def test_stays_positive_for_a_negative_raw(self):
        # The strain objective k(1 - cos 2psi) is linear in k: it has a floor
        # only because k cannot cross zero — and the window's floor, 1/64.
        head = CvffHead(8, 8, k_ref=1.5)
        _set_raw(head, "k", -3.0)
        k = head(FEATURES)["k"]
        assert torch.all(k > 1.5 / 64) and torch.all(k < 1.5 * math.exp(-2.0))


class TestLJHead:
    def test_starts_exactly_on_the_reference_well(self):
        sigma = torch.tensor([3.4, 3.4, 2.6, 3.0])
        epsilon = torch.tensor([0.1, 0.1, 0.02, 0.17])
        out = LJHead(8, 8)(FEATURES, sigma_ref=sigma, epsilon_ref=epsilon)
        torch.testing.assert_close(out["sigma"], sigma)
        torch.testing.assert_close(out["epsilon"], epsilon)

    def test_a_nan_reference_makes_a_nan_parameter(self):
        # An atom outside the reference table must not price silently.
        sigma = torch.tensor([3.4, float("nan"), 2.6, 3.0])
        out = LJHead(8, 8)(FEATURES, sigma_ref=sigma, epsilon_ref=torch.ones(4))
        assert torch.isnan(out["sigma"][1]) and not torch.isnan(out["sigma"][0])


class TestHeadProtocol:
    @pytest.mark.parametrize("head", [BondHead, AngleHead, PeriodicHead, CvffHead, LJHead])
    def test_every_output_layer_starts_at_zero(self, head):
        for mlp in head(8, 8).mlps.values():
            assert mlp[-1].weight.abs().sum().item() == 0.0
            assert mlp[-1].bias.abs().sum().item() == 0.0
