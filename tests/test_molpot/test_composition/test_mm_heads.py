"""Tests for continuous Class-I MM parameter heads (learnable-classical-ff-03)."""

from __future__ import annotations

import math

import torch

from molpot.composition.heads import ChargeHead, LJParameterHead
from molpot.composition.mm_heads import (
    AngleParamHead,
    BondParamHead,
    ImproperParamHead,
    ProperTorsionParamHead,
)
from molpot.composition.multihead import MultiHead

# ---------------------------------------------------------------------------
# BondParamHead (ac-001)
# ---------------------------------------------------------------------------


class TestBondParamHead:
    def test_shapes_and_positivity(self):
        head = BondParamHead(feature_dim=8, hidden_dim=16)
        features = torch.randn(5, 8)
        out = head(features)
        assert out["k"].shape == (5,)
        assert out["r0"].shape == (5,)
        assert torch.all(out["k"] > 0)
        assert torch.all(out["r0"] > 0)

    def test_softplus_on_large_negative_inputs(self):
        head = BondParamHead(feature_dim=4, hidden_dim=8, min_k=1e-3, min_r0=1e-3)
        # Force MLP weights so pre-activations are large-negative if features are huge negative.
        with torch.no_grad():
            for p in head.parameters():
                p.zero_()
                if p.ndim == 1:
                    p.fill_(-50.0)
        features = torch.full((3, 4), -100.0)
        out = head(features)
        assert torch.all(out["k"] >= head.min_k - 1e-12)
        assert torch.all(out["r0"] >= head.min_r0 - 1e-12)
        assert torch.all(out["k"] > 0)
        assert torch.all(out["r0"] > 0)

    def test_endpoint_symmetry_same_features(self):
        """Heads are pure MLPs: identical features → identical params (call-site symmetry).

        Endpoint symmetry for bonds (i,j)↔(j,i) is enforced by supplying
        order-invariant pooled features at the call site; the head itself does
        not reorder atoms.
        """
        head = BondParamHead(feature_dim=6, hidden_dim=12)
        feats = torch.randn(4, 6)
        a = head(feats)
        b = head(feats.clone())
        assert torch.allclose(a["k"], b["k"])
        assert torch.allclose(a["r0"], b["r0"])


# ---------------------------------------------------------------------------
# AngleParamHead (ac-002)
# ---------------------------------------------------------------------------


class TestAngleParamHead:
    def test_shapes_k_positive_theta0_in_open_interval(self):
        head = AngleParamHead(feature_dim=8, hidden_dim=16)
        features = torch.randn(7, 8)
        out = head(features)
        assert out["k"].shape == (7,)
        assert out["theta0"].shape == (7,)
        assert torch.all(out["k"] > 0)
        # theta0 ∈ (0, π)
        assert torch.all(out["theta0"] > 0)
        assert torch.all(out["theta0"] < math.pi)

    def test_softplus_k_on_large_negative(self):
        head = AngleParamHead(feature_dim=4, hidden_dim=8, min_k=1e-3)
        with torch.no_grad():
            for p in head.parameters():
                p.zero_()
                if p.ndim == 1:
                    p.fill_(-40.0)
        out = head(torch.full((2, 4), -80.0))
        assert torch.all(out["k"] >= head.min_k - 1e-12)
        assert torch.all(out["theta0"] > 0)
        assert torch.all(out["theta0"] < math.pi)

    def test_endpoint_symmetry_same_features(self):
        """(i,j,k)↔(k,j,i) symmetry is call-site feature pooling; head is pure MLP."""
        head = AngleParamHead(feature_dim=5, hidden_dim=10)
        feats = torch.randn(3, 5)
        a, b = head(feats), head(feats.clone())
        assert torch.allclose(a["k"], b["k"])
        assert torch.allclose(a["theta0"], b["theta0"])


# ---------------------------------------------------------------------------
# ProperTorsionParamHead (ac-003)
# ---------------------------------------------------------------------------


class TestProperTorsionParamHead:
    def test_multi_term_k_phase_shapes(self):
        n_terms = 3
        head = ProperTorsionParamHead(
            feature_dim=8,
            hidden_dim=16,
            n_terms=n_terms,
            periodicity=(1, 2, 3),
        )
        features = torch.randn(4, 8)
        out = head(features)
        assert out["k"].shape == (4, n_terms)
        assert out["phase"].shape == (4, n_terms)
        assert torch.all(out["k"] >= 0)
        # Fixed periodicity buffer
        assert out["periodicity"].shape == (n_terms,)
        assert out["periodicity"].tolist() == [1, 2, 3]
        assert out["idivf"].shape == (4,)
        assert torch.all(out["idivf"] > 0)

    def test_k_nonneg_large_negative(self):
        head = ProperTorsionParamHead(feature_dim=4, n_terms=2, periodicity=(1, 2))
        with torch.no_grad():
            for p in head.parameters():
                p.zero_()
                if p.ndim == 1:
                    p.fill_(-30.0)
        out = head(torch.full((2, 4), -50.0))
        assert torch.all(out["k"] >= 0)


# ---------------------------------------------------------------------------
# ImproperParamHead (ac-004)
# ---------------------------------------------------------------------------


class TestImproperParamHead:
    def test_harmonic_mode(self):
        head = ImproperParamHead(
            feature_dim=8,
            hidden_dim=16,
            include_harmonic=True,
            include_periodic=False,
        )
        out = head(torch.randn(3, 8))
        assert "k" in out and "chi0" in out
        assert out["k"].shape == (3,)
        assert out["chi0"].shape == (3,)
        assert torch.all(out["k"] > 0)
        assert "phase" not in out

    def test_periodic_mode(self):
        head = ImproperParamHead(
            feature_dim=8,
            n_terms=2,
            periodicity=(2, 2),
            include_harmonic=False,
            include_periodic=True,
        )
        out = head(torch.randn(2, 8))
        assert out["k"].shape == (2, 2)
        assert out["phase"].shape == (2, 2)
        assert torch.all(out["k"] >= 0)
        assert out["periodicity"].tolist() == [2, 2]
        assert "chi0" not in out

    def test_both_modes(self):
        head = ImproperParamHead(
            feature_dim=6,
            n_terms=1,
            periodicity=(2,),
            include_harmonic=True,
            include_periodic=True,
        )
        out = head(torch.randn(2, 6))
        assert "k_harmonic" in out and "chi0" in out
        assert "k_periodic" in out and "phase" in out
        assert torch.all(out["k_harmonic"] > 0)
        assert torch.all(out["k_periodic"] >= 0)


# ---------------------------------------------------------------------------
# MultiHead + ChargeHead + LJParameterHead reuse (ac-005)
# ---------------------------------------------------------------------------


class TestMultiHeadChargeLJReuse:
    def test_merge_epsilon_sigma_charge_neutrality(self):
        heads = MultiHead(
            {
                "lj": LJParameterHead(feature_dim=8, hidden_dim=16),
                "q": ChargeHead(feature_dim=8, hidden_dim=16, total_charge=0.0),
            }
        )
        node_features = torch.randn(5, 8)
        batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
        out = heads(node_features, batch=batch)
        assert set(out) == {"epsilon", "sigma", "charge"}
        assert out["epsilon"].shape == (5,)
        assert out["sigma"].shape == (5,)
        assert out["charge"].shape == (5,)
        assert torch.all(out["epsilon"] > 0)
        assert torch.all(out["sigma"] > 0)
        # Per-molecule neutrality
        assert out["charge"][:3].sum().abs() < 1e-5
        assert out["charge"][3:].sum().abs() < 1e-5
