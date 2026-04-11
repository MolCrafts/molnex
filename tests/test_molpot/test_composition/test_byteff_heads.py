"""Tests for ByteFF-Pol parameter heads and MultiHead."""

from __future__ import annotations

import pytest
import torch

from molpot.composition.heads import (
    RepulsionParameterHead,
    ChargeTransferParameterHead,
    ChargeHead,
    TSScalingHead,
)
from molpot.composition.multi_head import MultiHead


@pytest.fixture
def node_features():
    return torch.randn(5, 16)


@pytest.fixture
def batch():
    return torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)


# ---------------------------------------------------------------------------
# RepulsionParameterHead
# ---------------------------------------------------------------------------

class TestRepulsionParameterHead:
    def test_output_keys_and_shapes(self, node_features):
        head = RepulsionParameterHead(feature_dim=16)
        out = head(node_features)
        assert "eps_rep" in out
        assert "lam_rep" in out
        assert out["eps_rep"].shape == (5,)
        assert out["lam_rep"].shape == (5,)

    def test_outputs_positive(self, node_features):
        head = RepulsionParameterHead(feature_dim=16)
        out = head(node_features)
        assert torch.all(out["eps_rep"] > 0)
        assert torch.all(out["lam_rep"] > 0)

    def test_min_floor(self):
        head = RepulsionParameterHead(feature_dim=4, min_eps=0.5, min_lam=0.3)
        out = head(torch.zeros(3, 4))
        assert torch.all(out["eps_rep"] >= 0.5)
        assert torch.all(out["lam_rep"] >= 0.3)


# ---------------------------------------------------------------------------
# ChargeTransferParameterHead
# ---------------------------------------------------------------------------

class TestChargeTransferParameterHead:
    def test_output_keys_and_shapes(self, node_features):
        head = ChargeTransferParameterHead(feature_dim=16)
        out = head(node_features)
        assert "eps_ct" in out
        assert "lam_ct" in out
        assert out["eps_ct"].shape == (5,)
        assert out["lam_ct"].shape == (5,)

    def test_outputs_positive(self, node_features):
        head = ChargeTransferParameterHead(feature_dim=16)
        out = head(node_features)
        assert torch.all(out["eps_ct"] > 0)
        assert torch.all(out["lam_ct"] > 0)


# ---------------------------------------------------------------------------
# ChargeHead
# ---------------------------------------------------------------------------

class TestChargeHead:
    def test_output_shape(self, node_features, batch):
        head = ChargeHead(feature_dim=16)
        out = head(node_features, batch=batch)
        assert "charge" in out
        assert out["charge"].shape == (5,)

    def test_charge_conservation_neutral(self, node_features, batch):
        head = ChargeHead(feature_dim=16, total_charge=0.0)
        out = head(node_features, batch=batch)
        charge = out["charge"]
        # Sum per molecule should be ~0
        mol0_sum = charge[:3].sum()
        mol1_sum = charge[3:].sum()
        assert abs(mol0_sum.item()) < 1e-5
        assert abs(mol1_sum.item()) < 1e-5

    def test_charge_conservation_nonzero(self, node_features, batch):
        head = ChargeHead(feature_dim=16, total_charge=1.0)
        out = head(node_features, batch=batch)
        charge = out["charge"]
        mol0_sum = charge[:3].sum()
        mol1_sum = charge[3:].sum()
        assert abs(mol0_sum.item() - 1.0) < 1e-5
        assert abs(mol1_sum.item() - 1.0) < 1e-5

    def test_grad_flows(self, batch):
        head = ChargeHead(feature_dim=8)
        x = torch.randn(5, 8, requires_grad=True)
        out = head(x, batch=batch)
        out["charge"].sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# TSScalingHead
# ---------------------------------------------------------------------------

class TestTSScalingHead:
    @pytest.fixture
    def ts_head(self):
        num_elements = 10
        return TSScalingHead(
            feature_dim=16,
            c6_free=torch.rand(num_elements) * 10,
            alpha_free=torch.rand(num_elements) * 5,
            r_star_free=torch.rand(num_elements) * 2 + 1.0,
        )

    def test_output_keys_and_shapes(self, ts_head, node_features):
        Z = torch.tensor([1, 6, 8, 1, 6], dtype=torch.long)
        out = ts_head(node_features, Z=Z)
        assert "c6" in out
        assert "alpha" in out
        assert "r_star" in out
        assert out["c6"].shape == (5,)
        assert out["alpha"].shape == (5,)
        assert out["r_star"].shape == (5,)

    def test_outputs_positive(self, ts_head, node_features):
        Z = torch.tensor([1, 6, 8, 1, 6], dtype=torch.long)
        out = ts_head(node_features, Z=Z)
        assert torch.all(out["c6"] > 0)
        assert torch.all(out["alpha"] > 0)
        assert torch.all(out["r_star"] > 0)

    def test_buffers_on_device(self, ts_head):
        assert ts_head.c6_free is not None
        assert ts_head.alpha_free is not None
        assert ts_head.r_star_free is not None


# ---------------------------------------------------------------------------
# MultiHead
# ---------------------------------------------------------------------------

class TestMultiHead:
    def test_merges_outputs(self, node_features, batch):
        multi = MultiHead({
            "rep": RepulsionParameterHead(feature_dim=16),
            "ct": ChargeTransferParameterHead(feature_dim=16),
            "charge": ChargeHead(feature_dim=16),
        })
        out = multi(node_features, batch=batch)
        assert "eps_rep" in out
        assert "lam_rep" in out
        assert "eps_ct" in out
        assert "lam_ct" in out
        assert "charge" in out

    def test_duplicate_key_raises(self, node_features):
        multi = MultiHead({
            "a": RepulsionParameterHead(feature_dim=16),
            "b": RepulsionParameterHead(feature_dim=16),
        })
        with pytest.raises(ValueError, match="Duplicate key"):
            multi(node_features)

    def test_empty_heads_raises(self):
        with pytest.raises(ValueError, match="at least one head"):
            MultiHead({})

    def test_with_ts_head(self, node_features):
        Z = torch.tensor([1, 6, 8, 1, 6], dtype=torch.long)
        multi = MultiHead({
            "ts": TSScalingHead(
                feature_dim=16,
                c6_free=torch.rand(10) * 10,
                alpha_free=torch.rand(10) * 5,
                r_star_free=torch.rand(10) * 2 + 1.0,
            ),
        })
        out = multi(node_features, Z=Z)
        assert "c6" in out
        assert "alpha" in out
        assert "r_star" in out
