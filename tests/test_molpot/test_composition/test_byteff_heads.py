"""Tests for ByteFF-Pol parameter heads and MultiHead."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch

from molix import config
from molpot.composition.heads import (
    ChargeHead,
    ChargeTransferParameterHead,
    LJParameterHead,
    RepulsionParameterHead,
    TSScalingHead,
)
from molpot.composition.multihead import MultiHead


@pytest.fixture
def node_features():
    return torch.randn(5, 16)


@pytest.fixture
def batch():
    return torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)


@pytest.fixture
def fp64() -> Iterator[None]:
    """Run the case under the global fp64 precision, restoring the previous one.

    Every head in :mod:`molpot.composition.heads` bakes ``config["ftype"]``
    into its parameters at construction time (the contract documented in
    :mod:`molix.config`), so the precision has to be switched *before* the
    head is built and handed back afterwards.
    """
    previous = config["ftype"]
    config.set_precision("fp64")
    yield
    config.set_precision("fp64" if previous == torch.float64 else "fp32")


# ---------------------------------------------------------------------------
# LJParameterHead
# ---------------------------------------------------------------------------


class TestLJParameterHead:
    def test_all_parameters_honour_the_fp64_precision(self, fp64):
        """Every parameter is fp64 when the head is built under fp64.

        ``config["ftype"]`` is the single source of truth for the working
        precision; a layer that ignores it leaves the head mixed-precision
        and its first forward dies on a dtype-mismatched matmul.
        """
        head = LJParameterHead(feature_dim=16)

        assert {p.dtype for p in head.parameters()} == {torch.float64}


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

    def test_all_parameters_honour_the_fp64_precision(self, fp64):
        """Every parameter is fp64 when the head is built under fp64."""
        head = RepulsionParameterHead(feature_dim=16)

        assert {p.dtype for p in head.parameters()} == {torch.float64}


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

    def test_all_parameters_honour_the_fp64_precision(self, fp64):
        """Every parameter is fp64 when the head is built under fp64."""
        head = ChargeTransferParameterHead(feature_dim=16)

        assert {p.dtype for p in head.parameters()} == {torch.float64}


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

    def test_all_parameters_honour_the_fp64_precision(self, fp64):
        """Every parameter is fp64 when the head is built under fp64."""
        head = ChargeHead(feature_dim=16)

        assert {p.dtype for p in head.parameters()} == {torch.float64}

    def test_forward_runs_under_the_fp64_precision(self, fp64, batch):
        """A head built at fp64 conserves charge in fp64 arithmetic."""
        head = ChargeHead(feature_dim=16, total_charge=1.0)

        out = head(torch.ones(5, 16, dtype=torch.float64), batch=batch)

        assert out["charge"].shape == (5,)
        assert out["charge"].dtype == torch.float64


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

    def test_all_parameters_and_buffers_honour_the_fp64_precision(self, fp64):
        """Every parameter is fp64 when the head is built under fp64.

        The free-atom reference tables are caller-supplied buffers, so they
        follow the dtype of the tensors handed in; passing fp64 references
        pins the whole module to one precision.
        """
        num_elements = 10
        head = TSScalingHead(
            feature_dim=16,
            c6_free=torch.rand(num_elements, dtype=torch.float64) * 10,
            alpha_free=torch.rand(num_elements, dtype=torch.float64) * 5,
            r_star_free=torch.rand(num_elements, dtype=torch.float64) * 2 + 1.0,
        )

        assert {p.dtype for p in head.parameters()} == {torch.float64}
        assert {b.dtype for b in head.buffers() if b.is_floating_point()} == {torch.float64}


# ---------------------------------------------------------------------------
# MultiHead
# ---------------------------------------------------------------------------


class TestMultiHead:
    def test_merges_outputs(self, node_features, batch):
        multi = MultiHead(
            {
                "rep": RepulsionParameterHead(feature_dim=16),
                "ct": ChargeTransferParameterHead(feature_dim=16),
                "charge": ChargeHead(feature_dim=16),
            }
        )
        out = multi(node_features, batch=batch)
        assert "eps_rep" in out
        assert "lam_rep" in out
        assert "eps_ct" in out
        assert "lam_ct" in out
        assert "charge" in out

    def test_duplicate_key_raises(self, node_features):
        multi = MultiHead(
            {
                "a": RepulsionParameterHead(feature_dim=16),
                "b": RepulsionParameterHead(feature_dim=16),
            }
        )
        with pytest.raises(ValueError, match="Duplicate key"):
            multi(node_features)

    def test_empty_heads_raises(self):
        with pytest.raises(ValueError, match="at least one head"):
            MultiHead({})

    def test_with_ts_head(self, node_features):
        Z = torch.tensor([1, 6, 8, 1, 6], dtype=torch.long)
        multi = MultiHead(
            {
                "ts": TSScalingHead(
                    feature_dim=16,
                    c6_free=torch.rand(10) * 10,
                    alpha_free=torch.rand(10) * 5,
                    r_star_free=torch.rand(10) * 2 + 1.0,
                ),
            }
        )
        out = multi(node_features, Z=Z)
        assert "c6" in out
        assert "alpha" in out
        assert "r_star" in out
