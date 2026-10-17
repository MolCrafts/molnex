"""Parameter heads in :mod:`molpot.composition.heads`.

These are the ByteFF-Pol-family nonbonded heads — Lennard-Jones, Pauli
repulsion, charge transfer, partial charge, and Tang-Toennies damping — plus
the build dtype every one of them takes at construction.
"""

from __future__ import annotations

import pytest
import torch

from molpot.composition.heads import (
    ChargeHead,
    ChargeTransferParameterHead,
    LJParameterHead,
    RepulsionParameterHead,
    TSScalingHead,
)


@pytest.fixture
def node_features():
    return torch.randn(5, 16)


@pytest.fixture
def batch():
    return torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)


# ---------------------------------------------------------------------------
# LJParameterHead
# ---------------------------------------------------------------------------


class TestLJParameterHead:
    def test_every_parameter_is_built_at_the_given_dtype(self):
        """A layer that ignored ``dtype`` would leave the head mixed-precision
        and its first forward would die on a dtype-mismatched matmul."""
        head = LJParameterHead(feature_dim=16, dtype=torch.float64)

        assert {p.dtype for p in head.parameters()} == {torch.float64}

    def test_starts_at_zero_energy(self, node_features):
        """A fresh head predicts floor epsilon and moderate sigma.

        The default ``sigmoid(0)=0.5`` init would emit epsilon ≈ 0.5 kcal/mol
        and sigma ≈ 2.5 Å, placing a repulsive ``(sigma/r)**12`` wall on close
        contacts and blowing up the initial total-energy scale (calibrate
        gate ``energy_scale_ok=0``). Epsilon must start at its floor (no well
        depth) while sigma stays moderate (~1 Å): flooring sigma too would
        zero ``dE/dε`` and leave the head with no learning signal.
        """
        head = LJParameterHead(feature_dim=16, dtype=torch.float32)
        out = head(node_features)
        assert torch.all(out["epsilon"] < 5e-3), out["epsilon"]
        assert torch.all(out["epsilon"] > 0)
        expected_sigma = torch.full_like(out["sigma"], head.init_sigma)
        assert torch.allclose(out["sigma"], expected_sigma, atol=1e-3), out["sigma"]

    def test_init_values_are_configurable(self, node_features):
        head = LJParameterHead(
            feature_dim=16, init_epsilon=0.05, init_sigma=1.5, dtype=torch.float32
        )
        out = head(node_features)
        assert torch.allclose(out["epsilon"], torch.full_like(out["epsilon"], 0.05), atol=1e-3)
        assert torch.allclose(out["sigma"], torch.full_like(out["sigma"], 1.5), atol=1e-3)

    def test_init_values_must_respect_bounds(self):
        import pytest

        with pytest.raises(ValueError, match="init_sigma"):
            LJParameterHead(feature_dim=16, init_sigma=0.0, dtype=torch.float32)
        with pytest.raises(ValueError, match="init_epsilon"):
            LJParameterHead(feature_dim=16, init_epsilon=2.0, dtype=torch.float32)

    def test_relaxed_contact_has_a_slope_once_epsilon_leaves_the_floor(self):
        """A 2.4 Å pair is where a relaxed nonbonded contact sits.

        At the floor (epsilon ~1e-3, sigma 1 Å) both channels are flat:
        ``∂E/∂σ ∝ ε`` and ``(σ/r)¹²`` is ~0, so the bias gradient is noise.
        Epsilon 0.05 kcal/mol and sigma 1.5 Å sit on the slope of
        ``4ε[(σ/r)¹² − (σ/r)⁶]``. One 0.8 Å contact at that init is hundreds
        of kcal/mol, not the ~1e5 wall of epsilon 0.5 and sigma 2.5.
        """
        features = torch.zeros(1, 16)

        def energy(head: LJParameterHead, distance: float) -> torch.Tensor:
            predicted = head(features)
            ratio = predicted["sigma"][0] / distance
            sixth = ratio**6
            return 4.0 * predicted["epsilon"][0] * (sixth * sixth - sixth)

        floor = LJParameterHead(feature_dim=16, dtype=torch.float32)
        learnable = LJParameterHead(
            feature_dim=16, init_epsilon=0.05, init_sigma=1.5, dtype=torch.float32
        )
        energy(floor, 2.4).backward()
        energy(learnable, 2.4).backward()
        floor_bias = floor.mlp[-1].bias.grad.abs().sum()
        learnable_bias = learnable.mlp[-1].bias.grad.abs().sum()
        assert learnable_bias > 100.0 * floor_bias
        clash = energy(
            LJParameterHead(feature_dim=16, init_epsilon=0.05, init_sigma=1.5, dtype=torch.float32),
            0.8,
        )
        assert clash.abs() < 1000.0


# ---------------------------------------------------------------------------
# RepulsionParameterHead
# ---------------------------------------------------------------------------


class TestRepulsionParameterHead:
    def test_output_keys_and_shapes(self, node_features):
        head = RepulsionParameterHead(feature_dim=16, dtype=torch.float32)
        out = head(node_features)
        assert "eps_rep" in out
        assert "lam_rep" in out
        assert out["eps_rep"].shape == (5,)
        assert out["lam_rep"].shape == (5,)

    def test_outputs_positive(self, node_features):
        head = RepulsionParameterHead(feature_dim=16, dtype=torch.float32)
        out = head(node_features)
        assert torch.all(out["eps_rep"] > 0)
        assert torch.all(out["lam_rep"] > 0)

    def test_min_floor(self):
        head = RepulsionParameterHead(feature_dim=4, min_eps=0.5, min_lam=0.3, dtype=torch.float32)
        out = head(torch.zeros(3, 4))
        assert torch.all(out["eps_rep"] >= 0.5)
        assert torch.all(out["lam_rep"] >= 0.3)

    def test_every_parameter_is_built_at_the_given_dtype(self):
        head = RepulsionParameterHead(feature_dim=16, dtype=torch.float64)

        assert {p.dtype for p in head.parameters()} == {torch.float64}


# ---------------------------------------------------------------------------
# ChargeTransferParameterHead
# ---------------------------------------------------------------------------


class TestChargeTransferParameterHead:
    def test_output_keys_and_shapes(self, node_features):
        head = ChargeTransferParameterHead(feature_dim=16, dtype=torch.float32)
        out = head(node_features)
        assert "eps_ct" in out
        assert "lam_ct" in out
        assert out["eps_ct"].shape == (5,)
        assert out["lam_ct"].shape == (5,)

    def test_outputs_positive(self, node_features):
        head = ChargeTransferParameterHead(feature_dim=16, dtype=torch.float32)
        out = head(node_features)
        assert torch.all(out["eps_ct"] > 0)
        assert torch.all(out["lam_ct"] > 0)

    def test_every_parameter_is_built_at_the_given_dtype(self):
        head = ChargeTransferParameterHead(feature_dim=16, dtype=torch.float64)

        assert {p.dtype for p in head.parameters()} == {torch.float64}


# ---------------------------------------------------------------------------
# ChargeHead
# ---------------------------------------------------------------------------


class TestChargeHead:
    def test_output_shape(self, node_features, batch):
        head = ChargeHead(feature_dim=16, dtype=torch.float32)
        out = head(node_features, batch=batch)
        assert "charge" in out
        assert out["charge"].shape == (5,)

    def test_charge_conservation_neutral(self, node_features, batch):
        head = ChargeHead(feature_dim=16, total_charge=0.0, dtype=torch.float32)
        out = head(node_features, batch=batch)
        charge = out["charge"]
        # Sum per molecule should be ~0
        mol0_sum = charge[:3].sum()
        mol1_sum = charge[3:].sum()
        assert abs(mol0_sum.item()) < 1e-5
        assert abs(mol1_sum.item()) < 1e-5

    def test_charge_conservation_nonzero(self, node_features, batch):
        head = ChargeHead(feature_dim=16, total_charge=1.0, dtype=torch.float32)
        out = head(node_features, batch=batch)
        charge = out["charge"]
        mol0_sum = charge[:3].sum()
        mol1_sum = charge[3:].sum()
        assert abs(mol0_sum.item() - 1.0) < 1e-5
        assert abs(mol1_sum.item() - 1.0) < 1e-5

    def test_grad_flows(self, batch):
        head = ChargeHead(feature_dim=8, dtype=torch.float32)
        x = torch.randn(5, 8, requires_grad=True)
        out = head(x, batch=batch)
        out["charge"].sum().backward()
        assert x.grad is not None

    def test_every_parameter_is_built_at_the_given_dtype(self):
        head = ChargeHead(feature_dim=16, dtype=torch.float64)

        assert {p.dtype for p in head.parameters()} == {torch.float64}

    def test_forward_runs_at_the_build_dtype(self, batch):
        """A head built at float64 conserves charge in float64 arithmetic."""
        head = ChargeHead(feature_dim=16, total_charge=1.0, dtype=torch.float64)

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
            dtype=torch.float32,
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

    def test_every_parameter_and_buffer_is_built_at_the_given_dtype(self):
        """The free-atom reference tables are caller-supplied buffers, so they
        follow the dtype of the tensors handed in; float64 references with a
        float64 build pin the whole module to one precision.
        """
        num_elements = 10
        head = TSScalingHead(
            feature_dim=16,
            c6_free=torch.rand(num_elements, dtype=torch.float64) * 10,
            alpha_free=torch.rand(num_elements, dtype=torch.float64) * 5,
            r_star_free=torch.rand(num_elements, dtype=torch.float64) * 2 + 1.0,
            dtype=torch.float64,
        )

        assert {p.dtype for p in head.parameters()} == {torch.float64}
        assert {b.dtype for b in head.buffers() if b.is_floating_point()} == {torch.float64}
