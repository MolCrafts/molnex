"""Tests for continuous Class-I MM parameter heads (learnable-classical-ff-03).

Also covers OPLS-AA proper head form + reuse schema contracts
(oplsaa-parameterizer-02-heads).
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict

from molpot.charge import (
    AtomicChargeModel,
    DiagonalHardnessKernel,
    ElectronegativityEqualization,
)
from molpot.composition.heads import ChargeHead, LJParameterHead
from molpot.composition.mm_heads import (
    AngleParamHead,
    BondParamHead,
    ImproperParamHead,
    ProperTorsionParamHead,
)
from molpot.composition.multihead import MultiHead
from molpot.forcefield import ForceField

# ---------------------------------------------------------------------------
# BondParamHead
# ---------------------------------------------------------------------------


class TestBondParamHead:
    def test_starts_at_the_declared_physical_force_constant(self):
        """The init is the invariant, so it is asserted, not assumed.

        ``softplus(0) = 0.693`` kcal/mol/Å² is three orders of magnitude below
        what THEMol's QM partial-Hessian labels ask for: the 1-2 pairs of one
        batch have a mixed-block trace of median −932, and this container's bond
        term is ``E = ½k(r−r0)²``, whose mixed block is ``−k`` near equilibrium.
        Stage 2 of the curriculum then moved 0.747 → 0.728 in 3 893 steps, needing
        ~150 000 to arrive.
        """
        head = BondParamHead(feature_dim=8, hidden_dim=16)
        out = head(torch.randn(5, 8))
        assert torch.allclose(out["k"], torch.full((5,), 900.0), rtol=1e-5)
        assert torch.allclose(out["r0"], torch.full((5,), 1.20), rtol=1e-5)

    def test_the_init_is_a_constant_head_that_still_learns(self):
        """Zeroing the last layer starts every bond equal — it does not freeze it.

        The weight gradient is ``dL/draw * hidden``, so a uniform field is the
        prior, not the answer. A zero gradient here would be a dead head.
        """
        head = BondParamHead(feature_dim=8, hidden_dim=16)
        out = head(torch.randn(5, 8))
        out["k"].sum().backward()
        assert head.mlp[-1].weight.grad is not None
        assert float(head.mlp[-1].weight.grad.abs().sum()) > 0.0

    def test_a_requested_init_is_the_value_the_head_starts_at(self):
        """``init_k`` is a promise about the output, not about a pre-activation."""
        head = BondParamHead(feature_dim=4, hidden_dim=8, init_k=12.5, init_r0=2.0)
        out = head(torch.randn(3, 4))
        assert torch.allclose(out["k"], torch.full((3,), 12.5), rtol=1e-5)
        assert torch.allclose(out["r0"], torch.full((3,), 2.0), rtol=1e-5)

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
# AngleParamHead
# ---------------------------------------------------------------------------


class TestAngleParamHead:
    def test_starts_tetrahedral_at_a_live_force_constant(self):
        """``softplus(0)`` left the angle term contributing 0.004 kcal/mol — dead.

        Measured on a real batch: a within-scan spread of 1.3e-4 kcal/mol against
        a QM reference spanning 44 (run 2787862). ``sigmoid(0)`` also put the
        equilibrium angle at exactly π/2 rather than tetrahedral.
        """
        head = AngleParamHead(feature_dim=8, hidden_dim=16)
        out = head(torch.randn(6, 8))
        assert torch.allclose(out["k"], torch.full((6,), 80.0), rtol=1e-5)
        assert torch.allclose(out["theta0"], torch.full((6,), 1.9111), atol=2e-4)
        assert abs(math.degrees(float(out["theta0"][0])) - 109.47) < 0.1

    def test_theta0_init_and_forward_share_one_epsilon(self):
        """Two copies of the margin would make ``init_theta0`` a value it misses.

        The forward maps ``eps + (π − 2eps) · sigmoid`` and the init inverts it;
        asking for a value close to the boundary is where a second epsilon shows.
        """
        head = AngleParamHead(feature_dim=4, hidden_dim=8, init_theta0=0.05)
        out = head(torch.randn(3, 4))
        assert torch.allclose(out["theta0"], torch.full((3,), 0.05), atol=1e-3)

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
# ProperTorsionParamHead — unconstrained Cartesian (a, b), default n=1..6
# ---------------------------------------------------------------------------


class TestProperTorsionParamHead:
    def test_default_forward_emits_unconstrained_a_b(self):
        torch.manual_seed(0)
        head = ProperTorsionParamHead(feature_dim=8)
        out = head(torch.randn(4, 8))
        assert set(out) == {"a", "b", "periodicity"}
        assert out["a"].shape == (4, 6)
        assert out["b"].shape == (4, 6)
        assert out["periodicity"].tolist() == [[1, 2, 3, 4, 5, 6]] * out["periodicity"].shape[0]
        assert "k" not in out
        assert "phase" not in out
        assert "c0" not in out

    def test_n_terms_three_shapes(self):
        torch.manual_seed(0)
        head = ProperTorsionParamHead(
            feature_dim=8,
            hidden_dim=16,
            n_terms=3,
            periodicity=(1, 2, 3),
        )
        out = head(torch.randn(4, 8))
        assert out["a"].shape == (4, 3)
        assert out["b"].shape == (4, 3)
        assert out["periodicity"].tolist() == [[1, 2, 3]] * out["periodicity"].shape[0]
        assert "k" not in out
        assert "phase" not in out

    def test_large_negative_features_yield_finite_unconstrained_amplitudes(self):
        head = ProperTorsionParamHead(feature_dim=4, n_terms=2, periodicity=(1, 2))
        with torch.no_grad():
            for p in head.parameters():
                p.zero_()
                if p.ndim == 1:
                    p.fill_(-30.0)
        out = head(torch.full((2, 4), -50.0))
        assert torch.isfinite(out["a"]).all()
        assert torch.isfinite(out["b"]).all()
        assert (out["a"] < 0).all()
        assert (out["b"] < 0).all()

    def test_emits_exactly_the_canonical_amplitude_keys(self):
        torch.manual_seed(1)
        head = ProperTorsionParamHead(feature_dim=8)
        out = head(torch.randn(3, 8))
        assert set(out) == {"a", "b", "periodicity"}
        assert out["a"].shape == (3, 6)
        assert out["b"].shape == (3, 6)
        assert out["periodicity"].tolist() == [[1, 2, 3, 4, 5, 6]] * out["periodicity"].shape[0]

    def test_amplitudes_are_differentiable_wrt_features(self):
        """The Cartesian pair is the trainable interior — gradients must flow."""
        head = ProperTorsionParamHead(feature_dim=4, hidden_dim=8, n_terms=4)
        features = torch.randn(3, 4, requires_grad=True)
        out = head(features)
        (out["a"].sum() + out["b"].sum()).backward()
        assert features.grad is not None
        assert features.grad.shape == (3, 4)
        assert torch.isfinite(features.grad).all()

    def test_periodicity_length_must_match_n_terms(self):
        with pytest.raises(ValueError, match="periodicity"):
            ProperTorsionParamHead(feature_dim=4, n_terms=3, periodicity=(1, 2))


# ---------------------------------------------------------------------------
# ImproperParamHead — Universal-v0 default periodic n=1,2 (not Cartesian)
# ---------------------------------------------------------------------------


class TestImproperParamHead:
    def test_default_is_periodic_n_one_and_two(self):
        torch.manual_seed(0)
        head = ImproperParamHead(feature_dim=8)
        out = head(torch.randn(3, 8))
        assert "k" in out and "phase" in out
        assert "periodicity" in out
        assert out["periodicity"].tolist() == [[1, 2]] * out["periodicity"].shape[0]
        assert out["k"].shape == (3, 2)
        assert out["phase"].shape == (3, 2)
        assert torch.all(out["k"] >= 0)
        assert "chi0" not in out
        assert "a" not in out and "b" not in out

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
        assert out["periodicity"].tolist() == [[2, 2]] * out["periodicity"].shape[0]
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
# MultiHead + ChargeHead + LJParameterHead reuse
# ---------------------------------------------------------------------------


class TestMultiHeadChargeLJReuse:
    def test_merge_epsilon_sigma_charge_neutrality(self):
        heads = MultiHead(
            {
                "lj": LJParameterHead(feature_dim=8, hidden_dim=16),
                "charge": ChargeHead(feature_dim=8, hidden_dim=16, total_charge=0.0),
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


# ---------------------------------------------------------------------------
# OPLS-AA reuse schema contracts
# ---------------------------------------------------------------------------


class TestOPLSAAHeadSchemaContracts:
    def test_bond_param_head_keys(self):
        out = BondParamHead(feature_dim=4, hidden_dim=8)(torch.randn(2, 4))
        assert set(out) == {"k", "r0"}
        assert out["k"].shape == (2,)
        assert out["r0"].shape == (2,)

    def test_angle_param_head_keys(self):
        out = AngleParamHead(feature_dim=4, hidden_dim=8)(torch.randn(2, 4))
        assert set(out) == {"k", "theta0"}
        assert out["k"].shape == (2,)
        assert out["theta0"].shape == (2,)

    def test_improper_param_head_default_periodic_keys(self):
        out = ImproperParamHead(feature_dim=4, hidden_dim=8)(torch.randn(2, 4))
        assert set(out) == {"k", "phase", "periodicity"}
        assert out["k"].shape == (2, 2)
        assert out["phase"].shape == (2, 2)
        assert out["periodicity"].tolist() == [[1, 2]] * out["periodicity"].shape[0]

    def test_lj_parameter_head_keys(self):
        out = LJParameterHead(feature_dim=4, hidden_dim=8)(torch.randn(3, 4))
        assert set(out) == {"epsilon", "sigma"}
        assert out["epsilon"].shape == (3,)
        assert out["sigma"].shape == (3,)

    def test_atomic_charge_model_write_forcefield_writes_q(self):
        class _ConstChiEta(nn.Module):
            def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
                n = x.shape[0]
                return {
                    "chi": torch.full((n,), 2.0, dtype=x.dtype),
                    "eta": torch.full((n,), 1.0, dtype=x.dtype),
                }

        model = AtomicChargeModel(
            _ConstChiEta(),
            ElectronegativityEqualization(DiagonalHardnessKernel(), jitter=0.0),
        )
        batch = TensorDict(
            {
                "atoms": TensorDict(
                    {
                        "pos": torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                        "batch": torch.zeros(2, dtype=torch.long),
                    },
                    batch_size=[2],
                )
            },
            batch_size=[],
        )
        ff = model.write_forcefield(ForceField(name="schema"), torch.zeros(2, 4), batch)
        atoms = ff.params("atoms")
        assert "charge" in atoms
        assert atoms["charge"].shape == (2,)
