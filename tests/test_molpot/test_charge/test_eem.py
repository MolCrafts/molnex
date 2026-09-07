"""Tests for ElectronegativityEqualization (EEM/QEq-family)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from tensordict import TensorDict

from molpot.charge import (
    AtomicChargeModel,
    ChiEtaHead,
    DiagonalHardnessKernel,
    ElectronegativityEqualization,
    ShieldedCoulombKernel,
)
from molpot.charge.problem import ChargeEquilibrationProblem
from molpot.forcefield import ForceField, Parameterizer


class TestElectronegativityEqualizationDiagonal:
    def test_heterogeneous_packed_batch_padding_is_nonsingular(self):
        atom_batch = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
        problem = ChargeEquilibrationProblem(
            atom_batch=atom_batch,
            num_graphs=2,
            total_charge=torch.zeros(2, dtype=torch.float64),
        )
        chi = torch.tensor([1.0, 2.0, 1.0, 2.0, 3.0], dtype=torch.float64)
        eta = torch.ones(5, dtype=torch.float64)
        q = ElectronegativityEqualization(DiagonalHardnessKernel(), jitter=0.0).solve(
            chi=chi, eta=eta, problem=problem
        )
        torch.testing.assert_close(q[atom_batch == 0].sum(), torch.tensor(0.0, dtype=q.dtype))
        torch.testing.assert_close(q[atom_batch == 1].sum(), torch.tensor(0.0, dtype=q.dtype))

    def test_two_atom_analytic(self):
        """Diagonal A: q1 = (χ2 − χ1)/(η1+η2), q2 = −q1 for Q=0."""
        chi = torch.tensor([2.0, 4.0], dtype=torch.float64)
        eta = torch.tensor([1.0, 3.0], dtype=torch.float64)
        batch = torch.zeros(2, dtype=torch.long)
        problem = ChargeEquilibrationProblem(
            atom_batch=batch,
            num_graphs=1,
            total_charge=torch.zeros(1, dtype=torch.float64),
            pos=torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=torch.float64),
        )
        eq = ElectronegativityEqualization(DiagonalHardnessKernel(), jitter=0.0)
        q = eq.solve(chi=chi, eta=eta, problem=problem)
        q1 = (4.0 - 2.0) / (1.0 + 3.0)
        assert math.isclose(float(q[0]), q1, abs_tol=1e-10)
        assert math.isclose(float(q[1]), -q1, abs_tol=1e-10)
        assert math.isclose(float(q.sum()), 0.0, abs_tol=1e-10)

    def test_total_charge_constraint(self):
        chi = torch.tensor([1.0, 1.0, 2.0], dtype=torch.float64)
        eta = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64)
        batch = torch.zeros(3, dtype=torch.long)
        problem = ChargeEquilibrationProblem(
            atom_batch=batch,
            num_graphs=1,
            total_charge=torch.tensor([1.0], dtype=torch.float64),
        )
        eq = ElectronegativityEqualization(DiagonalHardnessKernel(), jitter=1e-12)
        q = eq.solve(chi=chi, eta=eta, problem=problem)
        assert math.isclose(float(q.sum()), 1.0, abs_tol=1e-8)

    def test_batched_two_molecules(self):
        chi = torch.tensor([2.0, 4.0, 1.0, 3.0], dtype=torch.float64)
        eta = torch.tensor([1.0, 1.0, 2.0, 2.0], dtype=torch.float64)
        batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        problem = ChargeEquilibrationProblem(
            atom_batch=batch,
            num_graphs=2,
            total_charge=torch.zeros(2, dtype=torch.float64),
        )
        q = ElectronegativityEqualization(DiagonalHardnessKernel(), jitter=0.0).solve(
            chi=chi, eta=eta, problem=problem
        )
        assert math.isclose(float(q[:2].sum()), 0.0, abs_tol=1e-10)
        assert math.isclose(float(q[2:].sum()), 0.0, abs_tol=1e-10)


class TestShieldedCoulomb:
    def test_two_atom_neutral_sum(self):
        chi = torch.tensor([2.0, 5.0], dtype=torch.float64)
        eta = torch.tensor([1.5, 2.5], dtype=torch.float64)
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], dtype=torch.float64)
        problem = ChargeEquilibrationProblem(
            atom_batch=torch.zeros(2, dtype=torch.long),
            num_graphs=1,
            total_charge=torch.zeros(1, dtype=torch.float64),
            pos=pos,
        )
        q = ElectronegativityEqualization(ShieldedCoulombKernel(), jitter=1e-12).solve(
            chi=chi, eta=eta, problem=problem
        )
        assert math.isclose(float(q.sum()), 0.0, abs_tol=1e-8)
        # more electronegative atom 1 should be more negative
        assert float(q[1]) < float(q[0])


class TestAtomicChargeModel:
    def test_write_forcefield_q(self):
        class _ConstChiEta(nn.Module):
            def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
                n = x.shape[0]
                return {
                    "chi": torch.tensor([2.0, 4.0], dtype=x.dtype)[:n],
                    "eta": torch.tensor([1.0, 3.0], dtype=x.dtype)[:n],
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
        feats = torch.zeros(2, 4)
        ff = model.write_forcefield(ForceField(name="c"), feats, batch)
        q = ff.params("atoms")["q"]
        assert math.isclose(float(q.sum()), 0.0, abs_tol=1e-10)

    def test_chi_eta_head_positive_eta(self):
        head = ChiEtaHead(feature_dim=8, hidden_dim=16)
        out = head(torch.randn(5, 8))
        assert out["chi"].shape == (5,)
        assert out["eta"].shape == (5,)
        assert (out["eta"] > 0).all()


class TestParameterizerCharge:
    def test_forward_writes_q(self):
        class FakeEnc(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.p = nn.Parameter(torch.zeros(1))

            def forward(self, td):
                return td

            def embeddings(self, td):
                class E:
                    def interaction_dict(self_inner):
                        return {"atoms": torch.zeros(2, 8)}

                return E()

        charge = AtomicChargeModel(
            ChiEtaHead(feature_dim=8, hidden_dim=16),
            ElectronegativityEqualization(DiagonalHardnessKernel()),
        )
        model = Parameterizer(
            FakeEnc(),
            ForceField(name="mm"),
            charge_model=charge,
        )
        batch = TensorDict(
            {
                "atoms": TensorDict(
                    {
                        "pos": torch.randn(2, 3),
                        "batch": torch.zeros(2, dtype=torch.long),
                        "Z": torch.tensor([6, 8]),
                    },
                    batch_size=[2],
                ),
                # Neutral CO fragment, spelled out: a Parameterizer with a
                # charge model refuses to assume ``Q = 0``.
                "graphs": TensorDict(
                    {
                        "num_atoms": torch.tensor([2], dtype=torch.long),
                        "total_charge": torch.zeros(1),
                    },
                    batch_size=[1],
                ),
            },
            batch_size=[],
        )
        ff = model(batch)
        assert ff.params("atoms") is not None
        assert "q" in ff.params("atoms")
        assert math.isclose(float(ff.params("atoms")["q"].sum()), 0.0, abs_tol=1e-5)
