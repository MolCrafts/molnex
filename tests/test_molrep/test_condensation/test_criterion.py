"""Tests for MergeCriterion budgets (learnable-classical-ff-06)."""

from __future__ import annotations

import math

import pytest
import torch

from molrep.condensation import (
    InteractionClass,
    MergeCriterion,
    bond_default_criterion,
    default_criterion,
)


class TestMergeCriterion:
    def test_bond_accepts_within_budget(self):
        crit = bond_default_criterion()
        proto = {"k": torch.tensor(300.0), "r0": torch.tensor(1.09)}
        cand = {"k": torch.tensor(310.0), "r0": torch.tensor(1.095)}  # ~3% k, 0.005 Å
        assert crit.accepts(proto, cand) is True

    def test_bond_rejects_r0_outside_budget(self):
        crit = bond_default_criterion()
        proto = {"k": torch.tensor(300.0), "r0": torch.tensor(1.09)}
        cand = {"k": torch.tensor(300.0), "r0": torch.tensor(1.12)}  # Δr0 = 0.03 > 0.01
        assert crit.accepts(proto, cand) is False

    def test_bond_rejects_k_outside_relative_budget(self):
        crit = bond_default_criterion()
        proto = {"k": torch.tensor(300.0), "r0": torch.tensor(1.09)}
        # 20% k change with r0 exact — relative k budget is 5%
        cand = {"k": torch.tensor(360.0), "r0": torch.tensor(1.09)}
        assert crit.accepts(proto, cand) is False

    def test_bond_units_are_class_i(self):
        crit = bond_default_criterion()
        assert crit.interaction is InteractionClass.BOND
        assert crit.abs_tol["r0"] == 0.01  # Å
        assert crit.rel_tol["k"] == 0.05
        assert crit.required_keys == ("k", "r0")

    def test_zero_budget_forces_exact_match_on_abs(self):
        crit = MergeCriterion(
            interaction=InteractionClass.BOND,
            abs_tol={"r0": 0.0, "k": 0.0},
            required_keys=("k", "r0"),
        )
        proto = {"k": 100.0, "r0": 1.0}
        assert crit.accepts(proto, {"k": 100.0, "r0": 1.0}) is True
        assert crit.accepts(proto, {"k": 100.0, "r0": 1.0001}) is False

    def test_angle_default_theta0_in_radians(self):
        crit = default_criterion(InteractionClass.ANGLE)
        assert math.isclose(crit.abs_tol["theta0"], math.radians(1.0))
        proto = {"k": torch.tensor(50.0), "theta0": torch.tensor(1.9)}
        near = {"k": torch.tensor(50.0), "theta0": torch.tensor(1.9 + math.radians(0.5))}
        far = {"k": torch.tensor(50.0), "theta0": torch.tensor(1.9 + math.radians(2.0))}
        assert crit.accepts(proto, near) is True
        assert crit.accepts(proto, far) is False

    def test_missing_required_key_raises(self):
        crit = bond_default_criterion()
        with pytest.raises(KeyError):
            crit.accepts({"k": 1.0}, {"k": 1.0, "r0": 1.0})
