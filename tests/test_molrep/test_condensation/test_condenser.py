"""Tests for Condenser — one type per environment."""

from __future__ import annotations

import pytest
import torch

from molrep.condensation import (
    Condenser,
    InteractionClass,
    MergeCriterion,
    PhysicalErrorMetrics,
    bond_default_criterion,
)


class TestCondenser:
    def test_same_environment_identical_params_one_type(self):
        result = Condenser().merge(
            [{"k": torch.tensor([300.0, 300.0, 300.0]), "r0": torch.tensor([1.09, 1.09, 1.09])}],
            interaction=InteractionClass.BOND,
            environments=[[0, 0, 0]],
            criterion=bond_default_criterion(),
        )
        assert result.type_system.n_types == 1
        assert result.type_system.get(0).member_count == 3
        assert result.assignment.type_ids.tolist() == [0, 0, 0]

    def test_same_environment_far_params_raises(self):
        with pytest.raises(ValueError, match="too coarse"):
            Condenser().merge(
                [{"k": torch.tensor([300.0, 300.0]), "r0": torch.tensor([1.09, 1.5])}],
                interaction=InteractionClass.BOND,
                environments=[[0, 0]],
                criterion=bond_default_criterion(),
            )

    def test_different_environments_stay_distinct(self):
        result = Condenser().merge(
            [{"k": torch.tensor([300.0, 300.0]), "r0": torch.tensor([1.09, 1.5])}],
            interaction=InteractionClass.BOND,
            environments=[["cc", "co"]],
            criterion=bond_default_criterion(),
        )
        assert result.type_system.n_types == 2
        assert result.assignment.type_ids.tolist() == [0, 1]

    def test_multi_system_shares_environment_ids(self):
        result = Condenser().merge(
            [
                {"k": torch.tensor([300.0]), "r0": torch.tensor([1.09])},
                {"k": torch.tensor([302.0, 500.0]), "r0": torch.tensor([1.091, 1.5])},
            ],
            interaction=InteractionClass.BOND,
            environments=[["cc"], ["cc", "co"]],
            criterion=bond_default_criterion(),
        )
        assert result.type_system.n_types == 2
        assert result.assignment.for_system(0).tolist() == [0]
        assert result.assignment.for_system(1).tolist() == [0, 1]

    def test_physics_gate_rejects_bucket(self):
        def always_hot(proto, cand, interaction):
            return {"energy_error": 10.0}

        with pytest.raises(ValueError, match="physics gate"):
            Condenser().merge(
                [{"k": torch.tensor([300.0, 301.0]), "r0": torch.tensor([1.09, 1.09])}],
                interaction=InteractionClass.BOND,
                environments=[[0, 0]],
                criterion=bond_default_criterion(),
                physical_eval=always_hot,
                energy_tol=0.1,
            )

    def test_physics_eval_records_without_gate(self):
        def mild(proto, cand, interaction):
            return 0.01

        result = Condenser().merge(
            [{"k": torch.tensor([300.0, 300.0]), "r0": torch.tensor([1.09, 1.09])}],
            interaction=InteractionClass.BOND,
            environments=[[0, 0]],
            criterion=bond_default_criterion(),
            physical_eval=mild,
        )
        assert result.type_system.n_types == 1
        assert isinstance(result.metrics, PhysicalErrorMetrics)
        assert result.metrics.n_compared == 2
        assert result.metrics.max_abs_energy_error == 0.01

    def test_zero_abs_budget_same_env_duplicate_ok(self):
        crit = MergeCriterion(
            interaction=InteractionClass.BOND,
            abs_tol={"r0": 0.0, "k": 0.0},
            required_keys=("k", "r0"),
        )
        result = Condenser().merge(
            [{"k": torch.tensor([100.0, 100.0]), "r0": torch.tensor([1.0, 1.0])}],
            interaction=InteractionClass.BOND,
            environments=[[0, 0]],
            criterion=crit,
        )
        assert result.type_system.n_types == 1

    def test_environment_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="environments"):
            Condenser().merge(
                [{"k": torch.tensor([1.0]), "r0": torch.tensor([1.0])}],
                interaction=InteractionClass.BOND,
                environments=[[0, 0]],
            )

    def test_string_environment_becomes_type_label(self):
        result = Condenser().merge(
            [{"k": torch.tensor([200.0, 200.0]), "r0": torch.tensor([1.53, 1.53])}],
            interaction=InteractionClass.BOND,
            environments=[["[#6;D1]", "[#6;D1]"]],
            criterion=bond_default_criterion(),
        )
        assert result.type_system.n_types == 1
        assert result.type_system.get(0).label == "[#6;D1]"

    def test_write_smarts_keys_then_def_type(self):
        from molpy import Atomistic
        from molpy.ff.forcefield import BondType, ForceField
        from molpy.perceive import SmartsPattern

        def write_smarts(mol, handle):
            return str(SmartsPattern.from_environment(mol, handle))

        mol = Atomistic()
        atoms = [mol.def_atom({"element": el}) for el in ("C", "C", "O")]
        mol.def_bond(atoms[0], atoms[1])
        mol.def_bond(atoms[1], atoms[2])
        envs = [
            [
                tuple(sorted((write_smarts(mol, b.itom.handle), write_smarts(mol, b.jtom.handle))))
                for b in mol.bonds
            ]
        ]
        result = Condenser().merge(
            [{"k": torch.tensor([200.0, 320.0]), "r0": torch.tensor([1.53, 1.41])}],
            interaction=InteractionClass.BOND,
            environments=envs,
        )
        ff = ForceField("t", "real")
        astyle = ff.def_style("atom", "full")
        bstyle = ff.def_style("bond", "harmonic")
        for rec in result.type_system.records():
            si, sj = envs[0][int(result.assignment.row_indices[rec.support_ids[0]])]
            ai = astyle.get_type_by_name(si) or astyle.def_type(si)
            aj = astyle.get_type_by_name(sj) or astyle.def_type(sj)
            bstyle.def_type(rec.label, ai, aj, k=rec.prototype["k"], r0=rec.prototype["r0"])
        assert len(list(ff.get_types(BondType))) == 2
