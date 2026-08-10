"""Tests for Condenser greedy merge (learnable-classical-ff-06)."""

from __future__ import annotations

import ast
from pathlib import Path

import torch

from molrep.condensation import (
    Condenser,
    InteractionClass,
    MergeCriterion,
    PhysicalErrorMetrics,
    bond_default_criterion,
)


class TestCondenser:
    def test_identical_params_merge_to_one_type(self):
        condenser = Condenser()
        params = {
            "k": torch.tensor([300.0, 300.0, 300.0]),
            "r0": torch.tensor([1.09, 1.09, 1.09]),
        }
        result = condenser.merge(
            [params],
            interaction=InteractionClass.BOND,
            criterion=bond_default_criterion(),
        )
        assert result.type_system.n_types == 1
        assert result.type_system.get(0).member_count == 3
        assert result.assignment.type_ids.tolist() == [0, 0, 0]

    def test_far_params_remain_distinct(self):
        condenser = Condenser()
        params = {
            "k": torch.tensor([300.0, 300.0]),
            "r0": torch.tensor([1.09, 1.5]),  # Δr0 >> 0.01 Å
        }
        result = condenser.merge(
            [params],
            interaction=InteractionClass.BOND,
            criterion=bond_default_criterion(),
        )
        assert result.type_system.n_types == 2
        assert result.assignment.type_ids.tolist() == [0, 1]

    def test_deterministic_sort_key(self):
        condenser = Condenser()
        params = {
            "k": torch.tensor([300.0, 500.0, 300.0]),
            "r0": torch.tensor([1.09, 1.5, 1.09]),
        }
        r1 = condenser.greedy_merge(
            [params],
            interaction=InteractionClass.BOND,
            criterion=bond_default_criterion(),
        )
        r2 = condenser.greedy_merge(
            [params],
            interaction=InteractionClass.BOND,
            criterion=bond_default_criterion(),
        )
        assert r1.type_system.n_types == r2.type_system.n_types
        assert r1.assignment.type_ids.tolist() == r2.assignment.type_ids.tolist()
        assert r1.type_system.prototypes_table() == r2.type_system.prototypes_table()

    def test_multi_system_global_ids(self):
        condenser = Condenser()
        sys_a = {"k": torch.tensor([300.0]), "r0": torch.tensor([1.09])}
        sys_b = {
            "k": torch.tensor([302.0, 500.0]),  # near 300/1.09 and far
            "r0": torch.tensor([1.091, 1.5]),
        }
        result = condenser.merge(
            [sys_a, sys_b],
            interaction=InteractionClass.BOND,
            criterion=bond_default_criterion(),
        )
        # Near-duplicate across systems → one global type; far → second type
        assert result.type_system.n_types == 2
        assert result.assignment.for_system(0).tolist() == [0]
        assert result.assignment.for_system(1).tolist() == [0, 1]
        assert result.assignment.type_id_at(0, 0) == 0
        assert result.assignment.type_id_at(1, 1) == 1

    def test_physics_gate_rejects_param_match(self):
        """physical_eval can block a param-budget merge."""
        condenser = Condenser()

        def always_hot(proto, cand, interaction):
            return {"energy_error": 10.0}

        params = {
            "k": torch.tensor([300.0, 301.0]),
            "r0": torch.tensor([1.09, 1.09]),
        }
        result = condenser.merge(
            [params],
            interaction=InteractionClass.BOND,
            criterion=bond_default_criterion(),
            physical_eval=always_hot,
            energy_tol=0.1,
        )
        # First row spawns type 0; second would match params but physics rejects
        assert result.type_system.n_types == 2
        assert result.metrics.rejected_by_physics == 1
        assert result.metrics.n_compared >= 1

    def test_physics_eval_records_metrics_without_gate(self):
        condenser = Condenser()

        def mild(proto, cand, interaction):
            return 0.01

        params = {
            "k": torch.tensor([300.0, 300.0]),
            "r0": torch.tensor([1.09, 1.09]),
        }
        result = condenser.merge(
            [params],
            interaction=InteractionClass.BOND,
            criterion=bond_default_criterion(),
            physical_eval=mild,
        )
        assert result.type_system.n_types == 1
        assert isinstance(result.metrics, PhysicalErrorMetrics)
        assert result.metrics.n_compared == 1
        assert result.metrics.max_abs_energy_error == 0.01

    def test_zero_abs_budget_one_type_per_unique_row(self):
        crit = MergeCriterion(
            interaction=InteractionClass.BOND,
            abs_tol={"r0": 0.0, "k": 0.0},
            required_keys=("k", "r0"),
        )
        condenser = Condenser()
        params = {
            "k": torch.tensor([100.0, 100.0, 200.0]),
            "r0": torch.tensor([1.0, 1.0, 1.0]),
        }
        result = condenser.merge([params], interaction=InteractionClass.BOND, criterion=crit)
        assert result.type_system.n_types == 2

    def test_no_smarts_emitters_in_package(self):
        root = Path(__file__).resolve().parents[3] / "src/molrep/condensation"
        for path in root.glob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    name = node.name.lower()
                    assert "smarts" not in name and "smirks" not in name
                if isinstance(node, ast.ClassDef):
                    name = node.name.lower()
                    assert "smarts" not in name and "smirks" not in name
            text = path.read_text().lower()
            # No function that returns pattern text as its purpose
            assert "def to_smarts" not in text
            assert "def emit_smarts" not in text
            assert "def to_smirks" not in text
