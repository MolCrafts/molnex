"""Tests for TypeSystem / TypeRecord (learnable-classical-ff-06)."""

from __future__ import annotations

import torch

from molrep.condensation import (
    UNMATCHED_TYPE_ID,
    InteractionClass,
    TypeRecord,
    TypeSystem,
    bond_default_criterion,
)


class TestTypeSystem:
    def test_from_prototypes_dense_ids(self):
        ts = TypeSystem.from_prototypes(
            InteractionClass.BOND,
            [{"k": 300.0, "r0": 1.09}, {"k": 500.0, "r0": 1.5}],
        )
        assert ts.n_types == 2
        assert ts.get(0).prototype["r0"] == 1.09
        assert ts.get(1).member_count == 0

    def test_assign_prototype_equal_returns_existing_id(self):
        ts = TypeSystem.from_prototypes(
            InteractionClass.BOND,
            [{"k": 300.0, "r0": 1.09}],
            criterion=bond_default_criterion(),
        )
        tid = ts.assign({"k": torch.tensor(300.0), "r0": torch.tensor(1.09)})
        assert tid == 0

    def test_assign_out_of_budget_soft_fails(self):
        """Out-of-budget params return UNMATCHED_TYPE_ID; assign never spawns."""
        ts = TypeSystem.from_prototypes(
            InteractionClass.BOND,
            [{"k": 300.0, "r0": 1.09}],
            criterion=bond_default_criterion(),
        )
        tid = ts.assign({"k": torch.tensor(300.0), "r0": torch.tensor(1.5)})
        assert tid == UNMATCHED_TYPE_ID
        assert ts.n_types == 1

    def test_assign_many(self):
        ts = TypeSystem.from_prototypes(
            InteractionClass.BOND,
            [{"k": 300.0, "r0": 1.09}, {"k": 500.0, "r0": 1.5}],
            criterion=bond_default_criterion(),
        )
        params = {
            "k": torch.tensor([300.0, 500.0, 100.0]),
            "r0": torch.tensor([1.09, 1.5, 2.0]),
        }
        ids = ts.assign_many(params)
        assert ids.tolist() == [0, 1, UNMATCHED_TYPE_ID]

    def test_prototypes_table_list(self):
        ts = TypeSystem.from_prototypes(
            InteractionClass.LJ,
            [{"epsilon": 0.1, "sigma": 3.0}],
        )
        table = ts.prototypes_table()
        assert isinstance(table, list)
        assert table[0]["sigma"] == 3.0

    def test_duplicate_type_id_rejected(self):
        import pytest

        with pytest.raises(ValueError, match="duplicate"):
            TypeSystem(
                InteractionClass.BOND,
                [
                    TypeRecord(type_id=0, prototype={"k": 1.0, "r0": 1.0}),
                    TypeRecord(type_id=0, prototype={"k": 2.0, "r0": 1.0}),
                ],
            )
