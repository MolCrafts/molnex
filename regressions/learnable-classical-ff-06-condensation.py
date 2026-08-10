"""Public-API regression for physics-aware chemical class condensation.

Spec: `learnable-classical-ff-06-condensation`.

Hard-coded goldens only — no third-party oracle. Pins:

1. MergeCriterion bond budgets (Å / relative k) accept/reject
2. Condenser: identical params → 1 type; far params → 2 types
3. Multi-system merge shares global type ids
4. TypeSystem.assign soft-fails out-of-budget rows (UNMATCHED_TYPE_ID)
5. TypeSystemLabeler satisfies Labeler Protocol
6. MultiTypeHead additive; TypeHead atom API unchanged
7. No SMARTS emitters under molrep.condensation; physical_eval is injected

Provenance
----------
    capture command  : PYTHONPATH=src python \
                      regressions/learnable-classical-ff-06-condensation.py
    date             : 2026-08-10
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import torch


def main() -> int:
    from molrep.condensation import (
        ClassAssignment,
        Condenser,
        InteractionClass,
        PhysicalErrorMetrics,
        TypeSystem,
        TypeSystemLabeler,
        UNMATCHED_TYPE_ID,
        bond_default_criterion,
    )
    from molrep.heads import Labeler, MultiTypeHead, TypeHead

    # --- 1. MergeCriterion ---
    crit = bond_default_criterion()
    assert crit.interaction is InteractionClass.BOND
    assert crit.abs_tol["r0"] == 0.01
    assert crit.accepts(
        {"k": torch.tensor(300.0), "r0": torch.tensor(1.09)},
        {"k": torch.tensor(310.0), "r0": torch.tensor(1.095)},
    )
    assert not crit.accepts(
        {"k": torch.tensor(300.0), "r0": torch.tensor(1.09)},
        {"k": torch.tensor(300.0), "r0": torch.tensor(1.12)},
    )

    # --- 2. Identical → 1 type ---
    condenser = Condenser()
    identical = {
        "k": torch.tensor([300.0, 300.0, 300.0]),
        "r0": torch.tensor([1.09, 1.09, 1.09]),
    }
    r_id = condenser.merge(
        [identical],
        interaction=InteractionClass.BOND,
        criterion=crit,
    )
    assert r_id.type_system.n_types == 1
    assert r_id.type_system.get(0).member_count == 3
    assert isinstance(r_id.assignment, ClassAssignment)
    assert isinstance(r_id.metrics, PhysicalErrorMetrics)

    # --- 3. Far params → 2 types (stable ids) ---
    far = {
        "k": torch.tensor([300.0, 300.0]),
        "r0": torch.tensor([1.09, 1.5]),
    }
    r_far = condenser.greedy_merge(
        [far],
        interaction=InteractionClass.BOND,
        criterion=crit,
    )
    assert r_far.type_system.n_types == 2
    assert r_far.assignment.type_ids.tolist() == [0, 1]

    # --- 4. Multi-system global ids ---
    sys_a = {"k": torch.tensor([300.0]), "r0": torch.tensor([1.09])}
    sys_b = {
        "k": torch.tensor([301.0, 500.0]),
        "r0": torch.tensor([1.091, 1.5]),
    }
    r_ms = condenser.merge(
        [sys_a, sys_b],
        interaction=InteractionClass.BOND,
        criterion=crit,
    )
    assert r_ms.type_system.n_types == 2
    assert r_ms.assignment.for_system(0).tolist() == [0]
    assert r_ms.assignment.for_system(1).tolist() == [0, 1]

    # --- 5. assign soft-fail ---
    ts = TypeSystem.from_prototypes(
        InteractionClass.BOND,
        [{"k": 300.0, "r0": 1.09}],
        criterion=crit,
    )
    assert ts.assign({"k": 300.0, "r0": 1.09}) == 0
    assert ts.assign({"k": 300.0, "r0": 1.5}) == UNMATCHED_TYPE_ID

    # --- 6. Labeler Protocol ---
    labeler = TypeSystemLabeler(r_ms.type_system)
    assert isinstance(labeler, Labeler)
    assert labeler.num_types == 2
    assert set(labeler.type_map) == {0, 1}
    ids = labeler.label(
        {
            "k": torch.tensor([300.0, 500.0]),
            "r0": torch.tensor([1.09, 1.5]),
        }
    )
    assert ids.tolist() == [0, 1]

    # --- 7. TypeHead / MultiTypeHead ---
    head = TypeHead(hidden_dim=4, num_types=5)
    assert head(torch.ones(2, 4)).shape == (2, 5)
    multi = MultiTypeHead.from_type_systems(
        4,
        {"bond": r_ms.type_system},
    )
    assert multi.num_types["bond"] == 2
    assert multi({"bond": torch.randn(3, 4)})["bond"].shape == (3, 2)

    # --- 8. Physics gate via injected callable (no molpot energy in package) ---
    def hot(_p, _c, _i):
        return 5.0

    r_phys = condenser.merge(
        [
            {
                "k": torch.tensor([300.0, 300.0]),
                "r0": torch.tensor([1.09, 1.09]),
            }
        ],
        interaction=InteractionClass.BOND,
        criterion=crit,
        physical_eval=hot,
        energy_tol=0.1,
    )
    assert r_phys.type_system.n_types == 2
    assert r_phys.metrics.rejected_by_physics == 1

    # --- 9. No SMARTS emitters; no molpot energy imports in condensation ---
    pkg = Path(__file__).resolve().parents[1] / "src/molrep/condensation"
    for path in sorted(pkg.glob("*.py")):
        text = path.read_text()
        lower = text.lower()
        assert "def to_smarts" not in lower
        assert "def emit_smarts" not in lower
        assert "def to_smirks" not in lower
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("molpot")
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("molpot")

    print("learnable-classical-ff-06-condensation: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
