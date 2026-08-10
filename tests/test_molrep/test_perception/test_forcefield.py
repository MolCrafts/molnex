"""SymbolicForceField records + match_molecule (learnable-classical-ff-07)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from molrep.condensation import InteractionClass, TypeRecord, TypeSystem
from molrep.perception import (
    ClassPatternRegistry,
    DiscreteClassRecord,
    FakeSmartsMatcher,
    SymbolicForceField,
    SymbolicPattern,
)

FORCEFIELD_PATH = (
    Path(__file__).resolve().parents[3] / "src" / "molrep" / "perception" / "forcefield.py"
)
PERCEPTION_ROOT = FORCEFIELD_PATH.parent


def _bond_type_system() -> TypeSystem:
    return TypeSystem(
        InteractionClass.BOND,
        [
            TypeRecord(
                type_id=0,
                prototype={"k": 300.0, "r0": 1.09},
                label="CT-CT",
            ),
            TypeRecord(
                type_id=1,
                prototype={"k": 320.0, "r0": 1.41},
                label="CT-OH",
            ),
        ],
    )


class TestSymbolicForceFieldRecords:
    def test_records_pair_prototypes_with_patterns(self):
        ts = _bond_type_system()
        reg = ClassPatternRegistry()
        reg.bind(
            InteractionClass.BOND,
            0,
            SymbolicPattern("[#6]-[#6]", arity=2),
        )
        reg.bind(
            InteractionClass.BOND,
            1,
            SymbolicPattern("[#6]-[#8]", arity=2),
        )
        ff = SymbolicForceField({InteractionClass.BOND: ts}, reg)
        recs = ff.records()
        assert len(recs) == 2
        assert all(isinstance(r, DiscreteClassRecord) for r in recs)
        by_id = {r.type_id: r for r in recs}
        assert by_id[0].smarts == "[#6]-[#6]"
        assert by_id[0].prototype["r0"] == 1.09
        assert by_id[0].label == "CT-CT"
        assert by_id[1].smarts == "[#6]-[#8]"
        assert by_id[1].smirks is None

    def test_unbound_type_has_none_smarts(self):
        ts = _bond_type_system()
        reg = ClassPatternRegistry()
        reg.bind(
            InteractionClass.BOND,
            0,
            SymbolicPattern("[#6]-[#6]", arity=2),
        )
        ff = SymbolicForceField({InteractionClass.BOND: ts}, reg)
        recs = {r.type_id: r for r in ff.records()}
        assert recs[0].smarts == "[#6]-[#6]"
        assert recs[1].smarts is None

    def test_smirks_kind_fills_smirks_field(self):
        ts = TypeSystem(
            InteractionClass.BOND,
            [TypeRecord(type_id=0, prototype={"k": 1.0, "r0": 1.0})],
        )
        reg = ClassPatternRegistry()
        reg.bind(
            InteractionClass.BOND,
            0,
            SymbolicPattern("[C:1][O:2]>>[C:1][O:2]", arity=2, kind="smirks"),
        )
        ff = SymbolicForceField({InteractionClass.BOND: ts}, reg)
        rec = ff.records()[0]
        assert rec.smirks == "[C:1][O:2]>>[C:1][O:2]"
        assert rec.smarts is None


class TestSymbolicForceFieldMatchMolecule:
    def test_match_molecule_assigns_types_via_fake(self):
        ts = _bond_type_system()
        reg = ClassPatternRegistry()
        reg.bind(
            InteractionClass.BOND,
            0,
            SymbolicPattern("[#6]-[#6]", arity=2),
        )
        reg.bind(
            InteractionClass.BOND,
            1,
            SymbolicPattern("[#6]-[#8]", arity=2),
        )
        ff = SymbolicForceField({InteractionClass.BOND: ts}, reg)

        matcher = FakeSmartsMatcher(
            {
                "[#6]-[#6]": torch.tensor([[0], [1]], dtype=torch.long),
                "[#6]-[#8]": torch.tensor([[1, 2], [2, 3]], dtype=torch.long),
            }
        )
        assigned = ff.match_molecule(mol=object(), matcher=matcher)
        assert InteractionClass.BOND in assigned
        matches = assigned[InteractionClass.BOND]["matches"]
        type_ids = assigned[InteractionClass.BOND]["type_ids"]
        assert matches.shape[0] == 2  # arity
        assert matches.shape[1] == 3  # 1 + 2 hits
        assert type_ids.tolist() == [0, 1, 1]
        # first column is the C-C hit
        assert matches[:, 0].tolist() == [0, 1]

    def test_no_hits_omits_interaction(self):
        ts = _bond_type_system()
        reg = ClassPatternRegistry()
        reg.bind(
            InteractionClass.BOND,
            0,
            SymbolicPattern("[#6]-[#6]", arity=2),
        )
        ff = SymbolicForceField({InteractionClass.BOND: ts}, reg)
        matcher = FakeSmartsMatcher()  # all empty
        assigned = ff.match_molecule(mol=None, matcher=matcher)
        assert assigned == {}

    def test_forcefield_module_has_no_molpot_imports(self):
        src = FORCEFIELD_PATH.read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("molpot"), alias.name
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                assert not mod.startswith("molpot"), mod

    def test_perception_package_has_no_molpot_energy_imports(self):
        for path in PERCEPTION_ROOT.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                    assert not mod.startswith("molpot"), f"{path.name}: {mod}"
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        assert not alias.name.startswith("molpot"), f"{path.name}: {alias.name}"

    def test_type_system_interaction_mismatch_raises(self):
        ts = TypeSystem(
            InteractionClass.ANGLE,
            [TypeRecord(type_id=0, prototype={"k": 50.0, "theta0": 1.9})],
        )
        reg = ClassPatternRegistry()
        with pytest.raises(ValueError, match="mapped under"):
            SymbolicForceField({InteractionClass.BOND: ts}, reg)
