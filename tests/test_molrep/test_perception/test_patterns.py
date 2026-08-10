"""SymbolicPattern + DiscreteClassRecord validation (learnable-classical-ff-07)."""

from __future__ import annotations

import pytest

from molrep.condensation import InteractionClass
from molrep.perception import DiscreteClassRecord, SymbolicPattern


class TestSymbolicPattern:
    def test_valid_smarts_arity_2(self):
        pat = SymbolicPattern(pattern="[#6]-[#6]", arity=2)
        assert pat.pattern == "[#6]-[#6]"
        assert pat.arity == 2
        assert pat.kind == "smarts"

    def test_empty_pattern_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            SymbolicPattern(pattern="", arity=1)

    def test_whitespace_only_pattern_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            SymbolicPattern(pattern="   ", arity=2)

    def test_arity_0_raises(self):
        with pytest.raises(ValueError, match="arity"):
            SymbolicPattern(pattern="[#6]", arity=0)

    def test_arity_5_raises(self):
        with pytest.raises(ValueError, match="arity"):
            SymbolicPattern(pattern="[#6]", arity=5)

    def test_valid_arities(self):
        for a in (1, 2, 3, 4):
            pat = SymbolicPattern(pattern="[#6]", arity=a)
            assert pat.arity == a

    def test_smirks_kind(self):
        pat = SymbolicPattern(
            pattern="[C:1][O:2]>>[C:1][O:2]",
            arity=2,
            kind="smirks",
        )
        assert pat.kind == "smirks"

    def test_atom_maps_frozen_copy(self):
        maps = {0: "1", 1: "2"}
        pat = SymbolicPattern(pattern="[C:1][O:2]", arity=2, atom_maps=maps)
        maps[0] = "99"
        assert pat.atom_maps is not None
        assert pat.atom_maps[0] == "1"


class TestDiscreteClassRecord:
    def test_construct_unbound(self):
        rec = DiscreteClassRecord(
            interaction=InteractionClass.BOND,
            type_id=0,
            prototype={"k": 300.0, "r0": 1.09},
        )
        assert rec.smarts is None
        assert rec.smirks is None
        assert rec.prototype["k"] == 300.0

    def test_construct_with_smarts(self):
        rec = DiscreteClassRecord(
            interaction=InteractionClass.BOND,
            type_id=1,
            prototype={"k": 200.0, "r0": 1.5},
            smarts="[#6]-[#8]",
            label="C-O",
        )
        assert rec.smarts == "[#6]-[#8]"
        assert rec.label == "C-O"

    def test_negative_type_id_raises(self):
        with pytest.raises(ValueError, match="type_id"):
            DiscreteClassRecord(
                interaction=InteractionClass.ANGLE,
                type_id=-1,
                prototype={"k": 50.0, "theta0": 1.9},
            )

    def test_prototype_is_copied(self):
        proto = {"k": 1.0, "r0": 1.0}
        rec = DiscreteClassRecord(
            interaction=InteractionClass.BOND,
            type_id=0,
            prototype=proto,
        )
        proto["k"] = 999.0
        assert rec.prototype["k"] == 1.0
