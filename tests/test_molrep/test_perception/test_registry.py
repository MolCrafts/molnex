"""ClassPatternRegistry bind/get/conflict (learnable-classical-ff-07)."""

from __future__ import annotations

import pytest

from molrep.condensation import InteractionClass
from molrep.perception import ClassPatternRegistry, SymbolicPattern


class TestClassPatternRegistry:
    def test_bind_then_get(self):
        reg = ClassPatternRegistry()
        pat = SymbolicPattern("[#6]-[#6]", arity=2)
        reg.bind(InteractionClass.BOND, 0, pat)
        got = reg.get(InteractionClass.BOND, 0)
        assert got is pat or got == pat
        assert got.pattern == "[#6]-[#6]"

    def test_idempotent_same_pattern(self):
        reg = ClassPatternRegistry()
        pat = SymbolicPattern("[#6]", arity=1)
        reg.bind(InteractionClass.LJ, 0, pat)
        reg.bind(InteractionClass.LJ, 0, pat)
        assert len(reg) == 1

    def test_conflicting_bind_raises(self):
        reg = ClassPatternRegistry()
        reg.bind(
            InteractionClass.BOND,
            0,
            SymbolicPattern("[#6]-[#6]", arity=2),
        )
        with pytest.raises(ValueError, match="conflicting"):
            reg.bind(
                InteractionClass.BOND,
                0,
                SymbolicPattern("[#6]-[#8]", arity=2),
            )

    def test_reverse_lookup(self):
        reg = ClassPatternRegistry()
        pat = SymbolicPattern("[#6]-[#8]", arity=2)
        reg.bind(InteractionClass.BOND, 3, pat)
        interaction, type_id = reg.reverse_lookup(pat)
        assert interaction is InteractionClass.BOND
        assert type_id == 3
        interaction2, type_id2 = reg.reverse_lookup("[#6]-[#8]")
        assert (interaction2, type_id2) == (InteractionClass.BOND, 3)

    def test_pattern_reuse_across_keys_raises(self):
        reg = ClassPatternRegistry()
        pat = SymbolicPattern("[#6]-[#6]", arity=2)
        reg.bind(InteractionClass.BOND, 0, pat)
        with pytest.raises(ValueError, match="already bound"):
            reg.bind(InteractionClass.BOND, 1, pat)

    def test_get_missing_raises(self):
        reg = ClassPatternRegistry()
        with pytest.raises(KeyError, match="no pattern"):
            reg.get(InteractionClass.ANGLE, 0)

    def test_get_optional_none(self):
        reg = ClassPatternRegistry()
        assert reg.get_optional(InteractionClass.ANGLE, 0) is None

    def test_contains(self):
        reg = ClassPatternRegistry()
        pat = SymbolicPattern("[#7]", arity=1)
        reg.bind(InteractionClass.CHARGE, 0, pat)
        assert (InteractionClass.CHARGE, 0) in reg
        assert "[#7]" in reg
        assert pat in reg
        assert (InteractionClass.BOND, 0) not in reg
