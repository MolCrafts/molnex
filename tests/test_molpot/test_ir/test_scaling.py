"""Per-interaction-class exclusion scaling — mirrors ``molpot/ir/scaling.py``.

Provenance: OpenMM §19 / SMIRNOFF nonbonded section / Cornell 1995 AMBER.
1-2 and 1-3 interactions are fully excluded (scale 0); 1-4 electrostatics are
scaled by 5/6 and 1-4 LJ by 1/2.

``composable-a0-02`` turns the six flat floats into a *keyed set*: one frozen
:class:`~molpot.ir.scaling.ExclusionScales` triple per interaction class, read
back through :meth:`~molpot.ir.scaling.NonbondedScaling.kinds` /
:meth:`~molpot.ir.scaling.NonbondedScaling.triple`. That is the whole point of
the shape — ``composable-a1`` adds ``induced=`` as another field and every
consumer that iterates ``kinds()`` keeps working unchanged, so what is pinned
here is the *contract* (the key set, the lookup, the loud miss), not six
attribute names.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from molpot.ir.scaling import ExclusionScales, NonbondedScaling


class TestExclusionScales:
    def test_defaults_exclude_the_bonded_pairs_and_keep_the_one_four(self) -> None:
        """``(0, 0, 1)`` — the neutral triple a new interaction class starts from."""
        scales = ExclusionScales()
        assert (scales.s12, scales.s13, scales.s14) == (0.0, 0.0, 1.0)

    def test_is_frozen(self) -> None:
        """A topology-fixed table is built from it once; mutating it later lies."""
        scales = ExclusionScales()
        with pytest.raises(dataclasses.FrozenInstanceError):
            scales.s14 = 0.5  # ty: ignore[invalid-assignment]


class TestNonbondedScaling:
    def test_class_i_defaults_are_amber_gaff_smirnoff(self) -> None:
        """1-2 / 1-3 fully excluded; 1-4 is 5/6 for ``q`` and 1/2 for ``lj``."""
        scaling = NonbondedScaling()
        q12, q13, q14 = scaling.triple("q")
        lj12, lj13, lj14 = scaling.triple("lj")
        assert (q12, q13) == (0.0, 0.0)
        assert math.isclose(q14, 5.0 / 6.0, rel_tol=1e-12, abs_tol=1e-12)
        assert (lj12, lj13) == (0.0, 0.0)
        assert math.isclose(lj14, 0.5, rel_tol=1e-12, abs_tol=1e-12)

    def test_kinds_are_the_declared_permanent_interaction_classes(self) -> None:
        """``("q", "lj")`` today; ``composable-a1`` appends ``"induced"``."""
        assert NonbondedScaling().kinds() == ("q", "lj")

    def test_as_dict_is_nested_by_kind(self) -> None:
        """One sub-dict per kind — the export adapters flatten it, not this type."""
        nested = NonbondedScaling().as_dict()
        assert set(nested) == {"q", "lj"}
        assert set(nested["q"]) == {"s12", "s13", "s14"}
        assert set(nested["lj"]) == {"s12", "s13", "s14"}
        assert nested["lj"]["s14"] == 0.5
        assert math.isclose(nested["q"]["s14"], 5.0 / 6.0, rel_tol=1e-12, abs_tol=1e-12)

    def test_unknown_kind_raises_listing_the_declared_kinds(self) -> None:
        """A silent 1.0 for an unmodelled class is how a term escapes exclusion."""
        with pytest.raises(KeyError) as excinfo:
            NonbondedScaling().triple("induced")
        message = str(excinfo.value)
        assert "induced" in message
        assert "q" in message
        assert "lj" in message

    def test_is_frozen(self) -> None:
        """Same reason as :class:`ExclusionScales`: the table caches its answers."""
        scaling = NonbondedScaling()
        with pytest.raises(dataclasses.FrozenInstanceError):
            scaling.q = ExclusionScales()  # ty: ignore[invalid-assignment]
