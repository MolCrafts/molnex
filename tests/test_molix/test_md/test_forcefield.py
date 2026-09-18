"""Unit tests for molix.md.forcefield — the MD-side force-field contract.

Mirrors ``src/molix/md/forcefield.py``: the re-export is molpy's own class and
``MDForceField`` is the nullary ``to_potentials`` + ``units`` protocol the
driver binds.
"""

from __future__ import annotations

from molpy import ForceField as MolpyForceField

from molix.md.forcefield import CellBound, ForceField, MDForceField


def test_forcefield_is_the_molpy_class() -> None:
    assert ForceField is MolpyForceField


def test_md_forcefield_is_satisfied_by_a_realized_molpot_field() -> None:
    from molpot.forcefield import ForceField as MolpotForceField

    field = MolpotForceField("stub", units="real")
    assert isinstance(field, MDForceField)
    assert field.units == "real"
    assert not isinstance(object(), MDForceField)


def test_cellbound_is_the_schema_protocol() -> None:
    from molix.schema import CellBound as SchemaCellBound

    assert CellBound is SchemaCellBound
