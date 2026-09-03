"""Tests for molix.md.forcefield — re-export surface only."""

import pytest

_DELETED_FORCEFIELD_NAMES = (
    "CallableForceField",
    "HarmonicForceField",
    "LennardJonesForceField",
    "LennardJonesCutForceField",
    "PotentialForceField",
    "PeriodicPotentialForceField",
)


class TestForceFieldReexport:
    """``molix.md.ForceField`` is identically ``molpy.ForceField``."""

    def test_forcefield_is_identically_molpy_forcefield(self):
        from molpy import ForceField as MolpyForceField

        from molix.md import ForceField

        assert ForceField is MolpyForceField

    def test_deleted_names_are_absent_from_package_all(self):
        import molix.md as md

        exported = set(md.__all__)
        for name in _DELETED_FORCEFIELD_NAMES:
            assert name not in exported

    def test_deleted_names_are_not_package_attributes(self):
        import molix.md as md

        for name in _DELETED_FORCEFIELD_NAMES:
            assert getattr(md, name, None) is None

    def test_importing_deleted_names_from_forcefield_module_raises(self):
        import importlib

        module = importlib.import_module("molix.md.forcefield")
        for name in _DELETED_FORCEFIELD_NAMES:
            with pytest.raises(ImportError):
                try:
                    getattr(module, name)
                except AttributeError as exc:
                    raise ImportError(
                        f"cannot import name {name!r} from 'molix.md.forcefield'"
                    ) from exc
                raise AssertionError(f"{name} is still importable from molix.md.forcefield")


class TestProtocolReexports:
    """``CellBound`` / ``CoreCompilable`` are re-exported here, declared in schema."""

    def test_protocols_are_aliased_beside_mdforcefield_without_restating_them(self):
        """Same objects, listed in ``__all__``; a restated contract is two contracts."""
        import inspect

        import molix.schema
        from molix.md import forcefield

        assert forcefield.CellBound is molix.schema.CellBound
        assert forcefield.CoreCompilable is molix.schema.CoreCompilable
        assert "CellBound" in forcefield.__all__
        assert "CoreCompilable" in forcefield.__all__
        assert "MDForceField" in forcefield.__all__
        source = inspect.getsource(forcefield).lower()
        assert "idempotent" not in source
        assert "never retain" not in source
        assert "never truncate" not in source
