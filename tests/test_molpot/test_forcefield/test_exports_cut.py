"""Hard-cut: banned Class-I product names absent; Parameterizer exports present."""

from __future__ import annotations

from pathlib import Path

import pytest

_BANNED = (
    "ClassIForceField",
    "OPLSForceField",
    "ClassicalIMM",
    "ClassIPredictions",
)

_ROOT = Path(__file__).resolve().parents[3]
_SCAN_PATHS = (
    _ROOT / "src" / "molpot" / "__init__.py",
    _ROOT / "src" / "molpot" / "forcefield" / "__init__.py",
    *(_ROOT / "scripts" / "mm_param_learning").rglob("*.py"),
)


class TestBannedNameAbsence:
    def test_source_trees_contain_zero_banned_identifiers(self) -> None:
        hits: list[str] = []
        for path in _SCAN_PATHS:
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            for name in _BANNED:
                if name in text:
                    hits.append(f"{path.relative_to(_ROOT)}:{name}")
        assert hits == [], f"banned names still present: {hits}"

    @pytest.mark.parametrize("name", list(_BANNED))
    def test_banned_name_raises_import_error_from_molpot(self, name: str) -> None:
        with pytest.raises(ImportError):
            exec(f"from molpot import {name}", {})

    @pytest.mark.parametrize("name", list(_BANNED))
    def test_banned_name_raises_import_error_from_forcefield(self, name: str) -> None:
        with pytest.raises(ImportError):
            exec(f"from molpot.forcefield import {name}", {})


class TestPublicExports:
    def test_parameterizer_oplsaa_predictions_in_molpot_all(self) -> None:
        import molpot

        for name in ("Parameterizer", "OPLSAAForceField", "ForceFieldPredictions"):
            assert name in molpot.__all__
            assert getattr(molpot, name) is not None

    def test_parameterizer_oplsaa_predictions_in_forcefield_all(self) -> None:
        import molpot.forcefield as ff

        for name in ("Parameterizer", "OPLSAAForceField", "ForceFieldPredictions"):
            assert name in ff.__all__
            assert getattr(ff, name) is not None

    def test_banned_names_absent_from_all_lists(self) -> None:
        import molpot
        import molpot.forcefield as ff

        for name in _BANNED:
            assert name not in molpot.__all__
            assert name not in ff.__all__
