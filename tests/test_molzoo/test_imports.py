"""Tests for the ``molzoo`` public import surface (``src/molzoo/__init__.py``).

Two contracts live in that file and nowhere else:

* the **names** the rest of the world imports from the top level —
  ``scripts/matpes_port/run_nve.py:45``, ``benchmarks/bench_mace_matpes.py:38``
  and ``benchmarks/bench_trainer_throughput.py:26`` bind them by hand, so the
  ``mace`` sub-package cutover must not move a single one;
* the **lazy policy**: no model symbol is imported at module level, so
  ``import molzoo`` costs nothing beyond ``typing`` and the cuEquivariance
  stack only appears once a model symbol is actually touched
  (``.claude/specs/mace-subpackage-restructure-06-wire.md`` §Design 2).

The lazy assertions run in a **subprocess**. This pytest session imported
cuEquivariance long before this module was collected (any molzoo model test
does), so an in-process ``sys.modules`` assertion is vacuously green.

:class:`TestMolzooMaceReexports` additionally pins the *identity* of the
``molzoo.mace`` re-export surface against its promoted ``molrep`` home. It was
moved here from ``tests/test_molrep/test_reexport_compat.py`` by
``mace-subpackage-restructure-06-wire`` ac-006 (no ``molzoo`` import may appear
under ``tests/test_molrep/``); the assertions are unchanged.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import molrep.embedding.mace
import molrep.interaction.mace.block
import molzoo
import molzoo.mace

#: Repo root — ``tests/test_molzoo/test_imports.py`` → ``tests/test_molzoo`` →
#: ``tests`` → root. Used to put ``src/`` on the probe interpreter's path
#: whether or not the package happens to be installed.
_REPO_ROOT = Path(__file__).resolve().parents[2]

#: The import package root, prepended to ``PYTHONPATH`` for the probe.
_SRC = _REPO_ROOT / "src"

#: Third-party module whose presence in ``sys.modules`` marks "the equivariance
#: stack got imported". It is the expensive one the lazy policy exists to defer.
_CUEQ = "cuequivariance_torch"

#: Seconds the probe interpreter may take. Importing the cuEquivariance stack
#: is the slow half and takes ~10 s cold; the budget is deliberately loose
#: because a *hang* (not a slow import) is what this guards against.
_PROBE_TIMEOUT = 600.0

#: Probe script: a clean interpreter reports whether :data:`_CUEQ` is loaded
#: right after ``import molzoo`` and again after touching a model symbol.
#: It writes JSON to ``argv[1]`` rather than stdout because importing this
#: stack prints to stdout ("opt_einsum_fx not available."), which would have to
#: be parsed around.
_PROBE = """
import json
import sys
from pathlib import Path

import molzoo

report = {"at_import": "cuequivariance_torch" in sys.modules}
molzoo.MACE
report["after_attribute_access"] = "cuequivariance_torch" in sys.modules
Path(sys.argv[1]).write_text(json.dumps(report))
"""


@pytest.fixture(scope="module")
def lazy_probe(tmp_path_factory: pytest.TempPathFactory) -> dict[str, bool]:
    """Run :data:`_PROBE` once in a clean interpreter and return its report.

    Args:
        tmp_path_factory: pytest's session-scoped temporary directory factory —
            the probe's only filesystem contact.

    Returns:
        ``{"at_import": bool, "after_attribute_access": bool}``: whether
        :data:`_CUEQ` was in ``sys.modules`` after ``import molzoo`` and after
        ``molzoo.MACE``.
    """
    report_path = tmp_path_factory.mktemp("molzoo_lazy") / "probe.json"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(_SRC), environment.get("PYTHONPATH", "")) if part
    )
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE, str(report_path)],
        capture_output=True,
        text=True,
        env=environment,
        cwd=str(_REPO_ROOT),
        timeout=_PROBE_TIMEOUT,
        check=False,
    )
    assert completed.returncode == 0, (
        f"probe interpreter failed ({completed.returncode}):\n{completed.stderr}"
    )
    report: dict[str, bool] = json.loads(report_path.read_text())
    return report


class TestPublicImportSurface:
    """The names and the import cost of the ``molzoo`` public surface.

    Both levels of it: ``molzoo/__init__.py`` and the ``molzoo.mace``
    re-export package, which runs the same PEP 562 policy.
    """

    def test_top_level_exports_the_symbols_the_scripts_bind(self) -> None:
        """The six names ``run_nve.py`` / ``bench_mace_matpes.py`` import."""
        from molzoo import (
            MACE,
            MACEMatpes,
            MACEOMol,
            MACESpec,
            load_matpes_state_dict,
            load_omol_state_dict,
        )

        resolved = (
            MACE,
            MACESpec,
            MACEMatpes,
            MACEOMol,
            load_matpes_state_dict,
            load_omol_state_dict,
        )
        assert all(symbol is not None for symbol in resolved)

    def test_mace_sub_package_re_exports_the_research_encoder(self) -> None:
        """``bench_trainer_throughput.py:26`` imports through the package."""
        from molzoo.mace import MACE

        assert isinstance(MACE, type)

    def test_importing_molzoo_does_not_import_cuequivariance(
        self, lazy_probe: dict[str, bool]
    ) -> None:
        """``import molzoo`` must not pay for the equivariance stack."""
        assert lazy_probe["at_import"] is False

    def test_touching_a_model_symbol_imports_cuequivariance(
        self, lazy_probe: dict[str, bool]
    ) -> None:
        """Lazy, not absent: the symbol still resolves to the real class."""
        assert lazy_probe["after_attribute_access"] is True

    def test_unknown_attribute_raises_attribute_error(self) -> None:
        """``__getattr__`` must not turn a typo into an import error."""
        with pytest.raises(AttributeError):
            molzoo.NotAThing

    def test_dir_reports_exactly_the_public_surface(self) -> None:
        """``__dir__`` keeps completion and ``from molzoo import *`` alive."""
        assert set(dir(molzoo)) == set(molzoo.__all__)

    def test_mace_subpackage_dir_covers_its_public_surface(self) -> None:
        """``molzoo.mace.__dir__`` reports exactly the sub-package's ``__all__``.

        The mirror of :meth:`test_dir_reports_exactly_the_public_surface` one
        level down, and the ac-001 lock. ``molzoo.mace`` imports only its
        torch-free config models eagerly, so without ``__dir__`` sixteen of its
        eighteen exported names would be absent from ``dir()`` — completion
        would hide every lazy symbol while ``from molzoo.mace import *`` still
        bound it.

        ac-001's gate is the subset form ``set(__all__) <= set(dir())``; the
        equality asserted here is what ``src/molzoo/mace/__init__.py``'s
        ``__dir__`` actually guarantees and is strictly stronger, since it also
        catches a ``dir()`` that grew a name ``__all__`` does not export.
        """
        assert set(dir(molzoo.mace)) == set(molzoo.mace.__all__)


class TestMolzooMaceReexports:
    """``molzoo.mace`` re-exports the promoted embedding / interaction blocks."""

    def test_embedding_block_is_same_object(self) -> None:
        """``molzoo.mace.EmbeddingBlock`` must be the promoted molrep class."""
        assert molzoo.mace.EmbeddingBlock is molrep.embedding.mace.EmbeddingBlock

    def test_embedding_spec_is_same_object(self) -> None:
        """``molzoo.mace.EmbeddingSpec`` must be the promoted molrep class."""
        assert molzoo.mace.EmbeddingSpec is molrep.embedding.mace.EmbeddingSpec

    def test_interaction_block_is_same_object(self) -> None:
        """``molzoo.mace.InteractionBlock`` must be the promoted molrep class."""
        assert molzoo.mace.InteractionBlock is molrep.interaction.mace.block.InteractionBlock

    def test_interaction_spec_is_same_object(self) -> None:
        """``molzoo.mace.InteractionSpec`` must be the promoted molrep class."""
        assert molzoo.mace.InteractionSpec is molrep.interaction.mace.block.InteractionSpec
