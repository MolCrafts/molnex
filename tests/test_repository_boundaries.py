"""Repository dependency-boundary checks."""

from __future__ import annotations

import ast
import importlib.util
import re
from collections.abc import Iterator
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

#: Trees that must never mention the dissolved ETL packages.
_SCANNED_ROOTS = ("src", "tests", "scripts", "regressions", "benchmarks")

#: Text formats worth scanning; prose rot counts as a violation, so ``.md`` and
#: the batch-submission scripts are in scope alongside Python.
_SCANNED_SUFFIXES = frozenset({".py", ".pyi", ".md", ".txt", ".yaml", ".yml", ".sbatch"})

#: Directory names that hold generated or vendored content, not repository code.
_SKIPPED_DIRS = frozenset({"__pycache__", ".git", ".tox", ".venv", ".pytest_cache", "node_modules"})

#: ``molix.data`` / ``molix.datasets``, in both dotted and path-slash form.
#: ``\b`` after the stem keeps ``molix.database`` / ``molix.data_module`` clean,
#: and the ``[./]`` separator keeps the ``molix-data-dissolution`` slug clean.
_MOLIX_DATA_REFERENCE = re.compile(r"\bmolix[./]data(?:sets)?\b")


def _is_generated_dir(name: str) -> bool:
    """Whether a directory component holds generated output rather than sources."""
    return name in _SKIPPED_DIRS or name == "build" or name.startswith("build-")


def _is_exempt(path: Path) -> bool:
    """Whether *path* is allowed to spell ``molix.data``.

    Two carve-outs, both of which exist precisely to assert the *absence* of
    the packages:

    1. this file — it carries the pattern that defines the rule;
    2. the ``molix-data-dissolution-*`` regression scripts at the root of
       ``regressions/``, whose negative assertions
       (``find_spec("molix.data.cache") is None``) cannot be written without
       naming the vacated module.

    Everything else is a violation, including comments and docstrings.
    """
    if path == Path(__file__).resolve():
        return True
    return path.parent == (_REPO_ROOT / "regressions").resolve() and path.name.startswith(
        "molix-data-dissolution-"
    )


def _scanned_files() -> Iterator[Path]:
    """Yield every text file under the scanned roots, skipping generated trees."""
    for root_name in _SCANNED_ROOTS:
        root = (_REPO_ROOT / root_name).resolve()
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in _SCANNED_SUFFIXES:
                continue
            directories = path.relative_to(root).parts[:-1]
            if any(_is_generated_dir(part) for part in directories):
                continue
            if _is_exempt(path):
                continue
            yield path


def test_molnex_library_does_not_import_molhub() -> None:
    """Concrete dataset adapters belong to MolHub, never MolNex library code."""
    source_root = Path(__file__).parents[1] / "src"
    violations: list[str] = []
    for path in sorted(source_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name == "molhub" or name.startswith("molhub.") for name in names):
                violations.append(str(path.relative_to(source_root.parent)))
                break

    assert violations == [], (
        "MolNex library code must remain dataset-agnostic; move these MolHub "
        f"imports to MolHub or the Molexp experiment: {violations}"
    )


def test_molix_data_packages_are_gone() -> None:
    """The ETL layer was deleted outright — no package, no shim, no alias."""
    resolvable = [
        name
        for name in ("molix.data", "molix.datasets")
        if importlib.util.find_spec(name) is not None
    ]

    assert resolvable == [], (
        "data acquisition belongs to MolHub; these packages must not resolve "
        "(stage: experimental — deleted without a deprecation path, so a bare "
        f"directory left behind as a namespace package also fails): {resolvable}"
    )


def test_no_molix_data_references() -> None:
    """No file may mention the dissolved packages — comments and prose included."""
    violations: list[str] = []
    for path in _scanned_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "molix" not in text:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if _MOLIX_DATA_REFERENCE.search(line):
                relative = path.relative_to(_REPO_ROOT)
                violations.append(f"{relative}:{lineno}: {line.strip()}")

    listing = "\n".join(violations)
    assert violations == [], (
        "molix.data / molix.datasets no longer exist; retarget these at "
        "molix.batch / molix.store / molix.io, or at MolHub for dataset "
        f"acquisition ({len(violations)} reference(s)):\n{listing}"
    )
