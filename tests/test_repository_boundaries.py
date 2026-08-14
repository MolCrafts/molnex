"""Repository dependency-boundary checks."""

from __future__ import annotations

import ast
from pathlib import Path


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
