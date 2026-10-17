"""The molix package as a whole: the one-way dependency molix <- molrep <- molzoo / molpot."""

from __future__ import annotations

import ast
from pathlib import Path

import molix

#: The one documented molix -> molpot edge (CLAUDE.md, ``.claude/notes/architecture.md``).
_EXCEPTIONS = {"ff_export"}


def test_molix_imports_no_science_package() -> None:
    root = Path(molix.__file__).resolve().parent
    offenders: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if path.relative_to(root).parts[0] in _EXCEPTIONS:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                names = [node.module]
            offenders += [
                f"{path.relative_to(root)}: {name}"
                for name in names
                if name.split(".")[0] in {"molpot", "molrep", "molzoo"}
            ]
    assert not offenders, offenders
