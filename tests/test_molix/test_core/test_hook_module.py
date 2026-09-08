"""Structural tests for the new ``molix.core.hook`` module.

These tests pin the invariants of the io-hooks-refactor: the contract
layer (`Hook` Protocol, `BaseHook`, `ScalarHook`) lives in
``molix.core.hook`` and depends on nothing from the concrete-hook /
io / recorder side-tiers.

"""

from __future__ import annotations

import ast
import inspect
import pathlib


def test_module_importable() -> None:
    """``from molix.core.hook import Hook, BaseHook, ScalarHook`` succeeds."""
    from molix.core.hook import BaseHook, Hook, ScalarHook  # noqa: F401


def test_module_does_not_import_concrete_layers() -> None:
    """``core/hook.py`` imports nothing from the side-tiers.

    Forbidden: ``molix.hooks``, ``molix.io``, ``molix.recorder``. The
    contract layer must stay free of dependencies on concrete hook
    implementations and the IO backend.
    """
    import molix.core.hook as mod

    source_path = pathlib.Path(inspect.getfile(mod))
    tree = ast.parse(source_path.read_text())

    forbidden = ("molix.hooks", "molix.io", "molix.recorder")
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            if any(node.module == f or node.module.startswith(f + ".") for f in forbidden):
                offenders.append(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if any(alias.name == f or alias.name.startswith(f + ".") for f in forbidden):
                    offenders.append(alias.name)

    assert not offenders, f"core/hook.py must not import {forbidden}; got {offenders}"
