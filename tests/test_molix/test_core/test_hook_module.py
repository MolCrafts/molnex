"""Structural tests for the new ``molix.core.hook`` module.

These tests pin the invariants of the io-hooks-refactor: the contract
layer (`Hook` Protocol, `BaseHook`, `ScalarHook`) lives in
``molix.core.hook`` and depends on nothing from the concrete-hook /
io / recorder side-tiers.

Trace to acceptance:
    ac-001 — core/hook.py exposes Hook, BaseHook, ScalarHook with no
             external imports.
    ac-008 — Trainer depends only on core.hook (verified indirectly
             via the import-isolation assertion below).
"""

from __future__ import annotations

import ast
import inspect
import pathlib
from typing import Protocol


def test_module_importable() -> None:
    """``from molix.core.hook import Hook, BaseHook, ScalarHook`` succeeds."""
    from molix.core.hook import BaseHook, Hook, ScalarHook  # noqa: F401


def test_hook_is_protocol() -> None:
    """``Hook`` is a typing.Protocol class."""
    from molix.core.hook import Hook

    assert issubclass(Hook, Protocol)


def test_hook_lifecycle_methods_present() -> None:
    """Every documented lifecycle callback is on the Protocol surface.

    The list mirrors the original `core/hooks.py:Hook` Protocol; this
    test fails loudly if a callback is silently dropped during the
    extract.
    """
    from molix.core.hook import Hook

    expected = {
        "on_train_start",
        "on_train_end",
        "on_epoch_start",
        "on_epoch_end",
        "on_train_batch_start",
        "on_train_batch_end",
        "on_eval_phase_start",
        "on_eval_batch_start",
        "on_after_backward",
        "on_eval_batch_end",
        "on_eval_step_complete",
    }
    actual = {name for name in dir(Hook) if name.startswith("on_")}
    assert expected == actual, f"missing or extra: {expected ^ actual}"


def test_base_hook_implements_protocol() -> None:
    """``BaseHook`` has no-op implementations of every Hook callback."""
    from molix.core.hook import BaseHook, Hook

    instance = BaseHook()
    for name in dir(Hook):
        if name.startswith("on_"):
            assert hasattr(instance, name)
            assert callable(getattr(instance, name))


def test_scalar_hook_extends_base_hook() -> None:
    """``ScalarHook`` is a subclass of ``BaseHook`` with ``scalar_keys`` attr."""
    from molix.core.hook import BaseHook, ScalarHook

    assert issubclass(ScalarHook, BaseHook)
    assert hasattr(ScalarHook, "scalar_keys")
    assert ScalarHook.scalar_keys == ()


def test_module_does_not_import_concrete_layers() -> None:
    """ac-001: ``core/hook.py`` imports nothing from the side-tiers.

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


def test_trainer_imports_hook_from_core_hook_module() -> None:
    """Trainer pulls the ``Hook`` Protocol from ``molix.core.hook``.

    The concrete-hook split is long done (``molix.core.hooks`` no longer
    exists anywhere under ``src/``); this asserts the Hook contract lives
    at its canonical home.
    """
    import molix.core.trainer as trainer_mod

    source_path = pathlib.Path(inspect.getfile(trainer_mod))
    tree = ast.parse(source_path.read_text())

    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "molix.core.hook":
            for alias in node.names:
                if alias.name == "Hook":
                    found = True
                    break
    assert found, "trainer.py must `from molix.core.hook import Hook`"
