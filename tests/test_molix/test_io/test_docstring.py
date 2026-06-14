"""Docstring discipline check for the new ``molix.io`` + ``molix.hooks.journal`` surface.

Acceptance trace:
    ac-012 — public docstrings on relocated classes follow Google
             style; ``JournalWriter``, ``JournalReader``,
             ``JournalHook`` each carry "Args:" sections on their
             public methods.
"""

from __future__ import annotations

import inspect

from molix.hooks import JournalHook
from molix.io import JournalReader, JournalWriter


def _has_args_section(docstring: str | None) -> bool:
    return docstring is not None and "Args:" in docstring


def _has_returns_section(docstring: str | None) -> bool:
    return docstring is not None and "Returns:" in docstring


def test_class_docstrings_present() -> None:
    """Every relocated public class carries a non-empty docstring."""
    for cls in (JournalWriter, JournalReader, JournalHook):
        assert cls.__doc__, f"{cls.__name__} missing class docstring"
        # Class docstring should describe arguments since these are
        # constructor-driven services.
        assert _has_args_section(cls.__doc__), (
            f"{cls.__name__} class docstring must include an Args: section"
        )


def test_journalwriter_methods_have_args() -> None:
    """``JournalWriter.append`` and other public methods document args."""
    assert _has_args_section(JournalWriter.append.__doc__), (
        "JournalWriter.append docstring must include Args: section"
    )
    # close / flush are simple lifecycle methods — Args: section optional.
    assert JournalWriter.close.__doc__
    assert JournalWriter.flush.__doc__


def test_journalreader_methods_document_returns() -> None:
    """``records``/``scalars``/``keys`` document return shapes."""
    assert _has_returns_section(JournalReader.scalars.__doc__), (
        "JournalReader.scalars must document its return tuple shape"
    )
    # records() yields plain dicts and documents the dict layout.
    assert JournalReader.records.__doc__ is not None
    assert "dict" in JournalReader.records.__doc__.lower()


def test_journalhook_documents_lifecycle() -> None:
    """``JournalHook`` lifecycle callbacks each have a docstring."""
    for name in (
        "on_train_start",
        "on_train_batch_end",
        "on_eval_step_complete",
        "on_epoch_end",
        "on_train_end",
    ):
        method = getattr(JournalHook, name)
        assert method.__doc__, f"JournalHook.{name} missing docstring"


def test_journalwriter_append_signature_matches_kwargs_contract() -> None:
    """ac-004 cross-check: ``append`` is keyword-only (post-refactor contract)."""
    sig = inspect.signature(JournalWriter.append)
    params = list(sig.parameters.values())
    # First parameter is `self`; everything after must be KEYWORD_ONLY.
    for p in params[1:]:
        assert p.kind == inspect.Parameter.KEYWORD_ONLY, (
            f"JournalWriter.append param {p.name!r} must be KEYWORD_ONLY; got {p.kind}"
        )
