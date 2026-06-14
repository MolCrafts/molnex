"""RED tests for the ``molix.hooks`` package re-exports and JournalHook rename.

Acceptance traces:
    ac-007 — hooks/ package re-exports all 12 concrete hook classes with
             JournalHook rename; ``from molix.hooks import Journal``
             raises ImportError.
    ac-009 — Top-level molix package re-exports JournalHook (not Journal).
"""

from __future__ import annotations

import pytest


def test_top_level_imports_succeed() -> None:
    """All 12 concrete hooks importable from ``molix.hooks``."""
    from molix.hooks import (  # noqa: F401
        ActivationCheckpointingHook,
        CheckpointHook,
        GPUMemoryHook,
        GPUUtilsHook,
        GradClipHook,
        JournalHook,
        Log,
        MetricsHook,
        ProfilerHook,
        ProgressBarHook,
        StepSpeedHook,
        TensorBoardHook,
    )


def test_legacy_journal_name_unimportable() -> None:
    """ac-007: ``from molix.hooks import Journal`` raises ImportError."""
    with pytest.raises(ImportError):
        from molix.hooks import Journal  # noqa: F401


def test_per_module_imports() -> None:
    """Each concrete hook is importable from its assigned submodule."""
    from molix.hooks.checkpoint import CheckpointHook  # noqa: F401
    from molix.hooks.gpu import GPUMemoryHook, GPUUtilsHook  # noqa: F401
    from molix.hooks.journal import JournalHook  # noqa: F401
    from molix.hooks.profiler import ProfilerHook  # noqa: F401
    from molix.hooks.progress import Log, ProgressBarHook  # noqa: F401
    from molix.hooks.scalar import MetricsHook, StepSpeedHook  # noqa: F401
    from molix.hooks.tensorboard import TensorBoardHook  # noqa: F401
    from molix.hooks.training import ActivationCheckpointingHook, GradClipHook  # noqa: F401


def test_journal_hook_identity_preserves_behaviour() -> None:
    """JournalHook is the (renamed) ``Journal`` class; behaviour preserved."""
    from molix.core.hook import BaseHook
    from molix.hooks import JournalHook

    assert issubclass(JournalHook, BaseHook)
    # Lifecycle callbacks present.
    for name in ("on_train_start", "on_train_end", "on_train_batch_end", "on_eval_step_complete"):
        assert hasattr(JournalHook, name)


def test_top_level_molix_exports_journal_hook() -> None:
    """ac-009: ``from molix import JournalHook`` works; ``Journal`` does not."""
    import molix

    assert hasattr(molix, "JournalHook")
    assert "JournalHook" in molix.__all__
    with pytest.raises(ImportError):
        from molix import Journal  # noqa: F401


def test_metrics_hook_phase_isolation_module_path() -> None:
    """``MetricsHook`` and ``StepSpeedHook`` co-locate in ``hooks/scalar.py``.

    Encodes the cycle-3 D1 decision (scalar-publishing vs training-loop
    intervention split): if a future refactor merges them back into
    ``training.py`` it will need to update this assertion.
    """
    from molix.hooks.scalar import MetricsHook, StepSpeedHook

    assert MetricsHook.__module__ == "molix.hooks.scalar"
    assert StepSpeedHook.__module__ == "molix.hooks.scalar"


def test_training_module_holds_only_loop_interventions() -> None:
    """``hooks/training.py`` contains only training-loop intervention hooks."""
    from molix.hooks.training import ActivationCheckpointingHook, GradClipHook

    assert GradClipHook.__module__ == "molix.hooks.training"
    assert ActivationCheckpointingHook.__module__ == "molix.hooks.training"
