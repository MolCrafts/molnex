"""Unit tests for :class:`molix.profiler.task.Profiler`.

Mirrors ``src/molix/profiler/task.py``, which had no test module of its own
— the dispatch it performs was entirely unpinned.

The contract under test is *structural*, not nominal (spec
molix-data-dissolution-03-transforms): the profiler times
``task.execute(sample)`` when the task carries an ``execute``, falls back to
calling the task, fits first when the task carries a callable ``fit``, and
refuses anything that is neither — all without importing the acquisition-side
task hierarchy it used to type-check against. That hierarchy has since been
deleted from the repository (spec molix-data-dissolution-04-etl-exit), so the
import pin below is expressed as an allowlist rather than a named module.
"""

from __future__ import annotations

import pytest

from molix.profiler.task import Profiler

# ---------------------------------------------------------------------------
# Stubs — one behaviour each, so a failure names the dispatch branch it broke
# ---------------------------------------------------------------------------


class _Source:
    """Minimal ``DataSource``: ``__len__`` + ``__getitem__`` + ``describe``.

    Samples are ``{"i": idx}`` so a task can record exactly which indices it
    was handed, in order.
    """

    def __init__(self, n: int = 4) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict[str, int]:
        return {"i": idx}

    def describe(self) -> str:
        return f"_Source(n={self.n})"


class _ExecuteTask:
    """Structural task: a ``task_id`` and an ``execute``, no base class."""

    task_id = "execute-task:v1"

    def __init__(self) -> None:
        self.seen: list[int] = []

    def execute(self, data: dict[str, int]) -> dict[str, int]:
        self.seen.append(data["i"])
        return data


class _CallableTask:
    """Plain callable — no ``execute`` attribute at all."""

    def __init__(self) -> None:
        self.seen: list[int] = []

    def __call__(self, data: dict[str, int]) -> dict[str, int]:
        self.seen.append(data["i"])
        return data


class _ExecuteAndCallableTask:
    """Carries both; ``execute`` is the documented winner."""

    def __init__(self) -> None:
        self.via_execute: list[int] = []
        self.via_call: list[int] = []

    def execute(self, data: dict[str, int]) -> dict[str, int]:
        self.via_execute.append(data["i"])
        return data

    def __call__(self, data: dict[str, int]) -> dict[str, int]:
        self.via_call.append(data["i"])
        return data


class _FittableTask:
    """Two-phase task: a callable ``fit`` plus ``execute``, no base class."""

    task_id = "fittable-task:v1"

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.fitted: list[dict[str, int]] | None = None

    def fit(self, samples: list[dict[str, int]]) -> None:
        self.calls.append("fit")
        self.fitted = list(samples)

    def execute(self, data: dict[str, int]) -> dict[str, int]:
        self.calls.append("execute")
        return data


class _NotATask:
    """Neither ``execute`` nor ``__call__``."""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestProfiler:
    def test_execute_object_is_timed_through_execute(self) -> None:
        """A task carrying ``execute`` is dispatched through it, once per sample."""
        task = _ExecuteTask()
        result = Profiler(task).run(_Source(n=4), n_samples=3, n_warmup=1)

        assert task.seen == [0, 1, 2, 3]  # warmup + measured, cycling the source
        assert result.n_samples == 3
        assert result.task_name == "_ExecuteTask"
        assert result.task_id == "execute-task:v1"
        assert result.timing.mean_ms >= 0.0

    def test_plain_callable_is_timed_through_call(self) -> None:
        """A task with no ``execute`` falls back to calling the task itself."""
        task = _CallableTask()
        result = Profiler(task).run(_Source(n=4), n_samples=3, n_warmup=1)

        assert task.seen == [0, 1, 2, 3]
        assert result.n_samples == 3

    def test_execute_wins_when_task_is_also_callable(self) -> None:
        """``execute`` takes precedence over ``__call__`` when both exist."""
        task = _ExecuteAndCallableTask()
        Profiler(task).run(_Source(n=2), n_samples=2, n_warmup=0)

        assert task.via_execute == [0, 1]
        assert task.via_call == []

    def test_fit_runs_once_over_the_whole_source_before_timing(self) -> None:
        """A callable ``fit`` is invoked once, with every sample, before timing."""
        task = _FittableTask()
        Profiler(task).run(_Source(n=4), n_samples=2, n_warmup=0)

        assert task.fitted == [{"i": 0}, {"i": 1}, {"i": 2}, {"i": 3}]
        assert task.calls == ["fit", "execute", "execute"]

    def test_non_callable_task_raises_type_error(self) -> None:
        """A task that is neither runnable nor callable is rejected, not skipped."""
        with pytest.raises(TypeError, match="_NotATask"):
            Profiler(_NotATask()).run(_Source(n=2), n_samples=1, n_warmup=0)

    def test_task_id_falls_back_to_type_name(self) -> None:
        """Without a ``task_id`` attribute the report is keyed by the type name."""
        result = Profiler(_CallableTask()).run(_Source(n=2), n_samples=1, n_warmup=0)

        assert result.task_id == "_CallableTask"
