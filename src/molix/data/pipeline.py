"""Declarative pipeline container.

A :class:`PipelineSpec` is a pure description of *what tasks to run and in what
order*. It holds nothing about execution, IO, caching, or DDP — those are
workflow concerns, implemented as free functions in :mod:`molix.data.execute`
and :mod:`molix.data.cache`.

Typical usage::

    from molix.data import Pipeline, AtomicDress, NeighborList
    from molix.data.execute import run
    from molix.data.cache import cache, cache_key, is_ready

    pipe = (
        Pipeline("qm9-u0")
        .add(AtomicDress(elements=[1, 6, 7, 8, 9], target_key="U0"))
        .add(NeighborList(cutoff=5.0))
        .build()
    )

    # Execute (workflow):
    samples = list(run(pipe, source, fit_source=train_subset))

    # Cache (workflow):
    if not is_ready(sink):
        cache(pipe, source, sink=sink, fit_source=train_subset)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from molix.data.task import BatchTask, DatasetTask, SampleTask


__all__ = ["TaskEntry", "PipelineSpec", "Pipeline"]


@dataclass(frozen=True)
class TaskEntry:
    """A single registered task inside a pipeline."""

    name: str
    task: Any


class PipelineSpec:
    """Compiled, immutable pipeline description.

    Holds only *what* the pipeline does (task list, grouping, identity).
    It does not execute, serialize, or cache anything — those are workflow
    concerns, surfaced via :mod:`molix.data.execute` and
    :mod:`molix.data.cache`.
    """

    __slots__ = ("name", "pipeline_id", "tasks")

    def __init__(
        self,
        name: str,
        pipeline_id: str,
        tasks: tuple[TaskEntry, ...],
    ) -> None:
        self.name = name
        self.pipeline_id = pipeline_id
        self.tasks = tasks

    # -- grouping ----------------------------------------------------------

    @property
    def prepare_tasks(self) -> tuple[TaskEntry, ...]:
        """Tasks executed *before* the DataLoader (sample- and dataset-level)."""
        return tuple(
            e
            for e in self.tasks
            if isinstance(e.task, (SampleTask, DatasetTask))
            or (callable(e.task) and not isinstance(e.task, BatchTask))
        )

    @property
    def batch_tasks(self) -> tuple[TaskEntry, ...]:
        """Tasks executed *post-collate* inside the DataLoader."""
        return tuple(e for e in self.tasks if isinstance(e.task, BatchTask))

    # -- introspection -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "pipeline_id": self.pipeline_id,
            "tasks": [
                {
                    "name": e.name,
                    "type": type(e.task).__name__,
                    "task_id": getattr(e.task, "task_id", type(e.task).__name__),
                }
                for e in self.tasks
            ],
        }

    def __repr__(self) -> str:
        names = ", ".join(e.name for e in self.tasks)
        return f"PipelineSpec(name={self.name!r}, tasks=[{names}], id={self.pipeline_id})"


# ---------------------------------------------------------------------------
# Builder DSL
# ---------------------------------------------------------------------------


class Pipeline:
    """Fluent builder for a :class:`PipelineSpec`.

    Three equivalent task-registration styles::

        Pipeline("p").add(NeighborList(cutoff=5.0))        # Task instance
        Pipeline("p").add(my_callable, name="normalize")   # bare callable
        Pipeline("p").task(lambda s: s, name="noop")       # @decorator
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._entries: list[TaskEntry] = []

    def task(self, fn: Any = None, *, name: str | None = None) -> Any:
        """Register a bare function as a sample-level task."""

        def decorator(f: Any) -> Any:
            entry_name = name or getattr(f, "__name__", "task")
            self._entries.append(TaskEntry(entry_name, f))
            return f

        if fn is not None:
            return decorator(fn)
        return decorator

    def add(self, task: Any, *, name: str | None = None) -> "Pipeline":
        """Add a Task instance or plain callable. Returns self for chaining."""
        _validate_task(task)
        entry_name = name or getattr(task, "task_id", type(task).__name__)
        self._entries.append(TaskEntry(entry_name, task))
        return self

    def build(self) -> PipelineSpec:
        tasks = tuple(self._entries)
        return PipelineSpec(self.name, _stable_pipeline_id(self.name, tasks), tasks)


# ---------------------------------------------------------------------------
# Validation & identity
# ---------------------------------------------------------------------------


def _validate_task(task: Any) -> None:
    """Enforce: every registered task is a Task subclass or a plain callable."""
    if isinstance(task, (SampleTask, DatasetTask, BatchTask)):
        return
    if callable(task):
        return
    raise TypeError(
        f"Task must be a SampleTask/DatasetTask/BatchTask or callable, "
        f"got {type(task).__name__}"
    )


def _stable_pipeline_id(name: str, tasks: tuple[TaskEntry, ...]) -> str:
    """Deterministic 16-hex id derived from name + task composition."""
    parts = [name]
    for e in tasks:
        tid = getattr(e.task, "task_id", type(e.task).__qualname__)
        parts.append(f"{e.name}:{tid}:{type(e.task).__name__}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
