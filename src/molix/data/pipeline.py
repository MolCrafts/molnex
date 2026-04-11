"""Pipeline DSL and spec for composing data tasks.

Usage::

    pipe = pipeline("qm9")
    pipe.add(AtomicDress(elements=[1, 6, 7, 8, 9]))
    pipe.add(NeighborList(cutoff=5.0))
    spec = pipe.build()

    samples = spec.prepare(source, cache_dir="./cache")
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import torch

from molix.data.task import BatchTask, DatasetTask, Runnable, SampleTask


class TaskEntry:
    """Registration of a single task inside a pipeline."""

    __slots__ = ("name", "task")

    def __init__(self, name: str, task: Any) -> None:
        self.name = name
        self.task = task


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def _call_task(task: Any, data: Any) -> Any:
    """Dispatch: Runnable.execute → callable → TypeError."""
    if isinstance(task, Runnable):
        return task.execute(data)
    if callable(task):
        return task(data)
    raise TypeError(f"Task is neither Runnable nor callable: {type(task)}")


# ---------------------------------------------------------------------------
# PipelineSpec
# ---------------------------------------------------------------------------

class PipelineSpec:
    """Compiled, immutable pipeline description.

    Structurally aligned with ``molexp.WorkflowSpec``: has a deterministic
    ``pipeline_id`` and a ``to_dict()`` for serialisation.
    """

    def __init__(
        self, name: str, pipeline_id: str, entries: list[TaskEntry]
    ) -> None:
        self.name = name
        self.pipeline_id = pipeline_id
        self.entries = entries

    # -- Task grouping (isinstance, zero strings) ---------------------------

    @property
    def prepare_tasks(self) -> list[TaskEntry]:
        """Tasks executed during :meth:`prepare` (SampleTask + DatasetTask + bare callables)."""
        return [
            e
            for e in self.entries
            if isinstance(e.task, (SampleTask, DatasetTask))
            or (callable(e.task) and not isinstance(e.task, BatchTask))
        ]

    @property
    def batch_tasks(self) -> list[TaskEntry]:
        """Tasks executed post-collate inside DataLoader."""
        return [e for e in self.entries if isinstance(e.task, BatchTask)]

    # -- Core: prepare ------------------------------------------------------

    def prepare(
        self,
        source: Any,
        *,
        fit_samples: list[dict] | None = None,
        cache_dir: str | Path | None = None,
    ) -> list[dict]:
        """Fit + transform + cache → ``list[Sample]``.

        In DDP the :class:`DataModule` ensures only rank 0 calls this
        with ``fit_samples``; other ranks call after a barrier and hit
        the cache path.

        Args:
            source: :class:`DataSource` providing raw samples.
            fit_samples: Samples for :class:`DatasetTask` fitting.
                If *None*, all source samples are used.
            cache_dir: Root directory for disk cache.
        """
        cache_path = self._cache_path(source, cache_dir)

        # Fast path: load from cache
        if cache_path is not None and (cache_path / "meta.json").exists():
            return self._load_cache(cache_path)

        # Fit DatasetTasks, then apply all prepare tasks
        samples = fit_samples if fit_samples is not None else [
            source[i] for i in range(len(source))
        ]
        current = list(samples)
        for entry in self.prepare_tasks:
            if isinstance(entry.task, DatasetTask):
                entry.task.fit(current)
            current = [_call_task(entry.task, s) for s in current]

        # If fit_samples was a subset, re-execute on the full source
        if fit_samples is not None:
            result: list[dict] = []
            for i in range(len(source)):
                s = source[i]
                for entry in self.prepare_tasks:
                    s = _call_task(entry.task, s)
                result.append(s)
        else:
            result = current

        if cache_path is not None:
            self._save_cache_atomic(cache_path, result)

        return result

    # -- Cache (single-file, atomic) ----------------------------------------

    def _cache_path(self, source: Any, cache_dir: str | Path | None) -> Path | None:
        if cache_dir is None:
            return None
        key = hashlib.sha256(
            f"{source.source_id}|{self.pipeline_id}".encode()
        ).hexdigest()[:16]
        return Path(cache_dir) / key

    def _save_cache_atomic(self, path: Path, samples: list[dict]) -> None:
        tmp = path.with_name(path.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)

        # Single-file save (performance: 1 syscall, not N)
        torch.save(samples, tmp / "samples.pt")

        # DatasetTask fitted state
        states: dict[str, Any] = {}
        for entry in self.entries:
            if isinstance(entry.task, DatasetTask):
                states[entry.name] = entry.task.state_dict()
        if states:
            torch.save(states, tmp / "task_states.pt")

        meta = {"n": len(samples), "pipeline_id": self.pipeline_id}
        (tmp / "meta.json").write_text(json.dumps(meta))

        # Atomic swap
        if path.exists():
            shutil.rmtree(path)
        tmp.rename(path)

    def _load_cache(self, path: Path) -> list[dict]:
        samples: list[dict] = torch.load(
            path / "samples.pt", weights_only=False
        )

        state_path = path / "task_states.pt"
        if state_path.exists():
            states = torch.load(state_path, weights_only=False)
            for entry in self.entries:
                if entry.name in states and isinstance(entry.task, DatasetTask):
                    entry.task.load_state_dict(states[entry.name])

        return samples

    # -- Serialisation ------------------------------------------------------

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
                for e in self.entries
            ],
        }


# ---------------------------------------------------------------------------
# DSL
# ---------------------------------------------------------------------------

class PipelineDSL:
    """Builder for constructing a :class:`PipelineSpec`.

    Supports three equivalent task registration methods:

    1. ``@pipe.task`` decorator (bare function → treated as SampleTask).
    2. ``pipe.add(TaskInstance())`` for typed tasks.
    3. ``pipe.add(any_callable, name="x")`` for third-party integration.
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

    def add(self, task: Any, *, name: str | None = None) -> PipelineDSL:
        """Add a Task instance or callable."""
        entry_name = name or getattr(task, "task_id", type(task).__name__)
        self._entries.append(TaskEntry(entry_name, task))
        return self

    def build(self) -> PipelineSpec:
        pid = _stable_pipeline_id(self.name, self._entries)
        return PipelineSpec(self.name, pid, list(self._entries))


def pipeline(name: str) -> PipelineDSL:
    """Create a new pipeline builder."""
    return PipelineDSL(name)


def _stable_pipeline_id(name: str, entries: list[TaskEntry]) -> str:
    parts = [name]
    for e in entries:
        tid = getattr(e.task, "task_id", type(e.task).__qualname__)
        parts.append(f"{e.name}:{tid}:{type(e.task).__name__}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
