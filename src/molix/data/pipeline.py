"""Pipeline DSL and spec for composing data transforms.

A :class:`PipelineSpec` is a pure transform description — it knows what
per-sample / per-dataset work to do, but not where results live on disk or
when to run. Those are scheduling concerns that belong to the caller
(workflow / user script).

Capabilities exposed:

* :meth:`PipelineSpec.transform` — apply prepare tasks to a single sample.
* :meth:`PipelineSpec.run` — iterate a :class:`~molix.data.DataSource`,
  fit any :class:`~molix.data.DatasetTask`, yield processed samples.
* :meth:`PipelineSpec.materialize` — convenience: run + write the
  molix-standard cache format to a resolved *sink* directory (the caller
  decides where the sink lives; the pipeline does not).
* :meth:`PipelineSpec.cache_identity` — stable hash for the
  (pipeline, source, fit_source) triple, suitable for naming cache dirs.
* :meth:`PipelineSpec.is_cache_ready` — probe a sink directory to see if
  a previous :meth:`materialize` committed successfully.

Usage::

    pipe = Pipeline("qm9").add(NeighborList(cutoff=5.0)).build()

    # In-memory:
    processed = list(pipe.run(source))

    # On-disk cache (workflow decides sink):
    sink = workspace / "data" / f"qm9__{pipe.cache_identity(source)}"
    if not pipe.is_cache_ready(sink):
        pipe.materialize(source, sink=sink)
    ds = MmapDataset.from_cache(sink)
"""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from molix.data.task import BatchTask, DatasetTask, Runnable, SampleTask

if TYPE_CHECKING:
    from molix.data.source import DataSource


# Bump when the cache identity schema changes in an identity-affecting way
# (e.g. new required meta fields, different serialization layout). Kept in
# sync with ``CACHE_SCHEMA_VERSION`` in ``molix.data.dataset``.
_IDENTITY_SCHEMA_VERSION = 1
_IDENTITY_HASH_LEN = 12


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
    :attr:`pipeline_id` and a :meth:`to_dict` for serialisation.

    Scheduling-free: this class exposes no ``workspace`` / ``cache_dir`` /
    ``cache_identity`` concepts. If a caller wants a shared on-disk asset,
    it resolves a concrete path and passes it as :meth:`materialize`'s
    ``sink`` argument.
    """

    def __init__(self, name: str, pipeline_id: str, entries: list[TaskEntry]) -> None:
        self.name = name
        self.pipeline_id = pipeline_id
        self.entries = entries

    # -- Task grouping (isinstance, zero strings) ---------------------------

    @property
    def prepare_tasks(self) -> list[TaskEntry]:
        """Tasks executed during :meth:`run` (SampleTask + DatasetTask + bare callables)."""
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

    # -- Transform (per-sample) --------------------------------------------

    def transform(self, sample: dict) -> dict:
        """Apply every prepare task to *sample* in order.

        Per-sample transform. Caller must ensure any
        :class:`~molix.data.DatasetTask` in the pipeline has been fitted
        (e.g. via a prior :meth:`run` / :meth:`materialize`); an unfitted
        :class:`DatasetTask` will raise per the task's own contract.
        """
        for entry in self.prepare_tasks:
            sample = _call_task(entry.task, sample)
        return sample

    # -- Run (iterate source) ----------------------------------------------

    def run(
        self,
        source: "DataSource",
        *,
        fit_source: "DataSource | None" = None,
    ) -> Iterator[dict]:
        """Iterate *source*, fit any :class:`DatasetTask`, yield processed samples.

        Pure in-memory; no disk I/O. Call :meth:`materialize` to persist
        results to the molix-standard cache format.

        Args:
            source: :class:`DataSource` providing raw samples.
            fit_source: Source used for :meth:`DatasetTask.fit`. Defaults to
                *source*; pass a distinct source (e.g. train-only split) to
                fit statistics without peeking at the full dataset.

        Yields:
            Processed sample dicts in *source* order.
        """
        prepare = self.prepare_tasks
        has_dataset_task = any(isinstance(e.task, DatasetTask) for e in prepare)

        # Case A: explicit fit_source (subset). Separate fit pass on the
        # subset, then apply tasks to the full source.
        if has_dataset_task and fit_source is not None:
            fit_data = [fit_source[i] for i in range(len(fit_source))]
            for entry in prepare:
                if isinstance(entry.task, DatasetTask):
                    entry.task.fit(fit_data)
                fit_data = [_call_task(entry.task, s) for s in fit_data]

            for i in range(len(source)):
                s = source[i]
                for entry in prepare:
                    s = _call_task(entry.task, s)
                yield s
            return

        # Case B: fit on the full source. Interleave fit() with execute() so
        # each task is applied to every sample exactly once (no double pass).
        buffered = [source[i] for i in range(len(source))]
        for entry in prepare:
            if isinstance(entry.task, DatasetTask):
                entry.task.fit(buffered)
            buffered = [_call_task(entry.task, s) for s in buffered]
        yield from buffered

    # -- Materialize (run + write standard cache format) -------------------

    def materialize(
        self,
        source: "DataSource",
        *,
        sink: str | Path,
        fit_source: "DataSource | None" = None,
        overwrite: bool = False,
        wait_timeout: float = 600.0,
    ) -> None:
        """Run the pipeline against *source* and write the result to *sink*.

        Convenience wrapper around :meth:`run` + :meth:`MmapDataset.write_cache`.

        *sink* is an already-resolved path — the caller (workflow / user
        script) decides where cache assets live and how they are named.
        Pipeline has no opinion on workspaces or directory conventions.

        Idempotent: if ``<sink>/meta.json`` has ``status == "ready"`` and
        ``overwrite=False``, this returns immediately. Callers typically
        invoke ``materialize`` unconditionally from their ``prepare_data``
        task; subsequent calls across sweep runs are no-ops.

        DDP defensive guard: when ``RANK`` env var is non-zero, this
        method polls ``<sink>/meta.json`` up to ``wait_timeout`` seconds
        instead of writing (protects against accidental concurrent
        materialization from worker ranks). Materialization should normally
        be driven from rank 0 / a dedicated ``prepare_data`` stage.

        Args:
            source: Source of raw samples.
            sink: Target cache directory. Its parent must exist or be
                creatable.
            fit_source: Optional fit-source (see :meth:`run`).
            overwrite: If True, replace a ready sink.
            wait_timeout: Seconds to poll for ``meta.json`` when invoked
                from a non-zero rank.
        """
        from molix.data.dataset import MmapDataset, _is_ready  # avoid import cycle

        sink = Path(sink)

        if _rank() != 0:
            _wait_for_ready(sink, timeout=wait_timeout)
            return

        if _is_ready(sink) and not overwrite:
            return

        processed = list(self.run(source, fit_source=fit_source))

        task_states: dict[str, Any] = {}
        for entry in self.entries:
            if isinstance(entry.task, DatasetTask):
                task_states[entry.name] = entry.task.state_dict()

        fit_source_id = (
            fit_source.source_id if fit_source is not None else source.source_id
        )

        MmapDataset.write_cache(
            sink,
            processed,
            pipeline_id=self.pipeline_id,
            source_id=source.source_id,
            fit_source_id=fit_source_id,
            pipeline_spec=self.to_dict(),
            task_states=task_states or None,
            overwrite=overwrite,
        )

    # -- Cache identity / readiness ----------------------------------------

    def cache_identity(
        self,
        source: "DataSource",
        *,
        fit_source: "DataSource | None" = None,
        extra: Mapping[str, str] | None = None,
    ) -> str:
        """Return a short stable hash for this ``(pipeline, source, fit_source)``.

        The formula folds in :attr:`pipeline_id`, ``source.source_id``, the
        fit source's id (defaults to *source*), and any workflow-local
        *extra* dimensions (e.g. ``{"impl": "v2"}``). Callers treat the
        return value as opaque and use it as part of a cache directory
        name.

        Args:
            source: Raw data source to be materialised.
            fit_source: Source used for :meth:`DatasetTask.fit`. Defaults
                to *source*; pass a distinct source (e.g. train-only
                split) when fitting statistics should not see the full
                dataset.
            extra: Additional string key/value pairs to fold into the
                hash. Keys are sorted for stability.
        """
        fs_id = fit_source.source_id if fit_source is not None else source.source_id
        parts = [
            f"schema={_IDENTITY_SCHEMA_VERSION}",
            f"pipeline_id={self.pipeline_id}",
            f"source_id={source.source_id}",
            f"fit_source_id={fs_id}",
        ]
        if extra:
            for k in sorted(extra):
                parts.append(f"{k}={extra[k]}")
        digest = hashlib.sha256("|".join(parts).encode()).hexdigest()
        return digest[:_IDENTITY_HASH_LEN]

    @staticmethod
    def is_cache_ready(path: str | Path) -> bool:
        """Return ``True`` if *path* is a cache dir committed by
        :meth:`materialize`.

        A cache is considered ready when its ``meta.json`` exists and
        carries ``"status": "ready"``. Use this in workflow
        ``prepare_data`` steps to skip expensive materialisation when a
        previous run already wrote the sink.
        """
        from molix.data.dataset import _is_ready  # avoid import cycle
        return _is_ready(Path(path))

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
# DDP helpers (defensive — materialize is expected to run on rank 0 only)
# ---------------------------------------------------------------------------


def _rank() -> int:
    try:
        return int(os.environ.get("RANK", "0"))
    except ValueError:
        return 0


def _wait_for_ready(sink: Path, *, timeout: float) -> None:
    """Poll ``<sink>/meta.json`` for ``status == "ready"`` up to *timeout* seconds."""
    from molix.data.dataset import _is_ready  # avoid import cycle

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _is_ready(sink):
            return
        time.sleep(0.5)
    raise TimeoutError(
        f"Timed out waiting {timeout:.0f}s for ready cache at {sink}. "
        "materialize() should be called from rank 0 (e.g. prepare_data) "
        "before workers start."
    )


# ---------------------------------------------------------------------------
# DSL
# ---------------------------------------------------------------------------


class Pipeline:
    """Builder for constructing a :class:`PipelineSpec`.

    Instantiate directly — there is no separate factory function.
    Supports three equivalent task registration styles::

        pipe = (
            Pipeline("qm9")
            .add(NeighborList(cutoff=5.0))          # typed task instance
            .add(my_callable, name="normalize")     # bare callable
            .task(                                  # @decorator form
                lambda s: {**s, "flag": True},
                name="flag",
            )
        )
        spec = pipe.build()                         # → PipelineSpec
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

    def add(self, task: Any, *, name: str | None = None) -> Pipeline:
        """Add a Task instance or callable. Returns *self* for chaining."""
        entry_name = name or getattr(task, "task_id", type(task).__name__)
        self._entries.append(TaskEntry(entry_name, task))
        return self

    def build(self) -> PipelineSpec:
        pid = _stable_pipeline_id(self.name, self._entries)
        return PipelineSpec(self.name, pid, list(self._entries))


def _stable_pipeline_id(name: str, entries: list[TaskEntry]) -> str:
    parts = [name]
    for e in entries:
        tid = getattr(e.task, "task_id", type(e.task).__qualname__)
        parts.append(f"{e.name}:{tid}:{type(e.task).__name__}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
