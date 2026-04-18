"""Pipeline execution — workflow-facing free functions.

A :class:`~molix.data.pipeline.PipelineSpec` only describes *what* to run.
This module contains the execution logic:

* :func:`run` — iterate a source, fit every :class:`DatasetTask`, apply each
  prepare task, yield processed samples.
* :func:`transform` — apply every prepare task to a single already-fitted
  sample (inference-time path).
* :func:`collect_task_states` — gather ``state_dict()`` from every
  :class:`DatasetTask` after :func:`run`/:func:`cache`.
* :func:`load_task_states` — restore fitted state back into a pipeline.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, TYPE_CHECKING

from molix.data.task import DatasetTask, Runnable

if TYPE_CHECKING:
    from molix.data.pipeline import PipelineSpec
    from molix.data.source import DataSource


__all__ = [
    "call_task",
    "run",
    "transform",
    "collect_task_states",
    "load_task_states",
]


def call_task(task: Any, data: Any) -> Any:
    """Dispatch a single sample through *task*.

    Accepts either a :class:`Runnable` (uses ``execute``) or a plain callable.
    """
    if isinstance(task, Runnable):
        return task.execute(data)
    if callable(task):
        return task(data)
    raise TypeError(f"Task is neither Runnable nor callable: {type(task)}")


def transform(pipeline: "PipelineSpec", sample: dict) -> dict:
    """Apply every prepare task to *sample* in order.

    Per-sample inference transform. Any :class:`DatasetTask` in the pipeline
    must have been fitted already (e.g. via a prior :func:`run`); the task
    will raise if called unfit.
    """
    for entry in pipeline.prepare_tasks:
        sample = call_task(entry.task, sample)
    return sample


def run(
    pipeline: "PipelineSpec",
    source: "DataSource",
    *,
    fit_source: "DataSource | None" = None,
) -> Iterator[dict]:
    """Iterate *source*, fit every :class:`DatasetTask`, yield processed samples.

    Pure in-memory; no disk IO. For persistence, pipe into
    :func:`molix.data.cache.cache` (or call :class:`list` and hand the result
    to :func:`~molix.data.cache.save`).

    Args:
        pipeline: Declarative pipeline.
        source: Raw :class:`DataSource` — every sample is processed.
        fit_source: Source used for :meth:`DatasetTask.fit`. Defaults to
            *source*. Pass a distinct (e.g. train-only) source so that
            fit-dependent tasks never peek at validation / test data.

    Yields:
        Processed sample dicts in *source* order.
    """
    prepare = pipeline.prepare_tasks
    has_dataset_task = any(isinstance(e.task, DatasetTask) for e in prepare)

    # Case A: explicit fit_source (subset). Fit on the subset, then apply
    # every prepare task to the full source. Two passes, but the fit pass is
    # small.
    if has_dataset_task and fit_source is not None:
        fit_data = [fit_source[i] for i in range(len(fit_source))]
        for entry in prepare:
            if isinstance(entry.task, DatasetTask):
                entry.task.fit(fit_data)
            fit_data = [call_task(entry.task, s) for s in fit_data]

        for i in range(len(source)):
            s = source[i]
            for entry in prepare:
                s = call_task(entry.task, s)
            yield s
        return

    # Case B: fit on the full source. Interleave fit() with execute() so each
    # task is applied to every sample exactly once.
    buffered = [source[i] for i in range(len(source))]
    for entry in prepare:
        if isinstance(entry.task, DatasetTask):
            entry.task.fit(buffered)
        buffered = [call_task(entry.task, s) for s in buffered]
    yield from buffered


def collect_task_states(pipeline: "PipelineSpec") -> dict[str, dict[str, Any]]:
    """Return ``{entry.name: task.state_dict()}`` for every :class:`DatasetTask`.

    Meant to be called right after :func:`run` / :func:`cache` so the fitted
    state is captured before the task instance goes out of scope.
    """
    states: dict[str, dict[str, Any]] = {}
    for entry in pipeline.tasks:
        if isinstance(entry.task, DatasetTask):
            states[entry.name] = entry.task.state_dict()
    return states


def load_task_states(
    pipeline: "PipelineSpec",
    states: dict[str, dict[str, Any]],
) -> None:
    """Restore fitted state into each :class:`DatasetTask` by entry name."""
    by_name = {e.name: e.task for e in pipeline.tasks}
    for name, state in states.items():
        task = by_name.get(name)
        if task is None or not isinstance(task, DatasetTask):
            continue
        task.load_state_dict(state)
