"""Task hierarchy for data pipeline processing.

Three task types by scope, dispatched via ``isinstance``:

- :class:`SampleTask` — per-sample, stateless, parallelisable.
- :class:`DatasetTask` — fit on full dataset, then apply per-sample.
- :class:`BatchTask` — post-collate batch processing.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Runnable(Protocol):
    """Sync task protocol, structurally aligned with molexp.Runnable."""

    def execute(self, data: Any) -> Any: ...


class Task:
    """Abstract base for all data pipeline tasks."""

    @property
    def task_id(self) -> str:
        """Deterministic identifier used in cache key computation."""
        return type(self).__name__

    def execute(self, data: Any) -> Any:
        raise NotImplementedError

    def __call__(self, data: Any) -> Any:
        return self.execute(data)


class SampleTask(Task):
    """Per-sample processing.  Each sample is independent.

    Best suited for parallelisation and caching.

    Examples: neighbor list, graph construction, augmentation.
    """

    def execute(self, sample: dict) -> dict:
        raise NotImplementedError


class DatasetTask(Task):
    """Two-phase task: fit on the full dataset, then apply per-sample.

    The pipeline runner calls :meth:`fit` once with all training
    samples, then calls :meth:`execute` on every sample.
    :meth:`state_dict` / :meth:`load_state_dict` are used for
    caching the fitted state.

    Examples: atomic dress, scaler, PCA, vocabulary.
    """

    def fit(self, samples: list[dict]) -> None:
        """Fit global parameters from the training set."""
        raise NotImplementedError

    def execute(self, sample: dict) -> dict:
        """Apply fitted parameters to a single sample."""
        raise NotImplementedError

    def state_dict(self) -> dict[str, Any]:
        """Serialise fitted state for caching."""
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore fitted state from cache."""


class BatchTask(Task):
    """Post-collate batch processing.

    Executed after ``collate_molecules`` inside the DataLoader's
    ``collate_fn``.  Runs on the hot path — keep it fast.

    Examples: force padding, negative sampling, batch augmentation.
    """

    def execute(self, batch: dict) -> dict:
        raise NotImplementedError
