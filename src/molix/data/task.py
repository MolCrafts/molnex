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

    def execute(self, data: Any) -> Any:
        """Transform *data* and return the result (structural ``execute`` contract)."""
        ...


class Task:
    """Abstract base for all data pipeline tasks."""

    @property
    def task_id(self) -> str:
        """Deterministic identifier used in cache key computation."""
        return type(self).__name__

    def execute(self, data: dict) -> dict:
        """Transform a sample dict — abstract; subclasses must override.

        Raises:
            NotImplementedError: Always, on the base class.
        """
        raise NotImplementedError

    def __call__(self, data: dict) -> dict:
        """Call the task, delegating to :meth:`execute`."""
        return self.execute(data)


class SampleTask(Task):
    """Per-sample processing.  Each sample is independent.

    Best suited for parallelisation and caching.

    Examples: neighbor list, graph construction, augmentation.
    """

    def execute(self, data: dict) -> dict:
        """Transform one sample dict — abstract; subclasses must override.

        Raises:
            NotImplementedError: Always, on this base class.
        """
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

    def execute(self, data: dict) -> dict:
        """Apply fitted parameters to a single sample."""
        raise NotImplementedError

    def state_dict(self) -> dict[str, Any]:
        """Serialise fitted state for caching."""
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore fitted state from cache."""


class BatchTask(Task):
    """Post-collate batch processing. **Extension point — no built-in subclasses.**

    Executed after :func:`~molix.data.collate.collate_molecules` inside the
    DataLoader's ``collate_fn``. Input and output are both
    ``TensorDict`` (nested), not raw
    sample dicts. Runs on the hot path of every training step — keep it
    fast.

    molix ships no built-in :class:`BatchTask` subclass. The base class
    exists so custom post-collate transforms can be plugged into
    :meth:`Pipeline.add` and routed through
    :attr:`~molix.data.pipeline.PipelineSpec.batch_nodes` to the
    :class:`~molix.data.datamodule._CollateFn`.

    Use this only when the transform **must** operate on the collated
    batch (e.g. batch-level augmentation that mixes atoms across graphs,
    dense-padding for ``torch.compile``). Transforms expressible per
    sample should subclass :class:`SampleTask` instead — they benefit
    from per-sample caching, which :class:`BatchTask` cannot (the input
    shape is not known until collate time).
    """

    def execute(self, data: dict) -> dict:
        """Transform a collated batch — abstract; custom subclasses override.

        Input and output are both nested ``TensorDict`` batches, not raw
        sample dicts.

        Raises:
            NotImplementedError: Always, on this base class.
        """
        raise NotImplementedError
