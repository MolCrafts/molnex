"""Post-collate transform base.

A :class:`BatchTask` runs inside the DataLoader's ``collate_fn``, after
:func:`~molix.batch.collate.collate_molecules` has built the nested batch —
it is not a pipeline transform and deliberately shares no base class with the
per-sample ETL task hierarchy.
"""

from __future__ import annotations


class BatchTask:
    """Post-collate batch processing. **Extension point — one built-in subclass.**

    Executed after :func:`~molix.batch.collate.collate_molecules` inside the
    DataLoader's ``collate_fn``. Input and output are both ``TensorDict``
    (nested), not raw sample dicts. Runs on the hot path of every training
    step — keep it fast.

    Instances are passed straight to
    :class:`~molix.batch.datamodule.DataModule` as ``batch_tasks=[...]``, in
    execution order; molix ships :class:`~molix.batch.pad.PadMolecularBatch`
    as the one built-in subclass.

    Use this only when the transform **must** operate on the collated batch
    (e.g. batch-level augmentation that mixes atoms across graphs,
    dense-padding for ``torch.compile``). Transforms expressible per sample
    belong in the per-sample pipeline instead — they benefit from per-sample
    caching, which a :class:`BatchTask` cannot (the input shape is not known
    until collate time).
    """

    @property
    def task_id(self) -> str:
        """Deterministic identifier — the concrete class name by default."""
        return type(self).__name__

    def execute(self, data: dict) -> dict:
        """Transform a collated batch — abstract; subclasses must override.

        Input and output are both nested ``TensorDict`` batches, not raw
        sample dicts.

        Raises:
            NotImplementedError: Always, on this base class.
        """
        raise NotImplementedError

    def __call__(self, data: dict) -> dict:
        """Call the task, delegating to :meth:`execute`."""
        return self.execute(data)
