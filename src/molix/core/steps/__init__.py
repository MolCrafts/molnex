"""Step implementations for Molix training.

This module provides the Step protocol and default implementations for
training and evaluation computation.
"""

from __future__ import annotations

from typing import Any

import torch

from molix.core.steps.base import Step
from molix.core.steps.eval import DefaultEvalStep
from molix.core.steps.multitask import (
    MultiTaskEvalStep,
    MultiTaskTrainStep,
    TaskRouter,
    TaskSpec,
)
from molix.core.steps.train import DefaultTrainStep
from molix.schema import FLOAT64_LABEL_KEYS


def cast_compute_floats(batch: Any, dtype: torch.dtype) -> Any:
    """Cast a batch's compute float leaves to *dtype*, keeping graph totals.

    The one thing about moving a batch that the library cannot do: every float
    leaf takes *dtype* **except** the graph totals at
    :data:`~molix.schema.FLOAT64_LABEL_KEYS`, which keep their incoming dtype. The
    skip matches the full NestedKey, never the leaf name ``"energy"`` — that name
    is also :data:`~molix.schema.ATOMIC_ENERGY_KEY`, a per-atom quantity that
    *should* follow the compute dtype. Integer and bool leaves are untouched.

    Device moves are **not** here: ``batch.to(device)`` is tensordict's own and
    needs no wrapper. This used to be ``batch_to(batch, device=, dtype=)``, which
    reimplemented the move as a per-leaf walk, carried a same-device fast path
    that duplicated the caller's own :meth:`Trainer._needs_device_move` guard, and
    grew dict / bare-tensor branches no caller in the tree used.

    Args:
        batch: Post-collate batch (a nested ``TensorDict``, which is the batch
            schema).
        dtype: Floating dtype for compute leaves.

    Returns:
        A batch with the compute floats cast. A late ``.double()`` is not applied:
        a float32 stored total stays float32.

    Raises:
        TypeError: *batch* is not a TensorDict — the NestedKey skip cannot be
            expressed on anything that has no nested keys, and silently casting a
            protected total is the failure this function exists to prevent.
    """
    if not callable(getattr(batch, "named_apply", None)):
        raise TypeError(
            f"cast_compute_floats needs a TensorDict batch, got {type(batch).__name__}: "
            "the graph totals it protects are addressed by NestedKey, which a plain "
            "dict or tensor has none of"
        )

    def _cast(key: object, tensor: torch.Tensor) -> torch.Tensor:
        if key in FLOAT64_LABEL_KEYS or not tensor.is_floating_point():
            return tensor
        return tensor.to(dtype)

    return batch.named_apply(_cast, nested_keys=True)


__all__ = [
    "DefaultEvalStep",
    "DefaultTrainStep",
    "MultiTaskEvalStep",
    "MultiTaskTrainStep",
    "Step",
    "TaskRouter",
    "TaskSpec",
    "cast_compute_floats",
]
