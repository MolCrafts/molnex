"""Step implementations for Molix training.

This module provides the Step protocol and default implementations for
training and evaluation computation.
"""

from __future__ import annotations

from typing import Any

import torch

from molix.core.steps.base import Step
from molix.core.steps.eval import DefaultEvalStep
from molix.core.steps.train import DefaultTrainStep


def batch_to(
    batch: Any,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> Any:
    """Move and/or cast a batch in a single pass.

    Handles TensorDict (via ``.to()``), plain dicts (recursive),
    and bare tensors. Non-tensor leaves are returned unchanged.

    Args:
        batch: TensorDict / nested dict / Tensor / other.
        device: Target device, or ``None`` to leave the device alone.
        dtype: Target floating-point dtype, or ``None`` to leave it alone.

    Returns:
        Batch with the requested transformations applied. Non-tensor
        leaves and tensors of unrelated dtype (when ``dtype`` is set)
        are returned unchanged.
    """
    if device is None and dtype is None:
        return batch

    def _move(t: torch.Tensor) -> torch.Tensor:
        eff_dtype = dtype if (dtype is not None and t.is_floating_point()) else None
        if device is None and eff_dtype is None:
            return t
        return t.to(device=device, dtype=eff_dtype)

    if hasattr(batch, "apply"):
        return batch.apply(_move)
    if isinstance(batch, torch.Tensor):
        return _move(batch)
    if isinstance(batch, dict):
        return {k: batch_to(v, device=device, dtype=dtype) for k, v in batch.items()}
    return batch


__all__ = [
    "Step",
    "DefaultTrainStep",
    "DefaultEvalStep",
    "batch_to",
]
