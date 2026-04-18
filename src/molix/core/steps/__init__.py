"""Step implementations for Molix training.

This module provides the Step protocol and default implementations for
training and evaluation computation.
"""

from __future__ import annotations

from typing import Any

import torch

from molix.core.steps.base import Step
from molix.core.steps.eval_step import DefaultEvalStep
from molix.core.steps.train_step import DefaultTrainStep

_NON_MODEL_KEYS = frozenset({"targets", "extras"})


def extract_model_inputs(batch: dict[str, Any]) -> dict[str, Any]:
    """Extract model-relevant inputs from a batch dict.

    Strips ``targets`` and ``extras`` so they are never forwarded to
    the model as keyword arguments.

    Args:
        batch: Full batch dictionary from the data pipeline.

    Returns:
        New dict containing only model input fields.
    """
    return {k: v for k, v in batch.items() if k not in _NON_MODEL_KEYS}


def batch_to_device(
    batch: Any,
    device: torch.device | str,
) -> Any:
    """Move a batch to the target device.

    Handles TensorDict/GraphBatch (via ``.to()``), plain dicts (recursive),
    and bare tensors. Non-tensor leaves are returned unchanged.

    Args:
        batch: Batch data — TensorDict, dict, Tensor, or other.
        device: Target device.

    Returns:
        Batch with all tensors on ``device``.
    """
    if hasattr(batch, "to"):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: batch_to_device(v, device) for k, v in batch.items()}
    return batch


__all__ = [
    "Step",
    "DefaultTrainStep",
    "DefaultEvalStep",
    "extract_model_inputs",
    "batch_to_device",
]
