"""Match two same-shaped nested columns of a ``TensorDict`` forward.

The prediction and the target are both ``(namespace, column)`` keys. The
forward that produced the prediction returned that ``TensorDict``; a teacher
stamp writes the target column on the student batch. A dense projection
would mix irreps; an equivariant aligner is a separate module that writes a
same-shaped column before this loss runs.

Reference:
    Kelvinius et al. "Accelerating Molecular Graph Neural Networks via
    Knowledge Distillation." NeurIPS 2023.
    https://arxiv.org/abs/2306.14818
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDictBase

from molix.core.losses.molecular import _reduce

NestedKey = tuple[str, str]


def _nested(key: object, *, what: str) -> NestedKey:
    """A ``(namespace, column)`` pair, or a :class:`TypeError` naming *what*."""
    if isinstance(key, tuple) and len(key) == 2:
        namespace, column = key
        if isinstance(namespace, str) and isinstance(column, str):
            return (namespace, column)
    raise TypeError(f"{what} must be a (namespace, column) pair of strings, got {key!r}")


def _row_weight(mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """*mask* as trailing singletons so a mean divides by rows, not by a broadcast.

    :func:`~molix.core.losses.molecular._reduce` treats ``weight.sum()`` as the
    row-weight sum and multiplies by the trailing element count. A mask
    expanded to the full column would be counted twice.
    """
    weight = mask.to(dtype=like.dtype)
    if weight.ndim > like.ndim or like.shape[: weight.ndim] != weight.shape:
        raise ValueError(
            f"mask shape {tuple(weight.shape)} is not a prefix of column shape {tuple(like.shape)}"
        )
    for _ in range(like.ndim - weight.ndim):
        weight = weight.unsqueeze(-1)
    return weight


def column_match(
    pred_key: NestedKey,
    target_key: NestedKey,
    *,
    elementwise: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    mask_key: NestedKey | None = None,
    reduction: str = "mean",
) -> Callable[[Any, Any], torch.Tensor]:
    """Elementwise match of one nested column against another.

    The prediction column is read from the ``TensorDict`` the forward
    returned. The target column is read from the batch the loss was given.
    When the forward returns the same batch it was given, both lookups hit
    that object under different keys.

    *mask_key*, when given and present, is multiplied into the residual before
    the reduction. An absent mask column is unweighted, the same rule as
    :func:`~molix.core.losses.force_loss`. The reduction never gathers rows.

    Args:
        pred_key: ``(namespace, column)`` on the prediction ``TensorDict``.
        target_key: ``(namespace, column)`` on the batch. The teacher stamp
            writes here.
        elementwise: Per-element residual of ``(pred, target)`` with no
            reduction. ``None`` is ``MSELoss(reduction="none")``.
        mask_key: Optional ``(namespace, column)`` row weight. Its shape must
            be a prefix of the column shape.
        reduction: ``'mean'``, ``'sum'``, or ``'none'``. ``'mean'`` divides by
            the weighted element count.

    Returns:
        ``loss_fn(preds, batch) -> Tensor``.

    Raises:
        ValueError: *reduction* is unknown, the two columns differ in shape,
            or the mask is not a prefix of that shape.
        TypeError: A key is not a nested pair, or *preds* is not a ``TensorDict``.
    """
    pred_key = _nested(pred_key, what="column_match pred_key")
    target_key = _nested(target_key, what="column_match target_key")
    if mask_key is not None:
        mask_key = _nested(mask_key, what="column_match mask_key")
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError(f"reduction must be 'mean', 'sum' or 'none', got {reduction!r}")
    measure = nn.MSELoss(reduction="none") if elementwise is None else elementwise

    def _fn(preds: Any, batch: Any) -> torch.Tensor:
        if not isinstance(preds, TensorDictBase):
            raise TypeError(
                f"column_match reads a TensorDict prediction, got {type(preds).__name__}"
            )
        pred = preds[pred_key]
        target = batch[target_key]
        if pred.shape != target.shape:
            raise ValueError(
                f"column_match {pred_key!r} shape {tuple(pred.shape)} != "
                f"{target_key!r} shape {tuple(target.shape)}"
            )
        weight = None
        if mask_key is not None:
            namespace, leaf = mask_key
            block = batch[namespace]
            if leaf in block.keys():
                weight = _row_weight(block[leaf], pred)
        return _reduce(measure(pred, target), weight, reduction)

    return _fn
