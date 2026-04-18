"""Scatter operations using PyTorch primitives.

These are thin wrappers around ``torch.Tensor.scatter_add_`` kept for API
stability; prefer calling the torch primitives directly in new code.
"""

from __future__ import annotations

import torch
from torch import Tensor


def scatter_sum(
    src: Tensor,
    index: Tensor,
    dim: int = 0,
    dim_size: int | None = None,
) -> Tensor:
    """Sum ``src`` values grouped by ``index`` along ``dim``.

    Args:
        src: Source tensor of any shape.
        index: 1-D or broadcastable index tensor. Values must be non-negative.
        dim: Reduction dimension.
        dim_size: Output size along ``dim``. Defaults to ``int(index.max()) + 1``.

    Returns:
        Tensor with ``src.shape`` except ``shape[dim] == dim_size``.
    """
    if dim < 0:
        dim += src.dim()
    if dim_size is None:
        dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0

    # Broadcast index to match src along non-reduction dims
    if index.dim() != src.dim():
        shape = [1] * src.dim()
        shape[dim] = index.shape[0]
        index = index.view(shape).expand_as(src)

    out_shape = list(src.shape)
    out_shape[dim] = dim_size
    out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
    return out.scatter_add_(dim, index, src)


def batch_add(src: Tensor, batch: Tensor, dim_size: int | None = None) -> Tensor:
    """Sum ``src[i]`` into bucket ``batch[i]`` along dim 0."""
    if dim_size is None:
        dim_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    return scatter_sum(src, batch, dim=0, dim_size=dim_size)
