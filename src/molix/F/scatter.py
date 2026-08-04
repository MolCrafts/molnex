"""Scatter operations using PyTorch primitives.

These are thin wrappers around ``torch.Tensor.scatter_add_`` kept for API
stability; prefer calling the torch primitives directly in new code.
"""

from __future__ import annotations

import os

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


# One-hot matmul allocates ``(E, dim_size)``. Above this element budget use
# ``index_add_`` to avoid OOM on large periodic systems (review: O(E·N) risk).
_ONEHOT_ELEMENT_BUDGET = 2_000_000

# Read once at import, not per call: ``scatter_sum_compile_safe`` runs inside
# ``torch.compile`` regions on the MACE interaction hot path, where an
# ``os.environ`` lookup is both a per-call dict hit and an opaque side effect
# the compiler must guard against. ``None`` (unset) = decide by element budget.
_ONEHOT_FLAG: str | None = os.environ.get("MOLNEX_SCATTER_ONEHOT")


def scatter_sum_compile_safe(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    """Sum ``src`` rows into ``dim_size`` buckets given by ``index`` (dim 0).

    For small ``E * dim_size`` (typical molecular graphs) uses a one-hot matmul
    that is bit-exact under ``torch.compile`` without global determinism flags
    (see historical note: ``index_add_`` scatter order can differ under
    inductor). For large systems (``E * dim_size > 2e6``) falls back to
    ``index_add_`` to avoid O(E·N) memory blow-ups.

    Force exact one-hot always with env ``MOLNEX_SCATTER_ONEHOT=1``.
    Force ``index_add_`` always with ``MOLNEX_SCATTER_ONEHOT=0``.
    Both are read once at import (see :data:`_ONEHOT_FLAG`).
    """
    use_onehot = _ONEHOT_FLAG == "1" or (
        _ONEHOT_FLAG is None and src.shape[0] * int(dim_size) <= _ONEHOT_ELEMENT_BUDGET
    )

    if not use_onehot:
        out_shape = (dim_size, *src.shape[1:])
        out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
        return out.index_add_(0, index, src)

    onehot = (index.view(-1, 1) == torch.arange(dim_size, device=src.device).view(1, -1)).to(
        src.dtype
    )
    return (onehot.t() @ src.reshape(src.shape[0], -1)).reshape(dim_size, *src.shape[1:])


def batch_add(src: Tensor, batch: Tensor, dim_size: int | None = None) -> Tensor:
    """Sum ``src[i]`` into bucket ``batch[i]`` along dim 0."""
    if dim_size is None:
        dim_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    return scatter_sum(src, batch, dim=0, dim_size=dim_size)
