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


# Read once at import, not per call: ``scatter_sum_compile_safe`` runs inside
# ``torch.compile`` regions on the MACE interaction hot path, where an
# ``os.environ`` lookup is both a per-call dict hit and an opaque side effect
# the compiler must guard against.
_ONEHOT_FLAG: bool = os.environ.get("MOLNEX_SCATTER_ONEHOT") == "1"


def scatter_sum_compile_safe(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    """Sum ``src`` rows into ``dim_size`` buckets given by ``index`` (dim 0).

    Uses ``index_add_`` — the O(E·D) scatter. The alternative one-hot matmul
    (bit-exact under ``torch.compile`` without global determinism flags, since
    ``index_add_``'s scatter order can differ under inductor) is an explicit
    opt-in via env ``MOLNEX_SCATTER_ONEHOT=1``: it turns the reduction into an
    O(E·N·D) GEMM measured 2.4–3.3x slower than ``index_add_`` at every
    profiled molecular-graph shape, so exactness must be a deliberate choice,
    never a size-heuristic default. The flag is read once at import.
    """
    if _ONEHOT_FLAG:
        onehot = (index.view(-1, 1) == torch.arange(dim_size, device=src.device).view(1, -1)).to(
            src.dtype
        )
        return (onehot.t() @ src.reshape(src.shape[0], -1)).reshape(dim_size, *src.shape[1:])

    out_shape = (dim_size, *src.shape[1:])
    out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
    return out.index_add_(0, index, src)


def batch_add(src: Tensor, batch: Tensor, dim_size: int | None = None) -> Tensor:
    """Sum ``src[i]`` into bucket ``batch[i]`` along dim 0."""
    if dim_size is None:
        dim_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    return scatter_sum(src, batch, dim=0, dim_size=dim_size)
