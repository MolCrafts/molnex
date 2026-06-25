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


def scatter_sum_compile_safe(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    """Sum ``src`` rows into ``dim_size`` buckets given by ``index`` (dim 0).

    Implemented as a one-hot matmul instead of ``index_add_``. The natural
    ``out.index_add_(0, index, src)`` is correct in eager, but under
    ``torch.compile`` its scatter reduction runs in a *different order* than
    eager (CUDA: non-deterministic ``atomicAdd``; CPU: it additionally trips
    the torch-2.12 ``index.is_vec`` SIMD-codegen crash). When the scattered
    energy is a small difference of large terms (catastrophic cancellation, as
    in MACE ``inter_e``), that float64 ordering difference is amplified to ~4%
    between compiled and eager. This is NOT a wrong-codegen miscompile: forcing
    ``torch.use_deterministic_algorithms(True)`` (CUDA) or
    ``torch._inductor.config.cpp.simdlen = 0`` (CPU) makes ``index_add_``
    bit-exact again. The one-hot matmul is deterministic and has no scatter
    node, so it is bit-exact under compile on both backends with NO global
    flags, and stays functorch-traceable. Cost: a dense ``(E, dim_size)``
    one-hot = ``O(E * dim_size)`` — negligible for molecular graphs, but heavy
    for very large periodic systems.

    TODO: switch back to the cheaper ``index_add_`` once we either (a) move to
    a torch where the compiled scatter is deterministic by default, or (b) set
    the determinism / ``cpp.simdlen`` knobs on the compile path — needed to
    drop the O(E*dim_size) memory for large periodic systems.
    """
    onehot = (index.view(-1, 1) == torch.arange(dim_size, device=src.device).view(1, -1)).to(
        src.dtype
    )
    return (onehot.t() @ src.reshape(src.shape[0], -1)).reshape(dim_size, *src.shape[1:])


def batch_add(src: Tensor, batch: Tensor, dim_size: int | None = None) -> Tensor:
    """Sum ``src[i]`` into bucket ``batch[i]`` along dim 0."""
    if dim_size is None:
        dim_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    return scatter_sum(src, batch, dim=0, dim_size=dim_size)
