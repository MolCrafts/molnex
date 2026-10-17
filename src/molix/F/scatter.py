"""Scatter-add along dim 0.

Generic COO reduction ``out[index[e]] += src[e]``. Same primitive as
``pytorch_scatter.scatter_add`` (rusty1s/pytorch_scatter) on **unsorted**
indices. We do **not** depend on that package:

* it is a third-party CUDA extension that must be rebuilt against this
  interpreter's torch (GH200 aarch64 + ``torch>=2.10`` is a packaging trap);
* inductor does not know its ops, so ``torch.compile`` graph-breaks or
  mis-fuses them;
* molnex only needs this one reduction (not segment CSR, softmax, …).

The native kernel ``torch.ops.molix.scatter_sum`` streams source rows with
atomics; workspace is the output ``(dim_size, *src.shape[1:])`` only — no
``(N, tile)`` scratch. Backward is gather. When the ``.so`` is missing, this
module falls back to ``index_add_`` (same math, inductor needs the
``register_inductor_fallbacks`` path).

``dim_size`` is always an argument, never ``int(index.max())``: the bucket
count is a static shape the caller already holds (``graphs.batch_size[0]``,
the atom count), and reading it off the index is a host sync. Prefer
:func:`scatter_sum_compile_safe` on compile / MACE hot paths.
"""

from __future__ import annotations

import os

import torch
from torch import Tensor

# Read once at import: ``scatter_sum_compile_safe`` runs inside ``torch.compile``
# regions, where an ``os.environ`` lookup is an opaque side effect.
_ONEHOT_FLAG: bool = os.environ.get("MOLNEX_SCATTER_ONEHOT") == "1"


def _scatter_sum_setup(ctx, inputs, output) -> None:  # noqa: ANN001
    _src, index, _dim_size = inputs
    ctx.save_for_backward(index)


def _scatter_sum_backward(ctx, grad: Tensor):  # noqa: ANN001
    (index,) = ctx.saved_tensors
    return grad.index_select(0, index), None, None


def _install_scatter_compile_hooks() -> None:
    """Opaque custom-op hooks: C++ Meta for dynamo, Python autograd for backward.

    The C++ library registers CPU + CUDA + Meta kernels only. AutogradCUDA
    ``Function::apply`` traces into ``data_ptr()`` on FakeTensor; Python
    ``register_autograd`` keeps the CUDA kernel opaque to dynamo.
    """
    if getattr(torch.ops.molix, "scatter_sum", None) is None:
        return
    torch.library.register_autograd(
        "molix::scatter_sum",
        _scatter_sum_backward,
        setup_context=_scatter_sum_setup,
    )


def _native_scatter_sum(src: Tensor, index: Tensor, dim_size: int) -> Tensor | None:
    """Call ``molix::scatter_sum`` when the op library loaded; else ``None``."""
    op = getattr(torch.ops.molix, "scatter_sum", None)
    if op is None:
        return None
    return op(src, index, dim_size)


def scatter_sum(
    src: Tensor,
    index: Tensor,
    dim: int = 0,
    *,
    dim_size: int,
) -> Tensor:
    """Sum ``src`` values grouped by ``index`` along ``dim``.

    Args:
        src: Source tensor of any shape.
        index: 1-D or broadcastable index tensor. Values must be non-negative.
        dim: Reduction dimension.
        dim_size: Output size along ``dim``.

    Returns:
        Tensor with ``src.shape`` except ``shape[dim] == dim_size``.
    """
    if dim < 0:
        dim += src.dim()
    if dim == 0 and index.dim() == 1 and index.shape[0] == src.shape[0]:
        native = _native_scatter_sum(src, index, dim_size)
        if native is not None:
            return native
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

    Native ``molix::scatter_sum`` when loaded; otherwise ``index_add_``.
    The one-hot matmul (bit-exact under ``torch.compile`` without global
    determinism flags) is opt-in via ``MOLNEX_SCATTER_ONEHOT=1``.
    """
    if _ONEHOT_FLAG:
        onehot = (index.view(-1, 1) == torch.arange(dim_size, device=src.device).view(1, -1)).to(
            src.dtype
        )
        return (onehot.t() @ src.reshape(src.shape[0], -1)).reshape(dim_size, *src.shape[1:])
    native = _native_scatter_sum(src, index, dim_size)
    if native is not None:
        return native
    out_shape = (dim_size, *src.shape[1:])
    out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
    return out.index_add_(0, index, src)


def batch_add(src: Tensor, batch: Tensor, dim_size: int) -> Tensor:
    """Sum ``src[i]`` into bucket ``batch[i]`` along dim 0.

    Args:
        src: Per-row values ``(N, ...)``.
        batch: Bucket index per row ``(N,)``.
        dim_size: Number of buckets ``B``.

    Returns:
        Per-bucket sums ``(B, ...)``.
    """
    return scatter_sum(src, batch, dim=0, dim_size=dim_size)


_install_scatter_compile_hooks()
