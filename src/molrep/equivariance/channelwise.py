"""MACE-compatible channelwise front end for the native contraction.

Only external per-edge weights, O3 ``ir_mul`` and MACE's natural-parity irreps
family are supported (see :mod:`.descriptor`). Unsupported options fail
explicitly; no cuEquivariance computation is called as a fallback. The
constructor uses cuEquivariance only to build its descriptor.

Build with ``-DMOLNEX_EQUIVARIANCE_ENABLE_CUDA=ON`` and point
``MOLNEX_EQUIVARIANCE_LIBRARY`` to the resulting shared library, or install it
into ``molrep/lib``. The kernels accept float32/float64 and uniform 32/64/128
feature channels. Every reduction runs in a fixed order over the edge schedule
— no atomics — so results are bitwise reproducible and PyTorch's deterministic
mode is honoured.
"""

from __future__ import annotations

from typing import NamedTuple

import cuequivariance as cue
import torch
from torch import nn

from .backend import NativeBackend
from .descriptor import ChannelwiseDescriptor


class EdgeSchedule(NamedTuple):
    """CSR views of one edge list, by target and by source (``int32``).

    Built once per batch (``ChannelWiseTensorProduct.schedule``) and shared by
    every contraction of every layer — the forward sums over a node's incoming
    edges, the node-operand adjoint over its outgoing ones.
    """

    order_t: torch.Tensor  #: edges sorted (stably) by target, ``(E,)``
    rowptr_t: torch.Tensor  #: ``(rows_d + 1,)``
    order_s: torch.Tensor  #: edges sorted (stably) by source, ``(E,)``
    rowptr_s: torch.Tensor  #: ``(rows_a + 1,)``


class ChannelWiseTensorProduct(nn.Module):
    """The cuEq 0.12 channelwise signature, restricted to MACE's indexed subset."""

    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        filter_irreps_out=None,
        *,
        layout=None,
        layout_in1=None,
        layout_in2=None,
        layout_out=None,
        shared_weights=True,
        internal_weights=None,
        device=None,
        dtype=None,
        math_dtype=None,
        use_fallback=None,
        method=None,
    ):
        super().__init__()
        if shared_weights or internal_weights not in (None, False):
            raise ValueError("native channelwise requires external, unshared weights")
        if any((value or layout) != cue.ir_mul for value in (layout_in1, layout_in2, layout_out)):
            raise ValueError("native channelwise requires ir_mul layout")
        if method not in (None, "uniform_1d") or use_fallback not in (None, False):
            raise ValueError("native channelwise has no cuEq execution fallback")
        dtype = torch.get_default_dtype() if dtype is None else dtype
        if dtype not in (torch.float32, torch.float64) or math_dtype not in (None, dtype):
            raise ValueError("native channelwise requires matching float32 or float64 math dtype")
        self.descriptor = ChannelwiseDescriptor(irreps_in1, irreps_in2, filter_irreps_out)
        self.irreps_in1, self.irreps_in2 = irreps_in1, irreps_in2
        self.irreps_out = self.descriptor.irreps_out
        self.weight_numel = self.descriptor.weight_numel
        self.shared_weights = False
        self.internal_weights = False
        self.register_parameter("weight", None)
        NativeBackend.load()

    @staticmethod
    def schedule(
        source: torch.Tensor, target: torch.Tensor, rows_source: int, rows_target: int
    ) -> EdgeSchedule:
        """The CSR edge schedule for ``(source, target)`` over the given row counts."""
        return EdgeSchedule(
            *torch.ops.molnex_equivariance.schedule(
                source.contiguous(), target.contiguous(), rows_source, rows_target
            )
        )

    def forward(
        self,
        x1,
        x2,
        weight=None,
        indices_1=None,
        indices_2=None,
        indices_out=None,
        size_out=None,
        schedule: EdgeSchedule | None = None,
    ):
        if weight is None or indices_1 is None or indices_out is None or size_out is None:
            raise ValueError("explicit weights, source/target indices and size_out are required")
        if indices_2 is not None:
            raise ValueError("indices_2 is not implemented")
        if x1.ndim != 2 or x2.ndim != 2 or weight.ndim != 2:
            raise ValueError("operands must be matrices")
        if (x1.shape[1], x2.shape[1]) != self.descriptor.input_dims:
            raise ValueError("operand widths differ from the descriptor")
        if weight.shape[1] != self.weight_numel:
            raise ValueError("weight width differs from weight_numel")
        source, target = indices_1.contiguous(), indices_out.contiguous()
        if schedule is None:
            schedule = self.schedule(source, target, x1.shape[0], size_out)
        d = self.descriptor
        return torch.ops.molnex_equivariance.product(
            x1.contiguous(),
            x2.contiguous(),
            weight.contiguous(),
            source,
            target,
            *schedule,
            d.coefficients,
            size_out,
            d.l1,
            d.l2,
            d.channels,
        )
