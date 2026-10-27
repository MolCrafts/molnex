"""Indexed channelwise tensor product for MACE's natural-parity O3 family.

``out[target, l3 m3, c] += C(l1 m1, l2 m2 | l3 m3) · a[source, l1 m1, c] · b[e, l2 m2] ·
w[e, path, c]`` over every path ``(l1, l2, l3)`` of cuEquivariance's ``channelwise_tensor_product``
descriptor (see :mod:`.descriptor` for the family: ``U×0e+U×1o+…+U×L1``,
``0e+1o+…+L2`` and the output filter ``0e+…+L2``; ``U`` in 32 / 64 / 128).
Weights are external and per edge — the one way MACE uses the product — and
the layout is ``ir_mul``. Anything else raises at construction.

On a CUDA tensor the contraction runs ``molnex_equivariance::product`` over a
CSR edge schedule: one CTA per target row, every reduction in a fixed order,
no atomics, bitwise reproducible, with fused adjoints for the first and second
derivative (:mod:`.backend`). On a CPU tensor the same term table is
evaluated by :mod:`.reference`.

Reference:
    Batatia et al. "MACE: Higher Order Equivariant Message Passing Neural
    Networks for Fast and Accurate Force Fields" NeurIPS 2022.
    https://arxiv.org/abs/2206.07697
"""

from __future__ import annotations

import torch
from torch import nn

import molrep.eq as meq

from . import reference
from .backend import EdgeSchedule, NativeBackend
from .descriptor import ChannelwiseDescriptor

__all__ = ["ChannelWiseTensorProduct", "EdgeSchedule"]


class ChannelWiseTensorProduct(nn.Module):
    """The indexed channelwise product with external per-edge weights.

    Args:
        irreps_in1: Node operand irreps, ``U×0e+U×1o+…`` (natural parity).
        irreps_in2: Angular operand irreps, ``0e+1o+…+L2``.
        filter_irreps_out: Output filter, ``0e+…+L2`` (``None`` keeps the
            descriptor's default, which the family check must accept).
        layout: Must be ``meq.ir_mul``.
        dtype: ``float32`` or ``float64``; the dtype of the coefficient buffer.
        device: Device of the term-table buffers.

    Attributes:
        irreps_out: Output irreps, one segment per path.
        weight_numel: Flat per-edge weight width.
    """

    def __init__(
        self,
        irreps_in1: meq.Irreps,
        irreps_in2: meq.Irreps,
        filter_irreps_out=None,
        *,
        layout=meq.ir_mul,
        dtype: torch.dtype,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if layout != meq.ir_mul:
            raise ValueError("native channelwise requires ir_mul layout")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("native channelwise requires float32 or float64")
        self.descriptor = ChannelwiseDescriptor(irreps_in1, irreps_in2, filter_irreps_out)
        self.irreps_in1, self.irreps_in2 = irreps_in1, irreps_in2
        self.irreps_out = self.descriptor.irreps_out
        self.weight_numel = self.descriptor.weight_numel
        self.channels = self.descriptor.channels
        self.output_segments = self.descriptor.output_dim // self.channels
        self.register_buffer(
            "terms",
            torch.tensor(self.descriptor.terms, dtype=torch.int64, device=device),
            persistent=False,
        )
        self.register_buffer(
            "coefficients",
            torch.tensor(self.descriptor.coefficients, dtype=dtype, device=device),
            persistent=False,
        )
        NativeBackend.load()

    def extra_repr(self) -> str:
        return (
            f"{self.irreps_in1} x {self.irreps_in2} -> {self.irreps_out}, "
            f"weight_numel={self.weight_numel}"
        )

    @staticmethod
    def schedule(
        source: torch.Tensor, target: torch.Tensor, rows_source: int, rows_target: int
    ) -> EdgeSchedule:
        """The CSR edge schedule for ``(source, target)`` over the given row counts."""
        if source.is_cuda:
            return EdgeSchedule(
                *torch.ops.molnex_equivariance.schedule(
                    source.contiguous(), target.contiguous(), rows_source, rows_target
                )
            )
        return reference.schedule(source, target, rows_source, rows_target)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        weight: torch.Tensor,
        *,
        indices_1: torch.Tensor,
        indices_out: torch.Tensor,
        size_out: int,
        schedule: EdgeSchedule | None = None,
    ) -> torch.Tensor:
        """Contract and sum into the target rows.

        Args:
            x1: Node operand ``(rows_1, irreps_in1.dim)``, ``ir_mul``.
            x2: Angular operand ``(E, irreps_in2.dim)``.
            weight: Per-edge weights ``(E, weight_numel)``.
            indices_1: ``(E,)`` row of ``x1`` per edge (the source).
            indices_out: ``(E,)`` output row per edge (the target).
            size_out: Output row count.
            schedule: The batch's edge schedule; built here when omitted.

        Returns:
            ``(size_out, irreps_out.dim)``, ``ir_mul``.
        """
        if x1.ndim != 2 or x2.ndim != 2 or weight.ndim != 2:
            raise ValueError("operands must be matrices")
        if (x1.shape[1], x2.shape[1]) != self.descriptor.input_dims:
            raise ValueError("operand widths differ from the descriptor")
        if weight.shape[1] != self.weight_numel:
            raise ValueError("weight width differs from weight_numel")
        source, target = indices_1.contiguous(), indices_out.contiguous()
        d = self.descriptor
        if x1.is_cuda:
            if schedule is None:
                schedule = self.schedule(source, target, x1.shape[0], size_out)
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
        return reference.channelwise(
            x1,
            x2,
            weight,
            source,
            target,
            self.coefficients,
            self.terms,
            size_out,
            self.channels,
            self.output_segments,
        )
