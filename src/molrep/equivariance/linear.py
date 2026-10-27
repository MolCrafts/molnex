"""Equivariant linear map: one GEMM per irrep block over cuEquivariance's descriptor.

cuEquivariance's ``descriptors.linear`` (subscripts ``uv,iu,iv``) evaluated
with ``torch.mm`` in ``ir_mul`` layout, with cue's flat ``(1, weight_numel)``
``weight`` so a checkpoint written for ``cuet.Linear`` loads unchanged.
Measured 2026-10-20 on a GH200 (MACE-MatPES, official weights, 193 / 1 544 /
4 425 atoms, fp32 and fp64) cue's ``fused_tp`` kernel ran the same linear maps
4–5× slower than plain GEMMs, and cue's ``naive`` method leaves
``opt_einsum_fx`` scalar constants on the CPU, which
``torch.compile(fullgraph=True)`` refuses as a device mismatch (operator ruling
2026-10-20: GPU only, the fastest path, full-graph compilation). The precision
is fixed at construction: build at the wanted ``dtype``.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import torch
import torch.nn as nn

import molrep.eq as meq

__all__ = ["Linear"]


class _LinearPath(NamedTuple):
    """One irrep block of the linear map: ``x[:, x_start:+d·u] (d, u) @ W (u, v)``."""

    x_start: int
    d: int
    u: int
    w_start: int
    v: int
    out_index: int
    coefficient: float


class Linear(nn.Module):
    """Equivariant linear map in ``meq.ir_mul`` layout, one GEMM per irrep block.

    ``meq.descriptors.linear(irreps_in, irreps_out)`` evaluated with
    ``torch.mm``: for every path — an irrep present in both ``irreps_in`` and
    ``irreps_out`` — the input block ``(…, d, u)`` is multiplied by the weight
    block ``(u, v)`` scaled by the descriptor's path coefficient (``1/√u``),
    and output blocks with no path are zero. The flat ``weight`` ``(1, Σ u·v)``
    keeps cuEquivariance's operand layout, so a checkpoint written for
    ``cuet.Linear`` loads unchanged; the per-path ``f.m.graphs.*.c*`` constants
    of cue's naive method are not parameters here (the family tables skip them).

    Args:
        irreps_in: Input irreps.
        irreps_out: Output irreps.
        layout: Must be ``meq.ir_mul`` — the layout every molnex block uses and
            the one in which an irrep block is ``(d, u)`` row-major, so no
            transpose is needed on either side.
        dtype: Weight dtype.

    Raises:
        ValueError: If ``layout`` is not ``meq.ir_mul``, or a path of the
            descriptor is not a scalar-scaled per-irrep linear block.

    Reference:
        cuEquivariance ``descriptors.linear`` (subscripts ``uv,iu,iv``), the
        operand the official MACE ``Linear`` wrapper evaluates with cue's
        ``naive`` method.
    """

    def __init__(
        self,
        irreps_in: meq.Irreps,
        irreps_out: meq.Irreps,
        *,
        layout: Any = meq.ir_mul,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if layout != meq.ir_mul:
            raise ValueError(f"Linear is laid out ir_mul; got layout {layout}")
        polynomial = meq.descriptors.linear(irreps_in, irreps_out).polynomial
        ((_, product),) = polynomial.operations
        weight_operand, x_operand = polynomial.inputs
        (out_operand,) = polynomial.outputs
        weight_starts = np.cumsum([0] + [int(np.prod(s)) for s in weight_operand.segments])
        x_starts = np.cumsum([0] + [int(np.prod(s)) for s in x_operand.segments])
        paths = []
        for path in product.paths:
            weight_index, x_index, out_index = path.indices
            u, v = weight_operand.segments[weight_index]
            d, u_x = x_operand.segments[x_index]
            coefficients = np.asarray(path.coefficients).reshape(-1)
            if u != u_x or coefficients.size != 1:
                raise ValueError("a path of the linear descriptor is not a per-irrep linear block")
            paths.append(
                _LinearPath(
                    x_start=int(x_starts[x_index]),
                    d=int(d),
                    u=int(u),
                    w_start=int(weight_starts[weight_index]),
                    v=int(v),
                    out_index=int(out_index),
                    coefficient=float(coefficients[0]),
                )
            )
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        #: Length of the flat weight, cuEquivariance's ``weight_numel``.
        self.weight_numel = int(weight_operand.size)
        self._paths: tuple[_LinearPath, ...] = tuple(paths)
        self._out_sizes = tuple(int(np.prod(s)) for s in out_operand.segments)
        self.weight = nn.Parameter(torch.randn(1, self.weight_numel, dtype=dtype))

    def extra_repr(self) -> str:
        return f"{self.irreps_in} -> {self.irreps_out}, weight_numel={self.weight_numel}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the map.

        Args:
            x: Features ``(…, irreps_in.dim)``, ``ir_mul``.

        Returns:
            ``(…, irreps_out.dim)``, ``ir_mul``.
        """
        lead = x.shape[:-1]
        flat = self.weight[0]
        outputs: dict[int, torch.Tensor] = {}
        for p in self._paths:
            block = flat[p.w_start : p.w_start + p.u * p.v].view(p.u, p.v) * p.coefficient
            a = x[..., p.x_start : p.x_start + p.d * p.u].reshape(-1, p.u)
            y = (a @ block).reshape(*lead, p.d * p.v)
            outputs[p.out_index] = y if p.out_index not in outputs else outputs[p.out_index] + y
        parts = [
            outputs[k] if k in outputs else x.new_zeros(*lead, size)
            for k, size in enumerate(self._out_sizes)
        ]
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
