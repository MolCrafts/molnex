"""Sparse uvu tensor product in native ``cue.ir_mul`` (Polar field path).

Only two connection types, matching ACEsuit ``SparseUvuTensorProduct``:

* ``ℓ × ℓ → 0`` — invariant contraction
* ``ℓ × 0 → ℓ`` — scalar modulation

Each path is divided by ``√(2ℓ+1)`` and then scaled by the e3nn
``path_normalization='element'`` coefficient so a Polar-1-M overlay
matches ``o3.TensorProduct``. No e3nn, no runtime ``mul_ir`` transpose.

Reference:
    Batatia et al. "MACE: Higher Order Equivariant Message Passing Neural
    Networks for Fast and Accurate Force Fields" NeurIPS 2022.
    https://arxiv.org/abs/2206.07697
"""

from __future__ import annotations

import math

import cuequivariance as cue
import torch
import torch.nn as nn
import torch.nn.functional as F

from molix import config


def _slices(irreps: cue.Irreps) -> list[tuple[int, int, int, int, int, int]]:
    """``(start, stop, mul, l, p, dim)`` per irrep in ``ir_mul`` storage order."""
    out: list[tuple[int, int, int, int, int, int]] = []
    start = 0
    for mi in irreps:
        dim = mi.ir.dim
        stop = start + mi.mul * dim
        out.append((start, stop, mi.mul, mi.ir.l, int(mi.ir.p), dim))
        start = stop
    return out


class SparseUvuTensorProduct(nn.Module):
    """Shared-weight uvu product on flattened ``(N, irreps.dim)`` features.

    Args:
        irreps_in1: First operand, e.g. ``"8x0e+8x1o"``.
        irreps_in2: Second operand.
        irreps_out: Output irreps. For Polar dots this is ``n_0e x 0e``;
            for ``tp_out`` it matches ``irreps_in1``.
    """

    def __init__(self, irreps_in1: str, irreps_in2: str, irreps_out: str) -> None:
        super().__init__()
        in1 = cue.Irreps("O3", irreps_in1)
        in2 = cue.Irreps("O3", irreps_in2)
        out = cue.Irreps("O3", irreps_out)
        self.irreps_in1 = in1
        self.irreps_in2 = in2
        self.irreps_out = out
        slices1 = _slices(in1)
        slices2 = _slices(in2)
        slices_out = _slices(out)
        if not slices_out:
            raise ValueError("irreps_out is empty")

        pending: list[tuple[int, int, int, int, int, int, int, int, int, int, int, int, int]] = []
        x_out: dict[tuple[int, int], float] = {}
        w_off = 0
        for s1, e1, mul1, l1, p1, d1 in slices1:
            for s2, e2, mul2, l2, p2, d2 in slices2:
                for so, eo, mulo, lo, po, do in slices_out:
                    if mulo != mul1:
                        continue
                    if do == 1 and d1 == d2 and l1 == l2 and p1 * p2 == po:
                        mode = 0
                    elif d2 == 1 and do == d1 and lo == l1 and p1 * p2 == po:
                        mode = 1
                    else:
                        continue
                    w_size = mul1 * mul2
                    pending.append(
                        (s1, e1, s2, e2, so, eo, w_off, w_off + w_size, mode, mul1, mul2, d1, do)
                    )
                    x_out[(so, eo)] = x_out.get((so, eo), 0.0) + float(mul2)
                    w_off += w_size
        if not pending:
            raise NotImplementedError(
                "SparseUvuTensorProduct only supports (l×l→0) and (l×0→l) uvu paths "
                f"for {irreps_in1} ⊗ {irreps_in2} → {irreps_out}"
            )
        # e3nn element path_normalization for uvu: x = Σ mul_in2 over paths
        # that share this output irrep; weight = √(dim_out / x).
        self._paths = [
            (
                s1,
                e1,
                s2,
                e2,
                so,
                eo,
                ws,
                we,
                mode,
                mul1,
                mul2,
                d1,
                math.sqrt(float(do) / x_out[(so, eo)]),
            )
            for s1, e1, s2, e2, so, eo, ws, we, mode, mul1, mul2, d1, do in pending
        ]
        self.weight = nn.Parameter(torch.randn(w_off, dtype=config.ftype))
        self.register_buffer("output_mask", torch.ones(out.dim, dtype=config.ftype))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Contract two flattened irreps tensors.

        Args:
            x1: ``(N, irreps_in1.dim)``.
            x2: ``(N, irreps_in2.dim)``.

        Returns:
            ``(N, irreps_out.dim)``.
        """
        if x1.ndim != 2 or x2.ndim != 2:
            raise ValueError(
                "SparseUvuTensorProduct expects 2-D tensors (N, irreps.dim), "
                f"got {tuple(x1.shape)} and {tuple(x2.shape)}"
            )
        batch = x1.shape[0]
        out = x1.new_zeros((batch, self.irreps_out.dim))
        weight = self.weight
        for s1, e1, s2, e2, so, eo, ws, we, mode, mul1, mul2, d1, path_weight in self._paths:
            scale = path_weight / math.sqrt(float(d1))
            w = weight[ws:we].reshape(mul1, mul2)
            if mode == 0:
                a = x1[:, s1:e1].reshape(batch, d1, mul1).transpose(1, 2)
                b = x2[:, s2:e2].reshape(batch, d1, mul2).transpose(1, 2)
                # Reassociate the same uvu contraction before materialising
                # it: sum_v b[b,v,d] * w[u,v] first, then dot with a over d.
                # The literal pair tensor is (N, mul1, mul2); Polar-1-M would
                # allocate (N, 512, 512), or 34.57 GiB at N=35400 in fp32.
                # F.linear uses the shared (u,v) weights as one GEMM and keeps
                # the transient at (N, d, mul1), where d <= 2*l_max+1.
                projected = F.linear(b.transpose(1, 2), w).transpose(1, 2)
                mixed = (a * projected).sum(dim=-1) * scale
                out[:, so:eo] = out[:, so:eo] + mixed
            else:
                a = x1[:, s1:e1].reshape(batch, d1, mul1).transpose(1, 2)
                scalars = x2[:, s2:e2].reshape(batch, mul2)
                mixed = torch.einsum("bv,uv->bu", scalars, w)
                contrib = (a * mixed.unsqueeze(-1) * scale).transpose(1, 2).reshape(batch, -1)
                out[:, so:eo] = out[:, so:eo] + contrib
        return out * self.output_mask
