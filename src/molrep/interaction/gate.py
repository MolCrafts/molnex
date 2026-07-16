"""Gated equivariant nonlinearity (parameter-free).

A drop-in, graph-break-free equivalent of ``e3nn.nn.Gate`` with identical
numerics, expressed with ``cuequivariance`` irreps so molrep keeps a single
irreps backend (no ``e3nn`` dependency). The block has **no learnable
parameters** — only activation functions and their ``normalize2mom`` constants.

The input is the sorted direct sum of ``(scalars, gates, gated)`` irreps, the
same convention as ``e3nn.nn.Gate`` and MACE's ``GatedEquivariantBlock``.
Scalars are activated directly; each gated (l>0) irrep is multiplied by a
sigmoid-activated scalar gate. Supports ``ir_mul`` (cuequivariance) and
``mul_ir`` (e3nn) layouts.

Reference:
    e3nn ``nn.Gate``; MACE ``mace.modules.gate``.
    https://arxiv.org/abs/2206.07697
"""

from __future__ import annotations

from typing import Callable, Sequence

import cuequivariance as cue
import torch

_NORM_CACHE: dict[int, float] = {}


def _normalize2mom_cst(fn: Callable) -> float:
    """Return ``sqrt(1 / E[f(z)^2])`` for ``z ~ N(0, 1)`` (e3nn convention).

    Matches ``e3nn.math.normalize2mom``: 1M standard-normal samples, cached by
    function identity with a fixed seed for reproducibility.
    """
    key = id(fn)
    cached = _NORM_CACHE.get(key)
    if cached is not None:
        return cached
    gen = torch.Generator(device="cpu").manual_seed(0)
    z = torch.randn(1_000_000, generator=gen, dtype=torch.float64)
    result = fn(z).pow(2).mean().pow(-0.5).item()
    _NORM_CACHE[key] = result
    return result


def _irreps(spec) -> cue.Irreps:
    return spec if isinstance(spec, cue.Irreps) else cue.Irreps("O3", spec)


class GatedNonlinearity(torch.nn.Module):
    """Parameter-free gated equivariant nonlinearity (e3nn ``Gate`` numerics).

    Args:
        irreps_scalars: Scalar (l=0) irreps activated directly.
        act_scalars: One activation per scalar group (e.g. ``[silu]``).
        irreps_gates: Scalar (l=0) irreps used as gates; ``num_irreps`` must
            equal that of ``irreps_gated``.
        act_gates: One activation per gate group (e.g. ``[sigmoid]``).
        irreps_gated: l>0 irreps gated by ``irreps_gates``.
        layout: ``"ir_mul"`` (cuequivariance) or ``"mul_ir"`` (e3nn).
    """

    def __init__(
        self,
        irreps_scalars,
        act_scalars: Sequence[Callable | None],
        irreps_gates,
        act_gates: Sequence[Callable | None],
        irreps_gated,
        layout: str = "ir_mul",
    ) -> None:
        super().__init__()
        irreps_scalars = _irreps(irreps_scalars)
        irreps_gates = _irreps(irreps_gates)
        irreps_gated = _irreps(irreps_gated)

        def _lmax(irr: cue.Irreps) -> int:
            return max((mi.ir.l for mi in irr), default=0)

        if _lmax(irreps_gates) > 0:
            raise ValueError(f"gates must be scalars, got {irreps_gates}")
        if _lmax(irreps_scalars) > 0:
            raise ValueError(f"scalars must be scalars, got {irreps_scalars}")
        if irreps_gates.num_irreps != irreps_gated.num_irreps:
            raise ValueError(
                f"irreps_gated has {irreps_gated.num_irreps} irreps but "
                f"irreps_gates has {irreps_gates.num_irreps}"
            )
        if layout not in ("mul_ir", "ir_mul"):
            raise ValueError(f"layout must be 'mul_ir' or 'ir_mul', got '{layout}'")
        self._layout_is_ir_mul = layout == "ir_mul"

        scalars = list(irreps_scalars)
        gates = list(irreps_gates)
        gated = list(irreps_gated)
        unsorted = scalars + gates + gated

        # sort the concatenated irreps (stable), as e3nn / MACE do
        sort = cue.Irreps("O3", [mi for mi in unsorted]).sort()
        sorted_mis = list(sort.irreps)
        perm = list(sort.perm)  # sorted[i] == unsorted[perm[i]]
        self._irreps_in = sort.irreps.simplify()

        n_scalar = len(scalars)
        n_gate = len(gates)

        scalar_sorted, gate_sorted, gated_sorted = [], [], []
        for si in range(len(sorted_mis)):
            orig = int(perm[si])
            if orig < n_scalar:
                scalar_sorted.append(si)
            elif orig < n_scalar + n_gate:
                gate_sorted.append(si)
            else:
                gated_sorted.append(si)

        def _d(mi) -> int:
            return mi.mul * mi.ir.dim

        offsets, off = [], 0
        for mi in sorted_mis:
            offsets.append(off)
            off += _d(mi)

        if scalar_sorted:
            self._s_start = offsets[scalar_sorted[0]]
            self._s_len = sum(_d(sorted_mis[i]) for i in scalar_sorted)
        else:
            self._s_start, self._s_len = 0, 0
        if gate_sorted:
            self._g_start = offsets[gate_sorted[0]]
            self._g_len = sum(_d(sorted_mis[i]) for i in gate_sorted)
        else:
            self._g_start, self._g_len = 0, 0

        gate_off_by_gated, cum = [], 0
        for mi in gated:
            gate_off_by_gated.append(cum)
            cum += mi.mul

        gated_info = []
        for si in gated_sorted:
            mi = sorted_mis[si]
            gated_orig = int(perm[si]) - n_scalar - n_gate
            gated_info.append(
                (offsets[si], _d(mi), mi.ir.dim, mi.mul, gate_off_by_gated[gated_orig])
            )
        self._gated_info = gated_info

        acts = list(act_scalars)
        if len(acts) == 1 and len(scalars) > 1:
            acts = acts * len(scalars)
        self._act_scalar = acts[0] if acts and acts[0] is not None else None
        self._scalar_cst = (
            _normalize2mom_cst(self._act_scalar) if self._act_scalar is not None else 1.0
        )

        actg = list(act_gates)
        if len(actg) == 1 and len(gates) > 1:
            actg = actg * len(gates)
        self._act_gate = actg[0] if actg and actg[0] is not None else None
        self._gate_cst = _normalize2mom_cst(self._act_gate) if self._act_gate is not None else 1.0

    @property
    def irreps_in(self) -> cue.Irreps:
        """Sorted+simplified input irreps the block expects."""
        return self._irreps_in

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply the gated nonlinearity.

        Args:
            features: ``(..., irreps_in.dim)`` in the configured layout.

        Returns:
            ``(..., irreps_out.dim)`` with scalars activated and l>0 irreps gated.
        """
        scalars = features.narrow(-1, self._s_start, self._s_len)
        if self._act_scalar is not None:
            scalars = self._act_scalar(scalars) * self._scalar_cst

        if not self._gated_info:
            return scalars

        gates = features.narrow(-1, self._g_start, self._g_len)
        if self._act_gate is not None:
            gates = self._act_gate(gates) * self._gate_cst

        ir_mul = self._layout_is_ir_mul
        batch_shape = features.shape[:-1]
        parts = [scalars]
        for gd_start, gd_len, ir_dim, mul, g_off in self._gated_info:
            gated_chunk = features.narrow(-1, gd_start, gd_len)
            gate_chunk = gates.narrow(-1, g_off, mul)
            if ir_mul:
                gated_3d = gated_chunk.reshape(*batch_shape, ir_dim, mul)
                result = (gated_3d * gate_chunk.unsqueeze(-2)).reshape(*batch_shape, ir_dim * mul)
            else:
                gated_3d = gated_chunk.reshape(*batch_shape, mul, ir_dim)
                result = (gated_3d * gate_chunk.unsqueeze(-1)).reshape(*batch_shape, mul * ir_dim)
            parts.append(result)
        return torch.cat(parts, dim=-1)
