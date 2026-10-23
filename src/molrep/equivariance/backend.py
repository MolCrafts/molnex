"""Load the native library and register the exact adjoints of its contractions.

Every output of the product is linear in each of the other operands, so each
adjoint is the same contraction with a cotangent in one slot. ``product`` has
``adjoints`` as its backward: one fused pass over the edge schedule by target
returning the gradients of all three operands. ``adjoints`` has
``adjoints(second=True)`` as its backward — the same kernel with the cotangents
of its outputs as further operands — so second derivatives (force training) are
exact and one launch. Third derivatives are not provided. CUDA and Meta kernels
come from the LibTorch library, usable by a process without Python once an
AOTInductor graph has captured the adjoints.
"""

from __future__ import annotations

import os
from pathlib import Path
from threading import Lock

import torch


class NativeBackend:
    """Load once, at construction time; never load or dispatch Python in forward."""

    _loaded = False
    _lock = Lock()

    @classmethod
    def load(cls) -> None:
        with cls._lock:
            if cls._loaded:
                return
            configured = os.environ.get("MOLNEX_EQUIVARIANCE_LIBRARY")
            path = (
                Path(configured)
                if configured
                else Path(__file__).resolve().parents[1] / "lib/libmolnex_equivariance.so"
            )
            if not path.is_file():
                raise FileNotFoundError(
                    f"native equivariance library not found: {path}; build csrc/equivariance "
                    "and set MOLNEX_EQUIVARIANCE_LIBRARY"
                )
            torch.ops.load_library(str(path))
            torch.library.register_autograd(
                "molnex_equivariance::product",
                cls.product_backward,
                setup_context=cls.product_context,
            )
            torch.library.register_autograd(
                "molnex_equivariance::adjoints",
                cls.adjoints_backward,
                setup_context=cls.adjoints_context,
            )
            cls._loaded = True

    @staticmethod
    def product_context(ctx, inputs, output) -> None:
        # a, b, w, source, target, order_t, rowptr_t, order_s, rowptr_s |
        # coefficients, rows, l1, l2, channels
        ctx.save_for_backward(*inputs[:9])
        ctx.coefficients = inputs[9]
        ctx.config = inputs[11:14]

    @staticmethod
    def product_backward(ctx, grad: torch.Tensor):
        a, b, w, *schedule = ctx.saved_tensors
        need = ctx.needs_input_grad
        if not (need[0] or need[1] or need[2]):
            return (None,) * 14
        empty = a.new_empty(0, 0)
        ga, gb, gw, _ = torch.ops.molnex_equivariance.adjoints(
            a,
            b,
            w,
            grad.contiguous(),
            empty,
            empty,
            empty,
            *schedule,
            ctx.coefficients,
            False,
            *ctx.config,
        )
        return (
            ga if need[0] else None,
            gb if need[1] else None,
            gw if need[2] else None,
            *(None,) * 11,
        )

    @staticmethod
    def adjoints_context(ctx, inputs, output) -> None:
        # a, b, w, g, da, db, dw, source, target, order_t, rowptr_t, order_s, rowptr_s |
        # coefficients, second, l1, l2, channels
        ctx.save_for_backward(*inputs[:4], *inputs[7:13])
        ctx.coefficients = inputs[13]
        ctx.second = inputs[14]
        ctx.config = inputs[15:18]

    @staticmethod
    def adjoints_backward(ctx, da, db, dw, dgrad_g):
        if ctx.second:
            raise NotImplementedError(
                "third derivatives of the native channelwise product are not implemented"
            )
        a, b, w, g, *schedule = ctx.saved_tensors
        need = ctx.needs_input_grad
        if not any(need[:4]):
            return (None,) * 18
        # An output that was not used has no cotangent: it contributes nothing.
        da = a.new_zeros(a.shape) if da is None else da.contiguous()
        db = b.new_zeros(b.shape) if db is None else db.contiguous()
        dw = w.new_zeros(w.shape) if dw is None else dw.contiguous()
        grad_a, grad_b, grad_w, grad_g = torch.ops.molnex_equivariance.adjoints(
            a, b, w, g, da, db, dw, *schedule, ctx.coefficients, True, *ctx.config
        )
        return (
            grad_a if need[0] else None,
            grad_b if need[1] else None,
            grad_w if need[2] else None,
            grad_g if need[3] else None,
            *(None,) * 14,
        )
