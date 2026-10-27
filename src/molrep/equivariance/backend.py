"""Load the native library and register the exact adjoints of its contractions.

Every output of the product is linear in each of the other operands, so each
adjoint is the same contraction with a cotangent in one slot. ``product`` has
``adjoints`` as its backward: one fused pass over the edge schedule by target
returning the gradients of all three operands. ``adjoints`` has
``adjoints(second=True)`` as its backward — the same kernel with the cotangents
of its outputs as further operands — so second derivatives (force training) are
exact and one launch. Third derivatives are not provided. The symmetric
contraction (``contraction`` / ``contraction_adjoints``), the spherical
harmonics (``harmonics`` / ``harmonics_adjoints``), the Allegro contracter
(``contracter`` / ``contracter_adjoints``), and Allegro weighted channels
(``weighted_channels`` / ``weighted_channels_adjoints``) register in this
``load`` too, so a process that loaded the channelwise product first still
registers them. CUDA and Meta kernels come from the LibTorch library, usable by
a process without Python once an AOTInductor graph has captured the adjoints.

The library holds CUDA kernels only. A module built on a machine without a
CUDA device skips the load and evaluates its term table with
:mod:`molrep.equivariance.reference` on CPU tensors; on a machine with one,
the library must be built (``csrc/``) and found through
``MOLNEX_EQUIVARIANCE_LIBRARY`` (an explicit file), a directory of
``MOLNEX_OP_LIBRARY_PATH`` — the variable molnex's other native libraries and
its C++ runtime search — or ``molrep/lib/``, where a wheel installs it. The
file is tagged with the machine, ``libmolnex_equivariance.<machine>.so``, so
builds for several architectures share one directory; the untagged name is
accepted too.
"""

from __future__ import annotations

import os
import platform
import warnings
from pathlib import Path
from threading import Lock
from typing import NamedTuple

import torch

#: The library's base name; the C++ runtime loads ``lib<name>.<machine>.so``.
LIBRARY = "molnex_equivariance"

_BUILD_HINT = (
    "build it from csrc/ with CUDA enabled and put the build directory on "
    "MOLNEX_OP_LIBRARY_PATH:\n"
    "  cmake -S csrc -B build/csrc-$(uname -m) -DMOLNEX_OP_ENABLE_CUDA=ON "
    "-DTorch_DIR=$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')/Torch\n"
    "  cmake --build build/csrc-$(uname -m) -j   # -> build/csrc-<arch>/equivariance/\n"
    "(scripts/env/activate.sh sets MOLNEX_OP_LIBRARY_PATH)"
)


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


class NativeBackend:
    """Load once, at construction time; never load or dispatch Python in forward."""

    _loaded = False
    _warned = False
    _lock = Lock()

    @staticmethod
    def library_path() -> Path | None:
        """The built library for this machine, or ``None`` when none is found."""
        explicit = os.environ.get("MOLNEX_EQUIVARIANCE_LIBRARY")
        if explicit:
            return Path(explicit) if Path(explicit).is_file() else None
        listed = os.environ.get("MOLNEX_OP_LIBRARY_PATH", "")
        directories = [Path(d) for d in listed.split(os.pathsep) if d] + [
            Path(__file__).resolve().parents[1] / "lib"
        ]
        names = (f"lib{LIBRARY}.{platform.machine()}.so", f"lib{LIBRARY}.so")
        for directory in directories:
            for name in names:
                if (directory / name).is_file():
                    return directory / name
        return None

    @classmethod
    def load(cls) -> None:
        """Load the library and register its adjoints, once.

        A no-op without a CUDA device. With one and no built library on the
        search path it warns once with the build command and returns: CPU
        tensors still evaluate through :mod:`.reference`, and the first CUDA
        call fails on the unregistered operator.
        """
        if not torch.cuda.is_available():
            return
        with cls._lock:
            if cls._loaded or cls._warned:
                return
            path = cls.library_path()
            if path is None:
                cls._warned = True
                warnings.warn(
                    f"native equivariance library lib{LIBRARY}.{platform.machine()}.so not "
                    f"found; CUDA tensors cannot be evaluated until it is built: {_BUILD_HINT}",
                    stacklevel=3,
                )
                return
            torch.ops.load_library(str(path))
            for name, backward, context in (
                ("product", cls.product_backward, cls.product_context),
                ("adjoints", cls.adjoints_backward, cls.adjoints_context),
                ("contraction", cls.contraction_backward, cls.contraction_context),
                (
                    "contraction_adjoints",
                    cls.contraction_adjoints_backward,
                    cls.contraction_adjoints_context,
                ),
                ("harmonics", cls.harmonics_backward, cls.harmonics_context),
                (
                    "harmonics_adjoints",
                    cls.harmonics_adjoints_backward,
                    cls.harmonics_adjoints_context,
                ),
                ("contracter", cls.contracter_backward, cls.contracter_context),
                (
                    "contracter_adjoints",
                    cls.contracter_adjoints_backward,
                    cls.contracter_adjoints_context,
                ),
                (
                    "weighted_channels",
                    cls.weighted_channels_backward,
                    cls.weighted_channels_context,
                ),
                (
                    "weighted_channels_adjoints",
                    cls.weighted_channels_adjoints_backward,
                    cls.weighted_channels_adjoints_context,
                ),
            ):
                torch.library.register_autograd(
                    f"{LIBRARY}::{name}", backward, setup_context=context
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

    @staticmethod
    def contraction_context(ctx, inputs, output) -> None:
        # x, w, indices, packed, coefficients, out_offsets | values, config, channels, segments_out
        ctx.save_for_backward(*inputs[:6])
        ctx.config = inputs[6:10]

    @staticmethod
    def contraction_backward(ctx, grad: torch.Tensor):
        x, w, indices, *tables = ctx.saved_tensors
        need = ctx.needs_input_grad
        if not (need[0] or need[1]):
            return (None,) * 10
        empty = x.new_empty(0)
        gx, gw, _grad_g = torch.ops.molnex_equivariance.contraction_adjoints(
            x,
            w,
            indices,
            grad.contiguous(),
            empty,
            empty,
            empty,
            *tables,
            *ctx.config[:2],
            False,
            *ctx.config[2:],
        )
        return (gx if need[0] else None, gw if need[1] else None, *(None,) * 8)

    @staticmethod
    def contraction_adjoints_context(ctx, inputs, output) -> None:
        # x, w, indices, g, dgx, dgw, dgy, packed, coefficients, out_offsets |
        # values, config, second, channels, segments_out
        ctx.save_for_backward(*inputs[:4], *inputs[7:10])
        ctx.values_config = inputs[10:12]
        ctx.second = inputs[12]
        ctx.shape = inputs[13:15]

    @staticmethod
    def contraction_adjoints_backward(ctx, dgx, dgw, _dgrad_g):
        if ctx.second:
            raise NotImplementedError(
                "third derivatives of the native symmetric contraction are not implemented"
            )
        x, w, indices, g, *tables = ctx.saved_tensors
        need = ctx.needs_input_grad
        if not (need[0] or need[1] or need[3]):
            return (None,) * 15
        # An output that was not used has no cotangent: it contributes nothing.
        # dgy is unused. dw does not depend on dgw; dx and dg do.
        dgx = x.new_zeros(x.shape) if dgx is None else dgx.contiguous()
        dgw = w.new_zeros(w.shape) if dgw is None else dgw.contiguous()
        dx, dw, grad_g = torch.ops.molnex_equivariance.contraction_adjoints(
            x,
            w,
            indices,
            g,
            dgx,
            dgw,
            x.new_empty(0),
            *tables,
            *ctx.values_config,
            True,
            *ctx.shape,
        )
        return (
            dx if need[0] else None,
            dw if need[1] else None,
            None,
            grad_g if need[3] else None,
            *(None,) * 11,
        )

    @staticmethod
    def harmonics_context(ctx, inputs, output) -> None:
        # v, terms, coefficients | components
        ctx.save_for_backward(*inputs[:3])
        ctx.components = inputs[3]

    @staticmethod
    def harmonics_backward(ctx, grad: torch.Tensor):
        v, terms, coefficients = ctx.saved_tensors
        if not ctx.needs_input_grad[0]:
            return (None,) * 4
        empty = v.new_empty(0)
        gv, _grad_g = torch.ops.molnex_equivariance.harmonics_adjoints(
            v,
            grad.contiguous(),
            empty,
            empty,
            terms,
            coefficients,
            False,
            ctx.components,
        )
        return (gv, None, None, None)

    @staticmethod
    def harmonics_adjoints_context(ctx, inputs, output) -> None:
        # v, g, dgx, dgy, terms, coefficients | second, components
        ctx.save_for_backward(inputs[0], inputs[1], inputs[4], inputs[5])
        ctx.second = inputs[6]
        ctx.components = inputs[7]

    @staticmethod
    def harmonics_adjoints_backward(ctx, dgv, _dgrad_g):
        if ctx.second:
            raise NotImplementedError(
                "third derivatives of the native spherical harmonics are not implemented"
            )
        v, g, terms, coefficients = ctx.saved_tensors
        need = ctx.needs_input_grad
        if not (need[0] or need[1]):
            return (None,) * 8
        dgv = v.new_zeros(v.shape) if dgv is None else dgv.contiguous()
        dv, grad_g = torch.ops.molnex_equivariance.harmonics_adjoints(
            v, g, dgv, v.new_empty(0), terms, coefficients, True, ctx.components
        )
        return (
            dv if need[0] else None,
            grad_g if need[1] else None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    @staticmethod
    def contracter_context(ctx, inputs, output) -> None:
        # x1, x2, w, index, paths | cg, values, config, channels, segments_out
        ctx.save_for_backward(*inputs[:5])
        ctx.tables = inputs[5:8]
        ctx.shape = inputs[8:10]

    @staticmethod
    def contracter_backward(ctx, grad: torch.Tensor):
        x1, x2, w, index, paths = ctx.saved_tensors
        need = ctx.needs_input_grad
        if not (need[0] or need[1] or need[2]):
            return (None,) * 10
        empty = x1.new_empty(0)
        gx1, gx2, gw, _grad_g = torch.ops.molnex_equivariance.contracter_adjoints(
            x1,
            x2,
            w,
            index,
            grad.contiguous(),
            empty,
            empty,
            empty,
            empty,
            paths,
            *ctx.tables,
            False,
            *ctx.shape,
        )
        return (
            gx1 if need[0] else None,
            gx2 if need[1] else None,
            gw if need[2] else None,
            *(None,) * 7,
        )

    @staticmethod
    def contracter_adjoints_context(ctx, inputs, output) -> None:
        # x1, x2, w, index, g, dgx1, dgx2, dgw, dgy, paths | cg, values, config, second,
        # channels, segments_out
        ctx.save_for_backward(*inputs[:5], inputs[9])
        ctx.tables = inputs[10:13]
        ctx.second = inputs[13]
        ctx.shape = inputs[14:16]

    @staticmethod
    def contracter_adjoints_backward(ctx, dgx1, dgx2, dgw, _dgrad_g):
        if ctx.second:
            raise NotImplementedError(
                "third derivatives of the native contracter are not implemented"
            )
        x1, x2, w, index, g, paths = ctx.saved_tensors
        need = ctx.needs_input_grad
        if not (need[0] or need[1] or need[2] or need[4]):
            return (None,) * 16
        dgx1 = x1.new_zeros(x1.shape) if dgx1 is None else dgx1.contiguous()
        dgx2 = x2.new_zeros(x2.shape) if dgx2 is None else dgx2.contiguous()
        dgw = w.new_zeros(w.shape) if dgw is None else dgw.contiguous()
        dx1, dx2, dw, grad_g = torch.ops.molnex_equivariance.contracter_adjoints(
            x1,
            x2,
            w,
            index,
            g,
            dgx1,
            dgx2,
            dgw,
            x1.new_empty(0),
            paths,
            *ctx.tables,
            True,
            *ctx.shape,
        )
        return (
            dx1 if need[0] else None,
            dx2 if need[1] else None,
            dw if need[2] else None,
            None,
            grad_g if need[4] else None,
            *(None,) * 11,
        )

    @staticmethod
    def weighted_channels_context(ctx, inputs, output) -> None:
        # basis, weight, index, terms, coefficients | channels, num_irreps, num_rows
        ctx.save_for_backward(*inputs[:5])
        ctx.channels = inputs[5]
        ctx.num_irreps = inputs[6]
        ctx.num_rows = inputs[7]

    @staticmethod
    def weighted_channels_backward(ctx, grad: torch.Tensor):
        basis, weight, index, terms, coefficients = ctx.saved_tensors
        need = ctx.needs_input_grad
        if not (need[0] or need[1]):
            return (None,) * 8
        empty = basis.new_empty(0)
        grad_basis, grad_weight, _grad_g = torch.ops.molnex_equivariance.weighted_channels_adjoints(
            basis,
            weight,
            index,
            grad.contiguous(),
            empty,
            empty,
            empty,
            terms,
            coefficients,
            False,
            ctx.channels,
            ctx.num_irreps,
            ctx.num_rows,
        )
        return (
            grad_basis if need[0] else None,
            grad_weight if need[1] else None,
            *(None,) * 6,
        )

    @staticmethod
    def weighted_channels_adjoints_context(ctx, inputs, output) -> None:
        # basis, weight, index, g, dga, dgw, dgy, terms, coefficients |
        # second, channels, num_irreps, num_rows
        ctx.save_for_backward(*inputs[:4], inputs[7], inputs[8])
        ctx.second = inputs[9]
        ctx.channels = inputs[10]
        ctx.num_irreps = inputs[11]
        ctx.num_rows = inputs[12]

    @staticmethod
    def weighted_channels_adjoints_backward(ctx, dga, dgw, _dgrad_g):
        if ctx.second:
            raise NotImplementedError(
                "third derivatives of the native weighted channels are not implemented"
            )
        basis, weight, index, g, terms, coefficients = ctx.saved_tensors
        need = ctx.needs_input_grad
        if not (need[0] or need[1] or need[3]):
            return (None,) * 13
        dga = basis.new_zeros(basis.shape) if dga is None else dga.contiguous()
        dgw = weight.new_zeros(weight.shape) if dgw is None else dgw.contiguous()
        da, dw, grad_g = torch.ops.molnex_equivariance.weighted_channels_adjoints(
            basis,
            weight,
            index,
            g,
            dga,
            dgw,
            basis.new_empty(0),
            terms,
            coefficients,
            True,
            ctx.channels,
            ctx.num_irreps,
            ctx.num_rows,
        )
        return (
            da if need[0] else None,
            dw if need[1] else None,
            None,
            grad_g if need[3] else None,
            *(None,) * 9,
        )
