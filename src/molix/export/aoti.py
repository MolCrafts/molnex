"""The AOTInductor format: a ``.pt2`` package for a flat tensor-IO ``nn.Module``.

The path documented for PyTorch 2.14:

    https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler_aot_inductor.html

:func:`package` is those two calls — ``torch.export.export`` then
``torch._inductor.aoti_compile_and_package`` — plus the inductor fallback
``index_add`` needs, registered immediately before the package step, and one
step before them: the forward is traced with ``make_fx`` below autograd first,
so a forward that calls ``torch.autograd.grad`` (forces as ``-dE/dpos``)
reaches ``torch.export`` as plain ATen forward-and-backward ops. A C++
runtime loads the ``.pt2`` with ``torch::inductor::AOTIModelPackageLoader``;
operators outside ATen run through the dispatcher there, so their namespaces
(:func:`custom_op_namespaces`) must be registered natively in that process.

That trace takes the parameters and buffers as inputs and the exported
module re-registers them under the source module's own names, so parameter
fully-qualified names are preserved and
``AOTIModelPackageLoader::load_constants`` can replace weights by state-dict
name; ``aot_inductor.use_runtime_constant_folding`` keeps the parameters as
reloadable buffers.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn as nn
from torch.fx.experimental.proxy_tensor import make_fx

from molix import logger as _logger_mod
from molix.compile import register_inductor_fallbacks

logger = _logger_mod.getLogger(__name__)

#: Operator namespaces every AOTInductor runtime provides.
_RUNTIME_NAMESPACES = frozenset({"aten", "prims", "inductor"})


class Package(NamedTuple):
    """A written package and the non-ATen operator namespaces it calls."""

    path: Path
    op_namespaces: frozenset[str]


def package(
    module: nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    path: str | Path,
    *,
    dynamic_shapes: tuple[dict[int, torch.export.Dim] | None, ...] | None = None,
    inductor_configs: dict[str, object] | None = None,
) -> Package:
    """Package *module* with ``torch.export.export`` and ``aoti_compile_and_package``.

    Args:
        module: Flat tensor-IO ``nn.Module``. ``forward`` receives *example_inputs*
            and returns tensors.
        example_inputs: Example arguments. Values must form a valid forward;
            sizes are placeholders when *dynamic_shapes* is set.
        path: Output ``.pt2`` path. Parent directories are created.
        dynamic_shapes: ``torch.export`` dynamic-shape spec — a per-input
            ``{dim_index: torch.export.Dim}`` tuple. ``None`` bakes the example
            sizes in.
        inductor_configs: Forwarded to ``aoti_compile_and_package``; e.g.
            ``{"aot_inductor.use_runtime_constant_folding": True}`` keeps the
            parameters reloadable constant buffers.

    Returns:
        The written ``.pt2`` path and :func:`custom_op_namespaces` of the
        exported program.

    Raises:
        TypeError: If *module* is not an ``nn.Module``, or *example_inputs*
            is not a tuple.
    """
    if not isinstance(module, nn.Module):
        raise TypeError(f"module must be an nn.Module, got {type(module).__name__}")
    if not isinstance(example_inputs, tuple):
        raise TypeError(f"example_inputs must be a tuple, got {type(example_inputs).__name__}")
    module.eval()
    # Materialise lazy parameters and run eager-only checks before tracing.
    module(*example_inputs)
    # The traced module takes ``*args``; its spec is the per-input tuple under "args".
    exported = torch.export.export(
        _AutogradFree(module, example_inputs),
        example_inputs,
        dynamic_shapes=None if dynamic_shapes is None else {"args": tuple(dynamic_shapes)},
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # index_add's fused atomic scatter mis-counts padded lanes unless the ATen
    # fallback is registered before this lowering.
    register_inductor_fallbacks()
    with _cpu_codegen_workaround(_compile_device(module, example_inputs)):
        torch._inductor.aoti_compile_and_package(
            exported, package_path=str(path), inductor_configs=inductor_configs
        )
    logger.info(f"Exported {module.__class__.__name__} to {path}")
    return Package(path, custom_op_namespaces(exported))


class _AutogradFree(nn.Module):
    """*module* with autograd traced away: forward and backward as one ATen graph.

    ``torch.export`` cannot trace ``torch.autograd.grad`` over a module with
    parameters: the tensors the backward reads (a saved weight's transpose)
    reach it untracked and leak into the program as fake constants (torch
    2.14.1 and 2.15 nightly, strict or not). ``make_fx`` runs autograd during
    tracing instead and records what it executes. The parameters and buffers
    are inputs of that trace and are re-registered here under *module*'s own
    names, so the exported program lifts them by state-dict name.

    Args:
        module: Module to trace, already called once eagerly (lazy parameters
            materialised, eager-only checks done).
        example_inputs: Example arguments of ``module.forward``.
    """

    def __init__(self, module: nn.Module, example_inputs: tuple[torch.Tensor, ...]) -> None:
        super().__init__()
        for name, child in module.named_children():
            self.add_module(name, child)
        for name, parameter in module.named_parameters(recurse=False):
            self.register_parameter(name, parameter)
        for name, buffer in module.named_buffers(recurse=False):
            self.register_buffer(name, buffer)
        self._parameter_names = [name for name, _ in module.named_parameters()]
        self._buffer_names = [name for name, _ in module.named_buffers()]

        def call(parameters, buffers, *args):
            return torch.func.functional_call(module, (parameters, buffers), args)

        # No duck sizing: the trace gives parameters symbolic sizes too, and a
        # weight dimension equal to an example size (8 atoms) would share its
        # symbol, so a guard on the weight would pin the atom count.
        with torch.fx.experimental._config.patch(use_duck_shape=False):
            self.graph = make_fx(call, tracing_mode="symbolic", _allow_non_fake_inputs=True)(
                dict(module.named_parameters()), dict(module.named_buffers()), *example_inputs
            )

    def forward(self, *args: torch.Tensor) -> object:
        parameters = {name: self.get_parameter(name) for name in self._parameter_names}
        buffers = {name: self.get_buffer(name) for name in self._buffer_names}
        return self.graph(parameters, buffers, *args)


def load(path: str | Path) -> object:
    """Load a ``.pt2`` produced by :func:`package` into this Python process."""
    return torch._inductor.aoti_load_package(str(path))


def custom_op_namespaces(exported: torch.export.ExportedProgram) -> frozenset[str]:
    """Namespaces of the operators in ``exported`` that are not ATen's.

    A C++ runtime needs each of them registered natively before it can load
    the package.
    """
    namespaces = {
        node.target.namespace
        for node in exported.graph.nodes
        if node.op == "call_function" and isinstance(node.target, torch._ops.OpOverload)
    }
    return frozenset(namespaces - _RUNTIME_NAMESPACES)


def _compile_device(module: nn.Module, example_inputs: tuple[torch.Tensor, ...]) -> torch.device:
    for tensor in example_inputs:
        if isinstance(tensor, torch.Tensor):
            return tensor.device
    for parameter in module.parameters():
        return parameter.device
    return torch.device("cpu")


@contextlib.contextmanager
def _cpu_codegen_workaround(device: torch.device) -> Iterator[None]:
    """Disable CPU SIMD vectorization while compiling a CPU package.

    torch 2.12's CPU inductor codegen asserts ``index.is_vec`` on the integer
    index stores an atom-type gather emits (``_inductor/codegen/cpp.py``).
    ``cpp.simdlen = 0`` sidesteps it. A CUDA package is unchanged.
    """
    cpp_cfg = torch._inductor.config.cpp
    if device.type != "cpu":
        yield
        return
    saved = cpp_cfg.simdlen
    cpp_cfg.simdlen = 0
    try:
        yield
    finally:
        cpp_cfg.simdlen = saved
