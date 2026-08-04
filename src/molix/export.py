"""AOTInductor export for MolNex models.

:class:`Exporter` packages a **flat tensor-IO** ``nn.Module`` (tensors in,
tensors / pytree out — e.g. :class:`molix.engine.EngineForward` wrapping a
potential) into a single ``.pt2`` loadable from Python (:meth:`Exporter.load`)
or C++ (``AOTIModelPackageLoader``). It exposes two explicit entry points — the
caller picks; there is no auto-detection or silent fallback:

* :meth:`Exporter.export` — direct ``torch.export``. For plain inference /
  energy modules and analytic-force modules. Preserves parameter FQNs (C++
  weight hot-reload works).
* :meth:`Exporter.export_pretraced` — ``make_fx`` pretrace then export, for any
  module that runs autograd **inside** ``forward`` (the common case: a force
  head computing ``F = -dE/dx``).

The split is by **mechanism, not by output**: both can emit forces; they differ
only in whether a ``make_fx`` pretrace runs first.

Why ``export_pretraced`` exists: handing a live in-forward gradient op straight to
``torch.export`` fails on torch 2.12 — functorch raises *"Cannot access data
pointer of Tensor that doesn't have storage"*; autograd raises *"fake tensor in
the exported program constant's list"*. The fix — validated on PiNet and shipped
by NequIP/Allegro (``nequip-compile --mode aotinductor``; arXiv 2504.16068;
pytorch#153251) — is to pre-trace forward+backward into one flat aten graph with
``make_fx`` first (running real autograd under a tracing mode), then export THAT,
so ``export`` never sees a live gradient op.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path

import torch
import torch.nn as nn
from torch._decomp import core_aten_decompositions
from torch.fx.experimental.proxy_tensor import make_fx

from molix import logger as _logger_mod

logger = _logger_mod.getLogger(__name__)


class Exporter:
    """Compile a flat tensor-IO module to a deployable AOTInductor ``.pt2``.

    Two explicit entry points — pick by whether the forward differentiates
    internally (no auto-detection, no silent fallback):

    * :meth:`export` — direct ``torch.export``. For plain inference / energy
      modules and analytic-force modules. **Preserves parameter FQNs**, so the
      C++ runtime can hot-reload weights by name.
    * :meth:`export_pretraced` — ``make_fx`` pretrace then export. For any module
      that runs autograd inside ``forward`` (functorch / ``autograd.grad``) — the
      common case being a force head ``F = -dE/dx`` — which defeats direct
      ``torch.export``. Named for the mechanism (pretrace), not the output: it is
      not "forces-only", and :meth:`export` can also emit analytic forces.

    Example (PiNet forces, varying atom/edge counts → no padding needed)::

        from molix import Exporter
        from molix.engine import EngineForward

        flat = EngineForward(pinet_potential)           # tensors in, (E, F) out
        n = torch.export.Dim("n_atoms", min=2)
        e = torch.export.Dim("n_edges", min=1)
        Exporter(flat).export_pretraced(
            (Z, pos, edge_index),
            "pinet.pt2",
            dynamic_shapes=({0: n}, {0: n}, {0: e}),
        )
        runner = Exporter.load("pinet.pt2")
        energy, forces = runner(Z2, pos2, edge_index2)   # any N / E
    """

    def __init__(self, module: nn.Module) -> None:
        if not isinstance(module, nn.Module):
            raise TypeError(f"module must be an nn.Module, got {type(module).__name__}")
        self.module = module

    def export(
        self,
        example_inputs: tuple[torch.Tensor, ...],
        path: str | Path,
        *,
        dynamic_shapes: object | None = None,
        inductor_configs: dict[str, object] | None = None,
    ) -> Path:
        """Export a non-differentiating module directly to a ``.pt2`` package.

        Use for plain inference / energy modules and analytic-force modules.
        Preserves the original parameter FQNs, so the C++ runtime can hot-reload
        weights by name (``ModelRunner.update_weights``).

        A forward that differentiates internally (``F = -dE/dx`` via autograd or
        functorch) will make the underlying ``torch.export`` raise — use
        :meth:`export_pretraced` for those.

        Args:
            example_inputs: Tuple of example input tensors. Values must form a
                valid forward; sizes are placeholders when *dynamic_shapes* is set.
            path: Output ``.pt2`` file path. Parent dirs are created.
            dynamic_shapes: ``torch.export`` dynamic-shape spec — a per-input
                ``{dim_index: torch.export.Dim}`` tuple — so one ``.pt2`` serves
                varying sizes. ``None`` bakes the example sizes in.
            inductor_configs: Optional inductor config overrides forwarded to
                ``aoti_compile_and_package`` — e.g.
                ``{"aot_inductor.use_runtime_constant_folding": True}`` to keep
                every parameter as a reloadable constant buffer.

        Returns:
            The written ``.pt2`` path.

        Raises:
            TypeError: If *example_inputs* is not a tuple.
        """
        self._prepare(example_inputs)
        exported = torch.export.export(self.module, example_inputs, dynamic_shapes=dynamic_shapes)
        return self._package(exported, path, inductor_configs)

    def export_pretraced(
        self,
        example_inputs: tuple[torch.Tensor, ...],
        path: str | Path,
        *,
        dynamic_shapes: object | None = None,
        inductor_configs: dict[str, object] | None = None,
    ) -> Path:
        """Export a module that differentiates inside ``forward`` to a ``.pt2``.

        Pre-traces forward + internal backward into one flat aten graph with
        ``make_fx`` (running real autograd under a tracing mode), then exports
        THAT — ``torch.export`` never sees a live gradient op, so neither the
        functorch "no storage" nor the autograd "fake constant" failure fires.
        ``core_aten_decompositions`` lowers second-order backward ops (e.g.
        ``silu_backward``) to exportable primitives.

        **Caveat:** ``make_fx`` lifts parameters as positional constants,
        dropping their FQNs — so weight hot-reload by name is unavailable for
        these packages (re-export to update weights). Args identical to
        :meth:`export`.

        Raises:
            RuntimeError: If the export target is CPU. torch 2.12's CPU inductor
                miscompiles the baked-in backward scatter (forces ~7% wrong while
                energy stays exact); refusing beats silently shipping bad forces.
                Export on CUDA, or use TorchScript + ``torch::autograd::grad`` for
                a CPU force path.
        """
        if self._device().type == "cpu":
            raise RuntimeError(
                "export_pretraced is disabled on CPU: torch 2.12's CPU AOTInductor "
                "miscompiles the baked-in backward scatter (atomic_add under "
                "simdlen=0) — energy is bit-exact but forces come out ~7% wrong. "
                "Export on CUDA (Triton backend is correct), or for a CPU force "
                "path use TorchScript trace + torch::autograd::grad at runtime."
            )
        self._prepare(example_inputs)
        graph = make_fx(
            self.module,
            decomposition_table=core_aten_decompositions(),
            tracing_mode="symbolic",
            _allow_non_fake_inputs=True,
        )(*example_inputs)
        exported = torch.export.export(graph, example_inputs, dynamic_shapes=dynamic_shapes)
        return self._package(exported, path, inductor_configs)

    @staticmethod
    def load(path: str | Path) -> object:
        """Load a ``.pt2`` produced by :meth:`export` / :meth:`export_pretraced`."""
        return torch._inductor.aoti_load_package(str(path))

    def _prepare(self, example_inputs: tuple[torch.Tensor, ...]) -> None:
        if not isinstance(example_inputs, tuple):
            raise TypeError(f"example_inputs must be a tuple, got {type(example_inputs).__name__}")
        self.module.eval()
        # Warm up: materialize lazy params at concrete shapes BEFORE tracing
        # (materializing them inside a functorch trace corrupts its wrappers).
        self.module(*example_inputs)

    def _package(
        self,
        exported: object,
        path: str | Path,
        inductor_configs: dict[str, object] | None,
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._cpu_codegen_workaround():
            torch._inductor.aoti_compile_and_package(
                exported, package_path=str(path), inductor_configs=inductor_configs
            )
        logger.info(f"Exported {self.module.__class__.__name__} to {path}")
        return path

    def _device(self) -> torch.device:
        for p in self.module.parameters():
            return p.device
        return torch.device("cpu")

    @contextlib.contextmanager
    def _cpu_codegen_workaround(self) -> Iterator[None]:
        """Disable CPU SIMD vectorization during compile on CPU targets.

        torch 2.12's CPU inductor codegen asserts ``index.is_vec`` on integer
        index stores (``_inductor/codegen/cpp.py``), which the atom-type gather
        triggers. ``cpp.simdlen = 0`` sidesteps it. CPU-only; GPU/Triton is
        unaffected.
        """
        cpp_cfg = torch._inductor.config.cpp
        if self._device().type != "cpu":
            yield
            return
        saved = cpp_cfg.simdlen
        cpp_cfg.simdlen = 0
        try:
            yield
        finally:
            cpp_cfg.simdlen = saved
