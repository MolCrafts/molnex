""":mod:`torch.compile` integration for MolNex.

:class:`Compiler` configures ``torch.compile`` once and applies it to a module
*or any other callable* — the compile seam on a potential is often a flat bound
method (an ``energy_core``), not an :class:`~torch.nn.Module`, and the call
returns the same type it was handed. :meth:`Compiler.explain` is the companion
measurement: one call yields graph count, graph-break count, break reasons and
the **source location** of each break, as a molnex-owned
:class:`ExplainSummary`.

Two regimes (see ``docs/molix/explanation/throughput-and-compilation.md``):

* **ragged / energy-only / general** — default inductor; dynamic shapes fine.
* **force training on PADDED fixed shapes** — ``Compiler(cuda_graphs=True)``
  applies the benchmarked CUDA-graph preset (``reduce-overhead``): inductor
  fusion + CUDA graphs reach ~10x eager on PiNet (measured ~130 steps/s on a
  GH200). REQUIRES static shapes — register
  :class:`molix.batch.PadMolecularBatch` and set the train loader's
  ``drop_last=True``. Without static shapes inductor recompiles every step and
  ``reduce-overhead`` is *slower* than eager.
* **both regimes, and AOTInductor** — :func:`register_inductor_fallbacks` keeps
  ``aten.index_add`` an extern ATen kernel, without which inductor's fused
  atomic scatter silently adds a padded-lane contribution to every scatter
  target (14.24 eV on the Polar PEO box). :class:`Compiler` and
  :class:`molix.export.Exporter` call it for you; a direct ``torch.compile``
  caller must call it first.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple, cast

import torch

from molix import logger as _logger_mod

logger = _logger_mod.getLogger(__name__)

#: Process-global latch for :func:`register_inductor_fallbacks` — the ATen
#: fallback is registered on inductor's global lowering table, so once is enough
#: and a second registration would trip inductor's duplicate-lowering assert.
_INDUCTOR_FALLBACKS_REGISTERED = False


def register_inductor_fallbacks() -> None:
    """Keep ``aten.index_add`` an extern ATen kernel under inductor.

    **The bug** (torch 2.12.0+cu130, GH200 sm_90; notes topic
    ``compile-no-graph-breaks``, paragraph "Inductor fused-scatter leak").
    Inductor decomposes ``index_add_`` to ``index_put_(accumulate=True)`` and
    lowers that to a Triton atomic ``ir.Scatter``, inlining the pointwise
    producer into the scatter's loader. The load is masked — padded lanes read
    ``0`` — but the fused epilogue is still evaluated and atomically added on
    those lanes, so every scatter target picks up
    ``(roundup(N, 32) - N) * f(0)``, where ``f`` is the fused producer. Any
    molnex scatter whose producer has ``f(0) != 0`` is affected: in Polar the
    producer is ``scale * node_es + shift``, giving ``29 padded lanes * shift =
    14.24 eV`` on the 4995-atom PEO box (forces stay exact — AOTAutograd's
    backward is a separate gather kernel). ``aot_eager`` and eager never see it,
    which is why break-count and ``aot_eager`` parity tests pass regardless.

    **Why a fallback.** Removing the ``index_add`` decompositions makes inductor
    emit the op as an extern ATen kernel: the source buffer is realised and the
    correct, unfused ``at::index_add`` runs, so no padded lane is ever
    evaluated. (Realising the source buffer *without* the fallback does not fix
    it — measured, GH200 job 1974354.) The extern call is an inductor lowering,
    not a Python callback, so it costs no graph break and works identically
    under AOTInductor, whose lowering path is the same. Validated on GH200 job
    1974355: PEO-box compiled-core ``dE = 0.0`` and the minimal probe down to
    round-off.

    **When to call it.** Before the first inductor lowering in the process:
    inductor caches its decomposition table behind ``lru_cache``, so the entries
    must be dropped *and* those caches cleared before a compile fills them in.
    :meth:`Compiler.__call__` and :meth:`molix.export.Exporter._package` call
    this, so molnex's own compile and AOTI-package seams are covered; code that
    calls ``torch.compile`` / ``torch._inductor.aoti_compile_and_package``
    directly must call it itself, first.

    The mutation is process-global (it edits torch's own decomposition and
    lowering tables) and this function is idempotent — repeat calls after the
    first return immediately, so calling it on every compile is cheap.
    """
    global _INDUCTOR_FALLBACKS_REGISTERED
    if _INDUCTOR_FALLBACKS_REGISTERED:
        return

    # Imported here, not at module scope: ``torch._inductor`` pulls in the whole
    # codegen stack, and ``import molix`` must stay cheap for non-compiled paths.
    from torch._decomp import decomposition_table
    from torch._inductor import decomposition as inductor_decomposition
    from torch._inductor import lowering as inductor_lowering

    aten = torch.ops.aten
    for op in (aten.index_add.default, aten.index_add_.default, aten.index_add.out):
        inductor_decomposition.decompositions.pop(op, None)
        decomposition_table.pop(op, None)
    # Both build the effective decomposition table under ``lru_cache``; a cached
    # table from an earlier compile in this process still holds the entries.
    for name in ("fast_random_decomps", "select_decomp_table"):
        fn = getattr(inductor_decomposition, name, None)
        if fn is not None and hasattr(fn, "cache_clear"):
            fn.cache_clear()
    for op in (aten.index_add.default, aten.index_add_.default):
        inductor_lowering.make_fallback(op)

    # Compiled energy is differentiated with ``autograd.grad(create_graph=…)``
    # for forces. Donated buffers require create_graph=False.
    torch._functorch.config.donated_buffer = False

    _INDUCTOR_FALLBACKS_REGISTERED = True
    logger.debug("Registered inductor ATen fallback for aten.index_add / aten.index_add_")


class ExplainSummary(NamedTuple):
    """What dynamo traced, projected onto molnex-owned types.

    Returned by :meth:`Compiler.explain`. Deliberately *not* dynamo's own
    result object: that dataclass lives at
    ``torch._dynamo.backends.debugging.ExplainOutput``, a private backends
    submodule whose path has already moved between torch releases — naming it
    in a molnex signature would pin a private path into molnex's types.

    Attributes:
        graph_count: Number of FX graphs dynamo produced.
        graph_break_count: Number of graph breaks; upstream defines it as
            ``graph_count - 1`` (so it is ``-1`` when nothing traced at all).
        break_reasons: One reason string per break, in trace order.
        break_sites: ``"<filename>:<lineno>"`` of the innermost user frame for
            each break, one per entry of :attr:`break_reasons` and in the same
            order; ``"?"`` when dynamo recorded no user stack. This is the
            answer to "which line broke the graph".
    """

    graph_count: int
    graph_break_count: int
    break_reasons: tuple[str, ...]
    break_sites: tuple[str, ...]


class Compiler:
    """Configure and apply ``torch.compile`` to a molnex module or callable.

    Construct with the desired config (or ``cuda_graphs=True`` for the
    force-training preset), then call the instance on the target to get the
    compiled wrapper. The call is generic in its argument, so a module stays a
    module and a bound method stays callable with the same signature::

        model = Compiler()(model)                  # default inductor
        model = Compiler(cuda_graphs=True)(model)  # CUDA-graph force preset
        model = Compiler(mode="max-autotune")(model)
        core = Compiler(fullgraph=True)(pots.energy_core)  # bound method

    :meth:`explain` is the measurement counterpart and is a ``@staticmethod``:
    it reports on dynamo's default configuration, independent of any instance.

    Every :meth:`__call__` first runs :func:`register_inductor_fallbacks`, so a
    model compiled through this class never hits inductor's fused-atomic-scatter
    leak on ``index_add``.
    """

    #: Benchmarked winning config for force training on padded fixed shapes.
    CUDA_GRAPH_PRESET: dict[str, object] = {
        "backend": "inductor",
        "fullgraph": True,
        "dynamic": False,
        "mode": "reduce-overhead",
    }

    def __init__(
        self,
        *,
        cuda_graphs: bool = False,
        backend: str = "inductor",
        fullgraph: bool = False,
        dynamic: bool | None = None,
        mode: str | None = None,
        donated_buffer: bool | None = None,
    ) -> None:
        """Capture a ``torch.compile`` configuration.

        Args:
            cuda_graphs: Apply :data:`CUDA_GRAPH_PRESET`, overriding the four
                params below. REQUIRES static shapes (``PadMolecularBatch`` +
                ``drop_last=True``).
            backend: Compile backend (default ``"inductor"``).
            fullgraph: Require a single graph (error on graph breaks).
            dynamic: Enable dynamic-shape tracing.
            mode: Compile mode (``"default"`` / ``"reduce-overhead"`` /
                ``"max-autotune"``).
            donated_buffer: AOTAutograd buffer donation. ``True`` reuses dead
                saved tensors during a ``create_graph=False`` backward (MD
                forces / virial) and cuts peak HBM. ``None`` (default) leaves
                the process default that :func:`register_inductor_fallbacks`
                sets to ``False``, which is required for force-supervised
                ``create_graph=True`` training.
        """
        if cuda_graphs:
            p = self.CUDA_GRAPH_PRESET
            backend = p["backend"]
            fullgraph = p["fullgraph"]
            dynamic = p["dynamic"]
            mode = p["mode"]
        self.backend = backend
        self.fullgraph = fullgraph
        self.dynamic = dynamic
        self.mode = mode
        self.donated_buffer = donated_buffer

    def __call__[T: Callable[..., Any]](self, target: T) -> T:
        """Return ``target`` compiled with this instance's configuration.

        Args:
            target: An :class:`~torch.nn.Module`, a plain function, or any
                other callable ``torch.compile`` accepts (a bound method is the
                usual non-module case).

        Returns:
            The compiled wrapper, typed as the same callable type as ``target``
            so a compiled module is still usable where a module is required.
        """
        # Unconditional (cheap after the first call): a non-inductor backend can
        # still hand a later graph to inductor, and the tables must be patched
        # before the process' first lowering, not after.
        register_inductor_fallbacks()
        if self.donated_buffer is not None:
            torch._functorch.config.donated_buffer = self.donated_buffer
        logger.info(
            f"Compiling {getattr(target, '__qualname__', type(target).__name__)} with "
            f"backend={self.backend}, fullgraph={self.fullgraph}, "
            f"dynamic={self.dynamic}, mode={self.mode}, "
            f"donated_buffer={self.donated_buffer}"
        )
        return cast(
            T,
            torch.compile(
                target,
                backend=self.backend,
                fullgraph=self.fullgraph,
                dynamic=self.dynamic,
                mode=self.mode,
            ),
        )

    @staticmethod
    def explain(target: Callable[..., Any], *args: object, **kwargs: object) -> ExplainSummary:
        """Trace ``target(*args, **kwargs)`` under dynamo and summarise the result.

        Two things this reports that a bare break count cannot:

        1. Upstream defines ``graph_break_count = graph_count - 1``, so a
           target that never traced at all reports ``-1``, not ``0``. Assert
           **both** fields together — ``graph_count == 1 and
           graph_break_count == 0`` — or a total tracing failure reads as
           "better than clean".
        2. :attr:`ExplainSummary.break_sites` names the file and line of each
           break, so "zero breaks" regressions can be attributed instead of
           grepped out of a reason string.

        This is a ``@staticmethod``: it measures under **dynamo's default
        configuration**, not the calling instance's ``backend`` / ``mode`` /
        ``fullgraph``. To measure under a different backend, set it at the call
        site. ``torch._dynamo.explain`` resets dynamo itself, before and after
        the trace, so callers must not add their own ``reset()``.

        Args:
            target: Module, function or bound method to trace.
            *args: Positional arguments forwarded to ``target``.
            **kwargs: Keyword arguments forwarded to ``target``.

        Returns:
            The projected :class:`ExplainSummary`.
        """
        out = torch._dynamo.explain(target)(*args, **kwargs)
        return ExplainSummary(
            graph_count=int(out.graph_count),
            graph_break_count=int(out.graph_break_count),
            break_reasons=tuple(str(r.reason) for r in out.break_reasons),
            break_sites=tuple(
                f"{r.user_stack[-1].filename}:{r.user_stack[-1].lineno}" if r.user_stack else "?"
                for r in out.break_reasons
            ),
        )
