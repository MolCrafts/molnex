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
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple, cast

import torch

from molix import logger as _logger_mod

logger = _logger_mod.getLogger(__name__)


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
        logger.info(
            f"Compiling {getattr(target, '__qualname__', type(target).__name__)} with "
            f"backend={self.backend}, fullgraph={self.fullgraph}, "
            f"dynamic={self.dynamic}, mode={self.mode}"
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
