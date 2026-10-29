""":mod:`torch.compile` integration for MolNex.

:class:`Compiler` configures ``torch.compile`` once and applies it to a module
*or any other callable* — a compile seam may be a bound method (a potential's
``energy``), not only an :class:`~torch.nn.Module`, and the call
returns the same type it was handed. :meth:`Compiler.explain` is the companion
measurement: one call yields graph count, graph-break count, break reasons and
the **source location** of each break, as a molnex-owned
:class:`ExplainSummary`. :func:`unwrap` is the one reader of the module under a
compiled (or DDP / FSDP) wrapper.

Two regimes (see ``docs/molix/explanation/throughput-and-compilation.md``):

* **ragged / energy-only / general** — default inductor; ``dynamic=True``.
  TensorDict batches are Python objects, so dynamo still guards leaf sizes
  unless :func:`mark_dynamic_batch` runs on them *before* the compiled
  region (``Compiler`` does this when ``dynamic=True``).
* **force training on PADDED fixed shapes** — ``Compiler(cuda_graphs=True)``
  applies the benchmarked CUDA-graph preset (``reduce-overhead``): inductor
  fusion + CUDA graphs reach ~10x eager on PiNet (measured ~130 steps/s on a
  GH200). REQUIRES static shapes — register
  :class:`molix.data.MoleculePadder` and set the train loader's
  ``drop_last=True``. Without static shapes inductor recompiles every step and
  ``reduce-overhead`` is *slower* than eager.
* **energy → force seams** — :class:`Compiler` turns on
  ``torch._dynamo.config.trace_autograd_ops`` (process-wide, off by default
  upstream as of torch 2.15), so a ``torch.autograd.grad`` on a position leaf
  created *inside* the traced function is one graph instead of a break at the
  ``autograd.grad`` call. Inference (``create_graph=False``, forces returned
  detached) is a single graph today; training, which returns the force still
  attached to the parameters, additionally needs pytorch/pytorch#196653, and a
  strain leaf made by ``torch.zeros(..., requires_grad=True)`` needs
  pytorch/pytorch#196652 (both 2026-09-10, see notes topic
  ``compile-no-graph-breaks``).

* **buffer donation off by default** — compiled energy is differentiated with
  ``autograd.grad(create_graph=True)`` for force-supervised training, which
  AOTAutograd serves only without donated buffers; ``Compiler(donated_buffer=True)``
  turns it on for ``create_graph=False`` use (MD forces).

``aten.index_add`` is lowered by inductor as torch ships it. molnex used to force
it to an extern ATen kernel (``register_inductor_fallbacks``) against a fused
atomic-scatter leak on torch 2.12 + CUDA (notes topic ``compile-no-graph-breaks``);
measured absent on torch 2.14.1, CPU and GH200 (2026-10-29), and the forced
fallback broke AOTInductor packages on torch ≤ 2.14 (the proxy executor counts
``index_add``'s defaulted ``alpha``), so it was removed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple, cast

import torch
from torch._dynamo.eval_frame import OptimizedModule
from torch.nn.parallel import DistributedDataParallel

from molix import logger as _logger_mod

logger = _logger_mod.getLogger(__name__)

# COO connectivity whose *count* axis is 1, matching ``molix.data.collate.INDEX_KEYS``.
# ``pair_index`` is ``(E, 2)`` (count axis 0); ``bond_index`` is ``(2, N)``.
_COO_COUNT_AXIS_1: frozenset[str] = frozenset({"bond_index", "angle_index", "dihedral_index"})


def mark_dynamic_batch(batch: object) -> None:
    """Mark ragged TensorDict count axes so dynamo does not recompile per size.

    ``torch.compile(..., dynamic=True)`` unspecs sizes of *formal tensor
    inputs*. A nested TensorDict is a Python object; dynamo still installs a
    guard on ``atoms.Z.shape[0]`` (and the other namespace lengths) and
    recompiles until ``recompile_limit`` (Class-I GH200 profile 2362410:
    ``expected 172, actual 190``). Call this on the batch *eagerly*, before
    the compiled region — :func:`torch._dynamo.mark_dynamic` is forbidden
    inside a traced function.

    Count axis is dim 0 for atoms / edges / graphs / valence columns, and
    dim 1 for ``bond_index``-shaped COO ``(arity, N)``.

    Args:
        batch: A TensorDict / TensorFrame, or anything else (ignored).
    """
    items = getattr(batch, "items", None)
    if not callable(items):
        return
    try:
        pairs = items(include_nested=True, leaves_only=True)
    except TypeError:
        return
    for key, val in pairs:
        if not torch.is_tensor(val) or val.ndim < 1:
            continue
        name = key[-1] if isinstance(key, tuple) else key
        if name in _COO_COUNT_AXIS_1 and val.ndim >= 2:
            torch._dynamo.mark_dynamic(val, 1)
        else:
            torch._dynamo.mark_dynamic(val, 0)


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


def unwrap(module: torch.nn.Module) -> torch.nn.Module:
    """The module under every ``torch.compile`` / DDP / FSDP wrapper.

    ``torch.compile`` returns an ``OptimizedModule`` that holds the original
    module as ``_orig_mod`` (and prefixes every ``state_dict`` key with it);
    :class:`~torch.nn.parallel.DistributedDataParallel` and
    :class:`~torch.distributed.fsdp.FullyShardedDataParallel` hold theirs as
    ``module``. The wrappers nest in either order (``torch.compile(DDP(m))``
    is how the Trainer stacks them), so this peels by type until the module
    is plain. Every reader of a model's own attributes — the step prologue's
    derivative op, the joint step's protocol primitives, a checkpoint's
    ``state_dict`` — goes through here; a plain module is returned as is.

    Args:
        module: A module, possibly under compile / DDP / FSDP wrappers.

    Returns:
        The innermost plain module.
    """
    # Imported here, not at module scope: ``torch.distributed.fsdp`` costs a
    # third of a second to import and nothing else in molix needs it.
    from torch.distributed.fsdp import FullyShardedDataParallel

    while True:
        if isinstance(module, OptimizedModule):
            module = module._orig_mod
        elif isinstance(module, (DistributedDataParallel, FullyShardedDataParallel)):
            module = module.module
        else:
            return module


class Compiler:
    """Configure and apply ``torch.compile`` to a molnex module or callable.

    Construct with the desired config (or ``cuda_graphs=True`` for the
    force-training preset), then call the instance on the target to get the
    compiled wrapper. The call is generic in its argument, so a module stays a
    module and a bound method stays callable with the same signature::

        model = Compiler()(model)                  # default inductor
        model = Compiler(cuda_graphs=True)(model)  # CUDA-graph force preset
        model = Compiler(mode="max-autotune")(model)
        energy = Compiler(fullgraph=True)(pots.energy)  # bound method

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
        donated_buffer: bool = False,
    ) -> None:
        """Capture a ``torch.compile`` configuration.

        Args:
            cuda_graphs: Apply :data:`CUDA_GRAPH_PRESET`, overriding the four
                params below. REQUIRES static shapes (``MoleculePadder`` +
                ``drop_last=True``).
            backend: Compile backend (default ``"inductor"``).
            fullgraph: Require a single graph (error on graph breaks).
            dynamic: Enable dynamic-shape tracing.
            mode: Compile mode (``"default"`` / ``"reduce-overhead"`` /
                ``"max-autotune"``).
            donated_buffer: AOTAutograd buffer donation. ``True`` reuses dead
                saved tensors during a ``create_graph=False`` backward (MD
                forces / virial) and cuts peak HBM. ``False`` (default) is
                required for force-supervised ``create_graph=True`` training.
                Set on every compile (it is a process-wide AOTAutograd flag).
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

    @property
    def cuda_graphs(self) -> bool:
        """Whether this configuration wraps the compiled graph in CUDA graphs.

        True for the :data:`CUDA_GRAPH_PRESET` and for either inductor mode that
        records and replays the graph (``reduce-overhead``, ``max-autotune``).
        A caller that holds the compiled object across steps reads this to
        apply the CUDA-graph step contract — call
        ``torch.compiler.cudagraph_mark_step_begin()`` before each replay and
        clone any output it keeps, since replayed outputs live in a static
        memory pool the next replay overwrites — and skips both otherwise
        (:class:`molix.md.driver.MD` does exactly this).
        """
        return self.backend == "inductor" and self.mode in ("reduce-overhead", "max-autotune")

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
        # Trace ``torch.autograd.grad`` in-graph: the energy → force seam is one
        # graph instead of a break at the ``autograd.grad`` call. Process-wide;
        # dynamo reads it at trace time, which is the compiled wrapper's first
        # call, not here.
        torch._dynamo.config.trace_autograd_ops = True
        torch._functorch.config.donated_buffer = self.donated_buffer
        logger.info(
            f"Compiling {getattr(target, '__qualname__', type(target).__name__)} with "
            f"backend={self.backend}, fullgraph={self.fullgraph}, "
            f"dynamic={self.dynamic}, mode={self.mode}, "
            f"donated_buffer={self.donated_buffer}"
        )
        compiled = torch.compile(
            target,
            backend=self.backend,
            fullgraph=self.fullgraph,
            dynamic=self.dynamic,
            mode=self.mode,
        )
        # ``dynamic=True`` does not unspec TensorDict leaf sizes. An eager
        # wrapper marks count axes so dynamo does not specialize them.
        # Modules keep their compiled type (checkpoints / ``state_dict``);
        # the Trainer step path compiles a bound method, which is wrapped.
        if self.dynamic is True and not isinstance(target, torch.nn.Module):

            def wrapper(*args: object, **kwargs: object) -> object:
                for arg in args:
                    mark_dynamic_batch(arg)
                for arg in kwargs.values():
                    mark_dynamic_batch(arg)
                return compiled(*args, **kwargs)

            wrapper.__qualname__ = getattr(target, "__qualname__", "compiled")
            wrapper.__name__ = getattr(target, "__name__", "compiled")
            return cast(T, wrapper)
        return cast(T, compiled)

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
        configuration** plus the one flag :meth:`__call__` sets
        (``trace_autograd_ops=True``, patched for the duration of the trace so
        the measurement matches what a :class:`Compiler` compiles), not the
        calling instance's ``backend`` / ``mode`` / ``fullgraph``. To measure
        under a different backend, set it at the call site.
        ``torch._dynamo.explain`` resets dynamo itself, before and after the
        trace, so callers must not add their own ``reset()``.

        Args:
            target: Module, function or bound method to trace.
            *args: Positional arguments forwarded to ``target``.
            **kwargs: Keyword arguments forwarded to ``target``.

        Returns:
            The projected :class:`ExplainSummary`.
        """
        with torch._dynamo.config.patch(trace_autograd_ops=True):
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
