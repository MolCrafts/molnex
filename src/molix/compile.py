"""torch.compile integration for MolNex.

Provides a toggleable compile wrapper and graph-break counting utility.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn

from molix import logger as _logger_mod

logger = _logger_mod.getLogger(__name__)

# The benchmarked winning configuration for force training (energy → functorch
# force → loss as one fullgraph). Inductor fusion **and** CUDA graphs together
# on padded, fixed-length shapes reach ~10x eager / ~4-5x plain compile on
# PiNet (measured ~130 steps/s on a GH200; see
# ``docs/molix/explanation/throughput-and-compilation.md``).
#
# PRECONDITION — static shapes. CUDA graphs replay one recorded launch
# sequence, so every step must see identical tensor shapes. Register
# :class:`molix.data.tasks.PadMolecularBatch` (pads atoms/edges to a fixed cap
# with a masked ghost region that is numerically inert) and set the train
# loader's ``drop_last=True`` (fixes the graph count). Without static shapes
# inductor recompiles every step and ``reduce-overhead`` is *slower* than eager.
CUDA_GRAPH_PRESET: dict[str, object] = {
    "backend": "inductor",
    "fullgraph": True,
    "dynamic": False,
    "mode": "reduce-overhead",
}


def maybe_compile(
    module: nn.Module,
    *,
    compile: bool = False,
    cuda_graphs: bool = False,
    backend: str = "inductor",
    fullgraph: bool = False,
    dynamic: bool | None = None,
    mode: str | None = None,
) -> nn.Module:
    """Optionally compile a module with torch.compile.

    Args:
        module: The PyTorch module.
        compile: If False, return module unchanged.
        cuda_graphs: Apply :data:`CUDA_GRAPH_PRESET` — the benchmarked winning
            force-training config (``backend="inductor", fullgraph=True,
            dynamic=False, mode="reduce-overhead"``). Overrides ``backend`` /
            ``fullgraph`` / ``dynamic`` / ``mode``. REQUIRES static shapes
            (``PadMolecularBatch`` + ``drop_last=True``); see the module
            docstring.
        backend: Compile backend (default: ``"inductor"``). Ignored when
            ``cuda_graphs`` is set.
        fullgraph: If True, require single graph (error on graph breaks).
            Ignored when ``cuda_graphs`` is set.
        dynamic: Enable dynamic shape tracing. Ignored when ``cuda_graphs`` is
            set.
        mode: Compile mode (``"default"``, ``"reduce-overhead"``,
            ``"max-autotune"``). Ignored when ``cuda_graphs`` is set.

    Returns:
        The original or compiled module.
    """
    if not compile:
        return module
    if cuda_graphs:
        backend = cast(str, CUDA_GRAPH_PRESET["backend"])
        fullgraph = cast(bool, CUDA_GRAPH_PRESET["fullgraph"])
        dynamic = cast("bool | None", CUDA_GRAPH_PRESET["dynamic"])
        mode = cast("str | None", CUDA_GRAPH_PRESET["mode"])
    logger.info(
        f"Compiling module {module.__class__.__name__} with "
        f"backend={backend}, fullgraph={fullgraph}, dynamic={dynamic}, mode={mode}"
    )
    return cast(
        nn.Module,
        torch.compile(
            module,
            backend=backend,
            fullgraph=fullgraph,
            dynamic=dynamic,
            mode=mode,
        ),
    )


def count_graph_breaks(
    module: nn.Module,
    *args,
    **kwargs,
) -> int:
    """Count graph breaks when compiling a module.

    Uses ``torch._dynamo.explain()`` to analyze graph breaks without
    actually compiling for execution.

    Args:
        module: Module to analyze.
        *args: Example forward arguments.
        **kwargs: Example forward keyword arguments.

    Returns:
        Number of graph breaks detected.
    """
    explanation = torch._dynamo.explain(module)(*args, **kwargs)
    return explanation.graph_break_count
