"""torch.compile integration for MolNex.

Provides a toggleable compile wrapper and graph-break counting utility.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from molix import logger as _logger_mod

logger = _logger_mod.getLogger(__name__)


def maybe_compile(
    module: nn.Module,
    *,
    compile: bool = False,
    backend: str = "inductor",
    fullgraph: bool = False,
    dynamic: bool | None = None,
    mode: str | None = None,
) -> nn.Module:
    """Optionally compile a module with torch.compile.

    Args:
        module: The PyTorch module.
        compile: If False, return module unchanged.
        backend: Compile backend (default: ``"inductor"``).
        fullgraph: If True, require single graph (error on graph breaks).
        dynamic: Enable dynamic shape tracing.
        mode: Compile mode (``"default"``, ``"reduce-overhead"``,
            ``"max-autotune"``).

    Returns:
        The original or compiled module.
    """
    if not compile:
        return module
    logger.info(
        "Compiling module %s with backend=%s, fullgraph=%s, mode=%s",
        module.__class__.__name__,
        backend,
        fullgraph,
        mode,
    )
    return torch.compile(
        module,
        backend=backend,
        fullgraph=fullgraph,
        dynamic=dynamic,
        mode=mode,
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
