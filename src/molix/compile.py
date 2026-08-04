"""``torch.compile`` integration for MolNex.

:class:`Compiler` configures ``torch.compile`` once and applies it to a module.
Two regimes (see ``docs/molix/explanation/throughput-and-compilation.md``):

* **ragged / energy-only / general** — default inductor; dynamic shapes fine.
* **force training on PADDED fixed shapes** — ``Compiler(cuda_graphs=True)``
  applies the benchmarked CUDA-graph preset (``reduce-overhead``): inductor
  fusion + CUDA graphs reach ~10x eager on PiNet (measured ~130 steps/s on a
  GH200). REQUIRES static shapes — register
  :class:`molix.data.tasks.PadMolecularBatch` and set the train loader's
  ``drop_last=True``. Without static shapes inductor recompiles every step and
  ``reduce-overhead`` is *slower* than eager.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from molix import logger as _logger_mod

logger = _logger_mod.getLogger(__name__)


class Compiler:
    """Configure and apply ``torch.compile`` to a molnex module.

    Construct with the desired config (or ``cuda_graphs=True`` for the
    force-training preset), then call the instance on a module to get the
    compiled wrapper::

        model = Compiler()(model)                  # default inductor
        model = Compiler(cuda_graphs=True)(model)  # CUDA-graph force preset
        model = Compiler(mode="max-autotune")(model)
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
            backend = p["backend"]  # type: ignore[assignment]
            fullgraph = p["fullgraph"]  # type: ignore[assignment]
            dynamic = p["dynamic"]  # type: ignore[assignment]
            mode = p["mode"]  # type: ignore[assignment]
        self.backend = backend
        self.fullgraph = fullgraph
        self.dynamic = dynamic
        self.mode = mode

    def __call__(self, module: nn.Module) -> nn.Module:
        """Return ``module`` compiled with this instance's configuration."""
        logger.info(
            f"Compiling {module.__class__.__name__} with backend={self.backend}, "
            f"fullgraph={self.fullgraph}, dynamic={self.dynamic}, mode={self.mode}"
        )
        return torch.compile(
            module,
            backend=self.backend,
            fullgraph=self.fullgraph,
            dynamic=self.dynamic,
            mode=self.mode,
        )

    @staticmethod
    def count_graph_breaks(module: nn.Module, *args: object, **kwargs: object) -> int:
        """Count graph breaks for ``module(*args, **kwargs)`` via ``torch._dynamo.explain``."""
        return torch._dynamo.explain(module)(*args, **kwargs).graph_break_count
