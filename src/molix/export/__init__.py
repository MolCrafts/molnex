"""Model packaging, one module per format.

:mod:`molix.export.aoti` — AOTInductor ``.pt2`` (``package`` / ``load`` /
``custom_op_namespaces``). A format module packages any flat tensor-IO
``nn.Module``; it knows nothing of what the module computes or which engine
will run it.
"""

from __future__ import annotations

from molix.export import aoti

__all__ = ["aoti"]
