"""Internal utilities shared between concrete hook modules.

Kept private (single leading underscore) to make the
:mod:`molix.hooks` public surface narrow — only the concrete hook
classes are re-exported by ``__init__``.
"""

from __future__ import annotations

from typing import Any


def _as_scalar(value: Any) -> float | int | None:
    """Coerce ``value`` to a Python scalar or return ``None`` if non-scalar.

    Accepts Python numbers and 0-d tensors / numpy scalars. Anything else
    (vectors, matrices, strings, enums, ``None``) returns ``None`` so callers
    can skip silently instead of raising.
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (RuntimeError, ValueError):
            return None
    return None
