"""Activation lookup by name: the one table every config-driven layer uses."""

from __future__ import annotations

import torch.nn as nn

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "swish": nn.SiLU,
    "tanh": nn.Tanh,
}


def activation_from_name(name: str | type[nn.Module] | None) -> type[nn.Module] | None:
    """Resolve an activation name or module class to a zero-argument module type.

    Args:
        name: ``"gelu"`` / ``"relu"`` / ``"silu"`` (alias ``"swish"``) /
            ``"tanh"`` (case-insensitive), an ``nn.Module`` subclass (returned
            as is), or ``None`` (no activation).

    Returns:
        The module class, or ``None``.

    Raises:
        ValueError: If ``name`` is an unknown string.
    """
    if name is None:
        return None
    if isinstance(name, type) and issubclass(name, nn.Module):
        return name
    try:
        return _ACTIVATIONS[str(name).lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported activation {name!r}; known: {sorted(_ACTIVATIONS)}") from exc
