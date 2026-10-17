"""Which molpot potential implements which molrs style.

:func:`register` files a potential class under its ``style``
(``"bond:harmonic"``); :func:`resolve` looks one up. Building the potentials
of a whole force field is ``Potentials.from_forcefield`` — a potential depends
on the force field, so the constructor lives on the collection. There is no
second description of a force field on the molnex side: what is priced is what
the ``ForceField`` says.
"""

from __future__ import annotations

from typing import TypeVar

import torch.nn as nn

__all__ = ["register", "registered", "resolve"]

_REGISTRY: dict[str, type[nn.Module]] = {}

P = TypeVar("P", bound=type[nn.Module])


def register(cls: P) -> P:
    """File a potential class under its ``style``; a second claim on a style raises."""
    style = cls.style  # type: ignore[attr-defined]
    owner = _REGISTRY.get(style)
    if owner is not None and owner is not cls:
        raise ValueError(f"style {style!r} is already implemented by {owner.__qualname__}")
    _REGISTRY[style] = cls
    return cls


def registered() -> tuple[str, ...]:
    """The styles a potential is registered for, sorted."""
    return tuple(sorted(_REGISTRY))


def resolve(style: str) -> type[nn.Module] | None:
    """The potential class registered for ``style`` (``"bond:harmonic"``), or ``None``."""
    return _REGISTRY.get(style)
