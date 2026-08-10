"""DiscreteClassRecord — condensed type + optional symbolic pattern binding."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from molrep.condensation.classes import InteractionClass

__all__ = ["DiscreteClassRecord"]


@dataclass(frozen=True)
class DiscreteClassRecord:
    """One discrete chemical class with optional SMARTS/SMIRKS text.

    Produced by :meth:`SymbolicForceField.records` for human inspection and
    export. ``smarts`` / ``smirks`` are ``None`` until a pattern is bound in
    the registry.

    Attributes:
        interaction: Interaction class this type belongs to.
        type_id: Discrete id within the interaction's :class:`TypeSystem`.
        prototype: Named parameter prototype values.
        smarts: Optional SMARTS pattern string when bound.
        smirks: Optional SMIRKS string for parameter-bearing transforms.
        label: Optional human-readable type label.
    """

    interaction: InteractionClass
    type_id: int
    prototype: Mapping[str, float | tuple[float, ...]]
    smarts: str | None = None
    smirks: str | None = None
    label: str | None = None

    def __post_init__(self) -> None:
        if self.type_id < 0:
            raise ValueError(f"type_id must be non-negative, got {self.type_id}")
        object.__setattr__(self, "prototype", dict(self.prototype))
