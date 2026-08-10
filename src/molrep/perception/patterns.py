"""SymbolicPattern — validated SMARTS/SMIRKS string + arity."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

__all__ = ["SymbolicPattern"]

_VALID_ARITIES = frozenset({1, 2, 3, 4})
PatternKind = Literal["smarts", "smirks"]


@dataclass(frozen=True)
class SymbolicPattern:
    """A symbolic chemical pattern with fixed interaction arity.

    Args:
        pattern: Non-empty SMARTS or SMIRKS string.
        arity: Number of atoms in a hit (1 atom, 2 bond, 3 angle, 4 torsion).
            Must be in ``{1, 2, 3, 4}``.
        atom_maps: Optional documentation of Daylight atom-map labels
            (query atom index → map name). Not used for matching constraints.
        kind: Whether ``pattern`` is SMARTS or SMIRKS. Default ``"smarts"``.

    Raises:
        ValueError: Empty pattern string or arity outside ``{1,2,3,4}``.
    """

    pattern: str
    arity: int
    atom_maps: Mapping[int, str] | None = None
    kind: PatternKind = "smarts"

    def __post_init__(self) -> None:
        if not isinstance(self.pattern, str) or not self.pattern.strip():
            raise ValueError(
                f"SymbolicPattern.pattern must be a non-empty string; got {self.pattern!r}"
            )
        if self.arity not in _VALID_ARITIES:
            raise ValueError(f"SymbolicPattern.arity must be in {{1,2,3,4}}; got {self.arity}")
        if self.kind not in ("smarts", "smirks"):
            raise ValueError(
                f"SymbolicPattern.kind must be 'smarts' or 'smirks'; got {self.kind!r}"
            )
        if self.atom_maps is not None:
            object.__setattr__(self, "atom_maps", dict(self.atom_maps))
