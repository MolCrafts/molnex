"""ClassPatternRegistry — bind discrete types to SymbolicPattern."""

from __future__ import annotations

from collections.abc import Iterator

from molrep.condensation.classes import InteractionClass
from molrep.perception.patterns import SymbolicPattern

__all__ = ["ClassPatternRegistry"]

_Key = tuple[InteractionClass, int]


class ClassPatternRegistry:
    """Map ``(InteractionClass, type_id) → SymbolicPattern`` with reverse lookup.

    Binding is idempotent for the exact same pattern. Binding a *different*
    pattern to an already-bound key, or reusing a pattern string for a different
    key, raises ``ValueError``.

    Examples:
        >>> from molrep.condensation import InteractionClass
        >>> from molrep.perception import ClassPatternRegistry, SymbolicPattern
        >>> reg = ClassPatternRegistry()
        >>> reg.bind(InteractionClass.BOND, 0, SymbolicPattern("[#6]-[#6]", 2))
        >>> reg.get(InteractionClass.BOND, 0).pattern
        '[#6]-[#6]'
    """

    def __init__(self) -> None:
        self._by_key: dict[_Key, SymbolicPattern] = {}
        self._by_pattern: dict[str, _Key] = {}

    def bind(
        self,
        interaction: InteractionClass,
        type_id: int,
        pattern: SymbolicPattern,
    ) -> None:
        """Bind a pattern to ``(interaction, type_id)``.

        Args:
            interaction: Interaction class.
            type_id: Discrete type id.
            pattern: Symbolic pattern to store.

        Raises:
            ValueError: Conflicting bind (different pattern for same key, or
                same pattern string already bound to another key); negative
                ``type_id``.
            TypeError: ``pattern`` is not a :class:`SymbolicPattern`.
        """
        if type_id < 0:
            raise ValueError(f"type_id must be non-negative, got {type_id}")
        if not isinstance(pattern, SymbolicPattern):
            raise TypeError(f"pattern must be SymbolicPattern, got {type(pattern)!r}")

        key: _Key = (interaction, type_id)
        existing = self._by_key.get(key)
        if existing is not None:
            if existing == pattern:
                return  # idempotent
            raise ValueError(
                f"conflicting bind for ({interaction.value}, type_id={type_id}): "
                f"already bound to {existing.pattern!r}, "
                f"refusing {pattern.pattern!r}"
            )

        owner = self._by_pattern.get(pattern.pattern)
        if owner is not None and owner != key:
            oi, ot = owner
            raise ValueError(
                f"pattern {pattern.pattern!r} already bound to "
                f"({oi.value}, type_id={ot}); cannot rebind to "
                f"({interaction.value}, type_id={type_id})"
            )

        self._by_key[key] = pattern
        self._by_pattern[pattern.pattern] = key

    def get(self, interaction: InteractionClass, type_id: int) -> SymbolicPattern:
        """Return the bound pattern.

        Args:
            interaction: Interaction class.
            type_id: Discrete type id.

        Returns:
            Bound :class:`SymbolicPattern`.

        Raises:
            KeyError: No pattern bound for this key.
        """
        key: _Key = (interaction, type_id)
        try:
            return self._by_key[key]
        except KeyError as exc:
            raise KeyError(
                f"no pattern bound for ({interaction.value}, type_id={type_id})"
            ) from exc

    def get_optional(self, interaction: InteractionClass, type_id: int) -> SymbolicPattern | None:
        """Return bound pattern or ``None`` if unbound."""
        return self._by_key.get((interaction, type_id))

    def reverse_lookup(self, pattern: str | SymbolicPattern) -> tuple[InteractionClass, int]:
        """Look up ``(interaction, type_id)`` for a pattern string.

        Args:
            pattern: Pattern string or :class:`SymbolicPattern`.

        Returns:
            Bound ``(InteractionClass, type_id)``.

        Raises:
            KeyError: Pattern string is not registered.
        """
        key_str = pattern.pattern if isinstance(pattern, SymbolicPattern) else pattern
        try:
            return self._by_pattern[key_str]
        except KeyError as exc:
            raise KeyError(f"no type bound for pattern {key_str!r}") from exc

    def items(self) -> Iterator[tuple[_Key, SymbolicPattern]]:
        """Iterate ``((interaction, type_id), pattern)`` pairs."""
        return iter(self._by_key.items())

    def __contains__(self, key: object) -> bool:
        if (
            isinstance(key, tuple)
            and len(key) == 2
            and isinstance(key[0], InteractionClass)
            and isinstance(key[1], int)
        ):
            return key in self._by_key
        if isinstance(key, str):
            return key in self._by_pattern
        if isinstance(key, SymbolicPattern):
            return key.pattern in self._by_pattern
        return False

    def __len__(self) -> int:
        return len(self._by_key)

    def __repr__(self) -> str:
        return f"ClassPatternRegistry(n_bindings={len(self._by_key)})"
