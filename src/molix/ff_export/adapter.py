"""BackendAdapter protocol + self-registering peer types.

Mirrors :class:`molix.engine.EngineAdapter`: each backend is a named
peer class (``OpenMMAdapter``, future ``GromacsAdapter``), **not** a
``Translator(method="openmm")`` switch.

References:
    Spec: learnable-classical-ff-08-ff-export
    Placement: ``.claude/notes/learnable-classical-ff.md`` (no method=)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from molix.ff_export.force_spec import ForceSpec
from molpot.ir import PotentialIR

__all__ = ["BackendAdapter"]


class BackendAdapter(ABC):
    """Strategy that translates :class:`~molpot.ir.PotentialIR` → :class:`ForceSpec`.

    Concrete subclasses set a class-level :attr:`name` (auto-registered) and
    implement :meth:`translate`. Resolve by name via :meth:`from_name`.
    """

    name: str = ""
    _registry: dict[str, type[BackendAdapter]] = {}

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if cls.name:
            BackendAdapter._registry[cls.name] = cls

    @classmethod
    def from_name(cls, name: str) -> BackendAdapter:
        """Instantiate the registered adapter for ``name`` (e.g. ``"openmm"``)."""
        try:
            return cls._registry[name]()
        except KeyError:
            valid = ", ".join(sorted(cls._registry)) or "(none)"
            raise ValueError(f"unknown adapter {name!r}; valid adapters: {valid}") from None

    @classmethod
    def names(cls) -> tuple[str, ...]:
        """All registered adapter names."""
        return tuple(sorted(cls._registry))

    @abstractmethod
    def translate(
        self,
        ir: PotentialIR,
        *,
        meta: dict[str, Any] | None = None,
    ) -> ForceSpec:
        """Translate ``ir`` into a backend force specification.

        Args:
            ir: Class-I parameter bags (+ optional scaling).
            meta: Optional attachments (type systems, symbolic FF, …).

        Returns:
            A serializable :class:`ForceSpec`.

        Raises:
            UnsupportedTermError: When an IR bag has no faithful mapping.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
