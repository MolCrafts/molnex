"""Backend-neutral force specification (JSON-serializable).

:class:`ForceSpec` is the portable product of a :class:`BackendAdapter`
translation. It is **not** a live OpenMM System — consumers may materialize
one later; unit tests pin hard-coded dict goldens only.

References:
    OpenMM User Guide §19 "Forces"
    Spec: learnable-classical-ff-08-ff-export
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = ["ForceSpec"]


@dataclass
class ForceSpec:
    """Serializable force-field specification for one backend.

    Attributes:
        backend: Adapter name (e.g. ``"openmm"``).
        forces: List of force-group records (dicts with at least ``"type"``).
        scaling: Optional nonbonded scale factors (1–2 / 1–3 / 1–4).
        metadata: Free-form provenance / type-system attachments.
    """

    backend: str
    forces: list[dict[str, Any]] = field(default_factory=list)
    scaling: dict[str, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly nested dict (deep copy of fields)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ForceSpec:
        """Alternate constructor: rebuild from :meth:`to_dict` output.

        Args:
            data: Mapping with keys ``backend``, ``forces``, optional
                ``scaling`` and ``metadata``.

        Returns:
            A new :class:`ForceSpec`.
        """
        return cls(
            backend=data["backend"],
            forces=list(data.get("forces") or []),
            scaling=data.get("scaling"),
            metadata=dict(data.get("metadata") or {}),
        )
