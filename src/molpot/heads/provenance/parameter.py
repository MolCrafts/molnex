"""Frozen parameter provenance records for Class-I FF audit trails.

Reference:
    Spec: learnable-classical-ff-09-provenance
"""

from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from molpot.heads.provenance.regime import CoverageRegime
from molrep.condensation.classes import InteractionClass

__all__ = ["ParameterProvenance", "attach_provenance"]


@dataclass(frozen=True)
class ParameterProvenance:
    """Immutable audit record for one predicted / exported parameter row.

    Construct with :class:`ParameterProvenance` directly (no factory). Keep
    lists of records **alongside** IR / ForceSpec payloads — do not fold into
    a mega context bag.

    Args:
        interaction: Interaction class name (e.g. ``"bond"``) or
            :class:`~molrep.condensation.classes.InteractionClass`.
        type_id: Discrete condensed type id, if any.
        confidence: Softmax-max confidence from
            :meth:`~molrep.heads.type.TypeHead.decode_with_confidence`, if any.
        regime: Chemical-space coverage regime.
        source: Provenance tag — e.g. ``"neural_continuous"``,
            ``"condensed_type"``, ``"symbolic"``.
        pattern: Optional SMARTS / pattern id when known.
        ir_units: Unit system tag (default Class-I canonical).
        notes: Free-form short note.
    """

    interaction: str | InteractionClass
    type_id: int | None
    confidence: float | None
    regime: CoverageRegime
    source: str
    pattern: str | None = None
    ir_units: str = "class_i_canonical"
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly dict (enums → stable string names)."""
        d = asdict(self)
        inter = self.interaction
        d["interaction"] = inter.value if isinstance(inter, InteractionClass) else str(inter)
        d["regime"] = self.regime.name
        return d


def attach_provenance(
    target: MutableMapping[str, Any] | Any,
    records: Sequence[ParameterProvenance],
    *,
    key: str = "provenance",
) -> Any:
    """Attach provenance records as metadata **alongside** a payload.

    Does not invent a god context object: writes a JSON-friendly list under
    ``key`` on a mapping, or on an object that exposes a ``metadata`` dict
    (e.g. :class:`~molix.ff_export.force_spec.ForceSpec`).

    Args:
        target: Mutable mapping, or object with ``.metadata`` mapping, or a
            mapping-like ForceSpec-style record.
        records: Provenance rows to attach.
        key: Metadata key (default ``"provenance"``).

    Returns:
        The same ``target`` (mutated) for call chaining.
    """
    payload = [r.as_dict() for r in records]
    if isinstance(target, MutableMapping):
        target[key] = payload
        return target
    meta = getattr(target, "metadata", None)
    if isinstance(meta, MutableMapping):
        meta[key] = payload
        return target
    raise TypeError(
        "attach_provenance requires a mutable mapping or an object "
        f"with mutable .metadata; got {type(target)!r}"
    )
