"""TypeSystem / TypeRecord — discrete condensed parameter prototypes.

Integer type ids + numeric prototypes only. No SMARTS / SMIRKS emission
(owned by sub-spec 07). Mutable helpers ``_spawn`` / ``_absorb`` are used by
:class:`~molrep.condensation.condenser.Condenser` during greedy merge; the
public table is otherwise treated as frozen after merge.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from molrep.condensation.classes import InteractionClass
from molrep.condensation.criterion import MergeCriterion, default_criterion

__all__ = ["TypeRecord", "TypeSystem", "UNMATCHED_TYPE_ID", "Prototype"]

# Soft-fail sentinel for :meth:`TypeSystem.assign` when no prototype accepts.
UNMATCHED_TYPE_ID: int = -1

Prototype = Mapping[str, float | tuple[float, ...] | torch.Tensor]


def _to_tensor(value: float | tuple[float, ...] | torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(dtype=torch.float64).clone()
    if isinstance(value, (tuple, list)):
        return torch.as_tensor(list(value), dtype=torch.float64)
    return torch.as_tensor(value, dtype=torch.float64)


def _to_python(value: torch.Tensor) -> float | tuple[float, ...]:
    t = value.detach().cpu().reshape(-1)
    if t.numel() == 1:
        return float(t.item())
    return tuple(float(x) for x in t.tolist())


def _prototype_tensors(
    prototype: Mapping[str, float | tuple[float, ...] | torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {k: _to_tensor(v) for k, v in prototype.items()}


def _prototype_python(
    prototype: Mapping[str, float | tuple[float, ...] | torch.Tensor],
) -> dict[str, float | tuple[float, ...]]:
    out: dict[str, float | tuple[float, ...]] = {}
    for k, v in prototype.items():
        if isinstance(v, torch.Tensor):
            out[k] = _to_python(v)
        elif isinstance(v, (tuple, list)):
            out[k] = tuple(float(x) for x in v)
        else:
            out[k] = float(v)
    return out


def _running_mean_tensors(
    prototype: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
    member_count: int,
) -> dict[str, torch.Tensor]:
    """Online mean: mean_new = mean_old + (x - mean_old) / (n+1)."""
    updated: dict[str, torch.Tensor] = {}
    for key, pval in prototype.items():
        if key not in candidate or pval.shape != candidate[key].shape:
            updated[key] = pval.clone()
            continue
        cval = candidate[key].to(dtype=pval.dtype)
        updated[key] = pval + (cval - pval) / float(member_count + 1)
    for key, cval in candidate.items():
        if key not in updated:
            updated[key] = cval.clone()
    return updated


@dataclass(frozen=True)
class TypeRecord:
    """One discrete chemical type and its parameter prototype.

    Attributes:
        type_id: Non-negative integer id within its :class:`TypeSystem`.
        prototype: Named parameter values in Class-I units
            (e.g. ``{"k": 300.0, "r0": 1.09}``). Tensor values are accepted
            at construction and stored as Python floats / tuples.
        member_count: How many interactions were merged into this type.
        label: Optional human-readable name.
        support_ids: Optional member indices into the condensation input stream.
    """

    type_id: int
    prototype: Mapping[str, float | tuple[float, ...]]
    member_count: int = 0
    label: str | None = None
    support_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.type_id < 0:
            raise ValueError(f"type_id must be non-negative, got {self.type_id}")
        if self.member_count < 0:
            raise ValueError(f"member_count must be non-negative, got {self.member_count}")
        object.__setattr__(self, "prototype", _prototype_python(self.prototype))
        object.__setattr__(self, "support_ids", tuple(self.support_ids))

    def prototype_tensors(self) -> dict[str, torch.Tensor]:
        """Prototype as float64 tensors (for criterion / assign)."""
        return _prototype_tensors(self.prototype)


class TypeSystem:
    """Ordered discrete type table for one :class:`InteractionClass`.

    Built by :class:`~molrep.condensation.condenser.Condenser`. At inference,
    :meth:`assign` maps new continuous parameters to an existing type when a
    :class:`MergeCriterion` accepts the prototype; otherwise returns
    :data:`UNMATCHED_TYPE_ID` (soft reject — does **not** spawn types).

    Args:
        interaction: Owning interaction family.
        records: Ordered type records (ids should be unique).
        criterion: Optional criterion used by :meth:`assign`. Defaults to
            the documented class default when omitted.
    """

    def __init__(
        self,
        interaction: InteractionClass,
        records: Sequence[TypeRecord] | None = None,
        *,
        criterion: MergeCriterion | None = None,
    ) -> None:
        self._interaction = interaction
        recs = list(records or ())
        seen: set[int] = set()
        for rec in recs:
            if rec.type_id in seen:
                raise ValueError(
                    f"duplicate type_id {rec.type_id} in TypeSystem for {interaction.value}"
                )
            seen.add(rec.type_id)
        self._records: list[TypeRecord] = recs
        self._by_id: dict[int, TypeRecord] = {r.type_id: r for r in recs}
        self._criterion = criterion if criterion is not None else default_criterion(interaction)

    @property
    def interaction(self) -> InteractionClass:
        """Interaction class this type table covers."""
        return self._interaction

    @property
    def n_types(self) -> int:
        """Number of discrete types."""
        return len(self._records)

    @property
    def criterion(self) -> MergeCriterion:
        """Merge criterion used for :meth:`assign`."""
        return self._criterion

    def records(self) -> list[TypeRecord]:
        """Return ordered type records (copy)."""
        return list(self._records)

    def get(self, type_id: int) -> TypeRecord:
        """Look up a type by id.

        Args:
            type_id: Discrete type identifier.

        Returns:
            The matching :class:`TypeRecord`.

        Raises:
            KeyError: If ``type_id`` is unknown.
        """
        try:
            return self._by_id[type_id]
        except KeyError as exc:
            raise KeyError(
                f"type_id {type_id} not in TypeSystem for {self._interaction.value}"
            ) from exc

    def prototypes_table(self) -> list[dict[str, Any]]:
        """Return prototypes as a list of plain dicts ordered by table order."""
        return [dict(r.prototype) for r in self._records]

    def prototypes_tensor_table(self) -> dict[str, torch.Tensor]:
        """Stack shared prototype keys into tensors ``(n_types, ...)``.

        Keys missing on any record are omitted.
        """
        if not self._records:
            return {}
        keys = set(self._records[0].prototype)
        for rec in self._records[1:]:
            keys &= set(rec.prototype)
        table: dict[str, torch.Tensor] = {}
        for key in sorted(keys):
            table[key] = torch.stack(
                [_to_tensor(rec.prototype[key]) for rec in self._records],
                dim=0,
            )
        return table

    def assign(
        self,
        params: Mapping[str, torch.Tensor | float | tuple[float, ...]],
        *,
        criterion: MergeCriterion | None = None,
    ) -> int:
        """Map params to the first acceptable existing type id.

        Soft-fail contract: if no prototype accepts ``params``, return
        :data:`UNMATCHED_TYPE_ID` (``-1``). Never creates a new type.

        Args:
            params: Continuous parameters for one interaction row.
            criterion: Optional override; defaults to the system criterion.

        Returns:
            Existing ``type_id`` or :data:`UNMATCHED_TYPE_ID`.
        """
        crit = criterion if criterion is not None else self._criterion
        row = _prototype_tensors(params)
        for rec in self._records:
            if crit.accepts(rec.prototype_tensors(), row):
                return rec.type_id
        return UNMATCHED_TYPE_ID

    def assign_many(
        self,
        params: Mapping[str, torch.Tensor],
        *,
        criterion: MergeCriterion | None = None,
    ) -> torch.Tensor:
        """Vectorized :meth:`assign` over a batch of parameter rows.

        Args:
            params: Mapping of tensors with leading batch dim ``(N, ...)``.
            criterion: Optional criterion override.

        Returns:
            Long tensor ``(N,)`` of type ids (``-1`` for unmatched).
        """
        if not params:
            return torch.zeros(0, dtype=torch.long)
        first = next(iter(params.values()))
        n = int(first.shape[0])
        out = torch.full((n,), UNMATCHED_TYPE_ID, dtype=torch.long)
        for i in range(n):
            row = {k: v[i] for k, v in params.items()}
            out[i] = self.assign(row, criterion=criterion)
        return out

    def __iter__(self) -> Iterator[TypeRecord]:
        return iter(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return f"TypeSystem(interaction={self._interaction!r}, n_types={self.n_types})"

    @classmethod
    def from_prototypes(
        cls,
        interaction: InteractionClass,
        prototypes: Iterable[Prototype],
        *,
        labels: Sequence[str | None] | None = None,
        criterion: MergeCriterion | None = None,
    ) -> TypeSystem:
        """Build a type system with sequential type ids from prototypes.

        Args:
            interaction: Interaction class.
            prototypes: One prototype mapping per type (order defines type_id).
            labels: Optional labels parallel to ``prototypes``.
            criterion: Optional assign criterion.

        Returns:
            A new :class:`TypeSystem`.
        """
        protos = list(prototypes)
        labs = list(labels) if labels is not None else [None] * len(protos)
        if len(labs) != len(protos):
            raise ValueError("labels length must match prototypes length")
        records = [
            TypeRecord(type_id=i, prototype=p, member_count=0, label=lab)
            for i, (p, lab) in enumerate(zip(protos, labs, strict=True))
        ]
        return cls(interaction, records, criterion=criterion)

    # --- mutation used by Condenser (package-internal) ---------------------

    def _spawn(
        self,
        params: Mapping[str, torch.Tensor | float | tuple[float, ...]],
        *,
        support_id: int | None = None,
        label: str | None = None,
    ) -> int:
        """Append a new type from ``params``; return its type_id."""
        type_id = self.n_types
        support = (support_id,) if support_id is not None else ()
        rec = TypeRecord(
            type_id=type_id,
            prototype=_prototype_python(params),
            member_count=1,
            label=label,
            support_ids=support,
        )
        self._records.append(rec)
        self._by_id[type_id] = rec
        return type_id

    def _absorb(
        self,
        type_id: int,
        params: Mapping[str, torch.Tensor | float | tuple[float, ...]],
        *,
        support_id: int | None = None,
        update_centroid: bool = True,
    ) -> None:
        """Merge ``params`` into an existing type, updating the centroid."""
        rec = self._by_id[type_id]
        proto_t = rec.prototype_tensors()
        cand_t = _prototype_tensors(params)
        if update_centroid:
            new_proto = _running_mean_tensors(proto_t, cand_t, rec.member_count)
        else:
            new_proto = proto_t
        support = rec.support_ids + ((support_id,) if support_id is not None else ())
        updated = TypeRecord(
            type_id=rec.type_id,
            prototype=_prototype_python(new_proto),
            member_count=rec.member_count + 1,
            label=rec.label,
            support_ids=support,
        )
        # Replace in ordered list
        for i, existing in enumerate(self._records):
            if existing.type_id == type_id:
                self._records[i] = updated
                break
        self._by_id[type_id] = updated
