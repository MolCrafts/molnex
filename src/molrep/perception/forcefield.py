"""SymbolicForceField — discrete classes + patterns + prototypes (no energy)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from molrep.condensation.classes import InteractionClass
from molrep.condensation.type_system import TypeSystem
from molrep.perception.matcher import SmartsMatcher
from molrep.perception.records import DiscreteClassRecord
from molrep.perception.registry import ClassPatternRegistry

__all__ = ["SymbolicForceField"]


class SymbolicForceField:
    """Discrete classes + patterns + prototypes for human/export consumption.

    Pairs condensed :class:`TypeSystem` tables with a
    :class:`ClassPatternRegistry`. Matching is pure perception — this module
    never evaluates energy or imports molpot kernels.

    Args:
        type_systems: Map of interaction class → condensed type table.
        registry: Pattern bindings for discrete type ids.

    Spec:
        learnable-classical-ff-07-smarts
    """

    def __init__(
        self,
        type_systems: Mapping[InteractionClass, TypeSystem],
        registry: ClassPatternRegistry,
    ) -> None:
        if not isinstance(registry, ClassPatternRegistry):
            raise TypeError(f"registry must be ClassPatternRegistry, got {type(registry)!r}")
        systems: dict[InteractionClass, TypeSystem] = {}
        for interaction, ts in type_systems.items():
            if not isinstance(interaction, InteractionClass):
                raise TypeError(
                    f"type_systems keys must be InteractionClass, got {type(interaction)!r}"
                )
            if not isinstance(ts, TypeSystem):
                raise TypeError(f"type_systems values must be TypeSystem, got {type(ts)!r}")
            if ts.interaction != interaction:
                raise ValueError(
                    f"TypeSystem.interaction is {ts.interaction.value!r} but "
                    f"mapped under {interaction.value!r}"
                )
            systems[interaction] = ts
        self._type_systems = systems
        self._registry = registry

    @property
    def type_systems(self) -> Mapping[InteractionClass, TypeSystem]:
        """Frozen view of interaction → type system."""
        return self._type_systems

    @property
    def registry(self) -> ClassPatternRegistry:
        """Pattern registry."""
        return self._registry

    def records(self) -> list[DiscreteClassRecord]:
        """Assemble discrete class records with bound SMARTS/SMIRKS when present.

        Returns:
            One :class:`DiscreteClassRecord` per type across all type systems,
            ordered by interaction enum order then table order. Unbound types
            have ``smarts=None`` and ``smirks=None``.
        """
        out: list[DiscreteClassRecord] = []
        for interaction in InteractionClass:
            ts = self._type_systems.get(interaction)
            if ts is None:
                continue
            for rec in ts.records():
                pat = self._registry.get_optional(interaction, rec.type_id)
                smarts: str | None = None
                smirks: str | None = None
                if pat is not None:
                    if pat.kind == "smirks":
                        smirks = pat.pattern
                    else:
                        smarts = pat.pattern
                out.append(
                    DiscreteClassRecord(
                        interaction=interaction,
                        type_id=rec.type_id,
                        prototype=dict(rec.prototype),
                        smarts=smarts,
                        smirks=smirks,
                        label=rec.label,
                    )
                )
        return out

    def match_molecule(
        self,
        mol: Any,
        matcher: SmartsMatcher,
    ) -> dict[InteractionClass, dict[str, torch.Tensor]]:
        """Assign type ids by matching bound patterns against ``mol``.

        For each bound ``(interaction, type_id)`` pattern, run ``matcher`` and
        collect hits. **No energy evaluation.**

        Args:
            mol: Molecule graph accepted by ``matcher``.
            matcher: Object implementing :class:`SmartsMatcher`.

        Returns:
            Mapping ``interaction → {"matches": LongTensor [arity, K],
            "type_ids": LongTensor [K]}`` for interactions with at least one
            hit. Interactions with zero hits are omitted.
        """
        buckets: dict[InteractionClass, dict[str, list[torch.Tensor]]] = {}

        for (interaction, type_id), pattern in self._registry.items():
            hits = matcher.match(mol, pattern)
            if hits.ndim != 2:
                raise ValueError(
                    f"matcher returned shape {tuple(hits.shape)}; expected [arity, n_hits]"
                )
            if hits.shape[1] == 0:
                continue
            type_ids = torch.full((hits.shape[1],), int(type_id), dtype=torch.long)
            bucket = buckets.setdefault(interaction, {"matches": [], "type_ids": []})
            bucket["matches"].append(hits.long())
            bucket["type_ids"].append(type_ids)

        result: dict[InteractionClass, dict[str, torch.Tensor]] = {}
        for interaction, bucket in buckets.items():
            # Patterns for the same interaction may have different arities
            # (e.g. atom LJ vs bond); group by leading dim.
            by_arity: dict[int, list[torch.Tensor]] = {}
            type_by_arity: dict[int, list[torch.Tensor]] = {}
            for m, t in zip(bucket["matches"], bucket["type_ids"], strict=True):
                a = int(m.shape[0])
                by_arity.setdefault(a, []).append(m)
                type_by_arity.setdefault(a, []).append(t)

            if len(by_arity) == 1:
                arity = next(iter(by_arity))
                result[interaction] = {
                    "matches": torch.cat(by_arity[arity], dim=1),
                    "type_ids": torch.cat(type_by_arity[arity], dim=0),
                }
            else:
                # Mixed arities under one InteractionClass — return a flat
                # concatenation only if all share arity; otherwise keep the
                # dominant path documented as single-arity per class.
                # Spec assumes one arity per InteractionClass (bond=2, …).
                raise ValueError(
                    f"mixed arities under {interaction.value}: "
                    f"{sorted(by_arity)}; bind patterns of one arity per class"
                )
        return result
