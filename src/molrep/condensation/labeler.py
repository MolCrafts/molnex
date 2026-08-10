"""TypeSystemLabeler — Labeler Protocol for condensed discrete types."""

from __future__ import annotations

from typing import Mapping

import torch

from molrep.condensation.criterion import MergeCriterion
from molrep.condensation.type_system import UNMATCHED_TYPE_ID, TypeSystem

__all__ = ["TypeSystemLabeler"]


class TypeSystemLabeler:
    """Map continuous parameter rows to condensed type ids.

    Implements the :class:`~molrep.heads.labeler.Labeler` surface
    (``num_types``, ``type_map``, ``label(...)``) so callers can treat
    condensed ids like atom-type labels. ``label`` accepts a parameter
    mapping (preferred) or a tensor of precomputed type ids; it does
    **not** emit SMARTS text.

    Args:
        type_system: Frozen (or post-merge) discrete type table.
        criterion: Optional criterion override for nearest-prototype assign.
        type_map: Optional human-readable names; defaults to
            ``{i: "<interaction>_<i>"}``.
    """

    def __init__(
        self,
        type_system: TypeSystem,
        *,
        criterion: MergeCriterion | None = None,
        type_map: dict[int, str] | None = None,
    ) -> None:
        self._type_system = type_system
        self._criterion = criterion
        if type_map is None:
            prefix = type_system.interaction.value
            type_map = {i: f"{prefix}_{i}" for i in range(type_system.n_types)}
        self._type_map = dict(type_map)

    @property
    def type_system(self) -> TypeSystem:
        """Underlying condensed type table."""
        return self._type_system

    @property
    def num_types(self) -> int:
        """Number of discrete types (Labeler Protocol)."""
        return self._type_system.n_types

    @property
    def type_map(self) -> dict[int, str]:
        """Mapping from type id to a display name (Labeler Protocol)."""
        return self._type_map

    def label(
        self,
        params: Mapping[str, torch.Tensor] | torch.Tensor,
    ) -> torch.Tensor:
        """Assign condensed type ids for each parameter row.

        Args:
            params: Either

                * a mapping of Class-I parameter tensors with batch dim
                  ``(N, ...)``, assigned via :meth:`TypeSystem.assign_many`, or
                * a long tensor of already-known type ids (pass-through
                  clamp / validate path).

        Returns:
            Long tensor ``(N,)`` with ids in ``0 .. num_types-1``. Unmatched
            rows become ``0`` only when ``num_types == 0`` is impossible;
            otherwise unmatched rows stay :data:`UNMATCHED_TYPE_ID` (``-1``).
        """
        if isinstance(params, torch.Tensor):
            ids = params.long().reshape(-1)
            # Validate range for known ids; leave -1 soft-fails intact.
            if self.num_types > 0:
                bad = (ids >= self.num_types) | ((ids < 0) & (ids != UNMATCHED_TYPE_ID))
                if bool(bad.any()):
                    raise ValueError(
                        f"type ids out of range for num_types={self.num_types}: {ids[bad].tolist()}"
                    )
            return ids

        return self._type_system.assign_many(params, criterion=self._criterion)
