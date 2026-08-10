"""Class assignment tables produced by condensation.

Maps every condensed interaction row to a global type id and its source
system. Integer ids only — no SMARTS text.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from molrep.condensation.classes import InteractionClass

__all__ = ["ClassAssignment"]


@dataclass
class ClassAssignment:
    """Global type-id assignment for one :class:`InteractionClass`.

    Attributes:
        interaction: Interaction family these rows belong to.
        type_ids: Global type ids ``(N,)`` long, values in
            ``0 .. n_types-1`` (or ``-1`` for unmatched soft-fail).
        system_ids: Source system index per row ``(N,)`` long.
        row_indices: Original row index within each system ``(N,)`` long.
    """

    interaction: InteractionClass
    type_ids: torch.Tensor
    system_ids: torch.Tensor
    row_indices: torch.Tensor

    def __post_init__(self) -> None:
        n = int(self.type_ids.numel())
        if self.system_ids.numel() != n or self.row_indices.numel() != n:
            raise ValueError(
                "ClassAssignment tensors must share length: "
                f"type_ids={n}, system_ids={self.system_ids.numel()}, "
                f"row_indices={self.row_indices.numel()}"
            )
        self.type_ids = self.type_ids.long().reshape(-1)
        self.system_ids = self.system_ids.long().reshape(-1)
        self.row_indices = self.row_indices.long().reshape(-1)

    @property
    def n_rows(self) -> int:
        """Total number of assigned interaction rows."""
        return int(self.type_ids.numel())

    def for_system(self, system_id: int) -> torch.Tensor:
        """Type ids for rows belonging to ``system_id``.

        Args:
            system_id: System index used during multi-system merge.

        Returns:
            Long tensor of type ids in original row order for that system.
        """
        mask = self.system_ids == int(system_id)
        ids = self.type_ids[mask]
        order = self.row_indices[mask]
        if ids.numel() == 0:
            return ids
        # Restore original within-system order.
        sorted_idx = torch.argsort(order)
        return ids[sorted_idx]

    def type_id_at(self, system_id: int, row_index: int) -> int:
        """Return the type id for one ``(system, row)`` pair.

        Raises:
            KeyError: If the pair is not present.
        """
        mask = (self.system_ids == int(system_id)) & (self.row_indices == int(row_index))
        hits = self.type_ids[mask]
        if hits.numel() == 0:
            raise KeyError(f"No assignment for system={system_id}, row={row_index}")
        return int(hits[0].item())
