"""Neighbor enumeration for ForceField nonbonded terms.

Uses molrs :class:`NeighborQuery` when a cell is available; otherwise
all intramolecular half-pairs (open boundary).
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["NeighborListBuilder"]


class NeighborListBuilder:
    """Build half-pair lists for nonbonded evaluation.

    Single responsibility: geometry → ``(edge_index, distances)``.
    Does not own cutoff semantics beyond what the caller requests
    (cutoff is still a property of the Potential that consumes the list).
    """

    def build(
        self,
        *,
        pos: torch.Tensor,
        atom_batch: torch.Tensor,
        cell: torch.Tensor | None,
        cutoff: float | None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return half-list ``edge_index (E,2)`` and ``distances (E,)``.

        Args:
            pos: ``(N, 3)``.
            atom_batch: molecule id ``(N,)``.
            cell: optional ``(3, 3)`` periodic cell.
            cutoff: if set, only pairs with ``r < cutoff`` (periodic uses
                molrs :class:`NeighborQuery`; open uses all-pairs filter).
        """
        n = pos.shape[0]
        if n < 2:
            return None
        if cell is not None and cutoff is not None and cutoff > 0:
            return self._periodic(pos, atom_batch, cell, cutoff)
        return self._open(pos, atom_batch, cutoff)

    def _open(
        self,
        pos: torch.Tensor,
        atom_batch: torch.Tensor,
        cutoff: float | None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        n = pos.shape[0]
        device = pos.device
        idx_i = torch.arange(n, device=device).unsqueeze(1).expand(n, n)
        idx_j = torch.arange(n, device=device).unsqueeze(0).expand(n, n)
        upper = idx_i < idx_j
        same = atom_batch.unsqueeze(0) == atom_batch.unsqueeze(1)
        pair_mask = upper & same
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)
        r = torch.linalg.norm(diff, dim=-1)
        if cutoff is not None:
            pair_mask = pair_mask & (r < cutoff) & (r > 0)
        ii = idx_i[pair_mask]
        jj = idx_j[pair_mask]
        if ii.numel() == 0:
            return None
        edge_index = torch.stack([ii, jj], dim=1)
        dist = r[pair_mask].clamp(min=1e-12)
        return edge_index, dist

    def _periodic(
        self,
        pos: torch.Tensor,
        atom_batch: torch.Tensor,
        cell: torch.Tensor,
        cutoff: float,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """molrs NeighborQuery (self) under PBC; keep intramolecular pairs only."""
        import molrs

        device, dtype = pos.device, pos.dtype
        pts = pos.detach().cpu().numpy().astype(np.float64, copy=False)
        h = cell.detach().cpu().numpy().astype(np.float64, copy=False)
        box = molrs.Box(h)
        neigh = molrs.NeighborQuery(box, pts, float(cutoff)).query_self()
        if neigh.n_pairs == 0:
            return None
        # query_point_indices = i, point_indices = j (half list)
        i = torch.as_tensor(
            np.asarray(neigh.query_point_indices()), device=device, dtype=torch.long
        )
        j = torch.as_tensor(np.asarray(neigh.point_indices()), device=device, dtype=torch.long)
        dist_sq = torch.as_tensor(np.asarray(neigh.dist_sq()), device=device, dtype=dtype)
        # intramolecular only (batch multi-mol in one cell is rare; still filter)
        same = atom_batch[i] == atom_batch[j]
        i, j, dist_sq = i[same], j[same], dist_sq[same]
        if i.numel() == 0:
            return None
        edge_index = torch.stack([i, j], dim=1)
        dist = dist_sq.clamp(min=1e-24).sqrt()
        return edge_index, dist
