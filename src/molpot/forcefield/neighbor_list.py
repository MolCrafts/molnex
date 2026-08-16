"""Neighbor enumeration for ForceField nonbonded terms.

Uses molpy :class:`NeighborQuery` when a cell is available; otherwise
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
        edges: object | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return half-list ``edge_index (E,2)`` and ``distances (E,)``.

        Args:
            pos: ``(N, 3)`` in Å.
            atom_batch: molecule id ``(N,)``.
            cell: optional ``(3, 3)`` periodic cell.
            cutoff: if set, only pairs with ``r < cutoff`` (periodic uses
                molpy :class:`NeighborQuery`; open uses all-pairs filter).
            edges: optional mapping with ``edge_index`` and ``edge_diff``
                or ``edge_dist``. When present, those already-placed
                distances are used and PBC is not applied again.
        """
        n = pos.shape[0]
        if n < 2:
            return None
        placed = self._from_placed(pos, edges, cutoff)
        if placed is not None:
            return placed
        if cell is not None and cutoff is not None and cutoff > 0:
            return self._periodic(pos, atom_batch, cell, cutoff)
        return self._open(pos, atom_batch, cutoff)

    def _from_placed(
        self,
        pos: torch.Tensor,
        edges: object | None,
        cutoff: float | None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if edges is None:
            return None
        getter = edges.get if hasattr(edges, "get") else None
        if getter is None:
            try:
                edge_index = edges["edge_index"]  # type: ignore[index]
            except Exception:
                return None
        else:
            edge_index = getter("edge_index")
            if edge_index is None:
                try:
                    edge_index = edges["edges", "edge_index"]  # type: ignore[index]
                except Exception:
                    return None
        if edge_index is None:
            return None
        if getter is not None:
            edge_diff = getter("edge_diff")
            edge_dist = getter("edge_dist")
            num_edges = getter("num_edges")
            shifts = getter("shifts")
        else:
            edge_diff = edges["edge_diff"] if "edge_diff" in edges else None  # type: ignore[operator]
            edge_dist = edges["edge_dist"] if "edge_dist" in edges else None  # type: ignore[operator]
            num_edges = edges["num_edges"] if "num_edges" in edges else None  # type: ignore[operator]
            shifts = edges["shifts"] if "shifts" in edges else None  # type: ignore[operator]
        if edge_diff is None and edge_dist is None:
            return None
        n_live = int(num_edges) if num_edges is not None else int(edge_index.shape[0])
        edge_index = edge_index[:n_live]
        src, tgt = edge_index[:, 0], edge_index[:, 1]
        if edge_diff is not None:
            edge_diff = edge_diff[:n_live]
            implied = edge_diff.detach() - (pos.detach()[tgt] - pos.detach()[src])
            vectors = pos[tgt] - pos[src] + implied
            dist = vectors.norm(dim=-1)
        elif shifts is not None:
            shifts = shifts[:n_live]
            dist = (pos[tgt] - pos[src] + shifts[:n_live]).norm(dim=-1)
        else:
            dist = edge_dist[:n_live]
        keep = src < tgt
        if cutoff is not None:
            keep = keep & (dist < cutoff) & (dist > 0)
        edge_index = edge_index[keep]
        dist = dist[keep]
        if edge_index.shape[0] == 0:
            return None
        return edge_index, dist

    def _open(
        self,
        pos: torch.Tensor,
        atom_batch: torch.Tensor,
        cutoff: float | None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        device = pos.device
        num_graphs = int(atom_batch[-1].item()) + 1
        counts = torch.bincount(atom_batch, minlength=num_graphs)
        max_atoms = int(counts.max().item())
        if max_atoms < 2:
            return None
        # Generate a padded half-list per graph. Memory scales as B*max(N_g)^2,
        # not sum(N_g)^2, so heterogeneous packed batches remain practical.
        local = torch.triu_indices(max_atoms, max_atoms, offset=1, device=device)
        offsets = torch.cumsum(counts, dim=0) - counts
        valid = local[1].unsqueeze(0) < counts.unsqueeze(1)
        ii = (offsets.unsqueeze(1) + local[0].unsqueeze(0))[valid]
        jj = (offsets.unsqueeze(1) + local[1].unsqueeze(0))[valid]
        edge_index = torch.stack([ii, jj], dim=1)
        dist = torch.linalg.norm(pos[ii] - pos[jj], dim=-1).clamp(min=1e-12)
        if cutoff is not None:
            within = (dist < cutoff) & (dist > 0)
            edge_index = edge_index[within]
            dist = dist[within]
        return edge_index, dist

    def _periodic(
        self,
        pos: torch.Tensor,
        atom_batch: torch.Tensor,
        cell: torch.Tensor,
        cutoff: float,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """molpy NeighborQuery (self) under PBC; keep intramolecular pairs only."""
        from molpy import Box, NeighborQuery

        device, dtype = pos.device, pos.dtype
        pts = pos.detach().cpu().numpy().astype(np.float64, copy=False)
        h = cell.detach().cpu().numpy().astype(np.float64, copy=False)
        box = Box(h)
        neigh = NeighborQuery(box, pts, float(cutoff)).query_self()
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
