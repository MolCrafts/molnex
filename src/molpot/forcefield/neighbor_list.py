"""Neighbor enumeration for ForceField nonbonded terms.

Uses molpy :class:`NeighborQuery` when a cell is available; otherwise
all intramolecular half-pairs (open boundary).

Lists are **capacity-shaped**: a list is described by ``(edge_index, dist,
live_mask)`` and never compacted. Selecting rows with a boolean mask makes the
output shape depend on the data, which is a graph break on the compiled force
path; the pair terms multiply by ``live_mask`` instead. Only the open /
periodic constructions filter while they enumerate — they hand back an all-true
mask over a row count that depends on the geometry.

Which is why they are eager-only. :meth:`NeighborListBuilder.build` runs them
inline on **every** call that arrives without a placed list, not once per
rebuild, and both are data-dependent throughout (a host sync on the atom
counts, boolean-mask selection, and molpy ``NeighborQuery`` on numpy under
PBC). A compiled force path must therefore supply a placed, capacity-shaped
list in the batch ``edges`` namespace — what :class:`molix.md.NeighborList`
writes — which ``_from_placed`` reuses at its own capacity with no
data-dependent shape anywhere.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["NeighborListBuilder"]


class NeighborListBuilder:
    """Build half-pair lists for nonbonded evaluation.

    Single responsibility: geometry → ``(edge_index, distances, live_mask)``.
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Return ``edge_index (E,2)``, ``distances (E,)`` and ``live_mask (E,)``.

        Args:
            pos: ``(N, 3)`` in Å.
            atom_batch: molecule id ``(N,)``.
            cell: optional ``(3, 3)`` periodic cell.
            cutoff: if set, only pairs with ``r < cutoff`` count (periodic uses
                molpy :class:`NeighborQuery`; open uses all-pairs filter).
            edges: optional mapping with ``edge_index`` and ``edge_diff``
                or ``edge_dist``. When present, those already-placed
                distances are used at their own capacity and PBC is not
                applied again.

        Returns:
            ``(edge_index, distances, live_mask)`` — pair endpoints
            ``(E, 2)``, separations ``(E,)`` in Å and a boolean ``(E,)``.
            ``E`` is the capacity of the placed list, or the number of pairs
            the open / periodic construction enumerated. ``live_mask`` is
            ``True`` exactly on the rows that contribute; dead rows keep their
            stored endpoints and distance. Wherever a distance is *computed*
            here it is floored at ``1e-12`` Å, so a padding row whose two ends
            coincide reads a tiny positive separation instead of an exact
            zero — a zero would make the ``1/r`` kernels infinite and the
            gradient of ``√`` at a zero vector ``NaN``, neither of which the
            later multiply-by-zero can undo. The one exception is the
            ``edge_dist`` passthrough of :meth:`_from_placed`, which forwards
            the caller's stored distances untouched. ``None`` when no list
            exists at all: fewer than two atoms, or a construction that
            produced no pair.
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Reuse an already-placed edge list at the capacity it was written at.

        Args:
            pos: Atom positions ``(N, 3)`` in Å.
            edges: The batch ``edges`` namespace, or ``None``.
            cutoff: Real-space cutoff in Å, or ``None`` to keep every pair.

        Returns:
            ``(edge_index, dist, live)`` at the list's capacity — ``(E, 2)``,
            ``(E,)`` in Å and a boolean ``(E,)`` — or ``None`` when ``edges``
            carries no usable list. ``live`` is ``source < target`` (the half
            list) ∧ ``row < num_edges`` ∧ ``0 < r < cutoff``. Dead rows keep
            their stored endpoints — a padding row is an in-range self-loop,
            so every downstream gather and ``index_add_`` stays valid and
            contributes exactly zero once multiplied by the mask. Distances
            recomputed here (from ``edge_diff`` or from ``shifts``) are
            floored at ``1e-12`` Å; a stored ``edge_dist`` is forwarded as
            given.
        """
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
        # A list with no rows at all is structurally absent, not empty-by-data:
        # fall back to enumeration. Every other row stays where it was written.
        if edge_index.shape[0] == 0:
            return None
        src, tgt = edge_index[:, 0], edge_index[:, 1]
        if edge_diff is not None:
            implied = edge_diff.detach() - (pos.detach()[tgt] - pos.detach()[src])
            vectors = pos[tgt] - pos[src] + implied
            # Clamped before the sqrt, not after: a padded row whose two ends
            # coincide is a zero vector, and ``norm``'s gradient there is NaN,
            # which the ``0 * scale`` masking downstream would propagate into
            # the autograd Coulomb forces. Value-identical on every live row
            # (and the same form ``_periodic`` already uses).
            dist = (vectors * vectors).sum(-1).clamp_min(1e-24).sqrt()
        elif shifts is not None:
            # Same clamp, same reason as the branch above: a padding row whose
            # ends coincide under a zero shift is a zero vector.
            shifted = pos[tgt] - pos[src] + shifts
            dist = (shifted * shifted).sum(-1).clamp_min(1e-24).sqrt()
        elif edge_dist is not None:
            dist = edge_dist
        else:  # unreachable — the guard above requires a diff or a distance
            return None
        live = src < tgt
        if num_edges is not None:
            # Compared as tensors: reading the live count out to Python would
            # be a host sync, and slicing by it a data-dependent shape.
            rows = torch.arange(edge_index.shape[0], device=edge_index.device)
            live = live & (rows < num_edges)
        if cutoff is not None:
            live = live & (dist < cutoff) & (dist > 0)
        return edge_index, dist, live

    def _open(
        self,
        pos: torch.Tensor,
        atom_batch: torch.Tensor,
        cutoff: float | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """All intramolecular half-pairs within ``cutoff`` (Å), open boundary.

        Enumerates the upper triangle per molecule, filters by distance, and
        returns an all-true ``live_mask`` — the row count therefore depends on
        the geometry, which is why this path is eager-only.
        """
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
        return edge_index, dist, torch.ones_like(dist, dtype=torch.bool)

    def _periodic(
        self,
        pos: torch.Tensor,
        atom_batch: torch.Tensor,
        cell: torch.Tensor,
        cutoff: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
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
        return edge_index, dist, torch.ones_like(dist, dtype=torch.bool)
