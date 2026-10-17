"""Neighbour-list construction — the owner of the repo Edge Convention.

A :class:`~molix.data.task.BatchTask`: the cutoff graph is a function of
positions and a cutoff, not dataset content, so it is built on the collated
batch rather than stored.

Builds the cutoff graph (``pair_index`` / ``pair_diff`` / ``pair_dist``) that
the collate schema in :mod:`molix.data.collate` consumes, which is why it
lives beside it rather than in an acquisition-side task tree.

The class is deliberately base-class free: it carries a ``task_id`` and an
``execute``, which is the whole structural contract an external ETL pipeline
needs in order to wrap and cache it.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from molix.F.locality import get_neighbor_pairs

#: Graph count from which :meth:`NeighborList.execute` enumerates the whole
#: batch at once instead of looping molecule by molecule. Above it the loop is
#: one Python iteration and one kernel launch per molecule — 2388 of each for a
#: packed Class-I batch.
_BATCHED_FROM_GRAPHS = 32

#: Candidate half-pairs (``sum n_i(n_i-1)/2``) under which
#: :meth:`NeighborList.execute` enumerates the batch at once **regardless** of
#: how few graphs it carries. Graph count alone is the wrong quantity to route
#: on: what makes the vectorised path expensive is candidate pairs, and a batch
#: can be tiny in both. A packed Class-I batch sits near 2e6 candidates and is
#: already taken by :data:`_BATCHED_FROM_GRAPHS`; one 100k-atom MD graph is 5e9
#: and must stay on the kernel's cell list. This bound only *widens* the
#: vectorised path, so nothing that takes it today stops taking it.
_BATCHED_MAX_CANDIDATES = 1 << 20


def _normalize_to_E2(pair_index: torch.Tensor) -> torch.Tensor:
    """Normalise pair_index to canonical ``[E, 2]``."""
    if pair_index.ndim != 2:
        raise ValueError(f"pair_index must be 2D, got {tuple(pair_index.shape)}")
    if pair_index.shape[1] == 2:
        return pair_index.long()
    if pair_index.shape[0] == 2:
        return pair_index.t().contiguous().long()
    raise ValueError(f"pair_index shape {tuple(pair_index.shape)} is invalid")


class NeighborList:
    """Compute neighbor list for a single sample.

    Wraps the compiled C++ backend ``molix.F.locality.get_neighbor_pairs``,
    which internally enumerates all O(N²) candidate pairs and returns those
    within ``cutoff`` as upper-triangle (half) pairs.

    Args:
        cutoff: Cutoff distance in Angstroms.
        max_num_pairs: Buffer size passed to the C++ kernel; sized for the
            *half*-pair count.  With ``symmetry=True`` the final edge tensor
            holds up to ``2 * max_num_pairs`` rows.
        pbc: Apply periodic boundary conditions.
        filter_padding: Strip NaN-padded rows from the C++ output.
        symmetry: If ``True`` (default), add the reverse edge for every pair so
            the output is a **full bidirectional** neighbour list
            (``E = 2 * n_pairs``).  Every atom then sees its complete
            neighbourhood, as required by Allegro, MACE, and all models that
            aggregate to the source node.
            If ``False``, only the upper-triangle pairs are returned
            (``E = n_pairs``, ``pair_index[:, 0] > pair_index[:, 1]``).

    The cutoff graph of one molecule depends only on that molecule, so
    :meth:`~molix.data.prepared.PreparedRow.capture` may run this task per row
    and the loader then skips it for rows captured with it. Stored per row it
    is ~36 B a pair for every conformer (a relax trajectory row 3.3 MB, a
    shard 285 GB); rebuilt per packed batch it is 0.2 s
    (:meth:`_execute_batched`), so the Class-I loader rebuilds it.
    """

    def __init__(
        self,
        cutoff: float = 5.0,
        max_num_pairs: int = 512,
        pbc: bool = False,
        filter_padding: bool = True,
        symmetry: bool = True,
    ) -> None:
        self.cutoff = cutoff
        self.max_num_pairs = max_num_pairs
        self.pbc = pbc
        self.filter_padding = filter_padding
        self.symmetry = symmetry

    @property
    def task_id(self) -> str:
        """Cache-key identity ``nlist:cut=..:max=..:pbc=..:sym=..``.

        The ``sym`` field keeps full-bidirectional and half-pair caches
        separate, since they produce different edge tensors.
        """
        return (
            f"nlist:cut={self.cutoff}:max={self.max_num_pairs}:pbc={self.pbc}:sym={self.symmetry}"
        )

    def __call__(self, data: dict) -> dict:
        """Alias of :meth:`execute`, preserved from the former ``Task`` base."""
        return self.execute(data)

    def pairs(self, pos: torch.Tensor, box_vectors: torch.Tensor | None = None) -> tuple:
        """Build one system's edges: ``(pair_index, pair_diff, pair_dist)``.

        Calls the C++ kernel on the sample's ``pos`` (and ``cell`` under
        PBC), normalises pair indices to ``(E, 2)``, optionally strips
        NaN-padded rows, and negates the kernel's delta so that
        ``pair_diff = pos[target] - pos[source]`` per the repo Edge
        Convention. With ``symmetry=True`` the reverse edge is appended for
        every pair (``E = 2 * n_pairs``), with the sign-flipped
        ``pair_diff`` and duplicated distances.

        Args:
            pos: Positions ``(N, 3)`` of one molecule.
            box_vectors: Lattice ``(3, 3)`` under PBC, else ``None``.

        Returns:
            ``pair_index`` ``(E, 2)`` (source in column 0, target in column 1),
            ``pair_diff`` ``(E, 3)`` (``pos[target] - pos[source]``) and
            ``pair_dist`` ``(E,)``, all molecule-local.
        """

        # O(N²) pair enum in the C++ kernel — size the buffer from N when the
        # caller left the small default (512) on a larger system.
        n_atoms = int(pos.shape[0])
        half_complete = n_atoms * (n_atoms - 1) // 2
        max_pairs = self.max_num_pairs
        if max_pairs > 0 and half_complete > max_pairs:
            max_pairs = half_complete

        neighbors, deltas, distances, num_pairs = get_neighbor_pairs(
            positions=pos,
            cutoff=self.cutoff,
            max_num_pairs=max_pairs,
            box_vectors=box_vectors,
        )
        n_found = int(num_pairs.reshape(-1)[0].item()) if num_pairs.numel() else 0
        # Strict greater: n_found == max_pairs is a full-but-valid buffer (e.g.
        # all half-pairs within cutoff when max_pairs == N(N-1)/2).
        if max_pairs > 0 and n_found > max_pairs:
            raise RuntimeError(
                f"Neighbor buffer overflow: found {n_found} pairs > "
                f"max_num_pairs={max_pairs} (N={n_atoms}, cutoff={self.cutoff}). "
                "Increase max_num_pairs or lower the cutoff. "
                "Note: the kernel is O(N²); large condensed-phase systems need a "
                "cell-list/Verlet neighbour algorithm (not yet in the native op library, csrc/)."
            )

        pair_index = _normalize_to_E2(neighbors)

        if self.filter_padding:
            valid = ~torch.isnan(distances)
            pair_index = pair_index[valid]
            deltas = deltas[valid]
            distances = distances[valid]

        # Convention: pair_diff = pos[target] - pos[source]  (source → target).
        # The C++ kernel returns deltas = pos[rows] - pos[cols] = pos[source] - pos[target],
        # so we negate.  See CLAUDE.md "Edge Convention" for the full spec.
        pair_diff = -deltas

        if self.symmetry:
            # Add reverse edges: for each (src→tgt), append (tgt→src).
            # pair_diff reverses sign: pos[new_tgt] - pos[new_src] = -pair_diff.
            pair_index = torch.cat([pair_index, pair_index[:, [1, 0]]], dim=0)
            pair_diff = torch.cat([pair_diff, -pair_diff], dim=0)
            distances = torch.cat([distances, distances], dim=0)

        return pair_index, pair_diff, distances

    def _execute_batched(
        self,
        batch: TensorDict,
        pos: torch.Tensor,
        membership: torch.Tensor,
        n_graphs: int,
    ) -> TensorDict:
        """The same cutoff graph, enumerated for every molecule at once.

        The loop this replaces ran one Python iteration, one boolean mask and
        one kernel launch **per molecule** — 2388 of each for a packed Class-I
        batch, which is the "avoidable per-molecule serial execution" the
        production checklist forbids on the input path. Here the candidate
        pairs of every molecule are decoded from one flat index and filtered in
        a handful of tensor ops, so the work is proportional to the pairs, not
        to the molecule count, and it runs on whatever device the batch is on.

        Restricted to the open-boundary, padding-filtered case: PBC needs a
        per-molecule box and ``filter_padding=False`` exists to expose the
        kernel's own NaN rows, neither of which this path produces.

        Args:
            batch: The collated batch to write ``pairs`` into.
            pos: Positions ``(N, 3)``.
            membership: Graph membership ``(N,)``.
            n_graphs: Number of graphs in the batch.

        Returns:
            *batch*, with ``pairs`` written.
        """
        device = pos.device
        counts = torch.bincount(membership, minlength=n_graphs)
        # Atoms of one molecule need not be contiguous; sort once and index
        # through the permutation rather than assuming the packed layout.
        order = torch.argsort(membership, stable=True)
        offsets = torch.cumsum(counts, 0) - counts

        n = counts.to(torch.float64)
        per_graph = ((n * (n - 1)) / 2).to(torch.long)
        total = int(per_graph.sum().item())
        if total == 0:
            return self._write_pairs(
                batch,
                torch.zeros(0, 2, dtype=torch.long, device=device),
                torch.zeros(0, 3, dtype=pos.dtype, device=device),
                torch.zeros(0, dtype=pos.dtype, device=device),
            )

        ends = torch.cumsum(per_graph, 0)
        flat = torch.arange(total, device=device)
        graph = torch.searchsorted(ends, flat, right=True)
        within = flat - (ends - per_graph)[graph]

        # Decode the q-th upper-triangle cell (i < j) of an n x n block. The
        # closed form is exact in float64 for any molecule that fits a batch.
        ng = counts[graph].to(torch.float64)
        q = within.to(torch.float64)
        i = (ng - 2 - (torch.sqrt(-8 * q + 4 * ng * (ng - 1) - 7) / 2 - 0.5).floor()).to(torch.long)
        fi = i.to(torch.float64)
        j = (q + fi + 1 - ng * (ng - 1) / 2 + (ng - fi) * (ng - fi - 1) / 2).to(torch.long)

        base = offsets[graph]
        # The decode gives ``i < j``; the half-pair convention of this task is
        # ``pair_index[:, 0] > pair_index[:, 1]`` (source > target), so the
        # larger local index is the source. With ``symmetry=True`` both
        # directions are emitted and the choice is invisible — which is exactly
        # why getting it backwards would only show up in the half-pair case.
        source = order[base + j]
        target = order[base + i]

        # Convention: pair_diff = pos[target] - pos[source].
        diff = pos[target] - pos[source]
        dist = diff.norm(dim=-1)
        keep = dist <= self.cutoff
        source, target = source[keep], target[keep]
        diff, dist = diff[keep], dist[keep]

        # No buffer guard here, deliberately. ``max_num_pairs`` sizes the C++
        # kernel's output buffer; this path allocates nothing from it and has
        # already materialised every candidate by now, so a cap could only
        # reject a correct answer after computing it. The loop path agrees:
        # it grows ``max_pairs`` to the molecule's complete half graph whenever
        # the caller asked for less (see :meth:`pairs`), which is why its own
        # guard is unreachable in production. A cap that fires on one path and
        # not the other would break the one invariant the routing rests on —
        # that the two paths are the same answer — and it did: routing a small
        # tail batch here raised `found 6 pairs over 1 graphs` on inputs the
        # loop returns 6 pairs for.

        pair_index = torch.stack([source, target], dim=-1)
        if self.symmetry:
            pair_index = torch.cat([pair_index, pair_index[:, [1, 0]]], dim=0)
            diff = torch.cat([diff, -diff], dim=0)
            dist = torch.cat([dist, dist], dim=0)
        return self._write_pairs(batch, pair_index, diff, dist)

    @staticmethod
    def _write_pairs(
        batch: TensorDict,
        pair_index: torch.Tensor,
        pair_diff: torch.Tensor,
        pair_dist: torch.Tensor,
    ) -> TensorDict:
        """Replace the batch's ``pairs`` namespace. One spelling, two paths."""
        batch["pairs"] = TensorDict(
            {"pair_index": pair_index, "pair_diff": pair_diff, "pair_dist": pair_dist},
            batch_size=[int(pair_index.shape[0])],
        )
        return batch

    def _prefer_batched(self, membership: torch.Tensor, n_graphs: int) -> bool:
        """Whether to enumerate the whole batch at once rather than loop.

        Two accepts, both widening: many graphs, where the loop's per-molecule
        Python iteration and kernel launch are the cost the checklist forbids on
        the input path; or few enough candidate half-pairs that enumerating them
        all is trivial, which is where a packing remainder lands.

        That second accept is not a tuning knob. Routing on graph count alone
        sent an end-of-epoch tail batch — one THEMol torsion-scan row is a
        single scan of 10-24 conformers, so a one-row batch carries fewer than
        :data:`_BATCHED_FROM_GRAPHS` graphs — down the per-molecule loop and
        into the C++ op, which is how a job that had been training for 8 h 48 m
        discovered that its frozen snapshot carried no native library
        (2026-09-22, stage ``torsion-scan`` of run 2787862). The graph count of
        a batch tail is luck; the work it implies is not.

        Args:
            membership: Graph membership ``(N,)``.
            n_graphs: Number of graphs in the batch.

        Returns:
            ``True`` to take :meth:`_execute_batched`.
        """
        if n_graphs >= _BATCHED_FROM_GRAPHS:
            return True
        if n_graphs == 0:
            return True
        counts = torch.bincount(membership, minlength=n_graphs).to(torch.float64)
        candidates = float(((counts * (counts - 1)) / 2).sum().item())
        return candidates <= _BATCHED_MAX_CANDIDATES

    def execute(self, batch: TensorDict) -> TensorDict:
        """Build the cutoff graph of every molecule in *batch* and write ``pairs``.

        Blocks carry no neighbour graph — it is a function of positions and a
        cutoff, not dataset content — so it is built here, on the collated
        batch, one molecule at a time with indices rebased into batch space.

        Args:
            batch: A collated batch with ``atoms.pos`` ``(N, 3)``,
                ``atoms.batch`` ``(N,)`` graph membership and, under PBC,
                ``graphs.cell`` ``(B, 3, 3)``.

        Returns:
            The same batch with its ``pairs`` namespace replaced by
            ``pair_index`` ``(E, 2)``, ``pair_diff`` ``(E, 3)`` and
            ``pair_dist`` ``(E,)``, indices global to the batch.
        """
        pos = batch["atoms", "pos"]
        membership = batch["atoms", "batch"]
        n_graphs = int(membership.max().item()) + 1 if membership.numel() else 0
        cells = batch["graphs", "cell"] if self.pbc and "cell" in batch["graphs"].keys() else None

        if cells is None and self.filter_padding and self._prefer_batched(membership, n_graphs):
            return self._execute_batched(batch, pos, membership, n_graphs)

        index_parts: list[torch.Tensor] = []
        diff_parts: list[torch.Tensor] = []
        dist_parts: list[torch.Tensor] = []
        offset = 0
        for graph in range(n_graphs):
            rows = membership == graph
            local_pos = pos[rows]
            box = cells[graph] if cells is not None else None
            pair_index, pair_diff, pair_dist = self.pairs(local_pos, box)
            index_parts.append(pair_index + offset)
            diff_parts.append(pair_diff)
            dist_parts.append(pair_dist)
            offset += int(local_pos.shape[0])

        pair_index = torch.cat(index_parts) if index_parts else torch.zeros(0, 2, dtype=torch.long)
        batch["pairs"] = TensorDict(
            {
                "pair_index": pair_index,
                "pair_diff": torch.cat(diff_parts) if diff_parts else torch.zeros(0, 3),
                "pair_dist": torch.cat(dist_parts) if dist_parts else torch.zeros(0),
            },
            batch_size=[int(pair_index.shape[0])],
        )
        return batch


class DenseNeighborList:
    """Pairwise geometry of padded molecules, indices implicit.

    Takes positions as ``(B, N, 3)`` — one row per molecule, padded to a common
    ``N`` — and returns the geometry of every pair as ``(B, N, N, …)``. There is
    no pair *list*: ``(b, i, j)`` **is** the pair, so no index tensor is built,
    transferred, or gathered through.

    **Why dense.** On OMol25 4m the 10 Å cutoff list is 72.5 % of the complete
    per-molecule graph (mean 54.85 atoms/molecule; 3769 list rows against 5200
    complete), so the sparse form pays two ``int64`` indices per pair to skip
    barely a quarter of the work. Measured on the Class-I production batch that
    list is 317 MB of a 344 MB batch (92.2 %), and its gather's backward —
    ``index_put_(accumulate=True)``, 9.9 M atomics into 131 072 atoms — is 48.9 %
    of eager device time. Here the same geometry is a broadcast subtract whose
    backward is a **sum reduction**: no atomics, no indices, no list to ship.

    **Why the cutoff is optional, and why ``None`` is right for a molecule.**
    Truncating a ``1/r`` sum at a radius is a periodic-system technique, where
    the radius bounds the real-space half of an Ewald/PME split. An
    open-boundary molecule has no lattice sum to split: the exact answer is
    every pair, which dense form gives for free. Pass a *cutoff* only for what a
    cutoff is actually for — a short-ranged term (Lennard-Jones), or the
    real-space half of PME.

    Attributes:
        cutoff: Radius in Å, or ``None`` for every pair.
        half: Whether to keep only ``i < j`` (each pair once) rather than both
            orientations.
    """

    def __init__(self, cutoff: float | None = None, *, half: bool = True) -> None:
        """Declare the radius and the orientation convention.

        Args:
            cutoff: Radius in Å. ``None`` keeps every pair — the exact
                open-boundary sum.
            half: Keep ``i < j`` only. ``False`` keeps both orientations, which
                a per-atom (rather than per-pair) accumulation wants.

        Raises:
            ValueError: *cutoff* is given and is not positive.
        """
        if cutoff is not None and cutoff <= 0:
            raise ValueError(f"cutoff must be positive, got {cutoff}")
        self.cutoff = None if cutoff is None else float(cutoff)
        self.half = bool(half)

    def geometry(
        self, pos: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Displacements, distances and the live mask of every pair.

        Shapes are a function of ``pos.shape`` alone, so a compiled region sees
        static shapes and no data-dependent control flow.

        The diagonal is why this is not two lines. ``‖pos_i − pos_i‖`` is zero
        and ``d‖v‖/dv = v/‖v‖`` is **NaN** there, and a NaN gradient on a dead
        row poisons the whole backward even though the forward is finite —
        masking the *output* afterwards does not undo it, because it has already
        flowed into ``pos``. The squared distance is therefore substituted to one
        on every entry that will be masked out, before the square root sees it.

        Args:
            pos: Positions ``(B, N, 3)``.
            mask: Which slots hold a real atom, ``(B, N)``. ``None`` means all.

        Returns:
            ``(diff, dist, live)`` — ``diff`` ``(B, N, N, 3)`` is
            ``pos[:, j] - pos[:, i]`` (source ``i``, target ``j``, the repo Edge
            Convention), ``dist`` ``(B, N, N)``, and ``live`` ``(B, N, N)``
            ``bool``. Dead entries carry a finite ``dist`` of no physical
            meaning: multiply an energy by *live*, never slice by it.

        Raises:
            ValueError: *pos* is not ``(B, N, 3)``, or *mask* does not match it.
        """
        if pos.ndim != 3 or pos.shape[-1] != 3:
            raise ValueError(f"pos must be (B, N, 3), got {tuple(pos.shape)}")
        n = int(pos.shape[1])
        if mask is not None and tuple(mask.shape) != tuple(pos.shape[:2]):
            raise ValueError(
                f"mask must be (B, N) = {tuple(pos.shape[:2])}, got {tuple(mask.shape)}"
            )

        # Edge Convention: pair_diff = pos[target] - pos[source].
        diff = pos.unsqueeze(1) - pos.unsqueeze(2)

        ones = torch.ones((n, n), dtype=torch.bool, device=pos.device)
        live = (
            torch.triu(ones, diagonal=1)
            if self.half
            else ~torch.eye(n, dtype=torch.bool, device=pos.device)
        )
        live = live.expand(pos.shape[0], n, n)
        if mask is not None:
            live = live & mask.unsqueeze(1) & mask.unsqueeze(2)

        square = diff.pow(2).sum(-1)
        dist = torch.where(live, square, torch.ones_like(square)).sqrt()

        if self.cutoff is not None:
            live = live & (dist <= self.cutoff)
        return diff, dist, live
