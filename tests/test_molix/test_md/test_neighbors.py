"""Tests for molix.md.neighbors."""

import pytest
import torch

from molix.md.neighbors import NeighborStrategy, PeriodicNeighborList


def _lattice(n_side: int = 3, spacing: float = 3.0) -> tuple[torch.Tensor, torch.Tensor]:
    """A simple cubic lattice and its cell — a periodic system with real edges."""
    grid = torch.arange(n_side, dtype=torch.float64) * spacing
    pos = torch.stack(torch.meshgrid(grid, grid, grid, indexing="ij"), dim=-1).reshape(-1, 3)
    cell = torch.eye(3, dtype=torch.float64) * (n_side * spacing)
    return pos, cell


def _triclinic() -> tuple[torch.Tensor, torch.Tensor]:
    """The golden triclinic counterexample and ten atoms placed inside it.

    The cell ``[[10, 0, 0], [6, 8, 0], [0, 0, 10]]`` has ``V = 800 A^3`` and
    perpendicular widths ``w_i = V / ||a_j x a_k|| = (8.0, 8.0, 10.0) A``, so
    minimum-image completeness holds only up to ``min_i w_i / 2 = 4.000 A``.
    Its shortest **row norm** is ``10.0 A``, which a row-norm guard reads as an
    admissible cutoff of ``5.000 A`` — the gap between ``4.000`` and ``5.000``
    is the bug this cell pins.

    Positions are literal fractional coordinates mapped by ``frac @ cell`` (no
    RNG): the closest minimum-image pairs sit at ``2.0 A`` and the next shell
    at ``4.0 A``, so an accepted ``cutoff = 3.9 A`` yields a non-empty edge set
    with no pair inside ``0.1 A`` of the cutoff.

    Returns:
        ``(positions (10, 3), cell (3, 3))`` in Angstrom, ``float64``.
    """
    cell = torch.tensor([[10.0, 0.0, 0.0], [6.0, 8.0, 0.0], [0.0, 0.0, 10.0]], dtype=torch.float64)
    frac = torch.tensor(
        [
            [0.05, 0.10, 0.10],
            [0.25, 0.10, 0.10],
            [0.45, 0.10, 0.10],
            [0.65, 0.10, 0.10],
            [0.05, 0.40, 0.45],
            [0.25, 0.40, 0.45],
            [0.45, 0.40, 0.45],
            [0.05, 0.70, 0.80],
            [0.25, 0.70, 0.80],
            [0.45, 0.70, 0.80],
        ],
        dtype=torch.float64,
    )
    return frac @ cell, cell


@pytest.fixture
def nlist():
    pos, cell = _lattice()
    return PeriodicNeighborList(cell=cell, cutoff=3.5, positions=pos), pos


class TestPeriodicNeighborList:
    """Test the rebuilding fixed-capacity neighbour list."""

    def test_satisfies_the_neighbor_strategy_protocol(self, nlist):
        nl, _ = nlist
        assert isinstance(nl, NeighborStrategy)

    def test_capacity_exceeds_initial_edges(self, nlist):
        """Headroom is what lets the edge count grow without reallocating."""
        nl, _ = nlist
        assert nl.capacity > nl.num_edges

    def test_buffer_shapes_are_the_capacity(self, nlist):
        """Shapes must be the capacity, not the live edge count — that is the
        whole point: a CUDA graph sees constant shapes across rebuilds. The
        edge buffer is ``(E, 2)`` per the repo-wide edge convention (``(2, N)``
        is reserved for ``bond_index`` as an anti-alias guard)."""
        nl, _ = nlist
        assert nl.edge_index.shape == (nl.capacity, 2)
        assert nl.shifts.shape == (nl.capacity, 3)

    def test_rebuild_keeps_shapes_constant(self, nlist):
        """Displacing every atom changes the edge set, never the shapes."""
        nl, pos = nlist
        shapes = (nl.edge_index.shape, nl.shifts.shape)
        torch.manual_seed(0)
        nl.rebuild(pos + torch.randn_like(pos) * 0.3)
        assert (nl.edge_index.shape, nl.shifts.shape) == shapes

    def test_rebuild_tracks_a_changed_neighbour_set(self, nlist):
        """A real displacement must actually change the recorded edges."""
        nl, pos = nlist
        before = nl.num_edges
        nl.rebuild(pos * 1.15)  # dilate: fewer pairs inside the cutoff
        assert nl.num_edges != before

    def test_dead_edges_are_self_loops_on_atom_zero(self, nlist):
        """Padding rows must not point at real atoms."""
        nl, _ = nlist
        tail = nl.edge_index[nl.num_edges :]
        assert torch.count_nonzero(tail) == 0

    def test_dead_edge_shift_exceeds_the_cutoff(self, nlist):
        """Beyond the cutoff every envelope is 0, which is what zeroes them."""
        nl, _ = nlist
        tail = nl.shifts[nl.num_edges :]
        assert bool((torch.linalg.norm(tail, dim=-1) > nl.cutoff).all())

    def test_shifts_reconstruct_minimum_image_displacements(self, nlist):
        """``pos[t] - pos[s] + shift`` must be the minimum-image vector, i.e.
        no live edge is longer than the cutoff."""
        nl, pos = nlist
        n = nl.num_edges
        src, tgt = nl.edge_index[:n, 0], nl.edge_index[:n, 1]
        vectors = pos[tgt] - pos[src] + nl.shifts[:n]
        assert float(torch.linalg.norm(vectors, dim=-1).max()) <= nl.cutoff + 1e-9

    def test_rebuild_count_increments(self, nlist):
        """Diagnostics: callers need to know the cadence actually fired."""
        nl, pos = nlist
        assert nl.rebuild_count == 0
        nl.rebuild(pos)
        nl.rebuild(pos)
        assert nl.rebuild_count == 2

    def test_to_accepts_positional_dtype(self, nlist):
        """``nl.to(torch.float32)`` must work — Tensor.to semantics, as documented."""
        nl, _ = nlist
        out = nl.to(torch.float32)
        assert out is nl
        assert nl.shifts.dtype == torch.float32
        assert nl.cell.dtype == torch.float32
        assert nl.edge_index.dtype == torch.long  # indices are never cast

    def test_rejects_cutoff_beyond_half_the_cell(self):
        """Minimum image silently misses images past L/2 — refuse instead."""
        pos, cell = _lattice()
        with pytest.raises(ValueError, match="exceeds half the minimum perpendicular cell width"):
            PeriodicNeighborList(cell=cell, cutoff=5.0, positions=pos)

    def test_accepts_orthorhombic_cutoff_just_below_half_the_cell(self):
        """Orthorhombic parity: for a cube ``w_i = ||a_i||``, so the bound stays
        ``4.500 A`` on the 9 A cell and ``4.4 A`` must still build."""
        pos, cell = _lattice()
        nl = PeriodicNeighborList(cell=cell, cutoff=4.4, positions=pos)
        assert nl.cutoff == 4.4
        assert nl.num_edges > 0

    def test_rejects_triclinic_cutoff_admitted_by_the_row_norm(self):
        """The bug, asserted directly: the golden cell's shortest row norm is
        ``10 A`` (row-norm bound ``5.000 A``) but its narrowest perpendicular
        width is ``8 A`` (true bound ``4.000 A``), so ``5.0 A`` must be refused
        instead of silently dropping pairs inside the cutoff."""
        pos, cell = _triclinic()
        with pytest.raises(ValueError):
            PeriodicNeighborList(cell=cell, cutoff=5.0, positions=pos)

    def test_triclinic_rejection_names_the_perpendicular_bound(self):
        """The measured bound must be observable through the public error, not
        just the refusal: ``min_i w_i / 2 = 4.000 A`` for the golden cell."""
        pos, cell = _triclinic()
        with pytest.raises(ValueError, match=r"4\.000 A"):
            PeriodicNeighborList(cell=cell, cutoff=4.5, positions=pos)

    def test_accepts_triclinic_cutoff_below_the_perpendicular_bound(self):
        """Acceptance must mean "actually built", not "did not raise": below the
        ``4.000 A`` bound the list constructs and reports real edges (closest
        golden pairs are at ``2.0 A``)."""
        pos, cell = _triclinic()
        nl = PeriodicNeighborList(cell=cell, cutoff=3.9, positions=pos)
        assert nl.cutoff == 3.9
        assert nl.num_edges > 0

    def test_rejects_a_singular_cell(self):
        """A zero-volume cell has no finite width. ``V / area`` would be ``nan``
        and ``cutoff > nan`` is ``False`` — the guard must raise rather than let
        a degenerate cell through the hole it opens."""
        cell = torch.tensor(
            [[10.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 0.0, 10.0]], dtype=torch.float64
        )
        pos = torch.tensor(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
            dtype=torch.float64,
        )
        with pytest.raises(ValueError):
            PeriodicNeighborList(cell=cell, cutoff=3.0, positions=pos)

    def test_rejects_a_non_3x3_cell(self):
        """A batched ``(1, 3, 3)`` cell must fail with a clean ``ValueError`` at
        the guard, not an opaque indexing error deeper in the kernel path."""
        pos, cell = _lattice()
        with pytest.raises(ValueError):
            PeriodicNeighborList(cell=cell.unsqueeze(0), cutoff=3.5, positions=pos)

    def test_overflow_raises_rather_than_truncating(self):
        """A truncated neighbour list is a silently wrong energy."""
        pos, cell = _lattice()
        nl = PeriodicNeighborList(cell=cell, cutoff=3.5, positions=pos, capacity_factor=1.0)
        with pytest.raises(RuntimeError, match="overflow"):
            nl.rebuild(pos * 0.5)  # compress: many more pairs inside the cutoff
