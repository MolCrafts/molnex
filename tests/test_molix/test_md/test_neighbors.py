"""Tests for molix.md.neighbors."""

import math
from typing import NamedTuple

import pytest
import torch
from tensordict import TensorDict

import molix.data.tasks.neighbor
import molix.md
import molix.md.neighbors
from molix.md import (
    EV_PER_AMU_A2_FS2,
    MD,
    LennardJonesCutForceField,
    MaxwellBoltzmann,
    MDHook,
    MDObservables,
    MDRunner,
)
from molix.md.neighbors import NeighborList, NeighborStrategy


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
    return NeighborList(cell=cell, cutoff=3.5, positions=pos), pos


class TestNeighborList:
    """Test the rebuilding fixed-capacity neighbour list."""

    def test_renamed_symbol_is_the_md_export(self):
        """The MD list is exported as ``NeighborList``; the old name is gone.

        There is no back-compat alias (``stage: experimental``, repo norm), so
        the old name must be absent from the module *and* from ``__all__`` — a
        ``PeriodicNeighborList = NeighborList`` shim would defeat this guard.
        """
        from molix.md import NeighborList

        assert NeighborList is molix.md.neighbors.NeighborList
        assert not hasattr(molix.md, "PeriodicNeighborList")

        names = list(molix.md.__all__)
        assert "PeriodicNeighborList" not in names
        assert "NeighborList" in names
        # ``__all__`` stays alphabetized (notes, 2026-08-09) inside the
        # PascalCase block that follows the CONSTANT_CASE block, so the entry
        # sorts above ``NeighborListHook`` instead of keeping the old "P" slot.
        pascal = [name for name in names if not name.isupper()]
        assert pascal == sorted(pascal)
        assert names.index("NeighborList") < names.index("NeighborListHook")

    def test_md_list_is_not_the_pipeline_task(self):
        """Anti-shadow: two deliberate same-name types in different layers.

        ``molix.md.neighbors`` imports the pipeline ``SampleTask`` — the one
        owner of kernel-output normalisation — into its own namespace. Once the
        MD buffer owner takes the bare name, that import must be aliased to
        ``NeighborListTask``, or the class definition rebinds the module-level
        name and the constructor calls *itself* recursively.
        """
        assert molix.md.NeighborList is not molix.data.tasks.neighbor.NeighborList
        assert molix.md.neighbors.NeighborListTask is molix.data.tasks.neighbor.NeighborList

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
            NeighborList(cell=cell, cutoff=5.0, positions=pos)

    def test_accepts_orthorhombic_cutoff_just_below_half_the_cell(self):
        """Orthorhombic parity: for a cube ``w_i = ||a_i||``, so the bound stays
        ``4.500 A`` on the 9 A cell and ``4.4 A`` must still build."""
        pos, cell = _lattice()
        nl = NeighborList(cell=cell, cutoff=4.4, positions=pos)
        assert nl.cutoff == 4.4
        assert nl.num_edges > 0

    def test_rejects_triclinic_cutoff_admitted_by_the_row_norm(self):
        """The bug, asserted directly: the golden cell's shortest row norm is
        ``10 A`` (row-norm bound ``5.000 A``) but its narrowest perpendicular
        width is ``8 A`` (true bound ``4.000 A``), so ``5.0 A`` must be refused
        instead of silently dropping pairs inside the cutoff."""
        pos, cell = _triclinic()
        with pytest.raises(ValueError):
            NeighborList(cell=cell, cutoff=5.0, positions=pos)

    def test_triclinic_rejection_names_the_perpendicular_bound(self):
        """The measured bound must be observable through the public error, not
        just the refusal: ``min_i w_i / 2 = 4.000 A`` for the golden cell."""
        pos, cell = _triclinic()
        with pytest.raises(ValueError, match=r"4\.000 A"):
            NeighborList(cell=cell, cutoff=4.5, positions=pos)

    def test_accepts_triclinic_cutoff_below_the_perpendicular_bound(self):
        """Acceptance must mean "actually built", not "did not raise": below the
        ``4.000 A`` bound the list constructs and reports real edges (closest
        golden pairs are at ``2.0 A``)."""
        pos, cell = _triclinic()
        nl = NeighborList(cell=cell, cutoff=3.9, positions=pos)
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
            NeighborList(cell=cell, cutoff=3.0, positions=pos)

    def test_rejects_a_non_3x3_cell(self):
        """A batched ``(1, 3, 3)`` cell must fail with a clean ``ValueError`` at
        the guard, not an opaque indexing error deeper in the kernel path."""
        pos, cell = _lattice()
        with pytest.raises(ValueError):
            NeighborList(cell=cell.unsqueeze(0), cutoff=3.5, positions=pos)

    def test_overflow_raises_rather_than_truncating(self):
        """A truncated neighbour list is a silently wrong energy."""
        pos, cell = _lattice()
        nl = NeighborList(cell=cell, cutoff=3.5, positions=pos, capacity_factor=1.0)
        with pytest.raises(RuntimeError, match="overflow"):
            nl.rebuild(pos * 0.5)  # compress: many more pairs inside the cutoff


def _policy_list(
    *,
    skin: float = 0.0,
    every: int = 1,
    delay: int = 0,
    check: bool = True,
    cutoff: float = 3.5,
    capacity_factor: float = 1.35,
) -> tuple[NeighborList, torch.Tensor]:
    """A policy-configured list over the 4x4x4 lattice, plus its positions.

    The shared fixture of the policy suite: 64 atoms in a 12 A cube, whose
    perpendicular half-width is 6.0 A and whose neighbour counts are exact
    integers from crystallography (6 at 3.0 A, 12 at 4.2426 A, 8 at 5.196 A).
    """
    pos, cell = _lattice(n_side=4, spacing=3.0)
    nl = NeighborList(
        cell=cell,
        cutoff=cutoff,
        positions=pos,
        skin=skin,
        every=every,
        delay=delay,
        check=check,
        capacity_factor=capacity_factor,
    )
    return nl, pos


def _displaced(pos: torch.Tensor, distance: float, *, atom: int = 0) -> torch.Tensor:
    """``pos`` with one atom translated ``distance`` A along x.

    Raw and unwrapped, as an MD trajectory drifts: the displacement test the
    policy runs is a raw difference, never a minimum image.
    """
    moved = pos.clone()
    moved[atom, 0] += distance
    return moved


class _Frame(NamedTuple):
    """One observed step: the evaluated configuration and the list behind it."""

    pos: torch.Tensor
    edge_index: torch.Tensor
    shifts: torch.Tensor
    total: torch.Tensor
    forces: torch.Tensor


class _PolicyForceField(LennardJonesCutForceField):
    """Drive the rebuild *policy* from the force-evaluation seam.

    A three-line preview of link 07's wiring: ``MD(rebuild_every=1)`` calls
    ``rebuild_neighbors`` once per force evaluation, *at the positions being
    evaluated*, so handing that call to :meth:`NeighborList.update` puts the
    gate exactly where the integrator will put it.
    """

    def rebuild_neighbors(self, pos: torch.Tensor) -> None:
        """Ask the policy instead of forcing a rebuild."""
        self.neighbors.update(pos)


class _FrameRecorder(MDHook):
    """Capture ``obs.pos`` together with the live list it was evaluated against."""

    def __init__(self, neighbors: NeighborList) -> None:
        self._neighbors = neighbors
        self.frames: list[_Frame] = []

    def on_step_end(self, runner: MDRunner, step: int, obs: MDObservables) -> None:
        """Snapshot the step; the live edges are the ones ``obs.forces`` used."""
        n = self._neighbors.num_edges
        self.frames.append(
            _Frame(
                pos=obs.pos.detach().clone(),
                edge_index=self._neighbors.edge_index[:n].clone(),
                shifts=self._neighbors.shifts[:n].clone(),
                total=obs.total.detach().clone(),
                forces=obs.forces.detach().clone(),
            )
        )


def _run_lj_lattice(
    *, skin: float, every: int = 1, delay: int = 0, check: bool = True, n_steps: int = 100
) -> tuple[NeighborList, list[_Frame]]:
    """NVE argon over the 64-atom lattice, driving the policy once per force eval.

    Deterministic CPU float64 throughout: seeded Maxwell-Boltzmann velocities,
    gamma = 0, no wall clock, no filesystem, no network. Argon in (amu, A, fs):
    eps = 0.0103 eV, sigma = 2.5 A, cutoff = 3.5 A, m = 39.95 amu, dt = 4 fs.

    ``capacity_factor=2.5`` is measured, not defensive: at ``skin=0.5`` this run
    reaches 600 live edges against the 519 rows the default 1.35 would allocate
    from the initial 384, and the overflow guard is not what these tests pin.
    """
    pos, cell = _lattice(n_side=4, spacing=3.0)
    neighbors = NeighborList(
        cell=cell,
        cutoff=3.5,
        positions=pos,
        skin=skin,
        every=every,
        delay=delay,
        check=check,
        capacity_factor=2.5,
    )
    force = _PolicyForceField(
        epsilon=0.0103 / EV_PER_AMU_A2_FS2,  # argon well depth, eV -> amu A^2/fs^2
        sigma=2.5,
        neighbors=neighbors,
        cutoff=3.5,
    )
    recorder = _FrameRecorder(neighbors)
    velocities = MaxwellBoltzmann(39.95, n_atoms=64).sample(300.0, seed=0)
    md = MD(
        force,
        mass=39.95,
        dt=4.0,
        gamma=0.0,
        dtype=torch.float64,
        rebuild_every=1,
        hooks=[recorder],
    )
    md.set_potential_dtype(torch.float64)
    md.run(pos, velocities, n_steps, chunk=1)
    return neighbors, recorder.frames


def _reference_pairs(
    pos: torch.Tensor, cell: torch.Tensor, cutoff: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The exact O(N^2) minimum-image pair set within ``cutoff`` — the oracle.

    Computed from fractional coordinates with ``round`` to the nearest image
    (exact for a cubic cell), so it holds for the unwrapped positions an MD run
    drifts into. Shares no code with the neighbour list under test.

    Returns:
        ``(source, target, distance)`` for every **ordered** pair inside
        ``cutoff``, with ``distance = ||pos[target] - pos[source] + shift||``.
    """
    fractional = pos @ torch.linalg.inv(cell)
    delta = fractional.unsqueeze(0) - fractional.unsqueeze(1)  # [i, j] = frac[j] - frac[i]
    delta = delta - torch.round(delta)
    distance = torch.linalg.norm(delta @ cell, dim=-1)
    inside = (distance < cutoff) & ~torch.eye(pos.shape[0], dtype=torch.bool)
    source, target = torch.nonzero(inside, as_tuple=True)
    return source, target, distance[source, target]


class TestNeighborListPolicy:
    """Verlet skin + LAMMPS ``neigh_modify every/delay/check`` rebuild policy.

    Reference:
        ``lammps/lammps`` develop, ``src/neighbor.cpp`` — ``Neighbor::decide``
        (2408-2424), ``Neighbor::check_distance`` (2438-2490), ``Neighbor::init``.
        K. Nordlund, *Introduction to molecular dynamics simulations*, lecture 3,
        for the half-skin two-atom criterion.
    """

    # --- construction contract: r_build derivation, sizing, guards ----------

    def test_build_radius_is_the_cutoff_plus_the_skin(self):
        """``cutoff`` stays the *interaction* cutoff every consumer means, and
        ``r_build = cutoff + skin`` is the derived build radius."""
        nl, _ = _policy_list(skin=1.5)
        assert nl.cutoff == 3.5
        assert nl.skin == 1.5
        assert nl.r_build == pytest.approx(5.0, abs=1e-12)

    def test_build_radius_is_not_settable(self):
        """Derived, never assigned: a writable ``r_build`` could drift out of
        step with the capacity, the kernel radius and the half-width guard that
        were all sized from it at construction."""
        nl, _ = _policy_list(skin=1.5)
        with pytest.raises(AttributeError):
            setattr(nl, "r_build", 6.0)

    def test_skin_extends_the_build_to_the_second_neighbour_shell(self):
        """Crystallography, not a fit: at ``r_build = 5.0 A`` every atom of the
        simple-cubic lattice sees 6 neighbours at 3.0 A and 12 at 3*sqrt(2) =
        4.2426 A (the 8 body-diagonal ones at 5.196 A stay out), so the
        bidirectional list holds 64 * 18 = 1152 edges."""
        nl, _ = _policy_list(skin=1.5)
        assert nl.num_edges == 1152

    def test_zero_skin_builds_only_the_first_shell(self):
        """The same lattice at ``r_build = cutoff = 3.5 A``: 6 neighbours each,
        64 * 6 = 384 edges. The 1152/384 = 3.0x ratio is the direct measurement
        of the ``(1 + s/r_cut)^3`` growth the capacity has to absorb."""
        nl, _ = _policy_list(skin=0.0)
        assert nl.num_edges == 384

    def test_capacity_is_sized_from_the_build_radius(self):
        """Sizing from ``cutoff`` instead would allocate 519 rows for a list
        that starts with 1152 live edges — an overflow on the constructor's own
        build, before a single MD step."""
        nl, _ = _policy_list(skin=1.5)
        assert nl.capacity >= math.ceil(1.35 * 1152)

    def test_buffer_shapes_stay_the_capacity_under_a_skin(self):
        """The skin must not cost the constant shapes a CUDA graph needs."""
        nl, _ = _policy_list(skin=1.5)
        assert nl.edge_index.shape == (nl.capacity, 2)
        assert nl.shifts.shape == (nl.capacity, 3)

    def test_a_fresh_list_reports_no_rebuild_history(self):
        """Defaults are the pre-skin behaviour: ``skin=0``, and the
        constructor's initial build is not counted as a rebuild."""
        pos, cell = _lattice(n_side=4, spacing=3.0)
        nl = NeighborList(cell=cell, cutoff=3.5, positions=pos)
        assert (nl.skin, nl.ago, nl.rebuild_count, nl.ndanger) == (0.0, 0, 0, 0)

    def test_to_casts_the_displacement_reference(self):
        """``_x_hold`` must follow ``shifts`` / ``cell`` through a cast.

        Asserted on the buffer dtype rather than through behaviour on purpose:
        a float64 ``_x_hold`` differenced against float32 positions *promotes
        silently* — no error, just a mixed-precision comparison nobody asked
        for — so the dtype is the only observable.
        """
        nl, pos = _policy_list(skin=1.5)
        nl.to(torch.float32)
        assert nl._x_hold.dtype == torch.float32
        assert nl.update(_displaced(pos.to(torch.float32), 1.0)) is True

    def test_rejects_a_skin_that_pushes_the_build_past_half_the_cell(self):
        """The link-01 half-width guard, re-derived on ``r_build``: ``cutoff =
        3.5 A`` alone is admissible in the 12 A cube (half-width 6.0 A), but
        ``skin = 3.0`` makes ``r_build = 6.5 A``, past which the kernel's
        minimum-image reduction silently drops pairs inside the cutoff."""
        pos, cell = _lattice(n_side=4, spacing=3.0)
        NeighborList(cell=cell, cutoff=3.5, positions=pos)  # the cutoff alone passes
        with pytest.raises(ValueError, match="r_build"):
            NeighborList(cell=cell, cutoff=3.5, positions=pos, skin=3.0)

    def test_rejects_a_skin_that_reaches_the_dead_edge_shift(self):
        """Dead padding edges sit at ``DEAD_EDGE_CUTOFF_FACTOR * cutoff = 10x``
        the cutoff; a skin of ``9x`` puts them exactly on the build radius,
        where they stop being inert and start being counted as real pairs. The
        60 A cell keeps the half-width guard — checked first — out of the way,
        so this pins the dead-edge assertion specifically."""
        pos, _ = _lattice(n_side=4, spacing=3.0)
        cell = torch.eye(3, dtype=torch.float64) * 60.0
        with pytest.raises(ValueError, match="dead"):
            NeighborList(cell=cell, cutoff=1.0, positions=pos, skin=9.0)

    @pytest.mark.parametrize(
        ("skin", "every", "delay"),
        [(-0.1, 1, 0), (0.0, 0, 0), (0.0, -1, 0), (0.0, 1, -1)],
        ids=["negative-skin", "zero-every", "negative-every", "negative-delay"],
    )
    def test_rejects_out_of_domain_policy_parameters(self, skin: float, every: int, delay: int):
        """Each arm has a domain; a silently clamped one disables the gate."""
        pos, cell = _lattice(n_side=4, spacing=3.0)
        with pytest.raises(ValueError):
            NeighborList(cell=cell, cutoff=3.5, positions=pos, skin=skin, every=every, delay=delay)

    def test_rejects_a_delay_that_is_not_a_multiple_of_every(self):
        """LAMMPS ``Neighbor::init`` parity: with ``every=4, delay=10`` the
        danger threshold ``max(every, delay) = 10`` is an ``ago`` the gate never
        permits (10 % 4 != 0), so ``ndanger`` could never fire and the
        correctness alarm would be silently dead."""
        pos, cell = _lattice(n_side=4, spacing=3.0)
        with pytest.raises(ValueError, match="multiple"):
            NeighborList(cell=cell, cutoff=3.5, positions=pos, every=4, delay=10)

    def test_accepts_a_delay_that_is_a_multiple_of_every(self):
        """The guard rejects a non-multiple, not a coarse gate as such."""
        nl, _ = _policy_list(every=4, delay=8)
        assert (nl.every, nl.delay, nl.ago) == (4, 8, 0)

    def test_accepts_zero_delay_under_any_every(self):
        """``0 % every == 0`` for every legal ``every``, so the default
        ``delay=0`` is always LAMMPS-legal — the guard must not read a falsy
        delay as unset and reject it."""
        nl, _ = _policy_list(every=7, delay=0)
        assert (nl.every, nl.delay) == (7, 0)

    # --- gating arithmetic (LAMMPS ``Neighbor::decide``) --------------------

    def test_the_gate_is_conjunctive_and_first_permits_ago_eight(self):
        """``every=4, delay=8``: both arms must agree. A displacement four times
        the half-skin cannot pull a rebuild forward — ``ago = 4`` clears
        ``ago % every`` but not ``ago >= delay``, and nothing before 8 clears
        both."""
        nl, pos = _policy_list(skin=1.0, every=4, delay=8)
        moved = _displaced(pos, 2.0)
        assert [nl.update(moved) for _ in range(8)] == [False] * 7 + [True]
        assert (nl.rebuild_count, nl.ago) == (1, 0)

    def test_the_every_arm_alone_first_permits_ago_four(self):
        """``delay=0`` leaves the cadence arm as the only gate."""
        nl, pos = _policy_list(skin=1.0, every=4, delay=0)
        moved = _displaced(pos, 2.0)
        assert [nl.update(moved) for _ in range(4)] == [False, False, False, True]

    def test_the_delay_arm_alone_first_permits_ago_ten(self):
        """``every=1`` leaves the delay arm as the only gate."""
        nl, pos = _policy_list(skin=1.0, every=1, delay=10)
        moved = _displaced(pos, 2.0)
        assert [nl.update(moved) for _ in range(10)] == [False] * 9 + [True]

    def test_a_forced_rebuild_rephases_the_schedule(self):
        """``ago`` counts from the last build, not from an absolute step index:
        after two blocked updates a forced ``rebuild`` restarts the clock, so
        the next permitted opportunity is four updates later, not two."""
        nl, pos = _policy_list(skin=1.0, every=4, delay=0)
        moved = _displaced(pos, 2.0)
        assert [nl.update(moved) for _ in range(2)] == [False, False]
        nl.rebuild(pos)
        assert nl.ago == 0
        assert [nl.update(moved) for _ in range(4)] == [False, False, False, True]

    def test_a_displacement_of_exactly_half_the_skin_does_not_rebuild(self):
        """LAMMPS compares with a strict ``>``: at ``max_i d_i == s/2`` the
        two-atom criterion ``d_(1) + d_(2) <= s`` still holds, so the list is
        still provably complete and a rebuild would be wasted work."""
        nl, pos = _policy_list(skin=1.0)
        assert nl.update(_displaced(pos, 0.5)) is False
        assert nl.rebuild_count == 0

    def test_a_displacement_just_past_half_the_skin_rebuilds(self):
        """The other side of the same strict comparison, one ulp-scale step
        away: 0.5 + 1e-9 A is no longer covered by the completeness proof."""
        nl, pos = _policy_list(skin=1.0)
        assert nl.update(_displaced(pos, 0.5 + 1e-9)) is True
        assert nl.rebuild_count == 1

    def test_zero_skin_holds_the_list_when_nothing_moved(self):
        """The degenerate limit is not "rebuild unconditionally": with
        ``max_d2 == 0`` and a strict ``>``, the list is already exactly right,
        so the bookkeeping says so."""
        nl, pos = _policy_list(skin=0.0)
        assert nl.update(pos.clone()) is False
        assert nl.rebuild_count == 0

    def test_zero_skin_rebuilds_on_any_motion(self):
        """``half_skin_sq = 0`` reproduces today's rebuild-every-ask behaviour
        edge for edge: any nonzero displacement rebuilds."""
        nl, pos = _policy_list(skin=0.0)
        assert nl.update(_displaced(pos, 1e-6)) is True
        assert nl.rebuild_count == 1

    def test_an_unchecked_policy_rebuilds_on_cadence_alone(self):
        """``check=False`` buys speed by dropping the distance test entirely:
        every permitted opportunity rebuilds even though nothing has moved."""
        nl, pos = _policy_list(skin=1.0, every=3, check=False)
        frozen = pos.clone()
        assert [nl.update(frozen) for _ in range(6)] == [False, False, True] * 2
        assert nl.rebuild_count == 2

    def test_ndanger_counts_a_rebuild_at_the_first_permitted_opportunity(self):
        """``every=1, delay=0`` puts ``_danger_ago`` at 1, so a rebuild that
        fires immediately may already have been overdue on the step before —
        which is exactly what the counter is for."""
        nl, pos = _policy_list(skin=1.0)
        assert nl.update(_displaced(pos, 1.0)) is True
        assert nl.ndanger == 1
        assert nl.update(_displaced(pos, 2.0)) is True
        assert nl.ndanger == 2

    def test_ndanger_stays_zero_when_the_rebuild_was_not_overdue(self):
        """``every=2, delay=6`` puts ``_danger_ago`` at 6. The half-skin is
        crossed only *after* that first opportunity, so the rebuild lands at
        ``ago = 8`` (7 % 2 != 0) and nothing was missed."""
        nl, pos = _policy_list(skin=1.0, every=2, delay=6)
        small, large = _displaced(pos, 0.2), _displaced(pos, 1.0)
        assert [nl.update(small) for _ in range(6)] == [False] * 6
        assert [nl.update(large) for _ in range(2)] == [False, True]
        assert (nl.rebuild_count, nl.ndanger) == (1, 0)

    def test_ndanger_fires_at_the_lammps_threshold(self):
        """The same coarse gate, but the half-skin is already crossed at the
        first permitted opportunity: the rebuild lands at ``ago = 6 =
        max(every, delay)``, ``neighbor.cpp:2488`` verbatim, on a LAMMPS-legal
        configuration."""
        nl, pos = _policy_list(skin=1.0, every=2, delay=6)
        moved = _displaced(pos, 1.0)
        assert [nl.update(moved) for _ in range(6)] == [False] * 5 + [True]
        assert (nl.rebuild_count, nl.ndanger) == (1, 1)

    def test_a_full_cell_translation_is_refused_as_unwrapped(self):
        """Frozen shifts plus a raw displacement test are correct only while
        positions stay unwrapped. A 12 A jump — one full cell vector, past the
        6.0 A half-width — means mid-run wrapping, a changed cell, or a blown-up
        trajectory; all three are fatal and none is recoverable."""
        nl, pos = _policy_list(skin=1.0)
        with pytest.raises(RuntimeError, match="unwrapped"):
            nl.update(_displaced(pos, 12.0))

    def test_an_unchecked_policy_skips_the_unwrapped_guard(self):
        """Documented consequence, pinned so the trade stays visible: the guard
        lives inside the displacement branch, so ``check=False`` switches off
        the invariant alarm along with the criterion."""
        nl, pos = _policy_list(skin=1.0, check=False)
        assert nl.update(_displaced(pos, 12.0)) is True

    # --- protocol -----------------------------------------------------------

    def test_a_skinned_list_satisfies_the_widened_protocol(self):
        """Widening ``NeighborStrategy`` must not push its own implementation
        out of the contract."""
        nl, _ = _policy_list(skin=1.5)
        assert isinstance(nl, NeighborStrategy)

    def test_a_list_without_the_policy_members_fails_the_protocol(self):
        """The widening is what lets ``forcefield.py`` drop its
        ``getattr(neighbors, "cutoff", None)`` duck-read, so the protocol has to
        actually *require* ``cutoff`` / ``skin`` / ``update``: a stub carrying
        only the buffer members must no longer pass."""

        class _BufferOnly:
            edge_index: torch.Tensor = torch.zeros(1, 2, dtype=torch.long)
            shifts: torch.Tensor = torch.zeros(1, 3)
            num_edges: int = 0
            capacity: int = 1

            def rebuild(self, positions: torch.Tensor) -> None:
                """No-op build."""

            def to(
                self,
                device: torch.device | str | torch.dtype | None = None,
                dtype: torch.dtype | None = None,
            ) -> "_BufferOnly":
                return self

        assert not isinstance(_BufferOnly(), NeighborStrategy)

    def test_the_interaction_cutoff_is_the_bar_not_the_build_radius(self):
        """A skinned list holds pairs out to ``r_build = 5.0 A``, but they are
        only *complete* out to ``cutoff`` between rebuilds — so a 5.0 A
        interaction cutoff over this list is precisely the silent truncation
        that check exists to prevent."""
        nl, _ = _policy_list(skin=1.5)
        with pytest.raises(ValueError):
            LennardJonesCutForceField(epsilon=1.0, sigma=2.5, neighbors=nl, cutoff=5.0)

    # --- trajectory falsification (NVE argon over the lattice) -------------

    def test_every_pair_within_the_cutoff_stays_in_the_live_list(self):
        """PRIMARY falsification: a gated list must never miss a pair.

        At every step the exact minimum-image O(N^2) pair set within the
        interaction cutoff — recomputed from the very configuration the forces
        were evaluated at — must be a subset of the live list, and each pair's
        list-reconstructed distance ``||pos[t] - pos[s] + shift||`` must match
        the reference. The distance half is not redundant: a stale *shift* on a
        surviving index pair is the frozen-shift failure mode, and an
        index-subset check alone would sail straight past it.
        """
        _, cell = _lattice(n_side=4, spacing=3.0)
        _, frames = _run_lj_lattice(skin=0.5)
        n_atoms = 64
        for step, frame in enumerate(frames):
            source, target, reference = _reference_pairs(frame.pos, cell, 3.5)
            rows = torch.full((n_atoms * n_atoms,), -1, dtype=torch.long)
            live = frame.edge_index
            rows[live[:, 0] * n_atoms + live[:, 1]] = torch.arange(live.shape[0])
            found = rows[source * n_atoms + target]
            missing = int((found < 0).sum())
            assert missing == 0, f"step {step}: {missing} pairs inside the cutoff are not listed"
            reconstructed = torch.linalg.norm(
                frame.pos[target] - frame.pos[source] + frame.shifts[found], dim=-1
            )
            torch.testing.assert_close(reconstructed, reference, atol=1e-9, rtol=0)

    def test_a_skinned_policy_matches_rebuilding_every_force_evaluation(self):
        """Policy equivalence: the gated ``skin=0.5`` run and a ``skin=0.0`` run
        that rebuilds at every force evaluation must trace the same physics.
        Compared at ``atol=1e-10, rtol=0`` rather than bitwise on purpose — the
        masked-zero skin edges reorder the ``index_add_`` accumulation."""
        _, gated = _run_lj_lattice(skin=0.5)
        _, every_eval = _run_lj_lattice(skin=0.0)
        assert len(gated) == len(every_eval) == 100
        for step, (a, b) in enumerate(zip(gated, every_eval, strict=True)):
            torch.testing.assert_close(a.total, b.total, atol=1e-10, rtol=0, msg=f"step {step}")
            torch.testing.assert_close(a.forces, b.forces, atol=1e-10, rtol=0, msg=f"step {step}")

    def test_the_standard_run_reports_no_dangerous_builds(self):
        """``skin=0.5`` gives a half-skin of 0.25 A against ~0.01 A of motion
        per step, so no rebuild is ever overdue. ``ndanger`` is the cheapest
        correctness alarm available and must stay silent on a sane run — while
        the gate itself stays alive (``rebuild_count > 0``)."""
        neighbors, _ = _run_lj_lattice(skin=0.5)
        assert neighbors.ndanger == 0
        assert neighbors.rebuild_count > 0

    def test_rebuild_count_falls_as_the_skin_grows(self):
        """The point of the skin, measured over the same trajectory. The strict
        inequality at the ends is what catches a dead or inverted gate — plain
        non-increasing monotonicity is also satisfied by a policy that never
        rebuilds at all."""
        counts = [_run_lj_lattice(skin=skin)[0].rebuild_count for skin in (0.0, 0.25, 0.5, 1.0)]
        assert counts == sorted(counts, reverse=True)
        assert counts[-1] < counts[0]


def _batch(pos: torch.Tensor, cell: torch.Tensor | None = None) -> TensorDict:
    """A single-system MD working batch over ``pos``, optionally carrying a cell.

    The same shape ``_periodic_template`` builds in ``test_forcefield.py`` — the
    batch :class:`~molix.md.forcefield.PeriodicPotentialForceField` binds into —
    plus the optional ``("graphs", "cell")`` the bind path validates.

    ``graphs`` carries ``batch_size=[]`` on purpose: the container must not be
    the thing that decides the cell's leading dimension, or ``(3, 3)`` and
    ``(1, 3, 3)`` could not both be stored and the accept/refuse decision under
    test would be made by the fixture instead of by ``build``.

    Args:
        pos: Positions ``(N, 3)`` in Angstrom.
        cell: Optional cell vectors in Angstrom, any shape — the point of
            several of these tests is that ``build`` judges the shape.

    Returns:
        A batch ``TensorDict`` with root ``batch_size=[]``.
    """
    n = int(pos.shape[0])
    data: dict[str, TensorDict] = {
        "atoms": TensorDict(
            {
                "pos": pos,
                "Z": torch.ones(n, dtype=torch.long),
                "batch": torch.zeros(n, dtype=torch.long),
            },
            batch_size=[n],
        )
    }
    if cell is not None:
        data["graphs"] = TensorDict({"cell": cell}, batch_size=[])
    return TensorDict(data, batch_size=[])


class TestNeighborListBind:
    """``build(batch)`` / ``update(batch)`` — the TensorDict side of the list.

    The list owns ``edges`` once bound: ``build`` refreshes the buffers at
    ``batch["atoms", "pos"]``, validates that the batch describes the system the
    list was constructed for, writes the *live* buffers into ``batch["edges"]``
    **by reference** and hands the same batch back so it composes with the
    repo-wide ``forward(td) -> td`` convention.
    """

    # --- happy path: bind -------------------------------------------------

    def test_build_returns_the_argument_itself(self):
        """``potential(nl.build(batch))`` only composes if the return is the
        argument — a copy would leave the caller holding an unbound batch."""
        nl, pos = _policy_list(skin=1.5, capacity_factor=4.0)
        batch = _batch(pos)
        assert nl.build(batch) is batch

    def test_build_binds_the_live_buffers_by_reference(self):
        """The load-bearing property of the whole link: **identity**, not equality.

        A container that copied on assignment would give the potential a
        snapshot, so every later in-place rebuild would be invisible and the PES
        would silently freeze. This assertion is that alarm.
        """
        nl, pos = _policy_list(skin=1.5, capacity_factor=4.0)
        batch = nl.build(_batch(pos))
        assert batch["edges", "edge_index"] is nl.edge_index
        assert batch["edges", "shifts"] is nl.shifts

    def test_bound_edges_span_the_capacity_not_the_live_count(self):
        """Fixed-capacity buffers go in whole: the tail of dead padding edges is
        part of the contract (constant shapes for a CUDA graph), so the bound
        namespace is sized by ``capacity``, never by ``num_edges``."""
        nl, pos = _policy_list(skin=1.5, capacity_factor=4.0)
        batch = nl.build(_batch(pos))
        assert batch["edges"].batch_size == torch.Size([nl.capacity])

    def test_bound_edges_hold_exactly_the_index_and_the_shifts(self):
        """The bind writes the two keys the periodic potentials read and nothing
        else — a derived ``edge_diff`` / ``edge_dist`` written here would be a
        stale straight-through value the moment an atom moves."""
        nl, pos = _policy_list(skin=1.5, capacity_factor=4.0)
        batch = nl.build(_batch(pos))
        assert set(batch["edges"].keys()) == {"edge_index", "shifts"}

    def test_build_is_never_counted_as_a_rebuild(self):
        """``rebuild_count`` means "rebuilds driven during the run". ``build`` is
        a binding operation — it also runs on every ``.to()`` re-sync — so
        counting it would make a dtype cast look like physics."""
        nl, pos = _policy_list(skin=1.5, capacity_factor=4.0)
        assert nl.rebuild_count == 0
        nl.build(_batch(pos))
        assert nl.rebuild_count == 0
        nl.rebuild(pos)
        nl.build(_batch(pos))
        assert nl.rebuild_count == 1

    def test_build_restarts_the_policy_clock(self):
        """The buffers are fresh after a bind, so the ``ago`` clock the
        ``every`` / ``delay`` gate runs on has to start there too — otherwise the
        next ``update`` measures a staleness that was just eliminated."""
        nl, pos = _policy_list(skin=1.0, every=4, capacity_factor=4.0)
        frozen = pos.clone()
        assert [nl.update(frozen) for _ in range(2)] == [False, False]
        assert nl.ago == 2
        nl.build(_batch(pos))
        assert nl.ago == 0

    def test_build_uses_the_positions_in_the_batch(self):
        """The bind is also a build: it must land on the batch's geometry, not
        re-emit the constructor's. Measured against a list constructed directly
        at the dilated lattice — the same kernel, so any difference is the bind
        having built at the wrong positions."""
        pos, cell = _lattice(n_side=4, spacing=3.0)
        nl, _ = _policy_list(skin=1.5, capacity_factor=4.0)
        assert nl.num_edges == 1152  # crystallography, link 04
        dilated = pos * 1.05
        reference = NeighborList(
            cell=cell, cutoff=3.5, positions=dilated, skin=1.5, capacity_factor=4.0
        )
        nl.build(_batch(dilated))
        assert reference.num_edges != 1152
        assert nl.num_edges == reference.num_edges

    # --- liveness of the tie ----------------------------------------------

    def test_a_forced_rebuild_is_visible_through_the_bound_batch(self):
        """An in-place rebuild must reach the batch with no re-binding at all.

        The value read is not redundant with the identity check: a stale
        *shift* on a surviving index pair is the frozen-shift failure mode, and
        it is invisible to an index comparison alone.
        """
        pos, cell = _lattice(n_side=4, spacing=3.0)
        nl, _ = _policy_list(skin=0.0, capacity_factor=4.0)
        batch = nl.build(_batch(pos))
        before = nl.num_edges
        compressed = pos * 0.8  # second-neighbour pairs enter the cutoff
        nl.rebuild(compressed)
        reference = NeighborList(cell=cell, cutoff=3.5, positions=compressed, capacity_factor=4.0)
        assert nl.num_edges != before
        assert nl.num_edges == reference.num_edges
        assert batch["edges", "edge_index"] is nl.edge_index
        assert batch["edges", "shifts"] is nl.shifts
        n = nl.num_edges
        assert torch.equal(batch["edges", "edge_index"][:n], reference.edge_index[:n])
        assert torch.equal(batch["edges", "shifts"][:n], reference.shifts[:n])

    def test_a_policy_rebuild_is_visible_through_the_bound_batch(self):
        """Same liveness, driven through the batch entry point instead: the
        per-step idiom is ``nl.update(batch)``, so that is the path that has to
        keep the tie alive."""
        pos, cell = _lattice(n_side=4, spacing=3.0)
        nl, _ = _policy_list(skin=0.0, capacity_factor=4.0)
        batch = nl.build(_batch(pos))
        before = nl.num_edges
        compressed = pos * 0.8
        assert nl.update(_batch(compressed)) is True
        reference = NeighborList(cell=cell, cutoff=3.5, positions=compressed, capacity_factor=4.0)
        assert nl.num_edges != before
        assert nl.num_edges == reference.num_edges
        assert batch["edges", "edge_index"] is nl.edge_index
        assert batch["edges", "shifts"] is nl.shifts
        n = nl.num_edges
        assert torch.equal(batch["edges", "edge_index"][:n], reference.edge_index[:n])
        assert torch.equal(batch["edges", "shifts"][:n], reference.shifts[:n])

    def test_a_bare_cast_severs_the_tie(self):
        """The documented consequence, pinned so the trade stays visible.

        ``to(dtype)`` rebinds ``shifts`` to a new tensor, so a batch bound
        beforehand keeps pointing at the old one. Auto-re-binding from inside
        ``to()`` was rejected — it would make the list hold a reference to a
        batch it does not own — so the **owner** re-binds; the force-field half
        of this trade is pinned in ``test_forcefield.py``.
        """
        nl, pos = _policy_list(skin=1.5, capacity_factor=4.0)
        batch = nl.build(_batch(pos))
        nl.to(torch.float32)
        assert batch["edges", "shifts"] is not nl.shifts
        assert batch["edges", "shifts"].dtype == torch.float64
        assert nl.shifts.dtype == torch.float32

    # --- the list owns ``edges`` once bound --------------------------------

    def test_build_replaces_a_stale_edges_namespace_wholesale(self):
        """Whatever was under ``edges`` is dropped, not merged.

        A precomputed ``edge_diff`` / ``edge_dist`` pair — exactly what
        ``PotentialForceField._STALE_EDGE_KEYS`` strips at construction — would
        be used straight through by a potential and freeze the PES, and a
        surviving shorter ``edge_index`` would disagree with the capacity.
        """
        nl, pos = _policy_list(skin=1.5, capacity_factor=4.0)
        batch = _batch(pos)
        batch["edges"] = TensorDict(
            {
                "edge_index": torch.zeros(3, 2, dtype=torch.long),
                "edge_diff": torch.zeros(3, 3, dtype=torch.float64),
                "edge_dist": torch.zeros(3, dtype=torch.float64),
            },
            batch_size=[3],
        )
        nl.build(batch)
        assert set(batch["edges"].keys()) == {"edge_index", "shifts"}
        assert batch["edges"].batch_size == torch.Size([nl.capacity])

    # --- validation: the batch must describe *this* system -----------------

    def test_build_without_positions_names_the_missing_key(self):
        """``KeyError(('atoms', 'pos'))`` from three frames deep is the classic
        two-tier data-contract confusion; the bind names the key it wanted."""
        nl, pos = _policy_list(skin=1.5, capacity_factor=4.0)
        n = int(pos.shape[0])
        batch = TensorDict(
            {"atoms": TensorDict({"Z": torch.ones(n, dtype=torch.long)}, batch_size=[n])},
            batch_size=[],
        )
        with pytest.raises(ValueError, match="pos") as excinfo:
            nl.build(batch)
        assert "atoms" in str(excinfo.value)

    def test_build_rejects_a_different_atom_count(self):
        """A different ``N`` is a different system: ``_x_hold.copy_`` would raise
        somewhere unhelpful, and the capacity was sized for the original."""
        nl, _ = _policy_list(skin=1.5, capacity_factor=4.0)
        eight_atoms, _ = _lattice(n_side=2, spacing=3.0)
        with pytest.raises(ValueError, match=r"\b8\b") as excinfo:
            nl.build(_batch(eight_atoms))
        assert "64" in str(excinfo.value)

    def test_build_rejects_recast_positions(self):
        """No silent cast: a float32 ``pos`` differenced against a float64
        ``_x_hold`` promotes silently — a mixed-precision comparison nobody
        asked for — so the bind names both sides and the owner casts."""
        nl, pos = _policy_list(skin=1.5, capacity_factor=4.0)
        with pytest.raises(ValueError, match="float32") as excinfo:
            nl.build(_batch(pos.to(torch.float32)))
        assert "float64" in str(excinfo.value)

    def test_build_rejects_positions_on_another_device(self):
        """The ``meta`` device stands in for a real second device: validation
        precedes any kernel call, so this needs no CUDA in CI."""
        nl, pos = _policy_list(skin=1.5, capacity_factor=4.0)
        elsewhere = torch.empty_like(pos, device="meta")
        with pytest.raises(ValueError, match="meta") as excinfo:
            nl.build(_batch(elsewhere))
        assert "cpu" in str(excinfo.value)

    def test_build_accepts_the_constructor_cell(self):
        """A batch may carry its cell; agreeing with the list's is the point of
        checking it. The constructor cell stays the **owner** — the batch's copy
        is validated, never adopted."""
        pos, cell = _lattice(n_side=4, spacing=3.0)
        nl, _ = _policy_list(skin=1.5, capacity_factor=4.0)
        before = nl.cell.clone()
        nl.build(_batch(pos, cell))
        assert nl.num_edges == 1152
        assert torch.equal(nl.cell, before)

    def test_build_accepts_a_single_system_batched_cell(self):
        """``(1, 3, 3)`` is what a collated batch's ``graphs`` namespace holds for
        one system, so the bind must read through the leading batch dim."""
        pos, cell = _lattice(n_side=4, spacing=3.0)
        nl, _ = _policy_list(skin=1.5, capacity_factor=4.0)
        before = nl.cell.clone()
        nl.build(_batch(pos, cell.unsqueeze(0)))
        assert nl.num_edges == 1152
        assert torch.equal(nl.cell, before)

    def test_build_rejects_a_disagreeing_cell(self):
        """A cell the list did not build against silently invalidates every
        frozen shift, so a disagreement is refused rather than adopted — 0.01 A
        is far below anything physical and far above the fp32 round-trip
        tolerance the check allows."""
        pos, cell = _lattice(n_side=4, spacing=3.0)
        nl, _ = _policy_list(skin=1.5, capacity_factor=4.0)
        before = nl.cell.clone()
        perturbed = cell.clone()
        perturbed[0, 0] += 0.01
        with pytest.raises(ValueError, match="cell"):
            nl.build(_batch(pos, perturbed))
        assert torch.equal(nl.cell, before)

    def test_build_rejects_a_multi_system_cell(self):
        """This list is single-system: one cell, one ``_x_hold``, one capacity.
        A ``B > 1`` batch has no meaningful semantics here and is refused by
        name rather than silently reduced to its first row."""
        pos, cell = _lattice(n_side=4, spacing=3.0)
        nl, _ = _policy_list(skin=1.5, capacity_factor=4.0)
        with pytest.raises(ValueError, match=r"\b2\b") as excinfo:
            nl.build(_batch(pos, cell.repeat(2, 1, 1)))
        assert "cell" in str(excinfo.value)

    # --- one ``update``, two input types -----------------------------------

    def test_the_batch_and_tensor_paths_decide_alike(self):
        """The dispatch is a type test at the top of one method, so both inputs
        must trace the same policy exactly: same decisions, same bookkeeping,
        same buffers. A schedule that both rebuilds and holds, so neither arm of
        the comparison is vacuous."""
        nl_tensor, pos = _policy_list(skin=1.0, capacity_factor=4.0)
        nl_batch, _ = _policy_list(skin=1.0, capacity_factor=4.0)
        schedule = [_displaced(pos, 0.1 * k) for k in range(1, 13)]
        by_tensor = [nl_tensor.update(step) for step in schedule]
        by_batch = [nl_batch.update(_batch(step)) for step in schedule]
        assert set(by_tensor) == {True, False}  # the schedule exercises both arms
        assert by_batch == by_tensor
        assert (nl_batch.ago, nl_batch.rebuild_count, nl_batch.ndanger) == (
            nl_tensor.ago,
            nl_tensor.rebuild_count,
            nl_tensor.ndanger,
        )
        assert nl_batch.num_edges == nl_tensor.num_edges
        assert torch.equal(nl_batch.edge_index, nl_tensor.edge_index)
        assert torch.equal(nl_batch.shifts, nl_tensor.shifts)

    def test_update_validates_a_batch_exactly_as_build_does(self):
        """Shared validation, not a second copy: a batch whose ``pos`` was
        re-cast must fail loud on the hot path too, instead of promoting
        silently against ``_x_hold`` for the rest of the run."""
        nl, pos = _policy_list(skin=1.0, capacity_factor=4.0)
        with pytest.raises(ValueError, match="float32") as excinfo:
            nl.update(_batch(pos.to(torch.float32)))
        assert "float64" in str(excinfo.value)

    def test_the_dispatch_has_no_twin_entry_points(self):
        """One method, two accepted types — ``update_td`` / ``update_pos`` twins
        are rejected by design: callers would have to know which one their
        driver holds, and the policy state would live behind two doors."""
        assert not hasattr(molix.md.neighbors, "update_td")
        assert not hasattr(molix.md.neighbors, "update_pos")
        assert not hasattr(NeighborList, "update_td")
        assert not hasattr(NeighborList, "update_pos")

    # --- protocol -----------------------------------------------------------

    def test_a_list_without_build_fails_the_widened_protocol(self):
        """``PeriodicPotentialForceField`` calls ``build`` through the
        ``NeighborStrategy`` annotation, so the protocol has to actually
        *require* it: a stub carrying every link-04 member but no ``build`` must
        no longer pass."""

        class _PolicyWithoutBuild:
            edge_index: torch.Tensor = torch.zeros(1, 2, dtype=torch.long)
            shifts: torch.Tensor = torch.zeros(1, 3)
            num_edges: int = 0
            capacity: int = 1
            cutoff: float = 3.5
            skin: float = 0.0

            def rebuild(self, positions: torch.Tensor) -> None:
                """No-op build."""

            def update(self, positions: TensorDict | torch.Tensor) -> bool:
                return False

            def to(
                self,
                device: torch.device | str | torch.dtype | None = None,
                dtype: torch.dtype | None = None,
            ) -> "_PolicyWithoutBuild":
                return self

        assert not isinstance(_PolicyWithoutBuild(), NeighborStrategy)
