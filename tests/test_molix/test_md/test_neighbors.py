"""Tests for molix.md.neighbors."""

import math
import re
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
from tests.test_molix.test_md.conftest import make_cubic_lattice


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
    pos, cell = make_cubic_lattice()
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
        # sorts into the "N" run instead of keeping the old "P" slot. Pinned
        # against ``NeighborStrategy`` (a permanent export) rather than the
        # ``NeighborListHook`` this chain's link 07 deletes.
        pascal = [name for name in names if not name.isupper()]
        assert pascal == sorted(pascal)
        assert names.index("NeighborList") < names.index("NeighborStrategy")

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
        pos, cell = make_cubic_lattice()
        with pytest.raises(ValueError, match="exceeds half the minimum perpendicular cell width"):
            NeighborList(cell=cell, cutoff=5.0, positions=pos)

    def test_accepts_orthorhombic_cutoff_just_below_half_the_cell(self):
        """Orthorhombic parity: for a cube ``w_i = ||a_i||``, so the bound stays
        ``4.500 A`` on the 9 A cell and ``4.4 A`` must still build."""
        pos, cell = make_cubic_lattice()
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
        pos, cell = make_cubic_lattice()
        with pytest.raises(ValueError):
            NeighborList(cell=cell.unsqueeze(0), cutoff=3.5, positions=pos)

    def test_overflow_raises_rather_than_truncating(self):
        """A truncated neighbour list is a silently wrong energy."""
        pos, cell = make_cubic_lattice()
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
    pos, cell = make_cubic_lattice(n_side=4, spacing=3.0)
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

    No cadence knob is passed: link 07 makes the *force field* declare that it
    owns a live list (``LennardJonesCutForceField.rebuilds_neighbors``), the
    integrator derive its static switch from that, and ``rebuild_neighbors``
    land on :meth:`NeighborList.update` — so the policy runs once per force
    evaluation, at the positions being evaluated, with no driver kwarg and no
    ``_PolicyForceField`` preview subclass in the way.

    ``capacity_factor=2.5`` is measured, not defensive: at ``skin=0.5`` this run
    reaches 600 live edges against the 519 rows the default 1.35 would allocate
    from the initial 384, and the overflow guard is not what these tests pin.
    """
    pos, cell = make_cubic_lattice(n_side=4, spacing=3.0)
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
    force = LennardJonesCutForceField(
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
        pos, cell = make_cubic_lattice(n_side=4, spacing=3.0)
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
        pos, cell = make_cubic_lattice(n_side=4, spacing=3.0)
        NeighborList(cell=cell, cutoff=3.5, positions=pos)  # the cutoff alone passes
        with pytest.raises(ValueError, match="r_build"):
            NeighborList(cell=cell, cutoff=3.5, positions=pos, skin=3.0)

    def test_rejects_a_skin_that_reaches_the_dead_edge_shift(self):
        """Dead padding edges sit at ``DEAD_EDGE_CUTOFF_FACTOR * cutoff = 10x``
        the cutoff; a skin of ``9x`` puts them exactly on the build radius,
        where they stop being inert and start being counted as real pairs. The
        60 A cell keeps the half-width guard — checked first — out of the way,
        so this pins the dead-edge assertion specifically."""
        pos, _ = make_cubic_lattice(n_side=4, spacing=3.0)
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
        pos, cell = make_cubic_lattice(n_side=4, spacing=3.0)
        with pytest.raises(ValueError):
            NeighborList(cell=cell, cutoff=3.5, positions=pos, skin=skin, every=every, delay=delay)

    def test_rejects_a_delay_that_is_not_a_multiple_of_every(self):
        """LAMMPS ``Neighbor::init`` parity: with ``every=4, delay=10`` the
        danger threshold ``max(every, delay) = 10`` is an ``ago`` the gate never
        permits (10 % 4 != 0), so ``ndanger`` could never fire and the
        correctness alarm would be silently dead."""
        pos, cell = make_cubic_lattice(n_side=4, spacing=3.0)
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
        _, cell = make_cubic_lattice(n_side=4, spacing=3.0)
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
        pos, cell = make_cubic_lattice(n_side=4, spacing=3.0)
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
        pos, cell = make_cubic_lattice(n_side=4, spacing=3.0)
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
        pos, cell = make_cubic_lattice(n_side=4, spacing=3.0)
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
        eight_atoms, _ = make_cubic_lattice(n_side=2, spacing=3.0)
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
        pos, cell = make_cubic_lattice(n_side=4, spacing=3.0)
        nl, _ = _policy_list(skin=1.5, capacity_factor=4.0)
        before = nl.cell.clone()
        nl.build(_batch(pos, cell))
        assert nl.num_edges == 1152
        assert torch.equal(nl.cell, before)

    def test_build_accepts_a_single_system_batched_cell(self):
        """``(1, 3, 3)`` is what a collated batch's ``graphs`` namespace holds for
        one system, so the bind must read through the leading batch dim."""
        pos, cell = make_cubic_lattice(n_side=4, spacing=3.0)
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
        pos, cell = make_cubic_lattice(n_side=4, spacing=3.0)
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
        pos, cell = make_cubic_lattice(n_side=4, spacing=3.0)
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


# ---------------------------------------------------------------------------
# The binned (cell-list) build path: fixtures, edge keys, comparison contract
# ---------------------------------------------------------------------------

#: ``(source, target, shift)`` with the shift rounded to 6 decimals — one live
#: edge, reduced to something hashable. Used both directed (as emitted) and
#: canonicalised low->high (see :func:`_canonical_keys`).
_EdgeKey = tuple[int, int, tuple[float, ...]]


def _directed_keys(nl: NeighborList) -> list[_EdgeKey]:
    """The live edges as hashable ``(source, target, shift)`` keys, as emitted.

    Only ``[0, num_edges)`` is read: the tail is dead padding, not edges.

    Shift components are integer combinations of the cell rows — exact values
    like ``12.0`` or ``-6.0`` — so rounding at 6 decimals absorbs the ~1e-13
    disagreement between two different minimum-image reductions without ever
    landing near a rounding boundary.

    Args:
        nl: A built list.

    Returns:
        One key per live edge, in buffer order. Duplicates here are real
        duplicates: the same directed edge emitted twice.
    """
    n = nl.num_edges
    sources = nl.edge_index[:n, 0].tolist()
    targets = nl.edge_index[:n, 1].tolist()
    shifts = nl.shifts[:n].tolist()
    return [
        (source, target, tuple(round(component, 6) for component in shift))
        for source, target, shift in zip(sources, targets, shifts, strict=True)
    ]


def _canonical_keys(nl: NeighborList) -> list[_EdgeKey]:
    """The live edges as orientation-free ``(low, high, shift)`` keys.

    Edge **order is not part of the contract** between the two build backends —
    the binned path emits bin-sorted edges, the kernel upper-triangle-sorted
    ones — so equality is compared as a *set*. The key must still separate
    periodic images, which is what the shift carries: for an edge ``(s, t, D)``
    the reverse edge is ``(t, s, -D)`` (``edge_diff = pos[t] - pos[s] + D``
    flips sign wholesale), so orienting the shift low->high makes the key
    independent of which way the edge was emitted, while keeping ``(i, j)``
    across the ``+x`` face distinct from ``(i, j)`` across the ``-x`` one.

    Note:
        Both paths emit a **full bidirectional** list, so every undirected pair
        contributes *two* live edges that share one canonical key: the unique
        canonical keys number ``num_edges / 2``, not ``num_edges``. The
        duplicate-free clause is therefore asserted on the *directed* keys (see
        :func:`_assert_paths_agree`), which is where a wrapped stencil's double
        emission would actually show up.

    Args:
        nl: A built list.

    Returns:
        One key per live edge, in buffer order.
    """
    keys: list[_EdgeKey] = []
    for source, target, shift in _directed_keys(nl):
        if source < target:
            keys.append((source, target, shift))
        else:
            keys.append((target, source, tuple(-component for component in shift)))
    return keys


def _closest_approach_to(pos: torch.Tensor, cell: torch.Tensor, radius: float) -> float:
    """``min_{i != j} |r_ij - radius|`` over minimum-image pairs, in Angstrom.

    The float-tie precondition of every binned-vs-kernel comparison. The two
    backends reduce to the minimum image differently — fractional rounding
    against the kernel's sequential subtraction — so a pair sitting *on*
    ``r_build`` could fall on either side of the closed ``r <= r_build`` filter
    for reasons that have nothing to do with the stencil. Every fixture asserts
    it keeps clear of that radius, so a failure is always a real disagreement.

    Computed independently of both paths, from fractional rounding, which is
    exact here: ``r_build <= min_i w_i / 2`` bounds the fractional offset of
    every in-range image by 1/2 (the lemma the binned path rests on).

    Args:
        pos: Positions ``(N, 3)`` in Angstrom, wrapped or not.
        cell: Cell vectors ``(3, 3)`` in Angstrom, one per row.
        radius: The radius to measure the closest approach to, in Angstrom.

    Returns:
        The smallest ``|r_ij - radius|`` over all ordered ``i != j`` pairs.
    """
    vectors = cell.detach().to(torch.float64)
    fractional = pos.detach().to(torch.float64) @ torch.linalg.inv(vectors)
    delta = fractional.unsqueeze(0) - fractional.unsqueeze(1)
    distance = torch.linalg.norm((delta - torch.round(delta)) @ vectors, dim=-1)
    off_diagonal = ~torch.eye(pos.shape[0], dtype=torch.bool)
    return float((distance[off_diagonal] - radius).abs().min())


def _assert_paths_agree(binned: NeighborList, kernel: NeighborList, pos: torch.Tensor) -> None:
    """The full equality contract between the binned build and the kernel oracle.

    Four clauses, all of them load-bearing:

    1. **Precondition** — no pair within 1e-9 A of ``r_build``, so no clause
       below can be decided by a float tie (:func:`_closest_approach_to`).
    2. **Counts** — ``num_edges`` equal. Set equality alone cannot see a
       duplicated edge; a wrapped stencil that emitted every pair twice would
       double this.
    3. **Duplicate-free** — every *directed* key occurs once on each path, i.e.
       ``len(set(directed)) == num_edges``, and the canonical keys come out at
       exactly ``num_edges / 2`` because both paths emit a full bidirectional
       list (``(s,t,D)`` and ``(t,s,-D)`` share one canonical key). This is the
       intrinsic check the aliasing fixture is built to trip, and it does not
       depend on the oracle.
    4. **Set equality** of the canonical keys — catches *missing* edges, which
       counts alone cannot.

    Args:
        binned: The list built with a ``bin`` grid.
        kernel: A list over the same geometry and parameters with ``bin=None``.
        pos: The positions both were built at ``(N, 3)`` in Angstrom.
    """
    margin = _closest_approach_to(pos, binned.cell, binned.r_build)
    assert margin > 1e-9, (
        f"fixture precondition violated: a pair sits {margin:.3e} A from r_build "
        f"{binned.r_build} A, where the closed r <= r_build filter is a float coin-flip"
    )
    assert binned.num_edges == kernel.num_edges
    directed_binned, directed_kernel = _directed_keys(binned), _directed_keys(kernel)
    assert len(set(directed_binned)) == binned.num_edges, "the binned path emitted a duplicate edge"
    assert len(set(directed_kernel)) == kernel.num_edges, "the kernel path emitted a duplicate edge"
    canonical_binned = set(_canonical_keys(binned))
    assert 2 * len(canonical_binned) == binned.num_edges, "the binned list is not bidirectional"
    assert canonical_binned == set(_canonical_keys(kernel))


def _pair_distances(nl: NeighborList, pos: torch.Tensor) -> list[tuple[int, int, float]]:
    """Live edges as a sorted ``(low, high, |displacement|)`` multiset, in Angstrom.

    Reconstructed the way a consumer does it — ``pos[t] - pos[s] + shift`` —
    so it reads the *stored* positions through the *stored* shifts. That makes
    it the observable that catches a build-time coordinate wrap leaking into
    either of them.

    Args:
        nl: A built list.
        pos: The positions it was built at ``(N, 3)`` in Angstrom.

    Returns:
        One entry per live edge, sorted (a multiset, duplicates kept).
    """
    n = nl.num_edges
    source, target = nl.edge_index[:n, 0], nl.edge_index[:n, 1]
    distance = torch.linalg.norm(pos[target] - pos[source] + nl.shifts[:n], dim=-1)
    return sorted(
        (min(s, t), max(s, t), round(d, 6))
        for s, t, d in zip(source.tolist(), target.tolist(), distance.tolist(), strict=True)
    )


def _jittered_box() -> tuple[torch.Tensor, torch.Tensor]:
    """512 atoms in a 24 A cube, jittered off the lattice — the pruning regime.

    The 8x8x8 simple-cubic lattice at 3.0 A displaced by +/-0.4 A uniform under
    ``torch.manual_seed(0)``. At ``r_build = 5.0 A`` the grid is 9x9x9 with
    ``k_i = 2``, so the stencil searches 125 of 729 bins: real pruning, and no
    lattice symmetry left for a broken stencil to hide behind. The closest any
    pair comes to ``r_build`` is 3.8e-4 A (measured), clear of the 1e-9 tie band.

    Returns:
        ``(positions (512, 3), cell (3, 3))`` in Angstrom, ``float64``.
    """
    pos, cell = make_cubic_lattice(n_side=8, spacing=3.0)
    torch.manual_seed(0)
    return pos + (torch.rand(pos.shape, dtype=torch.float64) * 2.0 - 1.0) * 0.4, cell


def _triclinic_forty() -> tuple[torch.Tensor, torch.Tensor]:
    """40 atoms in the golden triclinic cell — where the ``|f_i| <= 1/2`` lemma is tested.

    The cell ``[[10,0,0],[6,8,0],[0,0,10]]`` has ``V = 800 A^3`` and
    perpendicular widths ``w = (8, 8, 10) A`` against row norms that are all
    ``10 A``, so a grid sized on the wrong quantity is immediately visible in
    ``n_bins``. At ``cutoff = 3.0``, ``skin = 0.5`` the build radius ``3.5 A``
    stays strictly inside the ``min_i w_i / 2 = 4.000 A`` guard, which is what
    makes fractional-rounding minimum image and the kernel's sequential
    reduction provably the same image.

    Fractional coordinates are **literals**: generated offline once with
    ``torch.manual_seed(1); torch.rand(40, 3, dtype=torch.float64)``, rounded to
    6 decimals and pasted, so no RNG runs at test time and the geometry cannot
    drift with a torch RNG change. As pasted, the closest approach to
    ``r_build`` is 1.1e-3 A (measured), clear of the 1e-9 tie band.

    Returns:
        ``(positions (40, 3), cell (3, 3))`` in Angstrom, ``float64``.
    """
    cell = torch.tensor([[10.0, 0.0, 0.0], [6.0, 8.0, 0.0], [0.0, 0.0, 10.0]], dtype=torch.float64)
    frac = torch.tensor(
        [
            [0.061053, 0.224555, 0.234253],
            [0.177099, 0.556068, 0.109444],
            [0.460913, 0.708365, 0.579776],
            [0.496667, 0.510375, 0.329538],
            [0.718206, 0.384511, 0.089797],
            [0.117456, 0.640239, 0.196767],
            [0.512447, 0.711838, 0.924872],
            [0.999699, 0.892730, 0.876720],
            [0.844972, 0.154448, 0.170536],
            [0.984198, 0.812706, 0.435849],
            [0.414321, 0.428408, 0.757762],
            [0.922513, 0.964327, 0.176018],
            [0.953894, 0.313379, 0.454398],
            [0.295552, 0.187507, 0.243258],
            [0.349296, 0.444072, 0.406873],
            [0.285938, 0.803593, 0.321766],
            [0.363903, 0.298510, 0.663531],
            [0.255167, 0.414372, 0.839555],
            [0.741833, 0.286491, 0.792859],
            [0.500116, 0.897740, 0.105125],
            [0.580914, 0.986660, 0.131524],
            [0.239137, 0.304684, 0.515845],
            [0.451441, 0.492893, 0.530066],
            [0.264720, 0.167118, 0.548191],
            [0.237952, 0.537363, 0.442156],
            [0.645389, 0.537569, 0.224480],
            [0.663186, 0.843878, 0.010876],
            [0.280679, 0.930112, 0.543798],
            [0.812327, 0.774969, 0.730758],
            [0.992421, 0.728189, 0.232834],
            [0.999747, 0.554004, 0.420049],
            [0.541916, 0.864175, 0.431247],
            [0.121250, 0.895592, 0.878425],
            [0.912789, 0.968760, 0.415001],
            [0.409411, 0.688470, 0.679978],
            [0.641520, 0.401901, 0.487456],
            [0.956891, 0.517200, 0.953366],
            [0.854016, 0.955512, 0.083597],
            [0.168356, 0.188330, 0.938444],
            [0.354260, 0.202702, 0.506931],
        ],
        dtype=torch.float64,
    )
    return frac @ cell, cell


def _unwrapped_lattice() -> tuple[torch.Tensor, torch.Tensor]:
    """The 64-atom lattice with 8 atoms pushed out of the box by whole cell vectors.

    Physically the identical system — a lattice vector is a symmetry — but the
    coordinates are no longer inside ``[0, L)``, which is exactly the state an
    unwrapped MD trajectory drifts into. The binned path wraps *fractionally* to
    index its bins; this fixture is what pins that the wrap is an indexing
    device only and never reaches the stored positions or the shifts.

    Returns:
        ``(positions (64, 3), cell (3, 3))`` in Angstrom, ``float64``.
    """
    pos, cell = make_cubic_lattice(n_side=4, spacing=3.0)
    moved = pos.clone()
    rows = (0, 1, 2, 0, 1, 2, 0, 1)
    signs = (1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0)
    for atom, (row, sign) in enumerate(zip(rows, signs, strict=True)):
        moved[atom] = moved[atom] + sign * cell[row]
    return moved, cell


def _coincident_lattice() -> tuple[torch.Tensor, torch.Tensor]:
    """The 64-atom lattice carrying both flavours of zero-distance pair.

    ``pos[1]`` is moved onto ``pos[0]`` — coincident in real space — and
    ``pos[2]`` onto ``pos[3] + a_1``, i.e. coincident only *through* a lattice
    vector: raw separation 12 A, minimum image the zero vector. The compiled
    kernel rejects both (``distances > 0``), so the binned path's ``r > 0``
    filter has to reject exactly the same two and nothing else.

    Every atom still sits on a lattice site, so the pair distances stay
    ``{0, 3.0, 4.2426, 5.196, ...} A`` and the closest approach to
    ``r_build = 5.0 A`` remains 0.196 A.

    Returns:
        ``(positions (64, 3), cell (3, 3))`` in Angstrom, ``float64``.
    """
    pos, cell = make_cubic_lattice(n_side=4, spacing=3.0)
    moved = pos.clone()
    moved[1] = pos[0]
    moved[2] = pos[3] + cell[0]
    return moved, cell


def _binned_pair(
    pos: torch.Tensor,
    cell: torch.Tensor,
    *,
    bin: float = 0.0,
    cutoff: float = 3.5,
    skin: float = 1.5,
    capacity_factor: float = 1.35,
) -> tuple[NeighborList, NeighborList]:
    """Two lists over one geometry: the binned build and its ``bin=None`` oracle.

    The compiled path is not modified by this link, which is precisely what
    lets it stand as the reference for the new one.

    Args:
        pos: Positions ``(N, 3)`` in Angstrom.
        cell: Cell vectors ``(3, 3)`` in Angstrom.
        bin: Requested perpendicular bin thickness in Angstrom; ``0.0`` selects
            the automatic ``r_build / 2``.
        cutoff: Interaction cutoff in Angstrom.
        skin: Verlet skin in Angstrom.
        capacity_factor: Buffer headroom, as in the constructor.

    Returns:
        ``(binned, kernel)`` — both built at ``pos``.
    """
    binned = NeighborList(
        cell=cell,
        cutoff=cutoff,
        positions=pos,
        skin=skin,
        bin=bin,
        capacity_factor=capacity_factor,
    )
    kernel = NeighborList(
        cell=cell,
        cutoff=cutoff,
        positions=pos,
        skin=skin,
        bin=None,
        capacity_factor=capacity_factor,
    )
    return binned, kernel


def _policy_schedule(pos: torch.Tensor) -> list[torch.Tensor]:
    """One atom drifting 0.1 A per step along x, twelve steps.

    Against ``skin = 1.0`` (half-skin 0.5 A) this crosses the rebuild criterion
    twice, so both arms of the gate are exercised and neither half of a
    comparison over it is vacuous. Every configuration keeps its closest pair
    0.034 A away from ``r_build = 4.5 A`` (measured), so the per-rebuild edge-set
    comparison never rides a float tie.

    Args:
        pos: The reference positions ``(N, 3)`` in Angstrom.

    Returns:
        Twelve position tensors, each ``(N, 3)`` in Angstrom.
    """
    return [_displaced(pos, 0.1 * step) for step in range(1, 13)]


class TestNeighborListBinned:
    """The pure-torch binned (cell-list) build path behind ``bin=``.

    ``bin=None`` (the default) keeps the compiled O(N^2) kernel and is therefore
    available as the *oracle*: for every fixture the two backends must return
    the identical set of ``(source, target, shift)`` triples. Edge order is not
    part of that contract (see :func:`_canonical_keys`); completeness,
    duplicate-freedom and the ``0 < r <= r_build`` filter are.

    Reference:
        Allen, M. P.; Tildesley, D. J. *Computer Simulation of Liquids*, 2nd
        ed.; Oxford University Press, 2017.
        https://doi.org/10.1093/oso/9780198803195.001.0001 — cell lists, the
        27-cell stencil and minimum-image validity.

        Thompson, A. P. et al. *Comput. Phys. Commun.* **271** (2022) 108171,
        https://doi.org/10.1016/j.cpc.2021.108171; binning policy in
        ``lammps/lammps`` develop ``src/nbin_standard.cpp``
        (``binsize_optimal = 0.5 * cutneighmax``), which is the ``bin=0.0``
        automatic size adopted here.
    """

    # --- grid derivation: n_bins is the only public window onto the stencil ---

    def test_the_auto_bin_is_half_the_build_radius(self):
        """``bin=0.0`` requests ``b = r_build / 2`` (LAMMPS ``nbin_standard``).

        On the 12 A cube at ``r_build = 5.0 A`` that is 2.5 A, and
        ``n_i = floor(w_i / b) = floor(4.8) = 4``, giving effective bins of
        3.0 A and ``k_i = ceil(5.0 / 3.0) = 2``. So ``2k + 1 = 5 > 4``: the raw
        stencil wraps onto the same bin twice, which is the aliasing regime the
        equivalence fixtures below are built to trip.
        """
        pos, cell = make_cubic_lattice(n_side=4, spacing=3.0)
        nl = NeighborList(cell=cell, cutoff=3.5, positions=pos, skin=1.5, bin=0.0)
        assert nl.n_bins == (4, 4, 4)
        assert isinstance(nl.n_bins, tuple)

    def test_the_auto_grid_scales_with_the_cell(self):
        """Same 2.5 A request in a 24 A cube: ``floor(24 / 2.5) = 9`` per axis.

        The count is what makes the path O(N): 9^3 = 729 bins searched 125 at a
        time, against the 4^3 = 64 bins of the 12 A cube where the stencil still
        covers everything.
        """
        pos, cell = make_cubic_lattice(n_side=8, spacing=3.0)
        nl = NeighborList(cell=cell, cutoff=3.5, positions=pos, skin=1.5, bin=0.0)
        assert nl.n_bins == (9, 9, 9)

    def test_the_auto_grid_follows_the_build_radius_not_the_cutoff(self):
        """``skin=0.0`` moves ``r_build`` to 3.5 A, so the auto bin is 1.75 A and
        ``floor(12 / 1.75) = 6``. A grid derived from ``cutoff`` would report the
        same ``(6, 6, 6)`` here *only because* the skin is zero — which is why
        the skinned case above pins ``(4, 4, 4)`` and this one pins the move."""
        pos, cell = make_cubic_lattice(n_side=4, spacing=3.0)
        nl = NeighborList(cell=cell, cutoff=3.5, positions=pos, skin=0.0, bin=0.0)
        assert nl.n_bins == (6, 6, 6)

    def test_an_explicit_bin_thickness_sizes_the_grid(self):
        """``bin=5.0`` in the 12 A cube: ``floor(12 / 5) = 2`` bins of 6.0 A,
        ``k_i = ceil(5.0 / 6.0) = 1``. The explicit knob is a *requested*
        thickness — the effective one is ``w_i / n_i``, never smaller."""
        pos, cell = make_cubic_lattice(n_side=4, spacing=3.0)
        nl = NeighborList(cell=cell, cutoff=3.5, positions=pos, skin=1.5, bin=5.0)
        assert nl.n_bins == (2, 2, 2)

    def test_a_bin_as_wide_as_the_cell_degenerates_to_one_bin(self):
        """``max(1, floor(w_i / b))`` — a bin request at or beyond the cell width
        must clamp to a single bin per axis rather than produce a zero-bin grid
        (a division by zero in the flat-id arithmetic). One bin per axis is the
        graceful all-pairs degeneration: correct, just not faster."""
        pos, cell = make_cubic_lattice(n_side=4, spacing=3.0)
        nl = NeighborList(cell=cell, cutoff=3.5, positions=pos, skin=1.5, bin=12.0)
        assert nl.n_bins == (1, 1, 1)

    def test_the_grid_is_sized_on_perpendicular_widths_not_row_norms(self):
        """The discriminating golden — triclinic sizing, asserted directly.

        For ``[[10,0,0],[6,8,0],[0,0,10]]`` the perpendicular widths are
        ``w = (8, 8, 10) A`` while **all three row norms are 10 A**. At
        ``r_build = 3.5 A`` the auto bin is 1.75 A, so sizing on ``w`` gives
        ``(floor(8/1.75), floor(8/1.75), floor(10/1.75)) = (4, 4, 5)`` where an
        implementation sizing on ``||a_i||`` would report ``(5, 5, 5)``. That
        wrong grid makes the bins thinner than requested along the sheared axes
        and the stencil incomplete — a silently short neighbour list.
        """
        pos, cell = _triclinic_forty()
        nl = NeighborList(cell=cell, cutoff=3.0, positions=pos, skin=0.5, bin=0.0)
        assert nl.n_bins == (4, 4, 5)

    def test_the_default_backend_reports_no_grid(self):
        """``bin=None`` — passed or omitted — is the untouched kernel path.

        ``n_bins is None`` is the observable that says "no grid was derived",
        and the crystallographic 1152 edges say the build itself is bit-for-bit
        the pre-link behaviour. The explicit form is constructed first so this
        also pins that ``None`` is *accepted*, not just defaulted to.
        """
        pos, cell = make_cubic_lattice(n_side=4, spacing=3.0)
        explicit = NeighborList(cell=cell, cutoff=3.5, positions=pos, skin=1.5, bin=None)
        omitted = NeighborList(cell=cell, cutoff=3.5, positions=pos, skin=1.5)
        assert explicit.bin is None
        assert explicit.n_bins is None
        assert omitted.n_bins is None
        assert explicit.num_edges == omitted.num_edges == 1152

    # --- construction validation: refuse before allocating anything ---------

    def test_a_negative_bin_is_refused(self):
        """A negative thickness is a typo, and ``0.0`` already means "choose for
        me" — so the message has to name both, or the reader's next guess is
        that ``-1`` was the way to ask for the automatic size."""
        pos, cell = make_cubic_lattice(n_side=4, spacing=3.0)
        with pytest.raises(ValueError, match=r"\bbin\b") as excinfo:
            NeighborList(cell=cell, cutoff=3.5, positions=pos, skin=1.5, bin=-1.0)
        assert "0.0" in str(excinfo.value)

    def test_a_bin_far_below_the_build_radius_is_refused(self):
        """A 0.05 A bin at ``r_build = 5.0 A`` is a hang, not a fine grid.

        It gives 240 bins per axis, ``k_i = ceil(5.0 / 0.05) = 100`` and a
        ``201^3 ~ 8.1e6``-offset stencil — a Python loop that never finishes.
        Refused at construction against the half-width cap of 8, with the
        measured numbers in the message: the implied 100 and the cap it broke.
        """
        pos, cell = make_cubic_lattice(n_side=4, spacing=3.0)
        with pytest.raises(ValueError, match=r"\bbin\b") as excinfo:
            NeighborList(cell=cell, cutoff=3.5, positions=pos, skin=1.5, bin=0.05)
        message = str(excinfo.value)
        assert "100" in message
        assert re.search(r"\b8\b", message), f"the stencil cap is not named in: {message}"
        assert "0.0" in message

    # --- equivalence with the kernel oracle ---------------------------------

    def test_the_binned_path_matches_the_kernel_on_the_aliasing_lattice(self):
        """PRIMARY falsification, in the regime that breaks a naive stencil.

        64 atoms, ``n_bins = (4, 4, 4)``, ``k_i = 2``: the raw offsets
        ``-2..2 (mod 4)`` visit bins 2 and 3 twice each, so an implementation
        that does not reduce the stencil to *distinct residues* emits a large
        share of the pairs twice — which the duplicate-free and count clauses
        catch, and set equality alone would not.
        """
        pos, cell = make_cubic_lattice(n_side=4, spacing=3.0)
        binned, kernel = _binned_pair(pos, cell)
        _assert_paths_agree(binned, kernel, pos)

    @pytest.mark.parametrize(("skin", "edges"), [(0.0, 384), (1.5, 1152)], ids=["bare", "skinned"])
    def test_the_binned_path_builds_at_the_build_radius(self, skin: float, edges: int):
        """``skin`` sets the radius, ``bin`` only the search strategy.

        Crystallography, not a fit: the simple-cubic lattice has 6 neighbours at
        3.0 A and 12 more at ``3*sqrt(2) = 4.2426 A`` (the 8 body diagonals at
        5.196 A stay out of both radii), so 64*6 = 384 edges at ``r_build =
        3.5 A`` and 64*18 = 1152 at 5.0 A. A binned path that built at ``cutoff``
        would report 384 in both rows, leaving the Verlet skin dead while every
        energy still looked plausible.
        """
        pos, cell = make_cubic_lattice(n_side=4, spacing=3.0)
        binned, kernel = _binned_pair(pos, cell, skin=skin)
        assert binned.num_edges == edges
        assert kernel.num_edges == edges

    def test_the_binned_path_matches_the_kernel_on_a_jittered_dense_box(self):
        """512 atoms off-lattice: the stencil actually prunes, 125 bins of 729.

        No lattice symmetry is left to make a half-wrong stencil look right, and
        no count golden is available — the kernel is the oracle and the only
        claim is that the two agree edge for edge.
        """
        pos, cell = _jittered_box()
        binned, kernel = _binned_pair(pos, cell)
        _assert_paths_agree(binned, kernel, pos)

    def test_the_binned_path_matches_the_kernel_on_a_triclinic_cell(self):
        """Where fractional rounding has to reproduce the kernel's reduction.

        In a sheared cell the two minimum-image algorithms are visibly different
        procedures; they agree only because ``r_build <= min_i w_i / 2`` bounds
        every in-range image's fractional offset by 1/2. This fixture is the
        measurement of that lemma.
        """
        pos, cell = _triclinic_forty()
        binned, kernel = _binned_pair(pos, cell, cutoff=3.0, skin=0.5)
        _assert_paths_agree(binned, kernel, pos)

    def test_the_binned_path_matches_the_kernel_on_unwrapped_positions(self):
        """Eight atoms sitting a whole cell vector outside the box.

        The binned path wraps fractionally to index its bins. If that wrapped
        copy leaked into the displacement or the shift, this fixture — the same
        physical system as the plain lattice — would disagree with the kernel.
        """
        pos, cell = _unwrapped_lattice()
        binned, kernel = _binned_pair(pos, cell)
        _assert_paths_agree(binned, kernel, pos)

    def test_a_whole_cell_translation_leaves_the_pair_distances_unchanged(self):
        """The other half of the wrap invariant, read through the shifts.

        Translating atoms by lattice vectors is a symmetry, so the multiset of
        ``(low, high, ||pos[t] - pos[s] + shift||)`` must be *identical* to the
        untranslated lattice's. Set equality against the kernel cannot see this:
        it would also hold if both paths reconstructed the same wrong geometry.
        """
        pos, cell = make_cubic_lattice(n_side=4, spacing=3.0)
        moved, _ = _unwrapped_lattice()
        binned = NeighborList(cell=cell, cutoff=3.5, positions=pos, skin=1.5, bin=0.0)
        translated = NeighborList(cell=cell, cutoff=3.5, positions=moved, skin=1.5, bin=0.0)
        assert _pair_distances(translated, moved) == _pair_distances(binned, pos)

    def test_both_paths_drop_the_zero_distance_pairs(self):
        """``0 < r <= r_build`` filter parity, both flavours of ``r == 0``.

        Atom 1 is coincident with atom 0 in real space; atom 2 is coincident
        with atom 3 only through ``a_1``, so its raw separation is 12 A and its
        minimum image is the zero vector. The compiled backends reject both
        (``distances > 0`` in the C++ path, ``distance2 == 0`` in the CUDA one),
        so a binned path filtering on ``r <= r_build`` alone would build two
        self-cancelling edges the oracle does not have — and, worse, a division
        by zero anywhere a unit vector is taken.
        """
        pos, cell = _coincident_lattice()
        binned, kernel = _binned_pair(pos, cell)
        _assert_paths_agree(binned, kernel, pos)
        pairs = {(low, high) for low, high, _ in _canonical_keys(binned)}
        assert (0, 1) not in pairs
        assert (2, 3) not in pairs

    def test_the_edge_set_is_independent_of_the_bin_size(self):
        """``bin`` is a cost knob, never a physics knob.

        The sweep spans every regime the derivation has: 0.0 and 2.5 A give the
        9x9x9 pruning grid, 5.0 A a coarse 4x4x4 one, and 24.0 A the single-bin
        full degeneration. A bin size that changes the edge set is a broken
        stencil derivation, full stop — so all four are compared against the one
        kernel oracle rather than against each other.
        """
        pos, cell = _jittered_box()
        kernel = NeighborList(cell=cell, cutoff=3.5, positions=pos, skin=1.5, bin=None)
        for thickness in (0.0, 2.5, 5.0, 24.0):
            binned = NeighborList(cell=cell, cutoff=3.5, positions=pos, skin=1.5, bin=thickness)
            _assert_paths_agree(binned, kernel, pos)

    # --- orthogonality with the rest of the class ---------------------------

    def test_the_rebuild_policy_is_untouched_by_the_binned_backend(self):
        """``skin`` decides the radius, ``bin`` the strategy, ``update`` the moment.

        Over one scripted drift the gate's decisions and all three counters must
        be identical with and without a grid: the policy reads displacements and
        a clock, neither of which the build backend touches.
        """
        pos, cell = make_cubic_lattice(n_side=4, spacing=3.0)
        binned, kernel = _binned_pair(pos, cell, skin=1.0, capacity_factor=4.0)
        schedule = _policy_schedule(pos)
        by_binned = [binned.update(step) for step in schedule]
        by_kernel = [kernel.update(step) for step in schedule]
        assert set(by_kernel) == {True, False}  # the schedule exercises both arms
        assert by_binned == by_kernel
        assert (binned.ago, binned.rebuild_count, binned.ndanger) == (
            kernel.ago,
            kernel.rebuild_count,
            kernel.ndanger,
        )

    def test_the_live_edges_after_a_policy_rebuild_match_the_kernel(self):
        """The same schedule, but checking *what* was built, not *when*.

        A rebuild driven through ``update`` goes down the same dispatch as the
        constructor's initial build, so the equivalence has to survive it — a
        path that only agreed on the first build would leave the run drifting
        away from the oracle one rebuild at a time.
        """
        pos, cell = make_cubic_lattice(n_side=4, spacing=3.0)
        binned, kernel = _binned_pair(pos, cell, skin=1.0, capacity_factor=4.0)
        compared = 0
        for moved in _policy_schedule(pos):
            rebuilt = binned.update(moved)
            assert kernel.update(moved) is rebuilt
            if rebuilt:
                _assert_paths_agree(binned, kernel, moved)
                compared += 1
        assert compared == 2  # the schedule crosses the half-skin twice

    def test_a_cast_carries_the_grid_and_the_inverse_cell(self):
        """``to(dtype)`` must move the binned state with the buffers.

        The stencil and the cached inverse cell are the two tensors nothing else
        in the class owns, so they are exactly what a ``to`` that only knows
        about ``edge_index`` / ``shifts`` / ``cell`` / ``_x_hold`` leaves behind
        — in the wrong dtype (a float64 inverse cell against float32 positions
        promotes *silently*) or on the wrong device. The rebuilt 1152 is the
        cheapest observable that both survived.
        """
        pos, cell = make_cubic_lattice(n_side=4, spacing=3.0)
        nl = NeighborList(cell=cell, cutoff=3.5, positions=pos, skin=1.5, bin=0.0)
        nl.to(torch.float32)
        nl.rebuild(pos.to(torch.float32))
        assert nl.n_bins == (4, 4, 4)
        assert nl.num_edges == 1152
        assert nl.edge_index.dtype == torch.long  # indices are never cast

    def test_the_protocol_does_not_learn_about_bins(self):
        """The build backend is an implementation detail of *this* class.

        ``NeighborStrategy`` stays as link 04 left it: a binned list satisfies
        it unchanged, and ``bin`` / ``n_bins`` must not appear in the protocol,
        which would be creep that every other strategy — including the stubs in
        this file — would then have to carry.
        """
        pos, cell = make_cubic_lattice(n_side=4, spacing=3.0)
        nl = NeighborList(cell=cell, cutoff=3.5, positions=pos, skin=1.5, bin=0.0)
        assert isinstance(nl, NeighborStrategy)
        assert "bin" not in NeighborStrategy.__annotations__
        assert "n_bins" not in NeighborStrategy.__annotations__
