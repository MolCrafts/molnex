"""Tests for the GTO reciprocal-space surface (``molpot.potentials.gto.kspace``).

Only ``kspace.py``'s own public face lives here: :func:`kspace_cutoff` and
:class:`KSpaceGrid`. The ``GTOHartreeEnergy`` cases that used to sit in this
file moved to ``test_hartree.py`` when ``hartree_energy`` was folded into its
single caller (spec ``polar-compile-clean-01-kgrid``).
"""

from __future__ import annotations

import inspect
import math

import pytest
import torch

from molix import config
from molpot.potentials.gto.kspace import KSpaceGrid, kspace_cutoff

#: Polar dump reciprocal cutoff at ``σ = 1.5 Å``, ``l_max = 1``, ``κ = 1.0``
#: in Å⁻¹ — hard-coded golden, equal to the official checkpoint's
#: ``kspace_cutoff`` buffer.
K_CUT = 1.8467166200173746

#: Triclinic probe cell (Å), one lattice vector per row.
TRICLINIC_CELL: list[list[float]] = [[10.0, 0.0, 0.0], [3.0, 9.0, 0.0], [1.0, 2.0, 8.0]]

#: Lexicographically sorted fp64 k-vectors (Å⁻¹) that the pre-change
#: ``half_sphere_kvectors`` selected on the 10 Å cubic cell at
#: :data:`K_CUT`. Hard-coded golden captured from this tree before the
#: fixed-capacity rewrite (molnex @ ``feat/oplsaa-parameterizer``,
#: ``PYTHONPATH=src python capture_kgrid_goldens.py``, 2026-09-03) — no live
#: oracle.
CUBIC10_HALF_SPHERE: list[list[float]] = [
    [0.0, 0.0, 0.6283185307179586],
    [0.0, 0.0, 1.2566370614359172],
    [0.0, 0.6283185307179586, -1.2566370614359172],
    [0.0, 0.6283185307179586, -0.6283185307179586],
    [0.0, 0.6283185307179586, 0.0],
    [0.0, 0.6283185307179586, 0.6283185307179586],
    [0.0, 0.6283185307179586, 1.2566370614359172],
    [0.0, 1.2566370614359172, -1.2566370614359172],
    [0.0, 1.2566370614359172, -0.6283185307179586],
    [0.0, 1.2566370614359172, 0.0],
    [0.0, 1.2566370614359172, 0.6283185307179586],
    [0.0, 1.2566370614359172, 1.2566370614359172],
    [0.6283185307179586, -1.2566370614359172, -0.6283185307179586],
    [0.6283185307179586, -1.2566370614359172, 0.0],
    [0.6283185307179586, -1.2566370614359172, 0.6283185307179586],
    [0.6283185307179586, -0.6283185307179586, -1.2566370614359172],
    [0.6283185307179586, -0.6283185307179586, -0.6283185307179586],
    [0.6283185307179586, -0.6283185307179586, 0.0],
    [0.6283185307179586, -0.6283185307179586, 0.6283185307179586],
    [0.6283185307179586, -0.6283185307179586, 1.2566370614359172],
    [0.6283185307179586, 0.0, -1.2566370614359172],
    [0.6283185307179586, 0.0, -0.6283185307179586],
    [0.6283185307179586, 0.0, 0.0],
    [0.6283185307179586, 0.0, 0.6283185307179586],
    [0.6283185307179586, 0.0, 1.2566370614359172],
    [0.6283185307179586, 0.6283185307179586, -1.2566370614359172],
    [0.6283185307179586, 0.6283185307179586, -0.6283185307179586],
    [0.6283185307179586, 0.6283185307179586, 0.0],
    [0.6283185307179586, 0.6283185307179586, 0.6283185307179586],
    [0.6283185307179586, 0.6283185307179586, 1.2566370614359172],
    [0.6283185307179586, 1.2566370614359172, -0.6283185307179586],
    [0.6283185307179586, 1.2566370614359172, 0.0],
    [0.6283185307179586, 1.2566370614359172, 0.6283185307179586],
    [1.2566370614359172, -1.2566370614359172, 0.0],
    [1.2566370614359172, -0.6283185307179586, -0.6283185307179586],
    [1.2566370614359172, -0.6283185307179586, 0.0],
    [1.2566370614359172, -0.6283185307179586, 0.6283185307179586],
    [1.2566370614359172, 0.0, -1.2566370614359172],
    [1.2566370614359172, 0.0, -0.6283185307179586],
    [1.2566370614359172, 0.0, 0.0],
    [1.2566370614359172, 0.0, 0.6283185307179586],
    [1.2566370614359172, 0.0, 1.2566370614359172],
    [1.2566370614359172, 0.6283185307179586, -0.6283185307179586],
    [1.2566370614359172, 0.6283185307179586, 0.0],
    [1.2566370614359172, 0.6283185307179586, 0.6283185307179586],
    [1.2566370614359172, 1.2566370614359172, 0.0],
]

#: Same golden for :data:`TRICLINIC_CELL` (38 vectors).
TRICLINIC_HALF_SPHERE: list[list[float]] = [
    [0.0, 0.0, 0.7853981633974483],
    [0.0, 0.0, 1.5707963267948966],
    [0.0, 0.6981317007977318, -0.9599310885968813],
    [0.0, 0.6981317007977318, -0.17453292519943295],
    [0.0, 0.6981317007977318, 0.6108652381980153],
    [0.0, 0.6981317007977318, 1.3962634015954636],
    [0.0, 1.3962634015954636, -1.1344640137963142],
    [0.0, 1.3962634015954636, -0.3490658503988659],
    [0.0, 1.3962634015954636, 0.4363323129985824],
    [0.6283185307179586, -1.6057029118347832, -0.4625122517784973],
    [0.6283185307179586, -1.6057029118347832, 0.32288591161895097],
    [0.6283185307179586, -0.9075712110370514, -1.4224433403753785],
    [0.6283185307179586, -0.9075712110370514, -0.6370451769779303],
    [0.6283185307179586, -0.9075712110370514, 0.148352986419518],
    [0.6283185307179586, -0.9075712110370514, 0.9337511498169663],
    [0.6283185307179586, -0.2094395102393196, -1.5969762655748114],
    [0.6283185307179586, -0.2094395102393196, -0.8115781021773633],
    [0.6283185307179586, -0.2094395102393196, -0.026179938779914938],
    [0.6283185307179586, -0.2094395102393196, 0.7592182246175333],
    [0.6283185307179586, -0.2094395102393196, 1.5446163880149817],
    [0.6283185307179586, 0.4886921905584122, -0.9861110273767961],
    [0.6283185307179586, 0.4886921905584122, -0.2007128639793479],
    [0.6283185307179586, 0.4886921905584122, 0.5846852994181004],
    [0.6283185307179586, 0.4886921905584122, 1.3700834628155487],
    [0.6283185307179586, 1.186823891356144, -1.160643952576229],
    [0.6283185307179586, 1.186823891356144, -0.3752457891787808],
    [0.6283185307179586, 1.186823891356144, 0.41015237421866746],
    [0.6283185307179586, 1.186823891356144, 1.1955505376161157],
    [1.2566370614359172, -1.117010721276371, -0.6632251157578453],
    [1.2566370614359172, -1.117010721276371, 0.12217304763960307],
    [1.2566370614359172, -0.4188790204786392, -0.8377580409572781],
    [1.2566370614359172, -0.4188790204786392, -0.052359877559829876],
    [1.2566370614359172, -0.4188790204786392, 0.7330382858376184],
    [1.2566370614359172, 0.2792526803190926, -1.0122909661567112],
    [1.2566370614359172, 0.2792526803190926, -0.22689280275926282],
    [1.2566370614359172, 0.2792526803190926, 0.5585053606381855],
    [1.2566370614359172, 0.9773843811168244, -0.4014257279586958],
    [1.2566370614359172, 0.9773843811168244, 0.3839724354387525],
]


@pytest.fixture(autouse=True)
def fp64():
    previous = config["ftype"]
    config.set_precision("fp64")
    yield
    config.set_precision("fp64" if previous == torch.float64 else "fp32")


class TestKspaceCutoff:
    def test_polar_dump_defaults(self):
        """σ=1.5 Å, l_max=1, κ=1.0 → 2.25/σ · 2^{0.3}."""
        expect = 2.25 / 1.5 * (2.0**0.3)
        assert kspace_cutoff(1.5, 1, 1.0) == pytest.approx(expect, rel=1e-12)


class TestKSpaceGrid:
    """Fixed-capacity integer half-box: sized once, masked multiplicatively."""

    def test_first_bind_sizes_capacity_from_lattice_norms(self) -> None:
        """10 Å cubic, f=1.2 → N_c = (4, 4, 4) and M = (9³ − 1)/2 = 364."""
        grid = KSpaceGrid(k_cutoff=K_CUT, capacity_factor=1.2)
        grid.bind(torch.eye(3, dtype=torch.float64) * 10.0)
        assert grid.capacity == (4, 4, 4)
        assert tuple(grid.n.shape) == (364, 3)
        assert grid.n.dtype == torch.int64

    def test_half_box_excludes_the_origin_and_keeps_one_of_each_pair(self) -> None:
        """No all-zero row; every row's first non-zero component is > 0."""
        grid = KSpaceGrid(k_cutoff=K_CUT, capacity_factor=1.2)
        grid.bind(torch.eye(3, dtype=torch.float64) * 10.0)
        n = grid.n
        assert int((n == 0).all(dim=1).sum()) == 0
        first = torch.argmax((n != 0).to(torch.long), dim=1)
        sign = torch.gather(n, 1, first.unsqueeze(1)).squeeze(-1)
        assert int((sign > 0).sum()) == n.shape[0]

    def test_triclinic_capacity_is_ceil_of_scaled_lattice_norms(self) -> None:
        """D4: N_c,i = ceil(f · ‖a_i‖ k_cut / 2π), exact for a triclinic cell."""
        cell = torch.tensor(TRICLINIC_CELL, dtype=torch.float64)
        grid = KSpaceGrid(k_cutoff=K_CUT, capacity_factor=1.2)
        grid.bind(cell)
        expect = tuple(
            math.ceil(1.2 * float(torch.linalg.norm(cell[i])) * K_CUT / (2.0 * math.pi))
            for i in range(3)
        )
        assert expect == (4, 4, 3)
        assert grid.capacity == expect
        assert tuple(grid.n.shape) == (283, 3)

    def test_triclinic_weighted_set_matches_brute_force_enumeration(self) -> None:
        """Nothing inside the k-sphere is dropped by the fixed box (38 vectors)."""
        cell = torch.tensor(TRICLINIC_CELL, dtype=torch.float64)
        g = 2.0 * math.pi * torch.linalg.inv(cell).T
        axis = torch.arange(-6, 7)
        nvec = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1).reshape(-1, 3)
        kvec = nvec.to(torch.float64) @ g
        k_sq = (kvec * kvec).sum(dim=1)
        keep = (k_sq > 0.0) & (k_sq < K_CUT * K_CUT)
        nvec, kvec = nvec[keep], kvec[keep]
        first = torch.argmax((nvec != 0).to(torch.long), dim=1)
        sign = torch.gather(nvec, 1, first.unsqueeze(1)).squeeze(-1)
        expect = sorted(kvec[sign > 0].tolist(), key=tuple)
        assert len(expect) == 38

        grid = KSpaceGrid(k_cutoff=K_CUT, capacity_factor=1.2)
        grid.bind(cell)
        grid_kvec, _, weights = grid.vectors(cell)
        got = sorted(grid_kvec[weights > 0.0].tolist(), key=tuple)
        assert len(got) == 38
        assert torch.allclose(
            torch.tensor(got, dtype=torch.float64),
            torch.tensor(expect, dtype=torch.float64),
            atol=1e-12,
            rtol=0.0,
        )
        assert torch.allclose(
            torch.tensor(got, dtype=torch.float64),
            torch.tensor(TRICLINIC_HALF_SPHERE, dtype=torch.float64),
            atol=1e-12,
            rtol=0.0,
        )

    def test_weights_are_zero_or_two(self) -> None:
        """Conjugate symmetry is a multiplicative 0/2 mask, never an index."""
        cell = torch.eye(3, dtype=torch.float64) * 10.0
        grid = KSpaceGrid(k_cutoff=K_CUT, capacity_factor=1.2)
        grid.bind(cell)
        _, _, weights = grid.vectors(cell)
        assert tuple(weights.shape) == (364,)
        assert set(weights.tolist()) <= {0.0, 2.0}

    def test_weighted_kvectors_match_the_pre_change_half_sphere(self) -> None:
        """The 46 selected vectors of the removed ``half_sphere_kvectors``."""
        cell = torch.eye(3, dtype=torch.float64) * 10.0
        grid = KSpaceGrid(k_cutoff=K_CUT, capacity_factor=1.2)
        grid.bind(cell)
        kvec, _, weights = grid.vectors(cell)
        picked = kvec[weights > 0.0]
        assert picked.shape[0] == 46
        got = sorted(picked.tolist(), key=tuple)
        assert torch.allclose(
            torch.tensor(got, dtype=torch.float64),
            torch.tensor(CUBIC10_HALF_SPHERE, dtype=torch.float64),
            atol=1e-12,
            rtol=0.0,
        )

    def test_cutoff_comparison_is_strict(self) -> None:
        """|k|² == k_cut² gets weight 0.0 — the old mask used ``<`` too."""
        cell = torch.eye(3, dtype=torch.float64) * 10.0
        g = 2.0 * math.pi * torch.linalg.inv(cell).T
        # k of n = (1, 0, 0) with the module's own formula: the two zero
        # components make |k|² == g00 · g00 bitwise, so the boundary is exact.
        k_edge = float(g[0, 0])
        grid = KSpaceGrid(k_cutoff=k_edge, capacity_factor=1.2)
        grid.bind(cell)
        kvec, _, weights = grid.vectors(cell)
        k_sq = (kvec * kvec).sum(dim=-1)
        boundary = k_sq == k_edge * k_edge
        assert int(boundary.sum()) >= 1  # not a vacuous mask
        assert torch.all(weights[boundary] == 0.0)

    def test_rebinding_the_same_cell_keeps_the_integer_box(self) -> None:
        """Sizing happens on the first bind only — no rebuild, no resize."""
        cell = torch.eye(3, dtype=torch.float64) * 10.0
        grid = KSpaceGrid(k_cutoff=K_CUT, capacity_factor=1.2)
        grid.bind(cell)
        box = grid.n
        grid.bind(cell)
        assert grid.n is box

    def test_shrinking_the_cell_never_rebuilds(self) -> None:
        """A 0.5× cell fits by construction and must not shrink the box."""
        cell = torch.eye(3, dtype=torch.float64) * 10.0
        grid = KSpaceGrid(k_cutoff=K_CUT, capacity_factor=1.2)
        grid.bind(cell)
        box = grid.n
        grid.bind(cell * 0.5)
        assert grid.n is box
        assert grid.capacity == (4, 4, 4)

    def test_expansion_within_capacity_neither_raises_nor_rebuilds(self) -> None:
        """1.1× < f = 1.2: accepted, same object (NPT head-room)."""
        cell = torch.eye(3, dtype=torch.float64) * 10.0
        grid = KSpaceGrid(k_cutoff=K_CUT, capacity_factor=1.2)
        grid.bind(cell)
        box = grid.n
        grid.bind(cell * 1.1)
        grid.check_capacity(cell * 1.1)
        assert grid.n is box

    def test_over_capacity_bind_names_the_grid_and_the_factor(self) -> None:
        """2× overflows; the message must not read like a NeighborList one.

        Scale arithmetic (D4), 10 Å cubic at ``k_cut = 1.8467166200173746``:
        ``x = ‖a‖ k_cut / 2π = 2.9391``, ``N_c = ceil(1.2 x) = 4``. The guard
        is ``floor(x(t)) > N_c``, so it first fires at ``x ≥ 5``, i.e. at
        ``λ ≥ 5/2.9391 ≈ 1.7013``. ``λ = 2`` is a *genuine* overflow: ``n =
        (5, 0, 0)`` has ``|k| = 1.5708 < k_cut`` and would be truncated.
        """
        cell = torch.eye(3, dtype=torch.float64) * 10.0
        grid = KSpaceGrid(k_cutoff=K_CUT, capacity_factor=1.2)
        grid.bind(cell)
        with pytest.raises(RuntimeError) as excinfo:
            grid.bind(cell * 2.0)
        message = str(excinfo.value)
        assert "KSpaceGrid" in message
        assert "capacity_factor" in message

    def test_over_capacity_check_capacity_raises(self) -> None:
        """``check_capacity`` is the eager half of the ``floor(x) > N_c`` guard."""
        cell = torch.eye(3, dtype=torch.float64) * 10.0
        grid = KSpaceGrid(k_cutoff=K_CUT, capacity_factor=1.2)
        grid.bind(cell)
        with pytest.raises(RuntimeError) as excinfo:
            grid.check_capacity(cell * 2.0)
        message = str(excinfo.value)
        assert "KSpaceGrid" in message
        assert "capacity_factor" in message

    def test_expansion_beyond_the_factor_but_inside_the_box_is_accepted(self) -> None:
        """1.5× > f = 1.2 yet nothing is truncated — ``ceil`` bought head-room.

        ``x(1.5) = 4.4087`` and ``|k| < k_cut ⇒ |n_i| < x_i``, so every live
        k-point still satisfies ``|n_i| ≤ 4 = N_c``. The guard must stay
        ``floor(x) > N_c`` (D4): tightening it to ``x > N_c`` would reject a
        cell whose k-sum is exactly complete.
        """
        cell = torch.eye(3, dtype=torch.float64) * 10.0
        grid = KSpaceGrid(k_cutoff=K_CUT, capacity_factor=1.2)
        grid.bind(cell)
        box = grid.n
        grid.check_capacity(cell * 1.5)
        grid.bind(cell * 1.5)
        assert grid.n is box
        _, _, weights = grid.vectors(cell * 1.5)
        assert not bool(torch.isnan(weights).any())

    def test_guarded_denominator_is_strictly_positive(self) -> None:
        """``k_sq_guarded`` is a divisor only: guard first, divide second."""
        cell = torch.eye(3, dtype=torch.float64) * 10.0
        grid = KSpaceGrid(k_cutoff=K_CUT, capacity_factor=1.2)
        grid.bind(cell)
        kvec, k_sq_guarded, weights = grid.vectors(cell)
        k_sq = (kvec * kvec).sum(dim=-1)
        assert bool((k_sq_guarded > 0.0).all())
        assert bool((k_sq > 0.0).all())  # the half box excludes the origin
        assert not bool(((k_sq == 0.0) & (weights != 0.0)).any())

    def test_over_capacity_vectors_return_nan_weights(self) -> None:
        """In-graph the sentinel poisons the sum — it never truncates.

        ``2×`` is the same event the eager ``floor(x) > N_c`` guard rejects
        (see ``test_over_capacity_bind_names_the_grid_and_the_factor``); no
        ``check_capacity`` is called here, so this is the compiled route.
        """
        cell = torch.eye(3, dtype=torch.float64) * 10.0
        grid = KSpaceGrid(k_cutoff=K_CUT, capacity_factor=1.2)
        grid.bind(cell)
        _, _, weights = grid.vectors(cell * 2.0)
        assert bool(torch.isnan(weights).all())

    def test_expansion_within_capacity_has_no_nan(self) -> None:
        """1.1× stays inside the box, so nothing is poisoned."""
        cell = torch.eye(3, dtype=torch.float64) * 10.0
        grid = KSpaceGrid(k_cutoff=K_CUT, capacity_factor=1.2)
        grid.bind(cell)
        kvec, k_sq_guarded, weights = grid.vectors(cell * 1.1)
        assert bool(torch.isfinite(kvec).all())
        assert bool(torch.isfinite(k_sq_guarded).all())
        assert bool(torch.isfinite(weights).all())

    def test_vectors_has_no_graph_breaks(self) -> None:
        """The overflow sentinel is a tensor compare — no host sync."""
        cell = torch.eye(3, dtype=torch.float64) * 10.0
        grid = KSpaceGrid(k_cutoff=K_CUT, capacity_factor=1.2)
        grid.bind(cell)
        torch._dynamo.reset()
        explanation = torch._dynamo.explain(grid.vectors)(cell)
        assert explanation.graph_break_count == 0

    def test_batch_cell_is_bitwise_identical(self) -> None:
        """``(1, 3, 3)`` is what ``graphs.cell`` hands over — squeeze, not reshape."""
        cell = torch.eye(3, dtype=torch.float64) * 10.0
        grid = KSpaceGrid(k_cutoff=K_CUT, capacity_factor=1.2)
        grid.bind(cell)
        flat = grid.vectors(cell)
        batched = grid.vectors(cell.unsqueeze(0))
        assert torch.equal(flat[0], batched[0])
        assert torch.equal(flat[1], batched[1])
        assert torch.equal(flat[2], batched[2])

    def test_bind_and_check_capacity_accept_a_batch_cell(self) -> None:
        """``_as_cell`` is the first statement of all three entry points."""
        cell = (torch.eye(3, dtype=torch.float64) * 10.0).unsqueeze(0)
        grid = KSpaceGrid(k_cutoff=K_CUT, capacity_factor=1.2)
        grid.bind(cell)
        grid.check_capacity(cell)
        assert grid.capacity == (4, 4, 4)

    def test_two_system_cell_is_rejected(self) -> None:
        cell = (torch.eye(3, dtype=torch.float64) * 10.0).expand(2, 3, 3)
        grid = KSpaceGrid(k_cutoff=K_CUT, capacity_factor=1.2)
        with pytest.raises(ValueError, match="single-system"):
            grid.bind(cell)

    def test_unbound_vectors_name_bind_cell(self) -> None:
        """Trace-time failure must say what to call, not a matmul shape."""
        grid = KSpaceGrid(k_cutoff=K_CUT, capacity_factor=1.2)
        with pytest.raises(RuntimeError, match="bind_cell"):
            grid.vectors(torch.eye(3, dtype=torch.float64) * 10.0)

    def test_device_mismatch_names_both_devices(self) -> None:
        """CPU box vs a cell elsewhere: say where each one is."""
        grid = KSpaceGrid(k_cutoff=K_CUT, capacity_factor=1.2)
        grid.bind(torch.eye(3, dtype=torch.float64) * 10.0)
        elsewhere = torch.eye(3, dtype=torch.float64, device="meta") * 10.0
        with pytest.raises(RuntimeError) as excinfo:
            grid.vectors(elsewhere)
        message = str(excinfo.value)
        assert "meta" in message
        assert "cpu" in message

    def test_first_bind_beyond_max_axis_index_explodes(self) -> None:
        """Per-axis cap, named ``max_axis_index``; fail closed, do not allocate."""
        grid = KSpaceGrid(k_cutoff=K_CUT, capacity_factor=1.2, max_axis_index=2)
        with pytest.raises(RuntimeError, match="GTO k-grid exploded"):
            grid.bind(torch.eye(3, dtype=torch.float64) * 10.0)

    def test_to_returns_self(self) -> None:
        """``NeighborList.to`` semantics: mutate in place and return self."""
        cell = torch.eye(3, dtype=torch.float64) * 10.0
        grid = KSpaceGrid(k_cutoff=K_CUT, capacity_factor=1.2)
        grid.bind(cell)
        cpu = torch.device("cpu")
        assert grid.to(cpu) is grid
        assert grid.n.device.type == "cpu"
        kvec, k_sq_guarded, weights = grid.vectors(cell)
        assert kvec.device.type == "cpu"
        assert k_sq_guarded.device.type == "cpu"
        assert weights.device.type == "cpu"

    def test_no_resize_entry_points(self) -> None:
        """Resizing means building a new grid — there is no ``rebind``."""
        grid = KSpaceGrid(k_cutoff=K_CUT, capacity_factor=1.2)
        assert not hasattr(grid, "rebind")
        assert not hasattr(grid, "set_cell")

    def test_grid_is_not_a_module(self) -> None:
        """A runtime geometry cache, not a parameter — it stays out of state_dict."""
        grid = KSpaceGrid(k_cutoff=K_CUT, capacity_factor=1.2)
        assert not isinstance(grid, torch.nn.Module)

    def test_k_cutoff_is_a_readable_attribute(self) -> None:
        """The cutoff is frozen at construction and readable for assertions."""
        grid = KSpaceGrid(k_cutoff=K_CUT, capacity_factor=1.2)
        assert grid.k_cutoff == K_CUT

    def test_constructor_keywords_and_defaults(self) -> None:
        """``capacity_factor=1.2`` / ``max_axis_index=32``, keyword-only."""
        params = inspect.signature(KSpaceGrid).parameters
        assert params["capacity_factor"].default == 1.2
        assert params["capacity_factor"].kind is inspect.Parameter.KEYWORD_ONLY
        assert params["max_axis_index"].default == 32
        assert params["max_axis_index"].kind is inspect.Parameter.KEYWORD_ONLY
        assert "max_capacity" not in params
