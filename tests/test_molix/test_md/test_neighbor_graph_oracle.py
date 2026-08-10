"""NeighborList graph vs independent multi-image brute-force oracle.

SUT: ``molix.md.NeighborList`` (the graph MACE consumes). Production cutoff
filter is ``0 < r <= r_c`` (see ``get_neighbor_pairs`` / binned path).

Does **not** import production neighbor backends into the oracle module.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from molix.md import NeighborList
from tests.test_molix.test_md.oracle_bruteforce_neighbors import (
    assert_graphs_equal,
    bruteforce_edges,
    neighborlist_edge_keys,
    physical_dr_multiset,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _list(
    pos: torch.Tensor,
    cell: torch.Tensor,
    cutoff: float,
    *,
    skin: float = 0.0,
    bin: float | None = None,
) -> NeighborList:
    return NeighborList(
        cell=cell,
        cutoff=cutoff,
        positions=pos,
        skin=skin,
        every=1,
        delay=0,
        check=True,
        capacity_factor=2.0,
        bin=bin,
    )


def _sut_keys(nl: NeighborList, cell: torch.Tensor) -> set:
    return neighborlist_edge_keys(nl.edge_index, nl.shifts, nl.num_edges, cell, open_system=False)


def _compare(nl: NeighborList, pos: torch.Tensor, cell: torch.Tensor, cutoff: float, label: str):
    ref = bruteforce_edges(pos, cell=cell, cutoff=cutoff, pbc=(True, True, True))
    sut = _sut_keys(nl, cell)
    return assert_graphs_equal(ref, sut, label=label)


# ---------------------------------------------------------------------------
# ac-001 / oracle self-checks
# ---------------------------------------------------------------------------


class TestBruteforceOracle:
    def test_excludes_only_true_self(self):
        pos = torch.zeros(1, 3, dtype=torch.float64)
        cell = torch.eye(3, dtype=torch.float64) * 5.0
        keys = bruteforce_edges(pos, cell=cell, cutoff=2.0, pbc=(True, True, True))
        assert (0, 0, 0, 0, 0) not in keys
        # Self-image along ±e_x at distance L; need r_c >= L.
        cell2 = torch.eye(3, dtype=torch.float64) * 3.0
        keys2 = bruteforce_edges(pos, cell=cell2, cutoff=3.0, pbc=(True, True, True))
        assert (0, 0, 1, 0, 0) in keys2 or (0, 0, -1, 0, 0) in keys2

    def test_open_no_pbc(self):
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
        keys = bruteforce_edges(pos, cell=None, cutoff=1.5, pbc=(False, False, False))
        assert keys == {(0, 1, 0, 0, 0), (1, 0, 0, 0, 0)}


# ---------------------------------------------------------------------------
# ac-002 open random (large cell ≈ open under MIC guard)
# ---------------------------------------------------------------------------


class TestOpenRandom:
    def test_random_open_system_matches_oracle(self):
        torch.manual_seed(0)
        n = 20
        pos = torch.rand(n, 3, dtype=torch.float64) * 8.0 + 1.0  # stay away from edges
        cell = torch.eye(3, dtype=torch.float64) * 20.0  # half-width 10 > r_c
        cutoff = 3.5
        nl = _list(pos, cell, cutoff)
        _compare(nl, pos, cell, cutoff, "open-random: ")


# ---------------------------------------------------------------------------
# ac-003 cutoff boundary ≤
# ---------------------------------------------------------------------------


class TestCutoffBoundary:
    @pytest.mark.parametrize(
        "delta,expect_edge",
        [
            (-1e-3, True),
            (-1e-6, True),
            (0.0, True),  # r == r_c included
            (1e-6, False),
            (1e-3, False),
        ],
    )
    def test_edge_presence_at_cutoff(self, delta: float, expect_edge: bool):
        r_c = 2.0
        # two atoms along x in a large box (no PBC contact)
        pos = torch.tensor([[0.0, 5.0, 5.0], [r_c + delta, 5.0, 5.0]], dtype=torch.float64)
        cell = torch.eye(3, dtype=torch.float64) * 20.0
        nl = _list(pos, cell, r_c)
        keys = _sut_keys(nl, cell)
        has = (0, 1, 0, 0, 0) in keys and (1, 0, 0, 0, 0) in keys
        assert has is expect_edge, (
            f"cutoff convention is 0 < r <= r_c; at r=r_c+{delta:g} expected "
            f"edge={expect_edge}, got has={has}; num_edges={nl.num_edges}"
        )
        ref = bruteforce_edges(pos, cell=cell, cutoff=r_c)
        assert ((0, 1, 0, 0, 0) in ref) is expect_edge


# ---------------------------------------------------------------------------
# ac-004 PBC wrap
# ---------------------------------------------------------------------------


class TestPeriodicWrap:
    def test_atoms_across_box_face(self):
        pos = torch.tensor([[0.1, 5.0, 5.0], [9.9, 5.0, 5.0]], dtype=torch.float64)
        cell = torch.eye(3, dtype=torch.float64) * 10.0
        cutoff = 1.0
        nl = _list(pos, cell, cutoff)
        cmp = _compare(nl, pos, cell, cutoff, "pbc-wrap: ")
        assert cmp.n_ref >= 2
        # physical |dr| ≈ 0.2
        src, tgt = nl.edge_index[0, 0], nl.edge_index[0, 1]
        dr = pos[int(tgt)] - pos[int(src)] + nl.shifts[0]
        assert float(dr.norm()) == pytest.approx(0.2, abs=1e-9)


# ---------------------------------------------------------------------------
# ac-005a graph wrap invariance
# ---------------------------------------------------------------------------


class TestWrapInvarianceGraph:
    def _base(self):
        torch.manual_seed(1)
        pos = torch.rand(12, 3, dtype=torch.float64) * 6.0 + 1.0
        cell = torch.eye(3, dtype=torch.float64) * 10.0
        cutoff = 2.5
        return pos, cell, cutoff

    def test_wrap_translate_physical_dr_multiset(self):
        pos, cell, cutoff = self._base()
        nl0 = _list(pos, cell, cutoff)
        keys0 = _sut_keys(nl0, cell)
        dr0 = physical_dr_multiset(keys0, pos, cell)

        # wrap atom 0 by +cell_x
        pos_w = pos.clone()
        pos_w[0] = pos_w[0] + cell[0]
        nl_w = _list(pos_w, cell, cutoff)
        keys_w = _sut_keys(nl_w, cell)
        dr_w = physical_dr_multiset(keys_w, pos_w, cell)
        assert dr0 == dr_w, "wrap one atom: physical dr multiset must match"
        _compare(nl_w, pos_w, cell, cutoff, "wrap-atom: ")

        # translate whole structure by lattice vector
        pos_t = pos + cell[1]
        nl_t = _list(pos_t, cell, cutoff)
        keys_t = _sut_keys(nl_t, cell)
        dr_t = physical_dr_multiset(keys_t, pos_t, cell)
        assert dr0 == dr_t
        _compare(nl_t, pos_t, cell, cutoff, "translate-all: ")


# ---------------------------------------------------------------------------
# ac-005b E/F slow
# ---------------------------------------------------------------------------


def _matpes_weights() -> Path | None:
    p = Path("/home/jicli594/work/mace_models")
    if (p / "matpes_r2scan_config.json").is_file() and (
        p / "matpes_r2scan_cueq_state.pt"
    ).is_file():
        return p
    return None


@pytest.mark.slow
class TestWrapInvarianceEnergyForces:
    def test_energy_forces_invariant_under_wrap(self):
        weights = _matpes_weights()
        if weights is None:
            pytest.skip("MatPES weights not available")
        from molix import config
        from molpot.derivation.force import autograd_forces_from_energy
        from molzoo.mace import MACEPotential

        config.set_precision("fp64")
        model = (
            MACEPotential.from_checkpoint(
                weights / "matpes_r2scan_config.json",
                weights / "matpes_r2scan_cueq_state.pt",
                use_fallback=True,
            )
            .eval()
            .to(dtype=torch.float64)
        )
        r_c = float(model.cutoff_fn.r_cut)
        # small water-like openish box: need L/2 >= r_c
        L = 2.0 * r_c + 0.5
        torch.manual_seed(2)
        n = 8
        # place atoms in interior
        pos = (torch.rand(n, 3, dtype=torch.float64) * 0.4 + 0.3) * L
        cell = torch.eye(3, dtype=torch.float64) * L
        Z = torch.full((n,), 8, dtype=torch.long)  # oxygen

        def ef(p: torch.Tensor) -> tuple[float, torch.Tensor]:
            nl = _list(p, cell, r_c)
            leaf = p.detach().requires_grad_(True)
            batch = torch.zeros(n, dtype=torch.long)
            with torch.enable_grad():
                e = model.energy_core(
                    leaf,
                    Z,
                    nl.edge_index[: nl.num_edges],
                    batch,
                    1,
                    nl.shifts[: nl.num_edges],
                ).sum()
            f = autograd_forces_from_energy(e, leaf)
            return float(e.detach()), f.detach()

        e0, f0 = ef(pos)
        pos_w = pos.clone()
        pos_w[0] = pos_w[0] + cell[0]
        e1, f1 = ef(pos_w)
        assert e0 == pytest.approx(e1, abs=1e-6, rel=1e-8)
        # force on atom 0 after wrap: still same Cartesian F
        assert torch.allclose(f0, f1, atol=1e-5, rtol=1e-6)


# ---------------------------------------------------------------------------
# ac-006 triclinic
# ---------------------------------------------------------------------------


class TestTriclinic:
    def test_tilted_cell_matches_oracle(self):
        torch.manual_seed(3)
        pos = torch.rand(10, 3, dtype=torch.float64) * 4.0 + 0.5
        cell = torch.tensor(
            [[8.0, 0.0, 0.0], [1.5, 7.5, 0.0], [0.5, 0.8, 7.0]],
            dtype=torch.float64,
        )
        cutoff = 2.0
        # ensure r_build ok
        nl = _list(pos, cell, cutoff)
        _compare(nl, pos, cell, cutoff, "triclinic: ")


# ---------------------------------------------------------------------------
# ac-007 multi-image domain (oracle multi-image + SUT under r_c <= L/2)
# ---------------------------------------------------------------------------


class TestMultiImage:
    def test_oracle_finds_self_images_when_rc_exceeds_half_box(self):
        pos = torch.zeros(1, 3, dtype=torch.float64)
        cell = torch.eye(3, dtype=torch.float64) * 3.0
        keys = bruteforce_edges(pos, cell=cell, cutoff=3.0)
        assert any(k[0] == 0 and k[1] == 0 and k[2:] != (0, 0, 0) for k in keys)

    def test_neighborlist_matches_oracle_near_half_width(self):
        # NeighborList forbids r_build > L/2; stay just under.
        L = 10.0
        cutoff = 4.9  # half-width = 5.0
        torch.manual_seed(4)
        pos = torch.rand(15, 3, dtype=torch.float64) * (L - 1.0) + 0.5
        cell = torch.eye(3, dtype=torch.float64) * L
        nl = _list(pos, cell, cutoff)
        _compare(nl, pos, cell, cutoff, "near-half-width: ")


# ---------------------------------------------------------------------------
# ac-008 cutoff-crossing frames
# ---------------------------------------------------------------------------


class TestCutoffCrossing:
    def test_per_frame_rebuild_no_hysteresis(self):
        r_c = 2.0
        cell = torch.eye(3, dtype=torch.float64) * 20.0
        # approach from outside, enter, leave
        distances = [2.5, 2.01, 1.99, 1.5, 1.99, 2.01, 2.5]
        states = []
        for r in distances:
            pos = torch.tensor([[0.0, 5.0, 5.0], [r, 5.0, 5.0]], dtype=torch.float64)
            nl = _list(pos, cell, r_c)  # fresh each frame
            keys = _sut_keys(nl, cell)
            has = (0, 1, 0, 0, 0) in keys
            ref = bruteforce_edges(pos, cell=cell, cutoff=r_c)
            assert_graphs_equal(ref, keys, label=f"crossing r={r}: ")
            states.append(has)
        # expected: outside False, inside True, no sticky True after leaving
        assert states == [False, False, True, True, True, False, False]


# ---------------------------------------------------------------------------
# ac-012 dual backend
# ---------------------------------------------------------------------------


class TestDualBackend:
    def test_bin_matches_default_on_wrap_case(self):
        pos = torch.tensor([[0.1, 5.0, 5.0], [9.9, 5.0, 5.0]], dtype=torch.float64)
        cell = torch.eye(3, dtype=torch.float64) * 10.0
        cutoff = 1.0
        nl_def = _list(pos, cell, cutoff, bin=None)
        try:
            nl_bin = _list(pos, cell, cutoff, bin=0.0)  # auto bin size
        except Exception as exc:
            pytest.skip(f"binned backend unavailable: {exc}")
        k_def = _sut_keys(nl_def, cell)
        k_bin = _sut_keys(nl_bin, cell)
        assert_graphs_equal(k_def, k_bin, label="bin-vs-default: ")
        ref = bruteforce_edges(pos, cell=cell, cutoff=cutoff)
        assert_graphs_equal(ref, k_bin, label="bin-vs-oracle: ")
