"""Unit tests for ``MoleculePadder()`` — the fixed-length padding task.

Pure function tests: each exercises ``MoleculePadder()`` on a collated
batch and asserts the structure of its return value (fixed shapes, the real-atom
mask, ghost-atom fields, dead self-loop padding edges at the shared dead-edge
displacement, dead graph slots and the ghosts' sink, overflow errors) or on
the caps it owns (``from_counts`` / ``budgets`` / ``check``, and the contract
that a ``TokenBudgetPacker`` fed those budgets never emits a batch the padder
would refuse). No model, no ``torch.compile``, no training loop. That a MACE
model prices a padded batch exactly as the unpadded one is asserted at the
model, in ``tests/test_molzoo/test_mace/test_potential.py``.
"""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from molix.data import MoleculePadder, TokenBudgetPacker
from molix.data.pad import N_GHOSTS
from molix.schema import dead_pair_length
from tests.conftest import make_graph_batch

#: Cutoff (Å) the padder is told; dead edges sit at ``dead_pair_length(CUTOFF)``.
CUTOFF = 4.5


def _batch() -> tuple[object, int, int]:
    """A 2-molecule collated batch (4 + 3 atoms, fully-connected per mol)."""
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.1, 0.1, 0.0],
            [0.3, 1.2, 0.2],
            [1.4, 1.1, -0.1],
            [0.0, 0.0, 0.0],
            [1.0, 0.2, 0.1],
            [0.2, 1.0, -0.2],
        ],
        dtype=torch.float32,
    )
    z = torch.tensor([1, 6, 7, 8, 1, 6, 8], dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1], dtype=torch.long)

    def full_edges(idx):
        return [[i, j] for i in idx for j in idx if i != j]

    pair_index = torch.tensor(full_edges([0, 1, 2, 3]) + full_edges([4, 5, 6]), dtype=torch.long)
    b = make_graph_batch(pos, z, pair_index, batch)
    return b, pos.shape[0], pair_index.shape[0]


def test_execute_sets_fixed_shapes():
    b, n, e = _batch()
    out = MoleculePadder(max_atoms=n + 5, max_pairs=e + 9, cutoff=CUTOFF)(b)
    assert out["atoms"].batch_size[0] == n + 5
    assert out["pairs"].batch_size[0] == e + 9


def test_execute_mask_marks_real_atoms():
    b, n, e = _batch()
    out = MoleculePadder(max_atoms=n + 5, max_pairs=e + 9, cutoff=CUTOFF)(b)
    mask = out["atoms", "mask"]
    assert mask.dtype == torch.bool
    assert mask.shape == (n + 5,)
    assert mask[:n].all()
    assert not mask[n:].any()
    assert int(mask.sum()) == n


def test_execute_ghost_atoms_get_pad_type_and_zeroed_fields():
    b, n, e = _batch()
    out = MoleculePadder(max_atoms=n + 5, max_pairs=e + 9, cutoff=CUTOFF, pad_atom_type=1)(b)
    # ghost Z == pad_atom_type; first ghost at origin; next three sit on the
    # axes at dead_pair_length so dummy valence has finite geometry.
    assert torch.equal(out["atoms", "Z"][n:], torch.ones(5, dtype=torch.long))
    assert out["atoms", "pos"][n].abs().max().item() == 0.0
    assert int(out["atoms", "batch"][n:].abs().max()) == 0
    length = dead_pair_length(CUTOFF)
    assert torch.allclose(out["atoms", "pos"][n + 1], torch.tensor([length, 0.0, 0.0]))
    assert torch.allclose(out["atoms", "pos"][n + 2], torch.tensor([0.0, length, 0.0]))
    assert torch.allclose(out["atoms", "pos"][n + 3], torch.tensor([0.0, 0.0, length]))
    assert out["atoms", "pos"][n + 4].abs().max().item() == 0.0


def test_execute_padding_edges_are_ghost_self_loops():
    b, n, e = _batch()
    out = MoleculePadder(max_atoms=n + 5, max_pairs=e + 9, cutoff=CUTOFF)(b)
    pad_edges = out["pairs", "pair_index"][e:]
    # every padding edge self-loops on the first ghost atom (index n)
    assert torch.equal(pad_edges, torch.full((9, 2), n, dtype=pad_edges.dtype))
    # padding edges are *dead*: displaced by the shared dead-edge vector, so
    # every cutoff envelope is exactly zero on them and no ``1/r`` term sees
    # ``r = 0`` (zero-padding made them live ``r = 0`` self-loops before).
    dead = dead_pair_length(CUTOFF)
    assert torch.equal(
        out["pairs", "pair_diff"][e:],
        torch.tensor([[dead, 0.0, 0.0]] * 9, dtype=out["pairs", "pair_diff"].dtype),
    )
    assert torch.equal(
        out["pairs", "pair_dist"][e:], torch.full((9,), dead, dtype=out["pairs", "pair_dist"].dtype)
    )


def test_execute_pads_a_shifts_column_with_the_dead_edge_vector():
    """A periodic batch's ``edges.shifts`` pads to the same dead vector.

    The derivative op rebuilds ``pair_diff = pos[t] - pos[s] + shifts`` from its
    own leaf; a ghost self-loop has ``pos[t] - pos[s] = 0``, so the shift alone
    has to carry the dead displacement or the rebuilt padding edge is live.
    """
    b, n, e = _batch()
    b["pairs", "shifts"] = torch.zeros(e, 3, dtype=torch.float32)
    out = MoleculePadder(max_atoms=n + 5, max_pairs=e + 9, cutoff=CUTOFF)(b)
    dead = dead_pair_length(CUTOFF)
    assert torch.equal(out["pairs", "shifts"][:e], torch.zeros(e, 3, dtype=torch.float32))
    assert torch.equal(
        out["pairs", "shifts"][e:], torch.tensor([[dead, 0.0, 0.0]] * 9, dtype=torch.float32)
    )


def test_execute_real_rows_preserved():
    b, n, e = _batch()
    z_real = b["atoms", "Z"][:n].clone()
    ei_real = b["pairs", "pair_index"][:e].clone()
    out = MoleculePadder(max_atoms=n + 5, max_pairs=e + 9, cutoff=CUTOFF)(b)
    assert torch.equal(out["atoms", "Z"][:n], z_real)
    assert torch.equal(out["pairs", "pair_index"][:e], ei_real)


def test_execute_overflow_raises():
    b, n, e = _batch()
    # too few atom slots (need >= n + 1 for the ghost anchor) must raise, not truncate
    with pytest.raises(ValueError):
        MoleculePadder(max_atoms=n, max_pairs=e + 9, cutoff=CUTOFF)(b)
    b2, n2, e2 = _batch()
    with pytest.raises(ValueError):
        MoleculePadder(max_atoms=n2 + 5, max_pairs=e2 - 1, cutoff=CUTOFF)(b2)


def test_execute_replaces_atoms_when_pad_is_much_larger_than_n():
    """In-place setitem kept batch_size=[n] (Class-I 516 vs 16385)."""
    b, n, e = _batch()
    out = MoleculePadder(max_atoms=1024, max_pairs=e + 9, cutoff=CUTOFF)(b)
    assert out["atoms"].batch_size[0] == 1024
    assert out["atoms", "Z"].shape[0] == 1024


def _valence_batch() -> tuple[object, int, int]:
    b, n, e = _batch()
    b["bonds"] = TensorDict(
        {"bond_index": torch.tensor([[0, 4], [1, 5]], dtype=torch.long)},
        batch_size=[],
    )
    b["angles"] = TensorDict(
        {
            "atomi": torch.tensor([0, 4], dtype=torch.long),
            "atomj": torch.tensor([1, 5], dtype=torch.long),
            "atomk": torch.tensor([2, 6], dtype=torch.long),
        },
        batch_size=[2],
    )
    b["propers"] = TensorDict(
        {
            "atomi": torch.tensor([0], dtype=torch.long),
            "atomj": torch.tensor([1], dtype=torch.long),
            "atomk": torch.tensor([2], dtype=torch.long),
            "atoml": torch.tensor([3], dtype=torch.long),
        },
        batch_size=[1],
    )
    b["impropers"] = TensorDict(
        {
            "atomi": torch.tensor([0], dtype=torch.long),
            "atomj": torch.tensor([1], dtype=torch.long),
            "atomk": torch.tensor([2], dtype=torch.long),
            "atoml": torch.tensor([3], dtype=torch.long),
        },
        batch_size=[1],
    )
    return b, n, e


def test_pads_all_valence_families_to_caps():
    b, n, e = _valence_batch()
    out = MoleculePadder(
        max_atoms=n + 5,
        max_pairs=e + 9,
        cutoff=CUTOFF,
        max_bonds=8,
        max_angles=8,
        max_propers=4,
        max_impropers=4,
    )(b)
    assert out["bonds", "bond_index"].shape == (2, 8)
    assert out["angles"].batch_size[0] == 8
    assert out["propers"].batch_size[0] == 4
    assert out["impropers"].batch_size[0] == 4
    assert int(out["bonds", "mask"].sum()) == 2
    assert int(out["angles", "mask"].sum()) == 2
    assert int(out["propers", "mask"].sum()) == 1
    ghost0 = n
    dummy_bonds = out["bonds", "bond_index"][:, 2:]
    assert torch.equal(dummy_bonds[0], torch.full((6,), ghost0))
    assert torch.equal(dummy_bonds[1], torch.full((6,), ghost0 + 1))
    assert torch.equal(out["angles", "atomi"][2:], torch.full((6,), ghost0))
    assert torch.equal(out["angles", "atomj"][2:], torch.full((6,), ghost0 + 1))
    assert torch.equal(out["angles", "atomk"][2:], torch.full((6,), ghost0 + 2))
    assert torch.equal(out["propers", "atoml"][1:], torch.full((3,), ghost0 + 3))


def test_valence_overflow_raises():
    b, n, e = _valence_batch()
    with pytest.raises(ValueError, match="max_bonds"):
        MoleculePadder(max_atoms=n + 5, max_pairs=e + 9, cutoff=CUTOFF, max_bonds=1)(b)


def _one_molecule_batch() -> object:
    """A 1-molecule collated batch (3 atoms) — a different live graph count."""
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.2, 0.1], [0.2, 1.0, -0.2]], dtype=torch.float32)
    z = torch.tensor([1, 6, 8], dtype=torch.long)
    batch = torch.zeros(3, dtype=torch.long)
    pair_index = torch.tensor([[i, j] for i in range(3) for j in range(3) if i != j])
    return make_graph_batch(pos, z, pair_index, batch)


class TestMoleculePadderGraphs:
    """``max_graphs`` pads the graph count: dead slots, ``graphs.mask``, the sink."""

    def test_pads_graphs_to_max_graphs_with_mask_and_sink(self):
        b, n, e = _batch()
        out = MoleculePadder(max_atoms=n + 5, max_pairs=e + 9, cutoff=CUTOFF, max_graphs=5)(b)
        graphs = out["graphs"]
        assert graphs.batch_size[0] == 5
        assert graphs["num_atoms"].tolist() == [4, 3, 0, 0, 0]
        assert graphs["mask"].dtype == torch.bool
        assert graphs["mask"].tolist() == [True, True, False, False, False]
        # live atoms keep their graph; ghosts sit in the sink (last slot).
        assert out["atoms", "batch"][:n].tolist() == [0, 0, 0, 0, 1, 1, 1]
        assert out["atoms", "batch"][n:].tolist() == [4] * 5

    def test_same_static_shape_for_different_live_graph_counts(self):
        """Two batches with different B (and N, E) pad to identical shapes everywhere."""
        two, n, e = _batch()
        one = _one_molecule_batch()
        padder = MoleculePadder(max_atoms=n + 5, max_pairs=e + 9, cutoff=CUTOFF, max_graphs=4)
        out_two = padder(two)
        out_one = padder(one)
        for ns in ("atoms", "pairs", "graphs"):
            assert out_two[ns].batch_size == out_one[ns].batch_size
            for key in out_two[ns].keys():
                assert out_two[ns][key].shape == out_one[ns][key].shape, (ns, key)

    def test_graph_overflow_raises_without_a_sink_slot(self):
        b, n, e = _batch()
        with pytest.raises(ValueError, match="max_graphs"):
            MoleculePadder(max_atoms=n + 5, max_pairs=e + 9, cutoff=CUTOFF, max_graphs=2)(b)

    def test_max_graphs_requires_a_graphs_namespace(self):
        b, n, e = _batch()
        del b["graphs"]
        with pytest.raises(ValueError, match="graphs"):
            MoleculePadder(max_atoms=n + 5, max_pairs=e + 9, cutoff=CUTOFF, max_graphs=4)(b)

    def test_max_graphs_below_two_is_rejected(self):
        with pytest.raises(ValueError, match="sink"):
            MoleculePadder(max_atoms=8, max_pairs=8, cutoff=CUTOFF, max_graphs=1)

    def test_without_max_graphs_ghosts_stay_on_graph_zero(self):
        b, n, e = _batch()
        out = MoleculePadder(max_atoms=n + 5, max_pairs=e + 9, cutoff=CUTOFF)(b)
        assert out["graphs"].batch_size[0] == 2
        assert "mask" not in out["graphs"].keys()
        assert out["atoms", "batch"][n:].tolist() == [0] * 5


#: Four molecules of 10 atoms / 20 edges; one carries 900 propers, the rest none.
#: Hand-derivable: Σatoms = 40, edge density 2.0, proper density 22.5, mean atoms 10.
OUTLIER_COUNTS = {
    "atoms": torch.tensor([10, 10, 10, 10]),
    "pairs": torch.tensor([20, 20, 20, 20]),
    "propers": torch.tensor([0, 0, 0, 900]),
}


class TestMoleculePadderFromCounts:
    """``from_counts`` sizes every cap from density × atom budget, floored at the max."""

    def test_density_caps_and_graph_slots_at_unit_rounding(self):
        # atoms=16 → atom budget 12; edges 2.0×12×1.1 = 26.4 → 27 (> max 20);
        # propers 22.5×12×1.1 = 297 < max 900 → the max floor binds; graphs
        # ceil(1.25×12/10) + 1 = 3.
        padder = MoleculePadder.from_counts(OUTLIER_COUNTS, atoms=16, cutoff=CUTOFF, round_to=1)
        assert padder.caps() == {"atoms": 16, "pairs": 27, "propers": 900, "graphs": 3}
        assert padder.budgets() == {"atoms": 12, "pairs": 27, "propers": 900, "graphs": 2}
        assert padder.max_bonds == 0 and padder.max_angles == 0 and padder.max_impropers == 0

    def test_density_caps_round_up_to_multiple(self):
        # atoms=1024 → budget 1020; edges 2244 → 2304; propers 25245 → 25344;
        # graphs ceil(1.25×1020/10) + 1 = 129.
        padder = MoleculePadder.from_counts(OUTLIER_COUNTS, atoms=1024, cutoff=CUTOFF)
        assert padder.caps() == {"atoms": 1024, "pairs": 2304, "propers": 25344, "graphs": 129}

    def test_atoms_must_hold_the_largest_molecule_plus_ghosts(self):
        with pytest.raises(ValueError, match=f"{10 + N_GHOSTS}"):
            MoleculePadder.from_counts(OUTLIER_COUNTS, atoms=13, cutoff=CUTOFF)

    def test_requires_atoms_and_edges_tables(self):
        with pytest.raises(ValueError, match="pairs"):
            MoleculePadder.from_counts({"atoms": torch.tensor([3, 4])}, atoms=16, cutoff=CUTOFF)

    def test_unknown_table_is_rejected(self):
        counts = dict(OUTLIER_COUNTS, proper=torch.tensor([1, 1, 1, 1]))
        with pytest.raises(ValueError, match="proper"):
            MoleculePadder.from_counts(counts, atoms=16, cutoff=CUTOFF)

    def test_check_refuses_a_molecule_no_batch_can_carry(self):
        padder = MoleculePadder.from_counts(OUTLIER_COUNTS, atoms=16, cutoff=CUTOFF, round_to=1)
        padder.check(OUTLIER_COUNTS)
        with pytest.raises(ValueError, match="propers"):
            padder.check({"propers": torch.tensor([0, 901])})
        with pytest.raises(ValueError, match="atoms"):
            padder.check({"atoms": torch.tensor([13])})

    def test_budgets_keep_every_packed_batch_within_caps(self):
        """The packer fed ``budgets()`` never emits a batch the padder would refuse."""
        gen = torch.Generator().manual_seed(0)
        n = 400
        atoms = torch.randint(3, 60, (n,), generator=gen)
        atoms[7] = 350  # the largest molecule
        pairs = atoms * 6
        propers = atoms * 2
        propers[123] = 8516  # the proper outlier, on a small molecule
        counts = {"atoms": atoms, "pairs": pairs, "propers": propers}
        padder = MoleculePadder.from_counts(counts, atoms=1024, cutoff=CUTOFF)
        padder.check(counts)
        budgets = padder.budgets()
        packer = TokenBudgetPacker(
            [(budgets[t], counts[t]) for t in ("atoms", "pairs", "propers")]
            + [(budgets["graphs"], torch.ones(n, dtype=torch.long))]
        )
        assert not packer.oversize_mask().any()
        order = torch.randperm(n, generator=gen).tolist()
        plan = packer.pack(order)
        assert sorted(i for batch in plan for i in batch) == list(range(n))
        caps = padder.caps()
        for batch in plan:
            idx = torch.tensor(batch)
            assert int(atoms[idx].sum()) + N_GHOSTS <= caps["atoms"]
            assert int(pairs[idx].sum()) <= caps["pairs"]
            assert int(propers[idx].sum()) <= caps["propers"]
            assert len(batch) + 1 <= caps["graphs"]
