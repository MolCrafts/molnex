"""Unit tests for :class:`molix.neighbor.NeighborList`.

Sign-invariant pins for the edge geometry (pair_diff / pair_dist): guards the
load-bearing convention that survives the bond_*→edge_* rename,
``pair_diff[e] = pos[target] - pos[source]`` and, under ``symmetry=True``,
the reverse edge carries the exact negation (spherical-harmonic parity
``Y_l(-r) = (-1)^l Y_l(r)``). See spec graph-connectivity-alignment-02-rename.

``TestNeighborList`` moved here from ``test_data/test_transform.py`` with the
class (spec molix-data-dissolution-03-transforms).
"""

import re
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict

from molix.neighbor import DenseNeighborList, NeighborList

#: The two backends of ``torch.ops.molix.get_neighbor_pairs``.
_KERNELS = Path(__file__).resolve().parents[2] / "csrc/molix/src/neighbors"


def _one_graph(pos: torch.Tensor, cell: torch.Tensor | None = None) -> TensorDict:
    """Wrap one molecule's positions as the collated batch the task consumes."""
    n = int(pos.shape[0])
    graphs: dict = {"num_atoms": torch.tensor([n])}
    if cell is not None:
        graphs["cell"] = cell.unsqueeze(0)
    return TensorDict(
        atoms=TensorDict({"pos": pos, "batch": torch.zeros(n, dtype=torch.long)}, batch_size=[n]),
        graphs=TensorDict(graphs, batch_size=[1]),
        pairs=TensorDict({}, batch_size=[0]),
        batch_size=[],
    )


def test_pair_enumeration_indexes_in_64_bit():
    """The triangular pair index must not be int32 in either backend.

    ``row = floor((sqrt(8*index + 1) + 1) / 2)`` computed in int32 wraps
    negative once ``8*index`` passes 2^31 — from N(N-1)/2 > 2.68e8, i.e. about
    23 200 atoms. On CUDA that was an illegal memory access (found on a
    35 400-atom PEO melt, 6.3e8 candidate pairs); on CPU it silently produced
    wrong rows. A 2.7e8-pair regression test would need tens of GB, so the
    invariant is pinned statically, at the two lines that own it.
    """
    cuda = (_KERNELS / "get_neighbor_pairs.cu").read_text()
    assert re.search(r"const int64_t index = static_cast<int64_t>\(blockIdx\.x\)", cuda)
    assert "int64_t row" in cuda and "int64_t column" in cuda
    assert "sqrtf(8 * index" not in cuda

    cpu = (_KERNELS / "get_neighbor_pairs.cpp").read_text()
    assert "arange(0, num_all_pairs, positions.options().dtype(kInt64))" in cpu
    assert "dtype(kInt32))" not in cpu.split("Tensor neighbors")[0]


def _triangle():
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    return {"Z": torch.tensor([8, 1, 1]), "pos": positions, "targets": {"U0": torch.tensor([0.0])}}


def test_edge_diff_is_target_minus_source():
    sample = _triangle()
    pos = sample["pos"]
    for symmetry in (True, False):
        out = NeighborList(cutoff=2.0, max_num_pairs=10, symmetry=symmetry)(
            _one_graph(sample["pos"], sample.get("cell"))
        )
        pair_index = out["pairs", "pair_index"]
        expected = pos[pair_index[:, 1]] - pos[pair_index[:, 0]]
        assert torch.allclose(out["pairs", "pair_diff"], expected, atol=1e-5)
        assert torch.allclose(out["pairs", "pair_dist"], expected.norm(dim=1), atol=1e-5)


def test_symmetry_appends_reverse_edge_with_negated_edge_diff():
    sample = _triangle()
    out = NeighborList(cutoff=2.0, max_num_pairs=10, symmetry=True)(
        _one_graph(sample["pos"], sample.get("cell"))
    )
    pair_index = out["pairs", "pair_index"]
    pair_diff = out["pairs", "pair_diff"]
    n_pairs = pair_index.shape[0] // 2

    # Reverse edges are appended after the forward half: row k and row k+n_pairs
    # are (src→tgt) and (tgt→src) of the same pair, with negated displacement.
    fwd, rev = pair_index[:n_pairs], pair_index[n_pairs:]
    assert torch.equal(rev, fwd[:, [1, 0]])
    assert torch.allclose(pair_diff[n_pairs:], -pair_diff[:n_pairs], atol=1e-6)


def _compute_neighbor_list_naive(positions: torch.Tensor, cutoff: float):
    """Upper-triangle (i > j) reference pairs + distances via O(N^2) scan."""
    diff = positions.unsqueeze(0) - positions.unsqueeze(1)
    dist = torch.norm(diff, dim=2)
    mask = (dist < cutoff) & (dist > 0)
    i_idx, j_idx = torch.where(mask)
    valid = i_idx > j_idx
    i_idx = i_idx[valid]
    j_idx = j_idx[valid]
    edge_vec = positions[i_idx] - positions[j_idx]
    pair_dist = dist[i_idx, j_idx]
    return i_idx, j_idx, edge_vec, pair_dist


class TestNeighborList:
    def test_small_system_pairs_symmetric_default(self):
        """Default ``symmetry=True`` returns full bidirectional edges.

        Every upper-triangle pair from the naive reference must appear in
        both directions, so ``E == 2 * n_pairs`` and every distance is
        duplicated. This is the Allegro / MACE-facing production path.
        """
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        )
        sample = {
            "Z": torch.tensor([8, 1, 1]),
            "pos": positions,
            "targets": {"U0": torch.tensor([0.0])},
        }

        result = NeighborList(cutoff=2.0, max_num_pairs=10)(
            _one_graph(sample["pos"], sample.get("cell"))
        )
        pair_index = result["pairs", "pair_index"]

        ref_i, ref_j, _, ref_dist = _compute_neighbor_list_naive(positions, 2.0)
        assert pair_index.ndim == 2 and pair_index.shape[1] == 2
        assert pair_index.shape[0] == 2 * ref_i.numel()

        pairs = {(int(a), int(b)) for a, b in pair_index.tolist()}
        for a, b in zip(ref_i.tolist(), ref_j.tolist()):
            assert (a, b) in pairs and (b, a) in pairs

        doubled = torch.cat([ref_dist, ref_dist]).sort().values
        assert torch.allclose(result["pairs", "pair_dist"].sort().values, doubled, atol=1e-5)

    def test_small_system_pairs_half(self):
        """With ``symmetry=False`` the output matches the upper-triangle reference."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        )
        sample = {
            "Z": torch.tensor([8, 1, 1]),
            "pos": positions,
            "targets": {"U0": torch.tensor([0.0])},
        }

        result = NeighborList(cutoff=2.0, max_num_pairs=10, symmetry=False)(
            _one_graph(sample["pos"], sample.get("cell"))
        )
        pair_index = result["pairs", "pair_index"]

        ref_i, ref_j, _, ref_dist = _compute_neighbor_list_naive(positions, 2.0)
        assert pair_index.shape == (ref_i.numel(), 2)
        assert torch.allclose(
            result["pairs", "pair_dist"].sort().values, ref_dist.sort().values, atol=1e-5
        )

    def test_bond_diff_source_to_target_convention(self):
        """``pair_diff[k] == pos[target_k] - pos[source_k]`` for every edge.

        Locks in the edge convention documented in CLAUDE.md: pair_index[:,0]
        is the source, pair_index[:,1] is the target, and pair_diff points
        source → target. Required by SphericalHarmonics and cuEquivariance.
        """
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        )
        sample = {
            "Z": torch.tensor([8, 1, 1]),
            "pos": positions,
            "targets": {"U0": torch.tensor([0.0])},
        }

        # Test both modes to make sure the convention holds bidirectionally.
        for symmetry in (True, False):
            result = NeighborList(
                cutoff=2.0,
                max_num_pairs=10,
                symmetry=symmetry,
            )(_one_graph(sample["pos"], sample.get("cell")))
            pair_index = result["pairs", "pair_index"]
            expected = positions[pair_index[:, 1]] - positions[pair_index[:, 0]]
            assert torch.allclose(result["pairs", "pair_diff"], expected, atol=1e-5), (
                f"pair_diff violates source→target convention (symmetry={symmetry})"
            )

    def test_cutoff_behavior(self):
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            dtype=torch.float32,
        )
        sample = {
            "Z": torch.tensor([1, 1, 1]),
            "pos": positions,
            "targets": {"U0": torch.tensor([0.0])},
        }

        result = NeighborList(cutoff=2.0, max_num_pairs=10)(
            _one_graph(sample["pos"], sample.get("cell"))
        )
        assert torch.all(result["pairs", "pair_dist"] <= 2.0)

    def test_task_id_deterministic(self):
        t1 = NeighborList(cutoff=5.0, max_num_pairs=512)
        t2 = NeighborList(cutoff=5.0, max_num_pairs=512)
        assert t1.task_id == t2.task_id

        t3 = NeighborList(cutoff=6.0)
        assert t1.task_id != t3.task_id

    def test_result_is_independent_of_max_num_pairs_buffer(self):
        """Buffer size is an allocation detail, never a result detail.

        ``execute`` grows the kernel buffer to ``N(N-1)/2`` whenever the
        caller's ``max_num_pairs`` is smaller, so a tight explicit buffer, the
        512 default, and the "all pairs" sentinel ``-1`` must all produce the
        byte-identical edge tensors.
        """
        pos = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        sample = {"Z": torch.tensor([8, 1, 1, 1]), "pos": pos}

        # half_complete = 4*3/2 = 6; buffers below/above/at that, plus -1.
        outs = [
            NeighborList(cutoff=2.0, max_num_pairs=m, symmetry=False)(
                _one_graph(sample["pos"], sample.get("cell"))
            )
            for m in (1, 6, 512, -1)
        ]
        reference = outs[0]
        assert reference["pairs", "pair_index"].shape == (6, 2)
        for out in outs[1:]:
            assert torch.equal(out["pairs", "pair_index"], reference["pairs", "pair_index"])
            assert torch.equal(out["pairs", "pair_diff"], reference["pairs", "pair_diff"])
            assert torch.equal(out["pairs", "pair_dist"], reference["pairs", "pair_dist"])

    def test_tight_max_num_pairs_is_auto_grown_to_complete_half_pairs(self):
        """A 1-slot buffer on a fully connected 4-atom system still finds 6 pairs.

        Pins the auto-sizing branch itself (``half_complete > max_num_pairs``):
        the returned pair count follows the geometry, not the buffer request.
        """
        pos = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, 0.0, 0.5],
            ],
            dtype=torch.float64,
        )
        sample = {"Z": torch.ones(4, dtype=torch.long), "pos": pos}

        out = NeighborList(cutoff=5.0, max_num_pairs=1, symmetry=False)(
            _one_graph(sample["pos"], sample.get("cell"))
        )

        assert out["pairs", "pair_index"].shape == (6, 2)
        assert {(int(a), int(b)) for a, b in out["pairs", "pair_index"].tolist()} == {
            (1, 0),
            (2, 0),
            (2, 1),
            (3, 0),
            (3, 1),
            (3, 2),
        }

    def test_buffer_overflow_raises_runtime_error(self, monkeypatch: pytest.MonkeyPatch):
        """The overflow guard raises ``RuntimeError`` with actionable text.

        The guard compares the kernel's reported ``num_pairs`` against the
        (auto-grown) buffer.  A faked kernel is used to drive the branch: with
        the real kernel the buffer is always grown to ``N(N-1)/2`` first, so
        ``n_found > max_pairs`` is currently unreachable in production
        (reported as a finding — do not "fix" this test by relaxing it).
        """

        def fake_kernel(
            positions: torch.Tensor,
            cutoff: float,
            max_num_pairs: int,
            box_vectors: torch.Tensor | None,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            neighbors = torch.tensor([[1, 2, 2], [0, 0, 1]], dtype=torch.int32)
            deltas = torch.zeros((3, 3), dtype=positions.dtype)
            distances = torch.ones(3, dtype=positions.dtype)
            # One more pair than the buffer can hold -> overflow.
            return neighbors, deltas, distances, torch.tensor([4], dtype=torch.int32)

        monkeypatch.setattr("molix.neighbor.get_neighbor_pairs", fake_kernel)
        # The guard under test belongs to the kernel path, so keep this input on
        # it: a 3-atom single graph is otherwise cheap enough to be routed to the
        # vectorised path, which calls no kernel and so cannot overflow one.
        monkeypatch.setattr("molix.neighbor._BATCHED_MAX_CANDIDATES", 0)

        pos = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float64,
        )
        sample = {"Z": torch.tensor([8, 1, 1]), "pos": pos}

        with pytest.raises(
            RuntimeError,
            match=r"Neighbor buffer overflow: found 4 pairs > max_num_pairs=3",
        ):
            NeighborList(cutoff=2.0, max_num_pairs=3)(_one_graph(sample["pos"], sample.get("cell")))

    def test_nan_padding_rows_are_stripped_by_default(self):
        """``filter_padding=True`` leaves exactly the true pairs, no NaN rows.

        The kernel pads its fixed-width output with ``-1`` indices and NaN
        deltas/distances; a padded row surviving into ``pair_diff`` poisons
        every downstream sum.  3-atom right triangle: 3 half-pairs at
        1, 1, sqrt(2); buffer of 16 => 13 padded rows to strip.
        """
        pos = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float64,
        )
        sample = {"Z": torch.tensor([8, 1, 1]), "pos": pos}

        out = NeighborList(cutoff=2.0, max_num_pairs=16, symmetry=False)(
            _one_graph(sample["pos"], sample.get("cell"))
        )

        assert out["pairs", "pair_index"].tolist() == [[1, 0], [2, 0], [2, 1]]
        assert not torch.isnan(out["pairs", "pair_diff"]).any()
        assert not torch.isnan(out["pairs", "pair_dist"]).any()
        assert (out["pairs", "pair_index"] >= 0).all()
        assert torch.allclose(
            out["pairs", "pair_dist"],
            torch.tensor([1.0, 1.0, 2.0**0.5], dtype=torch.float64),
            atol=1e-12,
        )

    def test_filter_padding_false_keeps_the_nan_padded_rows(self):
        """Opting out returns the raw fixed-width kernel output (NaN + -1 rows).

        Characterises the escape hatch so the stripping test above cannot pass
        vacuously (i.e. proves the kernel really does emit padding here).
        """
        pos = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float64,
        )
        sample = {"Z": torch.tensor([8, 1, 1]), "pos": pos}

        out = NeighborList(cutoff=2.0, max_num_pairs=5, symmetry=False, filter_padding=False)(
            _one_graph(sample["pos"])
        )

        assert out["pairs", "pair_index"].tolist() == [[1, 0], [2, 0], [2, 1], [-1, -1], [-1, -1]]
        assert torch.isnan(out["pairs", "pair_dist"][3:]).all()
        assert torch.isnan(out["pairs", "pair_diff"][3:]).all()

    def test_pbc_true_returns_minimum_image_edge_diff(self):
        """Minimum-image convention across the periodic boundary (hard-coded golden).

        Two atoms in a 2 A cubic cell at x = 0.1 and x = 1.9: the direct
        separation is 1.8 A (outside a 1.0 A cutoff), the minimum image is
        1.8 - 2.0 = -0.2 A.  With ``pair_diff = pos[target] - pos[source]``
        and the kernel emitting source = 1, target = 0, the forward edge is
        ``+0.2 x_hat`` and the appended reverse edge ``-0.2 x_hat``.
        """
        pos = torch.tensor([[0.1, 0.0, 0.0], [1.9, 0.0, 0.0]], dtype=torch.float64)
        cell = torch.tensor(
            [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]], dtype=torch.float64
        )
        sample = {"Z": torch.tensor([1, 1]), "pos": pos, "cell": cell}

        out = NeighborList(cutoff=1.0, max_num_pairs=8, pbc=True)(
            _one_graph(sample["pos"], sample.get("cell"))
        )

        assert out["pairs", "pair_index"].tolist() == [[1, 0], [0, 1]]
        assert torch.allclose(
            out["pairs", "pair_diff"],
            torch.tensor([[0.2, 0.0, 0.0], [-0.2, 0.0, 0.0]], dtype=torch.float64),
            atol=1e-12,
        )
        assert torch.allclose(
            out["pairs", "pair_dist"],
            torch.tensor([0.2, 0.2], dtype=torch.float64),
            atol=1e-12,
        )

    def test_pbc_false_ignores_cell_and_finds_no_pair(self):
        """Open boundaries: the same sample has no edge inside the 1.0 A cutoff.

        Guards against a silently-always-on PBC path — ``cell`` is only read
        when ``pbc=True``.
        """
        pos = torch.tensor([[0.1, 0.0, 0.0], [1.9, 0.0, 0.0]], dtype=torch.float64)
        cell = torch.tensor(
            [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]], dtype=torch.float64
        )
        sample = {"Z": torch.tensor([1, 1]), "pos": pos, "cell": cell}

        out = NeighborList(cutoff=1.0, max_num_pairs=8, pbc=False)(
            _one_graph(sample["pos"], sample.get("cell"))
        )

        assert out["pairs", "pair_index"].shape == (0, 2)
        assert out["pairs", "pair_dist"].numel() == 0

    def test_task_id_distinguishes_pbc(self):
        """``pbc`` is part of the cache key — PBC and open caches must not alias."""
        open_bc = NeighborList(cutoff=1.0, max_num_pairs=8, pbc=False)
        periodic = NeighborList(cutoff=1.0, max_num_pairs=8, pbc=True)

        assert open_bc.task_id == "nlist:cut=1.0:max=8:pbc=False:sym=True"
        assert periodic.task_id == "nlist:cut=1.0:max=8:pbc=True:sym=True"
        assert open_bc.task_id != periodic.task_id


class TestBatchedEnumeration:
    """The whole-batch path must be the loop's answer, not a similar one."""

    @staticmethod
    def _batch(n_graphs: int, seed: int = 2666):
        torch.manual_seed(seed)
        sizes = torch.randint(2, 9, (n_graphs,))
        pos = torch.cat([torch.randn(int(n), 3) * 1.5 for n in sizes])
        membership = torch.repeat_interleave(torch.arange(n_graphs), sizes)
        return TensorDict(
            {
                "atoms": TensorDict(
                    {"pos": pos, "batch": membership}, batch_size=[int(pos.shape[0])]
                ),
                "graphs": TensorDict({}, batch_size=[n_graphs]),
            },
            batch_size=[],
        )

    @staticmethod
    def _sorted_rows(out):
        pairs = out["pairs"]
        idx = pairs["pair_index"]
        key = idx[:, 0] * (idx.max() + 1) + idx[:, 1]
        order = torch.argsort(key)
        return idx[order], pairs["pair_diff"][order], pairs["pair_dist"][order]

    @pytest.mark.parametrize("symmetry", [True, False])
    def test_batched_path_matches_the_per_molecule_loop(self, symmetry: bool) -> None:
        import molix.neighbor as nb

        n_graphs = nb._BATCHED_FROM_GRAPHS + 8
        nlist = NeighborList(cutoff=2.0, max_num_pairs=4096, symmetry=symmetry)

        batched = self._sorted_rows(nlist(self._batch(n_graphs)))
        # Same input, forced down the loop. Both accepts have to be closed:
        # these molecules are 2-8 atoms, so the candidate-pair accept would keep
        # taking the batched path and the comparison would assert nothing.
        original = (nb._BATCHED_FROM_GRAPHS, nb._BATCHED_MAX_CANDIDATES)
        nb._BATCHED_FROM_GRAPHS = n_graphs + 1
        nb._BATCHED_MAX_CANDIDATES = 0
        try:
            looped = self._sorted_rows(nlist(self._batch(n_graphs)))
        finally:
            nb._BATCHED_FROM_GRAPHS, nb._BATCHED_MAX_CANDIDATES = original

        torch.testing.assert_close(batched[0], looped[0])
        torch.testing.assert_close(batched[1], looped[1], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(batched[2], looped[2], atol=1e-6, rtol=1e-6)

    def test_a_small_tail_batch_is_routed_away_from_the_native_kernel(self) -> None:
        """A packing remainder must not need the C++ op to be present.

        Routing on graph count alone put an end-of-epoch tail batch on the
        per-molecule loop, and the loop needs the native library — which a
        frozen snapshot does not carry. Few graphs of few atoms each is the
        cheapest case there is; it belongs on the vectorised path.
        """
        import molix.neighbor as nb

        nlist = NeighborList(cutoff=2.0, max_num_pairs=4096)
        # One THEMol torsion-scan row: a single scan of 22 conformers.
        tail = torch.repeat_interleave(torch.arange(22), torch.full((22,), 24))
        assert 22 < nb._BATCHED_FROM_GRAPHS
        assert nlist._prefer_batched(tail, 22) is True

    def test_one_large_graph_stays_on_the_native_kernel(self) -> None:
        """The case the graph-count threshold existed to protect.

        A single MD graph enumerates ``n(n-1)/2`` candidates, so the vectorised
        path would materialise millions of rows where the kernel's cell list
        walks neighbours. Widening the batched path must not reach this.
        """
        nlist = NeighborList(cutoff=2.0, max_num_pairs=4096)
        one_big = torch.zeros(2100, dtype=torch.long)
        assert nlist._prefer_batched(one_big, 1) is False

    def test_batched_path_never_pairs_atoms_of_different_molecules(self) -> None:
        import molix.neighbor as nb

        n_graphs = nb._BATCHED_FROM_GRAPHS + 4
        batch = self._batch(n_graphs, seed=7)
        out = NeighborList(cutoff=100.0, max_num_pairs=1 << 20)(batch)
        membership = batch["atoms", "batch"]
        idx = out["pairs"]["pair_index"]
        assert bool((membership[idx[:, 0]] == membership[idx[:, 1]]).all())

    def test_batched_path_tolerates_unsorted_membership(self) -> None:
        """A collate need not hand atoms over grouped by molecule."""
        import molix.neighbor as nb

        n_graphs = nb._BATCHED_FROM_GRAPHS + 2
        batch = self._batch(n_graphs, seed=11)
        perm = torch.randperm(int(batch["atoms", "pos"].shape[0]))
        shuffled = TensorDict(
            {
                "atoms": TensorDict(
                    {
                        "pos": batch["atoms", "pos"][perm],
                        "batch": batch["atoms", "batch"][perm],
                    },
                    batch_size=[int(perm.shape[0])],
                ),
                "graphs": TensorDict({}, batch_size=[n_graphs]),
            },
            batch_size=[],
        )
        out = NeighborList(cutoff=2.0, max_num_pairs=4096)(shuffled)
        idx = out["pairs"]["pair_index"]
        member = shuffled["atoms", "batch"]
        assert bool((member[idx[:, 0]] == member[idx[:, 1]]).all())


class TestDenseNeighborList:
    """Dense pairwise geometry: ``(b, i, j)`` is the pair, so there is no list.

    The sparse :class:`NeighborList` is the oracle for the geometry itself; what
    these pin is what dense form is *for* — no index tensor, a backward that is a
    reduction rather than a scatter, and a diagonal that does not poison it.
    """

    @staticmethod
    def _two_molecules() -> tuple[torch.Tensor, torch.Tensor]:
        """Two molecules of 3 and 2 atoms, padded to 3."""
        pos = torch.tensor(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
                [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [9.0, 9.0, 9.0]],
            ],
            dtype=torch.float64,
        )
        mask = torch.tensor([[True, True, True], [True, True, False]])
        return pos, mask

    def test_half_pairs_are_the_upper_triangle(self) -> None:
        pos, mask = self._two_molecules()
        _diff, _dist, live = DenseNeighborList().geometry(pos, mask)
        assert live[0].sum() == 3  # 3 atoms -> 3 pairs
        assert live[1].sum() == 1  # 2 live atoms -> 1 pair
        assert not live[0].diagonal().any()

    def test_both_orientations_double_the_live_pairs(self) -> None:
        pos, mask = self._two_molecules()
        _d, _r, half = DenseNeighborList(half=True).geometry(pos, mask)
        _d, _r, full = DenseNeighborList(half=False).geometry(pos, mask)
        assert int(full.sum()) == 2 * int(half.sum())

    def test_distance_matches_the_norm_of_the_displacement(self) -> None:
        pos, mask = self._two_molecules()
        diff, dist, live = DenseNeighborList().geometry(pos, mask)
        assert torch.allclose(dist[live], diff[live].norm(dim=-1))

    def test_displacement_follows_the_edge_convention(self) -> None:
        """``pair_diff = pos[target] - pos[source]``; source is the first index."""
        pos, mask = self._two_molecules()
        diff, _dist, _live = DenseNeighborList().geometry(pos, mask)
        assert torch.allclose(diff[0, 0, 1], pos[0, 1] - pos[0, 0])

    def test_padding_is_never_live(self) -> None:
        pos, mask = self._two_molecules()
        _diff, _dist, live = DenseNeighborList().geometry(pos, mask)
        assert not live[1, 2].any() and not live[1, :, 2].any()

    def test_a_cutoff_drops_the_pairs_beyond_it(self) -> None:
        pos, mask = self._two_molecules()
        _d, _r, all_pairs = DenseNeighborList().geometry(pos, mask)
        _d, _r, near = DenseNeighborList(cutoff=1.5).geometry(pos, mask)
        assert int(near.sum()) < int(all_pairs.sum())
        assert bool(near[0, 0, 1])  # r = 1.0
        assert not bool(near[0, 0, 2])  # r = 2.0

    def test_no_cutoff_keeps_every_pair(self) -> None:
        """An open-boundary molecule has no lattice sum to split, so 1/r is exact."""
        pos, mask = self._two_molecules()
        _d, _r, live = DenseNeighborList(cutoff=None).geometry(pos, mask)
        assert int(live[0].sum()) == 3  # the complete graph of 3 atoms

    def test_the_diagonal_does_not_produce_a_nan_gradient(self) -> None:
        """``d‖v‖/dv`` is NaN at ``v = 0`` and the diagonal is always ``v = 0``.

        The forward is finite either way, so only the backward shows it — the
        exact shape of bug that reached production once already (54 NaN
        parameters from ``(0, 0)`` padded rows) and was invisible to a
        forward-only check.
        """
        pos, mask = self._two_molecules()
        pos = pos.clone().requires_grad_(True)
        _diff, dist, live = DenseNeighborList().geometry(pos, mask)
        (dist * live).sum().backward()
        assert pos.grad is not None
        assert torch.isfinite(pos.grad).all()

    def test_zeroed_padding_slots_do_not_produce_a_nan_gradient(self) -> None:
        """Padding is usually all zeros, so every dead pair sits at ``v = 0``.

        A padder leaves the tail of a molecule row zeroed, which makes *every*
        dead entry a coincident pair — the worst case for the substitution, and
        the one a real batch hits on every step.
        """
        pos = torch.tensor(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
            dtype=torch.float64,
            requires_grad=True,
        )
        mask = torch.tensor([[True, True, False, False]])
        _diff, dist, live = DenseNeighborList().geometry(pos, mask)
        (dist * live).sum().backward()
        assert torch.isfinite(pos.grad).all()
        assert torch.equal(pos.grad[0, 2:], torch.zeros(2, 3, dtype=torch.float64))

    def test_two_live_atoms_at_one_point_are_left_undefined(self) -> None:
        """Coincident *live* atoms are degenerate input, not a case to paper over.

        ``d‖v‖/dv = v/‖v‖`` has no value at ``v = 0``, and two atoms at one point
        have no Coulomb energy either. The substitution covers what this class
        controls — the diagonal and the padding it was told about — and does not
        invent a derivative for physics that has none. Pinned so the silence is
        a decision and not an oversight.
        """
        pos = torch.zeros(1, 2, 3, dtype=torch.float64, requires_grad=True)
        _diff, dist, live = DenseNeighborList().geometry(pos)
        (dist * live).sum().backward()
        assert not torch.isfinite(pos.grad).all()

    def test_it_agrees_with_the_sparse_list_on_one_molecule(self) -> None:
        """The sparse half-pair list is the oracle for which pairs exist."""
        pos = torch.tensor(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [5.0, 0.0, 0.0]]],
            dtype=torch.float64,
        )
        sparse = NeighborList(cutoff=3.0, symmetry=False)
        pair_index, _pair_diff, pair_dist = sparse.pairs(pos[0])
        _d, dense_dist, dense_live = DenseNeighborList(cutoff=3.0).geometry(pos)

        want = sorted(
            (min(int(a), int(b)), max(int(a), int(b)), round(float(r), 9))
            for (a, b), r in zip(pair_index, pair_dist, strict=True)
        )
        rows, cols = torch.nonzero(dense_live[0], as_tuple=True)
        got = sorted(
            (int(i), int(j), round(float(dense_dist[0, i, j]), 9))
            for i, j in zip(rows, cols, strict=True)
        )
        assert got == want

    def test_it_refuses_a_flat_position_tensor(self) -> None:
        """``(N, 3)`` is the sparse shape; taking it as ``(B, N)`` would be silent."""
        with pytest.raises(ValueError, match=r"\(B, N, 3\)"):
            DenseNeighborList().geometry(torch.zeros(4, 3))

    def test_it_refuses_a_mask_of_the_wrong_shape(self) -> None:
        with pytest.raises(ValueError, match=r"\(B, N\)"):
            DenseNeighborList().geometry(torch.zeros(2, 3, 3), torch.ones(2, 4, dtype=torch.bool))

    def test_a_non_positive_cutoff_is_refused(self) -> None:
        with pytest.raises(ValueError, match="cutoff"):
            DenseNeighborList(cutoff=0.0)

    @pytest.mark.slow
    def test_it_traces_as_one_graph_with_no_break(self) -> None:
        """A hot-path module compiles whole — CLAUDE.md's iron law.

        Asserts ``graph_count == 1`` alongside the break count: upstream defines
        ``graph_break_count = graph_count - 1``, so a target that never traced
        reports ``-1`` and a bare ``== 0`` check would read a total tracing
        failure as success.
        """
        from molix.compile import Compiler

        nlist = DenseNeighborList(cutoff=3.0)
        pos = torch.randn(2, 5, 3, dtype=torch.float64)
        mask = torch.ones(2, 5, dtype=torch.bool)
        summary = Compiler.explain(nlist.geometry, pos, mask)

        assert summary.graph_count == 1
        assert summary.graph_break_count == 0
