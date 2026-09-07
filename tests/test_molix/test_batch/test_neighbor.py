"""Unit tests for :class:`molix.batch.neighbor.NeighborList`.

Sign-invariant pins for the edge geometry (edge_diff / edge_dist): guards the
load-bearing convention that survives the bond_*→edge_* rename,
``edge_diff[e] = pos[target] - pos[source]`` and, under ``symmetry=True``,
the reverse edge carries the exact negation (spherical-harmonic parity
``Y_l(-r) = (-1)^l Y_l(r)``). See spec graph-connectivity-alignment-02-rename.

``TestNeighborList`` moved here from ``test_data/test_transform.py`` with the
class (spec molix-data-dissolution-03-transforms).
"""

import re
from pathlib import Path

import pytest
import torch

from molix.batch.neighbor import NeighborList

#: The two backends of ``torch.ops.molix.get_neighbor_pairs``.
_KERNELS = Path(__file__).resolve().parents[3] / "src/molix/op/src/neighbors"


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
        out = NeighborList(cutoff=2.0, max_num_pairs=10, symmetry=symmetry)(sample)
        edge_index = out["edge_index"]
        expected = pos[edge_index[:, 1]] - pos[edge_index[:, 0]]
        assert torch.allclose(out["edge_diff"], expected, atol=1e-5)
        assert torch.allclose(out["edge_dist"], expected.norm(dim=1), atol=1e-5)


def test_symmetry_appends_reverse_edge_with_negated_edge_diff():
    sample = _triangle()
    out = NeighborList(cutoff=2.0, max_num_pairs=10, symmetry=True)(sample)
    edge_index = out["edge_index"]
    edge_diff = out["edge_diff"]
    n_pairs = edge_index.shape[0] // 2

    # Reverse edges are appended after the forward half: row k and row k+n_pairs
    # are (src→tgt) and (tgt→src) of the same pair, with negated displacement.
    fwd, rev = edge_index[:n_pairs], edge_index[n_pairs:]
    assert torch.equal(rev, fwd[:, [1, 0]])
    assert torch.allclose(edge_diff[n_pairs:], -edge_diff[:n_pairs], atol=1e-6)


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
    edge_dist = dist[i_idx, j_idx]
    return i_idx, j_idx, edge_vec, edge_dist


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

        result = NeighborList(cutoff=2.0, max_num_pairs=10)(sample)
        edge_index = result["edge_index"]

        ref_i, ref_j, _, ref_dist = _compute_neighbor_list_naive(positions, 2.0)
        assert edge_index.ndim == 2 and edge_index.shape[1] == 2
        assert edge_index.shape[0] == 2 * ref_i.numel()

        edges = {(int(a), int(b)) for a, b in edge_index.tolist()}
        for a, b in zip(ref_i.tolist(), ref_j.tolist()):
            assert (a, b) in edges and (b, a) in edges

        doubled = torch.cat([ref_dist, ref_dist]).sort().values
        assert torch.allclose(result["edge_dist"].sort().values, doubled, atol=1e-5)

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

        result = NeighborList(cutoff=2.0, max_num_pairs=10, symmetry=False)(sample)
        edge_index = result["edge_index"]

        ref_i, ref_j, _, ref_dist = _compute_neighbor_list_naive(positions, 2.0)
        assert edge_index.shape == (ref_i.numel(), 2)
        assert torch.allclose(result["edge_dist"].sort().values, ref_dist.sort().values, atol=1e-5)

    def test_bond_diff_source_to_target_convention(self):
        """``edge_diff[k] == pos[target_k] - pos[source_k]`` for every edge.

        Locks in the edge convention documented in CLAUDE.md: edge_index[:,0]
        is the source, edge_index[:,1] is the target, and edge_diff points
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
            )(sample)
            edge_index = result["edge_index"]
            expected = positions[edge_index[:, 1]] - positions[edge_index[:, 0]]
            assert torch.allclose(result["edge_diff"], expected, atol=1e-5), (
                f"edge_diff violates source→target convention (symmetry={symmetry})"
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

        result = NeighborList(cutoff=2.0, max_num_pairs=10)(sample)
        assert torch.all(result["edge_dist"] <= 2.0)

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
            NeighborList(cutoff=2.0, max_num_pairs=m, symmetry=False)(sample)
            for m in (1, 6, 512, -1)
        ]
        reference = outs[0]
        assert reference["edge_index"].shape == (6, 2)
        for out in outs[1:]:
            assert torch.equal(out["edge_index"], reference["edge_index"])
            assert torch.equal(out["edge_diff"], reference["edge_diff"])
            assert torch.equal(out["edge_dist"], reference["edge_dist"])

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

        out = NeighborList(cutoff=5.0, max_num_pairs=1, symmetry=False)(sample)

        assert out["edge_index"].shape == (6, 2)
        assert {(int(a), int(b)) for a, b in out["edge_index"].tolist()} == {
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

        monkeypatch.setattr("molix.batch.neighbor.get_neighbor_pairs", fake_kernel)

        pos = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float64,
        )
        sample = {"Z": torch.tensor([8, 1, 1]), "pos": pos}

        with pytest.raises(
            RuntimeError,
            match=r"Neighbor buffer overflow: found 4 pairs > max_num_pairs=3",
        ):
            NeighborList(cutoff=2.0, max_num_pairs=3)(sample)

    def test_nan_padding_rows_are_stripped_by_default(self):
        """``filter_padding=True`` leaves exactly the true pairs, no NaN rows.

        The kernel pads its fixed-width output with ``-1`` indices and NaN
        deltas/distances; a padded row surviving into ``edge_diff`` poisons
        every downstream sum.  3-atom right triangle: 3 half-pairs at
        1, 1, sqrt(2); buffer of 16 => 13 padded rows to strip.
        """
        pos = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float64,
        )
        sample = {"Z": torch.tensor([8, 1, 1]), "pos": pos}

        out = NeighborList(cutoff=2.0, max_num_pairs=16, symmetry=False)(sample)

        assert out["edge_index"].tolist() == [[1, 0], [2, 0], [2, 1]]
        assert not torch.isnan(out["edge_diff"]).any()
        assert not torch.isnan(out["edge_dist"]).any()
        assert (out["edge_index"] >= 0).all()
        assert torch.allclose(
            out["edge_dist"],
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
            sample
        )

        assert out["edge_index"].tolist() == [[1, 0], [2, 0], [2, 1], [-1, -1], [-1, -1]]
        assert torch.isnan(out["edge_dist"][3:]).all()
        assert torch.isnan(out["edge_diff"][3:]).all()

    def test_pbc_true_returns_minimum_image_edge_diff(self):
        """Minimum-image convention across the periodic boundary (hard-coded golden).

        Two atoms in a 2 A cubic cell at x = 0.1 and x = 1.9: the direct
        separation is 1.8 A (outside a 1.0 A cutoff), the minimum image is
        1.8 - 2.0 = -0.2 A.  With ``edge_diff = pos[target] - pos[source]``
        and the kernel emitting source = 1, target = 0, the forward edge is
        ``+0.2 x_hat`` and the appended reverse edge ``-0.2 x_hat``.
        """
        pos = torch.tensor([[0.1, 0.0, 0.0], [1.9, 0.0, 0.0]], dtype=torch.float64)
        cell = torch.tensor(
            [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]], dtype=torch.float64
        )
        sample = {"Z": torch.tensor([1, 1]), "pos": pos, "cell": cell}

        out = NeighborList(cutoff=1.0, max_num_pairs=8, pbc=True)(sample)

        assert out["edge_index"].tolist() == [[1, 0], [0, 1]]
        assert torch.allclose(
            out["edge_diff"],
            torch.tensor([[0.2, 0.0, 0.0], [-0.2, 0.0, 0.0]], dtype=torch.float64),
            atol=1e-12,
        )
        assert torch.allclose(
            out["edge_dist"],
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

        out = NeighborList(cutoff=1.0, max_num_pairs=8, pbc=False)(sample)

        assert out["edge_index"].shape == (0, 2)
        assert out["edge_dist"].numel() == 0

    def test_task_id_distinguishes_pbc(self):
        """``pbc`` is part of the cache key — PBC and open caches must not alias."""
        open_bc = NeighborList(cutoff=1.0, max_num_pairs=8, pbc=False)
        periodic = NeighborList(cutoff=1.0, max_num_pairs=8, pbc=True)

        assert open_bc.task_id == "nlist:cut=1.0:max=8:pbc=False:sym=True"
        assert periodic.task_id == "nlist:cut=1.0:max=8:pbc=True:sym=True"
        assert open_bc.task_id != periodic.task_id
