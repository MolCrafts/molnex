"""Sign-invariant pins for the renamed edge geometry (edge_diff / edge_dist).

Guards the load-bearing convention that survives the bond_*→edge_* rename:
``edge_diff[e] = pos[target] - pos[source]`` and, under ``symmetry=True``,
the reverse edge carries the exact negation (spherical-harmonic parity
``Y_l(-r) = (-1)^l Y_l(r)``). See spec graph-connectivity-alignment-02-rename.
"""

import torch

from molix.data.tasks import NeighborList


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
