"""Unit tests for ``PadMolecularBatch.execute()`` — the fixed-length padding task.

Pure function tests: each exercises ``PadMolecularBatch.execute()`` on a collated
batch and asserts the structure of its return value (fixed shapes, the real-atom
mask, ghost-atom fields, self-loop padding edges, zero-padded tensors, overflow
errors). No model, no ``torch.compile``, no training loop.
"""

from __future__ import annotations

import pytest
import torch
from tests.conftest import make_graph_batch

from molix.data import PadMolecularBatch


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

    edge_index = torch.tensor(full_edges([0, 1, 2, 3]) + full_edges([4, 5, 6]), dtype=torch.long)
    b = make_graph_batch(pos, z, edge_index, batch)
    return b, pos.shape[0], edge_index.shape[0]


def test_execute_sets_fixed_shapes():
    b, n, e = _batch()
    out = PadMolecularBatch(max_atoms=n + 5, max_edges=e + 9).execute(b)
    assert out["atoms"].batch_size[0] == n + 5
    assert out["edges"].batch_size[0] == e + 9


def test_execute_mask_marks_real_atoms():
    b, n, e = _batch()
    out = PadMolecularBatch(max_atoms=n + 5, max_edges=e + 9).execute(b)
    mask = out["atoms", "mask"]
    assert mask.dtype == torch.bool
    assert mask.shape == (n + 5,)
    assert mask[:n].all()
    assert not mask[n:].any()
    assert int(mask.sum()) == n


def test_execute_ghost_atoms_get_pad_type_and_zeroed_fields():
    b, n, e = _batch()
    out = PadMolecularBatch(max_atoms=n + 5, max_edges=e + 9, pad_atom_type=1).execute(b)
    # ghost Z == pad_atom_type; ghost pos / batch zero-padded; real rows untouched.
    assert torch.equal(out["atoms", "Z"][n:], torch.ones(5, dtype=torch.long))
    assert out["atoms", "pos"][n:].abs().max().item() == 0.0
    assert int(out["atoms", "batch"][n:].abs().max()) == 0


def test_execute_padding_edges_are_ghost_self_loops():
    b, n, e = _batch()
    out = PadMolecularBatch(max_atoms=n + 5, max_edges=e + 9).execute(b)
    pad_edges = out["edges", "edge_index"][e:]
    # every padding edge self-loops on the first ghost atom (index n)
    assert torch.equal(pad_edges, torch.full((9, 2), n, dtype=pad_edges.dtype))
    # non-index edge fields are zero-padded
    assert out["edges", "edge_diff"][e:].abs().max().item() == 0.0
    assert out["edges", "edge_dist"][e:].abs().max().item() == 0.0


def test_execute_real_rows_preserved():
    b, n, e = _batch()
    z_real = b["atoms", "Z"][:n].clone()
    ei_real = b["edges", "edge_index"][:e].clone()
    out = PadMolecularBatch(max_atoms=n + 5, max_edges=e + 9).execute(b)
    assert torch.equal(out["atoms", "Z"][:n], z_real)
    assert torch.equal(out["edges", "edge_index"][:e], ei_real)


def test_execute_overflow_raises():
    b, n, e = _batch()
    # too few atom slots (need >= n + 1 for the ghost anchor) must raise, not truncate
    with pytest.raises(ValueError):
        PadMolecularBatch(max_atoms=n, max_edges=e + 9).execute(b)
    b2, n2, e2 = _batch()
    with pytest.raises(ValueError):
        PadMolecularBatch(max_atoms=n2 + 5, max_edges=e2 - 1).execute(b2)


def test_task_id_encodes_caps():
    task = PadMolecularBatch(max_atoms=40, max_edges=400)
    assert task.task_id == "PadMolecularBatch(a=40,e=400)"
