"""Unit tests for molzoo.pinet.geometry."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from molzoo.pinet.geometry import compute_d5, edge_bond_diff


def test_edge_bond_diff_open_system():
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edges = TensorDict({}, batch_size=[2])
    diff = edge_bond_diff(edges, pos, edge_index)
    torch.testing.assert_close(diff[0], torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(diff[1], torch.tensor([-1.0, 0.0, 0.0]))


def test_edge_bond_diff_pbc_ste_grad():
    pos = torch.randn(3, 3, requires_grad=True)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    supplied = torch.randn(2, 3)
    edges = TensorDict(edge_diff=supplied, batch_size=[2])
    diff = edge_bond_diff(edges, pos, edge_index)
    # value follows supplied (detached), gradient follows raw.
    torch.testing.assert_close(diff, supplied)
    loss = diff.sum()
    loss.backward()
    assert pos.grad is not None
    assert pos.grad.abs().sum() > 0


def test_compute_d5_shape():
    d3 = torch.nn.functional.normalize(torch.randn(5, 3), dim=-1)
    d5 = compute_d5(d3)
    assert d5.shape == (5, 5)
