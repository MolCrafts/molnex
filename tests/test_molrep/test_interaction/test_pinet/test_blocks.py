"""Unit tests for PiNet composite blocks (GCBlock, OutLayer)."""

from __future__ import annotations

import torch

from molrep.interaction.pinet import GCBlock, OutLayer


def test_outlayer_residual_accumulation():
    head = OutLayer([8], in_dim=4, out_units=1, activation="tanh")
    px = torch.randn(3, 4)
    prev = torch.zeros(3, 1)
    out1 = head(px, prev)
    out2 = head(px, out1)
    assert out1.shape == (3, 1)
    # Second call adds another residual delta onto the previous output.
    assert not torch.allclose(out2, out1)


def test_gcblock_rank3_output_keys():
    block = GCBlock(
        rank=3,
        weighted=False,
        pp_nodes=[8, 8],
        pi_nodes=[8, 8],
        ii_nodes=[8, 8],
        n_basis=3,
        p1_in_dim=4,
        p3_in_dim=1,
        activation="tanh",
    )
    n, e = 4, 6
    edge_index = torch.tensor([[0, 1], [1, 0], [0, 2], [2, 0], [1, 3], [3, 1]], dtype=torch.long)
    tensors = {
        "edge_index": edge_index,
        "p1": torch.randn(n, 4),
        "p3": torch.zeros(n, 3, 1),
        "d3": torch.nn.functional.normalize(torch.randn(e, 3), dim=-1),
    }
    basis = torch.randn(e, 3)
    out = block(tensors, basis)
    assert out["p1"].shape == (n, 8)
    assert out["p3"].shape == (n, 3, 8)
    assert out["i1"].shape[0] == e
