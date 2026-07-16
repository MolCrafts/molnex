"""Unit tests for molrep.interaction.pinet message layers."""

from __future__ import annotations

import torch

from molrep.interaction.pinet import DotLayer, PIXLayer


def test_pix_unweighted_gathers_target_property():
    px = torch.arange(4 * 3 * 2, dtype=torch.float32).reshape(4, 3, 2)
    edge_index = torch.tensor([[0, 1], [2, 3]])
    src, dst = edge_index[:, 0], edge_index[:, 1]
    out = PIXLayer(channels=2, weighted=False)(src, dst, px)
    torch.testing.assert_close(out, px[dst])


def test_dot_weighted_shape():
    x = torch.randn(5, 3, 4)
    out = DotLayer(channels=4, weighted=True)(x)
    assert out.shape == (5, 4)


def test_dot_unweighted_matches_manual():
    x = torch.randn(3, 3, 2)
    out = DotLayer(channels=2, weighted=False)(x)
    torch.testing.assert_close(out, (x * x).sum(1))
