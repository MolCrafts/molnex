"""Unit tests for molrep.interaction.pinet.ResUpdate."""

from __future__ import annotations

import torch

from molrep.interaction.pinet import ResUpdate


def test_identity_when_dims_match():
    layer = ResUpdate(in_dim=4, out_dim=4)
    old = torch.randn(3, 4)
    new = torch.randn(3, 4)
    torch.testing.assert_close(layer(old, new), old + new)


def test_projects_when_dims_differ():
    layer = ResUpdate(in_dim=2, out_dim=5)
    old = torch.randn(3, 2)
    new = torch.randn(3, 5)
    out = layer(old, new)
    assert out.shape == (3, 5)
