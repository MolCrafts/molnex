"""Unit tests for molrep.interaction.pinet.FFLayer (explicit Linear stack)."""

from __future__ import annotations

import torch

from molrep.interaction.pinet import FFLayer


def test_fflayer_shapes_and_no_lazy():
    layer = FFLayer([8, 4], in_dim=6, activation="tanh")
    # Fully materialised: every parameter is a concrete Linear weight.
    for m in layer.modules():
        if isinstance(m, torch.nn.Linear):
            assert m.weight.shape[1] > 0
            assert m.weight.shape[0] > 0
    x = torch.randn(5, 6)
    y = layer(x)
    assert y.shape == (5, 4)
    assert layer.output_dim == 4


def test_fflayer_empty_is_identity():
    layer = FFLayer([], in_dim=3, activation=None)
    x = torch.randn(2, 3)
    torch.testing.assert_close(layer(x), x)
