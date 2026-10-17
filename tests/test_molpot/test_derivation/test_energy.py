"""Unit tests for molpot.derivation.energy.EnergyAggregation."""

from __future__ import annotations

import pytest
import torch

from molpot.derivation.energy import EnergyAggregation


def test_sum_pooling():
    node = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    batch = torch.tensor([0, 0, 0, 1, 1])
    out = EnergyAggregation(pooling="sum")(node, batch, num_graphs=2)
    torch.testing.assert_close(out, torch.tensor([0.6, 0.9]))


def test_mean_pooling():
    node = torch.tensor([1.0, 3.0, 5.0, 7.0])
    batch = torch.tensor([0, 0, 1, 1])
    out = EnergyAggregation(pooling="mean")(node, batch, num_graphs=2)
    torch.testing.assert_close(out, torch.tensor([2.0, 6.0]))


def test_invalid_pooling_is_rejected():
    with pytest.raises(ValueError):
        EnergyAggregation(pooling="invalid")


def test_mean_pooling_is_differentiable():
    node = torch.randn(10, requires_grad=True)
    batch = torch.tensor([0] * 5 + [1] * 5)
    EnergyAggregation(pooling="mean")(node, batch, num_graphs=2).sum().backward()
    assert node.grad is not None
    assert not torch.isnan(node.grad).any()
