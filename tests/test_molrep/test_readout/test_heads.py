"""Tests for physical derivation heads (now in molpot.derivation)."""

from __future__ import annotations

import pytest
import torch

from molpot.derivation import EnergyAggregation, ForceDerivation


class TestEnergyAggregation:
    """Test EnergyAggregation pooling layer."""

    def test_initialization(self):
        head = EnergyAggregation(pooling="mean")
        assert head.pooling == "mean"

    def test_invalid_pooling(self):
        with pytest.raises(ValueError):
            EnergyAggregation(pooling="invalid")

    def test_forward_shape_mean_pooling(self):
        head = EnergyAggregation(pooling="mean")
        node_energy = torch.randn(10)
        batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        energy = head(node_energy, batch, num_graphs=2)
        assert energy.shape == (2,)

    def test_forward_shape_sum_pooling(self):
        head = EnergyAggregation(pooling="sum")
        node_energy = torch.randn(15)
        batch = torch.tensor([0] * 5 + [1] * 5 + [2] * 5)
        energy = head(node_energy, batch, num_graphs=3)
        assert energy.shape == (3,)

    def test_differentiable(self):
        head = EnergyAggregation(pooling="mean")
        node_energy = torch.randn(10, requires_grad=True)
        batch = torch.tensor([0] * 5 + [1] * 5)
        energy = head(node_energy, batch, num_graphs=2)
        loss = energy.sum()
        loss.backward()
        assert node_energy.grad is not None
        assert not torch.isnan(node_energy.grad).any()


class TestForceDerivation:
    """Test ForceDerivation layer."""

    def test_forward_shape(self):
        head = ForceDerivation()
        pos = torch.randn(10, 3)
        forces = head(lambda p: p.pow(2).sum(), pos)
        assert forces.shape == (10, 3)
        assert not torch.isnan(forces).any()
