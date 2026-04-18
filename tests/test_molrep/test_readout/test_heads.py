"""Tests for physical derivation heads (now in molpot.derivation)."""

from __future__ import annotations

import pytest
import torch
from molpot.derivation import EnergyAggregation, ForceDerivation
from tests.utils import assert_module_compiles, assert_module_exports, assert_outputs_close


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
        energy = head(node_energy, batch)
        assert energy.shape == (2,)

    def test_forward_shape_sum_pooling(self):
        head = EnergyAggregation(pooling="sum")
        node_energy = torch.randn(15)
        batch = torch.tensor([0]*5 + [1]*5 + [2]*5)
        energy = head(node_energy, batch)
        assert energy.shape == (3,)

    def test_differentiable(self):
        head = EnergyAggregation(pooling="mean")
        node_energy = torch.randn(10, requires_grad=True)
        batch = torch.tensor([0]*5 + [1]*5)
        energy = head(node_energy, batch, num_graphs=2)
        loss = energy.sum()
        loss.backward()
        assert node_energy.grad is not None
        assert not torch.isnan(node_energy.grad).any()

    def test_compile(self):
        head = EnergyAggregation(pooling="mean")
        node_energy = torch.randn(10)
        batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        output_uncompiled, output_compiled = assert_module_compiles(
            head, node_energy, batch, 2
        )
        assert_outputs_close(output_uncompiled, output_compiled)

    def test_export(self):
        head = EnergyAggregation(pooling="mean")
        node_energy = torch.randn(10)
        batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        exported_program, output_original, output_exported = assert_module_exports(
            head, args_tuple=(node_energy, batch, 2),
        )
        assert_outputs_close(output_original, output_exported)


class TestForceDerivation:
    """Test ForceDerivation layer."""

    def test_forward_shape(self):
        head = ForceDerivation()
        pos = torch.randn(10, 3, requires_grad=True)
        energy = pos.pow(2).sum()
        forces = head(energy, pos)
        assert forces.shape == (10, 3)
        assert not torch.isnan(forces).any()
