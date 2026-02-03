"""Tests for molrep.readout.heads module."""

import pytest
import torch
from molrep.readout.heads import EnergyHead, ForceHead
from tests.utils import assert_module_compiles, assert_module_exports, assert_outputs_close


class TestEnergyHead:
    """Test EnergyHead prediction layer."""
    
    def test_initialization(self):
        """Test EnergyHead initialization."""
        head = EnergyHead(pooling="mean")
        assert head.pooling == "mean"
    
    def test_invalid_pooling(self):
        """Test validation for pooling strategy."""
        with pytest.raises(ValueError):
            EnergyHead(pooling="invalid")
    
    def test_forward_shape_mean_pooling(self):
        """Test output shape with mean pooling."""
        head = EnergyHead(pooling="mean")
        
        n_atoms = 10
        node_energy = torch.randn(n_atoms)
        batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        
        energy = head(node_energy, batch)
        
        # Should output one energy per molecule
        assert energy.shape == (2,)
    
    def test_forward_shape_sum_pooling(self):
        """Test output shape with sum pooling."""
        head = EnergyHead(pooling="sum")
        
        n_atoms = 15
        node_energy = torch.randn(n_atoms)
        batch = torch.tensor([0]*5 + [1]*5 + [2]*5)
        
        energy = head(node_energy, batch)
        assert energy.shape == (3,)
    
    def test_differentiable(self):
        """Test that gradients flow through head."""
        head = EnergyHead(pooling="mean")

        node_energy = torch.randn(10, requires_grad=True)
        batch = torch.tensor([0]*5 + [1]*5)
        num_graphs = 2

        energy = head(node_energy, batch, num_graphs)
        loss = energy.sum()
        loss.backward()

        assert node_energy.grad is not None
        assert not torch.isnan(node_energy.grad).any()

    def test_compile(self):
        """Test that EnergyHead can be compiled with torch.compile."""
        head = EnergyHead(pooling="mean")

        n_atoms = 10
        node_energy = torch.randn(n_atoms)
        batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        num_graphs = 2

        # Test compilation
        output_uncompiled, output_compiled = assert_module_compiles(
            head,
            node_energy, batch, num_graphs
        )

        # Check outputs match
        assert_outputs_close(output_uncompiled, output_compiled)

    def test_export(self):
        """Test that EnergyHead can be exported with torch.export."""
        head = EnergyHead(pooling="mean")

        n_atoms = 10
        node_energy = torch.randn(n_atoms)
        batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        num_graphs = 2

        # Test export with explicit num_graphs
        exported_program, output_original, output_exported = assert_module_exports(
            head,
            args_tuple=(node_energy, batch, num_graphs),
        )

        # Check outputs match
        assert_outputs_close(output_original, output_exported)


class TestForceHead:
    """Test ForceHead prediction layer."""

    def test_forward_shape(self):
        """Test force computation shape."""
        head = ForceHead()

        pos = torch.randn(10, 3, requires_grad=True)
        energy = pos.pow(2).sum() # Dummy energy function

        forces = head(energy, pos)
        assert forces.shape == (10, 3)
        assert not torch.isnan(forces).any()
