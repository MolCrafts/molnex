"""Tests for molrep.interaction.symmetric_contraction module."""

import pytest
import torch
from molrep.interaction.symmetric_contraction import SymmetricContraction, SymmetricContractionSpec
from molix import config


class TestSymmetricContractionSpec:
    """Test SymmetricContractionSpec configuration."""
    
    def test_valid_config(self):
        """Test creation with valid parameters."""
        spec = SymmetricContractionSpec(
            hidden_dim=64,
            num_species=10,
            max_body_order=2,
        )
        assert spec.hidden_dim == 64
        assert spec.num_species == 10
        assert spec.max_body_order == 2
    
    def test_invalid_hidden_dim(self):
        """Test validation for hidden_dim."""
        with pytest.raises(ValueError):
            SymmetricContractionSpec(hidden_dim=0, num_species=10)
    
    def test_invalid_num_species(self):
        """Test validation for num_species."""
        with pytest.raises(ValueError):
            SymmetricContractionSpec(hidden_dim=64, num_species=0)
    
    def test_invalid_max_body_order(self):
        """Test validation for max_body_order."""
        with pytest.raises(ValueError):
            SymmetricContractionSpec(hidden_dim=64, num_species=10, max_body_order=0)
        
        with pytest.raises(ValueError):
            SymmetricContractionSpec(hidden_dim=64, num_species=10, max_body_order=4)


class TestSymmetricContraction:
    """Test SymmetricContraction multi-body basis construction."""
    
    def test_initialization(self):
        """Test SymmetricContraction initialization."""
        contraction = SymmetricContraction(
            hidden_dim=64,
            num_species=10,
            max_body_order=2,
        )
        assert contraction.config.hidden_dim == 64
        assert contraction.config.num_species == 10
        assert contraction.config.max_body_order == 2
    
    def test_forward_shape(self):
        """Test output shape matches input."""
        contraction = SymmetricContraction(
            hidden_dim=64,
            num_species=10,
            max_body_order=2,
        )
        
        n_nodes = 20
        node_features = torch.randn(n_nodes, 64, dtype=config.ftype)
        atom_types = torch.randint(0, 10, (n_nodes,), dtype=torch.long)
        
        output = contraction(node_features, atom_types)
        assert output.shape == (n_nodes, 64)
    
    def test_different_body_orders(self):
        """Test with different max_body_order values."""
        for max_body_order in [1, 2, 3]:
            contraction = SymmetricContraction(
                hidden_dim=32,
                num_species=5,
                max_body_order=max_body_order,
            )
            
            node_features = torch.randn(10, 32, dtype=config.ftype)
            atom_types = torch.randint(0, 5, (10,), dtype=torch.long)
            
            output = contraction(node_features, atom_types)
            assert output.shape == (10, 32)
    
    def test_element_specific(self):
        """Test that different elements produce different contractions."""
        contraction = SymmetricContraction(
            hidden_dim=32,
            num_species=5,
            max_body_order=2,
        )
        
        # Same features, different atom types
        node_features = torch.ones(2, 32, dtype=config.ftype)
        atom_types = torch.tensor([0, 1], dtype=torch.long)
        
        output = contraction(node_features, atom_types)
        
        # Different species should produce different outputs
        assert not torch.allclose(output[0], output[1], atol=1e-5)
    
    def test_differentiable(self):
        """Test that gradients flow through contraction."""
        contraction = SymmetricContraction(
            hidden_dim=32,
            num_species=5,
            max_body_order=2,
        )
        
        node_features = torch.randn(10, 32, requires_grad=True, dtype=config.ftype)
        atom_types = torch.randint(0, 5, (10,), dtype=torch.long)
        
        output = contraction(node_features, atom_types)
        loss = output.sum()
        loss.backward()
        
        assert node_features.grad is not None
        assert not torch.isnan(node_features.grad).any()
