"""Tests for molrep.readout.product_head module."""

import pytest
import torch
from molrep.readout.product_head import ProductHead, ProductHeadSpec
from molix import config


class TestProductHeadSpec:
    """Test ProductHeadSpec configuration."""
    
    def test_valid_config(self):
        """Test creation with valid parameters."""
        spec = ProductHeadSpec(
            hidden_dim=64,
            out_dim=1,
            num_radial=8,
            l_max=2,
            max_body_order=2,
            num_species=10,
        )
        assert spec.hidden_dim == 64
        assert spec.out_dim == 1
        assert spec.num_radial == 8
        assert spec.l_max == 2
        assert spec.max_body_order == 2
        assert spec.num_species == 10
    
    def test_invalid_hidden_dim(self):
        """Test validation for hidden_dim."""
        with pytest.raises(ValueError):
            ProductHeadSpec(hidden_dim=0, out_dim=1)
    
    def test_invalid_out_dim(self):
        """Test validation for out_dim."""
        with pytest.raises(ValueError):
            ProductHeadSpec(hidden_dim=64, out_dim=0)


class TestProductHead:
    """Test ProductHead prediction layer."""
    
    def test_initialization(self):
        """Test ProductHead initialization."""
        head = ProductHead(
            hidden_dim=64,
            out_dim=1,
            num_radial=8,
            l_max=2,
            max_body_order=2,
            num_species=10,
        )
        assert head.config.hidden_dim == 64
        assert head.config.out_dim == 1
    
    def test_forward_shape(self):
        """Test output shape."""
        head = ProductHead(
            hidden_dim=64,
            out_dim=1,
            num_radial=8,
            l_max=2,
            max_body_order=2,
            num_species=10,
        )
        
        n_nodes = 20
        node_features = torch.randn(n_nodes, 64, dtype=config.ftype)
        atom_types = torch.randint(0, 10, (n_nodes,), dtype=torch.long)
        
        output = head(node_features, atom_types)
        assert output.shape == (n_nodes, 1)
    
    def test_different_output_dims(self):
        """Test with different output dimensions."""
        for out_dim in [1, 3, 5]:
            head = ProductHead(
                hidden_dim=32,
                out_dim=out_dim,
                num_species=5,
            )
            
            node_features = torch.randn(10, 32, dtype=config.ftype)
            atom_types = torch.randint(0, 5, (10,), dtype=torch.long)
            
            output = head(node_features, atom_types)
            assert output.shape == (10, out_dim)
    
    def test_differentiable(self):
        """Test that gradients flow through head."""
        head = ProductHead(
            hidden_dim=32,
            out_dim=1,
            num_species=5,
        )
        
        node_features = torch.randn(10, 32, requires_grad=True, dtype=config.ftype)
        atom_types = torch.randint(0, 5, (10,), dtype=torch.long)
        
        output = head(node_features, atom_types)
        loss = output.sum()
        loss.backward()
        
        assert node_features.grad is not None
        assert not torch.isnan(node_features.grad).any()
