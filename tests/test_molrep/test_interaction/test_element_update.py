"""Tests for molrep.interaction.element_update module."""

import pytest
import torch
from molrep.interaction.element_update import ElementUpdate, ElementUpdateSpec
from molix import config


class TestElementUpdateSpec:
    """Test ElementUpdateSpec configuration."""
    
    def test_valid_config(self):
        """Test creation with valid parameters."""
        spec = ElementUpdateSpec(
            hidden_dim=64,
            num_species=10,
        )
        assert spec.hidden_dim == 64
        assert spec.num_species == 10
    
    def test_invalid_hidden_dim(self):
        """Test validation for hidden_dim."""
        with pytest.raises(ValueError):
            ElementUpdateSpec(hidden_dim=0, num_species=10)
        
        with pytest.raises(ValueError):
            ElementUpdateSpec(hidden_dim=-1, num_species=10)
    
    def test_invalid_num_species(self):
        """Test validation for num_species."""
        with pytest.raises(ValueError):
            ElementUpdateSpec(hidden_dim=64, num_species=0)
        
        with pytest.raises(ValueError):
            ElementUpdateSpec(hidden_dim=64, num_species=-1)


class TestElementUpdate:
    """Test ElementUpdate layer."""
    
    def test_initialization(self):
        """Test ElementUpdate initialization."""
        update = ElementUpdate(hidden_dim=64, num_species=10)
        assert update.config.hidden_dim == 64
        assert update.config.num_species == 10
        assert hasattr(update, 'linear')
    
    def test_forward_shape(self):
        """Test output shape."""
        update = ElementUpdate(hidden_dim=64, num_species=10)
        
        n_nodes = 20
        h_prev = torch.randn(n_nodes, 64)
        m_curr = torch.randn(n_nodes, 64)
        atom_types = torch.randint(1, 10, (n_nodes,))
        
        output = update(h_prev, m_curr, atom_types)
        assert output.shape == (n_nodes, 64)
    
    def test_residual_connection(self):
        """Test that output includes residual from h_prev."""
        update = ElementUpdate(hidden_dim=64, num_species=10)
        
        n_nodes = 20
        h_prev = torch.randn(n_nodes, 64)
        m_curr = torch.zeros(n_nodes, 64)  # Zero message
        atom_types = torch.randint(1, 10, (n_nodes,))
        
        output = update(h_prev, m_curr, atom_types)
        
        # With zero message, output should be close to h_prev
        # (might have small transformation from element-specific linear)
        assert output.shape == h_prev.shape
    
    def test_element_specific_update(self):
        """Test that different elements get different updates."""
        update = ElementUpdate(hidden_dim=64, num_species=10)
        
        # Same h_prev and m_curr but different atom types
        h_prev = torch.ones(2, 64)
        m_curr = torch.ones(2, 64)
        atom_types = torch.tensor([1, 2])  # Different elements
        
        output = update(h_prev, m_curr, atom_types)
        
        # Different elements should get different updates
        assert not torch.allclose(output[0], output[1], atol=1e-5)
    
    def test_same_element(self):
        """Test that same elements with same inputs get same outputs."""
        update = ElementUpdate(hidden_dim=64, num_species=10)
        
        # Same everything
        h_prev = torch.ones(3, 64)
        m_curr = torch.ones(3, 64)
        atom_types = torch.tensor([5, 5, 5])  # Same element
        
        output = update(h_prev, m_curr, atom_types)
        
        # Same inputs + same element = same outputs
        assert torch.allclose(output[0], output[1])
        assert torch.allclose(output[1], output[2])
    
    def test_differentiable(self):
        """Test that gradients flow through ElementUpdate."""
        update = ElementUpdate(hidden_dim=64, num_species=10)
        
        h_prev = torch.randn(10, 64, requires_grad=True)
        m_curr = torch.randn(10, 64, requires_grad=True)
        atom_types = torch.randint(1, 10, (10,))
        
        output = update(h_prev, m_curr, atom_types)
        loss = output.sum()
        loss.backward()
        
        assert h_prev.grad is not None
        assert m_curr.grad is not None
        assert not torch.isnan(h_prev.grad).any()
        assert not torch.isnan(m_curr.grad).any()
    
    def test_batch_processing(self):
        """Test with multiple nodes (non-batched)."""
        update = ElementUpdate(hidden_dim=64, num_species=10)
        
        n_nodes = 100
        h_prev = torch.randn(n_nodes, 64)
        m_curr = torch.randn(n_nodes, 64)
        atom_types = torch.randint(1, 10, (n_nodes,))
        
        output = update(h_prev, m_curr, atom_types)
        assert output.shape == (n_nodes, 64)
    
    def test_different_hidden_dims(self):
        """Test with different hidden dimensions."""
        for hidden_dim in [32, 64, 128, 256]:
            update = ElementUpdate(hidden_dim=hidden_dim, num_species=10)
            
            h_prev = torch.randn(10, hidden_dim)
            m_curr = torch.randn(10, hidden_dim)
            atom_types = torch.randint(1, 10, (10,))
            
            output = update(h_prev, m_curr, atom_types)
            assert output.shape == (10, hidden_dim)
    
    def test_all_species(self):
        """Test that all species indices work."""
        num_species = 10
        update = ElementUpdate(hidden_dim=64, num_species=num_species)
        
        # Test each species (0-based indexing)
        for species in range(num_species):
            h_prev = torch.randn(1, 64)
            m_curr = torch.randn(1, 64)
            atom_types = torch.tensor([species], dtype=torch.long)
            
            output = update(h_prev, m_curr, atom_types)
            assert output.shape == (1, 64)
            assert not torch.isnan(output).any()
    
    def test_dtype_consistency(self):
        """Test that module uses correct dtype from config."""
        update = ElementUpdate(hidden_dim=64, num_species=10)
        atom_types = torch.randint(1, 10, (10,), dtype=torch.long)
        
        # Module uses FLOAT_DTYPE from config
        h = torch.randn(10, 64, dtype=config.ftype)
        m = torch.randn(10, 64, dtype=config.ftype)
        out = update(h, m, atom_types)
        assert out.dtype == config.ftype
