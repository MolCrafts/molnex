"""Tests for molrep.readout.basis_projection module."""

import pytest
import torch
from molrep.readout.basis_projection import BasisProjection, BasisProjectionSpec

from tests.utils import assert_compile_compatible


class TestBasisProjectionSpec:
    """Test BasisProjectionSpec configuration."""
    
    def test_valid_config(self):
        """Test creation with valid parameters."""
        spec = BasisProjectionSpec(
            hidden_dim=64,
            num_radial=8,
            l_max=2,
            max_body_order=2,
        )
        assert spec.hidden_dim == 64
        assert spec.num_radial == 8
        assert spec.l_max == 2
        assert spec.max_body_order == 2
    
    def test_invalid_hidden_dim(self):
        """Test validation for hidden_dim."""
        with pytest.raises(ValueError):
            BasisProjectionSpec(hidden_dim=0)


class TestBasisProjection:
    """Test BasisProjection layer."""
    
    def test_initialization(self):
        """Test BasisProjection initialization."""
        proj = BasisProjection(
            hidden_dim=64,
            num_radial=8,
            l_max=2,
            max_body_order=2,
        )
        assert proj.config.hidden_dim == 64
        assert proj.config.num_radial == 8
    
    def test_forward_shape(self):
        """Test that output shape matches input (passthrough)."""
        proj = BasisProjection(
            hidden_dim=64,
            num_radial=8,
            l_max=2,
            max_body_order=2,
        )
        
        basis_tensor = torch.randn(20, 64)
        output = proj(basis_tensor)
        
        # Should be identity/passthrough
        assert output.shape == basis_tensor.shape
    
    def test_passthrough_behavior(self):
        """Test that projection acts as passthrough (identity)."""
        proj = BasisProjection(
            hidden_dim=32,
            num_radial=8,
            l_max=2,
            max_body_order=2,
        )
        
        basis_tensor = torch.randn(10, 32)
        output = proj(basis_tensor)
        
        # Should return input unchanged (passthrough)
        assert torch.allclose(output, basis_tensor)
    
    def test_different_hidden_dims(self):
        """Test with different hidden dimensions."""
        for hidden_dim in [32, 64, 128]:
            proj = BasisProjection(
                hidden_dim=hidden_dim,
                num_radial=8,
                l_max=2,
                max_body_order=2,
            )
            
            basis_tensor = torch.randn(15, hidden_dim)
            output = proj(basis_tensor)
            
            assert output.shape == (15, hidden_dim)
    
    def test_differentiable(self):
        """Test that gradients flow through projection."""
        proj = BasisProjection(
            hidden_dim=32,
            num_radial=8,
            l_max=2,
            max_body_order=2,
        )
        
        basis_tensor = torch.randn(10, 32, requires_grad=True)
        output = proj(basis_tensor)
        loss = output.sum()
        loss.backward()
        
        assert basis_tensor.grad is not None
        # For passthrough, gradient should be ones
        assert torch.allclose(basis_tensor.grad, torch.ones_like(basis_tensor))

    def test_compile(self):
        """Test that BasisProjection can be compiled with torch.compile."""
        proj = BasisProjection(
            hidden_dim=64,
            num_radial=8,
            l_max=2,
            max_body_order=2,
        )
        basis_tensor = torch.randn(20, 64)
        assert_compile_compatible(proj, basis_tensor, strict=False)
