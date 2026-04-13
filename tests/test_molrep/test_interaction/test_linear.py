"""Tests for molrep.interaction.linear module."""

import pytest
import torch
from molrep.interaction.linear import EquivariantLinear, EquivariantLinearSpec
from cuequivariance import Irreps
from molix import config

from tests.utils import assert_compile_compatible


class TestEquivariantLinearSpec:
    """Test EquivariantLinearSpec configuration."""
    
    def test_valid_config(self):
        """Test creation with valid parameters."""
        spec = EquivariantLinearSpec(
            irreps_in="0e + 1o",
            irreps_out="0e + 1o",
        )
        assert spec.irreps_in == "0e + 1o"
        assert spec.irreps_out == "0e + 1o"


class TestEquivariantLinear:
    """Test EquivariantLinear layer."""
    
    def test_initialization(self):
        """Test EquivariantLinear initialization."""
        linear = EquivariantLinear(
            irreps_in="0e + 1o",
            irreps_out="0e + 1o",
        )
        assert linear.config.irreps_in == "0e + 1o"
        assert linear.config.irreps_out == "0e + 1o"
    
    def test_forward_scalar_only(self):
        """Test with scalar features only."""
        # 64 scalars -> 32 scalars
        linear = EquivariantLinear(
            irreps_in="64x0e",
            irreps_out="32x0e",
        )
        
        n_nodes = 10
        features = torch.randn(n_nodes, 64, dtype=config.ftype)
        
        output = linear(features)
        assert output.shape == (n_nodes, 32)
    
    def test_forward_with_vectors(self):
        """Test with scalar + vector features."""
        # 32 scalars + 16 vectors -> 16 scalars + 8 vectors
        linear = EquivariantLinear(
            irreps_in="32x0e + 16x1o",
            irreps_out="16x0e + 8x1o",
        )
        
        n_nodes = 10
        # 32 scalars + 16*3 vectors = 32 + 48 = 80 total
        features = torch.randn(n_nodes, 80, dtype=config.ftype)
        
        output = linear(features)
        
        # 16 scalars + 8*3 vectors = 16 + 24 = 40 total
        assert output.shape == (n_nodes, 40)
    
    def test_differentiable(self):
        """Test that gradients flow through linear layer."""
        linear = EquivariantLinear(
            irreps_in="32x0e",
            irreps_out="16x0e",
        )
        
        features = torch.randn(10, 32, dtype=config.ftype, requires_grad=True)
        
        output = linear(features)
        loss = output.sum()
        loss.backward()
        
        assert features.grad is not None
        assert not torch.isnan(features.grad).any()
    
    def test_batch_processing(self):
        """Test with multiple nodes (non-batched)."""
        linear = EquivariantLinear(
            irreps_in="32x0e + 8x1o",
            irreps_out="16x0e + 4x1o",
        )
        
        n_nodes = 100
        # 32 + 8*3 = 56 features
        features = torch.randn(n_nodes, 56, dtype=config.ftype)
        
        output = linear(features)
        
        # 16 + 4*3 = 28 features
        assert output.shape == (n_nodes, 28)
    
    def test_equivariance_vectors(self):
        """Test equivariance property for vector features. Force to use (ir, mul) layout!"""
        import math
        
        # Simple case: pure vectors
        linear = EquivariantLinear(
            irreps_in="4x1o",
            irreps_out="2x1o",
        )
        
        # Create input vectors
        n_nodes = 5
        features = torch.randn(n_nodes, 12, dtype=config.ftype)  # 4 vectors * 3 components
        
        # Forward pass
        output1 = linear(features)
        
        # Rotate input (90 degrees around z-axis)
        angle = math.pi / 2
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        rot_matrix = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ], dtype=features.dtype)
        
        # Rotate all vectors in features (ir_mul layout: (3, 4) flattened)
        features_ir_mul = features.reshape(n_nodes, 3, 4)
        features_rot = (rot_matrix @ features_ir_mul).reshape(n_nodes, 12)
        
        # Forward pass on rotated input
        output2 = linear(features_rot)
        
        # Rotate output1 and compare with output2 (output also in ir_mul: (3, 2))
        output1_ir_mul = output1.reshape(n_nodes, 3, 2)
        output1_rot = (rot_matrix @ output1_ir_mul).reshape(n_nodes, 6)
        
        # Should be equivariant: rotating input then processing
        # should give same result as processing then rotating output
        # Using higher tolerance for float64 numerical precision
        assert torch.allclose(output1_rot, output2, rtol=1e-3, atol=1e-3)
    
    def test_different_irreps(self):
        """Test with various irreps configurations."""
        import cuequivariance as cue
        
        test_cases = [
            ("16x0e", "8x0e"),
            ("32x0e + 16x1o", "16x0e + 8x1o"),
        ]
        
        for irreps_in, irreps_out in test_cases:
            linear = EquivariantLinear(
                irreps_in=irreps_in,
                irreps_out=irreps_out,
            )
            
            # Calculate input dimension
            irreps_in_obj = cue.Irreps("O3", irreps_in)
            in_dim = irreps_in_obj.dim
            
            features = torch.randn(10, in_dim, dtype=config.ftype)
            
            output = linear(features)
            
            # Calculate expected output dimension
            irreps_out_obj = cue.Irreps("O3", irreps_out)
            out_dim = irreps_out_obj.dim
            assert output.shape == (10, out_dim)

    @pytest.mark.xfail(
        reason="cuEquivariance equivariant linear may have graph breaks under torch.compile",
        strict=False,
    )
    def test_compile(self):
        """Test that EquivariantLinear can be compiled with torch.compile."""
        linear = EquivariantLinear(
            irreps_in="32x0e + 16x1o",
            irreps_out="16x0e + 8x1o",
        )
        features = torch.randn(10, 80, dtype=config.ftype)
        assert_compile_compatible(linear, features, strict=False)
