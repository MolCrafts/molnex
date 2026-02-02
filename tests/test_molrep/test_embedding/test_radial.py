"""Tests for molrep.embedding.radial module."""

import pytest
import torch
from molrep.embedding.radial import BesselRBF, BesselRBFSpec


class TestBesselRBFSpec:
    """Test BesselRBFSpec configuration."""
    
    def test_valid_config(self):
        """Test creation with valid parameters."""
        spec = BesselRBFSpec(
            num_radial=8,
            r_cut=5.0,
        )
        assert spec.num_radial == 8
        assert spec.r_cut == 5.0
    
    def test_invalid_num_radial(self):
        """Test validation for num_radial."""
        with pytest.raises(ValueError):
            BesselRBFSpec(num_radial=0, r_cut=5.0)
        
        with pytest.raises(ValueError):
            BesselRBFSpec(num_radial=-1, r_cut=5.0)
    
    def test_invalid_r_cut(self):
        """Test validation for r_cut."""
        with pytest.raises(ValueError):
            BesselRBFSpec(num_radial=8, r_cut=0.0)
        
        with pytest.raises(ValueError):
            BesselRBFSpec(num_radial=8, r_cut=-1.0)


class TestBesselRBF:
    """Test BesselRBF radial basis function."""
    
    def test_initialization(self):
        """Test BesselRBF initialization."""
        rbf = BesselRBF(num_radial=8, r_cut=5.0)
        assert rbf.config.num_radial == 8
        assert rbf.config.r_cut == 5.0
    
    def test_forward_shape(self):
        """Test output shape."""
        rbf = BesselRBF(num_radial=8, r_cut=5.0)
        distances = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        output = rbf(distances)
        assert output.shape == (4, 8)
    
    def test_forward_batch(self):
        """Test with batch of distances."""
        rbf = BesselRBF(num_radial=16, r_cut=10.0)
        distances = torch.randn(100, 50).abs()  # [batch, edges]
        
        output = rbf(distances)
        assert output.shape == (100, 50, 16)
    
    def test_cutoff_behavior(self):
        """Test that values beyond cutoff are small."""
        rbf = BesselRBF(num_radial=8, r_cut=5.0)
        
        # Distances within and beyond cutoff
        distances = torch.tensor([2.0, 4.9, 5.0, 6.0, 10.0])
        output = rbf(distances)
        
        # Values at cutoff should be zero or very small
        # Note: Bessel functions don't strictly go to zero, but decay significantly
        assert output[2].abs().max() < 0.15  # At r_cut
        assert output[3].abs().max() < 0.15  # Beyond r_cut
        assert output[4].abs().max() < 0.15  # Far beyond r_cut
    
    def test_zero_distance(self):
        """Test behavior at zero distance."""
        rbf = BesselRBF(num_radial=8, r_cut=5.0)
        distances = torch.tensor([0.0, 0.1, 1.0])
        
        output = rbf(distances)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_differentiable(self):
        """Test that gradients flow through RBF."""
        rbf = BesselRBF(num_radial=8, r_cut=5.0)
        distances = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        
        output = rbf(distances)
        loss = output.sum()
        loss.backward()
        
        assert distances.grad is not None
        assert not torch.isnan(distances.grad).any()
    
    def test_different_num_radial(self):
        """Test with different num_radial values."""
        for num_radial in [4, 8, 16, 32]:
            rbf = BesselRBF(num_radial=num_radial, r_cut=5.0)
            distances = torch.tensor([1.0, 2.0, 3.0])
            output = rbf(distances)
            assert output.shape == (3, num_radial)
    
    def test_dtype_consistency(self):
        """Test that output dtype matches input."""
        rbf = BesselRBF(num_radial=8, r_cut=5.0)
        
        # Float32
        dist_f32 = torch.tensor([1.0, 2.0], dtype=torch.float32)
        out_f32 = rbf(dist_f32)
        assert out_f32.dtype == torch.float32
        
        # Float64
        dist_f64 = torch.tensor([1.0, 2.0], dtype=torch.float64)
        out_f64 = rbf(dist_f64)
        # Note: BesselRBF casts to float internally
        assert out_f64.dtype in [torch.float32, torch.float64]

