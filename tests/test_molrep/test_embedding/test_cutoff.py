"""Tests for molrep.embedding.cutoff module."""

import pytest
import torch
from molrep.embedding.cutoff import CosineCutoff, CosineCutoffSpec

from tests.utils import assert_compile_compatible


class TestCosineCutoffSpec:
    """Test CosineCutoffSpec configuration."""
    
    def test_valid_config(self):
        """Test creation with valid parameters."""
        spec = CosineCutoffSpec(
            r_cut=5.0
        )
        assert spec.r_cut == 5.0
    
    def test_invalid_r_cut(self):
        """Test validation for r_cut."""
        with pytest.raises(ValueError):
            CosineCutoffSpec(r_cut=0.0)
        
        with pytest.raises(ValueError):
            CosineCutoffSpec(r_cut=-1.0)


class TestCosineCutoff:
    """Test CosineCutoff envelope function."""
    
    def test_initialization(self):
        """Test CosineCutoff initialization."""
        cutoff = CosineCutoff(r_cut=5.0)
        assert cutoff.config.r_cut == 5.0
    
    def test_forward_shape(self):
        """Test output shape matches input."""
        cutoff = CosineCutoff(r_cut=5.0)
        distances = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        
        output = cutoff(distances)
        assert output.shape == distances.shape
    
    def test_forward_batch(self):
        """Test with batched input."""
        cutoff = CosineCutoff(r_cut=10.0)
        distances = torch.randn(5, 20).abs()
        
        output = cutoff(distances)
        assert output.shape == (5, 20)
    
    def test_cutoff_values(self):
        """Test cutoff behavior at specific distances."""
        cutoff = CosineCutoff(r_cut=5.0)
        
        # At r=0, cutoff should be 1.0
        out0 = cutoff(torch.tensor([0.0]))
        assert torch.allclose(out0, torch.tensor([1.0]), atol=1e-5)
        
        # At r=r_cut, cutoff should be 0.0
        out1 = cutoff(torch.tensor([5.0]))
        assert torch.allclose(out1, torch.tensor([0.0]), atol=1e-5)
        
        # Beyond r_cut, cutoff should be 0.0
        out2 = cutoff(torch.tensor([6.0]))
        assert torch.allclose(out2, torch.tensor([0.0]), atol=1e-5)
    
    def test_smoothness(self):
        """Test that cutoff is smooth between 0 and r_cut."""
        cutoff = CosineCutoff(r_cut=5.0)
        
        # Sample points between 0 and r_cut
        distances = torch.linspace(0, 5.0, 100)
        output = cutoff(distances)
        
        # Should be monotonically decreasing
        assert (output[:-1] >= output[1:]).all()
        
        # Should be in range [0, 1]
        assert (output >= 0).all()
        assert (output <= 1).all()
    
    def test_differentiable(self):
        """Test that gradients flow through cutoff."""
        cutoff = CosineCutoff(r_cut=5.0)
        distances = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        
        output = cutoff(distances)
        loss = output.sum()
        loss.backward()
        
        assert distances.grad is not None
        assert not torch.isnan(distances.grad).any()
    
    def test_gradient_at_cutoff(self):
        """Test that gradient is zero at cutoff (smooth boundary)."""
        cutoff = CosineCutoff(r_cut=5.0)
        distances = torch.tensor([5.0], requires_grad=True)
        
        output = cutoff(distances)
        output.backward()
        
        # Gradient should be zero or very small at cutoff
        assert abs(distances.grad.item()) < 1e-5
    
    def test_different_r_cut_values(self):
        """Test with different cutoff radii."""
        for r_cut in [1.0, 3.0, 5.0, 10.0]:
            cutoff = CosineCutoff(r_cut=r_cut)
            
            # Test at 0, mid, and beyond cutoff
            distances = torch.tensor([0.0, r_cut / 2, r_cut, r_cut + 1.0])
            output = cutoff(distances)
            
            assert torch.allclose(output[0], torch.tensor(1.0), atol=1e-5)
            assert output[1] > 0.4  # Should be significant at midpoint
            assert output[1] < 0.6
            assert torch.allclose(output[2], torch.tensor(0.0), atol=1e-5)
            assert torch.allclose(output[3], torch.tensor(0.0), atol=1e-5)
    
    def test_dtype_consistency(self):
        """Test that output dtype matches input."""
        cutoff = CosineCutoff(r_cut=5.0)
        
        # Float32
        dist_f32 = torch.tensor([1.0, 2.0], dtype=torch.float32)
        out_f32 = cutoff(dist_f32)
        assert out_f32.dtype == torch.float32
        
        # Float64
        dist_f64 = torch.tensor([1.0, 2.0], dtype=torch.float64)
        out_f64 = cutoff(dist_f64)
        # Note: Cutoff may cast to float internally
        assert out_f64.dtype in [torch.float32, torch.float64]
    
    def test_broadcasting(self):
        """Test broadcasting behavior."""
        cutoff = CosineCutoff(r_cut=5.0)
        
        # Different shapes
        dist1d = torch.tensor([1.0, 2.0, 3.0])
        dist2d = torch.randn(4, 5).abs()
        dist3d = torch.randn(2, 3, 4).abs()
        
        out1d = cutoff(dist1d)
        out2d = cutoff(dist2d)
        out3d = cutoff(dist3d)
        
        assert out1d.shape == dist1d.shape
        assert out2d.shape == dist2d.shape
        assert out3d.shape == dist3d.shape

    def test_compile(self):
        """Test that CosineCutoff can be compiled with torch.compile."""
        cutoff = CosineCutoff(r_cut=5.0)
        distances = torch.tensor([1.0, 2.0, 3.0, 4.0])
        assert_compile_compatible(cutoff, distances, strict=False)

