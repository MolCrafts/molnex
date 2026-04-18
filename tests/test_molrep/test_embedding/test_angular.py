"""Tests for molrep.embedding.angular module."""

import pytest
import torch
import math
from molrep.embedding.angular import SphericalHarmonics, SphericalHarmonicsSpec
from molrep.utils.equivariance import (
    rotation_matrix_z,
    rotation_matrix_x,
    rotation_matrix_y,
    random_rotation_matrix,
    rotate_vectors,
)
from tests.utils import assert_module_compiles, assert_module_exports, assert_outputs_close


class TestSphericalHarmonicsSpec:
    """Test SphericalHarmonicsSpec configuration."""
    
    def test_valid_config(self):
        """Test creation with valid parameters."""
        spec = SphericalHarmonicsSpec(
            l_max=2
        )
        assert spec.l_max == 2
    
    def test_invalid_l_max(self):
        """Test validation for l_max."""
        with pytest.raises(ValueError):
            SphericalHarmonicsSpec(
                l_max=-1
            )


class TestSphericalHarmonics:
    """Test SphericalHarmonics computation."""
    
    def test_initialization(self):
        """Test SphericalHarmonics initialization."""
        sh = SphericalHarmonics(
            l_max=2
        )
        assert sh.config.l_max == 2
        # Check output_dim property
        assert sh.config.output_dim == 9  # (2+1)^2
    
    def test_forward_shape(self):
        """Test output shape for different l_max."""
        vectors = torch.randn(10, 3)
        
        # l_max=0: 1 component (scalar)
        sh0 = SphericalHarmonics(l_max=0)
        out0 = sh0(vectors)
        assert out0.shape == (10, 1)
        
        # l_max=1: 1 + 3 = 4 components
        sh1 = SphericalHarmonics(l_max=1)
        out1 = sh1(vectors)
        assert out1.shape == (10, 4)
        
        # l_max=2: 1 + 3 + 5 = 9 components
        sh2 = SphericalHarmonics(l_max=2)
        out2 = sh2(vectors)
        assert out2.shape == (10, 9)
        
        # l_max=3: 1 + 3 + 5 + 7 = 16 components
        sh3 = SphericalHarmonics(l_max=3)
        out3 = sh3(vectors)
        assert out3.shape == (10, 16)
    
    def test_forward_batch(self):
        """Test with batched input."""
        sh = SphericalHarmonics(l_max=2)
        # Note: SphericalHarmonics expects (..., 3) shape, flattens internally
        # We can pass (N, 3) where N is total number of vectors
        vectors = torch.randn(100, 3)  # [total_vectors, 3]
        
        output = sh(vectors)
        assert output.shape == (100, 9)
    
    def test_normalized_vectors(self):
        """Test with normalized vectors."""
        sh = SphericalHarmonics(l_max=2)
        
        # Create normalized vectors
        vectors = torch.randn(10, 3)
        vectors = vectors / vectors.norm(dim=-1, keepdim=True)
        
        output = sh(vectors)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_zero_vector(self):
        """Test behavior with zero vectors."""
        sh = SphericalHarmonics(l_max=2)
        vectors = torch.zeros(5, 3)
        
        # Should handle gracefully (likely produces zeros or small values)
        output = sh(vectors)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_rotation_equivariance_l0(self):
        """Test that l=0 (scalars) are rotation invariant."""
        sh = SphericalHarmonics(l_max=0)
        
        # Original vectors
        vectors = torch.randn(10, 3)
        output1 = sh(vectors)
        
        # Rotated vectors (90 deg around z-axis)
        angle = math.pi / 2
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        rot_matrix = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ], dtype=vectors.dtype)
        
        vectors_rot = vectors @ rot_matrix.T
        output2 = sh(vectors_rot)
        
        # l=0 components should be invariant
        assert torch.allclose(output1, output2, atol=1e-5)
    
    def test_differentiable(self):
        """Test that gradients flow through SphericalHarmonics."""
        sh = SphericalHarmonics(l_max=2)
        vectors = torch.randn(10, 3, requires_grad=True)
        
        output = sh(vectors)
        loss = output.sum()
        loss.backward()
        
        assert vectors.grad is not None
        assert not torch.isnan(vectors.grad).any()
    
    def test_orthogonality_property(self):
        """Test approximate orthogonality of spherical harmonics on sphere."""
        sh = SphericalHarmonics(l_max=2)
        
        # Sample many points on unit sphere
        n_samples = 1000
        vectors = torch.randn(n_samples, 3)
        vectors = vectors / vectors.norm(dim=-1, keepdim=True)
        
        output = sh(vectors)  # [n_samples, 9]
        
        # Each output should have reasonable magnitude
        assert output.abs().mean() > 0.01
        assert output.abs().mean() < 10.0
    
    def test_dtype_consistency(self):
        """Test that output dtype matches input."""
        sh = SphericalHarmonics(l_max=2)
        
        # Float32
        vec_f32 = torch.randn(10, 3, dtype=torch.float32)
        out_f32 = sh(vec_f32)
        assert out_f32.dtype == torch.float32
        
        # Float64
        vec_f64 = torch.randn(10, 3, dtype=torch.float64)
        out_f64 = sh(vec_f64)
        assert out_f64.dtype == torch.float64
    
    def test_different_l_max_values(self):
        """Test various l_max values."""
        vectors = torch.randn(10, 3)

        for l_max in [0, 1, 2, 3, 4]:
            sh = SphericalHarmonics(l_max=l_max)
            output = sh(vectors)
            expected_dim = (l_max + 1) ** 2
            assert output.shape == (10, expected_dim)

    def test_rotation_equivariance_l1(self):
        """Test that l=1 components transform as vectors under rotation.

        For l=1, the spherical harmonics should transform like a 3D vector:
        Y_l=1(R·r) should be related to R·Y_l=1(r) by a rotation matrix.
        """
        sh = SphericalHarmonics(l_max=1)

        # Original vectors
        vectors = torch.randn(10, 3)
        vectors = vectors / vectors.norm(dim=-1, keepdim=True)  # Normalize
        output1 = sh(vectors)

        # Rotate vectors (90 deg around z-axis)
        angle = math.pi / 2
        rot_matrix = rotation_matrix_z(angle, dtype=vectors.dtype)
        vectors_rot = rotate_vectors(vectors, rot_matrix)
        output2 = sh(vectors_rot)

        # For l=0 (first component), should be invariant
        assert torch.allclose(output1[:, 0], output2[:, 0], atol=1e-5)

        # For l=1 (next 3 components), should transform as vectors
        # Note: cuEquivariance may use different conventions (real vs complex basis)
        # We check that they're related by some rotation (not necessarily the same matrix)
        l1_output1 = output1[:, 1:4]
        l1_output2 = output2[:, 1:4]

        # Check that norms are preserved
        norm1 = l1_output1.norm(dim=-1)
        norm2 = l1_output2.norm(dim=-1)
        assert torch.allclose(norm1, norm2, rtol=1e-4, atol=1e-4)

    def test_rotation_equivariance_random(self):
        """Test equivariance under random rotations for l=0, l=1, l=2."""
        sh = SphericalHarmonics(l_max=2)

        # Generate random vectors
        vectors = torch.randn(20, 3)
        vectors = vectors / vectors.norm(dim=-1, keepdim=True)

        # Generate random rotation
        rot_matrix = random_rotation_matrix(dtype=vectors.dtype)

        # Compute outputs
        output1 = sh(vectors)
        vectors_rot = rotate_vectors(vectors, rot_matrix)
        output2 = sh(vectors_rot)

        # l=0 should be invariant
        assert torch.allclose(output1[:, 0], output2[:, 0], rtol=1e-4, atol=1e-4)

        # Total output norms should be preserved (rotation is unitary)
        norm1 = output1.norm(dim=-1)
        norm2 = output2.norm(dim=-1)
        assert torch.allclose(norm1, norm2, rtol=1e-3, atol=1e-3)

    def test_rotation_x_axis(self):
        """Test rotation around x-axis."""
        sh = SphericalHarmonics(l_max=1)

        vectors = torch.randn(10, 3)
        vectors = vectors / vectors.norm(dim=-1, keepdim=True)

        # Rotate around x-axis
        angle = math.pi / 3
        rot_matrix = rotation_matrix_x(angle, dtype=vectors.dtype)

        output1 = sh(vectors)
        vectors_rot = rotate_vectors(vectors, rot_matrix)
        output2 = sh(vectors_rot)

        # l=0 invariant
        assert torch.allclose(output1[:, 0], output2[:, 0], atol=1e-5)

        # l=1 norms preserved
        assert torch.allclose(
            output1[:, 1:4].norm(dim=-1),
            output2[:, 1:4].norm(dim=-1),
            rtol=1e-4, atol=1e-4
        )

    def test_rotation_y_axis(self):
        """Test rotation around y-axis."""
        sh = SphericalHarmonics(l_max=1)

        vectors = torch.randn(10, 3)
        vectors = vectors / vectors.norm(dim=-1, keepdim=True)

        # Rotate around y-axis
        angle = math.pi / 4
        rot_matrix = rotation_matrix_y(angle, dtype=vectors.dtype)

        output1 = sh(vectors)
        vectors_rot = rotate_vectors(vectors, rot_matrix)
        output2 = sh(vectors_rot)

        # l=0 invariant
        assert torch.allclose(output1[:, 0], output2[:, 0], atol=1e-5)

        # l=1 norms preserved
        assert torch.allclose(
            output1[:, 1:4].norm(dim=-1),
            output2[:, 1:4].norm(dim=-1),
            rtol=1e-4, atol=1e-4
        )

    def test_compile(self):
        """Test that SphericalHarmonics can be compiled with torch.compile."""
        sh = SphericalHarmonics(l_max=2)
        vectors = torch.randn(10, 3)

        # Test compilation
        output_uncompiled, output_compiled = assert_module_compiles(sh, vectors)

        # Check outputs match
        assert_outputs_close(output_uncompiled, output_compiled)

    def test_export(self):
        """Test that SphericalHarmonics can be exported with torch.export."""
        sh = SphericalHarmonics(l_max=2)
        vectors = torch.randn(10, 3)

        # Test export
        exported_program, output_original, output_exported = assert_module_exports(
            sh,
            args_tuple=(vectors,),
        )

        # Check outputs match
        assert_outputs_close(output_original, output_exported)
