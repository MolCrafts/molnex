"""Tests for molrep.interaction.aggregation module."""

import pytest
import torch
import math
from molrep.interaction.aggregation import MessageAggregation, MessageAggregationSpec
from molrep.utils.equivariance import (
    rotation_matrix_z,
    rotate_irreps_features_simple,
    check_equivariance,
)
from tests.utils import assert_module_compiles, assert_module_exports, assert_outputs_close


class TestMessageAggregationSpec:
    """Test MessageAggregationSpec configuration."""
    
    def test_valid_config(self):
        """Test creation with valid parameters."""
        spec = MessageAggregationSpec(
            irreps="64x0e + 32x1o",
            apply_cutoff=True,
        )
        assert spec.irreps == "64x0e + 32x1o"
        assert spec.apply_cutoff == True


class TestMessageAggregation:
    """Test MessageAggregation layer."""
    
    def test_initialization(self):
        """Test MessageAggregation initialization."""
        agg = MessageAggregation(
            irreps="64x0e",
            apply_cutoff=True,
        )
        assert agg.config.irreps == "64x0e"
        assert agg.config.apply_cutoff == True
    
    def test_forward_shape(self):
        """Test output shape."""
        # 64 scalars -> 64 scalars (EquivariantLinear internally)
        agg = MessageAggregation(
            irreps="64x0e",
            apply_cutoff=False,
        )
        
        n_nodes = 10
        n_edges = 30
        messages = torch.randn(n_edges, 64)
        edge_index = torch.randint(0, n_nodes, (n_edges, 2))
        
        output = agg(messages, edge_index, n_nodes=n_nodes)
        
        assert output.shape == (n_nodes, 64)
    
    def test_with_cutoff(self):
        """Test aggregation with cutoff weighting."""
        agg = MessageAggregation(
            irreps="32x0e",
            apply_cutoff=True,
        )
        
        n_nodes = 5
        n_edges = 10
        messages = torch.randn(n_edges, 32)
        edge_index = torch.randint(0, n_nodes, (n_edges, 2))
        edge_cutoff = torch.rand(n_edges)
        
        output = agg(messages, edge_index, edge_cutoff=edge_cutoff, n_nodes=n_nodes)
        
        assert output.shape == (n_nodes, 32)
    
    def test_different_irreps(self):
        """Test with different irreps configurations."""
        for irreps in ["32x0e", "64x0e", "128x0e"]:
            agg = MessageAggregation(
                irreps=irreps,
                apply_cutoff=False,
            )
            assert agg.config.irreps == irreps

            # Basic forward check
            messages = torch.randn(5, 32 if irreps == "32x0e" else 64 if irreps == "64x0e" else 128)
            edge_index = torch.randint(0, 3, (5, 2))
            output = agg(messages, edge_index, n_nodes=3)
            assert output.shape[1] == (32 if irreps == "32x0e" else 64 if irreps == "64x0e" else 128)

    def test_equivariance_scalars(self):
        """Test that scalar features maintain rotation invariance.

        For scalar-only features, aggregation should be rotation invariant.
        """
        agg = MessageAggregation(
            irreps="32x0e",
            apply_cutoff=False,
        )

        n_nodes = 5
        n_edges = 10
        messages = torch.randn(n_edges, 32)
        edge_index = torch.randint(0, n_nodes, (n_edges, 2))

        # Forward pass
        output1 = agg(messages, edge_index, n_nodes=n_nodes)

        # Scalars don't change under rotation, so output should be same
        output2 = agg(messages, edge_index, n_nodes=n_nodes)

        # Should be identical
        assert torch.allclose(output1, output2, rtol=1e-5, atol=1e-5)

    def test_equivariance_vectors(self):
        """Test equivariance property for vector features.

        MessageAggregation should preserve SO(3) equivariance:
            agg(R·messages) = R·agg(messages)
        """
        # Use vector irreps (4 vectors)
        irreps = "4x1o"
        agg = MessageAggregation(
            irreps=irreps,
            apply_cutoff=False,
        )

        n_nodes = 5
        n_edges = 10
        messages = torch.randn(n_edges, 12)  # 4 vectors * 3 components
        edge_index = torch.randint(0, n_nodes, (n_edges, 2))

        # Forward pass on original messages
        output1 = agg(messages, edge_index, n_nodes=n_nodes)

        # Rotate messages (90 degrees around z-axis)
        angle = math.pi / 2
        rot_matrix = rotation_matrix_z(angle, dtype=messages.dtype)

        messages_rotated = rotate_irreps_features_simple(messages, rot_matrix, irreps)

        # Forward pass on rotated messages
        output2 = agg(messages_rotated, edge_index, n_nodes=n_nodes)

        # Rotate output1 and compare with output2
        output1_rotated = rotate_irreps_features_simple(output1, rot_matrix, irreps)

        # Should be equivariant
        assert check_equivariance(output1_rotated, output2, rtol=1e-3, atol=1e-3)

    def test_equivariance_mixed_irreps(self):
        """Test equivariance with mixed scalar and vector features."""
        # Mix scalars and vectors
        irreps = "8x0e + 4x1o"
        agg = MessageAggregation(
            irreps=irreps,
            apply_cutoff=False,
        )

        n_nodes = 5
        n_edges = 10
        # 8 scalars + 4*3 vectors = 8 + 12 = 20 features
        messages = torch.randn(n_edges, 20)
        edge_index = torch.randint(0, n_nodes, (n_edges, 2))

        # Forward pass
        output1 = agg(messages, edge_index, n_nodes=n_nodes)

        # Rotate (90 degrees around z)
        angle = math.pi / 2
        rot_matrix = rotation_matrix_z(angle, dtype=messages.dtype)

        messages_rotated = rotate_irreps_features_simple(messages, rot_matrix, irreps)

        # Forward on rotated
        output2 = agg(messages_rotated, edge_index, n_nodes=n_nodes)

        # Rotate output1
        output1_rotated = rotate_irreps_features_simple(output1, rot_matrix, irreps)

        # Check equivariance
        assert check_equivariance(output1_rotated, output2, rtol=1e-3, atol=1e-3)

    def test_compile(self):
        """Test that MessageAggregation can be compiled with torch.compile."""
        agg = MessageAggregation(
            irreps="64x0e",
            apply_cutoff=False,
        )

        n_nodes = 10
        n_edges = 30
        messages = torch.randn(n_edges, 64)
        edge_index = torch.randint(0, n_nodes, (n_edges, 2))

        # Test compilation
        output_uncompiled, output_compiled = assert_module_compiles(
            agg,
            messages, edge_index, n_nodes=n_nodes
        )

        # Check outputs match
        assert_outputs_close(output_uncompiled, output_compiled)

    def test_export(self):
        """Test that MessageAggregation can be exported with torch.export."""
        agg = MessageAggregation(
            irreps="64x0e",
            apply_cutoff=False,
        )

        n_nodes = 10
        n_edges = 30
        messages = torch.randn(n_edges, 64)
        edge_index = torch.randint(0, n_nodes, (n_edges, 2))

        # Test export
        exported_program, output_original, output_exported = assert_module_exports(
            agg,
            args_tuple=(messages, edge_index),
            kwargs_dict={"n_nodes": n_nodes}
        )

        # Check outputs match
        assert_outputs_close(output_original, output_exported)
