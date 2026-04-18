"""Tests for molrep.interaction.tensor_product module."""

import pytest
import torch
import math
from molrep.interaction.tensor_product import ConvTP, ConvTPSpec
from molrep.utils.equivariance import (
    rotation_matrix_z,
    rotate_irreps_features_simple,
    rotate_vectors,
    check_equivariance,
)
from tests.utils import assert_module_compiles, assert_module_exports, assert_outputs_close


class TestConvTPSpec:
    """Test ConvTPSpec configuration."""
    
    def test_valid_config(self):
        """Test creation with valid parameters."""
        spec = ConvTPSpec(
            in_irreps="64x0e",
            out_irreps="64x0e",
            sh_irreps="1x0e + 1x1o",
        )
        assert spec.in_irreps == "64x0e"
        assert spec.out_irreps == "64x0e"
        assert spec.sh_irreps == "1x0e + 1x1o"


class TestConvTP:
    """Test ConvTP tensor product layer."""
    
    def test_initialization(self):
        """Test ConvTP initialization."""
        tp = ConvTP(
            in_irreps="64x0e",
            out_irreps="64x0e",
            sh_irreps="1x0e + 1x1o",
        )
        assert tp.config.in_irreps == "64x0e"
        assert tp.config.out_irreps == "64x0e"
    
    def test_forward_shape(self):
        """Test output shape."""
        tp = ConvTP(
            in_irreps="16x0e",
            out_irreps="16x0e",
            sh_irreps="1x0e + 1x1o",
        )
        
        n_nodes = 10
        n_edges = 30
        node_features = torch.randn(n_nodes, 16)
        edge_angular = torch.randn(n_edges, 4)  # 1 + 3 = 4 dims
        edge_index = torch.randint(0, n_nodes, (n_edges, 2))
        
        # ConvTP weights usually depend on radial embedding
        # cuet.Linear handles the weights if passed correctly.
        # Wait, ConvTP expects tp_weights.
        # Let's check how many weights are needed.
        # In ConvTP, self.cue_tp is initialized with weight_dim.
        weight_dim = tp.cue_tp.weight_numel
        tp_weights = torch.randn(n_edges, weight_dim)
        
        output = tp(node_features, edge_angular, edge_index, tp_weights)
        
        # Messages should be per node after aggregation, not per edge
        assert output.shape[0] == n_nodes
        assert output.shape[1] == 16
    
    def test_different_irreps(self):
        """Test with different input/output irreps."""
        tp = ConvTP(
            in_irreps="32x0e",
            out_irreps="64x0e",
            sh_irreps="1x0e + 1x1o + 1x2e",
        )
        assert tp.config.in_irreps == "32x0e"
        assert tp.config.out_irreps == "64x0e"

    def test_equivariance_scalars(self):
        """Test equivariance with scalar features only.

        For scalar inputs, tensor product with spherical harmonics should
        maintain equivariance properties.
        """
        tp = ConvTP(
            in_irreps="16x0e",
            out_irreps="16x0e",
            sh_irreps="1x0e + 1x1o",
        )

        n_nodes = 5
        n_edges = 10
        node_features = torch.randn(n_nodes, 16)
        edge_angular = torch.randn(n_edges, 4)  # 1 + 3 dims
        edge_index = torch.randint(0, n_nodes, (n_edges, 2))

        weight_dim = tp.weight_numel
        tp_weights = torch.randn(n_edges, weight_dim)

        # Forward pass
        output1 = tp(node_features, edge_angular, edge_index, tp_weights)

        # Rotate edge angular features (spherical harmonics)
        angle = math.pi / 2
        rot_matrix = rotation_matrix_z(angle, dtype=edge_angular.dtype)

        # For spherical harmonics, we need to rotate the vector components (l=1)
        # l=0 is invariant, l=1 needs rotation
        edge_angular_rotated = edge_angular.clone()
        edge_angular_rotated[:, 1:4] = rotate_vectors(
            edge_angular[:, 1:4],
            rot_matrix
        )

        # Forward on rotated
        output2 = tp(node_features, edge_angular_rotated, edge_index, tp_weights)

        # For scalar outputs, they should be approximately equal (rotation invariant)
        assert torch.allclose(output1, output2, rtol=1e-3, atol=1e-3)

    def test_equivariance_vectors(self):
        """Test equivariance with vector features.

        ConvTP should satisfy: TP(R·h, R·Y) = R·TP(h, Y)
        where h are node features, Y are spherical harmonics, R is rotation.
        """
        # Vector input and output
        in_irreps = "4x1o"
        out_irreps = "4x1o"
        sh_irreps = "1x0e + 1x1o"

        tp = ConvTP(
            in_irreps=in_irreps,
            out_irreps=out_irreps,
            sh_irreps=sh_irreps,
        )

        n_nodes = 5
        n_edges = 10
        node_features = torch.randn(n_nodes, 12)  # 4 vectors * 3
        edge_angular = torch.randn(n_edges, 4)  # 1 + 3
        edge_index = torch.randint(0, n_nodes, (n_edges, 2))

        weight_dim = tp.weight_numel
        tp_weights = torch.randn(n_edges, weight_dim)

        # Forward pass
        output1 = tp(node_features, edge_angular, edge_index, tp_weights)

        # Rotate everything
        angle = math.pi / 2
        rot_matrix = rotation_matrix_z(angle, dtype=node_features.dtype)

        # Rotate node features
        node_features_rot = rotate_irreps_features_simple(
            node_features, rot_matrix, in_irreps
        )

        # Rotate edge angular (spherical harmonics)
        edge_angular_rot = edge_angular.clone()
        edge_angular_rot[:, 1:4] = rotate_vectors(
            edge_angular[:, 1:4],
            rot_matrix
        )

        # Forward on rotated inputs
        output2 = tp(node_features_rot, edge_angular_rot, edge_index, tp_weights)

        # Rotate output1
        output1_rot = rotate_irreps_features_simple(output1, rot_matrix, out_irreps)

        # Check equivariance
        assert check_equivariance(output1_rot, output2, rtol=1e-3, atol=1e-3)

    def test_differentiable(self):
        """Test that gradients flow through ConvTP."""
        tp = ConvTP(
            in_irreps="8x0e",
            out_irreps="8x0e",
            sh_irreps="1x0e + 1x1o",
        )

        n_nodes = 5
        n_edges = 10
        node_features = torch.randn(n_nodes, 8, requires_grad=True)
        edge_angular = torch.randn(n_edges, 4, requires_grad=True)
        edge_index = torch.randint(0, n_nodes, (n_edges, 2))

        weight_dim = tp.weight_numel
        tp_weights = torch.randn(n_edges, weight_dim, requires_grad=True)

        output = tp(node_features, edge_angular, edge_index, tp_weights)
        loss = output.sum()
        loss.backward()

        assert node_features.grad is not None
        assert edge_angular.grad is not None
        assert tp_weights.grad is not None
        assert not torch.isnan(node_features.grad).any()
        assert not torch.isnan(edge_angular.grad).any()
        assert not torch.isnan(tp_weights.grad).any()

    def test_compile(self):
        """Test that ConvTP can be compiled with torch.compile."""
        tp = ConvTP(
            in_irreps="16x0e",
            out_irreps="16x0e",
            sh_irreps="1x0e + 1x1o",
        )

        n_nodes = 10
        n_edges = 30
        node_features = torch.randn(n_nodes, 16)
        edge_angular = torch.randn(n_edges, 4)
        edge_index = torch.randint(0, n_nodes, (n_edges, 2))

        weight_dim = tp.weight_numel
        tp_weights = torch.randn(n_edges, weight_dim)

        # Test compilation
        output_uncompiled, output_compiled = assert_module_compiles(
            tp,
            node_features, edge_angular, edge_index, tp_weights
        )

        # Check outputs match
        assert_outputs_close(output_uncompiled, output_compiled)

    def test_export(self):
        """Test that ConvTP can be exported with torch.export."""
        tp = ConvTP(
            in_irreps="16x0e",
            out_irreps="16x0e",
            sh_irreps="1x0e + 1x1o",
        )

        n_nodes = 10
        n_edges = 30
        node_features = torch.randn(n_nodes, 16)
        edge_angular = torch.randn(n_edges, 4)
        edge_index = torch.randint(0, n_nodes, (n_edges, 2))

        weight_dim = tp.weight_numel
        tp_weights = torch.randn(n_edges, weight_dim)

        # Test export
        exported_program, output_original, output_exported = assert_module_exports(
            tp,
            args_tuple=(node_features, edge_angular, edge_index, tp_weights),
        )

        # Check outputs match
        assert_outputs_close(output_original, output_exported)
