"""Tests for molrep.interaction.mace.conv module."""

import math

import pytest
import torch

from molrep.interaction.mace.conv import ConvTP, ConvTPSpec
from molrep.utils.equivariance import (
    check_equivariance,
    rotate_irreps_features_simple,
    rotate_vectors,
    rotation_matrix_z,
)


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
        edge_angular_rotated[:, 1:4] = rotate_vectors(edge_angular[:, 1:4], rot_matrix)

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
        node_features_rot = rotate_irreps_features_simple(node_features, rot_matrix, in_irreps)

        # Rotate edge angular (spherical harmonics)
        edge_angular_rot = edge_angular.clone()
        edge_angular_rot[:, 1:4] = rotate_vectors(edge_angular[:, 1:4], rot_matrix)

        # Forward on rotated inputs
        output2 = tp(node_features_rot, edge_angular_rot, edge_index, tp_weights)

        # Rotate output1
        output1_rot = rotate_irreps_features_simple(output1, rot_matrix, out_irreps)

        # Check equivariance
        assert check_equivariance(output1_rot, output2, rtol=1e-3, atol=1e-3)


def _pack_target_blocks(
    edge_index: torch.Tensor,
    edge_angular: torch.Tensor,
    weights: torch.Tensor,
    *,
    num_nodes: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Small oracle: ``block_size`` is a maximum and intervals are balanced."""
    num_blocks = math.ceil(num_nodes / block_size)
    block_span = math.ceil(num_nodes / num_blocks)
    counts = torch.bincount(edge_index[:, 1] // block_span, minlength=num_blocks)
    edges_per_block = int(counts.max()) + 2
    packed_index, packed_angular, packed_weights = [], [], []
    for block in range(num_blocks):
        selected = torch.where(edge_index[:, 1] // block_span == block)[0]
        count = int(selected.numel())
        anchor = block * block_span
        pad = edges_per_block - count
        packed_index.append(
            torch.cat([edge_index[selected], torch.full((pad, 2), anchor, dtype=torch.long)])
        )
        packed_angular.append(
            torch.cat([edge_angular[selected], edge_angular.new_zeros(pad, edge_angular.shape[1])])
        )
        packed_weights.append(
            torch.cat([weights[selected], weights.new_zeros(pad, weights.shape[1])])
        )
    return (
        torch.cat(packed_index),
        torch.cat(packed_angular),
        torch.cat(packed_weights),
    )


class TestTargetBlockedConvTP:
    """Receiver blocking changes cuEq's working set, not its mathematics."""

    @staticmethod
    def _case():
        torch.manual_seed(0)
        tp = ConvTP(
            in_irreps="8x0e + 8x1o",
            out_irreps="8x0e + 8x1o",
            sh_irreps="1x0e + 1x1o",
        ).double()
        node = torch.randn(5, 32, dtype=torch.float64, requires_grad=True)
        angular = torch.randn(17, 4, dtype=torch.float64, requires_grad=True)
        edge_index = torch.randint(0, 5, (17, 2))
        weights = torch.randn(17, tp.weight_numel, dtype=torch.float64, requires_grad=True)
        return tp, node, angular, edge_index, weights

    def test_matches_full_official_convolution(self) -> None:
        tp, node, angular, edge_index, weights = self._case()
        expected = tp(node, angular, edge_index, weights)
        packed = _pack_target_blocks(
            edge_index, angular, weights, num_nodes=node.shape[0], block_size=4
        )
        tp.configure_target_blocks(4)
        actual = tp(node, packed[1], packed[0], packed[2])
        assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)

    def test_backward_reaches_all_three_inputs(self) -> None:
        tp, node, angular, edge_index, weights = self._case()
        packed = _pack_target_blocks(
            edge_index, angular, weights, num_nodes=node.shape[0], block_size=4
        )
        tp.configure_target_blocks(4)
        tp(node, packed[1], packed[0], packed[2]).square().sum().backward()
        assert node.grad is not None and torch.isfinite(node.grad).all()
        assert angular.grad is not None and torch.isfinite(angular.grad).all()
        assert weights.grad is not None and torch.isfinite(weights.grad).all()

    @pytest.mark.slow
    def test_compiles_fullgraph_with_static_block_slices(self) -> None:
        tp, node, angular, edge_index, weights = self._case()
        packed = _pack_target_blocks(
            edge_index, angular, weights, num_nodes=node.shape[0], block_size=4
        )
        tp.configure_target_blocks(4)

        def fn(n: torch.Tensor, a: torch.Tensor, ei: torch.Tensor, w: torch.Tensor):
            return tp(n, a, ei, w)

        compiled = torch.compile(fn, backend="aot_eager", fullgraph=True, dynamic=True)
        expected = fn(node, packed[1], packed[0], packed[2])
        assert torch.allclose(
            compiled(node, packed[1], packed[0], packed[2]), expected, atol=1e-12, rtol=1e-12
        )
