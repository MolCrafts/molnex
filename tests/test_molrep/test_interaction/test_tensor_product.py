"""Tests for molrep.interaction.tensor_product module."""

import pytest
import torch
from molrep.interaction.tensor_product import ConvTP, ConvTPSpec


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
