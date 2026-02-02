"""Tests for molrep.interaction.aggregation module."""

import pytest
import torch
from molrep.interaction.aggregation import MessageAggregation, MessageAggregationSpec


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
