"""Benchmarks for molrep.interaction.radial_mlp.RadialWeightMLP."""

import pytest
import torch
import torch._dynamo

from molrep.interaction.radial_mlp import RadialWeightMLP


class BMRadialWeightMLP:
    """Benchmarks for RadialWeightMLP with in_dim=8, hidden_dim=64, out_dim=128."""

    @pytest.fixture
    def module(self):
        return RadialWeightMLP(
            in_dim=8,
            hidden_dim=64,
            out_dim=128,
        )

    def test_forward(self, benchmark, module, graph_data):
        n_edges = graph_data["n_edges"]
        edge_feats = torch.randn(n_edges, 8)

        benchmark(module, edge_feats)

    def test_backward(self, benchmark, module, graph_data):
        n_edges = graph_data["n_edges"]
        edge_feats = torch.randn(n_edges, 8, requires_grad=True)

        def forward_backward():
            out = module(edge_feats)
            out.sum().backward()

        benchmark(forward_backward)

    def test_forward_compiled(self, benchmark, module, graph_data):
        n_edges = graph_data["n_edges"]
        edge_feats = torch.randn(n_edges, 8)

        compiled = torch.compile(module, backend="inductor")
        # warmup
        compiled(edge_feats)

        benchmark(compiled, edge_feats)

    def test_graph_breaks(self, module, graph_data):
        n_edges = graph_data["n_edges"]
        edge_feats = torch.randn(n_edges, 8)

        explanation = torch._dynamo.explain(module)(edge_feats)
        print(f"Graph break count: {explanation.graph_break_count}")
        print(f"Break reasons: {explanation.break_reasons}")
        # Pure MLP: no graph breaks expected
        assert explanation.graph_break_count == 0, (
            f"Expected 0 graph breaks, got {explanation.graph_break_count}: "
            f"{explanation.break_reasons}"
        )
