"""Benchmarks for molrep.interaction.aggregation.MessageAggregation."""

import pytest
import torch
import torch._dynamo

from molrep.interaction.aggregation import MessageAggregation


class BMMessageAggregation:
    """Benchmarks for MessageAggregation with scalar irreps "16x0e"."""

    @pytest.fixture
    def module(self):
        return MessageAggregation(irreps="16x0e")

    def test_forward(self, benchmark, module, graph_data):
        n_edges = graph_data["n_edges"]
        n_nodes = graph_data["n_nodes"]
        messages = torch.randn(n_edges, 16)
        edge_index = graph_data["edge_index"]
        edge_cutoff = torch.rand(n_edges)

        benchmark(module, messages, edge_index, edge_cutoff, n_nodes)

    def test_backward(self, benchmark, module, graph_data):
        n_edges = graph_data["n_edges"]
        n_nodes = graph_data["n_nodes"]
        messages = torch.randn(n_edges, 16, requires_grad=True)
        edge_index = graph_data["edge_index"]
        edge_cutoff = torch.rand(n_edges)

        def forward_backward():
            out = module(messages, edge_index, edge_cutoff, n_nodes)
            out.sum().backward()

        benchmark(forward_backward)

    def test_forward_compiled(self, benchmark, module, graph_data):
        n_edges = graph_data["n_edges"]
        n_nodes = graph_data["n_nodes"]
        messages = torch.randn(n_edges, 16)
        edge_index = graph_data["edge_index"]
        edge_cutoff = torch.rand(n_edges)

        compiled = torch.compile(module, backend="inductor")
        # warmup
        compiled(messages, edge_index, edge_cutoff, n_nodes)

        benchmark(compiled, messages, edge_index, edge_cutoff, n_nodes)

    def test_graph_breaks(self, module, graph_data):
        n_edges = graph_data["n_edges"]
        n_nodes = graph_data["n_nodes"]
        messages = torch.randn(n_edges, 16)
        edge_index = graph_data["edge_index"]
        edge_cutoff = torch.rand(n_edges)

        explanation = torch._dynamo.explain(module)(
            messages, edge_index, edge_cutoff, n_nodes
        )
        print(f"Graph break count: {explanation.graph_break_count}")
        print(f"Break reasons: {explanation.break_reasons}")
        # Uses EquivariantLinear (cuEquivariance): graph breaks are expected
