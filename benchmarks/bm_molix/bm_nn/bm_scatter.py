"""Benchmarks for molix.nn.scatter module."""

import pytest
import torch
import torch._dynamo

from molix.nn.scatter import BatchAggregation, ScatterSum


class BMScatterSum:
    """Benchmarks for ScatterSum(dim=0)."""

    @pytest.fixture
    def module(self):
        return ScatterSum(dim=0)

    def test_forward(self, benchmark, module, graph_data):
        n_nodes = graph_data["n_nodes"]
        n_edges = graph_data["n_edges"]
        src = torch.randn(n_edges, 16)
        index = graph_data["edge_index"][:, 1]

        benchmark(module, src, index, dim_size=n_nodes)

    def test_backward(self, benchmark, module, graph_data):
        n_nodes = graph_data["n_nodes"]
        n_edges = graph_data["n_edges"]
        src = torch.randn(n_edges, 16, requires_grad=True)
        index = graph_data["edge_index"][:, 1]

        def forward_backward():
            out = module(src, index, dim_size=n_nodes)
            out.sum().backward()

        benchmark(forward_backward)

    def test_forward_compiled(self, benchmark, module, graph_data):
        n_nodes = graph_data["n_nodes"]
        n_edges = graph_data["n_edges"]
        src = torch.randn(n_edges, 16)
        index = graph_data["edge_index"][:, 1]

        compiled = torch.compile(module, backend="inductor")
        # warmup
        compiled(src, index, dim_size=n_nodes)

        benchmark(compiled, src, index, dim_size=n_nodes)

    def test_graph_breaks(self, module, graph_data):
        n_nodes = graph_data["n_nodes"]
        n_edges = graph_data["n_edges"]
        src = torch.randn(n_edges, 16)
        index = graph_data["edge_index"][:, 1]

        explanation = torch._dynamo.explain(module)(src, index, dim_size=n_nodes)
        print(f"Graph break count: {explanation.graph_break_count}")
        print(f"Break reasons: {explanation.break_reasons}")
        assert explanation.graph_break_count == 0, (
            f"Expected 0 graph breaks, got {explanation.graph_break_count}: "
            f"{explanation.break_reasons}"
        )


class BMBatchAggregation:
    """Benchmarks for BatchAggregation()."""

    @pytest.fixture
    def module(self):
        return BatchAggregation()

    def test_forward(self, benchmark, module, graph_data):
        n_nodes = graph_data["n_nodes"]
        n_graphs = graph_data["n_graphs"]
        src = torch.randn(n_nodes, 16)
        batch = graph_data["batch"]

        benchmark(module, src, batch, dim_size=n_graphs)

    def test_backward(self, benchmark, module, graph_data):
        n_nodes = graph_data["n_nodes"]
        n_graphs = graph_data["n_graphs"]
        src = torch.randn(n_nodes, 16, requires_grad=True)
        batch = graph_data["batch"]

        def forward_backward():
            out = module(src, batch, dim_size=n_graphs)
            out.sum().backward()

        benchmark(forward_backward)

    def test_forward_compiled(self, benchmark, module, graph_data):
        n_nodes = graph_data["n_nodes"]
        n_graphs = graph_data["n_graphs"]
        src = torch.randn(n_nodes, 16)
        batch = graph_data["batch"]

        compiled = torch.compile(module, backend="inductor")
        # warmup
        compiled(src, batch, dim_size=n_graphs)

        benchmark(compiled, src, batch, dim_size=n_graphs)

    def test_graph_breaks(self, module, graph_data):
        n_nodes = graph_data["n_nodes"]
        n_graphs = graph_data["n_graphs"]
        src = torch.randn(n_nodes, 16)
        batch = graph_data["batch"]

        explanation = torch._dynamo.explain(module)(src, batch, dim_size=n_graphs)
        print(f"Graph break count: {explanation.graph_break_count}")
        print(f"Break reasons: {explanation.break_reasons}")
        assert explanation.graph_break_count == 0, (
            f"Expected 0 graph breaks, got {explanation.graph_break_count}: "
            f"{explanation.break_reasons}"
        )
