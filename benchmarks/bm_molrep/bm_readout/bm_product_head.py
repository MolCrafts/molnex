"""Benchmarks for ProductHead readout module."""

import pytest
import torch
import torch._dynamo

from molrep.readout.product_head import ProductHead


@pytest.fixture
def module():
    """Create a ProductHead module."""
    return ProductHead(
        hidden_dim=16,
        out_dim=1,
        num_radial=8,
        l_max=2,
        max_body_order=2,
        num_species=5,
    )


class BMProductHead:
    """Benchmarks for ProductHead."""

    def test_forward(self, benchmark, module, graph_data):
        node_features = torch.randn(graph_data["n_nodes"], 16)
        atom_types = graph_data["Z"]
        benchmark(module, node_features, atom_types)

    def test_backward(self, benchmark, module, graph_data):
        node_features = torch.randn(graph_data["n_nodes"], 16, requires_grad=True)
        atom_types = graph_data["Z"]

        def _forward_backward():
            out = module(node_features, atom_types)
            out.sum().backward()
            if node_features.grad is not None:
                node_features.grad.zero_()

        benchmark(_forward_backward)

    def test_forward_compiled(self, benchmark, module, graph_data):
        node_features = torch.randn(graph_data["n_nodes"], 16)
        atom_types = graph_data["Z"]
        compiled = torch.compile(module, backend="inductor")
        # warmup
        compiled(node_features, atom_types)
        benchmark(compiled, node_features, atom_types)

    def test_graph_breaks(self, module, graph_data):
        node_features = torch.randn(graph_data["n_nodes"], 16)
        atom_types = graph_data["Z"]
        explanation = torch._dynamo.explain(module)(node_features, atom_types)
        print(f"Graph break count: {explanation.graph_break_count}")
        print(f"Break reasons: {explanation.break_reasons}")
