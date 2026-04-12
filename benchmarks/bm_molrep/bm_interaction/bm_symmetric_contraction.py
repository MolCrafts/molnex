"""Benchmarks for molrep.interaction.symmetric_contraction.SymmetricContraction."""

import pytest
import torch
import torch._dynamo

from molrep.interaction.symmetric_contraction import SymmetricContraction


class BMSymmetricContraction:
    """Benchmarks for SymmetricContraction with hidden_dim=16, num_species=5, max_body_order=2."""

    @pytest.fixture
    def module(self):
        return SymmetricContraction(
            hidden_dim=16,
            num_species=5,
            max_body_order=2,
        )

    def test_forward(self, benchmark, module, graph_data):
        n_nodes = graph_data["n_nodes"]
        node_features = torch.randn(n_nodes, 16)
        atom_types = graph_data["Z"]

        benchmark(module, node_features, atom_types)

    def test_backward(self, benchmark, module, graph_data):
        n_nodes = graph_data["n_nodes"]
        node_features = torch.randn(n_nodes, 16, requires_grad=True)
        atom_types = graph_data["Z"]

        def forward_backward():
            out = module(node_features, atom_types)
            out.sum().backward()

        benchmark(forward_backward)

    def test_forward_compiled(self, benchmark, module, graph_data):
        n_nodes = graph_data["n_nodes"]
        node_features = torch.randn(n_nodes, 16)
        atom_types = graph_data["Z"]

        compiled = torch.compile(module, backend="inductor")
        # warmup
        compiled(node_features, atom_types)

        benchmark(compiled, node_features, atom_types)

    def test_graph_breaks(self, module, graph_data):
        n_nodes = graph_data["n_nodes"]
        node_features = torch.randn(n_nodes, 16)
        atom_types = graph_data["Z"]

        explanation = torch._dynamo.explain(module)(node_features, atom_types)
        print(f"Graph break count: {explanation.graph_break_count}")
        print(f"Break reasons: {explanation.break_reasons}")
        # cuEquivariance-based module: graph breaks are expected
