"""Benchmarks for molrep.interaction.element_update.ElementUpdate."""

import pytest
import torch
import torch._dynamo

from molrep.interaction.element_update import ElementUpdate


class BMElementUpdate:
    """Benchmarks for ElementUpdate with hidden_dim=16, num_species=5."""

    @pytest.fixture
    def module(self):
        return ElementUpdate(
            hidden_dim=16,
            num_species=5,
        )

    def test_forward(self, benchmark, module, graph_data):
        n_nodes = graph_data["n_nodes"]
        h_prev = torch.randn(n_nodes, 16)
        m_curr = torch.randn(n_nodes, 16)
        atom_types = graph_data["Z"]

        benchmark(module, h_prev, m_curr, atom_types)

    def test_backward(self, benchmark, module, graph_data):
        n_nodes = graph_data["n_nodes"]
        h_prev = torch.randn(n_nodes, 16, requires_grad=True)
        m_curr = torch.randn(n_nodes, 16, requires_grad=True)
        atom_types = graph_data["Z"]

        def forward_backward():
            out = module(h_prev, m_curr, atom_types)
            out.sum().backward()

        benchmark(forward_backward)

    def test_forward_compiled(self, benchmark, module, graph_data):
        n_nodes = graph_data["n_nodes"]
        h_prev = torch.randn(n_nodes, 16)
        m_curr = torch.randn(n_nodes, 16)
        atom_types = graph_data["Z"]

        compiled = torch.compile(module, backend="inductor")
        # warmup
        compiled(h_prev, m_curr, atom_types)

        benchmark(compiled, h_prev, m_curr, atom_types)

    def test_graph_breaks(self, module, graph_data):
        n_nodes = graph_data["n_nodes"]
        h_prev = torch.randn(n_nodes, 16)
        m_curr = torch.randn(n_nodes, 16)
        atom_types = graph_data["Z"]

        explanation = torch._dynamo.explain(module)(h_prev, m_curr, atom_types)
        print(f"Graph break count: {explanation.graph_break_count}")
        print(f"Break reasons: {explanation.break_reasons}")
        # cuEquivariance-based module: graph breaks are expected
