"""Benchmarks for molrep.interaction.tensor_product.ConvTP."""

import pytest
import torch
import torch._dynamo

from molrep.interaction.tensor_product import ConvTP, irreps_from_l_max, sh_irreps_from_l_max


class BMConvTP:
    """Benchmarks for ConvTP with l_max=2, hidden_dim=16."""

    @pytest.fixture
    def module(self):
        return ConvTP(
            in_irreps=irreps_from_l_max(2, 16),
            out_irreps=irreps_from_l_max(2, 16),
            sh_irreps=sh_irreps_from_l_max(2),
        )

    @pytest.fixture
    def inputs(self, module, graph_data):
        n_nodes = graph_data["n_nodes"]
        n_edges = graph_data["n_edges"]
        # dim = sum((2*l+1)*16 for l in [0,1,2]) = 16 + 48 + 80 = 144
        dim = 144
        # edge_angular dim = (l_max+1)^2 = 9
        sh_dim = 9
        node_features = torch.randn(n_nodes, dim)
        edge_angular = torch.randn(n_edges, sh_dim)
        edge_index = graph_data["edge_index"]
        tp_weights = torch.randn(n_edges, module.weight_numel)
        return node_features, edge_angular, edge_index, tp_weights

    def test_forward(self, benchmark, module, inputs):
        node_features, edge_angular, edge_index, tp_weights = inputs

        benchmark(module, node_features, edge_angular, edge_index, tp_weights)

    def test_backward(self, benchmark, module, inputs):
        node_features, edge_angular, edge_index, tp_weights = inputs
        node_features = node_features.clone().requires_grad_(True)
        tp_weights = tp_weights.clone().requires_grad_(True)

        def forward_backward():
            out = module(node_features, edge_angular, edge_index, tp_weights)
            out.sum().backward()

        benchmark(forward_backward)

    def test_forward_compiled(self, benchmark, module, inputs):
        node_features, edge_angular, edge_index, tp_weights = inputs

        compiled = torch.compile(module, backend="inductor")
        # warmup
        compiled(node_features, edge_angular, edge_index, tp_weights)

        benchmark(compiled, node_features, edge_angular, edge_index, tp_weights)

    def test_graph_breaks(self, module, inputs):
        node_features, edge_angular, edge_index, tp_weights = inputs

        explanation = torch._dynamo.explain(module)(
            node_features, edge_angular, edge_index, tp_weights
        )
        print(f"Graph break count: {explanation.graph_break_count}")
        print(f"Break reasons: {explanation.break_reasons}")
        # cuEquivariance-based module: graph breaks are expected
