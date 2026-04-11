"""Benchmarks for molix.nn.mlp module."""

import pytest
import torch
import torch._dynamo

from molix.nn.mlp import KeyedMLP


class BMKeyedMLP:
    """Benchmarks for KeyedMLP."""

    @pytest.fixture
    def module(self):
        return KeyedMLP(
            input_key="features",
            output_key="output",
            in_dim=64,
            hidden_dims=[128, 128],
            out_dim=32,
        )

    def test_forward(self, benchmark, module, graph_data):
        n_nodes = graph_data["n_nodes"]
        data = {"features": torch.randn(n_nodes, 64)}

        benchmark(module, data)

    def test_backward(self, benchmark, module, graph_data):
        n_nodes = graph_data["n_nodes"]

        def forward_backward():
            data = {"features": torch.randn(n_nodes, 64)}
            result = module(data)
            result["output"].sum().backward()

        benchmark(forward_backward)

    def test_forward_compiled(self, benchmark, module, graph_data):
        n_nodes = graph_data["n_nodes"]
        data = {"features": torch.randn(n_nodes, 64)}

        compiled = torch.compile(module, backend="inductor")
        # warmup
        compiled(data)

        benchmark(compiled, {"features": torch.randn(n_nodes, 64)})

    def test_graph_breaks(self, module, graph_data):
        n_nodes = graph_data["n_nodes"]
        data = {"features": torch.randn(n_nodes, 64)}

        explanation = torch._dynamo.explain(module)(data)
        print(f"Graph break count: {explanation.graph_break_count}")
        print(f"Break reasons: {explanation.break_reasons}")
        assert explanation.graph_break_count == 0, (
            f"Expected 0 graph breaks, got {explanation.graph_break_count}: "
            f"{explanation.break_reasons}"
        )
