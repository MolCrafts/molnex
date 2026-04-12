"""Benchmarks for Allegro encoder."""

import pytest
import torch
import torch._dynamo

from molzoo.allegro import Allegro


@pytest.fixture
def module():
    """Create an Allegro encoder."""
    return Allegro(
        num_elements=5,
        num_scalar_features=16,
        num_tensor_features=8,
        r_max=5.0,
        num_bessel=8,
        l_max=1,
        num_layers=2,
    )


class BMAllegro:
    """Benchmarks for Allegro."""

    def test_forward(self, benchmark, module, graph_batch_td):
        with torch.no_grad():
            benchmark(module, graph_batch_td)

    def test_backward(self, benchmark, module, graph_batch_td):
        def _forward_backward():
            td = module(graph_batch_td)
            td["edges", "edge_features"].sum().backward()

        benchmark(_forward_backward)

    def test_forward_compiled(self, benchmark, module, graph_batch_td):
        compiled = torch.compile(module, backend="inductor")
        # warmup
        with torch.no_grad():
            compiled(graph_batch_td)
        with torch.no_grad():
            benchmark(compiled, graph_batch_td)

    def test_graph_breaks(self, module, graph_batch_td):
        explanation = torch._dynamo.explain(module)(graph_batch_td)
        print(f"Graph break count: {explanation.graph_break_count}")
        print(f"Break reasons: {explanation.break_reasons}")
