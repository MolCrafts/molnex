"""Benchmarks for MACE encoder."""

import pytest
import torch
import torch._dynamo

from molrep.embedding.node import DiscreteEmbeddingSpec
from molzoo.mace import MACE


@pytest.fixture
def module():
    """Create a MACE encoder."""
    return MACE(
        node_attr_specs=[
            DiscreteEmbeddingSpec(input_key="Z", num_classes=5, emb_dim=16)
        ],
        num_elements=5,
        num_features=16,
        r_max=5.0,
        num_bessel=8,
        l_max=1,
        num_interactions=2,
        correlation=2,
    )


class BMMACE:
    """Benchmarks for MACE."""

    def test_forward(self, benchmark, module, graph_batch_td):
        with torch.no_grad():
            benchmark(module, graph_batch_td)

    def test_backward(self, benchmark, module, graph_batch_td):
        def _forward_backward():
            td = module(graph_batch_td)
            td["atoms", "node_features"].sum().backward()

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
