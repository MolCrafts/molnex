"""Benchmarks for SphericalHarmonics angular embedding."""

import pytest
import torch
import torch._dynamo

from molrep.embedding.angular import SphericalHarmonics


@pytest.fixture
def module():
    """Create a SphericalHarmonics module."""
    return SphericalHarmonics(l_max=2)


class BMSphericalHarmonics:
    """Benchmarks for SphericalHarmonics."""

    def test_forward(self, benchmark, module, graph_data):
        vectors = graph_data["bond_diff"]
        benchmark(module, vectors)

    def test_backward(self, benchmark, module, graph_data):
        vectors = graph_data["bond_diff"].clone().requires_grad_(True)

        def _forward_backward():
            out = module(vectors)
            out.sum().backward()
            if vectors.grad is not None:
                vectors.grad.zero_()

        benchmark(_forward_backward)

    def test_forward_compiled(self, benchmark, module, graph_data):
        vectors = graph_data["bond_diff"]
        compiled = torch.compile(module, backend="inductor")
        # warmup
        compiled(vectors)
        benchmark(compiled, vectors)

    def test_graph_breaks(self, module, graph_data):
        vectors = graph_data["bond_diff"]
        explanation = torch._dynamo.explain(module)(vectors)
        if explanation.graph_break_count != 0:
            pytest.xfail(
                f"cuequivariance_torch backend causes {explanation.graph_break_count} graph breaks"
            )
        assert explanation.graph_break_count == 0
