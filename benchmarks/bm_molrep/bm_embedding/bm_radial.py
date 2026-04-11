"""Benchmarks for BesselRBF radial embedding."""

import pytest
import torch
import torch._dynamo

from molrep.embedding.radial import BesselRBF


@pytest.fixture
def module():
    """Create a BesselRBF module."""
    return BesselRBF(r_cut=5.0, num_radial=8)


class BMBesselRBF:
    """Benchmarks for BesselRBF."""

    def test_forward(self, benchmark, module, graph_data):
        distances = graph_data["bond_dist"]
        benchmark(module, distances)

    def test_backward(self, benchmark, module, graph_data):
        distances = graph_data["bond_dist"].clone().requires_grad_(True)

        def _forward_backward():
            out = module(distances)
            out.sum().backward()
            if distances.grad is not None:
                distances.grad.zero_()

        benchmark(_forward_backward)

    def test_forward_compiled(self, benchmark, module, graph_data):
        distances = graph_data["bond_dist"]
        compiled = torch.compile(module, backend="inductor")
        # warmup
        compiled(distances)
        benchmark(compiled, distances)

    def test_graph_breaks(self, module, graph_data):
        distances = graph_data["bond_dist"]
        explanation = torch._dynamo.explain(module)(distances)
        assert explanation.graph_break_count == 0
