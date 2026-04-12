"""Benchmarks for CosineCutoff and PolynomialCutoff modules."""

import pytest
import torch
import torch._dynamo

from molrep.embedding.cutoff import CosineCutoff, PolynomialCutoff


# --- CosineCutoff ---


@pytest.fixture
def cosine_module():
    """Create a CosineCutoff module."""
    return CosineCutoff(r_cut=5.0)


class BMCosineCutoff:
    """Benchmarks for CosineCutoff."""

    def test_forward(self, benchmark, cosine_module, graph_data):
        distances = graph_data["bond_dist"]
        benchmark(cosine_module, distances)

    def test_backward(self, benchmark, cosine_module, graph_data):
        distances = graph_data["bond_dist"].clone().requires_grad_(True)

        def _forward_backward():
            out = cosine_module(distances)
            out.sum().backward()
            if distances.grad is not None:
                distances.grad.zero_()

        benchmark(_forward_backward)

    def test_forward_compiled(self, benchmark, cosine_module, graph_data):
        distances = graph_data["bond_dist"]
        compiled = torch.compile(cosine_module, backend="inductor")
        # warmup
        compiled(distances)
        benchmark(compiled, distances)

    def test_graph_breaks(self, cosine_module, graph_data):
        distances = graph_data["bond_dist"]
        explanation = torch._dynamo.explain(cosine_module)(distances)
        assert explanation.graph_break_count == 0


# --- PolynomialCutoff ---


@pytest.fixture
def polynomial_module():
    """Create a PolynomialCutoff module."""
    return PolynomialCutoff(r_cut=5.0)


class BMPolynomialCutoff:
    """Benchmarks for PolynomialCutoff."""

    def test_forward(self, benchmark, polynomial_module, graph_data):
        distances = graph_data["bond_dist"]
        benchmark(polynomial_module, distances)

    def test_backward(self, benchmark, polynomial_module, graph_data):
        distances = graph_data["bond_dist"].clone().requires_grad_(True)

        def _forward_backward():
            out = polynomial_module(distances)
            out.sum().backward()
            if distances.grad is not None:
                distances.grad.zero_()

        benchmark(_forward_backward)

    def test_forward_compiled(self, benchmark, polynomial_module, graph_data):
        distances = graph_data["bond_dist"]
        compiled = torch.compile(polynomial_module, backend="inductor")
        # warmup
        compiled(distances)
        benchmark(compiled, distances)

    def test_graph_breaks(self, polynomial_module, graph_data):
        distances = graph_data["bond_dist"]
        explanation = torch._dynamo.explain(polynomial_module)(distances)
        assert explanation.graph_break_count == 0
