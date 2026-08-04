"""Tests for molix.compile.Compiler."""

import torch
import torch.nn as nn

from molix.compile import Compiler


class TestCompiler:
    """Tests for Compiler configuration + application."""

    def test_call_returns_compiled_wrapper(self):
        module = nn.Linear(10, 5)
        result = Compiler(backend="eager")(module)
        assert result is not module

    def test_compiled_module_produces_same_output(self):
        module = nn.Linear(10, 5)
        x = torch.randn(3, 10)
        with torch.no_grad():
            expected = module(x)
        compiled = Compiler(backend="eager")(module)
        with torch.no_grad():
            actual = compiled(x)
        assert torch.allclose(expected, actual)

    def test_cuda_graph_preset_values(self):
        """The named preset is exactly the benchmarked winning force-training config."""
        assert Compiler.CUDA_GRAPH_PRESET == {
            "backend": "inductor",
            "fullgraph": True,
            "dynamic": False,
            "mode": "reduce-overhead",
        }

    def test_cuda_graphs_applies_preset(self):
        c = Compiler(cuda_graphs=True)
        assert (c.backend, c.fullgraph, c.dynamic, c.mode) == (
            "inductor",
            True,
            False,
            "reduce-overhead",
        )

    def test_cuda_graphs_overrides_explicit_args(self):
        c = Compiler(cuda_graphs=True, dynamic=True, mode="max-autotune")
        assert c.dynamic is False
        assert c.mode == "reduce-overhead"


class TestCountGraphBreaks:
    """Tests for Compiler.count_graph_breaks()."""

    def test_simple_linear_no_breaks(self):
        module = nn.Linear(10, 5)
        x = torch.randn(3, 10)
        assert Compiler.count_graph_breaks(module, x) == 0

    def test_sequential_no_breaks(self):
        module = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
        x = torch.randn(3, 10)
        assert Compiler.count_graph_breaks(module, x) == 0
