"""Tests for molix.compile module."""

import torch
import torch.nn as nn

from molix.compile import count_graph_breaks, maybe_compile


class TestMaybeCompile:
    """Tests for maybe_compile()."""

    def test_compile_false_returns_same_module(self):
        module = nn.Linear(10, 5)
        result = maybe_compile(module, compile=False)
        assert result is module

    def test_compile_true_returns_compiled(self):
        module = nn.Linear(10, 5)
        result = maybe_compile(module, compile=True, backend="eager")
        # Compiled module is not the same object
        assert result is not module

    def test_compiled_module_produces_same_output(self):
        module = nn.Linear(10, 5)
        x = torch.randn(3, 10)

        with torch.no_grad():
            expected = module(x)

        compiled = maybe_compile(module, compile=True, backend="eager")
        with torch.no_grad():
            actual = compiled(x)

        assert torch.allclose(expected, actual)

    def test_default_is_no_compile(self):
        module = nn.Linear(10, 5)
        result = maybe_compile(module)
        assert result is module


class TestCountGraphBreaks:
    """Tests for count_graph_breaks()."""

    def test_simple_linear_no_breaks(self):
        module = nn.Linear(10, 5)
        x = torch.randn(3, 10)
        breaks = count_graph_breaks(module, x)
        assert breaks == 0

    def test_sequential_no_breaks(self):
        module = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )
        x = torch.randn(3, 10)
        breaks = count_graph_breaks(module, x)
        assert breaks == 0
