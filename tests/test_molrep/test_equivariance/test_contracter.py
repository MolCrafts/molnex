"""The Allegro contracter against an independent monomial replay.

The Allegro module that owns the ``(channels, paths)`` parameter is covered in
``tests/test_molzoo/test_allegro/test_layers.py``.
"""

from __future__ import annotations

import pytest
import torch

import molrep.eq as meq
from molrep.equivariance import Contracter


def _layer(spec: str, channels: int, dtype: torch.dtype, device) -> Contracter:
    irreps = meq.Irreps("O3", spec)
    return Contracter(irreps, irreps, irreps, channels, dtype=dtype, device=device)


def _replay(layer, x1, x2, weight, index):
    terms = layer.terms.detach().cpu()
    coeff = layer.coefficients
    edges = x1.shape[0]
    y = x1.new_zeros(edges, layer.output_segments, layer.channels)
    left = x1.reshape(edges, layer.input1_segments, layer.channels)
    right = x2.reshape(x2.shape[0], layer.input2_segments, layer.channels)
    shared = weight.reshape(layer.weight_segments, layer.channels)
    for term, coefficient in zip(terms, coeff, strict=True):
        wseg, i, j, oseg = (int(item) for item in term.tolist())
        y[:, oseg] = y[:, oseg] + coefficient * shared[wseg] * left[:, i] * right[index, j]
    return y.reshape(edges, -1)


class TestContracter:
    def test_rejects_a_narrow_dtype_and_a_missing_path(self):
        with pytest.raises(ValueError, match="float32 or float64"):
            _layer("0e+1o", 2, torch.float16, None)
        with pytest.raises(ValueError, match="no tensor-product path"):
            Contracter(
                meq.Irreps("O3", "1x0e"),
                meq.Irreps("O3", "1x0e"),
                meq.Irreps("O3", "1x1o"),
                2,
                dtype=torch.float64,
            )

    def test_forward_and_grads_match_the_replay(self, device):
        torch.manual_seed(12)
        layer = _layer("0e+1o+2e", 4, torch.float64, device)
        if device.type == "cuda":
            assert layer.config >= 0  # an Allegro layer shape: the compiled table
        edges, atoms = 4, 4
        x1 = torch.randn(
            edges, layer.input1_dim, device=device, dtype=torch.float64, requires_grad=True
        )
        x2 = torch.randn(
            atoms, layer.input2_dim, device=device, dtype=torch.float64, requires_grad=True
        )
        weight = torch.randn(
            layer.weight_segments, 4, device=device, dtype=torch.float64, requires_grad=True
        )
        # Two edges share atom 0. Atom 3 is unused.
        index = torch.tensor([0, 0, 1, 2], device=device)
        actual = layer(x1, x2, weight, index)
        expected = _replay(layer, x1, x2, weight, index)
        assert actual.shape == (edges, 9 * 4)
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)
        cotangent = torch.randn_like(actual, requires_grad=True)
        gx1, gx2, gw = torch.autograd.grad(actual, (x1, x2, weight), cotangent, create_graph=True)
        egx1, egx2, egw = torch.autograd.grad(
            expected, (x1, x2, weight), cotangent, create_graph=True
        )
        torch.testing.assert_close(gx1, egx1, atol=1e-10, rtol=1e-9)
        torch.testing.assert_close(gx2, egx2, atol=1e-10, rtol=1e-9)
        torch.testing.assert_close(gw, egw, atol=1e-10, rtol=1e-9)
        assert torch.count_nonzero(gx2.reshape(atoms, layer.input2_segments, 4)[3]) == 0
        dgx1 = torch.randn_like(gx1)
        dgx2 = torch.randn_like(gx2)
        dgw = torch.randn_like(gw)
        second = (gx1 * dgx1).sum() + (gx2 * dgx2).sum() + (gw * dgw).sum()
        expected_second = (egx1 * dgx1).sum() + (egx2 * dgx2).sum() + (egw * dgw).sum()
        dx1, dx2, dw, dg = torch.autograd.grad(second, (x1, x2, weight, cotangent))
        edx1, edx2, edw, edg = torch.autograd.grad(expected_second, (x1, x2, weight, cotangent))
        torch.testing.assert_close(dx1, edx1, atol=1e-9, rtol=1e-8)
        torch.testing.assert_close(dx2, edx2, atol=1e-9, rtol=1e-8)
        torch.testing.assert_close(dw, edw, atol=1e-9, rtol=1e-8)
        torch.testing.assert_close(dg, edg, atol=1e-9, rtol=1e-8)

    def test_outside_the_family_runs_the_generic_kernel(self, device):
        """A triple no Allegro layer builds has no compiled table: the class-blocked kernel."""
        torch.manual_seed(16)
        layer = Contracter(
            meq.Irreps("O3", "0e+1o+2e"),
            meq.Irreps("O3", "0e+1o"),
            meq.Irreps("O3", "1o+2e"),
            4,
            dtype=torch.float64,
            device=device,
        )
        assert layer.config == -1
        x1 = torch.randn(
            3, layer.input1_dim, device=device, dtype=torch.float64, requires_grad=True
        )
        x2 = torch.randn(
            2, layer.input2_dim, device=device, dtype=torch.float64, requires_grad=True
        )
        weight = torch.randn(
            layer.weight_segments, 4, device=device, dtype=torch.float64, requires_grad=True
        )
        index = torch.tensor([1, 0, 1], device=device)
        actual = layer(x1, x2, weight, index)
        expected = _replay(layer, x1, x2, weight, index)
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)
        cotangent = torch.randn_like(actual, requires_grad=True)
        grads = torch.autograd.grad(actual, (x1, x2, weight), cotangent, create_graph=True)
        expected_grads = torch.autograd.grad(
            expected, (x1, x2, weight), cotangent, create_graph=True
        )
        for a, b in zip(grads, expected_grads, strict=True):
            torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-9)
        directions = [torch.randn_like(g) for g in grads]
        second = torch.autograd.grad(
            sum((g * d).sum() for g, d in zip(grads, directions, strict=True)),
            (x1, x2, weight, cotangent),
        )
        expected_second = torch.autograd.grad(
            sum((g * d).sum() for g, d in zip(expected_grads, directions, strict=True)),
            (x1, x2, weight, cotangent),
        )
        for a, b in zip(second, expected_second, strict=True):
            torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-8)

    def test_float32_against_float64_and_empty(self, device):
        torch.manual_seed(13)
        wide = _layer("0e+1o", 2, torch.float64, device)
        narrow = _layer("0e+1o", 2, torch.float32, device)
        weight = torch.randn(wide.weight_segments, 2, device=device, dtype=torch.float64)
        x1 = torch.randn(2, wide.input1_dim, device=device, dtype=torch.float64)
        x2 = torch.randn(3, wide.input2_dim, device=device, dtype=torch.float64)
        index = torch.tensor([2, 0], device=device)
        reference = _replay(wide, x1, x2, weight, index)
        actual = narrow(x1.float(), x2.float(), weight.float(), index)
        torch.testing.assert_close(actual.double(), reference, atol=1e-8, rtol=1e-5)
        empty_x1 = torch.randn(
            0, wide.input1_dim, device=device, dtype=torch.float64, requires_grad=True
        )
        empty_x2 = torch.randn(
            2, wide.input2_dim, device=device, dtype=torch.float64, requires_grad=True
        )
        empty_index = torch.empty(0, dtype=torch.int64, device=device)
        output = wide(empty_x1, empty_x2, weight, empty_index)
        assert output.shape == (0, wide.output_dim)
        gx1, gx2 = torch.autograd.grad(output.sum(), (empty_x1, empty_x2))
        assert gx1.shape == empty_x1.shape
        assert gx2.shape == empty_x2.shape
        assert torch.count_nonzero(gx2) == 0

    @pytest.mark.slow
    def test_gradcheck(self, device):
        torch.manual_seed(14)
        layer = _layer("0e+1o", 2, torch.float64, device)
        index = torch.tensor([0, 1, 0], device=device)
        x1 = torch.randn(
            3, layer.input1_dim, device=device, dtype=torch.float64, requires_grad=True
        )
        x2 = torch.randn(
            2, layer.input2_dim, device=device, dtype=torch.float64, requires_grad=True
        )
        weight = torch.randn(
            layer.weight_segments, 2, device=device, dtype=torch.float64, requires_grad=True
        )

        def fn(pair, atom, paths):
            return layer(pair, atom, paths, index)

        assert torch.autograd.gradcheck(fn, (x1, x2, weight), fast_mode=True)
        assert torch.autograd.gradgradcheck(fn, (x1, x2, weight), fast_mode=True)

    @pytest.mark.slow
    def test_compile_and_cuda_graph(self, cuda):
        from molix.compile import Compiler

        torch.manual_seed(15)
        layer = _layer("0e+1o", 4, torch.float64, cuda)
        index = torch.tensor([1, 0, 1], device=cuda)
        weight = torch.randn(layer.weight_segments, 4, device=cuda, dtype=torch.float64)
        x1 = torch.randn(3, layer.input1_dim, device=cuda, dtype=torch.float64, requires_grad=True)
        x2 = torch.randn(2, layer.input2_dim, device=cuda, dtype=torch.float64, requires_grad=True)
        cotangent = torch.randn(3, layer.output_dim, device=cuda, dtype=torch.float64)

        def fn(pair, atom):
            output = layer(pair, atom, weight, index)
            gx1, gx2 = torch.autograd.grad(output, (pair, atom), cotangent, create_graph=True)
            return gx1.square().sum() + gx2.square().sum()

        explain = Compiler.explain(fn, x1, x2)
        assert explain.graph_count == 1 and explain.graph_break_count == 0
        compiled = Compiler(fullgraph=True, dynamic=False)(fn)
        actual = compiled(x1, x2)
        expected = fn(x1, x2)
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-9)
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                compiled(x1, x2)
        torch.cuda.current_stream().wait_stream(stream)
        with torch.cuda.graph(graph):
            captured = compiled(x1, x2)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, expected, atol=1e-7, rtol=1e-9)
