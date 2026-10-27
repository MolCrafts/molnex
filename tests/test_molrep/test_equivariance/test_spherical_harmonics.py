"""The spherical-harmonics polynomial against an independent monomial replay."""

from __future__ import annotations

import pytest
import torch

from molrep.equivariance import SphericalHarmonics


def _replay(layer, vectors):
    y = vectors.new_zeros(vectors.shape[0], layer.components)
    terms = layer.terms.detach().cpu()
    coeff = layer.coefficients.detach()
    for term, coefficient in zip(terms, coeff, strict=True):
        i, j, k, oseg = (int(item) for item in term.tolist())
        prod = vectors.new_full((vectors.shape[0],), float(coefficient))
        if i >= 0:
            prod = prod * vectors[:, i]
        if j >= 0:
            prod = prod * vectors[:, j]
        if k >= 0:
            prod = prod * vectors[:, k]
        y[:, oseg] = y[:, oseg] + prod
    return y


class TestSphericalHarmonics:
    def test_rejects_a_narrow_dtype_and_a_degree_above_three(self):
        with pytest.raises(ValueError, match="float32 or float64"):
            SphericalHarmonics([0, 1], dtype=torch.float16)
        with pytest.raises(ValueError, match="0..3"):
            SphericalHarmonics([0, 1, 2, 3, 4], dtype=torch.float64)

    def test_forward_and_grads_match_the_replay(self, device):
        torch.manual_seed(8)
        layer = SphericalHarmonics([0, 1, 2, 3], dtype=torch.float64, device=device)
        stored = torch.randn(7, 4, device=device, dtype=torch.float64)
        vectors = stored[:, :3].detach().requires_grad_(True)
        actual = layer(vectors)
        expected = _replay(layer, vectors)
        assert actual.shape == (7, 16)
        assert actual.dtype == torch.float64
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)
        cotangent = torch.randn_like(actual, requires_grad=True)
        (gx,) = torch.autograd.grad(actual, vectors, cotangent, create_graph=True)
        (egx,) = torch.autograd.grad(expected, vectors, cotangent, create_graph=True)
        torch.testing.assert_close(gx, egx, atol=1e-10, rtol=1e-9)
        dgx = torch.randn_like(gx)
        second = (gx * dgx).sum()
        expected_second = (egx * dgx).sum()
        dx, dg = torch.autograd.grad(second, (vectors, cotangent))
        edx, edg = torch.autograd.grad(expected_second, (vectors, cotangent))
        torch.testing.assert_close(dx, edx, atol=1e-9, rtol=1e-8)
        torch.testing.assert_close(dg, edg, atol=1e-9, rtol=1e-8)

    def test_float32_against_float64_and_empty(self, device):
        torch.manual_seed(9)
        wide = SphericalHarmonics([0, 1, 2], dtype=torch.float64, device=device)
        narrow = SphericalHarmonics([0, 1, 2], dtype=torch.float32, device=device)
        vectors = torch.randn(1, 3, device=device, dtype=torch.float64)
        reference = _replay(wide, vectors)
        actual = narrow(vectors.float())
        torch.testing.assert_close(actual.double(), reference, atol=1e-8, rtol=1e-5)
        empty = torch.randn(0, 3, device=device, dtype=torch.float64, requires_grad=True)
        output = wide(empty)
        assert output.shape == (0, 9)
        (gx,) = torch.autograd.grad(output.sum(), empty)
        assert gx.shape == empty.shape

    @pytest.mark.slow
    def test_gradcheck(self, device):
        torch.manual_seed(10)
        layer = SphericalHarmonics([0, 1, 2, 3], dtype=torch.float64, device=device)
        vectors = torch.randn(4, 3, device=device, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(layer, (vectors,), fast_mode=True)
        assert torch.autograd.gradgradcheck(layer, (vectors,), fast_mode=True)

    @pytest.mark.slow
    def test_compile_and_cuda_graph(self, cuda):
        from molix.compile import Compiler

        torch.manual_seed(11)
        layer = SphericalHarmonics([0, 1, 2], dtype=torch.float64, device=cuda)
        vectors = torch.randn(5, 3, device=cuda, dtype=torch.float64, requires_grad=True)
        cotangent = torch.randn(5, 9, device=cuda, dtype=torch.float64)

        def fn(v):
            output = layer(v)
            (gx,) = torch.autograd.grad(output, v, cotangent, create_graph=True)
            return gx.square().sum()

        explain = Compiler.explain(fn, vectors)
        assert explain.graph_count == 1 and explain.graph_break_count == 0
        compiled = Compiler(fullgraph=True, dynamic=False)(fn)
        actual = compiled(vectors)
        expected = fn(vectors)
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-9)
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                compiled(vectors)
        torch.cuda.current_stream().wait_stream(stream)
        with torch.cuda.graph(graph):
            captured = compiled(vectors)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, expected, atol=1e-7, rtol=1e-9)
