"""Allegro weighted channels against an independent term replay.

The Allegro module that reads channel-major weights is covered in
``tests/test_molzoo/test_allegro/test_layers.py``; here the weights are
irrep-major, as the operator takes them.
"""

from __future__ import annotations

import pytest
import torch

from molrep.equivariance import WeightedChannels


def _replay(layer, basis, weight, index, num_rows):
    edges = basis.shape[0]
    terms = layer.terms.detach().cpu()
    coeff = layer.coefficients
    output = basis.new_zeros(edges, layer.components, layer.channels)
    irrep_major = weight.reshape(edges, layer.num_irreps, layer.channels)
    for term, coefficient in zip(terms, coeff, strict=True):
        component, irrep = (int(item) for item in term.tolist())
        output[:, component] = (
            output[:, component]
            + coefficient * basis[:, component].unsqueeze(-1) * irrep_major[:, irrep]
        )
    flat = output.reshape(edges, -1)
    if index is None:
        return flat
    summed = flat.new_zeros(num_rows, flat.shape[1])
    return summed.index_add(0, index, flat)


class TestWeightedChannels:
    def test_rejects_a_narrow_dtype_and_a_lone_index(self):
        with pytest.raises(ValueError, match="float32 or float64"):
            WeightedChannels([1], 1, dtype=torch.float16)
        layer = WeightedChannels([1], 1, dtype=torch.float64)
        with pytest.raises(ValueError, match="num_rows"):
            layer(
                torch.ones(1, 1, dtype=torch.float64),
                torch.ones(1, 1, 1, dtype=torch.float64),
                torch.tensor([0]),
            )

    def test_forward_and_grads_match_the_replay(self, device):
        torch.manual_seed(12)
        layer = WeightedChannels([1, 3, 5], 4, dtype=torch.float64, device=device)
        edges, atoms = 5, 4
        basis = torch.randn(edges, 9, device=device, dtype=torch.float64, requires_grad=True)
        weights = torch.randn(edges, 3, 4, device=device, dtype=torch.float64, requires_grad=True)
        # Two edges share atom 0. Atom 3 is unused.
        index = torch.tensor([0, 0, 1, 2, 1], device=device)
        actual = layer(basis, weights, index, atoms)
        expected = _replay(layer, basis, weights, index, atoms)
        assert actual.shape == (atoms, 9 * 4)
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)
        assert torch.count_nonzero(actual.view(atoms, 9, 4)[3]) == 0
        cotangent = torch.randn_like(actual, requires_grad=True)
        gb, gw = torch.autograd.grad(actual, (basis, weights), cotangent, create_graph=True)
        egb, egw = torch.autograd.grad(expected, (basis, weights), cotangent, create_graph=True)
        torch.testing.assert_close(gb, egb, atol=1e-10, rtol=1e-9)
        torch.testing.assert_close(gw, egw, atol=1e-10, rtol=1e-9)
        dgb = torch.randn_like(gb)
        dgw = torch.randn_like(gw)
        second = (gb * dgb).sum() + (gw * dgw).sum()
        expected_second = (egb * dgb).sum() + (egw * dgw).sum()
        db, dw, dg = torch.autograd.grad(second, (basis, weights, cotangent))
        edb, edw, edg = torch.autograd.grad(expected_second, (basis, weights, cotangent))
        torch.testing.assert_close(db, edb, atol=1e-9, rtol=1e-8)
        torch.testing.assert_close(dw, edw, atol=1e-9, rtol=1e-8)
        torch.testing.assert_close(dg, edg, atol=1e-9, rtol=1e-8)

    def test_per_edge_output_and_empty(self, device):
        torch.manual_seed(13)
        wide = WeightedChannels([1, 3], 2, dtype=torch.float64, device=device)
        narrow = WeightedChannels([1, 3], 2, dtype=torch.float32, device=device)
        basis = torch.randn(3, 4, device=device, dtype=torch.float64)
        weights = torch.randn(3, 2, 2, device=device, dtype=torch.float64)
        reference = _replay(wide, basis, weights, None, None)
        actual = narrow(basis.float(), weights.float())
        torch.testing.assert_close(actual.double(), reference, atol=1e-8, rtol=1e-5)
        empty_basis = torch.randn(0, 4, device=device, dtype=torch.float64, requires_grad=True)
        empty_weights = torch.randn(0, 2, 2, device=device, dtype=torch.float64, requires_grad=True)
        output = wide(empty_basis, empty_weights)
        assert output.shape == (0, 8)
        gb, gw = torch.autograd.grad(output.sum(), (empty_basis, empty_weights))
        assert gb.shape == empty_basis.shape
        assert gw.shape == empty_weights.shape
        indexed = wide(
            empty_basis, empty_weights, torch.empty(0, dtype=torch.int64, device=device), 3
        )
        assert indexed.shape == (3, 8)
        assert torch.count_nonzero(indexed) == 0

    @pytest.mark.slow
    def test_gradcheck(self, device):
        torch.manual_seed(14)
        layer = WeightedChannels([1, 3], 2, dtype=torch.float64, device=device)
        basis = torch.randn(3, 4, device=device, dtype=torch.float64, requires_grad=True)
        weights = torch.randn(3, 2, 2, device=device, dtype=torch.float64, requires_grad=True)
        index = torch.tensor([0, 0, 1], device=device)

        def indexed(values, rows):
            return layer(values, rows, index, 3)

        assert torch.autograd.gradcheck(layer, (basis, weights), fast_mode=True)
        assert torch.autograd.gradcheck(indexed, (basis, weights), fast_mode=True)
        assert torch.autograd.gradgradcheck(indexed, (basis, weights), fast_mode=True)

    @pytest.mark.slow
    def test_compile_and_cuda_graph(self, cuda):
        from molix.compile import Compiler

        torch.manual_seed(15)
        layer = WeightedChannels([1, 3], 4, dtype=torch.float64, device=cuda)
        index = torch.tensor([1, 0, 1], device=cuda)
        basis = torch.randn(3, 4, device=cuda, dtype=torch.float64, requires_grad=True)
        weights = torch.randn(3, 2, 4, device=cuda, dtype=torch.float64, requires_grad=True)
        cotangent = torch.randn(2, 16, device=cuda, dtype=torch.float64)

        def fn(values, rows):
            output = layer(values, rows, index, 2)
            gb, gw = torch.autograd.grad(output, (values, rows), cotangent, create_graph=True)
            return gb.square().sum() + gw.square().sum()

        explain = Compiler.explain(fn, basis, weights)
        assert explain.graph_count == 1 and explain.graph_break_count == 0
        compiled = Compiler(fullgraph=True, dynamic=False)(fn)
        actual = compiled(basis, weights)
        expected = fn(basis, weights)
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-9)
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                compiled(basis, weights)
        torch.cuda.current_stream().wait_stream(stream)
        with torch.cuda.graph(graph):
            captured = compiled(basis, weights)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, expected, atol=1e-7, rtol=1e-9)
