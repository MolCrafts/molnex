"""The symmetric contraction against an independent term replay.

The replay multiplies the term table in PyTorch, one product per term; it is
neither the CUDA loop nor the chunked reference evaluator. Float64 agrees
tightly; float32 is compared with the float64 replay.
"""

from __future__ import annotations

import pytest
import torch

import molrep.eq as meq
from molrep.equivariance import SymmetricContraction
from molrep.equivariance.descriptor import natural_irreps


def _layer(irreps_in, irreps_out, degree, elements, *, dtype, device):
    return SymmetricContraction(irreps_in, irreps_out, degree, elements, dtype=dtype, device=device)


def _replay(layer, features, index):
    """Projected weight, then one product per term. The oracle for the module."""
    weight = layer.projected_weight().reshape(-1, layer.weight_segments, layer.channels)
    x = features.reshape(-1, layer.input_segments, layer.channels)
    y = features.new_zeros(features.shape[0], layer.output_segments, layer.channels)
    terms = layer.terms.detach().cpu()
    coeff = layer.coefficients.detach()
    for term, coefficient in zip(terms, coeff, strict=True):
        wseg, i, j, k, oseg = (int(v) for v in term.tolist())
        prod = weight[index.long(), wseg] * x[:, i]
        if j >= 0:
            prod = prod * x[:, j]
        if k >= 0:
            prod = prod * x[:, k]
        y[:, oseg] = y[:, oseg] + coefficient * prod
    return y.reshape(features.shape[0], -1)


class TestSymmetricContraction:
    def test_rejects_a_narrow_dtype_and_a_mixed_multiplicity(self):
        with pytest.raises(ValueError, match="float32 or float64"):
            _layer(
                meq.Irreps("O3", "2x0e"),
                meq.Irreps("O3", "2x0e"),
                1,
                1,
                dtype=torch.float16,
                device=None,
            )
        with pytest.raises(ValueError, match="multiplicity"):
            _layer(
                meq.Irreps("O3", "2x0e+3x1o"),
                meq.Irreps("O3", "2x0e"),
                1,
                1,
                dtype=torch.float64,
                device=None,
            )

    def test_state_dict_is_mace_weight_and_projection(self):
        layer = _layer(
            meq.Irreps("O3", "4x0e+4x1o"),
            meq.Irreps("O3", "4x0e"),
            2,
            3,
            dtype=torch.float64,
            device=None,
        )
        assert set(layer.state_dict()) == {"weight", "projection"}
        assert layer.weight.shape == (3, layer.projection.shape[0], 4)
        assert layer.projection.shape[1] == layer.weight_segments

    @pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
    def test_forward_and_grads_match_the_replay(self, device, index_dtype):
        torch.manual_seed(4)
        irreps_in = meq.Irreps("O3", "7x0e+7x1o")
        irreps_out = meq.Irreps("O3", "7x0e")
        layer = _layer(irreps_in, irreps_out, 2, 4, dtype=torch.float64, device=device)
        nodes = 7
        # Non-contiguous view. The module copies before the kernel.
        stored = torch.randn(nodes, irreps_in.dim + 1, device=device, dtype=torch.float64)
        features = stored[:, :-1].detach().requires_grad_(True)
        index = torch.tensor([0, 0, 2, 1, 2, 0, 1], device=device, dtype=index_dtype)
        actual = layer(features, index)
        expected = _replay(layer, features, index)
        assert actual.shape == (nodes, irreps_out.dim)
        assert actual.dtype == torch.float64
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)

        cotangent = torch.randn_like(actual, requires_grad=True)
        (gx, gw) = torch.autograd.grad(
            actual, (features, layer.weight), cotangent, create_graph=True
        )
        (egx, egw) = torch.autograd.grad(
            expected, (features, layer.weight), cotangent, create_graph=True
        )
        torch.testing.assert_close(gx, egx, atol=1e-10, rtol=1e-9)
        torch.testing.assert_close(gw, egw, atol=1e-10, rtol=1e-9)
        # Element 3 is unused.
        unused = gw.reshape(4, -1)[3]
        assert torch.count_nonzero(unused) == 0

        dgx = torch.randn_like(gx)
        dgw = torch.randn_like(gw)
        second = (gx * dgx).sum() + (gw * dgw).sum()
        expected_second = (egx * dgx).sum() + (egw * dgw).sum()
        dx, dw, dg = torch.autograd.grad(second, (features, layer.weight, cotangent))
        edx, edw, edg = torch.autograd.grad(expected_second, (features, layer.weight, cotangent))
        torch.testing.assert_close(dx, edx, atol=1e-9, rtol=1e-8)
        torch.testing.assert_close(dw, edw, atol=1e-9, rtol=1e-8)
        torch.testing.assert_close(dg, edg, atol=1e-9, rtol=1e-8)

    def test_degree_three_and_float32_against_float64(self, device):
        torch.manual_seed(5)
        irreps_in = meq.Irreps("O3", "2x0e+2x1o")
        irreps_out = meq.Irreps("O3", "2x0e")
        wide = _layer(irreps_in, irreps_out, 3, 1, dtype=torch.float64, device=device)
        narrow = _layer(irreps_in, irreps_out, 3, 1, dtype=torch.float32, device=device)
        with torch.no_grad():
            narrow.weight.copy_(wide.weight.float())
            narrow.projection.copy_(wide.projection.float())
        features = torch.randn(7, irreps_in.dim, device=device, dtype=torch.float64)
        index = torch.zeros(7, device=device, dtype=torch.int32)
        reference = _replay(wide, features, index)
        actual = narrow(features.float(), index)
        torch.testing.assert_close(actual.double(), reference, atol=1e-8, rtol=1e-5)

    @pytest.mark.parametrize("family", [(3, 1, 3), (3, 2, 3), (2, 2, 2)])
    def test_mace_family_matches_the_replay(self, device, family):
        """MACE's own shapes — the compiled table on CUDA — against the per-term replay."""
        torch.manual_seed(21)
        l_in, l_out, degree = family
        irreps_in = natural_irreps(32, l_in)
        irreps_out = natural_irreps(32, l_out)
        # Six elements of which three occur, with gaps, over more nodes than
        # one chunk of the element sum holds.
        layer = _layer(irreps_in, irreps_out, degree, 6, dtype=torch.float64, device=device)
        if device.type == "cuda":
            assert layer.config == l_in * 100 + l_out * 10 + degree
        features = torch.randn(
            40, irreps_in.dim, device=device, dtype=torch.float64, requires_grad=True
        )
        index = torch.tensor([0, 3, 5] * 13 + [5], device=device)
        actual = layer(features, index)
        expected = _replay(layer, features, index)
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)
        cotangent = torch.randn_like(actual, requires_grad=True)
        gx, gw = torch.autograd.grad(actual, (features, layer.weight), cotangent, create_graph=True)
        egx, egw = torch.autograd.grad(
            expected, (features, layer.weight), cotangent, create_graph=True
        )
        torch.testing.assert_close(gx, egx, atol=1e-10, rtol=1e-9)
        torch.testing.assert_close(gw, egw, atol=1e-10, rtol=1e-9)
        dgx, dgw = torch.randn_like(gx), torch.randn_like(gw)
        second = (gx * dgx).sum() + (gw * dgw).sum()
        expected_second = (egx * dgx).sum() + (egw * dgw).sum()
        dx, dw, dg = torch.autograd.grad(second, (features, layer.weight, cotangent))
        edx, edw, edg = torch.autograd.grad(expected_second, (features, layer.weight, cotangent))
        torch.testing.assert_close(dx, edx, atol=1e-9, rtol=1e-8)
        torch.testing.assert_close(dw, edw, atol=1e-9, rtol=1e-8)
        torch.testing.assert_close(dg, edg, atol=1e-9, rtol=1e-8)

    def test_outside_the_family_runs_the_generic_table(self, device):
        """Scalar-only irreps have no compiled table: the generic kernel, same replay."""
        torch.manual_seed(22)
        irreps = meq.Irreps("O3", "3x0e")
        layer = _layer(irreps, irreps, 3, 4, dtype=torch.float64, device=device)
        assert layer.config == -1
        features = torch.randn(37, 3, device=device, dtype=torch.float64, requires_grad=True)
        index = torch.tensor([3, 0, 3] * 12 + [0], device=device)
        actual = layer(features, index)
        expected = _replay(layer, features, index)
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)
        cotangent = torch.randn_like(actual, requires_grad=True)
        gx, gw = torch.autograd.grad(actual, (features, layer.weight), cotangent, create_graph=True)
        egx, egw = torch.autograd.grad(
            expected, (features, layer.weight), cotangent, create_graph=True
        )
        torch.testing.assert_close(gx, egx, atol=1e-10, rtol=1e-9)
        torch.testing.assert_close(gw, egw, atol=1e-10, rtol=1e-9)
        dgx, dgw = torch.randn_like(gx), torch.randn_like(gw)
        dx, dw, dg = torch.autograd.grad(
            (gx * dgx).sum() + (gw * dgw).sum(), (features, layer.weight, cotangent)
        )
        edx, edw, edg = torch.autograd.grad(
            (egx * dgx).sum() + (egw * dgw).sum(), (features, layer.weight, cotangent)
        )
        torch.testing.assert_close(dx, edx, atol=1e-9, rtol=1e-8)
        torch.testing.assert_close(dw, edw, atol=1e-9, rtol=1e-8)
        torch.testing.assert_close(dg, edg, atol=1e-9, rtol=1e-8)

    def test_empty_batch(self, device):
        irreps_in = meq.Irreps("O3", "2x0e")
        irreps_out = meq.Irreps("O3", "2x0e")
        layer = _layer(irreps_in, irreps_out, 1, 2, dtype=torch.float64, device=device)
        features = torch.randn(
            0, irreps_in.dim, device=device, dtype=torch.float64, requires_grad=True
        )
        index = torch.empty(0, device=device, dtype=torch.int64)
        output = layer(features, index)
        assert output.shape == (0, irreps_out.dim)
        assert output.dtype == torch.float64
        gx, gw = torch.autograd.grad(output.sum(), (features, layer.weight))
        assert gx.shape == features.shape
        assert torch.count_nonzero(gw) == 0

    @pytest.mark.slow
    def test_gradcheck(self, device):
        torch.manual_seed(6)
        irreps_in = meq.Irreps("O3", "2x0e+2x1o")
        irreps_out = meq.Irreps("O3", "2x0e")
        layer = _layer(irreps_in, irreps_out, 2, 2, dtype=torch.float64, device=device)
        features = torch.randn(
            3, irreps_in.dim, device=device, dtype=torch.float64, requires_grad=True
        )
        index = torch.tensor([0, 1, 0], device=device)

        def fn(x, weight):
            return torch.func.functional_call(layer, {"weight": weight}, (x, index))

        args = (features, layer.weight.detach().clone().requires_grad_(True))
        assert torch.autograd.gradcheck(fn, args, fast_mode=True)
        assert torch.autograd.gradgradcheck(fn, args, fast_mode=True)

    @pytest.mark.slow
    def test_compile_and_cuda_graph(self, cuda):
        from molix.compile import Compiler

        torch.manual_seed(7)
        irreps_in = meq.Irreps("O3", "2x0e+2x1o")
        irreps_out = meq.Irreps("O3", "2x0e")
        layer = _layer(irreps_in, irreps_out, 2, 3, dtype=torch.float64, device=cuda)
        features = torch.randn(
            4, irreps_in.dim, device=cuda, dtype=torch.float64, requires_grad=True
        )
        index = torch.tensor([0, 1, 1, 2], device=cuda, dtype=torch.int32)
        cotangent = torch.randn(4, irreps_out.dim, device=cuda, dtype=torch.float64)

        def fn(x):
            output = layer(x, index)
            (gx, gw) = torch.autograd.grad(output, (x, layer.weight), cotangent, create_graph=True)
            return gx.square().sum() + gw.square().sum()

        explain = Compiler.explain(fn, features)
        assert explain.graph_count == 1 and explain.graph_break_count == 0
        compiled = Compiler(fullgraph=True, dynamic=False)(fn)
        actual = compiled(features)
        expected = fn(features)
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-9)
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                compiled(features)
        torch.cuda.current_stream().wait_stream(stream)
        with torch.cuda.graph(graph):
            captured = compiled(features)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, expected, atol=1e-7, rtol=1e-9)
