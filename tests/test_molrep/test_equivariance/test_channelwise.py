"""The indexed channelwise product against an explicit per-term sum and the cue frontend.

Runs on the CPU (the reference path) and on CUDA (the kernels) alike; what
only the kernels own — bitwise reproducibility, CUDA-graph replay with index
changes in place, the fused adjoints — runs on CUDA.
"""

import pytest
import torch

import molrep.eq as meq
import molrep.eq.torch as cuet
from molrep.equivariance import ChannelWiseTensorProduct
from molrep.equivariance.descriptor import natural_irreps

from .conftest import torch_contraction


class TestChannelWiseTensorProduct:
    def test_schedule_is_the_stable_csr(self, product, device):
        torch.manual_seed(3)
        rows_a, rows_d, n_edges = 11, 7, 500
        source = torch.randint(rows_a, (n_edges,), device=device)
        target = torch.randint(rows_d, (n_edges,), device=device)
        target[target == 4] = 5  # one row without edges
        schedule = product.schedule(source, target, rows_a, rows_d)
        for order, rowptr, index, rows in [
            (schedule.order_t, schedule.rowptr_t, target, rows_d),
            (schedule.order_s, schedule.rowptr_s, source, rows_a),
        ]:
            assert order.dtype == rowptr.dtype == torch.int32
            expected = torch.argsort(index, stable=True).to(torch.int32)
            torch.testing.assert_close(order, expected)
            counts = torch.bincount(index, minlength=rows)
            expected_ptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)]).to(torch.int32)
            torch.testing.assert_close(rowptr, expected_ptr)

    def test_precomputed_schedule_is_the_default(self, product, device):
        torch.manual_seed(5)
        args = tuple(
            torch.randn(shape, device=device, dtype=torch.float64)
            for shape in [(6, 128), (40, 9), (40, product.weight_numel)]
        )
        source = torch.randint(6, (40,), device=device)
        target = torch.randint(6, (40,), device=device)
        schedule = product.schedule(source, target, 6, 6)
        default = product(*args, indices_1=source, indices_out=target, size_out=6)
        explicit = product(
            *args, indices_1=source, indices_out=target, size_out=6, schedule=schedule
        )
        assert torch.equal(default, explicit)

    @pytest.mark.parametrize(
        "channels,l1,l2",
        [(c, l1, 3) for c in (32, 64, 128) for l1 in (0, 1, 2, 3)] + [(64, 1, 2), (32, 3, 2)],
    )
    def test_angular_orders_channels_and_streams(self, device, channels, l1, l2):
        # MACE has scalar first-layer features and l<=3 edge harmonics; the
        # second interaction adds vector features. Cover every compiled (l1, l2).
        irreps = (natural_irreps(channels, l1), natural_irreps(1, l2), natural_irreps(channels, l2))
        native = ChannelWiseTensorProduct(*irreps, dtype=torch.float64, device=device)
        reference = cuet.ChannelWiseTensorProduct(
            *irreps,
            layout=meq.ir_mul,
            shared_weights=False,
            internal_weights=False,
            device=device,
            dtype=torch.float64,
            method="uniform_1d" if device.type == "cuda" else "naive",
        )
        # Several edges per node, nodes of unequal degree, an incomplete last
        # CTA stride in the edge kernel, every channel width.
        n_edges = 259
        args = tuple(
            torch.randn(shape, device=device, dtype=torch.float64, requires_grad=True)
            for shape in [
                (7, irreps[0].dim),
                (n_edges, irreps[1].dim),
                (n_edges, native.weight_numel),
            ]
        )
        s = torch.arange(n_edges, device=device) % 7
        t = (s * s + 1) % 7
        actual = native(*args, indices_1=s, indices_out=t, size_out=7)
        first = torch.autograd.grad(actual.square().sum(), args, create_graph=True)
        second = torch.autograd.grad(sum(g.square().sum() for g in first), args)
        expected = reference(*args, indices_1=s, indices_out=t, size_out=7)
        rf = torch.autograd.grad(expected.square().sum(), args, create_graph=True)
        rs = torch.autograd.grad(sum(g.square().sum() for g in rf), args)
        for a, b in zip((actual, *first, *second), (expected, *rf, *rs), strict=True):
            error = (a - b).square().mean().sqrt() / b.square().mean().sqrt().clamp_min(1)
            assert error < 1e-10
        # Relabel nodes and reorder edges: output equivariance must survive.
        permutation = torch.tensor([3, 0, 6, 1, 4, 2, 5], device=device)
        inverse = torch.argsort(permutation)
        edges = torch.randperm(n_edges, device=device)
        permuted = native(
            args[0][permutation],
            args[1][edges],
            args[2][edges],
            indices_1=inverse[s[edges]],
            indices_out=inverse[t[edges]],
            size_out=7,
        )
        torch.testing.assert_close(permuted, actual[permutation], atol=1e-10, rtol=1e-10)

    def test_unsupported_options_fail_before_loading(self, irreps):
        with pytest.raises(ValueError, match="layout"):
            ChannelWiseTensorProduct(*irreps, layout=meq.mul_ir, dtype=torch.float64)
        with pytest.raises(ValueError, match="float32 or float64"):
            ChannelWiseTensorProduct(*irreps, dtype=torch.float16)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_outputs_and_two_derivatives(self, irreps, device, dtype):
        native = ChannelWiseTensorProduct(*irreps, dtype=dtype, device=device)
        torch.manual_seed(17)
        args = [
            torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
            for shape in [(5, 128), (9, 9), (9, native.weight_numel)]
        ]
        # Duplicate destinations, one isolated node (5) and one source without edges (4 as target).
        edges = torch.tensor(
            [[0, 1], [1, 1], [2, 1], [3, 2], [0, 2], [4, 0], [1, 0], [2, 2], [3, 1]], device=device
        )
        s, t = edges[:, 0], edges[:, 1]
        actual = native(*args, indices_1=s, indices_out=t, size_out=6)
        expected = torch_contraction(native, *args, s, t, 6)
        atol, rtol = (1e-10, 1e-9) if dtype == torch.float64 else (1e-5, 1e-4)
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        cotangent = torch.randn_like(actual)
        direction = [torch.randn_like(x) for x in args]
        derivatives = []
        for out in (actual, expected):
            first = torch.autograd.grad(out, args, cotangent, create_graph=True)
            second = torch.autograd.grad(sum((g * v).sum() for g, v in zip(first, direction)), args)
            derivatives.append((first, second))
        for a, b in zip(derivatives[0][0], derivatives[1][0], strict=True):
            torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
        for a, b in zip(derivatives[0][1], derivatives[1][1], strict=True):
            torch.testing.assert_close(a, b, atol=atol * 10, rtol=rtol * 10)
        assert torch.count_nonzero(actual[5]) == 0

    def test_bitwise_reproducible(self, irreps, cuda):
        torch.manual_seed(11)
        native = ChannelWiseTensorProduct(*irreps, dtype=torch.float32, device=cuda)
        n_edges, rows = 3000, 13
        args = tuple(
            torch.randn(shape, device=cuda, dtype=torch.float32, requires_grad=True)
            for shape in [(rows, 128), (n_edges, 9), (n_edges, native.weight_numel)]
        )
        s = torch.randint(rows, (n_edges,), device=cuda)
        t = torch.randint(rows, (n_edges,), device=cuda)
        runs = []
        previous = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            for _ in range(2):
                out = native(*args, indices_1=s, indices_out=t, size_out=rows)
                runs.append((out, *torch.autograd.grad(out.square().sum(), args)))
        finally:
            torch.use_deterministic_algorithms(previous)
        for x, y in zip(*runs, strict=True):
            assert torch.equal(x, y)

    def test_empty_edges_and_same_storage_index_changes(self, product, device):
        a = torch.randn(4, 128, device=device, dtype=torch.float64, requires_grad=True)
        b = torch.empty(0, 9, device=device, dtype=torch.float64, requires_grad=True)
        w = torch.empty(
            0, product.weight_numel, device=device, dtype=torch.float64, requires_grad=True
        )
        index = torch.empty(0, device=device, dtype=torch.long)
        out = product(a, b, w, indices_1=index, indices_out=index, size_out=4)
        assert torch.count_nonzero(out) == 0
        grads = torch.autograd.grad(out.sum(), (a, b, w))
        assert all(torch.count_nonzero(x) == 0 for x in grads)
        b = torch.randn(3, 9, device=device, dtype=torch.float64)
        w = torch.randn(3, product.weight_numel, device=device, dtype=torch.float64)
        s = torch.tensor([0, 1, 2], device=device)
        t = torch.tensor([1, 1, 1], device=device)
        product(a, b, w, indices_1=s, indices_out=t, size_out=4)
        t.fill_(3)
        expected = torch_contraction(product, a, b, w, s, t, 4)
        actual = product(a, b, w, indices_1=s, indices_out=t, size_out=4)
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)

    @pytest.mark.slow
    def test_gradcheck(self, product, device):
        args = tuple(
            torch.randn(shape, device=device, dtype=torch.float64, requires_grad=True)
            for shape in [(2, 128), (2, 9), (2, product.weight_numel)]
        )
        s = torch.tensor([0, 1], device=device)
        t = torch.tensor([1, 1], device=device)

        def fn(*xs):
            return product(*xs, indices_1=s, indices_out=t, size_out=2)

        assert torch.autograd.gradcheck(fn, args, fast_mode=True)
        assert torch.autograd.gradgradcheck(fn, args, fast_mode=True)

    @pytest.mark.slow
    def test_compile_and_cuda_graph(self, irreps, cuda):
        from molix.compile import Compiler

        native = ChannelWiseTensorProduct(*irreps, dtype=torch.float64, device=cuda)
        args = tuple(
            torch.randn(shape, device=cuda, dtype=torch.float64, requires_grad=True)
            for shape in [(4, 128), (8, 9), (8, native.weight_numel)]
        )
        s = torch.arange(8, device=cuda) % 4
        t = (s + 1) % 4

        def fn(*xs):
            out = native(*xs, indices_1=s, indices_out=t, size_out=4)
            grad = torch.autograd.grad(out.square().sum(), xs, create_graph=True)
            return sum(g.square().sum() for g in grad)

        explain = Compiler.explain(fn, *args)
        assert explain.graph_count == 1 and explain.graph_break_count == 0
        compiled = Compiler(fullgraph=True, dynamic=False)(fn)
        actual = compiled(*args)
        expected = fn(*args)
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-9)
        ag = torch.autograd.grad(actual, args)
        eg = torch.autograd.grad(expected, args)
        for x, y in zip(ag, eg):
            torch.testing.assert_close(x, y, atol=1e-6, rtol=1e-8)
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                compiled(*args)
        torch.cuda.current_stream().wait_stream(stream)
        with torch.cuda.graph(graph):
            out = compiled(*args)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out, expected, atol=1e-7, rtol=1e-9)
        # A neighbor update changes values, not pointers. Replay must read the
        # new targets rather than reuse a topology cache keyed by storage.
        t.copy_((t + 2) % 4)
        graph.replay()
        changed = fn(*args)
        torch.cuda.synchronize()
        torch.testing.assert_close(out, changed, atol=1e-7, rtol=1e-9)
