"""Tests for :mod:`molix.F.scatter`."""

from __future__ import annotations

import pytest
import torch

from molix.F.scatter import scatter_sum, scatter_sum_compile_safe


class TestScatterSum:
    """COO scatter-add along dim 0."""

    def test_matches_index_add(self) -> None:
        torch.manual_seed(0)
        src = torch.randn(7, 3, dtype=torch.float64)
        index = torch.tensor([0, 2, 1, 1, 0, 2, 1])
        dim_size = 3
        got = scatter_sum(src, index, dim=0, dim_size=dim_size)
        want = torch.zeros(dim_size, 3, dtype=torch.float64).index_add_(0, index, src)
        assert torch.allclose(got, want)

    def test_compile_safe_matches_index_add(self) -> None:
        torch.manual_seed(1)
        src = torch.randn(5, 4, dtype=torch.float64)
        index = torch.tensor([1, 0, 1, 2, 0])
        got = scatter_sum_compile_safe(src, index, 3)
        want = torch.zeros(3, 4, dtype=torch.float64).index_add_(0, index, src)
        assert torch.allclose(got, want)

    def test_empty_source(self) -> None:
        src = torch.zeros(0, 2, dtype=torch.float64)
        index = torch.zeros(0, dtype=torch.long)
        got = scatter_sum_compile_safe(src, index, 4)
        assert got.shape == (4, 2)
        assert torch.equal(got, torch.zeros(4, 2, dtype=torch.float64))

    def test_gradient_is_gather(self) -> None:
        torch.manual_seed(2)
        src = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
        index = torch.tensor([0, 1, 0, 1])
        scatter_sum_compile_safe(src, index, 2).sum().backward()
        src2 = src.detach().requires_grad_(True)
        torch.zeros(2, 2, dtype=torch.float64).index_add_(0, index, src2).sum().backward()
        assert src.grad is not None and src2.grad is not None
        assert torch.allclose(src.grad, src2.grad)

    def test_meta_kernel_does_not_call_numel(self) -> None:
        """``torch.compile`` traces ``molix::scatter_sum`` on meta tensors.

        The Meta kernel must use ``sym_sizes`` / ``empty_symint``, not
        ``numel()`` / ``sizes()``. Skip when the native op is not loaded.
        """
        if getattr(torch.ops.molix, "scatter_sum", None) is None:
            pytest.skip("molix::scatter_sum not registered")
        src = torch.empty(8, 3, device="meta")
        index = torch.empty(8, dtype=torch.long, device="meta")
        out = torch.ops.molix.scatter_sum(src, index, 4)
        assert tuple(out.shape) == (4, 3)
        assert out.device.type == "meta"

    def test_native_dim_size_is_symbolic(self) -> None:
        """An ``int`` here pins an exported program's atom count to the example's."""
        if getattr(torch.ops.molix, "scatter_sum", None) is None:
            pytest.skip("molix::scatter_sum not registered")
        dim_size = torch.ops.molix.scatter_sum.default._schema.arguments[2]
        assert isinstance(dim_size.real_type, torch.SymIntType)


@pytest.mark.gpu
class TestScatterSumCuda:
    """Native ``molix::scatter_sum`` on GPU."""

    def test_cuda_matches_index_add(self) -> None:
        torch.manual_seed(0)
        src = torch.randn(64, 8, device="cuda", dtype=torch.float32)
        index = torch.randint(0, 11, (64,), device="cuda")
        got = scatter_sum_compile_safe(src, index, 11)
        want = torch.zeros(11, 8, device="cuda", dtype=torch.float32).index_add_(0, index, src)
        assert torch.allclose(got, want, atol=1e-5, rtol=1e-5)

    def test_cuda_gradient_is_gather(self) -> None:
        torch.manual_seed(1)
        src = torch.randn(9, 3, device="cuda", dtype=torch.float32, requires_grad=True)
        index = torch.tensor([0, 2, 1, 1, 0, 2, 1, 0, 2], device="cuda")
        scatter_sum_compile_safe(src, index, 3).sum().backward()
        src2 = src.detach().requires_grad_(True)
        torch.zeros(3, 3, device="cuda").index_add_(0, index, src2).sum().backward()
        assert src.grad is not None and src2.grad is not None
        assert torch.allclose(src.grad, src2.grad, atol=1e-5, rtol=1e-5)

    def test_cuda_compile_matches_index_add(self) -> None:
        """Dynamo must use the Meta kernel, not trace into CUDA ``data_ptr()``."""
        torch.manual_seed(0)
        src = torch.randn(32, 4, device="cuda", dtype=torch.float32)
        index = torch.randint(0, 5, (32,), device="cuda")

        def fn(s: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
            return scatter_sum_compile_safe(s, i, 5)

        got = torch.compile(fn, fullgraph=False, dynamic=True)(src, index)
        want = torch.zeros(5, 4, device="cuda").index_add_(0, index, src)
        assert torch.allclose(got, want, atol=1e-5, rtol=1e-5)
