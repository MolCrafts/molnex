import pytest
import torch

from molpot.pooling import SumPooling

from tests.utils import assert_compile_compatible


class TestSumPooling:
    def test_sum_pooling(self):
        pooling = SumPooling()
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [10.0, 20.0]])
        batch = torch.tensor([0, 0, 1])
        out = pooling(x, batch)
        assert torch.allclose(out, torch.tensor([[4.0, 6.0], [10.0, 20.0]]))

    @pytest.mark.xfail(reason="pooling uses scatter/index_add which can break graph", strict=False)
    def test_compile(self):
        pooling = SumPooling()
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [10.0, 20.0]])
        batch = torch.tensor([0, 0, 1])
        assert_compile_compatible(pooling, x, batch, strict=False)
