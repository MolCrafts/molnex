import pytest
import torch

from molpot.pooling import MaxPooling

from tests.utils import assert_compile_compatible


class TestMaxPooling:
    def test_max_pooling(self):
        pooling = MaxPooling()
        x = torch.tensor([[1.0, 2.0], [3.0, 1.0], [10.0, 20.0]])
        batch = torch.tensor([0, 0, 1])
        out = pooling(x, batch)
        assert torch.allclose(out, torch.tensor([[3.0, 2.0], [10.0, 20.0]]))

    @pytest.mark.xfail(reason="pooling uses scatter_reduce which can break graph", strict=False)
    def test_compile(self):
        pooling = MaxPooling()
        x = torch.tensor([[1.0, 2.0], [3.0, 1.0], [10.0, 20.0]])
        batch = torch.tensor([0, 0, 1])
        assert_compile_compatible(pooling, x, batch, strict=False)
