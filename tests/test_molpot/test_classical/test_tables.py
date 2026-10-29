import pytest
import torch

from molpot.classical.tables import ParameterMap


class TestParameterMap:
    def test_conformers_share_and_sum_gradients(self):
        mapping = ParameterMap({"bonds": torch.tensor([0, 1, 0, 1])}, {"bonds": 2})
        k = torch.tensor([2.0, 3.0], requires_grad=True)
        out = mapping({"bonds": {"k": k}})["bonds"]["k"]
        (out * torch.tensor([1.0, 2.0, 3.0, 4.0])).sum().backward()
        torch.testing.assert_close(k.grad, torch.tensor([4.0, 6.0]))

    def test_out_of_bounds_rejected_before_compile(self):
        with pytest.raises(ValueError):
            ParameterMap({"atoms": torch.tensor([2])}, {"atoms": 2})

    @pytest.mark.slow
    def test_single_graph(self):
        from molix.compile import Compiler

        mapping = ParameterMap({"atoms": torch.tensor([1, 0, 1])}, {"atoms": 2})
        report = Compiler.explain(mapping, {"atoms": {"charge": torch.tensor([-0.5, 0.5])}})
        assert report.graph_count == 1 and report.graph_break_count == 0
