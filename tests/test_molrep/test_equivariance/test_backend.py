import torch

from molrep.equivariance import ChannelWiseTensorProduct


class TestNativeBackend:
    def test_adjoint_reaches_output_cotangent(self, irreps, cuda):
        """The fused first-order adjoint is differentiable: its backward is the second order."""
        native = ChannelWiseTensorProduct(*irreps, dtype=torch.float64, device=cuda)
        args = [
            torch.randn(s, device=cuda, dtype=torch.float64, requires_grad=True)
            for s in [(3, 128), (4, 9), (4, native.weight_numel)]
        ]
        index = torch.tensor([0, 1, 2, 1], device=cuda)
        out = native(*args, indices_1=index, indices_out=index, size_out=3)
        cotangent = torch.randn_like(out, requires_grad=True)
        first = torch.autograd.grad(out, args, cotangent, create_graph=True)
        dc = torch.autograd.grad(sum(x.square().sum() for x in first), cotangent)[0]
        assert torch.isfinite(dc).all() and dc.abs().sum() > 0
