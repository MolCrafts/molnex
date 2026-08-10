"""Tests for molrep.interaction.product module."""

import torch

from molrep.interaction.product import EquivariantPolynomialTP


class TestEquivariantPolynomialTP:
    """Generic equivariant polynomial wrapper (model-specific descriptors live in molzoo)."""

    def test_channelwise_forward_shape(self):
        """Passing a prebuilt channelwise descriptor gives the same shapes as ConvTP."""
        import cuequivariance as cue

        irreps_in = cue.Irreps("O3", "8x0e + 8x1o")
        irreps_sh = cue.Irreps("O3", "1x0e + 1x1o")
        poly = cue.descriptors.channelwise_tensor_product(irreps_in, irreps_sh)
        tp = EquivariantPolynomialTP(
            poly, shared_weights=False, internal_weights=False, method="naive"
        )

        n_edges = 5
        lhs = torch.randn(n_edges, irreps_in.dim)
        rhs = torch.randn(n_edges, irreps_sh.dim)
        w = torch.randn(n_edges, tp.weight_numel)
        out = tp(lhs, rhs, weight=w)
        assert out.shape == (n_edges, tp.irreps_out.dim)

    def test_shared_weights_param(self):
        """shared_weights=True stores an internal (1, weight_numel) parameter."""
        import cuequivariance as cue

        irreps_in = cue.Irreps("O3", "4x0e + 4x1o")
        irreps_sh = cue.Irreps("O3", "1x0e + 1x1o")
        poly = cue.descriptors.channelwise_tensor_product(irreps_in, irreps_sh)

        tp = EquivariantPolynomialTP(
            poly, shared_weights=True, internal_weights=True, method="naive"
        )
        assert tp.weight is not None
        assert tp.weight.shape == (1, tp.weight_numel)
