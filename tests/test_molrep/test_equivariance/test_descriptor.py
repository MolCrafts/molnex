import cuequivariance as cue
import cuequivariance_torch as cuet
import pytest
import torch

from molrep.equivariance.descriptor import ChannelwiseDescriptor, natural_irreps


class TestChannelwiseDescriptor:
    @pytest.mark.parametrize("l1,l2", [(0, 2), (1, 2), (1, 3), (3, 3)])
    def test_term_table_reproduces_upstream(self, l1, l2):
        irreps = (natural_irreps(32, l1), natural_irreps(1, l2), natural_irreps(32, l2))
        descriptor = ChannelwiseDescriptor(*irreps)
        reference = cuet.ChannelWiseTensorProduct(
            *irreps,
            layout=cue.ir_mul,
            shared_weights=False,
            internal_weights=False,
            dtype=torch.float64,
            method="naive",
        )
        assert descriptor.weight_numel == reference.weight_numel
        assert descriptor.irreps_out == reference.irreps_out
        assert (descriptor.l1, descriptor.l2, descriptor.channels) == (l1, l2, 32)
        assert len(descriptor.terms) == len(descriptor.coefficients)
        # Every term of the compiled table against upstream's dense evaluation,
        # including the ir_mul layout and the path / segment order.
        u = 32
        a = torch.randn(5, descriptor.input_dims[0], dtype=torch.float64)
        b = torch.randn(5, descriptor.input_dims[1], dtype=torch.float64)
        w = torch.randn(5, descriptor.weight_numel, dtype=torch.float64)
        expected = reference(a, b, w)
        actual = torch.zeros_like(expected)
        for (i, j, p, k), coefficient in zip(
            descriptor.terms, descriptor.coefficients, strict=True
        ):
            actual[:, k * u : (k + 1) * u] += (
                coefficient
                * a[:, i * u : (i + 1) * u]
                * b[:, j : j + 1]
                * w[:, p * u : (p + 1) * u]
            )
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)

    def test_unsupported_multiplicity_is_rejected(self):
        with pytest.raises(ValueError, match="channels"):
            ChannelwiseDescriptor(natural_irreps(16, 1), natural_irreps(1, 2), None)

    def test_mixed_channel_multiplicities_are_rejected(self):
        with pytest.raises(ValueError, match="uniform"):
            ChannelwiseDescriptor(
                cue.Irreps("O3", "32x0e+64x1o"), natural_irreps(1, 2), natural_irreps(32, 2)
            )

    def test_non_natural_parity_is_rejected(self):
        with pytest.raises(ValueError, match="node irreps"):
            ChannelwiseDescriptor(
                cue.Irreps("O3", "32x0e+32x1e"), natural_irreps(1, 2), natural_irreps(32, 2)
            )
        with pytest.raises(ValueError, match="angular irreps"):
            ChannelwiseDescriptor(natural_irreps(32, 1), cue.Irreps("O3", "0e+1o"), None)

    def test_filter_beyond_l2_is_rejected(self):
        # No filter: cue adds 1o×1o→... up to l3 = 2 fine, but 1o×2e→3o is
        # outside the compiled output range.
        with pytest.raises(ValueError, match="filter"):
            ChannelwiseDescriptor(natural_irreps(32, 1), natural_irreps(1, 2), None)
