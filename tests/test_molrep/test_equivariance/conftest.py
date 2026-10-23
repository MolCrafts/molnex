"""Small indexed channelwise fixtures and an independent torch contraction."""

import cuequivariance as cue
import pytest
import torch

from molrep.equivariance import ChannelWiseTensorProduct


@pytest.fixture
def irreps():
    return (
        cue.Irreps("O3", "32x0e+32x1o"),
        cue.Irreps("O3", "0e+1o+2e"),
        cue.Irreps("O3", "32x0e+32x1o+32x2e"),
    )


@pytest.fixture
def native(irreps):
    if not torch.cuda.is_available():
        pytest.skip("native test requires a CUDA device")
    return ChannelWiseTensorProduct(
        *irreps,
        layout=cue.ir_mul,
        shared_weights=False,
        internal_weights=False,
        dtype=torch.float64,
        device="cuda",
    )


def torch_contraction(module, x1, x2, weight, source, target, size_out):
    """Explicit sparse sum, deliberately materializing edge messages as an oracle."""
    d = module.descriptor
    u, width = d.channels, module.irreps_out.dim
    result = x1.new_zeros((size_out, width))
    for (i, j, p, k), coefficient in zip(d.terms, d.coefficients, strict=True):
        values = (
            x1[source, i * u : (i + 1) * u]
            * x2[:, j : j + 1]
            * weight[:, p * u : (p + 1) * u]
            * coefficient
        )
        padded = torch.nn.functional.pad(values, (k * u, width - (k + 1) * u))
        result = result.index_add(0, target, padded)
    return result
