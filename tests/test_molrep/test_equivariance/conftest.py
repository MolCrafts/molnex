"""Devices, small channelwise fixtures and an independent torch contraction.

Every operator test runs on the CPU (the :mod:`molrep.equivariance.reference`
path) and, when a CUDA device and the built library are present, on CUDA (the
kernels), against the same independent per-term replay.
"""

import pytest
import torch

import molrep.eq as meq
from molrep.equivariance import ChannelWiseTensorProduct
from molrep.equivariance.backend import NativeBackend


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("no CUDA device")
        if NativeBackend.library_path() is None:
            pytest.skip("native equivariance library not built")
    return torch.device(request.param)


@pytest.fixture
def cuda():
    """A CUDA device with the library, for what only the kernels have (fused adjoints)."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    if NativeBackend.library_path() is None:
        pytest.skip("native equivariance library not built")
    return torch.device("cuda")


@pytest.fixture
def irreps():
    return (
        meq.Irreps("O3", "32x0e+32x1o"),
        meq.Irreps("O3", "0e+1o+2e"),
        meq.Irreps("O3", "32x0e+32x1o+32x2e"),
    )


@pytest.fixture
def product(irreps, device):
    return ChannelWiseTensorProduct(*irreps, dtype=torch.float64, device=device)


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
