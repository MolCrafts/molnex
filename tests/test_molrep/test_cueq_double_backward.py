"""Force-supervised double-backward through the molrep l>0 cuet path.

The energy must depend on |r| (``normalize=False`` on the spherical
harmonics). A single normalised direction has rotation invariants that are
constant, so ``∂E/∂x`` is identically zero and a zero weight-grad is
degenerate, not a severed graph — see spec ``cuet-force-doublebackward``
Findings run 2.
"""

from collections.abc import Iterator

import cuequivariance as cue
import pytest
import torch

from molix import config
from molrep.embedding.angular import SphericalHarmonics
from molrep.interaction.linear import EquivariantLinear


@pytest.fixture
def fp64() -> Iterator[None]:
    """Run the case under global fp64, restoring the previous precision.

    ``config.set_precision`` is process-global: leaking fp64 out of this test
    builds fp64 buffers inside every later module in the same worker, which
    then meets fp32 inputs.
    """
    previous = config["ftype"]
    config.set_precision("fp64")
    yield
    config.set_precision("fp64" if previous == torch.float64 else "fp32")


def test_force_loss_reaches_linear_weights(fp64: None) -> None:
    """g = ∂E/∂x with create_graph; backprop g² must hit cuet.Linear weights."""
    torch.manual_seed(0)
    sh = SphericalHarmonics(l_max=2, normalize=False)
    linear = EquivariantLinear(irreps_in="1x0e+1x1o+1x2e", irreps_out="4x0e+4x1o+4x2e")
    pos = torch.randn(8, 3, dtype=torch.float64, requires_grad=True)
    energy = linear(sh(pos)).pow(2).sum()
    force = torch.autograd.grad(energy, pos, create_graph=True)[0]
    assert force.detach().abs().max() > 0.0
    linear.zero_grad(set_to_none=True)
    (force**2).mean().backward()
    weight_l1 = sum(
        float(param.grad.abs().sum()) for param in linear.parameters() if param.grad is not None
    )
    assert weight_l1 > 0.0, "force loss did not reach the l>0 cuet.Linear weights"
    assert cue.Irreps("O3", "1x0e+1x1o+1x2e").dim == 9
