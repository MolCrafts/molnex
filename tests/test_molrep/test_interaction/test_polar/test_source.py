"""Tests for EnvironmentDependentSpinSourceBlock."""

from __future__ import annotations

import cuequivariance as cue
import cuequivariance_torch as cuet
import pytest
import torch

from molix import config
from molrep.interaction.linear import EquivariantLinear
from molrep.interaction.polar.source import EnvironmentDependentSpinSourceBlock


@pytest.fixture(autouse=True)
def fp64():
    previous = config["ftype"]
    config.set_precision("fp64")
    yield
    config.set_precision("fp64" if previous == torch.float64 else "fp32")


class TestEnvironmentDependentSpinSourceBlock:
    """Test the Polar spin-resolved multipole source."""

    def test_forward_view_is_two_y_copies(self):
        """``l_max=1``: flat 8 → ``(N, 2, 4)`` with monopoles at ``[:, :, 0]``."""
        torch.manual_seed(0)
        block = EnvironmentDependentSpinSourceBlock(irreps_in="4x0e+4x1o", l_max=1)
        n = 3
        x = torch.randn(n, cue.Irreps("O3", "4x0e+4x1o").dim, dtype=torch.float64)
        out = block(x)
        assert out.shape == (n, 2, 4)
        flat = block.linear(x)
        assert flat.shape == (n, 8)
        assert torch.equal(out[:, 0, 0], flat[:, 0])
        assert torch.equal(out[:, 1, 0], flat[:, 4])

    def test_linear_is_bare_cuet_not_equivariant_wrapper(self):
        """Checkpoint key is ``linear.weight``, not ``linear.linear.weight``."""
        block = EnvironmentDependentSpinSourceBlock(irreps_in="4x0e+4x1o", l_max=1)
        assert isinstance(block.linear, cuet.Linear)
        assert not isinstance(block.linear, EquivariantLinear)
        keys = list(block.state_dict())
        assert any(k == "linear.weight" or k.endswith("linear.weight") for k in keys)
        assert not any(k.startswith("linear.linear") for k in keys)

    def test_zero_charges_zeros_monopoles_only(self):
        """Upstream ``zero_charges``: 0e slice exact 0, dipoles may stay."""
        torch.manual_seed(1)
        block = EnvironmentDependentSpinSourceBlock(
            irreps_in="4x0e+4x1o", l_max=1, zero_charges=True
        )
        x = torch.randn(2, cue.Irreps("O3", "4x0e+4x1o").dim, dtype=torch.float64)
        out = block(x)
        assert out.shape == (2, 2, 4)
        assert torch.equal(out[..., 0], torch.zeros(2, 2, dtype=out.dtype))

    def test_y_dim_is_l_plus_one_squared(self):
        """``dim Y_l = (l+1)^2``; two copies → ``2*(l+1)^2`` flat width."""
        for l_max, y_dim, flat in ((0, 1, 2), (1, 4, 8), (2, 9, 18)):
            block = EnvironmentDependentSpinSourceBlock(irreps_in="2x0e", l_max=l_max)
            x = torch.zeros(1, cue.Irreps("O3", "2x0e").dim, dtype=torch.float64)
            out = block(x)
            assert out.shape[-1] == y_dim
            assert block.linear(x).shape[-1] == flat
