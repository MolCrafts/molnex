"""Tests for molrep.embedding.mlp module."""

import math

import pytest
import torch
import torch.nn.functional as F

from molrep.embedding.mlp import MomentNormalizedMLP, normalize2mom


class TestNormalize2Mom:
    """Test the activation second-moment normalisation constant."""

    def test_scaled_activation_has_unit_second_moment(self):
        """``E[(c·act(z))²] = 1`` for ``z ~ N(0, 1)`` — the defining property.

        Checked on an independent draw, so the tolerance covers the Monte-Carlo
        error of both the constant (1e6 samples) and this estimate.
        """
        constant = normalize2mom(F.silu)
        z = torch.randn(2_000_000, dtype=torch.float64)
        assert float((constant * F.silu(z)).pow(2).mean()) == pytest.approx(1.0, abs=1e-2)

    def test_is_deterministic(self):
        """A fixed draw means two processes agree — weights must transfer."""
        assert normalize2mom(F.silu) == normalize2mom(F.silu)

    def test_identity_activation_is_unscaled(self):
        """``E[z²] = 1`` already, so the identity needs no rescaling."""
        assert normalize2mom(lambda x: x) == pytest.approx(1.0, abs=1e-2)

    def test_larger_activation_gets_smaller_constant(self):
        """A activation with bigger output needs a smaller constant."""
        assert normalize2mom(lambda x: 2.0 * x) < normalize2mom(lambda x: x)


class TestMomentNormalizedMLP:
    """Test the e3nn-compatible scalar MLP."""

    def test_output_shape(self):
        """Maps the last dimension from ``channels[0]`` to ``channels[-1]``."""
        mlp = MomentNormalizedMLP([10, 64, 512])
        assert mlp(torch.randn(7, 10)).shape == (7, 512)

    def test_layer_names_mirror_the_reference(self):
        """Official weights transfer by direct copy, so names must match."""
        names = {n for n, _ in MomentNormalizedMLP([10, 64, 64, 512]).named_parameters()}
        assert names == {f"layer{i}.weight" for i in range(3)}

    def test_has_no_biases(self):
        """e3nn's FullyConnectedNet is bias-free; a bias would break transfer."""
        assert not any("bias" in n for n, _ in MomentNormalizedMLP([4, 8, 2]).named_parameters())

    def test_maps_zero_to_zero(self):
        """Bias-free and SiLU(0)=0, so a dead edge contributes nothing."""
        mlp = MomentNormalizedMLP([6, 12, 3])
        assert float(mlp(torch.zeros(2, 6)).abs().max()) == 0.0

    def test_single_layer_is_a_bare_scaled_linear(self):
        """With two channel entries there is no hidden layer and no activation."""
        mlp = MomentNormalizedMLP([4, 1]).double()
        x = torch.randn(3, 4, dtype=torch.float64)
        want = x @ (mlp.layer0.weight * (1.0 / math.sqrt(4)))
        assert torch.allclose(mlp(x), want, atol=1e-12)

    def test_weight_scaling_is_one_over_sqrt_fan_in(self):
        """The ``1/√fan_in`` factor is applied at forward, not baked into W."""
        mlp = MomentNormalizedMLP([9, 1]).double()
        with torch.no_grad():
            mlp.layer0.weight.fill_(1.0)
        x = torch.ones(1, 9, dtype=torch.float64)
        assert float(mlp(x)) == pytest.approx(9.0 / 3.0, abs=1e-12)

    def test_rejects_too_few_channels(self):
        """A channel list needs at least an input and an output width."""
        with pytest.raises(ValueError, match="channels"):
            MomentNormalizedMLP([8])

    def test_is_differentiable(self):
        """It generates tensor-product weights, so gradients must flow."""
        mlp = MomentNormalizedMLP([4, 8, 2])
        x = torch.randn(3, 4, requires_grad=True)
        mlp(x).sum().backward()
        assert x.grad is not None and float(x.grad.abs().sum()) > 0.0
