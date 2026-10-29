"""Tests for molrep.embedding.mlp module."""

import math

import pytest
import torch
import torch.nn.functional as F

from molrep.embedding.mlp import (
    DenseMLP,
    FFLayer,
    MomentNormalizedMLP,
    ScalarLinearLayer,
    ScalarMLPFunction,
    normalize2mom,
)

#: Floating dtype every module under test is built at.
DTYPE = torch.float32


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
    def test_first_layer_split_reproduces_forward(self):
        """Layer 0 evaluated from a partition of its inputs is the same function.

        The bias-free counterpart of ``RadialMLP.first_layer_parts``; the MACE
        interactions use whichever of the two their radial MLP is.
        """
        torch.manual_seed(0)
        mlp = MomentNormalizedMLP([12, 8, 5], dtype=DTYPE).double()
        x = torch.randn(6, 12, dtype=torch.float64)
        widths = [4, 5, 3]
        parts = mlp.first_layer_parts(widths)
        hidden = sum(p @ w for p, w in zip(x.split(widths, dim=-1), parts, strict=True))
        assert mlp.first_layer_bias() is None
        torch.testing.assert_close(mlp.forward_from_first(hidden), mlp(x), atol=1e-12, rtol=0)

    """Test the e3nn-compatible scalar MLP."""

    def test_output_shape(self):
        """Maps the last dimension from ``channels[0]`` to ``channels[-1]``."""
        mlp = MomentNormalizedMLP([10, 64, 512], dtype=DTYPE)
        assert mlp(torch.randn(7, 10)).shape == (7, 512)

    def test_each_layer_holds_its_weight_in_a_contained_linear(self):
        """``layerN.linear.weight`` ``(out, in)``: upstream's ``layerN.weight``
        ``(in, out)`` lands there through the loader's declared layout."""
        mlp = MomentNormalizedMLP([10, 64, 64, 512], dtype=DTYPE)
        names = {n for n, _ in mlp.named_parameters()}
        assert names == {f"layer{i}.linear.weight" for i in range(3)}
        assert set(mlp.state_dict()) == names
        assert mlp.layer0.linear.weight.shape == (64, 10)
        assert MomentNormalizedMLP._Layer.upstream_linear_layout == "in_out"

    def test_a_seeded_build_is_upstreams_draw_transposed(self):
        """The weight is upstream's ``randn(in, out)`` transposed in — no extra
        RNG draw — so every seeded fixture keeps the model it had."""
        torch.manual_seed(0)
        mlp = MomentNormalizedMLP([4, 8], dtype=DTYPE)
        torch.manual_seed(0)
        want = torch.randn(4, 8, dtype=mlp.layer0.linear.weight.dtype).t()
        assert torch.equal(mlp.layer0.linear.weight, want)

    def test_has_no_biases(self):
        """e3nn's FullyConnectedNet is bias-free; a bias would break transfer."""
        assert not any(
            "bias" in n for n, _ in MomentNormalizedMLP([4, 8, 2], dtype=DTYPE).named_parameters()
        )

    def test_maps_zero_to_zero(self):
        """Bias-free and SiLU(0)=0, so a dead edge contributes nothing."""
        mlp = MomentNormalizedMLP([6, 12, 3], dtype=DTYPE)
        assert float(mlp(torch.zeros(2, 6)).abs().max()) == 0.0

    def test_single_layer_is_a_bare_scaled_linear(self):
        """With two channel entries there is no hidden layer and no activation."""
        mlp = MomentNormalizedMLP([4, 1], dtype=DTYPE).double()
        x = torch.randn(3, 4, dtype=torch.float64)
        want = F.linear(x, mlp.layer0.linear.weight) * (1.0 / math.sqrt(4))
        assert torch.allclose(mlp(x), want, atol=1e-12)

    def test_weight_scaling_is_one_over_sqrt_fan_in(self):
        """The ``1/√fan_in`` factor is applied at forward, not baked into W."""
        mlp = MomentNormalizedMLP([9, 1], dtype=DTYPE).double()
        with torch.no_grad():
            mlp.layer0.linear.weight.fill_(1.0)
        x = torch.ones(1, 9, dtype=torch.float64)
        assert float(mlp(x)) == pytest.approx(9.0 / 3.0, abs=1e-12)

    def test_rejects_too_few_channels(self):
        """A channel list needs at least an input and an output width."""
        with pytest.raises(ValueError, match="channels"):
            MomentNormalizedMLP([8], dtype=DTYPE)

    def test_is_differentiable(self):
        """It generates tensor-product weights, so gradients must flow."""
        mlp = MomentNormalizedMLP([4, 8, 2], dtype=DTYPE)
        x = torch.randn(3, 4, requires_grad=True)
        mlp(x).sum().backward()
        assert x.grad is not None and float(x.grad.abs().sum()) > 0.0


class TestScalarMLPFunction:
    def test_activation_name_equals_the_module_form(self):
        """``"silu"`` resolves to ``nn.SiLU`` exactly as ``FFLayer`` does."""
        torch.manual_seed(0)
        by_name = ScalarMLPFunction(
            6, 3, hidden_layers_depth=1, hidden_layers_width=4, nonlinearity="silu", dtype=DTYPE
        )
        torch.manual_seed(0)
        by_module = ScalarMLPFunction(
            6,
            3,
            hidden_layers_depth=1,
            hidden_layers_width=4,
            nonlinearity=torch.nn.SiLU,
            dtype=DTYPE,
        )
        x = torch.randn(5, 6, dtype=by_name.mlp[0].linear.weight.dtype)
        torch.testing.assert_close(by_name(x), by_module(x))


class TestScalarLinearLayer:
    def test_forward_scales_the_contained_linear_and_adds_the_bias_unscaled(self):
        """``linear(x)·α + b``: nequip scales the weight by α, never the bias."""
        torch.manual_seed(0)
        layer = ScalarLinearLayer(4, 3, alpha=0.5, bias=True, dtype=DTYPE).double()
        with torch.no_grad():
            layer.bias.normal_()
        x = torch.randn(6, 4, dtype=torch.float64)
        want = F.linear(x, layer.linear.weight) * 0.5 + layer.bias
        torch.testing.assert_close(layer(x), want, atol=1e-12, rtol=0)

    def test_weight_is_upstreams_uniform_draw_transposed(self):
        """``U(-√3, √3)`` drawn in upstream's ``(in, out)`` shape and transposed in:
        an ``(out, in)`` weight, a zero bias, and a seeded build equal to the old layout."""
        torch.manual_seed(0)
        layer = ScalarLinearLayer(32, 16, bias=True, dtype=DTYPE)
        torch.manual_seed(0)
        want = torch.empty(32, 16).uniform_(-math.sqrt(3), math.sqrt(3)).t()
        assert layer.linear.weight.shape == (16, 32)
        assert torch.equal(layer.linear.weight, want)
        assert torch.equal(layer.bias, torch.zeros(16))
        assert ScalarLinearLayer.upstream_linear_layout == "in_out"

    def test_state_dict_keys_are_linear_weight_and_bias(self):
        """The weight moved into the contained linear; the bias kept its key."""
        assert set(ScalarLinearLayer(4, 3, bias=True, dtype=DTYPE).state_dict()) == {
            "linear.weight",
            "bias",
        }
        assert set(ScalarLinearLayer(4, 3, dtype=DTYPE).state_dict()) == {"linear.weight"}


class TestFFLayer:
    def test_widths_and_output_dim(self):
        layer = FFLayer([8, 4], in_dim=6, activation="tanh", dtype=DTYPE)
        assert layer(torch.randn(5, 6, dtype=layer.layers[0].weight.dtype)).shape == (5, 4)
        assert layer.output_dim == 4

    def test_default_is_linear_like_keras_dense(self):
        """PiNN's ``Dense`` has no activation unless one is passed."""
        layer = FFLayer([3], in_dim=2, dtype=DTYPE)
        x = torch.randn(4, 2, dtype=layer.layers[0].weight.dtype)
        torch.testing.assert_close(layer(2 * x) - layer(0 * x), 2 * (layer(x) - layer(0 * x)))

    def test_empty_stack_is_identity(self):
        x = torch.randn(2, 3)
        torch.testing.assert_close(FFLayer([], in_dim=3, dtype=DTYPE)(x), x)


class TestDenseMLP:
    def test_the_activation_sits_between_the_linears_and_not_after_the_last(self):
        mlp = DenseMLP(3, 4, 2, 3, dtype=torch.float64)
        linears = [m for m in mlp.modules() if isinstance(m, torch.nn.Linear)]
        assert [(m.in_features, m.out_features) for m in linears] == [(3, 4), (4, 4), (4, 2)]
        assert isinstance(mlp[1], torch.nn.Linear)

    def test_one_layer_is_a_single_linear(self):
        mlp = DenseMLP(3, 8, 2, 1, bias=False, dtype=torch.float64)
        linears = [m for m in mlp.modules() if isinstance(m, torch.nn.Linear)]
        assert len(linears) == 1 and linears[0].bias is None
