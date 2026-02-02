"""Unit tests for keyed MLP module."""

import pytest
import torch
from tensordict import TensorDict

from molix.nn.mlp import KeyedMLP, KeyedMLPSpec


class TestKeyedMLPSpec:
    """Test KeyedMLPSpec configuration."""

    def test_spec_basic(self):
        spec = KeyedMLPSpec(
            input_key="edge_rbf",
            output_key="edge_weights",
            in_dim=8,
            hidden_dims=[64, 64],
            out_dim=128,
            activation="silu",
            use_bias=True,
        )

        assert spec.input_key == "edge_rbf"
        assert spec.output_key == "edge_weights"
        assert spec.in_dim == 8
        assert spec.hidden_dims == [64, 64]
        assert spec.out_dim == 128
        assert spec.activation == "silu"
        assert spec.use_bias is True

    def test_spec_tuple_keys(self):
        spec = KeyedMLPSpec(
            input_key=("edge", "rbf"),
            output_key=("edge", "weights"),
            in_dim=16,
            hidden_dims=[32],
            out_dim=64,
        )

        assert spec.input_key == ("edge", "rbf")
        assert spec.output_key == ("edge", "weights")

    def test_spec_key_property(self):
        spec = KeyedMLPSpec(
            input_key="rbf",
            output_key="weights",
            in_dim=8,
            hidden_dims=[32],
            out_dim=64,
        )

        assert spec.key == "rbf"

    def test_spec_multiple_hidden_layers(self):
        spec = KeyedMLPSpec(
            input_key="rbf",
            output_key="weights",
            in_dim=8,
            hidden_dims=[64, 64, 64],
            out_dim=128,
        )

        assert len(spec.hidden_dims) == 3

    def test_spec_activation_validation(self):
        for act in ["silu", "relu", "gelu", "tanh"]:
            spec = KeyedMLPSpec(
                input_key="rbf",
                output_key="weights",
                in_dim=8,
                hidden_dims=[32],
                out_dim=64,
                activation=act,
            )
            assert spec.activation == act

        with pytest.raises(Exception):
            KeyedMLPSpec(
                input_key="rbf",
                output_key="weights",
                in_dim=8,
                hidden_dims=[32],
                out_dim=64,
                activation="invalid",
            )

    def test_spec_dimension_validation(self):
        KeyedMLPSpec(
            input_key="rbf",
            output_key="weights",
            in_dim=1,
            hidden_dims=[1],
            out_dim=1,
        )

        with pytest.raises(Exception):
            KeyedMLPSpec(
                input_key="rbf",
                output_key="weights",
                in_dim=0,
                hidden_dims=[32],
                out_dim=64,
            )

    def test_spec_hidden_dims_not_empty(self):
        with pytest.raises(Exception):
            KeyedMLPSpec(
                input_key="rbf",
                output_key="weights",
                in_dim=8,
                hidden_dims=[],
                out_dim=64,
            )

    def test_spec_serialization(self):
        spec = KeyedMLPSpec(
            input_key="edge_rbf",
            output_key="edge_weights",
            in_dim=8,
            hidden_dims=[64, 64],
            out_dim=128,
            activation="silu",
            use_bias=True,
        )

        config = spec.model_dump()
        spec_restored = KeyedMLPSpec(**config)

        assert spec_restored.input_key == spec.input_key
        assert spec_restored.output_key == spec.output_key
        assert spec_restored.in_dim == spec.in_dim
        assert spec_restored.hidden_dims == spec.hidden_dims
        assert spec_restored.out_dim == spec.out_dim
        assert spec_restored.activation == spec.activation
        assert spec_restored.use_bias == spec.use_bias


class TestKeyedMLP:
    """Test KeyedMLP module."""

    def test_module_initialization(self):
        mlp = KeyedMLP(
            input_key="rbf",
            output_key="weights",
            in_dim=8,
            hidden_dims=[64, 64],
            out_dim=128,
        )

        assert mlp.config.in_dim == 8
        assert mlp.config.hidden_dims == [64, 64]
        assert mlp.config.out_dim == 128
        assert mlp.in_keys == ["rbf"]
        assert mlp.out_keys == ["weights"]

    def test_forward_basic(self):
        mlp = KeyedMLP(
            input_key="rbf",
            output_key="weights",
            in_dim=8,
            hidden_dims=[32, 32],
            out_dim=64,
        )

        features = torch.randn(100, 8)
        td = TensorDict({"rbf": features}, batch_size=[100])

        td_out = mlp(td)

        assert "weights" in td_out
        assert td_out["weights"].shape == (100, 64)

    def test_forward_single_hidden_layer(self):
        mlp = KeyedMLP(
            input_key="rbf",
            output_key="weights",
            in_dim=8,
            hidden_dims=[32],
            out_dim=64,
        )

        features = torch.randn(50, 8)
        td = TensorDict({"rbf": features}, batch_size=[50])

        td_out = mlp(td)

        assert td_out["weights"].shape == (50, 64)

    def test_forward_multiple_hidden_layers(self):
        mlp = KeyedMLP(
            input_key="rbf",
            output_key="weights",
            in_dim=8,
            hidden_dims=[64, 64, 64],
            out_dim=128,
        )

        features = torch.randn(30, 8)
        td = TensorDict({"rbf": features}, batch_size=[30])

        td_out = mlp(td)

        assert td_out["weights"].shape == (30, 128)

    def test_forward_different_activations(self):
        for activation in ["silu", "relu", "gelu", "tanh"]:
            mlp = KeyedMLP(
                input_key="rbf",
                output_key="weights",
                in_dim=8,
                hidden_dims=[32],
                out_dim=64,
                activation=activation,
            )

            features = torch.randn(10, 8)
            td = TensorDict({"rbf": features}, batch_size=[10])

            td_out = mlp(td)

            assert td_out["weights"].shape == (10, 64)
            assert torch.all(torch.isfinite(td_out["weights"]))

    def test_forward_no_bias(self):
        mlp = KeyedMLP(
            input_key="rbf",
            output_key="weights",
            in_dim=8,
            hidden_dims=[32],
            out_dim=64,
            use_bias=False,
        )

        features = torch.randn(20, 8)
        td = TensorDict({"rbf": features}, batch_size=[20])

        td_out = mlp(td)

        assert td_out["weights"].shape == (20, 64)

    def test_forward_tuple_keys(self):
        mlp = KeyedMLP(
            input_key=("edge", "rbf"),
            output_key=("edge", "weights"),
            in_dim=8,
            hidden_dims=[32],
            out_dim=64,
        )

        features = torch.randn(15, 8)
        td = TensorDict({("edge", "rbf"): features}, batch_size=[15])

        td_out = mlp(td)

        assert ("edge", "weights") in td_out
        assert td_out[("edge", "weights")].shape == (15, 64)

    def test_forward_batched(self):
        mlp = KeyedMLP(
            input_key="rbf",
            output_key="weights",
            in_dim=8,
            hidden_dims=[32],
            out_dim=64,
        )

        features = torch.randn(5, 20, 8)
        td = TensorDict({"rbf": features}, batch_size=[5, 20])

        td_out = mlp(td)

        assert td_out["weights"].shape == (5, 20, 64)

    def test_gradient_flow(self):
        mlp = KeyedMLP(
            input_key="rbf",
            output_key="weights",
            in_dim=8,
            hidden_dims=[32],
            out_dim=64,
        )

        features = torch.randn(10, 8, requires_grad=True)
        td = TensorDict({"rbf": features}, batch_size=[10])

        td_out = mlp(td)
        loss = td_out["weights"].sum()
        loss.backward()

        assert features.grad is not None
        assert torch.any(features.grad != 0)

    def test_serialization_roundtrip(self):
        mlp = KeyedMLP(
            input_key="edge_rbf",
            output_key="edge_weights",
            in_dim=8,
            hidden_dims=[64, 64],
            out_dim=128,
            activation="silu",
            use_bias=True,
        )

        config = mlp.config.model_dump()
        mlp_restored = KeyedMLP(**config)

        features = torch.randn(10, 8)
        td = TensorDict({"edge_rbf": features}, batch_size=[10])

        with torch.no_grad():
            out1 = mlp(td.clone())
            out2 = mlp_restored(td.clone())

        assert out1["edge_weights"].shape == out2["edge_weights"].shape

    def test_missing_input_key_error(self):
        mlp = KeyedMLP(
            input_key="rbf",
            output_key="weights",
            in_dim=8,
            hidden_dims=[32],
            out_dim=64,
        )

        td = TensorDict({"wrong_key": torch.randn(10, 8)}, batch_size=[10])

        with pytest.raises(KeyError):
            mlp(td)

    def test_wrong_input_shape(self):
        mlp = KeyedMLP(
            input_key="rbf",
            output_key="weights",
            in_dim=8,
            hidden_dims=[32],
            out_dim=64,
        )

        features = torch.randn(10, 6)
        td = TensorDict({"rbf": features}, batch_size=[10])

        with pytest.raises(RuntimeError):
            mlp(td)

    def test_output_values_reasonable(self):
        mlp = KeyedMLP(
            input_key="rbf",
            output_key="weights",
            in_dim=8,
            hidden_dims=[32, 32],
            out_dim=64,
        )

        features = torch.randn(10, 8)
        td = TensorDict({"rbf": features}, batch_size=[10])

        td_out = mlp(td)

        assert torch.all(torch.isfinite(td_out["weights"]))
