"""Unit tests for KeyedMLP with dict-based inputs."""

import pytest
import torch

from molix.nn.mlp import KeyedMLP, KeyedMLPSpec


class TestKeyedMLPSpec:
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

    def test_spec_validation(self):
        with pytest.raises(Exception):
            KeyedMLPSpec(
                input_key="rbf",
                output_key="weights",
                in_dim=0,
                hidden_dims=[32],
                out_dim=64,
            )

        with pytest.raises(Exception):
            KeyedMLPSpec(
                input_key="rbf",
                output_key="weights",
                in_dim=8,
                hidden_dims=[],
                out_dim=64,
            )


class TestKeyedMLP:
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

    def test_forward_basic_dict(self):
        mlp = KeyedMLP(
            input_key="rbf",
            output_key="weights",
            in_dim=8,
            hidden_dims=[32, 32],
            out_dim=64,
        )

        batch = {"rbf": torch.randn(100, 8)}
        out = mlp(batch)

        assert "weights" in out
        assert out["weights"].shape == (100, 64)

    def test_forward_tuple_key(self):
        mlp = KeyedMLP(
            input_key=("edge", "rbf"),
            output_key=("edge", "weights"),
            in_dim=8,
            hidden_dims=[16],
            out_dim=4,
        )

        batch = {("edge", "rbf"): torch.randn(15, 8)}
        out = mlp(batch)

        assert ("edge", "weights") in out
        assert out[("edge", "weights")].shape == (15, 4)

    def test_forward_activations(self):
        for activation in ["silu", "relu", "gelu", "tanh"]:
            mlp = KeyedMLP(
                input_key="rbf",
                output_key="weights",
                in_dim=8,
                hidden_dims=[32],
                out_dim=64,
                activation=activation,
            )
            out = mlp({"rbf": torch.randn(10, 8)})
            assert out["weights"].shape == (10, 64)
            assert torch.all(torch.isfinite(out["weights"]))

    def test_gradient_flow(self):
        mlp = KeyedMLP(
            input_key="rbf",
            output_key="weights",
            in_dim=8,
            hidden_dims=[32],
            out_dim=64,
        )

        x = torch.randn(10, 8, requires_grad=True)
        out = mlp({"rbf": x})
        loss = out["weights"].sum()
        loss.backward()

        assert x.grad is not None
        assert torch.any(x.grad != 0)

    def test_missing_input_key_error(self):
        mlp = KeyedMLP(
            input_key="rbf",
            output_key="weights",
            in_dim=8,
            hidden_dims=[32],
            out_dim=64,
        )

        with pytest.raises(KeyError):
            mlp({"wrong": torch.randn(10, 8)})

    def test_wrong_input_shape(self):
        mlp = KeyedMLP(
            input_key="rbf",
            output_key="weights",
            in_dim=8,
            hidden_dims=[32],
            out_dim=64,
        )

        with pytest.raises(RuntimeError):
            mlp({"rbf": torch.randn(10, 6)})

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

        restored = KeyedMLP(**mlp.config.model_dump())
        sample = {"edge_rbf": torch.randn(10, 8)}

        with torch.no_grad():
            out1 = mlp(dict(sample))
            out2 = restored(dict(sample))

        assert out1["edge_weights"].shape == out2["edge_weights"].shape
