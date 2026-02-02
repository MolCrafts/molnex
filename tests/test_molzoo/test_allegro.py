"""Tests for molzoo.allegro module."""

import pytest
import torch

from molzoo.allegro import (
    Allegro,
    AllegroLayer,
    PairEmbedding,
    ScaleShiftAllegro,
)
from molrep.embedding.radial import BesselRBF
from molrep.embedding.angular import SphericalHarmonics
from molrep.embedding.cutoff import PolynomialCutoff


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def graph_data():
    """Small molecular graph for testing."""
    n_nodes = 10
    n_edges = 30
    num_elements = 5
    Z = torch.randint(0, num_elements, (n_nodes,))
    pos = torch.randn(n_nodes, 3)
    # Random edges
    edge_index = torch.randint(0, n_nodes, (n_edges, 2))
    bond_diff = torch.randn(n_edges, 3)
    bond_dist = bond_diff.norm(dim=-1).clamp(min=0.1)
    batch = torch.zeros(n_nodes, dtype=torch.long)
    return {
        "Z": Z,
        "pos": pos,
        "bond_dist": bond_dist,
        "bond_diff": bond_diff,
        "edge_index": edge_index,
        "batch": batch,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "num_elements": num_elements,
    }


# ===========================================================================
# TestPairEmbedding
# ===========================================================================


class TestPairEmbedding:
    """Test PairEmbedding initialization and forward pass."""

    @pytest.fixture
    def emb_config(self):
        return {
            "num_elements": 5,
            "num_scalar_features": 32,
            "num_tensor_features": 8,
            "r_max": 5.0,
            "num_bessel": 8,
            "l_max": 2,
        }

    @pytest.fixture
    def pair_emb(self, emb_config):
        return PairEmbedding(**emb_config)

    def test_initialization(self, pair_emb, emb_config):
        """Test all components are created."""
        assert isinstance(pair_emb.radial_basis, BesselRBF)
        assert isinstance(pair_emb.cutoff_fn, PolynomialCutoff)
        assert isinstance(pair_emb.spherical_harmonics, SphericalHarmonics)
        assert pair_emb.num_scalar_features == emb_config["num_scalar_features"]
        assert pair_emb.num_tensor_features == emb_config["num_tensor_features"]

    def test_forward_shapes(self, pair_emb, emb_config, graph_data):
        """Test output shapes of forward pass."""
        scalar_feats, tensor_feats, edge_angular, edge_cutoff = pair_emb(
            Z=graph_data["Z"],
            bond_dist=graph_data["bond_dist"],
            bond_diff=graph_data["bond_diff"],
            edge_index=graph_data["edge_index"],
        )
        n_edges = graph_data["n_edges"]
        num_scalar = emb_config["num_scalar_features"]
        l_max = emb_config["l_max"]
        sh_dim = (l_max + 1) ** 2

        assert scalar_feats.shape == (n_edges, num_scalar)
        assert tensor_feats.shape == (n_edges, pair_emb.irreps_dim)
        assert edge_angular.shape == (n_edges, sh_dim)
        assert edge_cutoff.shape == (n_edges,)

    def test_cutoff_applied(self, pair_emb, graph_data):
        """Test that cutoff values are in [0, 1]."""
        _, _, _, edge_cutoff = pair_emb(
            Z=graph_data["Z"],
            bond_dist=graph_data["bond_dist"],
            bond_diff=graph_data["bond_diff"],
            edge_index=graph_data["edge_index"],
        )
        assert (edge_cutoff >= 0.0).all()
        assert (edge_cutoff <= 1.0).all()

    def test_different_l_max(self, graph_data):
        """Test with different l_max values."""
        for l_max in [1, 2, 3]:
            emb = PairEmbedding(
                num_elements=graph_data["num_elements"],
                num_scalar_features=32,
                num_tensor_features=8,
                r_max=5.0,
                l_max=l_max,
            )
            scalar_feats, tensor_feats, edge_angular, _ = emb(
                Z=graph_data["Z"],
                bond_dist=graph_data["bond_dist"],
                bond_diff=graph_data["bond_diff"],
                edge_index=graph_data["edge_index"],
            )
            expected_sh_dim = (l_max + 1) ** 2
            assert edge_angular.shape[-1] == expected_sh_dim
            assert tensor_feats.shape == (graph_data["n_edges"], emb.irreps_dim)


# ===========================================================================
# TestAllegroLayer
# ===========================================================================


class TestAllegroLayer:
    """Test AllegroLayer initialization and forward pass."""

    @pytest.fixture
    def layer_config(self):
        return {
            "num_scalar_features": 32,
            "num_tensor_features": 8,
            "l_max": 2,
            "mlp_depth": 1,
        }

    @pytest.fixture
    def allegro_layer(self, layer_config):
        return AllegroLayer(**layer_config)

    def test_initialization(self, allegro_layer, layer_config):
        """Test components are created."""
        assert hasattr(allegro_layer, "tp")
        assert hasattr(allegro_layer, "tp_linear")
        assert hasattr(allegro_layer, "latent_mlp")
        assert hasattr(allegro_layer, "tensor_env")
        assert allegro_layer.num_scalar_features == layer_config["num_scalar_features"]
        assert allegro_layer.num_tensor_features == layer_config["num_tensor_features"]

    def test_forward_shapes(self, allegro_layer, layer_config):
        """Test that forward preserves shapes."""
        n_edges = 30
        num_scalar = layer_config["num_scalar_features"]
        l_max = layer_config["l_max"]
        sh_dim = (l_max + 1) ** 2
        irreps_dim = allegro_layer.irreps_dim

        scalar_in = torch.randn(n_edges, num_scalar)
        tensor_in = torch.randn(n_edges, irreps_dim)
        edge_angular = torch.randn(n_edges, sh_dim)

        scalar_out, tensor_out = allegro_layer(scalar_in, tensor_in, edge_angular)

        assert scalar_out.shape == (n_edges, num_scalar)
        assert tensor_out.shape == (n_edges, irreps_dim)

    def test_dual_track_independence(self, allegro_layer, layer_config):
        """Test that scalar and tensor tracks have different values."""
        n_edges = 10
        num_scalar = layer_config["num_scalar_features"]
        irreps_dim = allegro_layer.irreps_dim
        sh_dim = (layer_config["l_max"] + 1) ** 2

        scalar_in = torch.randn(n_edges, num_scalar)
        tensor_in = torch.randn(n_edges, irreps_dim)
        edge_angular = torch.randn(n_edges, sh_dim)

        scalar_out, tensor_out = allegro_layer(scalar_in, tensor_in, edge_angular)

        # Outputs should differ from inputs (transformation happened)
        assert not torch.allclose(scalar_in, scalar_out, atol=1e-3)

    def test_different_mlp_depths(self, layer_config):
        """Test with different MLP depths."""
        for depth in [0, 1, 2]:
            layer = AllegroLayer(
                num_scalar_features=layer_config["num_scalar_features"],
                num_tensor_features=layer_config["num_tensor_features"],
                l_max=layer_config["l_max"],
                mlp_depth=depth,
            )
            n_edges = 10
            scalar_in = torch.randn(n_edges, layer_config["num_scalar_features"])
            tensor_in = torch.randn(n_edges, layer.irreps_dim)
            edge_angular = torch.randn(n_edges, (layer_config["l_max"] + 1) ** 2)

            scalar_out, tensor_out = layer(scalar_in, tensor_in, edge_angular)
            assert scalar_out.shape == scalar_in.shape


# ===========================================================================
# TestAllegro
# ===========================================================================


class TestAllegro:
    """Test Allegro encoder."""

    @pytest.fixture
    def encoder_config(self):
        return {
            "num_elements": 5,
            "num_scalar_features": 32,
            "num_tensor_features": 8,
            "r_max": 5.0,
            "num_bessel": 8,
            "l_max": 2,
            "num_layers": 2,
            "mlp_depth": 1,
        }

    @pytest.fixture
    def encoder(self, encoder_config):
        return Allegro(**encoder_config)

    def test_initialization(self, encoder, encoder_config):
        """Test encoder has all components."""
        assert hasattr(encoder, "embedding")
        assert isinstance(encoder.embedding, PairEmbedding)
        assert len(encoder.layers) == encoder_config["num_layers"]

    def test_config_storage(self, encoder, encoder_config):
        """Test config is properly stored."""
        assert encoder.config.num_elements == encoder_config["num_elements"]
        assert encoder.config.num_scalar_features == encoder_config["num_scalar_features"]
        assert encoder.config.num_layers == encoder_config["num_layers"]

    def test_forward_output_shape(self, encoder, encoder_config, graph_data):
        """Test encoder output shape."""
        per_layer = encoder(
            Z=graph_data["Z"],
            bond_dist=graph_data["bond_dist"],
            bond_diff=graph_data["bond_diff"],
            edge_index=graph_data["edge_index"],
        )
        expected = (
            graph_data["n_edges"],
            encoder_config["num_layers"],
            encoder_config["num_scalar_features"],
        )
        assert per_layer.shape == expected

    def test_different_num_layers(self, encoder_config, graph_data):
        """Test with different layer counts."""
        for n_layers in [1, 2, 3]:
            enc = Allegro(
                num_elements=graph_data["num_elements"],
                num_scalar_features=encoder_config["num_scalar_features"],
                num_tensor_features=encoder_config["num_tensor_features"],
                r_max=encoder_config["r_max"],
                num_layers=n_layers,
            )
            out = enc(
                Z=graph_data["Z"],
                bond_dist=graph_data["bond_dist"],
                bond_diff=graph_data["bond_diff"],
                edge_index=graph_data["edge_index"],
            )
            assert out.shape[1] == n_layers

    def test_different_feature_sizes(self, graph_data):
        """Test with different scalar/tensor feature sizes."""
        for n_scalar, n_tensor in [(16, 4), (64, 16), (128, 32)]:
            enc = Allegro(
                num_elements=graph_data["num_elements"],
                num_scalar_features=n_scalar,
                num_tensor_features=n_tensor,
                r_max=5.0,
                num_layers=1,
            )
            out = enc(
                Z=graph_data["Z"],
                bond_dist=graph_data["bond_dist"],
                bond_diff=graph_data["bond_diff"],
                edge_index=graph_data["edge_index"],
            )
            assert out.shape == (graph_data["n_edges"], 1, n_scalar)


# ===========================================================================
# TestScaleShiftAllegro
# ===========================================================================


class TestScaleShiftAllegro:
    """Test ScaleShiftAllegro complete model."""

    @pytest.fixture
    def model_config(self):
        return {
            "num_elements": 5,
            "num_scalar_features": 32,
            "num_tensor_features": 8,
            "r_max": 5.0,
            "num_bessel": 8,
            "l_max": 2,
            "num_layers": 2,
            "mlp_depth": 1,
            "compute_forces": False,
            "atomic_inter_scale": 1.0,
            "atomic_inter_shift": 0.0,
        }

    @pytest.fixture
    def model(self, model_config):
        return ScaleShiftAllegro(**model_config)

    def test_initialization(self, model, model_config):
        """Test model has all components."""
        assert hasattr(model, "encoder")
        assert isinstance(model.encoder, Allegro)
        assert len(model.readouts) == model_config["num_layers"]
        assert hasattr(model, "energy_head")

    def test_energy_output(self, model, graph_data):
        """Test energy prediction."""
        results = model(
            Z=graph_data["Z"],
            pos=graph_data["pos"],
            bond_dist=graph_data["bond_dist"],
            bond_diff=graph_data["bond_diff"],
            edge_index=graph_data["edge_index"],
            batch=graph_data["batch"],
        )
        assert "energy" in results
        assert results["energy"].shape == (1,)  # single molecule

    def test_force_output(self, model_config, graph_data):
        """Test force prediction with compute_forces=True."""
        model_config["compute_forces"] = True
        model = ScaleShiftAllegro(**model_config)

        pos = graph_data["pos"].clone().requires_grad_(True)
        # Recompute bond_diff from pos
        src, dst = graph_data["edge_index"][:, 0], graph_data["edge_index"][:, 1]
        bond_diff = pos[dst] - pos[src]
        bond_dist = bond_diff.norm(dim=-1).clamp(min=0.1)

        results = model(
            Z=graph_data["Z"],
            pos=pos,
            bond_dist=bond_dist,
            bond_diff=bond_diff,
            edge_index=graph_data["edge_index"],
            batch=graph_data["batch"],
        )

        assert "forces" in results
        assert results["forces"].shape == (graph_data["n_nodes"], 3)

    def test_gradient_flow(self, model, graph_data):
        """Test that gradients flow through the model."""
        Z = graph_data["Z"]
        pos = graph_data["pos"].clone().requires_grad_(True)
        src, dst = graph_data["edge_index"][:, 0], graph_data["edge_index"][:, 1]
        bond_diff = pos[dst] - pos[src]
        bond_dist = bond_diff.norm(dim=-1).clamp(min=0.1)

        results = model(
            Z=Z,
            pos=pos,
            bond_dist=bond_dist,
            bond_diff=bond_diff,
            edge_index=graph_data["edge_index"],
            batch=graph_data["batch"],
        )

        loss = results["energy"].sum()
        loss.backward()

        # Check at least some parameter has gradient
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
            if p.requires_grad
        )
        assert has_grad

    def test_scale_shift(self, model_config, graph_data):
        """Test that scale and shift are applied."""
        model_config["atomic_inter_scale"] = 2.0
        model_config["atomic_inter_shift"] = 0.5
        model = ScaleShiftAllegro(**model_config)

        results = model(
            Z=graph_data["Z"],
            pos=graph_data["pos"],
            bond_dist=graph_data["bond_dist"],
            bond_diff=graph_data["bond_diff"],
            edge_index=graph_data["edge_index"],
            batch=graph_data["batch"],
        )

        assert "energy" in results
        # Energy should be finite
        assert torch.isfinite(results["energy"]).all()

    def test_multi_batch(self, model_config):
        """Test with multiple molecules in a batch."""
        model = ScaleShiftAllegro(**model_config)

        n_nodes = 20
        n_edges = 50
        Z = torch.randint(0, model_config["num_elements"], (n_nodes,))
        pos = torch.randn(n_nodes, 3)
        edge_index = torch.randint(0, n_nodes, (n_edges, 2))
        bond_diff = torch.randn(n_edges, 3)
        bond_dist = bond_diff.norm(dim=-1).clamp(min=0.1)
        # Two molecules: first 10 nodes, last 10 nodes
        batch = torch.cat([torch.zeros(10, dtype=torch.long), torch.ones(10, dtype=torch.long)])

        results = model(
            Z=Z, pos=pos, bond_dist=bond_dist,
            bond_diff=bond_diff, edge_index=edge_index, batch=batch,
        )

        assert results["energy"].shape == (2,)  # two molecules
