"""Tests for molzoo.allegro module."""

import pytest
import torch
import math

from molzoo.allegro import (
    Allegro,
    AllegroLayer,
    PairEmbedding,
    ScaleShiftAllegro,
)
from molrep.embedding.radial import BesselRBF
from molrep.embedding.angular import SphericalHarmonics
from molrep.embedding.cutoff import PolynomialCutoff
from molrep.utils.equivariance import (
    rotation_matrix_z,
    random_rotation_matrix,
    rotate_vectors,
)
from tests.utils import assert_module_compiles, assert_module_exports, assert_outputs_close


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

    def test_compile(self, pair_emb, graph_data):
        """Test that PairEmbedding can be compiled with torch.compile."""
        # Test compilation using positional arguments
        output_uncompiled, output_compiled = assert_module_compiles(
            pair_emb,
            graph_data["Z"],
            graph_data["bond_dist"],
            graph_data["bond_diff"],
            graph_data["edge_index"],
        )

        # Check outputs match
        assert_outputs_close(output_uncompiled, output_compiled)

    def test_export(self, pair_emb, graph_data):
        """Test that PairEmbedding can be exported with torch.export."""
        # Test export using positional arguments
        exported_program, output_original, output_exported = assert_module_exports(
            pair_emb,
            args_tuple=(
                graph_data["Z"],
                graph_data["bond_dist"],
                graph_data["bond_diff"],
                graph_data["edge_index"],
            ),
        )

        # Check outputs match
        assert_outputs_close(output_original, output_exported)


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

    def test_compile(self, allegro_layer, layer_config):
        """Test that AllegroLayer can be compiled with torch.compile."""
        n_edges = 30
        num_scalar = layer_config["num_scalar_features"]
        l_max = layer_config["l_max"]
        sh_dim = (l_max + 1) ** 2
        irreps_dim = allegro_layer.irreps_dim

        scalar_in = torch.randn(n_edges, num_scalar)
        tensor_in = torch.randn(n_edges, irreps_dim)
        edge_angular = torch.randn(n_edges, sh_dim)

        # Test compilation
        output_uncompiled, output_compiled = assert_module_compiles(
            allegro_layer,
            scalar_in, tensor_in, edge_angular
        )

        # Check outputs match
        assert_outputs_close(output_uncompiled, output_compiled)

    def test_export(self, allegro_layer, layer_config):
        """Test that AllegroLayer can be exported with torch.export."""
        n_edges = 30
        num_scalar = layer_config["num_scalar_features"]
        l_max = layer_config["l_max"]
        sh_dim = (l_max + 1) ** 2
        irreps_dim = allegro_layer.irreps_dim

        scalar_in = torch.randn(n_edges, num_scalar)
        tensor_in = torch.randn(n_edges, irreps_dim)
        edge_angular = torch.randn(n_edges, sh_dim)

        # Test export
        exported_program, output_original, output_exported = assert_module_exports(
            allegro_layer,
            args_tuple=(scalar_in, tensor_in, edge_angular),
        )

        # Check outputs match
        assert_outputs_close(output_original, output_exported)


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

    def test_compile(self, encoder, graph_data):
        """Test that Allegro encoder can be compiled with torch.compile."""
        # Test compilation using positional arguments
        output_uncompiled, output_compiled = assert_module_compiles(
            encoder,
            graph_data["Z"],
            graph_data["bond_dist"],
            graph_data["bond_diff"],
            graph_data["edge_index"],
        )

        # Check outputs match
        assert_outputs_close(output_uncompiled, output_compiled, rtol=1e-3, atol=1e-3)

    def test_export(self, encoder, graph_data):
        """Test that Allegro encoder can be exported with torch.export."""
        # Test export using positional arguments
        exported_program, output_original, output_exported = assert_module_exports(
            encoder,
            args_tuple=(
                graph_data["Z"],
                graph_data["bond_dist"],
                graph_data["bond_diff"],
                graph_data["edge_index"],
            ),
        )

        # Check outputs match
        assert_outputs_close(output_original, output_exported, rtol=1e-3, atol=1e-3)


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

    def test_compile(self, model, graph_data):
        """Test that ScaleShiftAllegro can be compiled with torch.compile."""
        num_graphs = 1

        # Test compilation using positional arguments
        output_uncompiled, output_compiled = assert_module_compiles(
            model,
            graph_data["Z"],
            graph_data["pos"],
            graph_data["bond_dist"],
            graph_data["bond_diff"],
            graph_data["edge_index"],
            graph_data["batch"],
            num_graphs,
        )

        # Check outputs match
        assert_outputs_close(output_uncompiled, output_compiled, rtol=1e-3, atol=1e-3)

    def test_export(self, model, graph_data):
        """Test that ScaleShiftAllegro can be exported with torch.export."""
        num_graphs = 1

        # Test export using positional arguments
        exported_program, output_original, output_exported = assert_module_exports(
            model,
            args_tuple=(
                graph_data["Z"],
                graph_data["pos"],
                graph_data["bond_dist"],
                graph_data["bond_diff"],
                graph_data["edge_index"],
                graph_data["batch"],
                num_graphs,
            ),
        )

        # Check outputs match
        assert_outputs_close(output_original, output_exported, rtol=1e-3, atol=1e-3)


# ===========================================================================
# Equivariance Tests
# ===========================================================================


class TestPairEmbeddingEquivariance:
    """Test equivariance properties of PairEmbedding."""

    @pytest.fixture
    def pair_emb(self):
        """Create PairEmbedding for testing."""
        return PairEmbedding(
            num_elements=5,
            num_scalar_features=32,
            num_tensor_features=8,
            r_max=5.0,
            num_bessel=8,
            l_max=2,
        )

    def test_radial_features_rotation_invariance(self, pair_emb, graph_data):
        """Test that scalar features are rotation invariant."""
        # Forward pass
        scalar1, _, _, cutoff1 = pair_emb(
            Z=graph_data["Z"],
            bond_dist=graph_data["bond_dist"],
            bond_diff=graph_data["bond_diff"],
            edge_index=graph_data["edge_index"],
        )

        # Rotate bond vectors
        rot_matrix = random_rotation_matrix(dtype=graph_data["bond_diff"].dtype)
        bond_diff_rot = rotate_vectors(graph_data["bond_diff"], rot_matrix)

        # Forward pass on rotated
        scalar2, _, _, cutoff2 = pair_emb(
            Z=graph_data["Z"],
            bond_dist=graph_data["bond_dist"],
            bond_diff=bond_diff_rot,
            edge_index=graph_data["edge_index"],
        )

        # Scalar features should be invariant
        assert torch.allclose(scalar1, scalar2, rtol=1e-4, atol=1e-4)
        assert torch.allclose(cutoff1, cutoff2, rtol=1e-5, atol=1e-5)

    def test_spherical_harmonics_equivariance(self, pair_emb, graph_data):
        """Test that spherical harmonics transform correctly under rotation."""
        # Forward pass
        _, _, edge_angular1, _ = pair_emb(
            Z=graph_data["Z"],
            bond_dist=graph_data["bond_dist"],
            bond_diff=graph_data["bond_diff"],
            edge_index=graph_data["edge_index"],
        )

        # Rotate bond vectors
        angle = math.pi / 2
        rot_matrix = rotation_matrix_z(angle, dtype=graph_data["bond_diff"].dtype)
        bond_diff_rot = rotate_vectors(graph_data["bond_diff"], rot_matrix)

        # Forward pass on rotated
        _, _, edge_angular2, _ = pair_emb(
            Z=graph_data["Z"],
            bond_dist=graph_data["bond_dist"],
            bond_diff=bond_diff_rot,
            edge_index=graph_data["edge_index"],
        )

        # l=0 component should be invariant
        assert torch.allclose(edge_angular1[:, 0], edge_angular2[:, 0], atol=1e-5)

        # Overall norms should be preserved
        norm1 = edge_angular1.norm(dim=-1)
        norm2 = edge_angular2.norm(dim=-1)
        assert torch.allclose(norm1, norm2, rtol=1e-4, atol=1e-4)


class TestScaleShiftAllegroEquivariance:
    """Test equivariance properties of complete Allegro model."""

    @pytest.fixture
    def model(self):
        """Create Allegro model for testing."""
        return ScaleShiftAllegro(
            num_elements=5,
            num_scalar_features=32,
            num_tensor_features=8,
            r_max=5.0,
            num_bessel=8,
            l_max=2,
            num_layers=2,
            mlp_depth=1,
            compute_forces=False,
        )

    def test_energy_rotation_invariance(self, model):
        """Test that energy is invariant under rotation.

        Rotating the entire molecular geometry should not change the energy.
        """
        # Create molecular data
        n_nodes = 8
        n_edges = 20
        Z = torch.randint(0, 5, (n_nodes,))
        pos = torch.randn(n_nodes, 3)
        edge_index = torch.randint(0, n_nodes, (n_edges, 2))
        batch = torch.zeros(n_nodes, dtype=torch.long)

        # Compute bond vectors
        src, dst = edge_index[:, 0], edge_index[:, 1]
        bond_diff = pos[dst] - pos[src]
        bond_dist = bond_diff.norm(dim=-1).clamp(min=0.1)

        # Forward pass
        results1 = model(
            Z=Z, pos=pos, bond_dist=bond_dist,
            bond_diff=bond_diff, edge_index=edge_index, batch=batch,
        )
        energy1 = results1["energy"]

        # Rotate entire molecule
        rot_matrix = random_rotation_matrix(dtype=pos.dtype)
        pos_rot = rotate_vectors(pos, rot_matrix)
        bond_diff_rot = rotate_vectors(bond_diff, rot_matrix)

        # Forward pass on rotated
        results2 = model(
            Z=Z, pos=pos_rot, bond_dist=bond_dist,
            bond_diff=bond_diff_rot, edge_index=edge_index, batch=batch,
        )
        energy2 = results2["energy"]

        # Energy should be invariant
        assert torch.allclose(energy1, energy2, rtol=1e-4, atol=1e-4)

    def test_force_computation(self):
        """Test that force computation works correctly.

        This is a basic test to ensure forces can be computed without errors.
        Full force equivariance testing requires more complex setup.
        """
        model = ScaleShiftAllegro(
            num_elements=5,
            num_scalar_features=32,
            num_tensor_features=8,
            r_max=5.0,
            num_layers=2,
            compute_forces=True,
        )

        # Create molecular data
        n_nodes = 8
        n_edges = 20
        Z = torch.randint(0, 5, (n_nodes,))
        pos = torch.randn(n_nodes, 3).requires_grad_(True)
        edge_index = torch.randint(0, n_nodes, (n_edges, 2))
        batch = torch.zeros(n_nodes, dtype=torch.long)

        # Compute bond vectors from positions (ensures grad connection)
        src, dst = edge_index[:, 0], edge_index[:, 1]
        bond_diff = pos[dst] - pos[src]
        bond_dist = bond_diff.norm(dim=-1).clamp(min=0.1)

        # Forward pass - this computes forces via autograd
        results = model(
            Z=Z, pos=pos, bond_dist=bond_dist,
            bond_diff=bond_diff, edge_index=edge_index, batch=batch,
        )

        # Check forces exist and have correct shape
        assert "forces" in results
        assert results["forces"].shape == (n_nodes, 3)

        # Check forces are finite
        assert torch.isfinite(results["forces"]).all()

    def test_translation_invariance(self, model):
        """Test that energy is invariant under translation."""
        # Create molecular data
        n_nodes = 8
        n_edges = 20
        Z = torch.randint(0, 5, (n_nodes,))
        pos = torch.randn(n_nodes, 3)
        edge_index = torch.randint(0, n_nodes, (n_edges, 2))
        batch = torch.zeros(n_nodes, dtype=torch.long)

        # Compute bond vectors
        src, dst = edge_index[:, 0], edge_index[:, 1]
        bond_diff = pos[dst] - pos[src]
        bond_dist = bond_diff.norm(dim=-1).clamp(min=0.1)

        # Forward pass
        results1 = model(
            Z=Z, pos=pos, bond_dist=bond_dist,
            bond_diff=bond_diff, edge_index=edge_index, batch=batch,
        )

        # Translate molecule
        translation = torch.tensor([10.0, -5.0, 3.0])
        pos_trans = pos + translation
        # bond_diff remains the same (translation invariant)

        # Forward pass on translated
        results2 = model(
            Z=Z, pos=pos_trans, bond_dist=bond_dist,
            bond_diff=bond_diff, edge_index=edge_index, batch=batch,
        )

        # Energy should be identical
        assert torch.allclose(results1["energy"], results2["energy"], rtol=1e-5, atol=1e-5)
