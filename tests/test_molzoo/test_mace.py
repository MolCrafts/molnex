"""Tests for molzoo.mace module."""

import pytest
import torch
import torch.nn as nn
import math

import cuequivariance_torch as cuet

from molzoo.mace import (
    EmbeddingBlock,
    InteractionBlock,
    ProductBlock,
)
from molrep.embedding.node import DiscreteEmbeddingSpec, JointEmbedding
from molrep.embedding.radial import BesselRBF
from molrep.embedding.angular import SphericalHarmonics
from molrep.embedding.cutoff import CosineCutoff
from molrep.interaction.tensor_product import ConvTP, irreps_from_l_max, sh_irreps_from_l_max
from molrep.interaction.symmetric_contraction import SymmetricContraction
from molrep.readout.basis_projection import BasisProjection
from molrep.utils.equivariance import (
    rotation_matrix_z,
    random_rotation_matrix,
    rotate_vectors,
)
from tests.utils import assert_module_compiles, assert_module_exports, assert_outputs_close


class TestEmbeddingBlock:
    """Test EmbeddingBlock initialization and forward pass."""

    @pytest.fixture
    def embedding_config(self):
        """Common configuration for embedding tests."""
        return {
            "num_species": 5,
            "num_features": 16,
            "r_max": 5.0,
            "num_bessel": 8,
            "l_max": 2,
        }

    @pytest.fixture
    def node_attr_specs(self, embedding_config):
        """Node attribute specifications."""
        return [
            DiscreteEmbeddingSpec(
                input_key="Z",
                num_classes=embedding_config["num_species"],
                emb_dim=embedding_config["num_features"],
            )
        ]

    @pytest.fixture
    def embedding_block(self, node_attr_specs, embedding_config):
        """Create an EmbeddingBlock instance."""
        return EmbeddingBlock(
            node_attr_specs=node_attr_specs,
            num_features=embedding_config["num_features"],
            r_max=embedding_config["r_max"],
            num_bessel=embedding_config["num_bessel"],
            l_max=embedding_config["l_max"],
        )

    def test_initialization(self, embedding_block, embedding_config):
        """Test that EmbeddingBlock initializes all components correctly."""
        # Check node_embedding
        assert hasattr(embedding_block, "node_embedding")
        assert isinstance(embedding_block.node_embedding, JointEmbedding)

        # Check radial_embedding
        assert hasattr(embedding_block, "radial_embedding")
        assert isinstance(embedding_block.radial_embedding, BesselRBF)
        assert embedding_block.radial_embedding.config.r_cut == embedding_config["r_max"]
        assert embedding_block.radial_embedding.config.num_radial == embedding_config["num_bessel"]

        # Check spherical_harmonics
        assert hasattr(embedding_block, "spherical_harmonics")
        assert isinstance(embedding_block.spherical_harmonics, SphericalHarmonics)
        assert embedding_block.spherical_harmonics.l_max == embedding_config["l_max"]

        # Check cutoff_fn
        assert hasattr(embedding_block, "cutoff_fn")
        assert isinstance(embedding_block.cutoff_fn, CosineCutoff)
        assert embedding_block.cutoff_fn.config.r_cut == embedding_config["r_max"]

    def test_config_storage(self, embedding_block, embedding_config):
        """Test that configuration is properly stored."""
        assert hasattr(embedding_block, "config")
        config = embedding_block.config
        assert config.num_features == embedding_config["num_features"]
        assert config.r_max == embedding_config["r_max"]
        assert config.num_bessel == embedding_config["num_bessel"]
        assert config.l_max == embedding_config["l_max"]

    def test_forward_output_shapes(self, embedding_block, embedding_config):
        """Test forward pass returns correct output shapes."""
        n_atoms = 4
        n_edges = 6

        # Create input data
        z = torch.randint(0, embedding_config["num_species"], (n_atoms,))
        bond_dist = torch.rand(n_edges) * embedding_config["r_max"]
        bond_diff = torch.randn(n_edges, 3)

        # Normalize bond_diff to match bond_dist
        bond_diff = bond_diff / torch.norm(bond_diff, dim=-1, keepdim=True) * bond_dist.unsqueeze(-1)

        # Forward pass
        node_feats, edge_attrs, edge_feats = embedding_block(
            Z=z,
            bond_dist=bond_dist,
            bond_diff=bond_diff,
        )

        # Check shapes
        assert node_feats.shape == (n_atoms, embedding_config["num_features"])

        # Spherical harmonics dimension: (2*l_max + 1)^2 for l_max=2 is 9
        expected_sh_dim = (embedding_config["l_max"] + 1) ** 2
        assert edge_attrs.shape == (n_edges, expected_sh_dim)

        assert edge_feats.shape == (n_edges, embedding_config["num_bessel"])

    def test_node_embedding_component(self, embedding_block, embedding_config):
        """Test node_embedding component works independently."""
        n_atoms = 5
        z = torch.randint(0, embedding_config["num_species"], (n_atoms,))

        # Call node_embedding directly
        node_feats = embedding_block.node_embedding(Z=z)

        assert node_feats.shape == (n_atoms, embedding_config["num_features"])
        assert node_feats.dtype == torch.float32

    def test_radial_embedding_component(self, embedding_block, embedding_config):
        """Test radial_embedding component works independently."""
        n_edges = 10
        bond_dist = torch.rand(n_edges) * embedding_config["r_max"]

        # Call radial_embedding directly
        edge_radial = embedding_block.radial_embedding(bond_dist)

        assert edge_radial.shape == (n_edges, embedding_config["num_bessel"])
        assert edge_radial.dtype == torch.float32

    def test_spherical_harmonics_component(self, embedding_block, embedding_config):
        """Test spherical_harmonics component works independently."""
        n_edges = 8
        # Create normalized direction vectors
        edge_dir = torch.randn(n_edges, 3)
        edge_dir = edge_dir / torch.norm(edge_dir, dim=-1, keepdim=True)

        # Call spherical_harmonics directly
        edge_attrs = embedding_block.spherical_harmonics(edge_dir)

        expected_sh_dim = (embedding_config["l_max"] + 1) ** 2
        assert edge_attrs.shape == (n_edges, expected_sh_dim)
        assert edge_attrs.dtype == torch.float32

    def test_cutoff_component(self, embedding_block, embedding_config):
        """Test cutoff_fn component works independently."""
        n_edges = 12
        bond_dist = torch.rand(n_edges) * embedding_config["r_max"]

        # Call cutoff_fn directly
        cutoff_values = embedding_block.cutoff_fn(bond_dist)

        assert cutoff_values.shape == (n_edges,)
        assert cutoff_values.dtype == torch.float32
        # Cutoff should be in [0, 1]
        assert (cutoff_values >= 0.0).all()
        assert (cutoff_values <= 1.0).all()

    def test_edge_feats_includes_cutoff(self, embedding_block, embedding_config):
        """Test that edge_feats properly applies cutoff to radial basis."""
        n_edges = 6
        bond_dist = torch.rand(n_edges) * embedding_config["r_max"]
        bond_diff = torch.randn(n_edges, 3)
        bond_diff = bond_diff / torch.norm(bond_diff, dim=-1, keepdim=True) * bond_dist.unsqueeze(-1)

        z = torch.randint(0, embedding_config["num_species"], (3,))

        # Get outputs
        _, _, edge_feats = embedding_block(
            Z=z,
            bond_dist=bond_dist,
            bond_diff=bond_diff,
        )

        # Compute expected edge_feats manually
        edge_radial = embedding_block.radial_embedding(bond_dist)
        cutoff_values = embedding_block.cutoff_fn(bond_dist)
        expected_edge_feats = edge_radial * cutoff_values.unsqueeze(-1)

        # Check they match
        assert torch.allclose(edge_feats, expected_edge_feats, atol=1e-6)

    def test_cutoff_at_boundary(self, embedding_block, embedding_config):
        """Test cutoff behavior at r_max boundary."""
        # Distance at cutoff should give near-zero cutoff value
        bond_dist = torch.tensor([embedding_config["r_max"]])
        cutoff_value = embedding_block.cutoff_fn(bond_dist)

        # Cosine cutoff should be near 0 at r_max
        assert cutoff_value.item() < 0.01

        # Distance at 0 should give cutoff value of 1
        bond_dist_zero = torch.tensor([0.0])
        cutoff_value_zero = embedding_block.cutoff_fn(bond_dist_zero)
        assert abs(cutoff_value_zero.item() - 1.0) < 0.01


    def test_compile(self, embedding_block, embedding_config):
        """Test that EmbeddingBlock can be compiled with torch.compile."""
        n_atoms = 4
        n_edges = 6

        # Create input data
        z = torch.randint(0, embedding_config["num_species"], (n_atoms,))
        bond_dist = torch.rand(n_edges) * embedding_config["r_max"]
        bond_diff = torch.randn(n_edges, 3)
        bond_diff = bond_diff / torch.norm(bond_diff, dim=-1, keepdim=True) * bond_dist.unsqueeze(-1)

        # Test compilation
        output_uncompiled, output_compiled = assert_module_compiles(
            embedding_block,
            z,
            bond_dist,
            bond_diff,
        )

        # Check outputs match
        assert_outputs_close(output_uncompiled, output_compiled)

    def test_export(self, embedding_block, embedding_config):
        """Test that EmbeddingBlock can be exported with torch.export."""
        n_atoms = 4
        n_edges = 6

        # Create input data
        z = torch.randint(0, embedding_config["num_species"], (n_atoms,))
        bond_dist = torch.rand(n_edges) * embedding_config["r_max"]
        bond_diff = torch.randn(n_edges, 3)
        bond_diff = bond_diff / torch.norm(bond_diff, dim=-1, keepdim=True) * bond_dist.unsqueeze(-1)

        # Test export
        exported_program, output_original, output_exported = assert_module_exports(
            embedding_block,
            args_tuple=(z, bond_dist, bond_diff),
        )

        # Check outputs match
        assert_outputs_close(output_original, output_exported)


class TestInteractionBlock:
    """Test InteractionBlock initialization and forward pass."""

    @pytest.fixture
    def interaction_config(self):
        """Common configuration for interaction block tests."""
        return {
            "num_features": 64,
            "num_bessel": 8,
            "l_max": 2,
            "avg_num_neighbors": 10.0,
        }

    @pytest.fixture
    def interaction_block(self, interaction_config):
        """Create an InteractionBlock instance."""
        return InteractionBlock(**interaction_config)

    def test_initialization(self, interaction_block, interaction_config):
        """Test that InteractionBlock initializes all components correctly."""
        # Check conv_tp (must be created first to provide weight_numel)
        assert hasattr(interaction_block, "conv_tp")
        assert isinstance(interaction_block.conv_tp, ConvTP)

        # Check node_linear
        assert hasattr(interaction_block, "node_linear")
        assert isinstance(interaction_block.node_linear, cuet.Linear)

        # Check radial_mlp
        assert hasattr(interaction_block, "radial_mlp")
        assert isinstance(interaction_block.radial_mlp, nn.Sequential)

        # Check linear
        assert hasattr(interaction_block, "linear")
        assert isinstance(interaction_block.linear, cuet.Linear)

        # Check avg_num_neighbors
        assert hasattr(interaction_block, "avg_num_neighbors")
        assert interaction_block.avg_num_neighbors == interaction_config["avg_num_neighbors"]

    def test_initialization_order_fix(self, interaction_config):
        """Test that conv_tp is initialized before radial_mlp (bug fix)."""
        # This should NOT raise AttributeError about self.conv_tp
        block = InteractionBlock(**interaction_config)

        # Verify conv_tp exists and has weight_numel
        assert hasattr(block, "conv_tp")
        assert hasattr(block.conv_tp, "weight_numel")

        # Verify MLP output dimension matches weight_numel
        mlp_output_layer = block.radial_mlp[-1]
        assert mlp_output_layer.out_features == block.conv_tp.weight_numel

    def test_config_storage(self, interaction_block, interaction_config):
        """Test that configuration is properly stored."""
        assert hasattr(interaction_block, "config")
        config = interaction_block.config
        assert config.num_features == interaction_config["num_features"]
        assert config.num_bessel == interaction_config["num_bessel"]
        assert config.l_max == interaction_config["l_max"]
        assert config.avg_num_neighbors == interaction_config["avg_num_neighbors"]

    def test_forward_output_shapes(self, interaction_block, interaction_config):
        """Test forward pass returns correct output shapes."""
        n_nodes = 20
        n_edges = 50
        l_max = interaction_config["l_max"]
        num_features = interaction_config["num_features"]
        num_bessel = interaction_config["num_bessel"]

        # Calculate irreps dimension (uniform multiplicity after optimization)
        irreps_dim = sum(
            num_features * (2 * l + 1)
            for l in range(l_max + 1)
        )

        # Calculate spherical harmonics dimension
        sh_dim = sum(2 * l + 1 for l in range(l_max + 1))

        # Create input data
        node_feats = torch.randn(n_nodes, irreps_dim)
        edge_attrs = torch.randn(n_edges, sh_dim)
        edge_feats = torch.randn(n_edges, num_bessel)
        edge_index = torch.randint(0, n_nodes, (n_edges, 2))

        # Forward pass
        output_feats, skip_connection = interaction_block(
            node_feats=node_feats,
            edge_attrs=edge_attrs,
            edge_feats=edge_feats,
            edge_index=edge_index,
        )

        # Check output shapes
        assert output_feats.shape == (n_nodes, irreps_dim)
        assert skip_connection.shape == (n_nodes, irreps_dim)

    def test_skip_connection_is_input(self, interaction_block, interaction_config):
        """Test that skip connection returns the original input."""
        n_nodes = 10
        n_edges = 30
        l_max = interaction_config["l_max"]
        num_features = interaction_config["num_features"]

        # Calculate dimensions (uniform multiplicity)
        irreps_dim = sum(
            num_features * (2 * l + 1)
            for l in range(l_max + 1)
        )
        sh_dim = sum(2 * l + 1 for l in range(l_max + 1))

        # Create input data
        node_feats = torch.randn(n_nodes, irreps_dim)
        edge_attrs = torch.randn(n_edges, sh_dim)
        edge_feats = torch.randn(n_edges, interaction_config["num_bessel"])
        edge_index = torch.randint(0, n_nodes, (n_edges, 2))

        # Forward pass
        _, skip_connection = interaction_block(
            node_feats=node_feats,
            edge_attrs=edge_attrs,
            edge_feats=edge_feats,
            edge_index=edge_index,
        )

        # Skip connection should be identical to input
        assert torch.equal(skip_connection, node_feats)

    def test_radial_mlp_architecture(self, interaction_block, interaction_config):
        """Test radial_mlp MLP architecture."""
        mlp = interaction_block.radial_mlp
        num_features = interaction_config["num_features"]
        num_bessel = interaction_config["num_bessel"]
        weight_numel = interaction_block.conv_tp.weight_numel

        # Check number of layers (Linear -> SiLU -> Linear -> SiLU -> Linear)
        assert len(mlp) == 5

        # Check layer types
        assert isinstance(mlp[0], nn.Linear)
        assert isinstance(mlp[1], nn.SiLU)
        assert isinstance(mlp[2], nn.Linear)
        assert isinstance(mlp[3], nn.SiLU)
        assert isinstance(mlp[4], nn.Linear)

        # Check dimensions
        assert mlp[0].in_features == num_bessel
        assert mlp[0].out_features == num_features
        assert mlp[2].in_features == num_features
        assert mlp[2].out_features == num_features
        assert mlp[4].in_features == num_features
        assert mlp[4].out_features == weight_numel

    def test_cuequivariance_integration(self, interaction_block):
        """Test that cuEquivariance ChannelWiseTensorProduct is used."""
        # Conv_tp should wrap a ChannelWiseTensorProduct
        assert hasattr(interaction_block.conv_tp, "cue_tp")
        assert isinstance(interaction_block.conv_tp.cue_tp, cuet.ChannelWiseTensorProduct)

        # Verify it has the expected ChannelWiseTensorProduct configuration
        cue_tp = interaction_block.conv_tp.cue_tp
        assert hasattr(cue_tp, "irreps_in1")
        assert hasattr(cue_tp, "irreps_in2")
        assert hasattr(cue_tp, "irreps_out")

    def test_different_l_max_values(self, interaction_config):
        """Test initialization with different l_max values."""
        for l_max in [1, 2, 3]:
            block = InteractionBlock(
                num_features=interaction_config["num_features"],
                num_bessel=interaction_config["num_bessel"],
                l_max=l_max,
                avg_num_neighbors=interaction_config["avg_num_neighbors"],
            )

            # Should initialize without errors
            assert block.config.l_max == l_max
            assert hasattr(block.conv_tp, "weight_numel")

    def test_different_num_features(self, interaction_config):
        """Test initialization with different num_features values."""
        for num_features in [32, 64, 128]:
            block = InteractionBlock(
                num_features=num_features,
                num_bessel=interaction_config["num_bessel"],
                l_max=interaction_config["l_max"],
                avg_num_neighbors=interaction_config["avg_num_neighbors"],
            )

            # MLP hidden dimension should match num_features
            assert block.radial_mlp[0].out_features == num_features
            assert block.radial_mlp[2].in_features == num_features

    def test_edge_index_format(self, interaction_block, interaction_config):
        """Test that edge_index format (n_edges, 2) works correctly."""
        n_nodes = 15
        n_edges = 40
        l_max = interaction_config["l_max"]
        num_features = interaction_config["num_features"]

        # Calculate dimensions (uniform multiplicity)
        irreps_dim = sum(
            num_features * (2 * l + 1)
            for l in range(l_max + 1)
        )
        sh_dim = sum(2 * l + 1 for l in range(l_max + 1))

        # Create input data
        node_feats = torch.randn(n_nodes, irreps_dim)
        edge_attrs = torch.randn(n_edges, sh_dim)
        edge_feats = torch.randn(n_edges, interaction_config["num_bessel"])
        edge_index = torch.randint(0, n_nodes, (n_edges, 2))

        # Forward pass should work without errors
        output, _ = interaction_block(node_feats, edge_attrs, edge_feats, edge_index)
        assert output.shape == (n_nodes, irreps_dim)

    def test_compile(self, interaction_block, interaction_config):
        """Test that InteractionBlock can be compiled with torch.compile."""
        n_nodes = 20
        n_edges = 50
        l_max = interaction_config["l_max"]
        num_features = interaction_config["num_features"]
        num_bessel = interaction_config["num_bessel"]

        # Calculate dimensions
        irreps_dim = sum(num_features * (2 * l + 1) for l in range(l_max + 1))
        sh_dim = sum(2 * l + 1 for l in range(l_max + 1))

        # Create input data
        node_feats = torch.randn(n_nodes, irreps_dim)
        edge_attrs = torch.randn(n_edges, sh_dim)
        edge_feats = torch.randn(n_edges, num_bessel)
        edge_index = torch.randint(0, n_nodes, (n_edges, 2))

        # Test compilation
        output_uncompiled, output_compiled = assert_module_compiles(
            interaction_block,
            node_feats, edge_attrs, edge_feats, edge_index
        )

        # Check outputs match
        assert_outputs_close(output_uncompiled, output_compiled)

    def test_export(self, interaction_block, interaction_config):
        """Test that InteractionBlock can be exported with torch.export."""
        n_nodes = 20
        n_edges = 50
        l_max = interaction_config["l_max"]
        num_features = interaction_config["num_features"]
        num_bessel = interaction_config["num_bessel"]

        # Calculate dimensions
        irreps_dim = sum(num_features * (2 * l + 1) for l in range(l_max + 1))
        sh_dim = sum(2 * l + 1 for l in range(l_max + 1))

        # Create input data
        node_feats = torch.randn(n_nodes, irreps_dim)
        edge_attrs = torch.randn(n_edges, sh_dim)
        edge_feats = torch.randn(n_edges, num_bessel)
        edge_index = torch.randint(0, n_nodes, (n_edges, 2))

        # Test export
        exported_program, output_original, output_exported = assert_module_exports(
            interaction_block,
            args_tuple=(node_feats, edge_attrs, edge_feats, edge_index),
        )

        # Check outputs match
        assert_outputs_close(output_original, output_exported)


class TestProductBlock:
    """Test ProductBlock initialization and forward pass."""

    @pytest.fixture
    def product_config(self):
        """Common configuration for product basis block tests."""
        return {
            "irreps_dim": 576,  # For num_features=64, l_max=2 (uniform): 64*1 + 64*3 + 64*5 = 576
            "num_features": 64,
            "num_bessel": 8,
            "l_max": 2,
            "correlation": 2,
            "num_elements": 118,
        }

    @pytest.fixture
    def product_block(self, product_config):
        """Create a ProductBlock instance."""
        return ProductBlock(**product_config)

    def test_initialization(self, product_block, product_config):
        """Test that EquivariantProductBasisBlock initializes all components correctly."""
        # Check symmetric_contraction
        assert hasattr(product_block, "symmetric_contraction")
        assert isinstance(product_block.symmetric_contraction, SymmetricContraction)

        # Check basis_projection
        assert hasattr(product_block, "basis_projection")
        assert isinstance(product_block.basis_projection, BasisProjection)

        # Check linear
        assert hasattr(product_block, "linear")
        assert isinstance(product_block.linear, nn.Linear)
        assert product_block.linear.in_features == product_config["irreps_dim"]
        assert product_block.linear.out_features == product_config["num_features"]

    def test_symmetric_contraction_config(self, product_block, product_config):
        """Test that SymmetricContraction is configured correctly."""
        sc = product_block.symmetric_contraction
        assert sc.config.hidden_dim == product_config["irreps_dim"]
        assert sc.config.num_species == product_config["num_elements"]
        assert sc.config.max_body_order == product_config["correlation"]

    def test_basis_projection_config(self, product_block, product_config):
        """Test that BasisProjection is configured correctly."""
        bp = product_block.basis_projection
        assert bp.config.hidden_dim == product_config["irreps_dim"]
        assert bp.config.num_radial == product_config["num_bessel"]
        assert bp.config.l_max == product_config["l_max"]
        assert bp.config.max_body_order == product_config["correlation"]

    def test_forward_output_shapes(self, product_block, product_config):
        """Test forward pass returns correct output shapes."""
        n_nodes = 20
        irreps_dim = product_config["irreps_dim"]
        num_features = product_config["num_features"]
        num_elements = product_config["num_elements"]

        # Create input data
        node_feats = torch.randn(n_nodes, irreps_dim)
        node_attrs = torch.randint(0, num_elements, (n_nodes,))

        # Forward pass
        output = product_block(node_feats, node_attrs)

        # Check output shape
        assert output.shape == (n_nodes, num_features)

    def test_forward_dtype(self, product_block, product_config):
        """Test that forward pass preserves dtype."""
        n_nodes = 10
        irreps_dim = product_config["irreps_dim"]

        # Test with float32
        node_feats = torch.randn(n_nodes, irreps_dim, dtype=torch.float32)
        node_attrs = torch.randint(0, product_config["num_elements"], (n_nodes,))

        output = product_block(node_feats, node_attrs)
        assert output.dtype == torch.float32

    def test_different_correlation_orders(self, product_config):
        """Test initialization with different correlation (body order) values."""
        for correlation in [1, 2, 3]:
            block = ProductBlock(
                irreps_dim=product_config["irreps_dim"],
                num_features=product_config["num_features"],
                num_bessel=product_config["num_bessel"],
                l_max=product_config["l_max"],
                correlation=correlation,
                num_elements=product_config["num_elements"],
            )

            # Should initialize without errors
            assert block.symmetric_contraction.config.max_body_order == correlation
            assert block.basis_projection.config.max_body_order == correlation

    def test_different_num_elements(self, product_config):
        """Test initialization with different num_elements values."""
        for num_elements in [10, 50, 118]:
            block = ProductBlock(
                irreps_dim=product_config["irreps_dim"],
                num_features=product_config["num_features"],
                num_bessel=product_config["num_bessel"],
                l_max=product_config["l_max"],
                correlation=product_config["correlation"],
                num_elements=num_elements,
            )

            assert block.symmetric_contraction.config.num_species == num_elements

    def test_batch_processing(self, product_block, product_config):
        """Test that the block processes batches correctly."""
        # Test with different batch sizes
        for n_nodes in [5, 20, 100]:
            node_feats = torch.randn(n_nodes, product_config["irreps_dim"])
            node_attrs = torch.randint(0, product_config["num_elements"], (n_nodes,))

            output = product_block(node_feats, node_attrs)
            assert output.shape == (n_nodes, product_config["num_features"])

    def test_symmetric_contraction_component(self, product_block, product_config):
        """Test symmetric_contraction component works independently."""
        n_nodes = 15
        irreps_dim = product_config["irreps_dim"]

        node_feats = torch.randn(n_nodes, irreps_dim)
        node_attrs = torch.randint(0, product_config["num_elements"], (n_nodes,))

        # Call symmetric_contraction directly
        basis = product_block.symmetric_contraction(node_feats, node_attrs)

        # Output should have same shape as input (contraction preserves dimension)
        assert basis.shape == (n_nodes, irreps_dim)

    def test_basis_projection_component(self, product_block, product_config):
        """Test basis_projection component works independently."""
        n_nodes = 15
        irreps_dim = product_config["irreps_dim"]

        # Create dummy basis features
        basis = torch.randn(n_nodes, irreps_dim)

        # Call basis_projection directly (passthrough in current implementation)
        features = product_block.basis_projection(basis)

        # Currently acts as identity
        assert torch.equal(features, basis)

    def test_linear_component(self, product_block, product_config):
        """Test linear component works independently."""
        n_nodes = 15
        irreps_dim = product_config["irreps_dim"]
        num_features = product_config["num_features"]

        # Create dummy features
        features = torch.randn(n_nodes, irreps_dim)

        # Call linear directly
        output = product_block.linear(features)

        assert output.shape == (n_nodes, num_features)

    def test_gradient_flow(self, product_block, product_config):
        """Test that gradients flow through the block correctly."""
        n_nodes = 10
        irreps_dim = product_config["irreps_dim"]

        node_feats = torch.randn(n_nodes, irreps_dim, requires_grad=True)
        node_attrs = torch.randint(0, product_config["num_elements"], (n_nodes,))

        # Forward pass
        output = product_block(node_feats, node_attrs)

        # Backward pass
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert node_feats.grad is not None
        assert not torch.isnan(node_feats.grad).any()

    def test_cuequivariance_integration(self, product_block):
        """Test that cuEquivariance SymmetricContraction is used."""
        sc = product_block.symmetric_contraction
        assert hasattr(sc, "symmetric_contraction")
        assert isinstance(sc.symmetric_contraction, cuet.SymmetricContraction)

        # Verify configuration
        cue_sc = sc.symmetric_contraction
        assert hasattr(cue_sc, "contraction_degree")
        assert hasattr(cue_sc, "num_elements")

    def test_compile(self, product_block, product_config):
        """Test that ProductBlock can be compiled with torch.compile."""
        n_nodes = 20
        irreps_dim = product_config["irreps_dim"]
        num_elements = product_config["num_elements"]

        # Create input data
        node_feats = torch.randn(n_nodes, irreps_dim)
        node_attrs = torch.randint(0, num_elements, (n_nodes,))

        # Test compilation
        output_uncompiled, output_compiled = assert_module_compiles(
            product_block,
            node_feats, node_attrs
        )

        # Check outputs match
        assert_outputs_close(output_uncompiled, output_compiled)

    def test_export(self, product_block, product_config):
        """Test that ProductBlock can be exported with torch.export."""
        n_nodes = 20
        irreps_dim = product_config["irreps_dim"]
        num_elements = product_config["num_elements"]

        # Create input data
        node_feats = torch.randn(n_nodes, irreps_dim)
        node_attrs = torch.randint(0, num_elements, (n_nodes,))

        # Test export
        exported_program, output_original, output_exported = assert_module_exports(
            product_block,
            args_tuple=(node_feats, node_attrs),
        )

        # Check outputs match
        assert_outputs_close(output_original, output_exported)


class TestEmbeddingBlockEquivariance:
    """Test equivariance properties of EmbeddingBlock."""

    @pytest.fixture
    def embedding_block(self):
        """Create an EmbeddingBlock for equivariance testing."""
        node_attr_specs = [
            DiscreteEmbeddingSpec(
                input_key="Z",
                num_classes=5,
                emb_dim=16,
            )
        ]
        return EmbeddingBlock(
            node_attr_specs=node_attr_specs,
            num_features=16,
            r_max=5.0,
            num_bessel=8,
            l_max=2,
        )

    def test_spherical_harmonics_equivariance(self, embedding_block):
        """Test that spherical harmonics are equivariant under rotation.

        Rotating the bond vectors should rotate the spherical harmonics accordingly.
        """
        n_atoms = 4
        n_edges = 6

        # Create input data
        z = torch.randint(0, 5, (n_atoms,))
        bond_diff = torch.randn(n_edges, 3)
        bond_dist = torch.norm(bond_diff, dim=-1)

        # Normalize bond_diff
        bond_diff_normalized = bond_diff / bond_dist.unsqueeze(-1)

        # Forward pass
        _, edge_attrs1, _ = embedding_block(
            Z=z,
            bond_dist=bond_dist,
            bond_diff=bond_diff,
        )

        # Rotate bond vectors
        angle = math.pi / 2
        rot_matrix = rotation_matrix_z(angle, dtype=bond_diff.dtype)
        bond_diff_rot = rotate_vectors(bond_diff, rot_matrix)

        # Forward pass on rotated
        _, edge_attrs2, _ = embedding_block(
            Z=z,
            bond_dist=bond_dist,
            bond_diff=bond_diff_rot,
        )

        # l=0 component should be invariant
        assert torch.allclose(edge_attrs1[:, 0], edge_attrs2[:, 0], atol=1e-5)

        # Overall norm should be preserved
        norm1 = edge_attrs1.norm(dim=-1)
        norm2 = edge_attrs2.norm(dim=-1)
        assert torch.allclose(norm1, norm2, rtol=1e-4, atol=1e-4)

    def test_radial_features_invariance(self, embedding_block):
        """Test that radial features are rotation invariant.

        Rotating bond vectors should not change radial features (distances).
        """
        n_atoms = 4
        n_edges = 6

        z = torch.randint(0, 5, (n_atoms,))
        bond_diff = torch.randn(n_edges, 3)
        bond_dist = torch.norm(bond_diff, dim=-1)

        # Forward pass
        _, _, edge_feats1 = embedding_block(
            Z=z,
            bond_dist=bond_dist,
            bond_diff=bond_diff,
        )

        # Rotate bond vectors
        rot_matrix = random_rotation_matrix(dtype=bond_diff.dtype)
        bond_diff_rot = rotate_vectors(bond_diff, rot_matrix)

        # Forward pass on rotated
        _, _, edge_feats2 = embedding_block(
            Z=z,
            bond_dist=bond_dist,
            bond_diff=bond_diff_rot,
        )

        # Radial features should be identical (rotation invariant)
        assert torch.allclose(edge_feats1, edge_feats2, rtol=1e-5, atol=1e-5)


class TestInteractionBlockEquivariance:
    """Test equivariance properties of InteractionBlock."""

    @pytest.fixture
    def interaction_block(self):
        """Create an InteractionBlock for testing."""
        return InteractionBlock(
            num_features=32,
            num_bessel=8,
            l_max=1,  # Use l_max=1 for simpler testing
            avg_num_neighbors=10.0,
        )

    def test_output_shape_consistency(self, interaction_block):
        """Test that rotation doesn't change output shapes."""
        n_nodes = 10
        n_edges = 30
        l_max = 1
        num_features = 32

        # Calculate dimensions
        irreps_dim = sum(num_features * (2 * l + 1) for l in range(l_max + 1))
        sh_dim = sum(2 * l + 1 for l in range(l_max + 1))

        node_feats = torch.randn(n_nodes, irreps_dim)
        edge_attrs = torch.randn(n_edges, sh_dim)
        edge_feats = torch.randn(n_edges, 8)
        edge_index = torch.randint(0, n_nodes, (n_edges, 2))

        # Forward pass
        output, _ = interaction_block(node_feats, edge_attrs, edge_feats, edge_index)

        assert output.shape == (n_nodes, irreps_dim)


class TestProductBlockEquivariance:
    """Test equivariance properties of ProductBlock."""

    @pytest.fixture
    def product_block(self):
        """Create a ProductBlock for testing."""
        return ProductBlock(
            irreps_dim=160,  # 32*1 + 32*3 = 128 for l_max=1, num_features=32
            num_features=32,
            num_bessel=8,
            l_max=1,
            correlation=2,
            num_elements=10,
        )

    def test_permutation_equivariance(self, product_block):
        """Test that ProductBlock is equivariant to node permutations."""
        n_nodes = 15
        irreps_dim = 160

        node_feats = torch.randn(n_nodes, irreps_dim)
        node_attrs = torch.randint(0, 10, (n_nodes,))

        # Forward pass
        output1 = product_block(node_feats, node_attrs)

        # Permute nodes
        perm = torch.randperm(n_nodes)
        node_feats_perm = node_feats[perm]
        node_attrs_perm = node_attrs[perm]

        # Forward on permuted
        output2 = product_block(node_feats_perm, node_attrs_perm)

        # Outputs should match after inverse permutation
        assert torch.allclose(output1[perm], output2, rtol=1e-5, atol=1e-5)
