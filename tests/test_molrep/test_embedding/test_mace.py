"""Tests for molrep.embedding.mace module."""

import math

import pytest
import torch

from molrep.embedding.angular import SphericalHarmonics
from molrep.embedding.cutoff import CosineCutoff
from molrep.embedding.mace import EmbeddingBlock, EmbeddingSpec
from molrep.embedding.node import DiscreteEmbeddingSpec, JointEmbedding
from molrep.embedding.radial import BesselRBF
from molrep.utils.equivariance import (
    random_rotation_matrix,
    rotate_vectors,
    rotation_matrix_z,
)


class TestEmbeddingSpec:
    """Test EmbeddingSpec configuration."""

    def test_requires_at_least_one_node_attr_spec(self):
        """``node_attr_specs`` has ``min_length=1`` — an empty list is rejected."""
        with pytest.raises(ValueError):
            EmbeddingSpec(node_attr_specs=[], num_features=16, r_max=5.0)


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
        edge_dist = torch.rand(n_edges) * embedding_config["r_max"]
        edge_diff = torch.randn(n_edges, 3)

        # Normalize edge_diff to match edge_dist
        edge_diff = (
            edge_diff / torch.norm(edge_diff, dim=-1, keepdim=True) * edge_dist.unsqueeze(-1)
        )

        # Forward pass
        node_feats, edge_attrs, edge_feats = embedding_block(
            Z=z,
            edge_dist=edge_dist,
            edge_diff=edge_diff,
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
        edge_dist = torch.rand(n_edges) * embedding_config["r_max"]

        # Call radial_embedding directly
        edge_radial = embedding_block.radial_embedding(edge_dist)

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
        edge_dist = torch.rand(n_edges) * embedding_config["r_max"]

        # Call cutoff_fn directly
        cutoff_values = embedding_block.cutoff_fn(edge_dist)

        assert cutoff_values.shape == (n_edges,)
        assert cutoff_values.dtype == torch.float32
        # Cutoff should be in [0, 1]
        assert (cutoff_values >= 0.0).all()
        assert (cutoff_values <= 1.0).all()

    def test_edge_feats_includes_cutoff(self, embedding_block, embedding_config):
        """Test that edge_feats properly applies cutoff to radial basis."""
        n_edges = 6
        edge_dist = torch.rand(n_edges) * embedding_config["r_max"]
        edge_diff = torch.randn(n_edges, 3)
        edge_diff = (
            edge_diff / torch.norm(edge_diff, dim=-1, keepdim=True) * edge_dist.unsqueeze(-1)
        )

        z = torch.randint(0, embedding_config["num_species"], (3,))

        # Get outputs
        _, _, edge_feats = embedding_block(
            Z=z,
            edge_dist=edge_dist,
            edge_diff=edge_diff,
        )

        # Compute expected edge_feats manually
        edge_radial = embedding_block.radial_embedding(edge_dist)
        cutoff_values = embedding_block.cutoff_fn(edge_dist)
        expected_edge_feats = edge_radial * cutoff_values.unsqueeze(-1)

        # Check they match
        assert torch.allclose(edge_feats, expected_edge_feats, atol=1e-6)

    def test_cutoff_at_boundary(self, embedding_block, embedding_config):
        """Test cutoff behavior at r_max boundary."""
        # Distance at cutoff should give near-zero cutoff value
        edge_dist = torch.tensor([embedding_config["r_max"]])
        cutoff_value = embedding_block.cutoff_fn(edge_dist)

        # Cosine cutoff should be near 0 at r_max
        assert cutoff_value.item() < 0.01

        # Distance at 0 should give cutoff value of 1
        bond_dist_zero = torch.tensor([0.0])
        cutoff_value_zero = embedding_block.cutoff_fn(bond_dist_zero)
        assert abs(cutoff_value_zero.item() - 1.0) < 0.01


class TestEmbeddingBlockEquivariance:
    """Test equivariance properties of EmbeddingBlock.

    Migrated from ``tests/test_molzoo/test_mace.py`` by
    mace-subpackage-restructure-02-core (the block now lives in molrep, and
    the molzoo module/package name clash forced the file's removal).
    """

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
        edge_diff = torch.randn(n_edges, 3)
        edge_dist = torch.norm(edge_diff, dim=-1)

        # Forward pass
        _, edge_attrs1, _ = embedding_block(
            Z=z,
            edge_dist=edge_dist,
            edge_diff=edge_diff,
        )

        # Rotate bond vectors
        angle = math.pi / 2
        rot_matrix = rotation_matrix_z(angle, dtype=edge_diff.dtype)
        bond_diff_rot = rotate_vectors(edge_diff, rot_matrix)

        # Forward pass on rotated
        _, edge_attrs2, _ = embedding_block(
            Z=z,
            edge_dist=edge_dist,
            edge_diff=bond_diff_rot,
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
        edge_diff = torch.randn(n_edges, 3)
        edge_dist = torch.norm(edge_diff, dim=-1)

        # Forward pass
        _, _, edge_feats1 = embedding_block(
            Z=z,
            edge_dist=edge_dist,
            edge_diff=edge_diff,
        )

        # Rotate bond vectors
        rot_matrix = random_rotation_matrix(dtype=edge_diff.dtype)
        bond_diff_rot = rotate_vectors(edge_diff, rot_matrix)

        # Forward pass on rotated
        _, _, edge_feats2 = embedding_block(
            Z=z,
            edge_dist=edge_dist,
            edge_diff=bond_diff_rot,
        )

        # Radial features should be identical (rotation invariant)
        assert torch.allclose(edge_feats1, edge_feats2, rtol=1e-5, atol=1e-5)
