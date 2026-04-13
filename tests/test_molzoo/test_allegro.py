"""Tests for encoder-only Allegro modules."""

from __future__ import annotations

import pytest
import torch

from molix.data.types import AtomData, EdgeData, GraphBatch
from molrep.utils.equivariance import rotate_vectors, rotation_matrix_z
from molzoo.allegro import Allegro, AllegroLayer, PairEmbedding
from tests.utils import assert_compile_compatible


@pytest.fixture
def graph_data():
    n_nodes = 4
    edge_index = torch.tensor(
        [
            [0, 1],
            [1, 0],
            [1, 2],
            [2, 1],
            [2, 3],
            [3, 2],
        ],
        dtype=torch.long,
    )
    pos = torch.randn(n_nodes, 3)
    bond_diff = pos[edge_index[:, 1]] - pos[edge_index[:, 0]]
    bond_dist = bond_diff.norm(dim=-1).clamp(min=1e-4)
    Z = torch.randint(0, 5, (n_nodes,))
    n_edges = edge_index.shape[0]

    return GraphBatch(
        atoms=AtomData(
            Z=Z, pos=pos, batch=torch.zeros(n_nodes, dtype=torch.long),
            batch_size=[n_nodes],
        ),
        edges=EdgeData(
            edge_index=edge_index,
            bond_diff=bond_diff,
            bond_dist=bond_dist,
            batch_size=[n_edges],
        ),
        batch_size=[],
    )


class TestPairEmbedding:
    """PairEmbedding output shapes and compile compatibility."""

    def test_output_shapes(self, graph_data):
        module = PairEmbedding(
            num_elements=5,
            num_scalar_features=16,
            num_tensor_features=8,
            r_max=5.0,
            l_max=2,
        )
        scalars, tensors, edge_angular, edge_cutoff = module(
            Z=graph_data["atoms", "Z"],
            bond_dist=graph_data["edges", "bond_dist"],
            bond_diff=graph_data["edges", "bond_diff"],
            edge_index=graph_data["edges", "edge_index"],
        )

        n_edges = graph_data["edges", "edge_index"].shape[0]
        assert scalars.shape == (n_edges, 16)
        assert tensors.shape == (n_edges, module.irreps_dim)
        assert edge_angular.shape == (n_edges, 9)
        assert edge_cutoff.shape == (n_edges,)
        assert torch.all(edge_cutoff >= 0.0)
        assert torch.all(edge_cutoff <= 1.0)

    @pytest.mark.xfail(
        reason="cuEquivariance tensor products introduce graph breaks under fullgraph",
        strict=False,
    )
    def test_compile(self, graph_data):
        module = PairEmbedding(
            num_elements=5,
            num_scalar_features=16,
            num_tensor_features=8,
            r_max=5.0,
            l_max=2,
        )
        assert_compile_compatible(
            module,
            strict=False,
            Z=graph_data["atoms", "Z"],
            bond_dist=graph_data["edges", "bond_dist"],
            bond_diff=graph_data["edges", "bond_diff"],
            edge_index=graph_data["edges", "edge_index"],
        )


class TestAllegroLayer:
    """AllegroLayer shape preservation and compile compatibility."""

    def test_preserves_batch_dimension(self):
        num_scalar = 16
        num_tensor = 8
        # latent_in_dim for layer 0: 1 * num_scalar + num_tensor
        module = AllegroLayer(
            num_scalar_features=num_scalar,
            num_tensor_features=num_tensor,
            l_max=2,
            mlp_depth=1,
            latent_in_dim=num_scalar + num_tensor,
        )
        n_nodes = 4
        n_edges = 6
        edge_index = torch.tensor(
            [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]], dtype=torch.long
        )
        # accumulated_scalars for layer 0 = initial embedding scalars
        accumulated_scalars = torch.randn(n_edges, num_scalar)
        tensor_in = torch.randn(n_edges, module.irreps_dim)
        edge_angular = torch.randn(n_edges, 9)
        scalar_out, tensor_out = module(
            accumulated_scalars, tensor_in, edge_angular, edge_index, n_nodes
        )
        assert scalar_out.shape == (n_edges, num_scalar)
        assert tensor_out.shape == tensor_in.shape

    @pytest.mark.xfail(
        reason="cuEquivariance tensor products introduce graph breaks under fullgraph",
        strict=False,
    )
    def test_compile(self):
        num_scalar = 16
        num_tensor = 8
        module = AllegroLayer(
            num_scalar_features=num_scalar,
            num_tensor_features=num_tensor,
            l_max=2,
            mlp_depth=1,
            latent_in_dim=num_scalar + num_tensor,
        )
        n_nodes = 4
        n_edges = 6
        edge_index = torch.tensor(
            [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]], dtype=torch.long
        )
        accumulated_scalars = torch.randn(n_edges, num_scalar)
        tensor_in = torch.randn(n_edges, module.irreps_dim)
        edge_angular = torch.randn(n_edges, 9)
        assert_compile_compatible(
            module,
            accumulated_scalars,
            tensor_in,
            edge_angular,
            edge_index,
            n_nodes,
            strict=False,
        )


class TestAllegro:
    """Full Allegro encoder contract, equivariance, and compile."""

    def test_forward_encoder_contract(self, graph_data):
        encoder = Allegro(
            num_elements=5,
            num_scalar_features=16,
            num_tensor_features=8,
            r_max=5.0,
            l_max=2,
            num_layers=3,
            mlp_depth=1,
        )
        result = encoder(graph_data)
        edge_features = result["edges", "edge_features"]
        n_edges = graph_data["edges", "edge_index"].shape[0]
        assert edge_features.shape == (n_edges, 3, 16)

    def test_scalar_output_rotation_invariant(self, graph_data):
        encoder = Allegro(
            num_elements=5,
            num_scalar_features=16,
            num_tensor_features=8,
            r_max=5.0,
            l_max=2,
            num_layers=2,
            mlp_depth=1,
        )
        rotation = rotation_matrix_z(0.73)
        orig_diff = graph_data["edges", "bond_diff"]
        rotated_diff = rotate_vectors(orig_diff, rotation)
        rotated_dist = rotated_diff.norm(dim=-1).clamp(min=1e-4)

        n_nodes = graph_data["atoms", "Z"].shape[0]
        n_edges = graph_data["edges", "edge_index"].shape[0]
        rotated_batch = GraphBatch(
            atoms=AtomData(
                Z=graph_data["atoms", "Z"],
                pos=graph_data["atoms", "pos"],
                batch=graph_data["atoms", "batch"],
                batch_size=[n_nodes],
            ),
            edges=EdgeData(
                edge_index=graph_data["edges", "edge_index"],
                bond_diff=rotated_diff,
                bond_dist=rotated_dist,
                batch_size=[n_edges],
            ),
            batch_size=[],
        )

        base = encoder(graph_data)["edges", "edge_features"]
        rotated = encoder(rotated_batch)["edges", "edge_features"]
        assert torch.allclose(base, rotated, rtol=1e-4, atol=1e-4)

    @pytest.mark.xfail(
        reason="TensorDict/GraphBatch access patterns not yet fullgraph-compatible",
        strict=False,
    )
    def test_compile(self, graph_data):
        encoder = Allegro(
            num_elements=5,
            num_scalar_features=16,
            num_tensor_features=8,
            r_max=5.0,
            num_layers=2,
        )
        assert_compile_compatible(encoder, graph_data, strict=False)
