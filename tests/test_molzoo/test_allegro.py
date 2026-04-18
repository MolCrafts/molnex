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
            strict=True,
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
        module = AllegroLayer(
            num_scalar_features=num_scalar,
            num_tensor_features=num_tensor,
            l_max=2,
        )
        n_nodes = 4
        n_edges = 6
        edge_index = torch.tensor(
            [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]], dtype=torch.long
        )
        scalar_features = torch.randn(n_edges, num_scalar)
        tensor_in = torch.randn(n_edges, module.irreps_dim)
        edge_angular = torch.randn(n_edges, 9)
        scalar_out, tensor_out = module(
            scalar_features, tensor_in, edge_angular, edge_index, n_nodes
        )
        assert scalar_out.shape == (n_edges, num_scalar)
        assert tensor_out.shape == tensor_in.shape

    def test_linear_latent_mlp(self):
        """``latent_activation=None`` yields a pure linear stack (3BPA setup)."""
        module = AllegroLayer(
            num_scalar_features=16,
            num_tensor_features=8,
            l_max=2,
            latent_mlp_hiddens=[32, 32, 32],
            latent_activation=None,
        )
        activations = [
            m for m in module.latent_mlp if not isinstance(m, torch.nn.Linear)
        ]
        assert activations == []

    def test_avg_num_neighbors_constant(self):
        """When ``avg_num_neighbors`` is set the layer stores the constant."""
        module = AllegroLayer(
            num_scalar_features=16,
            num_tensor_features=8,
            l_max=2,
            avg_num_neighbors=4.0,
        )
        assert module.avg_num_neighbors == 4.0

    def test_residual_alpha_coefficients(self):
        module = AllegroLayer(
            num_scalar_features=16,
            num_tensor_features=8,
            l_max=2,
            residual_alpha=0.5,
        )
        # a = 1/√1.25, b = 0.5/√1.25, and a² + b² = 1.
        assert abs(module.residual_a ** 2 + module.residual_b ** 2 - 1.0) < 1e-6

    def test_compile(self):
        num_scalar = 16
        num_tensor = 8
        module = AllegroLayer(
            num_scalar_features=num_scalar,
            num_tensor_features=num_tensor,
            l_max=2,
        )
        n_nodes = 4
        n_edges = 6
        edge_index = torch.tensor(
            [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]], dtype=torch.long
        )
        scalar_features = torch.randn(n_edges, num_scalar)
        tensor_in = torch.randn(n_edges, module.irreps_dim)
        edge_angular = torch.randn(n_edges, 9)
        assert_compile_compatible(
            module,
            scalar_features,
            tensor_in,
            edge_angular,
            edge_index,
            n_nodes,
            strict=True,
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

    def test_compile(self, graph_data):
        """Full encoder compiles under strict fullgraph=True (inductor).

        Requires cuequivariance >= 0.10.0rc4 (PR #270 fixed the naive-CPU
        ``Subscripts.__add__`` graph break).  We compare the ``edge_features``
        tensor directly because ``assert_outputs_close`` cannot compare the
        full TensorDict output (TensorDict has no boolean ``==``).
        """
        encoder = Allegro(
            num_elements=5,
            num_scalar_features=16,
            num_tensor_features=8,
            r_max=5.0,
            num_layers=2,
        ).eval()

        torch._dynamo.reset()
        compiled = torch.compile(encoder, backend="inductor", fullgraph=True)

        with torch.no_grad():
            ref = encoder(graph_data)["edges", "edge_features"]
            got = compiled(graph_data)["edges", "edge_features"]
        assert torch.allclose(ref, got, rtol=1e-4, atol=1e-4)
