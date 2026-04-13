"""Tests for encoder-only MACE API."""

from __future__ import annotations

import pytest
import torch

from molrep.embedding.node import DiscreteEmbeddingSpec
from molzoo import MACE
from molix.data.types import AtomData, EdgeData, GraphBatch
from tests.utils import assert_compile_compatible


@pytest.fixture
def graph_data():
    n_nodes = 5
    edge_index = torch.tensor(
        [
            [0, 1],
            [1, 0],
            [1, 2],
            [2, 1],
            [2, 3],
            [3, 2],
            [3, 4],
            [4, 3],
        ],
        dtype=torch.long,
    )
    pos = torch.randn(n_nodes, 3)
    bond_diff = pos[edge_index[:, 1]] - pos[edge_index[:, 0]]
    bond_dist = bond_diff.norm(dim=-1).clamp(min=1e-4)
    n_edges = edge_index.shape[0]

    atoms = AtomData(
        Z=torch.randint(0, 6, (n_nodes,)),
        pos=pos,
        batch=torch.zeros(n_nodes, dtype=torch.long),
        batch_size=[n_nodes],
    )
    edges = EdgeData(
        edge_index=edge_index,
        bond_diff=bond_diff,
        bond_dist=bond_dist,
        batch_size=[n_edges],
    )
    return GraphBatch(atoms=atoms, edges=edges, batch_size=[])


def _build_encoder() -> MACE:
    return MACE(
        node_attr_specs=[
            DiscreteEmbeddingSpec(
                input_key="Z",
                num_classes=6,
                emb_dim=16,
            )
        ],
        num_elements=6,
        num_features=16,
        r_max=5.0,
        num_interactions=2,
        l_max=2,
    )


class TestMACE:
    """Full MACE encoder contract and compile compatibility."""

    def test_forward_encoder_contract(self, graph_data):
        encoder = _build_encoder()
        output = encoder(graph_data)
        node_features = output["atoms", "node_features"]
        n_nodes = graph_data["atoms", "Z"].shape[0]
        assert isinstance(node_features, torch.Tensor)
        assert node_features.shape == (n_nodes, 2, 16)

    @pytest.mark.xfail(
        reason="TensorDict access + cuEquivariance not yet fullgraph-compatible",
        strict=False,
    )
    def test_compile(self, graph_data):
        encoder = _build_encoder()
        assert_compile_compatible(encoder, graph_data, strict=False)
