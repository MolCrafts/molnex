"""Tests for modular potential composition."""

from __future__ import annotations

import pytest
import torch

from molix.data.types import AtomData, EdgeData, GraphBatch
from molpot.pooling import EdgeToNodePooling, LayerPooling
from molpot.composition import LJParameterHead, PotentialComposer
from molpot.potentials import LJ126
from molrep.embedding.node import DiscreteEmbeddingSpec
from molzoo import Allegro, MACE


@pytest.fixture
def molecular_batch():
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.4, 0.0, 0.0],
            [2.8, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    edge_index = torch.tensor(
        [
            [0, 1],
            [1, 0],
            [1, 2],
            [2, 1],
            [3, 4],
            [4, 3],
        ],
        dtype=torch.long,
    )
    bond_diff = pos[edge_index[:, 1]] - pos[edge_index[:, 0]]
    bond_dist = bond_diff.norm(dim=-1)
    batch_idx = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    Z = torch.tensor([1, 6, 8, 1, 6], dtype=torch.long)

    return GraphBatch(
        atoms=AtomData(
            Z=Z, pos=pos, batch=batch_idx,
            batch_size=[5],
        ),
        edges=EdgeData(
            edge_index=edge_index,
            bond_diff=bond_diff,
            bond_dist=bond_dist,
            batch_size=[6],
        ),
        batch_size=[],
    )


def test_layer_pooling_mean():
    pool = LayerPooling("mean")
    features_3d = torch.randn(5, 3, 8)
    out = pool(features_3d)
    assert out.shape == (5, 8)
    assert torch.allclose(out, features_3d.mean(dim=1))

    features_2d = torch.randn(5, 8)
    out = pool(features_2d)
    assert out.shape == (5, 8)
    assert torch.equal(out, features_2d)


def test_layer_pooling_sum():
    pool = LayerPooling("sum")
    features = torch.randn(5, 3, 8)
    out = pool(features)
    assert torch.allclose(out, features.sum(dim=1))


def test_layer_pooling_last():
    pool = LayerPooling("last")
    features = torch.randn(5, 3, 8)
    out = pool(features)
    assert torch.equal(out, features[:, -1])


def test_edge_to_node_pooling():
    edge_features = torch.arange(4 * 2, dtype=torch.float32).reshape(4, 2)
    edge_index = torch.tensor([[0, 1], [1, 0], [1, 2], [2, 1]], dtype=torch.long)

    pool = EdgeToNodePooling("mean")
    node_features = pool(edge_features, edge_index, num_nodes=3)
    assert node_features.shape == (3, 2)

    pool_sum = EdgeToNodePooling("sum")
    node_features_sum = pool_sum(edge_features, edge_index, num_nodes=3)
    assert node_features_sum.shape == (3, 2)


def test_parameter_head_outputs_positive_lj_parameters():
    pool = LayerPooling("mean")
    head = LJParameterHead(feature_dim=16, hidden_dim=32)

    features = pool(torch.randn(6, 3, 16))
    params = head(features)

    assert params["epsilon"].shape == (6,)
    assert params["sigma"].shape == (6,)
    assert torch.all(params["epsilon"] > 0.0)
    assert torch.all(params["sigma"] > 0.0)


def test_potential_composer_aggregates_terms_and_derives_forces(molecular_batch):
    pool = LayerPooling("mean")
    node_features = pool(torch.randn(5, 3, 16))

    composer = PotentialComposer(
        head=LJParameterHead(feature_dim=16, hidden_dim=32),
        potentials={
            "lj_main": LJ126(energy_scale=1.0),
            "lj_aux": LJ126(energy_scale=0.25),
        },
    )

    data = {
        "pos": molecular_batch["atoms", "pos"].clone().requires_grad_(True),
        "edge_index": molecular_batch["edges", "edge_index"],
        "batch": molecular_batch["atoms", "batch"],
    }
    outputs = composer(node_features=node_features, data=data, compute_forces=True)

    assert outputs["energy"].shape == (2,)
    assert outputs["forces"].shape == (5, 3)
    term_sum = outputs["term_energies"]["lj_main"] + outputs["term_energies"]["lj_aux"]
    assert torch.allclose(outputs["energy"], term_sum, atol=1e-6, rtol=1e-6)


def test_integration_with_mace_encoder(molecular_batch):
    encoder = MACE(
        node_attr_specs=[DiscreteEmbeddingSpec(input_key="Z", num_classes=10, emb_dim=16)],
        num_elements=10,
        num_features=16,
        r_max=5.0,
        num_interactions=2,
    )
    result = encoder(molecular_batch)

    pool = LayerPooling("mean")
    node_features = pool(result["atoms", "node_features"])

    composer = PotentialComposer(
        head=LJParameterHead(feature_dim=16, hidden_dim=32),
        potentials={"lj126": LJ126()},
    )
    data = {
        "pos": molecular_batch["atoms", "pos"].clone().requires_grad_(True),
        "edge_index": molecular_batch["edges", "edge_index"],
        "batch": molecular_batch["atoms", "batch"],
    }
    outputs = composer(node_features=node_features, data=data, compute_forces=True)
    assert outputs["energy"].shape == (2,)
    assert outputs["forces"].shape == (5, 3)


def test_integration_with_allegro_encoder(molecular_batch):
    encoder = Allegro(
        num_elements=10,
        num_scalar_features=16,
        num_tensor_features=8,
        r_max=5.0,
        num_layers=2,
    )
    result = encoder(molecular_batch)

    layer_pool = LayerPooling("mean")
    edge_to_node = EdgeToNodePooling("mean")

    edge_features = layer_pool(result["edges", "edge_features"])
    node_features = edge_to_node(
        edge_features,
        molecular_batch["edges", "edge_index"],
        num_nodes=molecular_batch["atoms", "Z"].shape[0],
    )

    composer = PotentialComposer(
        head=LJParameterHead(feature_dim=16, hidden_dim=32),
        potentials={"lj126": LJ126()},
    )
    data = {
        "pos": molecular_batch["atoms", "pos"].clone().requires_grad_(True),
        "edge_index": molecular_batch["edges", "edge_index"],
        "batch": molecular_batch["atoms", "batch"],
    }
    outputs = composer(node_features=node_features, data=data, compute_forces=True)
    assert outputs["energy"].shape == (2,)
    assert outputs["forces"].shape == (5, 3)
