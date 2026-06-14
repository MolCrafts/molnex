"""Tests for ``molpot.heads.EdgeEnergyHead``."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from molpot.heads import EdgeEnergyHead


def _stub_batch(*, n_atoms=3, n_edges=6, feat_dim=8):
    torch.manual_seed(0)
    edge_index = torch.tensor([[0, 1], [1, 0], [1, 2], [2, 1], [0, 2], [2, 0]], dtype=torch.long)[
        :n_edges
    ]
    atoms = TensorDict(
        Z=torch.tensor([1, 6, 8][:n_atoms]),
        pos=torch.randn(n_atoms, 3),
        batch=torch.zeros(n_atoms, dtype=torch.long),
        batch_size=[n_atoms],
    )
    edges = TensorDict(
        edge_index=edge_index,
        bond_diff=torch.randn(n_edges, 3),
        bond_dist=torch.rand(n_edges) + 0.5,
        batch_size=[n_edges],
    )
    edges["edge_features"] = torch.randn(n_edges, feat_dim)
    graphs = TensorDict(num_atoms=torch.tensor([n_atoms]), batch_size=[1])
    return TensorDict(atoms=atoms, edges=edges, graphs=graphs, batch_size=[])


class TestShape:
    def test_returns_per_graph_energy(self):
        head = EdgeEnergyHead(input_dim=8, hidden_dim=4, avg_num_neighbors=3.0)
        out = head(_stub_batch())
        assert set(out.keys()) == {"energy"}
        assert out["energy"].shape == (1,)

    def test_custom_out_key(self):
        head = EdgeEnergyHead(input_dim=8, hidden_dim=4, avg_num_neighbors=3.0, out_key="E")
        out = head(_stub_batch())
        assert "E" in out


class TestNeighborNormalization:
    def test_sqrt_normalization_is_applied(self):
        """Halving ``avg_num_neighbors`` scales edge energy by √2."""
        b = _stub_batch()
        # two heads with different avg_num_neighbors but identical parameters
        h1 = EdgeEnergyHead(input_dim=8, hidden_dim=4, avg_num_neighbors=4.0)
        h2 = EdgeEnergyHead(input_dim=8, hidden_dim=4, avg_num_neighbors=16.0)
        h2.load_state_dict(h1.state_dict())
        with torch.no_grad():
            e1 = h1(b)["energy"]
            e2 = h2(b)["energy"]
        # √16 / √4 = 2 → e1 should be twice e2.
        torch.testing.assert_close(e1, 2.0 * e2, rtol=1e-5, atol=1e-5)

    def test_rejects_non_positive_avg_num_neighbors(self):
        with pytest.raises(ValueError, match="avg_num_neighbors"):
            EdgeEnergyHead(input_dim=8, hidden_dim=4, avg_num_neighbors=0.0)


class TestLinearReadout:
    """Upstream ``edge_eng_mlp_nonlinearity=null`` ⇒ readout is purely linear.

    With ``ScalarMLPFunction(nonlinearity=None)``, two ``Linear`` layers in
    series collapse to a single linear map of the input. Energy must therefore
    be linear in ``edge_features`` (up to scatter-sum, which is also linear).
    """

    def test_energy_is_linear_in_edge_features(self):
        torch.manual_seed(7)
        b = _stub_batch()
        head = EdgeEnergyHead(input_dim=8, hidden_dim=4, avg_num_neighbors=3.0)

        # Energy at (a) features f, (b) features 2f, (c) features f + g should
        # satisfy E(2f) = 2 E(f) and E(f+g) = E(f) + E(g) (within numerical noise).
        f = b["edges", "edge_features"]
        g = torch.randn_like(f)

        b1 = b.clone()
        b1["edges", "edge_features"] = f
        b2 = b.clone()
        b2["edges", "edge_features"] = 2.0 * f
        b3 = b.clone()
        b3["edges", "edge_features"] = f + g
        b4 = b.clone()
        b4["edges", "edge_features"] = g

        with torch.no_grad():
            e1 = head(b1)["energy"]
            e2 = head(b2)["energy"]
            e3 = head(b3)["energy"]
            e4 = head(b4)["energy"]

        torch.testing.assert_close(e2, 2.0 * e1, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(e3, e1 + e4, rtol=1e-5, atol=1e-5)


class TestEmptyGraph:
    def test_zero_edges_give_zero_energy(self):
        """Graph with no edges → scatter sum is zero tensor."""
        atoms = TensorDict(
            Z=torch.tensor([1]),
            pos=torch.zeros(1, 3),
            batch=torch.zeros(1, dtype=torch.long),
            batch_size=[1],
        )
        edges = TensorDict(
            edge_index=torch.empty(0, 2, dtype=torch.long),
            bond_diff=torch.empty(0, 3),
            bond_dist=torch.empty(0),
            batch_size=[0],
        )
        edges["edge_features"] = torch.empty(0, 8)
        graphs = TensorDict(num_atoms=torch.tensor([1]), batch_size=[1])
        b = TensorDict(atoms=atoms, edges=edges, graphs=graphs, batch_size=[])

        head = EdgeEnergyHead(input_dim=8, hidden_dim=4, avg_num_neighbors=1.0)
        with torch.no_grad():
            out = head(b)
        assert out["energy"].shape == (1,)
        assert out["energy"].item() == 0.0
