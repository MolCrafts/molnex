"""Tests for the Allegro encoder (faithful port of mir-group/allegro).

End-to-end physical-invariant checks of the encoder + ``EdgeEnergyHead``
energy pipeline. Module-level shape / parity tests have been retired
along with the previous ``AllegroLayer`` / ``PairEmbedding`` API; the
new encoder follows the reference's monolithic ``Allegro_Module`` layout
(DenseNet scalar accumulation, env weights sliced from the latent MLP
output) and exposes only the encoder ``Allegro`` class.

Coverage:
* Translation / rotation / permutation invariance of the total energy.
* Cutoff vanishing: edges past ``r_max`` produce zero contribution.
* Single-batch overfit: forward + backward must reduce loss to ~0.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from molpot.heads import EdgeEnergyHead
from molrep.utils.equivariance import (
    random_rotation_matrix,
    rotate_vectors,
)
from molzoo.allegro import Allegro

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_graph(
    pos: torch.Tensor,
    Z: torch.Tensor,
    r_cut: float,
    *,
    with_graphs: bool = True,
) -> TensorDict:
    """Build a full-connectivity TensorDict (all pairs within ``r_cut``)."""
    n = pos.shape[0]
    pairs = []
    diffs = []
    dists = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = pos[j] - pos[i]
            r = d.norm().item()
            if r < r_cut:
                pairs.append((i, j))
                diffs.append(d)
                dists.append(r)
    edge_index = torch.tensor(pairs, dtype=torch.long)
    bond_diff = torch.stack(diffs, dim=0)
    bond_dist = torch.tensor(dists, dtype=pos.dtype)

    atoms = TensorDict(
        Z=Z,
        pos=pos,
        batch=torch.zeros(n, dtype=torch.long),
        batch_size=[n],
    )
    edges = TensorDict(
        edge_index=edge_index,
        bond_diff=bond_diff,
        bond_dist=bond_dist,
        batch_size=[edge_index.shape[0]],
    )
    td = {"atoms": atoms, "edges": edges}
    if with_graphs:
        td["graphs"] = TensorDict(num_atoms=torch.tensor([n], dtype=torch.long), batch_size=[1])
    return TensorDict(**td, batch_size=[])


def _build_encoder(
    *,
    num_layers: int = 2,
    l_max: int = 2,
    r_cut: float = 5.0,
    avg_num_neighbors: float = 4.0,
    seed: int = 0,
) -> Allegro:
    torch.manual_seed(seed)
    return Allegro(
        num_elements=10,
        num_scalar_features=16,
        num_tensor_features=8,
        r_max=r_cut,
        num_bessel=4,
        l_max=l_max,
        num_layers=num_layers,
        type_embed_dim=16,
        latent_mlp_depth=1,
        latent_mlp_width=16,
        avg_num_neighbors=avg_num_neighbors,
    )


def _build_energy_model(encoder: Allegro, avg_nbr: float) -> nn.Module:
    """Compose the encoder with an EdgeEnergyHead readout."""
    head = EdgeEnergyHead(
        input_dim=encoder.output_dim,
        hidden_dim=16,
        avg_num_neighbors=avg_nbr,
    )

    class EnergyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.head = head

        def forward(self, batch: TensorDict) -> dict[str, torch.Tensor]:
            batch = self.encoder(batch)
            return self.head(batch)

    return EnergyModel()


# ---------------------------------------------------------------------------
# Encoder shape / output-dim contracts
# ---------------------------------------------------------------------------


class TestAllegroEncoder:
    def test_output_dim_is_densenet_stack(self):
        """Encoder output dim is ``F · (L + 1)`` (twobody scalar + per-layer)."""
        for num_layers in (1, 2, 3):
            enc = _build_encoder(num_layers=num_layers)
            assert enc.output_dim == enc.num_scalar_features * (num_layers + 1)

    def test_forward_writes_edge_features(self):
        torch.manual_seed(0)
        enc = _build_encoder(num_layers=2)
        pos = torch.randn(4, 3)
        Z = torch.randint(1, 5, (4,))
        g = _build_graph(pos, Z, r_cut=enc.r_max)
        out = enc(g)
        ef = out["edges", "edge_features"]
        assert ef.shape == (g["edges", "edge_index"].shape[0], enc.output_dim)


# ---------------------------------------------------------------------------
# Physical invariants of the full energy pipeline
# ---------------------------------------------------------------------------


class TestEnergyInvariants:
    def test_translation_invariance(self):
        torch.manual_seed(0)
        enc = _build_encoder()
        model = _build_energy_model(enc, avg_nbr=4.0)
        pos = torch.randn(4, 3)
        Z = torch.randint(1, 5, (4,))
        g1 = _build_graph(pos, Z, r_cut=enc.r_max)
        g2 = _build_graph(pos + torch.tensor([7.3, -2.1, 0.5]), Z, r_cut=enc.r_max)
        with torch.no_grad():
            e1 = model(g1)["energy"]
            e2 = model(g2)["energy"]
        assert torch.allclose(e1, e2, rtol=1e-4, atol=1e-5)

    def test_rotation_invariance(self):
        torch.manual_seed(1)
        enc = _build_encoder()
        model = _build_energy_model(enc, avg_nbr=4.0)
        pos = torch.randn(5, 3)
        Z = torch.randint(1, 5, (5,))
        g1 = _build_graph(pos, Z, r_cut=enc.r_max)
        torch.manual_seed(2)
        R = random_rotation_matrix()
        g2 = _build_graph(rotate_vectors(pos, R), Z, r_cut=enc.r_max)
        with torch.no_grad():
            e1 = model(g1)["energy"]
            e2 = model(g2)["energy"]
        assert torch.allclose(e1, e2, rtol=1e-4, atol=1e-4)

    def test_permutation_invariance(self):
        torch.manual_seed(2)
        enc = _build_encoder()
        model = _build_energy_model(enc, avg_nbr=4.0)
        pos = torch.randn(4, 3)
        Z = torch.randint(1, 5, (4,))
        g1 = _build_graph(pos, Z, r_cut=enc.r_max)
        perm = torch.tensor([2, 0, 3, 1], dtype=torch.long)
        g2 = _build_graph(pos[perm], Z[perm], r_cut=enc.r_max)
        with torch.no_grad():
            e1 = model(g1)["energy"]
            e2 = model(g2)["energy"]
        assert torch.allclose(e1, e2, rtol=1e-4, atol=1e-5)

    def test_cutoff_vanishing(self):
        """An edge past ``r_max`` contributes zero."""
        torch.manual_seed(3)
        r_cut = 3.0
        enc = _build_encoder(r_cut=r_cut, num_layers=1)
        model = _build_energy_model(enc, avg_nbr=2.0)
        # In-cutoff pair vs in-cutoff pair + far ghost atom.
        pos_a = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        Z_a = torch.tensor([1, 1], dtype=torch.long)
        pos_b = torch.cat([pos_a, torch.tensor([[10.0, 0.0, 0.0]])])
        Z_b = torch.cat([Z_a, torch.tensor([1], dtype=torch.long)])
        g_a = _build_graph(pos_a, Z_a, r_cut=r_cut)
        g_b = _build_graph(pos_b, Z_b, r_cut=r_cut)
        with torch.no_grad():
            e_a = model(g_a)["energy"]
            e_b = model(g_b)["energy"]
        # Both should give the same energy (no edge connects atom 2).
        assert torch.allclose(e_a, e_b, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Single-batch overfit (sanity that forward/backward train)
# ---------------------------------------------------------------------------


class TestOverfitSingleBatch:
    def test_overfit_constant_target(self):
        torch.manual_seed(42)
        pos = torch.tensor([[0.00, 0.00, 0.00], [0.96, 0.00, 0.00], [-0.24, 0.93, 0.00]])
        Z = torch.tensor([8, 1, 1])
        g = _build_graph(pos, Z, r_cut=3.0)

        encoder = Allegro(
            num_elements=10,
            num_scalar_features=16,
            num_tensor_features=8,
            r_max=3.0,
            num_bessel=4,
            l_max=1,
            num_layers=1,
            type_embed_dim=16,
            latent_mlp_depth=1,
            latent_mlp_width=16,
            avg_num_neighbors=6.0,
        )
        model = _build_energy_model(encoder, avg_nbr=6.0)
        target = torch.tensor([1.234])

        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        initial_loss = None
        for step in range(500):
            opt.zero_grad()
            pred = model(g.clone())["energy"]
            loss = (pred - target).pow(2).mean()
            if step == 0:
                initial_loss = loss.item()
            loss.backward()
            opt.step()
        final_loss = loss.item()
        assert final_loss < 1e-3, (
            f"single-batch overfit failed: initial={initial_loss:.3e}, final={final_loss:.3e}"
        )
