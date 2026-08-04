"""Tests for PiNet prediction models."""

from __future__ import annotations

import pytest
import torch

from molrep.utils.equivariance import random_rotation_matrix, rotate_vectors
from molzoo import PiNet
from molzoo.pinet import PiNetDipole, PiNetPolarizability, PiNetPotential
from tests.conftest import make_graph_batch, rotate_graph, translate_graph


def _graph(total_charge: float = 0.0):
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.1, 0.0, 0.0],
            [0.2, 1.0, 0.1],
            [1.0, 1.1, -0.1],
        ],
        dtype=torch.float32,
    )
    z = torch.tensor([1, 6, 7, 8], dtype=torch.long)
    edge_index = torch.tensor(
        [[0, 1], [1, 0], [0, 2], [2, 0], [1, 3], [3, 1], [2, 3], [3, 2]],
        dtype=torch.long,
    )
    batch = torch.zeros(4, dtype=torch.long)
    return make_graph_batch(
        pos,
        z,
        edge_index,
        batch,
        graphs={"total_charge": torch.tensor([total_charge], dtype=torch.float32)},
    )


def _encoder(rank: int = 3) -> PiNet:
    torch.manual_seed(4)
    return PiNet(
        atom_types=[1, 6, 7, 8],
        r_max=4.0,
        n_basis=3,
        pp_nodes=[8, 8],
        pi_nodes=[8, 8],
        ii_nodes=[8, 8],
        depth=2,
        rank=rank,
    )


class TestPiNetPotential:
    def test_energy_and_force_shapes(self):
        model = PiNetPotential(
            atom_types=[1, 6, 7, 8],
            r_max=4.0,
            n_basis=3,
            pp_nodes=[8, 8],
            pi_nodes=[8, 8],
            ii_nodes=[8, 8],
            depth=2,
            rank=3,
            hidden_dim=8,
        )
        g = _graph()
        g["atoms", "pos"] = g["atoms", "pos"].clone().requires_grad_(True)
        out = model(g, compute_forces=True)
        assert out["energy"].shape == (1,)
        assert out["forces"].shape == (4, 3)
        (out["energy"].sum() + out["forces"].square().sum()).backward()


class TestPiNetDipole:
    def test_ac_charge_projection_is_neutral(self):
        enc = _encoder(rank=3)
        model = PiNetDipole(encoder=enc, variant="ac")
        out = model(_graph())
        torch.testing.assert_close(out["charge_sum_post_proj"], torch.zeros(1), atol=1e-6, rtol=0)

    def test_ac_translation_invariance_after_neutralization(self):
        enc = _encoder(rank=3)
        model = PiNetDipole(encoder=enc, variant="ac")
        g = _graph()
        with torch.no_grad():
            ref = model(g.clone())["molecular_dipole"]
            shifted = model(translate_graph(g, torch.tensor([5.0, -2.0, 1.0])))["molecular_dipole"]
        torch.testing.assert_close(ref, shifted, atol=1e-5, rtol=1e-5)

    def test_ad_rotation_equivariance(self):
        enc = _encoder(rank=3)
        model = PiNetDipole(encoder=enc, variant="ad")
        g = _graph()
        torch.manual_seed(5)
        r = random_rotation_matrix()
        with torch.no_grad():
            ref = model(g.clone())["molecular_dipole"]
            got = model(rotate_graph(g, r))["molecular_dipole"]
        torch.testing.assert_close(rotate_vectors(ref, r), got, atol=1e-4, rtol=1e-4)

    def test_bc_outputs_regularization(self):
        enc = _encoder(rank=3)
        model = PiNetDipole(encoder=enc, variant="bc")
        out = model(_graph())
        assert out["bond_charges"].shape == (8,)
        assert out["bond_charge_l2"].ndim == 0


class TestPiNetPolarizability:
    def test_localchi_symmetry_and_sum_rule(self):
        enc = _encoder(rank=3)
        model = PiNetPolarizability(encoder=enc, variant="localchi")
        out = model(_graph())
        chi = out["chi"]
        torch.testing.assert_close(chi, chi.transpose(-1, -2), atol=1e-6, rtol=1e-6)
        chi_sum = chi.sum(dim=-1)
        torch.testing.assert_close(chi_sum, torch.zeros_like(chi_sum), atol=1e-5, rtol=1e-5)
        assert out["alpha"].shape == (1, 3, 3)

    def test_polarizability_rotation_covariance(self):
        enc = _encoder(rank=3)
        model = PiNetPolarizability(encoder=enc, variant="localchi")
        g = _graph()
        torch.manual_seed(7)
        r = random_rotation_matrix()
        with torch.no_grad():
            ref = model(g.clone())["alpha"]
            got = model(rotate_graph(g, r))["alpha"]
        expected = r @ ref @ r.T
        torch.testing.assert_close(expected, got, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("variant", ["etainv_iso", "eem", "acks2", "local"])
    def test_charge_response_variants_output_shapes(self, variant):
        enc = _encoder(rank=3)
        model = PiNetPolarizability(encoder=enc, variant=variant)
        out = model(_graph())
        assert out["alpha"].shape == (1, 3, 3)
        assert out["chi"].shape == (1, 4, 4)
