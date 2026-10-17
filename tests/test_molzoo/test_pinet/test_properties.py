"""Unit tests for molzoo.pinet.properties."""

from __future__ import annotations

import pytest
import torch

from molzoo.pinet import PiNetDipole, PiNetPolarizability, PiNetSpec
from tests.conftest import random_rotation_matrix, rotate_graph, rotate_vectors, translate_graph
from tests.test_molzoo.test_pinet.conftest import TINY

ATOL = 1e-10


def _dipole(variant: str) -> PiNetDipole:
    torch.manual_seed(4)
    return PiNetDipole(PiNetSpec(**TINY, rank=3), variant=variant)


def _polar(variant: str) -> PiNetPolarizability:
    torch.manual_seed(4)
    return PiNetPolarizability(PiNetSpec(**TINY, rank=3), variant=variant)


class TestPiNetDipole:
    def test_ac_charges_are_projected_to_the_total_charge(self, cluster):
        out = _dipole("ac")(cluster)
        torch.testing.assert_close(
            out["graphs", "charge_sum_post_proj"],
            torch.zeros(1, dtype=torch.float64),
            atol=1e-12,
            rtol=0,
        )

    def test_ac_dipole_is_translation_invariant(self, cluster):
        model = _dipole("ac")
        ref = model(cluster.clone())["graphs", "molecular_dipole"]
        shift = torch.tensor([5.0, -2.0, 1.0], dtype=torch.float64)
        got = model(translate_graph(cluster, shift))["graphs", "molecular_dipole"]
        torch.testing.assert_close(got, ref, atol=ATOL, rtol=0)

    def test_ad_dipole_rotates_with_the_molecule(self, cluster):
        model = _dipole("ad")
        torch.manual_seed(5)
        r = random_rotation_matrix(dtype=torch.float64)
        ref = model(cluster.clone())["graphs", "molecular_dipole"]
        got = model(rotate_graph(cluster, r))["graphs", "molecular_dipole"]
        torch.testing.assert_close(got, rotate_vectors(ref, r), atol=ATOL, rtol=0)

    def test_bc_writes_bond_charges_and_the_penalty(self, cluster):
        out = _dipole("bc")(cluster)
        assert out["pairs", "bond_charges"].shape == (12,)
        assert out["bond_charge_l2"].ndim == 0


class TestPiNetPolarizability:
    def test_localchi_is_symmetric_and_charge_conserving(self, cluster):
        out = _polar("localchi")(cluster)
        chi = out["graphs", "chi"]
        torch.testing.assert_close(chi, chi.transpose(-1, -2), atol=ATOL, rtol=0)
        torch.testing.assert_close(chi.sum(-1), torch.zeros_like(chi.sum(-1)), atol=1e-9, rtol=0)
        assert out["graphs", "alpha"].shape == (1, 3, 3)

    def test_alpha_rotates_as_a_rank_two_tensor(self, cluster):
        model = _polar("localchi")
        torch.manual_seed(7)
        r = random_rotation_matrix(dtype=torch.float64)
        ref = model(cluster.clone())["graphs", "alpha"]
        got = model(rotate_graph(cluster, r))["graphs", "alpha"]
        torch.testing.assert_close(got, r @ ref @ r.T, atol=1e-9, rtol=0)

    @pytest.mark.parametrize("variant", ["etainv_iso", "eem", "acks2", "local"])
    def test_every_response_variant_runs(self, variant, cluster):
        out = _polar(variant)(cluster)
        assert out["graphs", "alpha"].shape == (1, 3, 3)
        assert out["graphs", "chi"].shape == (1, 4, 4)
