"""Degree, π count and the VSEPR reference angle from the bond list alone."""

from __future__ import annotations

import math

import torch

from molrep.chem.hybridization import Hybridization


def _propyne():
    """H3C–C≡C–H: atoms 0 C(sp3), 1 C(sp), 2 C(sp), 3 H(on 2), 4-6 H(on 0)."""
    bond_index = torch.tensor([[0, 1, 2, 0, 0, 0], [1, 2, 3, 4, 5, 6]])
    bond_type = torch.tensor([1, 3, 1, 1, 1, 1])
    return bond_index, bond_type


class TestHybridization:
    def test_degree_counts_live_bonds_only(self):
        bond_index, bond_type = _propyne()
        mask = torch.tensor([True, True, True, True, True, False])
        degree, _ = Hybridization()(bond_index, bond_type, mask, 7)
        assert degree.tolist() == [3.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.0]

    def test_a_triple_bond_makes_a_linear_centre(self):
        bond_index, bond_type = _propyne()
        degree, pi = Hybridization()(bond_index, bond_type, None, 7)
        assert pi.tolist()[:3] == [0.0, 2.0, 2.0]
        assert Hybridization.linear(degree, pi).tolist()[:3] == [False, True, True]

    def test_an_aromatic_carbon_is_trigonal(self):
        # One benzene carbon with two aromatic ring bonds and one hydrogen.
        bond_index = torch.tensor([[0, 0, 0], [1, 2, 3]])
        bond_type = torch.tensor([4, 4, 1])
        hyb = Hybridization()
        degree, pi = hyb(bond_index, bond_type, None, 4)
        assert degree[0].item() == 3.0 and pi[0].item() == 1.0
        assert math.isclose(math.degrees(hyb.angle_reference(degree, pi)[0].item()), 120.0)

    def test_water_oxygen_is_not_linear(self):
        bond_index = torch.tensor([[0, 0], [1, 2]])
        bond_type = torch.tensor([1, 1])
        hyb = Hybridization()
        degree, pi = hyb(bond_index, bond_type, None, 3)
        assert not Hybridization.linear(degree, pi)[0]
        assert math.isclose(math.degrees(hyb.angle_reference(degree, pi)[0].item()), 109.47)

    def test_the_linear_reference_is_exactly_180_degrees(self):
        """Only θ0 = 180° makes the harmonic bend smooth at a linear centre."""
        bond_index, bond_type = _propyne()
        hyb = Hybridization()
        degree, pi = hyb(bond_index, bond_type, None, 7)
        ref = math.degrees(hyb.angle_reference(degree, pi)[1].item())
        assert math.isclose(ref, 180.0)
