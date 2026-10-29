"""The bonded terms: symmetric in the orders that name one term, converted to molrs once."""

import pytest
import torch
from tensordict import TensorDict

from molpot.classical import (
    AngleHarmonic,
    BondChargeCorrection,
    BondHarmonic,
    DihedralPeriodic,
    ImproperCvff,
)
from molpot.potentials.angle import AngleHarmonic as AnglePotential
from molpot.potentials.bond import BondHarmonic as BondPotential
from molpot.potentials.dihedral import DihedralPeriodic as DihedralPotential
from molpot.potentials.improper import ImproperCvff as ImproperPotential
from molrep.encoders.features import EncoderFeatures

DTYPE = torch.float64
SETTINGS = dict(
    hidden_dim=8, width=8, layers=2, out_hidden_dim=8, out_layers=2, tanh_output=20.0, dtype=DTYPE
)

#: 0-1-2-3 chain with a branch 1-4; atom 1 is a three-coordinate centre (legs 0, 2, 4).
BONDS = [(0, 1), (1, 2), (2, 3), (1, 4)]


def _frame(
    *,
    Z=(6, 6, 6, 6, 1),
    bonds=BONDS,
    angles=((0, 1, 2),),
    propers=((0, 1, 2, 3),),
    impropers=((1, 0, 2, 4),),
    linear=(False,),
) -> TensorDict:
    def block(rows, names):
        rows = torch.tensor(rows, dtype=torch.long).reshape(-1, len(names))
        return {name: rows[:, n] for n, name in enumerate(names)}, rows.shape[0]

    angle_block, n_angles = block(angles, ("atomi", "atomj", "atomk"))
    proper_block, n_propers = block(propers, ("atomi", "atomj", "atomk", "atoml"))
    proper_block["linear"] = torch.tensor(linear, dtype=torch.bool)
    improper_block, n_impropers = block(impropers, ("atomi", "atomj", "atomk", "atoml"))
    return TensorDict(
        {
            "atoms": TensorDict({"Z": torch.tensor(Z)}, batch_size=[len(Z)]),
            "bonds": TensorDict(
                {"bond_index": torch.tensor(bonds, dtype=torch.long).T.contiguous()}, batch_size=[]
            ),
            "angles": TensorDict(angle_block, batch_size=[n_angles]),
            "propers": TensorDict(proper_block, batch_size=[n_propers]),
            "impropers": TensorDict(improper_block, batch_size=[n_impropers]),
        },
        batch_size=[],
    )


def _features(n_atoms: int, n_bonds: int) -> EncoderFeatures:
    generator = torch.Generator().manual_seed(0)
    return EncoderFeatures(
        atom=torch.randn(n_atoms, 4, generator=generator, dtype=DTYPE),
        bond=torch.randn(n_bonds, 3, generator=generator, dtype=DTYPE),
    )


def _at_the_mean(term) -> None:
    """Zero every output linear, so the head's raw output is 0 and the parameter its window mean."""
    with torch.no_grad():
        for mlp in term.out.values():
            mlp[1].weight.zero_()
            mlp[1].bias.zero_()


class TestBondHarmonic:
    def test_reversing_a_bond_gives_the_same_parameters(self):
        term = BondHarmonic(4, 3, **SETTINGS)
        features = _features(5, 4)
        flipped = [(j, i) for i, j in BONDS]
        one = term.predict(features, _frame(), BondPotential())
        two = term.predict(features, _frame(bonds=flipped), BondPotential())
        for name in ("k", "r0"):
            torch.testing.assert_close(one[name], two[name])

    def test_k_is_half_the_head_k(self):
        # The head predicts k of ½k(r − r0)²; LAMMPS harmonic is K(r − r0)².
        term = BondHarmonic(4, 3, **SETTINGS)
        _at_the_mean(term)
        out = term.predict(_features(5, 4), _frame(), BondPotential())
        torch.testing.assert_close(out["k"], torch.full((4,), 350.0, dtype=DTYPE))
        torch.testing.assert_close(out["r0"], torch.full((4,), 1.3, dtype=DTYPE))

    def test_a_bond_to_a_hypervalent_atom_takes_the_patch(self):
        bonds = [(0, n) for n in range(1, 6)] + [(5, 6)]
        frame = _frame(
            Z=(15, 9, 9, 9, 9, 6, 1), bonds=bonds, angles=(), propers=(), impropers=(), linear=()
        )
        out = BondHarmonic(4, 3, **SETTINGS).predict(_features(7, 6), frame, BondPotential())
        torch.testing.assert_close(out["k"][:5], torch.full((5,), 500.0, dtype=DTYPE))
        assert out["r0"][:5].abs().max() == 0.0 and out["r0"][5] > 0.0


class TestAngleHarmonic:
    def test_reversing_an_angle_gives_the_same_parameters(self):
        term = AngleHarmonic(4, 3, **SETTINGS)
        features = _features(5, 4)
        one = term.predict(features, _frame(angles=((0, 1, 2),)), AnglePotential())
        two = term.predict(features, _frame(angles=((2, 1, 0),)), AnglePotential())
        for name in ("k", "theta0"):
            torch.testing.assert_close(one[name], two[name])

    def test_k_is_half_the_head_k_and_theta0_is_in_degrees(self):
        term = AngleHarmonic(4, 3, **SETTINGS)
        _at_the_mean(term)
        out = term.predict(_features(5, 4), _frame(), AnglePotential())
        torch.testing.assert_close(out["k"], torch.full((1,), 65.0, dtype=DTYPE))
        torch.testing.assert_close(out["theta0"], torch.full((1,), 120.0, dtype=DTYPE))


class TestDihedralPeriodic:
    def test_reversing_a_torsion_gives_the_same_parameters(self):
        term = DihedralPeriodic(4, 3, **SETTINGS)
        features = _features(5, 4)
        one = term.predict(features, _frame(propers=((0, 1, 2, 3),)), DihedralPotential())
        two = term.predict(features, _frame(propers=((3, 2, 1, 0),)), DihedralPotential())
        torch.testing.assert_close(one["k"], two["k"])

    def test_a_torsion_through_a_linear_centre_is_silent(self):
        term = DihedralPeriodic(4, 3, **SETTINGS)
        frame = _frame(propers=((0, 1, 2, 3), (0, 1, 2, 3)), linear=(False, True))
        k = term.predict(_features(5, 4), frame, DihedralPotential())["k"]
        assert k.shape == (2, 4) and k[1].abs().max() == 0.0 and k[0].abs().max() > 0.0

    def test_the_series_is_n_1_to_4_at_phases_0_and_180(self):
        term = DihedralPeriodic(4, 3, **SETTINGS)
        params = term.potential_params({"propers": {"k": torch.zeros(2, 4, dtype=DTYPE)}})
        assert params["periodicity"][0].tolist() == [1.0, 2.0, 3.0, 4.0]
        assert params["phase"][0].tolist() == [0.0, 180.0, 0.0, 180.0]

    def test_a_batch_without_the_linear_column_is_refused_by_name(self):
        frame = _frame()
        del frame["propers", "linear"]
        with pytest.raises(KeyError, match="propers.linear"):
            DihedralPeriodic(4, 3, **SETTINGS).predict(_features(5, 4), frame, DihedralPotential())


class TestImproperCvff:
    def test_the_order_of_the_legs_does_not_change_k(self):
        term = ImproperCvff(4, 3, **SETTINGS)
        features = _features(5, 4)
        one = term.predict(features, _frame(impropers=((1, 0, 2, 4),)), ImproperPotential())
        two = term.predict(features, _frame(impropers=((1, 4, 0, 2),)), ImproperPotential())
        torch.testing.assert_close(one["k"], two["k"])

    def test_an_explicit_chemical_centre_gives_the_same_k_in_amber_order(self):
        term = ImproperCvff(4, 3, **SETTINGS)
        features = _features(5, 4)
        expected = term.predict(features, _frame(impropers=((1, 0, 2, 4),)), ImproperPotential())
        amber = _frame(impropers=((0, 2, 1, 4),))  # centre stored third
        for name, value in zip(("centre", "leg1", "leg2", "leg3"), (1, 0, 2, 4), strict=True):
            amber["impropers", name] = torch.tensor([value])
        actual = term.predict(features, amber, ImproperPotential())
        torch.testing.assert_close(actual["k"], expected["k"])

    def test_only_a_carbon_or_nitrogen_centre_carries_the_term(self):
        torch.manual_seed(0)
        term = ImproperCvff(4, 3, **SETTINGS)
        features = _features(5, 4)
        k = {
            z: term.predict(features, _frame(Z=(6, z, 6, 6, 1)), ImproperPotential())["k"]
            for z in (5, 6, 7, 16)
        }
        # The head's own value for this centre, before the centre rule.
        _, value = term._values(features, _frame(), ImproperPotential())
        unmasked = value["k"][:, 0]
        assert bool(unmasked.ne(0).all()), "seeded so the head's window leaves k above its floor"
        torch.testing.assert_close(k[6], unmasked)
        torch.testing.assert_close(k[7], unmasked)
        assert k[5].tolist() == [0.0] and k[16].tolist() == [0.0]

    def test_the_form_is_k_one_minus_cos_two_phi(self):
        # k(1 + cos(2φ − π)) = cvff with d = −1, n = 2.
        term = ImproperCvff(4, 3, **SETTINGS)
        params = term.potential_params({"impropers": {"k": torch.ones(3, dtype=DTYPE)}})
        assert params["sign"].tolist() == [-1.0] * 3
        assert params["periodicity"].tolist() == [2.0] * 3


class TestBondChargeCorrection:
    def test_the_total_charge_is_the_total_formal_charge(self):
        term = BondChargeCorrection(
            4,
            3,
            hidden_dim=8,
            width=8,
            layers=2,
            out_hidden_dim=8,
            out_layers=2,
            tanh_output=20.0,
            dtype=DTYPE,
        )
        frame = _frame()
        frame["atoms", "formal_charge"] = torch.tensor([0, -1, 0, 0, 0])
        frame["atoms", "equi_index"] = torch.tensor([0, 1, 2, 3, 4])
        charge = term.predict(_features(5, 4), frame, None)["charge"]
        torch.testing.assert_close(charge.sum(), torch.tensor(-1.0, dtype=DTYPE))
        assert charge.std() > 0.0
