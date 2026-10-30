"""``Potentials``: the sum of its members, keyed by style."""

from __future__ import annotations

import molpy as mp
import pytest
import torch

from molpot.potentials.angle import AngleHarmonic
from molpot.potentials.bond import BondHarmonic
from molpot.potentials.collection import Potentials
from tests.test_molpot.test_potentials.conftest import chain_batch

PARAMS = {
    "bond:harmonic": {
        "k": torch.full((6,), 300.0, dtype=torch.float64),
        "r0": torch.full((6,), 1.4, dtype=torch.float64),
    },
    "angle:harmonic": {
        "k": torch.full((4,), 50.0, dtype=torch.float64),
        "theta0": torch.full((4,), 109.5, dtype=torch.float64),
    },
}


class TestPotentials:
    def test_energy_is_the_sum_of_the_members(self):
        pots = Potentials({"bond:harmonic": BondHarmonic(), "angle:harmonic": AngleHarmonic()})
        batch = chain_batch()
        parts = [pots[key](batch, **PARAMS[key]) for key in pots]
        torch.testing.assert_close(pots(batch, PARAMS), parts[0] + parts[1])

    def test_named_styles_price_only_those(self):
        pots = Potentials({"bond:harmonic": BondHarmonic(), "angle:harmonic": AngleHarmonic()})
        batch = chain_batch()
        bond = pots["bond:harmonic"](batch, **PARAMS["bond:harmonic"])
        torch.testing.assert_close(pots(batch, PARAMS, styles=("bond:harmonic",)), bond)

    def test_atom_energies_split_each_term_equally_and_sum_to_the_graph(self):
        pots = Potentials({"bond:harmonic": BondHarmonic(), "angle:harmonic": AngleHarmonic()})
        batch = chain_batch()
        atoms = pots.atom_energy(batch, PARAMS)
        expected = torch.zeros_like(atoms)
        for key in pots:
            index, energy = pots[key].term_energies(batch, **PARAMS[key])
            for column, e in zip(index.T.tolist(), energy.tolist(), strict=True):
                for atom in column:
                    expected[atom] += e / len(column)
        torch.testing.assert_close(atoms, expected)
        graphs = torch.zeros_like(pots(batch, PARAMS)).index_add(0, batch["atoms", "batch"], atoms)
        torch.testing.assert_close(graphs, pots(batch, PARAMS))

    def test_a_named_style_it_does_not_hold_is_refused(self):
        pots = Potentials({"bond:harmonic": BondHarmonic()})
        with pytest.raises(ValueError, match="angle:harmonic"):
            pots(chain_batch(), styles=("angle:harmonic",))

    def test_params_for_a_style_it_does_not_hold_are_refused(self):
        pots = Potentials({"bond:harmonic": BondHarmonic()})
        with pytest.raises(ValueError, match="angle:harmonic"):
            pots(chain_batch(), PARAMS)

    def test_an_empty_collection_prices_nothing(self):
        with pytest.raises(ValueError, match="empty"):
            Potentials()(chain_batch())


class TestPotentialsCompile:
    @pytest.mark.slow
    def test_a_full_class_i_field_is_one_graph(self):
        """Every built-in style, parameters carried by type, traced as one graph."""
        import molpy as mp

        from molix.compile import Compiler
        from tests.test_molpot.test_potentials.test_pair import pair_batch

        ff = mp.ff.forcefield.ForceField(name="t", units="real")
        c = ff.def_style("atom", "full").def_type("c", mass=12.0)
        ff.def_style("bond", "harmonic").def_type("cc", c, c, k=300.0, r0=1.4)
        ff.def_style("angle", "harmonic").def_type("ccc", c, c, c, k=50.0, theta0=109.5)
        ff.def_style("dihedral", "periodic").def_type(
            "cccc",
            c,
            c,
            c,
            c,
            k1=1.3,
            periodicity1=1,
            phase1=0.0,
            k2=0.4,
            periodicity2=2,
            phase2=180.0,
        )
        ff.def_style("improper", "cvff").def_type("x", c, c, c, c, k=2.0, sign=-1.0, periodicity=2)
        ff.def_style("pair", "lj/cut", params={"cutoff": 10.0}).def_type(
            "c", c, epsilon=0.1, sigma=3.0
        )
        ff.def_style(
            "pair", "coul/cut", params={"cutoff": 10.0, "coulomb": 332.06371, "dielectric": 1.0}
        )
        ff.set_special_bonds([0.0, 0.0, 0.5], [0.0, 0.0, 0.8333])
        batch = pair_batch()
        # The collate spells a bond's type ``bond_types`` and every other term's ``type``.
        batch["bonds", "bond_types"] = torch.zeros(batch["bonds"].batch_size[0], dtype=torch.long)
        for ns in ("angles", "propers", "impropers"):
            batch[ns, "type"] = torch.zeros(batch[ns].batch_size[0], dtype=torch.long)
        batch["atoms", "type"] = torch.zeros(8, dtype=torch.long)
        batch["atoms", "charge"] = torch.linspace(-0.3, 0.3, 8, dtype=torch.float64)

        explanation = Compiler.explain(Potentials.from_forcefield(ff), batch)
        assert explanation.graph_count == 1, explanation.break_sites
        assert explanation.graph_break_count == 0, explanation.break_sites


def _forcefield() -> mp.ForceField:
    ff = mp.ff.forcefield.ForceField(name="t", units="real")
    ff.set_special_bonds([0.0, 0.0, 0.5], [0.0, 0.0, 0.5])
    c = ff.def_style("atom", "full").def_type("c", mass=12.0)
    ff.def_style("bond", "harmonic").def_type("cc", c, c, k=300.0, r0=1.4)
    ff.def_style("angle", "harmonic")
    ff.def_style("pair", "lj/cut", params={"cutoff": 9.0}).def_type("c", c, epsilon=0.1, sigma=3.0)
    return ff


class TestFromForcefield:
    """``from_forcefield`` returns one ``Potentials`` for the declared styles."""

    def test_every_declared_style_becomes_its_potential(self):
        pots = Potentials.from_forcefield(_forcefield())
        assert sorted(pots) == ["angle:harmonic", "bond:harmonic", "pair:lj/cut"]
        assert pots["bond:harmonic"].has_table and not pots["angle:harmonic"].has_table
        assert pots["pair:lj/cut"].cutoff == 9.0

    def test_an_unregistered_style_is_refused(self):
        ff = _forcefield()
        ff.def_style("pair", "exp6/pol")
        with pytest.raises(ValueError, match="pair:exp6/pol"):
            Potentials.from_forcefield(ff)
