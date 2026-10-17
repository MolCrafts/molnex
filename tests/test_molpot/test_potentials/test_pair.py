"""Pair styles priced as molpy prices them, special-bonds weights included."""

from __future__ import annotations

import math

import molpy as mp
import numpy as np
import pytest
import torch
from tensordict import TensorDict

from molpot.potentials.pair import LJCut, PairCoulCut
from tests.test_molpot.test_potentials.conftest import chain_batch, energy_and_forces, oracle

#: Every intramolecular pair of the two chains with its relation (1-2/1-3/1-4).
PAIRS = [(0, 1, 1), (1, 2, 1), (2, 3, 1), (0, 2, 2), (1, 3, 2), (0, 3, 3)]
PAIRS = PAIRS + [(i + 4, j + 4, c) for i, j, c in PAIRS]
SPECIAL = (0.0, 0.3, 0.5)
EPS = [0.10, 0.20, 0.15, 0.12, 0.10, 0.20, 0.15, 0.12]
SIG = [3.0, 3.4, 3.2, 2.9, 3.0, 3.4, 3.2, 2.9]
Q = [0.3, -0.2, 0.1, -0.2, -0.1, 0.25, -0.3, 0.15]
COULOMB = 332.06371


def pair_batch() -> TensorDict:
    """The chains with a bidirectional pair list, as a neighbour list writes it."""
    batch = chain_batch()
    rows = [(i, j, c) for i, j, c in PAIRS] + [(j, i, c) for i, j, c in PAIRS]
    batch["pairs"] = TensorDict(
        {
            "pair_index": torch.tensor([[i, j] for i, j, _ in rows]),
            "relation": torch.tensor([c for _, _, c in rows], dtype=torch.int8),
        },
        batch_size=[len(rows)],
    )
    return batch


def weighted_pairs() -> tuple[list[int], list[int], list[float]]:
    """Each real pair once, with its special-bonds weight; weight-0 pairs dropped."""
    weights = (1.0, *SPECIAL)
    kept = [(i, j, weights[c]) for i, j, c in PAIRS if weights[c] != 0.0]
    i, j, w = zip(*kept, strict=True)
    return list(i), list(j), list(w)


class TestLJCut:
    def test_prices_what_molpy_prices(self):
        lj = LJCut(cutoff=10.0, special=SPECIAL, mixing="arithmetic")
        energy, forces = energy_and_forces(
            lj,
            pair_batch(),
            epsilon=torch.tensor(EPS, dtype=torch.float64),
            sigma=torch.tensor(SIG, dtype=torch.float64),
        )
        i, j, w = weighted_pairs()
        eps = [w_ * math.sqrt(EPS[a] * EPS[b]) for a, b, w_ in zip(i, j, w, strict=True)]
        sig = [0.5 * (SIG[a] + SIG[b]) for a, b in zip(i, j, strict=True)]
        rows = [[a, b] for a, b in zip(i, j, strict=True)]
        ref_e, ref_f = oracle(
            ("pair", "lj/cut", rows, {"epsilon": eps, "sigma": sig, "cutoff": 10.0})
        )
        np.testing.assert_allclose(energy, ref_e, rtol=1e-12)
        np.testing.assert_allclose(forces, ref_f, rtol=1e-9, atol=1e-9)

    def test_relation_is_required(self):
        batch = pair_batch()
        del batch["pairs", "relation"]
        with pytest.raises(KeyError):
            LJCut(cutoff=10.0, special=SPECIAL)(
                batch,
                epsilon=torch.tensor(EPS, dtype=torch.float64),
                sigma=torch.tensor(SIG, dtype=torch.float64),
            )

    def test_a_style_carries_the_self_rows_of_each_atom_type(self):
        ff = mp.ff.forcefield.ForceField(name="t", units="real")
        atoms = ff.def_style("atom", "full")
        a, b = atoms.def_type("a", mass=12.0), atoms.def_type("b", mass=1.0)
        style = ff.def_style("pair", "lj/cut", params={"cutoff": 10.0, "mixing": "geometric"})
        style.def_type("a", a, epsilon=0.1, sigma=3.0)
        style.def_type("b", b, epsilon=0.2, sigma=2.5)
        ff.set_special_bonds([0.0, 0.0, 0.5], [0.0, 0.0, 0.5])
        lj = LJCut.from_style(style, forcefield=ff)
        batch = pair_batch()
        type_id = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0])
        batch["atoms", "type"] = type_id
        passed = LJCut(cutoff=10.0, special=(0.0, 0.0, 0.5), mixing="geometric")(
            batch,
            epsilon=torch.tensor([0.1, 0.2], dtype=torch.float64)[type_id],
            sigma=torch.tensor([3.0, 2.5], dtype=torch.float64)[type_id],
        )
        torch.testing.assert_close(lj(batch), passed)

    def test_a_cross_row_is_refused(self):
        ff = mp.ff.forcefield.ForceField(name="t", units="real")
        atoms = ff.def_style("atom", "full")
        a, b = atoms.def_type("a", mass=12.0), atoms.def_type("b", mass=1.0)
        style = ff.def_style("pair", "lj/cut", params={"cutoff": 10.0})
        style.def_type("ab", a, b, epsilon=0.1, sigma=3.0)
        with pytest.raises(ValueError, match="cross row"):
            LJCut.from_style(style, forcefield=ff)


class TestPairCoulCut:
    def test_prices_what_molpy_prices(self):
        coul = PairCoulCut(cutoff=10.0, special=SPECIAL, coulomb=COULOMB, dielectric=1.0)
        energy, forces = energy_and_forces(
            coul, pair_batch(), charge=torch.tensor(Q, dtype=torch.float64)
        )
        i, j, w = weighted_pairs()
        # The explicit kernel prices a pair row at full strength, so each
        # special-bonds class is its own term with the weight folded into C.
        by_weight: dict[float, list[list[int]]] = {}
        for a, b, w_ in zip(i, j, w, strict=True):
            by_weight.setdefault(w_, []).append([a, b])
        ref_e, ref_f = oracle(
            *(
                (
                    "pair",
                    "coul/cut",
                    rows,
                    {"charges": Q, "coulomb": COULOMB * w_, "dielectric": 1.0, "cutoff": 10.0},
                )
                for w_, rows in by_weight.items()
            )
        )
        np.testing.assert_allclose(energy, ref_e, rtol=1e-12)
        np.testing.assert_allclose(forces, ref_f, rtol=1e-9, atol=1e-9)

    def test_the_style_must_state_its_coulomb_constant(self):
        ff = mp.ff.forcefield.ForceField(name="t", units="real")
        style = ff.def_style("pair", "coul/cut", params={"cutoff": 10.0})
        with pytest.raises(ValueError, match="coulomb"):
            PairCoulCut.from_style(style, forcefield=ff)


class TestPairGeometry:
    """The pair vector: live positions, plus a lattice remainder or a stored difference."""

    @staticmethod
    def _energy(batch: TensorDict) -> torch.Tensor:
        lj = LJCut(cutoff=10.0, special=SPECIAL, mixing="arithmetic")
        eps = torch.tensor(EPS, dtype=torch.float64)
        sig = torch.tensor(SIG, dtype=torch.float64)
        return lj(batch, epsilon=eps, sigma=sig).sum()

    def test_shifts_move_the_pair_vector(self):
        batch = pair_batch()
        base = self._energy(batch)
        shifts = torch.zeros(batch["pairs"].batch_size[0], 3, dtype=torch.float64)
        batch["pairs", "shifts"] = shifts
        torch.testing.assert_close(self._energy(batch), base)
        shifts[3, 0] = 0.3  # the (0, 2) 1-3 pair: weight 0.3, so it is priced
        batch["pairs", "shifts"] = shifts
        assert not torch.isclose(self._energy(batch), base)

    def test_a_stored_pair_diff_is_priced_with_the_gradient_through_pos(self):
        batch = pair_batch()
        pos = batch["atoms", "pos"]
        src, tgt = batch["pairs", "pair_index"].T
        batch["pairs", "pair_diff"] = pos[tgt] - pos[src]
        torch.testing.assert_close(self._energy(batch), self._energy(pair_batch()))

        stale = pair_batch()
        stale["pairs", "pair_diff"] = (pos[tgt] - pos[src]) * 1.05
        assert not torch.isclose(self._energy(stale), self._energy(pair_batch()))
        leaf = pos.clone().requires_grad_(True)
        stale["atoms", "pos"] = leaf
        self._energy(stale).backward()
        assert leaf.grad is not None and float(leaf.grad.abs().sum()) > 0.0
