"""LJ 12-6 on ForceField.to_potentials() via existing lj126 kernel."""

from __future__ import annotations

import math

import torch
from tensordict import TensorDict

from molpot.forcefield import ForceField
from molpot.potentials.vdw.lj126 import lj126_pair_energy


class TestLJEnergy:
    def test_packed_graphs_return_one_energy_per_molecule(self):
        eps = torch.full((4,), 0.2, dtype=torch.float64)
        sig = torch.full((4,), 3.0, dtype=torch.float64)
        ff = (
            ForceField(name="packed-lj")
            .def_pairstyle("lj/cut", cutoff=10.0)
            .with_params(atoms={"epsilon": eps, "sigma": sig})
        )
        batch = TensorDict(
            {
                "atoms": TensorDict(
                    {
                        "pos": torch.tensor(
                            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
                            dtype=torch.float64,
                        ),
                        "batch": torch.tensor([0, 0, 1, 1], dtype=torch.long),
                    },
                    batch_size=[4],
                ),
                "graphs": TensorDict({}, batch_size=[2]),
            },
            batch_size=[],
        )
        energy = ff.to_potentials().calc_energy(batch)
        assert energy.shape == (2,)
        assert not torch.isclose(energy[0], energy[1])

    def test_two_atom_matches_kernel(self):
        eps = torch.tensor([0.1, 0.2], dtype=torch.float64)
        sig = torch.tensor([3.0, 3.0], dtype=torch.float64)
        r = 3.0
        # LB: eps_ij = sqrt(0.1*0.2), sig_ij = 3.0
        e_ref = lj126_pair_energy(
            torch.tensor([r], dtype=torch.float64),
            torch.sqrt(eps[0] * eps[1]).unsqueeze(0),
            torch.tensor([3.0], dtype=torch.float64),
        )
        ff = (
            ForceField(name="lj")
            .def_pairstyle("lj/cut", cutoff=10.0)
            .with_params(atoms={"epsilon": eps, "sigma": sig})
        )
        batch = TensorDict(
            {
                "atoms": TensorDict(
                    {
                        "pos": torch.tensor([[0.0, 0.0, 0.0], [r, 0.0, 0.0]], dtype=torch.float64),
                        "batch": torch.zeros(2, dtype=torch.long),
                    },
                    batch_size=[2],
                )
            },
            batch_size=[],
        )
        e = ff.to_potentials().calc_energy(batch)
        assert math.isclose(float(e), float(e_ref), rel_tol=1e-10)

    def test_bond_exclusion_zeros_lj(self):
        from molpot.ir.scaling import NonbondedScaling

        ff = (
            ForceField(name="lj", nonbonded_scaling=NonbondedScaling(scale_lj_12=0.0))
            .def_pairstyle("lj/cut", cutoff=10.0)
            .with_params(
                atoms={
                    "epsilon": torch.tensor([0.15, 0.15], dtype=torch.float64),
                    "sigma": torch.tensor([3.0, 3.0], dtype=torch.float64),
                }
            )
        )
        batch = TensorDict(
            {
                "atoms": TensorDict(
                    {
                        "pos": torch.tensor(
                            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64
                        ),
                        "batch": torch.zeros(2, dtype=torch.long),
                    },
                    batch_size=[2],
                ),
                "bonds": TensorDict(
                    {
                        "atomi": torch.tensor([0], dtype=torch.long),
                        "atomj": torch.tensor([1], dtype=torch.long),
                    },
                    batch_size=[1],
                ),
            },
            batch_size=[],
        )
        e = ff.to_potentials().calc_energy(batch)
        assert math.isclose(float(e), 0.0, abs_tol=1e-12)
