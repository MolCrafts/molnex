"""Coulomb term on ForceField.to_potentials() reading atoms.q."""

from __future__ import annotations

import math

import torch
from tensordict import TensorDict

from molpot.charge import (
    AtomicChargeModel,
    ChiEtaHead,
    DiagonalHardnessKernel,
    ElectronegativityEqualization,
)
from molpot.forcefield import ForceField
from molpot.ir.scaling import ExclusionScales, NonbondedScaling
from molpot.potentials.elec.prefactors import kcalmol_A


class TestCoulombEnergy:
    def test_two_charge_analytic(self):
        """q=±1 at r=1 Å → E = −ke."""
        ff = (
            ForceField(name="es")
            .def_pairstyle("coulomb")
            .with_params(
                atoms={
                    "q": torch.tensor([1.0, -1.0], dtype=torch.float64),
                }
            )
        )
        batch = TensorDict(
            {
                "atoms": TensorDict(
                    {
                        "pos": torch.tensor(
                            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64
                        ),
                        "batch": torch.zeros(2, dtype=torch.long),
                    },
                    batch_size=[2],
                )
            },
            batch_size=[],
        )
        e = ff.to_potentials().calc_energy(batch)
        assert math.isclose(float(e), -kcalmol_A, rel_tol=1e-10)

    def test_bond_exclusion_scale_zero(self):
        """1–2 pair with the Class-I default ``q.s12 = 0`` contributes nothing."""
        ff = (
            ForceField(
                name="es",
                nonbonded_scaling=NonbondedScaling(),
            )
            .def_pairstyle("coulomb")
            .with_params(atoms={"q": torch.tensor([1.0, -1.0], dtype=torch.float64)})
        )
        batch = TensorDict(
            {
                "atoms": TensorDict(
                    {
                        "pos": torch.tensor(
                            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64
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

    def test_scale_14(self):
        """Four atoms; only i–l is 1–4 with scale 0.5."""
        scaling = NonbondedScaling(q=ExclusionScales(0.0, 0.0, 0.5))
        q = torch.tensor([1.0, 0.0, 0.0, -1.0], dtype=torch.float64)
        # place 0 and 3 at r=2 so E_full = ke*(1)*(-1)/2 = -ke/2; with s14=0.5 → -ke/4
        pos = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],  # will move
            ],
            dtype=torch.float64,
        )
        pos[3] = torch.tensor([0.0, 0.0, 2.0], dtype=torch.float64)
        ff = (
            ForceField(name="es", nonbonded_scaling=scaling)
            .def_pairstyle("coulomb")
            .with_params(atoms={"q": q})
        )
        batch = TensorDict(
            {
                "atoms": TensorDict(
                    {"pos": pos, "batch": torch.zeros(4, dtype=torch.long)},
                    batch_size=[4],
                ),
                "bonds": TensorDict(
                    {
                        "atomi": torch.tensor([0, 1, 2], dtype=torch.long),
                        "atomj": torch.tensor([1, 2, 3], dtype=torch.long),
                    },
                    batch_size=[3],
                ),
                "propers": TensorDict(
                    {
                        "atomi": torch.tensor([0], dtype=torch.long),
                        "atomj": torch.tensor([1], dtype=torch.long),
                        "atomk": torch.tensor([2], dtype=torch.long),
                        "atoml": torch.tensor([3], dtype=torch.long),
                    },
                    batch_size=[1],
                ),
            },
            batch_size=[],
        )
        e = ff.to_potentials().calc_energy(batch)
        # only non-excluded pairs among (0,2)=1-3 scale0, (0,3)=1-4 scale0.5, (1,3)=1-3 scale0
        # r_03 = 2 → contribution 0.5 * ke * (1)*(-1)/2 = -0.25 ke
        expected = -0.25 * kcalmol_A
        assert math.isclose(float(e), expected, rel_tol=1e-8)

    def test_charge_model_into_coulomb(self):
        """EEM q written then Coulomb energy finite and neutral-consistent."""
        charge = AtomicChargeModel(
            ChiEtaHead(feature_dim=4, hidden_dim=8),
            ElectronegativityEqualization(DiagonalHardnessKernel()),
        )
        ff = (
            ForceField(name="mm")
            .def_pairstyle("coulomb")
            .with_params(atoms={"q": torch.tensor([0.3, -0.3], dtype=torch.float64)})
        )
        batch = TensorDict(
            {
                "atoms": TensorDict(
                    {
                        "pos": torch.tensor(
                            [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=torch.float64
                        ),
                        "batch": torch.zeros(2, dtype=torch.long),
                    },
                    batch_size=[2],
                )
            },
            batch_size=[],
        )
        e = ff.to_potentials().calc_energy(batch)
        expected = kcalmol_A * (0.3 * -0.3) / 1.5
        assert math.isclose(float(e), expected, rel_tol=1e-10)
        ff2 = charge.write_forcefield(
            ForceField(name="mm").def_pairstyle("coulomb"),
            torch.zeros(2, 4),
            batch,
        )
        e2 = ff2.to_potentials().calc_energy(batch)
        assert torch.isfinite(e2)

    def test_uses_coulomb_potential_kernel(self):
        """Open-boundary path must match CoulombPotential.from_dist."""
        from molpot.potentials.elec.potentials import CoulombPotential

        q = torch.tensor([0.5, -0.5], dtype=torch.float64)
        r = 2.0
        pot = CoulombPotential(prefactor=kcalmol_A)
        e_ref = (q[0] * q[1] * pot.from_dist(torch.tensor([r], dtype=torch.float64))).sum()
        ff = ForceField(name="es").def_pairstyle("coulomb").with_params(atoms={"q": q})
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
        assert math.isclose(float(e), float(e_ref), rel_tol=1e-12)

    def test_ewald_path_with_cell(self):
        """graphs.cell present → EwaldCalculator path (finite energy)."""
        a = 10.0
        q = torch.tensor([1.0, -1.0], dtype=torch.float64)
        ff = (
            ForceField(name="es")
            .def_pairstyle(
                "coulomb",
                calculator="ewald",
                smearing=4.0,
                lr_wavelength=1.5,
                real_space_cutoff=8.0,
            )
            .with_params(atoms={"q": q})
        )
        batch = TensorDict(
            {
                "atoms": TensorDict(
                    {
                        "pos": torch.tensor(
                            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float64
                        ),
                        "batch": torch.zeros(2, dtype=torch.long),
                    },
                    batch_size=[2],
                ),
                "graphs": TensorDict(
                    {
                        "cell": torch.eye(3, dtype=torch.float64) * a,
                    },
                    batch_size=[],
                ),
            },
            batch_size=[],
        )
        e = ff.to_potentials().calc_energy(batch)
        assert torch.isfinite(e)
        # opposite charges should give attractive (negative) total energy
        assert float(e) < 0.0

    def test_cutoff_lives_on_lj_potential_not_potentials(self):
        ff = ForceField(name="lj").def_pairstyle("lj/cut", cutoff=12.5)
        pots = ff.to_potentials()
        assert pots.lj is not None
        assert pots.lj.cutoff == 12.5
        assert not hasattr(pots, "lj_cutoff") or getattr(pots, "lj_cutoff", None) is None
