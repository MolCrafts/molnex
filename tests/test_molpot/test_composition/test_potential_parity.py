"""Validation B0 — Class-I PotentialIR + kernel parity goldens (kcal/mol, Å).

Hard-coded analytical references only. No live OpenMM / molpy ForceField.
Units: CLASS_I_CANONICAL (kcal/mol, Å, kcal/mol/Å). Improper center-first
(molrs row 0 = center). Forces via ForceDerivation(method="autograd").
"""

from __future__ import annotations

import math

import torch
from tensordict import TensorDict

from molpot.composition.classical_mm import ClassicalMMComposer
from molpot.derivation import ForceDerivation
from molpot.ir import NonbondedScaling, PotentialIR
from molpot.ir.bags import (
    BondBag,
)
from molpot.potentials.angles import AngleHarmonic
from molpot.potentials.bonds import BondHarmonic
from molpot.potentials.dihedrals.periodic import ProperTorsionPeriodic
from molpot.potentials.elec.potentials.coulomb import CoulombPotential
from molpot.potentials.elec.prefactors import kcalmol_A
from molpot.potentials.impropers.harmonic import ImproperHarmonic
from molpot.potentials.vdw.lj126 import lj126_pair_energy

_TOL_E = 1e-10
_TOL_F = 1e-6


class TestPotentialParityTerms:
    def test_bond_harmonic_golden(self):
        pot = BondHarmonic(
            k=torch.tensor([2.0], dtype=torch.float64), r0=torch.tensor([1.0], dtype=torch.float64)
        )
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=torch.float64)
        e = pot(
            pos=pos,
            bond_index=torch.tensor([[0], [1]], dtype=torch.long),
            bond_types=torch.tensor([0], dtype=torch.long),
        )
        assert abs(float(e) - 0.25) <= _TOL_E

    def test_angle_harmonic_golden(self):
        pot = AngleHarmonic(
            k=torch.tensor([2.0], dtype=torch.float64),
            theta0=torch.tensor([math.pi / 3], dtype=torch.float64),
        )
        # right angle at atom 1: (1,0,0)-(0,0,0)-(0,1,0)
        pos = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float64,
        )
        e = pot(
            pos=pos,
            angle_index=torch.tensor([[0], [1], [2]], dtype=torch.long),
            angle_types=torch.tensor([0], dtype=torch.long),
        )
        expected = (math.pi / 6) ** 2
        assert abs(float(e) - expected) <= _TOL_E

    def test_proper_cis_golden(self):
        # E = k/s * [1 + cos(n*φ - γ)]; cis φ≈0, n=1, γ=0, s=1 → 2k with k=1
        pot = ProperTorsionPeriodic(
            k=torch.tensor([[1.0]], dtype=torch.float64),
            periodicity=torch.tensor([1.0], dtype=torch.float64),
            phase=torch.tensor([[0.0]], dtype=torch.float64),
            idivf=torch.tensor([1.0], dtype=torch.float64),
        )
        # planar cis: i-j-k-l with φ=0
        pos = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=torch.float64,
        )
        e = pot(
            pos=pos,
            proper_index=torch.tensor([[0], [1], [2], [3]], dtype=torch.long),
            proper_types=torch.tensor([0], dtype=torch.long),
        )
        assert abs(float(e) - 2.0) <= _TOL_E

    def test_improper_harmonic_golden(self):
        """E = ½ k (χ − χ₀)² with k=2, χ₀=0, χ=π/6 → (π/6)²."""
        pot = ImproperHarmonic(
            k=torch.tensor([2.0], dtype=torch.float64),
            chi0=torch.tensor([0.0], dtype=torch.float64),
        )
        chi = math.pi / 6
        # molrs center-first improper_index [c,i,j,k]; dihedral uses (i,c,j,k).
        pos = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # center
                [1.0, 0.0, 0.0],  # i
                [0.0, 1.0, 0.0],  # j
                [math.cos(chi), 1.0, math.sin(chi)],  # k → |χ|=π/6
            ],
            dtype=torch.float64,
        )
        e = pot(
            pos=pos,
            improper_index=torch.tensor([[0], [1], [2], [3]], dtype=torch.long),
            improper_types=torch.tensor([0], dtype=torch.long),
        )
        expected = chi**2  # 0.5 * k=2 → E = χ²
        assert abs(float(e) - expected) <= 1e-8

    def test_coulomb_golden(self):
        pot = CoulombPotential(prefactor=kcalmol_A)
        r = torch.tensor([2.0], dtype=torch.float64)
        # pair energy q_i q_j / r * prefactor with q=±1
        e_pair = pot.from_dist(r) * (-1.0)  # charges product -1
        expected = -kcalmol_A / 2.0
        assert abs(float(e_pair) - expected) <= _TOL_E

    def test_lj_golden(self):
        r = torch.tensor([2.0], dtype=torch.float64)
        e = lj126_pair_energy(
            r, torch.tensor([1.0], dtype=torch.float64), torch.tensor([1.0], dtype=torch.float64)
        )
        assert abs(float(e) - (-0.0615234375)) <= _TOL_E


class TestPotentialParityTotal:
    def test_composer_bonded_bond_only(self):
        ir = PotentialIR(
            bonds=BondBag(
                k=torch.tensor([2.0], dtype=torch.float64),
                r0=torch.tensor([1.0], dtype=torch.float64),
            )
        )
        batch = TensorDict(
            {
                "atoms": TensorDict(
                    {"pos": torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=torch.float64)},
                    batch_size=[2],
                ),
                "bonds": TensorDict(
                    {
                        "bond_index": torch.tensor([[0], [1]], dtype=torch.long),
                    },
                    batch_size=[],
                ),
            },
            batch_size=[],
        )
        # ClassicalMMComposer may read bond_index differently - check helpers
        composer = ClassicalMMComposer()
        # Ensure bond_index layout matches composer expectations
        e = composer.energy(ir, batch)
        assert abs(float(e) - 0.25) <= _TOL_E
        terms = composer.term_energies(ir, batch)
        assert abs(float(terms["bonds"]) - 0.25) <= _TOL_E


class TestPotentialParityForces:
    def test_bond_forces_golden(self):
        pot = BondHarmonic(
            k=torch.tensor([2.0], dtype=torch.float64),
            r0=torch.tensor([1.0], dtype=torch.float64),
        )
        pos = torch.tensor(
            [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=torch.float64, requires_grad=True
        )
        bond_index = torch.tensor([[0], [1]], dtype=torch.long)
        bond_types = torch.tensor([0], dtype=torch.long)

        def energy_fn(p: torch.Tensor) -> torch.Tensor:
            return pot(pos=p, bond_index=bond_index, bond_types=bond_types)

        forces = ForceDerivation(method="autograd")(energy_fn, pos)
        assert abs(float(forces[0, 0].detach()) - 1.0) <= _TOL_F
        assert abs(float(forces[1, 0].detach()) - (-1.0)) <= _TOL_F

    def test_force_is_neg_grad(self):
        pot = BondHarmonic(
            k=torch.tensor([2.0], dtype=torch.float64),
            r0=torch.tensor([1.0], dtype=torch.float64),
        )
        pos = torch.tensor(
            [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=torch.float64, requires_grad=True
        )
        bond_index = torch.tensor([[0], [1]], dtype=torch.long)
        bond_types = torch.tensor([0], dtype=torch.long)
        e = pot(pos=pos, bond_index=bond_index, bond_types=bond_types)
        (g,) = torch.autograd.grad(e, pos, create_graph=False)
        forces = ForceDerivation(method="autograd")(
            lambda p: pot(pos=p, bond_index=bond_index, bond_types=bond_types),
            pos.detach().requires_grad_(True),
        )
        assert torch.allclose(forces, -g.detach(), atol=_TOL_F)


class TestPotentialParityUnits:
    def test_nonbonded_scaling_defaults(self):
        s = NonbondedScaling()
        assert s.scale_q_12 == 0.0
        assert s.scale_lj_12 == 0.0
        assert abs(s.scale_q_14 - 5.0 / 6.0) < 1e-12
        assert abs(s.scale_lj_14 - 0.5) < 1e-12
        ir = PotentialIR()
        assert ir.unit_system == "class_i_canonical"

    def test_kcalmol_prefactor(self):
        assert abs(kcalmol_A - 332.0637132991921) < 1e-9
