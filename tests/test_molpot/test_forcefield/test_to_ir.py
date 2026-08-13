"""ForceField.to_ir — Class-I bags from continuous param tables."""

from __future__ import annotations

import math
from dataclasses import fields

import pytest
import torch
from tensordict import TensorDict

from molpot.forcefield import ClassIForceField, OPLSForceField
from molpot.ir import BondBag, ProperTorsionBag


class TestForceFieldToIR:
    def test_bonds_and_opls_propers(self):
        ff = OPLSForceField().with_params(
            bonds={
                "k": torch.tensor([100.0, 200.0]),
                "r0": torch.tensor([1.5, 1.6]),
            },
            propers={
                "k1": torch.tensor([1.0]),
                "k2": torch.tensor([2.0]),
                "k3": torch.tensor([0.0]),
                "k4": torch.tensor([0.5]),
            },
        )
        ir = ff.to_ir()
        assert isinstance(ir.bonds, BondBag)
        assert torch.allclose(ir.bonds.k, torch.tensor([100.0, 200.0]))
        assert isinstance(ir.propers, ProperTorsionBag)
        # OPLS Ki → multi-term k = Ki/2
        assert torch.allclose(ir.propers.k[0], torch.tensor([0.5, 1.0, 0.0, 0.25]))
        assert torch.allclose(
            ir.propers.phase[0],
            torch.tensor([0.0, math.pi, 0.0, math.pi]),
        )
        assert ir.propers.periodicity.tolist() == [1, 2, 3, 4]

    def test_fourier_propers_passthrough(self):
        k = torch.tensor([[0.1, 0.2, 0.0, 0.0]])
        phase = torch.zeros(1, 4)
        phase[0, 1] = math.pi
        ff = ClassIForceField().with_params(
            propers={
                "k": k,
                "phase": phase,
                "periodicity": torch.tensor([1, 2, 3, 4], dtype=torch.long),
                "idivf": torch.ones(1),
            }
        )
        ir = ff.to_ir()
        assert ir.propers is not None
        assert torch.allclose(ir.propers.k, k)

    def test_atoms_q_becomes_charge_bag(self):
        ff = ClassIForceField().with_params(atoms={"q": torch.tensor([0.1, -0.1])})
        ir = ff.to_ir()
        assert ir.charges is not None
        assert torch.allclose(ir.charges.q, torch.tensor([0.1, -0.1]))

    def test_is_molpy_forcefield_and_realize(self):
        from molpy import BondType
        from molpy import ForceField as MolpyForceField

        ff = ClassIForceField().with_params(
            bonds={
                "k": torch.tensor([200.0]),
                "r0": torch.tensor([1.53]),
            }
        )
        assert isinstance(ff, MolpyForceField)
        system = TensorDict(
            {
                "atoms": TensorDict(
                    {"pos": torch.tensor([[0.0, 0.0, 0.0], [1.53, 0.0, 0.0]])},
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
        ff.realize(system)
        names = [t.name for t in ff.get_types(BondType)]
        assert names == ["1-2"]

    def test_cartesian_propers_to_ir_maps_to_cosine_table(self):
        a = torch.tensor([[0.6]], dtype=torch.float64)
        b = torch.tensor([[0.8]], dtype=torch.float64)
        ff = ClassIForceField().with_params(
            propers={
                "a": a,
                "b": b,
                "periodicity": torch.tensor([1], dtype=torch.long),
                "idivf": torch.ones(1, dtype=torch.float64),
            }
        )
        ir = ff.to_ir()
        assert isinstance(ir.propers, ProperTorsionBag)
        assert torch.allclose(ir.propers.k, torch.tensor([[1.0]], dtype=torch.float64), atol=1e-12)
        assert torch.allclose(
            ir.propers.phase,
            torch.tensor([[0.9272952180016123]], dtype=torch.float64),
            atol=1e-12,
        )
        assert {f.name for f in fields(type(ir.propers))} == {
            "k",
            "phase",
            "periodicity",
            "idivf",
        }
        assert not hasattr(ir.propers, "c0")

    def test_mixed_cartesian_and_polar_propers_raise(self):
        with pytest.raises(ValueError):
            ClassIForceField().with_params(
                propers={
                    "a": torch.tensor([[0.6]]),
                    "b": torch.tensor([[0.8]]),
                    "k": torch.tensor([[1.0]]),
                    "phase": torch.tensor([[0.0]]),
                    "periodicity": torch.tensor([1], dtype=torch.long),
                    "idivf": torch.ones(1),
                }
            ).to_ir()

    def test_cartesian_to_potentials_energy_matches_kernel_identity(self):
        # φ=π/3, n=2, k=2, γ=π/4 → E = k[1+cos(nφ-γ)] = 2.5176380902050415
        k = 2.0
        gamma = math.pi / 4.0
        a = k * math.cos(gamma)
        b = k * math.sin(gamma)
        ff = ClassIForceField().with_params(
            propers={
                "a": torch.tensor([[a]], dtype=torch.float64),
                "b": torch.tensor([[b]], dtype=torch.float64),
                "periodicity": torch.tensor([2], dtype=torch.long),
                "idivf": torch.ones(1, dtype=torch.float64),
            }
        )
        # i-j-k-l with l rotated about jk so φ = π/3 (same atan2 convention as cis/perp fixtures).
        pos = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.5, math.sqrt(3.0) / 2.0],
            ],
            dtype=torch.float64,
        )
        batch = TensorDict(
            {
                "atoms": TensorDict({"pos": pos}, batch_size=[4]),
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
        energy = ff.to_potentials().calc_energy(batch)
        phi = math.pi / 3.0
        n = 2.0
        expected_kernel = k * (1.0 + math.cos(n * phi - gamma))
        expected_fourier = k + a * math.cos(n * phi) + b * math.sin(n * phi)
        assert math.isclose(expected_kernel, 2.5176380902050415, abs_tol=1e-10)
        assert math.isclose(expected_fourier, 2.5176380902050415, abs_tol=1e-10)
        # ProperTorsionPeriodic uses a 1e-8 normalizer in _dihedral_phi, so φ
        # is ~1e-9 off π/3; match the algebraic identity at 1e-7, not 1e-10.
        assert math.isclose(float(energy), expected_kernel, abs_tol=1e-7)
