"""LammpsAdapter — Class-I IR → LAMMPS units real force-spec goldens."""

from __future__ import annotations

import math

import torch

from molix.ff_export import ForceFieldCompiler, LammpsAdapter
from molpot.ir import AngleBag, BondBag, PotentialIR, ProperTorsionBag


class TestLammpsAdapter:
    def test_bond_k_half_convention(self):
        """molnex E=½k dr² → LAMMPS K = k/2."""
        ir = PotentialIR(bonds=BondBag(k=torch.tensor([100.0]), r0=torch.tensor([1.5])))
        spec = LammpsAdapter().translate(ir)
        bond = next(f for f in spec.forces if f["category"] == "bond")
        assert bond["style"] == "harmonic"
        assert bond["parameters"][0]["K"] == 50.0
        assert bond["parameters"][0]["r0"] == 1.5
        assert bond["parameters"][0]["type_index"] == 1

    def test_angle_degrees(self):
        ir = PotentialIR(
            angles=AngleBag(
                k=torch.tensor([100.0]),
                theta0=torch.tensor([math.radians(109.5)]),
            )
        )
        spec = LammpsAdapter().translate(ir)
        ang = next(f for f in spec.forces if f["category"] == "angle")
        assert abs(ang["parameters"][0]["K"] - 50.0) < 1e-12
        assert abs(ang["parameters"][0]["theta0_deg"] - 109.5) < 1e-5

    def test_opls_manifold_dihedral(self):
        k = 0.5 * torch.tensor([[1.0, 2.0, 0.5, 0.25]])
        phase = torch.zeros(1, 4)
        phase[0, 1] = math.pi
        phase[0, 3] = math.pi
        ir = PotentialIR(
            propers=ProperTorsionBag(
                k=k,
                periodicity=torch.tensor([1, 2, 3, 4], dtype=torch.long),
                phase=phase,
                idivf=torch.ones(1),
            )
        )
        spec = LammpsAdapter().translate(ir)
        dih = next(f for f in spec.forces if f["category"] == "dihedral")
        assert dih["style"] == "opls"
        p = dih["parameters"][0]
        assert abs(p["K1"] - 1.0) < 1e-12
        assert abs(p["K2"] - 2.0) < 1e-12
        assert abs(p["K3"] - 0.5) < 1e-12
        assert abs(p["K4"] - 0.25) < 1e-12

    def test_non_opls_emits_fourier(self):
        ir = PotentialIR(
            propers=ProperTorsionBag(
                k=torch.tensor([[0.4, 0.1]], dtype=torch.float64),
                periodicity=torch.tensor([1, 3], dtype=torch.long),
                phase=torch.zeros(1, 2, dtype=torch.float64),
                idivf=torch.ones(1, dtype=torch.float64),
            )
        )
        spec = LammpsAdapter().translate(ir)
        dih = next(f for f in spec.forces if f["category"] == "dihedral")
        assert dih["style"] == "fourier"
        assert dih["parameters"][0]["terms"][0]["periodicity"] == 1
        assert dih["parameters"][0]["terms"][0]["k"] == 0.4

    def test_compiler_name_lammps(self):
        ir = PotentialIR(bonds=BondBag(k=torch.tensor([100.0]), r0=torch.tensor([1.0])))
        spec = ForceFieldCompiler("lammps").compile(ir)
        assert spec.backend == "lammps"
        assert spec.metadata["units"] == "real"
