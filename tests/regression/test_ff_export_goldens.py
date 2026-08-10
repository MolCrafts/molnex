"""Hard-coded force-export unit goldens (no live OpenMM).

Spec: learnable-classical-ff-08-ff-export.

Auto-marked ``regression`` by this directory's conftest; not part of the
default unit run. Public-API scenario script lives at
``regressions/learnable-classical-ff-08-ff-export.py``.
"""

from __future__ import annotations

import math

import torch

from molix.ff_export import (
    ForceFieldCompiler,
    scale_amber_vn,
    scale_bond_k,
)
from molpot.ir import BondBag, PotentialIR, ProperTorsionBag


def test_bond_k_golden_100_to_41840():
    assert scale_bond_k(100.0) == 41840.0


def test_torsion_vn_golden_2_to_4_184():
    assert scale_amber_vn(2.0) == 4.184


def test_compiler_emits_bond_and_torsion_goldens():
    ir = PotentialIR(
        bonds=BondBag(
            k=torch.tensor([100.0]),
            r0=torch.tensor([1.0]),
        ),
        propers=ProperTorsionBag(
            k=torch.tensor([[1.0]]),  # half-barrier for AMBER Vn=2
            periodicity=torch.tensor([1], dtype=torch.long),
            phase=torch.tensor([[0.0]]),
            idivf=torch.tensor([1.0]),
        ),
    )
    spec = ForceFieldCompiler("openmm").compile(ir)
    bond = next(f for f in spec.forces if f["type"] == "HarmonicBondForce")
    tor = next(f for f in spec.forces if f["type"] == "PeriodicTorsionForce")
    assert bond["parameters"][0]["k"] == 41840.0
    assert math.isclose(tor["parameters"][0]["k"], 4.184, abs_tol=0.0)
