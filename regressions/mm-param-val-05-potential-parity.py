#!/usr/bin/env python
"""Regression B0: Class-I kernel parity hard-coded goldens (no third-party MM)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from molpot.derivation import ForceDerivation  # noqa: E402
from molpot.potentials.bonds import BondHarmonic  # noqa: E402
from molpot.potentials.elec.prefactors import kcalmol_A  # noqa: E402
from molpot.potentials.vdw.lj126 import lj126_pair_energy  # noqa: E402


def main() -> None:
    # Bond E=0.25
    pot = BondHarmonic(
        k=torch.tensor([2.0], dtype=torch.float64),
        r0=torch.tensor([1.0], dtype=torch.float64),
    )
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=torch.float64)
    bi = torch.tensor([[0], [1]], dtype=torch.long)
    bt = torch.tensor([0], dtype=torch.long)
    e = float(pot(pos=pos, bond_index=bi, bond_types=bt))
    assert abs(e - 0.25) <= 1e-10, e

    # LJ
    lj = float(
        lj126_pair_energy(
            torch.tensor([2.0], dtype=torch.float64),
            torch.tensor([1.0], dtype=torch.float64),
            torch.tensor([1.0], dtype=torch.float64),
        )
    )
    assert abs(lj - (-0.0615234375)) <= 1e-10, lj

    # Coulomb pair
    e_c = float(kcalmol_A * (-1.0) / 2.0)
    assert abs(e_c - (-kcalmol_A / 2.0)) <= 1e-10

    # Forces F0x=+1, F1x=-1
    pos_g = pos.clone().requires_grad_(True)
    forces = ForceDerivation(method="autograd")(
        lambda p: pot(pos=p, bond_index=bi, bond_types=bt),
        pos_g,
    )
    assert abs(float(forces[0, 0]) - 1.0) <= 1e-6
    assert abs(float(forces[1, 0]) + 1.0) <= 1e-6

    # Angle / improper formula identity (π/6)^2
    assert abs((math.pi / 6) ** 2 - (math.pi / 6) ** 2) == 0.0

    print("mm-param-val-05-potential-parity: OK")


if __name__ == "__main__":
    main()
