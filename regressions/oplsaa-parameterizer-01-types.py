"""Public-API regression for OPLSAAForceField types + OPLS-AA f14.

Spec: ``oplsaa-parameterizer-01-types``.

Hard-coded goldens only — no OpenMM / third-party oracle. Pins:

1. Default styles: bond/angle harmonic, dihedral:opls, improper:periodic,
   pair:lj/cut, pair:coulomb
2. Default NonbondedScaling ``scale_q_14 == scale_lj_14 == 0.5`` (not AMBER 5/6)
3. Four-atom 1–4 Coulomb-only energy:
       E = 0.5 * kcalmol_A * q_i * q_j / r
       q=[1,0,0,1], r_03=2 Å → E = 0.25 * kcalmol_A

Provenance
----------
    formula          : OPLS-AA f14=1/2 on Coulomb; E = f14 * ke * qi*qj / r
    ke               : molpot.potentials.elec.prefactors.kcalmol_A
                       = 332.0637132991921 (LAMMPS real units)
    geometry         : chain bonds 0-1,1-2,2-3; proper 0-1-2-3;
                       pos 0=(0,0,0), 1=(1,0,0), 2=(2,0,0), 3=(0,0,2)
    citation         : Jorgensen et al. JACS 1996 DOI 10.1021/ja9621760
    capture command  : PYTHONPATH=src python regressions/oplsaa-parameterizer-01-types.py
    date             : 2026-08-20
    tolerance        : energy atol 1e-10
"""

from __future__ import annotations

import math
import sys

import torch
from tensordict import TensorDict

from molpot.forcefield import OPLSAAForceField
from molpot.potentials.elec.prefactors import kcalmol_A

# Literal golden: 0.25 * ke with ke from prefactors.py (2026-08-20)
_GOLDEN_E = 0.25 * 332.0637132991921
_ENERGY_ATOL = 1e-10


def main() -> int:
    ff = OPLSAAForceField()
    for key in (
        "bond:harmonic",
        "angle:harmonic",
        "dihedral:opls",
        "improper:periodic",
        "pair:lj/cut",
        "pair:coulomb",
    ):
        if not ff.has_style(key):
            raise AssertionError(f"missing style {key}")

    scaling = ff.nonbonded_scaling
    if scaling.scale_q_14 != 0.5 or scaling.scale_lj_14 != 0.5:
        raise AssertionError(
            f"f14 scales q={scaling.scale_q_14} lj={scaling.scale_lj_14}, want 0.5/0.5"
        )

    # Four-atom chain; only 1–4 pair (0,3) contributes Coulomb with f14=0.5
    q = torch.tensor([1.0, 0.0, 0.0, 1.0], dtype=torch.float64)
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0],
        ],
        dtype=torch.float64,
    )
    ff = ff.with_params(atoms={"q": q})
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
    energy = float(ff.to_potentials().calc_energy(batch))
    expected = 0.25 * kcalmol_A
    if not math.isclose(energy, expected, abs_tol=_ENERGY_ATOL):
        raise AssertionError(
            f"1-4 Coulomb E={energy}, want {expected} (0.25*kcalmol_A; golden literal {_GOLDEN_E})"
        )
    if not math.isclose(expected, _GOLDEN_E, abs_tol=_ENERGY_ATOL):
        raise AssertionError(f"kcalmol_A drifted: 0.25*kcalmol_A={expected}, literal={_GOLDEN_E}")

    print("oplsaa-parameterizer-01-types: styles + f14 + 1-4 Coulomb pin OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
