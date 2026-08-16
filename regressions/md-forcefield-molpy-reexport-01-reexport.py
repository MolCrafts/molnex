"""Public-API regression for molix.md ForceField re-export.

Spec: `md-forcefield-molpy-reexport-01-reexport` (ac-009).

Hard-coded goldens only — no third-party oracle. Pins:

1. ``from molix.md import ForceField`` is identically ``molpy.ForceField``
2. Six deleted homemade names are absent from ``molix.md.__all__``
3. Two-atom ``ClassIForceField`` harmonic bond via ``to_potentials().calc_energy``
       E = 0.5 * k * (r - r0)^2
       k=2, r0=1 Å, r=1.5 Å → E = 0.25 kcal/mol
   Class-I default ``units="real"``. This script checks ``pots.calc_energy``
   directly; MD does not convert.
4. ``MD.set_neighbors`` is fluent (returns the same ``MD``)

Provenance
----------
    formula          : E = 0.5 * k * (r - r0)^2
    geometry         : two atoms on x, r = 1.5 Å
    capture command  : PYTHONPATH=src python regressions/md-forcefield-molpy-reexport-01-reexport.py
    commit           : 556f167 (556f167a9c48d4d72493d566ae848b4d3a749a5e)
    torch            : 2.12.1+cpu (python 3.14.5)
    date             : 2026-08-16
    device/precision : cpu, float64
    oracle           : none — analytic harmonic bond, no third-party, no
                       network, no subprocess
    tolerance        : energy atol 1e-10 (exact band)
"""

from __future__ import annotations

import math
import sys

import torch
from molpy import ForceField as MolpyForceField
from tensordict import TensorDict

import molix.md as md
from molix.md import MD, ForceField, NeighborList
from molpot.forcefield import ClassIForceField

_DELETED_FORCEFIELD_NAMES = (
    "CallableForceField",
    "HarmonicForceField",
    "LennardJonesForceField",
    "LennardJonesCutForceField",
    "PotentialForceField",
    "PeriodicPotentialForceField",
)

_K = 2.0
_R0 = 1.0
_R = 1.5
_GOLDEN_ENERGY = 0.25
_ENERGY_ATOL = 1e-10
_DTYPE = torch.float64


def _two_atom_bond(*, r: float = _R) -> TensorDict:
    """Two atoms on x with one covalent bond (column topology)."""
    pos = torch.tensor([[0.0, 0.0, 0.0], [r, 0.0, 0.0]], dtype=_DTYPE)
    return TensorDict(
        {
            "atoms": TensorDict({"pos": pos}, batch_size=[2]),
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


def main() -> int:
    if ForceField is not MolpyForceField:
        raise AssertionError("molix.md.ForceField is not identically molpy.ForceField")

    exported = set(md.__all__)
    still_exported = [name for name in _DELETED_FORCEFIELD_NAMES if name in exported]
    if still_exported:
        raise AssertionError(
            f"deleted ForceField names still in molix.md.__all__: {still_exported}"
        )

    system = _two_atom_bond()
    ff = ClassIForceField().with_params(
        bonds={
            "k": torch.tensor([_K], dtype=_DTYPE),
            "r0": torch.tensor([_R0], dtype=_DTYPE),
        }
    )
    energy = float(ff.to_potentials().calc_energy(system))
    if not math.isclose(energy, _GOLDEN_ENERGY, abs_tol=_ENERGY_ATOL):
        raise AssertionError(
            f"two-atom harmonic bond E={energy}, want {_GOLDEN_ENERGY} "
            f"(E = 0.5*k*(r-r0)^2, k={_K}, r0={_R0} Å, r={_R} Å)"
        )

    pos = system["atoms", "pos"]
    cell = torch.eye(3, dtype=_DTYPE) * 10.0
    neighbors = NeighborList(cell=cell, cutoff=3.5, positions=pos)
    composer = MD()
    if composer.set_neighbors(neighbors) is not composer:
        raise AssertionError("MD.set_neighbors must return the same MD")

    print("md-forcefield-molpy-reexport-01-reexport: all hard-coded goldens OK")
    print("  ForceField is molpy.ForceField")
    print("  deleted names absent from molix.md.__all__")
    print(f"  two-atom harmonic bond E = {energy}")
    print("  set_neighbors is fluent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
