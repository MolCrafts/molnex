"""Ewald / PME / P3M reproduce analytic Madelung constants on tiny cells.

One length scale, half neighbour list, float64. The full grid (seven
crystals, three scales, both list kinds, Wigner solids, GROMACS and
espressomd parity) lives in ``regressions/elec-reference-parity.py``.
"""

import math

import pytest
import torch

from molpot.potentials.elec import (
    CoulombPotential,
    EwaldCalculator,
    InversePowerLawPotential,
    P3MCalculator,
    PMECalculator,
)
from tests.test_molpot.test_elec.conftest import periodic_neighbor_list

DTYPE = torch.float64

# name: (positions, charges, cell rows, Madelung constant, formula units)
CRYSTALS = {
    # cubic, 1:1
    "CsCl": (
        [[0, 0, 0], [0.5, 0.5, 0.5]],
        [-1.0, 1.0],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        2.0353610945260,
        1,
    ),
    # fcc primitive, 1:1
    "NaCl_primitive": (
        [[0, 0, 0], [1, 0, 0]],
        [1.0, -1.0],
        [[0, 1, 1], [1, 0, 1], [1, 1, 0]],
        1.7475645946,
        1,
    ),
    # fcc primitive, 1:2
    "fluorite": (
        [[0.25, 0.25, 0.25], [0.75, 0.75, 0.75], [0, 0, 0]],
        [-1.0, -1.0, 2.0],
        [[0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
        11.6365752270768,
        1,
    ),
}


def _crystal(name):
    pos, q, cell, ref, units = CRYSTALS[name]
    return (
        torch.tensor(pos, dtype=DTYPE),
        torch.tensor(q, dtype=DTYPE).reshape(-1, 1),
        torch.tensor(cell, dtype=DTYPE),
        ref,
        units,
    )


def _calculator(name, smearing):
    if name == "ewald":
        return EwaldCalculator(
            InversePowerLawPotential(exponent=1, smearing=smearing), lr_wavelength=0.5 * smearing
        ), 4e-6
    if name == "pme":
        return PMECalculator(
            InversePowerLawPotential(exponent=1, smearing=smearing), mesh_spacing=smearing / 8
        ), 9e-4
    return P3MCalculator(CoulombPotential(smearing=smearing), mesh_spacing=smearing / 8), 9e-4


@pytest.mark.parametrize("calc_name", ["ewald", "pme", "p3m"])
@pytest.mark.parametrize("crystal", sorted(CRYSTALS))
def test_madelung_constant(crystal, calc_name):
    pos, charges, cell, ref, units = _crystal(crystal)
    cutoff = 1.0 if calc_name == "ewald" else 2.0
    calc, rtol = _calculator(calc_name, smearing=cutoff / 5.0)
    calc.to(DTYPE)
    pairs, _, dist = periodic_neighbor_list(pos, cell, cutoff, full_list=False, periodic=True)
    potentials = calc.forward(
        positions=pos,
        charges=charges,
        cell=cell,
        neighbor_indices=pairs,
        neighbor_distances=dist,
    )
    madelung = -torch.sum(potentials * charges) / units
    torch.testing.assert_close(madelung, torch.tensor(ref, dtype=DTYPE), atol=0.0, rtol=rtol)


def test_wigner_simple_cubic_with_background():
    """Net-charge cell: one +1 ion per cube, neutralised by a uniform background."""
    pos = torch.zeros(1, 3, dtype=DTYPE)
    charges = torch.ones(1, 1, dtype=DTYPE)
    cell = torch.eye(3, dtype=DTYPE)
    madelung = 1.7601188 / (3 / (4 * math.pi)) ** (1 / 3)  # Wigner-Seitz radius → a = 1
    pairs, _, dist = periodic_neighbor_list(pos, cell, 0.5 - 1e-6, full_list=False, periodic=True)
    calc = EwaldCalculator(InversePowerLawPotential(exponent=1, smearing=0.1), lr_wavelength=0.05)
    calc.to(DTYPE)
    potentials = calc.forward(
        positions=pos,
        charges=charges,
        cell=cell,
        neighbor_indices=pairs,
        neighbor_distances=dist,
    )
    torch.testing.assert_close(
        potentials * charges,
        torch.full_like(potentials, -madelung / 2),
        atol=0.0,
        rtol=4.2e-6,
    )
