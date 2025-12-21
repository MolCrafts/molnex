"""Classic molecular mechanics potentials.

This module provides implementations of classic force field potentials:
- LJ126: Lennard-Jones 12-6 potential
- BondHarmonic: Harmonic bond stretching potential
- AngleHarmonic: Harmonic angle bending potential
- DihedralHarmonic: Harmonic dihedral torsion potential

All potentials follow the same interface:
- Parameters are type-indexed and stored in __init__
- forward() accepts only AtomicTD and returns energy
"""

from molpot.potentials.base import BasePotential
from molpot.potentials.lj126 import LJ126
from molpot.potentials.bond_harmonic import BondHarmonic
from molpot.potentials.angle_harmonic import AngleHarmonic
from molpot.potentials.dihedral_harmonic import DihedralHarmonic

__all__ = [
    "BasePotential",
    "LJ126",
    "BondHarmonic",
    "AngleHarmonic",
    "DihedralHarmonic",
]
