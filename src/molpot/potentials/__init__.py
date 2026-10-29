"""Molecular mechanics potentials.

- LJ126: Lennard-Jones 12-6
- BondHarmonic / AngleHarmonic / DihedralPeriodic / DihedralOpls / ImproperCvff /
  LJCut / PairCoulCut: the molrs styles, LAMMPS conventions, one class per
  style; ``Potentials.from_forcefield(ff)`` builds them from a molpy ForceField
- the fixed (tabulated) path's forms — ``bonds.harmonic``, ``angles.harmonic``,
  ``dihedrals``, ``impropers`` — are reached by module path only
- RepulsionExp6: Buckingham-style exponential repulsion
- DispersionC6: rationally damped C6 dispersion
- ChargeTransfer: Charge-transfer potential
- Polarization: Self-consistent induced-dipole polarization
- ZBLRepulsion: Ziegler-Biersack-Littmark screened nuclear repulsion (MACE form)
- UniversalZBL: LAMMPS pair_zbl universal form (nequip / Allegro)
"""

# The torch twins of the molrs styles. Importing each class registers its
# style, which is what lets ``Potentials.from_forcefield`` resolve every built-in one.
from molpot.potentials.angle import AngleHarmonic
from molpot.potentials.base import BasePotential
from molpot.potentials.bond import BondHarmonic
from molpot.potentials.collection import Potentials
from molpot.potentials.dihedral import DihedralOpls, DihedralPeriodic
from molpot.potentials.elec import (
    EwaldMultipoleEnergy,
    EwaldMultipoleEnergySpec,
)
from molpot.potentials.improper import ImproperCvff, ImproperPeriodic
from molpot.potentials.mixing import geometric_arithmetic_mixing
from molpot.potentials.nonbonded import (
    ChargeTransfer,
    DispersionC6,
    RepulsionExp6,
    ct_mixing,
    dispersion_mixing,
    repulsion_mixing,
)
from molpot.potentials.pair import LJCut, PairCoulCut
from molpot.potentials.polarization import Polarization
from molpot.potentials.registry import register, resolve
from molpot.potentials.repulsion import UniversalZBL, ZBLRepulsion
from molpot.potentials.vdw import (
    LJ126,
    MixingRule,
    geometric_mean,
    geometric_mixing,
    lorentz_berthelot,
)

__all__ = [
    "BasePotential",
    "LJ126",
    "MixingRule",
    "geometric_mean",
    "geometric_mixing",
    "lorentz_berthelot",
    "BondHarmonic",
    "AngleHarmonic",
    "DihedralPeriodic",
    "DihedralOpls",
    "ImproperCvff",
    "ImproperPeriodic",
    "LJCut",
    "PairCoulCut",
    "EwaldMultipoleEnergy",
    "EwaldMultipoleEnergySpec",
    "geometric_arithmetic_mixing",
    "RepulsionExp6",
    "DispersionC6",
    "ChargeTransfer",
    "repulsion_mixing",
    "dispersion_mixing",
    "ct_mixing",
    "Polarization",
    "UniversalZBL",
    "ZBLRepulsion",
    "Potentials",
    "register",
    "resolve",
]
