"""Gaussian-type-orbital electrostatics for PolarMACE.

A Gaussian-type orbital (GTO) charge model smears each atom's charge and dipole
over a Gaussian of width ``σ`` instead of leaving them at a point. The official
pair kernel is ``erf(r/(2σ))/r`` — the convention of the released MACE-POLAR-1
checkpoint ("the dump" throughout this package), not paper Eq. 37. Coulomb
prefactor is :data:`molpot.potentials.elec.prefactors.eV_A` (eV·Å·e⁻²).

Periodic electrostatics are the SI V.6 reciprocal-space (k-space) sum, not PME
(particle-mesh Ewald). Both periodic blocks —
:class:`~molpot.potentials.gto.hartree.GTOHartreeEnergy` (energy) and
:class:`~molpot.potentials.gto.features.GTOFieldFeatures` (potential / field
descriptors) — evaluate that sum over a
:class:`~molpot.potentials.gto.kspace.KSpaceGrid`, a fixed-capacity integer
grid of reciprocal indices sized **once**. Its lifecycle is
``construct → .to(device) → bind_cell(cell) → compile / export → forward`` each
step: ``bind_cell`` is eager-only, and resizing means building a new module,
never re-binding an existing grid.

Reference:
    Batatia et al. "MACE-POLAR-1: A Polarisable Electrostatic Foundation Model
    for Molecular Chemistry" arXiv:2602.19411.
"""

from molpot.potentials.gto.features import GTOFieldFeatures
from molpot.potentials.gto.fukui import FukuiEquilibration
from molpot.potentials.gto.hartree import GTOHartreeEnergy
from molpot.potentials.gto.kernel import GTOCoulombKernel
from molpot.potentials.gto.kspace import KSpaceGrid, kspace_cutoff

__all__ = [
    "FukuiEquilibration",
    "GTOCoulombKernel",
    "GTOFieldFeatures",
    "GTOHartreeEnergy",
    "KSpaceGrid",
    "kspace_cutoff",
]
