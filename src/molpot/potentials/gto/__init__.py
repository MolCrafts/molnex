"""Gaussian-type-orbital electrostatics for PolarMACE.

Official kernel is ``erf(r/(2σ))/r`` (dump, not paper Eq.37). Periodic
electrostatics are the SI V.6 k-space sum, not PME. Coulomb prefactor is
:data:`molpot.potentials.elec.prefactors.eV_A` (eV·Å·e⁻²).

Reference:
    Batatia et al. "MACE-POLAR-1: A Polarisable Electrostatic Foundation Model
    for Molecular Chemistry" arXiv:2602.19411.
"""

from molpot.potentials.gto.features import GTOFieldFeatures
from molpot.potentials.gto.fukui import FukuiEquilibration
from molpot.potentials.gto.hartree import GTOHartreeEnergy
from molpot.potentials.gto.kernel import GTOCoulombKernel
from molpot.potentials.gto.kspace import hartree_energy, kspace_cutoff

__all__ = [
    "FukuiEquilibration",
    "GTOCoulombKernel",
    "GTOFieldFeatures",
    "GTOHartreeEnergy",
    "hartree_energy",
    "kspace_cutoff",
]
