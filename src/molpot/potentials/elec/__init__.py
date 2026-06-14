"""molpot.potentials.elec — Differentiable long-range electrostatics with Ewald/PME/P3M.

Provides GPU-accelerated, autograd-compatible computation of long-range
electrostatic and dispersion interactions for periodic molecular systems.

References:
    Ewald, P. Ann. Phys. 369, 253–287 (1921)
    Darden, T. et al. J. Chem. Phys. 98, 10089–10092 (1993)
    Deserno, M. & Holm, C. J. Chem. Phys. 109, 7678–7693 (1998)
"""

from molpot.potentials.elec.calculators import (
    Calculator,
    CalculatorDipole,
    EwaldCalculator,
    P3MCalculator,
    PMECalculator,
)
from molpot.potentials.elec.ewald_multipole import (
    EwaldMultipoleEnergy,
    EwaldMultipoleEnergySpec,
)
from molpot.potentials.elec.potentials import (
    CombinedPotential,
    CoulombPotential,
    InversePowerLawPotential,
    Potential,
    PotentialDipole,
    SplinePotential,
)

__all__ = [
    "Calculator",
    "EwaldCalculator",
    "P3MCalculator",
    "PMECalculator",
    "CalculatorDipole",
    "EwaldMultipoleEnergy",
    "EwaldMultipoleEnergySpec",
    "CoulombPotential",
    "Potential",
    "InversePowerLawPotential",
    "SplinePotential",
    "CombinedPotential",
    "PotentialDipole",
]
