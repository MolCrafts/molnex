"""Charge equilibration: χ,η → partial charges (algorithm-named solvers).

::

    from molpot.charge import (
        AtomicChargeModel,
        ChiEtaHead,
        DiagonalHardnessKernel,
        ElectronegativityEqualization,
    )

    model = AtomicChargeModel(
        ChiEtaHead(feature_dim=128),
        ElectronegativityEqualization(DiagonalHardnessKernel()),
    )
    q = model.charges(atom_features, batch)
    ff = model.write_forcefield(ff_template, atom_features, batch)
"""

from molpot.charge.assignment import AtomicChargeModel
from molpot.charge.chi_eta import ChiEtaHead
from molpot.charge.kernels import DiagonalHardnessKernel, ShieldedCoulombKernel
from molpot.charge.problem import ChargeEquilibrationProblem
from molpot.charge.solvers import ElectronegativityEqualization

__all__ = [
    "AtomicChargeModel",
    "ChargeEquilibrationProblem",
    "ChiEtaHead",
    "DiagonalHardnessKernel",
    "ElectronegativityEqualization",
    "ShieldedCoulombKernel",
]
