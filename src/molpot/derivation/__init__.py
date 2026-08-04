"""Physical quantity derivation: aggregation + in-place derivative readouts.

Public call shape (``model`` is always the first argument)::

    # energy only — no Derivative session
    batch = EnergyReadout(model, method="func", backward=False)(batch)

    # energy + force as a sequential pair (one energy eval when backward=True)
    batch = EnergyReadout(model, method="func", backward=True)(batch)
    batch = ForceReadout(model, method="func")(batch)

Or a potential that fixes forces at init (monomorphic pipeline, preferred)::

    model = PiNetPotential(..., compute_forces=True, method="func")
    batch = model(batch)
    model.compile()  # optional torch.compile of that static forward
"""

from molpot.derivation.energy import EnergyAggregation
from molpot.derivation.energy_readout import EnergyReadout
from molpot.derivation.force import (
    ForceDerivation,
    autograd_forces,
    autograd_forces_from_energy,
    functorch_forces,
    functorch_forces_with_aux,
)
from molpot.derivation.force_readout import ForceReadout
from molpot.derivation.stress import StressDerivation

__all__ = [
    "EnergyAggregation",
    "EnergyReadout",
    "ForceReadout",
    "ForceDerivation",
    "StressDerivation",
    "autograd_forces",
    "autograd_forces_from_energy",
    "functorch_forces",
    "functorch_forces_with_aux",
]
