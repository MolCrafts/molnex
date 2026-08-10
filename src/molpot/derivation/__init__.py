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

Potentials and modes do not hand-roll the pass body: they bind one of the two
shared batch-level kernels (``energy_core(batch) -> batch`` in, batch with
``graphs.energy`` / ``atoms.forces`` out)::

    batch = grad_force_pass(self._write_energy, batch, detach_energy=False)
    batch = func_force_pass(self._write_energy, batch)
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
from molpot.derivation.kernels import func_force_pass, grad_force_pass
from molpot.derivation.stress import StressDerivation

__all__ = [
    "EnergyAggregation",
    "EnergyReadout",
    "ForceReadout",
    "ForceDerivation",
    "StressDerivation",
    "autograd_forces",
    "autograd_forces_from_energy",
    "func_force_pass",
    "functorch_forces",
    "functorch_forces_with_aux",
    "grad_force_pass",
]
