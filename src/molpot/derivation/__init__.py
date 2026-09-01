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

Beside those batch-level passes sit the tensor-level derivation types — one
stateless :class:`torch.nn.Module` per differentiated quantity, reusable across
batches: :class:`~molpot.derivation.force.ForceDerivation` for the first
derivative ``F = -∂E/∂pos`` (eV/Å for an eV core),
:class:`~molpot.derivation.hessian.HessianDerivation` for the Cartesian second
derivative ``H = ∂²E/∂pos²`` (energy/Å²; dense ``(3N, 3N)`` or one
``(3Nᵢ, 3Nᵢ)`` block per molecule), and
:class:`~molpot.derivation.stress.StressDerivation` for the stress tensor
``σ = (1/V) ∂E/∂ε`` (energy/Å³).
"""

from molpot.derivation.energy import EnergyAggregation
from molpot.derivation.energy_readout import EnergyReadout
from molpot.derivation.force import (
    ForceDerivation,
    autograd_forces,
    autograd_forces_and_virial,
    autograd_forces_from_energy,
    functorch_forces,
    functorch_forces_with_aux,
)
from molpot.derivation.force_readout import ForceReadout
from molpot.derivation.hessian import HessianDerivation
from molpot.derivation.kernels import func_force_pass, grad_force_pass
from molpot.derivation.stress import StressDerivation

__all__ = [
    "EnergyAggregation",
    "EnergyReadout",
    "ForceDerivation",
    "ForceReadout",
    "HessianDerivation",
    "StressDerivation",
    "autograd_forces",
    "autograd_forces_and_virial",
    "autograd_forces_from_energy",
    "func_force_pass",
    "functorch_forces",
    "functorch_forces_with_aux",
    "grad_force_pass",
]
