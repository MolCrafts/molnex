"""Physical quantity derivation: aggregation and the tensor-level derivatives.

Batch-level derivatives — ``F = -∂E/∂pos``, ``W = ∂E/∂ε`` — are
:class:`molix.autograd.Derivative`, composed over any ``energy(frame) ->
frame`` (a potential's ``energy`` method, or a potential built with
``wrt=...``). What lives here is the raw-tensor side:
:class:`~molpot.derivation.force.ForceDerivation` /
:func:`~molpot.derivation.force.autograd_forces` for an ``energy_fn(pos) ->
scalar``, :class:`~molpot.derivation.hessian.HessianDerivation` for the
Cartesian second derivative ``H = ∂²E/∂pos²``, and
:class:`~molpot.derivation.energy.EnergyAggregation` for the per-graph sum.
"""

from molpot.derivation.energy import EnergyAggregation
from molpot.derivation.force import (
    ForceDerivation,
    autograd_forces,
    autograd_forces_from_energy,
)
from molpot.derivation.hessian import HessianDerivation

__all__ = [
    "EnergyAggregation",
    "ForceDerivation",
    "HessianDerivation",
    "autograd_forces",
    "autograd_forces_from_energy",
]
