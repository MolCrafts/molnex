"""Batch-level differentiation: one op, addressed by TensorFrame keys.

:class:`Derivative` is the only derivative in molnex — ``F = -∂E/∂pos``,
``W = ∂E/∂ε``, ``∂E/∂(any column)`` — composed around any ``energy(frame) ->
frame`` and compiled like any module with :class:`molix.compile.Compiler`.
:func:`leaf` is the ``requires_grad`` copy it differentiates against and
:func:`symmetrize_strain_grad` the strain-gradient convention every virial
goes through; both are pure math with no natural owner beyond this package.
"""

from molix.autograd.derivative import Derivative
from molix.autograd.variables import leaf, symmetrize_strain_grad

__all__ = ["Derivative", "leaf", "symmetrize_strain_grad"]
