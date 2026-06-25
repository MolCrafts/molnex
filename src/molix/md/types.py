"""Typed tensor containers for the MD engine — compile-friendly pytrees.

:class:`ForceOutput` and :class:`MDState` are ``NamedTuple``s. PyTorch registers
namedtuples as pytree nodes, so they flatten / unflatten transparently under
:func:`torch.compile`, ``torch.func`` and ``torch.utils._pytree`` with no custom
registration. They are the **only** data contract crossing component boundaries
(``ForceField`` → ``Integrator`` → ``MDRunner``); the heterogeneous topology
``TensorDict`` stays inside :class:`~molix.md.forcefield.ForceField` (model I/O).
This is the deliberate ``TensorDict`` ↔ typed-state split: rich dict where the
model needs it, static tensors where the hot loop and type-checking need it.
"""

from __future__ import annotations

from typing import NamedTuple

import torch


class ForceOutput(NamedTuple):
    """Energy + forces from a :class:`~molix.md.forcefield.ForceField`.

    Attributes:
        energy: Scalar total energy ``()``.
        forces: Per-atom forces ``(N, 3)`` ``= -∂E/∂pos``.
    """

    energy: torch.Tensor
    forces: torch.Tensor


class MDState(NamedTuple):
    """Dynamical state advanced one BAOAB step by an ``Integrator``.

    Carries the force-cache (``force`` / ``energy`` at ``pos``, both from the
    same force-field evaluation) so the loop does one evaluation per step.

    Attributes:
        pos: Positions ``(N, 3)``.
        vel: Velocities ``(N, 3)``.
        force: Cached force ``(N, 3)`` at ``pos``.
        energy: Cached scalar energy ``()`` at ``pos``.
    """

    pos: torch.Tensor
    vel: torch.Tensor
    force: torch.Tensor
    energy: torch.Tensor
