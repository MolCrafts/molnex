"""Typed tensor containers for the MD engine — compile-friendly pytrees.

:class:`MDState` and :class:`MDObservables` are ``NamedTuple``s. PyTorch
registers namedtuples as pytree nodes, so they flatten / unflatten under
:func:`torch.compile` / ``torch.func`` with no custom registration.

The force provider is **not** a third type: it returns a plain
``dict[str, Tensor]`` with required ``energy`` / ``forces`` and optional
``virial``. Extra keys are allowed and ignored by the integrator (charges
and the like already live on the work ``TensorDict``). The dict is a pytree;
:meth:`~molix.md.integrators.Integrator.eval_force` tree-maps it to the
state dtype. That is the TensorDict ↔ typed-state split: a dict at the
force seam, a NamedTuple only for the state that the integrator actually
steps.
"""

from __future__ import annotations

from typing import NamedTuple

import torch


class MDState(NamedTuple):
    """Dynamical state advanced one step by an ``Integrator``.

    Carries the force-cache (``forces`` / ``energy`` at ``pos``, both from the
    same force-field evaluation) so the loop does one evaluation per step.

    Attributes:
        pos: Positions ``(N, 3)``.
        vel: Velocities ``(N, 3)``.
        forces: Cached forces ``(N, 3)`` at ``pos``.
        energy: Cached scalar energy ``()`` at ``pos``.
        cell: Optional lattice vectors ``(3, 3)`` Å (NPT).
        virial: Optional cached ``∂E/∂ε`` ``(3, 3)`` at ``pos`` / ``cell``.
    """

    pos: torch.Tensor
    vel: torch.Tensor
    forces: torch.Tensor
    energy: torch.Tensor
    cell: torch.Tensor | None = None
    virial: torch.Tensor | None = None


class MDObservables(NamedTuple):
    """Per-observation thermodynamic snapshot handed to MD hooks.

    Produced by :class:`~molix.md.runner.MDRunner` after each hook-visible
    chunk of steps; consumed by :class:`~molix.md.runner.MDHook.on_step_end`
    implementations (trajectory capture, checkpointing, logging).

    Attributes:
        pos: Positions ``(N, 3)``.
        vel: Velocities ``(N, 3)``.
        forces: Forces ``(N, 3)`` at ``pos``.
        potential: Scalar potential energy ``()``.
        kinetic: Scalar kinetic energy ``()``.
        total: Scalar total energy ``()`` (``potential + kinetic``).
        temperature: Scalar instantaneous temperature ``()`` in kelvin (for the
            runner's default ``kb``).
        cell: Optional lattice ``(3, 3)`` Å.
        virial: Optional ``∂E/∂ε`` ``(3, 3)``.
        volume: Optional cell volume ``()`` Å³.
        pressure: Optional instantaneous pressure ``()`` in MD units
            (amu Å⁻¹ fs⁻²).
    """

    pos: torch.Tensor
    vel: torch.Tensor
    forces: torch.Tensor
    potential: torch.Tensor
    kinetic: torch.Tensor
    total: torch.Tensor
    temperature: torch.Tensor
    cell: torch.Tensor | None = None
    virial: torch.Tensor | None = None
    volume: torch.Tensor | None = None
    pressure: torch.Tensor | None = None
