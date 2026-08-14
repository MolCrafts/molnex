"""Component-based, compilable in-process MD engine.

:class:`~molix.md.driver.MD` is the composer: ``MD(dtype).set_forcefield(ff)
.set_integrator(Integrator(dt)).to(device)``. It owns the MD-side precision
only. The neighbour-list cadence belongs to
:class:`~molix.md.neighbors.NeighborList`. The lower layers are the primitives
the caller composes:

* :class:`~molix.md.types.ForceOutput` / :class:`~molix.md.types.MDState` /
  :class:`~molix.md.types.MDObservables` — typed pytree contracts crossing
  component boundaries.
* :class:`~molix.md.forcefield.ForceField` (``PotentialForceField`` /
  ``PeriodicPotentialForceField`` over a TensorDict potential;
  ``CallableForceField`` over any ``pos -> (energy, forces)`` callable;
  ``HarmonicForceField`` / ``LennardJonesForceField`` analytic;
  ``LennardJonesCutForceField`` — periodic truncated-shifted LJ over a
  rebuildable neighbour list, the bulk lj/cut production path) — binds a
  model to a system, maps positions to ``(energy, forces)``.
* :class:`~molix.md.integrators.Integrator` /
  :class:`~molix.md.integrators.VelocityVerletIntegrator` (NVE) /
  :class:`~molix.md.integrators.LangevinVerletIntegrator` (γ>0) — constructed
  with ``dt`` (Langevin also γ / kT); mass is ``atoms.mass``, bound at run.
* :class:`~molix.md.runner.MDRunner` — drives the integrator through the
  :class:`~molix.md.runner.MDHook` lifecycle;
  :class:`~molix.md.runner.TrajectoryHook` captures trajectories,
  :class:`~molix.md.runner.MDCheckpointHook` persists restartable state.
* :class:`~molix.md.driver.MaxwellBoltzmann` — writes ``atoms.vel`` from ``atoms.mass``.

Periodic systems are supported through
:class:`~molix.md.neighbors.NeighborList`, which **owns the rebuild policy**:
a Verlet ``skin`` under the LAMMPS ``every`` / ``delay`` / ``check`` gate,
rebuilt in place into fixed-capacity buffers so the force path can stay inside
a CUDA graph. There is exactly one caller —
:meth:`~molix.md.integrators.Integrator.eval_force` asks once per force
evaluation, at the positions being evaluated, iff the force field declares
:attr:`~molix.md.forcefield.ForceField.rebuilds_neighbors`; no driver kwarg and
no step-start hook. A force field that keeps its list frozen (the default for
:class:`~molix.md.forcefield.PotentialForceField`, and any integrator built with
``rebuild=False``) remains valid only for open systems or trajectories short
enough that no atom changes neighbours.
"""

from molix.md.driver import MD, MaxwellBoltzmann
from molix.md.forcefield import (
    CallableForceField,
    ForceField,
    HarmonicForceField,
    LennardJonesCutForceField,
    LennardJonesForceField,
    PeriodicPotentialForceField,
    PotentialForceField,
)
from molix.md.integrators import Integrator, LangevinVerletIntegrator, VelocityVerletIntegrator
from molix.md.neighbors import NeighborList, NeighborStrategy
from molix.md.runner import (
    MDCheckpointHook,
    MDHook,
    MDRunner,
    TrajectoryHook,
)
from molix.md.types import ForceOutput, MDObservables, MDState
from molix.units import units

__all__ = [
    "MD",
    "CallableForceField",
    "ForceField",
    "ForceOutput",
    "HarmonicForceField",
    "Integrator",
    "LangevinVerletIntegrator",
    "LennardJonesCutForceField",
    "LennardJonesForceField",
    "MDCheckpointHook",
    "MDHook",
    "MDObservables",
    "MDRunner",
    "MDState",
    "MaxwellBoltzmann",
    "NeighborList",
    "NeighborStrategy",
    "PeriodicPotentialForceField",
    "PotentialForceField",
    "TrajectoryHook",
    "VelocityVerletIntegrator",
    "units",
]
