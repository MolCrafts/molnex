"""Component-based, compilable in-process MD engine.

:class:`~molix.md.driver.MD` is the composer: ``MD(dtype).set_forcefield(ff)
.set_neighbors(nl).set_integrator(Integrator(dt)).to(device)``. It owns the
MD-side precision only. :class:`~molix.md.forcefield.ForceField` is
identically :class:`molpy.ForceField`. The neighbour-list cadence belongs to
:class:`~molix.md.neighbors.NeighborList`. The lower layers are the primitives
the caller composes:

* :class:`~molix.md.types.ForceOutput` / :class:`~molix.md.types.MDState` /
  :class:`~molix.md.types.MDObservables` — typed pytree contracts crossing
  component boundaries.
* :class:`~molix.md.forcefield.ForceField` — ``molpy.ForceField`` re-export;
  mount with :meth:`~molix.md.driver.MD.set_forcefield`.
* :class:`~molix.md.integrators.Integrator` /
  :class:`~molix.md.integrators.VelocityVerletIntegrator` (NVE) /
  :class:`~molix.md.integrators.LangevinVerletIntegrator` (γ>0) — constructed
  with ``dt`` (Langevin also γ / kT); mass is ``atoms.mass``, bound at run.
* :class:`~molix.md.runner.MDRunner` — drives the integrator through the
  :class:`~molix.md.runner.MDHook` lifecycle;
  :class:`~molix.md.runner.TrajectoryHook` captures trajectories,
  :class:`~molix.md.runner.MDCheckpointHook` persists restartable state.
* :class:`~molix.md.driver.MaxwellBoltzmann` — velocity distribution;
  :class:`~molix.md.runner.VelocityInitHook` applies it at ``on_init``.

Periodic systems are supported through
:class:`~molix.md.neighbors.NeighborList`, which **owns the rebuild policy**:
a Verlet ``skin`` under the LAMMPS ``every`` / ``delay`` / ``check`` gate,
rebuilt in place into fixed-capacity buffers so the force path can stay inside
a CUDA graph. Mount it with :meth:`~molix.md.driver.MD.set_neighbors`. There
is exactly one caller — :meth:`~molix.md.integrators.Integrator.eval_force`
asks once per force evaluation, at the positions being evaluated, iff a list
is mounted. An integrator built with ``rebuild=False`` freezes the list.
"""

from molix.md.driver import MD, MaxwellBoltzmann
from molix.md.forcefield import ForceField
from molix.md.integrators import Integrator, LangevinVerletIntegrator, VelocityVerletIntegrator
from molix.md.neighbors import NeighborList, NeighborStrategy
from molix.md.runner import (
    MDCheckpointHook,
    MDHook,
    MDRunner,
    TrajectoryHook,
    VelocityInitHook,
)
from molix.md.types import ForceOutput, MDObservables, MDState
from molix.units import units

__all__ = [
    "ForceField",
    "ForceOutput",
    "Integrator",
    "LangevinVerletIntegrator",
    "MD",
    "MDCheckpointHook",
    "MDHook",
    "MDObservables",
    "MDRunner",
    "MDState",
    "MaxwellBoltzmann",
    "NeighborList",
    "NeighborStrategy",
    "TrajectoryHook",
    "VelocityInitHook",
    "VelocityVerletIntegrator",
    "units",
]
