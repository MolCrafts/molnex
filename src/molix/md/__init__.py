"""Component-based, compilable in-process MD engine.

:class:`~molix.md.driver.MD` is **the** entry point: it binds a force field to
an integrator, owns the MD-side precision and the neighbour-list cadence, and
delegates the loop to :class:`~molix.md.runner.MDRunner`. The lower layers are
the primitives it composes (use them directly only when you need a custom
loop):

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
  :class:`~molix.md.integrators.LangevinVerletIntegrator` — advances an
  ``MDState`` (BAOAB); ``step`` / ``rollout`` ``torch.compile(fullgraph=True)``
  to a single graph including a traceable force field. ``advance_n`` is the
  eager chunk driver (γ=0 skips the noise draw; bit-identical dynamics).
* :class:`~molix.md.runner.MDRunner` — drives the integrator through the
  :class:`~molix.md.runner.MDHook` lifecycle;
  :class:`~molix.md.runner.TrajectoryHook` captures trajectories,
  :class:`~molix.md.runner.NeighborListHook` refreshes the neighbour list,
  :class:`~molix.md.runner.MDCheckpointHook` persists restartable state.
* :class:`~molix.md.driver.MaxwellBoltzmann` — initial-velocity sampler.

Periodic systems are supported through
:class:`~molix.md.neighbors.NeighborList`, which rebuilds the neighbour
list on a step cadence into fixed-capacity buffers so the force path can stay
inside a CUDA graph. A force field that keeps its list frozen (the default for
:class:`~molix.md.forcefield.PotentialForceField`) remains valid only for open
systems or trajectories short enough that no atom changes neighbours.
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
from molix.md.integrators import Integrator, LangevinVerletIntegrator
from molix.md.neighbors import NeighborList, NeighborStrategy
from molix.md.runner import (
    MDCheckpointHook,
    MDHook,
    MDRunner,
    NeighborListHook,
    TrajectoryHook,
)
from molix.md.types import ForceOutput, MDObservables, MDState
from molix.units import EV_PER_AMU_A2_FS2, KB_AMU_A_FS, KB_EV_PER_K

__all__ = [
    "EV_PER_AMU_A2_FS2",
    "KB_AMU_A_FS",
    "KB_EV_PER_K",
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
    "NeighborListHook",
    "NeighborStrategy",
    "PeriodicPotentialForceField",
    "PotentialForceField",
    "TrajectoryHook",
]
