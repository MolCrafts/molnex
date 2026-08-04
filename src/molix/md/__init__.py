"""Component-based, compilable in-process MD engine.

Component layecake (mirrors molpy's ``Potential`` vs ``ForceField`` split):

* :class:`~molix.md.types.ForceOutput` / :class:`~molix.md.types.MDState` —
  typed pytree contracts crossing component boundaries.
* :class:`~molix.md.forcefield.ForceField` (``PotentialForceField`` over a
  molpot Potential; ``HarmonicForceField`` / ``LennardJonesForceField`` analytic)
  — binds a model to a system, maps positions to ``(energy, forces)``.
* :class:`~molix.md.integrators.LangevinVerletIntegrator` — advances an
  ``MDState`` (BAOAB); ``step`` / ``rollout`` ``torch.compile(fullgraph=True)``
  to a single graph including a traceable force field.
* :class:`~molix.md.runner.MDRunner` — drives the integrator through the molix
  hook lifecycle; :class:`~molix.md.runner.TrajectoryHook` captures trajectories.

Scope: open (non-periodic) systems, short small-displacement trajectories. The
neighbour list (``edge_index``) is frozen for the whole run — there is no
rebuild — so this is a study/inference engine for near-equilibrium dynamics, not
general production MD. See :class:`~molix.md.forcefield.PotentialForceField`.
"""

from molix.md.dynamics import (
    TrajectoryArtifact,
    build_paired_trajectory,
    evaluate_delta_along_trajectory,
    run_trajectory,
)
from molix.md.forcefield import (
    ForceField,
    HarmonicForceField,
    LennardJonesForceField,
    PotentialForceField,
)
from molix.md.integrators import (
    EV_PER_AMU_A2_FS2,
    Integrator,
    LangevinVerletIntegrator,
    as_mass_col,
)
from molix.md.runner import MDRunner, TrajectoryHook
from molix.md.types import ForceOutput, MDState

__all__ = [
    "EV_PER_AMU_A2_FS2",
    "ForceField",
    "ForceOutput",
    "HarmonicForceField",
    "Integrator",
    "LangevinVerletIntegrator",
    "LennardJonesForceField",
    "MDRunner",
    "MDState",
    "PotentialForceField",
    "TrajectoryArtifact",
    "TrajectoryHook",
    "as_mass_col",
    "build_paired_trajectory",
    "evaluate_delta_along_trajectory",
    "run_trajectory",
]
