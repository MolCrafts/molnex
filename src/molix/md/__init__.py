"""In-process molecular-dynamics driver: Langevin velocity-Verlet integration.

Model-agnostic integrator (BAOAB Langevin splitting) plus a force seam that wraps
``PiNetPotential`` with a live (grad-tracking) position leaf, paired reference vs
quantized trajectory runners, and an optional ASE-Calculator shell. Used by the
PiNet quantization-as-thermal-noise study to generate paired trajectories whose
force residual ΔF(t) is analyzed for Langevin-noise behavior.
"""

from molix.md.ase_shim import HAS_ASE, make_pinet_calculator
from molix.md.dynamics import (
    TrajectoryArtifact,
    build_paired_trajectory,
    evaluate_delta_along_trajectory,
    run_trajectory,
)
from molix.md.force_seam import build_force_fn
from molix.md.integrators import LangevinVerletIntegrator
from molix.md.runner import MDRunner, TrajectoryHook

__all__ = [
    "HAS_ASE",
    "LangevinVerletIntegrator",
    "MDRunner",
    "TrajectoryArtifact",
    "TrajectoryHook",
    "build_force_fn",
    "build_paired_trajectory",
    "evaluate_delta_along_trajectory",
    "make_pinet_calculator",
    "run_trajectory",
]
