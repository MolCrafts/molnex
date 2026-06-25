"""Paired reference/quantized trajectory protocol and the trajectory artifact.

The reference (fp64) potential drives a trajectory x(t). Along that same x(t) both
the reference and the quantized potential are evaluated to give the per-frame force
residual ΔF(t) = F_quant - F_ref — the time series the thermal-noise diagnostics
(spec -03) consume. A second, independent quantized-driven trajectory is available
for observable comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from tensordict import TensorDict
from torch import nn

from molix.md.forcefield import PotentialForceField
from molix.md.integrators import LangevinVerletIntegrator

_INTEGRATOR_NAME = "velocity-verlet+langevin-baoab"


@dataclass(frozen=True)
class TrajectoryArtifact:
    """Per-step paired-trajectory record.

    Units follow the caller's ``force_fn`` / integrator (the engine is
    unit-agnostic — see :mod:`molix.md.integrators`); they are not assumed to be
    eV/Å. Shapes: ``pos``/``vel``/``f_ref``/``f_quant``/``df`` are ``(T, N, 3)``;
    ``energy`` is ``(T,)``. ``metadata`` carries the run scalars (``dt``,
    ``gamma``, ``kbt``, ``mass``, ``N``, ``dof``, ``seed``, ``integrator``, plus
    any caller-supplied condition keys). ``dof`` is ``3N`` under Langevin and
    ``3N-3`` under NVE, matching :class:`molix.md.MDRunner`.
    """

    pos: torch.Tensor
    vel: torch.Tensor
    energy: torch.Tensor
    f_ref: torch.Tensor
    f_quant: torch.Tensor
    df: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        return int(self.pos.shape[0])

    @property
    def n_atoms(self) -> int:
        return int(self.pos.shape[1])

    def to_dict(self) -> dict[str, Any]:
        """Flatten to a plain dict for ``torch.save`` (the on-disk ``.pt`` schema)."""
        return {
            "pos": self.pos,
            "vel": self.vel,
            "energy": self.energy,
            "f_ref": self.f_ref,
            "f_quant": self.f_quant,
            "df": self.df,
            "metadata": self.metadata,
        }


def run_trajectory(
    model: nn.Module,
    template: TensorDict,
    pos0: torch.Tensor,
    vel0: torch.Tensor,
    n_steps: int,
    *,
    dt: float,
    gamma: float,
    kbt: float,
    mass: float | torch.Tensor,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Drive ``n_steps`` of Langevin velocity-Verlet using ``model``'s forces."""
    force = PotentialForceField(model, template)
    integ = LangevinVerletIntegrator(force, dt=dt, gamma=gamma, kbt=kbt, mass=mass, seed=seed)
    return integ.run(pos0, vel0, n_steps)


def evaluate_delta_along_trajectory(
    model_ref: nn.Module,
    model_quant: nn.Module,
    template: TensorDict,
    pos_traj: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-frame ``(F_ref, F_quant, ΔF)`` along a fixed position trajectory.

    ``ΔF = F_quant - F_ref`` is a catastrophic-cancellation site (the quantized
    force tracks the reference, so ``|F| / |ΔF| ≫ 1``). The subtraction is done
    in float64 to avoid fp32 round-off (~1e-7·|F|) swamping the residual; for the
    result to be meaningful **both models should also be evaluated in float64**
    (the rounding inside an fp32 forward cannot be recovered here).
    """
    ref_ff = PotentialForceField(model_ref, template)
    quant_ff = PotentialForceField(model_quant, template)
    f_ref = []
    f_quant = []
    for pos in pos_traj:
        f_ref.append(ref_ff.calc_forces(pos))
        f_quant.append(quant_ff.calc_forces(pos))
    f_ref_t = torch.stack(f_ref)
    f_quant_t = torch.stack(f_quant)
    df = (f_quant_t.to(torch.float64) - f_ref_t.to(torch.float64)).to(f_quant_t.dtype)
    return f_ref_t, f_quant_t, df


def build_paired_trajectory(
    model_ref: nn.Module,
    model_quant: nn.Module,
    template: TensorDict,
    pos0: torch.Tensor,
    vel0: torch.Tensor,
    n_steps: int,
    *,
    dt: float,
    gamma: float,
    kbt: float,
    mass: float,
    seed: int = 0,
    condition: dict[str, Any] | None = None,
) -> TrajectoryArtifact:
    """Run the reference trajectory and evaluate ΔF(t) along it.

    Returns a :class:`TrajectoryArtifact` with the reference pos/vel/energy plus the
    paired ``f_ref`` / ``f_quant`` / ``df`` time series and run metadata.
    """
    ref = run_trajectory(
        model_ref, template, pos0, vel0, n_steps, dt=dt, gamma=gamma, kbt=kbt, mass=mass, seed=seed
    )
    f_ref, f_quant, df = evaluate_delta_along_trajectory(
        model_ref, model_quant, template, ref["pos"]
    )
    n_atoms = int(pos0.shape[0])
    # Match MDRunner's temperature convention: Langevin (γ>0) thermostats all 3N
    # DoF including the COM; NVE with COM removed leaves 3N-3.
    dof = 3 * n_atoms - (0 if gamma > 0.0 else 3)
    metadata: dict[str, Any] = {
        "dt": dt,
        "gamma": gamma,
        "kbt": kbt,
        "mass": mass,
        "N": n_atoms,
        "dof": dof,
        "seed": seed,
        "n_steps": n_steps,
        "integrator": _INTEGRATOR_NAME,
    }
    if condition:
        metadata.update(condition)
    return TrajectoryArtifact(
        pos=ref["pos"],
        vel=ref["vel"],
        energy=ref["energy"],
        f_ref=f_ref,
        f_quant=f_quant,
        df=df,
        metadata=metadata,
    )
