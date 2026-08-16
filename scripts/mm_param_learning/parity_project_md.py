#!/usr/bin/env python3
"""Parity: continuous Class-I (Fourier) vs project(OPLS) — E/F + short NVE MD.

Uses a synthetic OPLS-manifold molecule (n=1..4, γ∈{0,π}) so projection is
exact. Prints single-point and trajectory agreement metrics.

Example::

    PYTHONPATH=src python scripts/mm_param_learning/parity_project_md.py
    PYTHONPATH=src python scripts/mm_param_learning/parity_project_md.py --steps 1000
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
from tensordict import TensorDict

# repo root on path when run as a script
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from molix.md import (  # noqa: E402
    MD,
    MDHook,
    MDObservables,
    MDRunner,
    VelocityVerletIntegrator,
)
from molix.units import units  # noqa: E402

_KCAL_PER_MD = (
    (1.0 * units.kilocalorie_per_mole).to("amu * angstrom ** 2 / femtosecond ** 2").magnitude
)
from molpot.forcefield import ClassicalIMM, ClassIForceField, ForceField  # noqa: E402


class _PosHook(MDHook):
    """Record positions and total energy every step."""

    def __init__(self) -> None:
        self.pos: list[torch.Tensor] = []
        self.etot: list[float] = []

    def on_step_end(self, runner: MDRunner, step: int, obs: MDObservables) -> None:
        self.pos.append(obs.pos.detach().cpu().clone())
        self.etot.append(float(obs.total.detach().cpu()))


class _ConstBond(torch.nn.Module):
    def __init__(self, k: float, r0: float) -> None:
        super().__init__()
        self.k, self.r0 = k, r0

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        n = features.shape[0]
        return {
            "k": torch.full((n,), self.k, dtype=features.dtype, device=features.device),
            "r0": torch.full((n,), self.r0, dtype=features.dtype, device=features.device),
        }


class _OplsFourierProper(torch.nn.Module):
    """OPLS-manifold multi-term Fourier (exact project target)."""

    def __init__(self, K: tuple[float, float, float, float] = (1.5, 3.0, 0.5, 0.1)) -> None:
        super().__init__()
        self.K = K

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        n = features.shape[0]
        device, dtype = features.device, features.dtype
        K = torch.tensor(self.K, device=device, dtype=dtype)
        k = (0.5 * K).unsqueeze(0).expand(n, 4).contiguous()
        phase = torch.zeros(n, 4, device=device, dtype=dtype)
        phase[:, 1] = math.pi
        phase[:, 3] = math.pi
        return {
            "k": k,
            "phase": phase,
            "periodicity": torch.tensor([1, 2, 3, 4], device=device, dtype=torch.long),
            "idivf": torch.ones(n, device=device, dtype=dtype),
        }


class _FakeEncoder(torch.nn.Module):
    def __init__(self, features: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self._features = features
        self._p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, td: TensorDict) -> TensorDict:
        return td

    def embeddings(self, td: TensorDict):
        class _E:
            def __init__(self, f):
                self._f = f

            def interaction_dict(self):
                return dict(self._f)

        return _E(self._features)


def butane_like_batch(*, dtype: torch.dtype = torch.float64) -> TensorDict:
    pos = torch.tensor(
        [
            [1.0, 0.0, 0.2],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 1.0, 0.3],
        ],
        dtype=dtype,
    )
    return TensorDict(
        {
            "atoms": TensorDict(
                {
                    "pos": pos,
                    "Z": torch.tensor([6, 6, 6, 6]),
                    "batch": torch.zeros(4, dtype=torch.long),
                },
                batch_size=[4],
            ),
            "bonds": TensorDict(
                {
                    "atomi": torch.tensor([0, 1, 2], dtype=torch.long),
                    "atomj": torch.tensor([1, 2, 3], dtype=torch.long),
                },
                batch_size=[3],
            ),
            "propers": TensorDict(
                {
                    "atomi": torch.tensor([0], dtype=torch.long),
                    "atomj": torch.tensor([1], dtype=torch.long),
                    "atomk": torch.tensor([2], dtype=torch.long),
                    "atoml": torch.tensor([3], dtype=torch.long),
                },
                batch_size=[1],
            ),
        },
        batch_size=[],
    )


def run_nve(
    ff: ForceField,
    batch: TensorDict,
    *,
    vel: torch.Tensor,
    mass: torch.Tensor,
    dt: float,
    n_steps: int,
    dtype: torch.dtype,
) -> tuple[list[torch.Tensor], list[float], float]:
    hook = _PosHook()
    md = (
        MD(dtype)
        .set_forcefield(ff)
        .set_integrator(VelocityVerletIntegrator(dt))
        .set_hooks([hook])
    )
    batch["atoms", "pos"] = batch["atoms", "pos"].to(dtype).clone()
    batch["atoms", "vel"] = vel.to(dtype)
    batch["atoms", "mass"] = mass.to(dtype)
    t0 = time.perf_counter()
    md.run(batch, n_steps)
    elapsed = time.perf_counter() - t0
    return hook.pos, hook.etot, elapsed


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--dt", type=float, default=0.5, help="timestep fs")
    p.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    args = p.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    torch.manual_seed(0)
    batch = butane_like_batch(dtype=dtype)

    src_ff = ClassIForceField()
    model = ClassicalIMM(
        _FakeEncoder(
            {
                "bonds": torch.zeros(3, 2, dtype=dtype),
                "propers": torch.zeros(1, 4, dtype=dtype),
            }
        ),
        forcefield=src_ff,
        bond_head=_ConstBond(200.0, 1.53),
        proper_head=_OplsFourierProper(),
    ).to(dtype=dtype)

    filled = model(batch)
    # Target form matches source terms (no angles in this fixture)
    tgt = ForceField(name="OPLS", units="real").def_bondstyle("harmonic").def_dihedralstyle("opls")
    projected = model.project(tgt, source=filled)

    # --- single-point ---
    e_src = filled.to_potentials().calc_energy(batch).detach()
    e_prj = projected.to_potentials().calc_energy(batch).detach()
    f_src = filled.to_potentials().calc_forces(batch).detach()
    f_prj = projected.to_potentials().calc_forces(batch).detach()
    de = float((e_src - e_prj).abs())
    df = float((f_src - f_prj).abs().max())
    print("=== single-point (kcal/mol) ===")
    print(f"  E_fourier = {float(e_src):.10f}")
    print(f"  E_opls    = {float(e_prj):.10f}")
    print(f"  |ΔE|      = {de:.3e}")
    print(f"  max|ΔF|   = {df:.3e}")

    # --- NVE ---
    n = batch["atoms", "pos"].shape[0]
    mass = torch.full((n,), 12.011, dtype=dtype)
    # small thermal kick in MD units
    vel = 1e-4 * torch.randn(n, 3, dtype=dtype)
    vel -= vel.mean(dim=0, keepdim=True)

    pos_s, etot_s, t_s = run_nve(
        filled, batch, vel=vel, mass=mass, dt=args.dt, n_steps=args.steps, dtype=dtype
    )
    pos_p, etot_p, t_p = run_nve(
        projected, batch, vel=vel, mass=mass, dt=args.dt, n_steps=args.steps, dtype=dtype
    )

    max_dx = max(float((a - b).norm(dim=-1).max()) for a, b in zip(pos_s, pos_p, strict=True))
    drift_s = max(etot_s) - min(etot_s) if etot_s else float("nan")
    drift_p = max(etot_p) - min(etot_p) if etot_p else float("nan")
    max_detot = max(abs(a - b) for a, b in zip(etot_s, etot_p, strict=True))

    print(f"\n=== NVE ({args.steps} steps, dt={args.dt} fs, γ=0) ===")
    print(f"  max ‖Δx‖ (fourier vs opls) = {max_dx:.3e} Å")
    print(f"  max |ΔE_tot| between runs  = {max_detot:.3e} (MD energy units)")
    print(f"  E_tot range fourier        = {drift_s:.3e}")
    print(f"  E_tot range opls           = {drift_p:.3e}")
    print(f"  wall fourier               = {t_s * 1e3:.2f} ms  ({args.steps / t_s:.0f} steps/s)")
    print(f"  wall opls                  = {t_p * 1e3:.2f} ms  ({args.steps / t_p:.0f} steps/s)")

    ok_e = de < 1e-9
    ok_f = df < 1e-7
    ok_x = max_dx < 1e-6
    ok = ok_e and ok_f and ok_x
    print(f"\n{'PASS' if ok else 'FAIL'}: E={ok_e} F={ok_f} traj={ok_x}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
