"""LJ-cluster NVE energy-conservation validation (spec md-component-engine ac-008).

Drives a Lennard-Jones cluster (argon parameters, the LJ13 icosahedral minimum)
under NVE (γ=0) with the compiled MD engine for a long trajectory, and checks the
total energy shows no systematic drift — the gold-standard integrator test.

Units: (amu, Å, fs) with energy in amu·Å²/fs² (see molix.md.integrators). All
pairs interact (no cutoff/neighbour list) so the PES is exact for any
displacement. The single BAOAB step is ``torch.compile``d (LJ is analytic — no
scatter, so the Inductor CPU backend works) and looped in Python; 5 ns is only
feasible because the step is compiled.

Run::

    PYTHONPATH=src:. python benchmarks/verify_md_lj_nve.py            # full 5 ns
    PYTHONPATH=src:. python benchmarks/verify_md_lj_nve.py --ps 100   # short smoke

Pass when ``|slope·duration| / |E_tot(0)| < 1e-3`` with bounded RMS fluctuation.
"""

from __future__ import annotations

import argparse
import time

import torch

from molix.md import LangevinVerletIntegrator, LennardJonesForceField
from molix.md.runner import KB_AMU_A_FS

# Argon in (amu, Å, fs): ε = 0.0103 eV, σ = 3.4 Å, m = 39.95 amu.
_EPS = 0.0103 / 103.6426965638  # eV -> amu·Å²/fs²
_SIGMA = 3.4
_MASS = 39.95
_DT = 5.0  # fs — ~1/350 of the LJ vibrational period; comfortably stable
_DTYPE = torch.float64


def _icosahedron13(r_min: float) -> torch.Tensor:
    """LJ13 icosahedral minimum: centre + 12 vertices at the LJ nearest distance."""
    phi = (1.0 + 5.0**0.5) / 2.0
    verts = []
    for a, b in ((1.0, phi), (-1.0, phi), (1.0, -phi), (-1.0, -phi)):
        verts += [(0.0, a, b), (a, b, 0.0), (b, 0.0, a)]
    v = torch.tensor(verts, dtype=_DTYPE)
    v = v / v.norm(dim=1, keepdim=True) * r_min  # centre→vertex = r_min
    return torch.cat([torch.zeros(1, 3, dtype=_DTYPE), v], dim=0)  # (13, 3)


def _maxwell_velocities(n: int, kbt: float, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    sigma_v = (kbt / _MASS) ** 0.5
    vel = sigma_v * torch.randn(n, 3, generator=g, dtype=_DTYPE)
    vel = vel - vel.mean(0, keepdim=True)  # remove COM momentum
    return vel


def _total_energy(pe: torch.Tensor, vel: torch.Tensor) -> float:
    return float(pe + 0.5 * _MASS * (vel * vel).sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ps", type=float, default=5000.0, help="trajectory length in ps (5 ns = 5000)"
    )
    ap.add_argument("--t0", type=float, default=20.0, help="initial temperature (K)")
    ap.add_argument("--sample-every", type=int, default=2000, help="steps between energy samples")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()

    n_steps = int(args.ps * 1000.0 / _DT)
    kbt = KB_AMU_A_FS * args.t0
    pos = _icosahedron13(2.0 ** (1.0 / 6.0) * _SIGMA)
    vel = _maxwell_velocities(pos.shape[0], kbt, args.seed)

    ff = LennardJonesForceField(epsilon=_EPS, sigma=_SIGMA).to(_DTYPE)
    ig = LangevinVerletIntegrator(ff, dt=_DT, gamma=0.0, kbt=0.0, mass=_MASS, seed=args.seed)
    step = ig.step if args.no_compile else torch.compile(ig.step, fullgraph=True)
    noise = torch.zeros_like(vel)  # NVE: O step is the identity (γ=0)

    state = ig.initial(pos, vel)
    e0 = _total_energy(state.energy, state.vel)
    ts: list[float] = []
    es: list[float] = []
    t_wall = time.perf_counter()
    for i in range(n_steps):
        state = step(state, noise)
        if i % args.sample_every == 0:
            ts.append(i * _DT / 1000.0)  # ps
            es.append(_total_energy(state.energy, state.vel))
    wall = time.perf_counter() - t_wall

    t = torch.tensor(ts, dtype=_DTYPE)
    e = torch.tensor(es, dtype=_DTYPE)
    # Linear fit E(t) = a + b·t  → slope b
    tc = t - t.mean()
    slope = (tc * (e - e.mean())).sum() / (tc * tc).sum()
    duration = n_steps * _DT / 1000.0  # ps
    rel_drift = float(abs(slope * duration) / abs(e0))
    rms_rel = float((e - e.mean()).pow(2).mean().sqrt() / abs(e0))
    rate = n_steps / wall if wall > 0 else float("nan")

    ns = duration / 1000
    print(f"LJ13 NVE: N={pos.shape[0]} dt={_DT} fs  steps={n_steps}  duration={ns:.3f} ns")
    print(f"  T0={args.t0} K  E_tot(0)={e0:.6e}  compiled={not args.no_compile}")
    print(f"  rel energy drift (|slope·dur|/|E0|) = {rel_drift:.3e}  (bound 1e-3)")
    print(f"  rel RMS energy fluctuation          = {rms_rel:.3e}")
    print(f"  temperature finite: {torch.isfinite(e).all().item()}   steps/s = {rate:.0f}")
    ok = rel_drift < 1e-3 and bool(torch.isfinite(e).all())
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
