"""Bulk lj/cut NVE on GPU with a compiled force field (LAMMPS-melt state point).

Drives an FCC argon lattice at the classic ``melt`` state point (ρ* = 0.8442,
T0* = 1.44, r_c = 2.5σ) under NVE with :class:`molix.md.LennardJonesCutForceField`
over a rebuilding :class:`molix.md.NeighborList`. The force evaluation
is ``torch.compile``d — fullgraph inductor by default, ``--cuda-graphs`` for the
``reduce-overhead`` preset — while the *list* decides when to rebuild, under its
own Verlet skin and LAMMPS ``every`` / ``delay`` / ``check`` gate
(``--skin`` / ``--every`` / ``--delay`` / ``--no-check``). ``Integrator.eval_force``
asks it once per force evaluation, at the positions being evaluated, and runs the
answer eagerly *between* compiled force calls; the fixed-capacity buffers keep
every tensor shape static across rebuilds, which is what lets the compiled force
path survive the whole run.

The default ``--skin 1.02`` Å is 0.3 σ — the ``neighbor 0.3 bin`` setting shipped
with the LAMMPS ``melt`` example — giving a half-skin of 0.51 Å against a
ballistic per-step displacement of order 0.013 Å at T0, i.e. a rebuild every few
tens of steps rather than every one. ``--skin 0`` is the no-skin limit (rebuild
whenever anything moved at all), useful as the reference arm.

Pure-GPU requires the molix op built with ``MOLNEX_OP_ENABLE_CUDA=ON`` so the
neighbour rebuild runs on-device (a CPU-only op build fails at list
construction with a dispatch error).

Units: (amu, Å, fs) with energy in amu·Å²/fs² (see molix.md.integrators);
``skin`` / ``r_build`` in Å, ``every`` / ``delay`` in MD steps.

Run::

    PYTHONPATH=src:. python benchmarks/verify_md_ljcut_nve.py            # 100 ps
    PYTHONPATH=src:. python benchmarks/verify_md_ljcut_nve.py --ps 5     # smoke
    PYTHONPATH=src:. python benchmarks/verify_md_ljcut_nve.py --cuda-graphs
    PYTHONPATH=src:. python benchmarks/verify_md_ljcut_nve.py --skin 0   # no-skin arm

Pass when ``|slope·duration| / |E_tot(0)| < 1e-3`` with bounded RMS fluctuation
and — at ``skin > 0``, where the counter carries information — no dangerous
builds (``ndanger == 0``).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from molix.compile import Compiler
from molix.md import (
    KB_EV_PER_K,
    MD,
    LennardJonesCutForceField,
    MaxwellBoltzmann,
    MDHook,
    NeighborList,
)

# Persistent artifact dir (versioned with the repo for the paper).
_OUT_DEFAULT = Path(__file__).resolve().parent / "results" / "md_ljcut_nve"

# Argon in (amu, Å, fs): ε = 0.0103 eV, σ = 3.4 Å, m = 39.95 amu.
_EPS_EV = 0.0103
_EPS = _EPS_EV / 103.6426965638  # eV -> amu·Å²/fs²
_SIGMA = 3.4
_MASS = 39.95
_RHO_STAR = 0.8442  # LAMMPS melt reduced density
_T0_STAR = 1.44  # LAMMPS melt initial reduced temperature
_CUTOFF = 2.5 * _SIGMA
_DTYPE = torch.float64


def _fcc(n_cells: int, a: float) -> tuple[torch.Tensor, torch.Tensor]:
    """FCC lattice: ``4 n³`` atoms in a cubic box of side ``n·a``."""
    basis = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]], dtype=_DTYPE
    )
    grid = torch.arange(n_cells, dtype=_DTYPE)
    offsets = torch.stack(torch.meshgrid(grid, grid, grid, indexing="ij"), dim=-1).reshape(-1, 3)
    pos = (offsets.unsqueeze(1) + basis.unsqueeze(0)).reshape(-1, 3) * a
    cell = torch.eye(3, dtype=_DTYPE) * (n_cells * a)
    return pos, cell


class _EnergySampler(MDHook):
    """Record (t, PE, KE, E_tot, T) at every hook-visible chunk boundary."""

    def __init__(self, dt_fs: float) -> None:
        self._dt = float(dt_fs)
        self.t_ps: list[float] = []
        self.pe: list[float] = []
        self.ke: list[float] = []
        self.etot: list[float] = []
        self.temp: list[float] = []

    def clear(self) -> None:
        for series in (self.t_ps, self.pe, self.ke, self.etot, self.temp):
            series.clear()

    def on_step_end(self, runner, step, obs) -> None:
        self.t_ps.append(step * self._dt / 1000.0)
        self.pe.append(float(obs.potential))
        self.ke.append(float(obs.kinetic))
        self.etot.append(float(obs.total))
        self.temp.append(float(obs.temperature))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ps", type=float, default=100.0, help="trajectory length in ps")
    ap.add_argument("--n", type=int, default=5, help="FCC cells per side (N = 4n^3 atoms)")
    ap.add_argument("--dt", type=float, default=4.0, help="timestep (fs)")
    ap.add_argument(
        "--t0",
        type=float,
        default=_T0_STAR * _EPS_EV / KB_EV_PER_K,
        help="initial temperature (K); default T0* = 1.44",
    )
    ap.add_argument(
        "--skin",
        type=float,
        default=1.02,
        help="Verlet skin in A (default 1.02 = 0.3 sigma, the LAMMPS melt setting); "
        "the list is built at cutoff + skin and stays complete to cutoff while no "
        "atom has moved more than skin/2. 0 = the no-skin limit",
    )
    ap.add_argument(
        "--every", type=int, default=1, help="attempt a rebuild only every N steps (LAMMPS every)"
    )
    ap.add_argument(
        "--delay",
        type=int,
        default=0,
        help="attempt no rebuild until N steps after the last one (LAMMPS delay; "
        "must be a multiple of --every)",
    )
    ap.add_argument(
        "--no-check",
        action="store_true",
        help="rebuild on cadence alone, without the half-skin displacement test "
        "(cheaper, and never a free optimisation: it accepts missed pairs)",
    )
    ap.add_argument("--capacity-factor", type=float, default=1.5)
    ap.add_argument("--sample-every", type=int, default=100, help="steps between energy samples")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument(
        "--cuda-graphs",
        action="store_true",
        help="compile with the reduce-overhead CUDA-graph preset instead of default inductor",
    )
    ap.add_argument("--out", type=Path, default=_OUT_DEFAULT, help="artifact dir (npz + png)")
    ap.add_argument("--no-save", action="store_true", help="skip writing artifacts")
    args = ap.parse_args()

    device = torch.device(args.device)
    n_steps = int(args.ps * 1000.0 / args.dt)
    a = _SIGMA * (4.0 / _RHO_STAR) ** (1.0 / 3.0)
    pos, cell = _fcc(args.n, a)
    n_atoms = pos.shape[0]
    pos, cell = pos.to(device), cell.to(device)

    try:
        neighbors = NeighborList(
            cell=cell,
            cutoff=_CUTOFF,
            positions=pos,
            skin=args.skin,
            every=args.every,
            delay=args.delay,
            check=not args.no_check,
            capacity_factor=args.capacity_factor,
        )
    except (RuntimeError, NotImplementedError) as err:
        if device.type == "cuda":
            raise SystemExit(
                f"on-device neighbour rebuild failed ({err}).\n"
                "Build the molix op with CUDA kernels: "
                "cmake -S src/molix/op -B src/molix/op/build -DMOLNEX_OP_ENABLE_CUDA=ON"
            ) from err
        raise

    ff = LennardJonesCutForceField(epsilon=_EPS, sigma=_SIGMA, neighbors=neighbors)
    ff = ff.to(device=device, dtype=_DTYPE)
    compiled = not args.no_compile
    if compiled:
        ff = Compiler(cuda_graphs=args.cuda_graphs, fullgraph=True)(ff)

    sampler = _EnergySampler(args.dt)
    # No cadence kwarg: the list owns the policy and the integrator derives its
    # switch from LennardJonesCutForceField.rebuilds_neighbors.
    md = MD(
        ff,
        mass=_MASS,
        dt=args.dt,
        gamma=0.0,  # NVE
        dtype=_DTYPE,
        device=device,
        hooks=[sampler],
    )
    vel = MaxwellBoltzmann(_MASS, n_atoms=n_atoms).sample(args.t0, seed=args.seed)

    md.run(pos, vel, min(3 * args.sample_every, n_steps), chunk=args.sample_every)  # warmup/compile
    sampler.clear()
    # The warmup left the list built at the *warmup's* final configuration, and
    # its counters carrying the warmup's rebuilds. Re-phase it onto the timed
    # run's initial configuration with the forced-build escape hatch (so the
    # first force evaluation is not served a list held elsewhere under a coarse
    # every/delay gate), then read the counters, and report deltas: what is
    # printed is exactly what the timed trajectory paid.
    neighbors.rebuild(pos)
    rebuilds_before, ndanger_before = neighbors.rebuild_count, neighbors.ndanger
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_wall = time.perf_counter()
    md.run(pos, vel, n_steps, chunk=args.sample_every)
    if device.type == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t_wall
    rebuilds = neighbors.rebuild_count - rebuilds_before
    ndanger = neighbors.ndanger - ndanger_before

    t = torch.tensor(sampler.t_ps, dtype=_DTYPE)
    e = torch.tensor(sampler.etot, dtype=_DTYPE)
    e0 = float(e[0])
    tc = t - t.mean()
    slope = (tc * (e - e.mean())).sum() / (tc * tc).sum()
    duration = n_steps * args.dt / 1000.0  # ps
    rel_drift = float(abs(slope * duration) / abs(e0))
    rms_rel = float((e - e.mean()).pow(2).mean().sqrt() / abs(e0))
    rate = n_steps / wall if wall > 0 else float("nan")
    t_mean = sum(sampler.temp[len(sampler.temp) // 2 :]) / max(1, len(sampler.temp) // 2)

    print(
        f"lj/cut NVE melt: N={n_atoms} (fcc {args.n}^3)  rho*={_RHO_STAR}  rc={_CUTOFF:.2f} A  "
        f"dt={args.dt} fs  steps={n_steps}  duration={duration / 1000:.3f} ns"
    )
    print(f"  device={device}  compiled={compiled}  cuda_graphs={args.cuda_graphs}")
    print(
        f"  policy: skin={neighbors.skin:g} A  every={neighbors.every}  delay={neighbors.delay}  "
        f"check={neighbors.check}  r_build={neighbors.r_build:.2f} A"
    )
    print(
        f"  rebuilds={rebuilds} ({rebuilds / max(1, n_steps):.3f} of {n_steps} steps)  "
        f"ndanger={ndanger}"
    )
    print(f"  T0={args.t0:.1f} K  <T>(2nd half)={t_mean:.1f} K  E_tot(0)={e0:.6e}")
    print(f"  rel energy drift (|slope*dur|/|E0|) = {rel_drift:.3e}  (bound 1e-3)")
    print(f"  rel RMS energy fluctuation          = {rms_rel:.3e}")
    print(f"  steps/s = {rate:.0f}   ({rate * args.dt / 1e6 * 86400:.1f} ns/day)")
    # At skin=0 every rebuild lands on the first permitted opportunity by
    # construction, so ndanger just counts rebuilds and carries no information;
    # at skin>0 a nonzero count means a rebuild came too late and pairs were
    # missed, which is a wrong PES however small the drift happens to look.
    ok = rel_drift < 1e-3 and bool(torch.isfinite(e).all())
    if neighbors.skin > 0.0:
        ok = ok and ndanger == 0
    print("RESULT:", "PASS" if ok else "FAIL")

    if not args.no_save:
        _save_artifacts(args.out, sampler, e0, rel_drift, rms_rel, duration, args, n_atoms)

    return 0 if ok else 1


def _save_artifacts(out, sampler, e0, rel_drift, rms_rel, duration, args, n_atoms):
    """Write the energy/temperature series (.npz) and a conservation figure (.png)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out.mkdir(parents=True, exist_ok=True)
    t = np.asarray(sampler.t_ps)
    e = np.asarray(sampler.etot)
    pe = np.asarray(sampler.pe)
    ke = np.asarray(sampler.ke)
    temp = np.asarray(sampler.temp)

    npz = out / "ljcut_nve.npz"
    np.savez_compressed(
        npz,
        time_ps=t,
        e_total=e,
        e_pot=pe,
        e_kin=ke,
        temperature=temp,
        meta=np.array(
            f"lj/cut argon NVE melt; N={n_atoms}; rho*={_RHO_STAR}; rc={_CUTOFF}A; "
            f"dt={args.dt}fs; T0={args.t0:.1f}K; duration={duration:.1f}ps; seed={args.seed}; "
            f"skin={args.skin}A; every={args.every}; delay={args.delay}; "
            f"check={not args.no_check}; "
            f"rel_drift={rel_drift:.3e}; rel_rms={rms_rel:.3e}; units=(amu,A,fs)"
        ),
    )

    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(6.0, 7.0), sharex=True)
    ax0.plot(t, (e - e0) / abs(e0) * 1e6, lw=0.8, color="C3")
    ax0.axhline(0.0, color="k", lw=0.5, ls=":")
    ax0.set_ylabel(r"$(E_{\rm tot}-E_0)/|E_0|$  [ppm]")
    ax0.set_title(
        f"lj/cut NVE melt, N={n_atoms}, {duration:.0f} ps, dt={args.dt:g} fs — "
        f"drift {rel_drift:.1e}, RMS {rms_rel:.1e}"
    )
    ax1.plot(t, pe, lw=0.7, color="C0", label="potential")
    ax1.plot(t, ke, lw=0.7, color="C1", label="kinetic")
    ax1.plot(t, e, lw=0.9, color="k", label="total")
    ax1.set_ylabel(r"energy [amu$\cdot$Å$^2$/fs$^2$]")
    ax1.legend(loc="best", fontsize=8, ncol=3)
    ax2.plot(t, temp, lw=0.7, color="C2")
    ax2.set_xlabel("time [ps]")
    ax2.set_ylabel("T [K]")
    fig.tight_layout()
    png = out / "ljcut_nve_energy.png"
    fig.savefig(png, dpi=200)
    plt.close(fig)

    print(f"  saved: {npz}")
    print(f"  saved: {png}")


if __name__ == "__main__":
    raise SystemExit(main())
