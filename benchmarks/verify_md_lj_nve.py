"""LJ-cluster NVE energy-conservation validation (spec md-component-engine ac-008).

Drives a Lennard-Jones cluster (argon parameters, the LJ13 icosahedral minimum)
under NVE with the compiled MD engine for a long trajectory, and checks the
total energy shows no systematic drift — the gold-standard integrator test.

Units: (amu, Å, fs) with energy in amu·Å²/fs² (see molix.md.integrators). Every
pair interacts — the cutoff is far larger than the cluster — so the PES is
exact for any displacement the trajectory reaches. The hot path is the force
field's analytic energy/force core, ``torch.compile``d through
:meth:`molix.md.MD.set_compiler` (LJ is analytic — no scatter, so the Inductor
CPU backend works); 5 ns is only feasible because that core is compiled.

Run::

    PYTHONPATH=src:. python benchmarks/verify_md_lj_nve.py            # full 5 ns
    PYTHONPATH=src:. python benchmarks/verify_md_lj_nve.py --ps 100   # short smoke

The conservation criterion itself is :class:`molix.md.EnergyDriftHook`'s, not
this script's: the hook is mounted on the run and fed ``(t, E_tot)`` by the
runner once per ``--sample-every`` steps, and publishes the fit as
:attr:`~molix.md.EnergyDriftHook.result` at run end. ``_ThermoSeries`` beside it
holds the PE / KE / frames the figure wants. Both are ordinary
:class:`molix.md.MDHook` s: this script drives ``MD`` through its public
setters only.

Pass when ``rel_drift = |slope · duration_fs| / |e0| < 1e-3`` with bounded RMS
fluctuation, where ``e0`` is the first *sampled* total energy and
``duration_fs`` the sampled span.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from molpy import Box, Frame
from tensordict import TensorDict

from molix.compile import Compiler
from molix.data import TotalCharge
from molix.data.collate import collate_molecules
from molix.md import MD, EnergyDriftHook, MDHook, NeighborList, VelocityVerletIntegrator
from molix.units import units
from molpot.forcefield import ForceField

_KB_MD = (1.0 * units.k_B).to("amu * angstrom ** 2 / femtosecond ** 2 / kelvin").magnitude

# Persistent artifact dir (versioned with the repo for the paper).
_OUT_DEFAULT = Path(__file__).resolve().parent / "results" / "md_lj_nve"

# Argon in (amu, Å, fs): ε = 0.0103 eV, σ = 3.4 Å, m = 39.95 amu.
_EPS = 0.0103 / 103.6426965638  # eV -> amu·Å²/fs²
_SIGMA = 3.4
_MASS = 39.95
_DT = 5.0  # fs — ~1/350 of the LJ vibrational period; comfortably stable
_DTYPE = torch.float64
# The cluster is ~7.6 Å across, so a 30 Å cutoff keeps every pair in range and
# admits no periodic image inside the 80 Å box. It is also the largest radius
# the list will build at here: ``NeighborList`` refuses cutoff + skin beyond
# half the minimum perpendicular width, because the kernel's sequential
# minimum-image reduction would silently drop pairs — the 50 Å this script used
# to ask for was such a refusal, and the run died at construction.
_CUTOFF = 30.0
_BOX = 80.0


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


class _ThermoSeries(MDHook):
    """The PE / KE / frame series the figure wants, beside the drift gate.

    ``EnergyDriftHook`` owns ``(t, E_tot)`` and the pass/fail criterion; this
    second, smaller observer holds what only the artifacts need. It drains at
    ``on_run_start`` so a second run on the same object never fits one line
    across two trajectories.
    """

    def __init__(self) -> None:
        self.pe: list[float] = []
        self.ke: list[float] = []
        self.frames: list[torch.Tensor] = []

    def on_run_start(self, runner) -> None:
        self.pe.clear()
        self.ke.clear()
        self.frames.clear()

    def on_step_end(self, runner, step, obs) -> None:
        self.pe.append(float(obs.potential))
        self.ke.append(float(obs.kinetic))
        self.frames.append(obs.pos.detach().cpu().clone())


def _cluster_frame(pos: torch.Tensor, vel: torch.Tensor) -> Frame:
    """The argon cluster as one molpy ``Frame`` — this run's data at rest.

    A ``Frame`` is the unit of data at rest in molnex and
    :func:`~molix.data.collate.collate_molecules` is the only conversion to the
    batch a model reads, so a benchmark that builds a system from scratch builds
    the same object a dataset would rather than assembling batch tensors by
    hand. One frame: the cluster is one periodic cell (**F20**), and that cell
    is the frame's box.

    ``mass`` rides as a per-atom column, the velocities as ``vx``/``vy``/``vz``
    and the (zero) net charge as a ``meta`` scalar, so all of it is the frame's
    own data.

    Args:
        pos: Positions ``(N, 3)`` in Å, already inside the box.
        vel: Velocities ``(N, 3)`` in Å/fs.

    Returns:
        The frame.
    """
    xyz = pos.detach().cpu().double().numpy()
    vxyz = vel.detach().cpu().double().numpy()
    n = int(xyz.shape[0])
    frame = Frame(meta={"total_charge": 0.0})
    frame.box = Box(np.eye(3) * _BOX)
    frame["atoms"] = {
        # Argon: the species this benchmark's epsilon / sigma / mass are.
        "number": np.full(n, 18, dtype=np.int64),
        "x": np.ascontiguousarray(xyz[:, 0]),
        "y": np.ascontiguousarray(xyz[:, 1]),
        "z": np.ascontiguousarray(xyz[:, 2]),
        "vx": np.ascontiguousarray(vxyz[:, 0]),
        "vy": np.ascontiguousarray(vxyz[:, 1]),
        "vz": np.ascontiguousarray(vxyz[:, 2]),
    }
    frame["atoms"]["mass"] = np.full(n, _MASS, dtype=np.float64)
    return frame


def _cluster_batch(pos: torch.Tensor, vel: torch.Tensor) -> TensorDict:
    """The rollout's batch: :func:`_cluster_frame` through the real collate, at this run's dtype.

    The lattice, the masses and the velocities are the frame's, and the
    collate builds the batch at ``_DTYPE`` directly — a 5 ns energy-drift
    gate is never given single precision's rounding of its own initial
    geometry. ``graphs.cell`` comes out ``(B, 3, 3)`` under a sized
    ``graphs``, the layout MD writes a cast / barostat-scaled lattice back
    into.

    Args:
        pos: Positions ``(N, 3)`` in Å.
        vel: Velocities ``(N, 3)`` in Å/fs.

    Returns:
        The batch ``MD.run`` integrates.
    """
    batch = collate_molecules([_cluster_frame(pos, vel)], dtype=_DTYPE)
    # The frame declared its net charge; this only normalises the dtype.
    TotalCharge().execute(batch)
    return batch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ps", type=float, default=5000.0, help="trajectory length in ps (5 ns = 5000)"
    )
    ap.add_argument("--t0", type=float, default=20.0, help="initial temperature (K)")
    ap.add_argument("--sample-every", type=int, default=2000, help="steps between energy samples")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--out", type=Path, default=_OUT_DEFAULT, help="artifact dir (npz + png)")
    ap.add_argument("--no-save", action="store_true", help="skip writing artifacts")
    args = ap.parse_args()

    n_steps = int(args.ps * 1000.0 / _DT)
    kbt = _KB_MD * args.t0
    pos = _icosahedron13(2.0 ** (1.0 / 6.0) * _SIGMA)
    vel = _maxwell_velocities(pos.shape[0], kbt, args.seed)

    n = int(pos.shape[0])
    ff = (
        ForceField("lj13", units="metal")
        .def_pairstyle("lj/cut", cutoff=_CUTOFF)
        .with_params(
            atoms={
                "epsilon": torch.full((n,), 0.0103, dtype=_DTYPE),
                "sigma": torch.full((n,), _SIGMA, dtype=_DTYPE),
            }
        )
    )
    cell = torch.eye(3, dtype=_DTYPE) * _BOX
    pos = pos + 0.5 * _BOX
    nl = NeighborList(cell=cell, cutoff=_CUTOFF, positions=pos, skin=1.0)
    system = _cluster_batch(pos, vel)
    hook = EnergyDriftHook()
    thermo = _ThermoSeries()
    md = (
        MD(_DTYPE)
        .set_forcefield(ff)
        .set_neighbors(nl)
        .set_integrator(VelocityVerletIntegrator(_DT))
        .set_hooks([hook, thermo])
    )
    if not args.no_compile:
        # The one compile seam of a run: the force field's classical
        # energy/force core. Applied through the setter so it survives the
        # potential rebuild every other setter performs.
        md.set_compiler(Compiler())
    t_wall = time.perf_counter()
    md.run(system, n_steps, chunk=args.sample_every)
    wall = time.perf_counter() - t_wall
    pes, kes, traj = thermo.pe, thermo.ke, thermo.frames

    drift = hook.result
    if drift is None:
        raise SystemExit(
            f"the run recorded {len(hook.totals)} energy sample(s): {n_steps} steps at "
            f"--sample-every {args.sample_every} give fewer than the two the drift fit "
            "needs. Lower --sample-every or raise --ps."
        )
    t = torch.tensor(hook.times_fs, dtype=_DTYPE) / 1000.0  # ps
    e = torch.tensor(hook.totals, dtype=_DTYPE)
    e0, rel_drift, rms_rel = drift.e0, drift.rel_drift, drift.rms_rel
    rate = n_steps / wall if wall > 0 else float("nan")

    ns = drift.duration_fs / 1e6  # sampled window, ns
    print(f"LJ13 NVE: N={pos.shape[0]} dt={_DT} fs  steps={n_steps}  duration={ns:.3f} ns")
    print(f"  T0={args.t0} K  E_tot(0)={e0:.6e}  compiled={not args.no_compile}")
    print(f"  rel energy drift (|slope·dur|/|E0|) = {rel_drift:.3e}  (bound 1e-3)")
    print(f"  rel RMS energy fluctuation          = {rms_rel:.3e}")
    print(f"  temperature finite: {torch.isfinite(e).all().item()}   steps/s = {rate:.0f}")
    ok = rel_drift < 1e-3 and bool(torch.isfinite(e).all())
    print("RESULT:", "PASS" if ok else "FAIL")

    if not args.no_save:
        _save_artifacts(args.out, t, e, pes, kes, traj, e0, rel_drift, rms_rel, ns, args)

    return 0 if ok else 1


def _save_artifacts(out, t, e, pes, kes, traj, e0, rel_drift, rms_rel, ns, args):
    """Write trajectory + energy series (.npz) and an energy-conservation figure (.png)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    pe = torch.tensor(pes, dtype=_DTYPE)
    ke = torch.tensor(kes, dtype=_DTYPE)
    positions = torch.stack(traj).numpy()  # (frames, N, 3)

    npz = out / "lj13_nve.npz"
    import numpy as np

    np.savez_compressed(
        npz,
        time_ps=t.numpy(),
        e_total=e.numpy(),
        e_pot=pe.numpy(),
        e_kin=ke.numpy(),
        positions=positions,
        meta=np.array(
            f"LJ13 argon NVE; dt={_DT}fs; T0={args.t0}K; duration={ns:.3f}ns; "
            f"eps={_EPS:.6e} amuA2/fs2; sigma={_SIGMA}A; mass={_MASS}amu; seed={args.seed}; "
            f"rel_drift={rel_drift:.3e}; rel_rms={rms_rel:.3e}; units=(amu,A,fs)"
        ),
    )

    # Energy-conservation figure: relative total-energy drift (ppm) + PE/KE exchange.
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(6.0, 5.0), sharex=True)
    ax0.plot(t.numpy(), ((e - e0) / abs(e0) * 1e6).numpy(), lw=0.8, color="C3")
    ax0.axhline(0.0, color="k", lw=0.5, ls=":")
    ax0.set_ylabel(r"$(E_{\rm tot}-E_0)/|E_0|$  [ppm]")
    ax0.set_title(
        f"LJ$_{{13}}$ NVE, {ns:.1f} ns, dt={_DT:g} fs — drift {rel_drift:.1e}, RMS {rms_rel:.1e}"
    )
    ax1.plot(t.numpy(), pe.numpy(), lw=0.7, color="C0", label="potential")
    ax1.plot(t.numpy(), ke.numpy(), lw=0.7, color="C1", label="kinetic")
    ax1.plot(t.numpy(), e.numpy(), lw=0.9, color="k", label="total")
    ax1.set_xlabel("time [ps]")
    ax1.set_ylabel(r"energy [amu$\cdot$Å$^2$/fs$^2$]")
    ax1.legend(loc="best", fontsize=8, ncol=3)
    fig.tight_layout()
    png = out / "lj13_nve_energy.png"
    fig.savefig(png, dpi=200)
    plt.close(fig)

    print(f"  saved: {npz}")
    print(f"  saved: {png}")


if __name__ == "__main__":
    raise SystemExit(main())
