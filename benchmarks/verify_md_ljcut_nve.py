"""Bulk lj/cut NVE on GPU at the LAMMPS-melt state point.

Drives an FCC argon lattice at the classic ``melt`` state point (ρ* = 0.8442,
T0* = 1.44, r_c = 2.5σ) under NVE with a molpy ``ForceField`` (``lj/cut``)
over a rebuilding :class:`molix.md.NeighborList`, while the *list* decides when
to rebuild, under its own Verlet skin and LAMMPS ``every`` / ``delay`` /
``check`` gate (``--skin`` / ``--every`` / ``--delay`` / ``--no-check``).
``Integrator.eval_force`` asks it once per force evaluation, at the positions
being evaluated; the fixed-capacity buffers keep every tensor shape static
across rebuilds.

Two arms ride on :class:`molix.md.MD`'s own seams. ``--compile`` /
``--no-compile`` (compiled by default) installs a ``torch.compile``d build of
the force field's classical energy/force core with
:meth:`molix.md.MD.set_compiler`, and ``--cuda-graphs`` asks for the
fixed-shape CUDA-graph preset instead (``Compiler(cuda_graphs=True)``, which
the list's fixed capacity is what makes admissible). ``--potential-precision``
is the split-precision arm: the force field's parameter tables and the
geometry handed to it go to that dtype (``MD(..., potential_dtype=…)``) while
the trajectory, the integrator and the neighbour list stay at ``--precision``
— the fp64-integrator over fp32-potential operating point of the
quantization-as-thermal-noise study (``molix.quant``). Both were dead between
d9043cee and the seam that restored them (finding **F28** in
``.claude/notes/composable-potentials.md``).

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

    PYTHONPATH=src:. python benchmarks/verify_md_ljcut_nve.py            # 100 ps default
    PYTHONPATH=src:. python benchmarks/verify_md_ljcut_nve.py --ps 5000  # 5 ns production
    PYTHONPATH=src:. python benchmarks/verify_md_ljcut_nve.py --skin 0   # no-skin arm

The energy-conservation criterion is **not** this script's: it belongs to
:class:`molix.md.EnergyDriftHook`, which accumulates ``(t, E_tot)`` on every
observation and fits the secular slope. This script mounts that hook and,
beside it, the small ``_ThermoSeries`` observer below holding the potential /
kinetic / temperature series the figure wants, and reads the gate back off
:attr:`~molix.md.EnergyDriftHook.result` as an :class:`molix.md.EnergyDrift`
record.

Pass when ``rel_drift = |slope · duration_fs| / |e0| < 1e-3``, every sampled
total energy is finite, and — at ``skin > 0``, where the counter carries
information — no rebuild came late (``ndanger == 0``). ``e0`` is the first
sampled total energy and ``duration_fs`` the sampled span ``t_last − t_first``:
both observers drain at the timed run's ``on_run_start``, so the warm-up
segment is gone and both refer to the timed trajectory only. The bounded RMS fluctuation
``rms_rel`` is printed beside the drift as the scale the slope has to be read
against — it is reported, not asserted, because its size is set by ``dt`` and
is not an error.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
from tensordict import TensorDict

from molix.md import (
    MD,
    EnergyDrift,
    EnergyDriftHook,
    MaxwellBoltzmann,
    MDHook,
    NeighborList,
    VelocityInitHook,
    VelocityVerletIntegrator,
)
from molix.units import units

_KB_EV = (1.0 * units.k_B).to("eV / kelvin").magnitude

# Persistent artifact dir (versioned with the repo for the paper).
_OUT_DEFAULT = Path(__file__).resolve().parent / "results" / "md_ljcut_nve"

# Argon in (amu, Å, fs): ε = 0.0103 eV, σ = 3.4 Å, m = 39.95 amu.
_EPS_EV = 0.0103  # eV; the force field is declared in metal units and rescales
_SIGMA = 3.4
_MASS = 39.95
_RHO_STAR = 0.8442  # LAMMPS melt reduced density
_T0_STAR = 1.44  # LAMMPS melt initial reduced temperature
_CUTOFF = 2.5 * _SIGMA

_DTYPE_MAP = {"fp64": torch.float64, "fp32": torch.float32}


def _fcc(n_cells: int, a: float, *, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """FCC lattice: ``4 n³`` atoms in a cubic box of side ``n·a``."""
    basis = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]], dtype=dtype
    )
    grid = torch.arange(n_cells, dtype=dtype)
    offsets = torch.stack(torch.meshgrid(grid, grid, grid, indexing="ij"), dim=-1).reshape(-1, 3)
    pos = (offsets.unsqueeze(1) + basis.unsqueeze(0)).reshape(-1, 3) * a
    cell = torch.eye(3, dtype=dtype) * (n_cells * a)
    return pos, cell


class _ThermoSeries(MDHook):
    """The PE / KE / T series this script plots — mounted beside the drift hook.

    ``EnergyDriftHook`` owns ``(t, E_tot)`` and the gate; these three series
    exist only for the figure and the ``<T>`` line, which is why they are a
    second, smaller observer rather than a subclass of it. Both drain at
    ``on_run_start``, so the warm-up run's samples never reach the timed run's
    figure and the two series stay index-aligned.
    """

    def __init__(self) -> None:
        self.pe: list[float] = []
        self.ke: list[float] = []
        self.temp: list[float] = []

    def on_run_start(self, runner) -> None:
        for series in (self.pe, self.ke, self.temp):
            series.clear()

    def on_step_end(self, runner, step, obs) -> None:
        self.pe.append(float(obs.potential))
        self.ke.append(float(obs.kinetic))
        self.temp.append(float(obs.temperature))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ps", type=float, default=100.0, help="trajectory length in ps")
    ap.add_argument("--n", type=int, default=5, help="FCC cells per side (N = 4n^3 atoms)")
    ap.add_argument("--dt", type=float, default=4.0, help="timestep (fs)")
    ap.add_argument(
        "--t0",
        type=float,
        default=_T0_STAR * _EPS_EV / _KB_EV,
        help="initial temperature (K); default T0* = 1.44",
    )
    ap.add_argument(
        "--precision",
        choices=("fp64", "fp32"),
        default="fp64",
        help="MD-side precision (trajectory state, integrator, mass)",
    )
    ap.add_argument(
        "--potential-precision",
        choices=("fp64", "fp32"),
        default=None,
        help="force-field / inference precision; defaults to --precision. "
        "Setting it below --precision is the split-precision arm "
        "(e.g. MD fp64 over pot fp32)",
    )
    ap.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="compile the force field's classical energy/force core "
        "(MD.set_compiler(Compiler())); --no-compile runs it eager",
    )
    ap.add_argument(
        "--cuda-graphs",
        action="store_true",
        help="compile with the fixed-shape CUDA-graph preset "
        "(Compiler(cuda_graphs=True)) instead of the default inductor config; "
        "implies --compile",
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
    ap.add_argument("--out", type=Path, default=_OUT_DEFAULT, help="artifact dir (npz + png)")
    ap.add_argument("--no-save", action="store_true", help="skip writing artifacts")
    args = ap.parse_args()

    md_dtype = _DTYPE_MAP[args.precision]
    pot_name = args.potential_precision or args.precision
    # None when the two agree: MD then adds no cast at all on the force path,
    # so the single-precision arm is byte-identical to having no seam.
    pot_dtype = _DTYPE_MAP[pot_name] if pot_name != args.precision else None
    compiled = args.compile or args.cuda_graphs
    device = torch.device(args.device)
    n_steps = int(args.ps * 1000.0 / args.dt)
    a = _SIGMA * (4.0 / _RHO_STAR) ** (1.0 / 3.0)
    pos, cell = _fcc(args.n, a, dtype=md_dtype)
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

    # Everything this script builds — geometry, the nblist, the LJ tables — is
    # declared at the MD-state dtype, and the split-precision arm is left
    # entirely to ``MD(potential_dtype=…)``. That seam already restates the
    # container's parameter tables through ``ForceField.to_dtype`` (ε / σ are
    # read live off the container, so no module cast can reach them), moves the
    # assembled module's own buffers, and casts the per-step geometry. Declaring
    # the tables at the potential dtype here as well would quantize them twice —
    # harmless while the second cast is a widening no-op, and a silent second
    # rounding the moment the two dtypes are read in the other order.
    from molpot.forcefield import ForceField

    n_atoms = int(pos.shape[0])
    ff = (
        ForceField("ljcut", units="metal")
        .def_pairstyle("lj/cut", cutoff=2.5 * _SIGMA)
        .with_params(
            atoms={
                # eV: ForceField(units="metal") rescales to MD units itself.
                # Passing the pre-converted _EPS here (as d9043cee did when it
                # replaced the MD-unit LennardJonesCutForceField module) divided
                # epsilon by 103.6 a second time and the crystal free-streamed
                # into the r^-12 core.
                "epsilon": torch.full((n_atoms,), _EPS_EV, dtype=md_dtype, device=device),
                "sigma": torch.full((n_atoms,), _SIGMA, dtype=md_dtype, device=device),
            }
        )
    )
    drift_hook = EnergyDriftHook()
    thermo = _ThermoSeries()
    md = (
        MD(md_dtype, potential_dtype=pot_dtype)
        .set_forcefield(ff)
        .set_neighbors(neighbors)
        .set_integrator(VelocityVerletIntegrator(args.dt))
        .set_hooks(
            [VelocityInitHook(MaxwellBoltzmann(args.t0, seed=args.seed)), drift_hook, thermo]
        )
        .to(device)
    )
    if compiled:
        # set_compiler is a setter like the others: it rebuilds the potential
        # and re-installs the compiled core, so it is safe here at the end of
        # the chain and equally safe in the middle of it.
        from molix.compile import Compiler

        md.set_compiler(Compiler(cuda_graphs=True) if args.cuda_graphs else Compiler())
    atoms = TensorDict(
        {
            "atoms": TensorDict(
                {
                    "pos": pos,
                    "vel": torch.zeros_like(pos),
                    "mass": torch.full((n_atoms,), _MASS, device=device, dtype=md_dtype),
                },
                batch_size=[n_atoms],
            )
        },
        batch_size=[],
    )
    md.run(atoms.clone(), min(3 * args.sample_every, n_steps), chunk=args.sample_every)
    # Both observers drain at the timed run's on_run_start, so the warmup's
    # samples are gone without a reset call here.
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
    md.run(atoms, n_steps, chunk=args.sample_every)
    if device.type == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t_wall
    rebuilds = neighbors.rebuild_count - rebuilds_before
    ndanger = neighbors.ndanger - ndanger_before

    # A run too short to fit (one observation at a coarse --sample-every)
    # publishes result=None rather than raising. The timed run is the one that
    # must produce a gate, so read its record and fail closed if there is none.
    drift = drift_hook.result
    if drift is None:
        raise SystemExit(
            f"the timed run recorded {len(drift_hook.totals)} energy sample(s): "
            f"{n_steps} steps at --sample-every {args.sample_every} give fewer than the "
            "two the drift fit needs. Lower --sample-every or raise --ps."
        )
    duration = drift.duration_fs / 1000.0  # ps, the sampled window
    rate = n_steps / wall if wall > 0 else float("nan")
    t_mean = sum(thermo.temp[len(thermo.temp) // 2 :]) / max(1, len(thermo.temp) // 2)

    print(
        f"lj/cut NVE melt: N={n_atoms} (fcc {args.n}^3)  rho*={_RHO_STAR}  rc={_CUTOFF:.2f} A  "
        f"dt={args.dt} fs  steps={n_steps}  duration={duration / 1000:.3f} ns"
    )
    print(
        f"  device={device}  precision: MD {args.precision}, potential {pot_name}  "
        f"compiled={compiled}{' (cuda graphs)' if compiled and args.cuda_graphs else ''}"
    )
    print(
        f"  policy: skin={neighbors.skin:g} A  every={neighbors.every}  delay={neighbors.delay}  "
        f"check={neighbors.check}  r_build={neighbors.r_build:.2f} A"
    )
    print(
        f"  rebuilds={rebuilds} ({rebuilds / max(1, n_steps):.3f} of {n_steps} steps)  "
        f"ndanger={ndanger}"
    )
    print(f"  T0={args.t0:.1f} K  <T>(2nd half)={t_mean:.1f} K  E_tot(0)={drift.e0:.6e}")
    print(f"  rel energy drift (|slope*dur|/|E0|) = {drift.rel_drift:.3e}  (bound 1e-3)")
    print(f"  rel RMS energy fluctuation          = {drift.rms_rel:.3e}")
    print(f"  steps/s = {rate:.0f}   ({rate * args.dt / 1e6 * 86400:.1f} ns/day)")
    # At skin=0 every rebuild lands on the first permitted opportunity by
    # construction, so ndanger just counts rebuilds and carries no information;
    # at skin>0 a nonzero count means a rebuild came too late and pairs were
    # missed, which is a wrong PES however small the drift happens to look.
    ok = drift.rel_drift < 1e-3 and all(math.isfinite(v) for v in drift_hook.totals)
    if neighbors.skin > 0.0:
        ok = ok and ndanger == 0
    print("RESULT:", "PASS" if ok else "FAIL")

    if not args.no_save:
        _save_artifacts(args.out, drift_hook, thermo, drift, args, n_atoms, pot_name)

    return 0 if ok else 1


def _save_artifacts(
    out: Path,
    drift_hook: EnergyDriftHook,
    thermo: _ThermoSeries,
    drift: EnergyDrift,
    args: argparse.Namespace,
    n_atoms: int,
    pot_name: str,
) -> None:
    """Write the energy/temperature series (.npz) and a conservation figure (.png)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out.mkdir(parents=True, exist_ok=True)
    e0 = drift.e0
    duration = drift.duration_fs / 1000.0  # ps
    t = np.asarray(drift_hook.times_fs) / 1000.0  # ps
    e = np.asarray(drift_hook.totals)
    pe = np.asarray(thermo.pe)
    ke = np.asarray(thermo.ke)
    temp = np.asarray(thermo.temp)

    tag = f"md{args.precision[-2:]}_pot{pot_name[-2:]}"
    npz = out / f"ljcut_nve_{tag}.npz"
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
            f"check={not args.no_check}; MD={args.precision}; pot={pot_name}; "
            f"rel_drift={drift.rel_drift:.3e}; rel_rms={drift.rms_rel:.3e}; units=(amu,A,fs)"
        ),
    )

    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(6.0, 7.0), sharex=True)
    ax0.plot(t, (e - e0) / abs(e0) * 1e6, lw=0.8, color="C3")
    ax0.axhline(0.0, color="k", lw=0.5, ls=":")
    ax0.set_ylabel(r"$(E_{\rm tot}-E_0)/|E_0|$  [ppm]")
    ax0.set_title(
        f"lj/cut NVE melt, N={n_atoms}, {duration:.0f} ps, MD {args.precision}/pot {pot_name} — "
        f"drift {drift.rel_drift:.1e}, RMS {drift.rms_rel:.1e}"
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
    png = out / f"ljcut_nve_{tag}_energy.png"
    fig.savefig(png, dpi=200)
    plt.close(fig)

    print(f"  saved: {npz}")
    print(f"  saved: {png}")


if __name__ == "__main__":
    raise SystemExit(main())
