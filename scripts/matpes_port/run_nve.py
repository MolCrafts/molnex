"""Run an NVE trajectory with MACE-MatPES through molnex's own MD engine.

molnex-only by design (see ``README.md``): no ASE, e3nn or ``mace-torch``
import anywhere in this file. The official-model comparison is a separate,
out-of-tree step that consumes the ``.pt`` this script writes.

Usage::

    PYTHONPATH=src python scripts/matpes_port/run_nve.py \\
        --structure /path/to/wat64_h3o+.vasp \\
        --weights-dir /path/to/mace_models \\
        --out /path/to/nve.pt --steps 200 --dt 0.5 --temperature 300
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch

from molix import config

# Default; overridden by --precision before any model construction.
config.set_precision("fp64")

from molix.compile import Compiler  # noqa: E402
from molix.md import (  # noqa: E402
    EV_PER_AMU_A2_FS2,
    MD,
    CallableForceField,
    MaxwellBoltzmann,
    MDCheckpointHook,
    PeriodicNeighborList,
    TrajectoryHook,
)

#: Re-exported for the out-of-tree comparison scripts that share this system
#: preparation; molix.units is the single source.
from molix.units import KB_AMU_A_FS  # noqa: E402,F401
from molpot.derivation.force import autograd_forces_from_energy  # noqa: E402
from molzoo.mace import MACEPotential  # noqa: E402


def read_poscar(path: Path) -> dict[str, torch.Tensor]:
    """Read a VASP POSCAR/CONTCAR into a system dict.

    Inline rather than in ``molix.datasets`` because this is the only POSCAR
    reader in the tree; promote it if a second caller appears.

    ``molpy`` is imported inside the function (the pattern ``molix.datasets``
    already uses for ``Element``) so that ``--system`` runs never touch it. That
    matters on aarch64/GH200, where the available molrs build predates
    ``molpy.Element``: system preparation happens wherever molpy works, and the
    compute node only replays the resulting ``system.pt``.

    Args:
        path: POSCAR file (Cartesian or Direct coordinates, no selective dynamics).

    Returns:
        ``{"Z": (N,), "pos": (N, 3) in Angstrom, "cell": (3, 3), "mass": (N,)}``.
    """
    from molpy import Element

    lines = [ln.strip() for ln in path.read_text().splitlines()]
    scale = float(lines[1])
    cell = torch.tensor(
        [[float(v) for v in lines[i].split()] for i in (2, 3, 4)], dtype=config.ftype
    ) * scale
    symbols = lines[5].split()
    counts = [int(v) for v in lines[6].split()]
    mode = lines[7].lower()
    if mode.startswith("s"):
        raise ValueError("selective dynamics POSCAR is not supported")
    direct = mode.startswith("d")

    n_atoms = sum(counts)
    coords = torch.tensor(
        [[float(v) for v in lines[8 + i].split()[:3]] for i in range(n_atoms)],
        dtype=config.ftype,
    )
    pos = coords @ cell if direct else coords * scale
    elements = [Element(sym) for sym, n in zip(symbols, counts) for _ in range(n)]
    return {
        "Z": torch.tensor([e.number for e in elements], dtype=torch.long),
        "pos": pos,
        "cell": cell,
        "mass": torch.tensor([e.mass for e in elements], dtype=config.ftype),
    }


def _matpes_energy_forces(
    model: MACEPotential,
    *,
    Z: torch.Tensor,
    neighbors: PeriodicNeighborList,
    compile_energy: bool = False,
    autocast_dtype: torch.dtype | None = None,
):
    """Closure ``pos -> (energy, forces)`` over MACE-MatPES' compiled energy core.

    Only the model-specific part lives here — the compiled/autocast energy plus
    in-place autograd forces (MACE's ``get_outputs`` shape, written out rather
    than calling ``model.energy_forces`` so the compiled closure can substitute
    the eager one). The ForceField contract, neighbour-rebuild cadence and unit
    bridge are :class:`molix.md.CallableForceField`'s job.

    ``compile_energy`` wraps the energy in ``Compiler(cuda_graphs=True)``
    (inductor + ``reduce-overhead`` + CUDA graphs). Only the energy is
    compiled; ``autograd.grad`` runs outside the graph. Valid here because the
    fixed-capacity neighbour list keeps every shape static — 8.5x (fp64) /
    14.6x (fp32) on GH200, see
    ``docs/molix/explanation/throughput-and-compilation.md``.
    """
    core = model.energy_core
    if autocast_dtype is not None:
        # The autocast region must sit INSIDE the compiled callable: wrapped
        # outside, it invalidates the CUDA-graph capture and the "compiled"
        # bf16 arm runs 4x slower than fp32 (measured 52 vs 12 ms/step).
        # Inside, dynamo traces the region and inductor fuses through it.
        def core(p, Z, ei, batch, ng, shifts, _f=model.energy_core, _d=autocast_dtype):
            with torch.autocast(device_type="cuda", dtype=_d):
                return _f(p, Z, ei, batch, ng, shifts)

    energy_fn = Compiler(cuda_graphs=True)(core) if compile_energy else core
    batch = torch.zeros(Z.shape[0], dtype=torch.long, device=Z.device)

    def energy_forces(pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        leaf = pos.detach().requires_grad_(True)
        with torch.enable_grad():
            # (E, 2) end to end: the list's rebuilt-in-place buffer feeds the
            # core directly — one storage, no per-step transpose.
            energy = energy_fn(
                leaf, Z, neighbors.edge_index, batch, 1, neighbors.shifts
            )
        forces = autograd_forces_from_energy(energy, leaf)
        return energy.sum().detach(), forces.detach()

    return energy_forces


def main() -> None:
    """Parse arguments, run NVE, and write the trajectory."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--structure", type=Path, help="POSCAR (needs molpy.Element)")
    source.add_argument("--system", type=Path, help="system.pt from a previous --structure run")
    parser.add_argument(
        "--dump-system",
        type=Path,
        help="write the parsed system to this path and exit (prepare on a molpy host)",
    )
    parser.add_argument("--weights-dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--dt", type=float, default=0.5, help="timestep in fs")
    parser.add_argument("--temperature", type=float, default=300.0, help="initial T in K")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--precision",
        choices=("fp64", "fp32"),
        default="fp64",
        help="parameter/state precision (bf16 is --autocast-bf16 on top of fp32)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100_000,
        help="steps between restartable checkpoints (0 disables)",
    )
    parser.add_argument("--resume", type=Path, help="checkpoint to resume from")
    parser.add_argument(
        "--flush-every",
        type=int,
        default=10_000,
        help="trajectory frames buffered in host RAM before spilling a shard",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--fallback",
        choices=("auto", "on", "off"),
        default="auto",
        help="cuEquivariance pure-torch fallback; auto = on for CPU, off for CUDA",
    )
    parser.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    parser.add_argument(
        "--rebuild-every",
        type=int,
        default=5,
        help="rebuild the neighbour list every N steps (0 = never, frozen list)",
    )
    parser.add_argument(
        "--capacity-factor",
        type=float,
        default=1.35,
        help="edge-buffer capacity as a multiple of the initial edge count",
    )
    parser.add_argument(
        "--autocast-bf16",
        action="store_true",
        help="run the model under torch.autocast(bfloat16); state stays at --precision",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="compile the energy with Compiler(cuda_graphs=True); needs static shapes, "
             "which the frozen neighbour list provides",
    )
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    config.set_precision(args.precision)

    device = torch.device(args.device)
    # The fused cuEquivariance kernels need cuequivariance_ops_torch and a GPU;
    # on CPU the pure-torch fallback is the only path.
    use_fallback = device.type != "cuda" if args.fallback == "auto" else args.fallback == "on"
    # Report actual capability, not the request: without the ops wheel cuEq
    # silently degrades ~30x while still honouring use_fallback=False.
    try:
        import cuequivariance_ops_torch  # noqa: F401

        fused_available = True
    except ImportError:
        fused_available = False
    print(f"device: {device} (cuEq fused kernels: requested={not use_fallback}, "
          f"available={fused_available})")
    if not use_fallback and not fused_available:
        print(
            "WARNING: fused kernels requested but cuequivariance-ops-torch is not "
            "installed (pip install 'molnex[cueq-cu13]'); cuEq degrades to its "
            "pure-torch path, ~30x slower on this model"
        )

    if args.structure is not None:
        system = read_poscar(args.structure)
        if args.dump_system:
            args.dump_system.parent.mkdir(parents=True, exist_ok=True)
            torch.save(system, args.dump_system)
            print(f"wrote {args.dump_system}")
            return
    else:
        system = torch.load(args.system, map_location="cpu", weights_only=True)
    if args.weights_dir is None or args.out is None:
        parser.error("--weights-dir and --out are required unless --dump-system is given")
    Z, pos, cell = system["Z"], system["pos"].to(config.ftype), system["cell"].to(config.ftype)

    # from_checkpoint leaves the model in training mode by contract; .eval() is
    # the caller's step, as the collapsed build_model used to do.
    model = MACEPotential.from_checkpoint(
        args.weights_dir / "matpes_r2scan_config.json",
        args.weights_dir / "matpes_r2scan_cueq_state.pt",
        use_fallback=use_fallback,
    ).eval()
    r_max = float(model.cutoff_fn.r_cut)
    print(f"system: {Z.numel()} atoms, cell diag {torch.diagonal(cell).tolist()}, r_max {r_max}")

    pos = pos.to(device)
    mass = system["mass"].to(dtype=config.ftype, device=device)
    # PeriodicNeighborList validates r_max <= L/2 itself and sizes its buffers
    # from this configuration; MD(rebuild_every=) drives the refresh cadence.
    neighbors = PeriodicNeighborList(
        cell=cell.to(device), cutoff=r_max, positions=pos, capacity_factor=args.capacity_factor
    )
    print(
        f"edges: {neighbors.num_edges} (mean {neighbors.num_edges / Z.numel():.1f} per atom), "
        f"buffer capacity {neighbors.capacity}"
    )

    model = model.to(device)
    force_field = CallableForceField(
        _matpes_energy_forces(
            model,
            Z=Z.to(device),
            neighbors=neighbors,
            compile_energy=args.compile,
            autocast_dtype=torch.bfloat16 if args.autocast_bf16 else None,
        ),
        neighbors=neighbors,
        energy_scale=1.0 / EV_PER_AMU_A2_FS2,
    ).to(device)

    t0 = time.perf_counter()
    out0 = force_field(pos)
    print(
        f"single point: E = {float(out0.energy) * EV_PER_AMU_A2_FS2:.6f} eV, "
        f"|F|max = {float(out0.forces.abs().max()) * EV_PER_AMU_A2_FS2:.6f} eV/A "
        f"({time.perf_counter() - t0:.2f} s)"
    )

    start_step = 0
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
        start_step = int(ckpt["step"])
        pos = ckpt["pos"].to(dtype=config.ftype, device=device)
        resume_vel = ckpt["vel"].to(dtype=config.ftype, device=device)
        args.out = args.out.with_name(f"{args.out.stem}.from{start_step}{args.out.suffix}")
        print(f"resuming at step {start_step}; trajectory continues in {args.out}")

    run_hooks: list = [
        TrajectoryHook(
            args.out,
            stride=args.stride,
            numbers=Z,
            write_xyz=False,
            with_forces=True,
            flush_every=args.flush_every,
        )
    ]
    if args.checkpoint_every:
        run_hooks.append(
            MDCheckpointHook(
                args.out.with_suffix(".ckpt.pt"),
                every=args.checkpoint_every,
                step_offset=start_step,
            )
        )
    md = MD(
        force_field,
        mass=mass,
        dt=args.dt,
        gamma=0.0,  # NVE
        dtype=config.ftype,
        # autocast lives inside the force field's compiled callable (above);
        # wrapping again here would just add per-call context overhead.
        rebuild_every=args.rebuild_every or None,
        hooks=run_hooks,
        seed=args.seed,
        device=device,
    )
    vel = resume_vel if args.resume is not None else MaxwellBoltzmann(mass).sample(
        args.temperature, seed=args.seed
    )
    remaining = args.steps - start_step
    if remaining <= 0:
        print(f"nothing to do: checkpoint already at step {start_step} >= {args.steps}")
        return

    # Observation cadence only — neighbour rebuild runs inside each force
    # evaluation (Integrator.eval_force) and must not force chunk=1.
    chunk = max(1, int(args.stride))
    if args.checkpoint_every:
        chunk = math.gcd(chunk, int(args.checkpoint_every))
    print(
        f"advancing in chunks of {chunk} steps "
        f"(rebuild_every={args.rebuild_every} at force-eval positions)"
    )

    t0 = time.perf_counter()
    md.run(pos, vel, remaining, chunk=chunk)
    elapsed = time.perf_counter() - t0
    print(
        f"NVE: {remaining} steps in {elapsed:.1f} s ({elapsed / remaining:.4f} s/step); "
        f"neighbour rebuilds: {neighbors.rebuild_count}"
    )

    traj = torch.load(args.out, map_location="cpu", weights_only=True)
    etot = traj["etot"] * EV_PER_AMU_A2_FS2
    n_frames = int(etot.numel())
    if n_frames < 2:
        print(
            f"E_tot drift: only {n_frames} trajectory frame(s) "
            f"(stride={args.stride}, steps={args.steps}) — cannot estimate; "
            f"use checkpoint log instead"
        )
    else:
        # Frames are kept at step stride, 2*stride, ...; span ≈ (n_frames)*stride*dt
        t_ps = n_frames * args.stride * args.dt * 1e-3
        dE_meV_atom = float(etot[-1] - etot[0]) / Z.numel() * 1e3
        rate = dE_meV_atom / t_ps if t_ps > 0 else float("nan")
        print(
            f"E_tot drift = {dE_meV_atom:.4f} meV/atom over ~{t_ps:.2f} ps "
            f"({n_frames} frames, {rate:.6f} meV/atom/ps); "
            f"T range {float(traj['temp'].min()):.1f}-{float(traj['temp'].max()):.1f} K"
        )
    print(
        f"neighbour rebuilds this segment: {neighbors.rebuild_count} "
        f"(expect ~{remaining if args.rebuild_every == 1 else remaining // max(args.rebuild_every, 1)} "
        f"for rebuild_every={args.rebuild_every} at force-eval)"
    )
    # The comparison step needs the exact graph the trajectory was produced on.
    # edge_index is (E, 2) [source, target] — the repo edge convention.
    torch.save(
        {
            "Z": Z.cpu(),
            "cell": cell.cpu(),
            "edge_index": neighbors.edge_index.cpu(),
            "shifts": neighbors.shifts.cpu(),
            "num_edges": neighbors.num_edges,
            "rebuild_count": neighbors.rebuild_count,
            "energy_scale": EV_PER_AMU_A2_FS2,
        },
        args.out.with_suffix(".graph.pt"),
    )
    print(f"wrote {args.out} and {args.out.with_suffix('.graph.pt')}")


if __name__ == "__main__":
    main()
