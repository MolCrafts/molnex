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
from tensordict import TensorDict

from molix import config

# Default; overridden by --precision before any model construction.
config.set_precision("fp64")

from molix.compile import Compiler  # noqa: E402
from molix.md import (  # noqa: E402
    MD,
    ForceOutput,
    LangevinVerletIntegrator,
    MaxwellBoltzmann,
    MDCheckpointHook,
    MDRunner,
    NeighborList,
    TrajectoryHook,
    VelocityInitHook,
    VelocityVerletIntegrator,
)
from molix.units import units  # noqa: E402

_EV_TO_MD = (1.0 * units.eV).to("amu * angstrom ** 2 / femtosecond ** 2").magnitude
_KB_MD = (1.0 * units.k_B).to("amu * angstrom ** 2 / femtosecond ** 2 / kelvin").magnitude
from molpot.derivation.force import autograd_forces_from_energy  # noqa: E402
from molzoo.mace import MACEPotential  # noqa: E402


def read_poscar(path: Path) -> dict[str, torch.Tensor]:
    """Read a VASP POSCAR/CONTCAR into a system dict.

    Inline rather than in a library package because this is the only POSCAR
    reader in the tree, and dataset acquisition is MolHub's; promote it there
    if a second caller appears.

    ``molpy`` is imported inside the function (the usual pattern for
    ``Element``) so that ``--system`` runs never touch it. That
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
    cell = (
        torch.tensor([[float(v) for v in lines[i].split()] for i in (2, 3, 4)], dtype=config.ftype)
        * scale
    )
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
    neighbors: NeighborList,
    potential_dtype: torch.dtype,
    compile_energy: bool = False,
    autocast_dtype: torch.dtype | None = None,
):
    """Closure ``pos -> (energy, forces)`` over MACE-MatPES' compiled energy core.

    Only the model-specific part lives here — the compiled/autocast energy plus
    in-place autograd forces (MACE's ``get_outputs`` shape, written out rather
    than calling ``model.energy_forces`` so the compiled closure can substitute
    the eager one). Neighbour rebuild stays on NeighborList; the unit
    bridge is applied in ``_PosForce`` before the integrator.

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
        # The force field owns its precision: the leaf lives in the MODEL's
        # dtype (a same-dtype .to is the identity), forces come back in it,
        # and Integrator.eval_force casts to the state dtype at the boundary —
        # the split-precision contract of molix.md.driver.MD.
        leaf = pos.detach().to(potential_dtype).requires_grad_(True)
        with torch.enable_grad():
            # (E, 2) end to end: the list's rebuilt-in-place buffer feeds the
            # core directly — one storage, no per-step transpose.
            energy = energy_fn(
                leaf, Z, neighbors.edge_index, batch, 1, neighbors.shifts.to(potential_dtype)
            )
        forces = autograd_forces_from_energy(energy, leaf)
        return energy.sum().detach(), forces.detach()

    return energy_forces


class _PosForce:
    """PotentialLike: pos -> ForceOutput. MD.set_forcefield is molpy-only."""

    def __init__(self, energy_forces, neighbors: NeighborList, scale: float) -> None:
        self._ef = energy_forces
        self._nl = neighbors
        self._scale = float(scale)

    @property
    def rebuilds_neighbors(self) -> bool:
        return True

    def rebuild_neighbors(self, pos: torch.Tensor) -> None:
        self._nl.update(pos)

    def bind_work(self, _work) -> None:
        return None

    def to_pots(self, _device) -> None:
        return None

    def __call__(self, pos: torch.Tensor) -> ForceOutput:
        energy, forces = self._ef(pos)
        return ForceOutput(energy * self._scale, forces * self._scale)


def _bind_md(dtype, force_field, integrator, neighbors, hooks, device) -> MD:
    """Public MD composition: set_forcefield + set_neighbors + set_integrator."""
    return (
        MD(dtype)
        .set_forcefield(force_field)
        .set_neighbors(neighbors)
        .set_integrator(integrator)
        .set_hooks(hooks)
        .to(device)
    )


def minimize_fire(
    force_field,
    pos: torch.Tensor,
    neighbors: NeighborList,
    *,
    fmax_eV_A: float,
    max_steps: int,
    ev_to_md: float,
) -> tuple[torch.Tensor, dict]:
    """FIRE on the live force field. ``pos`` is updated in place and returned."""
    dt = 0.1
    dt_max = 1.0
    n_min = 5
    f_inc = 1.1
    f_dec = 0.5
    alpha_start = 0.1
    alpha = alpha_start
    vel = torch.zeros_like(pos)
    n_pos = 0
    e_eV = 0.0
    fmax = float("inf")
    step = 0
    for step in range(1, max_steps + 1):
        neighbors.update(pos)
        work = TensorDict(
            {"atoms": TensorDict({"pos": pos}, batch_size=[int(pos.shape[0])])},
            batch_size=[],
        )
        neighbors.build(work)
        pots = force_field.to_potentials()
        energy = pots.calc_energy(work)
        forces = pots.calc_forces(work)

        class _Out:
            pass

        out = _Out()
        out.energy = energy.sum()
        out.forces = forces
        forces = out.forces
        e_eV = float(out.energy) / ev_to_md
        fmax = float((forces / ev_to_md).abs().max())
        if fmax < fmax_eV_A:
            break
        power = float((forces * vel).sum())
        if power > 0.0:
            n_pos += 1
            if n_pos > n_min:
                dt = min(dt * f_inc, dt_max)
                alpha *= 0.99
        else:
            vel.zero_()
            dt *= f_dec
            alpha = alpha_start
            n_pos = 0
        fnorm = float(forces.norm())
        vnorm = float(vel.norm())
        if fnorm > 0.0:
            vel = (1.0 - alpha) * vel + alpha * forces * (vnorm / fnorm)
        vel = vel + dt * forces
        pos = pos + dt * vel
    neighbors.update(pos)
    return pos, {
        "steps": step,
        "E_eV": e_eV,
        "fmax_eV_A": fmax,
        "converged": fmax < fmax_eV_A,
        "max_steps": max_steps,
        "fmax_target": fmax_eV_A,
    }


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
    parser.add_argument("--temperature", type=float, default=300.0, help="initial / NVT target T in K")
    parser.add_argument(
        "--nvt-steps",
        type=int,
        default=0,
        help="Langevin NVT steps at --temperature before NVE (0 = Maxwell-Boltzmann then NVE)",
    )
    parser.add_argument(
        "--nvt-gamma",
        type=float,
        default=0.001,
        help="Langevin friction γ in 1/fs (used only when --nvt-steps > 0)",
    )
    parser.add_argument(
        "--minimize-fmax",
        type=float,
        default=0.0,
        help="FIRE |F|_max in eV/A before NVT/NVE. 0 skips minimization.",
    )
    parser.add_argument(
        "--minimize-steps",
        type=int,
        default=2000,
        help="FIRE step cap when --minimize-fmax > 0.",
    )
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--potential-precision",
        choices=["fp64", "fp32"],
        default=None,
        help="model/inference precision; defaults to --precision. Setting it below "
        "--precision runs the split-precision configuration (e.g. an fp64 "
        "trajectory over fp32 inference; MD state stays at --precision)",
    )
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
        "--skin",
        type=float,
        default=1.0,
        help="Verlet skin in A: the list is built at cutoff + skin and stays complete "
        "to cutoff while no atom has moved more than skin/2 (0 = the no-skin limit, "
        "rebuild whenever anything moved)",
    )
    parser.add_argument(
        "--every", type=int, default=1, help="attempt a rebuild only every N steps (LAMMPS every)"
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=0,
        help="attempt no rebuild until N steps after the last one (LAMMPS delay; "
        "must be a multiple of --every)",
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="rebuild on cadence alone, without the half-skin displacement test "
        "(cheaper, and never free: it accepts missed pairs and the energy leak they cause)",
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
        "which the neighbour list's fixed-capacity buffers provide across rebuilds "
        "(the list is refreshed in place, never reallocated)",
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
    print(
        f"device: {device} (cuEq fused kernels: requested={not use_fallback}, "
        f"available={fused_available})"
    )
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
    # NeighborList validates r_build <= L/2 itself, sizes its buffers from this
    # configuration, and owns the refresh policy (skin / every / delay / check);
    # Integrator.eval_force asks it once per force evaluation.
    neighbors = NeighborList(
        cell=cell.to(device),
        cutoff=r_max,
        positions=pos,
        skin=args.skin,
        every=args.every,
        delay=args.delay,
        check=not args.no_check,
        capacity_factor=args.capacity_factor,
    )
    print(
        f"edges: {neighbors.num_edges} (mean {neighbors.num_edges / Z.numel():.1f} per atom), "
        f"buffer capacity {neighbors.capacity}"
    )
    print(
        f"neighbour policy: skin={neighbors.skin:g} A, every={neighbors.every}, "
        f"delay={neighbors.delay}, check={neighbors.check}, r_build={neighbors.r_build:.2f} A"
    )

    potential_dtype = {"fp64": torch.float64, "fp32": torch.float32}[
        args.potential_precision or args.precision
    ]
    model = model.to(device=device, dtype=potential_dtype)
    print(
        f"precision: MD state {args.precision}, "
        f"potential {args.potential_precision or args.precision}"
    )
    from molpot.forcefield import ForceField as MolpotForceField

    if args.compile:
        _matpes_energy_forces(
            model,
            Z=Z.to(device),
            neighbors=neighbors,
            potential_dtype=potential_dtype,
            compile_energy=True,
            autocast_dtype=torch.bfloat16 if args.autocast_bf16 else None,
        )
    ff = MolpotForceField("matpes", units="metal")
    ff.def_pairstyle(model)
    force_field = ff

    t0 = time.perf_counter()
    _probe = TensorDict(
        {
            "atoms": TensorDict(
                {
                    "pos": pos,
                    "Z": Z.to(device),
                    "batch": torch.zeros(Z.numel(), dtype=torch.long, device=device),
                    "mass": mass,
                },
                batch_size=[int(Z.numel())],
            ),
            "graphs": TensorDict({"cell": cell.to(device)}, batch_size=[1]),
        },
        batch_size=[],
    )
    neighbors.build(_probe)
    pots0 = force_field.to_potentials()
    e0 = pots0.calc_energy(_probe)
    f0 = pots0.calc_forces(_probe)
    print(
        f"single point: E = {float(e0.sum()) :.6f} eV, "
        f"|F|max = {float(f0.abs().max()):.6f} eV/A "
        f"({time.perf_counter() - t0:.2f} s)"
    )

    if args.minimize_fmax > 0.0 and args.resume is None:
        if args.minimize_steps < 1:
            raise ValueError(f"--minimize-steps must be >= 1, got {args.minimize_steps}")
        print(
            f"FIRE minimize fmax={args.minimize_fmax:g} eV/A  max_steps={args.minimize_steps}"
        )
        t_min = time.perf_counter()
        pos, min_info = minimize_fire(
            force_field,
            pos,
            neighbors,
            fmax_eV_A=args.minimize_fmax,
            max_steps=args.minimize_steps,
            ev_to_md=_EV_TO_MD,
        )
        print(
            f"FIRE done in {time.perf_counter() - t_min:.1f} s: conv={min_info['converged']} "
            f"steps={min_info['steps']} E={min_info['E_eV']:.6f} eV "
            f"|F|max={min_info['fmax_eV_A']:.6f} eV/A"
        )

    start_step = 0
    resume_vel = None
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
        start_step = int(ckpt["step"])
        pos = ckpt["pos"].to(dtype=config.ftype, device=device)
        resume_vel = ckpt["vel"].to(dtype=config.ftype, device=device)
        args.out = args.out.with_name(f"{args.out.stem}.from{start_step}{args.out.suffix}")
        print(f"resuming at step {start_step}; trajectory continues in {args.out}")
        # NeighborList was built at the IC; resume positions are far away in
        # unwrapped space. Rebuild here so the unwrap/displacement test is
        # anchored at the resume frame, not the IC.
        neighbors.update(pos)

    if args.nvt_steps < 0:
        raise ValueError(f"--nvt-steps must be >= 0, got {args.nvt_steps}")
    if args.nvt_steps > 0:
        if args.nvt_gamma <= 0.0:
            raise ValueError(
                f"--nvt-gamma must be > 0 when --nvt-steps > 0, got {args.nvt_gamma}"
            )
        if args.temperature <= 0.0:
            raise ValueError(f"--temperature must be > 0 for NVT, got {args.temperature}")

    n_atoms = int(pos.shape[0])
    atoms = TensorDict(
        {
            "atoms": TensorDict(
                {
                    "pos": pos,
                    "vel": torch.zeros_like(pos) if resume_vel is None else resume_vel,
                    "mass": mass,
                },
                batch_size=[n_atoms],
            )
        },
        batch_size=[],
    )

    def _instant_T(*, remove_com: bool) -> float:
        vel = atoms["atoms", "vel"]
        mass_col = atoms["atoms", "mass"].reshape(-1, 1)
        if remove_com:
            vel = vel - (mass_col * vel).sum(0) / mass_col.sum()
        ke = 0.5 * (mass_col * vel * vel).sum()
        dof = 3 * n_atoms - (3 if remove_com else 0)
        return float(2.0 * ke / (dof * _KB_MD))

    if args.nvt_steps > 0 and args.resume is None:
        nvt_out = args.out.with_name("nvt.pt")
        if nvt_out.is_file():
            nvt_traj = torch.load(nvt_out, map_location="cpu", weights_only=True)
            atoms["atoms", "pos"] = nvt_traj["pos"][-1].to(dtype=config.ftype, device=device)
            vel = nvt_traj["vel"][-1].to(dtype=config.ftype, device=device)
            mass_col = atoms["atoms", "mass"].reshape(-1, 1)
            atoms["atoms", "vel"] = vel - (mass_col * vel).sum(0) / mass_col.sum()
            neighbors.rebuild(atoms["atoms", "pos"])
            t_mean = float(nvt_traj["temp"][-min(20, nvt_traj["temp"].numel()) :].mean())
            print(
                f"reuse {nvt_out} last-window mean T={t_mean:.1f} K; "
                f"NVE start T={_instant_T(remove_com=True):.1f} K"
            )
        else:
            nvt_stride = max(1, min(args.nvt_steps, 1000))
            nvt_chunk = nvt_stride
            print(
                f"NVT: {args.nvt_steps} steps Langevin at {args.temperature:g} K, "
                f"gamma={args.nvt_gamma:g} /fs, dt={args.dt:g} fs"
            )
            nvt = _bind_md(
                config.ftype,
                force_field,
                LangevinVerletIntegrator(
                    args.dt,
                    gamma=args.nvt_gamma,
                    kbt=_KB_MD * args.temperature,
                    seed=args.seed,
                ),
                neighbors,
                [
                    VelocityInitHook(MaxwellBoltzmann(args.temperature, seed=args.seed)),
                    TrajectoryHook(
                        nvt_out,
                        stride=nvt_stride,
                        numbers=Z,
                        write_xyz=False,
                        with_forces=True,
                        flush_every=args.flush_every,
                    ),
                ],
                device,
            )
            t_nvt = time.perf_counter()
            nvt.run(atoms, args.nvt_steps, chunk=nvt_chunk)
            print(
                f"NVT done in {time.perf_counter() - t_nvt:.1f} s; "
                f"T_inst (3N) = {_instant_T(remove_com=False):.1f} K"
            )
            if not nvt_out.is_file():
                raise FileNotFoundError(f"NVT trajectory {nvt_out} was not written")
            nvt_traj = torch.load(nvt_out, map_location="cpu", weights_only=True)
            t_last = float(nvt_traj["temp"][-1])
            t_mean = float(nvt_traj["temp"][-min(20, nvt_traj["temp"].numel()) :].mean())
            print(
                f"NVT last-frame T = {t_last:.1f} K, last-window mean T = {t_mean:.1f} K "
                f"(target {args.temperature:g} K)"
            )
            if t_mean > args.temperature + 150.0:
                raise RuntimeError(
                    f"NVT did not cool: last-window mean T={t_mean:.1f} K "
                    f"vs target {args.temperature:g} K"
                )
            vel = atoms["atoms", "vel"]
            mass_col = atoms["atoms", "mass"].reshape(-1, 1)
            atoms["atoms", "vel"] = vel - (mass_col * vel).sum(0) / mass_col.sum()
            neighbors.update(atoms["atoms", "pos"])
            print(f"NVE start T (3N-3, COM removed) = {_instant_T(remove_com=True):.1f} K")

    run_hooks: list = []
    if args.resume is None and args.nvt_steps == 0:
        run_hooks.append(VelocityInitHook(MaxwellBoltzmann(args.temperature, seed=args.seed)))
    run_hooks.append(
        TrajectoryHook(
            args.out,
            stride=args.stride,
            numbers=Z,
            write_xyz=False,
            with_forces=True,
            flush_every=args.flush_every,
        )
    )
    if args.checkpoint_every:
        run_hooks.append(
            MDCheckpointHook(
                args.out.with_suffix(".ckpt.pt"),
                every=args.checkpoint_every,
                step_offset=start_step,
            )
        )
    md = _bind_md(
        config.ftype,
        force_field,
        VelocityVerletIntegrator(args.dt),
        neighbors,
        run_hooks,
        device,
    )
    remaining = args.steps - start_step
    if remaining <= 0:
        print(f"nothing to do: checkpoint already at step {start_step} >= {args.steps}")
        return

    # Observation cadence only — the neighbour policy runs inside each force
    # evaluation (Integrator.eval_force) and must not force chunk=1.
    chunk = max(1, int(args.stride))
    if args.checkpoint_every:
        chunk = math.gcd(chunk, int(args.checkpoint_every))
    print(f"advancing in chunks of {chunk} steps (policy asked at every force evaluation)")

    rebuilds_before, ndanger_before = neighbors.rebuild_count, neighbors.ndanger
    t0 = time.perf_counter()
    md.run(atoms, remaining, chunk=chunk)
    elapsed = time.perf_counter() - t0
    rebuilds = neighbors.rebuild_count - rebuilds_before
    ndanger = neighbors.ndanger - ndanger_before
    print(
        f"NVE: {remaining} steps in {elapsed:.1f} s ({elapsed / remaining:.4f} s/step); "
        f"neighbour rebuilds: {rebuilds}"
    )

    if not args.out.is_file():
        raise FileNotFoundError(
            f"trajectory {args.out} was not written (steps={args.steps}, stride={args.stride})"
        )
    traj = torch.load(args.out, map_location="cpu", weights_only=True)
    etot = traj["etot"] / _EV_TO_MD
    n_frames = int(etot.numel())
    if n_frames < 2:
        raise RuntimeError(
            f"trajectory has {n_frames} frame(s); need >= 2 "
            f"(steps={args.steps}, stride={args.stride})"
        )
    # Frames are kept at step stride, 2*stride, ...; span ≈ (n_frames)*stride*dt
    t_ps = n_frames * args.stride * args.dt * 1e-3
    dE_meV_atom = float(etot[-1] - etot[0]) / Z.numel() * 1e3
    rate = dE_meV_atom / t_ps if t_ps > 0 else float("nan")
    print(
        f"E_tot drift = {dE_meV_atom:.4f} meV/atom over ~{t_ps:.2f} ps "
        f"({n_frames} frames, {rate:.6f} meV/atom/ps); "
        f"T range {float(traj['temp'].min()):.1f}-{float(traj['temp'].max()):.1f} K"
    )
    # Measured, not predicted: the cadence is the list's, driven by how far the
    # atoms actually moved, so there is no step-count formula to check against.
    # ndanger is the alarm — nonzero at skin > 0 means a rebuild came too late
    # and pairs were missed, so rerun with a larger skin or a tighter gate.
    print(
        f"neighbour rebuilds this segment: {rebuilds} of {remaining} steps "
        f"({rebuilds / max(1, remaining):.3f} per step); ndanger={ndanger} "
        f"(skin={neighbors.skin:g} A, every={neighbors.every}, delay={neighbors.delay}, "
        f"check={neighbors.check})"
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
            "energy_scale": 1.0 / _EV_TO_MD,
        },
        args.out.with_suffix(".graph.pt"),
    )
    print(f"wrote {args.out} and {args.out.with_suffix('.graph.pt')}")


if __name__ == "__main__":
    main()
