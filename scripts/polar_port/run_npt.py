"""100-step (default) Polar NPT on a packed PEO box through molix.md.

Loads the official Polar dump plus the out-of-tree convert_e3nn_cueq overlay.
No ASE / e3nn / mace-torch. GPU by default when CUDA is visible.

    PYTHONPATH=src python scripts/polar_port/run_npt.py \\
        --system /tmp/peo_box.pt --steps 100 --device cuda
"""

from __future__ import annotations

import argparse
import importlib.metadata as _metadata
import time
from pathlib import Path

# L40 venv ships molrs 0.13 metadata against an editable molpy 0.14 tree.
_real_version = _metadata.version


def _compat_version(name: str) -> str:
    if name == "molcrafts-molrs":
        return "0.14.0"
    return _real_version(name)


_metadata.version = _compat_version  # type: ignore[method-assign]

import torch
from tensordict import TensorDict

from molix import config
from molix.md import (
    MD,
    BerendsenBarostat,
    LangevinVerletIntegrator,
    MaxwellBoltzmann,
    MDHook,
    NeighborList,
    VelocityInitHook,
)
from molix.units import units
from molpot.forcefield import ForceField as MolpotForceField
from molzoo.mace.checkpoint import CheckpointRemap, polar_keys
from molzoo.mace.polar import MacePolar

_KB_MD = (1.0 * units.k_B).to("amu * angstrom ** 2 / femtosecond ** 2 / kelvin").magnitude


class _Beat(MDHook):
    """Print a heartbeat so a 100-step job is not silent until the end."""

    cadence = 10

    def on_step_end(self, runner, step, obs) -> None:  # noqa: ANN001
        if step % 10 == 0:
            vol = float(obs.volume) if obs.volume is not None else float("nan")
            fmax = (
                float(obs.forces.detach().abs().max()) if obs.forces is not None else float("nan")
            )
            print(
                f"step {step:4d}  E={float(obs.total):.4f}  "
                f"T={float(obs.temperature):.1f} K  V={vol:.1f} Å³  "
                f"|F|max={fmax:.4f}",
                flush=True,
            )


def _system_td(
    *,
    pos: torch.Tensor,
    vel: torch.Tensor,
    mass: torch.Tensor,
    z: torch.Tensor,
    cell: torch.Tensor,
    device: torch.device,
) -> TensorDict:
    n = int(pos.shape[0])
    cell_b = cell if cell.dim() == 3 else cell.unsqueeze(0)
    return TensorDict(
        {
            "atoms": TensorDict(
                {
                    "pos": pos,
                    "vel": vel,
                    "mass": mass,
                    "Z": z.to(device=device, dtype=torch.long),
                    "batch": torch.zeros(n, dtype=torch.long, device=device),
                },
                batch_size=[n],
            ),
            "graphs": TensorDict(
                {
                    "cell": cell_b,
                    "total_charge": torch.zeros(1, dtype=pos.dtype, device=device),
                    "total_spin": torch.ones(1, dtype=pos.dtype, device=device),
                },
                batch_size=[1],
            ),
        },
        batch_size=[],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", type=Path, default=Path("/tmp/peo_box.pt"))
    parser.add_argument("--dump", type=Path, default=Path("/tmp/mace-polar/MACE-POLAR-1-M.model"))
    parser.add_argument("--cueq-state", type=Path, default=Path("/tmp/polar_cueq_state.pt"))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--dt", type=float, default=0.5, help="fs")
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--pressure", type=float, default=1.0, help="bar")
    parser.add_argument("--gamma", type=float, default=0.01, help="Langevin γ in 1/fs")
    parser.add_argument("--tau", type=float, default=200.0, help="Berendsen τ_p in fs")
    parser.add_argument("--skin", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=("fp32", "fp64"), default="fp32")
    args = parser.parse_args()

    config.set_precision(args.precision)
    dtype = torch.float64 if args.precision == "fp64" else torch.float32
    device = torch.device(args.device)
    print(f"device: {device}  precision: {args.precision}  cuda: {torch.cuda.is_available()}")
    print(f"inference: grad_enabled={torch.is_grad_enabled()}  (MD.run uses no_grad)")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    blob = torch.load(args.system, map_location="cpu", weights_only=True)
    z = blob["Z"]
    pos = blob["pos"].to(dtype=dtype, device=device)
    mass = blob["mass"].to(dtype=dtype, device=device)
    cell = blob["cell"].to(dtype=dtype, device=device)
    print(f"system: {int(z.numel())} atoms, L={float(cell[0, 0]):.3f} Å")

    if device.type != "cuda":
        raise RuntimeError("Polar NPT on GPU only; CPU fallback is forbidden")
    try:
        import cuequivariance_ops_torch  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "cuequivariance_ops_torch is required (fused cuEq). "
            "Missing ops is an environment error, not a fallback."
        ) from exc
    print("cuEq fused kernels: True")
    model = MacePolar.load(
        args.dump,
        format="official",
        use_fallback=False,
        compute_forces=False,
    )
    cueq_sd = torch.load(args.cueq_state, map_location="cpu", weights_only=True)
    CheckpointRemap(polar_keys, on_unexpected="return").load(model, cueq_sd)
    model = model.to(device=device, dtype=dtype).eval()
    model.requires_grad_(False)
    r_max = float(model._spec.r_max)
    energy_only = model._pipeline.__func__ is model._write_energy.__func__
    print(
        f"Polar r_max={r_max} Å  params={sum(p.numel() for p in model.parameters())}  "
        f"use_fallback={model._spec.use_fallback}  torch.compile=False  "
        f"pipeline={'energy' if energy_only else 'ef'}"
    )

    neighbors = NeighborList(
        cell=cell,
        cutoff=r_max,
        positions=pos,
        skin=args.skin,
        every=1,
        delay=0,
        # ASE's MACE calculator rebuilds every force call. Verlet check=True
        # skipped rebuilds until skin/2 (~20 fs for H at 300 K) — the same
        # step where T exploded.
        check=False,
        capacity_factor=1.2,
        device=device,
    )
    print(f"neighbors: r_build={neighbors.r_build:.2f} Å  capacity={neighbors.capacity}")

    ff = MolpotForceField("polar", units="metal")
    ff.def_pairstyle(model)
    kbt = _KB_MD * float(args.temperature)
    md = (
        MD(dtype)
        .set_forcefield(ff)
        .set_neighbors(neighbors)
        .set_integrator(
            LangevinVerletIntegrator(args.dt, gamma=args.gamma, kbt=kbt, seed=args.seed)
        )
        .set_barostat(BerendsenBarostat(pressure=args.pressure, tau=args.tau))
        .set_hooks(
            [
                VelocityInitHook(MaxwellBoltzmann(args.temperature, seed=args.seed)),
                _Beat(),
            ]
        )
        .to(device)
    )
    work = _system_td(
        pos=pos,
        vel=torch.zeros_like(pos),
        mass=mass,
        z=z,
        cell=cell,
        device=device,
    )

    t_sp = time.perf_counter()
    neighbors.build(work)
    n_live = neighbors.num_edges
    full_edges = work["edges"]
    work["edges"] = TensorDict(
        {key: full_edges[key][:n_live] for key in full_edges.keys()},
        batch_size=[n_live],
    )
    pots = ff.to_potentials()
    e0 = pots.calc_energy(work)
    f0 = pots.calc_forces(work)
    work["edges"] = full_edges
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(
        f"single point: E={float(e0.detach().sum()):.4f} eV  "
        f"|F|max={float(f0.detach().abs().max()):.4f} eV/Å  "
        f"({time.perf_counter() - t_sp:.2f} s)"
    )

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    state = md.run(work, args.steps)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    vol0 = float(torch.linalg.det(cell).abs())
    vol1 = float(torch.linalg.det(state.cell).abs()) if state.cell is not None else vol0
    peak = torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else float("nan")
    print(
        f"NPT {args.steps} steps  dt={args.dt:g} fs  "
        f"T={args.temperature:g} K  P={args.pressure:g} bar"
    )
    print(
        f"wall {elapsed:.2f} s  {elapsed / args.steps * 1000:.1f} ms/step  "
        f"{args.steps / elapsed:.2f} step/s  ns/day={args.steps * args.dt / elapsed * 86.4:.3f}"
    )
    print(f"V {vol0:.1f} → {vol1:.1f} Å³  L {vol1 ** (1.0 / 3.0):.3f} Å")
    if device.type == "cuda":
        print(f"peak GPU memory {peak:.2f} GiB")


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    main()
