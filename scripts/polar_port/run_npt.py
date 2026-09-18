"""100-step (default) Polar NPT on a packed PEO box through molix.md.

Loads official cuEq-converted weights once, with JSON or trusted dump metadata.
No ASE / e3nn / mace-torch. GPU by default when CUDA is visible.

    PYTHONPATH=src python scripts/polar_port/run_npt.py \\
        --system /tmp/peo_box.pt --cueq-state /path/polar_cueq_state.pt \\
        --config /path/polar_config.json --steps 100 --device cuda
"""

from __future__ import annotations

import argparse
import importlib.metadata as _metadata
import json
import statistics
import time
from pathlib import Path

import torch
from tensordict import TensorDict

from molix import config
from molix.autograd import Derivative
from molix.compile import Compiler
from molix.md import (
    MD,
    BerendsenBarostat,
    LangevinVerletIntegrator,
    MaxwellBoltzmann,
    MDHook,
    NeighborList,
    VelocityInitHook,
)
from molix.units import preset_energy_to_md, units
from molpot.forcefield import ForceField as MolpotForceField
from molzoo.mace.polar import MacePolar

_KB_MD = (1.0 * units.k_B).to("amu * angstrom ** 2 / femtosecond ** 2 / kelvin").magnitude


class _Beat(MDHook):
    """Print a heartbeat so a 100-step job is not silent until the end."""

    cadence = 10

    def on_step_end(self, runner, step, obs) -> None:  # noqa: ANN001
        if step % 10 == 0:
            scale = preset_energy_to_md("metal")
            vol = float(obs.volume) if obs.volume is not None else float("nan")
            fmax = (
                float(obs.forces.detach().abs().max()) / scale
                if obs.forces is not None
                else float("nan")
            )
            print(
                f"step {step:4d}  E_total={float(obs.total) / scale:.4f} eV  "
                f"T={float(obs.temperature):.1f} K  V={vol:.1f} Å³  "
                f"|F|max={fmax:.4f} eV/Å",
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
    parser.add_argument(
        "--neighbor-bin",
        type=float,
        default=None,
        help="cell-list bin width in Å (0 selects automatic); omitted uses pair kernel",
    )
    parser.add_argument(
        "--benchmark-repeats",
        type=int,
        default=0,
        help="benchmark fixed-coordinate E/F/virial then exit without MD (0 disables)",
    )
    parser.add_argument("--benchmark-warmup", type=int, default=2)
    parser.add_argument("--benchmark-output", type=Path, default=None)
    parser.add_argument(
        "--compile-core", action="store_true", help="benchmark fullgraph Inductor core"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=("fp32", "fp64"), default="fp32")
    parser.add_argument("--potential-precision", choices=("fp32", "fp64"))
    parser.add_argument("--energy-precision", choices=("model", "fp32", "fp64"), default="fp64")
    parser.add_argument(
        "--config", type=Path, help="JSON config; avoids loading the original pickle"
    )
    parser.add_argument(
        "--trust-checkpoint", action="store_true", help="allow verified local dump pickle"
    )
    parser.add_argument(
        "--matmul-precision",
        choices=("highest", "high"),
        default="highest",
        help="highest forbids TF32 GEMMs; validate high against fp64 before MD",
    )
    args = parser.parse_args()
    if args.steps < 1 or args.benchmark_repeats < 0 or args.benchmark_warmup < 1:
        parser.error("steps/warmup must be positive and benchmark-repeats nonnegative")
    if args.compile_core and not args.benchmark_repeats:
        parser.error("--compile-core requires --benchmark-repeats until MD parity is validated")

    potential_precision = args.potential_precision or args.precision
    config.set_precision(potential_precision, matmul_precision=args.matmul_precision)
    dtype = torch.float64 if args.precision == "fp64" else torch.float32
    potential_dtype = torch.float64 if potential_precision == "fp64" else torch.float32
    energy_dtype = {"model": None, "fp32": torch.float32, "fp64": torch.float64}[
        args.energy_precision
    ]
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
        args.cueq_state if args.config else args.dump,
        format="official",
        config=args.config,
        converted_weights=None if args.config else args.cueq_state,
        trusted=args.trust_checkpoint,
        use_fallback=False,
        energy_dtype=energy_dtype,
    )
    model = model.to(device=device, dtype=potential_dtype).eval()
    model.configure_input_dtype(potential_dtype)
    model.requires_grad_(False)
    print(
        f"precision: state={args.precision} model={potential_precision} "
        f"energy={args.energy_precision} matmul={args.matmul_precision}"
    )
    r_max = float(model._spec.r_max)
    energy_only = True  # built with wrt=(): MD composes the force/virial op over it
    print(
        f"Polar r_max={r_max} Å  params={sum(p.numel() for p in model.parameters())}  "
        f"use_fallback={model._spec.use_fallback}  torch.compile=False  "
        f"pipeline={'energy' if energy_only else 'ef'}"
    )
    print(f"conv weight_numel: {[layer.conv_tp.weight_numel for layer in model.interactions]}")

    neighbors = NeighborList(
        cell=cell,
        cutoff=r_max,
        positions=pos,
        skin=args.skin,
        every=1,
        delay=0,
        # Rebuild every MD step for this bounded validation runner. This is
        # not evidence against displacement-gated Verlet rebuilding.
        check=False,
        bin=args.neighbor_bin,
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
    n_live = neighbors.num_pairs
    full_edges = work["pairs"]
    work["pairs"] = TensorDict(
        {key: full_edges[key][:n_live] for key in full_edges.keys()},
        batch_size=[n_live],
    )
    pots = ff.to_potentials()
    if args.benchmark_repeats:
        # Fixed geometry: do not compare timings from evolving NPT states or
        # include cuEq JIT/cache construction in the steady-state samples.
        cold_start = time.perf_counter()
        with torch.no_grad():
            npt = Derivative(pots, wrt=("atoms.pos", "graphs.strain"))

            def energy_forces_virial(frame: TensorDict) -> tuple[torch.Tensor, ...]:
                out = npt(frame)
                return out["graphs", "energy"], out["atoms", "forces"], out["graphs", "virial"][0]

            eager_reference = None
            if args.compile_core:
                eager_reference = tuple(t.detach().clone() for t in energy_forces_virial(work))
                npt = Compiler(fullgraph=True, dynamic=False, donated_buffer=True)(npt)
            for _ in range(args.benchmark_warmup):
                energy, forces, virial = energy_forces_virial(work)
            torch.cuda.synchronize()
            cold_seconds = time.perf_counter() - cold_start
            reference = tuple(t.detach().clone() for t in (energy, forces, virial))
            compiled_deltas = (
                [float((a - b).abs().max()) for a, b in zip(reference, eager_reference)]
                if eager_reference is not None
                else None
            )
            torch.cuda.reset_peak_memory_stats()
            samples = []
            max_deltas = [0.0, 0.0, 0.0]
            for _ in range(args.benchmark_repeats):
                torch.cuda.synchronize()
                start = time.perf_counter()
                energy, forces, virial = energy_forces_virial(work)
                torch.cuda.synchronize()
                samples.append(time.perf_counter() - start)
                for i, (observed, expected) in enumerate(zip((energy, forces, virial), reference)):
                    if not torch.isfinite(observed).all():
                        raise RuntimeError("nonfinite fixed-coordinate E/F/virial")
                    max_deltas[i] = max(max_deltas[i], float((observed - expected).abs().max()))
            model_peak = torch.cuda.max_memory_allocated() / 1024**3
            model_reserved = torch.cuda.max_memory_reserved() / 1024**3
            force_only = Derivative(pots, wrt=("atoms.pos",))(work)["atoms", "forces"]
            force_route_delta = float((force_only - forces).abs().max())
            work["atoms", "pos"] = pos
            # Report neighbor rebuilding separately; no topology change is
            # allowed to contaminate the fixed-edge model samples above.
            work["pairs"] = full_edges
            neighbor_samples = []
            torch.cuda.reset_peak_memory_stats()
            for _ in range(args.benchmark_repeats):
                torch.cuda.synchronize()
                start = time.perf_counter()
                neighbors.rebuild(pos)
                torch.cuda.synchronize()
                neighbor_samples.append(time.perf_counter() - start)
        if args.benchmark_output is not None:
            torch.save(
                {
                    "energy": energy.detach().cpu(),
                    "forces": forces.detach().cpu(),
                    "virial": virial.detach().cpu(),
                },
                args.benchmark_output,
            )
        print(
            "BENCHMARK "
            + json.dumps(
                {
                    "atoms": int(z.numel()),
                    "system": str(args.system),
                    "precision": args.precision,
                    "potential_precision": potential_precision,
                    "energy_precision": args.energy_precision,
                    "compiled": args.compile_core,
                    "compiled_vs_eager_max_delta_efv": compiled_deltas,
                    "matmul_precision": torch.get_float32_matmul_precision(),
                    "gpu": torch.cuda.get_device_name(device),
                    "gpu_total_gib": torch.cuda.get_device_properties(device).total_memory
                    / 1024**3,
                    "torch": torch.__version__,
                    "cueq": _metadata.version("cuequivariance"),
                    "edge_capacity": neighbors.capacity,
                    "live_edges": n_live,
                    "neighbor_bin": args.neighbor_bin,
                    "warmup_seconds": cold_seconds,
                    "warmup_calls": args.benchmark_warmup,
                    "observable": "energy_forces_virial",
                    "seconds": samples,
                    "median_seconds": statistics.median(samples),
                    "model_peak_allocated_gib": model_peak,
                    "model_peak_reserved_gib": model_reserved,
                    "neighbor_seconds": neighbor_samples,
                    "neighbor_median_seconds": statistics.median(neighbor_samples),
                    "neighbor_peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                    "energy_ev": float(energy.detach().sum()),
                    "force_max_ev_per_a": float(forces.abs().max()),
                    "net_force_ev_per_a": forces.sum(0).tolist(),
                    "virial_ev": virial.tolist(),
                    "repeat_max_delta_efv": max_deltas,
                    "force_only_vs_virial_max_delta": force_route_delta,
                }
            ),
            flush=True,
        )
        return
    # Evaluation must inherit no-grad so force derivation uses
    # create_graph=False. Otherwise this one-off probe retains every backward
    # intermediate and can consume the entire GH200 before MD even starts.
    with torch.no_grad():
        single_point = Derivative(pots, wrt=("atoms.pos",))(work)
        e0, f0 = single_point["graphs", "energy"], single_point["atoms", "forces"]
    work["pairs"] = full_edges
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
    vol0 = float(torch.linalg.det(cell).abs())
    t0 = time.perf_counter()
    state = md.run(work, args.steps)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
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
