#!/usr/bin/env python3
"""Compare molnex classical MD vs dump→LAMMPS-only MD (accuracy + performance).

Pipeline:

1. Build a synthetic OPLS-shaped Class-I ForceField + butane-like topology.
2. Single-point E/F in molnex.
3. ``ForceField.to_ir`` → ``LammpsAdapter`` → ``LammpsWriter`` data/settings +
   a script-local NVE deck.
4. ``lmp -in in.nve`` (or path from ``--lmp``).
5. Compare PE at step 0 and (optionally) short trajectory dump vs molnex NVE.
6. Report ms/step both sides.

Example::

    PYTHONPATH=src python scripts/mm_param_learning/parity_lammps_classical_md.py
    PYTHONPATH=src python scripts/mm_param_learning/parity_lammps_classical_md.py \\
        --lmp /home/jicli594/.local/x86_64/bin/lmp --steps 200
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
from tensordict import TensorDict

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from molix.ff_export import LammpsWriter  # noqa: E402
from molix.md import (  # noqa: E402
    MD,
    CallableForceField,
    MDHook,
    MDObservables,
    MDRunner,
    VelocityVerletIntegrator,
)
from molix.units import units  # noqa: E402

_KCAL_PER_MD = (
    (1.0 * units.kilocalorie_per_mole).to("amu * angstrom ** 2 / femtosecond ** 2").magnitude
)
from molpot.forcefield import ForceField, OPLSForceField  # noqa: E402


class _PosEnergyHook(MDHook):
    def __init__(self) -> None:
        self.pos: list[torch.Tensor] = []
        self.pe: list[float] = []
        self.etot: list[float] = []

    def on_step_end(self, runner: MDRunner, step: int, obs: MDObservables) -> None:
        self.pos.append(obs.pos.detach().cpu().clone())
        self.pe.append(float(obs.potential.detach().cpu()))
        self.etot.append(float(obs.total.detach().cpu()))


def butane_batch(dtype: torch.dtype = torch.float64) -> TensorDict:
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.53, 0.0, 0.0],
            [2.05, 1.43, 0.0],
            [3.58, 1.43, 0.0],
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
                    "q": torch.zeros(4, dtype=dtype),
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
            "angles": TensorDict(
                {
                    "atomi": torch.tensor([0, 1], dtype=torch.long),
                    "atomj": torch.tensor([1, 2], dtype=torch.long),
                    "atomk": torch.tensor([2, 3], dtype=torch.long),
                },
                batch_size=[2],
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


def synthetic_opls_ff(dtype: torch.dtype = torch.float64) -> ForceField:
    return OPLSForceField().with_params(
        bonds={
            "k": torch.tensor([200.0, 200.0, 200.0], dtype=dtype),
            "r0": torch.tensor([1.53, 1.53, 1.53], dtype=dtype),
        },
        angles={
            "k": torch.tensor([100.0, 100.0], dtype=dtype),
            "theta0": torch.tensor([math.radians(112.0), math.radians(112.0)], dtype=dtype),
        },
        propers={
            "k1": torch.tensor([1.5], dtype=dtype),
            "k2": torch.tensor([3.0], dtype=dtype),
            "k3": torch.tensor([0.5], dtype=dtype),
            "k4": torch.tensor([0.1], dtype=dtype),
        },
    )


def bind_md_force(ff: ForceField, batch: TensorDict) -> CallableForceField:
    pots = ff.to_potentials()
    template = batch.clone()

    def _fn(pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        template["atoms", "pos"] = pos
        return pots.calc_energy_forces(template, pos=pos)

    return CallableForceField(_fn, energy_scale=1.0 / _KCAL_PER_MD)


def run_molnex_nve(
    ff: ForceField,
    batch: TensorDict,
    *,
    vel: torch.Tensor,
    mass: torch.Tensor,
    dt: float,
    n_steps: int,
    dtype: torch.dtype,
) -> tuple[_PosEnergyHook, float]:
    force = bind_md_force(ff, batch)
    hook = _PosEnergyHook()
    md = (
        MD(dtype)
        .set_forcefield(force)
        .set_integrator(VelocityVerletIntegrator(dt))
        .set_hooks([hook])
    )
    batch["atoms", "pos"] = batch["atoms", "pos"].to(dtype).clone()
    batch["atoms", "vel"] = vel.to(dtype)
    batch["atoms", "mass"] = mass.to(dtype)
    t0 = time.perf_counter()
    md.run(batch, n_steps)
    return hook, time.perf_counter() - t0


def write_lammps_job(
    out_dir: Path,
    ff: ForceField,
    batch: TensorDict,
    *,
    masses: torch.Tensor,
    vel: torch.Tensor,
    dt: float,
    n_steps: int,
) -> Path:
    writer = LammpsWriter()
    writer.write_data(out_dir / "system.data", batch, masses=masses)
    writer.write_settings(out_dir / "system.in.settings", ff, batch, masses=masses)
    has_pair = bool(ff.pair_styles())
    thermo_every = max(1, n_steps // 10)
    dump_every = max(1, n_steps // 10)
    in_path = out_dir / "in.nve"
    pair_block = (
        "pair_style lj/cut 10.0\n" if has_pair else "pair_style zero 10.0\npair_coeff * *\n"
    )
    special = "special_bonds lj/coul 0.0 0.0 0.5\n" if has_pair else ""
    text = f"""# NVE deck (script-local; not part of LammpsWriter)
units real
atom_style full
boundary f f f
newton on

read_data system.data
include system.in.settings
{pair_block}{special}
neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes

velocity all set 0.0 0.0 0.0 units box

fix 1 all nve
timestep {dt}

thermo_style custom step pe ke etotal temp
thermo {thermo_every}
dump 1 all custom {dump_every} dump.lammpstrj id type x y z fx fy fz
dump_modify 1 sort id

run {n_steps}
"""
    # Per-atom velocities (Å/fs, units real / box)
    vel_cmds = ["# explicit initial velocities (Å/fs)"]
    for i, v in enumerate(vel.detach().cpu().tolist(), start=1):
        vel_cmds.append(f"group a{i} id {i}")
        vel_cmds.append(f"velocity a{i} set {v[0]:.16g} {v[1]:.16g} {v[2]:.16g} units box")
    text = re.sub(
        r"velocity all set 0\.0 0\.0 0\.0 units box\n",
        "\n".join(vel_cmds) + "\n",
        text,
        count=1,
    )
    # run-0 input for single-point PE / forces
    dump_every = max(1, n_steps // 10)
    run0 = out_dir / "in.run0"
    run0.write_text(
        text.replace(f"run {n_steps}", "run 0").replace(
            f"dump 1 all custom {dump_every}", "dump 1 all custom 1"
        )
    )
    in_path.write_text(text)
    return in_path


def run_lammps(lmp: str, workdir: Path, input_name: str) -> tuple[str, float]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")
    t0 = time.perf_counter()
    proc = subprocess.run(
        [lmp, "-in", input_name, "-log", "log.lammps", "-screen", "none"],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - t0
    log = (workdir / "log.lammps").read_text() if (workdir / "log.lammps").exists() else ""
    if proc.returncode != 0:
        raise RuntimeError(
            f"lmp failed ({proc.returncode}):\n{proc.stderr[-2000:]}\nlog tail:\n{log[-2000:]}"
        )
    return log, elapsed


def parse_thermo_pe(log: str) -> list[tuple[int, float]]:
    """Parse thermo rows: step pe ke etotal temp."""
    rows: list[tuple[int, float]] = []
    for line in log.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            step = int(parts[0])
            pe = float(parts[1])
        except ValueError:
            continue
        # skip header-like
        if step < 0:
            continue
        rows.append((step, pe))
    return rows


def parse_dump_forces(dump_path: Path) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return (pos, forces) from the first frame of a custom dump."""
    if not dump_path.exists():
        return None
    text = dump_path.read_text().splitlines()
    # find first ITEM: ATOMS
    i = 0
    while i < len(text) and not text[i].startswith("ITEM: ATOMS"):
        i += 1
    if i >= len(text):
        return None
    header = text[i].split()[2:]  # columns after ATOMS
    i += 1
    rows = []
    while i < len(text) and not text[i].startswith("ITEM:"):
        rows.append(text[i].split())
        i += 1
    if not rows:
        return None
    col = {name: k for k, name in enumerate(header)}
    # sort by id
    rows.sort(key=lambda r: int(r[col["id"]]))
    pos = torch.tensor(
        [[float(r[col["x"]]), float(r[col["y"]]), float(r[col["z"]])] for r in rows],
        dtype=torch.float64,
    )
    forces = torch.tensor(
        [[float(r[col["fx"]]), float(r[col["fy"]]), float(r[col["fz"]])] for r in rows],
        dtype=torch.float64,
    )
    return pos, forces


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument(
        "--lmp",
        default=os.environ.get("LMP", shutil.which("lmp") or "lmp"),
        help="path to lmp binary",
    )
    ap.add_argument("--workdir", type=Path, default=None)
    ap.add_argument("--keep", action="store_true", help="keep workdir")
    args = ap.parse_args()

    dtype = torch.float64
    torch.manual_seed(0)
    batch = butane_batch(dtype)
    ff = synthetic_opls_ff(dtype)
    pots = ff.to_potentials()

    e0 = float(pots.calc_energy(batch).detach())
    f0 = pots.calc_forces(batch).detach()
    print("=== molnex single-point (kcal/mol, kcal/mol/Å) ===")
    print(f"  E = {e0:.10f}")
    print(f"  |F|_max = {float(f0.abs().max()):.6e}")

    n = batch["atoms", "pos"].shape[0]
    mass = torch.full((n,), 12.011, dtype=dtype)
    vel = 1e-4 * torch.randn(n, 3, dtype=dtype)
    vel -= vel.mean(dim=0, keepdim=True)

    tmp_ctx = (
        tempfile.TemporaryDirectory(prefix="molnex_lammps_parity_")
        if args.workdir is None
        else None
    )
    workdir = Path(tmp_ctx.name) if tmp_ctx is not None else Path(args.workdir)  # type: ignore[arg-type]
    workdir.mkdir(parents=True, exist_ok=True)
    print(f"\nworkdir: {workdir}")

    try:
        write_lammps_job(
            workdir,
            ff,
            batch,
            masses=mass,
            vel=vel,
            dt=args.dt,
            n_steps=args.steps,
        )

        # run 0 for PE / forces
        log0, _ = run_lammps(args.lmp, workdir, "in.run0")
        thermo0 = parse_thermo_pe(log0)
        pe_lmp0 = thermo0[0][1] if thermo0 else float("nan")
        dump_forces = parse_dump_forces(workdir / "dump.lammpstrj")
        print("\n=== LAMMPS run 0 ===")
        print(f"  PE = {pe_lmp0:.10f} kcal/mol")
        dpe = abs(pe_lmp0 - e0)
        print(f"  |ΔPE| vs molnex = {dpe:.3e}")
        df_max = float("nan")
        if dump_forces is not None:
            _, f_lmp = dump_forces
            df_max = float((f_lmp - f0.cpu()).abs().max())
            print(f"  max|ΔF| = {df_max:.3e}")

        # MD both sides
        hook, t_mx = run_molnex_nve(
            ff, batch, vel=vel, mass=mass, dt=args.dt, n_steps=args.steps, dtype=dtype
        )
        # remove previous dump
        dump = workdir / "dump.lammpstrj"
        if dump.exists():
            dump.unlink()
        log, t_lmp = run_lammps(args.lmp, workdir, "in.nve")
        thermo = parse_thermo_pe(log)

        print(f"\n=== NVE ({args.steps} steps, dt={args.dt} fs) ===")
        print(
            f"  molnex: {t_mx * 1e3:.2f} ms total, "
            f"{t_mx / args.steps * 1e3:.4f} ms/step, "
            f"{args.steps / t_mx:.0f} steps/s"
        )
        print(
            f"  lammps: {t_lmp * 1e3:.2f} ms total, "
            f"{t_lmp / args.steps * 1e3:.4f} ms/step, "
            f"{args.steps / t_lmp:.0f} steps/s"
        )
        if t_mx > 0 and t_lmp > 0:
            print(f"  speedup (molnex_ms / lammps_ms) = {t_mx / t_lmp:.2f}x")

        if thermo and hook.pe:
            # PE series: molnex pe is in MD units; convert back to kcal for compare
            scale = _KCAL_PER_MD
            pe_mx_kcal = [p * scale for p in hook.pe]
            # thermo has sparse samples; compare final PE if available
            pe_lmp_final = thermo[-1][1]
            pe_mx_final = pe_mx_kcal[-1]
            print(f"  final PE molnex  = {pe_mx_final:.6f} kcal/mol")
            print(f"  final PE lammps  = {pe_lmp_final:.6f} kcal/mol")
            print(f"  |ΔPE_final|      = {abs(pe_mx_final - pe_lmp_final):.3e}")

        summary = {
            "e0_molnex": e0,
            "e0_lammps": pe_lmp0,
            "dpe0": dpe,
            "df_max": df_max,
            "steps": args.steps,
            "dt_fs": args.dt,
            "ms_per_step_molnex": t_mx / args.steps * 1e3,
            "ms_per_step_lammps": t_lmp / args.steps * 1e3,
            "steps_per_s_molnex": args.steps / t_mx,
            "steps_per_s_lammps": args.steps / t_lmp,
        }
        (workdir / "parity_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"\nsummary → {workdir / 'parity_summary.json'}")

        ok = dpe < 1e-5 and (math.isnan(df_max) or df_max < 1e-4)
        print(f"\n{'PASS' if ok else 'FAIL'}: single-point gate dpe={dpe:.3e} df={df_max:.3e}")
        return 0 if ok else 1
    finally:
        if tmp_ctx is not None and not args.keep:
            tmp_ctx.cleanup()
        elif tmp_ctx is not None and args.keep:
            print(f"kept workdir at {workdir}")
            # prevent cleanup
            tmp_ctx._finalizer.detach()  # type: ignore[attr-defined]


if __name__ == "__main__":
    raise SystemExit(main())
