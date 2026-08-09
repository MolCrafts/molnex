"""MACE-MatPES perf guard: fused-vs-fallback cuEq kernels + compiled energy core.

The MACE family carries the repo's headline GH200 compile numbers, yet had no
benchmark of its own — which is exactly how a `use_fallback=True` default (a
measured **35.7x** per-step regression with zero correctness upside) shipped
unnoticed. This script is the guard:

  * eager ``energy_forces`` with ``use_fallback=False`` (fused) vs ``True``
    (pure-torch) — asserts the fused arm is at least ``--min-speedup`` faster
    when the ``cuequivariance-ops-torch`` wheel is importable;
  * the ``Compiler(cuda_graphs=True)``-compiled energy core (the
    ``run_nve.py`` production path) vs eager, on the fused arm.

Random weights (perf only, no physics); a periodic random-dense system builds
real ``(E, 2)`` edges + shifts through :class:`molix.md.PeriodicNeighborList`.
Run on a GPU node:

    python benchmarks/bench_mace_matpes.py            # defaults: N=192, fp32
    python benchmarks/bench_mace_matpes.py --fp64 --steps 30
"""

from __future__ import annotations

import argparse
import time

import torch

from molix import config
from molix.compile import Compiler
from molix.md import PeriodicNeighborList
from molpot.derivation.force import autograd_forces_from_energy

_Z_TABLE = [1, 6, 7, 8, 14, 26]  # small table; dims below are the MatPES-class ones


def _build_model(use_fallback: bool):
    from molzoo import MACEMatpes

    torch.manual_seed(0)
    return MACEMatpes(
        atomic_numbers=_Z_TABLE,
        atomic_energies=torch.zeros(len(_Z_TABLE), dtype=config.ftype),
        r_max=6.0,
        num_bessel=10,
        num_polynomial_cutoff=5,
        l_max=3,
        num_features=128,
        max_hidden_l=1,
        num_interactions=2,
        correlation=3,
        mlp_dim=16,
        use_fallback=use_fallback,
    ).eval()


def _system(n_atoms: int, device: torch.device):
    torch.manual_seed(1)
    # ~0.045 atoms/A^3 — condensed-phase-ish density for a real edge count.
    box = float((n_atoms / 0.045) ** (1.0 / 3.0))
    pos = torch.rand(n_atoms, 3, dtype=config.ftype, device=device) * box
    cell = torch.eye(3, dtype=config.ftype, device=device) * box
    Z = _Z_TABLE[0] + torch.zeros(n_atoms, dtype=torch.long, device=device)
    Z[::3] = _Z_TABLE[2]
    Z[::5] = _Z_TABLE[3]
    neighbors = PeriodicNeighborList(cell=cell, cutoff=6.0, positions=pos)
    batch = torch.zeros(n_atoms, dtype=torch.long, device=device)
    return pos, Z, batch, neighbors


def _time(fn, steps: int, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / steps * 1e3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-atoms", type=int, default=192)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--fp64", action="store_true")
    ap.add_argument(
        "--min-speedup",
        type=float,
        default=3.0,
        help="required fused/fallback per-step ratio when the ops wheel is present",
    )
    args = ap.parse_args()

    config.set_precision("fp64" if args.fp64 else "fp32")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        import cuequivariance_ops_torch  # noqa: F401

        fused_available = True
    except ImportError:
        fused_available = False
    print(f"device={device} dtype={config.ftype} fused_ops_available={fused_available}")

    pos, Z, batch, neighbors = _system(args.n_atoms, device)
    print(f"system: N={args.n_atoms} E={neighbors.num_edges} (capacity {neighbors.capacity})")

    times: dict[str, float] = {}
    for label, use_fallback in (("fused", False), ("fallback", True)):
        model = _build_model(use_fallback).to(device)

        def step(model=model):
            out = model.energy_forces(
                pos, Z, neighbors.edge_index, batch, num_graphs=1, shifts=neighbors.shifts
            )
            return out["forces"]

        times[label] = _time(step, args.steps)
        print(f"eager energy+forces [{label:8s}]: {times[label]:9.3f} ms/step")

    ratio = times["fallback"] / times["fused"]
    print(f"fallback/fused ratio: {ratio:.2f}x (guard: >= {args.min_speedup} when fused available)")

    # Compiled energy core on the fused arm — the run_nve.py production path.
    if device.type == "cuda":
        model = _build_model(False).to(device)
        energy_fn = Compiler(cuda_graphs=True)(model.energy_core)

        def compiled_step():
            leaf = pos.detach().requires_grad_(True)
            with torch.enable_grad():
                energy = energy_fn(leaf, Z, neighbors.edge_index, batch, 1, neighbors.shifts)
            return autograd_forces_from_energy(energy, leaf)

        t = _time(compiled_step, args.steps)
        print(f"compiled energy core [fused  ]: {t:9.3f} ms/step ({times['fused'] / t:.2f}x eager)")

    ok = (not fused_available) or ratio >= args.min_speedup
    if not ok:
        print(
            "RESULT: FAIL — fused kernels available but the speedup collapsed; "
            "check use_fallback plumbing / the ops wheel / cuEq versions"
        )
    else:
        print("RESULT: PASS")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
