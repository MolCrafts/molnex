"""PiNet force path: eager vs torch.compile + peak GPU memory.

Regimes:

* energy-only (baseline compile surface)
* force inf / force train — method=func and method=grad
* eager vs compile(fullgraph=False) vs compile(fullgraph=True)
* optional reduce-overhead (CUDA graphs; needs static shapes — same MockBatch size)

Reports ms/step and peak allocated / reserved MiB (torch.cuda memory stats).

Usage (GPU node)::

    PYTHONPATH=src python benchmarks/bench_pinet_force_compile.py
"""

from __future__ import annotations

import argparse
import gc
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

_orig_load_library = torch.ops.load_library


def _soft_load_library(path):  # noqa: ANN001
    try:
        return _orig_load_library(path)
    except OSError as exc:
        print(f"[bench] skip native op library ({path}): {exc}")


torch.ops.load_library = _soft_load_library  # type: ignore[method-assign]

from molix.profiler import MockBatch  # noqa: E402
from molzoo.pinet import PiNetPotential  # noqa: E402

ATOM_TYPES = list(range(1, 8))


@dataclass
class Row:
    name: str
    ms: float
    peak_alloc_mib: float
    peak_reserved_mib: float
    status: str = "ok"
    notes: str = ""


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _reset_mem() -> None:
    if not torch.cuda.is_available():
        return
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()


def _peak_mib() -> tuple[float, float]:
    if not torch.cuda.is_available():
        return float("nan"), float("nan")
    alloc = torch.cuda.max_memory_allocated() / (1024**2)
    reserved = torch.cuda.max_memory_reserved() / (1024**2)
    return alloc, reserved


def build(
    *,
    compute_forces: bool,
    method: str = "func",
    rank: int = 3,
    depth: int = 3,
    hidden: int = 64,
) -> PiNetPotential:
    return PiNetPotential(
        atom_types=ATOM_TYPES,
        r_max=5.0,
        n_basis=5,
        pp_nodes=[hidden, hidden],
        pi_nodes=[hidden, hidden],
        ii_nodes=[hidden, hidden],
        depth=depth,
        rank=rank,
        hidden_dim=hidden,
        compute_forces=compute_forces,
        method=method,  # type: ignore[arg-type]
        emit_property_features=False,
    )


def factory(device: str, n_atoms: int, n_edges: int, n_graphs: int) -> Callable[[], object]:
    return MockBatch(
        n_atoms=n_atoms,
        n_edges=n_edges,
        n_graphs=n_graphs,
        atomic_numbers=7,
        device=device,
        seed=0,
    )


def _ef_loss(batch) -> torch.Tensor:
    return (
        batch["graphs", "energy"].float().pow(2).mean()
        + batch["atoms", "forces"].float().pow(2).mean()
    )


def _e_loss(batch) -> torch.Tensor:
    return batch["graphs", "energy"].float().pow(2).mean()


def time_and_mem(
    step: Callable[[], None],
    *,
    n_warmup: int,
    n_steps: int,
) -> tuple[float, float, float]:
    for _ in range(n_warmup):
        step()
    _sync()
    _reset_mem()
    # one step after reset so peak includes steady-state activations
    step()
    _sync()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        step()
    _sync()
    ms = (time.perf_counter() - t0) * 1000.0 / n_steps
    peak_a, peak_r = _peak_mib()
    return ms, peak_a, peak_r


def run_cell(
    name: str,
    *,
    compute_forces: bool,
    method: str,
    train: bool,
    compile_mode: str | None,
    fac: Callable[[], object],
    device: torch.device,
    n_warmup: int,
    n_steps: int,
) -> Row:
    """compile_mode: None | 'default' | 'fullgraph' | 'reduce-overhead'."""
    torch.compiler.reset()
    try:
        pot = build(compute_forces=compute_forces, method=method).to(device)
        pot.train(train)

        notes = f"method={method}"
        if compile_mode is not None:
            kw: dict = {"backend": "inductor"}
            if compile_mode == "fullgraph":
                kw.update(fullgraph=True, dynamic=False)
                notes += " fullgraph=True"
            elif compile_mode == "reduce-overhead":
                kw.update(fullgraph=True, dynamic=False, mode="reduce-overhead")
                notes += " reduce-overhead"
            elif compile_mode == "default":
                kw.update(fullgraph=False)
                notes += " fullgraph=False"
            else:
                raise ValueError(compile_mode)
            pot = torch.compile(pot, **kw)  # type: ignore[assignment]
            # warm compile
            for _ in range(3):
                if train:
                    out = pot(fac())
                    if compute_forces:
                        _ef_loss(out).backward()
                    else:
                        _e_loss(out).backward()
                    pot.zero_grad(set_to_none=True)
                else:
                    with torch.enable_grad() if compute_forces else torch.no_grad():
                        pot(fac())
            _sync()

        opt = torch.optim.Adam(pot.parameters(), lr=1e-3) if train else None

        def step():
            if train:
                assert opt is not None
                opt.zero_grad(set_to_none=True)
                out = pot(fac())
                loss = _ef_loss(out) if compute_forces else _e_loss(out)
                loss.backward()
                opt.step()
            else:
                if compute_forces:
                    with torch.enable_grad():
                        pot(fac())
                else:
                    with torch.no_grad():
                        pot(fac())

        ms, pa, pr = time_and_mem(step, n_warmup=n_warmup, n_steps=n_steps)
        return Row(name=name, ms=ms, peak_alloc_mib=pa, peak_reserved_mib=pr, notes=notes)
    except Exception as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {str(e).splitlines()[0][:80]}"
        return Row(
            name=name,
            ms=float("nan"),
            peak_alloc_mib=float("nan"),
            peak_reserved_mib=float("nan"),
            status="FAIL",
            notes=msg,
        )
    finally:
        torch.compiler.reset()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def print_table(rows: list[Row]) -> None:
    print(f"\n{'name':<48} {'ms/step':>9} {'peak_alloc':>12} {'peak_rsrv':>10}  status  notes")
    print("-" * 110)
    for r in rows:
        ms = f"{r.ms:9.3f}" if r.ms == r.ms else f"{'nan':>9}"
        pa = (
            f"{r.peak_alloc_mib:10.1f} MiB"
            if r.peak_alloc_mib == r.peak_alloc_mib
            else f"{'nan':>12}"
        )
        pr = (
            f"{r.peak_reserved_mib:8.1f} MiB"
            if r.peak_reserved_mib == r.peak_reserved_mib
            else f"{'nan':>10}"
        )
        print(f"{r.name:<48} {ms} {pa:>12} {pr:>10}  {r.status:<5}  {r.notes}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=12)
    ap.add_argument("--atoms", type=int, default=512)
    ap.add_argument("--edges", type=int, default=4096)
    ap.add_argument("--graphs", type=int, default=16)
    ap.add_argument(
        "--skip-reduce-overhead",
        action="store_true",
        help="Skip CUDA-graph reduce-overhead cells (slow first compile)",
    )
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")

    device = torch.device(args.device)
    print(f"torch {torch.__version__} device={device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"total VRAM: {props.total_memory / (1024**3):.1f} GiB")

    fac = factory(str(device), args.atoms, args.edges, args.graphs)
    print(
        f"batch atoms={args.atoms} edges={args.edges} graphs={args.graphs} "
        f"steps={args.steps} warmup={args.warmup}"
    )

    rows: list[Row] = []

    # --- energy only --------------------------------------------------------
    for cmode, tag in (
        (None, "energy-inf eager"),
        ("default", "energy-inf compile"),
        ("fullgraph", "energy-inf compile fullgraph"),
    ):
        rows.append(
            run_cell(
                tag,
                compute_forces=False,
                method="func",
                train=False,
                compile_mode=cmode,
                fac=fac,
                device=device,
                n_warmup=args.warmup,
                n_steps=args.steps,
            )
        )

    # --- force inference ----------------------------------------------------
    for method in ("func", "grad"):
        for cmode, tag in (
            (None, f"force-inf {method} eager"),
            ("default", f"force-inf {method} compile"),
            ("fullgraph", f"force-inf {method} fullgraph"),
        ):
            rows.append(
                run_cell(
                    tag,
                    compute_forces=True,
                    method=method,
                    train=False,
                    compile_mode=cmode,
                    fac=fac,
                    device=device,
                    n_warmup=args.warmup,
                    n_steps=args.steps,
                )
            )

    # --- force train --------------------------------------------------------
    for method in ("func", "grad"):
        for cmode, tag in (
            (None, f"force-train {method} eager"),
            ("default", f"force-train {method} compile"),
            ("fullgraph", f"force-train {method} fullgraph"),
        ):
            rows.append(
                run_cell(
                    tag,
                    compute_forces=True,
                    method=method,
                    train=True,
                    compile_mode=cmode,
                    fac=fac,
                    device=device,
                    n_warmup=args.warmup,
                    n_steps=args.steps,
                )
            )

    if not args.skip_reduce_overhead:
        for method, train, tag in (
            ("func", False, "force-inf func reduce-overhead"),
            ("func", True, "force-train func reduce-overhead"),
            ("grad", True, "force-train grad reduce-overhead"),
        ):
            rows.append(
                run_cell(
                    tag,
                    compute_forces=True,
                    method=method,
                    train=train,
                    compile_mode="reduce-overhead",
                    fac=fac,
                    device=device,
                    n_warmup=max(args.warmup, 15),
                    n_steps=args.steps,
                )
            )

    print_table(rows)

    # quick ratios vs eager force-inf func
    by = {r.name: r for r in rows if r.status == "ok" and r.ms == r.ms}
    base = by.get("force-inf func eager")
    if base is not None:
        print("\n# force-inf func vs eager")
        for k in (
            "force-inf func compile",
            "force-inf func fullgraph",
            "force-inf func reduce-overhead",
        ):
            if k in by:
                print(
                    f"  {k}: {by[k].ms / base.ms:.2f}x time, "
                    f"alloc {by[k].peak_alloc_mib / base.peak_alloc_mib:.2f}x"
                )

    base_tr = by.get("force-train func eager")
    if base_tr is not None:
        print("\n# force-train func vs eager")
        for k in (
            "force-train func compile",
            "force-train func fullgraph",
            "force-train func reduce-overhead",
        ):
            if k in by:
                print(
                    f"  {k}: {by[k].ms / base_tr.ms:.2f}x time, "
                    f"alloc {by[k].peak_alloc_mib / base_tr.peak_alloc_mib:.2f}x"
                )

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
