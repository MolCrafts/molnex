#!/usr/bin/env python3
"""Audit NeighborList graphs vs an independent multi-image brute-force oracle.

Trajectory CLI (every frame independently rebuilt — no Verlet cache)::

    PYTHONPATH=src:. python scripts/matpes_port/neighbor_graph_audit.py \\
        path/to/traj.xyz --cutoff 6.0 --metrics-dir /tmp/run

Writes:
  * table to stdout
  * optional ``metrics/metrics.jsonl`` (molrec scalars for molexp molplot)
  * optional ``neighbor_compare.txt`` under --out

**LAMMPS / ML-IAP (optional, not implemented as invasive dump):** the production
``interface/`` pair_style feeds flat ``(Z, pos, edge_index)`` into the exported
model; molnex MD rebuilds via ``NeighborList``. A three-way
oracle vs molix vs LAMMPS-fed list needs a non-default debug dump at the
ML-IAP neighbor handoff — tracked as ac-011 (document seam; no production
behavior change in this suite).

Cutoff convention (production): ``0 < r <= cutoff``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# Allow running from repo root without install.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))
_TESTS = _REPO / "tests"
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from test_molix.test_md.oracle_bruteforce_neighbors import (  # noqa: E402
    bruteforce_edges,
    compare_graphs,
    neighborlist_edge_keys,
)

from molix.datasets._extxyz import parse_extxyz_frames  # noqa: E402
from molix.md import NeighborList  # noqa: E402


def _frame_cell(frame) -> torch.Tensor:
    cell = getattr(frame, "cell", None)
    if cell is None:
        # open: large orthorhombic box
        return torch.eye(3, dtype=torch.float64) * 100.0
    t = torch.as_tensor(cell, dtype=torch.float64)
    if t.shape == (3, 3):
        return t
    raise SystemExit(f"unexpected cell shape {tuple(t.shape)}")


def audit_frame(
    pos: torch.Tensor,
    cell: torch.Tensor,
    cutoff: float,
    *,
    bin: float | None = None,
):
    nl = NeighborList(
        cell=cell,
        cutoff=cutoff,
        positions=pos,
        skin=0.0,
        capacity_factor=2.0,
        bin=bin,
    )
    ref = bruteforce_edges(pos, cell=cell, cutoff=cutoff, pbc=(True, True, True))
    sut = neighborlist_edge_keys(nl.edge_index, nl.shifts, nl.num_edges, cell)
    return compare_graphs(ref, sut), nl


def _append_metrics(metrics_dir: Path, frame: int, cmp) -> None:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    path = metrics_dir / "metrics.jsonl"
    rows = [
        {"t": "scalar", "k": "n_ref", "s": frame, "v": cmp.n_ref},
        {"t": "scalar", "k": "n_mace", "s": frame, "v": cmp.n_sut},
        {"t": "scalar", "k": "missing", "s": frame, "v": len(cmp.missing)},
        {"t": "scalar", "k": "extra", "s": frame, "v": len(cmp.extra)},
        {"t": "scalar", "k": "max_dr_mismatch", "s": frame, "v": cmp.max_dr_mismatch},
    ]
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trajectory", type=Path, help="multi-frame extxyz")
    ap.add_argument("--cutoff", type=float, required=True, help="interaction cutoff (A)")
    ap.add_argument("--bin", type=float, default=None, help="NeighborList bin= thickness")
    ap.add_argument(
        "--metrics-dir",
        type=Path,
        default=None,
        help="directory for metrics/metrics.jsonl (molplot)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="directory for neighbor_compare.txt",
    )
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all")
    args = ap.parse_args(argv)

    frames = parse_extxyz_frames(args.trajectory)
    if args.max_frames > 0:
        frames = frames[: args.max_frames]

    metrics_root = None
    if args.metrics_dir is not None:
        metrics_root = args.metrics_dir / "metrics"

    lines = ["frame  n_ref  n_mace  missing  extra"]
    any_bad = False
    dumps: list[str] = []

    print("neighbor-graph audit  cutoff_convention='0 < r <= r_c'  (production filter)")
    print(f"trajectory={args.trajectory}  frames={len(frames)}  cutoff={args.cutoff}")

    for fi, fr in enumerate(frames):
        pos = torch.as_tensor(fr.pos, dtype=torch.float64)
        cell = _frame_cell(fr)
        try:
            cmp, _nl = audit_frame(pos, cell, args.cutoff, bin=args.bin)
        except Exception as exc:  # noqa: BLE001
            print(f"{fi:5d}  ERROR {type(exc).__name__}: {exc}")
            any_bad = True
            continue
        lines.append(
            f"{fi:5d}  {cmp.n_ref:5d}  {cmp.n_sut:6d}  {len(cmp.missing):7d}  {len(cmp.extra):5d}"
        )
        print(lines[-1])
        if metrics_root is not None:
            _append_metrics(metrics_root, fi, cmp)
        if not cmp.ok:
            any_bad = True
            for kind, bag in (("missing", cmp.missing), ("extra", cmp.extra)):
                for i, j, sx, sy, sz in sorted(bag)[:20]:
                    shift = float(sx) * cell[0] + float(sy) * cell[1] + float(sz) * cell[2]
                    dr = pos[j] - pos[i] + shift
                    dumps.append(
                        f"frame={fi} {kind} i={i} j={j} S=({sx},{sy},{sz}) "
                        f"|dr|={float(dr.norm()):.6f}"
                    )

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "neighbor_compare.txt").write_text(
            "\n".join(lines) + "\n" + ("\n".join(dumps) + "\n" if dumps else ""),
            encoding="utf-8",
        )
        print(f"wrote {args.out / 'neighbor_compare.txt'}")
    if dumps:
        print("--- mismatch detail (truncated) ---")
        print("\n".join(dumps[:50]))

    # ac-011 seam note
    print(
        "LAMMPS/ML-IAP: no invasive dump in this tool; "
        "molnex MD uses NeighborList; pair_style path is interface/ (ac-011)."
    )
    return 1 if any_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
