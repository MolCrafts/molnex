"""Freeze stage-1 force-constant targets from THEMol's QM Hessians.

ByteFF's first stage fits the harmonic valence terms to a *partial Hessian*
(spec ``src/molzoo/specs/rondo.md`` §5: staged partial-Hessian -> torsion
profiles -> off-equilibrium E/F). The target is one force constant per bond and
per angle, recovered from the record's QM Hessian by
:class:`molpot.forcefield.seminario.SeminarioForceConstants`. Under the spec's
convention ``E = sum 1/2 k (q - q0)^2`` (line 265), ``d2E/dq2 = k`` exactly, so
there is no factor of two anywhere in this path.

**Not the Wilson pseudo-inverse projection.** ``H_q = (B^+)^T H_x B^+`` is exact
for a non-redundant coordinate set, but real molecules are redundant — 38 bonds
and 59 angles on a 30-atom THEMol record is 97 coordinates for ``3N - 6 = 84``
degrees of freedom. Measured on this shard (2026-09-20): ``cond(B)`` about
``2e16``, and angle constants with a median of 1789 and a maximum of ``2e12``
kcal mol⁻¹ rad⁻² against a physical 35-160. Restricting to bends alone does not
help, and the smallest molecule in the first 400 records is still redundant by
one coordinate. Seminario's method is local — one 3x3 interaction block per term
— so redundancy cannot reach it; on 120 records it gives a bond median of 706.7
and an angle median of 146.5, with per-pair medians within 10-20 % of literature
Class-I values (C-H 717 against ~700, C-C 598 against ~620, N-H 982 against
~900).

**Why this is a precompute and not a step.** The geometry and the QM Hessian are
both fixed per record, so the force constants are a **label** — nothing the model
participates in. Precomputed, stage 1 is an MSE against a cached vector, the
``(3N)^2`` Hessian never enters a batch, and the batch schema needs no new
namespace: the labels ride as columns on the existing ``bonds`` and ``angles``.

Writes one npz per shard: concatenated ``bond_k`` / ``angle_k`` with ``offsets``
so a row's slice is ``[offsets[i]:offsets[i + 1]]``. Term order is the order
``ClassIPrepare.topology`` produces, which is the order the batch's ``bonds`` and
``angles`` rows carry — a label list in any other order is silently wrong.
"""

from __future__ import annotations

import hashlib
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from molhub.config import config as molhub_config
from molhub.dataset.themol import THEMolHessian

from molpot.forcefield.seminario import SeminarioForceConstants
from scripts.mm_param_learning.class_i_fit import ClassIPrepare

MOLNEX = Path(__file__).resolve().parents[2]
HOME = MOLNEX.parent / "workspaces" / "molhub-home"


def targets_path(source_id: str) -> Path:
    """Where this shard's frozen force-constant targets live."""
    digest = hashlib.sha1(f"{source_id}:class-i-hessian-targets".encode()).hexdigest()[:16]
    path = Path(molhub_config.home_dir) / "derived" / "class-i-hessian" / f"{digest}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


_RowTargets = tuple[int, list[np.ndarray], list[np.ndarray]]


def _row_targets(payload: tuple[object, int, int]) -> _RowTargets:
    """Worker: ``(bond_k, angle_k)`` per row of ``dataset[start:stop]``."""
    dataset, start, stop = payload
    prepare = ClassIPrepare()
    seminario = SeminarioForceConstants()
    bonds_out: list[np.ndarray] = []
    angles_out: list[np.ndarray] = []
    for idx in range(start, stop):
        sample = dataset[idx]
        frame = prepare.topology(sample[0])
        n = int(frame["atoms"].nrows)
        pos = torch.tensor(
            np.stack([np.asarray(frame["atoms"][c], dtype=np.float64) for c in ("x", "y", "z")], -1)
        )
        bond_index = torch.tensor(
            np.stack([np.asarray(frame["bonds"][c], dtype=np.int64) for c in ("atomi", "atomj")])
        )
        if "angles" in frame and frame["angles"].nrows:
            angle_index = torch.tensor(
                np.stack(
                    [
                        np.asarray(frame["angles"][c], dtype=np.int64)
                        for c in ("atomi", "atomj", "atomk")
                    ]
                )
            )
        else:
            angle_index = torch.zeros(3, 0, dtype=torch.long)
        hessian = torch.tensor(
            np.asarray(sample[0]["hessian"]["value"], dtype=np.float64).reshape(3 * n, 3 * n)
        )
        bonds_out.append(
            seminario.bond_constants(pos, hessian, bond_index).numpy().astype(np.float64)
        )
        angles_out.append(
            seminario.angle_constants(pos, hessian, angle_index).numpy().astype(np.float64)
        )
    return start, bonds_out, angles_out


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    role = sys.argv[2] if len(sys.argv) > 2 else "shard-000"
    molhub_config.home_dir = HOME
    dataset = THEMolHessian().load(role)
    n = len(dataset)
    out = targets_path(dataset.source_id)
    print(f"source={dataset.source_id} rows={n} workers={workers} -> {out}", flush=True)

    t0 = perf_counter()
    chunk = max(int(np.ceil(n / max(workers, 1))), 1)
    spans = [(s, min(s + chunk, n)) for s in range(0, n, chunk)]
    parts: dict[int, tuple[list[np.ndarray], list[np.ndarray]]] = {}
    if workers <= 1:
        start, b, a = _row_targets((dataset, 0, n))
        parts[start] = (b, a)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_row_targets, (dataset, s, e)) for s, e in spans]
            done = 0
            for future in futures:
                start, b, a = future.result()
                parts[start] = (b, a)
                done += len(b)
                print(f"hessian-targets {done}/{n}", flush=True)

    bond_rows: list[np.ndarray] = []
    angle_rows: list[np.ndarray] = []
    for start, _ in spans:
        b, a = parts[start]
        bond_rows.extend(b)
        angle_rows.extend(a)

    def ragged(rows: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        offsets = np.zeros(len(rows) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([r.size for r in rows])
        return (np.concatenate(rows) if rows else np.zeros(0)), offsets

    bond_k, bond_offsets = ragged(bond_rows)
    angle_k, angle_offsets = ragged(angle_rows)
    np.savez_compressed(
        out,
        _source_id=np.asarray(dataset.source_id),
        bond_k=bond_k,
        bond_offsets=bond_offsets,
        angle_k=angle_k,
        angle_offsets=angle_offsets,
    )
    dt = perf_counter() - t0
    print(
        f"wrote {out} in {dt:.1f}s — {len(bond_rows)} rows, "
        f"{bond_k.size} bond terms, {angle_k.size} angle terms",
        flush=True,
    )
    for name, arr in (("bond k", bond_k), ("angle k", angle_k)):
        if arr.size:
            print(
                f"  {name}: median {np.median(arr):9.1f} p1 {np.percentile(arr, 1):9.1f} "
                f"p99 {np.percentile(arr, 99):9.1f} negative {int((arr < 0).sum())}",
                flush=True,
            )


if __name__ == "__main__":
    main()
