---
title: MACE neighbor-graph correctness vs independent brute-force oracle
status: code-complete
slug: mace-neighbor-graph-correctness
created: 2026-08-10
revised: 2026-08-10
grilled: true
knowledge_home: mace-nve/projects/mace-r2san (Note + experiment neighbor-graph-correctness)
supersede: grill-2026-08-10 (cutoff ≤; E/F slow gate; no MIC oracle reuse; no O3 equivariance)
---

# MACE neighbor-graph correctness vs independent brute-force oracle

## Summary

Implement a **self-contained neighbor-graph correctness suite** for debugging NVE energy drift: compare an **independent brute-force O(N²) oracle** against the neighbor graph that **molnex MACE actually consumes** (`molix.md.NeighborList` → `edge_index` `(E,2)` + integer/image `shifts`), with optional instrumentation of the **LAMMPS/ML-IAP → molnex** path when available without invasive production changes.

This is **not** a re-test of matscipy/ASE/LAMMPS against themselves. The oracle enumerates atoms × periodic images and returns directed edges `(i, j, sx, sy, sz)` with **`0 < |dr| ≤ cutoff`** (matching production filter parity).

**Plotting / observability:** diagnostics are written as **molrec-compatible `metrics/metrics.jsonl`** (keys `n_ref`, `n_mace`, `missing`, `extra`, `max_dr_mismatch`) so the **molexp molplot plugin** can render curves from the run; human tables go to `artifacts/*.txt`. No ad-hoc matplotlib as the primary chart path.

**Primary knowledge home:** design also lives as a molexp **Note** under the `mace-nve` workspace project `mace-r2san`, experiment `neighbor-graph-correctness`.

## Domain basis

- Edge convention (molnex hard rule): `edge_index[:,0]=source`, `edge_index[:,1]=target`,
  `edge_diff = pos[target] - pos[source] (+ shift for PBC)`,
  `dr = pos[j] - pos[i] + S @ cell` for integer image vector `S=(sx,sy,sz)`.
- Exclude only the true self-edge `i==j && S==(0,0,0)`. Periodic self-images inside cutoff **remain valid**.
- **Cutoff convention (pinned to code):** production pair filter is **`0 < r ≤ r_c`** (equivalently `distance2 > cutoff2 || distance2 == 0` drops; **`r == r_c` is kept**). Documented in `neighbors.py` filter parity and `get_neighbor_pairs.cu`. Oracle and tests **must** use the same inequality.
- **Multiple images:** when any box edge ≲ `2 r_c`, minimum-image alone is insufficient; the oracle uses an image range large enough that all images with `0 < |dr| ≤ r_c` are found.
- Verlet skin / every / delay affect **when** the list rebuilds in MD, not completeness of a **fresh** build. Fresh-build tests use construction / `rebuild`; cutoff-crossing evaluates **each frame independently** (no stale list).
- **PBC wrap / lattice reparametrization:** same physical configuration → **E invariant**, **F numerically the same** on corresponding atoms (invariance under reparametrization — **not** SO(3) equivariance).
- **O(3) equivariance** (E invariant + F transforms as vectors under rotation) is **out of scope** for this suite.
- Implementation target: **molnex `NeighborList` + edges fed to MACE**. Not `mace.data.get_neighborhood` / mace-torch (forbidden under `src/` / `tests/`). In-tree there is **no matscipy** neighbor path.

## Design

### Discovered neighbor surface (molnex)

| Layer | Role |
|-------|------|
| `molix.md.NeighborList` | Stateful fixed-capacity list; `rebuild`/`update`/`build(td)`; skin/every/delay/check; `edge_index` `(capacity,2)` + `shifts` `(capacity,3)` Å |
| `get_neighbor_pairs` (C++/CUDA) | Pair search kernel; filter `distance2 > cutoff2 \|\| distance2 == 0` |
| `NeighborList._build_binned` | Pure-torch cell-list (`bin=`); same `0 < r ≤ r_build` filter |
| `molzoo.mace.MACEPotential.energy_core` | Consumes caller `edge_index` + `shifts` — **does not** build neighbors |
| MD bind | Force fields call `neighbors.update` then read live buffers |

The MACE energy graph is only as correct as the **list supplied to it**; this suite validates that list.

### Graph representation & comparison

```text
EdgeKey = (i, j, sx, sy, sz)   # ints; S @ cell → Cartesian
dr      = pos[j] - pos[i] + (S @ cell)
```

- Live edges: `[0, num_edges)`. Convert continuous `shifts` (Å) to integer `S` via `S ≈ shifts @ inv(cell)`, nearest-int; fail if residual large.
- Compare sets: missing = ref \ sut, extra = sut \ ref. **Hard-fail** if either non-empty.
- For matching keys report max ‖dr_ref − dr_sut‖.

### Brute-force oracle (independent)

Dedicated test-only module (e.g. `tests/test_molix/test_md/oracle_bruteforce_neighbors.py`):

- Inputs: `pos (N,3)`, `cell (3,3)|None`, `cutoff`, `pbc (3,) bool`.
- Image range from perpendicular cell widths vs cutoff (not MIC-only).
- Emit directed pairs with **`0 < |dr| ≤ cutoff`**, exclude true self only.
- **Forbidden imports:** `molix.md.neighbors`, `molix.op` / `get_neighbor_pairs`, matscipy, ASE, freud, LAMMPS.
- **Must not** reuse `test_neighbors._reference_pairs` (that helper is **MIC-only** and would fail multi-image cases).

### SUT coverage

- **Default:** production `NeighborList(...)` construction / `rebuild` (whatever backend the environment selects).
- **Light dual-backend:** parametrize **at least one** synthetic case with `bin=` vs non-bin/kernel path when both are available — not a full 2× matrix of every test.

### Required synthetic tests (pytest)

1. **Open random** — N∈[10,30], no PBC: missing=extra=0 (**unit**).
2. **Cutoff boundary** — `r_c±1e-3`, `r_c±1e-6`; **`r=r_c` has edge** (because ≤). Failures must state the `≤` convention (**unit**).
3. **PBC wrap** — cubic L=10 Å, x=0.1 vs 9.9, r_c=1 Å → |dr|≈0.2 Å; correct `S` (**unit**).
4. **Translation / wrap**
   - **Graph (unit):** physical `dr` multisets equal across wrap / lattice translate / whole-structure lattice translate.
   - **E/F (slow):** same representations → MACE `energy_core` E and forces agree within fixed tol when list is **rebuilt** each time; mark `@pytest.mark.slow` and **skip if MatPES weights unavailable**.
5. **Triclinic** — small tilted cell (**unit**).
6. **Small cell / multi-image** — box edge ≲ 2 r_c; multi-image oracle (**unit**).
7. **Cutoff-crossing frames** — r>r_c → r<r_c → r>r_c; **per-frame rebuild**; no hysteresis (**unit**).
8. **Trajectory CLI** — multi-frame file; table + mismatch dump + **metrics.jsonl** for molplot.

### Diagnostics & molplot contract

Per frame / comparison, append to run `metrics/metrics.jsonl`:

```json
{"t":"scalar","k":"n_ref","s":<frame>,"v":...}
{"t":"scalar","k":"n_mace","s":<frame>,"v":...}
{"t":"scalar","k":"missing","s":<frame>,"v":...}
{"t":"scalar","k":"extra","s":<frame>,"v":...}
{"t":"scalar","k":"max_dr_mismatch","s":<frame>,"v":...}
```

Human table: `artifacts/neighbor_compare.txt`.
molexp molplot reads **metrics.jsonl** (SoT), not PNG.

### Optional LAMMPS / ML-IAP

Non-invasive dump of neighbor payload into molnex if feasible; else document the seam under the knowledge Note and mark ac-011 done-with-doc. No production behavior change by default.

### Reuse decision

| Candidate | Decision |
|-----------|----------|
| `NeighborList` + MD tests | **reuse** as SUT |
| Skin/ndanger tests | **pattern** only (policy ≠ fresh-build geometry) |
| `_reference_pairs` in `test_neighbors.py` | **do not reuse** as multi-image oracle |
| New multi-image brute force | **new** under tests |
| matscipy / freud / ASE / mace-torch | **forbid** |

## Files

- `tests/test_molix/test_md/oracle_bruteforce_neighbors.py` — independent multi-image oracle + compare helpers
- `tests/test_molix/test_md/test_neighbor_graph_oracle.py` — synthetic cases 1–7 (E/F under slow)
- `scripts/matpes_port/neighbor_graph_audit.py` — trajectory CLI + metrics.jsonl (+ optional experiment wiring notes)
- Knowledge: `mace-nve/projects/mace-r2san/mace-neighbor-graph-correctness/`
- Experiment: `…/experiments/neighbor-graph-correctness/`

## Tasks

- [x] Pin cutoff comments: production filter is `0 < r ≤ r_c`; note MACE does not build its own list
- [x] Implement independent multi-image brute-force oracle + integer-shift key helpers (no MIC-only reuse)
- [x] Pytest unit: open, boundary (≤), PBC wrap, graph wrap-invariance, triclinic, multi-image, cutoff-crossing
- [x] Pytest slow (or skip-no-weights): E/F invariance under PBC-equivalent reparametrization
- [x] One dual-backend smoke (`bin=` vs default) on at least one case
- [x] CLI trajectory audit + `neighbor_compare.txt` + metrics.jsonl
- [x] Optional LAMMPS dump or documented seam
- [x] `ruff` + targeted pytest green

## Testing

- Synthetic **unit** cases hard-fail on any missing/extra edge; print diagnostics on failure.
- E/F path is **not** required for default CI green without weights.
- CLI non-zero exit if any frame mismatches.
- No third-party neighbor library as oracle; no mace-torch / ASE / e3nn.

## Out of scope

- Changing production cutoff convention or NeighborList algorithm (separate decision).
- **O(3) energy invariance / force equivariance** under rotation (separate suite).
- Full LAMMPS CI matrix if dump needs invasive patches (document only).
- Oracle performance.
- Training accuracy / general equivariance ports.
