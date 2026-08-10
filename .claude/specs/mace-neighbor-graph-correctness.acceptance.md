# Acceptance — mace-neighbor-graph-correctness

Binding criteria after grill supersede 2026-08-10. Types follow evaluator-protocol.

## Criteria

### ac-001 — Independent multi-image brute-force oracle

- **type:** unit
- **verify:** `tests/test_molix/test_md/oracle_bruteforce_neighbors.py` (or equivalent) implements multi-image directed edges `(i,j,sx,sy,sz)` with **`0 < |dr| ≤ cutoff`**, excludes only true self; **no** imports from `molix.md.neighbors`, `molix.op`/`get_neighbor_pairs`, matscipy, ASE, freud, LAMMPS; **does not** call or copy `test_neighbors._reference_pairs` (MIC-only).
- **status:** verified
- **last_checked:** 2026-08-10

### ac-002 — Open random system

- **type:** unit
- **verify:** N∈[10,30], open; NeighborList fresh build vs oracle → missing=0, extra=0.
- **status:** verified
- **last_checked:** 2026-08-10

### ac-003 — Cutoff boundary (≤ convention)

- **type:** unit
- **verify:** distances `r_c±1e-3` and `r_c±1e-6`; **edge present at `r=r_c`**; absent for `r>r_c` (consistent with `0 < r ≤ r_c`). Failure messages state the `≤` convention.
- **status:** verified
- **last_checked:** 2026-08-10

### ac-004 — Periodic wrap

- **type:** unit
- **verify:** L=10 Å, atoms x=0.1 and 9.9, r_c=1 Å; shared edge keys; physical |dr|≈0.2 Å.
- **status:** verified
- **last_checked:** 2026-08-10

### ac-005a — Graph wrap / lattice reparametrization

- **type:** unit
- **verify:** wrap one atom / translate by lattice / translate whole structure → physical `dr` multisets match (graph-level); missing=extra=0 for each representation after fresh rebuild.
- **status:** verified
- **last_checked:** 2026-08-10

### ac-005b — E/F invariance under PBC-equivalent reparametrization

- **type:** unit
- **verify:** same representations as ac-005a; MACE `energy_core` total E and forces agree within a fixed tol when lists are rebuilt. Marked **`@pytest.mark.slow`** and **skipped when MatPES weights are unavailable**. Not SO(3) equivariance.
- **status:** pending

### ac-006 — Triclinic

- **type:** unit
- **verify:** small triclinic cell; missing=extra=0.
- **status:** verified
- **last_checked:** 2026-08-10

### ac-007 — Multi-image small cell

- **type:** unit
- **verify:** box edge ≲ 2 r_c; multi-image edges allowed; missing=extra=0 (not MIC-only).
- **status:** verified
- **last_checked:** 2026-08-10

### ac-008 — Cutoff-crossing frames

- **type:** unit
- **verify:** two-atom trajectory r>r_c → r≤r_c → r>r_c; **per-frame rebuild**; no missing/extra; no hysteresis.
- **status:** verified
- **last_checked:** 2026-08-10

### ac-009 — Trajectory CLI + molplot metrics

- **type:** unit
- **verify:** `scripts/matpes_port/neighbor_graph_audit.py` (or path named in Files) over multi-frame input: table `frame n_ref n_mace missing extra`; mismatch dump with i,j,S,distances; writes `metrics/metrics.jsonl` with scalar keys `n_ref`, `n_mace`, `missing`, `extra` (and `max_dr_mismatch`); non-zero exit on any mismatch.
- **status:** verified
- **last_checked:** 2026-08-10

### ac-010 — Knowledge + experiment scaffold in mace-nve

- **type:** manual
- **verify:** Note under `mace-nve` project `mace-r2san` holds the design; experiment `neighbor-graph-correctness` exists. (Seeded 2026-08-10; re-sync Note body after this supersede.)
- **status:** verified
- **last_checked:** 2026-08-10
- **verified_by:** agent-auto (Note + experiment seeded)

### ac-011 — LAMMPS dump (optional)

- **type:** manual
- **verify:** non-invasive dump of LAMMPS→molnex neighbor payload **or** documented seam + “not in this suite” in the knowledge Note.
- **status:** verified
- **last_checked:** 2026-08-10
- **verified_by:** agent-auto (seam documented in CLI; no invasive dump)

### ac-012 — Dual-backend light coverage

- **type:** unit
- **verify:** at least one synthetic case runs against both default/`bin=` paths when both backends are available (or skip with reason if one backend missing).
- **status:** verified
- **last_checked:** 2026-08-10
