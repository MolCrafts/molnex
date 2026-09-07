---
title: Neighbour-list enumeration leaves molpot — Potentials consumes molix's placed list (composable-a0-03b)
status: in-progress
created: 2026-09-07
chain: composable-a0
---

# composable-a0-03b-neighbor-list-to-molix

## Summary

`src/molpot/forcefield/neighbor_list.py` (`NeighborListBuilder`, 249 lines,
single caller `Potentials`) is a third neighbour-list enumerator living in the
physics layer: `_open` (all-pairs) and `_periodic` (molpy `NeighborQuery`,
intramolecular-only filter — finding F4) duplicate the two owners molix already
has (`molix.batch.neighbor.NeighborList` for training batches, the Edge
Convention owner; `molix.md.neighbors.NeighborList` for MD). Only
`_from_placed` — "placed `edges` → live-position distances + live mask" — is a
potential-side concern. This step deletes the file: enumeration is molix's job,
`Potentials` **requires** a placed list and fails closed with a message naming
the molix owners, and the placed-list consumer becomes a private method on
`Potentials` (keeping a0-03's non-detached `shifts` branch, F13). Raised by the
operator on 2026-09-07 ("这不是基建嘛，不应该在 molix 下面嘛").

## Domain basis

CLAUDE.md dependency graph: batching / `NeighborList` are `molix.batch` and
`molix.md`; `molpot` consumes the post-collate `edges` namespace
(`edge_index (E, 2)`, `edge_diff = pos[target] − pos[source]`, `shifts`,
`num_edges`). Two `NeighborList` types already exist and are documented as the
owners (`.claude/notes/architecture.md` §Layer roles). A potential that
enumerates its own pairs re-implements infrastructure, hides the Edge
Convention's ownership, and (via `_periodic`'s same-molecule filter) carries
a latent training-batch bug (F4).

## Design

- Delete `src/molpot/forcefield/neighbor_list.py` and its export.
- `Potentials._placed_pairs(system, pos) -> tuple[edge_index, dist, live_mask]`
  (private): reads `edges.edge_index`, prefers `edges.shifts`
  (`pos[t] − pos[s] + shifts`, clamped sqrt — the F13-correct branch), falls
  back to the `edge_diff` remainder reconstruction (documented energy/force-only;
  the virial refusal from a0-03 stays), applies the `source < target`,
  `0 < r < cutoff`, `row < num_edges` liveness rule, keeps capacity shape.
  Missing `edges` (or missing both `edge_diff` and `shifts`) → `ValueError`
  naming `molix.batch.NeighborList` / `molix.md.NeighborList` — no silent
  all-pairs fallback.
- `_scaled_pairs` / the pair terms are unchanged apart from calling
  `_placed_pairs`.
- Fixtures that relied on the auto-enumeration (bare batches without `edges`)
  are rebuilt through `molix.batch.NeighborList(cutoff=…).execute(batch)` in
  the tests' shared helpers — one helper edit per module, not per test.
- F4 closes by deletion; the note's F4 row is updated.

## Files

- `src/molpot/forcefield/neighbor_list.py` (deleted), `src/molpot/forcefield/__init__.py`, `src/molpot/forcefield/potentials.py`
- `tests/test_molpot/test_forcefield/test_neighbor_list.py` (deleted — its placed-list tests move to `test_potentials.py` as `_placed_pairs` tests), `tests/test_molpot/test_forcefield/{conftest,test_potentials,test_lj,test_coulomb,test_analytic_forces,test_predictions}.py` and any other bare-batch fixture the sweep finds
- `.claude/notes/composable-potentials.md` (F4 row)

## Tasks

- [x] tester sweep: list every test / script that reaches `Potentials` energy without an `edges` namespace; rebuild those fixtures through `molix.batch.NeighborList`
- [x] RED: `Potentials` on a batch without `edges` raises naming the molix owners; `_placed_pairs` reproduces the a0-02/a0-03 placed-list contract (moved tests)
- [x] delete `neighbor_list.py` + export; `_placed_pairs` on `Potentials`; `_scaled_pairs` rewired
- [x] F4 row closed in the note; architecture note updated (two owners, molpot consumes)

## Testing

Unit: `test_potentials.py::TestPotentials` — `_placed_pairs` contract (capacity
shape, live mask, `shifts` preferred over `edge_diff`), fail-closed without
`edges`; the moved placed-list tests. Mirrored dirs green
(`tests/test_molpot`, `tests/test_molix/test_md`, `tests/test_molix/test_ff_export`);
slow: the a0-02/a0-03 explains still 0 breaks.

## Out of scope

Unifying the two molix `NeighborList` types; packed periodic training batches
(`num_graphs > 1` under a cell — still fail-closed from a0-03).
