---
title: Canonicalize covalent bond_index [2, N] with registry-driven offset and edge≠bond enforcement
status: approved
created: 2026-06-26
---

# Canonicalize covalent bond_index [2, N] with registry-driven offset and edge≠bond enforcement

## Summary

This sub-spec (03 of the `graph-connectivity-alignment` chain) makes covalent connectivity a single canonical batch field `bond_index` of shape `[2, num_bonds]` (PyG COO), always paired with `bond_types`. It routes `bond_index` through the lazy declarative offset registry pre-declared in sub-spec 01 so that multi-molecule batches rebase bond atom indices correctly — fixing the latent `BondHarmonic` bug where `bond_index` is never atom-offset during collation. It deletes the 3-way fallback in `src/molpot/potentials/bonds/harmonic.py` (`bond_index` → `edge_index` → `bonds["i"]`) so a geometric cutoff pair `[E, 2]` can never be silently consumed as a bond list `[2, N]`; the potential now requires the canonical contract and raises a clear error otherwise. It maps (without over-building) the molpy I/O boundary that stacks `bonds` columns `atomi`/`atomj` into `bond_index`, and documents both the bond_index/edge_index distinction and a torch_geometric collate/Batch feature-parity table in CLAUDE.md.

## Domain basis

Classical force-field bonded terms (`BondHarmonic`, angles, dihedrals) are defined **strictly on covalent topology**, which is a fixed topological object — conformation-independent. This is a different mathematical object from the neighbor-list graph:

> edge≠bond: NeighborList graph = geometric cutoff graph G(r_c)={(i,j):‖rᵢ−rⱼ‖≤r_c}, recomputed from coords, includes through-space/intermolecular contacts that are NOT chemical bonds. A covalent bond list is a fixed topological object, conformation-independent — different set, cardinality, invariances. Classical FF terms (BondHarmonic, angles, dihedrals) are defined strictly on covalent topology.

The two objects differ in set membership, cardinality, and invariances; feeding one into a term defined on the other is a category error, not an approximation. The canonical COO layout follows PyG:

> PyG COO reference: data.edge_index is [2, num_edges] torch.long, "not a list of index tuples." https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html

Harmonic stretch energy (units: energy in the model's working unit system; lengths in Å, `k` in energy·Å⁻², `r0` in Å):

```
E = Σ_b 0.5 · k[type_b] · (‖pos[j_b] − pos[i_b]‖ − r0[type_b])²
```

The transpose discrepancy between `bond_index = [2, N]` (covalent) and `edge_index = [E, 2]` (cutoff/neighbor) is **deliberate and load-bearing**: it is a type-level anti-alias guard. Indexing `pos[bond_index[0]]` on a `[2, N]` tensor yields per-bond source positions `[N, 3]`; indexing `pos[edge_index[0]]` on an `[E, 2]` tensor yields `pos[[src0, tgt0]]` of shape `[2, 3]` — silently wrong energy, and the `bond_index.size(1) == 0` empty guard is never true for an `[E, 2]` tensor. The transpose is what lets the potential detect the misuse instead of computing nonsense.

## Design

Entities touched:

- **`bond_index` (new canonical field)** — `torch.long`, shape `[2, num_bonds]`, COO; `bond_index[0]` = source atom, `bond_index[1]` = target atom. Lives under the batch `edges`-sibling `bonds` namespace conceptually but is offset by atom index like `edge_index`. Always paired with `bond_types` (`torch.long`, `[num_bonds]`).
- **Offset ownership** — the atom-rebasing rule for `bond_index` is the registry entry **declared in sub-spec 01**. This sub-spec only *consumes* it: collation registers `bond_index` as an atom-offset field so the generic registry-driven rebase applies. No new offset machinery is built here.
- **`BondHarmonic.forward`** — lifecycle change: it stops guessing the connectivity source. It requires `pos`, `bond_index` (`[2, N]`), and `bond_types`; it validates `bond_index.ndim == 2 and bond_index.shape[0] == 2`; an `[E, 2]` tensor (first dim ≠ 2, or a tensor whose shape matches the edge convention) raises a clear `ValueError` naming the edge≠bond contract. The `data["edge_index"]` and `data["bonds"]["i"]` fallbacks are removed. The legitimate empty-bond case (`num_bonds == 0`, i.e. `bond_index.shape == [2, 0]`) still returns `0`.
- **I/O boundary (mapped, not over-built)** — molpy/molrs `to_frame()` emits a `bonds` block with columns `atomi`/`atomj` (and a bond type column). The conversion that stacks these into `bond_index = [2, N]` and `bond_types = [N]` belongs in the frame-ingesting dataset adapter, co-located with the existing `frames[i]["atoms"]` ingest pattern in `src/molix/datasets/molrec.py`. A minimal pure helper performs the column→COO stack; no general molpy importer is introduced.

## Files to create or modify

- `src/molpot/potentials/bonds/harmonic.py` — remove the 3-way fallback and the `[E, 2]`-vulnerable empty guard; require canonical `bond_index [2, N]` + `bond_types`; raise on an `[E, 2]` / non-`[2, N]` tensor; document the intentional transpose.
- `src/molix/data/collate.py` — carry `bond_index` / `bond_types` through both `collate_molecules` and `collate_packed`, registering `bond_index` as an atom-offset field consuming the sub-spec 01 registry entry (offset along dim 1, concat along dim 1).
- `src/molix/data/cache.py` — pack a per-sample `bonds` bucket (`bond_index` concatenated along dim 1, `bond_ptr` cumsum, `bond_types`) so the packed fast path round-trips connectivity.
- `src/molix/datasets/_bond_adapter.py` (new) — minimal pure helper `bond_index_from_columns(atomi, atomj, bond_types)` mapping molpy/molrs `bonds` columns to canonical `bond_index [2, N]` + `bond_types`; documents that `frame["bonds"]` feeds it.
- `CLAUDE.md` — add the torch_geometric collate/Batch feature-parity table and the `bond_index` ([2, N], covalent) vs `edge_index` ([E, 2], cutoff/neighbor) distinction.

## Tasks

- [ ] Write failing tests for BondHarmonic canonical contract: `[E, 2]` edge_index rejected, missing `bond_types` rejected, canonical `[2, N]` path, empty `[2, 0]` returns 0 (`tests/test_molpot/test_potentials/test_bonds.py`)
- [ ] Implement canonical `bond_index` contract in `src/molpot/potentials/bonds/harmonic.py`: remove the `edge_index`/`bonds["i"]` fallback, validate `[2, N]`, raise clear edge≠bond error on `[E, 2]`
- [ ] Write failing tests for multi-molecule `bond_index` atom-offset and synthetic-`[2, N]` round-trip through collate (`tests/test_molix/test_data/test_bond_collate.py`)
- [ ] Wire `bond_index` / `bond_types` through `src/molix/data/collate.py` (both `collate_molecules` and `collate_packed`) and the `bonds` bucket in `src/molix/data/cache.py`, consuming the sub-spec 01 offset-registry entry
- [ ] Implement `bond_index_from_columns` in `src/molix/datasets/_bond_adapter.py` mapping molpy `atomi`/`atomj`/type columns to canonical `bond_index` + `bond_types`
- [ ] Add Google-style docstrings with units and the intentional-transpose note in `harmonic.py`
- [ ] Verify a 2-molecule batch BondHarmonic energy equals the sum of per-molecule energies (offset-fix validation case)
- [ ] Add the torch_geometric collate/Batch parity table and the `bond_index`/`edge_index` distinction to `CLAUDE.md`
- [ ] Run full check + test suite

## Testing strategy

- **Happy path** — canonical `bond_index [2, N]` + `bond_types` + `pos` yields the analytic `Σ 0.5·k·(r−r0)²`; single-molecule energy matches a hand-computed reference.
- **Edge cases** — empty bonds `[2, 0]` returns scalar `0`; missing `bond_types` raises; an `[E, 2]` edge_index passed as `bond_index` raises a clear `ValueError` (regression test on the exact bug, asserting it does **not** silently return a wrong scalar); grep gate asserts no `data["edge_index"]` / `data.get("bonds")` fallback path survives in `harmonic.py`.
- **Round-trip** — synthetic `bond_index [2, N]` (and molpy `atomi`/`atomj` columns via `bond_index_from_columns`) survive `collate_molecules` and the packed `collate_packed` fast path with connectivity preserved (same source/target atoms after rebase).
- **Domain validation** — multi-molecule (2+) batch: total BondHarmonic energy equals the sum of independently computed per-molecule energies, proving `bond_index` is atom-offset correctly via the phase-01 registry (translation/composition invariance of a topological term under batching).

## Out of scope

- The offset-registry mechanism itself (declaration, lazy application engine) — owned by sub-spec 01; this sub-spec only consumes its `bond_index` entry.
- Any `edge_index` / neighbor-list / `bond_diff` / `bond_dist` changes — owned by sub-spec 02; `harmonic.py` reads only `pos` / `bond_index` / `bond_types`, never `edge_diff` / `edge_dist`.
- Angle / dihedral / improper bonded terms — same covalent-topology contract applies but they are separate potentials, not part of this fix.
- A general molpy importer — only the minimal column→COO boundary helper is added; full structure/topology ingestion is deliberately not built.
