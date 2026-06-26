---
title: Atom-index offset registry for multi-molecule collation
status: approved
created: 2026-06-26
---

# Atom-index offset registry for multi-molecule collation

## Summary
Multi-molecule batching in `molix.data.collate` rebases atom indices by a running `atom_offset` so each molecule's local indices become global. Today only `edge_index` is rebased, and the rule is hardcoded in two places — `collate_molecules` (per-sample scalar add) and `collate_packed` (vectorized segment add) — which must stay byte-identical because they are each other's equivalence oracle. This sub-spec replaces both hardcodes with a single module-level registry (`INDEX_KEYS`) mapping each connectivity key to its `(cat_dim, index_axis)`, plus one tiny `rebase` helper consumed by both functions. The registry pre-declares `bond_index`, `angle_index`, and `dihedral_index` (COO `[k, N]`, offset over rows) so phase 03 only has to *produce* `bond_index` and it rebases correctly with zero further plumbing. No key renames, no `harmonic.py` changes, no `bond_diff`/`bond_dist` touching — this is the offset mechanism only.

## Domain basis
Not a physics spec; no equations or published references apply. Background only: the registry is the minimal lazy subset of PyTorch Geometric's `Data.__inc__` / `__cat_dim__` contract — `__cat_dim__` answers "along which axis are per-graph tensors concatenated" and `__inc__` answers "by how much do entries increment per graph (here: `num_atoms`)". We encode `cat_dim` (the count/concat axis) and `index_axis` (the axis carrying atom indices, over which the per-graph offset broadcasts). No DOI required.

## Design
- **Registry (`INDEX_KEYS`)** — a plain module-level `dict[str, tuple[int, int]]` in `collate.py`, value `(cat_dim, index_axis)`:
  - `edge_index`: `(0, 1)` — shape `[E, 2]`, count axis is `0` (the E axis), both columns (`index_axis=1`) are atom indices.
  - `bond_index`: `(1, 0)` — shape `[2, N]` COO, count axis is `1` (the N axis), both rows (`index_axis=0`) are atom indices.
  - `angle_index`: `(1, 0)` — shape `[3, N]`, same rule as `bond_index`.
  - `dihedral_index`: `(1, 0)` — shape `[4, N]`, same rule.
  Only `edge_index` is produced by any current task; the other three are declared-but-unproduced, reserved for sub-spec 03 / future.
- **Helper (`rebase`)** — one function `rebase(tensor, offset, key)`. When `offset` is a scalar (the per-sample `atom_offset` int in `collate_molecules`) it returns `tensor + offset` (shape-agnostic broadcast). When `offset` is a per-count-element tensor (the gathered segment offsets in `collate_packed`) it returns `tensor + offset.unsqueeze(index_axis)`. For `edge_index` this reproduces the existing `new_atom_offsets[e_seg].unsqueeze(1)` exactly; for `bond_index` the same call yields `.unsqueeze(0)` and offsets the `[2, N]` rows correctly. Kept ~15–25 lines, plain dict + one function, no class / no namedtuple.
- **Single source of truth** — both `collate_molecules` and `collate_packed` import key membership and per-key axes from `INDEX_KEYS` and apply offsets only via `rebase`. Neither function may re-state a cat_dim or an axis literal.
- **Registry home** — keep `INDEX_KEYS` and `rebase` module-level in `collate.py` for now (ponytail rule: do not extract a shared `_index_keys.py` until a second consumer appears). Recommendation recorded here so sub-specs 02/03 import from `molix.data.collate`; if phase 03 or `cache.py` schema inference becomes a second consumer, lift both into a new `src/molix/data/_index_keys.py` at that time.
- **Out of registry scope** — `ptr`/segment-pointer vectors are not rebased here; `bond_diff`/`bond_dist` are edge *features*, not indices, and are left exactly as they are.

## Files to create or modify
- `src/molix/data/collate.py` — add `INDEX_KEYS` registry + `rebase` helper; route the line-115 (`collate_molecules`) and line-304 (`collate_packed`) `edge_index` offsets through them.
- `tests/test_molix/test_data/test_collate.py` — add registry regression: `collate_molecules` edge rebase pinned, plus the synthetic `cat_dim=1` mock-`bond_index` helper test.
- `tests/test_molix/test_data/test_collate_packed.py` — add an explicit assertion that the equivalence oracle (packed == molecules) still holds on an edged batch after the registry routing.

## Tasks
- [ ] Write failing tests for `INDEX_KEYS` + `rebase` in `tests/test_molix/test_data/test_collate.py` (registry declares edge/bond/angle/dihedral with expected `(cat_dim, index_axis)`; `collate_molecules` rebases a 2-molecule edged batch to pinned values; `rebase` on a synthetic `[2, N]` mock `bond_index` with per-count offsets asserts row-broadcast offset)
- [ ] Implement `INDEX_KEYS` registry and `rebase(tensor, offset, key)` helper in `src/molix/data/collate.py`
- [ ] Route the `collate_molecules` `edge_index` offset (currently `edge_index + atom_offset` at ~line 115) through `rebase`, driving key membership from `INDEX_KEYS`
- [ ] Route the `collate_packed` vectorized `edge_index` offset (currently `+ new_atom_offsets[e_seg].unsqueeze(1)` at ~line 304) through `rebase`, asserting byte-identical output
- [ ] Add an edged-batch equivalence-oracle assertion in `tests/test_molix/test_data/test_collate_packed.py` confirming `collate_packed == collate_molecules` after registry routing
- [ ] Add Google-style docstrings (with tensor-shape annotations and the `(cat_dim, index_axis)` meaning) to `INDEX_KEYS` and `rebase` in `src/molix/data/collate.py`
- [ ] Run full check + test suite

## Testing strategy
- **Happy path** — `collate_molecules` on a 2-molecule batch with edges produces the same globally-rebased `edge_index` as before (regression-pinned to the existing `test_collate_basic_fields_and_offsets` expectation `[[2, 3], [2, 4]]` for the second molecule).
- **Equivalence oracle** — `collate_packed(view, indices) == collate_molecules([dataset[i] for i in indices])` leaf-for-leaf on an edged batch still holds with the offset routed through `rebase` (re-uses the GROUP 1 oracle in `test_collate_packed.py`).
- **Mechanism generalization (edge case)** — feed a synthetic registered key with `cat_dim=1` (mock `bond_index` `[2, N]`) and a per-count offset vector through `rebase`; assert each column is offset by its segment value broadcast across both rows, proving the differing-axis path works before phase 03 produces a real `bond_index`.
- **Registry contract** — assert `INDEX_KEYS` contains `edge_index`, `bond_index`, `angle_index`, `dihedral_index` with the declared `(cat_dim, index_axis)` tuples (guards against an accidental rename/drop before sub-specs 02/03 consume them).
- No domain validation: refactor declares no physics.

## Out of scope
- Key renames (`bond_*` → `edge_*`) — sub-spec 02.
- Producing `bond_index` / angle / dihedral tensors and any `harmonic.py` changes — sub-spec 03.
- `bond_diff` / `bond_dist` edge-feature handling — untouched.
- `ptr` / segment-pointer vectors — explicitly deferred.
- Extracting a shared `src/molix/data/_index_keys.py` — deferred until a second consumer exists (recommendation only).
