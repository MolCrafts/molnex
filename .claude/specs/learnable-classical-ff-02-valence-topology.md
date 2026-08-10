---
title: Learnable classical FF — valence topology collate namespaces
status: done
created: 2026-08-10
revised: 2026-08-10
grilled: true
chain: learnable-classical-ff
depends_on:
  - learnable-classical-ff-01-ir-kernels
---

# Learnable classical FF — valence topology collate namespaces

## Summary

Extend the molix two-tier **TensorDict** contract so covalent **angles**,
**propers**, and **impropers** live as nested namespaces with **per-role atom
index columns** (molpy/molrs Frame style: `atomi` / `atomj` / …), e.g.
`batch["angles"]["atomi"]` / `batch["angles", "atomi"]`. Collate rebases those
1-D index vectors by atom offset and packs them in `PackedCache`. Topology is
**not** stored in molpy; molpy may only *emit* columns when a sample is built.
No energy evaluation here.

## Design


### Placement supersede (2026-08-10 — binding)

Full rule: `.claude/notes/learnable-classical-ff.md`.

1. **Reuse first** — prefer molpy (≥0.13) / in-tree modules over new twins.
2. **No `Foo(method=…)` on new APIs**.
3. **Non-diff sinks to molpy/molrs** for *enumeration / SMARTS / classical
   non-torch E/F* — **not** for the live batch store.
4. **Batch topology store = molix TensorDict only.**

**02-specific (corrected):**
- **Primary schema is column form under TensorDict namespaces**, aligned with
  molpy Frame blocks (`atomi`/`atomj`/`atomk`/`atoml`), **not** packed
  `angle_index [3, N]` as the batch contract.
- Optional: stack columns → COO only at the **kernel call site** for 01
  potentials that still take `[arity, N]` (thin local helper; not the collate
  schema).
- Existing `bonds` may keep `bond_index [2, N]` for back-compat **or** also
  expose `atomi`/`atomj` under `bonds` — prefer adding columns; do not break
  current bonds consumers without a follow-up.
- Improper: molrs center-first means `impropers.atomi` is the **center**.

### Reuse decision

| Symbol | Action | Rationale |
|--------|--------|-----------|
| Nested `bonds` TensorDict collate | **pattern-reuse** | Same optional namespace + all-or-none presence |
| `rebase` | **generalize** | Support 1-D atom-index columns (`atomi` etc.), not only COO tables |
| `INDEX_KEYS` | **extend** | Register column keys that need atom offset (see below) |
| `collate_molecules` / `collate_packed` / `PackedCache` | **extend** | Valence namespaces with 1-D columns + ptrs |
| molpy Frame `atomi/atomj/...` | **naming reuse** | Same column names in TensorDict; Frame is not the runtime store |
| `bond_index_from_columns` | **reuse** | Kernel-local stack helper pattern when COO is needed |
| Potential kernels | **out of scope** | 01 |

### Canonical TensorDict schema (post-collate)

Access forms (equivalent):

```python
batch["angles"]["atomi"]   # preferred nested style
batch["angles", "atomi"]   # tuple key (same TensorDict)
```

```
TensorDict
├── atoms / edges / graphs
├── "bonds": TensorDict (batch_size=[])   # existing; see note
│   ├── bond_index: [2, N_b]              # legacy COO (keep green)
│   ├── atomi / atomj: [N_b]              # optional parallel columns
│   └── bond_types / type: [N_b]          # optional
├── "angles": TensorDict (batch_size=[N_a] or [] — prefer batch_size=[N_a]
│             when all leaves share length N_a)
│   ├── atomi: [N_a]   # i of angle i–j–k
│   ├── atomj: [N_a]   # central
│   ├── atomk: [N_a]
│   └── type:  [N_a]   # optional integer type id
├── "propers": TensorDict
│   ├── atomi, atomj, atomk, atoml: [N_p]
│   └── type: [N_p] optional
└── "impropers": TensorDict
    ├── atomi: [N_i]   # **center** (molrs)
    ├── atomj, atomk, atoml: [N_i]
    └── type: [N_i] optional
```

**`batch_size`:** Prefer `batch_size=[N_terms]` when every leaf in the
namespace is length `N_terms` (true for pure column form). That is cleaner
than `bonds`’ historical `batch_size=[]` forced by COO+types shape clash.
If only columns are present, use `[N]`.

Deliberately **not** geometric `edge_index`. Deliberately **not** requiring
`angle_index [3, N]` on the batch.

### Flat sample keys (pre-collate)

Either nested already or flat with a documented prefix convention. Preferred
flat dict for a sample:

```
# angles
"angles": {"atomi": Long[N_a], "atomj": Long[N_a], "atomk": Long[N_a], "type"?: Long[N_a]}
# or flat keys if pipeline is flat-only pre-collate:
"angle_atomi", "angle_atomj", "angle_atomk", "angle_type"?
```

Implementer picks one pre-collate style and documents it; post-collate is always
the nested form above.

### INDEX_KEYS / rebase

Register **1-D atom index columns** (cat on dim 0, index axis is the vector
itself — offset every element):

```python
# illustrative — exact registration style may use namespaced keys
"atomi": (0, 0)   # 1-D, cat dim 0
"atomj": (0, 0)
"atomk": (0, 0)
"atoml": (0, 0)
```

When collating a valence namespace, rebase **each present** of
`{atomi, atomj, atomk, atoml}` by `atom_offset`, then `torch.cat` on dim 0.
Do **not** invent a second COO path as the collate primary.

Optional kernel helper (datasets or molpot util, not collate schema):

```python
def stack_angle_index(angles_td: TensorDict) -> Tensor:
    return torch.stack([angles_td["atomi"], angles_td["atomj"], angles_td["atomk"]], dim=0)
```

### collate_molecules / collate_packed / PackedCache

- Mirror bonds: optional namespace if any sample carries angles (all-or-none).
- Packed buckets store concatenated 1-D columns + `angle_ptr` etc.
- `collate_packed` gather + rebase each column by segment atom base.
- Unpack restores nested or flat sample form.

### Documentation

Contract lives in this spec + `.claude/notes/learnable-classical-ff.md`.

## Files to create or modify

- `src/molix/data/collate.py` — valence namespaces as **column** TensorDicts; INDEX_KEYS for atomi/j/k/l
- `src/molix/data/cache.py` — pack/unpack angle/proper/improper **columns** + ptrs
- `src/molix/datasets/_valence_columns.py` (new) — optional stack helpers only (COO for kernels)
- `tests/test_molix/test_data/test_collate.py` (extend)
- `tests/test_molix/test_data/test_collate_packed.py` (extend)
- `tests/test_molix/test_data/test_cache_valence.py` (new or extend)
- `tests/test_molix/test_datasets/test_valence_columns.py` (new)

## Tasks

- [x] Write failing collate tests: multi-sample batch rebases `angles.atomi/j/k` by cumulative n_atoms; types concat; empty-N; access `batch["angles"]["atomi"]`
- [x] Implement collate_molecules valence namespaces as column TensorDicts (atomi/atomj/atomk/atoml + optional type)
- [x] Write failing PackedCache round-trip tests for column buckets + ptrs
- [x] Implement PackedCache pack/unpack + collate_packed for angles/propers/impropers columns
- [x] Write failing tests for optional stack helpers (columns → [3,N]/[4,N] for 01 kernels only)
- [x] Implement stack helpers; do **not** make packed COO the batch schema
- [x] Keep existing bonds collate green (no regression)
- [x] Run check + unit tests

## Testing strategy

- **Offset correctness (code):** two molecules with known local indices; after
  collate, second molecule's indices equal local + n_atoms_0.
- **Oracle parity (code):** collate_packed on a PackedCache equals
  collate_molecules on unpack_sample list (leaf-for-leaf torch.equal for new
  namespaces), same pattern as existing packed-collate suite.
- **All-or-none (code):** mixed presence of angle_index across samples raises
  ValueError with a clear message.
- **Empty valence (code):** all-zero counts produce either omitted namespace or
  empty `[3,0]` / `[4,0]` tensors consistent with bonds empty behavior.
- **Adapters (code):** column stacks produce `[arity, N]` long tensors;
  types length matches N.
- **No energy (code):** touched modules do not import molpot potentials.
- Full suite green; bonds/edge collate regressions still pass.

## Out of scope

- Energy / force evaluation (01 / 03 / 05).
- SMARTS matching or chemical perception (04 / 07).
- Generating angles/propers from bond graphs automatically (topology builders
  may land later; this spec only transports caller-supplied indices).
- Changing edge_index geometric convention.
- TargetSchema / loss wiring.
