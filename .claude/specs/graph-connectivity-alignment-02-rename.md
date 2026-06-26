---
title: Rename edges-namespace geometry keys bond_diff/bond_dist to edge_diff/edge_dist
status: approved
created: 2026-06-26
---

# Rename edges-namespace geometry keys bond_diff/bond_dist to edge_diff/edge_dist

## Summary
The `("edges", ...)` namespace currently stores the neighbor-graph geometry under the keys `bond_diff` and `bond_dist`. These describe a *geometric* radius-cutoff graph (NeighborList), not chemical bonds, so `bond_*` is a scientific misnomer that misleads readers into treating through-space contacts as bonded pairs. This sub-spec performs a single atomic key-string rename `bond_diff → edge_diff`, `bond_dist → edge_dist` across producers, every cross-package consumer, tests, and docs, unifying with `molrep/utils/geometry.py` which already uses `edge_*`. No mechanism, sign convention, or `bond_index`/harmonic behavior changes — only the dictionary key strings. The rename is committed atomically because any partial sweep raises `KeyError(("edges", "bond_dist"))` at the first un-renamed consumer.

## Domain basis
The NeighborList builds a geometric cutoff graph

```
G(r_c) = { (i, j) : ‖r_i − r_j‖ ≤ r_c }
```

recomputed from coordinates each step. This graph includes intermolecular / through-space contacts that are **not** chemical bonds; therefore the geometry tensors describing it must not be named `bond_*`. In GNN / message-passing literature "edge" is neutral directed adjacency (sender → receiver). Per CLAUDE.md, `edge_index[:, 0]` is the source/sender and `edge_index[:, 1]` is the target/receiver, with the load-bearing sign convention

```
edge_diff[e] = pos[edge_index[e, 1]] − pos[edge_index[e, 0]]   (target − source)
edge_dist[e] = ‖edge_diff[e]‖
```

With `symmetry=True` the NeighborList appends the reverse edge for every pair with a **negated** displacement, `edge_diff_rev = −edge_diff`. This negation is physically required so that spherical-harmonic edge features respect parity, `Y_ℓ(−r̂) = (−1)^ℓ Y_ℓ(r̂)`; the equivariant MLIPs in the repo (MACE, Allegro, PiNet) all consume this radius graph and rely on the sign relation. The rename must preserve both the target−source convention and the reverse-edge negation byte-for-byte. PyG COO adjacency reference: https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html

## Design
This is a pure rename of two string literals used as dictionary keys in the `("edges", ...)` namespace; no symbols, signatures, or tensor semantics change.

- **Producer** (`src/molix/data/tasks/neighbor.py`, `src/molix/data/collate.py`): the transform that emits the edges namespace writes `edge_diff` / `edge_dist` instead of `bond_diff` / `bond_dist`. The negation logic (`edge_diff = -deltas`; reverse edge `torch.cat([edge_diff, -edge_diff])`) is untouched; only the key strings and the docstrings naming them change.
- **Cache schema** (`src/molix/data/cache.py`): `edge_diff` / `edge_dist` become `PackedCache` bucket key names. Bump `PackedCache.FORMAT_VERSION` from `2` to `3`. **Decision: read-time alias, not forced rebuild.** A cache written at `format_version == 2` (carrying `bond_diff` / `bond_dist`) loads transparently and is remapped to `edge_diff` / `edge_dist` on read, emitting a one-time `DeprecationWarning`. This avoids forcing every cached HPC dataset to rebuild for a key rename. The alias covers exactly one prior version (`v2`); a `v1` or older cache continues to raise the existing version-mismatch error. The alias is documented as removable in the next format bump.
- **Consumers**: every reader of `batch["edges", "bond_diff"]` / `bond_dist` (molzoo encoders, molpot heads/composition, molix.md, molix.profiler, molix.lammps, molrep interaction) switches to the new keys. Because these reads are spread across packages, the commit is atomic.
- **Sign invariant**: an added test pins `edge_diff[e] = pos[target] − pos[source]` and the `symmetry=True` reverse-edge negation, so the rename cannot silently flip the convention.

The usual large-spec heuristic would flag this cross-package breadth for splitting, but a key rename is one indivisible logical change: a partial sweep is a broken tree (`KeyError`). The breadth is mechanical, not architectural, so it is kept as one atomic commit by design.

## Files to create or modify
Producer + schema:
- `src/molix/data/tasks/neighbor.py`
- `src/molix/data/collate.py`
- `src/molix/data/cache.py`

Consumers (src):
- `src/molix/md/forcefield.py`
- `src/molix/profiler/__init__.py`
- `src/molix/profiler/mock.py`
- `src/molix/lammps/adapter.py`
- `src/molzoo/allegro.py`
- `src/molzoo/mace.py`
- `src/molzoo/pinet.py`
- `src/molpot/composition/composer.py`
- `src/molpot/composition/sonata.py`
- `src/molpot/heads/charge_bond.py`
- `src/molpot/heads/charge_response.py`
- `src/molpot/heads/dipole.py`

Tests:
- `tests/test_molix/test_data/test_neighbor_sign.py` (new)
- `tests/test_molix/test_data/test_cache_alias.py` (new)
- `tests/symmetry_helpers.py`
- `tests/test_molix/test_md_dynamics.py`
- `tests/test_molix/test_lammps.py`
- `tests/test_molix/test_core/test_losses_molecular.py`
- `tests/test_molix/test_data/test_collate.py`
- `tests/test_molix/test_data/test_collate_packed.py`
- `tests/test_molix/test_data/test_datamodule.py`
- `tests/test_molix/test_data/test_dtype_propagation.py`
- `tests/test_molix/test_data/test_e2e_workers.py`
- `tests/test_molix/test_data/test_pipeline.py`
- `tests/test_molix/test_data/test_sampler.py`
- `tests/test_molix/test_data/test_transform.py`
- `tests/test_molpot/test_composition/conftest.py`
- `tests/test_molpot/test_composition/test_composition.py`
- `tests/test_molpot/test_composition/test_sonata.py`
- `tests/test_molpot/test_composition/test_sonata_batch.py`
- `tests/test_molpot/test_composition/test_sonata_boundary.py`
- `tests/test_molpot/test_composition/test_sonata_periodic.py`
- `tests/test_molpot/test_heads/test_charge_bond.py`
- `tests/test_molpot/test_heads/test_edge.py`
- `tests/test_molpot/test_heads/test_multipole_charged.py`
- `tests/test_molpot/test_heads/test_multipole_symmetry.py`
- `tests/test_molzoo/test_allegro.py`
- `tests/test_molzoo/test_mace.py`
- `tests/test_molzoo/test_mace_encoder.py`
- `tests/test_molzoo/test_mace_omol.py`
- `tests/test_molzoo/test_pinet_padding.py`
- `tests/test_molzoo/test_symmetry.py`

Docs + benchmarks (outside grep gate but kept correct):
- `CLAUDE.md` (Edge Convention + Post-collate batch schema sections)
- `docs/molix/explanation/batch-schema.md`
- `docs/molix/user-guide/data-loading.md`
- `docs/molix/tutorials/train-a-graph-model.md`
- `docs/molpot/tutorials/build-a-potential.md`
- `docs/molrep/explanation/representation-learning.md`
- `docs/molrep/tutorials/build-an-encoder.md`
- `docs/molzoo/tutorials/index.md`
- `docs/molzoo/user-guide/allegro.md`
- `docs/molzoo/specs/allegro.md`
- `docs/molzoo/specs/mace_omol.md`
- `src/molzoo/README.md`
- `src/molzoo/specs/allegro.md`
- `src/molzoo/specs/pinet2.md`
- `src/molpot/README.md`
- `src/molrep/README.md`
- `src/molix/data/README.md`
- `benchmarks/bench_pinet.py`
- `benchmarks/bench_trainer_overhead.py`
- `CHANGELOG.md`

## Tasks
- [ ] Write failing sign-invariant test for edge geometry (tests/test_molix/test_data/test_neighbor_sign.py) asserting edge_diff[e] == pos[target] - pos[source] and that symmetry=True appends reverse edges with negated edge_diff, using the new key names
- [ ] Write failing cache-alias test (tests/test_molix/test_data/test_cache_alias.py) that loads a format_version=2 cache carrying bond_diff/bond_dist and asserts it is read as edge_diff/edge_dist with a DeprecationWarning
- [ ] Implement edge_diff/edge_dist producer keys in src/molix/data/tasks/neighbor.py and src/molix/data/collate.py, updating docstrings and leaving the negation logic unchanged
- [ ] Bump PackedCache.FORMAT_VERSION 2->3 and add the read-time bond_*->edge_* alias with deprecation note in src/molix/data/cache.py
- [ ] Sweep all src consumer key-string reads to edge_diff/edge_dist (molzoo allegro/mace/pinet, molpot composition + heads, molix.md.forcefield, molix.profiler, molix.lammps.adapter)
- [ ] Sweep all test fixtures and assertions to edge_diff/edge_dist across tests/ (collate, datamodule, composition/sonata, heads, molzoo, symmetry_helpers, md/lammps)
- [ ] Update CLAUDE.md Edge Convention and Post-collate batch schema to edge_diff/edge_dist preserving the sign/direction wording, and sweep docs/, README files, benchmarks, and CHANGELOG
- [ ] Verify rotation + permutation equivariance tests (tests/test_molzoo/test_symmetry.py, tests/symmetry_helpers.py) pass post-rename
- [ ] Run full check + test suite

## Testing strategy
- **Happy path**: a collated batch exposes `batch["edges", "edge_diff"]` `(E, 3)` and `batch["edges", "edge_dist"]` `(E,)`; downstream encoders consume them and run end-to-end without `KeyError`.
- **Edge cases**: `symmetry=False` (half-pair count, no reverse edges); empty/NaN-padded edge rows; dtype propagation (`test_dtype_propagation.py`) unchanged under the new keys; worker/e2e path (`test_e2e_workers.py`) carries renamed keys across process boundaries.
- **Cache migration**: a `format_version=2` cache (old `bond_*` buckets) loads through the alias yielding `edge_*` plus one `DeprecationWarning`; a freshly written cache reports `format_version=3` and carries `edge_*` natively; a `v1` cache still raises the existing version error.
- **Domain validation**: sign-invariant test pins `edge_diff[e] = pos[target] − pos[source]` and the `symmetry=True` reverse-edge negation `edge_diff_rev = −edge_diff`; rotation-equivariance and permutation-equivariance suites for MACE/Allegro/PiNet remain green, confirming parity behavior is byte-for-byte preserved.
- **Grep gate**: no `"bond_diff"`/`"bond_dist"`/`'bond_diff'`/`'bond_dist'` key-string literal remains under `src/` or `tests/`.

## Out of scope
- The new edges-namespace mechanism / population changes — owned by sub-spec 01.
- `bond_index` and harmonic-bond (true chemical bond) changes — owned by sub-spec 03.
- Renaming `molrep/utils/geometry.py` symbols — it already uses `edge_*`; this spec aligns *to* it, not the reverse.
- Removing the `v2` read-time alias — deferred to the next `FORMAT_VERSION` bump; alias intentionally retained one version.
- Frozen historical records under `.claude/specs/*` — left as written; not rewritten by this sweep.
