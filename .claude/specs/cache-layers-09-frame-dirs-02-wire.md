---
title: Drop the count wrap; public source and indices; home is molhub_config.home_dir
status: code-complete
grilled: true
created: 2026-09-12
revised: 2026-09-12
chain: cache-layers
depends:
  - cache-layers-09-frame-dirs-01-types
  - cache-layers-08-molnex-counts
---

# Drop the count wrap; public source and indices; home is molhub_config.home_dir

## Summary

Molnex Class-I OMol scripts keep, prefix, and profile-filter with the numpy vector from `source.cache("graph.natoms")` under the exact predicate `1 ≤ n ≤ max_atoms`, without `_VectorCounts` and without any `LazyOMol25.atom_counts` (no constructor argument, no property, no packing field). `LazyOMol25` is a view with public `source` and `indices` (tuple of source rows). A decode that disagrees with `int(numbers.shape[0])` on the public `record()` path, or `int(frame["atoms"].nrows)` on the frame path, raises `ValueError` and does not write a sidecar. `_is_labeled_source` is `hasattr(obj, "task_labels")` only. `prod_day` / `quasi_prod` min/max are `np.asarray(view.source.cache("graph.natoms"))[list(view.indices)]`. The labeled `ClassIFit.datamodule` path is `DataModule(..., batch_size=8)` with no `DataMixture` and no token sampler. Process home is the `molhub_config.home_dir` attribute from cache-layers-09-frame-dirs-01-types, not `molhub_config["home"]`. There is still no `cache()` on `LazyOMol25` and no `Cache()` constructor.

## Domain basis

Count snapshots are not scientific labels: they do not define chemistry subsets, change energies or forces, or stand in for measured bounds (cache-layers-02-cache, cache-layers-08-molnex-counts). This slug does not introduce a physical model, conversion constant, or DOI.

## Design

Scope is molnex `scripts/mm_param_learning` and `tests/test_scripts/test_mm_param_learning` only. Do not edit `src/molix` or molhub production (`home_dir` lands in cache-layers-09-frame-dirs-01-types; `loaded.cache("graph.natoms")` already landed in cache-layers-02-cache; public `record()` already landed in cache-layers-01-sources). `scripts/` may import molhub; `src/` still must not.

Depends-on contract from 09-01-types, consumed here and not reimplemented: `molhub_config.home_dir` is a `pathlib.Path` attribute (read and assign). 09-01 does **not** write `frame.meta["n_atoms"]`. This slug does not add that attribute, does not write that meta key, and does not require it.

Current callers: `_official_vector_counts`, `_official_prefix`, `_index_omol_source`, `_lazy_ev`, `load_omol25_official` / `load_omol25_official_tiny` / `load_omol25_fairchem_4m`, `LazyOMol25.__init__` / `atom_counts` / `subset` / `__getitem__` / `_sample_from_record`, `_is_labeled_source`, `ClassIFit.datamodule`, `_e0_cache_path`, `profile_class_i.build_lazy_pair`, `prod_day._require_nobackup_molhub_home` and the coverage dicts in `prod_day.run` / `quasi_prod.calibrate`. Current tests: `TestOfficialVectorCounts`, `TestVectorCounts`, `TestLoadOmol25Official`, `TestE0CachePath`, `TestLazyOMol25`, `TestBuildLazyPair`, `TestRequireNobackupMolhubHome`, plus `test_batch_max_atoms_does_not_drop_a_500_atom_row`.

Law findings from the blocked persist of this slug, resolved here (caller / test named):

- 🔴 `LazyOMol25.atom_counts` (property, packing field, constructor arg) — deleted. One owner of the view subset is `view.source.cache("graph.natoms")` at `view.indices`. Caller: `ClassIFit.datamodule` uses `DataModule(batch_size)` (today's `DataMixture` wrap would raise `ValueError: LazyOMol25 does not expose atom_counts` after the name is gone). Test: `TestLazyOMol25` / `TestClassIFit`.
- 🔴 `_subset_natoms` over `view._source` / `view._indices` — deleted. Caller: `prod_day.run` / `quasi_prod.calibrate` inline gather on the public attributes. Test: those numbers, not private layout.
- 🟡 LMDB vs frame via `_tables` — LMDB is a callable public `record()` on the loaded view (cache-layers-01-sources). Test: `TestLazyOMol25` (LMDB stub exposes `record()`; frame stub does not).
- 🟡 `frame.meta["n_atoms"]` as grain-one — grain-one is `int(frame["atoms"].nrows)` (molpy `Block.nrows`). Test: `TestLazyOMol25` 3-vs-2 nrows mismatch. No `KeyError` for missing meta `n_atoms`.

No § VII carve-out is on file (`law.md` is absent).

### Numpy from `source.cache("graph.natoms")`

Delete `_VectorCounts`. Do not add a replacement wrap, gather type, or `Cache()` constructor. `from molhub.cache import Cache` stays absent.

Generalize `_official_vector_counts` to the one molnex conversion, returning the array:

```python
def _official_vector_counts(source: Any) -> np.ndarray:
    return np.asarray(source.cache("graph.natoms"))
```

No `getattr`, no `hasattr`, no `try`/`except`. Missing `cache` is `AttributeError` and propagates. An `AttributeError` raised inside a real `cache` also propagates. Unknown field names are MolHub `ValueError` and propagate. `"n_atoms"` and molnex `"graphs.num_atoms"` stay unknown.

Keep / prefix / profile filter that numpy with one predicate:

```text
1 <= n <= max_atoms
```

A measured zero is dropped. Negatives are dropped by the same predicate. Sites: `_index_omol_source`, `_official_prefix`, `build_lazy_pair`. `load_omol25_fairchem_4m` still does not filter by `batch_max_atoms`.

`_index_omol_source` calls `_official_vector_counts` once, then `keep = np.flatnonzero((n_atoms >= 1) & (n_atoms <= int(max_atoms)))`. Return type is `tuple[list[int], np.ndarray]` (keep, ε). It does not return a count wrap. `load_omol25_official` still fits ε through `_fit_e0_subsample` and still returns `tuple[LazyOMol25, LazyOMol25]` with kcal conversion and `val_role="val-neutral"`.

`_official_prefix` takes the numpy vector (no wrap, no LMDB decode):

```python
def _official_prefix(n_atoms: np.ndarray, n: int, max_atoms: int, scan_cap: int) -> list[int]:
    limit = min(int(n_atoms.shape[0]), int(scan_cap))
    keep: list[int] = []
    for i in range(limit):
        if len(keep) >= n:
            break
        size = int(n_atoms[i])
        if 1 <= size <= int(max_atoms):
            keep.append(i)
    return keep
```

`load_omol25_official_tiny` calls `_official_vector_counts` once per source, prefixes from that array, and constructs `LazyOMol25` without passing the array.

`_lazy_ev(source, indices)` drops the counts argument:

```python
def _lazy_ev(source: Any, indices: Sequence[int]) -> LazyOMol25:
    return LazyOMol25(source, indices, dressing=None, energy_unit=units.eV)
```

### `LazyOMol25` is a view, not a count owner

Constructor is `LazyOMol25(source, indices, *, dressing=None, energy_unit=...)`. There is no positional or keyword `atom_counts=`. There is no `@property atom_counts`. There is no packing tensor field named `atom_counts`. Accessing `view.atom_counts` is `AttributeError`. `subset` forwards `source`, kept indices, `dressing`, and `energy_unit` only. `view.cache("graph.natoms")` remains `AttributeError`.

Public attributes, set once in `__init__` (not factories, not private aliases tests must read):

```python
self.source = source
self.indices = tuple(int(i) for i in indices)
```

`indices` is the tuple of source rows. Keep / prefix / profile / decode / launcher min/max / tests read these names. Do not keep `_source` / `_indices` as the public contract. Existing `pair_counts` zeros stay as they are; they are not a count index and are not what launchers or `datamodule` read.

### Decode vs physical table length

The loaded view is LMDB iff it exposes the public `record()` method from cache-layers-01-sources (`OMol25Lmdb.record` / `OMol25.record`). Discriminator is:

```python
record_fn = getattr(self.source, "record", None)
if callable(record_fn):
    ...
```

Document that `getattr(..., "record", None)` callable **is** that public `record` API. Do not inspect private `_tables`. Packed `OMol25Dataset` and frame-dir / test stubs have no `record()`; they take the frame path. Do not unwrap `source._view` in `__getitem__` — `OMol25.record` already delegates.

After the LMDB `record()` path decodes `numbers`, if `row is not None`:

```python
cache_row = int(np.asarray(self.source.cache("graph.natoms"))[row])
decoded = int(numbers.shape[0])
if cache_row != decoded:
    raise ValueError(
        f"row {row}: cache count {cache_row} disagrees with decoded count {decoded}"
    )
```

On the frame path (`__getitem__` else branch: tests / packed NPZ / frame directories), compare the same `cache_row` to `int(frame["atoms"].nrows)` on the source frame before convert. Disagreement raises `ValueError` naming the row, the cache count, and the table length. Do not read `frame.meta["n_atoms"]`. Missing that meta key is not a failure of this slug. MetaValue unwrap `getattr(v, "value", v)` stays only in `_to_energy_unit`. Do not overwrite anything.

`_is_labeled_source` is exactly:

```python
def _is_labeled_source(obj: Any) -> bool:
    return hasattr(obj, "task_labels")
```

Do not test `atom_counts` here. `hasattr` is this discriminator only — not cache discovery and not the LMDB/frame branch.

### Labeled `datamodule` is `DataModule(batch_size)`, not `DataMixture`

Today `ClassIFit.datamodule` always wraps `self.train_ds` in `DataMixture({TASK_ENERGY: self.train_ds}, mixture={TASK_ENERGY: 1.0}, max_atoms=self.max_atoms)`. `DataMixture` `getattr`s `atom_counts` (`molix.data.mixture._snapshot_source_counts`) and would raise `ValueError: LazyOMol25 does not expose atom_counts` once that name is gone. That is why a packing field named `atom_counts` is forbidden: it would be the wrap API under a new story.

After `__init__`, `self.train_ds` is already labeled (`LazyOMol25` or `_LabeledFrames`, both expose `task_labels`). Generalize `datamodule` to:

```python
return DataModule(
    self.train_ds,
    self.val_ds,
    batch_tasks=[self.forcefield.neighbor_list(), TotalCharge(), *self.batch_tasks],
    batch_size=8,
    ...
)
```

No `DataMixture`. No `max_atoms_per_batch`. No `batch_sampler_source`. Drop the unused `DataMixture` import. `ClassIFit.max_atoms` remains the construction field callers still pass (`packed_max_atoms`); it is not a token-sampler budget after this slug.

`_LabeledFrames` may still copy `FrameDataset.atom_counts` (molix store packing, out of scope to delete). That copy is unused by `datamodule` once `DataMixture` is gone. Do not add `atom_counts` on `LazyOMol25` to keep a dead mix path green.

### Launchers gather cache at public `source` / `indices`

Delete `_subset_natoms`. Do not add a replacement gather helper. The two call sites (`prod_day.run` coverage, `quasi_prod.calibrate` coverage) inline:

```python
n_atoms = np.asarray(view.source.cache("graph.natoms"))[list(view.indices)]
```

Use that array for min/max when `len(view)` else `0`. Keep the existing coverage keys (`prod_day` max only; `quasi_prod` min and max). They do not read `view.atom_counts` and they do not read private `_source` / `_indices`.

### Home is `molhub_config.home_dir`

`_e0_cache_path` reads `Path(molhub_config.home_dir)`, not `molhub_config["home"]` and not `os.environ`. Path stays `home_dir / "derived" / "omol25-index" / f"{digest}-e0.npy"` with the existing sha1 of `root:len:max_atoms:n_fit:seed`.

`prod_day._require_nobackup_molhub_home` mkdirs the nobackup workspace path and assigns `molhub_config.home_dir = that Path`. It does not read or write `os.environ`. After this slug, `scripts/mm_param_learning` and `tests/test_scripts/test_mm_param_learning` contain no `molhub_config["home"]`. Tests that assign `home_dir` restore the previous Path.

`profile_class_i.build_lazy_pair` uses `_official_vector_counts(src)` as numpy, keeps `1 ≤ n ≤ max_atoms`, and constructs both `LazyOMol25` views with `(src, indices, dressing=...)` only. `run_profile` still calls `_require_nobackup_molhub_home` before `build_lazy_pair`. Zeros dressing stays `_historical_are(torch.zeros(95))`. Do not retarget the profile onto `load_omol25_fairchem_4m`.

### Reuse decision

Caller did not pass a librarian_report. Decisions below are from in-repo discovery of cache-layers-01/02/08 symbols and the blocked 09-02 persist.

- **reuse** `OMol25.cache` / `OMol25Lmdb.cache` / packed `OMol25Dataset.cache` — spec calls `source.cache("graph.natoms")`; molnex does not reimplement lookup/publish or construct `Cache`.
- **reuse** `OMol25.record` / `OMol25Lmdb.record` (cache-layers-01-sources) — public LMDB row dict; `getattr(view, "record", None)` callable is this API. Packed `OMol25Dataset` has no `record()`.
- **reuse** molpy `Block.nrows` — grain-one physical table length is `int(frame["atoms"].nrows)`.
- **reuse** `molhub_config.home_dir` (cache-layers-09-frame-dirs-01-types) — the Python home attribute; assignment is the write path. Do not reintroduce `molhub_config["home"]`.
- **reuse** `_fit_e0_subsample` — historical ε fit stays; only its on-disk root follows `home_dir`.
- **reuse** `_historical_are` — profile zeros dressing in the touched `build_lazy_pair`.
- **reuse** `_require_nobackup_molhub_home` as the one launcher write — generalize the key it assigns.
- **reuse** `DataModule` fixed-`batch_size` construction — labeled Class-I path; no token budget kwargs.
- **reuse** `_LabeledFrames` copy of `FrameDataset.atom_counts` — molix store packing; out of scope to delete.
- **reuse** `FrameStoreDataset.atom_counts` / `DataMixture.atom_counts` / `FrameStoreReader.atom_counts` — out of scope; different persistence. Do not edit or fork.
- **do not reuse** `_VectorCounts` — deleted; keep/prefix/profile/decode use numpy (or `int(cache_row)` / `int(frame["atoms"].nrows)`) directly.
- **do not reuse** `LazyOMol25.atom_counts` in any form (property, packing field, constructor arg). Operator delete applies to this name on `LazyOMol25`.
- **do not reuse** `_subset_natoms` — deleted; launchers inline the public gather.
- **do not reuse** `DataMixture` in `ClassIFit.datamodule` — that path `getattr`s `atom_counts`.
- **do not reuse** `molhub.cache.Cache()` — no constructor, no import.
- **do not** put `cache()` on `LazyOMol25`.
- **do not reuse** `frame.meta["n_atoms"]` — 09-01 will not write it; grain-one is `nrows`.
- **generalize** `_official_vector_counts` in `scripts/mm_param_learning/class_i_fit.py` to return `np.ndarray` from `source.cache("graph.natoms")`, no discovery wrapper.
- **generalize** `_official_prefix`, `_index_omol_source`, `_lazy_ev`, loaders, and `profile_class_i.build_lazy_pair` onto that numpy and two-arg `LazyOMol25`.
- **generalize** `LazyOMol25` to public `source` / `indices`; `__getitem__` LMDB vs frame by public `record()`; decode vs `numbers.shape[0]` / `frame["atoms"].nrows`.
- **generalize** `_is_labeled_source` to `hasattr(obj, "task_labels")` only.
- **generalize** `ClassIFit.datamodule` to `DataModule(self.train_ds, self.val_ds, batch_size=8)` with no `DataMixture`.
- **generalize** `_e0_cache_path` and `_require_nobackup_molhub_home` to `molhub_config.home_dir`.
- **generalize** `prod_day.run` / `quasi_prod.calibrate` coverage min/max to `np.asarray(view.source.cache("graph.natoms"))[list(view.indices)]`.
- **do not reuse** `espaloma_mm_protocol_train.py` per-conformer `atom_counts` — not a MolHub source index.

Closest pattern: MolHub `loaded.cache("graph.natoms")` as numpy; `LazyOMol25(source, indices)` as a view with those public attributes; `DataModule(dataset, batch_size=…)` without a mix/token policy; molcfg attribute assign `config.home_dir = path` with restore. Error handling: let `source.cache("graph.natoms")` fail naturally (`AttributeError` missing method, MolHub `ValueError` unknown field); decode mismatch is `ValueError`. No factory, no `make_cache`, no `cache()` on the lazy view.

## Files to create or modify

- `scripts/mm_param_learning/class_i_fit.py`
- `scripts/mm_param_learning/profile_class_i.py`
- `scripts/mm_param_learning/prod_day.py`
- `scripts/mm_param_learning/quasi_prod.py`
- `tests/test_scripts/test_mm_param_learning/test_class_i_fit.py`
- `tests/test_scripts/test_mm_param_learning/test_profile_class_i.py`
- `tests/test_scripts/test_mm_param_learning/test_prod_day.py`
- `tests/test_scripts/test_mm_param_learning/test_quasi_prod.py` (new)

## Tasks

- [x] Write failing unit tests for `_official_vector_counts`, `_official_prefix`, `_index_omol_source`, `LazyOMol25`, `_sample_from_record`, `_is_labeled_source`, `ClassIFit.datamodule`, and `_e0_cache_path` (tests/test_scripts/test_mm_param_learning/test_class_i_fit.py → TestOfficialVectorCounts, TestLoadOmol25Official, TestLazyOMol25, TestIsLabeledSource, TestClassIFit, TestE0CachePath); delete TestVectorCounts (rework: public `source`/`indices`, no `atom_counts` in any form, `record()` vs `nrows`, `DataModule(batch_size)` not DataMixture)
- [x] Generalize `_official_vector_counts`, `_official_prefix`, `_index_omol_source`, `_lazy_ev`, loaders, `LazyOMol25`, `_sample_from_record`, `_is_labeled_source`, `ClassIFit.datamodule`, and `_e0_cache_path` in scripts/mm_param_learning/class_i_fit.py onto cache numpy, public `source`/`indices`, and `molhub_config.home_dir`; delete `_VectorCounts` and every `LazyOMol25.atom_counts`; drop the `DataMixture` wrap (rework: no packing field named `atom_counts`; no `_subset_natoms`; decode vs `numbers.shape[0]` / `frame["atoms"].nrows`; LMDB is callable `record()`)
- [x] Write failing unit tests for `build_lazy_pair` (tests/test_scripts/test_mm_param_learning/test_profile_class_i.py → TestBuildLazyPair) (rework: assert public `indices` and cache-at-`source`/`indices`, not `_indices` / `atom_counts` / a third wrap)
- [x] Generalize `build_lazy_pair` in scripts/mm_param_learning/profile_class_i.py to numpy keep `1 ≤ n ≤ max_atoms` and two-arg `LazyOMol25`
- [x] Write failing unit tests for launcher coverage gather and `home_dir` (tests/test_scripts/test_mm_param_learning/test_prod_day.py → TestRequireNobackupMolhubHome; tests/test_scripts/test_mm_param_learning/test_quasi_prod.py → TestCalibrateCoverage) (rework: assert `np.asarray(view.source.cache("graph.natoms"))[list(view.indices)]`, not `_subset_natoms` / `_source` / `_indices`)
- [x] Generalize coverage in scripts/mm_param_learning/prod_day.py and scripts/mm_param_learning/quasi_prod.py to the public source/indices gather; generalize `_require_nobackup_molhub_home` to `molhub_config.home_dir` (rework: delete `_subset_natoms`; inline the gather)
- [x] Add Google-style docstrings with units on `_official_vector_counts`, `_official_prefix`, `_index_omol_source`, `LazyOMol25`, `_sample_from_record`, `_is_labeled_source`, `ClassIFit.datamodule`, and `_e0_cache_path` in scripts/mm_param_learning/class_i_fit.py and on `_require_nobackup_molhub_home` in scripts/mm_param_learning/prod_day.py (rework: name `record()` / `nrows` / public `source` and `indices`; drop `_subset_natoms` and meta `n_atoms`)
- [x] Run full check + test suite

## Testing strategy

Unit tests only, mirrored under `tests/test_scripts/test_mm_param_learning/`. One function per test. Green with `python -m pytest {path} -v` on that file only. No e2e under `tests/`. No FAIR/ASE/MACE/e3nn/lammps/openmm imports in `src/`, `tests/`, or `scripts/`. No `hasattr` in cache-discovery tests, no signature-default asserts, no `setenv("MOLHUB_HOME")`. Tests that assign `molhub_config.home_dir` restore the previous Path. Repo policy deleted `regressions/` (2026-09-07); this slug does not add `regressions/cache-layers-09-frame-dirs-02-wire.py`. Public-API goldens are hard-coded literals in the mirrors below. Tests assert public `source` / `indices`, not private `_source` / `_indices`.

`TestOfficialVectorCounts` (`test_class_i_fit.py`):

- Happy: `_RecordSource.cache("graph.natoms")` returns int64 `[3, 3]` (water-sized, two rows). `_official_vector_counts(source)` is a numpy array; `list(got) == [3, 3]`. There is no `source.atom_counts` call and no `.gather` on the result.
- Edge: an object whose type name is `NoCache` and that has no `cache` method raises `AttributeError` (message may contain `NoCache`). The production body of `_official_vector_counts` contains no `getattr`, `hasattr`, `try`, or `except`.
- Edge: a stub whose `cache` method raises `AttributeError("inner")` still raises `AttributeError` matching `inner`.
- Edge: `scripts/mm_param_learning/class_i_fit.py` does not contain `from molhub.cache import Cache`, `Cache(`, `class _VectorCounts`, or `atom_counts(cache=`. `_VectorCounts` is not importable. There is no `_subset_natoms`.

`TestLoadOmol25Official`:

- Happy: stub `_official_source` with `_RecordSource` whose `cache("graph.natoms")` is `[3, 400]`; stub `_fit_e0_subsample` to zeros so the test does not decode. `load_omol25_official(max_atoms=32)` keeps only row 0; `train.indices == (0,)` and `int(np.asarray(train.source.cache("graph.natoms"))[list(train.indices)][0]) == 3`. Exact filter, not a 512 hint.
- Edge: a `[0, 3]` vector with `max_atoms=32` drops the measured-zero row and keeps the 3-atom row (`indices == (1,)`).
- Edge: `load_omol25_official` still returns `tuple[LazyOMol25, LazyOMol25]` and still requests `val-neutral`.
- Tiny: `_RecordSource` of length 5, counts all 2, `n_train=1` keeps `[0]` and calls `_official_vector_counts` once per source. `LazyOMol25` is constructed without a third positional counts argument.
- Domain: counts are packing bounds, not labels — dropping the 400-atom row does not rewrite energies (`1.0` eV still converts through `molix.units` on the historical path). Official 4M `test_batch_max_atoms_does_not_drop_a_500_atom_row` still holds via cache-at-public-indices (`int(...) == 500`), not `train.atom_counts`.

`TestLazyOMol25`:

- Happy: `LazyOMol25(_RecordSource(energy_ev=1.0), [0], dressing=None, energy_unit=units.eV)` (two positional args). eV construction still keeps raw `1.0` eV as `1.0`; historical default still converts through `molix.units`; `subset()` forwards `energy_unit` and `dressing`. `view.source` is that stub; `view.indices == (0,)`.
- Edge: `view.cache("graph.natoms")` raises `AttributeError`. `view.atom_counts` raises `AttributeError`.
- Edge (LMDB): `_RecordSource` grows a public `record(idx)` (the 01-sources API; not a private `_tables` flag) with `counts=[3]` and decoded `n_atoms=2`; `view[0]` raises `ValueError` naming row `0`, cache count `3`, and decoded `2`. Production `LazyOMol25.__getitem__` source contains no `_tables`.
- Edge (frame): a stub **without** callable `record`, `cache("graph.natoms") == [3]`, `__getitem__` returning a molpy `Frame` whose `atoms` table has `nrows == 2` (e.g. `tests.conftest.frame` with two atoms); `view[0]` raises `ValueError` naming row, cache count `3`, and decoded `2`. Do not set `frame.meta["n_atoms"]`. A frame whose `nrows` matches the cache row does not raise for missing meta `n_atoms`.

`TestIsLabeledSource`:

- Happy: `LazyOMol25` is labeled. `_LabeledRecords` (has `task_labels`) is labeled.
- Edge: a list is not labeled. An object with `atom_counts` and no `task_labels` is not labeled. Production `_is_labeled_source` source is `hasattr(obj, "task_labels")` only (no `atom_counts` in that function).

`TestClassIFit`:

- Happy: `ClassIFit` constructed with two two-arg `LazyOMol25` views sets `train_ds` to the view (labeled path). `fit.datamodule()` does not raise a `ValueError` whose message contains `does not expose atom_counts`. `type(fit.datamodule().train_dataset) is LazyOMol25`. Production `datamodule` source does not construct `DataMixture(` and does not pass `max_atoms_per_batch`.
- Edge: `_LabeledFrames.__init__` still assigns `self.atom_counts = dataset.atom_counts`. `src/molix` is unmodified.

`TestE0CachePath`:

- Happy: assign `molhub_config.home_dir = tmp_path`, restore in finally. `_e0_cache_path(source, max_atoms=32, n_fit=8, seed=2666)` is `tmp_path / "derived" / "omol25-index" / f"{digest}-e0.npy"` with the existing sha1 of `root:len:max_atoms:n_fit:seed`.
- Edge: the path follows the attribute, not `molhub_config["home"]`. Production `_e0_cache_path` / `_require_nobackup_molhub_home` source contains `home_dir` and does not contain `molhub_config["home"]`.

`TestBuildLazyPair` (`test_profile_class_i.py`):

- Happy: fake loaded source with `cache("graph.natoms")` → `[3, 40, 2]`, `max_atoms=32`, `n_train=1`, `n_val=1`. Train `indices == (0,)`, val `indices == (2,)`. `int(np.asarray(train.source.cache("graph.natoms"))[list(train.indices)][0]) == 3`. `LazyOMol25` captured args have length 2 (no third wrap).
- Edge: vector `[0, 3, 40, 2]` drops measured 0; train is the 3-atom row (`indices == (1,)`), val is the 2-atom row (`indices == (3,)`).
- Edge: production `build_lazy_pair` does not call `src.atom_counts`, uses `_historical_are`, and does not construct `AtomicReferenceEnergy(` directly. `run_profile` source calls `_require_nobackup_molhub_home` before `build_lazy_pair`.

`TestRequireNobackupMolhubHome` (`test_prod_day.py`):

- Happy: monkeypatch `_DEFAULT_MOLHUB_HOME` to `tmp_path`. Save `molhub_config.home_dir` and `os.environ.get("MOLHUB_HOME")`. Call `_require_nobackup_molhub_home`. Then `molhub_config.home_dir == tmp_path`, `tmp_path.is_dir()`, and env is unchanged. Restore the singleton.
- Edge (static): `prod_day.run` source contains `source.cache("graph.natoms")` and `indices`, and does not contain `train.atom_counts` / `val.atom_counts` / `_subset_natoms`. `scripts/mm_param_learning/*.py` and the mirrored tests contain no `molhub_config["home"]`.

`TestCalibrateCoverage` (`test_quasi_prod.py`):

- Edge (static): `quasi_prod.calibrate` source contains `source.cache("graph.natoms")` and `indices`, and does not contain `train.atom_counts` / `val.atom_counts` / `_subset_natoms`.
- Happy (unit, no launcher): a two-arg `LazyOMol25` over indices `(0, 2)` of cache `[3, 40, 2]` yields `list(np.asarray(view.source.cache("graph.natoms"))[list(view.indices)]) == [3, 2]` (min `2`, max `3`). Empty `indices` yields an empty array.

Static: `scripts/mm_param_learning/` and `tests/test_scripts/test_mm_param_learning/` contain no `ase`, `e3nn`, `mace`, `fairchem`, `lammps`, or `openmm` imports.

## Out of scope

- Editing `src/molix` or molhub production (`MolhubConfig`, `Cache.lookup` / `publish`, frame-dir writers, `home_dir` definition, `OMol25.record` — that attribute and that method are 09-01 / 01-sources).
- Writing `frame.meta["n_atoms"]` in molhub (09-01 will not write it).
- Deleting `_LabeledFrames.atom_counts` or `FrameDataset.atom_counts` / `FrameStoreReader.atom_counts` / `DataMixture.atom_counts` / `molix.data.sampler` count budgets.
- Putting `cache()` on `LazyOMol25` or on `dataset.protocol.Dataset`.
- Constructing `molhub.cache.Cache()`.
- Deleting `_fit_e0_subsample` / `fit_atomic_reference` / historical kcal conversion / `load_omol25_official` as a named loader.
- Deleting `ClassIFit.max_atoms` as a construction field.
- Retargeting `prod_day.py` / `quasi_prod.py` / `overfit_gate.py` / `align.py` / `profile_class_i.py` onto `load_omol25_fairchem_4m`.
- `espaloma_mm_protocol_train.py` per-conformer `atom_counts`.
- Admitting fields other than `"graph.natoms"`; aliasing old `quantity="n_atoms"` seals.
- TypeScript `$MOLHUB_HOME/files/` (`molhub` `coordinate.py`).
- A `regressions/` script (deleted 2026-09-07).
- Production job submission.
