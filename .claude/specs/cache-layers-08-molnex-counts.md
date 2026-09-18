---
title: Molnex scripts read loaded.cache("graph.natoms"); Python home is molhub.config
status: code-complete
grilled: true
created: 2026-09-12
revised: 2026-09-12
chain: cache-layers
depends:
  - cache-layers-02-cache
  - cache-layers-07-protocol
supersedes: cache-layers-08-molnex-counts
---

# Molnex scripts read loaded.cache("graph.natoms"); Python home is molhub.config

## Summary

Molnex Class-I OMol scripts read per-row atom counts from a loaded MolHub view with `source.cache("graph.natoms")` and wrap that fully-known vector in `_VectorCounts` (`gather` plus `__len__`, no `record`). Public molnex calls to `atom_counts` and `atom_counts(cache=Cache())` go away, `AtomCountCache` and the `root:len` npy sidecar die with them, and every `max_atoms` keep this slug applies is the exact predicate `1 ≤ n ≤ max_atoms`. A decode that disagrees with the vector raises rather than overwriting. Process home is `molhub.config["home"]` assigned by `prod_day._require_nobackup_molhub_home`; sbatch and shell launchers no longer export `MOLHUB_HOME`. `LazyOMol25.atom_counts` stays as the in-memory subset of that immutable vector and does not grow a `cache()` method.

## Domain basis

Count snapshots are not scientific labels: they do not define chemistry subsets, change energies or forces, or stand in for measured bounds (cache-layers-02-cache). This slug does not introduce a physical model, conversion constant, or DOI.

## Design

Scope is molnex `scripts/mm_param_learning` and `tests/test_scripts/test_mm_param_learning` (plus the three launchers that still export `MOLHUB_HOME`). Do not edit `src/molix`, molhub production (cache-layers-02-cache already landed), `FrameStoreDataset.atom_counts`, or `DataMixture.atom_counts`. `scripts/` may import molhub; `src/` still must not.

Current callers: `_official_vector_counts` (official 4M), `load_omol25_official` / `load_omol25_official_tiny` / `_index_omol_source` / `_official_prefix` (historical kcal), `_e0_cache_path` / `_fit_e0_subsample`, `profile_class_i.build_lazy_pair`, `prod_day._require_nobackup_molhub_home` (also imported by `quasi_prod.py` / `align.py` / `scale_audit.py` / `overfit_gate.py`), `run_class_i_prod_day_gh200.sbatch`, `run_class_i_profile_gh200.sbatch`, `run_official_smoke.sh`. Current tests: `TestAtomCountCache`, `TestLoadOmol25Fairchem4m`, `TestLazyOMol25`, `_RecordSource.atom_counts`, `_patch_official_sources` boom on `_n_atoms_cache_path`.

### Public count read

MolHub's public verb is `loaded.cache("graph.natoms") -> np.ndarray` (int64, shape `(N,)`, fully known). Molnex does not construct `molhub.cache.Cache` and does not pass `store=` / `workers=` / `cache=`. Delete molnex calls to `atom_counts` and `atom_counts(cache=Cache())`. Drop `from molhub.cache import Cache`.

`_official_vector_counts(source)` is the one wrap. Call `source.cache("graph.natoms")` with no discovery wrapper and wrap the array. The function body has no `getattr`, no `hasattr`, no `try`/`except` (including `except AttributeError`). Missing `cache` is Python `AttributeError` and propagates. Do not translate it to `TypeError`. An `AttributeError` raised *inside* a real `cache` also propagates — it is not “has no cache”. Unknown field names are MolHub's `ValueError` and propagate. `"n_atoms"` and molnex `"graphs.num_atoms"` stay unknown.

```python
def _official_vector_counts(source: Any) -> _VectorCounts:
    return _VectorCounts(np.asarray(source.cache("graph.natoms")))
```

Each load of one source calls `_official_vector_counts` **once** (a Cache miss still scans; a second call would double that work).

### Immutable wrap

Generalize `_VectorCounts` in place as the immutable RAM wrap of that fully-known vector. Keep `gather` and the int32 copy. Add `__len__` returning the row count. Delete `record`. `_VectorCounts` is the only type that names the numpy storage attribute `_counts`. Callers use `len(counts)` and `counts.gather(...)` only.

`LazyOMol25._counts` remains the wrap *instance* passed at construction (the `gather` target), not the numpy buffer. Do not read `counts._counts` from `_official_prefix`, `_index_omol_source`, `build_lazy_pair`, or tests.

Do not add `cache()` on `LazyOMol25` — that would be a sibling of the source verb. `LazyOMol25.atom_counts` is the in-memory subset of the immutable vector: `torch.as_tensor(self._counts.gather(self._indices), dtype=torch.long)`. It is not a live hint cache and is not “as measured so far”.

`_sample_from_record` must not call `record`. After decode, if `row is not None`, compare `int(self._counts.gather((row,))[0])` to `int(numbers.shape[0])`. On disagreement raise `ValueError` naming the row, the cache count, and the decoded count. Do not silently overwrite.

### Historical loaders use the same vector

Delete `AtomCountCache`, `_n_atoms_cache_path`, `_omol_atom_counts`, and `ATOM_COUNT_HINT`. There is no packing hint and no `derived/omol25-index/*-n_atoms.npy` sidecar. Unmeasured rows no longer exist: the vector is fully known, so a keep is exact.

One keep predicate everywhere this slug filters the vector:

```text
1 <= n <= max_atoms
```

A measured zero is dropped (`n >= 1`); it is not a hint. Negatives are dropped by the same predicate. Sites: `_index_omol_source`, `_official_prefix`, `build_lazy_pair`. `load_omol25_fairchem_4m` still does not filter by `batch_max_atoms`.

Generalize `_index_omol_source` to `_official_vector_counts` once, then `n_atoms = counts.gather(range(len(counts)))` (or equivalent `np.arange(len(counts))`), then `keep = rows with 1 ≤ n ≤ max_atoms`. Return type is `tuple[list[int], _VectorCounts, np.ndarray]` (keep, vector, ε). `load_omol25_official` still fits ε through `_fit_e0_subsample` and still returns `tuple[LazyOMol25, LazyOMol25]` with kcal conversion and `val_role="val-neutral"`.

Generalize `_official_prefix` to operate on an already-wrapped `_VectorCounts` (no LMDB decode, no `record`):

```python
def _official_prefix(counts: _VectorCounts, n: int, max_atoms: int, scan_cap: int) -> list[int]:
    limit = min(len(counts), int(scan_cap))
    n_atoms = counts.gather(range(limit))
    keep: list[int] = []
    for i in range(limit):
        if len(keep) >= n:
            break
        size = int(n_atoms[i])
        if 1 <= size <= int(max_atoms):
            keep.append(i)
    return keep
```

`load_omol25_official_tiny` calls `_official_vector_counts` once per source, prefixes from that vector, and passes the **same** vector into `LazyOMol25`. Do not call `_official_vector_counts` again for the constructor.

### Home is molhub.config["home"]

`class_i_fit.py` already imports `molix.config` as `config`. Alias MolHub at module level:

```python
from molhub.config import config as molhub_config
```

`_e0_cache_path` reads `molhub_config["home"]` (not `os.environ`). Path is `home / "derived" / "omol25-index" / f"{digest}-e0.npy"` with the existing sha1 of `root:len:max_atoms:n_fit:seed`. Keep `_fit_e0_subsample`. After this slug, `os` is unused in `class_i_fit.py` and is removed.

`prod_day._require_nobackup_molhub_home` is the one Python home write for Class-I launchers. It `mkdir`s the nobackup workspace path (`REPO_ROOT.parent / "workspaces" / "molhub-home"`) and assigns `molhub_config["home"] = that Path`. It does not read or write `os.environ`. It does not skip when `$MOLHUB_HOME` is set — env is not a Python home. After this slug `os` is unused in `prod_day.py` and is removed. `quasi_prod` / `align` / `scale_audit` / `overfit_gate` keep calling this helper and need no source edit.

`profile_class_i.run_profile` calls `_require_nobackup_molhub_home` before `build_lazy_pair` (that is before `OMol25(...).load()`). `build_lazy_pair` uses `_official_vector_counts(src)` (not `src.atom_counts()`, not a list of ints as the third `LazyOMol25` argument). It gathers `range(len(counts))`, keeps `1 ≤ n ≤ max_atoms`, and passes the **same** `_VectorCounts` instance into both `LazyOMol25` views. The dense zeros dressing in that same function currently omits `supported=` and cannot construct after cache-layers-03-refs; replace it with `_historical_are(torch.zeros(95))` while rewriting the function. Drop the unused `AtomicReferenceEnergy` import. Do not retarget the profile onto `load_omol25_fairchem_4m`.

Sbatch and shell: remove `export MOLHUB_HOME`, the `mkdir -p "${MOLHUB_HOME}"` that exists only to back that export, and echoes of `MOLHUB_HOME=...`. Python entry sets home. After this slug, `scripts/mm_param_learning` has no live `os.environ.get("MOLHUB_HOME")`, `os.environ["MOLHUB_HOME"]`, or `export MOLHUB_HOME`. `_DEFAULT_MOLHUB_HOME` as a Path constant is not an env read.

### Tests

`_RecordSource.cache(field)` returns the int64 vector for `"graph.natoms"` and does not define `atom_counts`. Accept an optional per-row `counts=` array so `[3, 400]` and measured-zero fixtures do not need `hasattr`. Unknown fields raise `ValueError`. Delete `TestAtomCountCache`. `_patch_official_sources` must not patch `_n_atoms_cache_path`. Tests that assign `molhub_config["home"]` restore the previous Path (try/finally or equivalent). Tests do not `setenv("MOLHUB_HOME")` and do not use `hasattr`. A stub whose type name is `NoCache` and that has no `cache` method raises `AttributeError` (message may contain `NoCache`). `TestLazyOMol25` constructs with `_VectorCounts`, not `AtomCountCache`. Official-pair tests exercise real `_index_omol_source` (stub `_fit_e0_subsample` so they do not decode); do not stub the indexer with a deleted cache type. `test_omol25_fairchem_4m.py` needs no edit once `_RecordSource` grows `cache("graph.natoms")`.

New mirrors: `tests/test_scripts/test_mm_param_learning/test_profile_class_i.py` (`TestBuildLazyPair`) and `test_prod_day.py` (`TestRequireNobackupMolhubHome`). Helper tests monkeypatch `_DEFAULT_MOLHUB_HOME` to `tmp_path` so they do not mkdir the production workspace.

### Reuse decision

- **reuse** `OMol25.cache` / `OMol25Lmdb.cache` / packed `OMol25Dataset.cache` — spec calls `source.cache("graph.natoms")`; molnex does not reimplement lookup/publish or construct `Cache`.
- **reuse** `molhub.config["home"]` — the Python home; assignment is the write path.
- **generalize** `_VectorCounts` in `scripts/mm_param_learning/class_i_fit.py` to an immutable wrap with `__len__` and `gather` only (delete `record`; no `._counts` outside the type). Do not add a second count type.
- **reuse** `LazyOMol25.atom_counts` — subset projection of that immutable vector; do not put `cache()` on `LazyOMol25`.
- **reuse** `_fit_e0_subsample` — historical ε fit stays; only its on-disk root generalizes.
- **reuse** `_historical_are` — profile zeros dressing in the touched `build_lazy_pair`.
- **reuse** `_require_nobackup_molhub_home` as the one launcher write — `quasi_prod` / `align` / `scale_audit` / `overfit_gate` already call it; `run_profile` joins them.
- **reuse** `FrameStoreDataset.atom_counts` / `DataMixture.atom_counts` / `FrameStoreReader.atom_counts` — out of scope; different persistence (molix packed frames, not MolHub source indexes). Do not edit or fork.
- **reuse** `tests/test_scripts/test_mm_param_learning/test_omol25_fairchem_4m.py` as-is — stays green after `_RecordSource.cache`; `_LabeledRecords.atom_counts` is the molix packing tensor, not a MolHub source index.
- **do not reuse** `AtomCountCache` / `_n_atoms_cache_path` / `_omol_atom_counts` / `ATOM_COUNT_HINT` — deleted; they were a sibling of `loaded.cache("graph.natoms")`.
- **generalize** `_official_vector_counts` in `scripts/mm_param_learning/class_i_fit.py` to `source.cache("graph.natoms")` → `_VectorCounts`, no `getattr`/`hasattr`/`try`/`except`, no `TypeError` translation, no `Cache()` constructor. Missing `cache` is `AttributeError`.
- **generalize** historical `load_omol25_official` / `load_omol25_official_tiny` / `_index_omol_source` / `_official_prefix` onto that same vector (once per source); exact `1 ≤ n ≤ max_atoms` keep; prefix uses `len(counts)` then `gather(range(limit))`.
- **generalize** `_e0_cache_path` from `molhub_config["home"] / "derived" / "omol25-index"`.
- **generalize** `profile_class_i.build_lazy_pair` to `_official_vector_counts` and the same `1 ≤ n ≤ max_atoms` keep; `run_profile` to `_require_nobackup_molhub_home`.
- **generalize** `prod_day._require_nobackup_molhub_home` to assign `molhub_config["home"]` (mkdir, no setenv).
- **generalize** sbatch/shell launchers by deleting `export MOLHUB_HOME`.
- **generalize** test stubs: `_RecordSource.cache(field)`, delete `TestAtomCountCache`, restore `molhub_config["home"]`, no `setenv("MOLHUB_HOME")`.

Closest pattern: MolHub `loaded.cache("graph.natoms")` plus the existing `_VectorCounts` wrap with `__len__` so callers never leak storage; home assignment matching cache-layers-02 isolators (`config["home"] = path` and restore). Error handling: let `source.cache("graph.natoms")` fail naturally (`AttributeError` missing method, MolHub `ValueError` unknown field) — do not invent a `TypeError` discovery wrapper. No factory, no `make_cache`, no `cache()` on the lazy view.

## Files to create or modify

- `scripts/mm_param_learning/class_i_fit.py`
- `scripts/mm_param_learning/profile_class_i.py`
- `scripts/mm_param_learning/prod_day.py`
- `scripts/mm_param_learning/run_class_i_prod_day_gh200.sbatch`
- `scripts/mm_param_learning/run_class_i_profile_gh200.sbatch`
- `scripts/mm_param_learning/run_official_smoke.sh`
- `tests/test_scripts/test_mm_param_learning/test_class_i_fit.py`
- `tests/test_scripts/test_mm_param_learning/test_profile_class_i.py` (new)
- `tests/test_scripts/test_mm_param_learning/test_prod_day.py` (new)

## Tasks

- [x] Write failing unit tests for `_official_vector_counts`, `_VectorCounts`, historical loaders, `_e0_cache_path`, and `LazyOMol25` (tests/test_scripts/test_mm_param_learning/test_class_i_fit.py → TestOfficialVectorCounts, TestVectorCounts, TestLoadOmol25Official, TestE0CachePath, TestLazyOMol25); delete TestAtomCountCache; retarget `_RecordSource.cache(field)` (rework: AttributeError on NoCache, not TypeError; `__len__` / no `._counts` leak; no `record`; gather/decode disagreement raises)
- [x] Generalize `_VectorCounts`, `_official_vector_counts`, `load_omol25_official`, `load_omol25_official_tiny`, `_index_omol_source`, `_official_prefix`, and `_sample_from_record` in scripts/mm_param_learning/class_i_fit.py onto `source.cache("graph.natoms")`; add `_VectorCounts.__len__`; delete `record`, `AtomCountCache`, `_n_atoms_cache_path`, `_omol_atom_counts`, and `ATOM_COUNT_HINT`; alias `from molhub.config import config as molhub_config`; generalize `_e0_cache_path` from `molhub_config["home"] / "derived" / "omol25-index"`; drop `from molhub.cache import Cache` and `os` (rework: no try/except around cache; prefix/indexer use `len`+`gather`; decode mismatch raises; no TypeError translation)
- [x] Write failing unit tests for `build_lazy_pair` (tests/test_scripts/test_mm_param_learning/test_profile_class_i.py → TestBuildLazyPair) (rework: drop measured 0 with `1 ≤ n ≤ max_atoms`, same as TestLoadOmol25Official)
- [x] Generalize `build_lazy_pair` and `run_profile` in scripts/mm_param_learning/profile_class_i.py to `_official_vector_counts` and `_require_nobackup_molhub_home`; keep `1 ≤ n ≤ max_atoms`; replace the dense zeros dressing with `_historical_are(torch.zeros(95))` (rework: same keep predicate as historical loaders, not `n <= max_atoms`)
- [x] Write failing unit tests for `_require_nobackup_molhub_home` and launcher env (tests/test_scripts/test_mm_param_learning/test_prod_day.py → TestRequireNobackupMolhubHome)
- [x] Generalize `_require_nobackup_molhub_home` in scripts/mm_param_learning/prod_day.py to assign `molhub_config["home"]` (mkdir, no setenv); remove `export MOLHUB_HOME` from scripts/mm_param_learning/run_class_i_prod_day_gh200.sbatch, scripts/mm_param_learning/run_class_i_profile_gh200.sbatch, and scripts/mm_param_learning/run_official_smoke.sh
- [x] Add Google-style docstrings with units on `_official_vector_counts`, `_VectorCounts`, `_e0_cache_path`, `LazyOMol25.atom_counts`, and `_sample_from_record` in scripts/mm_param_learning/class_i_fit.py and on `_require_nobackup_molhub_home` in scripts/mm_param_learning/prod_day.py
- [x] Run full check + test suite

## Testing strategy

Unit tests only, mirrored under `tests/test_scripts/test_mm_param_learning/`. One function per test. Green with `python -m pytest {path} -v` on that file only. No e2e under `tests/`. No FAIR/ASE/MACE/e3nn/lammps/openmm imports in `src/`, `tests/`, or `scripts/`. No `hasattr`, no signature-default asserts, no `setenv("MOLHUB_HOME")`. Tests that assign `molhub_config["home"]` restore the previous Path. Repo policy deleted `regressions/` (2026-09-07); this slug does not add `regressions/cache-layers-08-molnex-counts.py`. Public-API goldens are hard-coded literals in the mirrors below.

`TestOfficialVectorCounts` (`test_class_i_fit.py`):

- Happy: `_RecordSource.cache("graph.natoms")` returns int64 `[3, 3]` (water-sized, two rows). `_official_vector_counts(source).gather([0, 1])` is `[3, 3]`. There is no `source.atom_counts` call in the production path.
- Edge: an object whose type name is `NoCache` and that has no `cache` method raises `AttributeError` (message may contain `NoCache`). The production body of `_official_vector_counts` contains no `getattr`, `hasattr`, `try`, or `except`.
- Edge: a stub whose `cache` method raises `AttributeError("inner")` still raises `AttributeError` matching `inner` — not a translated `TypeError`.
- Edge: `scripts/mm_param_learning/class_i_fit.py` does not contain `from molhub.cache import Cache` or `atom_counts(cache=`.

`TestVectorCounts`:

- Happy: `len(_VectorCounts(np.array([3, 3], dtype=np.int32))) == 2`. `gather(range(2))` is `[3, 3]`.
- Edge: `counts.record(0, 1)` raises `AttributeError`. Production `_official_prefix` / `_index_omol_source` source text does not contain `._counts`.

`TestLoadOmol25Official`:

- Happy: stub `_official_source` with `_RecordSource` whose `cache("graph.natoms")` is `[3, 400]`; stub `_fit_e0_subsample` to zeros so the test does not decode. `load_omol25_official(max_atoms=32)` keeps only row 0; `int(train.atom_counts[0]) == 3`. Exact filter, not a 512 hint.
- Edge: a `[0, 3]` vector with `max_atoms=32` drops the measured-zero row and keeps the 3-atom row.
- Edge: `load_omol25_official` still returns `tuple[LazyOMol25, LazyOMol25]` and still requests `val-neutral`. `AtomCountCache` / `ATOM_COUNT_HINT` are not importable from `class_i_fit`.
- Tiny: `_RecordSource` of length 5, counts all 2, `n_train=1` keeps `[0]` and calls `_official_vector_counts` once per source.
- Domain: counts are packing bounds, not labels — dropping the 400-atom row does not rewrite energies. Official 4M `test_batch_max_atoms_does_not_drop_a_500_atom_row` still holds (`load_omol25_fairchem_4m` does not filter by `batch_max_atoms`).

`TestE0CachePath`:

- Happy: assign `molhub_config["home"] = tmp_path`, restore in finally. `_e0_cache_path(source, max_atoms=32, n_fit=8, seed=2666)` is `tmp_path / "derived" / "omol25-index" / f"{digest}-e0.npy"` with the existing sha1 of `root:len:max_atoms:n_fit:seed`.
- Edge: `os.environ["MOLHUB_HOME"]` is not read (do not setenv in the test; the path follows the singleton, not `Path.home()/".cache"/"molhub"` when the singleton is `tmp_path`).

`TestLazyOMol25`:

- Happy: construct with `_VectorCounts(np.array([2], dtype=np.int32))` (not `AtomCountCache`). eV construction still keeps raw `1.0` eV as `1.0`; historical default still converts through `molix.units`; `subset()` forwards `energy_unit` and `dressing`.
- Edge: `view.cache("graph.natoms")` raises `AttributeError` — `LazyOMol25` has no `cache` method.
- Edge: construct with `_VectorCounts(np.array([3], dtype=np.int32))` over a record that decodes 2 atoms; `view[0]` raises `ValueError` naming row, cache count 3, and decoded 2. The wrap is unchanged (no overwrite).

`TestBuildLazyPair` (`test_profile_class_i.py`):

- Happy: fake loaded source with `cache("graph.natoms")` → `[3, 40, 2]`, `max_atoms=32`, `n_train=1`, `n_val=1`. Train indices `[0]`, val indices `[2]` (the 40-atom row is skipped). `int(train.atom_counts[0]) == 3`. Third argument to `LazyOMol25` is `_VectorCounts`, not a Python list, and both views share that instance.
- Edge: vector `[0, 3, 40, 2]` with the same `max_atoms`/`n_train`/`n_val` drops measured 0; train is the 3-atom row, val is the 2-atom row.
- Edge: production `build_lazy_pair` does not call `src.atom_counts`, uses `_historical_are`, and does not construct `AtomicReferenceEnergy(` directly. `run_profile` source calls `_require_nobackup_molhub_home` before `build_lazy_pair`.

`TestRequireNobackupMolhubHome` (`test_prod_day.py`):

- Happy: monkeypatch `_DEFAULT_MOLHUB_HOME` to `tmp_path`. Save `molhub_config["home"]` and `os.environ.get("MOLHUB_HOME")`. Call `_require_nobackup_molhub_home`. Then `molhub_config["home"] == tmp_path`, `tmp_path.is_dir()`, and `os.environ.get("MOLHUB_HOME")` equals the saved value (helper does not setenv). Restore the singleton.
- Edge: the helper does not skip assignment when a leftover `$MOLHUB_HOME` would previously have caused an early return — env is not a Python home. Prove by monkeypatching the default path, not by setenv as the write under test.
- Edge (static): `scripts/mm_param_learning/run_class_i_prod_day_gh200.sbatch`, `run_class_i_profile_gh200.sbatch`, and `run_official_smoke.sh` contain no `export MOLHUB_HOME`. `scripts/mm_param_learning/*.py` contain no `os.environ.get("MOLHUB_HOME")` and no `os.environ["MOLHUB_HOME"]`.

Static: `scripts/mm_param_learning/` and `tests/test_scripts/test_mm_param_learning/` contain no `ase`, `e3nn`, `mace`, `fairchem`, `lammps`, or `openmm` imports.

## Out of scope

- Editing `src/molix`, molhub production code, or cache-layers-02-cache types (`MolhubConfig`, `Cache.lookup` / `publish`, `CountIdentity`).
- `FrameStoreDataset.atom_counts`, `FrameStoreReader.atom_counts`, `DataMixture.atom_counts`, `molix.data.sampler` count budgets — different persistence; not MolHub source indexes.
- Putting `cache()` on `LazyOMol25` or on `dataset.protocol.Dataset`.
- Deleting `_fit_e0_subsample` / `fit_atomic_reference` / historical kcal conversion / `load_omol25_official` as a named loader.
- Retargeting `prod_day.py` / `quasi_prod.py` / `overfit_gate.py` / `align.py` / `profile_class_i.py` onto `load_omol25_fairchem_4m`.
- Wiring `omol25_fairchem_4m.run` to `_require_nobackup_molhub_home` (no current launcher exports `MOLHUB_HOME` for that entry; the helper is reusable when a later slug needs it).
- Changing `_is_labeled_source` / `ClassIFit` labeled-vs-frames detection.
- `espaloma_mm_protocol_train.py` per-conformer `atom_counts` (not a MolHub source index).
- Admitting fields other than `"graph.natoms"`; aliasing old `quantity="n_atoms"` seals; auto-deleting leftover `.molhub-n-atoms-*.npy`.
- TypeScript `$MOLHUB_HOME/files/` (`molhub` `coordinate.py`).
- A `regressions/` script (deleted 2026-09-07).
- Production job submission.
