---
title: MolhubConfig.home_dir; grain-one is atoms.nrows
status: code-complete
grilled: true
created: 2026-09-12
chain: cache-layers
depends: cache-layers-02-cache
supersedes: cache-layers-09-frame-dirs-01-types
---

# MolhubConfig.home_dir; grain-one is atoms.nrows

## Summary

Molhub process home is the pathlib attribute `molhub.config.home_dir`, default `Path.home() / ".cache" / "molhub"`. Callers assign `molhub.config.home_dir = Path(...)`. There is no `config["home"]` key and no `config.home`. `BlobStore`, `DownloadCache`, `Cache`, CLI `--home`, and `build_class1_splits --molhub-home` read and write `config.home_dir`; `Cache()` default root is `config.home_dir / "counts"`. Tests restore `config.home_dir`. Isolators assign `config.home_dir`, not `config["home"]`. Grain-one on a decoded OMol25 frame is `int(frame["atoms"].nrows)` (LMDB: that equals `len(numbers)` after `_checked_numbers`; packed: that equals `int(self._n_atoms[idx])`). `cache("graph.natoms")` remains the bulk vector of the same quantity; tests assert `int(frame["atoms"].nrows) == int(cache("graph.natoms")[idx])`. This slug does not write `n_atoms` onto `frame.meta` and does not invent `graphs.n_atoms`. There is no `atom_counts` method. `$MOLHUB_HOME` is not a Python home.

## Domain basis

Per-frame atom count and `cache("graph.natoms")` are packing bounds, not scientific labels: they do not define chemistry subsets, change energies or forces, or stand in for measured bounds (cache-layers-02-cache). The count is dimensionless. `home_dir` is a directory `Path`, not a physical unit. This slug does not introduce a physical model, conversion constant, or DOI.

## Design

Paths below are in the sibling molhub repository. Scope is molhub only. Do not edit molnex or `src/molix`. Do not change `Cache.lookup` / `Cache.publish` / `_seal` / `disable_writes` / `CountConflict` / retain-one. Do not change `CountIdentity.quantity` (`"graph.natoms"` only). Do not revive `$MOLHUB_HOME` as a Python home. Do not write `n_atoms` onto `frame.meta` (neither whole-table `frame.meta = {…, "n_atoms": …}` nor item assignment `frame.meta["n_atoms"] = …`). Do not add `graphs.n_atoms`.

Current callers of the home singleton: `BlobStore._resolve_root`, `DownloadCache._resolve_root`, `Cache.__init__`, `Fetcher()` → `BlobStore()`, `Molhub()` → `Fetcher()`, `DatasetBuilder(home=None)` → `BlobStore(home)`, CLI `_apply_home` / `--home` (including `molhub fetch --home`), `python -m molhub.scripts.build_class1_splits --molhub-home`. Current tests: `TestConfig`, `TestBlobStoreRoot`, `TestCache`, `TestDownloadCache`, `TestCacheLayering`, CLI home tests, `TestTierStore`, isolators that assign `config["home"]`. Current frame builders (unchanged production): `_frame_from_record` (called by `OMol25Lmdb.__getitem__`) and packed `OMol25Dataset.__getitem__`.

This is one L0 field rename plus a grain-one identity lock on existing `Block.nrows`. The file list is the existing reader set from cache-layers-02-cache, landed in the same slug so the suite never has two home names. Constructor injection stays: an explicit `root=` / `home=` / `blobs=BlobStore(PATH)` still wins and does not require writing the singleton first.

### MolhubConfig.home_dir (L0)

Keep `MolhubConfig(molcfg.Config)` with `__slots__ = ()` and the process singleton. Change the one stored field from `"home"` to `"home_dir"`:

```python
config = MolhubConfig(
    {
        "home_dir": Path.home() / ".cache" / "molhub",
    }
)
```

Public access is the molcfg attribute, not a dict key named `home` and not a second attribute named `home`:

```python
from molhub.config import config

config.home_dir                          # Path
config.home_dir = Path(...)              # write path
```

`config.py` still imports only `pathlib` and `molcfg`. No extra methods. No `writable` key. No factory. Do not re-export from `molhub/__init__.py`.

Because molcfg stores attributes in `_data`, `config["home_dir"]` is the same Path. Production code and tests use the attribute `config.home_dir`. They must not use `config["home"]` or `config.home`. `"home" not in config`. `config["home"]` is `KeyError`. `config.home` (read) is `AttributeError` (`Config has no attribute 'home'`). Do not add a property, alias, or `__getitem__` override to hide molcfg item access on the new name. Do not assign `config.home = …` (molcfg `__setattr__` would create a second key).

There is still no `os.environ` / `$MOLHUB_HOME` / `$MOLHUB_CACHE_DIR` for home. Do not migrate `$MOLHUB_REGISTRY`, `FIGSHARE_TOKEN`, `MOLHUB_SOURCE_BASE`, or `MOLNEX_SRC`. `Cache.__init__(root=None, *, writable=True)` stays the write switch; do not default `writable` from the singleton.

### Who reads the singleton

`root=None` / `home=None` / `blobs=None` reads `config.home_dir` at construction.

| Caller | Default (`None`) | Explicit |
|---|---|---|
| `BlobStore(root=None)` | `config.home_dir` | `BlobStore(root=PATH)` and `BlobStore(PATH)` / `BlobStore(home)` still for tests and CLI |
| `DownloadCache(root=None)` | `config.home_dir` (`urls/` under it) | `DownloadCache(root=tmp)` still for tests. Does not call `BlobStore()` to resolve home |
| `Cache(root=None, *, writable=True)` | root `config.home_dir / "counts"`; writable is the constructor flag | `Cache(root=tmp)` is the snapshot tree |
| `Fetcher(blobs=None)` | `BlobStore()` → `config.home_dir` | `Fetcher(blobs=BlobStore(PATH))` |
| `Molhub(..., fetcher=None)` | `Fetcher()` → `BlobStore()` | `Molhub(registry, fetcher=Fetcher(blobs=BlobStore(PATH)))` |
| `DatasetBuilder(home=None)` | `BlobStore(home).root` already; `home=None` follows `config.home_dir` | `DatasetBuilder(home=PATH)` |
| typer `--home` on `info` / `cache verify` / `cache status` / `cache coverage` / `fetch` | leave the singleton; construct with `root=None` | write `config.home_dir = Path(home).expanduser()` **and** pass the path |
| `python -m molhub.scripts.build_class1_splits --molhub-home` | argparse default `None` meaning `config.home_dir` | write `config.home_dir` and `Fetcher(blobs=BlobStore(home))` |

CLI `--home` help must not say `$MOLHUB_HOME`. Private `_apply_home(home: Path | None)` stays; it assigns `config.home_dir`. Do not forbid `BlobStore(home)`. After this slug, a grep of `src/molhub` for `config["home"]` and for `config.home` that is not `config.home_dir` is empty.

Isolators assign `config.home_dir` and restore the previous Path; they must not assign `config["home"]` and must not `setenv("MOLHUB_HOME")`. `tests/test_docs_examples.py` `fixture-home` preamble becomes `config.home_dir = Path(...)`. `TestConfig` / `TestBlobStoreRoot` may `setenv("MOLHUB_HOME")` only as negative proofs that env does not move `config.home_dir` or `BlobStore()`. `TestConfig` may name `config["home"]` / `config.home` only as those negative proofs (`KeyError` / `AttributeError` / `"home" not in config`).

TypeScript `$MOLHUB_HOME/files/` in `coordinate.py` stays out of scope.

### Grain-one is `atoms.nrows`; bulk is `cache("graph.natoms")`

Do not add `atom_counts` on `OMol25Lmdb`, `OMol25`, or `OMol25Dataset`. `cache("graph.natoms")` is unchanged: LMDB/façade still `lookup`/`publish` through `store=` / `workers=`; packed still returns `self._n_atoms.copy()` and refuses extra kwargs; `"n_atoms"` as a **cache field** still raises `ValueError`. Packed NPZ column `n_atoms` (the `_REQUIRED` array, `self._n_atoms`) stays the packed storage for that bulk vector. It is not a `frame.meta` key.

Grain-one on a decoded molpy `Frame` is already the atom-block row count. Tests lock the identity; `__getitem__` production does not grow a parallel count field.

```text
int(frame["atoms"].nrows) == int(cache("graph.natoms")[idx])
```

LMDB: `_frame_from_record` builds `atoms` from `numbers = _checked_numbers(record["numbers"])`, so `nrows` equals `len(numbers)`. Packed: `__getitem__` slices `[self._ptr[idx]:self._ptr[idx+1]]` from the NPZ columns, so `nrows` equals `int(self._n_atoms[idx])`. Do not use `len(frame["atoms"])` (`len(Block)` is column count). Do not write `n_atoms` onto `frame.meta`. Do not invent `graphs.n_atoms` (molnex collate already owns `graphs.num_atoms` from `Z.shape[0]`; FrameStore already reserves layout `n_atoms.bin`). `OMol25.__getitem__` still delegates to the loaded LMDB view.

Water-sized fixture is 3 (`_water` / `_water_record` numbers `[8, 1, 1]`). Packed two-row `[3, 5]` (`_water` plus `_n_atom_record(5)`): row 1 is 5. `OMol25Small._bonded` already copies every meta key; it needs no extra write. `OMol25SmallFilter.reason` may keep counting from `numbers`.

### Tests (owned seams)

No `hasattr`, no signature-default asserts, no e2e under `tests/`. `TestConfig` owns the attribute, the default Path, assignment, restore, absence of `home`, and that env does not move the singleton — not `BlobStore`/`Cache` construction and not the repo grep. `TestBlobStoreRoot` / `TestCache` / `TestDownloadCache` own that `root=None` follows `config.home_dir`. `TestCacheLayering` keeps the existing import and `$MOLHUB_HOME` greps and adds: walk `src/molhub` for leftover `config["home"]` and `config.home` (not `home_dir`); walk isolators for leftover `config["home"]` assignment. `TestOMol25Lmdb` / `TestOMol25Dataset` own `int(frame["atoms"].nrows) == int(cache("graph.natoms")[idx])` on `__getitem__`, that `"n_atoms"` is absent from `frame.meta`, and that there is no `graphs.n_atoms`. `cache("n_atoms")` still raises (`test_n_atoms_field_raises` stays a cache-field test).

### Reuse decision

Caller did not pass a librarian_report. Decisions below are from in-repo discovery of cache-layers-02-cache symbols, molcfg attribute access, and molpy `Block.nrows`.

- **reuse** `MolhubConfig` / `config` singleton — same named `molcfg.Config` root; only the field name and access style change. Do not add a second config type.
- **reuse** `molcfg.Config` attribute access — `config.home_dir` is `__getattribute__` / `__setattr__` on `_data["home_dir"]`. Do not add a `@property`, factory, or `make_home`.
- **generalize** `BlobStore._resolve_root` in `src/molhub/sources/blobs.py` from `config.home_dir`. Explicit `root=` still wins.
- **generalize** `DownloadCache._resolve_root` in `src/molhub/dataset/cache.py` from `config.home_dir`. Do not call `BlobStore()` to resolve home.
- **generalize** `Cache.__init__` default root in `src/molhub/cache/cache.py` from `config.home_dir / "counts"`. **reuse** `lookup` / `publish` / `_seal` / `disable_writes` / `CountConflict` / retain-one — do not replace with a compute façade. `writable=True` stays constructor-only.
- **reuse** `CountIdentity.quantity == "graph.natoms"` — hashed field name unchanged. `"n_atoms"` as a cache field still `ValueError`.
- **reuse** `OMol25Lmdb.cache` / `OMol25.cache` / packed `OMol25Dataset.cache` — bulk vector of the same quantity; do not reimplement lookup/publish.
- **reuse** molpy `frame["atoms"].nrows` — grain-one on a decoded frame. Do not add a meta key, a `graphs` block, or a public `atom_counts` method for the same fact.
- **reuse** `_checked_numbers` — LMDB `nrows` equals `len(numbers)` after that check. Packed `self._n_atoms` / `_atom_ptr` already size the sliced block.
- **reuse** `_meta_mapping` in `tests/test_dataset/test_omol25.py` to assert `"n_atoms"` is absent from meta (MetaValue unwrap).
- **generalize** CLI `_apply_home` in `src/molhub/cli.py` and `build_class1_splits` `--molhub-home` in `src/molhub/scripts/build_class1_splits.py` to write `config.home_dir`. Keep write-and-inject. Do not forbid `BlobStore(home)`.
- **generalize** isolators and `fixture-home` preamble to assign `config.home_dir` and restore.
- **reuse** `Fetcher` / `Molhub` / `DatasetBuilder(home=None)` construction — they already go through `BlobStore()`; only docstrings and the BlobStore default change.
- **new —** none. `home_dir` is the existing singleton field under a new name. Grain-one is the existing molpy Block property.
- **do not reuse** `config["home"]` / `config.home` — deleted names.
- **do not reuse** public `atom_counts` — still gone.
- **do not reuse** `frame.meta["n_atoms"]` — a third home for a count already owned as `atoms.nrows` and `cache("graph.natoms")` (architect 🔴 `law.md:one-home`).
- **do not reuse** molnex `graphs.num_atoms` / FrameStore `n_atoms.bin` as a molhub Frame field.

Closest pattern: molcfg attribute field (`config.home_dir = path`) matching how molcfg exposes `_data` keys; Frame grain-one matching existing `Block.nrows` (molpy I/O already reads `frame["atoms"].nrows`). Error handling: `KeyError` on `config["home"]`, `AttributeError` on `config.home`, MolHub `ValueError` on `cache("n_atoms")`. No factory.

## Files to create or modify

Paths are in the sibling molhub repository.

- `src/molhub/config.py`
- `tests/test_config.py`
- `src/molhub/cache/cache.py`
- `src/molhub/cache/__init__.py`
- `tests/test_cache/test_cache.py`
- `tests/test_cache/test_layering.py`
- `src/molhub/sources/blobs.py`
- `src/molhub/sources/fetcher.py`
- `tests/test_sources/test_blobs.py`
- `tests/test_sources/conftest.py`
- `src/molhub/dataset/cache.py`
- `tests/test_dataset/test_cache.py`
- `src/molhub/dataset/csv_dataset.py`
- `tests/test_dataset/test_csv_dataset.py`
- `tests/test_dataset/test_threebpa.py`
- `tests/test_dataset/test_revmd17.py`
- `tests/test_dataset/test_omol25.py`
- `src/molhub/cli.py`
- `tests/test_cli.py`
- `src/molhub/molhub.py`
- `src/molhub/loader/builder.py`
- `tests/test_loader/conftest.py`
- `src/molhub/scripts/build_class1_splits.py`
- `tests/test_scripts/test_build_class1_splits.py`
- `src/molhub/conveyor/__init__.py`
- `src/molhub/conveyor/tier.py`
- `tests/test_conveyor/test_tier.py`
- `tests/test_registry_data.py`
- `tests/test_docs_examples.py`

## Tasks

- [x] Write failing unit tests for MolhubConfig.home_dir (tests/test_config.py → TestConfig): default is Path.home()/".cache"/"molhub", assignment is config.home_dir = Path(...), tests restore config.home_dir, "home" not in config, config["home"] raises KeyError, config.home raises AttributeError, os.environ MOLHUB_HOME does not change config.home_dir, no writable key
- [x] Generalize MolhubConfig and the config singleton in src/molhub/config.py to the one field home_dir (attribute access; drop the home key)
- [x] Write failing unit tests for home_dir readers, isolators, CLI --home, and leftover-name grep (tests/test_cache/test_cache.py → TestCache; tests/test_cache/test_layering.py → TestCacheLayering; tests/test_sources/test_blobs.py → TestBlobStoreRoot; tests/test_dataset/test_cache.py → TestDownloadCache; tests/test_cli.py; tests/test_scripts/test_build_class1_splits.py; tests/test_conveyor/test_tier.py; tests/test_docs_examples.py; tests/test_sources/conftest.py; tests/test_loader/conftest.py; tests/test_registry_data.py; tests/test_dataset/test_csv_dataset.py; tests/test_dataset/test_threebpa.py; tests/test_dataset/test_revmd17.py): root=None follows config.home_dir; Cache() default root is config.home_dir/"counts"; isolators assign config.home_dir and restore; no config["home"] assignment; src/molhub has no config["home"] and no config.home that is not home_dir
- [x] Generalize BlobStore._resolve_root in src/molhub/sources/blobs.py, DownloadCache._resolve_root in src/molhub/dataset/cache.py, and Cache.__init__ default root in src/molhub/cache/cache.py from config.home_dir (Cache default config.home_dir/"counts"; writable stays constructor-only; keep lookup/publish); generalize _apply_home in src/molhub/cli.py and --molhub-home in src/molhub/scripts/build_class1_splits.py to write config.home_dir and still inject BlobStore(PATH); retarget isolators in tests/test_sources/conftest.py, tests/test_loader/conftest.py, tests/test_registry_data.py, tests/test_docs_examples.py, tests/test_dataset/test_csv_dataset.py, tests/test_dataset/test_threebpa.py, tests/test_dataset/test_revmd17.py, tests/test_cli.py, tests/test_scripts/test_build_class1_splits.py, tests/test_conveyor/test_tier.py, tests/test_dataset/test_cache.py, and tests/test_cache/test_cache.py to config.home_dir; rewrite home-resolver docstrings in src/molhub/cache/__init__.py, src/molhub/sources/fetcher.py, src/molhub/dataset/csv_dataset.py, src/molhub/molhub.py, src/molhub/loader/builder.py, src/molhub/conveyor/__init__.py, and src/molhub/conveyor/tier.py to name config.home_dir (not config["home"], not $MOLHUB_HOME)
- [x] Write unit tests for grain-one nrows identity (tests/test_dataset/test_omol25.py → TestOMol25Lmdb, TestOMol25Dataset): water fixture int(frame["atoms"].nrows) == 3 == int(cache("graph.natoms")[0]) and equals len(numbers) after _checked_numbers; packed two-row [3, 5] has int(ds[1][0]["atoms"].nrows) == 5 == int(cache("graph.natoms")[1]); "n_atoms" not in frame.meta; no graphs.n_atoms; no public atom_counts; cache("n_atoms") still ValueError; do not write n_atoms onto frame.meta
- [x] Add Google-style docstrings with units on MolhubConfig.home_dir in src/molhub/config.py, Cache default root in src/molhub/cache/cache.py, and BlobStore in src/molhub/sources/blobs.py (home_dir is a directory Path, not a physical unit)
- [x] Run full check + test suite

## Testing strategy

Unit tests only, path-mirrored, one method per test. No e2e under `tests/`. No `hasattr`, no signature-default asserts. Green is `uv run pytest {path} -v` on the mirrored file only. Tests that mutate `config.home_dir` restore the singleton. Caller and cache-layers-02-cache forbid a `regressions/` script; public-API goldens are hard-coded literals in the mirrors below.

- **MolhubConfig (`TestConfig`)** — after import, `config.home_dir == Path.home() / ".cache" / "molhub"` and `isinstance(..., Path)`. `"home" not in config`. `"writable" not in config`. `config["home"]` raises `KeyError`. `config.home` raises `AttributeError`. `config.home_dir = tmp_path` is visible on a later read of the singleton; restore returns the previous Path. `monkeypatch.setenv("MOLHUB_HOME", …)` does not change `config.home_dir`. This class does not construct `BlobStore`/`Cache` and does not grep the tree.
- **Home readers** — `TestBlobStoreRoot`: explicit `root=` wins over `config.home_dir`; `BlobStore()` follows the singleton, not env; `setenv("MOLHUB_HOME")` does not change `BlobStore()`. `TestDownloadCache`: explicit root wins; default is `config.home_dir`; `MOLHUB_CACHE_DIR` does not change the root. `TestCache`: `Cache()` with `config.home_dir = tmp` uses `tmp/"counts"` as root; `lookup`/`publish` of int32 `[3, 3]` still round-trips; `quantity="graph.natoms"` unchanged. CLI `--home PATH` then `config.home_dir == PATH` and constructors receive `PATH`; `--help` text does not contain `$MOLHUB_HOME`. `build_class1_splits` default home is `config.home_dir`. Isolators listed in Files assign `config.home_dir` and restore.
- **Layering (`TestCacheLayering`)** — existing import greps stay (`molhub.cache` may import `molhub.config`; must not import molix / dataset / sources / BlobStore / DownloadCache / loader / conveyor). Walk `src/molhub` (gone-comments stripped): no `os.environ.get("MOLHUB_HOME")`, no live `config["home"]`, no `config.home` that is not `config.home_dir`. Walk isolators: no `config["home"]` assignment, no `setenv("MOLHUB_HOME")`. `TestConfig` / `TestBlobStoreRoot` may `setenv` only as negative proofs.
- **OMol25 getitem (`TestOMol25Lmdb`, `TestOMol25Dataset`)** — Happy: water fixture `ds[0][0]` has `int(frame["atoms"].nrows) == 3` (hard-coded; 3 atoms in `_water` / `_water_record` numbers `[8, 1, 1]`). That equals `int(ds.cache("graph.natoms")[0])`. LMDB also equals `len(ds.record(0)["numbers"])` after the same check `_checked_numbers` applies. Packed two-row `[3, 5]` (`_water` plus `_n_atom_record(5)`): `int(ds[1][0]["atoms"].nrows) == 5 == int(ds.cache("graph.natoms")[1])`. Edge: `"n_atoms" not in _meta_mapping(frame)`; the decoded frame has no `graphs` / `n_atoms` graph field; there is no public `atom_counts`; `cache("n_atoms")` still raises `ValueError`; `cache("graph.natoms")` still returns int64 shape `(N,)`; tests use `nrows`, never `len(frame["atoms"])`. Domain: identity lock does not rewrite `energy` (water LMDB total stays `-76.4` eV; packed `_water(energy=-10.0)` stays `-10.0`).
- **Static** — `src/molhub` and listed tests contain no `ase` / `e3nn` / `mace` / `fairchem` / `lammps` / `openmm` imports.

## Out of scope

- Writing `n_atoms` onto `frame.meta` (whole-table or `frame.meta["n_atoms"] = …`) or inventing `graphs.n_atoms`.
- Editing molnex (`scripts/mm_param_learning`) or `src/molix`. Follow-up `cache-layers-09-frame-dirs-02-wire` landed the molnex `home_dir` callers; this slug does not shim `config["home"]`.
- Changing `Cache.lookup` / `publish` / seal / `CountConflict` / retain-one / `writable`.
- Changing `CountIdentity.quantity`; admitting cache fields other than `"graph.natoms"`; aliasing old `quantity="n_atoms"` seals.
- Reviving public `atom_counts` on OMol25 types, `LazyOMol25`, or `dataset.protocol.Dataset`.
- Putting `cache()` on `SubsetDataset` or `dataset.protocol.Dataset`.
- Rewriting `OMol25SmallFilter.reason` to read `nrows` (it may keep counting from `numbers`).
- Renaming the packed NPZ column `n_atoms` (`_REQUIRED` / `self._n_atoms`).
- TypeScript `$MOLHUB_HOME/files/` (`coordinate.py`).
- Migrating `$MOLHUB_REGISTRY`, `FIGSHARE_TOKEN`, `MOLHUB_SOURCE_BASE`, `MOLNEX_SRC`.
- A `writable` key on `MolhubConfig`.
- Renaming or merging `DownloadCache`; writing count snapshots into `BlobStore` `files/` or `DerivedLayout` `derived/`.
- A `regressions/` script (caller forbids; goldens live in unit mirrors).
- Production job submission.
