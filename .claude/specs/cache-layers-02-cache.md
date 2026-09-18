---
title: loaded.cache("graph.natoms"); Python home is molhub.config
status: code-complete
grilled: true
created: 2026-09-11
revised: 2026-09-12
chain: cache-layers
depends: cache-layers-01-sources
supersedes: cache-layers-01-cache (source-index publication slice)
---

# loaded.cache("graph.natoms"); Python home is molhub.config

## Summary

A loaded OMol25 view answers per-row atom counts with `loaded.cache("graph.natoms")`, returning a fully-known int64 vector of shape `(N,)`. Public `atom_counts` methods go away. Snapshot identity hashes the MolHub field name `graph.natoms` (not `n_atoms`, not molnex `graphs.num_atoms`). Process home is the `molhub.config` singleton with one key, `home` (default `Path.home()/".cache"/"molhub"`). `BlobStore`, `DownloadCache`, `Cache`, `Fetcher`, `Molhub`, `DatasetBuilder`, and CLI `--home` (including `molhub fetch --home`) read that key when no explicit path is given; `$MOLHUB_HOME` is not a Python home. A miss still costs only a rescan. Sealed snapshots stay optional, int32 on disk, one directory per key. Packed NPZ serves the RAM copy and does not publish.

## Domain basis

Count snapshots are **not scientific labels**. They do not define chemistry subsets, change energies or forces, or stand in for measured bounds. Unknown counts stay unknown and are distinct from a measured zero. Dataset units, record identity, and shard revisions are owned by `cache-layers-01-sources` (OMol25 native labels remain eV / eV·Å⁻¹ / Å as already declared on that source). This slug does not introduce a physical model, conversion constant, or DOI.

## Design

Paths below are in the sibling molhub repository. Current callers: `OMol25Lmdb` / `OMol25` / packed `OMol25Dataset` (public count read), `BlobStore` / `DownloadCache` / `Cache` / `Fetcher` / `Molhub` / `DatasetBuilder` / typer `--home` / `molhub fetch` / `build_class1_splits --molhub-home` (home root). Current tests: `TestCountIdentity`, `TestCache`, `TestCacheLayering`, `TestOMol25Lmdb`, `TestOMol25`, `TestOMol25Dataset`, `TestBlobStoreRoot`, `TestDownloadCache`, `TestFetch`, CLI home tests, `TestTierStore.test_the_default_root_is_the_blob_stores`, plus isolators that still `setenv("MOLHUB_HOME")`.

### molhub.config (L0)

New module `src/molhub/config.py`, same shape as `molix.config`: thin `MolhubConfig(molcfg.Config)` with `__slots__ = ()` and a process singleton. **One key only:**

```python
config = MolhubConfig({
    "home": Path.home() / ".cache" / "molhub",
})
```

No extra methods (molix needs `set_precision`; molhub does not). `config.py` imports only `pathlib` and `molcfg` — not sources, cache, dataset, loader, conveyor, or molix. Declare `molcrafts-molcfg>=1.5.0` on molhub (it is not a direct dependency today). Callers write `from molhub.config import config`. Do not re-export from `molhub/__init__.py`.

`config["home"]` is a `Path`. Assignment is the write path (`config["home"] = Path(...)`). There is no `writable` key and no `os.environ` / `$MOLHUB_HOME` / `$MOLHUB_CACHE_DIR` for home. Do not migrate `$MOLHUB_REGISTRY`, `FIGSHARE_TOKEN`, `MOLHUB_SOURCE_BASE`, or `MOLNEX_SRC`.

`Cache.__init__(root=None, *, writable=True)` stays the write switch (`writable=False` / `disable_writes()` unchanged). Do not default `writable` from the singleton. Nothing in CLI, `BlobStore`, or `DownloadCache` reads a process-wide writable flag.

### Who reads the singleton; constructor injection stays

`root=None` / `home=None` / `blobs=None` reads `config["home"]` at construction. An explicit path still wins and does not require writing the singleton first.

| Caller | Default (`None`) | Explicit |
|---|---|---|
| `BlobStore(root=None)` | `config["home"]` | `BlobStore(root=PATH)` and `BlobStore(PATH)` / `BlobStore(home)` still for tests and CLI |
| `DownloadCache(root=None)` | `config["home"]` (`urls/` under it) | `DownloadCache(root=tmp)` still for tests. Does not call `BlobStore()` to resolve home. Drops `$MOLHUB_CACHE_DIR` entirely (no warning path). |
| `Cache(root=None, *, writable=True)` | root `config["home"]/"counts"`; writable is the constructor flag | `Cache(root=tmp)` is the snapshot tree; `writable=False` stays keyword-only |
| `Fetcher(blobs=None)` | `BlobStore()` → `config["home"]` | `Fetcher(blobs=BlobStore(PATH))` |
| `Molhub(..., fetcher=None)` | `Fetcher()` → `BlobStore()` | `Molhub(registry, fetcher=Fetcher(blobs=BlobStore(PATH)))` |
| `DatasetBuilder(home=None)` | `BlobStore(home).root` already; `home=None` follows `config["home"]` | `DatasetBuilder(home=PATH)` |
| typer `--home` on `info` / `cache verify` / `cache status` / `cache coverage` / **`fetch`** | leave the singleton; construct with `root=None` | write `config["home"] = Path(home).expanduser()` **and** pass the path (`BlobStore(root=PATH)`, `DatasetBuilder(home=PATH)`, `Fetcher(blobs=BlobStore(PATH))`) |
| `python -m molhub.scripts.build_class1_splits --molhub-home` | argparse default `None` meaning `config["home"]` (delete `os.environ.get("MOLHUB_HOME", …)`) | write `config["home"]` and `Fetcher(blobs=BlobStore(home))`, same as CLI `--home` |

CLI `--home` help must not say `$MOLHUB_HOME`. Private `_apply_home(home: Path | None)` in `cli.py` is allowed (verbs share it). **`molhub fetch` takes `--home` in this slug** and applies it before `Molhub()` / `Fetcher()` / `BlobStore()` — today `MOLHUB_HOME=/data molhub fetch …` is that verb's only home seam; after `$MOLHUB_HOME` dies it would otherwise only write `~/.cache/molhub`. Do not forbid `BlobStore(home)`.

After this slug, a grep of `src/molhub` for `os.environ.get("MOLHUB_HOME")` is empty, and `MOLHUB_CACHE_DIR` appears only in comments that say it is gone. `_ENV_VAR = "MOLHUB_HOME"` and `os.environ.get(_ENV_VAR)` in `blobs.py` go away. Python-facing resolver docstrings must not say “defaults to `$MOLHUB_HOME`” (`molhub.py`, `loader/builder.py`, `sources/fetcher.py`, `sources/blobs.py`, `cache/cache.py`, `dataset/cache.py`, `dataset/csv_dataset.py`, `cli.py`, `conveyor/__init__.py`, `conveyor/tier.py`, `scripts/build_class1_splits.py`). `$MOLHUB_REGISTRY` / `FIGSHARE_TOKEN` / `MOLHUB_SOURCE_BASE` / `MOLNEX_SRC` reads remain.

Isolators assign `config["home"]` and restore the previous Path; they must not `setenv("MOLHUB_HOME")`. Retarget fixtures in the **same** change as the resolvers stop reading env, so the suite never writes into the developer `~/.cache/molhub`. `tests/test_docs_examples.py` `FixtureHome.env` drops `MOLHUB_HOME` (keep `$MOLHUB_REGISTRY`); the `fixture-home` runner prepends `from molhub.config import config; config["home"] = Path(...)` to the executed snippet.

TypeScript `$MOLHUB_HOME/files/` in `coordinate.py` is out of scope: the Python default directory is `Path.home()/".cache"/"molhub"`; the TS env is not a Python fallback. `export MOLHUB_HOME` must not drive `BlobStore()`.

Do not edit molnex. Named follow-up slug **`cache-layers-08-molnex-counts`**: retarget `class_i_fit._official_vector_counts`, `profile_class_i.py`, and `prod_day` `MOLHUB_HOME` after this lands.

### CountIdentity.quantity

Same frozen dataclass, same required fields, same `key()` (UTF-8 canonical JSON → sha256 hex). Do not add a second identity type. This slug admits `quantity == "graph.natoms"` only — that string is the MolHub source field name, hashed as the quantity field. Construction with `"n_atoms"` or any other string raises `ValueError`. `COUNT_SCHEMA_VERSION` stays `1`; dtype stays `"int32"`. Old seals hashed `quantity="n_atoms"` and therefore miss. That is honest; do not alias, copy, or migrate them.

### Public read: `cache("graph.natoms")`

Delete public `OMol25Lmdb.atom_counts`, `OMol25.atom_counts`, and `OMol25Dataset.atom_counts`. The public verb is

```python
loaded.cache("graph.natoms") -> np.ndarray  # int64, shape (N,)
```

`field` is positional. LMDB and the `OMol25` façade take keyword-only `workers=0` (LMDB scan only; adapter-owned, `Cache` never sees it) and `store=None` (injected snapshot store for tests; default `Cache()`). **Do not name the injected `Cache` `cache=` on a method named `cache`.** Packed `OMol25Dataset.cache(field)` takes `field` only; extra `store=` / `workers=` / `cache=` raise `TypeError` (do not swallow). Unknown field names raise `ValueError`. This slug admits exactly `"graph.natoms"`. `"n_atoms"` and molnex `"graphs.num_atoms"` are unknown.

LMDB path (generalize the current `atom_counts` body, do not add a parallel persist path): fill `CountIdentity` from the 01 table with `quantity="graph.natoms"`; `lookup`; on a fully-known hit cast int32 → int64; on any `known=False` raise (do not return zeros, do not republish into `CountConflict`); on a miss scan (`_count_table_atoms` / `ProcessPoolExecutor` stay private on the adapter), `publish` int32, return int64. No `.molhub-n-atoms-*.npy` read, write, or auto-delete. No public `revision=`. `OMol25.cache` forwards to the loaded view (`_require_loaded()`). Domain views still project `parent.cache("graph.natoms")[indices]`; `SubsetDataset` does not gain `cache()`. Do not put `cache` on `dataset.protocol.Dataset`.

Packed `OMol25Dataset.cache("graph.natoms")` returns `self._n_atoms.copy()` (already int64 in RAM). It does not construct `Cache` and does not `lookup`/`publish`.

### Cache lookup / publish / seal (keep)

`lookup` / `publish` / `disable_writes` / `CountConflict` / retain-one stay. Construction is `Cache(root=None, *, writable=True)` — keyword-only tail, no `make_cache`, no `Cache.value(name, compute)`, no `max_generations`, no `max_temp_bytes`, no `CountSnapshot.generation`. Default root is `config["home"]/"counts"`, **not** `BlobStore().root/"counts"`. `molhub.cache` must not import `molhub.sources`, `BlobStore`, `molhub.dataset`, `DownloadCache`, `molhub.loader`, `molhub.conveyor`, or molix. It may import `molhub.config`. Seal copy of FrameStore (candidate under `counts/tmp/<uuid>/` → validate → fsync → `sealed: true` → `os.replace` into `counts/<key>/`) is unchanged. Concurrent identical payloads share one directory; contradictory bytes raise `CountConflict` and leave the existing dir. `writable=False` / `disable_writes()` / `OSError` on replace: return the in-memory snapshot, write nothing durable, do not change labels.

`counts/` remains Python-local like `urls/`. Not `files/`, not `derived/`. `DownloadCache` is not renamed or merged.

### Layering

`molhub.cache` sits below L3 (`dataset`) and above L0 (`config`). dataset → cache is one-way. Guard in `tests/test_cache/test_layering.py` (AST/source scan, same shape as `tests/test_sources/test_layering.py`): cache source does not import molix, `molhub.dataset`, `molhub.sources`, `BlobStore`, `DownloadCache`, `molhub.loader`, or `molhub.conveyor`; it may import `molhub.config`. `molhub.sources`, `molhub.loader`, and `molhub.conveyor` still must not import `molhub.cache`.

Repo grep lives in that layering/boundary test, **not** in `TestConfig`: walk `src/molhub` (gone-comments stripped) for `os.environ.get("MOLHUB_HOME")`, leftover `MOLHUB_CACHE_DIR` uses, and live “defaults to `$MOLHUB_HOME`” resolver sentences. Walk isolators listed in Files for `setenv("MOLHUB_HOME")`. `TestConfig` and `TestBlobStoreRoot` may `setenv` only as negative proofs that env does not move `config["home"]` or `BlobStore()`.

### Tests (owned seams)

No `hasattr`, no signature-default asserts. `TestConfig` owns keys, the `home` default, assignment, and restore — not `BlobStore`/`Cache` construction and not the repo grep. `TestBlobStoreRoot` / `TestCache` / `TestDownloadCache` own that `root=None` follows `config["home"]`. `TestCache` owns seal vs partial vs `CountConflict` vs retain-one vs read-only, with a real `Cache(tmp_path)`. `TestOMol25Lmdb` / `TestOMol25` own `cache("graph.natoms")` with a fake `lookup`/`publish` or `Cache(root=tmp)`. `TestOMol25Dataset` owns the packed RAM path and `TypeError` on extra kwargs. `TestFetch` owns `molhub fetch --home PATH` applying the path before `Molhub()`/`Fetcher()`/`BlobStore()`. CLI tests that currently `monkeypatch.setenv("MOLHUB_HOME")` assign `config["home"]` instead and restore.

### Reuse decision

- **reuse** `Cache.lookup` / `Cache.publish` / `_seal` / `disable_writes` / retain-one — spec calls them; do not replace with a compute façade.
- **reuse** `CountConflict` — contradictory sealed bytes still raise and leave the existing directory.
- **reuse** `molcfg.Config` — `MolhubConfig` subclasses it; the singleton is `config = MolhubConfig({...})` (closest pattern: `MolnexConfig` at `molix/config.py`).
- **generalize** `CountIdentity.quantity` to `"graph.natoms"` (hashed). Do not add a second identity type. Old `n_atoms` quantity string is gone.
- **generalize** `Cache.__init__` default **root** from `config["home"]/"counts"`; drop the `BlobStore` import. **Do not** default `writable` from the singleton (librarian suggested `config["writable"]`; architect 🔴: no current caller, bag field). `writable=True` stays constructor-only.
- **generalize** `BlobStore._resolve_root` and `DownloadCache._resolve_root` from `config["home"]`. Explicit `root=` still for tests.
- **generalize** CLI `--home` (and `build_class1_splits --molhub-home`) to write `config["home"]` **and** pass `root=` / `home=` / `blobs=BlobStore(PATH)`. **Add `--home` on `molhub fetch`.** Do not forbid `BlobStore(home)`.
- **generalize** `OMol25Lmdb.atom_counts` / `OMol25.atom_counts` / packed `OMol25Dataset.atom_counts` — delete the public names; LMDB/façade become `cache("graph.natoms")` through Cache with `store=`; packed is `cache(field)` only and does not publish.
- **new —** `MolhubConfig` + module singleton: molhub has no config type today; a named root matching `MolnexConfig` is the molix pattern (`molcfg.Config` uses `__slots__`, so the project type is the subclass). One key, `home`. Not a second home resolver.
- **new —** FrameStore seal copy stays local (already landed). `molhub.cache` still must not import molix.
- **new —** `DownloadCache` stays a separate URL-addressed `urls/` cache. Different key space; do not rename or merge. Only its root resolution generalizes.

Do not reuse `molix.data.FrameStoreReader.atom_counts` (different persistence). Do not place types in molnex `class_i_fit.py` / `AtomCountCache` (follow-up `cache-layers-08-molnex-counts`).

Closest pattern for new code: `MolnexConfig` + `config = MolnexConfig({...})`; `BlobStore(root)` / `Cache(root)` / `Fetcher(blobs=BlobStore(PATH))` construction; `ValueError` on illegal identity or unknown field; dedicated `CountConflict` for contradictory snapshots. No factory.

## Files to create or modify

Paths are in the sibling molhub repository.

- `src/molhub/config.py` (new)
- `tests/test_config.py` (new)
- `pyproject.toml`
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
- `src/molhub/dataset/omol25.py`
- `tests/test_dataset/test_omol25.py`
- `tests/test_dataset/test_csv_dataset.py`
- `tests/test_dataset/test_threebpa.py`
- `tests/test_dataset/test_revmd17.py`
- `src/molhub/cli.py`
- `tests/test_cli.py`
- `src/molhub/molhub.py`
- `src/molhub/loader/builder.py`
- `src/molhub/loader/errors.py`
- `tests/test_loader/conftest.py`
- `src/molhub/scripts/build_class1_splits.py`
- `src/molhub/scripts/__init__.py`
- `scripts/build_class1_splits.py`
- `tests/test_scripts/test_build_class1_splits.py`
- `src/molhub/conveyor/__init__.py`
- `src/molhub/conveyor/tier.py`
- `src/molhub/conveyor/splits.py`
- `tests/test_conveyor/test_tier.py`
- `tests/test_registry_data.py`
- `tests/test_docs_examples.py`

## Tasks

- [x] Write failing unit tests for MolhubConfig (`tests/test_config.py` → `TestConfig`): default `home` is `Path.home()/".cache"/"molhub"`, assignment updates the singleton, tests restore the previous Path, there is no `writable` key, `os.environ` `MOLHUB_HOME` does not change `config["home"]`
- [x] Implement `MolhubConfig` and the `config` singleton in `src/molhub/config.py`; declare `molcrafts-molcfg>=1.5.0` in `pyproject.toml`
- [x] Write failing unit tests for CountIdentity quantity `"graph.natoms"`, Cache config defaults, home readers, fetch `--home`, and layering (`tests/test_cache/test_cache.py` → `TestCountIdentity`, `TestCache`; `tests/test_cache/test_layering.py` → `TestCacheLayering`; `tests/test_sources/test_blobs.py` → `TestBlobStoreRoot`; `tests/test_dataset/test_cache.py` → `TestDownloadCache`; `tests/test_cli.py` → `TestFetch` and CLI home tests; `tests/test_conveyor/test_tier.py`; `tests/test_scripts/test_build_class1_splits.py`; `tests/test_dataset/test_csv_dataset.py`; `tests/test_dataset/test_threebpa.py`; `tests/test_dataset/test_revmd17.py`; `tests/test_registry_data.py`; `tests/test_docs_examples.py`) (rework: quantity is `graph.natoms`; `root=None` follows `config["home"]` not `$MOLHUB_HOME`; TestConfig does not grep or construct BlobStore/Cache; grep lives in TestCacheLayering; CLI fetch has `--home`; no hasattr/signature-default)
- [x] Generalize `CountIdentity.quantity` to `"graph.natoms"` and `Cache.__init__` default root from `config["home"]/"counts"` in `src/molhub/cache/cache.py` (drop `BlobStore` import; `writable=True` stays constructor-only; re-export from `src/molhub/cache/__init__.py`); generalize `BlobStore._resolve_root` in `src/molhub/sources/blobs.py`, `DownloadCache._resolve_root` in `src/molhub/dataset/cache.py`, `Fetcher` in `src/molhub/sources/fetcher.py`, `Molhub` in `src/molhub/molhub.py`, and `DatasetBuilder(home=None)` in `src/molhub/loader/builder.py` from `config["home"]`; add `--home` on `molhub fetch` and apply it before `Molhub()`/`Fetcher()`/`BlobStore()` in `src/molhub/cli.py` (write `config["home"]` when `--home` is present AND pass `root=`/`home=`/`blobs=BlobStore(PATH)`; keep constructor injection; do not forbid `BlobStore(home)`); generalize `--molhub-home` in `src/molhub/scripts/build_class1_splits.py`; retarget isolators in `tests/test_sources/conftest.py`, `tests/test_loader/conftest.py`, `tests/test_registry_data.py`, `tests/test_docs_examples.py`, `tests/test_dataset/test_csv_dataset.py`, `tests/test_dataset/test_threebpa.py`, `tests/test_dataset/test_revmd17.py`, `tests/test_cli.py`, `tests/test_scripts/test_build_class1_splits.py`, `tests/test_conveyor/test_tier.py`, and `tests/test_dataset/test_cache.py` to assign `config["home"]` and restore (same change as the resolvers; they must not `setenv("MOLHUB_HOME")`); rewrite Python-facing home-resolver docstrings in `src/molhub/cache/__init__.py`, `src/molhub/dataset/csv_dataset.py`, `src/molhub/loader/errors.py`, `src/molhub/scripts/__init__.py`, `src/molhub/conveyor/__init__.py`, `src/molhub/conveyor/tier.py`, `src/molhub/conveyor/splits.py`, and `scripts/build_class1_splits.py` (rework: previous lookup/publish still seals, but default root, quantity check, env home, missing fetch `--home`, and setenv isolators do not match this design)
- [x] Write failing unit tests for `cache("graph.natoms")` (`tests/test_dataset/test_omol25.py` → `TestOMol25Lmdb`, `TestOMol25`, `TestOMol25Dataset`): fake or `Cache(root=tmp)` lookup/publish, fill-table `quantity="graph.natoms"`, no public `atom_counts`, unknown field `ValueError`, no `.npy` sidecar, packed `cache(field)` only (extra `store=`/`workers=`/`cache=` `TypeError`), LMDB/façade keyword-only `workers=` and `store=`, partial `known=False` is raise not zeros, public dtype int64 (rework: public verb, hashed quantity, and injected-store name changed)
- [x] Generalize `OMol25Lmdb`, `OMol25`, and packed `OMol25Dataset` in `src/molhub/dataset/omol25.py` to `cache("graph.natoms")`; delete public `atom_counts`; add a Google-style docstring with units on `OMol25Lmdb.cache` (rework: `atom_counts(cache=)` still exists and still fills `quantity="n_atoms"`)
- [x] Add Google-style docstrings with units on `MolhubConfig`, `CountIdentity.quantity`, and `Cache` in `src/molhub/config.py` and `src/molhub/cache/cache.py` (rework: current text still names `$MOLHUB_HOME` and `n_atoms`)
- [x] Run full check + test suite

## Testing strategy

Unit tests only, path-mirrored, one method per test. No e2e under `tests/`. No `hasattr`, no signature-default asserts. Molnex `regressions/` is gone; this slug does not add `molhub/regressions/` — public-API goldens are hard-coded literals in the mirrors. Green is `uv run pytest {path} -v` on the mirrored file only. Tests that mutate `config["home"]` restore the singleton.

- **MolhubConfig (`TestConfig`)** — after import, `config["home"] == Path.home() / ".cache" / "molhub"`. `"writable" not in config`. `config["home"] = tmp_path` is visible on a later read of the singleton; restore returns the previous Path. `monkeypatch.setenv("MOLHUB_HOME", …)` does not change `config["home"]`. This class does not construct `BlobStore`/`Cache` and does not grep the tree.
- **CountIdentity (`TestCountIdentity`)** — `quantity="graph.natoms"` is required. `quantity="n_atoms"` raises `ValueError`. Two identities that share length but differ in `shard_digests` or ordered `row_mapping_digest` have different `key()` values. Construction without `source_revision` / shard tokens / mapping digest raises. OMol25 fill-table fixtures use `record_identity="omol25-aselmdb-table-row"`, `row_domain="record"`, `quantity="graph.natoms"`, `dtype="int32"`, `schema_version=1`.
- **Cache.lookup happy path (`TestCache`)** — `Cache(tmp_path).publish` of int32 `[3, 3]`, known all-true; lookup returns the same. Hard-coded water-sized golden: 3 atoms/row. Identity A `[3, 3]` vs identity B (same length, different shard token) `[4, 4]` stay apart. `Cache()` with `config["home"] = tmp` uses `tmp/"counts"` as root. `Cache(tmp_path, writable=False)` does not read a config writable key.
- **Miss vs partial / retain-one / read-only** — missing key → `None`. Unsealed or truncated `counts.bin` → `None`. Second publish of different bytes raises `CountConflict` and leaves one sealed dir under `counts/<key>/` with no generation subdirectory. Identical bytes retain one dir. `Cache(tmp_path, writable=False).publish` returns the in-memory snapshot and writes no sealed dir. Simulated `OSError` on replace calls `disable_writes` and keeps labels. `known=[True, False]`: slot 1 is unknown, not a measured 0.
- **Layering (`TestCacheLayering`)** — `src/molhub/cache/` does not import molix, `molhub.dataset`, `molhub.sources`, `BlobStore`, `DownloadCache`, `molhub.loader`, or `molhub.conveyor`. It may import `molhub.config`. `molhub.dataset` may import `molhub.cache`. `molhub.sources`, `molhub.loader`, and `molhub.conveyor` must not import `molhub.cache`. Walk `src/molhub` (gone-comments stripped): no `os.environ.get("MOLHUB_HOME")`, `MOLHUB_CACHE_DIR` only in comments that say it is gone, no live “defaults to `$MOLHUB_HOME`” on Python resolvers. Walk isolators listed in Files: no `setenv("MOLHUB_HOME")`. `TestConfig` / `TestBlobStoreRoot` may `setenv` only as negative proofs. `$MOLHUB_REGISTRY` / `FIGSHARE_TOKEN` / `MOLHUB_SOURCE_BASE` / `MOLNEX_SRC` reads remain.
- **Home readers** — `TestBlobStoreRoot`: explicit `root=` wins over `config["home"]`; `BlobStore()` follows the singleton, not env; `setenv("MOLHUB_HOME")` does not change `BlobStore()`. `TestDownloadCache`: explicit root wins; default is `config["home"]`; setting `MOLHUB_CACHE_DIR` does not change the root and raises no deprecation warning. CLI `--home PATH` then `config["home"] == PATH` and constructors receive `PATH`; `--help` text does not contain `$MOLHUB_HOME`. `TestFetch`: `molhub fetch --home PATH` files under `PATH` (not the singleton’s previous value, not env). `TestTierStore` default root follows `BlobStore()` / the singleton. `build_class1_splits` default home is `config["home"]`. Isolators listed in Files assign `config["home"]` and restore.
- **OMol25 (`TestOMol25Lmdb`, `TestOMol25`, `TestOMol25Dataset`)** — `ds.cache("graph.natoms", store=fake)` (or `Cache(root=tmp)`): no `.molhub-n-atoms-*.npy`; fake receives `quantity="graph.natoms"` and the 01 fill table; leftover npy is neither read nor deleted; fully-known int32 `[3, 3]` hit returns int64 `[3, 3]` without decode; a sealed `known=False` hit **raises** (do not rescan or republish) — not `[3, 0]`; only `lookup is None` may scan; `cache("n_atoms")` and `cache("graphs.num_atoms")` raise `ValueError`; there is no public `atom_counts`; `max_rows=1` vs full extract produce different keys; packed `OMol25Dataset.cache("graph.natoms")` returns the NPZ column and does not call the fake / does not construct `Cache`; packed `cache(..., store=…)` / `workers=` raises `TypeError`.
- **Domain** — Cache miss does not drop rows or rewrite labels; the adapter's computed `[3, 3]` is what the caller sees.

## Out of scope

- Atomic reference energies, fitted offsets, `AtomicDressing`, residual columns, collate, TensorFrame precision, DDP / `DataMixture` scheduling, geometry memos, `ForceAwareModel`.
- Molnex `src/` (molnex must not import molhub). Follow-up slug **`cache-layers-08-molnex-counts`** retargets molnex scripts that still call `atom_counts(cache=Cache())`: `class_i_fit._official_vector_counts`, `profile_class_i.py`, and `prod_day` `MOLHUB_HOME`. Do not edit molnex files in this slug.
- TypeScript `$MOLHUB_HOME/files/` layout (`src/molhub/coordinate.py`): the Python default is `Path.home()/".cache"/"molhub"`; the TS env is not a Python fallback.
- molhub `regressions/hf-facade-03-builder-info.py` and `hf-facade-04` still `os.environ["MOLHUB_HOME"]` around `DatasetBuilder(home=None)`. After env is ignored they would miss the fixture tree. Not retargeted in this slug.
- Migrating `$MOLHUB_REGISTRY`, `FIGSHARE_TOKEN`, `MOLHUB_SOURCE_BASE`, `MOLNEX_SRC`.
- Deleting `AtomCountCache` in molnex `scripts/mm_param_learning/class_i_fit.py` (that is `cache-layers-08-molnex-counts`).
- A `writable` key on `MolhubConfig`.
- Reviving `PackedCache` / `TensorDict.memmap_()`.
- Consuming FrameStore `n_atoms.bin`; this slug only forbids duplicating it.
- Renaming or merging `DownloadCache`; writing count snapshots into `BlobStore` `files/` or `DerivedLayout` `derived/`.
- Importing molix from `molhub.cache`.
- Auto-deleting existing `.molhub-n-atoms-*.npy` user files; aliasing old `quantity="n_atoms"` seals.
- A public `revision=` on `OMol25Lmdb` (forbidden by 01-sources).
- `max_generations` / `max_temp_bytes` knobs and `CountSnapshot.generation`.
- `make_cache`, `Cache.value(name, compute)`, any factory.
- Putting `cache()` on `dataset.protocol.Dataset`; giving `SubsetDataset` a `cache()` method (views project the parent vector).
- Admitting any field other than `"graph.natoms"`.
- Fixing `SubsetDataset.source_id`'s `sorted(indices)` hash (`cache-layers-01-sources`).
- CLI `molhub cache *` for count snapshots; docs site prose that still names `$MOLHUB_HOME`.
- A `regressions/` script (molnex `regressions/` is gone; this slug does not add `molhub/regressions/`).
- Per-experiment or per-subset published families; worker shared-mmap; cross-job exactly-once.
