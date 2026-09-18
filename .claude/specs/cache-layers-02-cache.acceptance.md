---
slug: cache-layers-02-cache
created: 2026-09-12
criteria:
  - id: ac-001
    summary: MolhubConfig singleton owns one key, home
    type: code
    pass_when: |
      from molhub.config import config yields a molcfg.Config (MolhubConfig)
      singleton. Default is config["home"] == Path.home()/".cache"/"molhub".
      Assignment config["home"] = path is visible on a later read of that
      singleton. There is no config["writable"] key, no make_cache, and no
      factory around MolhubConfig. Cache.__init__(root=None, *, writable=True)
      remains the write switch and does not read the singleton for writable.
    status: verified
    last_checked: 2026-09-12
  - id: ac-002
    summary: Python home is config[home]; env and resolver docs do not teach it
    type: code
    pass_when: |
      Grep of src/molhub for os.environ.get("MOLHUB_HOME") is empty.
      MOLHUB_CACHE_DIR appears only in comments that say it is gone.
      BlobStore._resolve_root, DownloadCache._resolve_root, Cache() default
      root, Fetcher() default BlobStore, Molhub() default Fetcher, and
      DatasetBuilder(home=None) read config["home"] when the path is None.
      Python-facing resolver docstrings (molhub.py, loader/builder.py,
      sources/fetcher.py, sources/blobs.py, cache/cache.py, dataset/cache.py,
      dataset/csv_dataset.py, cli.py --home help, conveyor/__init__.py,
      conveyor/tier.py, scripts/build_class1_splits.py) do not say
      "defaults to $MOLHUB_HOME". tests/test_sources/conftest.py,
      tests/test_loader/conftest.py, tests/test_registry_data.py,
      tests/test_docs_examples.py, tests/test_dataset/test_csv_dataset.py,
      tests/test_dataset/test_threebpa.py, tests/test_dataset/test_revmd17.py,
      tests/test_cli.py, tests/test_scripts/test_build_class1_splits.py,
      tests/test_conveyor/test_tier.py, and tests/test_dataset/test_cache.py
      assign config["home"] and restore; they do not setenv("MOLHUB_HOME").
      $MOLHUB_REGISTRY, FIGSHARE_TOKEN, MOLHUB_SOURCE_BASE, and MOLNEX_SRC
      reads are unchanged.
    status: verified
    last_checked: 2026-09-12
  - id: ac-003
    summary: CountIdentity quantity is the field graph.natoms
    type: code
    pass_when: |
      CountIdentity accepts quantity="graph.natoms" and refuses "n_atoms"
      with ValueError. key() hashes quantity in field order. There is no
      second identity type. COUNT_SCHEMA_VERSION stays 1. Seals published
      under the old n_atoms quantity are misses for new identities.
    status: verified
    last_checked: 2026-09-12
  - id: ac-004
    summary: Public read is cache("graph.natoms") int64; atom_counts is gone
    type: code
    pass_when: |
      OMol25Lmdb, OMol25, and OMol25Dataset expose cache("graph.natoms")
      returning a fully-known int64 ndarray of shape (N,). There is no public
      atom_counts on those types. cache("n_atoms") and
      cache("graphs.num_atoms") raise ValueError. LMDB/façade keyword-only
      tails are workers= and store= (not cache=). Packed OMol25Dataset.cache
      takes field only; extra store=/workers=/cache= raise TypeError.
      On-disk snapshot counts stay int32 via one adapter cast, not a second
      file. cache is not on dataset.protocol.Dataset.
    status: verified
    last_checked: 2026-09-12
  - id: ac-005
    summary: lookup/publish seal one directory per key; CountConflict on clash
    type: code
    pass_when: |
      Cache(root=None, *, writable=True) has lookup(identity) and
      publish(identity, counts, *, known=None). Default root is
      config["home"]/"counts". publish writes a private candidate, validates,
      flushes, then exposes counts/<key>/ with sealed true plus payload
      digest, or leaves only an unreferenced candidate. Identical payloads
      share that one directory. Divergent bytes raise CountConflict and leave
      the original sealed directory. No generation subdirectory. No
      make_cache, Cache.value, max_generations, or max_temp_bytes.
      src/molhub/cache/ does not import molix or BlobStore.
    status: verified
    last_checked: 2026-09-12
  - id: ac-006
    summary: Packed RAM copy; unknown known=False never becomes int64 zeros
    type: code
    pass_when: |
      OMol25Dataset.cache("graph.natoms") returns a copy of the packed
      int64 n_atoms column and does not construct Cache or call
      lookup/publish. OMol25Lmdb.cache("graph.natoms") on a sealed
      known=False hit raises (does not rescan or republish) and never
      returns int64 zeros. Only lookup is None may scan.
      Leftover .molhub-n-atoms-*.npy beside the extract is neither read
      nor deleted. A complete scan publishes known all-true.
    status: verified
    last_checked: 2026-09-12
  - id: ac-007
    summary: CLI --home including fetch writes config and injects constructors
    type: code
    pass_when: |
      typer --home on info, cache verify, cache status, cache coverage, and
      fetch assigns config["home"] when the option is present and passes the
      path into BlobStore(root=PATH) / DatasetBuilder(home=PATH) /
      Fetcher(blobs=BlobStore(PATH)). molhub fetch --home PATH applies that
      path before Molhub()/Fetcher()/BlobStore() so files land under PATH.
      molhub cache status --help (and the shared --home help) does not
      contain the string $MOLHUB_HOME. BlobStore(home) / BlobStore(root=)
      remain valid. build_class1_splits --molhub-home writes the same
      singleton; --molhub-home argparse default is None; main()
      resolves config["home"] when the flag is omitted, then writes
      the singleton and Fetcher(blobs=BlobStore(home)). Not
      os.environ.get("MOLHUB_HOME").
    status: verified
    last_checked: 2026-09-12
  - id: ac-008
    summary: cache sits above config; dataset to cache is one-way
    type: code
    pass_when: |
      tests/test_cache/test_layering.py is green. src/molhub/cache/ does
      not import molix, molhub.dataset, molhub.sources, BlobStore,
      DownloadCache, molhub.loader, or molhub.conveyor, and may import
      molhub.config. molhub.dataset may import molhub.cache. molhub.sources,
      molhub.loader, and molhub.conveyor do not import molhub.cache.
    status: verified
    last_checked: 2026-09-12
  - id: ac-009
    summary: DownloadCache, files/, and derived/ stay separate trees
    type: code
    pass_when: |
      molhub.dataset.cache.DownloadCache is not renamed, merged, or imported
      by molhub.cache. Count snapshots are not filed under BlobStore files/
      or DerivedLayout derived/. Explicit BlobStore(root=) and Cache(root=)
      still pin a test tree. PackedCache is not reintroduced.
    status: verified
    last_checked: 2026-09-12
  - id: ac-010
    summary: Unit tests cover config, identity, cache("graph.natoms"), fetch --home
    type: runtime
    pass_when: |
      uv run pytest tests/test_config.py tests/test_cache/test_cache.py
      tests/test_dataset/test_omol25.py tests/test_cli.py
      tests/test_sources/test_blobs.py tests/test_cache/test_layering.py -v
      exits 0 and includes tests that (1) config["home"] assignment, not
      MOLHUB_HOME, drives BlobStore()/Cache() defaults, (2) two same-length
      identities with quantity "graph.natoms" and different shard tokens keep
      hard-coded arrays [3, 3] vs [4, 4], (3) OMol25Lmdb.cache("graph.natoms")
      with a fake or Cache(root=tmp) returns int64 [3, 3] for the water
      fixture and raises ValueError on an unknown field, (4)
      Cache(writable=False).publish writes no sealed directory, (5) a second
      publish of different bytes raises CountConflict and leaves exactly one
      sealed directory under counts/<key>/, (6) molhub fetch --home PATH
      files under PATH. No live OMol25 download in the default tier; no
      FAIR/ASE/MACE imports; no third-party oracle; no regressions/ script.
    status: verified
    last_checked: 2026-09-12
---

# Acceptance criteria

Done means `from molhub.config import config` is the one Python home (one key, `home`), `loaded.cache("graph.natoms")` is the one public count read, `molhub fetch --home PATH` still selects a tree after `$MOLHUB_HOME` dies, and `from molhub.cache import Cache, CountIdentity` still round-trips one sealed snapshot under an explicit tmp root. Misses cost a rescan only. 01-sources still owns identity ingredients; this slug is the only writer. Molnex files are untouched; `cache-layers-08-molnex-counts` owns the leftover `atom_counts` callers.

## AC-001 — singleton

`MolhubConfig` is a named `molcfg.Config` root, matching `MolnexConfig` at `molix/config.py`. One key. Tests assign `config["home"]` and restore it; they do not set `MOLHUB_HOME`. `writable` stays on `Cache.__init__`.

## AC-002 — no env home

Mechanical grep of `src/molhub` plus constructor defaults. Resolver docs that still teach `$MOLHUB_HOME` fail this criterion, not only `os.environ.get`. Isolators listed in `pass_when` must not `setenv("MOLHUB_HOME")`. Other env vars listed in the criterion stay. TypeScript `$MOLHUB_HOME/files/` is out of scope and is not a Python fallback.

## AC-003 — hashed field name

Fixture: `quantity="graph.natoms"`. `quantity="n_atoms"` cannot construct. A directory sealed under the old quantity string is not a hit for a new identity.

## AC-004 — public verb

`loaded.cache("graph.natoms")` is the call. No public `atom_counts`. Injected store is `store=`. Packed extra kwargs `TypeError`. Unknown field → `ValueError` listing that this slug admits `"graph.natoms"`.

## AC-005 — primitives

`Cache` still has two verbs. Adapters compute. Explicit `Cache(tmp_path)` is the snapshot tree for tests. Default `Cache()` is `config["home"]/"counts"`. `writable=True` is constructor-only.

## AC-006 — packed and unknown

Packed NPZ already has counts; publishing a second family is a bug. `known=False` is not a measured zero on the public int64 vector.

## AC-007 — CLI write-and-inject, including fetch

`--home PATH` mutates the singleton and passes `PATH` into constructors. `molhub fetch --home` is in scope. Help text names `config["home"]` or “cache root”, never `$MOLHUB_HOME`. `BlobStore(home)` remains valid.

## AC-008 — layering

`tests/test_cache/test_layering.py` is the guard. Cache may import L0 config; it must not import L1 `BlobStore` to resolve home. That file also owns the repo grep.

## AC-009 — neighbouring caches

`DownloadCache` still owns `urls/`. `BlobStore` still owns `files/`. `DerivedLayout` still owns `derived/`. Count snapshots are under `counts/` (or the injected root).

## AC-010 — runtime bar

Mirrored unit files are the public-API net. Hard-coded arrays `[3, 3]` and `[4, 4]`; water fixture 3 atoms/row. `fetch --home` lands under `PATH`. No `regressions/` script.
