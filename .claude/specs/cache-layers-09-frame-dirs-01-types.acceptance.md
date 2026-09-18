---
slug: cache-layers-09-frame-dirs-01-types
created: 2026-09-12
criteria:
  - id: ac-001
    summary: Process home is config.home_dir; home key is gone
    type: code
    pass_when: |
      from molhub.config import config yields a molcfg.Config (MolhubConfig)
      singleton. Default is config.home_dir == Path.home()/".cache"/"molhub"
      and isinstance(..., pathlib.Path). Assignment config.home_dir = path
      is visible on a later read of that singleton. "home" not in config.
      config["home"] raises KeyError. config.home raises AttributeError.
      There is no config["writable"] key and no make_cache / factory.
      Cache.__init__(root=None, *, writable=True) remains the write switch
      and does not read the singleton for writable.
    status: verified
    last_checked: 2026-09-12
  - id: ac-002
    summary: Readers and isolators use config.home_dir not config[home]
    type: code
    pass_when: |
      BlobStore._resolve_root, DownloadCache._resolve_root, Cache() default
      root, Fetcher() default BlobStore, Molhub() default Fetcher, and
      DatasetBuilder(home=None) read config.home_dir when the path is None.
      Cache() default root is config.home_dir/"counts". typer --home on
      info, cache verify, cache status, cache coverage, and fetch assigns
      config.home_dir when the option is present and passes the path into
      BlobStore(root=PATH) / DatasetBuilder(home=PATH) /
      Fetcher(blobs=BlobStore(PATH)). build_class1_splits --molhub-home
      writes config.home_dir; argparse default is None meaning
      config.home_dir. Grep of src/molhub has no config["home"] and no
      config.home that is not config.home_dir. Isolators in
      tests/test_sources/conftest.py, tests/test_loader/conftest.py,
      tests/test_registry_data.py, tests/test_docs_examples.py,
      tests/test_dataset/test_csv_dataset.py,
      tests/test_dataset/test_threebpa.py,
      tests/test_dataset/test_revmd17.py, tests/test_cli.py,
      tests/test_scripts/test_build_class1_splits.py,
      tests/test_conveyor/test_tier.py, tests/test_dataset/test_cache.py,
      and tests/test_cache/test_cache.py assign config.home_dir and restore;
      they do not assign config["home"] and do not setenv("MOLHUB_HOME").
      Python-facing resolver docstrings name config.home_dir, not
      config["home"] and not $MOLHUB_HOME.
    status: verified
    last_checked: 2026-09-12
  - id: ac-003
    summary: Cache lookup/publish and graph.natoms stay; no MOLHUB_HOME home
    type: code
    pass_when: |
      Cache still has lookup(identity) and publish(identity, counts, *,
      known=None). CountIdentity still requires quantity="graph.natoms"
      and refuses "n_atoms" with ValueError. OMol25Lmdb, OMol25, and
      OMol25Dataset still expose cache("graph.natoms") returning a
      fully-known int64 ndarray of shape (N,) and have no public
      atom_counts. Grep of src/molhub for os.environ.get("MOLHUB_HOME")
      is empty. $MOLHUB_REGISTRY, FIGSHARE_TOKEN, MOLHUB_SOURCE_BASE,
      and MOLNEX_SRC reads are unchanged.
    status: verified
    last_checked: 2026-09-12
  - id: ac-004
    summary: Grain-one is atoms.nrows equal to cache graph.natoms
    type: code
    pass_when: |
      OMol25Lmdb.__getitem__ and packed OMol25Dataset.__getitem__ do not
      write n_atoms onto frame.meta (neither whole-table assign nor
      frame.meta["n_atoms"] = …). Grain-one on a decoded frame is
      int(frame["atoms"].nrows), equal to len(numbers) after
      _checked_numbers on the LMDB path and equal to
      int(self._n_atoms[idx]) on the packed path. Tests assert
      int(frame["atoms"].nrows) == int(cache("graph.natoms")[idx]).
      There is no graphs.n_atoms field and no public atom_counts.
      cache("n_atoms") still raises ValueError. Packed NPZ column
      n_atoms stays the cache-vector source, not a Frame meta key.
    status: verified
    last_checked: 2026-09-12
  - id: ac-005
    summary: Unit tests hit home_dir default and water nrows==3
    type: runtime
    pass_when: |
      uv run pytest tests/test_config.py tests/test_cache/test_cache.py
      tests/test_cache/test_layering.py tests/test_sources/test_blobs.py
      tests/test_dataset/test_cache.py tests/test_dataset/test_omol25.py
      tests/test_cli.py -v exits 0 and includes tests that (1)
      config.home_dir default is Path.home()/".cache"/"molhub" and
      assignment not MOLHUB_HOME drives BlobStore()/Cache() defaults,
      (2) Cache() under config.home_dir=tmp uses tmp/"counts", (3)
      water OMol25Lmdb and packed OMol25Dataset getitem have
      int(frame["atoms"].nrows) == 3 matching cache("graph.natoms")[0],
      (4) packed two-row [3, 5] has int(ds[1][0]["atoms"].nrows) == 5,
      (5) "n_atoms" is absent from frame.meta, cache("n_atoms") still
      ValueError, (6) leftover config["home"] grep in TestCacheLayering
      is green. No live OMol25 download; no FAIR/ASE/MACE imports; no
      third-party oracle; no regressions/ script.
    status: verified
    last_checked: 2026-09-12
---

# Acceptance criteria

Done means `from molhub.config import config` exposes `config.home_dir` as the one Python home (no `config["home"]`, no `config.home`), every listed reader and isolator uses that attribute, `loaded.cache("graph.natoms")` is still the bulk count read, and each OMol25 `__getitem__` frame satisfies `int(frame["atoms"].nrows) == int(cache("graph.natoms")[idx])` with no `frame.meta` `n_atoms` and no `graphs.n_atoms`. Molnex files are untouched.

## AC-001 — singleton attribute

`MolhubConfig` stays a named `molcfg.Config` root. The stored field is `home_dir`. Tests assign `config.home_dir` and restore it. Negative proofs for the old names live in `TestConfig` only.

## AC-002 — readers

Mechanical grep plus constructor defaults. CLI `--home` and `--molhub-home` write `config.home_dir` and inject constructors. Isolators listed in `pass_when` must not assign `config["home"]`.

## AC-003 — keep 02-cache verbs

Lookup/publish, hashed quantity `graph.natoms`, and the ban on `$MOLHUB_HOME` as a Python home do not regress. `atom_counts` stays gone.

## AC-004 — grain-one identity

Decoded count is `frame["atoms"].nrows`. Bulk vector is `cache("graph.natoms")`. Same quantity, different grain. No third home on `frame.meta`. No `graphs.n_atoms`. Never `frame.meta["n_atoms"] = …`.

## AC-005 — runtime bar

Mirrored unit files are the public-API net. Hard-coded water 3 atoms/row; packed second row 5. No `regressions/` script.
