---
slug: cache-layers-08-molnex-counts
created: 2026-09-12
criteria:
  - id: ac-001
    summary: Official wrap is source.cache("graph.natoms") to _VectorCounts
    type: code
    pass_when: |
      _official_vector_counts(source) calls source.cache("graph.natoms") and
      returns _VectorCounts. It does not call atom_counts or construct
      molhub.cache.Cache. scripts/mm_param_learning/class_i_fit.py contains
      no "from molhub.cache import Cache" and no atom_counts(cache=. A
      source whose type name is NoCache and that has no cache method raises
      AttributeError (message may contain NoCache), not TypeError. The
      function body contains no getattr, hasattr, try, or except. An
      AttributeError raised inside a real cache method is not translated.
    status: verified
    last_checked: 2026-09-12
  - id: ac-002
    summary: Historical loaders share the vector; prefix uses len and gather
    type: code
    pass_when: |
      AtomCountCache, _n_atoms_cache_path, _omol_atom_counts, and
      ATOM_COUNT_HINT are deleted from class_i_fit.py and are not
      importable. load_omol25_official and load_omol25_official_tiny call
      _official_vector_counts once per source. A stub vector [3, 400] with
      max_atoms=32 keeps only the 3-atom row. A measured 0 is dropped.
      load_omol25_official still returns tuple[LazyOMol25, LazyOMol25] with
      val_role val-neutral. _official_prefix takes a _VectorCounts, uses
      limit = min(len(counts), scan_cap) then gather(range(limit)), and
      does not decode LMDB records. _index_omol_source uses
      gather(range(len(counts))) or equivalent. Neither function reads
      counts._counts. _VectorCounts has __len__ and has no record method.
    status: verified
    last_checked: 2026-09-12
  - id: ac-003
    summary: LazyOMol25.atom_counts is an immutable subset; mismatch raises
    type: code
    pass_when: |
      LazyOMol25.atom_counts still gathers the in-memory _VectorCounts at
      subset indices as torch.long and is documented as a subset of that
      immutable vector, not a live hint cache. view.cache("graph.natoms")
      raises AttributeError. _sample_from_record does not call record. A
      gather/decode disagreement raises ValueError naming the row, the
      cache count, and the decoded count, and does not overwrite the wrap.
      _fit_e0_subsample still exists. TestLazyOMol25 constructs with
      _VectorCounts, not AtomCountCache. TestAtomCountCache is deleted.
    status: verified
    last_checked: 2026-09-12
  - id: ac-004
    summary: e0 cache root is molhub_config[home]/derived/omol25-index
    type: code
    pass_when: |
      class_i_fit.py imports molhub.config as molhub_config (molix.config
      stays config). _e0_cache_path uses molhub_config["home"] /
      "derived" / "omol25-index" / "{digest}-e0.npy" with the existing sha1
      of root:len:max_atoms:n_fit:seed. It does not read os.environ
      MOLHUB_HOME. Tests assign molhub_config["home"] to tmp_path and
      restore.
    status: verified
    last_checked: 2026-09-12
  - id: ac-005
    summary: Profile wraps the same vector and drops measured 0
    type: code
    pass_when: |
      build_lazy_pair calls _official_vector_counts(src), not
      src.atom_counts(), and passes one _VectorCounts instance into both
      LazyOMol25 views (not a list of ints). Keep is 1 <= n <= max_atoms:
      a stub vector [0, 3, 40, 2] with max_atoms=32, n_train=1, n_val=1
      drops measured 0, trains the 3-atom row, and vals the 2-atom row.
      run_profile calls _require_nobackup_molhub_home before
      build_lazy_pair. The zeros dressing in build_lazy_pair is
      _historical_are(torch.zeros(95)). The script is not retargeted onto
      load_omol25_fairchem_4m.
    status: verified
    last_checked: 2026-09-12
  - id: ac-006
    summary: Python home assignment; launchers do not export MOLHUB_HOME
    type: code
    pass_when: |
      prod_day._require_nobackup_molhub_home mkdirs the nobackup workspace
      path and assigns molhub_config["home"] to that Path. It does not
      read or write os.environ MOLHUB_HOME. run_class_i_prod_day_gh200.sbatch,
      run_class_i_profile_gh200.sbatch, and run_official_smoke.sh contain
      no export MOLHUB_HOME. scripts/mm_param_learning/*.py contain no
      os.environ.get("MOLHUB_HOME") and no os.environ["MOLHUB_HOME"].
      Tests restore molhub_config["home"] and do not setenv MOLHUB_HOME.
    status: verified
    last_checked: 2026-09-12
  - id: ac-007
    summary: Mirrored unit goldens stand in for deleted regressions/
    type: runtime
    pass_when: |
      python -m pytest tests/test_scripts/test_mm_param_learning/test_class_i_fit.py
      tests/test_scripts/test_mm_param_learning/test_profile_class_i.py
      tests/test_scripts/test_mm_param_learning/test_prod_day.py -v is green.
      Hard-coded literals include water-sized [3, 3] from
      cache("graph.natoms"), historical max_atoms=32 dropping a 400-atom
      row, measured 0 dropped in both TestLoadOmol25Official and
      TestBuildLazyPair, _e0_cache_path under tmp/derived/omol25-index,
      AttributeError on NoCache, and ValueError on a 3-vs-2 gather/decode
      mismatch. No regressions/ script. No FAIR/ASE/MACE runtime.
    status: verified
    last_checked: 2026-09-12
  - id: ac-008
    summary: Google docstrings name dimensionless counts and Path home
    type: docs
    pass_when: |
      _official_vector_counts, _VectorCounts, _e0_cache_path,
      LazyOMol25.atom_counts, _sample_from_record, and
      _require_nobackup_molhub_home have Google-style docstrings. Counts are
      dimensionless atoms per row. Home is a pathlib Path. Field name
      graph.natoms is named. LazyOMol25.atom_counts is a subset of the
      immutable vector, not a live hint cache. Units are not invented for
      a non-physical index.
    status: verified
    last_checked: 2026-09-12
---

# Acceptance criteria

Done means molnex Class-I OMol scripts read `loaded.cache("graph.natoms")` through `_official_vector_counts` with no discovery wrapper, historical and profile loaders share that immutable `_VectorCounts` with one `1 ≤ n ≤ max_atoms` keep, `AtomCountCache` and `record` are gone, a gather/decode mismatch raises, Python home is `molhub_config["home"]` assigned by `prod_day._require_nobackup_molhub_home`, and launchers no longer export `MOLHUB_HOME`. `LazyOMol25.atom_counts` remains the subset projection.

## AC-001 — One wrap, no discovery

`_official_vector_counts` is the only molnex conversion of a MolHub count snapshot. Missing `cache` is `AttributeError`. `getattr` / `hasattr` / `except AttributeError` around that call is a failure of this criterion.

## AC-002 — Historical path, exact filter, hidden storage

Deleting the hint cache is the point: a 400-atom row cannot sneak through `max_atoms=32` because nobody has decoded it yet. Tiny prefix reads `len` and `gather`, not `numbers` and not `._counts`.

## AC-003 — Projection, not a second owner

`LazyOMol25.cache` would be a sibling of `OMol25.cache`. `record` would be a second mutable owner of a fully-known vector. Disagreement is an error, not a write. `_fit_e0_subsample` is not collateral damage.

## AC-004 — e0 root follows the singleton

ε npy stays under `derived/omol25-index`. Only the home resolver changes. Tests restore the singleton so the suite never writes the developer `~/.cache/molhub`.

## AC-005 — Profile uses the same wrap and the same keep

`build_lazy_pair` previously passed a Python list that cannot `gather` source indices and kept `n <= max_atoms` (measured 0 survived). `_official_vector_counts` plus `1 ≤ n ≤ max_atoms` is the fix. `run_profile` must set home itself once sbatch stops exporting it. The zeros `AtomicReferenceEnergy` without `supported=` is rot in that same function.

## AC-006 — Env is not Python home

The helper always assigns the nobackup Path. Early-return on `$MOLHUB_HOME` would keep env as a second home. Shell `export` is removed even when it only "defaults" the variable.

## AC-007 — Runtime goldens in mirrors

`regressions/` was deleted 2026-09-07. The three mirrored modules plus the literals in AC-001/002/003/004/005 are the public-API bar. `/mol:impl` verifies this criterion by running those paths.

## AC-008 — Docstrings

Dimensionless counts; `home` is a `Path`; field name `graph.natoms`; `atom_counts` is a subset of an immutable vector.
