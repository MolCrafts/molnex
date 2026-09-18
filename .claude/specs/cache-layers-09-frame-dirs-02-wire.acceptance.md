---
slug: cache-layers-09-frame-dirs-02-wire
created: 2026-09-12
criteria:
  - id: ac-001
    summary: Official counts are numpy from source.cache; wrap type is gone
    type: code
    pass_when: |
      _official_vector_counts(source) calls source.cache("graph.natoms") and
      returns a numpy array. _VectorCounts is deleted and not importable.
      scripts/mm_param_learning/class_i_fit.py contains no "from molhub.cache
      import Cache", no Cache(, no class _VectorCounts, and no _subset_natoms.
      A source whose type name is NoCache and that has no cache method raises
      AttributeError (message may contain NoCache). The function body contains
      no getattr, hasattr, try, or except. An AttributeError raised inside a
      real cache method is not translated.
    status: verified
    last_checked: 2026-09-12
  - id: ac-002
    summary: Keep, prefix, and profile filter numpy with 1 <= n <= max_atoms
    type: code
    pass_when: |
      load_omol25_official and load_omol25_official_tiny call
      _official_vector_counts once per source. A stub vector [3, 400] with
      max_atoms=32 keeps only the 3-atom row (indices == (0,); cache-at-indices
      is 3). A measured 0 is dropped. load_omol25_official still returns
      tuple[LazyOMol25, LazyOMol25] with val-neutral. _official_prefix takes a
      numpy array and uses limit = min(n_atoms.shape[0], scan_cap) then
      int(n_atoms[i]). _index_omol_source uses numpy keep and returns (keep, e0)
      with no wrap. build_lazy_pair on [0, 3, 40, 2] with max_atoms=32,
      n_train=1, n_val=1 drops measured 0, trains the 3-atom row (indices
      == (1,)), and vals the 2-atom row (indices == (3,)). LazyOMol25 is
      constructed with two positional args (source, indices).
    status: verified
    last_checked: 2026-09-12
  - id: ac-003
    summary: Decode mismatch vs numbers.shape[0] or atoms.nrows raises
    type: code
    pass_when: |
      LazyOMol25 has no cache method (view.cache("graph.natoms") raises
      AttributeError) and no atom_counts in any form (view.atom_counts raises
      AttributeError; no constructor atom_counts=). Public attributes are
      source and indices (tuple of source rows). LMDB vs frame is
      getattr(self.source, "record", None) callable — the public record() from
      cache-layers-01-sources. Production __getitem__ contains no _tables.
      A cache vector [3] over a record() that decodes 2 atoms raises ValueError
      naming row 0, cache count 3, and decoded 2. A frame-shaped source with no
      record(), cache [3], and frame["atoms"].nrows == 2 raises ValueError
      naming row, cache 3, and decoded 2. frame.meta["n_atoms"] is not read.
      _fit_e0_subsample still exists.
    status: verified
    last_checked: 2026-09-12
  - id: ac-004
    summary: Labeled-source check is hasattr task_labels only
    type: code
    pass_when: |
      _is_labeled_source(obj) is hasattr(obj, "task_labels") only. LazyOMol25
      and a stub with task_labels are labeled. A list is not. An object with
      atom_counts and no task_labels is not. The function source does not
      mention atom_counts.
    status: verified
    last_checked: 2026-09-12
  - id: ac-005
    summary: Labeled ClassIFit uses DataModule(batch_size), no atom_counts
    type: code
    pass_when: |
      ClassIFit constructed with two two-arg LazyOMol25 views sets train_ds
      to the view (labeled path). fit.datamodule() does not raise a ValueError
      whose message contains "does not expose atom_counts".
      type(fit.datamodule().train_dataset) is LazyOMol25. Production
      ClassIFit.datamodule does not construct DataMixture and does not pass
      max_atoms_per_batch. LazyOMol25 has no packing field named atom_counts.
      _LabeledFrames still assigns self.atom_counts = dataset.atom_counts.
      src/molix is unmodified.
    status: verified
    last_checked: 2026-09-12
  - id: ac-006
    summary: Launchers gather cache numpy at public source and indices
    type: code
    pass_when: |
      There is no _subset_natoms. prod_day.run and quasi_prod.calibrate use
      np.asarray(view.source.cache("graph.natoms"))[list(view.indices)] for
      n_atoms min/max (prod_day max only; quasi_prod min and max) when
      len(view) else 0. A view over indices (0, 2) of [3, 40, 2] is [3, 2]
      (min 2, max 3). Their source does not contain train.atom_counts,
      val.atom_counts, _source, or _indices as the gather target.
    status: verified
    last_checked: 2026-09-12
  - id: ac-007
    summary: Python home is molhub_config.home_dir, not config[home]
    type: code
    pass_when: |
      class_i_fit.py still imports molhub.config as molhub_config.
      _e0_cache_path uses molhub_config.home_dir / "derived" /
      "omol25-index" / "{digest}-e0.npy" with the existing sha1 of
      root:len:max_atoms:n_fit:seed. prod_day._require_nobackup_molhub_home
      mkdirs the nobackup workspace path and assigns molhub_config.home_dir
      to that Path. scripts/mm_param_learning and
      tests/test_scripts/test_mm_param_learning contain no
      molhub_config["home"]. Tests restore home_dir and do not setenv
      MOLHUB_HOME.
    status: verified
    last_checked: 2026-09-12
  - id: ac-008
    summary: Mirrored unit goldens stand in for deleted regressions/
    type: runtime
    pass_when: |
      python -m pytest tests/test_scripts/test_mm_param_learning/test_class_i_fit.py
      tests/test_scripts/test_mm_param_learning/test_profile_class_i.py
      tests/test_scripts/test_mm_param_learning/test_prod_day.py
      tests/test_scripts/test_mm_param_learning/test_quasi_prod.py -v is green.
      Hard-coded literals include water-sized [3, 3] from
      cache("graph.natoms"), historical max_atoms=32 dropping a 400-atom
      row, measured 0 dropped in TestLoadOmol25Official and TestBuildLazyPair,
      _e0_cache_path under tmp/derived/omol25-index, AttributeError on
      NoCache and on view.atom_counts, ValueError on a 3-vs-2 record()
      mismatch and on a 3-vs-2 frame["atoms"].nrows mismatch, and
      cache-at-indices [3, 2] from public indices (0, 2). No regressions/
      script. No FAIR/ASE/MACE runtime.
    status: verified
    last_checked: 2026-09-12
  - id: ac-009
    summary: Google docstrings name dimensionless counts and Path home_dir
    type: docs
    pass_when: |
      _official_vector_counts, _official_prefix, _index_omol_source,
      LazyOMol25, _sample_from_record, _is_labeled_source,
      ClassIFit.datamodule, _e0_cache_path, and
      _require_nobackup_molhub_home have Google-style docstrings. Counts are
      dimensionless atoms per row. Home is a pathlib Path via home_dir.
      Field name graph.natoms is named. Public source and indices are named.
      Decode mismatch names cache_row versus numbers.shape[0] or
      frame["atoms"].nrows. record() is named as the 01-sources LMDB API.
      Units are not invented for a non-physical index.
    status: verified
    last_checked: 2026-09-12
---

# Acceptance criteria

Done means molnex Class-I OMol scripts filter `source.cache("graph.natoms")` as numpy with `1 ≤ n ≤ max_atoms`, `_VectorCounts` and every `LazyOMol25.atom_counts` are gone, the view exposes public `source` / `indices`, decode disagreement is vs `numbers.shape[0]` or `frame["atoms"].nrows`, labeled detection is `task_labels` only, labeled `datamodule` is `DataModule(batch_size)` not `DataMixture`, launcher min/max gather the cache at those public attributes, and Python home is `molhub_config.home_dir`.

## AC-001 — Numpy, no wrap, no Cache()

`_official_vector_counts` is the only molnex conversion of a MolHub count snapshot. Missing `cache` is `AttributeError`. A new wrap type or `_subset_natoms` would fail this criterion.

## AC-002 — Exact filter on the array

A 400-atom row cannot sneak through `max_atoms=32`. Tiny prefix reads `shape[0]` and `n_atoms[i]`. Profile shares the same predicate and two-arg `LazyOMol25`. Kept rows are public `indices`.

## AC-003 — One home at decode

The source cache row is the count. LMDB is public `record()`; disagreement is `numbers.shape[0]`. Frame-dir / packed disagreement is `frame["atoms"].nrows`. Either way is `ValueError`, not a write. `frame.meta["n_atoms"]` is not a home.

## AC-004 — Labeled means task_labels

Dropping `atom_counts` must not send `LazyOMol25` down the `FrameDataset.from_records` path. `atom_counts` is not the discriminator.

## AC-005 — No packing field named atom_counts

Today's `DataMixture` wrap would raise `does not expose atom_counts`. The labeled path is `DataModule(batch_size)` with no token sampler. `_LabeledFrames` may still copy `FrameDataset.atom_counts`. `src/molix` stays unmodified.

## AC-006 — Launchers re-gather on public attributes

`prod_day` / `quasi_prod` must not treat a packing field or private `_source` / `_indices` as the MolHub index.

## AC-007 — Attribute home

`home_dir` is the 09-01-types write path. `["home"]` in these scripts/tests is a miss.

## AC-008 — Runtime goldens in mirrors

`regressions/` was deleted 2026-09-07. The four mirrored modules plus the literals in AC-001..007 are the public-API bar. `/mol:impl` verifies this criterion by running those paths.

## AC-009 — Docstrings

Dimensionless counts; `home_dir` is a `Path`; field name `graph.natoms`; public `source` / `indices`; mismatch names cache vs `numbers.shape[0]` / `nrows`.
```

```markdown
