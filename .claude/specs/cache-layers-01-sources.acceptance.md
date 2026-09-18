---
slug: cache-layers-01-sources
created: 2026-09-11
criteria:
  - id: ac-001
    summary: atom_counts identity is schema+shard+row-map, not path+len
    type: code
    pass_when: |
      OMol25Lmdb.atom_counts identity includes ADAPTER_SCHEMA_VERSION, ordered
      per-shard (filename, size) of _paths, and SHA-256 of _index.tobytes().
      There is no public revision= argument. A sidecar hit requires all
      components. path+len(self) alone never identifies the counts file.
      Equal-length shard replacement, reordered _index, schema bump, and
      max_rows change miss; a matching identity does not re-decode records.
      When OMol25.load knows the catalogued tarball size, that integer is
      folded into the private sidecar key so path-only OMol25Lmdb(root) does
      not share the file.
    status: verified
    last_checked: 2026-09-12
    verified_by: tests/test_dataset/test_omol25.py TestOMol25Lmdb identity tests
  - id: ac-002
    summary: Default 4M; pinned val; domain views via assignment
    type: code
    pass_when: |
      OMol25() defaults to role "4m". MANIFEST catalogs file roles 4m, val,
      val-neutral, and omol-elem-refs only. val filename is val.tar.gz, locator
      is https://dl.fbaipublicfiles.com/opencatalystproject/data/omol/250514/val.tar.gz,
      size is 21293980744. OMol25(role=) accepts only 4m, val, val-neutral.
      OMol25(role="omol-elem-refs") and OMol25(role="val_spice") raise ValueError
      listing those tarball roles (and assignment names separately). After
      OMol25(role="val").load(), assignment() maps val_metal_complexes,
      val_electrolytes, val_biomolecules, val_spice, val_neutralorganics to
      parent indices whose record["data"]["data_id"] is in the pinned set,
      parent order kept. Callers wrap with SubsetDataset(..., name=).
      val-neutral is not the default. max_rows is not a constructor argument
      for domain names because those names are not roles.
    status: verified
    last_checked: 2026-09-12
  - id: ac-003
    summary: Official frames stay raw totals; load uses ArtifactHub
    type: code
    pass_when: |
      Official OMol25 frames expose frame.meta["energy"] as the published DFT
      total with fx/fy/fz present, and have no e0 or energy_residual keys.
      OMol25.load fetches the tarball role through an injected ArtifactHub.
      DownloadCache remains defined in src/molhub/dataset/cache.py and is not
      the official OMol25 fetch path. Frames have no e0 or energy_residual.
    status: verified
    last_checked: 2026-09-12
  - id: ac-004
    summary: omol-elem-refs is a Manifest file; fetch returns Path
    type: code
    pass_when: |
      MANIFEST.artifacts["omol-elem-refs"] has filename uma_v1_hof_lin_refs.yaml,
      format yaml, locator containing commit 41d36dc6bf8ced6348dc73e6aaf54d0527ebd8c9,
      size 11549, and digest None. molhub.dataset does not export OMolElemRefs.
      hub.fetch(COORDINATE, roles=["omol-elem-refs"])["omol-elem-refs"] is a
      Path. No class on this module has load() -> Path for the YAML.
    status: verified
    last_checked: 2026-09-12
  - id: ac-005
    summary: Unit-test goldens for SHA-256 and native raw totals
    type: runtime
    pass_when: |
      uv run pytest tests/test_dataset/test_omol25.py -v is green. TestOMol25
      asserts SHA-256 literal e89e1abbcc000a810eb4aed62540389712e0c335bb960c85ff985920cd42f640
      of the YAML bytes served by the fake hub (comment: sha256sum, FAIR commit
      41d36dc6bf8ced6348dc73e6aaf54d0527ebd8c9, 2026-09-11). TestOMol25Lmdb/TestOMol25
      assert the water fixture energy is exactly -76.4 eV, forces eV/Å, positions Å,
      and frame.meta has no e0 or energy_residual. A tiny val fixture assignment()
      keeps parent order. No third-party runtime and no regressions/ file.
    status: verified
    last_checked: 2026-09-12
  - id: ac-006
    summary: Google docstrings declare eV, eV/Å, and Å
    type: docs
    pass_when: |
      OMol25Lmdb, OMol25, and MANIFEST module/class docstrings name energy eV,
      forces eV/Å, and positions Å, state that omol-elem-refs is hosted as a
      Manifest file role not fitted, and state that domain views are assignment()
      projections of official val.
    status: verified
    last_checked: 2026-09-12
    verified_by: agent-auto
---

# Acceptance criteria

Done means MolHub official OMol25 is a durable native-unit source with official 4M/val file roles, in-memory domain `assignment()`, a strong `atom_counts` identity, and a hosted `omol-elem-refs` YAML fetched through `ArtifactHub`, without residuals on disk and without a second public host type.

## AC-001 — atom_counts identity

Equal-length replacement of an aselmdb, a reordered `_index`, a schema bump, and a `max_rows` change must miss the sidecar. A matching identity must not re-decode records. No public `revision=`.

## AC-002 — official roles and assignment

Default is official 4M train. Official val is the pinned `val.tar.gz`, not `val-neutral`. Domain views are ordered projections of official val using one field, `record["data"]["data_id"]`, wrapped by the caller in `SubsetDataset`. `OMol25.role` never means YAML or a filter.

## AC-003 — raw totals and fetch seam

Frames keep published DFT totals. `load` talks to `ArtifactHub`. `DownloadCache` stays for URL datasets; `AtomicDress` stays off this training source. Tests assert frame/`load` behavior.

## AC-004 — YAML is a file role

Commit-pinned locator + size 11549 + `digest=None`. Callers compose `fetch`. SHA-256 is not catalogued on `Artifact`.

## AC-005 — goldens

The mirrored unit test is the public-API golden (no `regressions/` file). Literals are the YAML SHA-256 and the water raw total `-76.4` eV.

## AC-006 — units in docs

Google-style docstrings carry the native units so consumers do not convert twice.
