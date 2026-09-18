---
title: Official OMol25 is the durable source
status: code-complete
grilled: true
created: 2026-09-11
chain: cache-layers
depends: none
supersedes: cache-layers-01-cache (durable-source slice)
---

# Official OMol25 is the durable source

## Summary

MolHub's official OMol25 reader is the durable training source: `OMol25Lmdb` keeps published totals in native eV / eV Å⁻¹ / Å, `OMol25()` defaults to the official 4M train, official val is a pinned tarball (not `val-neutral`), domain views are in-memory `assignment()` projections of that val wrapped by the caller in `SubsetDataset`, and `omol-elem-refs` is a Manifest file role fetched with `ArtifactHub.fetch` — not a second public type and not an `OMol25` role. Frames on disk stay RAW DFT totals. SHA-256 of the pinned YAML bytes is a unit-test golden only. `DownloadCache` is not renamed.

## Domain basis

Levine et al., "Open Molecules 2025 (OMol25)", 2025. https://arxiv.org/html/2505.08762v1 — DOI `10.48550/arXiv.2505.08762`. Dataset documentation: https://fair-chem.github.io/omol25/. Raw labels are total energy in eV and forces in eV/Å; positions are Å. MolHub `omol25.py` already keeps those native units and must not convert.

Pinned FAIR commit `41d36dc6bf8ced6348dc73e6aaf54d0527ebd8c9`. Official 4M protocol dataset definition: `configs/allscaip/dataset/omol_4M.yaml`. Train is the public tarball `train_4M.tar.gz`. Validation is the official general val split, with domain views as ordered filters of that val by `data_ids` (the same string decode already stores as `frame.meta["domain"]`):

| Assignment name | `data_ids` membership |
|---|---|
| `val_metal_complexes` | `metal_complexes` |
| `val_electrolytes` | `elytes` |
| `val_biomolecules` | `biomolecules` |
| `val_spice` | `spice` |
| `val_neutralorganics` | `ani2x`, `orbnet_denali`, `geom_orca6`, `trans1x`, `rgd` |

This slug hosts the full official val and the full domain membership in parent order. The recipe's 20k monitoring caps and their seeds belong to a later alignment slug; they are not a new split and not a substitute for official val.

Combined element table `omol_elem_refs` lives in `configs/allscaip/element_refs/uma_v1_hof_lin_refs.yaml` at the pinned commit. SHA-256 of the complete upstream YAML bytes, verified 2026-09-11 (`sha256sum`):

```text
e89e1abbcc000a810eb4aed62540389712e0c335bb960c85ff985920cd42f640
```

That hex is a unit-test golden of those bytes. It is not a catalog digest. This slug only hosts the file and its locator/size. Do not fit, do not subtract, do not write residual/`e0` columns. Applied later:

```text
e0[g]    = sum(c[Z[i]] for real atoms i in graph g)   # not this slug
E_res[g] = E_DFT[g] - e0[g]                           # not this slug
```

Limitation: the table's exact historical fit-sample IDs are not established by the public configuration. Do not claim it was fit exclusively on 4M.

## Design

Entities: existing `OMol25`, `OMol25Lmdb`, `MANIFEST` / `Artifact` in `molhub.dataset.omol25`; `ArtifactHub.fetch` → `Molhub.fetch` → `BlobStore`; `SubsetDataset` and `assignment()` in `molhub.dataset.protocol`. No new public type. Current caller: MolHub dataset load plus later molix consumption of the YAML `Path` from `fetch`. Current tests: `tests/test_dataset/test_omol25.py` (`TestOMol25Lmdb`, `TestOMol25`).

`OMol25Lmdb` stays the one LMDB reader. Construction, `record`, `__getitem__` and pickle/reopen are unchanged. Frames keep published totals: `frame.meta["energy"]` is the raw DFT total (eV); `fx`/`fy`/`fz` are raw forces (eV/Å); `x`/`y`/`z` are Å. Missing energy or forces still refuse. No `energy_residual`, no `e0`, no unit conversion, no `AtomicDress.execute` on this path. Decode already writes `record["data"]["data_id"]` to `frame.meta["domain"]`; that mapping stays.

**Manifest roles are downloadable files only.** `MANIFEST` (`dataset:fairchem/omol25@1`) catalogs four artifacts. Closest pattern: ThreeBPA (commit-pinned GitHub-raw locator + `size`, `digest=None`).

- `4m` — keep `filename="train_4M.tar.gz"`, locator `https://dl.fbaipublicfiles.com/opencatalystproject/data/omol/250514/train_4M.tar.gz`, `size=19983081456`.
- `val` — official general validation archive. `filename="val.tar.gz"`, locator `https://dl.fbaipublicfiles.com/opencatalystproject/data/omol/250514/val.tar.gz`, `size=21293980744` (HEAD 2026-09-11, HTTP 200). Do not reuse `neutral_val.tar.gz`.
- `val-neutral` — keep current `neutral_val.tar.gz` locator and size; non-default smoke tarball, never protocol val.
- `omol-elem-refs` — `filename="uma_v1_hof_lin_refs.yaml"`, `format="yaml"`, locator `https://raw.githubusercontent.com/facebookresearch/fairchem/41d36dc6bf8ced6348dc73e6aaf54d0527ebd8c9/configs/allscaip/element_refs/uma_v1_hof_lin_refs.yaml`, `size=11549` (GitHub contents size at that commit, 2026-09-11; the integer a HEAD Content-Length on the raw locator must report). `digest=None`. Do not put SHA-256 `e89e1abb…` on `Artifact.digest` — MolHub copies platform-published digests only and never invents one (same rule as `Artifact.from_mapping`). Identity of the file is the commit already in the locator plus catalogued size.

Callers compose `hub.fetch(COORDINATE, roles=["omol-elem-refs"])` and receive a `Path`. There is no `OMolElemRefs`, no `load() -> Path` façade, and no class attributes copying `COORDINATE` / `ROLE` / `SHA256` / `FAIR_COMMIT`. SHA-256 of those bytes lives only in `TestOMol25` as a hard-coded golden.

**`OMol25.role` is a tarball LMDB split.** Published constructor roles: `4m`, `val`, `val-neutral`. Default `role="4m"`. `OMol25(role="omol-elem-refs")` raises `ValueError`. `OMol25(role="val_spice")` (and any other assignment name) raises `ValueError`. Unknown names raise `ValueError` listing the three tarball roles and, separately, the five assignment names — no homemade-split / `80/20` pattern match.

`OMol25.load` fetches that one tarball role through `ArtifactHub` / `BlobStore`, extracts once behind `.extracted`, and opens `OMol25Lmdb`. It does not fetch YAML and does not scan domain filters. Official splits do not use `DownloadCache`.

**Domain views are not roles.** After `OMol25(role="val").load()`, `OMol25.assignment()` (the method `protocol.assignment()` already looks up) returns `{name: [parent_idx, …]}` for the five names in the Domain-basis table. Predicate is one field: `record["data"]["data_id"]` against that table (the same string as `frame.meta["domain"]`). Multi-id views are membership in the set, parent order preserved. Indices are built in memory from `record()` when assignment is requested; persist nothing — no domain sidecar. `4m` and `val-neutral` raise `TypeError` from `assignment()` (no published assignment). The caller wraps:

```text
val = OMol25(role="val", hub=hub).load()
spice = SubsetDataset(val, assignment(val)["val_spice"], name="val_spice")
```

Named `SubsetDataset` keeps the `#split=` source_id convention. `atom_counts` stays on the parent; a view projects `parent.atom_counts()[indices]`. Do not write a second count array.

**`max_rows` applies only to tarball LMDB indexing** (`4m` / `val` / `val-neutral`). Domain views are projections of a fully indexed official val. Smoke uses an injected `ArtifactHub` plus a tiny val aselmdb fixture, not `max_rows` on production val. `OMol25(role="val_spice", max_rows=N)` cannot construct.

**Generalize `OMol25Lmdb.atom_counts` identity** (not a `Cache` type; sidecar home stays beside the extract until a later slug). No public `revision=`. Identity of the view that always exists, all required:

1. `OMol25Lmdb.ADAPTER_SCHEMA_VERSION` (int, start at 1; bump when decode or row mapping changes).
2. Ordered per-shard `(filename, size)` of `_paths`. Never hash full aselmdb payloads on startup.
3. SHA-256 of `_index.tobytes()` (table_idx, row_id in view order). Reorder is a different identity.

When `OMol25.load` knows the catalogued tarball `size`, fold that integer into the same **private** sidecar key so a path-only `OMol25Lmdb(root)` and a Manifest-backed view do not share a file. Do not add a public constructor knob the caller can omit. Hit only on exact identity match. Equal-length shard replacement, reordered `_index`, schema bump, or `max_rows` change miss and rescan. Atomic write stays tmp + `replace`; read-only extracts still return the scan without persisting.

Ownership: MolHub owns durable bytes, official file roles, and provenance. molix later reads the YAML path from `fetch`. Residual arithmetic is not this slug. molnex `src/` is not edited and must not import molhub.

Law findings from the blocked draft, resolved here (caller / test named):

- 🔴 `OMolElemRefs` — dropped. Caller: `ArtifactHub.fetch`. Test: `TestOMol25` (fetch path + SHA-256 golden of bytes, not a type).
- 🔴 invented `Artifact.digest` — dropped. Catalog: commit in locator + `size=11549`. Test: `MANIFEST.artifacts["omol-elem-refs"].digest is None`.
- 🔴 overloaded `OMol25.role` — tarball roles only. Caller: `assignment` + `SubsetDataset`. Test: `TestOMol25` rejects `omol-elem-refs` / assignment names as roles and checks parent-order membership.
- 🔴 unpinned `val` — filename / locator / size written above. Test: `TestOMol25` asserts those three fields.
- 🔴 multi-field domain predicate — one field `record["data"]["data_id"]`. Test: fixture rows distinguished only by that key.
- 🔴 `max_rows` × domain role — illegal because domain names are not roles. Test: tiny val fixture, no cap-as-protocol-view.
- 🟡 `revision=` — not public. 🟡 domain indices — RAM only. 🟡 unknown-role heuristic — list published names. 🟡 tests — frames / `load` / `assignment` / identity, not import choreography of `AtomicDress` / `DownloadCache`.

### Reuse decision

- reuse `OMol25Lmdb` — spec extends this reader; no second LMDB type.
- reuse `BlobStore` — official tarballs and `omol-elem-refs` fetch through `ArtifactHub` / `Molhub.fetch` into `BlobStore`; no parallel file layout.
- reuse `ArtifactHub.fetch` — YAML and tarball bytes; callers compose `fetch(COORDINATE, roles=[…])`. No fetch façade type.
- reuse `SubsetDataset` — domain views are `SubsetDataset(val, assignment(val)[name], name=name)`; do not add `OMol25.view`.
- generalize `OMol25Lmdb.atom_counts` identity — schema version, per-shard `(filename, size)`, SHA-256 of `_index.tobytes()`; not the `Cache` type.
- pattern-only `DownloadCache` — do not rename; not used for official OMol25.
- pattern-only `AtomicDress` — forbidden as the OMol25 training residual path; leave the type for other datasets.
- no `OMolElemRefs` — neither reuse nor new; Manifest artifact + `fetch` already return the `Path`.

## Files to create or modify

- `/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molhub/src/molhub/dataset/omol25.py`
- `/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molhub/tests/test_dataset/test_omol25.py`

## Tasks

- [x] Write failing unit tests for `OMol25Lmdb.atom_counts` identity (`tests/test_dataset/test_omol25.py` → `TestOMol25Lmdb`) (rework: identity is schema version + per-shard `(filename, size)` + SHA-256 of `_index.tobytes()`, not path+len and not a public `revision=`)
- [x] Generalize `OMol25Lmdb.atom_counts` identity in `src/molhub/dataset/omol25.py` to cover `ADAPTER_SCHEMA_VERSION`, per-shard `(filename, size)`, SHA-256 of `_index.tobytes()`, and a private Manifest size fold from `OMol25.load`
- [x] Write failing unit tests for `OMol25` official tarball roles, `assignment()` domain views, raw totals, and `omol-elem-refs` fetch (`tests/test_dataset/test_omol25.py` → `TestOMol25`) (rework: no domain-as-role, no `OMolElemRefs`; pin val locator/size; SHA-256 is a byte golden, not `Artifact.digest`)
- [x] Implement official 4M default, pinned `val` / `omol-elem-refs` Manifest entries, `assignment()` of official val, and role rejection in `src/molhub/dataset/omol25.py`
- [x] Add docstring per google style with units (eV, eV/Å, Å) on `OMol25Lmdb`, `OMol25`, and `MANIFEST` in `src/molhub/dataset/omol25.py` (rework: no `OMolElemRefs` type to document)
- [x] Embed hard-coded goldens in `tests/test_dataset/test_omol25.py` (SHA-256 `e89e1abbcc000a810eb4aed62540389712e0c335bb960c85ff985920cd42f640` of YAML bytes; water energy `-76.4` eV; comment: `sha256sum`, FAIR commit `41d36dc6bf8ced6348dc73e6aaf54d0527ebd8c9`, 2026-09-11; no `regressions/` file; no third-party runtime)
- [x] Run full check + test suite (mirrored omol25 tests 52 passed; pre-existing workers=2 import of molpy.io.mrec still fails)

## Testing strategy

Unit tests only under `tests/test_dataset/test_omol25.py`, mirroring `src/molhub/dataset/omol25.py`. Types: `TestOMol25Lmdb`, `TestOMol25`. One method per concern. No e2e, no network, no live FAIR/ASE import. Inject `ArtifactHub`. Smoke domain views use a tiny val aselmdb (optionally tarred for `load`), not `max_rows` on official val. Unit green is `uv run pytest tests/test_dataset/test_omol25.py -v` from the molhub repo. molnex `regressions/` was deleted 2026-09-07; this slug does not add `regressions/` in either repo.

Happy path: synthetic aselmdb water (`energy=-76.4` eV, zero forces eV/Å, positions Å) reads raw `frame.meta["energy"] == -76.4` with `fx` present; `OMol25()` default role is `4m`; `OMol25(role="val").load()` on a fake hub serving a tiny val fixture publishes `assignment()`; `SubsetDataset(val, assignment(val)["val_spice"], name="val_spice")` yields spice rows in parent order; `hub.fetch(COORDINATE, roles=["omol-elem-refs"])` returns a `Path`.

Edge cases: `atom_counts` sidecar miss when an equal-length shard is replaced (different file size), when `_index` row order changes, when `ADAPTER_SCHEMA_VERSION` differs, and when `max_rows` differs; hit when identity matches without re-decode; path-only `OMol25Lmdb` and Manifest-backed `OMol25.load` do not share a sidecar; unknown role and `omol-elem-refs` / `val_spice` as `OMol25.role` raise `ValueError` listing published tarball roles (and assignment names, separately); `val-neutral` is catalogued but is not `__init__` default; `assignment()` on `4m` raises `TypeError`; `val_neutralorganics` keeps parent order across `ani2x` then `orbnet_denali`; projected `parent.atom_counts()[indices]` matches view rows and writes no second npy; frames have no `e0` / `energy_residual` keys; `MANIFEST.artifacts["omol-elem-refs"].digest is None` and `size == 11549`; `MANIFEST.artifacts["val"]` has filename `val.tar.gz`, the pinned CDN locator, and `size == 21293980744`. Do not add a `TestOMol25` method whose only assertion is that `omol25.py` does not import `AtomicDress` or `DownloadCache`.

Domain validation (hard-coded): native units remain eV / eV Å⁻¹ / Å on the water fixture; YAML resource identity is SHA-256 `e89e1abbcc000a810eb4aed62540389712e0c335bb960c85ff985920cd42f640` at commit `41d36dc6bf8ced6348dc73e6aaf54d0527ebd8c9`. The fake hub may serve a local copy of those YAML bytes so the test can hash them; it never downloads GitHub.

## Out of scope

- `molhub.cache.Cache` type, coordinator publication, and AtomCountCache deletion (later cache slug). This slug only strengthens the existing beside-extract sidecar identity.
- Applying `omol_elem_refs` (fit, residual, `e0` columns, `AtomicReferenceEnergy`).
- `OMolElemRefs` or any other host type / `load() -> Path` façade for the YAML.
- Putting a self-computed SHA-256 on `Artifact.digest`.
- `AtomicDress.execute` as the OMol25 training residual path; renaming `DownloadCache`.
- molnex `src/` (`BatchTask`, collate, Residual, NeighborList, TensorFrame precision, compile). molnex must not import molhub.
- `PrepareJob` / conveyor / FrameStore materialization of OMol25.
- 100M All dump as a training source; homemade 80/20; using `val-neutral` as protocol val.
- Monitoring `sample_n=20000` caps and their seeds (alignment slug).
- Changing `OMol25Small` filter semantics.
- Adding `atom_counts` onto `SubsetDataset`; a public `OMol25Lmdb(revision=)`.
- A `regressions/` file in molnex or molhub.
