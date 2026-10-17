# Removing backward-compat apparatus before 1.0 (inventory, 2026-10-07)

**The user's ruling:** "1.0之前不要任何的schema version、migration、changelog这些向后兼容的东西". Before 1.0 there are no schema versions, migration guides, changelogs or release notes, deprecation shims, legacy spellings, readers for old layouts, or retired-name test lists.

**Keep:**
- the FFI ABI handshake;
- package-version gates for an installed dependency (molpy `check_molrs_version`);
- program/tool version fields (`creator.version`);
- third-party formats' own legacy handling (TRR headers, SDF without `$$$$`);
- structural rule tests (casing, door placement, module boundaries, "one public path" checks).

The inventory below was taken at molrs b7439db5, molpy e38f27d, molpack 31acb10 and molrec d18356d. Line numbers are approximate; re-grep before editing.

## molrs (molrs-p-registry)

**Docs and CI**
- Delete `molrs-python/docs/migration.md`, `molrs-python/docs/release-notes.md` and `docs/migration.md` (a stub).
- `zensical.toml:30-33`: remove the nav group "Release 0.16".
- Reword these so no version-history text remains:
  - `index.md:74-98`;
  - the READMEs (`README.md:165`, `molrs-python/README.md:56,69-70`, `molrs-wasm/README.md:10-11`);
  - `installation.md:158-162`, `quickstart-rust.md:88-89`;
  - `guides/forcefield-ir.md`: drop the "0.16" column from the style tables; lines 50-53, 78, 208, 788, 1389-1396;
  - `guides/records.md:59-63,144-145,229`;
  - `reference/python.md:36`.
- `docs/releasing.md:51-56`: drop the migration/release-notes steps.
- `.github/workflows/publish.yml:340-341`: release notes text → plain.

**Schema versions**
- **`molrec_version`:**
  - `io/mrec/record.rs` (`MOLREC_VERSION`, `RESERVED_META_KEYS`);
  - `io/mrec/mod.rs`;
  - `io/mrec/validation.rs` (`validate_meta`, `molrec_version_of`, `stamp`, and the tests at 99-195);
  - `zarr_storage/record_io.rs` (stamping, version validation, `read_upgrade`, `V1Upgrade` re-export, the upgrade paths, version tests);
  - `sequence.rs` (3715-3746 refusing appends at a non-current version; `MrecReader.upgrade`; test 7992-8006);
  - `forcefield_io.rs` (`upgrade` param);
  - `frame_io.rs:778-780` (doc);
  - Python: `io/mrec/mod.rs` (`MOLREC_VERSION`), `molrs/io/mrec/__init__.py`, `validation.py`, `_native.pyi:4149`;
  - `tests/test_mrec.py:69,98-129,266`.
- **`FRAME_VOCAB_VERSION`:**
  - `core/schema/mod.rs:67-83`;
  - `core/schema/document.rs` (`vocab_version`; id `/schema/frame/v1` → no version in the id);
  - capi `molrs_schema_vocab_version` (+ cpp tests);
  - wasm `schema_vocab_version`;
  - Python `VOCAB_VERSION`.
- **WASM catalog `version: 6`:** `molrs-wasm/src/compute/catalog.rs:271-272,1171-1186` (and its history comments); `compute/mod.rs:154`.

**Old-layout readers**
- Delete `zarr_storage/record_v1.rs` (V1Upgrade) and its `mod` line.
- Delete `zarr_storage/testdata/v1/`.
- Delete the v1 tests in `record_io.rs:2391-2590` and in Python `test_mrec.py:597-645`.
- Regenerate `molrs-wasm/tests/fixtures/precision.mrec.zip` without a version; it is used by `molrs-wasm/src/io/mrec.rs:542` and `record_io.rs:2042`.
- `sequence.rs` `LEGACY_FRAMES_GROUP` / `ensure_not_legacy` (the ≤0.13 layout): delete, along with its tests (7942-7985; `record_io.rs:1131,1992-2031`).
- `sequence.rs:352-366` `dtype_from_tag` accepts legacy `"float"`/`"int"`/`"uint"` tags: drop that acceptance; fix the test at 6895-6911.
- The retired `.zarr`/`.zarr.zip` suffix refusal: `validation.rs:16-30`, `record_io.rs` (65,110,859,1059,1120,1165-1180), `pack.rs:14,85,139-144`, Python `io/mrec/mod.rs` (58,102,139,190,967-980), `test_mrec.py:131`. Remove the special-casing; an unknown suffix gets the ordinary error.

**Code-level compatibility**
- `core/keys.rs:131-137`: `units_preset` accepts the bare preset string. Remove that, and delete `io/lammps/molecule.rs:1107-1110`.
- `io/lammps/columns.rs:95-97` and `data.rs:1639-1648`: the legacy `molecule`/`molecule_id` → `mol_id` spellings. Remove them.
- `test_stream.py:17`: an ignored `DeprecationWarning`.
- Reword the history wording in:
  - `compute/msd/mod.rs:52`;
  - `op/numeric.rs:38-51`;
  - `frame_io.rs:792-799,1100,1208,2180`, `sequence.rs:715`, `core/metadata.rs:204`;
  - the spectroscopy/dielectric/transport "reproduces legacy" docstrings, plus the pyo3 and `.pyi` copies;
  - `ff/ir_invariance.rs` ("0.15.1 values" → plain reference values).
- `compute/spectroscopy/dielectric_spectrum.rs:948-1210`: delete the test-only copies of removed functions, or recast them as plain oracles.

**Retired-name tests**
- `test_public_paths.py`: delete these sections:
  - `test_retired_modules_do_not_import` (115-144);
  - `test_retired_names_are_absent` (147-311);
  - the S3 retired section (314-353);
  - the S6 retired and `wave_s6_names_exist` sections (643-685);
  - the retired-constant regex (697-714).
- `module_boundaries.rs:163-230`: `RETIRED_FACTOR_CONSTANTS`. Keep the generic factor-literal rules.
- `molrs-capi/src/forcefield.rs:866,877-879`.
- Single `assert not hasattr` checks for removed names in `test_io`, `test_compute`, `test_md`, `test_forcefield_builder`, `test_forcefield_lammps_reader`, `test_mrec`, `test_stream`, `test_smiles_emit`, `test_smiles_error` (the "Backward compatibility" section), `test_cgsmiles`, `test_ecs_pybind`, `test_element_typifier`, `test_optimize`, `test_opls_typifier`, `test_trace` and `test_block_dtype_contract`. Keep the ones that enforce "one public path" (e.g. `test_forcefield_builder.py:511`).

## molpy (molpy-016)
- **Docs and repo files:**
  - Delete `docs/getting-started/whats-new.md` and its nav entry (`zensical.toml:32`).
  - Reword the links to it: `README.md:155`, `getting-started/index.md:11`, `installation.md:54`, `tutorials/index.md:15`, `developer/release-process.md:28-30`.
  - Delete the release-notes template (`release-process.md:113-130`).
  - Reword `release.yml:4`, `.github/copilot-instructions.md:43` (the "MUST maintain backward compatibility / migration note" rule contradicts the ruling) and `contributing.md:29`.
  - Reword `docs/api/compute.md:13-27` ("removed in 0.15" / migrating from `IonicConductivity`).
  - Reword `11_io.md:236-237` and `src/molpy/io/mrec.py:8` (`molrec_version`).
- **Tests:**
  - `tests/test_io/test_mrec.py:60,128-144`.
  - `test_public_paths.py`: delete `test_a_removed_module_is_gone` and `test_a_removed_name_is_gone`, plus the retired lists at 19-20.
  - Delete the `assert not hasattr` on removed names in `test_resources.py:70-72`, `test_ff/test_reexports.py:46-53`, `test_engine.py:100-103`, `test_lammps_relax.py:55` and `test_core/test_link.py:15`.
  - Reword the old-version comments in `test_lammps.py:334` and `test_lammps_molecule.py:135`.
- **`.pre-commit/check_molrs_pin_on_pypi.py`:** drop the "exact pin (legacy)" acceptance.

## molpack (molpack-016)
- **Tests:**
  - `python/tests/test_packing_methods.py:41-49`: delete the absence asserts and reword the comment.
  - Delete `test_region.py:56-65` and `test_state.py:101-104`.
  - `src/context/pack_state/tests.rs`: delete `legacy_unscaled` and the "matches_legacy" tests, or recast them as plain oracles.
- **Reword:**
  - `gencan_pack.rs:242-245`, `grow/tests/refusals.rs:370,472`, `grow/tests/mod.rs:269,318` ("deleted Molpack / migration record");
  - `objective.rs:509,1329`;
  - `docs/python/api-reference.md:359`.
- **`.github/PULL_REQUEST_TEMPLATE.md`:** remove the "Breaking changes" section.
- **Keep:** `script/parser.rs` `centerofmass`, which is a Packmol keyword.

## molrec (molrec-w3d)
- **Spec:**
  - Remove `molrec_version` and all version and migration text:
    - `docs/spec/overview.md:120,134-165`;
    - `docs/spec/forcefield.md` (delete "Reading a version-1 record" at 843-930; reword 95-102 and 285-286; drop the v1, reject-v1 and absent-version conformance cases at 968-975 and 993-1008);
    - `specification.md:17-20,108,112`;
    - `storage.md:134,141-142,259-266,72-78`;
    - `lmdb.md:40-72` (`layout_version`);
    - `ragged.md:391,463,483`, `zarr.md:54`, `observables.md:88-90`, `metrics-wal.md:68`;
    - `docs/layout.md`.
  - Reword `README.md:68-83`, `CLAUDE.md:37,110-126` and `.claude/specs/INDEX.md:7`.
  - `docs/design/2026-10-capabilities.md`: delete its version-decision and migration sections, or delete the whole design doc if it is history only.
- **Code:**
  - Delete `src/molrec/core/v1.py`, and keep `DEGREES_PER_RADIAN` where it is used.
  - Update `core/__init__.py`.
  - `core/model.py`: `MOLREC_VERSION`, `MolrecVersion`, `_VERSION_SCHEMA`, `stamp_version`, and the `molrec_version` fields of `MetaModel` and `CollectionMetaModel`.
  - The bindings: `zarr.py` (stamp, `upgrade_of`, upgrades) and `lmdb.py` (`LAYOUT_VERSION`, `_check_layout_version`, upgrades).
  - `observables/bindings/zarr.py`.
  - Reword `case.py`, `schema_export.py` and the observables/draft "v1/v2" docstrings.
  - Regenerate the JSON schemas under `schema/core/*`.
- **Suite:**
  - Delete `ffsuite.py` `_version_1` and the V1 block (839-1143).
  - Delete `suite.py` `_version_1_records` and the version cases (2124-2175, 2232-2270, 2627-2631).
- **Tests:**
  - Remove version use from `test_model.py`, `test_lmdb.py`, `test_zarr.py`, `test_observables_conformance.py`, `test_precision.py`, `test_forcefield.py` and `test_core_conformance.py`.
  - Delete `test_mrec_brand.py`.
  - Recast `test_pair14_is_retired…` as a test of the current unknown-category behaviour.
  - `tests/molrs_adapter.py:66-120`: delete the partner-version capability gates.
- **Fixtures:**
  - Delete the version-only fixtures (`valid-pre-1`, `invalid-newer-version`, `invalid-null-version`).
  - Regenerate the fixtures that carry `molrec_version` (meta, collection, record, `run-minimal`, `precision.mrec.zip`, README).
  - Rename `invalid-retired-*` fixtures to describe the current refusal.

## Atomiverse
Not in scope. Atomiverse is not one of the pre-1.0 molcrafts libraries under this ruling, so leave its changelog alone.
