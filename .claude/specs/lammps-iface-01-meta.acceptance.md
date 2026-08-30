---
slug: lammps-iface-01-meta
retired_ids: []   # ids are append-only within the chain; a dropped criterion is
                  # recorded here with its reason, never reused.
criteria:
  - id: ac-001
    summary: preset_energy_ratio resolves styles through molpy UnitSystem
    type: code
    pass_when: |
      src/molix/units.py defines preset_energy_ratio(source_style, target_style)
      resolving both styles via UnitSystem.preset(...).base_units["energy"] and
      pint .to(); the file contains no numeric conversion literal, and
      preset_energy_to_md keeps its existing signature and behavior.
    status: verified
    last_checked: 2026-08-30
  - id: ac-002
    summary: conversion ratios match the SI-defined kcal/mol and eV values
    type: scientific
    pass_when: |
      TestPresetEnergyRatio passes: preset_energy_ratio("real","metal") is within
      relative 1e-6 of the SI-exact hand-computed 0.043364104241800934,
      preset_energy_ratio("metal","real") within relative 1e-6 of
      23.06054783061903 (the 4-significant-figure short forms are NOT satisfiable
      at rel 1e-6), self-ratio == 1.0 exactly, round-trip product within 1e-12 of
      1.0, and an unknown style raises.
    status: pending
  - id: ac-003
    summary: meta lammps block carries exactly the v2 field set
    type: code
    pass_when: |
      TestLammpsMetaBlock passes: LAMMPS_META_SCHEMA == 2; the block key set equals
      {schema_version, contract, execution, cutoff, units, energy_unit, length_unit,
      energy_conv, species, model_dtype, inputs, outputs, adapter, supports_pbc,
      supports_virial} plus {n_atoms, e_max} iff execution == "cuda-graph"; the key
      "cuda_graph" is absent from the block.
    status: verified
    last_checked: 2026-08-30
  - id: ac-004
    summary: contract and execution are the only two meta discriminators
    type: code
    pass_when: |
      contract == "cutoff-graph" on both export paths; execution == "dynamic" when
      cuda_graph is False and "cuda-graph" when True; grep of src/molix/engine/export.py
      finds no occurrence of "molnex-tensordict-static"; the meta adapter value is a
      member of EngineAdapter.names() on every export path.
    status: verified
    last_checked: 2026-08-30
  - id: ac-005
    summary: silent meta misreports in export are fixed
    type: code
    pass_when: |
      outputs == ["energy","forces","virial"] when supports_virial is True and
      ["energy","forces"] otherwise; TestExportForLammpsValidation shows
      export_for_lammps(..., cuda_graph=True, adapter=<non-default>) raises ValueError
      instead of ignoring the argument.
    status: verified
    last_checked: 2026-08-30
  - id: ac-006
    summary: energy_conv and length_unit are stamped from molix.units
    type: code
    pass_when: |
      the block's energy_conv keys equal the supported unit-style set {"real","metal"},
      every value is produced by preset_energy_ratio, energy_conv[units] == 1.0, and
      length_unit == "angstrom".
    status: verified
    last_checked: 2026-08-30
  - id: ac-007
    summary: engine-neutral header carries no LAMMPS vocabulary
    type: code
    pass_when: |
      interface/include/molnex/interface/meta_schema.h includes no torch/, ATen/ or
      LAMMPS header, exposes only generic primitives (device, has_block, block,
      string and string-array reads) and contains neither the token "lammps" as an
      API name nor any unit-style, contract or execution vocabulary; LammpsContract,
      UnitConversion, Contract and Execution live in interface/lammps/lammps_contract.h,
      which includes no LAMMPS header.
    status: verified
    last_checked: 2026-08-30
  - id: ac-008
    summary: MetaSchema and LammpsContract ctests pass without LAMMPS or fixtures
    type: runtime
    pass_when: |
      ctest runs meta_device_from_bare_export, meta_scoped_block_shadowing,
      meta_string_array, meta_block_missing_returns_absent,
      contract_schema_version_mismatch_throws, contract_unknown_contract_throws,
      contract_unknown_execution_throws, contract_unsupported_unit_style_throws,
      contract_length_unit_not_angstrom_throws and contract_input_arity_mismatch_throws
      green from a build configured without LAMMPS_HEADER_DIR and without LAMMPS_LIB,
      none of them declaring FIXTURES_REQUIRED; test_lammps_contract reaches
      lammps_contract.h through an include path only, and no CMakeLists adds
      interface/lammps as a subdirectory.
    status: verified
    last_checked: 2026-08-30
  - id: ac-009
    summary: bare Exporter metas still load after the ModelRunner migration
    type: runtime
    pass_when: |
      the pre-existing interface ctest list (cpu_run_matches_reference,
      update_weights_from_path, update_weights_from_map, parameter_info,
      concurrent_run_and_update, load_bad_path_throws) is green against fixtures whose
      meta.json has no lammps block and no schema_version.
    status: verified
    last_checked: 2026-08-30
  - id: ac-010
    summary: no conversion float is restated in C++ or docs
    type: code
    pass_when: |
      grep for "0.0433641" and "23.0605" across interface/src, interface/include
      and interface/lammps returns no match (test goldens under interface/tests
      are spec-mandated literals, not restatements); energy_unit_in_eV is absent
      from interface/lammps/pair_molnex.cpp; the units section of
      interface/lammps/README.md documents the meta energy_conv table instead of
      tabulating factors.
    status: verified
    last_checked: 2026-08-30
  - id: ac-011
    summary: regression pins decimal-literal goldens with provenance, no subprocess
    type: runtime
    pass_when: |
      PYTHONPATH=src python regressions/lammps-iface-01-meta.py prints OK and exits 0,
      asserting preset_energy_ratio("real","metal") against the literal 0.043364104 and
      preset_energy_ratio("metal","real") against 23.060548 at rtol 1e-7, plus the
      literal LAMMPS_META_SCHEMA == 2; the file starts no subprocess, makes no network
      call, carries a Provenance block in the regressions/oplsaa-parameterizer-02-heads.py
      form, and nowhere compares a molpy-derived value against another molpy-derived value.
    status: verified
    last_checked: 2026-08-30
  - id: ac-012
    summary: schema v2 semantics and the version break are documented
    type: docs
    pass_when: |
      meta_schema.h and lammps_contract.h docstrings follow google style with units;
      they state that Execution is a strategy selector whose anticipated `torchscript`
      value requires a schema_version bump, that force = energy is licensed only by the
      length_unit == "angstrom" assertion, that device() is a top-level-only field which
      throws on a block() sub-document, that the Contract dispatch pattern's binding
      consumer is lammps-iface-04-facade, and that fixed-graph's future payload is
      bond_index (2, N) COO covalent topology and never edge_index (E, 2); export.py's
      module docstring lists the v2 field set and its export_for_lammps docstring
      carries a Raises: entry for the adapter-with-cuda_graph ValueError;
      interface/lammps/README.md records that v1 plugins must not load v2 exports
      because the retired cuda_graph key would silently default to false.
    status: pending
  - id: ac-013
    summary: the generated v2 fixture ties the Python and C++ schema constants
    type: runtime
    pass_when: |
      interface/tests/generate_fixtures.py calls _lammps_meta_block to emit one real
      schema-v2 meta, contract_reads_generated_v2_fixture parses it through
      LammpsContract green, and the required CI interface job in
      .github/workflows/ci.yml runs the no-fixture list via an explicit ctest filter
      (never bare ctest, which would trigger the FIXTURES_SETUP Python export);
      bumping either LAMMPS_META_SCHEMA or the C++ expected version alone turns
      contract_reads_generated_v2_fixture red.
    status: verified
    last_checked: 2026-08-30
  - id: ac-014
    summary: engine tests mirror their source modules with every case preserved
    type: code
    pass_when: |
      tests/test_molix/test_engine.py no longer exists; tests/test_molix/test_engine/
      holds __init__.py, test_adapter.py, test_export.py and test_static.py mirroring
      src/molix/engine/{adapter,export,static}.py; all 8 pre-existing cases survive the
      split (test_registry_has_builtin_adapters, test_unknown_adapter_raises,
      test_tensordict_adapter_edge_convention, test_lammps_forward_shapes_tensordict,
      test_lammps_forward_shapes_flat, test_export_for_lammps_validates_inputs,
      test_export_for_lammps_cpu_refused, test_export_for_lammps_roundtrip), the last
      keeping @pytest.mark.slow plus its CUDA skipif; test_static.py adds 2 real
      StaticForward cases (the mirror module had none), so post-split
      `pytest tests/test_molix/test_engine/ -q` reports 9 passed, 1 skipped on CPU
      (7+1 from the preserved originals, +2 new).
    status: verified
    last_checked: 2026-08-30
---

# Acceptance criteria

- **ac-001 / ac-002** 锁住"换算值由 molpy pint 求得、不复述浮点"这条 CLAUDE.md 硬规，并给出可独立核验的物理数值（Domain basis 的精确定义常量）。
- **ac-003 / ac-004 / ac-006** 是 schema v2 的契约本体。ac-004 特别锁住"判别子净数不增"：`contract` 与 `execution` 正交，`cuda_graph` 布尔与 `molnex-tensordict-static` 假名同时消失。
- **ac-005** 是本 spec 起草时发现的两处静默失真（iron law），必须与结构迁移同批落地。
- **ac-007 / ac-010** 锁住"引擎中立头零 LAMMPS 词汇"与"换算浮点零复述"两个删除类结果——删除很容易在后续 spec 中被悄悄复活，故用 grep 级判据。
- **ac-008** 的"不依赖 LAMMPS 构建"是 `02-view` / `03-graph` 组件 ctest 的先决形状。
- **ac-009** 是本次迁移最容易被忽略的回归面：裸 `Exporter` 的 meta 没有 `lammps` 块也没有 `schema_version`，正是"schema_version 校验必须落在 `LammpsContract` 构造而非 `MetaSchema` 构造"这一设计裁决的来源。
- **ac-011** 的容差是 1e-6 相对而非逐位相等——替换硬编码 `0.0433641043` 会带来 ~1e-8 的相对偏移，逐位判据在构造上不可满足。
- **ac-012** 防止 AOTI 假设被写死进字段语义，为已记录在案的 TorchScript 路线留门。`fixed_graph` 的"链终未实现即删除"义务是**链级待办**（见 spec Out of scope），不在本判据内——它无法在 01 关闭时判定。
- **ac-013** 把 Python 侧与 C++ 侧的 schema 常数在测试时绑死，并将 CI 门（过滤式 ctest）纳入判据——没有 CI 门，本链 C++ 判据全体退化为本地债。
- **ac-014** 的用例清单与计数是钉死的（8 例、7 通过 1 跳过），"preserving every case" 因此可验证而非口头。
