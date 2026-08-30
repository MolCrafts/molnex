---
slug: lammps-iface-02-view
retired_ids: []   # ids are append-only within the chain; a dropped criterion is
                  # recorded here with its reason, never reused.
criteria:
  - id: ac-001
    summary: LammpsView is a LAMMPS-free POD carrying only live-consumer fields
    type: code
    pass_when: |
      interface/lammps/bridge/lammps_view.h defines an aggregate LammpsView with
      exactly x, f, type, nlocal, nghost and nall(); it includes no LAMMPS header,
      declares no user-provided constructor, owns no memory, and contains no tag,
      charge, ilist, numneigh, firstneigh or NEIGHMASK member.
    status: verified
    last_checked: 2026-08-30
  - id: ac-002
    summary: bridge target builds and tests without any LAMMPS input
    type: runtime
    pass_when: |
      configuring interface/ without -DLAMMPS_HEADER_DIR and without -DLAMMPS_LIB
      builds molnex_lammps_bridge and runs the bridge_* ctest cases green; no file
      under interface/lammps/bridge/ includes a LAMMPS header or contains the token
      NEIGHMASK.
    status: verified
    last_checked: 2026-08-30
  - id: ac-003
    summary: SpeciesMap validates a 0-based dense table against meta species
    type: code
    pass_when: |
      SpeciesMap takes a 0-based z_by_type of length ntypes with no placeholder slot and
      derives ntypes() from it; the constructor throws std::runtime_error when any entry
      is non-positive or absent from meta_species, naming the offending LAMMPS type, its Z
      and every valid species; z_of throws for lammps_type <= 0 and for lammps_type >
      ntypes().
    status: verified
    last_checked: 2026-08-30
  - id: ac-004
    summary: bridge component ctest cases pass over a synthetic LammpsView
    type: runtime
    pass_when: |
      ctest runs bridge_species_map_maps_types, bridge_stage_nodes_shapes_and_values,
      bridge_force_scatter_accumulates, bridge_species_map_rejects_unknown_z,
      bridge_species_map_rejects_bad_table_size,
      bridge_species_map_rejects_nonpositive_z,
      bridge_species_map_z_of_rejects_out_of_range_type and
      bridge_force_scatter_rejects_wrong_row_count green, none of them declaring
      FIXTURES_REQUIRED.
    status: verified
    last_checked: 2026-08-30
  - id: ac-005
    summary: force scatter accumulates rather than overwrites, in LAMMPS units
    type: code
    pass_when: |
      bridge_force_scatter_accumulates starts from a non-zero f and asserts
      f_out == f_in + force_conv * forces within 1e-12 against hand-computed
      literals; ForceScatter throws when forces.size(0) != n_graph.
    status: verified
    last_checked: 2026-08-30
  - id: ac-006
    summary: meta species integer array is read whole, with its consumer present
    type: code
    pass_when: |
      LammpsContract::species() returns the full integer vector (not the first
      element), rejects non-integer and negative entries, and its only caller is the
      SpeciesMap construction site in PairMolnex::coeff.
    status: verified
    last_checked: 2026-08-30
  - id: ac-007
    summary: PairMolnex no longer owns the atom-side tables or loops
    type: code
    pass_when: |
      interface/lammps/pair_molnex.h declares no type2z vector and no force_conv member
      (ForceScatter owns the factor; energy_conv stays with its live readers);
      pair_molnex.cpp contains no per-atom type-to-Z, position-staging or
      force-accumulation loop and instead calls SpeciesMap, stage_nodes and
      ForceScatter, held as std::optional members guarded once at the top of compute();
      the nghost guard, energy and virial tally and CUDA-graph branch are unchanged.
    status: verified
    last_checked: 2026-08-30
  - id: ac-008
    summary: no tag-to-local mapping is introduced anywhere
    type: code
    pass_when: |
      grep for tag across interface/lammps/bridge/ and the 02 diff of
      interface/lammps/pair_molnex.{h,cpp} returns no atom->tag access and no
      tag-keyed container; the concern remains entirely unstarted and is routed to the
      future fixed-graph contract spec.
    status: verified
    last_checked: 2026-08-30
  - id: ac-009
    summary: no allocation or dtype behavior is smuggled into the migration
    type: code
    pass_when: |
      stage_nodes allocates Z and pos per call (no persistent buffer member) and
      returns pos as float64, leaving the model-dtype cast at its existing call site
      in compute(); no pinned, cached or reused host buffer is introduced.
    status: verified
    last_checked: 2026-08-30
  - id: ac-010
    summary: pre-existing interface ctests stay green after the migration
    type: runtime
    pass_when: |
      the interface ctest list carried over from earlier specs (cpu_run_matches_reference,
      update_weights_from_path, update_weights_from_map, parameter_info,
      concurrent_run_and_update, load_bad_path_throws and the meta_* / contract_* cases)
      is green after PairMolnex, MetaSchema and LammpsContract are extended.
    status: verified
    last_checked: 2026-08-30
  - id: ac-011
    summary: bridge ctest cases gate in CI, not only locally
    type: runtime
    pass_when: |
      .github/workflows/ci.yml has an interface job that installs the CPU torch wheel and
      cmake, configures interface/ without LAMMPS_HEADER_DIR and without LAMMPS_LIB, and
      runs the no-fixture ctest list including every bridge_* case added here; no
      regressions/lammps-iface-02-view.py exists, and no file under regressions/ starts a
      subprocess.
    status: verified
    last_checked: 2026-08-30
  - id: ac-012
    summary: bridge boundary is documented at the type level
    type: docs
    pass_when: |
      lammps_view.h, species_map.h, node_staging.h and force_scatter.h carry
      google-style docstrings with tensor shapes and units, stating that LammpsView
      borrows LAMMPS-owned memory, that bridge code must stay free of LAMMPS headers,
      and that neighbor spans plus neigh_mask arrive with GraphBridge in 03-graph.
    status: pending
  - id: ac-013
    summary: n_graph is validated before any indexing of LAMMPS-owned memory
    type: code
    pass_when: |
      stage_nodes and ForceScatter::scatter throw when n_graph < 0 or n_graph >
      view.nall(), and ctest cases bridge_stage_nodes_rejects_n_graph_over_nall and
      bridge_force_scatter_rejects_n_graph_over_nall assert the throw rather than an
      out-of-bounds write into f.
    status: verified
    last_checked: 2026-08-30
  - id: ac-014
    summary: the pbc energy over-count defect is named and routed
    type: docs
    pass_when: |
      the spec's Out of scope names pair_molnex.cpp:332-333 booking the model's total
      local+ghost energy into eng_vdwl under supports_pbc=true as a live defect, states it
      is reachable via export_for_lammps(supports_pbc=True), and routes it to
      lammps-iface-05-pbc; nothing in this spec attempts the fix.
    status: pending
---

# Acceptance criteria

- **ac-001 / ac-008 / ac-009** 是本 spec 的"没有夹带"三条守卫：视图字段不超前于消费者、tag 完全未开工、分配与 dtype 行为零变化。它们存在的原因是本 spec 的验收标准是 parity——任何夹带都会让 parity 判据失去意义。
- **ac-002** 把"bridge 静态目标由父侧声明"这一构建拓扑变成可执行判据：判定标准不是 CMake 写法，而是**在完全没有 LAMMPS 输入的配置下 ctest 能绿**。这条同时是 `03-graph` 组件测试的先决条件。
- **ac-003 / ac-005** 覆盖本 spec 修复的静默失真与最易写反的语义（`+=` 而非 `=`）。ac-005 特意要求以非零初始 `f` 起测——用零初始力测不出覆盖与累加的区别，而 `virial_fdotr_compute` 与其它 pair style 的贡献都依赖累加语义。
- **ac-006** 把整数数组读取器与其消费者绑定，防止 01 推迟的那一项落地成无人调用的通用读取器。
- **ac-007** 是结构迁移本身的判据，措辞刻意限定"其余不动"，把 facade 削薄挡在 `04-facade`。
- **ac-011** 记录本 spec 的门禁归属（ctest + CI job，无 regressions 脚本的显式例外）。
- **ac-013** 是 architect 审查发现的越界写风险的封口：`n_graph` 组件化后成为跨边界入参，校验不再可选。
- **ac-014** 按 iron law 把本 spec 相邻代码上的活缺陷具名留痕并路由 `05-pbc`。
