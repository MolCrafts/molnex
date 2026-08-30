---
slug: lammps-iface-03-graph
retired_ids: []   # ids are append-only within the chain; a dropped criterion is
                  # recorded here with its reason, never reused.
criteria:
  - id: ac-001
    summary: LammpsView gains neighbor spans in the shape 02 pinned
    type: code
    pass_when: |
      lammps_view.h declares ilist, numneigh, firstneigh, inum and an int neigh_mask
      data member plus a neighbor(i, jj) accessor applying that mask; the token
      NEIGHMASK appears nowhere under interface/lammps/bridge/; no tag or charge
      member is added.
    status: verified
    last_checked: 2026-08-30
  - id: ac-002
    summary: GraphBridge preserves the cutoff refilter and the source-target sort
    type: code
    pass_when: |
      GraphBridge::build refilters candidate pairs against cutoff squared and sorts
      collected pairs lexicographically by (source, target) via std::sort, keeping the
      reused edge_pairs_/edge_flat_ host buffers; it takes n_graph and does not take
      supports_pbc. (Forward note: lammps-iface-05-pbc supersedes the unconditional
      sort by demoting it to the requires_sorted_edges contract flag; within this
      spec's lifetime the sort is unconditional.)
    status: verified
    last_checked: 2026-08-30
  - id: ac-003
    summary: GraphBridge synthetic ctest cases pass without LAMMPS or CUDA
    type: runtime
    pass_when: |
      ctest runs bridge_graph_edge_convention, bridge_graph_cutoff_refilter,
      bridge_graph_sorted_by_source_target, bridge_graph_drops_beyond_n_graph,
      bridge_graph_applies_neigh_mask, bridge_graph_empty_neighbor_list and
      bridge_graph_reuses_buffers_across_calls green from a build configured without
      LAMMPS_HEADER_DIR, none of them declaring FIXTURES_REQUIRED.
    status: verified
    last_checked: 2026-08-30
  - id: ac-004
    summary: empty neighbor list yields a well-formed empty edge tensor
    type: code
    pass_when: |
      bridge_graph_empty_neighbor_list asserts build() returns a (0, 2) int64 tensor
      when no pair survives, and GraphBridge never passes a possibly-null empty
      vector data pointer to at::from_blob.
    status: verified
    last_checked: 2026-08-30
  - id: ac-005
    summary: reverse-comm relocation is verbatim, with no private buffer added
    type: code
    pass_when: |
      pack_reverse_forces and unpack_reverse_forces read and accumulate directly on the
      view's f, introduce no owned buffer, and their header comment records the
      shared-atom->f defect as belonging to lammps-iface-05-pbc; PairMolnex's two
      callbacks are one-line forwards.
    status: verified
    last_checked: 2026-08-30
  - id: ac-006
    summary: reverse-comm ctest locks layout and accumulation
    type: runtime
    pass_when: |
      bridge_comm_pack_reverse_layout asserts 3n doubles written in fx,fy,fz order with
      return value 3n, and bridge_comm_unpack_accumulates starts from a non-zero f and
      asserts f[list[i]] += buf against hand-computed literals.
    status: verified
    last_checked: 2026-08-30
  - id: ac-007
    summary: Exec interface is backend-neutral and owns its returned memory
    type: code
    pass_when: |
      exec.h declares run(Z, pos_f64, edge_index) -> ModelOutputs with no AOTI, .pt2 or
      package-loader type in the signature, ModelOutputs is an attribute-only struct
      using virial.defined() for presence, and its docstring records both that the
      TorchScript runtime-backward route from lammps-pair-molnex-01 attaches as a third
      implementation without changing the interface, and that returned tensors are owned
      CPU copies which must not alias runtime-owned or replay-owned memory because
      run_graphed outputs live at fixed replay addresses overwritten by the next replay.
    status: verified
    last_checked: 2026-08-30
  - id: ac-008
    summary: capture state is not duplicated outside ModelRunner
    type: code
    pass_when: |
      GraphedExec holds only the persistent input buffers, n_fixed and e_max and
      delegates to ModelRunner::run_graphed; no CUDAGraph object, graph_captured_ flag
      or captured-output vector exists under interface/lammps/bridge/.
    status: verified
    last_checked: 2026-08-30
  - id: ac-009
    summary: pad_edges is CPU-testable and matches the padding semantics
    type: runtime
    pass_when: |
      bridge_pad_edges passes on a CPU-only build (pad_edges is outside the
      MOLNEX_INTERFACE_CUDA guard), asserting the first ne rows carry the real edges,
      the remainder are (0,0), mask is true exactly on the first ne entries, and the
      ne == e_max and ne == 0 boundaries hold.
    status: verified
    last_checked: 2026-08-30
  - id: ac-010
    summary: DynamicExec reproduces the fixture reference outputs
    type: runtime
    pass_when: |
      generate_fixtures.py emits a three-input (Z, pos, edge_index) -> (energy, forces)
      graph-potential fixture via the bare Exporter, and
      bridge_dynamic_exec_matches_reference asserts DynamicExec's outputs match the
      stored reference within 1e-12, returning CPU float64 with contiguous forces and
      an undefined virial when supports_virial is false.
    status: verified
    last_checked: 2026-08-30
  - id: ac-011
    summary: PairMolnex no longer owns graph, comm, execution or dtype state
    type: code
    pass_when: |
      pair_molnex.h declares none of edge_pairs_, edge_flat_, edge_pad_flat_, Zg_,
      posg_, edgeg_, maskg_, use_cuda_graph, n_fixed, e_max or fp64_model; the logmesg
      dtype is read from the Exec's read-only dtype name; compute() contains no neighbor
      loop, no sort, no at::from_blob and no #ifdef MOLNEX_INTERFACE_CUDA branch;
      ev_init, the eng_vdwl and virial tally and the comm->reverse_comm call site are
      unchanged; a single guard at the top of compute() establishes that SpeciesMap,
      ForceScatter, GraphBridge and Exec are all present, with GraphBridge held as
      std::optional and Exec as std::unique_ptr.
    status: verified
    last_checked: 2026-08-30
  - id: ac-012
    summary: bridge stays schema-blind and CPU-buildable after linking molnex_interface
    type: runtime
    pass_when: |
      molnex_lammps_bridge links molnex_interface PUBLIC, molnex_interface is defined
      before add_subdirectory(lammps/bridge) in interface/CMakeLists.txt, and no CMakeLists
      adds interface/lammps as a subdirectory; no file under interface/lammps/bridge/
      includes meta_schema.h, lammps_contract.h or any LAMMPS header; configuring
      interface/ without LAMMPS_HEADER_DIR and without LAMMPS_LIB still builds the bridge
      and runs its ctest cases green; GraphedExec's class is declared unconditionally with
      its run body #ifdef-guarded in the .cpp, and on a CPU-only build constructing it
      throws a CPU-only-build diagnostic.
    status: verified
    last_checked: 2026-08-30
  - id: ac-013
    summary: graph, comm, pad and dynamic-exec cases gate in CI
    type: runtime
    pass_when: |
      the interface job in .github/workflows/ci.yml configures interface/ without
      LAMMPS_HEADER_DIR and without LAMMPS_LIB and runs an explicit ctest filter covering
      the no-fixture list plus bridge_dynamic_exec_matches_reference, with fixture_setup
      pulled in automatically by FIXTURES_REQUIRED; the filter excludes
      cuda_run_matches_reference and bridge_graphed_exec_matches_dynamic rather than
      relying on a runtime skip; no regressions/lammps-iface-03-graph.py exists, and no
      file under regressions/ starts a subprocess.
    status: verified
    last_checked: 2026-08-30
  - id: ac-014
    summary: GraphBridge validates n_graph before indexing LAMMPS-owned memory
    type: code
    pass_when: |
      GraphBridge::build throws when n_graph < 0 or n_graph > view.nall(), and ctest case
      bridge_graph_rejects_n_graph_over_nall asserts the throw rather than an
      out-of-bounds read of view.x.
    status: verified
    last_checked: 2026-08-30
  - id: ac-015
    summary: the ghost-force double-communication defect is named and routed
    type: docs
    pass_when: |
      Out of scope names both pair_molnex.cpp:351-352 (the pair-level
      comm->reverse_comm(this) call) and :359-382 (packing the shared atom->f), states
      the newton-on mechanism by which Verlet::run's global reverse comm delivers the
      same ghost contribution a second time, and routes the pair to lammps-iface-05-pbc
      alongside the energy over-count; this spec relocates both verbatim and alters
      neither.
    status: pending
  - id: ac-016
    summary: virial well-formedness is an Exec-side invariant, not a facade check
    type: runtime
    pass_when: |
      read_model_outputs defines ModelOutputs::virial only when supports_virial is true,
      the raw output carries the element, and its numel() == 6, throwing with the observed
      numel otherwise; both DynamicExec and GraphedExec route their output conversion
      through it rather than duplicating it; bridge_read_model_outputs_rejects_short_virial,
      _absent_virial_undefined and _ignores_virial_when_unsupported pass without fixtures
      or CUDA.
    status: verified
    last_checked: 2026-08-30
---

# Acceptance criteria

- **ac-002 / ac-003** 守住本 spec 最容易被"顺手优化"掉的两件事：cutoff 重过滤与 (source,target) 排序。LAMMPS 的邻居表带 skin，去掉重过滤会静默改能量。排序在本 spec 生命周期内无条件保留；`05-pbc` 将其**明示取代**为 `requires_sorted_edges` 契约标志（ac-002 的 forward note），这是链内先后接力而非互相矛盾。
- **ac-004** 是起草时发现的空指针边缘（`ne == 0` 时把可能为空的 `edge_flat_.data()` 交给 `at::from_blob`）。它落在本 spec 正在改写的那一行上，属本地可修范围，故按 iron law 在此修掉而非留给后续。
- **ac-005 / ac-015** 是"只搬不修"的守卫。组件按当下真实所为命名（`comm_reverse` 而非 `CommBuf`），ghost 力双重通信的机制、行号与路由以 F3 同等标准具名。
- **ac-007 / ac-008** 是执行轴的两条硬约束：接口不得烘焙 AOTI 假设、返回张量必须是自有 CPU 拷贝（run_graphed 的重放地址会被下一次 replay 覆写——现行正确纯属 `.to()` 拷贝的巧合，组件化后升为契约），capture 状态不得离开 `ModelRunner`。
- **ac-009 / ac-010** 解决"执行策略如何在没有 GPU、没有定形导出产物的机器上被单测"：padding 与输出转换抽为 CPU 可测的自由函数，dynamic 路径靠新增的三输入 fixture。
- **ac-011** 是结构迁移本身的判据，措辞刻意把 tally 与 `reverse_comm` 调用点划在界外，留给 `04-facade`；`fp64_model` 随执行策略迁出（与 02 的 `force_conv` 同规则：最后一个读者离开，值就离开）。
- **ac-012** 明确记录 bridge 依赖面的扩张（新增 `molnex_interface`），把"仍然 LAMMPS-free"重新验证一遍，并封死 bridge 对 meta schema 头的包含。
- **ac-013** 把 `bridge_dynamic_exec_matches_reference`（Exec 抽取的唯一数值 parity 证据）纳入 CI 显式过滤。
- **ac-016** 把 virial 良构性从 facade 检查升为执行层构造不变量——它同时是 `05-pbc` 输出下标位移（`out[2]`→`out[3]`）的保险。
