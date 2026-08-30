---
slug: lammps-iface-05-pbc
retired_ids: []   # ids are append-only within the chain; a dropped criterion is
                  # recorded here with its reason, never reused. ac-017 is the
                  # current maximum.
criteria:
  - id: ac-001
    summary: partition energy sums site energies over local atoms only
    type: scientific
    pass_when: |
      impl_energy_sums_locals_only builds a LammpsView with explicit ghost images and a
      stub Exec returning a constant per-site energy, and asserts EvalResult::e_total
      equals nlocal * constant * energy_conv against a hard-coded literal, not nall *
      constant; CutoffGraphImpl computes it as a sum over i < view.nlocal.
    status: pending
  - id: ac-002
    summary: per-atom energy is produced and tallied for locals
    type: code
    pass_when: |
      EvalResult carries e_atom, CutoffGraphImpl fills it with energy_conv-scaled
      per-atom energies, the facade accumulates eatom over the returned tensor's own
      extent without referencing atom->nlocal when eflag_atom is set, and the
      eflag_atom error->all path from lammps-iface-04-facade is removed while the
      vflag_atom one remains; e_atom is documented as length nlocal; ModelOutputs
      carries atomic_energy, read_model_outputs validates atomic_energy.size(0)
      against the expected atom count inside the single decoder (the Execs do only
      device/dtype movement), and virial is read from output index 3 under the v3
      output order [energy, forces, atomic_energy, (virial)] with indices resolved
      by the meta outputs name list backed by named constants.
    status: verified
    last_checked: 2026-08-30
  - id: ac-003
    summary: reduction ownership moves to the integrator through the seam, not around it
    type: code
    pass_when: |
      EvalResult still declares needs_pair_reverse_comm and pair_molnex.cpp still contains
      the single if (r.needs_pair_reverse_comm) comm->reverse_comm(this) call site
      unchanged from lammps-iface-04-facade; CutoffGraphImpl sets it to a constant false
      whose comment names CutoffGraphImpl::init's newton-pair enforcement as its
      precondition, and whose docstring records that the value is permanently false for
      the cutoff_graph contract and that a future contract would set it true only if it
      scatters ghost forces outside newton-reduced integration; compute() throws when
      supports_pbc && !newton_pair_ using the value snapshotted at init, with the comment
      stating that this catches internal ordering faults and not LAMMPS-side changes;
      CutoffGraphImpl::init throws when supports_pbc is true and newton_pair is off,
      naming newton on. impl_reports_no_pair_reverse_comm asserts the field is false in
      both the with-ghost and without-ghost cases, and impl_compute_reasserts_newton_invariant
      asserts the compute-time throw.
    status: verified
    last_checked: 2026-08-30
  - id: ac-004
    summary: the retained reverse-comm seam is documented as currently unexercised
    type: docs
    pass_when: |
      ContractImpl::pack_reverse and unpack_reverse remain per decision 2, and their
      docstrings state that on the cutoff-graph contract they are not invoked because the
      integrator's global reverse comm performs the reduction, while a fixed-graph or
      layer-wise-communication impl may re-enable them; the docstrings also state the
      single re-entry route: a contract that fills needs_pair_reverse_comm with true
      re-activates the facade's existing call site, and there is no second entrance.
    status: pending
  - id: ac-005
    summary: the ghost closure makes the builder aggregation-direction agnostic
    type: scientific
    pass_when: |
      GraphBridge::build iterates ii < view.inum + view.gnum with n_graph == view.nall();
      bridge_graph_ghost_closure asserts every local atom's complete incoming edge set is
      present, and that a target-aggregating stub (MACE direction) and a
      source-aggregating stub (PiNet direction) both reproduce the full-enumeration
      reference over that closure, with no per-model branch anywhere in the builder.
    status: pending
  - id: ac-006
    summary: the fat-ghost receptive-field condition is validated and fails closed
    type: code
    pass_when: |
      meta carries a required positive integer receptive_field_layers for cutoff-graph
      contracts, export_for_lammps raises ValueError when it is omitted (no default)
      naming both the keyword and the re-export command, LammpsRunEnv carries
      ghost_cutoff and skin snapshotted raw by the facade with no defaulting, the
      zero-means-unset rule (effective cutoff + skin) is applied only inside
      CutoffGraphImpl::init, and init throws when ghost_cutoff < T * cutoff + skin with
      a message naming comm_modify cutoff and the required value; no code path
      downgrades this to a warning. The check applies to supports_pbc contracts; the non-periodic path's receptive-field completeness is enforced by its zero-ghost guard instead, so a T>=2 single-molecule run is not spuriously rejected.
    status: verified
    last_checked: 2026-08-30
  - id: ac-007
    summary: strictly-local models are not rejected by the fat-ghost check
    type: runtime
    pass_when: |
      impl_init_accepts_strictly_local_default_cutoff passes with T == 1 and the default
      ghost cutoff of cutoff + skin, proving the D3.2 check does not break Allegro-class
      models or the existing non-periodic path.
    status: verified
    last_checked: 2026-08-30
  - id: ac-008
    summary: the sort is contract-driven and no unconditional order promise survives
    type: code
    pass_when: |
      GraphBridge sorts only when the contract's requires_sorted_edges is true; a content
      grep for the unconditional sorted-order phrasing ("sorts edges by", "sorted by
      (source, target)", "sorted by (src, tgt)") returns zero matches across
      src/molix/engine/, interface/lammps/ and interface/tests/, the anchors at
      src/molix/engine/adapter.py:23 and :77 now stating a conditional promise; the
      replacement comment records that order-independence holds for scatter reductions
      only and that segmented or CSR aggregation re-imposes sorted indices, which is what
      requires_sorted_edges exists to declare; src/molrep/interaction/aggregation.py
      carries the guard comment that adding CSR/segmented aggregation requires exports
      to declare requires_sorted_edges=True.
    status: verified
    last_checked: 2026-08-30
  - id: ac-009
    summary: aggregation is order-invariant, and sorting happens iff declared
    type: scientific
    pass_when: |
      bridge_graph_order_shuffle_invariance shuffles firstneigh order with
      requires_sorted_edges false and asserts the scatter-reduced energy is unchanged
      within 1e-12 in fp64, consistent with the (n-1)u reassociation bound;
      bridge_graph_sorts_only_when_contract_requires asserts the true case yields
      lexicographically non-decreasing rows and the false case preserves raw neighbor
      order, with an identical edge set either way.
    status: pending
  - id: ac-010
    summary: the export contract reaches schema v3 with local_mask and per-atom energy
    type: code
    pass_when: |
      LAMMPS_META_SCHEMA == 3 and lammps_contract.h expects 3; outputs are
      ["energy","forces","atomic_energy"] plus "virial" when supports_virial; inputs are
      4 entries for dynamic and 5 for cuda-graph with local_mask last; EngineForward and
      StaticForward compute energy as (atomic_energy * local_mask).sum() rather than by
      slicing on a runtime scalar; meta also carries requires_sorted_edges defaulting to
      false, the cuda-graph edge-validity mask is named edge_mask so that inputs read
      ["Z","pos","edge_index","edge_mask","local_mask"], and both the positional order
      and the dtype of every input are pinned in interface/lammps/README.md's input table.
    status: verified
    last_checked: 2026-08-30
  - id: ac-011
    summary: local_mask semantics are exact at both extremes
    type: runtime
    pass_when: |
      TestLocalMask asserts an all-ones mask reproduces the sum over all sites, a partial
      mask reproduces the masked subset against hard-coded literals, forces on masked-out
      sites are non-zero because they are cross-terms of local site energies, and an
      all-zero mask yields zero energy and zero forces.
    status: verified
    last_checked: 2026-08-30
  - id: ac-012
    summary: per-atom energy has exactly one flat spelling and is never silently dropped
    type: code
    pass_when: |
      src/molix/schema.py declares ATOMIC_ENERGY_FLAT_KEY beside ATOMIC_ENERGY_KEY and
      documents the nested-to-flat correspondence; protocol.py:166 and write_energy's
      keyword both reference it; TestAbsorbModelOutput shows a plain dict carrying
      per-atom energy is written through and that a non-canonical spelling no longer
      passes silently; the export outputs use the same single spelling and no third
      spelling such as node_energies exists anywhere in the tree.
    status: verified
    last_checked: 2026-08-30
  - id: ac-013
    summary: the edge-distance clamp is one shared constant
    type: code
    pass_when: |
      the 1e-6 and 1e-8 clamps at src/molix/engine/adapter.py:111 and
      src/molix/engine/static.py:62 are replaced by one named constant in
      src/molix/schema.py referenced from both, and TestEdgeDistanceClamp asserts both
      paths return identical edge_dist for a degenerate input; the spec records that
      standardizing on 1e-6 changes static-path behavior for distances in [1e-8, 1e-6),
      a physically degenerate and empty-in-practice band, so this is a sub-threshold
      behavior change rather than a pure refactor.
    status: verified
    last_checked: 2026-08-30
  - id: ac-014
    summary: periodic multi-rank parity holds against direct Python evaluation
    type: runtime
    pass_when: |
      scripts/lammps_iface_parity.py implements the periodic case as a LIVE
      three-way comparison (1 rank vs >=2 ranks vs direct in-process Python
      evaluation of the same configuration, rel 1e-6), refuses --capture for it,
      exits non-zero naming the missing prerequisite (LAMMPS binary, plugin, MPI
      launcher) and never skips or fabricates; the Python reference leg is
      validated in-tree (folded forces match a central-difference gradient of the
      masked local energy at rel <= 1.3e-7 against the v3 fixture); the aspirin
      case stays stdlib-only while the periodic verification may import torch;
      interface/lammps/README.md carries the runbook entry naming the owner and
      the re-run trigger ("re-run on any change under interface/ or
      src/molix/engine/").
    status: verified
    last_checked: 2026-08-30
  # CHAIN-LEVEL DEBT (not verifiable in this environment): the actual periodic
  # single-rank vs multi-rank vs Python three-way run on a real LAMMPS build —
  # the ONLY end-to-end evidence for the D2 double-communication removal — is
  # owed by the chain implementer before the branch merges, together with 04's
  # captured aspirin parity run. Attesting either as verified without attached
  # runs is forbidden.
  - id: ac-015
    summary: the schema-v3 fixture tie stays bidirectional
    type: runtime
    pass_when: |
      generate_fixtures.py emits a schema-v3 meta through _lammps_meta_block and
      contract_reads_generated_v3_fixture parses it green; bumping either
      LAMMPS_META_SCHEMA or the C++ expected version alone turns that case red.
    status: verified
    last_checked: 2026-08-30
  - id: ac-016
    summary: tag-to-local mapping is absent and rerouted
    type: docs
    pass_when: |
      no tag member, tag-keyed container or atom->tag access exists under
      interface/lammps/bridge/ or interface/lammps/impls/; Out of scope states that the
      fat-ghost single-shot scheme does not need it and that it belongs to the future
      fixed-graph contract spec, not to this one.
    status: pending
  - id: ac-017
    summary: the v2-to-v3 break is documented with a migration path
    type: docs
    pass_when: |
      interface/lammps/README.md states that every export produced before schema v3 is
      invalid and must be re-exported, gives the re-export command, and lists the v3
      field additions (atomic_energy output, local_mask input, edge_mask rename,
      receptive_field_layers, requires_sorted_edges); the four export_for_lammps call
      sites in tests/test_molix/test_engine/test_export.py and the prose examples at
      src/molix/engine/__init__.py:15 and interface/lammps/README.md:47,65,79,147 are all
      updated; no scripts/ call site exists to update. The six-atom transpose lock
      bridge_read_model_outputs_six_atoms_no_transpose passes (both size guards are
      simultaneously blind at N == 6, so this value-level case is the only defense
      against a virial/atomic_energy swap at that size).
    status: pending
---

# Acceptance criteria

- **ac-001 / ac-005 / ac-009** 是三条物理判据，分别锁 D1、D4、D6。ac-005 的两个方向桩聚合是本 spec 最有价值的一条测试：它把"对称闭包使 builder 与聚合方向无关"从论证变成可执行的证据，也是 `CLAUDE.md` "no per-model C++" 在 PBC 下继续成立的凭据。
- **ac-002 / ac-004** 处理 `04-facade` 留下的两个接口承诺：扩展点被兑现（`e_atom` 到场，输出下标按名单解析），保留的 seam 被诚实标注（pack/unpack 仍在、当前不被触发、唯一重启路径写明）。
- **ac-003** 与 ac-014 配对：前者是结构判据（归约所有权经缝移交、newton 前置 + 廉价重申），后者是唯一能真正证明双重通信已消失的端到端证据——单 rank 跑不出这个差异。
- **ac-006 / ac-007** 成对存在：一条锁 fail-closed，一条锁不误伤。只有前者会诱使实现把阈值调松以求测试通过。
- **ac-008 / ac-009** 把排序从"普适税"变为契约声明：默认路径零开销，分段聚合模型无法被喂进无序边，ac-014 的真模型 parity 兼作满量程顺序不变性证据。
- **ac-012 / ac-013** 是本表面上发现的两处工程失真。ac-012 在 05 之前只是功能缺失，在 05 之后会变成静默的零能量——这正是它必须与 D1 同批落地的原因。
- **ac-014** 沿用 `04-facade` ac-013 的形制并加强：runbook 条目使多 rank 证据有归属与失效期；该判据 pending 期间，**链的收尾提交被阻塞**。
- **ac-017** 的六原子锁值得单独一句：`numel==6` 与 `size(0)==n_atoms` 两道守卫在 N==6 时同时失明，而六原子恰是手写 fixture 最常见的尺寸——值级断言是唯一防线。实现期间对 virial 下标那一行要有 reviewer 的专门目光，不只靠测试。
