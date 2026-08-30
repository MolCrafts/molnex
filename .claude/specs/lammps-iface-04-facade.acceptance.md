---
slug: lammps-iface-04-facade
retired_ids:
  - id: ac-011
    reason: retired — cut together with the num_models pair_style keyword per the C3
      ruling; the grammar ships with the spec that introduces a hot-reload consumer.
      Never reused; the reverse-comm criterion landed as ac-016.
criteria:
  - id: ac-001
    summary: facade contains no semantic branch
    type: code
    pass_when: |
      interface/lammps/pair_molnex.cpp contains no reference to supports_pbc,
      use_cuda_graph, n_graph, cutsq, type2z, energy_conv or force_conv, no
      #ifdef MOLNEX_INTERFACE_CUDA, no neighbor loop and no at:: tensor construction
      beyond assembling LammpsView; pair_molnex.h declares only model_dir, model_name,
      the contract, the runner, the impl, the cached NeighborNeeds and the LAMMPS
      protocol flags.
    status: pending
  - id: ac-002
    summary: NeighborNeeds is executed mechanically by init_style
    type: code
    pass_when: |
      NeighborNeeds carries full, ghost, cutoff, comm_reverse and comm_reverse_off and
      no predicate field; init_style assembles NeighConst flags from full and ghost with
      no other condition, init_one returns needs.cutoff, and settings applies
      comm_reverse and comm_reverse_off from needs; impl_needs_matches_legacy_request
      asserts the values equal the pre-refactor ones.
    status: pending
  - id: ac-003
    summary: EvalResult carries no forces, no unproduced field, and one ownership seam
    type: code
    pass_when: |
      As of this spec, EvalResult declares exactly e_total, an optional virial and
      needs_pair_reverse_comm; its docstring records that the impl scatters forces into
      atom->f before returning, that needs_pair_reverse_comm is the reduction-ownership
      seam stating per step whether the pair style itself must reverse-communicate (not
      whether ghost forces were scattered), that it is independent of
      NeighborNeeds::ghost, and that per-atom energy has no producer until
      lammps-iface-05-pbc adds e_atom as the designed extension. EvalResult::virial is
      well-formed-or-undefined by construction because lammps-iface-03-graph's
      read_model_outputs validates numel == 6 before defining it, so the facade never
      receives a defined virial with fewer than six readable elements and performs no
      length check of its own. The docstring also carries the accretion rule that every
      field is a per-step output of one evaluation arriving with its producer and
      consumer, and that a future "the facade must also do X" obligation needs a
      different home, with needs_pair_reverse_comm explicitly grandfathered and labelled
      as that seam. No member is left unset by CutoffGraphImpl on any path.
    status: pending
  - id: ac-004
    summary: per-atom energy and virial requests error instead of returning zeros
    type: code
    pass_when: |
      compute() calls error->all when eflag_atom or vflag_atom is set, naming the
      contract and the offending per-atom compute; no code path leaves eatom or vatom
      unwritten while reporting success.
    status: pending
  - id: ac-005
    summary: the impl pipeline composes correctly under an injected stub Exec
    type: runtime
    pass_when: |
      ctest runs impl_compute_composes_pipeline, impl_compute_n_graph_local_only,
      impl_compute_n_graph_with_ghost, impl_compute_rejects_ghost_without_pbc and
      impl_virial_present_and_absent green from a build configured without
      LAMMPS_HEADER_DIR, without CUDA and without any .pt2, using an in-test Exec
      implementation constructed directly into CutoffGraphImpl's public constructor and
      never through ContractImpl::for_contract; e_total matches hand-computed literals
      including the energy_conv factor and atom->f is asserted from a non-zero initial
      value.
    status: pending
  - id: ac-006
    summary: Exec is injected, proving the interface is backend-neutral
    type: code
    pass_when: |
      CutoffGraphImpl takes std::unique_ptr<Exec> at construction and never selects an
      execution strategy internally; the Execution-to-Exec mapping lives in
      interface/lammps/impls/registry.cpp; the ctest stub Exec is not derived from any
      AOTI type and requires no model package.
    status: pending
  - id: ac-007
    summary: pair_coeff residual tokens are interpreted strictly by the impl
    type: code
    pass_when: |
      PairMolnex::coeff validates only that the first two tokens are literal * and
      forwards arg[2..] as std::vector<std::string> with atom->ntypes; token-count and
      integer parsing live in CutoffGraphImpl::coeff, which uses no LAMMPS symbol such as
      utils::inumeric and parses with std::from_chars requiring both a clean errc and an
      end pointer at the token's end; impl_coeff_token_count,
      impl_coeff_rejects_trailing_garbage, impl_coeff_rejects_empty_token and
      impl_coeff_rejects_out_of_range all pass, each asserting the original token appears
      in the diagnostic.
    status: pending
  - id: ac-008
    summary: special_bonds exclusion of bonded neighbors is intercepted
    type: code
    pass_when: |
      CutoffGraphImpl::init throws when env.molecular != 0 and any of special_lj[1..3]
      or special_coul[1..3] differs from 1.0, with a message naming
      "special_bonds lj/coul 1.0 1.0 1.0" and the reason; it does not throw when
      env.molecular == 0; impl_init_rejects_special_bonds_on_molecular covers both
      halves, and is documented as the only coverage of this fix because the aspirin
      parity case is molecular == 0.
    status: pending
  - id: ac-009
    summary: the single-domain check is carried over verbatim, not tightened
    type: code
    pass_when: |
      CutoffGraphImpl::init throws only when !supports_pbc && newton_pair == 1 &&
      nprocs > 1, reusing the pre-refactor message; impl_init_single_domain_condition
      asserts that newton_pair == 0 with nprocs > 1 does not throw.
    status: pending
  - id: ac-010
    summary: dispatch is an exhaustive switch over the closed contract set
    type: runtime
    pass_when: |
      ContractImpl::for_contract dispatches through a switch over the Contract enum with
      an explicit case per enumerator and no default label, so -Wswitch flags a future
      enumerator at compile time; the Execution-to-Exec mapping has the same shape; no
      function-local static map and no self-registration is introduced;
      registry_returns_cutoff_graph_impl and registry_rejects_fixed_graph pass with the
      miss message listing every valid contract name; each impl keeps a public direct
      constructor and no symbol named make_* or create_* is introduced.
    status: pending
  - id: ac-012
    summary: impls build and test without LAMMPS and gate in CI
    type: runtime
    pass_when: |
      molnex_lammps_impls is declared in interface/lammps/impls/CMakeLists.txt, added
      from interface/CMakeLists.txt and linked from interface/lammps/CMakeLists.txt; no
      CMakeLists adds interface/lammps as a subdirectory; no file under
      interface/lammps/impls/ includes a LAMMPS header; the interface job in
      .github/workflows/ci.yml runs every new impl_* and registry_* case green without
      LAMMPS_HEADER_DIR or LAMMPS_LIB.
    status: pending
  - id: ac-013
    summary: aspirin parity holds against embedded pre-refactor literals
    type: runtime
    pass_when: |
      scripts/lammps_iface_parity.py reproduces the embedded decimal-literal aspirin
      energy and forces at relative tolerance 1e-6, carries a Provenance block naming
      LAMMPS version, atom_style, plugin commit, torch version, device/precision, capture
      command and date, exits non-zero with a diagnostic when no LAMMPS build is present
      and never skips; the tool header and this spec both record that aspirin is
      molecular == 0 and therefore exercises none of the special_bonds path. A real run's
      output is attached at delivery; while this criterion is pending the chain-closing
      commit is blocked, not merely this spec's, and the named owner is the chain
      implementer.
    status: pending
  - id: ac-014
    summary: the pbc energy over-count stays named and unfixed
    type: docs
    pass_when: |
      Out of scope names the local+ghost energy over-count formerly at
      pair_molnex.cpp:332-333, states that this spec relocates it into EvalResult::e_total
      with unchanged semantics, and routes it with the shared-atom->f reverse-comm defect
      to lammps-iface-05-pbc; neither is altered here.
    status: pending
  - id: ac-015
    summary: the virial tally keeps all three arms, including the do-nothing one
    type: code
    pass_when: |
      the tally is written as if (r.virial.defined() && vflag_global) { accumulate }
      else if (vflag_fdotr) { virial_fdotr_compute(); } with no reordering of the
      conditions, so that vflag_global set with an undefined virial and vflag_fdotr unset
      still performs no virial work, exactly as pair_molnex.cpp:335-349 does today.
    status: pending
  - id: ac-016
    summary: the reverse-comm decision is produced by the impl and executed mechanically
    type: code
    pass_when: |
      As of this spec, CutoffGraphImpl::compute sets EvalResult::needs_pair_reverse_comm
      from supports_pbc && view.nghost > 0, the same expression over the same inputs as
      the current pair_molnex.cpp:351 condition; compute() in the facade reaches
      comm->reverse_comm(this) through that field alone and reads no capability flag;
      impl_reports_pair_reverse_comm_condition covers all four combinations of
      supports_pbc and non-zero nghost. lammps-iface-05-pbc keeps both the field and this
      call site and changes only the impl's answer, so this criterion's structural claim
      survives the chain unchanged.
    status: pending
---

# Acceptance criteria

- **ac-001 / ac-002** 共同定义"facade 削到协议机械"这句话的可检验含义：删掉的成员清单是硬判据，`NeighborNeeds` 被机械执行是软判据的硬化。
- **ac-003 / ac-004** 落实裁决③。`EvalResult` 不含力这一不对称必须写在 docstring 里，否则会被后来者当作遗漏"补全"，而补全它会立刻与 `reverse_comm` 读 `atom->f` 冲突；`needs_pair_reverse_comm` 作为归约所有权缝被明确豁免于累积规则。
- **ac-005 / ac-006** 是本 spec 最重要的方法学结果：注入 `Exec` 让 Layer-2+3 全栈在无 LAMMPS、无 CUDA、无 `.pt2` 的条件下可测，因而**进得了 CI**；直接构造（不经 `for_contract`）同时守住"直接构造是首要构造故事"。
- **ac-008** 是起草本 spec 时发现的活缺陷。它不是理论风险：LAMMPS 的 `special_bonds` 默认值就是 `0 0 0`，任何带键拓扑的体系都会让 `GraphBridge` 拿到一张缺了 1-2 / 1-3 / 1-4 近邻的图。判据刻意要求覆盖不抛出的那一半，并具名记录 aspirin parity 完全不覆盖此路径。
- **ac-009 / ac-015** 防止"顺手修正"：单域校验的 newton 条件与 virial tally 的"什么都不做"分支都是看着像漏洞、实为现行为的东西；收紧或重排属行为变更，归 `05-pbc` 或不属于本链。
- **ac-010** 把闭集的封闭性交给编译器（`-Wswitch`），并封死 `make_*` 形状。
- **ac-013** 是唯一无法进 CI 的判据（CI 无 LAMMPS），其 `pass_when` 显式要求交付时附真实运行记录，且 pending 期间**链的收尾提交被阻塞**——这是把"本地债"写成可核验条件而非口头承诺。
- **ac-016** 落在新号（旧 ac-011 已 tombstone，编号链内 append-only）：归约判断由 impl 生产、facade 机械执行；`05-pbc` 只换 impl 的答案，结构性主张全链存活。
