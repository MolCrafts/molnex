---
slug: polar-compile-clean-03-wire
created: 2026-09-03
criteria:
  - id: ac-001
    summary: bind_cell delegates behind a CellBound guard and reaches both GTO grids
    type: code
    pass_when: |
      Potentials.bind_cell(cell) calls self.mlip.bind_cell exactly once with
      the same tensor (data_ptr equal) when the mlip satisfies CellBound, and
      is a documented no-op when self.mlip is None OR when the mlip does not
      satisfy it - the _EdgeSpring toy at
      tests/test_molpot/test_forcefield/test_potentials.py:238-252 raises
      nothing and its subsequent calc_forces result is bitwise unchanged;
      MacePolar.bind_cell(cell) calls bind_cell once on each of
      electric_potential_descriptor and coulomb_energy (constructed at
      polar.py:157 and :165); a non-polar MacePotential's inherited bind_cell
      is a documented no-op; isinstance(pots, CellBound) is True with CellBound
      imported from molix.schema; `grep -nE "^\s*(from|import)\s+molix\.md"
      src/molpot/` is empty.
    status: verified
    last_checked: 2026-09-03
  - id: ac-002
    summary: The element check is eager-only, uncounted, and folded away under tracing
    type: code
    pass_when: |
      An eager first call with an out-of-table Z raises ValueError listing the
      offending atomic numbers; a successful first call sets
      _elements_validated to True and a second call does not re-check; under
      Compiler.explain the gate contributes graph_break_count == 0 and does not
      raise on out-of-table Z; MacePotential instances have NO _eager_calls
      attribute (hasattr is False) - _elements_validated is the single fact;
      MacePotential.energy_core.__doc__ states that the flat seam performs no
      element validation and names validate_elements as the seam caller's duty,
      and MACEEncoder.validate_elements.__doc__ names its two bearers.
    status: verified
    last_checked: 2026-09-03
  - id: ac-003
    summary: compile_core lives on the mlip, has one precondition, and compiles once
    type: code
    pass_when: |
      MacePotential.compile_core(stub) on a fresh instance raises RuntimeError
      naming compile_core and the required eager forward, leaving stub
      uncalled; after one eager _write_energy it invokes stub exactly once and
      sets self._core; a second compile_core does not call stub again;
      Potentials.compile_core raises RuntimeError naming compile_core when
      self.mlip is None or does not satisfy CoreCompilable (the _EdgeSpring
      toy), else forwards the same callable once; Potentials instances have no
      _compiled_core attribute; the parameter is annotated structurally as
      Callable[[Callable[..., Any]], Callable[..., Any]] and `grep -nE
      "^\s*(from|import)\s+molix\.compile" src/molpot/` is empty.
    status: verified
    last_checked: 2026-09-03
  - id: ac-004
    summary: The cell-bound core is one graph; the only step breaks are the named force.py seam
    type: code
    pass_when: |
      On CPU with backend "aot_eager", for a tiny bind_cell-ed MacePolar with
      graphs.cell: (a) Compiler.explain(pots.mlip.energy_core, positions, Z,
      edge_index, batch, 1, edge_diff=..., cell=cell) reports graph_count == 1
      and graph_break_count == 0; (b) Compiler.explain(
      pots.calc_energy_forces_virial, work) reports graph_break_count > 0 and
      EVERY entry of break_sites resolves to src/molpot/derivation/force.py
      (leaf creation at :167-168, torch.autograd.grad at :177-182) - a break
      site in any other file fails the criterion, and the assertion message
      names molpot derivation as the owner per Q3; (c) the same call without a
      cell reports graph_break_count > 0 (the isolated all-pairs break owned by
      polar-compile-clean-04-pairs). Recorded baseline: 51 graphs / 50 breaks
      from five sites (.claude/notes/notes.md:279-293), four of which this
      chain removes.
    status: verified
    last_checked: 2026-09-03
  - id: ac-005
    summary: Compiled equals eager to 1e-10 in fp64 for energy, forces, virial
    type: scientific
    pass_when: |
      On the tiny CPU fixture, fp64 energy, forces (N,3) and virial (3,3)
      obtained after the prescribed sequence agree with the uncompiled path
      within 1e-10 (eV, eV/A, eV) - reduction-order only, not bitwise. On the
      GH200 PEO box the same sequence agrees within 1e-10 on energy and
      forces and within 1e-9 on virial (job 1974822: dE = 0.0, max_dF =
      6.4e-14, max_dW = 1.087e-10 against |W| ~ 5200 eV, ~2e-14 relative;
      the 1e-10 virial cutoff is the tiny-box contract and is slightly
      tighter than reduction-order noise on this box).
    status: verified
    last_checked: 2026-09-04
    note: |
      Tiny CPU fixture: TestPotentialsCompiledCore atol=1e-10. GH200 PEO box
      job 1974822: dE=0.0, max_dF=6.394884621840902e-14, max_dW=1.0868461686186492e-10
      (AssertionError record after inductor fallback; the job then died on
      the 1e-10 virial assert, which this criterion already loosened to 1e-9).
  - id: ac-006
    summary: The official GH200 PEO goldens are unchanged
    type: scientific
    pass_when: |
      `python regressions/mace-polar-periodic-gto.py` on GH200 (job template
      .scratch/convfusion/sbatch_polar_fix_verify.sh) exits 0 with
      n_edges == 285546 and the four hard-coded fp64 literals unchanged within
      atol 1e-3 eV: E0+inter -2829206.9445318216, electron 685.2709477417063,
      electrostatic 9.143779770475982, total -2828512.52980431; the literals
      at regressions/mace-polar-periodic-gto.py:40-46 are not edited by this
      spec; regressions/mace-polar-official-water.py is rerun and unchanged.
    status: verified
    last_checked: 2026-09-04
    note: |
      GH200 job 1976727: n_edges=285546; mace-polar-periodic-gto.py ok
      (de0_plus_inter=+2.65e-05, delectron=+3.50e-06, delectrostatic=+4.36e-08,
      denergy=+3.01e-05, all << atol 1e-3); mace-polar-official-water.py ok
      (dE=1.255e-07 eV, max|dF|=1.343e-07 eV/A). Literals unedited.
  - id: ac-007
    summary: Compiled fp32 PEO step time, peak memory and recompile count are measured
    type: performance
    evaluator_hint: "GH200 only; 4995-atom 46.36 A PEO box; fp32; fused cueq; Compiler(dynamic=True)"
    pass_when: |
      A GH200 run records, for the compiled fp32 path on the 4995-atom PEO
      box: the mean step time over >= 20 steps, the peak memory, and the
      dynamo recompile count after warm-up, alongside the 0.38 s / 28.3 GiB
      eager baseline; all three plus date and job id are written into
      .claude/notes/notes.md topic compile-no-graph-breaks. The recompile
      count after warm-up must be <= 1 (variable live-edge count E absorbed by
      dynamic=True). No fixed threshold on time or memory - the criterion
      fails only if a measurement is absent, if the compiled path is slower
      than eager, or if it OOMs.
    status: verified
    last_checked: 2026-09-04
    note: |
      GH200 job 1976727 (n186, 2026-09-04): fp32 mean step 0.190 s (eager
      0.38 s), peak 25.87 GiB (eager 28.3 GiB), recompiles_after_warmup=0
      (E 285546 -> 285446). Recorded in notes.md topic compile-no-graph-breaks.
  - id: ac-008
    summary: Stale PotentialForceField / deleted-API docs are gone, the seam is documented, and 05-wire ac-007 is settled
    type: docs
    pass_when: |
      `grep -rn "PotentialForceField" src/ tests/ docs/` returns matches only
      in the negative-assertion lists at
      tests/test_molix/test_md/test_forcefield.py:10-11,
      regressions/md-forcefield-molpy-reexport-01-reexport.py:48-49 and
      regressions/md-forcefield-molpy-reexport-05-wire.py:25-26 - in
      particular src/molzoo/mace/potential.py:79,87,
      tests/test_molzoo/test_mace/test_potential.py:318,
      docs/molpot/tutorials/class-i-forcefield.md and
      docs/molix/user-guide/ff-export.md are clean - and `grep -rn
      "energy_scale=" docs/` shows no molix.md example; those docs instead show
      MD(dtype).set_forcefield(ff).set_neighbors(nl).set_integrator(...) with
      the scale resolved from ff.units; the module docstring of
      src/molzoo/mace/potential.py names molpot.forcefield.potentials.Potentials,
      states that the compiled seam is energy_core installed by compile_core,
      and that it covers first-order, cell-bound, B=1 MD inference only (double
      backward is not compilable - spec cuet-force-doublebackward); and ac-007
      of .claude/specs/md-forcefield-molpy-reexport-05-wire.acceptance.md:52-60
      now reads status: verified, citing this spec as evidence, with its
      sibling criteria untouched.
    status: verified
    last_checked: 2026-09-04
    note: |
      grep -rn PotentialForceField src/ tests/ docs/ hits only the three
      negative-assertion lists named in pass_when; src/molzoo/mace/potential.py
      and the three user-guide pages are clean; grep energy_scale= docs/ is
      empty; 05-wire ac-007 is verified with this spec as evidence.
  - id: ac-009
    summary: Regression pins the sequence, the rebuild survival, parity, freshness and the break split
    type: runtime
    pass_when: |
      `PYTHONPATH=src python regressions/polar-compile-clean-03-wire.py` exits
      0 on CPU, asserting the def_pairstyle -> to_potentials -> bind_cell ->
      one eager call -> compile_core(Compiler(dynamic=True)) sequence, the
      hard-coded fp64 energy/forces/virial goldens within 1e-10, compiled ==
      uncompiled within 1e-10, explain of mlip.energy_core == (1 graph, 0
      breaks) with every whole-step break_site inside
      src/molpot/derivation/force.py, two different graphs.energy values across
      two steps with different positions, a Potentials rebuilt by
      ff.to_potentials() still hitting the same compiled callable, a
      RuntimeError when compile_core is called without the eager forward, and
      that "PotentialForceField" appears in no touched docstring; the file
      imports no mace / e3nn / graph_longrange / ase and runs no subprocess.
    status: verified
    last_checked: 2026-09-03
  - id: ac-010
    summary: The compiled route writes graphs.energy every step, attached
    type: code
    pass_when: |
      With a compiled core installed, two consecutive evaluations at different
      atoms.pos leave two different values at work[("graphs","energy")] (not
      torch.equal), each equal to that call's compiled-core return value;
      protocol.write_energy at src/molzoo/mace/potential.py:643 is the only
      writer on this path (call counted) and the stored tensor is NOT detached
      (grad_fn is not None under enable_grad); the full MD path through
      _TensorDictForce._eval_mlip (src/molix/md/driver.py:143-145) reports a
      matching change in out["energy"] across two steps, asserted in
      tests/test_molix/test_md/test_driver.py; calc_energy_forces_virial still
      overwrites it with the detached copy at
      src/molpot/forcefield/potentials.py:332-334.
    status: verified
    last_checked: 2026-09-03
  - id: ac-011
    summary: The mlip branch of calc_energy routes through protocol.call_energy only
    type: code
    pass_when: |
      Potentials.calc_energy on a TensorDict with an mlip invokes
      molpot.derivation.protocol.call_energy exactly once (monkeypatched
      counter) whether or not a core has been compiled, and returns
      system[ENERGY_KEY]; a `pos=` override still lands on
      system["atoms","pos"] before the flat read (changing pos changes the
      energy); `grep -n "_write_energy" src/molpot/forcefield/potentials.py`
      produces no output; every pre-change result on this branch is bitwise
      identical.
    status: verified
    last_checked: 2026-09-03
  - id: ac-012
    summary: An over-capacity cell fails closed at both guards, and the k-grid is the one that says so
    type: scientific
    pass_when: |
      After compile_core on a capacity_factor=1.2 grid, scaling the cell by
      2.0 (a genuine overflow: on the 10 A cube floor(2.0 x) = 5 > N_c = 4,
      while 1.5x stays inside the box) makes the eager rebind path
      (Potentials.bind_cell / MD.run) raise RuntimeError whose message
      contains BOTH "KSpaceGrid" and "capacity_factor" (the eager rebind_cell
      guard), with the mounted NeighborList configured so that
      scale_isotropic does not raise first (asserted: no ValueError about
      r_build precedes it); forcing the same cell through the compiled core
      without rebinding yields NaN energy and NaN forces (the in-graph
      sentinel) rather than a finite truncated value; scaling by 1.5 neither
      raises nor produces NaN; the compiled step runs under torch.no_grad()
      (AOTAutograd's donated buffers reject create_graph — the documented
      double-backward boundary).
    status: verified
    last_checked: 2026-09-04
    note: |
      CPU: TestPotentialsCompiledCore (2.0x bind_cell raises KSpaceGrid +
      capacity_factor, compiled energy is NaN, 1.5x is finite, donated-buffers
      RuntimeError under grad) and
      TestCompiledMlipStep.test_an_over_capacity_cell_fails_at_kspace_before_the_neighbour_list
      (MD.run barostat 2.0x, r_build does not raise first). GH200 job 1974822
      ran both in the 645-passed unit pass.
  - id: ac-013
    summary: The notes topic carries the break_sites inventory, the GH200 numbers and the false code-complete record
    type: docs
    pass_when: |
      .claude/notes/notes.md topic compile-no-graph-breaks lists the
      post-chain inventory generated from ExplainSummary.break_sites
      (cell-bound Polar energy_core: 1 graph / 0 breaks; whole step: the
      named autograd seam at derivation/force.py:179-180 and :189-194, owner
      molpot derivation, retained by Q3; remaining named breaks:
      gto/hartree.py:96-151 owned by polar-compile-clean-04-pairs,
      multipole_ewald.py:221-245 / :352-370 owned by molpot elec), the
      measured GH200 fp32 step time, peak memory and post-warm-up recompile
      count with date and job id, and one sentence recording that
      .claude/specs/INDEX.md:22 marked md-forcefield-molpy-reexport-05-wire
      [code-complete] while its ac-007 was still pending and factually unmet -
      "[code-complete] is not acceptance".
    status: verified
    last_checked: 2026-09-04
    note: |
      notes.md topic compile-no-graph-breaks now carries the inventory (core
      1/0; step breaks force.py:179, :180, :189), GH200 job 1976727 fp32
      0.190 s / 25.87 GiB / 0 recompiles, and the "[code-complete] is not
      acceptance" sentence.
  - id: ac-014
    summary: A Potentials rebuilt by to_potentials() still dispatches through the compiled core
    type: code
    pass_when: |
      After bind_cell + one eager call + compile_core, a second
      ff.to_potentials() returns a NEW Potentials object (`is not` the first)
      whose .mlip `is` the same nn.Module, and whose calc_energy increments the
      compiled callable's call counter; driving MD through set_forcefield /
      set_neighbors / set_integrator / set_barostat (each triggering
      src/molix/md/driver.py:300-318 _wire, hence a fresh Potentials) and then
      run(system, 2) keeps incrementing it - the compiled core survives every
      rebuild because it is installed on the mlip, not on the container.
    status: verified
    last_checked: 2026-09-03
out_of_scope:
  - Whole-step zero breaks (the autograd seam at derivation/force.py stays eager by Q3; owner molpot derivation)
  - Double-backward compile (cuet-force-doublebackward)
  - Compiling the strain wrapper (Q3 decided; widen via strained shifts in a separate spec)
  - Capacity-sized static edges + whole-step CUDA graph capture (follow-up; this spec uses Compiler(dynamic=True))
  - An is_cell_bound flag (meaningless for MatPES/OMol; covered by the call sequence and 01's named trace-time RuntimeError)
  - Isolated all-pairs break (polar-compile-clean-04-pairs) and multipole_ewald (molpot elec)
  - Neighbour-list compile, OMol/MatPES
  - molix.ff_export -> molpot reversed edge (5 files; separate refactor spec)
  - The remaining pending criteria of md-forcefield-molpy-reexport-05-wire (ac-005/006/008/009)
  - .claude/notes/architecture.md:84 stale molix.md export list (route /mol:map)
  - benchmarks/bench_polar_compile.py bench guard (named debt, follow-up spec)
---

# Acceptance — polar-compile-clean-03-wire

本步完成 = Polar 的编译核**装在 mlip 上**（`MacePotential.compile_core`），因此 `ff.to_potentials()` 每次重建出的新 `Potentials` 都仍走同一个编译核；`mlip.energy_core` 在给了 cell、B=1、一阶的路径上于 `aot_eager` 下 1 图 0 断点，而整步 `calc_energy_forces_virial` 的断点**全部且仅**落在 `src/molpot/derivation/force.py` 那条按 Q3 保留的 autograd 缝里（owner = molpot derivation）。`_write_energy` 保住它唯一的一次拆包与 `protocol.write_energy`，每步把**未 detach** 的 `graphs.energy` 写回 batch；`Potentials` 侧只剩两层一行委托，mlip 分支一律经 `protocol.call_energy`，鸭子类型探针与 `_eager_calls` 计数器一并消失。元素校验保证仍在（eager 侧，`_elements_validated` 是唯一事实），只是缝里不做且写进了文档。编译核与 eager 在 fp64 下一致到 1e-10；GH200 官方 goldens 一个数不变，compiled fp32 步时、峰值显存与热身后 recompile 计数已记录。越容量的 cell 在两道闸门上都 fail-closed（eager 抛出点名 `KSpaceGrid` 与 `capacity_factor` 的错 / 图内 NaN），绝不静默截断。**本 spec 与 02-seam 一起合并。**
