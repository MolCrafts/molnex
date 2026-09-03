---
slug: polar-compile-clean-01-kgrid
created: 2026-09-03
criteria:
  - id: ac-001
    summary: The first bind sizes a fixed half-box from the real lattice norms
    type: code
    pass_when: |
      KSpaceGrid(k_cutoff=1.8467166200173746, capacity_factor=1.2).bind(10 A
      cubic cell) yields capacity (4, 4, 4) and an integer grid of shape
      (364, 3) with no all-zero row; every row's first non-zero component is
      > 0; a second bind with an unchanged cell returns the identical grid
      tensor (`is`-equal, no rebuild); a triclinic cell
      [[10,0,0],[3,9,0],[1,2,8]] gives
      capacity_i == ceil(1.2 * ||a_i|| * k_cutoff / (2*pi)) for i in 0..2.
    status: verified
    last_checked: 2026-09-03
  - id: ac-002
    summary: Weights are 0/2 and select exactly the old masked half-sphere
    type: code
    pass_when: |
      KSpaceGrid.vectors(cell) returns weights whose unique values are a subset
      of {0.0, 2.0}; the set of k-vectors with weight 2.0 equals, after
      sorting, the hard-coded fp64 output of the removed half_sphere_kvectors
      for the same cell and cutoff (46 vectors on the 10 A cubic cell, and the
      triclinic cell above); a k-vector with |k|**2 exactly equal to
      k_cutoff**2 gets weight 0.0 (strict <).
    status: verified
    last_checked: 2026-09-03
  - id: ac-003
    summary: Guarded denominator keeps energy and gradient finite at k=0
    type: scientific
    pass_when: |
      KSpaceGrid.vectors returns k_sq_guarded with all entries > 0 even when a
      degenerate/near-singular cell drives some |k|**2 to 0; the returned
      weights are 0.0 at those rows; the fp64 GTOHartreeEnergy value on the
      periodic branch and its autograd gradient w.r.t. positions are both
      torch.isfinite().all() - no NaN from 0 * inf.
    status: pending
  - id: ac-004
    summary: Sizing happens once; a later over-capacity cell raises and names KSpaceGrid
    type: code
    pass_when: |
      After bind on a 10 A cubic cell with capacity_factor=1.2, a second bind
      (and check_capacity) on that cell scaled by 2.0 raises RuntimeError whose
      message contains both "KSpaceGrid" and "capacity_factor" - so it is
      distinguishable from the neighbour-list overflow at
      src/molix/md/neighbors.py:838-842, which uses the same word; scaling by
      1.5, 1.1 and 0.5 neither raise nor rebuild (floor(1.5 x) = 4 = N_c: inside
      the box, no k-point truncated) (the integer box stays the same
      object, `is`-equal), i.e. the grid never re-sizes after the first bind;
      KSpaceGrid exposes no `rebind` and no `set_cell` attribute; a first bind
      whose max(N_c) > max_axis_index raises RuntimeError containing "GTO
      k-grid exploded"; the keyword is spelled max_axis_index (not
      max_capacity) and defaults to 32.
    status: verified
    last_checked: 2026-09-03
  - id: ac-005
    summary: num_graphs is explicit and the host sync survives only on the None branch
    type: code
    pass_when: |
      GTOHartreeEnergy.forward and GTOFieldFeatures.forward accept
      num_graphs: int | None = None; with num_graphs given, an AST scan of the
      periodic call chain (_kspace_energy, _kspace_potential,
      _kspace_features, _self_energy) finds no `.item()`, no `int(<Tensor>)`,
      no `float(<Tensor>)` and no boolean-mask subscript; `grep -n
      "from molix.F.scatter import"` in hartree.py and features.py shows
      scatter_sum_compile_safe and not scatter_sum.
    status: verified
    last_checked: 2026-09-03
  - id: ac-006
    summary: Both GTO modules compile to one graph with zero breaks on the periodic branch
    type: code
    pass_when: |
      On CPU with backend "aot_eager", torch._dynamo.explain(mod)(...) for
      GTOHartreeEnergy and GTOFieldFeatures after bind_cell, with a given cell
      and num_graphs=1, reports graph_count == 1 and graph_break_count == 0;
      with num_graphs=None the same call reports graph_break_count > 0 (the
      documented eager-only fallback).
    status: verified
    last_checked: 2026-09-03
  - id: ac-007
    summary: The isolated all-pairs branch is byte-identical and still breaks by design
    type: code
    pass_when: |
      inspect.getsource of _complete_directed_pairs,
      cached_complete_directed_pairs, _skip_diagonal and
      _skip_equal_orig_regular in src/molpot/potentials/gto/hartree.py equals
      the stored pre-change source literals character for character (a
      line-range git diff is not usable - imports above them shift); the
      isolated-branch fp64 energy matches the pre-change value with
      torch.allclose(atol=1e-12, rtol=0) - not bitwise, because the
      scatter_sum -> scatter_sum_compile_safe swap replaces scatter_add_ by
      index_add_ and reduction order is not a bitwise contract on CUDA; and an
      explain run on the isolated branch asserts graph_break_count > 0, with
      the assertion message naming polar-compile-clean-04-pairs as the owner.
    status: verified
    last_checked: 2026-09-03
  - id: ac-008
    summary: fp64 energy and strain derivative match the pre-change eager path
    type: scientific
    pass_when: |
      For two unit charges 1 A apart in a 10 A cubic cell at sigma=1.5 A,
      k_cutoff=1.8467166200173746 A^-1, the fp64 periodic GTOHartreeEnergy
      equals the hard-coded pre-change literal within 1e-10 eV; dE/dpos and
      the 3x3 dE/deps match their hard-coded pre-change literals within 1e-10
      (eV/A and eV respectively).
    status: pending
  - id: ac-009
    summary: Regression reproduces grid shape, weights, energy, zero breaks and both guards
    type: runtime
    pass_when: |
      `PYTHONPATH=src python regressions/gto-kspace-fixed-grid.py` exits 0,
      asserting M == 364, unique(weights) subset of {0.0, 2.0} with 46
      non-zero rows, the hard-coded fp64 Hartree golden within 1e-10 eV,
      bitwise equality between the (3,3) and (1,3,3) cell inputs, aot_eager
      explain == (1 graph, 0 breaks) after bind_cell, all-NaN weights when
      .vectors is called on a 2.0x-scaled cell, and a RuntimeError naming both
      KSpaceGrid and capacity_factor from check_capacity on that same cell;
      the file imports no mace / e3nn / graph_longrange / ase and runs no
      subprocess.
    status: verified
    last_checked: 2026-09-03
  - id: ac-010
    summary: An over-capacity grid returns NaN in-graph instead of truncating
    type: scientific
    pass_when: |
      With a grid bound to a 10 A cubic cell, KSpaceGrid.vectors on that cell
      scaled by 2.0 (no check_capacity call, i.e. the compiled route) returns
      weights that are entirely NaN, and the resulting GTOHartreeEnergy energy
      and its autograd force are NaN; the same call on a 1.5x cell (and a 1.1x cell) yields no
      NaN anywhere — floor(x) > N_c first fires at x >= 5, i.e. at 1.70x on
      this cell; torch._dynamo.explain(grid.vectors)(cell) reports
      graph_break_count == 0, proving the sentinel costs no host sync; the eager and in-graph
      guards use the same floor(x) > N_c predicate, so they fire on the same
      cell.
    status: pending
  - id: ac-011
    summary: An unbound or misplaced grid fails with a named error at trace time
    type: code
    pass_when: |
      Calling KSpaceGrid.vectors before any bind raises RuntimeError whose
      message contains "bind_cell"; compiling GTOHartreeEnergy.forward on the
      cell branch without a prior bind_cell surfaces that same named message
      (not a matmul shape error); a cell on a device different from the
      integer box raises a RuntimeError naming both devices; KSpaceGrid.to(
      device) returns self (`is`-equal) and moves the integer box and the
      .vectors output to that device.
    status: verified
    last_checked: 2026-09-03
  - id: ac-012
    summary: MacePolar.energy_core forwards num_graphs to both GTO submodules
    type: code
    pass_when: |
      With GTOHartreeEnergy.forward and GTOFieldFeatures.forward stubbed to
      record kwargs, MacePolar.energy_core(..., num_graphs=1, cell=cell)
      records num_graphs=1 on every call (the two field descriptors at
      polar.py:241-246 per recursion step and the coulomb energy at :266); no
      call reaches the int(batch.max().item()) fallback; on CPU / aot_eager,
      torch._dynamo.explain of a bind_cell-ed tiny MacePolar.energy_core with
      a cell reports (1 graph, 0 breaks). (01 uses the raw
      torch._dynamo.explain pattern of test_potential.py:740-748;
      Compiler.explain arrives in 02-seam, which migrates these sites.)
    status: verified
    last_checked: 2026-09-03
  - id: ac-013
    summary: The notes topic records inflation, both promotion triggers and the retained break
    type: docs
    pass_when: |
      .claude/notes/notes.md topic compile-no-graph-breaks records: the
      half-box vs half-sphere k-point counts for the PEO box (5318 ->
      21437 at f=1.2, 12194 at f=1.0) with the 408.5 MiB per (N,M) fp32
      tensor figure; multipole_ewald.py:221-245 / :352-370 as the same three
      defects with owner "molpot elec" and BOTH promotion triggers ("multipole
      _ewald becomes a second consumer" and "KSpaceGrid gains a second
      consumer first") pointing at elec/lib/kvectors.py; and the retained
      isolated-path break at hartree.py:96-151 with owner
      polar-compile-clean-04-pairs.
    status: pending
  - id: ac-014
    summary: A (1,3,3) batch cell is bitwise identical to the (3,3) cell
    type: code
    pass_when: |
      GTOHartreeEnergy.forward(feats, pos, batch, cell=cell.unsqueeze(0),
      num_graphs=1) is torch.equal to the same call with the (3,3) cell, and
      the same holds for GTOFieldFeatures.forward; _as_cell is the first
      statement of KSpaceGrid.bind, .check_capacity and .vectors; a (2,3,3)
      cell raises ValueError containing "single-system", and a two-graph batch
      still raises the owners' own ValueError at hartree.py:282-286 /
      features.py:205-209.
    status: verified
    last_checked: 2026-09-03
  - id: ac-015
    summary: The two existing Polar regressions are rerun and unchanged
    type: runtime
    pass_when: |
      `python regressions/mace-polar-periodic-gto.py` exits 0 with
      n_edges == 285546 and the four hard-coded fp64 literals at
      regressions/mace-polar-periodic-gto.py:40-46 unchanged within atol
      1e-3 eV (E0+inter -2829206.9445318216, electron 685.2709477417063,
      electrostatic 9.143779770475982, total -2828512.52980431); this spec
      edits no literal in that file; `python
      regressions/mace-polar-official-water.py` exits 0 with the isolated-path
      goldens at :28-33 reproduced within 1e-12.
    status: verified
    last_checked: 2026-09-03
  - id: ac-016
    summary: The ctor-derived GTOFieldFeatures buffers leave the state_dict
    type: code
    pass_when: |
      set(GTOFieldFeatures(density_sigma=1.5, density_lmax=1, feature_lmax=1,
      feature_widths=[1.5, 3.0]).state_dict()) is empty - combined_sigma,
      l0_factors, l1_factors, self_overlap_l0 and self_overlap_l1 are all
      registered with persistent=False; their attribute values and dtypes are
      unchanged (self_overlap_l0 ~ [19.199527, 12.142847], self_overlap_l1 ~
      [2.315467, 1.171544], float64), so
      tests/test_molpot/test_potentials/test_gto/test_features.py:87-94 passes
      unmodified; loading the official Polar checkpoint through CheckpointRemap
      reports no new missing or unexpected key.
    status: verified
    last_checked: 2026-09-03
out_of_scope:
  - Isolated batched all-pairs enumeration (retained; owner polar-compile-clean-04-pairs)
  - multipole_ewald._build_kvec_grid migration (two-way promotion trigger only)
  - A resize entry point on KSpaceGrid (rebind / set_cell); resize = build a new grid
  - GTOHartreeEnergy.self_overlap buffer persistence (untouched)
  - Neighbour-list compile, whole-step CUDA graphs, double-backward compile
  - Compiler widening, CellBound / CoreCompilable protocols, MD rebind, compile_core, element gate (02 / 03)
---

# Acceptance — polar-compile-clean-01-kgrid

本步完成 = `molpot.potentials.gto` 的**周期分支**跑在**只定尺一次**的整数半盒上，`num_graphs` 显式传入并一路到达 `MacePolar` 的两个 GTO 子模块，容量溢出在 eager 抛出同时点名 `KSpaceGrid` 与 `capacity_factor` 的错、在编译区内返回 NaN（永不截断），两侧用同一条 `floor(x) > N_c` 判据。`(1,3,3)` 的批量 cell 与 `(3,3)` 逐位一致（`_as_cell` 是三个入口的第一句）。`GTOHartreeEnergy` / `GTOFieldFeatures` / `MacePolar.energy_core` 在给 `cell` 时于 `aot_eager` 下 1 图 0 断点；孤立分支的四个枚举函数逐字节未改、断点仍在且被显式断言，其数值以 `atol=1e-12` 守（归约内核换了）。fp64 能量、力、应变导数与改动前 eager 路径一致到 1e-10，两个既有 Polar regression 重跑不改数。不碰 `Compiler`、不碰 MD driver、不碰 `Potentials`。
