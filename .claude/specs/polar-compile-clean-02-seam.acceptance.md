---
slug: polar-compile-clean-02-seam
created: 2026-09-03
criteria:
  - id: ac-001
    summary: Compiler compiles a bound method and a plain function
    type: code
    pass_when: |
      Compiler(backend="aot_eager")(obj.method) and Compiler(...)(plain_fn)
      both return callables whose output is torch.equal to the uncompiled
      result on a (4, 3) CPU tensor; the emitted log line contains the
      target's __qualname__ and the call does not raise AttributeError on
      __class__.__name__; the two existing nn.Linear tests in
      tests/test_molix/test_compile.py pass unchanged.
    status: pending
  - id: ac-002
    summary: Compiler.explain returns a molnex ExplainSummary carrying break sites
    type: code
    pass_when: |
      Compiler.explain(fn, x) returns an ExplainSummary with graph_count == 1,
      graph_break_count == 0 and break_reasons == break_sites == () for a
      break-free fn; for a fn containing a print() it reports
      graph_break_count >= 1 with len(break_reasons) == len(break_sites) ==
      graph_break_count, every element a str, and at least one break_sites
      entry of the form "<file>:<lineno>" naming the test file and the line of
      that print(); Compiler has no count_graph_breaks attribute; ExplainSummary
      is importable from molix and listed in molix.__all__; `grep -rn
      "torch._dynamo.ExplainOutput" src/ tests/ benchmarks/` is empty.
    status: pending
  - id: ac-003
    summary: bind_work rebinds the cell once on the freshly read live storage
    type: code
    pass_when: |
      With a stub pots satisfying CellBound, _TensorDictForce.bind_work(work)
      invokes rebind_cell(), which calls bind_cell exactly once with a (3, 3)
      tensor whose .data_ptr() equals that of work["graphs","cell"] (or its
      [0] slice when the stored cell is (1,3,3)); after work["graphs","cell"]
      is replaced by a different tensor, rebind_cell(new) copies into and
      dispatches on the REPLACEMENT (data_ptr follows), never a cached view;
      with a stub that is not CellBound the call is a no-op and the resolved
      _eval is identical to the pre-change resolution; with graphs.cell absent
      _cell_view is None and bind_cell is not invoked on either call path; a
      RuntimeError raised by bind_cell propagates out of MD.run.
    status: pending
  - id: ac-004
    summary: MD._sync_cell rebinds once per barostat update, not once per step
    type: code
    pass_when: |
      An MD run of 6 steps with chunk=2 and a barostat stub invokes bind_cell
      1 + 3 times (once from bind_work, once per _sync_cell), each time with a
      (3, 3) tensor sharing storage with the tensor currently at graphs.cell
      and equal to that update's barostat cell; the same run with chunk=1 over
      3 steps gives 1 + 3; an NVT run (no barostat) invokes it exactly once; a
      RuntimeError raised by bind_cell inside _sync_cell propagates out of
      MD.run; src/molix/md/driver.py contains no `cell.dim() == 3` branch
      outside _TensorDictForce.rebind_cell; and
      tests/test_molix/test_md/test_barostat.py:87-116 and :119-148 both pass
      UNMODIFIED.
    status: pending
  - id: ac-005
    summary: molix gains no import dependency on molpot
    type: code
    pass_when: |
      `grep -nE "^\s*(from|import)\s+molpot" src/molix/md/driver.py
      src/molix/md/forcefield.py src/molix/compile.py src/molix/schema.py`
      produces no output, and none of the four files adds any new import
      outside molix / torch / tensordict / typing / collections.abc. (The
      criterion is about imports: those files legitimately name molpot in
      prose at driver.py:222, forcefield.py:10,15 and schema.py:8.)
    status: pending
  - id: ac-007
    summary: CellBound and CoreCompilable are declared in molix.schema with written contracts
    type: code
    pass_when: |
      src/molix/schema.py declares @runtime_checkable class CellBound(Protocol)
      with bind_cell(cell: torch.Tensor) -> None and @runtime_checkable class
      CoreCompilable(Protocol) with compile_core(compiler: Callable[[Callable[
      ..., Any]], Callable[..., Any]]) -> None; CellBound's docstring states
      all four contract clauses (idempotent; O(1) host work; must re-read the
      handed live (3,3) tensor and never retain a copy; may raise RuntimeError
      on overflow and must never truncate silently); isinstance() returns True
      for a stub with the method and False for one without; both names are
      re-exported from src/molix/md/forcefield.py __all__ and are the SAME
      objects (`is`), with the contract text living only in schema.py.
    status: pending
  - id: ac-008
    summary: PiNetPotential no longer overrides nn.Module.compile and gained no alias
    type: code
    pass_when: |
      PiNetPotential.compile is nn.Module.compile and "compile" is not in
      PiNetPotential.__dict__; PiNetPotential has no `compiled` attribute;
      `grep -rn "\.compiled(" src/ tests/ benchmarks/ docs/` produces no
      output; `grep -rn "torch.compile(" src/molzoo/pinet/potential.py` is
      empty; docs/molix/explanation/throughput-and-compilation.md:153-160
      shows Compiler(cuda_graphs=True)(model) / trainer.compile(
      cuda_graphs=True) plus a migration note for the removed method.
    status: pending
  - id: ac-009
    summary: No raw torch._dynamo.explain call sites remain in tests or benchmarks
    type: code
    pass_when: |
      `grep -rn "torch._dynamo.explain" tests/ benchmarks/` produces no
      output; tests/test_molzoo/test_mace/test_potential.py's
      test_energy_core_compiles_to_one_graph asserts on
      Compiler.explain(...).graph_count == 1 and .graph_break_count == 0 and
      no longer calls torch._dynamo.reset() by hand; benchmarks/
      pinet_compile_probe.py probe E prints the four ExplainSummary fields.
    status: pending
  - id: ac-010
    summary: The blocking ty gate stays green through the widened Compiler
    type: code
    pass_when: |
      `ty check src/` reports no error at src/molix/core/trainer.py:656-662
      (the assignment of Compiler(...)(self.model) to self.model: nn.Module)
      nor anywhere in src/molix/compile.py or src/molix/schema.py; the full
      `$META.build.check` command exits 0.
    status: pending
out_of_scope:
  - The cross-package isinstance(Potentials, CellBound) check (carried by 03-wire ac-001; unverifiable at this spec's close)
  - Potentials.compile_core / bind_cell, MacePotential.compile_core, element gate, break-site inventory (03; one merge unit with this spec)
  - A 02-only regression file (the merge unit's public-API script is 03's regressions/polar-compile-clean-03-wire.py)
  - The compile-no-graph-breaks notes inventory (owned by 03)
  - MDRunner.run's MDState cell-layout resolution at src/molix/md/runner.py:178-181 (declared second site)
  - KSpaceGrid and the isolated all-pairs break (01 / 04-pairs)
  - tests/test_molix/test_md/oracle_bruteforce_neighbors.py layout violation (named, routed to /mol:fix)
---

# Acceptance — polar-compile-clean-02-seam

本步完成 = `Compiler` 能编译绑定方法而不破坏 blocking 的 `ty` 门，并一次给出"图数 + 断点数 + 断点原因 + **断点位置**"；只回一个数的 `count_graph_breaks` 与覆盖 `nn.Module.compile` 的 `PiNetPotential.compile` 两个零调用方陷阱都被删除。跨层的两条契约（`CellBound` / `CoreCompilable`）是 `molix.schema` 里声明过的 `runtime_checkable` 协议（molpot 已经 import 那个模块，因此不需要 import `molix.md`），并从 `molix.md.forcefield` 与 `MDForceField` 并排再导出；重绑定只有 `_TensorDictForce.rebind_cell` 一个所有者，**每次重新读** `graphs.cell`，在 `MD.run` 的一次性绑定处以及**每次 barostat 更新**（每 `chunk` 一次，不是每步）被调用，`tests/test_molix/test_md/test_barostat.py` 的两条既有测试一行不改地通过。molix 仍不 import molpot。不涉及任何物理数值，本步不新增 regression。**本 spec 与 03-wire 一起合并。**
