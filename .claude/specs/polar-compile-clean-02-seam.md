---
title: Compiler 泛型加宽、ExplainSummary 与 MD 的 CellBound 重绑定缝
status: approved
grilled: true
created: 2026-09-03
slug: polar-compile-clean-02-seam
chain: polar-compile-clean
---

# Compiler 泛型加宽、ExplainSummary 与 MD 的 CellBound 重绑定缝

## Summary

编译缝是 `energy_core` 这样的**扁平绑定方法**，不是 `nn.Module`，而 `molix.compile.Compiler.__call__`（`compile.py:78`）与 `count_graph_breaks`（`:92-95`）今天的签名都写死 `nn.Module`；同时规范断言需要"1 图 **且** 0 断点 **且** 断在哪一行"三样东西，而 `count_graph_breaks` 只回一个数，逼得每个测试自己手搓 `torch._dynamo.explain`（`tests/test_molzoo/test_mace/test_potential.py:746`、`benchmarks/pinet_compile_probe.py:156`）。本步把 `Compiler.__call__` 改成 PEP 695 泛型（返回类型跟着入参走，`Trainer.compile` 在 `trainer.py:656-662` 把结果赋回 `self.model: nn.Module` 因此仍过 blocking 的 `ty` 门），新增一个 molnex 自己的 `ExplainSummary`（图数 / 断点数 / 断点原因 / **断点位置**），删掉零生产调用方的 `count_graph_breaks`，并把两条跨层契约**声明成 `runtime_checkable` Protocol** —— `CellBound`（`bind_cell`）与 `CoreCompilable`（`compile_core`）—— 放进 `src/molix/schema.py`（molpot 已经在 `derivation/protocol.py:36` import 的那个基础层契约模块），再从 `src/molix/md/forcefield.py` 与 `MDForceField` 并排再导出。MD 侧由 `_TensorDictForce.rebind_cell` 这**唯一**一个所有者在 `bind_work` 与每次 barostat 改 cell 之后调用，且**每次都重新读** `graphs.cell`（缓存 view 会踩回 `tests/test_molix/test_md/test_barostat.py:119` 守着的孤儿存储 bug）。顺带删掉 `PiNetPotential.compile`（它覆盖了 `nn.Module.compile` 的原地语义，是个静默 eager 陷阱，且零树内调用方）。molix 不新增任何 molpot import。

**02-seam 与 03-wire 是一个 merge unit** —— molix 侧的协议与钩子不单独上线；它们的树内实现者是 03 的 `Potentials.bind_cell` / `Potentials.compile_core` / `MacePotential.compile_core`。

## Domain basis

本步不含物理断言 —— 它只搬编译与绑定的接缝。必须保住的物理不变量有两条，均由 03-wire 的对照测试守：(a) MD 每步的 cell 依旧通过 `MD._sync_cell`（`driver.py:382-400`）**原地 copy_** 进当前存放在 `graphs.cell` 的那个张量（该方法的 docstring `:383-388` 已把这条写成不变式），因此 k 权重自动跟随活 cell，不需要回调；关键在于"当前存放的那个"必须**每次重新读**：`Potentials.calc_energy_forces_virial` 的应变闭包在 `potentials.py:311` 会把 `graphs.cell` **换成另一个张量**，并且只在无异常路径上于 `:328` 换回来。(b) 整数盒容量在**编译区之外**检查（01 的 `KSpaceGrid.check_capacity`，由 `bind_cell` 内部触发）：`bind_work` 时一次，NPT 每次 barostat 更新（每 `chunk` 一次）时一次；NPT 收缩恒不溢出、膨胀比 ≤ `capacity_factor` 恒不溢出（01 的 D4），越过容量则当次抛错，永不静默截断；即使这条钩子缺席，01 的 in-graph NaN 哨兵也保证不会静默截断。参考单位与常数见 `polar-compile-clean-01-kgrid` 的 Domain basis。

## Design

**`molix.compile.Compiler` 加宽（`src/molix/compile.py`）。**

- `def __call__[T: Callable[..., Any]](self, target: T) -> T:` —— PEP 695 泛型，实现体 `return cast(T, torch.compile(target, ...))`。`torch.compile` 本来就接受任意可调用对象；写成泛型而不是 `Callable -> Callable`，是因为 `Trainer.compile`（`trainer.py:654-665`）把返回值赋回 `self.model: nn.Module` 再对它调 `.to` / `.state_dict` / `.parameters` / `.train`，返回类型一旦被拓宽成裸 `Callable`，**blocking 的 `ty` 门就会红**（`$META.build.check` 里的 `ty check src/`）。该泛型写法已在本仓 `ty` 下验证为绿，保留。参数由 `module` 改名 `target`（既有调用方 `trainer.py:662` 与 `tests/test_molix/test_compile.py:14,22` 都是位置传参，不受影响）；docstring 从"a module"改成"a module or any callable"。日志行不能再假设 `target.__class__.__name__`（绑定方法会打成 `method`），改用 `getattr(target, "__qualname__", type(target).__name__)`。模块 docstring（`:3`）与类 docstring（`:27-36`）同步更新。
- 新增 `class ExplainSummary(NamedTuple)`：`graph_count: int`、`graph_break_count: int`、`break_reasons: tuple[str, ...]`、`break_sites: tuple[str, ...]`。**不返回 dynamo 的对象**：`torch._dynamo.ExplainOutput` 这个路径**不存在** —— 类实际住在 `torch._dynamo.backends.debugging.ExplainOutput`（本仓 `.venv-l40` torch 的 `debugging.py:537`；`.venv-gh200` torch 的 `debugging.py:598`），是私有 backends 子包里的 dataclass，把它写进 molnex 的公开签名等于把私有路径钉进类型。`break_reasons` 在上游是 `list[GraphCompileReason]`（`.venv-l40` `output_graph.py:279-286`，`.venv-gh200` `output_graph.py:330-336`；字段 `reason: str`、`user_stack: list[traceback.FrameSummary]`），我们只取 `tuple(str(r.reason) for r in out.break_reasons)`。
  - `break_sites` 与 `break_reasons` 一一对应、等长：每项取 `r.user_stack[-1]` 的 `f"{frame.filename}:{frame.lineno}"`，`user_stack` 为空时填 `"?"`。**这是 03-wire 断言"断点只在 `force.py`"以及 notes 断点清单的数据来源** —— 没有它，"哪里断了"就只能靠人读字符串。
- `@staticmethod explain(target, *args, **kwargs) -> ExplainSummary` —— 内部调 `torch._dynamo.explain(target)(*args, **kwargs)` 并投影成 `ExplainSummary`。上游 `explain` 自己在首尾调 `reset()`（`.venv-l40` `eval_frame.py:1612,1653`；`.venv-gh200` `eval_frame.py:1819,1860`），所以本方法不重复 reset。docstring 点明两件事：(1) **`graph_break_count = graph_count - 1`**（`.venv-l40` `eval_frame.py:1649`；`.venv-gh200` `:1856`），一次都没成图时它是 **−1**，所以"0 断点"必须和"1 图"一起断言 —— 这正是本类型同时携带两个数的理由；(2) 它是 `@staticmethod`，**在 dynamo 的默认配置下测量**，不使用调用它的那个 `Compiler` 实例的 `backend` / `mode` / `fullgraph` —— 想在别的后端下量，就在调用点自己设。
- **删除 `count_graph_breaks`**（`compile.py:92-95`）。全树扫描后它**只有测试调用方**（`tests/test_molix/test_compile.py:57,62`），生产零调用；留一个只回一个数的第二条 explain 路径，正是本步要消灭的东西。`tests/test_molix/test_compile.py:51-62` 的 `TestCountGraphBreaks` 改写为对 `Compiler.explain(...)` 的断言。
- `ExplainSummary` 与 `Compiler` 一样从 `src/molix/__init__.py` 导出（`:102` 的 import 与 `:111`/`:126` 的 `__all__` 各加一行）—— 断言型 API 与它的返回类型不该住在两个可见性等级上。

**两个跨层协议（`src/molix/schema.py`）。** 放在 `schema.py` 而不是 `md/forcefield.py`，理由是**消费者**：molpot 已经 import `molix.schema`（`derivation/protocol.py:36`），而 molpot **不得**、也不需要 import `molix.md`。把跨层结构契约放进基础层的契约模块，03 的实现者才可以在 `src/molpot/` 里 `from molix.schema import CellBound` 而不违反单向包规则。

```python
@runtime_checkable
class CellBound(Protocol):
    def bind_cell(self, cell: torch.Tensor) -> None: ...


@runtime_checkable
class CoreCompilable(Protocol):
    def compile_core(
        self, compiler: Callable[[Callable[..., Any]], Callable[..., Any]]
    ) -> None: ...
```

`CellBound` 的四条契约（binding，写在 schema.py 的类 docstring 上，只此一份）：**幂等**；只做 O(1) 的 host 工作；必须**重新读**交给它的那个活 `(3, 3)` 张量（**绝不留副本** —— molix 用 `copy_` 原地改它）；容量溢出时**可以**抛 `RuntimeError`，**永远不得静默截断**。`CoreCompilable` 的契约：`compiler` 是结构类型（`Compiler.__call__` 的泛型形状天然满足），实现者自己决定编译什么、装在哪；重复调用不重编。

两者从 `src/molix/md/forcefield.py` **再导出**（加进 `:23` 的 `__all__`，与 `MDForceField` 并排），因为 MD 侧读者会在那里找它们；`MDForceField`（`:26-39`）的理由段同样适用，`forcefield.py:10,15` 已经在散文里合法地提到 molpot。**契约文本只住在 schema.py**，`forcefield.py` 只做 `from molix.schema import CellBound, CoreCompilable` + `__all__`，不复述。`molix.md.driver` 直接从 `molix.schema` import（它已经在 `driver.py:42` import 那个模块）。

**唯一的重绑定所有者（`src/molix/md/driver.py`）。**

- `def rebind_cell(self, new_cell: torch.Tensor | None = None) -> None:` —— 三步，顺序固定：
  1. **重新解析**：从 `self._work` 现在存的 `("graphs","cell")` 取张量，按 `cell[0] if cell.dim() == 3 else cell` 定 `self._cell_view`（即今天 `:71-73` 那条规则）；键不存在 ⇒ `self._cell_view = None`。**每次都重新读**，绝不复用上一次缓存的 view —— `potentials.py:311` 的应变闭包会把 `graphs.cell` 换成另一个张量，`:328` 只在无异常时换回来，缓存 view 会 copy 进孤儿存储，这正是 `tests/test_molix/test_md/test_barostat.py:119 test_sync_cell_follows_a_replaced_graphs_cell` 守的那个 bug。
  2. **落值**：`new_cell is not None and self._cell_view is not None` ⇒ `self._cell_view.copy_(new_cell)`。
  3. **派发**：`self._cell_view is not None and isinstance(self._pots, CellBound)` ⇒ `self._pots.bind_cell(self._cell_view)`。`_cell_view` 为 `None`（无 `graphs.cell`）时整段静默 no-op —— 不传 `None` 给一个签名要 `Tensor` 的方法。
- `bind_work`（`:68-79`）在 `self._work = work` 之后调一次 `rebind_cell()`（无 `new_cell`），删掉 `:70-73` 那段内联解析 —— 这是 `MD.run`（`:357-362`）在 `neighbors.build(system)` 之后、进入求值之前的唯一时刻。
- `MD._sync_cell`（`:382-400`）改为：保留 `state.cell is None / _adapter is None / work is None` 三道早退，然后 `self._adapter.rebind_cell(state.cell)`（把 `:395-397` 那三行连同它自己的 `cell.dim() == 3` 判断一起删掉 —— MD 不再自己判断 cell 布局），再 `work[POS_KEY] = state.pos`，最后 `self.neighbors.scale_isotropic(...)`。`_cell_view` 这个今天**只写不读**的属性（`:62,70,73`）由此获得它的第一个读者。
- **节奏更正**：`_sync_cell` 只在 barostat 存在时被 `MDRunner` 调用，且是**每 `chunk` 一次**（`runner.py:208-211`，`if self.barostat is not None:` 里的 `self.sync_state(md)`），**不是每步**。NVT（无 barostat）下 `bind_cell` 只在 `bind_work` 被调一次。
- **单一所有者的范围**：`rebind_cell` 是"cell 布局解析"在**通往势的那条路上**唯一的家。树内还有第二处同形代码 —— `MDRunner.run` 在 `runner.py:178-181` 为 `MDState` 解析 `(1,3,3) → (3,3)`。它服务的是**积分器状态**而不是势，本步不动，但在 Out of scope 里点名登记，避免"唯一所有者"这句话被读成"全树唯一一处"。
- pots 不满足 `CellBound` 时第 3 步是 no-op —— 经典力场路径行为逐位不变（`tests/test_molix/test_md/test_barostat.py:87-148` 的两条测试用的就是 `_TensorDictForce(object(), 1.0)`，它们必须**一行不改**地通过）。这是 `KSpaceFilter.update(cell, ns_mesh)`（`kspace_filter.py:89`）那条"cell 变了 → 带外重算"接缝在 MD 侧的对应物，也是 `NeighborList.scale_isotropic`（`neighbors.py:1012`）的同位物。

**不在 molix 侧加"第四个求值分支"。** 每步的求值器已经在 `bind_work` 里一次性解析成 `_eval_npt` / `_eval_mlip` / `_eval_classical`（`:74-79`）。再加一条"是否已编译"的分支会造出**死分支**（molix 无法在不 import molpot 的前提下判断编译核的语义），也违背"一次解析"的初衷。**一次性解析 = 编译核作为属性装在 mlip 上**（03 的 `MacePotential.compile_core`），`_write_energy` 每步调用同一个属性；**molix 侧无分支**，每步零解析开销，molix 也不多一条 import。

**删除 `PiNetPotential.compile`（`src/molzoo/pinet/potential.py:130-156`）。** 它有两个问题：(a) 复制了第二份 `torch.compile` kwargs；(b) 它**覆盖了 `nn.Module.compile`** —— 基类那个是原地生效、返回 `None` 的，这个返回一个新的 `OptimizedModule`，所以 `model.compile(...)` 而不接返回值会静默地什么也没做。**整只删除，不改名、不留替代方法**：全树 `grep -n "\.compile("` 后它**没有任何树内调用方**（`benchmarks/bench_pinet*.py` 用的是裸 `torch.compile`），一个零调用方的别名再包一层只是把陷阱换个名字；删除后 `PiNetPotential.compile is nn.Module.compile`，语义回到基类的原地契约。唯一的树内提及是 `docs/molix/explanation/throughput-and-compilation.md:153-160` 那段展示树外 `pqtn.training.train_pinet` 用法的示例，改写为 `Compiler(cuda_graphs=True)(model)` / `trainer.compile(cuda_graphs=True)`（`:162-164` 已经在推荐这条路），并加一句迁移说明"旧的 `PiNetPotential.compile(...)` 已删除；`nn.Module.compile` 是原地的，需要新对象就用 `Compiler`"。本步因此是"molix + 这一处 molzoo 清理"。

**把两处裸 `torch._dynamo.explain` 迁到 `Compiler.explain`**：`tests/test_molzoo/test_mace/test_potential.py:745-748`（连带删掉手写的 `torch._dynamo.reset()`）与 `benchmarks/pinet_compile_probe.py:156-162`（probe E 改印 `ExplainSummary` 的四个字段）。

**notes 归属**：`.claude/notes/notes.md` topic `compile-no-graph-breaks` 的清单更新由 **03-wire 一次性负责**（它才有链末的实测数与最终断点清单，且清单要从 `break_sites` 生成），本步不改 notes —— 单一所有者，避免同一 topic 被两次半更新。

### Reuse decision

- **(13) `molix/compile.py:26 Compiler` — `generalize`。** 见 Tasks 的 "Generalize Compiler…"：`__call__` 从 `nn.Module` 泛化为 PEP 695 的 `[T: Callable[..., Any]]`，新增 `ExplainSummary` + `explain`。不新建第二个编译包装类型，也不把 dynamo 的私有 `ExplainOutput` 抬进签名。
- **(19) `Compiler.count_graph_breaks`（`tests/test_molix/test_compile.py:51-62` 覆盖）— `new`（删除）。** 原判决是"保留公开名并委托"，被 architect 的 🟡 推翻：全树只有它自己的两个测试调用它，生产零调用方，保留就是保留一条只回一个数的第二 explain 路径。删除，测试改写到 `Compiler.explain`。
- **(14) `molix/md/driver.py:68 _TensorDictForce.bind_work` — `generalize`。** 见 Tasks 的 "Generalize the MD driver…"：在既有的一次性解析里换成 `rebind_cell()`，并让 `MD._sync_cell` 复用同一个方法。**原判决里的"缓存 `_cell_view` 再 copy_"被 architect 的 🚨 推翻**：`rebind_cell` 必须每次**重新读** `graphs.cell`（`potentials.py:311/:328` 会替换它），否则重开 `test_barostat.py:119` 守的孤儿存储 bug。`_sync_cell:395-397` 那份重复的 cell 布局解析被删除。
- **(16) `molzoo/pinet/potential.py:130 PiNetPotential.compile` — `new`（删除）。** 原判决是"改名 `compiled` 并委托 `Compiler`"，被 architect 的 🔴 推翻：那会新增一个零调用方的公开方法并重述 `Compiler` 的四个 kwargs。直接删，`compile` 回落到 `nn.Module.compile`。
- **(20) `tests/test_molix/test_md/test_compile.py:17 _BACKEND = "aot_eager"` — `pattern`（照抄）。**
- **(1)–(12) → 01-kgrid；(10)(15)(17)(18) → 03-wire。** 本步不碰。

**关于跨包**：Files 触及 `molix` 与 `molzoo`（一处 `pinet` 清理）。这是 architect 明确要求并入本步的、与 `Compiler` 加宽同一议题的单点删除（不删就留下第二份 kwargs + 一个静默陷阱，违反 iron law "no silent debt"），不是范围扩张。

## Files to create or modify

- src/molix/compile.py
- src/molix/__init__.py
- src/molix/schema.py
- src/molix/md/forcefield.py
- src/molix/md/driver.py
- src/molzoo/pinet/potential.py
- docs/molix/explanation/throughput-and-compilation.md
- tests/test_molix/test_compile.py
- tests/test_molix/test_schema.py
- tests/test_molix/test_md/test_forcefield.py
- tests/test_molix/test_md/test_driver.py
- tests/test_molix/test_md/test_barostat.py
- tests/test_molzoo/test_pinet/test_potential.py
- tests/test_molzoo/test_mace/test_potential.py
- benchmarks/pinet_compile_probe.py

## Tasks

- [ ] Write failing unit tests for Compiler on a bound method, for the generic return type and for Compiler.explain / ExplainSummary including break_sites (tests/test_molix/test_compile.py → TestCompiler, TestExplainSummary)
- [ ] Generalize Compiler.__call__ to a PEP 695 generic callable, add ExplainSummary with break_sites and Compiler.explain, delete count_graph_breaks, export ExplainSummary from src/molix/__init__.py, and refresh the stale docstrings at src/molix/compile.py:3 and :27-36
- [ ] Write failing unit tests for the CellBound / CoreCompilable protocols, their molix.md re-export, and rebind_cell at bind_work, per barostat update and on the missing-cell path (tests/test_molix/test_schema.py → TestCellBound, TestCoreCompilable; tests/test_molix/test_md/test_forcefield.py → TestProtocolReexports; tests/test_molix/test_md/test_driver.py → TestRebindCell; tests/test_molix/test_md/test_barostat.py)
- [ ] Declare the runtime_checkable CellBound and CoreCompilable protocols with the four-clause contract docstring in src/molix/schema.py, widen that module's docstring from "batch-schema keys + constants" to "keys, shared constants, and the cross-layer structural protocols" (same one-way-dependency rationale), and re-export both from src/molix/md/forcefield.py __all__
- [ ] Generalize the MD driver in src/molix/md/driver.py with _TensorDictForce.rebind_cell(new_cell=None) as the single owner of cell-layout resolution on the pots-binding path, re-reading graphs.cell on every call, invoked from bind_work and from MD._sync_cell
- [ ] Write failing unit tests for PiNetPotential no longer overriding nn.Module.compile (tests/test_molzoo/test_pinet/test_potential.py → TestPiNetPotentialCompile)
- [ ] Delete PiNetPotential.compile in src/molzoo/pinet/potential.py and repoint the example at docs/molix/explanation/throughput-and-compilation.md:153-160 to Compiler(cuda_graphs=True) / trainer.compile(cuda_graphs=True) with a migration note
- [ ] Migrate the raw torch._dynamo.explain call sites in tests/test_molzoo/test_mace/test_potential.py:745-748 and benchmarks/pinet_compile_probe.py:156-162 — plus the sites 01-kgrid added under tests/test_molpot/test_potentials/test_gto/ and in tests/test_molzoo/test_mace/test_polar.py — to Compiler.explain, so ac-009's grep is empty after this step
- [ ] Run full check + test suite

## Testing strategy

单元测试只测单函数/单模块，路径镜像 `src/`（`src/molix/compile.py` → `tests/test_molix/test_compile.py`，`src/molix/schema.py` → `tests/test_molix/test_schema.py`，`src/molix/md/driver.py` → `tests/test_molix/test_md/test_driver.py`，`src/molix/md/forcefield.py` → `tests/test_molix/test_md/test_forcefield.py`，`src/molzoo/pinet/potential.py` → `tests/test_molzoo/test_pinet/test_potential.py`），类型镜像（`Compiler` → `TestCompiler`）。

**本步不新增 regression。** 02 与 03 是一个 merge unit，公开面场景由 03 的 `regressions/polar-compile-clean-03-wire.py` 承载：它经 `pots.compile_core(Compiler(...))` 走本步的泛型 `__call__`（编译的是一个绑定方法），并用 `Compiler.explain` 断言图数 / 断点数 / `break_sites`。再造一个只驱动私有适配器、断言桩计数的 `regressions/polar-compile-clean-02-seam.py` 只会把单元测试抄一遍成"公开面示例"，那是假的公开面 —— 这条偏离是有意的，记在 Out of scope。跨包那条契约（`isinstance(Potentials(...), CellBound)`）由 03-wire 的 ac-001 守（在本 spec 关闭时不可验证，故不列为本 spec 的验收项）。

- `tests/test_molix/test_compile.py::TestCompiler`
  - happy path：`Compiler(backend="aot_eager")(obj.method)` 返回可调用体，对 `(4, 3)` 输入给出与未编译方法逐位相等的结果（`torch.equal`）；对普通 `def f(x)` 同理；对 `nn.Linear` 仍返回可当 `nn.Module` 用的对象（既有 `test_call_returns_compiled_wrapper` / `test_compiled_module_produces_same_output` 不改而通过）。
  - edge case：日志不再对绑定方法崩在 `__class__.__name__` —— 用 `caplog` 断言日志行含方法的 `__qualname__`。
- `tests/test_molix/test_compile.py::TestExplainSummary`
  - happy path：`Compiler.explain(fn, x)` 返回的对象同时暴露 `graph_count == 1` 与 `graph_break_count == 0`；`break_reasons` / `break_sites` 都是 `tuple[str, ...]`（不是 dynamo 对象）且为空。
  - edge case（断点位置）：一个函数体里插一行 `print(...)` ⇒ `graph_break_count >= 1`、`len(break_reasons) == len(break_sites) == graph_break_count`，且**至少一个 `break_sites` 项以本测试文件名开头并带着那行 `print` 的行号**（硬编码文件名与"至少 1"，不硬编码 dynamo 版本相关的精确断点数）。
  - edge case：`Compiler` 上不再有 `count_graph_breaks` 属性；`ExplainSummary` 可从 `molix` 顶层 import 且在 `molix.__all__` 里。
  - 静态：`grep -rn "torch._dynamo.ExplainOutput" src/ tests/ benchmarks/` 无输出（那个路径不存在）。
- `tests/test_molix/test_schema.py::TestCellBound` / `::TestCoreCompilable`
  - happy path：只有 `bind_cell` 的桩类 `isinstance(obj, CellBound)` 为 True，缺该方法的桩为 False；只有 `compile_core` 的桩对 `CoreCompilable` 同理。
  - 契约文本：`CellBound.__doc__` 含四条子句（幂等 / O(1) host / 重读活张量不留副本 / 可抛 RuntimeError 但绝不静默截断）。
- `tests/test_molix/test_md/test_forcefield.py::TestProtocolReexports`
  - happy path：`molix.md.forcefield.CellBound is molix.schema.CellBound`，`CoreCompilable` 同理；两者都在 `molix.md.forcefield.__all__` 里；契约 docstring **只**住在 schema.py（`forcefield.py` 不复述）。
- `tests/test_molix/test_md/test_driver.py::TestRebindCell`
  - happy path：pots 桩满足 `CellBound`，`bind_work(work)` 后记录到**恰好一次**调用，实参是 `(3, 3)` 且与 `work["graphs","cell"]` 共享存储（`.data_ptr()` 相等 —— 这是 `_sync_cell` 原地 `copy_` 能生效的前提）；`(1,3,3)` 批量 cell 时实参是 `[0]` 切片。
  - happy path（重读）：`bind_work` 之后把 `work["graphs","cell"]` 换成另一个张量，再 `rebind_cell(new)` ⇒ `copy_` 落在**新**张量上、`work["graphs","cell"] is` 新张量、`bind_cell` 收到的也是新张量的 view（缓存 view 的退化在这里被抓）。
  - happy path（节奏）：带 barostat 的 `MD` 桩跑 6 步、`chunk=2`，`bind_cell` 被调用 1（bind_work）+ 3（每 chunk 一次 `_sync_cell`）次，每次实参与 `graphs.cell` 共享存储且数值等于当次 barostat 的 cell；`chunk=1` 跑 3 步则是 1+3 次。
  - edge case：无 barostat（NVT）时只在 `bind_work` 被调一次。
  - edge case：pots 不满足 `CellBound` → 不抛、`_eval` 的解析结果与改动前一致；`graphs.cell` 缺失 → `_cell_view` 为 `None` 且 `bind_cell` **两条调用路径上都**不被调用（`rebind_cell()` 与 `rebind_cell(state.cell)` 各一条）。
  - edge case：`bind_cell` 抛 `RuntimeError`（容量溢出）时原样冒泡到 `MD.run` 调用方，不被吞（`bind_work` 与 `_sync_cell` 两条路径各一条）。
  - 静态（依赖方向护栏）：`grep -nE "^\s*(from|import)\s+molpot" src/molix/md/driver.py src/molix/md/forcefield.py src/molix/compile.py src/molix/schema.py` 无输出。**注意断言的是 import 而不是字符串"molpot"** —— 后者今天就有（`driver.py:222`、`forcefield.py:10,15`、`schema.py:8` 的 docstring 里合法地提到 molpot），拿字符串断言会误报。
- `tests/test_molix/test_md/test_barostat.py`
  - **既有两条必须一行不改地通过**：`test_sync_cell_keeps_batched_graphs_cell`（`:87-116`）与 `test_sync_cell_follows_a_replaced_graphs_cell`（`:119-148`）—— 它们用 `_TensorDictForce(object(), 1.0)`，即一个不满足 `CellBound` 的 pots，正好同时覆盖"派发 no-op"与"重读替换后的 cell"。
  - 新增一条：`graphs.cell` 不存在时 `_sync_cell` 不抛、不写 cell、`_cell_view` 为 `None`。
- `tests/test_molzoo/test_pinet/test_potential.py::TestPiNetPotentialCompile`
  - happy path：`PiNetPotential.compile is nn.Module.compile`（覆盖已删除）；类的 `__dict__` 里没有 `compile`，也没有新增的 `compiled` 属性。
  - edge case：`grep -rn "\.compiled(" src/ tests/ benchmarks/ docs/` 无输出（没有把陷阱改名留下）；`docs/molix/explanation/throughput-and-compilation.md:153-160` 展示 `Compiler(cuda_graphs=True)(model)` / `trainer.compile(cuda_graphs=True)` 并带一句迁移说明。

## Out of scope

- `Potentials.compile_core` / `Potentials.bind_cell` / `MacePotential.compile_core` / `MacePolar.bind_cell` 的 molpot·molzoo 侧实现、molzoo 元素门、整步 explain 断点归属（→ 03-wire）。**本步与 03-wire 是一个 merge unit**：molix 的两个协议不单独上线。
- **本步不新增 regression 文件**（理由见 Testing strategy）；merge unit 的公开面脚本是 03 的 `regressions/polar-compile-clean-03-wire.py`。
- `.claude/notes/notes.md` topic `compile-no-graph-breaks` 的清单更新 —— 单一所有者是 03-wire（清单从 `break_sites` 生成）。
- **`MDRunner.run` 的第二处 cell 布局解析**（`src/molix/md/runner.py:178-181`，为 `MDState` 把 `(1,3,3)` 取 `[0]`）—— 它服务积分器状态而不是势，本步不合并；"`rebind_cell` 是唯一所有者"这句话的范围是**通往势的那条路**。合并到一处属另一次清理，点名登记于此。
- `KSpaceGrid` 本身与孤立分支断点（→ 01-kgrid / 04-pairs）。
- 邻居表 build/update 的编译、整步 CUDA graph 捕获、二阶反传编译（`cuet-force-doublebackward`：AOTAutograd 不支持 double backward，本链的编译缝只覆盖一阶 MD 推理）。
- **点名的既有债务（本步只报，不修）**：`tests/test_molix/test_md/oracle_bruteforce_neighbors.py` 是 `tests/` 下的自由浮动 helper 模块，违反 CLAUDE.md "Test layout rules"（`tests/` 只放 `test_*.py` / `conftest.py` / `__init__.py` / 数据文件）→ 路由 `/mol:fix`，把它并进最近的 `conftest.py`。`tests/test_molix/test_md/test_compile.py` 没有 `src/molix/md/compile.py` 镜像 → 记录，暂不动作（它测的是 `integrators.py` 的编译性质，重命名属另一次清理）。
- blueprint 重生成（`/mol:map`）。
