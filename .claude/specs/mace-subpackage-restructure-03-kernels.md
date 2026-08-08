---
title: Extract shared force-pass kernels into molpot.derivation.kernels
status: approved
created: 2026-08-08
grilled: true
chain: mace-subpackage-restructure
slug: mace-subpackage-restructure-03-kernels
---

# Extract shared force-pass kernels into molpot.derivation.kernels

## Summary

把当前重复了两份的"批级力学传递体"（batch-level force pass）抽取为 `src/molpot/derivation/kernels.py` 中的两个共享可调用体：grad 内核（一次能量前向 + `torch.autograd.grad` 就地求导）与 func 内核（`torch.func.grad` 单变换一次性拿到能量与力）。`FuncMode` / `GradMode` 与 `PiNetPotential` 全部改为绑定这两个内核，使链条下一步的 `MACEPotential`（04）不必手搓第三、第四份拷贝。为了同时容纳 MACE 的形状（当它自己创建 grad 叶子时会 detach 返回的能量，`mace_matpes.py:368-377`）与 PiNet 的形状（不 detach），内核暴露 `detach_energy` 叶子归属旋钮。本 spec 是**零行为变更重构**：PiNet 的数值必须逐位保持，其 `torch.compile(fullgraph=True)` 零 graph-break 合同必须继续成立，`FuncMode` / `GradMode` 的公开行为不变。

## Domain basis

原子力是保守势能面对坐标的负梯度：

```
F_i = -∂E/∂r_i        E: eV,  r: Å,  F: eV/Å
```

两个内核实现的是同一个物理量的两条反向自动微分路径，因此在数学上恒等，差异仅在于图的构造方式与可编译性：

- **autograd 路径**：先物化 `E(r)` 的计算图，再对位置叶子调用 `torch.autograd.grad`。适用于 cuEquivariance 融合核（其 legacy `autograd.Function` 缺少 `setup_context`，被 `torch.func` 拒绝 — pytorch#170834）。MACE 官方 `get_outputs` 即此形状：一次前向 + 一次反向。
- **functorch 路径**：`torch.func.grad(..., has_aux=True)` 把求导整体 trace 进前向图，`energy → force → loss` 只有一次 backward，且可与 `torch.compile(fullgraph=True)` 组合。仅适用于纯 PyTorch 能量图（PiNet）。

两条路径在同一组算子上给出逐位相同的梯度；本 spec 不改变任何一条路径上的算子序列，故"逐位保持"是可检验的强约束，而不是近似声明。

参考：
- Batatia et al., *MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields*, NeurIPS 2022, arXiv:2206.07697（`get_outputs` 的一前向一反向形状）。
- Shao, Hellström, Mitev, Knijff, Zhang, *PiNN: A Python Library for Building Atomic Neural Networks of Molecules and Materials*, J. Chem. Inf. Model. 60, 1184 (2020), DOI 10.1021/acs.jcim.9b00994（PiNet 能量/力定义）。
- pytorch/pytorch#170834（cuEq 融合核与 `torch.func` 不兼容的上游根因）。
- 仓库决策记录 `.claude/notes/notes.md` §"Force derivation — dual explicit backends (2026-07-29)"：`molpot.derivation` 是 `F = -∂E/∂pos` 的唯一归属；模型不得自造第三条力路径。

单位约定沿用仓库全局：位置 Å、能量 eV、力 eV/Å；内核本身与单位无关（几何无关，只接收能量可调用体与 batch），单位约束在 docstring 中声明以防调用方混用。

## Design

### 新增符号（`src/molpot/derivation/kernels.py`）

两个模块级自由函数——无状态、无参数、无自有类型可归属，且已有两个（很快四个）调用点，符合 CLAUDE.md "Shape check" 第 1/3 条与"true free operations"例外；命名与形状对齐同目录的 `force.py`（最近模式：模块级函数 + 模块 docstring 划定后端边界）。

```python
def grad_force_pass(
    energy_core: Callable[[TensorDict], TensorDict | dict | None] | None,
    batch: TensorDict,
    *,
    create_graph: bool | None = None,
    detach_energy: bool | None = None,
) -> TensorDict: ...

def func_force_pass(
    energy_core: Callable[[TensorDict], TensorDict | dict | None],
    batch: TensorDict,
) -> TensorDict: ...
```

**`grad_force_pass` 语义（就地写 batch，返回同一 batch）**

1. `pos = batch[POS_KEY]`；`owns_leaf = not (pos.requires_grad and pos.is_leaf)`。`owns_leaf` 为真时 `pos = pos.detach().requires_grad_(True)` 并写回 `batch[POS_KEY]`（即 `mace_matpes.py:368-369` 的 `needs_leaf` 语义）。
2. `energy_core is None` → 跳过前向，直接对 batch 上已物化的能量求导（`GradMode` 顺序协议中能量已就绪的分支；此时若 `owns_leaf` 为真则抛 `RuntimeError`，因为不存在可求导的图）。否则 `with torch.enable_grad(): batch = absorb_model_output(batch, energy_core(batch))`，随后校验 `has_energy(batch)`，失败抛 `RuntimeError` 且消息含 `graphs.energy`。
3. 力：`autograd_forces_from_energy(batch[ENERGY_KEY], pos, create_graph=create_graph)`（**复用**，不重写 `torch.autograd.grad`）。`create_graph=None` 沿用该函数默认 `torch.is_grad_enabled()`。
4. `detach_energy` 三态：`False` = 永不 detach（PiNet / `GradMode` 行为）；`True` = 总是 detach；`None` = **仅当本次调用创建了位置叶子时** detach（MACE `get_outputs` 行为，`mace_matpes.py:376`）。
5. `write_forces(batch, forces)`。

**`func_force_pass` 语义**：完全保留今天 `modes/func.py:71-91` 与 `pinet/potential.py:121-138` 的字面算子序列——`pos = batch[POS_KEY].detach()`、`base = batch.clone()`、闭包内 `b = base.clone()` 后写入 `p`、`energy_core(b)` → `absorb_model_output` → `has_energy` 校验 → 返回 `(b[ENERGY_KEY].sum(), b)`，再交给**复用**的 `functorch_forces_with_aux`（其已返回 `-grad`），最后回写 `graphs.energy`、可选 `atoms.energy`、`write_forces`。无 `detach_energy` 旋钮（cuEq 融合核不走 functorch，MACE 不需要）。

**生命周期与所有权**：内核不持有任何状态，不读写 `protocol._SESSIONS` 侧信道（session 记账留在 `modes/`），不引入 `nn.Module`。batch 就地修改，返回值即入参对象。

**编译约束（硬）**：`func_force_pass` 位于 PiNet 的 fullgraph 路径上，因此禁止 `set_non_tensor`、`.item()`、张量值相关分支、`_SESSIONS` 字典查找；`detach_energy` / `energy_core is None` 这类分支只出现在 grad 内核且是 Python 级静态分支。`energy_core` 由调用方在构造期绑定（PiNet 直接传 `self._write_energy` 绑定方法；modes 传 `functools.partial(call_energy, model)`，避免每次调用新建 lambda）。

### 调用点重绑定

| 调用点 | 变更 |
|---|---|
| `modes/grad.py` `GradMode.run_forces` | 保留 session 前置检查与既有 `RuntimeError` 文案（"needs positions with requires_grad"），能量未就绪时仍调 `self.run_energy(deriv, batch, backward=True)`（`n_forward == 1` 的既有断言依赖它），随后以 `energy_core=None`、`create_graph=bool(getattr(deriv.model, "training", False))`、`detach_energy=False` 委托内核 |
| `modes/func.py` `FuncMode.run_forces` | pass 体整体替换为 `func_force_pass(partial(call_energy, model), batch)`，其后的 `_lazy_func` / `_energy_ready` / `_backward` 记账保持不变 |
| `molzoo/pinet/potential.py` `_pipeline_ef_grad` | `return grad_force_pass(self._write_energy, batch, create_graph=bool(self.training), detach_energy=False)`。注：旧代码的 `retain_graph=create_graph` 恰是 PyTorch 在 `retain_graph=None` 时的默认取值，故删除该显式实参不改变数值 |
| `molzoo/pinet/potential.py` `_pipeline_ef_func` | `return func_force_pass(self._write_energy, batch)`。`__init__` 中 `self._pipeline = self._pipeline_ef_*` 的 monomorphic 绑定形状不变 |

**已知的行为放宽（有意、需记录）**：grad 内核在能量前向外层加了 `with torch.enable_grad()`，且叶子创建改为 `owns_leaf` 条件式。在 grad 已启用且传入 `pos` 非叶子的常规场景（PiNet 全部现有调用、`GradMode` 全部现有调用）两者均为恒等变换，逐位数值不变；仅在 `torch.no_grad()` 下把"抛异常"放宽为"正确返回力"，以及在调用方已提供活叶子时复用该叶子而非新建。二者都需要专门的单元测试钉住。

### 放置理由与依赖方向

`molpot.derivation` 是 CLAUDE.md "Key Design Patterns" 明确指定的 `F = -dE/dpos` 唯一归属（"do not hand-roll a third force path inside encoders"）。`kernels.py` 只放 pass 体；`modes/` 与各 potential 负责绑定。`kernels.py` 不从 `molzoo` / `molix.md` 导入任何东西（仅 `torch` / `tensordict` / 同包 `force.py` / `protocol.py`），单向依赖 `molix ← molrep ← molzoo / molpot` 保持成立。

### Reuse decision

- `reuse molpot.derivation.force.autograd_forces_from_energy` — grad 内核的求导尾段直接调用，不重写 `torch.autograd.grad`。
- `reuse molpot.derivation.force.functorch_forces_with_aux` — func 内核的单变换求导直接调用（该函数已返回 `-grad`，回写时不得再取负）。
- `reuse molpot.derivation.protocol.{call_energy, absorb_model_output, has_energy, write_forces, ENERGY_KEY, POS_KEY}` — 键映射与模型返回值吸收语义原样复用，内核内不得内联等价逻辑。
- `pattern molpot.derivation.force` — 最近模式：模块级自由函数 + 模块 docstring 划定"哪个后端、何时用"、Google 风格 docstring 带张量形状。`kernels.py` 沿用同一文件形状与命名族（`*_force_pass` ↔ `force.py` 的 `*_forces`）。
- `new molpot.derivation.kernels.grad_force_pass / func_force_pass` — `force.py` 提供的是**张量级** API（`energy_fn(pos) -> scalar`），本 spec 需要的是**批级** pass（`energy_core(batch) -> batch`，含 in-place 键写入、位置叶子所有权、`detach_energy` 策略）。`force.py` 中无对应符号，故新增；新符号**在 `force.py` 之上组合**，不重新推导梯度。
- `new — 不扩展 ForceDerivation` — `ForceDerivation` 是 `nn.Module` 后端分发器；把批级 pass 挂进去会让它同时承担"选后端"与"写 batch 键"两个职责（违反单一职责），且 `nn.Module.__call__` 的 hook 层会给 PiNet 的 monomorphic fullgraph 路径增加不必要的 trace 表面。`ForceDerivation` 在本 spec 中**不改动**。

## Files to create or modify

- `src/molpot/derivation/kernels.py` (new) — 两个共享 pass 内核 + 模块 docstring（边界、单位、编译约束）
- `src/molpot/derivation/__init__.py` — 导出 `grad_force_pass` / `func_force_pass`，更新 `__all__` 与模块 docstring 的调用形状说明
- `src/molpot/derivation/modes/grad.py` — `GradMode.run_forces` 重绑定（session 检查与错误文案保留）
- `src/molpot/derivation/modes/func.py` — `FuncMode.run_forces` 重绑定（lazy/flush 记账保留）
- `src/molzoo/pinet/potential.py` — `_pipeline_ef_grad` / `_pipeline_ef_func` 重绑定，删除本地拷贝
- `docs/molpot/user-guide/gradients.md` — 新增"批级力学传递内核"小节，说明张量级（`force.py`）与批级（`kernels.py`）的分工及 `detach_energy` 三态
- `tests/test_molpot/test_derivation/test_kernels.py` (new) — 内核单元测试（镜像 `src/molpot/derivation/kernels.py`）
- `regressions/mace-subpackage-restructure-03-kernels.py` (new) — PiNet 公共 API 逐位基准脚本，硬编码 golden

## Tasks

- [ ] Capture PiNet fixed-seed energy/force goldens from pre-rebinding HEAD (float64, `torch.manual_seed(0)`, 4 atoms / 8 edges, `compute_forces=True` × `method="func"` 与 `"grad"`) as `repr()` float literals for `regressions/mace-subpackage-restructure-03-kernels.py`
- [ ] Write failing unit tests for the shared force kernels (`tests/test_molpot/test_derivation/test_kernels.py` → `TestGradForcePass` / `TestFuncForcePass`)
- [ ] Implement `grad_force_pass` and `func_force_pass` in `src/molpot/derivation/kernels.py` composing `autograd_forces_from_energy` / `functorch_forces_with_aux`, and export both from `src/molpot/derivation/__init__.py`
- [ ] Rebind `GradMode.run_forces` in `src/molpot/derivation/modes/grad.py` and `FuncMode.run_forces` in `src/molpot/derivation/modes/func.py` onto the kernels, preserving session bookkeeping and existing `RuntimeError` messages
- [ ] Rebind `PiNetPotential._pipeline_ef_grad` / `_pipeline_ef_func` in `src/molzoo/pinet/potential.py` onto the kernels with `detach_energy=False`, deleting the local copies
- [ ] Add Google-style docstrings with tensor shapes and units to `src/molpot/derivation/kernels.py` and a "批级力学传递内核" section to `docs/molpot/user-guide/gradients.md`
- [ ] Add regression example `regressions/mace-subpackage-restructure-03-kernels.py` (public API only; hard-coded goldens from task 1, no third-party runtime)
- [ ] Verify the PiNet compile contract: `tests/test_molix/test_md/test_compile.py::test_pinet_step_fullgraph_compiles_and_matches_eager` and `::test_pinet_rollout_fullgraph_nve` pass with `fullgraph=True` and zero graph breaks
- [ ] Run full check + test suite

## Testing strategy

单元测试位于 `tests/test_molpot/test_derivation/test_kernels.py`（镜像 `src/molpot/derivation/kernels.py`），每个测试只针对一个函数，类名 `TestGradForcePass` / `TestFuncForcePass`。夹具沿用 `test_readouts.py` 里的玩具势 `E = s · Σ‖r‖²`（解析力 `F = -2s·r`，并带 `n_forward` 计数器），放入本目录 `conftest.py`。

`TestGradForcePass`：
- happy path — 4 原子 / 2 图，`s = 1.0`，`pos = arange(12).reshape(4,3) * 0.1`，力逐元素等于硬编码的 `-0.2 * arange(12)`（eV/Å），`atol=1e-12`（float64）。
- `energy_core=None` 分支 — 传入已在活叶子上物化能量的 batch，断言模型 `n_forward == 0` 且力正确（钉住 `GradMode` 的单前向合同）。
- `energy_core=None` + 非叶子位置 — 抛 `RuntimeError`。
- 缺 `graphs.energy` 的能量核 — 抛 `RuntimeError`，消息含 `graphs.energy`。
- `detach_energy` 三态 — `False` → `batch[ENERGY_KEY].requires_grad is True`；`True` → `False`；`None` + 非叶子入参（内核建叶子）→ 能量被 detach；`None` + 调用方已给活叶子 → 能量保持 attached 且 `batch[POS_KEY] is` 传入的那个张量（未新建叶子）。
- `create_graph=True` → 力损失 `.backward()` 后玩具势参数 `scale.grad` 非零；`create_graph=False` → 力已断开（`requires_grad is False`）。
- `torch.no_grad()` 下仍返回正确的力（放宽行为的钉子）。

`TestFuncForcePass`：
- happy path — 同一玩具势与同一硬编码期望值，与 grad 内核结果 `atol=1e-12` 一致（float64）。
- 单前向 — `n_forward == 1`。
- `atoms.energy` 存在时被回写、不存在时不报错。
- 力损失单次 backward 触达参数。

领域校验（`$META.science.required`）：float64 下用中心差分 `h = 1e-5 Å` 对玩具势与一个小型 `PiNetPotential` 求 `-dE/dr`，与两个内核的输出比对，`max|ΔF| < 1e-6 eV/Å`（硬编码阈值，非相对比较）。

回归保持绿色（不新增、不修改）：`tests/test_molpot/test_derivation/test_force.py`、`test_readouts.py`、`test_energy.py`、`tests/test_molzoo/test_pinet/test_potential.py`。

编译合同：`tests/test_molix/test_md/test_compile.py` 的两个 PiNet 用例（`backend="aot_eager"`, `fullgraph=True`，float32 与 float64）必须在重绑定后继续通过——`fullgraph=True` 本身即 0-graph-break 断言。

**Regression example**：`regressions/mace-subpackage-restructure-03-kernels.py`（仓库根 `regressions/`，**不在 `tests/` 下**）。只用公共 API：构造 `PiNetPotential(..., compute_forces=True, method="func")` 与 `method="grad"` 各一次（`torch.manual_seed(0)`、float64、固定 4 原子 / 8 边 batch），调用 `model(batch)`，把 `graphs.energy` (eV) 与 `atoms.forces` (eV/Å) 与文件内**硬编码**的 golden 字面量比对，`atol=1e-12, rtol=0`；通过打印 `OK` 并 `exit 0`，否则打印最大偏差并非零退出。Golden 由重绑定前的 HEAD（`82c3091`）生成，脚本顶部注释记录：生成命令、commit sha、torch 版本、设备（CPU）、日期。运行期不导入任何第三方 oracle（无 ASE / e3nn / mace-torch）。

## Out of scope

- `MACEPotential` 本体及其对内核的绑定 —— 属链条 04-potential；本 spec 只保证 `detach_energy` 旋钮能表达 `mace_matpes.py:368-377` 的语义，不改 `mace_matpes.py`。
- 修改 `molpot/derivation/force.py` 的张量级 API 或 `ForceDerivation` 的分发行为。
- 应力 / virial 的共享 pass（`stress.py` 不动）。
- 给 func 内核加 `detach_energy` 旋钮（无调用方；cuEq 融合核不走 functorch）。
- 移除 `protocol._SESSIONS` 侧信道或重构 `EnergyReadout` / `ForceReadout` 的顺序协议。
- 任何性能优化 —— 本 spec 是零行为变更重构，`benchmarks/` 不变。
- 新增 `molzoo` 公共符号（链条不变量）；`edge_index` 约定与 post-collate batch schema 均不触碰。
- （已考虑并否决的替代方案：把两个 pass 做成 `ForceKernel` 类挂到 `ForceDerivation` 上 —— 见 Design 的 Reuse decision，会破坏单一职责并给 fullgraph 路径加 `nn.Module` trace 表面。）
