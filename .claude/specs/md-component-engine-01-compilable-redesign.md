---
title: MD component engine — compilable, component-based redesign of molix.md
status: code-complete
created: 2026-06-26
chain: md-component-engine
---

# MD component engine — compilable, component-based redesign of `molix.md`

## Summary

把 `molix.md` 从"闭包 + 动态分发"的研究脚本，重构为一套**基于组件、静态类型化、整步可 `torch.compile` 全图编译**的 MD 引擎。三条硬约束(来自 review 后的用户指令):

1. **基于组件,删除函数引用** —— 注入的 `force_fn: Callable` 闭包、`build_force_fn` 闭包工厂、测试里的 `_harmonic(k)` 闭包、`MDRunner._call_hooks` 的 `getattr` 动态分发,全部换成 `nn.Module` 组件 / 类型化直调。
2. **静态、类型化、可编译** —— 去掉热路径里的所有运行期判断(`isinstance(mass,…)`、`if gamma>0`、`noise=None` 分支、`getattr`);`Integrator.step` 与 `Integrator.rollout`(含 PiNet 力)必须 `torch.compile(fullgraph=True)` **零 graph-break**,fp32/fp64 与 eager 数值一致。
3. **参考 molpy 区分 `Potential` 与 `ForceField`**(两者都是 `nn.Module`,是不同概念,见 Domain basis)。

并在 `TensorDict` 与 typed state 之间取平衡:**拓扑/模型 I/O 用 `TensorDict`(模型母语),积分器边界跨越的动力学状态用 typed `MDState` NamedTuple(纯张量,pytree,编译友好)**。

本次重构必须**保留上一轮 review 修掉的全部正确性**:活体几何(冻结 PES bug)、ΔF fp64、Langevin dof=3N、有界显存(CPU 流式 + shard-flush)、单位桥接、正质量守卫。

## Domain basis

**molpy 的 `ForceField` vs `Potential`**(`/home/jicli594/work/molcrafts/molpy`,调研确认):

- `ForceField` 定义 styles/types/parameters(符号、可变、**不含求值核**),通过 `ff.to_potentials(frame)` **绑定到一个具体体系**,产出可求值的 `Potentials`。
- `Potentials`(=分子力学里的求值集合)是 frame-bound、不可变、可求值:`calc_energy(coords)` / `calc_forces(coords)`;求值接口抽象为 `PotentialLike` Protocol(`optimize/base.py`)。

映射到 molnex MD(ML 势场语境):

| molpy | molnex MD(本 spec) | 职责 |
|---|---|---|
| `Potential`(功能形式/模型) | **`Potential(nn.Module)`** = `molpot.BasePotential` / `PiNetPotential`(已存在,**复用不重写**) | 从 batch `TensorDict` 算能量;力经 `ForceDerivation(method="functorch")` 求 |
| `Potentials`(frame-bound 可求值) | **`ForceField(nn.Module)`**(新增) | 把一个 `Potential` **绑定到体系模板**,对 `pos` 暴露 `forward(pos)->ForceOutput` + `calc_energy/calc_forces` | 
| `PotentialLike` Protocol | `ForceField` 的 `calc_energy/calc_forces` | 积分器消费的统一求值接口 |
| `Frame`/`Block`(列式状态) | `MDState` NamedTuple(typed) + `ForceField` 内部持有的 `TensorDict` 模板 | tensordict↔typed 平衡点 |

**结论**:`Potential` 是"功能形式(能量模型)",`ForceField` 是"绑定体系后的能量+力求值组件"。积分器**只认 `ForceField`**,不认 `Potential`、不认闭包。

**可编译性事实(调研确认,决定本设计可行)**:`PiNetPotential` 用 `ForceDerivation(method="functorch")` → `torch.func.grad` **traced into the forward graph**,`energy→force` 单次 backward,`torch.compile(fullgraph=True)` **0 graph breaks**(证据:`tests/test_molzoo/test_pinet_functorch_force.py`、`docs/molix/explanation/throughput-and-compilation.md:21-31,94-99`,GH200 ~130 steps/s);fp32/fp64 均正确(`benchmarks/verify_pinet_cudagraph_ef.py`);力路径**不改输入 batch**(全在内部 clone,`pinet.py:443-469`)→ 跨步复用持久模板安全。编译入口 `molix.compile.maybe_compile(cuda_graphs=True)` = `CUDA_GRAPH_PRESET{fullgraph,dynamic=False,reduce-overhead}`,要求静态 shape(冻结邻居表天然满足)。

**积分器物理(不变)**:BAOAB(Leimkuhler & Matthews 2013, DOI 10.1093/amrx/abs010);O 步 `v ← c1·v + c2·σ·ξ`,`c1=e^{-γΔt}`,`c2=√(1-c1²)`,`σ=√(k_BT/m)`;γ=0 ⇒ c1=1,c2=0 ⇒ O 步恒等 ⇒ 退化为 velocity-Verlet(**这正是"always-on O 步去分支"成立的依据**:0·σ·ξ=0,数值不变)。

## Design

### 组件清单(全部 `nn.Module` 或 typed pytree)

**`molix/md/types.py`(新)** —— 共享 typed 容器,避免循环依赖:
- `class ForceOutput(NamedTuple): energy: Tensor; forces: Tensor` —— 取代 `(E,F)` 裸元组与 `ForceFn` Callable 别名。
- `class MDState(NamedTuple): pos: Tensor; vel: Tensor; force: Tensor` —— 积分器跨步的动力学状态;NamedTuple 是 pytree,`torch.compile` 原生支持。

**`molix/md/forcefield.py`(新,取代 `force_seam.py`)** —— `ForceField` 组件层:
- `class ForceField(nn.Module)`[抽象]:`forward(self, pos: Tensor) -> ForceOutput`;便捷 `calc_energy(pos)->Tensor` / `calc_forces(pos)->Tensor`(镜像 molpy `PotentialLike`)。
- `class PotentialForceField(ForceField)`:绑定一个 `Potential`(molpot 模型)+ 体系模板。`__init__` 里 `clone()` 模板一次、`del` 掉 `edges.bond_diff/bond_dist`(活体几何,**保留冻结-PES 修复**)、把 `pos` 之外的拓扑作为成员持有;`energy_scale: float` buffer 做单位桥接;`forward` 写入 `pos`(`.to(device,dtype)`)→ `model(batch, compute_forces=True)` → `ForceOutput(energy*scale, forces*scale)`。**无闭包、无每步 clone(模型内部自 clone)**。
- `class AnalyticForceField(ForceField)` 及具体 `HarmonicForceField(ForceField)`(`E=½k‖x‖²`, `F=-kx`)—— 取代测试与参考用的 `_harmonic(k)` 闭包,使积分器单测不依赖 PiNet 且走同一组件接口。
- `class LennardJonesForceField(ForceField)`(全对 LJ,`E=4ε[(σ/r)¹²−(σ/r)⁶]`,无 cutoff/无邻居表,**全对天然规避冻结邻居表问题**)—— 用于 NVE 长程能量守恒验收(ac-008):小 LJ 团簇(N~13–55,平衡构型附近)是各向异性、刚硬、非谐的真实 PES,比谐振子玩具强得多的积分器稳定性检验。力解析可微 → `rollout` 全图编译。

**`molix/md/integrators.py`(重构)** —— `Integrator` 组件层:
- `class Integrator(nn.Module)`[抽象]:持有 `self.force: ForceField`(组件,非 Callable);`initial(self, pos, vel) -> MDState`(seed 力);`step(self, state: MDState, noise: Tensor) -> MDState`;`rollout(self, state: MDState, n_steps: int) -> MDState`。
- `class LangevinVerletIntegrator(Integrator)`:`register_buffer` 存 `dt/gamma/kbt/c1/c2/mass_col/inv_mass/sigma`(`mass` 在 `__init__` 一律归一化为正张量 buffer,**去 isinstance**,**正质量守卫**);`step` 走 BAOAB,**O 步恒等式始终执行**(去 `if gamma>0`),`noise` 永远是真实张量(去 `None` 分支);`dof` 约定 γ>0→3N、γ=0→3N-3(**保留 dof 修复**,作为温度估计辅助,放在 runner)。
- **编译**:`step`/`rollout` 对 traceable `ForceField`(含 PiNet functorch)`torch.compile(fullgraph=True)` 零 break。`rollout` 内部逐步 `torch.randn`(dynamo functionalize RNG;`manual_seed` 复现)→ 满足"`torch.compile(md.run)` 全流程编译"的无 hook 快路径。生产路径见 runner。

**`molix/md/runner.py`(重构)** —— 驱动 + 观测:
- `MDRunner`:持有 `Integrator` 组件;`run(state0|pos,vel, n_steps)` 用 **eager Python 循环 + 已编译 `step`**(每步 eager 抽 `noise` 传入已编译 step,hook 在此触发——hook 有副作用不能进编译图);可选 `compile=True` 经 `maybe_compile` 包 `step`。**hook 分发改类型化直调**(`BaseHook` 已有 no-op 默认实现,`hook.on_train_batch_end(...)` 直接调,**删 `getattr`**)。物理量仍只走 `outputs` 通道(保 state 命名空间契约)。
- `TrajectoryHook`:消费 `MDState`/`ForceOutput`;**保留 shard-flush 有界显存 + 诚实同步拷贝**;`weights_only=True` 重载 shard。
- 共享质量列工具 `as_mass_col` 仍由 integrators 提供(去 runner 重复)。

**`molix/md/dynamics.py`(重构)** —— study 层迁到组件:`run_trajectory`/`evaluate_delta_along_trajectory`/`build_paired_trajectory` 改用 `PotentialForceField` 对(ref/quant),不再调 `build_force_fn` 闭包;**保留 ΔF fp64 与 dof 元数据**。`TrajectoryArtifact` 不变。
**`molix/md/ase_shim.py`(重构)** —— 基于 `ForceField` 组件而非闭包。
**`molix/md/__init__.py`** —— 导出组件词表:`ForceField/PotentialForceField/HarmonicForceField`、`ForceOutput/MDState`、`Integrator/LangevinVerletIntegrator`、`MDRunner/TrajectoryHook`、study 层符号。

### tensordict ↔ typed 平衡(指令 5)

- **`TensorDict`** 留在 `ForceField` 内部(模型 I/O 的异构拓扑:`atoms/edges/graphs`),不外泄到积分器。
- **`MDState`/`ForceOutput`**(typed NamedTuple,纯张量)是积分器/runner/hook 之间的**唯一**数据契约。NamedTuple 是 pytree → `torch.compile`/`vmap` 友好,且静态可标注。
- 边界函数:`ForceField.forward(pos: Tensor) -> ForceOutput`。一边是 TensorDict,一边是 typed tensor,转换点唯一且类型化。

## Files

- `src/molix/md/types.py` —— 新增 `ForceOutput`、`MDState`。
- `src/molix/md/forcefield.py` —— 新增,取代 `force_seam.py`;`ForceField`/`PotentialForceField`/`AnalyticForceField`/`HarmonicForceField`。
- `src/molix/md/force_seam.py` —— 删除(`build_force_fn` 迁为 `PotentialForceField`)。
- `src/molix/md/integrators.py` —— `Integrator` 基类 + `LangevinVerletIntegrator(nn.Module)`,`MDState` 步进,去分支/去 isinstance,`rollout` 可编译。
- `src/molix/md/runner.py` —— `MDRunner` 类型化 hook 直调 + 编译 step 循环;`TrajectoryHook` 适配 `MDState`。
- `src/molix/md/dynamics.py` —— study 层迁到 `PotentialForceField`。
- `src/molix/md/ase_shim.py` —— 基于 `ForceField`。
- `src/molix/md/__init__.py` —— 组件词表导出。
- `tests/test_molix/test_md_*.py` —— 迁到组件 + 新增编译/类型测试。
- `benchmarks/verify_md_lj_nve.py` —— 新增,LJ 团簇 NVE 5ns 能量守恒长程验证(ac-008)。
- `CLAUDE.md` —— 更新 `molix.md` 依赖图注记(组件化 + 可编译)。

## Tasks

1. `types.py`:`ForceOutput`、`MDState`(+ pytree 注册校验 round-trip 测试)。
2. `forcefield.py`:`ForceField` 抽象 + `PotentialForceField`(含活体几何 `del bond_diff`、`energy_scale`、持久模板)+ `HarmonicForceField`。RED:迁移 `test_force_seam_tracks_live_geometry`(活体几何回归)。
3. `integrators.py`:`Integrator`/`LangevinVerletIntegrator(nn.Module)`,`MDState` 步进,always-on O 步,mass buffer + 正质量守卫,`as_mass_col`。RED:NVE 守恒、Langevin 等分、force-caching、step==step_cached、compile==eager 全部迁移并通过。
4. `runner.py`:`MDRunner` 类型化 hook 直调 + 编译 step;`TrajectoryHook` 适配 `MDState`,保留 shard-flush/dof/单位。RED:迁移 runner 生命周期 + shard-flush 测试。
5. `dynamics.py`/`ase_shim.py`/`__init__.py`:study 层与 ASE 壳迁到 `ForceField`;保留 ΔF fp64、dof 元数据;迁移 paired/energy-varies 测试。
6. **编译验收**:新增 `test_md_compile.py`:`torch.compile(ig.rollout, fullgraph=True)` 对真实 tiny-PiNet 跑 N 步 0 graph-break 且 ≈ eager(fp32 与 fp64);`torch.compile(ig.step)` 同理。
7. **LJ NVE 5ns 能量守恒验收**:`LennardJonesForceField` + `benchmarks/verify_md_lj_nve.py`,LJ 团簇 γ=0 经编译 rollout 跑满 5 ns,验证总能量无系统漂移(ac-008)。5ns≈数百万步,**只有靠编译/CUDA-graph 才可行**,本验收同时压测性能。
8. **静态化验收**:热路径无 `getattr`/`isinstance`/`None`-分支(代码审查 + 可选 `ty`/`mypy` 门);`ruff check`+`ruff format` 干净。
9. 更新 `CLAUDE.md` + specs `INDEX.md`。

## Testing

- **正确性回归(必须全绿,防止重构倒退)**:活体几何(energy/force 随构型变)、能量守恒(NVE)、等分(Langevin,T 无偏 dof)、force-caching 位级一致、shard-flush==单缓冲、ΔF fp64、正质量守卫 raise。
- **编译**:`fullgraph=True` 的 `rollout`/`step` 对 PiNet 0 break(用 `torch._dynamo` 计数或 `fullgraph` 抛错即失败);compile==eager 数值 `allclose`(fp32 atol 适配、fp64 1e-10);NVE+compile 与 Langevin+compile 均覆盖。
- **类型/静态**:`ForceFn` Callable 别名已删;`grep` 断言热路径无 `getattr(`/`isinstance(`/`is None`(测试或 CI 检查);NamedTuple pytree round-trip。
- **LJ NVE 5ns 长程守恒**(ac-008):`benchmarks/verify_md_lj_nve.py` —— LJ 团簇(记录 N/σ/ε/m/dt/T0)γ=0 编译 rollout 跑满 5 ns(步数=5ns/dt);判据:E_tot(t) 线性拟合斜率 `|slope·5ns|/|E_tot(0)| < 1e-3`,RMS 涨落有界、无单调爬升,温度有限不爆。打印 drift/斜率 PASS/FAIL。属长程验证脚本(非快速单测)。
- 环境:无 editable 安装,`PYTHONPATH=src:. /nobackup/proj/disk/teoroo/personal/jicli594/work/.x86_64/bin/python -m pytest tests/test_molix/test_md_*.py`。

## Out of scope

- **PBC / 最小镜像 / 邻居表重建**:仍是开放体系、冻结邻居表、小位移研究引擎(在 `ForceField` 文档与 `__init__` 明示);活体 minimum-image 与 Verlet skin 另立 spec。
- **把 study 层(`TrajectoryArtifact`/`build_paired_trajectory`)迁出到 `pinet-quant/csmd`**:API review 提的跨包搬迁,独立处理,本 spec 只做组件化与类型化。
- **MACE**:走 `ForceDerivation(method="autograd")`(`allow_in_graph` 路径),其全图编译与 cuet 约束(pytorch#170834)是另一条线。
- **CUDA-graph/`reduce-overhead` 下的 RNG 确定性**:`rollout` 默认 in-graph `torch.randn`;CUDA-graph 路径的预抽噪声缓冲作为后续优化。
