---
title: 力训练路径的编译策略 — 开发者声明边界 / CUDA Graph 闸门 / functorch fullgraph
slug: trainer-hardening-01-compile-strategy
status: draft
created: 2026-06-12
---

# 力训练路径的编译策略

## Summary
让 `torch.compile` 在**力训练**路径上真正可用，分三个独立成熟度的阶段。核心原则（用户裁定）：**编译边界由模型开发者在定义处声明，不是给使用者的 API、也不是 molix 框架基建**。使用者只该 `torch.compile(model)`，边界由开发者挂在模块上的装饰器自动生效。
- **Phase 1（落地）**：统一"开发者声明 eager 边界"约定 —— 所有做二阶反向（`create_graph=self.training`）的模块都用 `@torch.compiler.disable` 围栏隔离；审计并补齐当前漏挂的几处；删除 PiNet 的 user-facing `compile_energy()`，让 `torch.compile(model)` 对 energy-only / 推理 / force 推理 默认安全。
- **Phase 2（仅文档，profile-gated）**：CUDA Graph 三道硬闸门，只在 profiler trace 证明 launch-bound 时才追加实现。
- **Phase 3（spike-gated）**：functorch fullgraph 路径 —— 用 `torch.func.grad` 把力计算物化进前向图，消除"作为编译障碍"的 double backward，使整个 energy→force→loss 成为单次 backward 可 fullgraph 编译。JAX 系 MLIP 天生无此问题的根因级对标，但侵入性最高，由两道门禁 + 证伪式 spike 决定是否开工。

## Domain basis
不是新物理实现，但编译策略必须保持 `F = −∂E/∂x` 的数值不变性（数值验证归 `trainer-hardening-03`）。硬约束来自 PyTorch 语义：
- `aot_autograd` / `make_graphed_callables` 不支持 grad-of-grad（二阶反向）。训练期 `create_graph=True` 的力路径无法被 inductor/cudagraph 捕获 —— 这是 Phase-1 eager 围栏的根因（`molpot/derivation/force.py:32` docstring 已记录）。
- `torch.func.grad` 是 functional autodiff transform：被 trace 成显式前向算子，于是"对一阶导再求导"变成"对增广前向图的一次常规 reverse"，aot_autograd 只需单次 backward 即可整体编译。**注意：消除的是编译障碍，不是二阶计算 —— 混合二阶导的 FLOPs/显存仍在。**收益是 fullgraph 融合 + 解锁 cudagraph。
- 参考：JAX-MD / e3nn-jax / MACE-jax 用 `jax.grad` 函数式组合，全程一个 traced 计算交给 XLA，天然没有 double-backward 编译墙。

## Design

### Phase 1 — 开发者声明的 eager 边界（落地）
现状（已核 2026-06-12）：只有 `src/molpot/derivation/force.py:28` 的 `ForceDerivation.forward` 挂了 `@torch.compiler.disable`。而 `StressDerivation.forward`（stress.py:30）、`PotentialComposer.forward`（composer.py:38）、`Sonata.forward`（sonata.py:325）都在 forward 内**内联** `autograd.grad(..., create_graph=self.training)`，**无 fence**，且因内联无法整体 disable（会把整个 potential 排除出编译）。

- **统一围栏模式**：把内联的二阶反向 grad 调用收进可被 `@torch.compiler.disable` 的子模块（复用 `ForceDerivation` / 新增对应 disabled helper），让 stress/composer/sonata 经由它求力/应力，而不是 forward 里裸调 `autograd.grad`。
- **plain `torch.compile(model)` 安全化**：力模型在 `training and compute_forces` 时，energy 区域也必须落在编译之外（二阶反向会穿过它）。模型在构造期/forward 入口判断该条件并让 energy 区走 eager，使用者无脑 `torch.compile(model)` 永不报 double-backward。
- **删除 user-facing `compile_energy()`**（pinet.py:440）：它把开发者的编译边界泄漏成了使用者配置，是被否决的 API 形态。同时不引入 `molix.compile.CompilePolicy` 基建。`molix/compile.py:maybe_compile` 保留为薄封装；`count_graph_breaks`（compile.py:59）保留作诊断。
- 用 `count_graph_breaks` 断言 energy 子图无 graph break（CUDA Graph 与 Phase-3 的前置）。

### Phase 2 — CUDA Graph 三闸门（仅文档，profile-gated）
仅文档 + `cuda_graph_preconditions()` 返回未满足闸门清单；**不实现捕获**，且仅当 profiler trace 显示 GPU launch-bound 空隙时才追加：
- gate-1 动态 `edge_index` shape → 需 padding 到固定 `max_E` + masked dummy edges（revMD17 每分子 N 固定，仅 E 变）。
- gate-2 double backward → 需手工 `torch.cuda.graph()` 静态训练步 + capturable optimizer（`Adam(capturable=True)`）；`make_graphed_callables` 不支持 grad-of-grad。**Phase-3 落地后此闸门消失。**
- gate-3 AMP `GradScaler` 的 inf/nan 检查是 host-side 同步（`src/molix/core/steps/train.py:96,102` 的 `scaler.get_scale()` 比较），破坏捕获 → 须丢 AMP 或用 capture-safe scaler。

### Phase 3 — functorch fullgraph 路径（spike-gated）
目标：`force = -torch.func.grad(lambda pos: energy_fn(pos, params).sum())(pos)`，使 energy→force→force_loss 成为单一前向函数，对 params 只需一次常规 backward；整体 `torch.compile(fullgraph=True)`。

两道决定成败的门禁（按"最便宜证伪"排序）：
- **门 1（最致命）cuEquivariance 的 functorch 覆盖度**：cuEq 遍布 MACE/Allegro/molrep（contraction/product/linear/angular/node 全用）。`torch.func.grad` 要穿过这些自定义 op，要求它们由可组合 ATen op 构成或注册了 functorch（vmap/grad）规则。若 cuEq op 不透明且无规则 → eager 下即挂，方案 dead on arrival。**~1h 可证伪，第一个做。**
- **门 2（最贵）functional 纯度 vs dict-mutate 架构**：encoder 契约是原地写 batch TensorDict（`mace.py:543` `td["atoms","node_features"] = ...`，CLAUDE.md 明文）。functorch 要求纯函数 + 参数经 `functional_call` 穿入 + 无容器突变。只读 buffer（freqs/mu/sigma/r_cut，`register_buffer(persistent=False)`）无害；横切所有 encoder 的 in-place 写 batch 是真正的重构代价，且与 dict-first 核心数据流冲突。MACE 纯 TP 结构最适合，但 molzoo MACE 仍原地写 node_features。
- **门 3 fullgraph HOP 覆盖**：`fullgraph=True` 严格，不支持的算子是硬报错。target torch≥2.10（CLAUDE.md）覆盖最好但需实测；复用 `count_graph_breaks` 审计。

定位：严格更强（解锁 force 训练的 fullgraph + cudagraph）但严格更侵入；与 Phase-1（回避）互补，是真正的"解决"。仍需 padding 才能上 cudagraph（gate-1 不因 Phase-3 消失）。

## Files to create or modify
- `src/molpot/derivation/force.py` — 只读校验 fence 仍在；docstring 交叉引用本策略（可选）。
- `src/molpot/derivation/stress.py`、`src/molpot/composition/composer.py`、`src/molpot/composition/sonata.py` — Phase-1：把内联二阶反向 grad 收进 `@torch.compiler.disable` 子模块。
- `src/molzoo/pinet.py` — Phase-1：删除 `compile_energy()`；构造期接好 force-training 时 energy 走 eager 的守卫。
- `src/molix/compile.py` — 保留 `maybe_compile` / `count_graph_breaks`；新增 `cuda_graph_preconditions()`（Phase-2 文档闸门检查）。
- `docs/compile_strategy.md` (new) — 三阶段、三闸门、Phase-3 门禁与 spike 计划成文。
- `tests/test_molix/test_core/test_compile.py` — Phase-1：force 训练 plain compile 安全、graph-break 闸门。
- `tests/test_molpot/test_compile_fences.py` (new) — 断言所有二阶反向模块都被 eager 围栏隔离。

## Tasks
- [ ] Audit @torch.compiler.disable coverage across all create_graph=self.training sites (force/stress/composer/sonata)
- [ ] Factor inline double-backward autograd.grad in stress/composer/sonata into a compiler-disabled submodule
- [ ] Make plain torch.compile(model) safe for force training by routing the energy region eager when training and compute_forces
- [ ] Remove the user-facing compile_energy() from PiNet and update its benchmark call sites
- [ ] Add cuda_graph_preconditions() gate-check and write docs/compile_strategy.md (3 phases, 3 gates)
- [ ] Write test_compile_fences.py asserting every double-backward module is fenced
- [ ] Phase-3 spike (separate, non-product): eager torch.func.grad through one MACE forward incl. cuEq; then fullgraph trace via count_graph_breaks; record coverage verdict
- [ ] Run full check + test suite

## Testing strategy
- Fence 覆盖：参数化测试，对每个做 `create_graph=self.training` 的模块断言其求导子区被 `@torch.compiler.disable` 隔离。
- 力不变性：Phase-1 改造后 force 与未编译 baseline `allclose`（严谨数值版归 spec-03）。
- plain compile 安全：`torch.compile(force_model); model.train(); loss.backward()` 不抛 double-backward RuntimeError（energy 区走 eager）。
- Graph-break 闸门：energy 子模块 `count_graph_breaks == 0`，否则 `cuda_graph_preconditions` 含 gate-1。
- Phase-3 spike 是探针不是回归：产出"cuEq 能否被 functorch 穿透 / fullgraph 是否无断点"的书面结论，不进 CI。

## Out of scope
- 实际手工捕获的 cudagraph 训练步、edge_index padding、capturable optimizer（Phase-2 实现，profiler-gated）。
- functorch fullgraph 的产品化实现与 encoder functional 重构（Phase-3，spike 全绿后另立 spec）。
- 数值一致性矩阵（spec-03）。
- 异步日志（spec-02，已实现并合入 dev）。
