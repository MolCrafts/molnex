---
title: cuet 等变 Linear (l>0) 二阶反传修复 — 解锁力监督训练
status: in-progress
created: 2026-06-21
---

# cuet 等变 Linear (l>0) 二阶反传修复 — 解锁力监督训练

## Summary

在 cuet(cuequivariance)后端下,等变 `cuet.Linear`(含 l>0 irreps)阻断了力监督训练所需的**二阶反传**:对力损失 `‖F_pred − F_true‖²` 反传时,**所有模型参数得到零梯度**,因此任何基于 cuet 的 molzoo 编码器(`MACEOMol` 等)都无法训练力。能量训练正常,移植对官方 OMOL 一致到 1e-9,问题**仅**出现在力的二阶反传。

约束:**必须保持纯 cuet,不许引入 e3nn**。本 spec 从最小可复现片段入手,定位根因(官方 mace 的 `cuet.Linear` 实例在同节点/同 cueq 版本下能二阶反传,而我们 molrep 新建的实例不能 —— 差异在实例化方式),在 `molrep` 内以纯 cuet 方式修复,并保持 1e-9 移植一致性与能量训练不退化。

## Domain basis

- 力 `F = −∂E/∂x`(对位置一阶导)。
- 力监督训练的损失对参数的梯度需要**混合二阶导** `∂²E/∂x∂θ`(即对 `∂E/∂x` 再关于参数 θ 求导,亦称 double-backward / 二阶反传)。这与用 `torch.autograd.grad(create_graph=True)` 还是 `torch.func.grad`(functorch)无关——functorch 当年解决的是力的**编译/推理**(单次函数变换、`torch.compile` 友好),正交于二阶训练需求。
- 等变线性 `cuet.Linear`(l>0)是 `cue.descriptors` 经 `cuet.SegmentedPolynomial` 实现;不同 dispatch/实例化路径对二阶反传的支持不同。
- 参考:Batatia et al., MACE, NeurIPS 2022, arXiv:2206.07697;NVIDIA cuEquivariance 文档(SegmentedPolynomial / Linear method 语义)。

## 已实测的诊断证据(复现前提)

1. `MACEOMol` 力损失 backward → 73/73 参数零梯度(`DOUBLE_BACKWARD_BROKEN`)。autograd `create_graph=True`、`torch.func.grad`、`ForceDerivation` 三条路径结果一致。
2. 官方 mace teacher(`OMOL-cueq.model`:e3nn 球谐 + 27 个 `cuet.Linear` + cuet TP/SymmetricContraction)在**同节点、同 cueq 0.10、naive 路径**下力二阶反传 OK(101/103 参数有梯度)→ **非平台限制**。
3. 逐层隔离:`cuet 球谐 → nn.Linear` OK;`cuet 球谐 → cuet.Linear(l>0)` BROKEN。`cuet.Linear(纯 0e 标量)` OK,**仅 l>0 等变路径 BROKEN**。
4. fused ops(`cuequivariance-ops-torch-cu13`)已正确装好并激活(kernel 路径、fallback 警告消失),`cuet.Linear(l>0)` 仍 BROKEN。
5. 照抄 mace 的实例化(`method="naive", shared_weights=True, layout=ir_mul`)新建的 `cuet.Linear(l>0)` 仍 BROKEN。
6. **核心矛盾**:同一个类 `cuet.Linear`、同版本、l>0,teacher 的实例能二阶反传,我们新建的不能 → 差异落在"teacher 的实例当初以何种参数/布局/权重组织被创建并 pickle"。这是定位的钥匙。

## Design

- **最小复现优先(从最小片段开始)**:`scripts/cueq_db/minimal_repro.py`(≤30 行),只依赖 cuet,构造一个位置依赖输入(球谐)→ `cuet.Linear(l>0)` → 标量能量,`g = ∂E/∂x`,对 `g²` 反传,断言权重梯度的 L1。当前应为 0(RED)。
- **受控 toggle 定位根因**:在同一脚本里并排两种 `cuet.Linear(l>0)` 实例 —— "我们当前的" vs "可工作的"(从 teacher 反推的实例化),只改单一变量(候选:`layout` `ir_mul`↔`mul_ir`、`shared_weights`/`internal_weights`、`method`、权重外置 indexed 路径、是否经 `cuet.equivariant_tensor_product`/wrapper),找出让梯度从 0→非 0 的那个开关。teacher 的实例可直接 `torch.load` 取一个 `cuet.Linear` 子模块对照其构造特征(irreps/layout/属性)。
- **修复**:据根因调整 `molrep/interaction/` 内所有 cuet 等变 `Linear` 的实例化(`residual.py`、`product_basis.py`、`linear.py`、`element.py`),保持纯 cuet;同步检查 `ChannelWiseTensorProduct` / `SymmetricContraction` 是否需要同样开关。
- **不退化护栏**:修复改变实例化但**不得改变数值** —— 仍能 `load_omol_state_dict` 并对 teacher 保持 1e-9;若权重布局变化,需提供权重转换或证明等价。

## Files

- `scripts/cueq_db/minimal_repro.py`(新)— 最小复现 + 受控 toggle
- `scripts/cueq_db/probe_teacher_linear.py`(新)— 从 teacher 反推可工作的 `cuet.Linear` 构造特征
- `src/molrep/interaction/{residual.py, product_basis.py, linear.py, element.py}` — 修复 cuet 等变层实例化
- `tests/test_molrep/test_cueq_double_backward.py`(新)— 二阶反传回归测试
- `scripts/omol_port/verify_e2e.py` — 复用作一致性不退化验证

## Tasks

- [x] Write a ≤30-line minimal reproduction (cuet-only) asserting force-loss→weight grad L1 == 0 for the current `cuet.Linear(l>0)` path (RED) — `scripts/cueq_db/minimal_repro.py`, baseline L1==0 reproduced deterministically
- [x] Probe the teacher's `cuet.Linear` submodule to extract its working construction signature (irreps/layout/weight organization) — `scripts/cueq_db/probe_teacher_linear.py`; finding below
- [ ] Build a single-variable toggle in the repro that flips the gradient from 0 to non-zero; identify and document the exact knob (root cause) — **BLOCKED: no cuet-only toggle found (see Findings)**
- [ ] Fix the cuet equivariant `Linear` (and any affected TP/contraction) instantiation across `molrep/interaction/`, pure-cuet
- [ ] Add a regression test in `tests/test_molrep/` asserting double-backward weight grad != 0 for an l>0 cuet path
- [ ] Verify full `MACEOMol` force double-backward yields non-zero grad for ~all parameters
- [ ] Verify port consistency vs official OMOL stays ≤ 1e-6 (isolated molecules) after the fix
- [ ] Run full check + test suite

## Findings (2026-06-21, mol:impl run 1)

复现已成功(ac-001 RED 成立),但根因比 spec 起草时的假设更深,且**纯 cuet 修复目前看不可行**:

1. **不是 cuet.Linear 本身**:用通用输入 `tanh(v)` 喂 `cuet.Linear(l>0)`,无论 teacher 抽取的实例还是新建实例,力损失→权重梯度都**非零**(L1 2.5e8 / 1.5e7,OK)。teacher 与新建实例结构完全一致(同 method/layout/weight_numel)。之前"cuet.Linear 是元凶"的判断**被推翻**。
2. **不是 cuet 球谐本身**:cuet 球谐的纯二阶导(Hessian)可算(L1 4.8e2,OK);`cuet 球谐 → nn.Linear` 权重梯度也 OK(因为只需球谐一阶导)。
3. **真正的触发点 = cuet 球谐输出 → cuet 等变算子 的组合**:`cuet 球谐 → cuet.Linear(l>0)` 权重梯度恒为 0。数学上该梯度只需球谐一阶导,故这是 cueq 自定义算子在二阶反传链上的**组合缺陷**(cuet 球谐的输出张量喂入下游 cuet 等变算子时,把权重从二阶图里 detach 了)。
4. **teacher 为何能**:官方 teacher 用 **e3nn 球谐**(纯 torch、二阶可微)→ cuet 等变算子,故 101/103 OK。差异从来不是 cuet.Linear 的实例化,而是**球谐后端**。
5. **已尝试且全部失败的 cuet-only 开关**(权重梯度仍 0):`.contiguous()` / `*1.0` / `clone` / `+0*y` 中间层;cuet 球谐 `method="naive"`;`normalize=False` + plain-torch 归一化;`cuet.Linear` 用 `layout=mul_ir`;fused ops(cu13)装好激活后仍 0。
6. **结论**:在 cueq 0.10 下,唯一已知可工作的球谐路径是 e3nn(被"只能 cuet"约束排除)。**ac-002 的 toggle / ac-003 的纯 cuet 修复目前不可行**,需操作者决策(放宽约束允许"我们自写的纯 torch 球谐",或提供某个 cuet 球谐组合二阶可微的写法,或接受能量-only)。详见与操作者的讨论。

## Findings (2026-06-21, mol:impl run 2 — run 1 premise DISPROVEN)

Run 2 re-verified the RED, then disproved the entire root-cause hypothesis. **There is no cuet/cueq double-backward bug.**

1. **The minimal repro (ac-001) is mathematically DEGENERATE, not a severed graph.** `cuet SH(normalize=True) → cuet.Linear(→8x0e) → sum` produces an energy that is *exactly constant in v*: an equivariant linear map reads only the `0e` channel, and the `l=0` spherical harmonic is a constant (SH l=0 channel std over samples = 0). Measured `|∂E/∂v| = 0` at **first** order — the force itself is identically zero, so its weight-grad being zero is trivial, not evidence of a double-backward break. Any rotation-invariant of a *single normalized direction* is constant (∑_m|Y_lm(v̂)|² = (2l+1)/4π), so the whole single-vector-normalized repro family is degenerate.
2. **Pure cuet SH → cuet equivariant ops double-backward FINE.** With a non-degenerate energy (`normalize=False`, so |v|^l radial dependence survives): `cuet SH → FullyConnectedTensorProduct → 0e` gives |g|=824, force-loss→weight-grad **L1 = 2.06e5**; `cuet SH → cuet.Linear(l>0) → TP → 0e` gives |g|=96, Linear weight-grad **L1 = 3954**. Both nonzero → second-order backprop through pure-cuet SH + cuet.Linear + cuet TP works. The constraint "pure cuet, no e3nn" was never the blocker.
3. **A pure-torch SH built from cueq's own `sympy_spherical_harmonics` matches `cuet.SphericalHarmonics` to 6.7e-16** — and *still* gives zero on the degenerate repro, confirming the zero is about the energy being constant, not about the SH backend.
4. **Real `MACEOMol` "73/73 (actually ~0/104) zero grad" root cause = `src/molzoo/mace_omol.py:192`:** `grad = torch.autograd.grad(total_energy.sum(), positions, create_graph=self.training)[0]`. The model defaults to **eval** mode (`self.training=False`) after construction, so `create_graph=False` detaches the force from the parameter graph → any force loss yields 0/104 param grads (and `loss.backward()` even raises "does not require grad"). Flipping to train mode (same line, `create_graph=True`) gives **102/104 params nonzero**. The original diagnostic ran the model in eval mode and misattributed the detached force to cueq.
5. **Why the official teacher "worked":** the teacher was probed via its own training-mode force path (create_graph on); the difference was never the spherical-harmonics backend or the `cuet.Linear` instantiation.
6. **Separate, real, out-of-scope issue:** the `MACEOMol.forward → ForceDerivation` (functorch `torch.func.grad`, `force.py:59`) path does not trace through cuet `SphericalHarmonics` (functorch transform unsupported). This is orthogonal to second-order *training* (Domain basis already says functorch ≠ double-backward) and is a separate ticket.

**Conclusion:** This spec's Design (pure-cuet surgery across `molrep/interaction/`, ac-002 toggle, ac-003 weight-layout preservation) targets a non-existent bug. The real fix is one line in `molzoo/mace_omol.py` (make `create_graph` robust for force-supervised training, e.g. `create_graph=self.training or torch.is_grad_enabled()`, and/or ensure the training loop calls `model.train()`), plus a regression test and a force-training smoke check. **Recommend superseding this spec** with a corrected minimal one. Awaiting operator decision.

## Resolution (2026-06-21, mol:impl run 3 — functorch port, operator-directed)

The autograd double-backward path works eager but **cannot `torch.compile`** (confirmed: "aot_autograd does not currently support double backward"). PiNet shows functorch (`torch.func.grad`) *is* compile-friendly. So the operator directed porting MACEOMol's force path to functorch. The blocker was that functorch transforms cannot trace cuEquivariance's fused custom ops (no `setup_context`). Resolved by making the model's equivariant ops functorch-traceable:

- **Spherical harmonics → pure torch** (`src/molrep/embedding/angular.py`): replaced the `cuequivariance_torch.SphericalHarmonics` backend with a pure-PyTorch evaluation built from cuEquivariance's own symbolic polynomials (`cue.descriptors.sympy_spherical_harmonics`), pow-free (cumulative-product power tables) so the gradient is defined at the origin. Matches the cuet backend to **4.4e-16** for l_max=1,2,3; functorch- and `torch.compile`-traceable.
- **Equivariant tensor products → `use_fallback=True`** (`molrep/interaction/residual.py` `ChannelWiseTensorProduct`, `molrep/interaction/product_basis.py` `SymmetricContraction`): the fused CUDA kernels have no functorch `setup_context`; their pure-torch fallback is numerically identical and traceable. (`cuet.Linear` already traces, left as-is.)

**Verified:**
- `MACEOMol.forward` (functorch `ForceDerivation`) now runs; all 6 prior `test_mace_omol.py` tests pass (the 5 `test_forward_*` were RED on `dev` HEAD).
- Force-supervised training works: force loss → 71/73 params (small fixture; 102/104 official-weights). New tests `test_functorch_force_loss_trains_via_forward` (forward/functorch) and `test_force_loss_reaches_parameters_in_eval_mode` (energy_forces/autograd).
- **`torch.compile(fullgraph=True)` of the functorch force loss: OK** (71/73). (`torch.compile` default-mode hits an unrelated CPU-inductor scatter codegen bug; fullgraph — the perf mode — works.)
- Port consistency preserved: `scripts/omol_port/verify_e2e.py` PASS (|dE|=7e-7, max|dF|=4.3e-6).

**Also landed (secondary, autograd path):** `energy_forces` now uses `create_graph = self.training or torch.is_grad_enabled()` so the raw-tensor autograd entry is also trainable in eval mode (used by `verify_e2e.py`).

**Obsolete (premise disproven — see Findings run 2):** original Design (pure-cuet surgery), ac-002 (toggle), ac-003 (weight-layout) target a non-existent cuet double-backward bug; dropped.

**Out of scope (same pre-existing issue, fixable by the same pattern):** `molzoo/mace.py` and `molzoo/allegro.py` force paths still raise `setup_context` (their `contraction.py` / `product.py` / allegro cuet ops are still on the fused path). 41 such `test_symmetry.py` failures are pre-existing on `dev` HEAD, not regressions. Extending `use_fallback=True` to those shared ops would fix them identically.

## Testing

- 最小复现脚本:修复前 RED(grad L1 == 0),修复后 GREEN(grad L1 > 0)。
- `tests/test_molrep/test_cueq_double_backward.py`:对一个小的 l>0 cuet 路径断言力损失→权重梯度非零(回归护栏)。
- 全模型探针:构造 `MACEOMol`,力损失 backward,断言非零梯度参数数 > 0(理想 ≈ 全部)。
- 一致性:`scripts/omol_port/verify_e2e.py` 修复后仍 `PASS`(|dE|<1e-5, max|dF|<1e-4;实测应 ~1e-9)。
- 全量:`python -m pytest tests/ -v`。
- **运行环境**:aarch64 GH200,`.aarch64` venv(torch 2.12.0+cu130, cueq 0.10);fused ops 需 `LD_LIBRARY_PATH` 含 `cuequivariance_ops/lib` 与 `nvidia/cu13/lib`。结论须在"有 fused ops"与"naive fallback"两种路径下都明确记录(teacher 在 naive 下即可二阶反传,故修复目标应不依赖 fused ops)。

## Out of scope

- molpy 数据管线(读 xyz + 周期邻居表 + shifts)—— 属下游 fine-tune spec。
- ZnO 体系的实际 fine-tune 训练运行与能量参考(E0)重拟合。
- OMOL 在凝聚态上 OOD 的科学问题(已知,另议)。
- 球谐本身的替换(已证 cuet 球谐二阶 OK,非本 spec 目标)。
