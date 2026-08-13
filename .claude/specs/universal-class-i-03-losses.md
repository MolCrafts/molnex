---
title: Universal Class-I 多任务损失原语
status: approved
created: 2026-08-13
grilled: true
---

# Universal Class-I 多任务损失原语

## Summary

在既有 `molix.core.losses.molecular` 上扩展四类可组合的 `nn.Module` 损失原语，供 Universal Class-I 多任务训练直接调用：用 Wilson B 矩阵把驻点笛卡尔 Hessian 投影到内坐标后再对 Class-I 对角力常数（或给定的 \(H_q\)）做 MSE；对扭转扫描能量做分子/扫描中心化 MSE，并可加垒高项；对 EEM 电荷与划分参考做电荷-only MSE；以及用训练集统计量 \(s_k\) 把各项变成无量纲 \(\tilde L_k=L_k/s_k^2\)，再交给已有 `WeightedLoss` 做 \(\sum w_k\tilde L_k\)。调用方只传入张量，不必经过新的损失包、编码器或势能头。

## Domain basis

Class-I 力场拟合的是相对构象能，不是生成热。分子（或扫描）内去中心：

\[
\Delta E_i = E_i - \langle E\rangle_{\mathrm{mol}}.
\]

绝对 QM 能量 / 生成热不可作为必选路径（Wang 2022 Fig.4 / §3；Takaba 2023 §3）。

Wilson GF（内坐标 \(s=Bx\)）。在驻点 \(\nabla E=0\) 处，笛卡尔二次型与内坐标二次型等价：

\[
\mathrm{d}x^{\mathsf T} H_x\,\mathrm{d}x = \mathrm{d}q^{\mathsf T} H_q\,\mathrm{d}q,
\qquad
H_q=(B^{+})^{\mathsf T} H_x B^{+},
\]

其中 \(B=\partial q/\partial x\) 形状为 \((\mathrm{n}_q,3N)\)，\(B^{+}\) 为 Moore–Penrose 伪逆，\(H_x\) 形状 \((3N,3N)\)，\(H_q\) 形状 \((\mathrm{n}_q,\mathrm{n}_q)\)。Class-I 只取对角：\(H_{rr}=k_b\)（键，kcal mol\(^{-1}\) Å\(^{-2}\)）、\(H_{\theta\theta}=k_\theta\)（角，kcal mol\(^{-1}\) rad\(^{-2}\)）。非对角是 Class-II，本损失的对角模式不把它们罚向 0。笛卡尔 \(H_x\) 还含 6 个刚体零模以及非键 Hessian；若两者同时存在，调用方须先减去（或联合评估）冻结非键 Hessian 再投影。本 spec 的单元测试与回归只用纯价键玩具，损失本身不减非键、不做 Eckart 投影。THEMol Hessian 给在极小值上（Liang 2026），驻点假设成立；本损失不检查 \(\nabla E=0\)。

**禁止**把笛卡尔 \(H_{ij}\) 全部摊平当作等权目标：那会把刚体零模、非键耦合和价键力常数混成同一个 MSE。

电荷项只比较划分参考（MBIS / AM1-BCC 类）：\(L_q=\mathrm{MSE}(q_{\mathrm{pred}},q_{\mathrm{ref}})\)，单位 \(e\)。电中性是 EEM KKT 求解器的约束，不是本损失的项；此处不做 \(E(q)\) 耦合，也不引入实时几何依赖 QEq。

无量纲标度（调用方传入训练集统计量 \(s_k\)，本模块不扫描数据集）：

\[
\tilde L_k = L_k / s_k^2.
\]

`WeightedLoss` 已实现 \(\sum_k w_k\tilde L_k\)（把 \(\tilde L_k\) 当作它的一项即可）。

单位（与 Class-I 约定一致，不在损失里做换算）：能量 kcal/mol；Hessian 按内坐标标注为 kcal mol\(^{-1}\) Å\(^{-2}\) 或 kcal mol\(^{-1}\) rad\(^{-2}\)；电荷 \(e\)。

参考文献：

- Wang, Fass, Kaminow, Herr, Rufa, Zhang, Pulido, Clark-Joseph, Chodera. arXiv:2010.01196
- Takaba, Pulido, Henry, Chodera. arXiv:2307.07085
- Liang et al., THEMol. arXiv:2605.14973
- Wilson, E. B. *J. Chem. Phys.* **9**, 76 (1941). DOI 10.1063/1.1750829

**硬编码 Wilson 投影金标（回归与域测试照抄，atol \(10^{-6}\)）**。二原子键沿 \(x\)，\(r=1.5\) Å（几何说明；沿 \(x\) 的单位键 \(B\) 与 \(r\) 无关），\(k=2.0\) kcal mol\(^{-1}\) Å\(^{-2}\)。\(q=\|x_2-x_1\|\)，\(\hat u=(1,0,0)\)，\(B=\partial q/\partial x\)：

```
B = [[-1.0, 0.0, 0.0, 1.0, 0.0, 0.0]]          # (1, 6)
```

驻点谐振子 \(H_x = k\,B^{\mathsf T}B\)（仅相对 \(x\) 耦合为 \(k\)）：

```
H_x = [
  [ 2.0, 0.0, 0.0, -2.0, 0.0, 0.0],
  [ 0.0, 0.0, 0.0,  0.0, 0.0, 0.0],
  [ 0.0, 0.0, 0.0,  0.0, 0.0, 0.0],
  [-2.0, 0.0, 0.0,  2.0, 0.0, 0.0],
  [ 0.0, 0.0, 0.0,  0.0, 0.0, 0.0],
  [ 0.0, 0.0, 0.0,  0.0, 0.0, 0.0],
]                                               # kcal/mol/Å²
```

\(BB^{\mathsf T}=2\)，\(B^{+}=[-1/2,0,0,1/2,0,0]^{\mathsf T}\)。解析结果：

\[
H_q=(B^{+})^{\mathsf T} H_x B^{+} = [[2.0]].
\]

对角目标 \(k_b=2.0\) 时 MSE \(=0\)；\(k_b=0\) 时 MSE \(=4.0\)。正交于 \(B\) 的笛卡尔噪声（例如只改 \(H_x[1,1]\)）不得改变 \(H_q\)。

扭转剖面金标（与现有 `TestMoleculeCenteredEnergyMse` 同一组数）：\(E_{\mathrm{pred}}=[1,2,5,7]\)，\(E_{\mathrm{ref}}=[10,11,0,2]\)，`group_ids=[0,0,1,1]`，中心化后 MSE \(=0\)。垒高：同一组 \(\Delta E=\max E-\min E\)；例 \(E_{\mathrm{pred}}=[0,4]\)，\(E_{\mathrm{ref}}=[0,2]\)，`group_ids=[0,0]` → 中心化 MSE \(=1.0\)，垒高残差 \((4-2)^2=4\)；`barrier_weight=0.5` 时总损失 \(=3.0\)。

电荷金标：\(q_{\mathrm{pred}}=[0.2,-0.2]\)，\(q_{\mathrm{ref}}=[0.1,-0.1]\) → MSE \(=0.01\)。

标度金标：内损失 \(L=4.0\)，\(s=2.0\) → \(\tilde L=1.0\)。

## Design

新符号全部落在已有模块 `src/molix/core/losses/molecular.py`，形状对齐 `MSELoss` / `WeightedLoss`（`nn.Module`，`forward(pred, target)`，缓冲器用 `register_buffer`），**不**再增加 `energy_mse` 那种工厂闭包，也**不**把既有工厂改写成类。`WeightedLoss` 的 `nn.ModuleList` 只收 `nn.Module`；新损失必须是模块才能直接加权。构造即 `Cls(...)`，禁止 `make_*` / `create_*` 作为主 API。损失不拥有几何、编码器或力场；无跨 step 累加器。`ScaleNormalizedLoss.scale` 是 buffer（随 `.to()` / `state_dict`），不是 `nn.Parameter`，dtype 用 `config.ftype`。

`ProjectedInternalHessianLoss`（`nn.Module`）：

- `@staticmethod project(cartesian_hessian, b_matrix) -> H_q`：`H_q = pinv(B).T @ H_x @ pinv(B)`，`torch.linalg.pinv`。输入无 batch 维：`H_x` 为 `(3N, 3N)`，`B` 为 `(n_q, 3N)`，输出 `(n_q, n_q)`。形状不合则 `ValueError`。不对称性检查、不减非键、不检驻点。
- `target_kind`：`"diagonal"`（默认）对 `diag(H_q)` 与 `target[diagonal_key]`（`(n_q,)` 的 \(k_b,k_\theta,\ldots\)，行序与 `B` 一致，本类不解释内坐标类型）做 MSE；`"hq_ref"` 对完整 `H_q` 与 `target[hq_ref_key]` 做 MSE。这是比较选择，不是后端开关，禁止再引入 `method=`。对角模式**不**把非对角罚向 0。
- `forward(pred, target)` 从 `pred[hessian_key]`（默认 `"hessian"`）与 `target[b_matrix_key]`（默认 `"B"`）取张量，先 `project` 再 MSE。任何实现路径若对 `H_x.reshape(-1)` 做等权 MSE 即视为违反本 spec。

`TorsionProfileLoss`（`nn.Module`）：

- `forward` 读 `pred[pred_key]`、`target[target_key]`、`target[group_key]`（默认能量键 `"energy"`，组键 `"molecule_id_index"`），对 pred/ref **分别**调用已有 `center_by_group`，再 `nn.MSELoss`。`group_ids` 由调用方提供（分子 id 或扫描 id），本类不区分。
- 可选 `barrier_weight`（Python float，默认 `0.0`）：每组 \(\Delta E=\max E-\min E\)，再 MSE 垒高并加到剖面项上。`barrier_weight=0` 时与 `center_by_group` + MSE 数值一致；不调用、不改写 `molecule_centered_energy_mse`（那条 TensorDict/`graphs` 路径保持原样）。
- 不做绝对能量 / HoF 项。

`ChargeMSELoss`（`nn.Module`）：对 `pred[pred_key]` 与 `target[target_key]`（默认 `"q"`）做 MSE，单位 \(e\)。不读能量键，不加中性罚，不加 \(E(q)\)。内部用 `nn.MSELoss`，键抽取对齐 `MSELoss`。

`ScaleNormalizedLoss`（`nn.Module`）：`ScaleNormalizedLoss(loss, scale=s)`，`forward` 返回 `loss(pred, target) / scale**2`。`scale` 由调用方从训练统计传入；`scale==0` 在 `__init__` 里 `ValueError`。组合方式：

```python
WeightedLoss([
    (w_h, ScaleNormalizedLoss(ProjectedInternalHessianLoss(...), scale=s_h)),
    (w_t, ScaleNormalizedLoss(TorsionProfileLoss(...), scale=s_t)),
    (w_q, ScaleNormalizedLoss(ChargeMSELoss(...), scale=s_q)),
])
```

不新增 `run_everything` / 多任务 façade；加权是调用方用 `WeightedLoss` 组装的。

测试路径：GLOB 显示 `tests/test_molix/test_core/test_losses/` **不存在**，主镜像已是 `tests/test_molix/test_core/test_losses_molecular.py`。在该文件扩展 `TestProjectedInternalHessianLoss` / `TestTorsionProfileLoss` / `TestChargeMSELoss` / `TestScaleNormalizedLoss`，不新造 `test_losses/` 树，也不把现有测试搬走。

`src/molix/core/losses/__init__.py` 再导出四个新名，并按仓库规则重排 `__all__` 字母序。不改 `src/molix/__init__.py`，不改 `combined.py`。

### Reuse decision

- reuse `center_by_group` — `TorsionProfileLoss.forward` 对 pred/ref 各调用一次
- reuse `molecule_centered_energy_mse` — 保留既有 TensorDict 能量闭包；本 spec 不改写、不平行实现
- reuse `energy_force_mse` — 不改、不重实现
- reuse `WeightedLoss` — 无量纲项作为其 `(weight, nn.Module)` 输入；本 spec 不改 `combined.py`
- reuse `parameter_bag_mse` — 参数袋诊断保留；Hessian 损失不得走摊平袋
- generalize `molix.core.losses.molecular` — 在同一模块加入上述四个 `nn.Module`（投影 / 扭转 PES / 电荷 / 标度归一）
- reuse `ChemEncoder`, `ClassicalIMM`, heads, IR, charge solvers, `Condenser`, `ff_export`, `group_split` — 本 spec 不修改、不重实现；损失只消费张量

## Files to create or modify

- `src/molix/core/losses/molecular.py`
- `src/molix/core/losses/__init__.py`
- `tests/test_molix/test_core/test_losses_molecular.py`
- `regressions/universal-class-i-03-losses.py` (new)

## Tasks

- [ ] Write failing unit tests for ProjectedInternalHessianLoss, TorsionProfileLoss, ChargeMSELoss, and ScaleNormalizedLoss (tests/test_molix/test_core/test_losses_molecular.py → TestProjectedInternalHessianLoss / TestTorsionProfileLoss / TestChargeMSELoss / TestScaleNormalizedLoss)
- [ ] Generalize `molix.core.losses.molecular` in `src/molix/core/losses/molecular.py` to cover projected Hessian, torsion profile, charge MSE, and scale-normalized wrapper
- [ ] Add docstring per google style with units on the new public symbols in `src/molix/core/losses/molecular.py`
- [ ] Export the new symbols from `src/molix/core/losses/__init__.py` and re-sort `__all__` alphabetically
- [ ] Add regression example `regressions/universal-class-i-03-losses.py` (public API only; hard-coded goldens, no third-party runtime)
- [ ] Verify against the 2-atom Wilson fixture (\(H_q=2.0\) kcal mol\(^{-1}\) Å\(^{-2}\), atol `1e-6`)
- [ ] Run full check + test suite

## Testing strategy

单元测试只写在已有镜像 `tests/test_molix/test_core/test_losses_molecular.py`（一类一 `Test*`，一方法一行为；`tests/` 下不做 e2e）。

`TestProjectedInternalHessianLoss`：

- Happy：Domain 金标 `project(H_x, B)` 得 `[[2.0]]`（atol `1e-6`）；`target_kind="diagonal"` 对 \(k=2.0\) 损失为 0；对 \(k=0\) 损失为 4.0；`target_kind="hq_ref"` 对 `[[2.0]]` 损失为 0。
- Edge：`H_x` 非方、`B` 列数 ≠ `3N` 时 `ValueError`；`H_x[1,1]` 加正交噪声后 \(H_q\) 不变（证明走投影而非摊平笛卡尔）。
- Domain：解析 Wilson 恒等式 \(H_q=(B^{+})^{\mathsf T}H_x B^{+}=k\)（Wilson 1941；驻点谐振子）。

`TestTorsionProfileLoss`：

- Happy：与 `TestMoleculeCenteredEnergyMse` 同一组数，损失为 0（证明调用 `center_by_group`）。
- Edge：`barrier_weight=0.5` 时总损失为 3.0（剖面 1.0 + 0.5×4）；空/单点组垒高为 0。
- Domain：相对能量 \(\Delta E_i=E_i-\langle E\rangle\)，无绝对能量项。

`TestChargeMSELoss`：

- Happy：金标 MSE `0.01`。
- Edge：只给 `q` 键即可 `forward`（不要求 energy 键）。

`TestScaleNormalizedLoss`：

- Happy：内损失 4、`scale=2` → `1.0`；`scale` 在 `named_buffers()` 且不在 `parameters()`。
- Edge：`scale=0` 在构造期 `ValueError`。
- Happy / 组合：`WeightedLoss([(1.0, ScaleNormalizedLoss(ChargeMSELoss(), scale=2.0))])` 对上述电荷金标得 `0.01/4=0.0025`。

回归脚本 `regressions/universal-class-i-03-losses.py`：只走 `molix.core.losses` 公有 API。脚本内写死 Domain basis 的 \(B\)、\(H_x\)、\(H_q=2.0\)、对角 MSE、以及 \(\tilde L=4/2^2=1.0\)（atol `1e-6`）。注释写明解析 Wilson GF（非第三方 oracle）、日期 `2026-08-13`。运行时不得 import / subprocess 任何第三方化学软件。

## Out of scope

- 摊平笛卡尔 Hessian 损失
- 实时几何依赖 QEq，以及任何 \(E(q)\) / 电荷–能量耦合
- 把气相 LJ 残差当成“非键损失”（以免解冻 \(\sigma,\varepsilon\)）
- 生成热或未中心化的绝对 QM 能量作为必选路径
- 新的公有 `run_everything` / 多任务 façade
- 新的 `molix.core.losses` 子树或新包
- 把既有 `energy_mse` / `molecule_centered_energy_mse` 工厂改写成类
- 非键 Hessian 扣除、Eckart / 刚体模过滤、驻点力检查
- molrep、molpot heads、数据 bundle、THEMol ingest、凝聚态可观测量
- `ChemEncoder`、`ClassicalIMM`、IR、电荷求解器、`Condenser`、`ff_export`、`group_split` 的修改
- 改 `src/molix/__init__.py` 或 `src/molix/core/losses/combined.py`
- 前置 01 encoder / 02 fourier（本 spec 只消费张量）
