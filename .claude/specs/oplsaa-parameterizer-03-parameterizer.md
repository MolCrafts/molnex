---
title: Parameterizer 硬改名与 ForceFieldPredictions
status: code-complete
grilled: true
slug: oplsaa-parameterizer-03-parameterizer
chain: oplsaa-parameterizer
created: 2026-08-20
---

# Parameterizer 硬改名与 ForceFieldPredictions

## Summary

将 `ClassicalIMM` 硬改名为可训练的 `Parameterizer(encoder, template: ForceField)`：模块文件 `classical_imm.py` → `parameterizer.py`，公开符号仅 `Parameterizer`（无 `ClassicalIMM` 别名）。构造时按 `template` 的 style / `PotentialMeta` schema 挂接已有 heads（含 LJ、charge、improper；proper 走链上 `02-heads` 的 OPLS `k1..k4` 路径），`forward(batch)` 只返回填参后的同型 `ForceField` 副本；能量与力一律经 `ff.to_potentials()`，本类不求值。同步将 `ClassIPredictions` 硬改名为 `ForceFieldPredictions`（无旧名别名），并更新 `molpot` / `molpot.forcefield` 导出面。删除 `project()`。本阶段不改训练脚本全集与 notes 清扫（由 `04-cut` 收尾）。

## Domain basis

本规格**不引入新势能形式**。填参后的能量仍是 OPLS-AA / Class-I 代数式，由既有 `ForceField.to_potentials()` 在坐标与邻表上求值：

- 键：\(E_b=\tfrac12 k_b(r-r_0)^2\)（kcal/mol；\(k_b\) 为 kcal mol⁻¹ Å⁻²，\(r,r_0\) 为 Å）
- 角：\(E_a=\tfrac12 k_a(\theta-\theta_0)^2\)（\(k_a\) 为 kcal mol⁻¹ rad⁻²，\(\theta\) 为 rad）
- OPLS proper（LAMMPS `dihedral_style opls`）：
  \(E=\tfrac12 K_1(1+\cos\phi)+\tfrac12 K_2(1-\cos 2\phi)+\tfrac12 K_3(1+\cos 3\phi)+\tfrac12 K_4(1-\cos 4\phi)\)，\(K_n\) 为 kcal/mol
- LJ 12-6 + Coulomb；部分电荷 \(q\) 为基本电荷 \(e\)

`Parameterizer` 只读拓扑（经 `ChemEncoder`），故 \(\theta=\theta(G)\)、\(d\theta/dx=0\)，部署力为 \(F=-\partial E(x;\theta(G))/\partial x\)，与导出力场一致。

参考文献：Jorgensen, W. L.; Maxwell, D. S.; Tirado-Rives, J. *J. Am. Chem. Soc.* **118**, 11225 (1996). https://doi.org/10.1021/ja9621760 ；Wang, Y.; Fass, J.; Chodera, J. D. *Chem. Sci.* **13**, 12016 (2022). https://doi.org/10.1039/D2SC02739A ；LAMMPS `dihedral_style opls`。

## Design

**实体。**

- `Parameterizer(nn.Module)`：公开构造 `Parameterizer(encoder, template, *, bond_head=…, angle_head=…, proper_head=…, improper_head=…, atom_head=…, charge_model=…)`。`encoder` 复用 `molrep.chem.encoder.ChemEncoder`（或同协议模块）；`template` 为 `ForceField` 子类实例（链上默认用法钉 `OPLSAAForceField`，类型面由 `01-types` 提供）。关键字由旧 `forcefield=` **改为** `template=`（位置第二参亦可）。无 `ClassicalIMM` 别名、无 `project`、无能量方法。
- **Schema 装配 / 校验。** 构造时读取 `template` 已注册 styles，经 `PotentialMeta.schema_for(...)`（或等价 style 查询）决定哪些 namespace 必须由 head 写入：`bond:harmonic` → `bond_head`；`angle:harmonic` → `angle_head`；`dihedral:opls` → `proper_head`（输出必须为 OPLS `k1..k4`，依赖 `02-heads`）；`improper:*` → `improper_head`；`pair:lj/cut` → `atom_head`（`LJParameterHead`）；`pair:coulomb` → `charge_model.write_forcefield`。缺必需 head / 键名与 schema 不符 → 构造或首次 `forward` **大声失败**（`ValueError` / `TypeError`）。未声明的 style 不写参。
- **生命周期。** `forward(batch)`：`features = encoder` → 各 head 产出参数表 → `template.with_params(...)`（及可选 charge 写入）→ 返回**副本**；永不就地改写传入的 `template`。调用方求 \(E/F\)：`filled.to_potentials().calc_energy / calc_forces(batch)`。
- `ForceFieldPredictions`：由 `ClassIPredictions` 硬改名；`__init__(model: Parameterizer, *, keys=…, hessian_derivation=…)`；一次 `model(batch)` + 至多一次 `to_potentials()`；支持键仍为 `energy` / `forces` / `hessian` / `q`。无 `ClassIPredictions` 别名。
- **删除。** `project`、`_propers_to_opls`、`_opls_to_periodic` 及一切 Fourier↔OPLS 投影路径（训练主路径直接预测 OPLS 表）。`classical_imm.py` 删除；测试镜像同步改名。

**Reuse decision**

- generalize `ClassicalIMM` → `Parameterizer`（同职责：encoder + heads → 填参 `ForceField`；收紧 schema 门禁；删 `project`）
- reuse `ChemEncoder`（拓扑 → `chem_features` / `embeddings`；不改感知语义）
- reuse `ForceField.to_potentials`（能量/力唯一路径）
- reuse `BondParamHead` / `AngleParamHead` / `ImproperParamHead` / `LJParameterHead` / `AtomicChargeModel.write_forcefield`；proper 复用 `02` 的 OPLS `k1..k4` 路径（`ProperTorsionParamHead` 或其并列 OPLS 输出）
- generalize `ClassIPredictions` 模式 → `ForceFieldPredictions`（类型标注与文档改挂 `Parameterizer`）
- reuse `PotentialMeta` / `OPLSAAForceField` / `ForceField`（假定 `01-types` 已落地；本规格不重做类型面）
- new — 无；禁止再实现第二套 encoder / 势核 / head

## Files to create or modify

- `src/molpot/forcefield/parameterizer.py` (new) — 自 `classical_imm.py` 迁入并改名实现
- `src/molpot/forcefield/classical_imm.py` — 删除
- `src/molpot/forcefield/predictions.py` — `ClassIPredictions` → `ForceFieldPredictions`
- `src/molpot/forcefield/__init__.py` — 导出 `Parameterizer` / `ForceFieldPredictions`；移除旧名
- `src/molpot/__init__.py` — 同上
- `tests/test_molpot/test_forcefield/test_parameterizer.py` (new) — 自 `test_classical_imm.py` 迁改
- `tests/test_molpot/test_forcefield/test_classical_imm.py` — 删除
- `tests/test_molpot/test_forcefield/test_predictions.py` — 改挂新名与 `Parameterizer`
- `regressions/oplsaa-parameterizer-03-parameterizer.py` (new)

## Tasks

- [x] Write failing unit tests for Parameterizer (tests/test_molpot/test_forcefield/test_parameterizer.py → TestParameterizer): schema 挂接 LJ/charge/improper/OPLS proper；`forward` 填参副本；无 `project`；`ClassicalIMM` 不可导入
- [x] Write failing unit tests for ForceFieldPredictions (tests/test_molpot/test_forcefield/test_predictions.py → TestForceFieldPredictions): 可导入；经 `to_potentials` 得 energy/forces；`ClassIPredictions` 不可导入
- [x] Generalize ClassicalIMM to Parameterizer in src/molpot/forcefield/parameterizer.py (delete classical_imm.py; drop project and Fourier↔OPLS helpers; template= API; schema-gated heads)
- [x] Generalize ClassIPredictions to ForceFieldPredictions in src/molpot/forcefield/predictions.py (type model: Parameterizer; no ClassIPredictions alias)
- [x] Update exports in src/molpot/forcefield/__init__.py and src/molpot/__init__.py (Parameterizer, ForceFieldPredictions only; remove ClassicalIMM / ClassIPredictions)
- [x] Add Google-style docstrings on Parameterizer and ForceFieldPredictions with units (kcal/mol, Å, e)
- [x] Add regression example regressions/oplsaa-parameterizer-03-parameterizer.py (public API only; hard-coded goldens, no third-party runtime)
- [x] Verify Parameterizer(ChemEncoder, OPLSAAForceField) forward fills params; ImportError for ClassicalIMM/ClassIPredictions; no project attribute
- [x] Run full check + test suite

## Testing strategy

- **单元（`tests/` 镜像，单方法）。** `TestParameterizer`：构造绑定 `OPLSAAForceField` + stub/`ChemEncoder` + 常量 heads → `forward` 后 `params("bonds"|"angles"|"propers"|"impropers"|"atoms")` 键与形状符合 schema（proper 含 `k1..k4`）；模板对象本身未被就地修改；`hasattr(model, "project") is False`；`from molpot.forcefield import ClassicalIMM` 抛 `ImportError`。`TestForceFieldPredictions`：两原子谐振子硬编码 \(k=100\)、\(r_0=1\)、\(r=1.5\) → \(E=12.5\) kcal/mol、\(F_{0,x}=+50\)（沿用既有 analytic 夹具）；`from molpot.forcefield import ForceFieldPredictions` 成功；`ClassIPredictions` 导入失败。
- **边沿。** template 声明 `dihedral:opls` 但 proper head 缺 `k1` → 失败；有 `pair:coulomb` 无 `charge_model` 且任务要写 `q` → 失败；encoder 无 atom features 却挂 charge → 失败。
- **回归。** `regressions/oplsaa-parameterizer-03-parameterizer.py`：`Parameterizer(ChemEncoder(...), OPLSAAForceField(), …heads…)` → `ff = model(batch)` → `ff.to_potentials().calc_energy(batch)` 复现硬编码标量；断言无 `project`；尝试导入旧名失败。第三方离线注释（若有）写入字面量旁，运行时不调用。

## Out of scope

- WaterLES 数据集训练与实验脚本
- freeze-encoder + head-swap 精度探针
- GAFF / 其它具名力场模板
- 保留或兼容 `project()` / Fourier→OPLS 投影训练路径
- `scripts/mm_param_learning/**` 与 `.claude/notes/learnable-classical-ff.md` / architecture 公开名全集硬切清零（本阶段只动 package 导出；由 `04-cut` 完成）
- 重写 `ChemEncoder`、势核、`BondParamHead` / `AngleParamHead` / `ImproperParamHead` / `LJParameterHead` / charge 求解器
