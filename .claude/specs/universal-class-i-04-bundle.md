---
title: 分子 PES 束原语（PesBundle）
status: approved
created: 2026-08-13
grilled: true
---

# 分子 PES 束原语（PesBundle）

## Summary

在 `molix.data` 增加名为 `PesBundle` 的数据/视图类型：一份共价拓扑加上该分子的一组构象坐标。调用方对同一张图只做一次化学感知与 Class-I 参数赋值（`topology_batch()` → `ChemEncoder` / `ClassicalIMM`），再按构象调用势能（`conformation_batch(i)` 或迭代 → `to_potentials().calc_energy(pos=…)`）。束内所有构象共享同一 `group_id`；分子级划分继续走已有的 `group_split_indices`，同一分子的构象不会被拆进不同 train/val/test。本规格不下载数据、不写训练脚本、不改 encoder / heads / losses。

## Domain basis

一张分子图 \(G\)（核电荷 + 共价键/角/二面角拓扑，可选图电荷 \(Q\)）决定一套参数 \(\Theta\)；势能面是构象坐标上的标量场：

\[
\Theta = f(G),\qquad E_c = E_\Theta(R_c),\quad c = 1,\ldots,C .
\]

同一 \(G\) 的全部构象共用 \(\Theta\)，不得按构象各自重新感知或重新赋参。划分必须按唯一分子图 / 异构体身份，而不是按构象；Wang / Takaba 的协议是按分子做 \(80/10/10\)。相对能量 \(\Delta E_c = E_c - \bar E_G\) 由后续 losses（规格 03，`center_by_group`）计算，**不**在束内预减。本类型不计算、不存储生成热（HoF）。

- Wang, Y. et al. End-to-end differentiable construction of molecular mechanics force fields. *Chem. Sci.* **2022**, 13, 12016–12033. https://doi.org/10.1039/D2SC02739A https://arxiv.org/abs/2010.01196
- Takaba, K. et al. Machine-learned molecular mechanics force field for the simulation of protein–ligand complexes and beyond. arXiv:2307.07085 (2023). https://arxiv.org/abs/2307.07085

单位（调用方经 `from molix.units import units` 自行换算；本类型不写死转换系数）：

| 量 | 符号 | 存储形状 | 单位 |
|---|---|---|---|
| 坐标 | \(R_c\) | `(C, N, 3)` | Å |
| 图电荷 | \(Q\) | `graphs.total_charge` `(1,)` | \(e\) |
| 能量（可选靶） | \(E_c\) | `(C,)` | 调用方（MM 常见 kcal/mol） |
| 力（可选靶） | \(F_c\) | `(C, N, 3)` | energy / Å |
| Hessian（可选靶） | \(H_c\) | `(C, 3N, 3N)` | energy / Å²；笛卡尔序 \(x_0,y_0,z_0,\ldots\) |
| 原子电荷（可选靶） | \(q_c\) | `(C, N)` | \(e\) |

## Design

`PesBundle` 是 `molix.data` 里的普通类（**不是** `TensorDict` 子类，**不是** `Dataset` / `DataSource`）。它拥有一份拓扑与一叠构象，对外只提供单职责视图；调用方自己组合「赋参一次、按构象求能」。禁止 `make_pes_bundle`、禁止 `run_everything`、禁止把 encoder / potential / split 塞进一个上下文袋。本模块不得 import `molrep` / `molpot` / `molzoo`。

**构造。** 唯一主构造器：

```python
PesBundle(topology, pos, *, group_id, targets=None)
```

- `topology`：嵌套 `TensorDict`（`batch_size=[]`），契约与当前 post-collate schema 对齐：
  - 必有 `atoms.Z` `(N,)` `long`；
  - 可选 `atoms.batch` `(N,)`（缺则填 0）、`atoms.mass` `(N,)` amu、`graphs.num_atoms` `(1,)`、`graphs.total_charge` `(1,)`（物理量 \(Q\)，单位 \(e\)；**不**新增 `graphs.Q`）；
  - 可选共价命名空间：`bonds.bond_index` `(2, N_b)` COO、`bonds.bond_types`；`angles` / `propers` / `impropers` 为 1-D 列（`atomi`/`atomj`/`atomk`[/`atoml`]，impropers 的 `atomi` 为中心）；
  - **禁止** `atoms.pos` / `atoms.vel` / `atoms.forces`，**禁止**整个 `edges` 命名空间（截断邻接图随构象变，不是共价拓扑）。违者 `ValueError`。
- `pos`：`(C, N, 3)`，`C >= 1`，`N` 与 `Z` 一致；否则 `ValueError`。
- `group_id`：`str | int`，分子图 / 异构体身份（由调用方从数据集 `molecule_id` 传入；本类不哈希拓扑）。
- `targets`：可选扁平 `TensorDict`，`batch_size=[C]`，键子集为 `energy` / `forces` / `hessian` / `q`，形状见上表；未知键或形状不符 → `ValueError`。

构造时 `topology.clone()` 与 `pos.clone()`，束是坐标与拓扑的源。

**交替构造器（语义不同，不是 `__init__` 别名）。** `PesBundle.from_conformer_list(conformers, *, group_id)`：输入为同一分子的若干**已嵌套**构象 `TensorDict`（每个含 `atoms.pos`）。校验全部 `atoms.Z` 与共价索引 `torch.equal`；堆叠 `pos`；从第一份克隆拓扑并剥掉 `atoms.pos`、`atoms.vel`、`atoms.forces`、`atoms.q`、整个 `edges`、以及 `graphs.energy` / `graphs.cell` 等构象量；若各构象都带对应靶则堆进 `targets`。空序列或拓扑不一致 → `ValueError`（指明构象下标与键）。

**视图（均返回新 `TensorDict`，不把 encoder 写入写回束内存储）。**

| 方法 | 职责 |
|---|---|
| `topology_batch()` | 克隆拓扑；**任何路径都没有 `pos`**。供 `ChemEncoder` / `ClassicalIMM` 一次赋参。 |
| `conformation_batch(i)` | 克隆拓扑，写入 `atoms.pos = pos[i]` `(N, 3)`；若有靶则映射 `graphs.energy`、`atoms.forces`、`atoms.q`、`graphs.hessian` `(3N, 3N)`。越界 `IndexError`。供 `ff.to_potentials().calc_energy(batch, pos=batch["atoms", "pos"])`。 |
| `__iter__` / `__len__` | 依次 `yield conformation_batch(i)`；`len` 为 \(C\)。不是 DataLoader 适配器。 |
| `conformer_group_ids()` | 长度为 \(C\) 的元组，元素全是 `group_id`。供展开后再交给 `group_split_indices`。 |

只读属性：`group_id`、`n_atoms`、`n_conformers`、`pos`、`topology`、`targets`。

**划分（组合，不包一层 façade）。** 按束：

```python
train, val, test = group_split_indices([b.group_id for b in bundles])
```

按展开构象（每个束 \(C\) 个样本）：

```python
ids = [g for b in bundles for g in b.conformer_group_ids()]
train, val, test = group_split_indices(ids)
```

两种写法都必须让同一 `group_id` 的全部构象落在同一 split。`PesBundle` **不**提供 `split` / `make_*` 方法，不改 `group_split.py`。

**后续规格。** `universal-class-i-01`…`03` 将消费本类型；本规格不实现它们。

### Reuse decision

- `reuse group_split_indices` — 划分只调用现有函数；本规格不包装、不修改。
- `reuse center_by_group` — 不调用、不修改；相对能量留在 03。
- `reuse ChemEncoder` / `reuse ClassicalIMM` — 不调用、不修改；`topology_batch()` 的 schema 就是它们已接受的拓扑 `TensorDict`。
- `new — PesBundle` — librarian 确认 `molix.data` 没有「一图多构象」类型；`ClassicalIMM` 已是拓扑-only，本类型只做数据包装。

## Files to create or modify

- `src/molix/data/pes_bundle.py` (new)
- `src/molix/data/__init__.py`
- `tests/test_molix/test_data/test_pes_bundle.py` (new)
- `regressions/universal-class-i-04-bundle.py` (new)

## Tasks

- [ ] Write failing unit tests for PesBundle constructor, topology_batch, and conformation_batch (tests/test_molix/test_data/test_pes_bundle.py → TestPesBundle)
- [ ] Implement PesBundle in src/molix/data/pes_bundle.py
- [ ] Write failing unit tests for PesBundle.from_conformer_list and group_split_indices leakage on conformer_group_ids (tests/test_molix/test_data/test_pes_bundle.py → TestPesBundle)
- [ ] Implement PesBundle.from_conformer_list in src/molix/data/pes_bundle.py
- [ ] Export PesBundle from src/molix/data/__init__.py (__all__ Group-splits 段字母序：PesBundle 在 group_split_indices 之前)
- [ ] Add docstring per google with units in src/molix/data/pes_bundle.py
- [ ] Add regression example regressions/universal-class-i-04-bundle.py (public API only; hard-coded goldens, no third-party runtime)
- [ ] Run full check + test suite

## Testing strategy

单元测试只在 `tests/test_molix/test_data/test_pes_bundle.py`，类名 `TestPesBundle`，每测一个方法。

- **Happy path：** 3 原子链或水分子、\(C=2\)。`topology_batch()` 含 `atoms.Z` 与 `bonds.bond_index`，扁平键集合不含 `pos`；`conformation_batch(0)["bonds", "bond_index"]` 与 `conformation_batch(1)` `torch.equal`；两构象 `atoms.pos` 不相等且等于构造时传入的 `(C, N, 3)` 切片（位置数值容差 `1e-8`）；`list(bundle)` 长度 2。
- **Edge：** 拓扑带 `atoms.pos` 或 `edges` → `ValueError`；`pos` 不是 `(C, N, 3)` 或 \(C=0\) → `ValueError`；`conformation_batch` 越界 → `IndexError`；`from_conformer_list([])` → `ValueError`；第二构象 `bond_index` 不同 → `ValueError`；未知 `targets` 键 → `ValueError`；缺 `group_id` 不可构造。
- **Domain：** 10 个 `PesBundle`、各 2 构象、`group_id="m{k}"`。对 `[g for b in bundles for g in b.conformer_group_ids()]` 调用 `group_split_indices(..., ratios=(0.8, 0.1, 0.1), seed=0)`：唯一分子数 train/val/test = 8/1/1；三集合两两不相交；同一 `group_id` 的两个展开下标落在同一 split。相对能量不在本测试计算。
- **Regression：** `regressions/universal-class-i-04-bundle.py` — 水分子两构象（硬编码 Å 坐标，见 acceptance ac-008）；断言拓扑无 `pos`、两构象 `bond_index` 相同、`pos` 与字面量一致。不 import / subprocess 第三方。

## Out of scope

- Encoder、MM heads、losses、`center_by_group` 的实现或改动（01–03 后续消费本类型）。
- 训练脚本、DataModule / PackedCache / collate 改动、Dataset 适配器。
- MolHub / THEMol / OMol25 或任何数据集下载与 I/O。
- 力场、IR、电荷求解、Condenser、ff_export、PiNet。
- 生成热（HoF）、束内相对能量、按构象重新赋参的 API。
- `make_pes_bundle` 工厂、god context、`run_everything` 门面。
- 将截断邻接图（`edges`）当作拓扑；PBC `cell` 进拓扑。
