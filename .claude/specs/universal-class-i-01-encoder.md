---
title: Universal Class-I 共价图化学编码器
status: code-complete
created: 2026-08-13
grilled: true
---

# Universal Class-I 共价图化学编码器

## Summary

把现有 `molrep.chem.encoder.ChemEncoder` 原地泛化成 Universal Class-I 的共价图消息传递化学编码器：同一张共价分子图在任意构象下给出同一套连续化学表征，供 `ClassicalIMM` 继续从 `ChemEmbeddings` / `*.chem_features` 读参数。编码器只读拓扑（`atoms.Z` 与价键命名空间下标），不读 `pos`；原子潜变量 \(h_i\) 与键潜变量 \(h_{ij}\) 由默认 5 层（构造期可选 4/5/6，对应 S/M/L）共价图块迭代更新，键满足 \((i,j)\equiv(j,i)\)。宽度 `atom_dim` / `bond_dim` 由构造函数显式传入（S/M/L 推荐 128/256/384），本规格不新增 molzoo Spec。角 / 真二面角 / 非真二面角仍由 `AngleContext` / `ProperContext` / `ImproperContext` 在消息传递之后从原子特征生成。无能量、无受力、无 `molpot` 导入。

## Domain basis

可迁移 Class-I 感知是**局部** SMARTS/SMIRKS 化学环境，不是全局分子指纹。参数只依赖共价拓扑，不依赖坐标。

- Transferable Class-I perception is local SMARTS/SMIRKS (Mobley 2018 bioRxiv 286542). Espaloma GraphSAGE depth searched {2,3,4,5}; small depth keeps repeating residues stable (Wang 2022 arXiv:2010.01196 §5; Takaba 2023 arXiv:2307.07085 SI E).
- 3 hops already cover a torsion; 4–6 hops is extra functional-group context, not a physical law. Do NOT claim global attention is required. Do NOT claim "SAGE depth 3 = 4–6 hops."
- Encoder must not read coordinates (parameters are topology-only).

References:

- Mobley et al., bioRxiv 286542 (2018). https://doi.org/10.1101/286542
- Wang et al., “Espaloma”, arXiv:2010.01196 (2022) §5. https://arxiv.org/abs/2010.01196
- Takaba et al., “Espaloma-0.3”, arXiv:2307.07085 (2023) SI E. https://arxiv.org/abs/2307.07085

表征（无量纲学习嵌入，不经 `molix.units` 换算）：

\[
h_i^{(0)}=\mathrm{AtomChemEmbedding}(Z_i),\qquad
h_{ij}^{(0)}=\mathrm{BondContext}\bigl(h_i^{(0)},h_j^{(0)}\bigr)=h_{ji}^{(0)}
\]

\[
\bigl(h^{(\ell+1)},\,h_{\mathrm{bond}}^{(\ell+1)}\bigr)
=\mathrm{ChemGraphBlock}^{(\ell)}\bigl(h^{(\ell)},\,h_{\mathrm{bond}}^{(\ell)};\,\mathcal{E}_{\mathrm{cov}}\bigr),
\quad \ell=0,\ldots,n_{\mathrm{blocks}}-1
\]

有效感受野约为沿共价键的 \(n_{\mathrm{blocks}}\) hop（4–6）。三跳已覆盖一个扭转四元组；更深只提供官能团上下文，**不是**物理定律，也不要求全局全分子注意力。角 / 真二面角 / 非真二面角上下文只读 \(h_i^{(L)}\)：

\[
h_{ijk}=\mathrm{AngleContext}(h^{(L)}),\quad
h_{ijkl}=\mathrm{ProperContext}(h^{(L)}),\quad
h_{\mathrm{imp}}=\mathrm{ImproperContext}(h^{(L)}).
\]

数值对称公差沿用现有 chem 套件：`rtol=1e-5`、`atol=1e-6`。构象不变是恒等（不读 `pos`），回归用 `atol=1e-7`（期望 max abs diff = 0）。

## Design

**实体与所有权。** 继续以类型为中心，无工厂、无 god-context、无一次跑完的 façade。`ChemEncoder` 仍是唯一公共编码器；`compose` / `forward` / `write_batch` / `embeddings` 契约不变。

| 符号 | 角色 |
|---|---|
| `AtomChemEmbedding` | \(h_i^{(0)}\leftarrow Z\)；现有 `JointEmbedding` 路径。本规格不新增形式电荷等特征仓库；额外离散通道若以后需要，只加 `DiscreteEmbeddingSpec`。 |
| `BondChemEmbedding` | 键潜变量的对称原语。原地升级：可选 `h_bond` 作为上一层键潜变量，输出仍是同一 \(h_{ij}\) 张量，**不**新建 `h_bond` 类型。 |
| `BondContext` | 价键 IR 薄封装。`forward` 把可选 `h_bond` 转给 `BondChemEmbedding`，不改写 \((i,j)\leftrightarrow(j,i)\)。编码器用于种子 \(h_{ij}^{(0)}\)；块内消息也走它。 |
| `AngleContext` / `ProperContext` / `ImproperContext` | 消息传递**之后**的相互作用表征层。只消费更新后的原子特征；对称合同（角反转、真二面角反转、非真正中心 `atomi` + 外侧置换）一字不改。 |
| `ChemEmbeddings` | 五路 payload，字段与计数合同不变。 |
| `ChemGraphBlock`（新） | 一层共价图更新。学 `GCBlock` 的「一块 = 一次状态更新、`ModuleList` 堆叠、显式 `nn.Linear(..., dtype=config.ftype)`」，但实现只放在 `molrep.chem`，**禁止**写入 `molrep.interaction.pinet/`，也禁止 import pinet。 |
| `ChemEncoder` | 拥有 `atom_embed`、种子 `bond_context`、`nn.ModuleList` 的 `n_blocks` 个 `ChemGraphBlock`、以及三个高阶 context。 |

**构造函数（保持 keyword-only，与现有一致）。** 新增并保留：

```text
ChemEncoder(*,
    atom_dim=32, bond_dim=32, angle_dim=32, proper_dim=32, improper_dim=32,
    num_elements=119, hidden_dim=None, num_bond_types=0,
    n_blocks=5, local_attention=False)
```

- `n_blocks` 必须属于 `{4, 5, 6}`（S/M/L 深度），默认 `5`（medium）。其它整数 `ValueError`。
- `atom_dim` / `bond_dim` 由调用方传入。S/M/L **推荐宽度** 128 / 256 / 384；**不**引入 molzoo `Spec`，**不**增加 `ChemEncoder.small()` 之类工厂。现有默认 32 保留，以免无参的 `ChemPerception()` / 旧单测 silently 膨胀。
- `local_attention` 默认 `False`。为真时只对每个原子的**共价邻域**做 softmax，禁止 \(N\times N\) 全分子注意力、禁止图级 token。
- 既有 `angle_dim` / `proper_dim` / `improper_dim` / `num_elements` / `num_bond_types` 保留，`molzoo.chem.ChemPerception` 不改文件即可继承 `n_blocks=5`。

**生命周期（`compose`）。** 禁止读取 `atoms.pos` / `edge_diff` / `edge_dist`（`pos` 可以留在 batch 上，只是不读）。

1. `h_atom = self.atom_embed(Z)`。
2. 从 `bonds.atomi`/`atomj` 或 COO `bond_index`（现有 `_bond_endpoints`）取无向共价边 \(\mathcal{E}_{\mathrm{cov}}\)；`h_bond = self.bond_context(h_atom, atomi, atomj, bond_type=...)`。无键 → `(0, bond_dim)`，跳过块循环。
3. 对每个 `ChemGraphBlock`：`h_atom, h_bond = block(h_atom, h_bond, atomi, atomj, bond_type=...)`。`bonds` 命名空间是**无向不重复**对；每条键的潜变量向两个端点各 scatter 一次。
4. `AngleContext` / `ProperContext` / `ImproperContext` 读 **MP 之后** 的 `h_atom`（缺省命名空间仍写 `(0, D)`）。
5. 返回 `ChemEmbeddings`；`forward` → `write_batch` 仍写 `atoms/bonds/angles/propers/impropers.chem_features`。`ClassicalIMM` 的 `embeddings()` / `*.chem_features` 路径不改。

**`BondChemEmbedding` 迭代升级。** `forward(h_atom, atomi, atomj, bond_type=None, h_bond=None)`：`h_bond is None` 时保持现式 \(h_{ij}=\tfrac12(\mathrm{MLP}[h_i,h_j]+\mathrm{MLP}[h_j,h_i])\)（外加既有 type table），现有单测数值路径不变；若给出 `h_bond`，在对称 pair 上做残差混合（`pair + mix(h_bond)` 或等价、对端点对换不变的形式）。`mix` 若有参数则 `nn.Linear(bond_dim, bond_dim, dtype=config.ftype)`。禁止为键潜变量新建类型。

**`ChemGraphBlock.forward`。** 入参是张量，不是 context 袋：`h_atom (N, D_a)`、`h_bond (E, D_b)`、`atomi/atomj (E,)`、可选 `bond_type`。

- 键更新 / 消息：`h_bond = BondContext(...)(h_atom, atomi, atomj, bond_type=, h_bond=)`（内部即 `BondChemEmbedding`）。
- 原子更新：`local_attention=False` 时用 `index_add_`（与 `PILayer`/`IPLayer` 同族，但不 import pinet）把每条 `h_bond` 累加到两个端点，再残差 `h_atom ← h_atom + MLP(agg)`（两层 SiLU，隐宽 `hidden_dim` 或 `atom_dim`）。
- `local_attention=True`：由**已对称**的键特征打分（`Linear(D_b→1)`），在每个原子的关联键上做分段 softmax，再加权求和。同一无向键的两个方向共用一个分数。禁止全局 softmax、禁止非邻接原子对。
- 一块的感受野是 1 hop。堆叠 \(L\) 块 → \(L\) hop。单测用「改非邻接原子的 \(Z\)，一块之后 \(h_i\) 不变」钉死局部性。

**错误处理（贴近现编码器）。** 缺 `atoms` / `atoms.Z` → `KeyError`；非法 `bond_index` 形状 → `ValueError`；`n_blocks ∉ {4,5,6}` → `ValueError`；`atom_dim`/`bond_dim` ≤ 0 沿用 embedding 既有校验。新 `Linear`/`Embedding` 一律 `dtype=config.ftype`。`__all__` 按笔记字母序插入 `ChemGraphBlock`。

**包边界。** `src/molrep/chem/**` 不得 `import molpot`、不得 `import molrep.interaction.pinet`。现有 `TestNoMolpotImport` 已递归扫描该树，新 `blocks.py` 自动落入。

### Reuse decision

- `generalize ChemEncoder`（`src/molrep/chem/encoder.py:79`）— 原地泛化，不写第二套编码器。
- `generalize ChemEncoder.__init__` — 并入上一任务：增加 `n_blocks`、`local_attention` 与块堆叠。
- `reuse AtomChemEmbedding` — 仍只从 `Z` 产 \(h_i^{(0)}\)。
- `generalize BondChemEmbedding` — 升级为可迭代键潜变量（可选 `h_bond`），不新建类型。
- `reuse BondContext` — 种子 + 块内消息；仅把 `h_bond` 转给 embedding，不改写对称。
- `reuse AngleContext` / `ProperContext` / `ImproperContext` — MP 之后调用，对称一字不改。
- `reuse ChemEmbeddings` — 五字段合同不变。
- `reuse molzoo.chem.ChemPerception` — **本规格 Files 不列入、不修改**；默认参数继承即可。
- `pattern GCBlock` / `PILayer` — 只学堆叠与 `index_add_` 风格；块落在 `molrep.chem`，不进 `pinet/`。
- `reuse ClassicalIMM` — 不改；仍读 `embeddings()` / `*.chem_features`。
- `reuse` heads / IR / charge / losses / `Condenser` / `ff_export` — 本规格不修改。

## Files to create or modify

- `src/molrep/chem/embed.py`
- `src/molrep/chem/context.py`
- `src/molrep/chem/blocks.py` (new)
- `src/molrep/chem/encoder.py`
- `src/molrep/chem/__init__.py`
- `tests/test_molrep/test_chem/test_embed.py`
- `tests/test_molrep/test_chem/test_context.py`
- `tests/test_molrep/test_chem/test_blocks.py` (new)
- `tests/test_molrep/test_chem/test_encoder.py`
- `regressions/universal-class-i-01-encoder.py` (new)

## Tasks

- [x] Write failing unit tests for ChemGraphBlock (tests/test_molrep/test_chem/test_blocks.py → TestChemGraphBlock)
- [x] Write failing unit tests for BondChemEmbedding iterable latent, BondContext h_bond passthrough, and ChemEncoder stacked blocks (tests/test_molrep/test_chem/test_embed.py → TestBondChemEmbedding; tests/test_molrep/test_chem/test_context.py → TestBondContext; tests/test_molrep/test_chem/test_encoder.py → TestChemEncoder)
- [x] Generalize BondChemEmbedding in src/molrep/chem/embed.py and pass optional h_bond through BondContext.forward in src/molrep/chem/context.py without rewriting reverse symmetry
- [x] Implement ChemGraphBlock in src/molrep/chem/blocks.py (GCBlock stack style; messages via BondContext / BondChemEmbedding; local neighborhood attention default off)
- [x] Generalize ChemEncoder in src/molrep/chem/encoder.py to stack n_blocks in {4,5,6} ChemGraphBlock then AngleContext / ProperContext / ImproperContext; ignore pos
- [x] Export ChemGraphBlock from src/molrep/chem/__init__.py (alphabetized __all__) and add Google-style docstrings with tensor shapes (dimensionless embeddings) on new and changed public methods
- [x] Add regression example regressions/universal-class-i-01-encoder.py (public API only; hard-coded goldens, no third-party runtime)
- [x] Verify against same-graph two-conformation identity and reverse-bond bond-feature identity
- [x] Run full check + test suite
- [x] Hygiene: /mol:simplify on 01 chem surface (no apply-able residue)

## Testing strategy

单测只在 `tests/`，路径镜像 `src/`，一类一 `Test*`，一测一方法。禁止在 `tests/` 写 e2e。`torch.manual_seed` 固定；公差：对称 `rtol=1e-5`、`atol=1e-6`；构象恒等 `atol=1e-7`。

- **ChemGraphBlock（`TestChemGraphBlock`）**
  - Happy：`forward` 保持 `(N, atom_dim)` / `(E, bond_dim)`。
  - 对称：交换 `atomi`/`atomj` 后 `h_bond` allclose。
  - 局部：一块之内，改非邻接原子的输入，目标原子输出不变；`local_attention=True` 仍只在共价邻域归一化（不是对全部 E 条键做一次 softmax）。
  - 默认：`local_attention` 为 `False`；构造函数不接收 `pos`。
- **BondChemEmbedding / BondContext**
  - `h_bond=None`：现有 shape / 端点对称 / 空键用例仍过。
  - `h_bond` 给定：输出 `(E, bond_dim)`，端点对换仍 allclose。
- **ChemEncoder（扩展既有 `TestChemEncoder`）**
  - Happy：`n_blocks=5`（默认）以及 4、6 都能 `forward`，五路 `*.chem_features` 计数与 `mini_batch` 一致（4/3/2/1/1）。
  - 宽度：`atom_dim=bond_dim` ∈ {128, 256, 384} 可构造（不经过 molzoo Spec）；单测用小宽度跑数值。
  - Edge：`n_blocks=3` 或 `7` 抛 `ValueError`；缺 angles/propers/impropers 仍 `(0, D)`；空键跳过块循环。
  - Domain：同一 `Z`+价键下标、两套不同 `atoms.pos` → 五路特征 allclose（`atol=1e-7`）；反转 `bonds.atomi`/`atomj` → `bonds.chem_features` allclose。
  - 边界：既有 `TestNoMolpotImport` 继续扫描整个 `src/molrep/chem`（含新 `blocks.py`）；`forward` 不写 energy/force 键。
- **回归示例** `regressions/universal-class-i-01-encoder.py`：公共 API only。固定 `torch.manual_seed(0)`，复用现有四原子迷你拓扑（O/H/H/C，3 键 / 2 角 / 1 proper / 1 improper）。硬编码断言（无第三方、无 subprocess）：
  1. 两套 `pos`（全零 vs 随机）→ 五路 `ChemEmbeddings` 逐叶 `allclose(atol=1e-7)`，期望 max abs diff = 0。
  2. 反转键端点 → `bonds.chem_features` `allclose(rtol=1e-5, atol=1e-6)`。
  3. 形状字面量：atom `(4, 8)`、bond `(3, 8)`、angle `(2, 8)`、proper `(1, 8)`、improper `(1, 8)`（脚本内 `atom_dim=bond_dim=8`，`n_blocks=5`）。
  4. 无 energy/force 键。
  页眉注释：capture command `PYTHONPATH=src python regressions/universal-class-i-01-encoder.py`、date `2026-08-13`、torch 版本、device/precision。

## Out of scope

- Fourier 参数头、损失、PES bundle、THEMol、LJ、电荷均衡 / `molpot.charge`。
- `molzoo.chem.ChemPerception` / `ChemPerceptionSpec` 与任何 molzoo recipe（本规格不改这些文件）。
- `scripts/`、训练循环、数据集。
- 能量 / 受力 / 任何 `molpot` 导入。
- 全局全分子注意力、图级 token、几何邻表（`edges.edge_index` / `pos`）。
- 新的形式电荷（或其它离散通道）特征仓库；不扩展 `AtomChemEmbedding` 的 v1 Z-only 合同。
- `ClassicalIMM`、IR、`Condenser`、`ff_export`、SMARTS。
- 把块放进 `molrep.interaction.pinet/`，或从 chem 引用 pinet 符号。
