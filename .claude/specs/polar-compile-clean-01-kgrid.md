---
title: GTO k 空间固定容量网格与周期分支 compile-clean 化
status: code-complete
grilled: true
created: 2026-09-03
slug: polar-compile-clean-01-kgrid
chain: polar-compile-clean
---

# GTO k 空间固定容量网格与周期分支 compile-clean 化

## Summary

把 `molpot.potentials.gto` 的**周期分支**（给了 `cell`、单体系 B=1，也就是 MD 热路径）从"数据依赖形状"改成"固定容量 + 乘性掩码"，使它在 `torch.compile` 下达到 1 图 0 断点。今天 `kspace.py:82-129 half_sphere_kvectors` 用 `.item()` 定 nk（`:101` 的推导式里三次读数）、再用布尔掩码选 k 点（`:118-127`），k 向量个数随 cell 变化；`hartree.py:209,219,281` 与 `features.py:204` 用 `int(batch.max().item())+1` 取 `num_graphs`；`features.py:213-217` 每次前向都读一次 `float(self.combined_sigma.min())`。本步用一个**只在第一次 bind 时定尺**的整数半盒 `KSpaceGrid`（形状只依赖容量、与 cell 无关，带自包含的 in-graph 溢出哨兵）、一个显式 `num_graphs` 关键字、以及构造期一次算好的 `k_cut` 替换它们，并把 `MacePolar.energy_core` 已有的 `num_graphs` 真正下传给两个 GTO 子模块（`polar.py:241-246` 与 `:266` 今天只传 `cell`，是纯接线缺口）。**孤立分子的批量全对枚举原样不动**（见 Design 的"保留的断点"），周期路径的数值与今天的 eager 路径在 fp64 下逐项一致。

## Domain basis

- **Hartree 能量（Eq. 40–42, SI V.6）**：\(E=\Omega/(2(2\pi)^6)\sum_{k\neq0}(4\pi/k^2)|\tilde\rho(k)|^2\)，\(\tilde\rho(k)=((2\pi)^3/\Omega)\sum_{i,lm}p_{i,lm}\tilde\varphi_{lm}(k)e^{-ik\cdot r_i}\)，\(\tilde\varphi_{lm}(k)=C_l(-i)^lY_{lm}(\hat k)I_{l,\sigma}(k)\)。取 \(l\le1\)、\(I_{0,\sigma}=e^{-\sigma^2k^2/2}\) 后化为 `kspace.py:13-14` 里的
  \(E=\mathrm{eV\_A}\cdot(2\pi/\Omega)\sum_{k\neq0}e^{-\sigma^2k^2}k^{-2}|S(k)|^2\)，\(S(k)=\sum_i(q_i+i\,k\cdot\mu_i)e^{ik\cdot r_i}\)；实空间搭档是 `erf(r/(2σ))/r`（`kspace.py:9-11`）。参考：Batatia et al., arXiv:2602.19411, §II.2.6b 与 SI V.6。
- **共轭对称（D2）**：q、μ、r 实值 ⇒ \(S(-k)=\overline{S(k)}\)，故 \(|S(-k)|^2=|S(k)|^2\)；在 \(\{n,-n\}\) 中任取其一并配权 2 得同一总和。选择规则 = "第一个非零整数分量 > 0"（`torch.argmax` 返回首个最大索引，`kspace.py:122-125`），n=0 自动得权 0。
- **固定盒等价（D3）**：取 \(n\in[-N_c,N_c]^3\)，\(w(n)=2\cdot\mathbb 1[\text{first nonzero}>0]\cdot\mathbb 1[0<|k(n)|^2<k_{\rm cut}^2]\in\{0,2\}\)，\(k(n)=\sum_i n_i b_i\)，\(b_i\) 为 \(2\pi\,\mathrm{cell}^{-\mathsf T}\) 的行；除法前先做保护分母 \(\tilde k^2=\mathrm{where}(|k|^2>0,|k|^2,1)\)。逐项等价；padding 贡献精确 `+0.0`（IEEE `x+0.0=x`）。k=0 绝不可先除后乘（`1/0=inf`，`0·inf=NaN` 同时毒化数值与梯度）。截断用严格 `<`，与 `kspace.py:118` 一致。
- **容量定界、溢出判据与它的保守边（D4）**：\(a_i\cdot b_j=2\pi\delta_{ij}\Rightarrow 2\pi|n_i|=|a_i\cdot k|\le\|a_i\||k|\)，故 \(|k|<k_{\rm cut}\Rightarrow|n_i|<x_i:=\|a_i\|k_{\rm cut}/2\pi\)（严格小于，因为截断本身是严格的），对三斜胞精确。容量取 \(N_{c,i}=\lceil f\cdot x_i(t_0)\rceil\)，必须用**实值** \(x_i\)（\(\lceil f\cdot n_{\max}\rceil\) 会偏小：x=2.9、f=1.5 时 floor(4.35)=4 > ceil(3)=3）。溢出判据 \(\lfloor x_i(t)\rfloor>N_{c,i}\) **同一条式子两种算法**：eager 用 Python float 算并抛错，编译区内用张量算（\(N_c\) 是 Python int 元组 ⇒ 张量常量），`weights = where(overflow, nan, weights)`。
  **严格不等号的保守边（写进 docstring，不要"修"）**：因为界是 \(|n_i|<x_i\)，盒子足够的充要条件是 \(N_{c,i}\ge\lceil x_i\rceil-1\)。当 \(x_i\) 恰为整数（零测集）时 \(\lfloor x_i\rfloor=x_i\)，判据在 \(x_i=N_{c,i}+1\) 处**提前一档**触发：其实没有任何 k 点会被截断，但网格仍然拒绝。这是**故意的保守 fail-closed**，不要改成 `>=`（那会触发得更早），也不要换成 \(\lceil x\rceil-1>N_c\) 的第二种写法（同一条界的第二个家）。eager 与图内用**同一条** `floor(x) > n_cap`，两侧于是在同一事件上失败。
  NPT 收缩恒不溢出；膨胀比 \(\lambda\le f\) 恒不溢出。**越界的网格永远不截断 —— eager 抛错，图内返回 NaN。**
- **半盒而非半球，以及 k 点膨胀的代价（D5）**：全盒 \((2N_c+1)^3\) 对半球 \(\approx\frac23\pi n_{\max}^3\) 的渐近比是 \((2f)^3/(4\pi/3)\)，f=1.2 时 3.30。**半球列表不是三斜胞的安全静态超集**（k 球在 n 空间是依赖应变的椭球），只有盒（或容量胞椭球×f）有保证，所以必须用整数半盒（\(M=(\prod_i(2N_{c,i}+1)-1)/2\)，first-nonzero>0，原点已剔除；整数三元组与 cell 无关，只有 w(n) 依赖 cell）。**实测膨胀**：PEO 盒（46.36 Å 立方、k_cut 1.8467166200173746）今天的半球 M = 5318；\(x_i=13.63\)，f=1.2 ⇒ \(N_c=17\) ⇒ \(M=(35^3-1)/2=21437\)（4.0×）；f=1.0 ⇒ \(N_c=14\) ⇒ \(M=(29^3-1)/2=12194\)（2.3×）。小胞更差：10 Å 立方 \(x_i=2.939\)，f=1.2 ⇒ \(N_c=4\) ⇒ M=364，而半球只有 46（7.9×）—— ceil 在小 x 上的相对开销大。`structure_factor` / k 空间势是 O(N·M)：N=4995 时单个 (N,M) fp32 张量从 101.3 MiB（M=5318）涨到 **408.5 MiB**（M=21437，4.28e8 B）；`structure_factor` 同时活着 `kr`/`cos_kr`/`sin_kr` 三张，约 +0.9 GiB fp32。默认 f=1.2 是给 NPT 留头寸（与 `regressions/mace-polar-periodic-gto.py:110` 的 `NeighborList(capacity_factor=1.2)` 同源）；**NVT 调用方可以传 1.0**。步时与峰值显存由 03-wire 在 GH200 实测（k 空间只占 0.38 s/step 的一小部分，见 `.claude/notes/notes.md:231-236`）。
- **应变导数不变（D7）**：\(pos_s=pos(I+\varepsilon)^{\mathsf T}\)、\(cell_s=cell(I+\varepsilon)^{\mathsf T}\Rightarrow k_s=(I+\varepsilon)^{-\mathsf T}k\)、\(k_s\cdot r_s=k\cdot r\)、\(\Omega(\varepsilon)=\det(I+\varepsilon)\Omega\)。半球/非零指示只依赖整数 n ⇒ \(\partial w/\partial\varepsilon=0\)；截断指示分段常数（只在零测集 \(|k_n(\varepsilon)|=k_{\rm cut}\) 上变）。两种写法都从比较处取零梯度，故 \(\partial E/\partial\varepsilon\) 一致。跨界跳变上界 \(2\,\mathrm{eV\_A}(2\pi/\Omega)e^{-\sigma^2k_{\rm cut}^2}k_{\rm cut}^{-2}|S|^2\)，σ=1.5、k_cut=1.8467 时标量因子 1.364e-4 —— 这是硬 k 截断的固有性质，不是回归。
- **孤立分支的成对枚举（D6，本步不改）**：`_skip_diagonal`（`hartree.py:69-77`）已是静态 N(N−1) 且无对角线（r=0 会让 \(\partial\|x\|/\partial x\) 成 NaN）；`_skip_equal_orig_regular`（`:80-93`）结构性只枚举 16N(N−1) 条活边。把它们改成"整批全对 + 乘性掩码"会把活边数从 16N(N−1) 抬到 \((4Bn)^2\)：B=128、n=20 时是 1.05e8 对 8.19e5，**训练路径上放大 128 倍**，且 `GTOFieldFeatures._pair_features` 在每次场更新里被调两次（`polar.py:240-246`，循环 `num_recursion_steps` 次）。因此本步保留这条路径原样，把它记为**已命名的保留断点**，由 `polar-compile-clean-04-pairs` 用块对角枚举解决（见 Design）。
- **常数（D8）**：`eV_A = 14.399645478425667` eV·Å/e²（=1/(4πε₀)），FIELD = 4π·eV_A = 180.95128179727823。κ 代码默认 1.0（`hartree.py:184`、`features.py:70`），已核对等于官方 checkpoint 的 `kspace_cutoff` buffer：σ=1.5、l_max=1 时 k_cut = 1.8467166200173746 Å⁻¹，\(x_i=0.29391\|a_i\|\)。`GTOFieldFeatures` 进入 k_cut 的 σ 是 `min(density_sigma, min(σ_comb)) = min(1.5, 1.5) = 1.5`（σ_comb=[1.5, 2.3717082451262845]，`features.py:82`），故两条路径共用同一 k_cut。**`k_cutoff = κ·2.25/σ_min·(l_max+1)^0.3`（`kspace.py:53`）是参考实现的约定**（`graph_longrange` 的 `gto_basis_kspace_cutoff = 0.75·3/σ_min·(l+1)^0.3`），论文里没有这条式子；模块 docstring 必须按"代码约定"而非"论文方程"署名。单位：电荷 e、位置 Å、能量 eV、k 为 Å⁻¹。

## Design

**`KSpaceGrid`（`src/molpot/potentials/gto/kspace.py`，普通类，非 `nn.Module`）。** 沿用 `NeighborList` 的形态理由（`neighbors.py:391-397`）：它是运行期几何缓存，不是模型参数，不进 `state_dict`。

- `KSpaceGrid(k_cutoff: float, *, capacity_factor: float = 1.2, max_axis_index: int = 32)` —— 直接构造，无 `make_*`。`max_axis_index` 是**每轴**上限（今天 `kspace.py:104` 那个 `_max_nk = 32` 的具名化），不是总点数。`capacity_factor` 是具名关键字而非内联字面量（照 `molix/units.py:102 DEAD_EDGE_CUTOFF_FACTOR` 的意图）；1.2 取自 `regressions/mace-polar-periodic-gto.py:110`，`NeighborList` 的类默认 1.35（`neighbors.py:479`）不在此复用。
- `.bind(cell) -> None` —— **eager 专用**，唯一允许 host sync 的入口。第一句是 `cell = _as_cell(cell)`。
  - **只有第一次调用定尺**：由 \(x_i=\|a_i\|k_{\rm cut}/2\pi\) 定 \(N_{c,i}=\lceil f\,x_i\rceil\)，建整数半盒 `n (M, 3) int64`，并记下 \(N_c\)。`max(N_c) > max_axis_index` 时抛现有的 `RuntimeError("GTO k-grid exploded: …")`（原文照搬 `kspace.py:106-111`，把 `nk` 换成 `N_c`、把 `_max_nk` 换成 `max_axis_index`）。
  - **之后每一次调用都只是 `check_capacity(cell)`，永不重新定尺**：cell 缩小或仍在容量内 ⇒ 原样接受、整数盒对象不变（整数三元组与 cell 无关，只有 `vectors` 里的 w(n) 依赖 cell）；越容量 ⇒ 抛 `RuntimeError`。这与 `NeighborList` 同形：容量在构造时算一次（`neighbors.py:552`），此后溢出永远抛错（`neighbors.py:838-842`），而不是偷偷放大缓冲区。**若真要换尺寸，就构造一个新的 `KSpaceGrid`** —— 类 docstring 明写这句。本类**不提供** `rebind` / `set_cell`：树内零消费者，按"第二次真实使用前不抽取"不造入口。
  - 直接后果：`check_capacity` 不再是恒真的装饰（旧稿的 `bind` 会顺手改尺寸，使它永远不可能失败），**eager 与编译区在同一事件上失败** —— 一个抛 `RuntimeError`，一个给 NaN。
- `.check_capacity(cell) -> None` —— eager 专用；第一句同样是 `_as_cell(cell)`。\(\lfloor x_i\rfloor>N_{c,i}\) 抛 `RuntimeError`，消息**同时点名 `KSpaceGrid` 与 `capacity_factor`**：`NeighborList._write`（`neighbors.py:838-842`）的溢出消息用的是同一个词，测试与用户必须能一眼分辨是哪一个容量爆了。措辞其余部分对齐那条消息。严格不等号的保守边见 D4，写进 docstring。
- `.vectors(cell) -> (kvec (M,3), k_sq_guarded (M,), weights (M,))` —— 第一句仍是 `_as_cell(cell)`（**这条不能省**：`_write_energy` 从 `graphs.cell` 取到的就是 `(1,3,3)`，见 `regressions/mace-polar-periodic-gto.py:93`；`g = 2π·inv(cell).T` 直接吃 `(1,3,3)` 会静默给出垃圾）。`_as_cell` 里那条 `"GTO k-space is single-system"` 的 `ValueError`（`kspace.py:70-79`）随之保留；持有者里 `n_graphs > 1` 的 `ValueError`（`hartree.py:282-286`、`features.py:205-209`）也保留 —— 两道检查针对两件不同的事（cell 形状 / batch 图数）。
  其余**全程可 trace，形状恒为 M**：`g = 2π·inv(cell).T`；`kvec = n.to(dtype) @ g`；`k_sq = (kvec*kvec).sum(-1)`；`weights = 2.0 * ((k_sq > 0) & (k_sq < k_cut²)).to(dtype)`；`k_sq_guarded = where(k_sq > 0, k_sq, ones)` **先保护再除**。布尔量只作为**乘性数值**使用，绝不作索引（照 `molix/engine/static.py:93-95` 的 where-中和 padding 做法；照 `elec/lib/kspace_filter.py:253-254` 的 divide-safe 做法）。`k_sq_guarded` **只有作为除数才有效**，任何把它当 \(|k|^2\) 读的用法都是错的；`weights` 是**唯一**的掩码来源。
  - **in-graph 溢出哨兵（自包含，无 host sync）**：`x = norm(cell, dim=1) * (k_cut / 2π)`（张量）；`overflow = (torch.floor(x) > n_cap).any()`，`n_cap` 由构造期的 Python int 元组变成张量常量；`weights = torch.where(overflow, weights.new_full((), float("nan")), weights)`。docstring 写明：**越界的网格永远不截断，它返回 NaN** —— 能量、力、virial 一起变 NaN，失败是响的、fail-closed 的，且不需要把标量读回 host。
  - **未绑定即调用**抛具名 `RuntimeError("KSpaceGrid is not bound — call bind_cell(cell) before compiling or exporting")`。判据读的是普通 Python 属性（`self._n is None`），因此在 **trace 期**就触发，报出的是这句话而不是一个莫名其妙的 matmul 形状错。
  - **设备不匹配**抛具名 `RuntimeError`，指明整数盒在哪个设备、cell 在哪个设备、以及修法（`.to(device)` 或重新构造）。
- `.to(device) -> KSpaceGrid` —— 照 `NeighborList.to`（`neighbors.py:1218-1264`）：**原地改自己的张量并 `return self`**（`neighbors.py:1264`），含一段 `Warning:` docstring（照 `:1230-1238`）说明"移动会重建整数盒张量，之前取到的 `vectors` 输出仍指向旧设备"。持有者用两行 `_apply` 覆写转发（`GTOHartreeEnergy` / `GTOFieldFeatures` 各自两行，不抽公共基类 —— 只有两个调用点，按"第二次真实使用前不抽取"）。**明确否决 `register_buffer(..., persistent=False)` 方案**：整数盒的 M 由容量决定而非由 checkpoint 决定，重建策略不属于模块树，而且 buffer 会跟着 `state_dict` 的 shape 检查走。
- 归属与生命周期：`GTOHartreeEnergy` 与 `GTOFieldFeatures` 各持一个普通属性 `self.kgrid`（不是子模块），并各暴露一个公开方法 **`bind_cell(cell: torch.Tensor) -> None`**（eager：`self.kgrid.bind(cell)`）。两者的 `forward` 在 `cell is not None` 分支里先做
  `if not torch.compiler.is_compiling(): self.bind_cell(cell)`，再 `self.kgrid.vectors(cell)`。`torch.compiler.is_compiling()` 在 trace 期是编译期常量，dynamo 直接折掉该分支 —— 零断点，同时 eager 调用方（现有测试 / regressions）签名不变。**这是 `torch.compiler.is_compiling()` 在 `src/` 下的首次使用**（全树 grep 无匹配）；03-wire 复用同一写法做元素门。一个 eager MD 步在 Polar 上调 `bind` 的次数是 `1 + 2·num_recursion_steps`（Hartree 一次 + 每轮场更新两次描述子），dump 默认 `num_recursion_steps=2` ⇒ 每步 5 次；第一次之后每次只读 3 个 float 做容量检查。
  **生命周期契约（写进两个类的 docstring）**：构造 → `.to(device)` → `bind_cell(cell)`（eager，定尺）→ 编译 / 导出 → 每步 `forward`（eager 侧顺带查容量）。跳过 `bind_cell` 直接编译，报的是上面那句具名错误。
- `half_sphere_kvectors`（`kspace.py:82-129`）整只删除 —— 它就是缺陷本体，留着即静默债务。`hartree_energy`（`:161-191`）与 `electrostatic_potential`（`:194-239`）**各只有一个生产调用方**（`hartree.py:288` 与 `features.py:218`），按"只有一个树内调用点就不抽取"折进各自的持有者：`GTOHartreeEnergy._kspace_energy` 吸收 `hartree_energy`，`GTOFieldFeatures._kspace_potential`（由 `_kspace_features` 调用）吸收 `electrostatic_potential`。`structure_factor`（`:132`）、`kspace_cutoff`（`:36`）、`cartesian_dipole`（`:56`）、`_as_cell`（`:68`，升为 `KSpaceGrid` 的内部第一句）、`KSpaceGrid` 保持在模块内。`gto/__init__.py:16,23` 去掉 `hartree_energy`、加上 `KSpaceGrid`。`if kvec.numel() == 0: return zeros` 早退（`:186`、`:224`）一并删除 —— 固定容量下 M 恒 >0，k_cut 过小时全零权重给出精确 0.0，分支多余。
- `k_cut` 只算一次：`GTOFieldFeatures.__init__` 里 `density_sigma` / `feature_widths` / `comb`（`features.py:82`）全是 Python float，`k_cut = kspace_cutoff(min(density_sigma, min(comb)), max(density_lmax, feature_lmax), κ)` 在那里求值并交给 `KSpaceGrid(k_cutoff=...)`；`features.py:213-217` 那次 `float(self.combined_sigma.min())` 消失，`_kspace_features` 不再读任何 host 标量。`GTOHartreeEnergy` 同理（`k_cutoff` property `hartree.py:197-200` 保留为只读展示，构造期把值交给网格）。
- **buffer 归属修正（一处两家的消除）**：`combined_sigma`（`features.py:83-85`）、`l0_factors`（`:90`）、`l1_factors`（`:98`）、`self_overlap_l0`（`:108`）、`self_overlap_l1`（`:113`）今天都是 `persistent=True`，也就是**可以被 `load_state_dict` 覆盖**；而同样由构造参数导出的 `k_cut` 在本步之后是冻结在 ctor 里的 Python float。同一组事实两个家，checkpoint 一改就静默不一致。五个 buffer 一律改 `persistent=False`（它们百分之百由 `__init__` 的 Python 参数算出，官方 checkpoint 从不提供它们；`CheckpointRemap` 容忍缺失 buffer）。已核对树内**没有任何测试断言它们出现在 `state_dict` 里**（`tests/test_molpot/test_potentials/test_gto/test_features.py:87-94,116,135-137` 都是直接读属性）。`GTOHartreeEnergy.self_overlap`（`hartree.py:195`）不在本次改动范围内，保持 `persistent=True`（本步不碰它，也不因"顺手"而改一个未经 checkpoint 核对的键）。

**`num_graphs` 显式化。** 按 molpot 既有约定（`nonbonded.py:61-79` 的 `_reduce_pair_energy`：`.item()` 只在 `None` 分支，`:74-75`；公开签名见 `:123`；另见 `lj126.py:81`、`polarization.py:138`）加 `num_graphs: int | None = None` 到 `GTOHartreeEnergy.forward` 与 `GTOFieldFeatures.forward`，并下传到 `_pair_energy` / `_self_energy` / `_kspace_energy` / `_pair_features` / `_kspace_features`。`int(batch.max().item())+1` 只保留在 `None` 分支并在 docstring 标注 **eager-only**。`_kspace_energy` 末尾的 `new_zeros(max(n,1)); out[0]=…`（`hartree.py:296-298`）换成 `energy.reshape(1)`（单体系已由 `:282-286` 的 ValueError 保证）。三处 `scatter_sum`（import 在 `hartree.py:22`、`features.py:25`；调用点 `hartree.py:215-216,223` 与 `features.py:139`）改用 `molix.F.scatter.scatter_sum_compile_safe`（`scatter.py:56`，签名 `(src, index, dim_size)`，dim 0；三个调用点本就是 dim 0 且已传 `dim_size`）—— `scatter_sum` 的 `dim_size=None` 路径在 `scatter.py:35` 做 `.item()`。**注意这次替换换掉的是归约内核**（`scatter_add_` → `index_add_`，`scatter.py:46` vs `:67-` 的 docstring），所以孤立分支的数值不变契约是 `allclose(atol=1e-12)` 而**不是**逐位相等；四个成对枚举函数的源码则是逐字节不变（见下）。

**`MacePolar.energy_core` 接线（`src/molzoo/mace/polar.py`）。** `energy_core` 已收 `num_graphs: int`（`polar.py:181-193`），只是没往下传，`num_graphs` 的管道在这里断掉。三处调用改为 `self.electric_potential_descriptor(..., cell=cell, num_graphs=num_graphs)`（`:241-243`、`:244-246`）与 `self.coulomb_energy(..., cell=cell, num_graphs=num_graphs)`（`:266`）。方向合法：`molzoo → molpot` 已经存在（`polar.py:24,26`）。**编译路线的假设是 B=1**：`num_graphs` 是 trace 期常量，每个 B 特化出一张图；周期分支本就由 `ValueError` 强制 B=1，孤立分支的多图仍走 eager。`MacePotential.energy_core`（`potential.py:349-362`）的同名契约与 `:56-65` 的理由段是本步照抄的样板。

**保留的断点（点名，不静默）。** 孤立分子的批量全对枚举 —— `_complete_directed_pairs`（`hartree.py:96-135`，`.item()` ×2 + `torch.equal` + 布尔掩码索引）与 `cached_complete_directed_pairs`（`:138-151`，`.item()` ×2 + host 侧 cache key）—— **本步一行不改**，理由见 D6。`_skip_diagonal`（`:69-77`）/ `_skip_equal_orig_regular`（`:80-93`）同样不动。这条断点记入 `.claude/notes/notes.md` topic `compile-no-graph-breaks`，owner = molpot gto，后继链节 **`polar-compile-clean-04-pairs`**，设计草图一并记录：*以构造期的 `max_atoms_per_graph` 容量做块对角逐图枚举，规模 O(B·n²·copies²)，索引全由 `arange` 算术得出，超容量在 bind 时抛错 —— 就是 `NeighborList` 的固定容量惯用法*。测试对这条分支断言"**仍然**有断点"，避免它退化成无意的静默契约。

### Reuse decision

按 librarian 报告逐条裁决（13–19 归属本链后两步，在此点名而不实现）：

- **(1) `elec/kernels/multipole_ewald.py:221-245 _build_kvec_grid` — `new`（generalize 已延期，记录双向触发条件）。** 同算法同三缺陷（`.item()` at `:235`、布尔掩码切片 at `:244-245`、半球掩码 at `:352-370`），但它与 multipole Ewald 核以及 `enumerate_kvec_indices:247-255` 的 FD-stress keep-set 冻结耦合，迁移需要它自己的 parity 证据。本步按 librarian 的 placement 建议把 `KSpaceGrid` 落在 `gto/kspace.py`。**两个方向的提升触发器都写进 notes**：(a) `multipole_ewald` 迁过来成为第二消费者时，`KSpaceGrid` 提升到 `src/molpot/potentials/elec/lib/kvectors.py`；(b) 反过来，若 `KSpaceGrid` 在 `multipole_ewald` 迁移之前先拿到第二个消费者，合并同样当场到期 —— 两条都不许静默拖着。
- **(2) `elec/lib/kvectors.py:23 _generate_kvectors` / `:88 generate_kvectors_for_ewald` — `new`（已先读再判）。** 它是树内唯一的定形无掩码 k 生成器，但按 FFT 频率排序、且 `ns` 是喂给 `fftfreq` 的张量，无法给出"半盒 + first-nonzero>0"的次序。`KSpaceGrid` **模仿**它的契约（整数网格入、`(M,3)` 出、不做掩码选择），不新写第三种排序语义。
- **(3) `elec/lib/kspace_filter.py:253-254 divide-safe where` — `pattern`（照抄）。** 保护分母先于除法。
- **(4) `kspace_filter.py:89 KSpaceFilter.update(cell, ns_mesh)` — `pattern`（照抄其"带外重算"的位置，不照抄"改尺寸"）。**
- **(5) `molix/md/neighbors.py:390 NeighborList` — `pattern`（照抄五处）。** 容量**只定一次**且此后溢出恒抛错 `:552` + `:838-842`、溢出 `RuntimeError` 点名 `capacity_factor` `:838-842`（本步再加上 `KSpaceGrid` 以便区分）、"不是 `nn.Module`…不进 state_dict" 理由 `:391-397`、`to()` 原地改并 `return self` 加 `Warning:` 块 `:1230-1264`、`scale_isotropic`（`:1012`）"缩放不重建"的语义。
- **(6) `molix/engine/static.py:36 StaticForward` — `pattern`（照抄 `:93-95`，不 import）。** padding 用 `where` 算术中和。
- **(7) `molix/units.py:102 DEAD_EDGE_CUTOFF_FACTOR` — `pattern`（照抄意图）。** 惰性行的常数具名，不内联；此处具名为 `KSpaceGrid.__init__` 的 `capacity_factor` / `max_axis_index`（k 网格参数属物理侧，不搬进 `molix.units`）。
- **(8) `molix/batch/pad.py:36 PadMolecularBatch` — `no reuse`。** 它 pad 的是批维，不是 k 点维。
- **(9) `nonbonded.py:61-79 _reduce_pair_energy` 的 `num_graphs: int | None = None` — `reuse`（原样沿用，含 `.item()` 只在 `None` 分支的隔离方式）。**
- **(10) `molzoo/mace/potential.py:349 energy_core` 的 `num_graphs` 契约（理由段 `:56-65`、被隔离的唯一 host sync `:612-615`）— `reuse`（本步落地）。** 照抄到 `MacePolar` 的接线缺口 `polar.py:241-246, :266`，不新建并行入口。
- **(11) `molix/F/scatter.py:56 scatter_sum_compile_safe` — `reuse`。** 替换 `scatter_sum`（`scatter.py:35` 的 `.item()`）。
- **(12) `hartree.py:69 _skip_diagonal` / `:80 _skip_equal_orig_regular` / `:96 _complete_directed_pairs` / `:138 cached_complete_directed_pairs` — `reuse`（源码零改动，本步不碰）。** 原判决是 `generalize`，被 architect 的 🚨 推翻：整批掩码会把训练路径的成对张量放大两个数量级（D6）。改判为保留 + 点名断点 + 后继链节 `polar-compile-clean-04-pairs`。
- **(20) `tests/test_molzoo/test_mace/test_potential.py:740-748` 与 `tests/test_molix/test_md/test_compile.py:17 _BACKEND = "aot_eager"` — `pattern`（照抄）。** explain→(1 图, 0 断点) 的断言写法与 CPU `aot_eager` 后端。
- **(15)(17)(18) → 03-wire；(13)(14)(16)(19) → 02-seam。** 本步不碰。

**关于跨包**：Files 触及 `molpot` 与 `molzoo` 两个包。这不是范围失控 —— `molzoo` 那一处是同一条 `num_graphs` 契约的三个关键字，architect 明确要求把它并进本步（否则 `num_graphs` 的管道在 01 结束时仍然断在 `polar.py`，本步的 explain→0 验收无从落地）。链本身就是拆分结果。

## Files to create or modify

- src/molpot/potentials/gto/kspace.py
- src/molpot/potentials/gto/hartree.py
- src/molpot/potentials/gto/features.py
- src/molpot/potentials/gto/__init__.py
- src/molzoo/mace/polar.py
- tests/test_molpot/test_potentials/test_gto/test_kspace.py
- tests/test_molpot/test_potentials/test_gto/test_hartree.py
- tests/test_molpot/test_potentials/test_gto/test_features.py
- tests/test_molzoo/test_mace/test_polar.py
- .claude/notes/notes.md
- regressions/gto-kspace-fixed-grid.py (new)

## Tasks

- [x] Write failing unit tests for KSpaceGrid first-bind sizing, the no-resize-on-rebind rule, half-box shape, 0/2 weights, the NaN overflow sentinel, the (1,3,3) cell contract and the named unbound/device/overflow errors (tests/test_molpot/test_potentials/test_gto/test_kspace.py → TestKSpaceGrid)
- [x] Implement KSpaceGrid with bind / check_capacity / vectors / to in src/molpot/potentials/gto/kspace.py, sizing only on the first bind, raising a RuntimeError that names both KSpaceGrid and capacity_factor afterwards, and retire half_sphere_kvectors
- [x] Write failing unit tests for bind_cell, num_graphs threading, the constructor-time k_cut, the non-persistent buffers and the periodic-branch explain → 1 graph / 0 breaks (tests/test_molpot/test_potentials/test_gto/test_hartree.py, tests/test_molpot/test_potentials/test_gto/test_features.py)
- [x] Implement bind_cell, the num_graphs keyword, scatter_sum_compile_safe, persistent=False on the five ctor-derived GTOFieldFeatures buffers and the absorbed _kspace_energy / _kspace_potential in src/molpot/potentials/gto/hartree.py and src/molpot/potentials/gto/features.py, and re-export KSpaceGrid from src/molpot/potentials/gto/__init__.py
- [x] Write failing unit tests for MacePolar.energy_core forwarding num_graphs to both GTO submodules and for the isolated branch still breaking (tests/test_molzoo/test_mace/test_polar.py → TestMacePolarKspaceCompile)
- [x] Implement the num_graphs pass-down at src/molzoo/mace/polar.py:241-246 and :266
- [x] Add google docstrings with units, the D3/D4 capacity argument incl. the conservative strict-inequality edge, the construct → to → bind_cell → compile lifecycle with "resize = build a new grid", and the k_cutoff "reference-implementation convention" attribution
- [x] Record the k-point inflation figures, the two-way multipole_ewald promotion trigger and the retained isolated-path break in .claude/notes/notes.md topic compile-no-graph-breaks
- [x] Add regression example regressions/gto-kspace-fixed-grid.py (public API only; hard-coded goldens, no third-party runtime)
- [x] Hygiene cleanup: /mol:simplify ran clean (1 apply: GTOFieldFeatures.forward resolves num_graphs only on the periodic branch; 2 manual → /mol:refactor: MacePolar.__init__ / energy_core length, pre-existing)
- [x] Run full check + test suite (x86: 3925 passed / 15 skipped, ruff+ty clean; GH200 job 1957227: 333 gto+mace passed, 3 regressions ok, full suite 4013 passed with the 11 documented env-only failures)

## Testing strategy

单元测试只测单函数/单模块，路径镜像 `src/`，类型镜像（`KSpaceGrid` → `TestKSpaceGrid`）。`$META.build.test_single`（`python -m pytest {path} -v`）单文件跑绿即算单元绿。**顺带修一处镜像卫生**：`test_kspace.py` 今天的 `TestGTOKspaceHartree`（`:31-118`）测的是 `hartree.py` 的 `GTOHartreeEnergy` 与将被吸收的两个自由函数，整类迁到 `test_hartree.py`；`test_kspace.py` 之后只留 `kspace.py` 自己的公开面（`TestKspaceCutoff`、`TestKSpaceGrid`、`structure_factor` / `cartesian_dipole`）。**迁移时必须改构造**：`test_zero_dipole_matches_monopole`（`:66-75`）、`test_half_sphere_matches_full_sphere`（`:77-99`）、`test_cell_gradient_is_nonzero`（`:101-109`）今天直接调自由函数 `hartree_energy`，它**没有自能项**；而 `GTOHartreeEnergy.forward` 的周期分支在默认 `include_self_interaction=False`（`hartree.py:182`）下会**减去** `_self_energy`（`:250-251`）。因此这三条一律改成 `GTOHartreeEnergy(sigma=…, l_max=…, include_self_interaction=True)` 的 `forward`，才是同一个量。

- `tests/test_molpot/test_potentials/test_gto/test_kspace.py::TestKSpaceGrid`
  - happy path：立方胞 a=10 Å、k_cut=1.8467166200173746 Å⁻¹ ⇒ \(x_i=2.9391\)，`capacity_factor=1.2` ⇒ \(N_c=(4,4,4)\)，\(M=(9^3-1)/2=364\)；`weights` 取值集合 ⊆ {0.0, 2.0}；权重非零的 k 集合（46 个）与硬编码的旧 `half_sphere_kvectors` 输出（改动前捕获的 fp64 字面量）排序后逐个相等。
  - 三斜胞（D4 精确性）：`cell = [[10,0,0],[3,9,0],[1,2,8]]`，断言所有 `weights>0` 的 n 都满足 \(|n_i|\le N_{c,i}\)，且没有 \(|k|<k_{\rm cut}\) 的整数点被漏掉（对同一 cell 用一次性 eager 暴力枚举做集合比较，硬编码计数）。
  - **定尺只发生一次**：同一 cell 连调两次 `bind` ⇒ 整数盒张量是同一对象（`is` 相等）；把 cell 缩小到 0.5× 再 `bind` ⇒ 不抛、整数盒**仍是同一对象**（不缩容）；放大到 1.1× 与 1.5× ⇒ 不抛、对象不变（x=4.41、⌊x⌋=4=N_c，盒内无截断）；放大到 2.0× ⇒ `RuntimeError`（⌊5.88⌋=5>4，n=(5,0,0) 的 |k|=1.571<k_cut 会被截断）。类上不存在 `rebind` / `set_cell` 属性。
  - edge case：k=0 保护 —— 半盒不含原点，仍断言 `k_sq_guarded` 全 >0、`weights` 在 `k_sq==0` 处恒 0、能量与梯度均 `isfinite`（0·inf=NaN 的护栏）。
  - edge case（NaN 哨兵）：`bind` 于 10 Å 立方胞后，直接对放大 2.0× 的 cell 调 `.vectors`（**不**先 `check_capacity`，模拟编译区内路径）⇒ `weights` 全 NaN、下游能量与 `autograd.grad` 力全 NaN；放大 1.1× 与 1.5× 时不出现任何 NaN（首次溢出在 x≥5，即本胞 1.70×）。这条测试同时断言 `.vectors` 全程不触发 host sync（用 `torch._dynamo.explain(grid.vectors)(cell)` 的 `graph_break_count == 0` 作代理 —— 本步不依赖 02 才引入的 `Compiler.explain`，沿用 `test_potential.py:740-748` 的裸 `torch._dynamo.explain` 写法；02 落地后统一迁移）。
  - edge case（`(1,3,3)` 契约）：`vectors(cell)` 与 `vectors(cell.unsqueeze(0))` 的三个输出逐位相等（`torch.equal`）；`(2,3,3)` ⇒ `ValueError` 含 `"single-system"`；`bind` / `check_capacity` 同样接受 `(1,3,3)`。
  - edge case（具名错误）：未 `bind` 就 `.vectors` ⇒ `RuntimeError`，消息含 `"bind_cell"`；整数盒在 CPU、cell 在另一设备 ⇒ 具名 `RuntimeError`（CPU-only CI 用 `torch.device("meta")` 制造不匹配）；第一次 `bind` 就 `max(N_c) > max_axis_index` ⇒ `RuntimeError("GTO k-grid exploded: …")`；越容量的 `check_capacity` ⇒ `RuntimeError` 且消息**同时**含 `"KSpaceGrid"` 与 `"capacity_factor"`（与 `neighbors.py:838-842` 区分）。
  - device：`to(device)` 返回**自身**（`is` 相等）且整数盒与 `.vectors` 输出在目标设备（CPU 下走 `torch.device("cpu")` 的恒等路径）。
- `tests/test_molpot/test_potentials/test_gto/test_hartree.py`
  - domain validation（硬编码期望值）：两个 ±1 e 电荷、σ=1.5 Å、10 Å 立方胞，`GTOHartreeEnergy` 的 fp64 周期能量等于改动前 eager 路径捕获的字面量，`atol=1e-10` eV；迁移过来的三条（见上，全部以 `include_self_interaction=True` 构造）走公开 `forward`。
  - 应变导数（D7）：同一构型 \(\partial E/\partial\varepsilon\)（`torch.autograd.grad` 对 3×3 应变叶子）与改动前捕获的 3×3 fp64 字面量比较，`atol=1e-10` eV。
  - compile（周期分支）：`torch._dynamo.explain(mod)(…)` 在 `backend="aot_eager"`、CPU、`bind_cell` 之后、传 `num_graphs=1` 时 `graph_count == 1 and graph_break_count == 0`；`cell` 以 `(3,3)` 与 `(1,3,3)` 各跑一次，两次能量逐位相等。
  - compile（孤立分支，**保留断点的守卫**）：同一模块不传 `cell`、传 `num_graphs=1` 时断言 `graph_break_count > 0`，并在断言消息里指向 `polar-compile-clean-04-pairs`。
  - edge case：不传 `num_graphs` 时周期分支**允许**断点（eager-only 回退，显式断言其存在）。
  - 不变式（符号级）：`inspect.getsource` 取 `_complete_directed_pairs` / `cached_complete_directed_pairs` / `_skip_diagonal` / `_skip_equal_orig_regular` 四个函数体，与改动前捕获的字符串字面量逐字相等；孤立路径的 fp64 能量与改动前捕获的字面量 `torch.allclose(atol=1e-12)`（**不是**逐位：`scatter_sum` → `scatter_sum_compile_safe` 把 `scatter_add_` 换成了 `index_add_`，归约次序在 CUDA 上不是逐位契约）。
- `tests/test_molpot/test_potentials/test_gto/test_features.py`：同上，针对 `GTOFieldFeatures`（`out_dim=8`、σ_comb=[1.5, 2.3717082451262845]、共享 k_cut），含 `bind_cell`、`(1,3,3)` 逐位一致、周期分支 explain→(1,0)、孤立分支断点仍在、以及 fp64 字面量对照；另加两条：`_kspace_features` 不再读 `combined_sigma.min()`（AST 扫描：函数体内无 `float(` 与 `.item(`），以及 `set(GTOFieldFeatures(...).state_dict()) == set()`（五个 ctor 导出 buffer 全部 `persistent=False`）。
- `tests/test_molzoo/test_mace/test_polar.py::TestMacePolarKspaceCompile`
  - happy path：给两个 GTO 子模块的 `forward` 打桩记录实参，断言 tiny `MacePolar.energy_core(..., num_graphs=1, cell=…)` 的三次调用（`num_recursion_steps=1` 时为 2 次描述子 + 1 次 coulomb）都收到 `num_graphs=1`，且没有一次落到 `int(batch.max().item())` 回退。
  - compile：tiny Polar（CPU、naive cueq、`aot_eager`），`bind_cell` 之后 `torch._dynamo.explain(model.energy_core)(…, cell=…)` 给 `(1, 0)`；不传 `cell` 的孤立路径断言 `graph_break_count > 0`。
  - domain validation：同一 tiny 周期构型的 fp64 能量等于改动前 eager 捕获的硬编码字面量，`atol=1e-10` eV。
- **Regression example**：`regressions/gto-kspace-fixed-grid.py`（按能力命名，而非按 slug —— 它是"固定容量 k 网格"这项能力的活文档）—— 只用公开 API（`KSpaceGrid`、`GTOHartreeEnergy`、`GTOFieldFeatures`），CPU 秒级：构造 10 Å 立方胞与 4 电荷构型，断言 (a) `M == 364`、(b) 权重集合 ⊆ {0,2} 且非零权重恰 46 个、(c) fp64 周期 Hartree 能量等于硬编码字面量 `atol=1e-10` eV、(d) `(3,3)` 与 `(1,3,3)` 两种 cell 输入逐位相等、(e) `bind_cell` 后 `aot_eager` explain 为 (1, 0)、(f) 对 2.0× 放大的 cell 直接 `.vectors` 得到全 NaN 而不是截断（1.5× 仍在盒内、无 NaN），而 `check_capacity` 在同一 cell 上抛出含 `KSpaceGrid` 与 `capacity_factor` 的 `RuntimeError`。goldens 是改动前本仓 eager 路径捕获的数值（注释注明：`molnex` @ `feat/oplsaa-parameterizer`、命令 `PYTHONPATH=src python regressions/gto-kspace-fixed-grid.py`、2026-09-03），**不 import、不 subprocess 任何第三方**（无 `mace` / `e3nn` / `graph_longrange` / ASE）。
- **既有 regressions 必须重跑，不得改数**：`regressions/mace-polar-periodic-gto.py`（周期路径，四个官方 golden + `n_edges=285546`，`atol=1e-3` eV，`:40-46`；它正是喂 `(1,3,3)` cell 的那条路径，`:93`）；`regressions/mace-polar-official-water.py`（孤立路径，`:28-33`）—— 孤立分支的枚举本步未改动，其数值按 `atol=1e-12` 断言不变。

## Out of scope

- **孤立分支的批量全对枚举**（`hartree.py:96-151`）—— 保留原样，断点已点名，owner = molpot gto，后继链节 `polar-compile-clean-04-pairs`（块对角逐图枚举 + 构造期 `max_atoms_per_graph` 容量 + bind 时溢出抛错）。
- `elec/kernels/multipole_ewald.py:221-245 _build_kvec_grid` / `:352-370` 半球掩码的同缺陷迁移。**双向提升触发器**：它成为第二消费者时（或 `KSpaceGrid` 先拿到第二消费者时），网格提升到 `src/molpot/potentials/elec/lib/kvectors.py`；同时同步 `enumerate_kvec_indices:247-255` 的 FD-stress keep-set。owner = molpot elec。
- `elec/lib/kvectors.py` 的 FFT 序生成器改造（只被模仿，不被改）。
- `KSpaceGrid` 的重定尺入口（`rebind` / `set_cell`）—— 树内零消费者，本步不造；换尺寸 = 新建对象。
- `GTOHartreeEnergy.self_overlap`（`hartree.py:195`）的 buffer 持久性 —— 本步不碰。
- 邻居表 build/update 的编译（另 spec）、整步 CUDA graph 捕获、OMol/MatPES（已 0 断点）、二阶反传编译（spec `cuet-force-doublebackward`：AOTAutograd 不支持 double backward，故本链的编译缝只对**一阶 MD 推理**成立，训练力损失继续走 eager）。
- `Compiler` 加宽、`CellBound` / `CoreCompilable` 协议、MD 重绑定、`compile_core`、元素门（→ 02-seam / 03-wire）。
- k 点膨胀带来的步时/显存代价本步只**记录估算**（D5），实测在 03-wire 的 GH200 作业里做。
- blueprint 重生成（`/mol:map`，另行执行；librarian 报告 `gto/` 处为盲区）。
