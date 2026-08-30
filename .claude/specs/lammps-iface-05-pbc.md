---
title: PBC 正确性 —— fat-ghost 单发方案、per-atom 能量分区与 schema v3
status: approved
created: 2026-08-30
slug: lammps-iface-05-pbc
chain: lammps-iface
grilled: false
---

# PBC 正确性 —— fat-ghost 单发方案、per-atom 能量分区与 schema v3

## Summary

修掉 `pair_style molnex` 在周期 / 多 rank 条件下的四处正确性缺陷，方案是 `mir-group/pair_nequip_allegro` Allegro 分支的 **fat-ghost 单发**方案，外加一条被校验的感受野条件。四处缺陷：**(1) 能量重复计入**——`pair_molnex.cpp:332-333` 把模型对 local + ghost 全部位点求得的总能量整体记入 `eng_vdwl`，多算 \(\sum_{i\in G} E_i\)，其量级由 per-species \(E_0\) 主导（MACE 的 `atomic_energies`，`src/molzoo/mace/potential.py:500`），每个 ghost 数十 eV；**(2) ghost 力双重通信**——pair 层自调 `comm->reverse_comm(this)`（`:351-352`）且 pack 读**共享的** `atom->f`（`:359-382`），`newton` 打开时 `Verlet::run` 的全局 reverse comm 会把同一份贡献再送回一次；**(3) 图残缺**——只对 local 建邻居表，ghost→local 的入边缺失，目标聚合模型（MACE）第 1 层即错、源聚合模型（PiNet）第 2 层起错；**(4) 感受野不足**——\(T\) 层 MPNN 的有效截断是 \(T\cdot r_c\)，LAMMPS 默认 ghost 截断 \(r_c+\text{skin}\) 对 \(T\ge2\) 不够，且当前完全不校验。

修法：导出契约从"标量能量"改为"per-atom 能量 + local mask"（schema v3），\(E^{(p)}=\sum_{i\in L}E_i\)，力对 local∪ghost 全体求导（pair 层不再自调：归约所有权经 `EvalResult::needs_pair_reverse_comm` 这条缝移交积分器，字段与 facade 调用点原样保留、恒为 `false` 而永不触发；pbc 路径强制 `newton_pair on`）；`GraphBridge` 请求 `REQ_FULL|REQ_GHOST` 并遍历 `inum+gnum` 取得 \(S_{T-1}\) 闭包，因对称闭包而**与聚合方向无关**（一套 builder 同时对 PiNet 与 MACE 正确，无 per-model 分支）；新增必填 meta 字段 `receptive_field_layers`，`init` 校验 ghost 截断 \(\ge T\cdot r_c+\text{skin}\)，失败即 `error->all` 并给出 `comm_modify` 提示（fail closed，绝不降级为 warning）。同时按 D6 把每步 `std::sort` 从**无条件执行**降级为**契约声明驱动**：meta v3 新增 `requires_sorted_edges`（默认 `false`），`GraphBridge::build` 当且仅当契约声明时排序。排序代码留着，普遍性的每步开销去掉，而未来采用分段 / CSR 聚合的模型只要以 `True` 导出就永远不可能被喂进无序边——与 `receptive_field_layers` 同一套 fail-closed 哲学。全树"按 (source,target) 有序"的**无条件承诺**随之改写为条件承诺。`04-facade` 预留的扩展点在此兑现：`EvalResult` 获得 `e_atom`，`eflag_atom` 从报错转为**受支持**。顺带修掉本表面上的两处工程失真（iron law）：`src/molpot/derivation/protocol.py:166` 的 per-atom 能量**静默丢弃**，以及 `adapter.py:111` 与 `static.py:62` 两条被文档声称等价的路径上**两个不同的小 r 正则化**（1e-6 vs 1e-8）。

## Domain basis

### D1 位点能量分区与域分解下的求和域

多体 ML 势的总能量是位点能量之和 \(E=\sum_i E_i\)（DeePMD，Wang, Zhang, Han, E, arXiv:[1712.03641](https://arxiv.org/abs/1712.03641)；Allegro，Musaelian et al., SC'23 §V.A, DOI [10.1145/3581784.3627041](https://doi.org/10.1145/3581784.3627041)；SevenNet，Park, Kim, Hwang, Han, *JCTC* **20**, 4857 (2024), DOI [10.1021/acs.jctc.4c00190](https://doi.org/10.1021/acs.jctc.4c00190) eq. 1–2）。域分解下每个 rank 拥有 local 集 \(L\) 与 ghost 集 \(G\)，ghost 是其他 rank（或周期镜像）所拥有原子的副本。分区总能量必须**只对 local 求和**：

\[
E^{(p)}=\sum_{i\in L}E_i,\qquad E=\sum_p E^{(p)}
\]

对 \(L\cup G\) 求和会多计 \(\sum_{i\in G}E_i\)。该误差**不是小量**：\(E_i\) 含 per-species 常数偏移 \(\mu_Z\)（MACE 的 `atomic_energies`），每个 ghost 位点数十 eV 量级，且 ghost 数随 rank 数与截断增大。参考实现：`mir-group/pair_nequip_allegro` `pair_nequip_allegro.cpp:387-400`，注释即为 *"energy = sum of local atomic energies to avoid energy shift issues for ghost atoms"*，代码为 `if (ii < inum) eng_vdwl += atomic_energies[i][0];`。

**这直接决定导出契约**：标量总能量不足以在 C++ 侧完成分区求和，模型必须输出 per-atom 能量。

### D2 力与 ghost 贡献的归约路径

由 \(E^{(p)}=\sum_{i\in L}E_i\) 出发，

\[
\mathbf F_j=-\frac{\partial E^{(p)}}{\partial \mathbf r_j},\qquad j\in L\cup G
\]

对 \(j\in G\) 的分量非零——它们是 local 位点能量对 ghost 坐标的交叉项，必须送回属主原子。**归约由积分器完成**：`Verlet::run` 在 `if (force->newton)` 时调用全局 `comm->reverse_comm()`，一次性归约所有 style 写进 `atom->f` 的 ghost 贡献。`pair_nequip_allegro` 中**没有任何** `comm->reverse_comm` 调用，并在 `init_style` 中强制 `newton_pair` 打开。当前 molnex 两者兼有（pair 层自调 + 依赖全局归约），故 ghost 贡献被送回两次。

真正域分解下的消息传递（每层通信）另有方案，见 SevenNet（arXiv:[2402.03789](https://arxiv.org/abs/2402.03789) eqs. 5–6, §3.1）——本 spec **不采用**，见 Design 的方案裁决。

### D3 感受野与 fat-ghost 条件

\(T\) 层消息传递网络中，位点 \(i\) 的能量依赖半径 \(T\cdot r_c\) 内的全部原子（Musaelian et al., *Nat. Commun.* **14**, 579 (2023), arXiv:[2204.05249](https://arxiv.org/abs/2204.05249)）。单发（single-shot）方案要求本 rank 持有每个 local 原子完整感受野内的全部原子副本，即

**D3.2（fat-ghost 条件）**：\(\ \text{ghost cutoff}\ \ge\ T\cdot r_c+\text{skin}\)

LAMMPS 默认 ghost 截断是 \(r_c+\text{skin}\)，对 \(T\ge2\) **不足**；须经 `comm_modify cutoff` 加宽。严格局域模型（Allegro，\(T=1\)）默认即足。该条件无法从 \(r_c\) 推断，故 \(T\) 必须由导出侧声明。

### D4 边集完整性与聚合方向无关性

只对 local 原子建邻居表，得到的边全部满足 \(\text{src}\in L\)。此时：

- **目标聚合**模型（MACE，`_scatter_sum` 聚到 target，`src/molrep/interaction/mace/density.py:176`）丢失 ghost→local 的消息，**第 1 层即错**；
- **源聚合**模型（PiNet，`index_add_(0, src)`，`src/molrep/interaction/pinet/message.py:65`）第 1 层 local 特征正确但 ghost 特征被冻结在初值，**第 2 层起错**。

**D4.2（精确性条件）**：枚举 \(S_{T-1}\)（\(L\) 的 \((T-1)\) 跳闭包）中每个原子的**完整入边集**。操作上即请求 `REQ_FULL|REQ_GHOST` 并遍历 `ii < inum + gnum`（`pair_nequip_allegro.cpp:146` 先例）。所得边集在方向上**对称闭合**，因而 builder 与模型的聚合方向无关——**一套 `GraphBridge` 同时对 PiNet 与 MACE 正确，不需要任何 per-model 分支**（`CLAUDE.md` "no per-model C++" 因此得以保持）。

### D5 位力

\[
W_{\alpha\beta}=\sum_{k\in L\cup G} r_{k\alpha}f_{k\beta}
\]

即对 local+ghost、在 reverse comm **之前**的部分力做 fdotr（LAMMPS `compute pressure` 文档中的 \(N'\) 含 ghost）。该式**当且仅当**力由 \(E_{\text{local}}\) 导出时成立——故 D1 必须先修。逐边形式为 \(W=\sum_e \mathbf d_e\otimes\mathbf f_e\)，\(\mathbf f_e=-\partial E/\partial\mathbf d_e\)；**符号为正**（力约定），\(-\sum \mathbf r\otimes\mathbf f\) 形式只对梯度成立。模型自报的 virial 必须是 \(E_{\text{local}}\) 的 virial。per-atom virial 仍不提供，`vflag_atom` 保持 `error->all`（`pair_nequip_allegro` 同样报错）。

### D6 边序与浮点重结合

`index_add_` / `scatter_add_` 类归约在精确算术下**与输入排列无关**；浮点下重结合带来的相对误差约 \((n-1)u\)（fp32 每位点最坏 ~1e-6）。CUDA 的 `index_add_` 本身即非确定性，故排序**买不到逐位可重现性**。`pair_nequip_allegro` 按原始邻居表顺序建边、不排序。

但"当前模型不需要有序"不等于"有序永远不需要"：**分段 / CSR 聚合（cuEquivariance 的 indexed 算子）要求按源索引分组**，其正确性依赖有序输入。因此排序的正确处置不是删除、也不是无条件保留，而是**由导出契约声明**：`requires_sorted_edges` 默认 `false`（scatter 类归约，无序即可），需要分段聚合的模型以 `true` 导出，`GraphBridge` 据此决定是否排序。这样普遍代价消失，而错误组合（分段聚合模型 + 无序边）在契约层被封死，无法悄悄发生。**已验证的树内事实**：`src/molrep` / `src/molzoo` / `src/molpot` 全树无 `segment_csr` / `segment_coo` / CSR 类聚合，现有 zoo 100% 是 scatter 家族——`src/molrep/interaction/aggregation.py` 须加一句守卫注释："若引入 CSR/分段聚合，导出必须声明 `requires_sorted_edges=True`"。

**端到端证据。** ac-014 的 parity 本身即全尺度的顺序不变性实验：05 之后 C++ 侧按 LAMMPS 邻居表的**原始顺序**建边，而作为参照的 Python 直接求值走 `nonzero()` 的**有序**输出；两者在真实导出模型上于 rel 1e-6 内一致，正是打乱顺序实验的满量程版本。桩聚合的 ctest（ac-009）保留为快速的结构锁。因此本条判据不依赖任何历史推断。

**保留的史料判断（不再承担论证重量）**：若当初加排序是为了消除某个观察到的差异，那个差异最可能的真实成因是 D1 或 D4，而非边序。

**单位**：能量 eV 或 kcal/mol（由 meta `units` 声明，换算表见 `01-meta`），长度 Å，力 = 能量/Å，virial 与能量同单位。

## Design

### 前置条件（硬依赖）

`01-meta` / `02-view` / `03-graph` / `04-facade` 全部落地。本 spec 修改的每一处 C++ 都落在 `04-facade` 划出的边界内（`GraphBridge`、`CutoffGraphImpl`、`LammpsRunEnv`、facade 的机械 tally），这正是三层拆分要换取的东西。

### 方案裁决（binding，不翻案）

采用 **fat-ghost 单发**：一次前向覆盖完整感受野，rank 间不做逐层通信。**不采用** SevenNet 的逐层通信（真域分解下更省显存、可扩展到大 \(T\)，但需要在每层插入 halo 交换，属另起的执行模型——列为未来 fork，本 spec 之外）。**不采用** `pair_nequip` 的 `nequip_mode` 单 rank tag 重映射（把 ghost 按 tag 映回 local 索引，只在单 rank 下成立，不是我们的先例）。

**推论（需明确记录）**：**tag↔local 映射本方案不需要**。它曾被暂记于 `05-pbc`，实际归属是未来的 `fixed-graph` 契约（拓扑型 FF 需要把 LAMMPS 的 tag 对上分子拓扑的原子编号）。本 spec 内不得出现 tag 相关字段或映射，`LammpsView` 也不新增 `tag`。

### D1 落地：per-atom 能量与分区求和

导出契约新增 per-atom 能量输出；`CutoffGraphImpl::compute` 以

`e_total = energy_conv * Σ_{i < view.nlocal} atomic_energy[i]`

填 `EvalResult::e_total`，并把 `(N_local,)` 的 per-atom 能量（乘过 `energy_conv`）填入**新增的** `EvalResult::e_atom`。**`e_atom` 的长度契约是 `nlocal`**，写进结构体 docstring；facade 的 tally 对**返回张量自身的长度**迭代累加 `eatom[i]`，**不引用 `atom->nlocal`**——facade 不得对 impl 的返回范围做独立假设，否则两处对"哪些原子是 local"的理解一旦分歧就会静默越界或漏算。facade 的 tally 相应扩展：`eflag_atom` 时累加 `eatom`——这正是 `04-facade` 记录的**设计扩展点在预定时刻被兑现**（04 当时没有生产者，故不设字段；05 是生产者到场的那一版），不是对 04 决定的推翻。`vflag_atom` 仍无生产者，保持 `error->all`。

`EvalResult` 与 `LammpsRunEnv` 的累积规则（04 已写在结构体 docstring 上）在此第一次被行使：`e_atom`、`ghost_cutoff`、`skin` 都与其生产者/消费者代码同批落地。

### D2 落地：归约所有权移交积分器

- **保留** `04-facade` 的 `EvalResult::needs_pair_reverse_comm` 字段与 facade 中 `if (r.needs_pair_reverse_comm) comm->reverse_comm(this);` 这一调用点，**一行 facade 代码都不改**。`CutoffGraphImpl::compute` 把它填为**恒定的 `false`**：ghost 力照常被 `ForceScatter` 写进 `atom->f`（含 ghost 行），归约由 `Verlet::run` 的全局 reverse comm 承担。这正是该字段作为**归约所有权缝**的价值兑现——05 改变的是归约由谁负责这一事实，而改变的方式是 impl 换一个返回值，不是 facade 换一段代码。删掉字段反而会**逼出一次 facade 编辑**，把一个本来干净的所有权移交变成对协议层的手术。
- 该 `false` 必须带注释写明其**前提条件**：`CutoffGraphImpl::init` 强制 `newton_pair` 打开，因而积分器的全局 reverse comm 必定运行——**常量 `false` 的安全性完全建立在这条强制之上**，二者是一体的，任何放松 init 中 newton 强制的改动都必须同时重新审视这个常量。docstring 的"未来为真的条件"必须写全其**后果**：某个在 newton 归约的积分之外散射 ghost 力的契约把本字段填为 `true` 时，它同时**重新激活 `ContractImpl::pack_reverse` / `unpack_reverse`**——那两个 override 只可能经由 `comm->reverse_comm(this)` 被 LAMMPS 调用，别无入口。这正是本字段必须保留的决定性理由：删掉调用点会让裁决②要求存在的那条缝**永久不可达**，退化成装饰，而任何未来需要 pair 层归约的契约都将被迫重新往 facade 里加一行。
- `compute()` 以**廉价方式重申该不变量**：impl 在 `init(env)` 时把 `env.newton_pair` 存为成员（该成员随本处的读取代码一同落地，符合 consumer-arrives 规则），`compute` 中若 `supports_pbc && !newton_pair_` 则抛出。**诚实说明其效力边界**：它读的是 impl 自己在 init 时的快照，因此**能**捕获"compute 在 init 未校验的情况下被触达"这类内部次序错误，**不能**捕获 LAMMPS 侧在 init 之后改变 newton 的情形（`init_style` 每次 `run` setup 都会重跑，该情形由 init 本身覆盖）。写下这条边界是为了让下一个读者知道它保的是什么，而不是误以为它保了一切。
- `CutoffGraphImpl::init` 对 `supports_pbc` 契约**强制 `newton_pair` 打开**（全局归约只在 `force->newton` 时发生；newton off + pbc ⇒ ghost 力永不归约）。这是对 04 中"照搬未收紧"的单域校验的**设计内继任修改**：04 明确把收紧留给 05。非 pbc 路径的原条件（`!supports_pbc && newton_pair == 1 && nprocs > 1`）保持不变。
- `ContractImpl::pack_reverse` / `unpack_reverse` 与 `NeighborNeeds::comm_reverse`（=3）**保留**（裁决②）：它们是 LAMMPS 虚函数 `pack_reverse_comm` / `unpack_reverse_comm` 的转发去处，那两个 override 无论是否被调用都必须存在。在 cutoff-graph 契约上它们目前不被触发，docstring 必须写明这一点及其原因与唯一重新启用路径（某契约把 `needs_pair_reverse_comm` 填 `true`）。
- **05 对 facade 的全部改动**：一处**替换**（`eflag_atom` 的报错臂改为对 `EvalResult::e_atom` 返回长度的 tally）、两行**新增**（把 `comm->cutghostuser` 与 `neighbor->skin` 原样快照进 `LammpsRunEnv`）。**没有任何删除，没有任何语义 if**——三处都是协议机械：多 tally 一个 per-atom 数组、多转述两个 LAMMPS 全局量。归约所有权这样一个实质性的物理决策，在 facade 侧的代价是零行。

### Exec 输出面随 v3 扩展

v3 让模型多返回一路 per-atom 能量，因此 `03-graph` 的执行面必须同步：`ModelOutputs`（`interface/lammps/bridge/exec.h`）新增 `at::Tensor atomic_energy`，`read_model_outputs` 扩签名以接收期望原子数（照 03 docstring 预告的 expectation 形态演化，避免布尔堆积），**在单一解码器内**完成 `atomic_energy.size(0) == n_atoms` 的校验（不符即库侧抛出并给出实测长度）——两个 `Exec` 只做设备/dtype 搬运，不各自复制输出契约校验。

**输出下标发生位移，这是本次改动最容易致命的一处**：v2 的输出顺序是 `[energy, forces, (virial)]`，virial 在 `out[2]`；v3 变为 `[energy, forces, atomic_energy, (virial)]`，virial 移到 `out[3]`。沿用旧下标会把一个 `(N,)` 的 per-atom 能量当成 `(6,)` 的 virial 读入。防线有三层，缺一不可：
1. `read_model_outputs` 内的**具名下标常量**（单一定义点），禁止散落的裸数字；
2. `numel == 6`（virial）与 `size(0) == n_atoms`（atomic_energy）两道校验——但**二者在 N == 6 时同时失明**（苯、双水分子、多数手写 fixture 恰是这个尺寸），故：
3. **六原子 ctest**：`bridge_read_model_outputs_six_atoms_no_transpose` 以 N=6 的构造输入断言 virial 与 atomic_energy 未被互换（值级断言，非仅形状）。meta v3 已携带 `outputs` 名单，实现**按名单顺序解析下标**（名解析优先于常量，二者一致性由该 ctest 锁定）。

### D3 落地：`receptive_field_layers` 与 fat-ghost 校验

meta 新增必填字段 `receptive_field_layers`（\(T\)，正整数）。Python 侧 `export_for_lammps` 增加**无默认值**的同名 keyword-only 参数：给默认值 1 会让 \(T=3\) 的 MACE 静默声明为严格局域，D3.2 校验随即以过小的阈值通过——正是 fail-closed 要杜绝的形态。缺参即 `ValueError`，文案**同时给出关键字名与重新导出的命令**——报错发生在用户已经跑完一次训练之后，此时最有价值的是可直接照抄的下一步。

`LammpsRunEnv` 新增 `ghost_cutoff` 与 `skin` 两个字段。**facade 只做原始快照，不做任何默认值推断**：`ghost_cutoff` 直接取 `comm->cutghostuser`（用户未设置时 LAMMPS 即为 `0.0`），`skin` 直接取 `neighbor->skin`。"`0` 表示未设置、此时有效 ghost 截断为 `cutoff + skin`"这条**解释规则完全落在 `CutoffGraphImpl::init` 内部**。两侧契约都必须钉死：facade 承诺原样转述，impl 承诺独家解释。把默认值推断放进 facade 会让同一条规则出现两个可能漂移的实现，也会在 facade 里重新长出一个语义 if。这两个字段按 04 写在该结构体 docstring 上的 consumer-arrives 规则添加——**与读取它们的代码同时落地**。

`CutoffGraphImpl::init` 校验 \(\text{ghost cutoff}\ge T\cdot r_c+\text{skin}\)，不满足即抛出，文案给出实测值、所需值与 `comm_modify cutoff <value>` 的具体建议。**fail closed，绝不 warning**。

### D4 落地：ghost 闭包

- `NeighborNeeds::ghost` 对 `supports_pbc` 契约置 **true**（04 定义该字段时即为此刻预留；facade 侧那行 `if (needs.ghost) flags |= NeighConst::REQ_GHOST;` 一字不改）。
- `GraphBridge::build` 遍历 `ii < view.inum + view.gnum`（`LammpsView` 因此新增 `gnum`，同样随消费者落地），`n_graph = view.nall()`。cutoff 重过滤、`j >= n_graph` 过滤、`neigh_mask` 剥除全部保留。
- `03-graph` 记录的"`ilist` 源索引被信任为 local"那句注释必须**改写**：加入 ghost 遍历后，源索引不再限于 local，其取值域是 \(L\cup G\)，这正是 D4.2 的对称闭包所要求的。

### D6 落地：排序降级为契约声明

`GraphBridge` 新增一个构造期确定的 `bool requires_sorted_edges`（来自 `LammpsContract`），`build` 中的 `std::sort` **仅在其为真时执行**；排序代码、`edge_pairs_` 的复用缓冲与其注释谱系（源自 `pair_molnex.cpp:236-239`）全部保留，只是不再无条件付费。默认路径（`false`）按 LAMMPS 邻居表的原始顺序建边，与 `pair_nequip_allegro` 一致。

同步把全树的**无条件**"按 (source,target) 有序"承诺改写为**条件**承诺：`src/molix/engine/adapter.py:23-24`（"The C++ side sorts edges by ``(source, target)`` before the call; adapters must not assume any other order."）与 `:77`（`edge_index` 参数说明中的 "sorted by ``(src, tgt)``"），以及 `interface/lammps/README.md` 中的对应措辞。新措辞：边序默认不受约束，聚合为 scatter 类归约因而排列不变；需要有序输入的模型（分段 / CSR 聚合）必须以 `requires_sorted_edges=True` 导出，届时且仅届时 C++ 侧保证有序。D6 的注意事项注释与之同处一地；`src/molrep/interaction/aggregation.py` 加守卫注释（见 Domain basis D6）。

**与 `03-graph` 的关系（明示取代）**：`03-graph` 无条件保留 `std::sort` 并把处置明确留给本 spec；本 spec 的裁决是**降级为契约标志**，而非早期设想的"删除"。03 的相应措辞由本 spec 取代（03 的 ac-002 带有 forward note）。

### Python 导出契约 schema v3

本 spec 的全部 Python 测试落在 `tests/test_molix/test_engine/` **镜像包**内（该包由 `01-meta` 的拆分建立，对应 `src/molix/engine/` 的三个模块），不新建任何平铺的 `test_engine.py`；`molpot` 侧的改动落在既有的 `tests/test_molpot/test_derivation/test_protocol.py`。

- `LAMMPS_META_SCHEMA = 3`；`lammps_contract.h` 的期望版本同步升到 3。`01-meta` 建立的 fixture 绑定使**任一侧单独改动即红**——这是设计意图。
- 输出：`outputs = ["energy", "forces", "atomic_energy"]`（`supports_virial` 时追加 `"virial"`）。`EngineForward` 返回 `(energy, forces, atomic_energy)`。
- 输入新增 `local_mask (N,) bool`。同一次 v3 bump 中，把 CUDA-graph 路径原有的**边有效性掩码由 `mask` 更名为 `edge_mask`**——两个语义完全不同的掩码在同一个输入列表里一个叫 `mask`、一个叫 `local_mask`，是可预见的误接线源。于是：`dynamic` 为 `["Z","pos","edge_index","local_mask"]`（4），`cuda-graph` 为 `["Z","pos","edge_index","edge_mask","local_mask"]`（5）。`lammps_contract.h` 的输入 arity 校验随之更新。**位置契约与 dtype 必须在 `interface/lammps/README.md` 的输入表中钉死**：`Z (N,) int64`、`pos (N,3) 模型 dtype`、`edge_index (E,2) int64`、`edge_mask (E_max,) bool`、`local_mask (N,) bool`。
- 能量以**掩码乘法**求得：`energy = (atomic_energy * local_mask).sum()`。之所以用掩码而非按 `n_local` 切片：切片长度是运行期标量，`torch.export` 无法 trace；掩码乘法可以。力仍为 `F = -∂energy/∂pos`，对全体原子求导，因而自动得到 D2 的 ghost 交叉项。非 pbc 路径传全 1 掩码——彼时 `nghost == 0`，\(\sum_{\text{all}}=\sum_{L}\)，**parity 得以保持**。
- `receptive_field_layers` 写入 meta；`contract == cutoff-graph` 时必填、**无默认值**。
- `requires_sorted_edges` 写入 meta；`export_for_lammps` 的同名参数默认 `False`。它与 `receptive_field_layers` 一同构成 v3 的字段增量，必须列进 `export.py` 的模块 docstring 与 `lammps_contract.h` 的 docstring 所记录的 v3 字段集。（默认值不对称是有意的：每个模型都**有**感受野，作者必须表态；几乎没有模型用分段聚合，opt-in 恰如其分。）

**v2→v3 是一次硬破坏，必须记录并给出出路。** v3 的输入与输出张量数目、名称都变了，因此**此前所有导出产物一律失效**，必须重新导出；`lammps_contract.h` 的版本校验会把旧 meta 响亮拒绝。`interface/lammps/README.md` 须写明这一点并给出重新导出的命令形态。

**树内需要同步更新的调用点**（已核实，无遗漏）：真正的调用点只有原 `tests/test_molix/test_engine.py` 中的四处（`:104, :106, :112, :127`），它们在 `01-meta` 的拆分后位于 `tests/test_molix/test_engine/test_export.py`；其余出现均为文字说明——`src/molix/engine/__init__.py:15` 的模块 docstring 示例与 `interface/lammps/README.md:47, 65, 79, 147` 的示例，需随 v3 一并更新。**`scripts/` 下没有调用点。**

### 顺带修复一：per-atom 能量的静默丢弃（iron law）

`src/molpot/derivation/protocol.py` 的 `absorb_model_output`：TensorDict 分支（`:158-159`）正确读嵌套 `("atoms","energy")`（`ATOMIC_ENERGY_KEY`，`src/molix/schema.py:22`）；**plain-dict 分支（`:161-167`）读的是无名魔法字符串 `out.get("atomic_energy")`**。该拼写没有任何常量为其背书。一个返回 plain dict、按 schema 文档理解 per-atom 能量的模型，只要拼写不是恰好这七个字母，`.get` 返回 `None`，`write_energy` 跳过写入，per-atom 能量**被静默丢弃**。在 05 之前这只是功能缺失；05 之后 per-atom 能量是 `eng_vdwl` 的唯一来源，静默丢弃会变成静默的零能量。

修法：在 `src/molix/schema.py` 中于 `ATOMIC_ENERGY_KEY` 旁新增具名常量 `ATOMIC_ENERGY_FLAT_KEY = "atomic_energy"`（即嵌套键的扁平别名，供 plain dict / 扁平导出约定使用），`protocol.py:166` 与 `write_energy` 的 kwarg 名统一引用它，并在 docstring 中把"嵌套键 ↔ 扁平别名"的对应关系写死。导出侧的 `outputs` 张量名同样用这一个拼写。**全树只留一个名字**（`node_energies` 之类的第三拼写不得出现）。

### 顺带修复二：小 r 正则化不统一（iron law）

`src/molix/engine/adapter.py:111` 以 `clamp(min=1e-6)` 截断 `edge_dist`，`src/molix/engine/static.py:62` 以 `clamp(min=1e-8)` 截断——两条被文档声称等价（`static.py` 的模块 docstring 明言 padded 边"不改变权重或物理"）的路径上跑着**两个相差两个量级的正则化**。统一为 `src/molix/schema.py` 中的一个具名常量（取更保守的 `1e-6`，与主路径一致），两处共同引用。

需要说明的是，这**不是纯粹的重构**：统一到 `1e-6` 会让 `static.py` 路径上距离处于 \([10^{-8},10^{-6})\) 区间的边的 `edge_dist` 发生改变。该区间在物理上是退化的（两原子重合到 0.01 pm 以内），任何真实构型都不会落在其中，且该路径的 padded 边被置于 \(10 r_c\) 之外、根本不靠 clamp 生效。因此这是一次**阈值以下的行为改变**，影响面为空但不为零——如实记录，不以"纯重构"之名带过。

### Reuse decision

- generalize `EvalResult`（`04-facade`）→ 新增 `e_atom`；这是 04 记录的扩展点在生产者到场时的兑现。`needs_pair_reverse_comm` 保留并置恒 `false`（归约所有权缝的 05 侧答案）。
- generalize `NeighborNeeds::ghost`（`04-facade`，当时恒 false）→ `supports_pbc` 契约置 true；facade 侧翻译代码一字不动。
- generalize `LammpsView`（`02-view` / `03-graph`）→ 随消费者新增 `gnum`。
- generalize `LammpsRunEnv`（`04-facade`）→ 随 D3.2 校验代码新增 `ghost_cutoff` / `skin`。
- generalize `GraphBridge`（`03-graph`）→ 遍历域扩到 `inum+gnum`，`std::sort` 降级为 `requires_sorted_edges` 契约驱动；cutoff 重过滤与 `neigh_mask` 逻辑不变。
- generalize `ModelOutputs` / `read_model_outputs` / `DynamicExec` / `GraphedExec`（`03-graph`）→ 输出面扩 `atomic_energy`，下标按 `outputs` 名单解析 + 具名常量，校验集中在单一解码器。
- generalize `CutoffGraphImpl::init`（`04-facade`）→ 追加 newton 强制与 D3.2 校验；单域校验的非 pbc 分支保持原样。
- generalize `LAMMPS_META_SCHEMA` / `LammpsContract`（`01-meta`）→ v2 升 v3，输入 arity 表更新，新增 `receptive_field_layers` / `requires_sorted_edges`。
- generalize `write_energy` / `absorb_model_output`（`src/molpot/derivation/protocol.py:118-169`）→ 扁平别名具名化，消除静默丢弃。
- generalize `MolnexTensorDictAdapter` / `StaticForward`（`src/molix/engine/adapter.py`、`static.py`）→ 新增 `local_mask` 输入与 per-atom 能量输出，统一 clamp 常量。
- reuse `ForceScatter` / `stage_nodes` / `SpeciesMap`（`02-view`）、`pad_edges` / reverse-comm 自由函数（`03-graph`）、穷举 switch 派发与 facade 机械（`04-facade`）—— 一概不改实现。
- reuse `ATOMIC_ENERGY_KEY`（`src/molix/schema.py:22`）与 `molpot` 既有的 per-atom 能量书写协议 —— 不新建第二条 per-atom 能量通路。
- pattern `mir-group/pair_nequip_allegro`（`pair_nequip_allegro.cpp:146, 387-400`）→ ghost 遍历范围与 local-only 能量求和的参考实现。
- new `receptive_field_layers` / `requires_sorted_edges` meta 字段与 D3.2 校验 —— 树内无先例；`src/molix/md/neighbors.py` 的 skin 逻辑管的是重建频率，不是感受野覆盖，二者不可复用。
- deferred SevenNet 逐层通信、`TorchScriptExec`、`fixed-graph` 契约（连同 **tag↔local 映射**与 `requires_sorted_edges=True` 的首个消费者）→ 本链之外。

## Files to create or modify

- `src/molix/schema.py`
- `src/molpot/derivation/protocol.py`
- `src/molix/engine/adapter.py`
- `src/molix/engine/static.py`
- `src/molix/engine/export.py`
- `src/molix/engine/__init__.py`
- `src/molrep/interaction/aggregation.py`
- `tests/test_molix/test_engine/test_adapter.py`
- `tests/test_molix/test_engine/test_static.py`
- `tests/test_molix/test_engine/test_export.py`
- `tests/test_molpot/test_derivation/test_protocol.py`
- `interface/lammps/lammps_contract.h`
- `interface/lammps/bridge/lammps_view.h`
- `interface/lammps/bridge/graph_bridge.h`
- `interface/lammps/bridge/graph_bridge.cpp`
- `interface/lammps/bridge/exec.h`
- `interface/lammps/bridge/exec.cpp`
- `interface/lammps/bridge/dynamic_exec.cpp`
- `interface/lammps/bridge/graphed_exec.cpp`
- `interface/lammps/impls/contract_impl.h`
- `interface/lammps/impls/cutoff_graph_impl.h`
- `interface/lammps/impls/cutoff_graph_impl.cpp`
- `interface/lammps/pair_molnex.cpp`
- `interface/lammps/pair_molnex.h`
- `interface/tests/test_bridge_graph.cpp`
- `interface/tests/test_contract_impl.cpp`
- `interface/tests/test_lammps_contract.cpp`
- `interface/tests/generate_fixtures.py`
- `interface/lammps/README.md`
- `scripts/lammps_iface_parity.py`
- `.github/workflows/ci.yml`

## Tasks

- [ ] Write failing unit tests for the flat per-atom-energy alias and the unified edge-distance clamp (tests/test_molpot/test_derivation/test_protocol.py → TestAbsorbModelOutput; tests/test_molix/test_engine/test_adapter.py and test_static.py → TestEdgeDistanceClamp)
- [ ] Pin ATOMIC_ENERGY_FLAT_KEY and the edge-distance clamp constant in src/molix/schema.py and consume them from src/molpot/derivation/protocol.py, src/molix/engine/adapter.py and src/molix/engine/static.py
- [ ] Write failing unit tests for schema v3, local_mask energy masking, receptive_field_layers and requires_sorted_edges (tests/test_molix/test_engine/test_export.py → TestLammpsMetaBlockV3, TestLocalMask)
- [ ] Implement schema v3 in src/molix/engine/export.py (edge_mask rename, local_mask input, atomic_energy output, receptive_field_layers without default, requires_sorted_edges defaulting False) and the local_mask input plus per-atom energy output in src/molix/engine/adapter.py and src/molix/engine/static.py, emit the v3 fixture from interface/tests/generate_fixtures.py, and update the prose examples in src/molix/engine/__init__.py
- [ ] Write failing ctest cases for the ghost closure, local-only site-energy partition, order-shuffle invariance, contract-driven sorting, the six-atom no-transpose lock and the newton, ghost-cutoff and receptive-field rejections in interface/tests/test_bridge_graph.cpp, test_contract_impl.cpp and test_lammps_contract.cpp
- [ ] Implement the ghost-inclusive edge closure in interface/lammps/bridge/graph_bridge.{h,cpp} and demote the per-step std::sort to the requires_sorted_edges contract flag, adding gnum to interface/lammps/bridge/lammps_view.h
- [ ] Implement per-atom energy of length nlocal and local-only summation in interface/lammps/impls/cutoff_graph_impl.{h,cpp}, adding e_atom to EvalResult and ghost_cutoff plus skin to LammpsRunEnv in interface/lammps/impls/contract_impl.h, setting needs_pair_reverse_comm to a documented constant false with its newton precondition and cheap re-assertion, and extending ModelOutputs plus read_model_outputs in interface/lammps/bridge/ to carry and validate atomic_energy with name-resolved output indices (virial moves from index 2 to 3)
- [ ] Implement the newton-pair enforcement and the D3.2 fat-ghost check in CutoffGraphImpl::init, and read receptive_field_layers, requires_sorted_edges, the v3 expected version and updated input arity in interface/lammps/lammps_contract.h
- [ ] Extend the facade in interface/lammps/pair_molnex.{cpp,h} to tally eatom over the returned extent and to snapshot the raw ghost cutoff and skin into LammpsRunEnv
- [ ] Rewrite the sorted-order contract language in src/molix/engine/adapter.py and interface/lammps/README.md as a conditional promise driven by requires_sorted_edges, add the CSR guard comment to src/molrep/interaction/aggregation.py, and record the v2-to-v3 break, the re-export command, the input positional/dtype table and the ac-014 runbook entry in interface/lammps/README.md
- [ ] Extend scripts/lammps_iface_parity.py with a periodic multi-rank case validated against direct Python evaluation, and extend the interface ctest job in .github/workflows/ci.yml
- [ ] Run full check + test suite

## Testing strategy

- **单元（Python，`tests/test_molix/test_engine/` 镜像包与 `tests/test_molpot/` 镜像，单函数/单方法）**
  - `TestAbsorbModelOutput`（目标 `src/molpot/derivation/protocol.py::absorb_model_output`）：plain dict 携带 per-atom 能量时被写入 `ATOMIC_ENERGY_KEY`；使用非规范拼写时**不再静默通过**；TensorDict 分支行为不变。
  - `TestEdgeDistanceClamp`（目标 `adapter.py` / `static.py` 各自的距离构建）：两条路径对同一退化输入产出**同一** `edge_dist`，且都等于 schema 常量。这条测试的存在本身就是"两条路径等价"这一文档声明的执行化。
  - `TestLocalMask`（目标 `EngineForward` / `StaticForward`）：全 1 掩码时能量等于全体位点能量之和；半掩码时等于被掩码子集之和（硬编码字面量）；掩码位点的力**非零**（这是 D2 的核心——ghost 位点没有自己的位点能量却必须有力）；掩码为全 0 时能量为 0 且力为 0。
  - `TestLammpsMetaBlockV3`（目标 `_lammps_meta_block`）：`schema_version == 3`；`outputs` 含 `atomic_energy` 且 `supports_virial` 时含 `virial`；`inputs` arity 按 execution 为 4 / 5 且 cuda-graph 下为 `["Z","pos","edge_index","edge_mask","local_mask"]`；`receptive_field_layers` 存在且为正整数、缺该参数时 `export_for_lammps` 抛 `ValueError` 且文案同时含关键字名与重新导出命令；`requires_sorted_edges` 存在且默认 `False`。
- **单元（C++ ctest，无 fixture，合成含 ghost 的 `LammpsView` + 桩 `Exec`，受 CI 门禁）**
  - `impl_energy_sums_locals_only`：手工构造一个小周期体系的 `LammpsView`，显式给出 ghost 镜像；桩 `Exec` 返回每位点常数能量；断言 `EvalResult::e_total` 等于 **local 位点数 × 常数 × energy_conv** 的字面量，而非 `nall` 倍。这是 D1 的直接锁。
  - `impl_e_atom_locals_only`：`e_atom` 长度为 `nlocal` 且内容正确。
  - `impl_reports_no_pair_reverse_comm`：`cutoff_graph` 契约在有无 ghost 的两种情形下 `needs_pair_reverse_comm` 皆为 `false`。
  - `impl_compute_reasserts_newton_invariant`：`supports_pbc && !newton_pair_` 时 `compute` 抛出。
  - `bridge_graph_ghost_closure`：遍历 `inum+gnum` 后，**每个 local 原子的完整入边集**都在边表中；进一步用两个桩聚合分别模拟 **MACE 方向（聚到 target）** 与 **PiNet 方向（聚到 source）**，断言二者在该闭包下都得到与全枚举参考一致的结果——D4 的方向无关性锁，也是"一套 builder 服务两类模型"这一主张的证据。
  - `bridge_graph_order_shuffle_invariance`：`requires_sorted_edges == false` 时打乱 `firstneigh` 顺序，经 scatter 归约的能量在 fp 容差内不变（fp64 用 1e-12，依 D6 的 \((n-1)u\) 估计）。
  - `bridge_graph_sorts_only_when_contract_requires`：同一输入下 `requires_sorted_edges == true` 产出按 (source,target) 字典序非降的边表，`false` 产出保持邻居表原始顺序——两者的边**集合**相同。
  - `bridge_read_model_outputs_six_atoms_no_transpose`：N=6 输入下 virial 与 atomic_energy 未被互换（两道尺寸校验在 N==6 同时失明，此例是唯一的值级防线）。
  - `impl_init_rejects_newton_off_with_pbc` / `impl_init_rejects_insufficient_ghost_cutoff` / `contract_requires_receptive_field_layers`：三条 fail-closed 拒绝路径，文案分别含 `newton on`、`comm_modify cutoff <value>` 与字段名。
  - `impl_init_accepts_strictly_local_default_cutoff`：\(T=1\) 且默认 ghost 截断时**不**报错——证明 D3.2 的校验不误伤严格局域模型。
  - 既有全部用例（`meta_*` / `contract_*` / `bridge_*` / `impl_*` / `registry_*`）保持绿；`contract_reads_generated_v3_fixture` 承接 `01-meta` 的双向绑定，任一侧版本单独改动即红。
- **覆盖边界**：facade 自身的回调体仍无法在无 LAMMPS 下执行（`04-facade` 已记录）；`eatom` 的 tally 由代码级判据与 parity 工具覆盖。
- **LAMMPS parity（本地门禁，含负责人与附运行记录要求）**：`scripts/lammps_iface_parity.py` 扩出一个**周期体系**算例（水盒或 NaCl）：(a) 单 rank 与**多 rank**（≥2）运行给出同一总能量与力，容差 rel 1e-6；(b) 该能量与**Python 侧直接求值**（同一模型、同一构型、全体原子一次前向）一致；(c) 既有 aspirin 算例保持 parity。**多 rank 一致性是 D1+D2 的唯一端到端证据**——单 rank 下 ghost 仍存在（周期镜像）故 (b) 已能暴露 D1，但 (a) 才同时暴露 D2 的双重通信。**ac-014 同时是 D6 的端到端证据**——C++ 侧按 LAMMPS 原始邻居顺序建边，Python 直接求值按 `nonzero()` 的有序输出，两者在真实导出模型上一致即为满量程的顺序不变性验证。Provenance 块须记录 LAMMPS 版本、rank 数、`atom_style`、`comm_modify` 设置、插件 commit、torch 版本、device/precision、capture command 与日期。无 LAMMPS 构建时**非零码响亮退出，绝不 skip**。
- **不设 `regressions/` 脚本**：沿用 `02`/`03`/`04` 已记录在案的例外（`regressions/README.md:3-6` 禁止 subprocess，本 spec 的门禁产物为 ctest + CI + 上述工具）。Python 侧的新增行为由 `tests/` 单测覆盖，不另起回归脚本。

## Out of scope

- **SevenNet 式逐层通信**（真域分解下每层 halo 交换）：显存与可扩展性更优，但属另一种执行模型，列为未来 fork。
- **tag↔local 映射**：本方案不需要（fat-ghost 单发不做索引重映射）。它归**未来的 `fixed-graph` 契约** spec——拓扑型 FF 需要把 LAMMPS tag 对上分子拓扑编号。本 spec 内不得出现。
- `TorchScriptExec`、espaloma / `fixed-graph` 契约实现、per-atom virial 的生产（`vflag_atom` 仍 `error->all`）。
- 性能：pinned memory、异步拷贝、持久缓冲、ghost 遍历带来的边数增长的任何优化。
- `num_models` 的 pair_style 文法（按 `04-facade` 的裁决，随引入热重载消费者的 spec 一同发布）。
- **链级待办（链收尾时处理，不属本 spec）**：`Contract::fixed_graph` 的"落地实现或按 schema 规则删除"（`01-meta` 定下的规矩；`requires_sorted_edges` 的 `true` 分支同属"无树内消费者"纪律）；`.claude/notes/architecture.md` 对 `interface/` 的蓝图补齐（建议链结束后一次性 `/mol:map`）；`04-facade` ac-013 与本 spec ac-014 的 parity 运行记录（pending 期间链收尾提交被阻塞）。
