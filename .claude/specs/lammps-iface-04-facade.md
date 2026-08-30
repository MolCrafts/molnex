---
title: ContractImpl 契约层、闭集派发与 facade 削薄至协议机械
status: approved
created: 2026-08-30
slug: lammps-iface-04-facade
chain: lammps-iface
grilled: false
---

# ContractImpl 契约层、闭集派发与 facade 削薄至协议机械

## Summary

落地三层架构的最后一层：`ContractImpl` 抽象基类（携带 `NeighborNeeds` / `EvalResult` / `LammpsRunEnv` 三个 attribute-only 契约类型）、以 `Contract` 枚举为键的**穷举 `switch` 派发**、以及组合 `02-view` / `03-graph` 全部组件的首个实现 `CutoffGraphImpl`（PiNet / MACE / Allegro 共享，按运行时契约族而非网络分类，故不触犯 "no per-model C++"）。`PairMolnex` 随之削到纯 LAMMPS 协议机械——参数管道、`allocate`、邻居请求、`compute` 转发、comm 转发、tally——**不含任何语义 if**：`supports_pbc`、`use_cuda_graph`、`n_graph` 的计算与 nghost 守卫全部迁入 impl。用户裁决的五条 facade 接口决定①-⑤逐条落地。同时修掉本表面上的两处静默失真（iron law）：`eflag_atom` / `vflag_atom` 被 LAMMPS 请求而无实现提供时**静默写零**（现行 `ev_init` 之后从不写 `eatom` / `vatom`），改为 `error->all`；以及新发现的 **`special_bonds` 缺陷**——分子体系下 LAMMPS 默认的 `special_bonds lj/coul 0 0 0` 会把 1-2 / 1-3 / 1-4 对**整个排除出邻居表**，`GraphBridge` 因而永远看不到这些最近邻，多体 ML 势的图残缺而无任何报错。另记录一处**已敞开但不在本 spec 铺设文法**的扩展口：`ModelRunner` 的 `num_models` 构造参数（`pair_molnex.cpp:135` 处 facade 调用点仍传 1）。除上述两条新增报错路径外行为不变，parity 判据沿用 rel 1e-6。

## Design

### 前置条件（硬依赖）

本 spec 在 `01-meta` / `02-view` / `03-graph` 全部落地前不可实现：它消费 `LammpsContract`（`Contract` / `Execution` 枚举、`units_for`、`species`）、`SpeciesMap` / `stage_nodes` / `ForceScatter` / `LammpsView`、`GraphBridge` / `Exec` / `read_model_outputs` / `comm_reverse`。

### 契约类型（`interface/lammps/impls/contract_impl.h`，无 LAMMPS 头）

三个 attribute-only 聚合体，形状照 `src/molix/md/types.py:21` `ForceOutput`（无方法），属性契约照 `src/molix/md/neighbors.py:329` `NeighborStrategy`：

```
struct NeighborNeeds {
  bool   full;              // 请求全邻居表（多体聚合必需）
  bool   ghost;             // 请求 ghost 邻居（REQ_GHOST）——仅描述邻居表请求
  double cutoff;            // Å
  int    comm_reverse;      // reverse 方向每原子 double 数
  int    comm_reverse_off;  // newton off 变体
};

struct EvalResult {
  double     e_total;                  // 模型能量单位已换算至 LAMMPS 单位
  at::Tensor virial;                   // 可选，(6,) xx,yy,zz,xy,xz,yz；virial.defined() 即存在性
  bool       needs_pair_reverse_comm;  // impl 是否要求 facade 在 pair 层发起 reverse comm
};

struct LammpsRunEnv {       // facade 从 LAMMPS 全局摘出的只读快照
  int           newton_pair;
  int           nprocs;
  int           molecular;   // 0 = atom_style atomic（无键拓扑）
  const double* special_lj;   // (4,)
  const double* special_coul; // (4,)
};
```

**`EvalResult` 不含力**（binding decision 2）：impl 在返回前已通过 `ForceScatter` 把力累加进 `atom->f`——这是必须的，`reverse_comm` 读的就是 `atom->f`。该不对称性必须写进结构体 docstring，否则会被读成遗漏。`EvalResult::virial` 因 `03-graph` 的 `read_model_outputs` 在定义前校验 `numel == 6`，**在构造上"良构或未定义"**——facade 不做任何 virial 长度检查。

**`needs_pair_reverse_comm` 是归约所有权缝，与力的不对称性写在一起。** 该字段每步回答一个问题：**pair style 自己是否必须发起 reverse comm**。`supports_pbc` 从 facade 删除之后，这个问题在 facade 侧没有干净的数据来源，于是把它交还给知道答案的一方：impl 用**与今天完全相同的输入**求值 `supports_pbc && view.nghost > 0` 并填入，facade 只做 `if (r.needs_pair_reverse_comm) comm->reverse_comm(this);`——对返回值的机械分支，不是对模型能力的读取，因此不构成语义 if。字段名描述的是 **facade 该做什么**，而非"ghost 力是否已散射"。

**这条缝是长期结构，不是过渡脚手架。** 04 的 impl 回答"有 ghost 就要"，`05-pbc` 的 impl 回答"永远不要，因为积分器拥有归约"——**同一个问题的两个答案，字段与 facade 调用点在两版之间一字不动**。这正是本层存在的意义：归约所有权的转移应当表现为 impl 换一个返回值，而不是对协议层动手术。`EvalResult` 的 docstring 须按此措辞标注，并说明未来某个"在 newton 归约的积分之外散射 ghost 力"的契约会把它重新填为 `true`。

**该字段不是无生产者的字段**：生产者就是本 spec 的 `CutoffGraphImpl::compute`。它与 `NeighborNeeds::ghost` **相互独立**——后者只描述向 LAMMPS 请求哪种邻居表（本 spec 恒为 `false`），前者描述归约由谁发起；`05-pbc` 会把 `ghost` 翻成 `true` 而把 `needs_pair_reverse_comm` 翻成 `false`，两者朝相反方向变化，正说明它们不可合并。

**`EvalResult` 不含 per-atom 能量或 per-atom virial 字段。** 裁决③要求"为 `eflag_atom` / `vflag_atom` 留位"；本 spec **没有任何生产者**能填充它们（模型输出只有 energy / forces / 可选全局 virial）。整条链一贯拒绝无消费者的字段，此处同理拒绝无生产者的字段。"留位"由 **facade 的显式报错路径** + 结构体 docstring 记录的扩展点共同满足。该扩展点**已确定在 `05-pbc` 兑现**：per-atom 能量正是其修复的核心（\(E^{(p)}=\sum_{i\in L}E_i\) 必须逐位点求和），届时 `EvalResult` 获得 `e_atom` 是**本设计预定的扩展，不是对本决定的推翻**——docstring 必须这样措辞。

**`EvalResult` 与 `LammpsRunEnv` 各自的 docstring 承载同一条累积规则（consumer/producer-arrives）**：`LammpsRunEnv` 只携带**当前有代码读取**的 LAMMPS 全局量，后续 spec 添加字段必须与读取它的代码同批落地（例如 `05-pbc` 为 D3.2 校验加入 ghost 截断值）；`EvalResult` 的字段是**一次求值的每步输出**，每个字段必须与其生产者和消费者同批落地，将来"请 facade 顺便再做 X"这类**义务**需要另一个归宿——`needs_pair_reverse_comm` 是唯一被明确豁免的成员，它记录的是归约所有权，是这条缝的定义性内容。把规则写在结构体上而非散在各 spec 里，是因为最容易违反它的地方就是这两个结构体。

**`NeighborNeeds` 不含 `single_rank`。** 裁决①把"单 rank 约束"列入 needs，但 needs 是 facade **机械执行的请求表**；把一个谓词塞进去会迫使 facade 解释它，即产生一个语义 if——恰恰违反同一批裁决里"facade 不含任何语义 if"的第一条。约束的**执行**按裁决⑤归 `init()` 钩子。

`ghost` 字段当前恒为 `false`（`CutoffGraphImpl` 现请求 `REQ_FULL`，`REQ_GHOST` 归 `05-pbc`），且**只描述向 LAMMPS 发起的邻居表请求**，不承载任何关于力归约的含义（那是 `EvalResult::needs_pair_reverse_comm` 的事）。它**不是**投机字段：facade 侧对应的是一行机械翻译 `if (needs.ghost) flags |= NeighConst::REQ_GHOST;`——请求表若含 facade 不执行的条目，这张表本身就是假的；`05-pbc` 只需把该字段翻成 `true` 而不必再动 facade，这正是本层存在的理由。

### `ContractImpl` 基类与五条接口决定

```
class ContractImpl {
 public:
  virtual ~ContractImpl();
  virtual NeighborNeeds needs() const = 0;                                       // ①
  virtual void coeff(const std::vector<std::string>& tokens, int ntypes) = 0;    // ④
  virtual void init(const LammpsRunEnv& env) = 0;                                // ⑤
  virtual EvalResult compute(const LammpsView& view) = 0;                        // ③
  virtual int  pack_reverse(const LammpsView&, int n, int first, double* buf) = 0;              // ②
  virtual void unpack_reverse(const LammpsView&, int n, const int* list, const double* buf) = 0; // ②
};
```

④ 的落法：facade 只负责剥掉 `* *` 这两个必须的 token 并校验其为字面 `*`，**剩余 token 原样以 `std::vector<std::string>` 交给 impl**，连 token 数校验（现 `cpp:167-168`）都归 impl。impl 用标准库解析整数——**不使用 `utils::inumeric`**，因为那是 LAMMPS 符号，会把 impls 层拉进 LAMMPS 依赖、毁掉其可 ctest 性。

**放弃 `utils::inumeric` 不得以放松严格性为代价。** 解析必须是**全量消费**式：`std::from_chars` 取得 `[ptr, ec]` 后，同时要求 `ec == std::errc()` **且** `ptr == token.data() + token.size()`。据此，`"6abc"`（尾部残留）、`""`（空 token）、超出 `int` 范围的字面量（`std::errc::result_out_of_range`）三类输入全部被拒并抛出含原 token 的诊断。`std::stoi` 在此不可接受——它对 `"6abc"` 静默返回 6，把一个拼写错误变成一个错误的元素。

### `CutoffGraphImpl`（`interface/lammps/impls/cutoff_graph_impl.{h,cpp}`）

状态：`LammpsContract` 的相关取值（cutoff、`supports_pbc`、`supports_virial`、`species`、`energy_conv`）、`std::optional<SpeciesMap>`、`GraphBridge`、`ForceScatter`、**构造注入的** `std::unique_ptr<Exec>`。

- `needs()` → `{full=true, ghost=false, cutoff=<contract cutoff>, comm_reverse=3, comm_reverse_off=3}`，与 `cpp:62-63, 191` 现值逐一相同。
- `coeff(tokens, ntypes)`：要求 `tokens.size() == ntypes`，逐个全量消费解析为 Z，构造 `SpeciesMap`（其构造即完成 Z↔meta species 校验，`02-view` 已落地）。
- `init(env)`：三项校验，前两项逐字保留现行语义，第三项是新增的缺陷拦截。
  1. 单域约束：`!supports_pbc && env.newton_pair == 1 && env.nprocs > 1` → 抛出，文案逐字沿用 `cpp:187-189`。该条件**照搬**，包括它只在 `newton_pair == 1` 时触发这一点——收紧它属行为变更，归 `05-pbc`。
  2. `execution == cuda_graph` 相关的既有前置（device、与 `supports_pbc` 互斥）在 `settings()` 时已由契约层校验，此处不重复。
  3. **`special_bonds` 拦截（新增，见下）**。
- `compute(view)`：计算 `n_graph = supports_pbc ? view.nall() : view.nlocal`（从 facade `cpp:219` 迁入）；`!supports_pbc && view.nghost != 0` 时抛出（从 `cpp:202-206` 迁入）；`stage_nodes` → `GraphBridge::build` → `Exec::run` → `ForceScatter::scatter`；把能量乘 `energy_conv` 装进 `EvalResult::e_total`，`supports_virial` 且模型给出 virial 时把乘过 `energy_conv` 的 `(6,)` 装进 `EvalResult::virial`；并以 `supports_pbc && view.nghost > 0` 填 `EvalResult::needs_pair_reverse_comm`——与 `cpp:351` 今天的判据逐字相同的输入、逐字相同的表达式。
- `pack_reverse` / `unpack_reverse`：转调 `03-graph` 的 `pack_reverse_forces` / `unpack_reverse_forces`。

**`Exec` 由构造注入**（`std::unique_ptr<Exec>`），不在 impl 内部按 `execution` 分支自建。这有三重收益：impl 不含执行策略的语义 if；`03-graph` 声明的 backend-neutral 性质被真正证明（测试注入一个非 AOTI 的桩实现）；**整个 Layer-2 + Layer-3 栈因此可以在没有 LAMMPS、没有 CUDA、没有 `.pt2` 的机器上被 ctest 驱动**，从而进 CI 门禁。`execution → Exec` 的映射归派发层。

### 新发现的活缺陷：`special_bonds` 排除键合近邻

LAMMPS 在 `special_bonds lj/coul` 系数为 `0.0` 时，把对应的 1-2 / 1-3 / 1-4 原子对**从邻居表中整体剔除**，而 `0.0 0.0 0.0` 正是 LAMMPS 的默认值。`pair_style molnex` 消费的是 cutoff 图：任何键合近邻被剔除，`GraphBridge` 就永远看不到它们，多体 ML 势因此在残缺图上求值——能量与力都错，且**全程无任何报错**。现行 `init_style()`（`cpp:185-192`）对此零校验。

该缺陷仅在体系带键拓扑时可达（`atom_style atomic` 无 special 对，故现有 aspirin 基准不受影响，parity 不受影响）。裁决⑤已把 `special_bonds` 校验明确划归 impl 的 init 钩子，故本 spec 修之而非仅记录：`CutoffGraphImpl::init` 在 `env.molecular != 0` 时要求 `special_lj[1..3]` 与 `special_coul[1..3]` **全为 `1.0`**，否则抛出，文案指明需 `special_bonds lj/coul 1.0 1.0 1.0` 并说明原因（molnex 不施加任何 special 缩放，非 1.0 即表示 LAMMPS 认为应有缩放而模型不会施加）。`env.molecular == 0` 时不作要求——这样既拦住错误配置，又不误伤现有的原子体系运行，parity 得以保持。

### 闭集派发（`interface/lammps/impls/registry.{h,cpp}`）

`static std::unique_ptr<ContractImpl> ContractImpl::for_contract(Contract, const LammpsContract&, molnex::interface::ModelRunner&)`。命名刻意避开 CLAUDE.md 禁止的 `make_*` / `create_*`：这不是对单一类型构造的包装别名，而是闭集上的多态派发，语义与 `Foo.from_file` 同类。

**派发用对枚举穷举的 `switch`，不用表。** 每个枚举量都有显式 `case`，**不写 `default:` 标签**——于是将来向 `Contract` 添加枚举量时 `-Wswitch` 会在**编译期**指出这里未处理，而 `std::map` 只会在运行期给出一次未命中。`Contract::fixed_graph` 的 `case` 显式抛出，文案列全部有效契约名（风格照 `src/molix/engine/adapter.py:58-60`）；`switch` 之后放一条不可达的 `throw` 以满足编译器对返回路径的要求。`Execution → Exec` 的映射同形处理。表机制（function-local `static const std::map`）在此不引入；若将来落地第三个契约、派发确实变得数据驱动，再重新评估表形式。

**`for_contract` 派发的是各实现的 public 直接构造函数**，不是它们的唯一入口：`CutoffGraphImpl` 的构造函数保持 public，直接构造仍是首要构造故事（CLAUDE.md）。因此 ctest **必须直接构造 `CutoffGraphImpl` 并注入桩 `Exec`，不得经由 `for_contract`**——后者会拽进 `LammpsContract` 与 `ModelRunner`，把一个组合测试变成集成测试，也会让"直接构造可用"这一性质失去覆盖。

本 spec 只实现 `Contract::cutoff_graph`；`fixed_graph` 是一次响亮的显式拒绝。`01-meta` 已定下规矩——链结束时若仍无 `fixed-graph` impl 落地，枚举值与本派发分支一并在 schema 升版下退化为直接构造。

### facade 削薄后的 `PairMolnex`

保留的成员仅：`model_dir` / `model_name`、`std::optional<LammpsContract>`、`std::unique_ptr<molnex::interface::ModelRunner>`、`std::unique_ptr<ContractImpl>`、`NeighborNeeds`、以及 LAMMPS 协议标志（`single_enable=0`、`restartinfo=0`、`one_coeff=1`、`manybody_flag=1`）。`cut_global`、`type2z`、`device`、`model_units`、`fp64_model`、`energy_conv`、`force_conv`、`supports_pbc`、`supports_virial`、`use_cuda_graph`、`n_fixed`、`e_max` 与全部张量/主机缓冲成员**一律删除**。

各回调：

- `settings()`：解析命令行 → 读 meta → `LammpsContract` → 构造 `ModelRunner(model_dir, /*num_models=*/1, model_name)` → `ContractImpl::for_contract(...)` → 缓存 `needs()` 并据其设置 `comm_reverse` / `comm_reverse_off` → `logmesg`。
- `coeff()`：`allocate()`；校验 `arg[0] == "*" && arg[1] == "*"`；把 `arg[2..]` 收成 `std::vector<std::string>` 转交 `impl_->coeff(tokens, atom->ntypes)`；置 `setflag`。
- `init_one()`：返回 `needs_.cutoff`。
- `init_style()`：按 `needs_` 组装 `NeighConst` 标志并 `neighbor->add_request(this, flags)`；组装 `LammpsRunEnv` 并调 `impl_->init(env)`。
- `compute()`：`ev_init` → 组装 `LammpsView`（含 `neigh_mask = NEIGHMASK`）→ `EvalResult r = impl_->compute(view)` → tally → `if (r.needs_pair_reverse_comm) comm->reverse_comm(this);`。
- `pack_reverse_comm` / `unpack_reverse_comm`：各一行转发。

tally 语义（③）——**三条分支的结构逐字保留**，特别是那条"什么都不做"的分支：

```
if (eflag_global) eng_vdwl += r.e_total;
if (eflag_atom)   error->all(FLERR, "pair_molnex: per-atom energy requested (e.g. compute pe/atom) "
                                    "but contract '<name>' provides none; remove the per-atom compute");
if (vflag_atom)   error->all(FLERR, "pair_molnex: per-atom virial requested (e.g. compute stress/atom) "
                                    "but contract '<name>' provides none; remove the per-atom compute");

if (r.virial.defined() && vflag_global) {
  accumulate(virial, r.virial);          // 现行 cpp:335-346（×energy_conv 已由 impl 完成）
} else if (vflag_fdotr) {
  virial_fdotr_compute();                // 现行 cpp:347-348
}

if (r.needs_pair_reverse_comm) comm->reverse_comm(this);   // 现行 cpp:351-352 的判据，由 impl 给出
```

**不得把 virial 分支重写成"先判 `vflag_global` 再判 `defined()`"。** 现行 `cpp:335-349` 的条件结构下存在第三种情形——`vflag_global` 为真、模型未给 virial、且 `vflag_fdotr` 为假——此时**什么也不发生**。重排条件会把它变成走 fdotr 或走报错，二者都是行为变更。本 spec 是 parity spec，这条"什么都不做"的分支必须原样活着，并由 AC 逐条钉住三个分支（含它）。

现行代码从不写 `eatom` / `vatom`，故 `compute pe/atom` / `compute stress/atom` 至今**静默给出全零**——用户拿到的是一个看似合法的零场。改为响亮报错。这三个 `if` 是 LAMMPS 协议机械（读 `ev_init` 设好的标志位），不是语义 if：facade 不据模型能力分支，只据 LAMMPS 的请求分支，而"能力不足"由 `EvalResult` 的字段存在性直接给出。

**`num_models`：门是记录在案地敞着的，不是焊死的，但本 spec 不铺文法。** `ModelRunner` 的构造参数 `num_models`（`model_runner.cpp:75-122`）分配多份常量缓冲，是 `update_weights` 双缓冲热重载（`:210-257`）赖以工作的前提；facade 调用点（现 `pair_molnex.cpp:135`）**仍传 1**。暴露它的 `pair_style` 文法随**引入热重载消费者的那个 spec**（`fix molnex/reparam` 或其后继）一同发布——在没有任何调用方需要热重载的情况下先造一套命令行文法与其校验规则，就是没挣来的语法。本段与 `interface/lammps/README.md` 中对应的一行共同构成该扩展点的记录。

### 构建面

新增 `interface/lammps/impls/CMakeLists.txt` 定义 `molnex_lammps_impls`（STATIC、PIC、链接 `molnex_lammps_bridge`），由 `interface/CMakeLists.txt` **父侧** `add_subdirectory(lammps/impls)` 声明；`interface/lammps/CMakeLists.txt` 链接之。理由与 `02-view` / `03-graph` 同：`interface/CMakeLists.txt` 从不处理 `interface/lammps/CMakeLists.txt`（后者是独立 `project(molnexplugin)`，缺 `-DLAMMPS_HEADER_DIR` / `-DLAMMPS_LIB` 即 `FATAL_ERROR`，见其 `:26-31`），因此**绝不可**用 `add_subdirectory(lammps)` 从父侧"修复"目标可见性。父侧只添加 `lammps/bridge` 与 `lammps/impls` 两个自身不引用任何 LAMMPS 变量的子目录。三层依赖梯度就此完整：`bridge/` = LAMMPS-free 且不识 meta schema 的原语，`impls/` = LAMMPS-free 但认识契约的策略层（meta schema 从这里才进入），facade = 唯一接触 LAMMPS 的层。

### Reuse decision

- generalize `init_style` 的邻居请求与单域校验（`interface/lammps/pair_molnex.cpp:185-192`）→ `NeighborNeeds` + `ContractImpl::init`，并补上 `special_bonds` 拦截。
- generalize `settings` 的能力标志读取与分支（`cpp:105-159`）→ `ContractImpl::for_contract` 穷举 switch；facade 只留参数管道。
- generalize `compute` 的 `n_graph` 计算与 nghost 守卫（`cpp:202-206, 219`）→ `CutoffGraphImpl::compute`。
- generalize 能量 / virial tally 与 reverse-comm 判据（`cpp:332-352`）→ `EvalResult`（含 `needs_pair_reverse_comm` 归约所有权缝）+ facade 的机械 tally，并把静默零改为报错；virial 长度检查已由 `03-graph` 的 `read_model_outputs` 吸收。
- generalize `pack_reverse_comm` / `unpack_reverse_comm`（`03-graph` 已迁为自由函数）→ 经 `ContractImpl` 转发（裁决②）。
- reuse `SpeciesMap` / `stage_nodes` / `ForceScatter` / `LammpsView`（`02-view`）与 `GraphBridge` / `Exec` / `DynamicExec` / `GraphedExec` / `read_model_outputs` / `pack_reverse_forces` / `unpack_reverse_forces`（`03-graph`）—— `CutoffGraphImpl` 是它们的组合点，实现一概不改。
- reuse `LammpsContract` / `Contract` / `Execution` / `UnitConversion`（`01-meta`）—— `Contract` 枚举在此获得其**绑定消费者**，即本派发。
- reuse `ModelRunner`（`interface/src/model_runner.cpp`）—— 仍由 facade 拥有并注入，`run` / `run_graphed` / capture 状态一概不动；`num_models` 参数记录为已敞开的扩展口。
- reuse `interface/tests` 的 no-fixture ctest 列表与 `.github/workflows/ci.yml` 的 `interface` job（01 引入）—— 扩其覆盖。
- pattern `src/molix/md/neighbors.py:329` `NeighborStrategy`（attribute-only 协议）→ `NeighborNeeds`；`src/molix/md/types.py:21` `ForceOutput` → `EvalResult` / `LammpsRunEnv`。
- pattern `src/molix/engine/adapter.py:58-60` → 派发未命中时列全有效名的文案。
- new `ContractImpl` / `CutoffGraphImpl` —— 现无任何类型承担"运行时契约实现"职责；`cpp:219, 273` 的隐式分支正是本 spec 要显式化的对象。
- deferred `REQ_GHOST`（`NeighborNeeds::ghost` 翻真）、per-atom energy 求和语义、newton 门控、排序降级 → `lammps-iface-05-pbc`。
- deferred `fixed-graph` impl（`bond_index` `(2,N)` COO 拓扑型 FF，含 tag↔local）、`TorchScriptExec`、`num_models` 文法与热重载调用面 → 本链之外的后继 spec。

## Files to create or modify

- `interface/lammps/impls/contract_impl.h` (new)
- `interface/lammps/impls/cutoff_graph_impl.h` (new)
- `interface/lammps/impls/cutoff_graph_impl.cpp` (new)
- `interface/lammps/impls/registry.h` (new)
- `interface/lammps/impls/registry.cpp` (new)
- `interface/lammps/impls/CMakeLists.txt` (new)
- `interface/tests/test_contract_impl.cpp` (new)
- `scripts/lammps_iface_parity.py` (new)
- `interface/lammps/pair_molnex.cpp`
- `interface/lammps/pair_molnex.h`
- `interface/lammps/CMakeLists.txt`
- `interface/lammps/README.md`
- `interface/CMakeLists.txt`
- `interface/tests/CMakeLists.txt`
- `.github/workflows/ci.yml`

## Tasks

- [ ] Write failing ctest cases for NeighborNeeds, EvalResult and CutoffGraphImpl composition using an in-test stub Exec over a synthetic LammpsView in interface/tests/test_contract_impl.cpp (new), registered in the no-fixture list of interface/tests/CMakeLists.txt
- [ ] Implement ContractImpl, NeighborNeeds, EvalResult and LammpsRunEnv in interface/lammps/impls/contract_impl.h (new) with docstrings per google style recording the forces asymmetry, the reduction-ownership seam, the designed e_atom extension point and the accretion rules
- [ ] Implement CutoffGraphImpl in interface/lammps/impls/cutoff_graph_impl.h and cutoff_graph_impl.cpp (new), composing SpeciesMap, stage_nodes, GraphBridge, the injected Exec, ForceScatter and the reverse-comm free functions
- [ ] Implement the special_bonds, single-domain and full-consumption token parsing checks in CutoffGraphImpl::coeff and CutoffGraphImpl::init
- [ ] Implement the exhaustive-switch contract dispatch and the Execution-to-Exec mapping in interface/lammps/impls/registry.h and registry.cpp (new)
- [ ] Declare the LAMMPS-free molnex_lammps_impls static target in interface/lammps/impls/CMakeLists.txt (new), add it from interface/CMakeLists.txt, and link it from interface/lammps/CMakeLists.txt
- [ ] Shrink PairMolnex to protocol mechanics in interface/lammps/pair_molnex.cpp and pair_molnex.h, deleting every capability member and semantic branch, forwarding settings, coeff, init_style, compute and both comm callbacks, and preserving the three-arm virial tally structure verbatim
- [ ] Implement the eflag_atom and vflag_atom error paths in interface/lammps/pair_molnex.cpp, and document them, the special_bonds requirement and the num_models hot-reload opening in interface/lammps/README.md
- [ ] Add the LAMMPS parity tool scripts/lammps_iface_parity.py (new) with embedded aspirin goldens, a Provenance block pinning atom_style, and fail-loud handling when no LAMMPS build is present
- [ ] Extend the interface ctest job in .github/workflows/ci.yml to cover the new contract-impl cases, then run full check + test suite

## Testing strategy

- **单元（C++ ctest，无 fixture，合成 `LammpsView` + 测内桩 `Exec`，不依赖 LAMMPS、CUDA 或 `.pt2`）**——注册进 01 引入的 no-fixture 列表，因而受 CI `interface` job 门禁。全部组合用例**直接构造 `CutoffGraphImpl` 并注入测内桩 `Exec`，不经 `for_contract`**：
  - `impl_needs_matches_legacy_request`：`needs()` 给出 `full=true, ghost=false, comm_reverse=3, comm_reverse_off=3` 且 `cutoff` 等于契约值，逐项对齐 `cpp:62-63, 191` 的现值。
  - `impl_compute_composes_pipeline`：桩 `Exec` 返回闭式能量/力；断言 `EvalResult::e_total` 等于手算字面量（含 `energy_conv` 换算）、`atom->f` 已被 impl 累加（**非零初始 `f`** 起测）、`EvalResult` 不携带力字段。
  - `impl_compute_n_graph_local_only` / `impl_compute_n_graph_with_ghost`：`supports_pbc` 两种取值下 `n_graph` 分别为 `nlocal` / `nall`，锁住从 facade 迁入的语义。
  - `impl_compute_rejects_ghost_without_pbc`：`!supports_pbc && nghost != 0` 抛出，文案沿用 `cpp:203-206`。
  - `impl_virial_present_and_absent`：`supports_virial` 且桩给出 virial 时 `virial.defined()` 且已乘 `energy_conv`；否则 `!defined()`。
  - `impl_reports_pair_reverse_comm_condition`：`supports_pbc` 与 `view.nghost` 的四种组合下，`EvalResult::needs_pair_reverse_comm` 恰等于 `supports_pbc && nghost > 0`。
  - `impl_coeff_token_count`（token 数不等于 `ntypes` 抛出）；`impl_coeff_rejects_trailing_garbage`（`"6abc"` 被拒——`std::stoi` 会静默接受为 6）；`impl_coeff_rejects_empty_token`；`impl_coeff_rejects_out_of_range`。四条均要求诊断文案含原 token。
  - `impl_init_rejects_special_bonds_on_molecular`：`molecular != 0` 且任一 `special_lj[1..3]` / `special_coul[1..3]` 非 `1.0` 时抛出；`molecular == 0` 时同样的系数**不**抛出（锁住不误伤原子体系这一点）。
  - `impl_init_single_domain_condition`：仅当 `!supports_pbc && newton_pair == 1 && nprocs > 1` 抛出；`newton_pair == 0 && nprocs > 1` **不**抛出，锁住"照搬而非收紧"。
  - `impl_reverse_comm_forwards`：`pack_reverse` / `unpack_reverse` 的结果与直接调用 `03-graph` 的自由函数逐位相同。
  - `registry_returns_cutoff_graph_impl` / `registry_rejects_fixed_graph`：后者显式抛出并在文案中列出全部有效契约名。
- **覆盖边界（明确记录）**：`PairMolnex` 自身的回调体**无法在无 LAMMPS 的情况下执行**，故 facade 侧（tally 分支、`eflag_atom` / `vflag_atom` 报错、`NeighborNeeds`→`NeighConst` 翻译）由代码级判据 + 下述 parity 工具覆盖，不由 ctest 覆盖。这是本 spec 的已知覆盖缺口，写在此处而非留给读者发现；把 facade 拆到可单测会要求引入 LAMMPS 的测试替身——那是"验证编排而非属主契约"的反模式，成本远超收益。
- **迁移不回归**：`interface/tests` 既有全部用例（`model_runner` 系、`meta_*` / `contract_*`、`bridge_*`）在本次迁移后仍绿。
- **LAMMPS parity（本地门禁，非 CI，含负责人）**：`scripts/lammps_iface_parity.py` 对同一 aspirin 输入跑 `pair_style molnex`，把**重构前**记录的能量与力以十进制字面量内嵌，判据为 rel 1e-6（非逐位相等——`01-meta` 已把换算因子从硬编码浮点换成 pint 求值，相对偏移约 1e-8）。脚本头带 `regressions/oplsaa-parameterizer-02-heads.py:12-18` 形制的 Provenance 块，记录 LAMMPS 版本、**`atom_style`**、插件 commit、torch 版本、device/precision、capture command 与日期。**工具头与本 spec 均记录：aspirin 是 `molecular == 0`，因此 parity 完全不覆盖 `special_bonds` 修复——`impl_init_rejects_special_bonds_on_molecular` 的两半是该修复的唯一覆盖。** 无 LAMMPS 构建时以非零码响亮退出，绝不 skip。CI 无 LAMMPS，故该门禁是**记录在案的本地债**：交付时必须附上一次真实运行的输出（数值 + 环境），负责人为本链实现者；未附运行记录则对应验收判据不得置为通过，且**链的收尾提交被阻塞**（不只是本 spec）。
- **不设 `regressions/` 脚本（沿用 `02-view` / `03-graph` 的记录在案例外）**：`regressions/README.md:3-6` 定义该目录为纯 Python 公开面场景脚本且禁止 subprocess；本 spec 全部产出为 C++ 与一个必须驱动外部 LAMMPS 二进制的工具，二者都不符合该目录的契约。绑定门禁是 ctest 用例 + CI job + 上述 parity 工具。`scripts/` 而非 `regressions/` 正是为此。

## Out of scope

- **已命名的活缺陷，归 `lammps-iface-05-pbc`（本 spec 只搬迁其相邻代码，不修）**：`pair_molnex.cpp:332-333` 在 `supports_pbc=true` 时把 local + ghost 全部位点的总能量整体计入 `eng_vdwl`，ghost / 镜像位点被重复计入（`02-view` 已点名）。本 spec 把该 tally 迁成 `EvalResult::e_total` 的机械累加，**语义原样保留**；`comm_reverse` 直接 pack 共享 `atom->f` 的缺陷（`03-graph` 已按 F3 标准具名）同源、同归 `05-pbc`。
- `REQ_GHOST`（把 `NeighborNeeds::ghost` 翻真）、per-atom energy 求和语义、newton 门控、单域约束的收紧、排序降级为契约标志：`lammps-iface-05-pbc`。
- `fixed-graph` 契约实现（拓扑型 FF，载荷为 `bond_index` `(2,N)` COO，含 tag↔local）与 espaloma 经典档 evaluator（默认走 LammpsWriter 原生系数，裁决 5）。
- `TorchScriptExec` 本体（`Exec` 接口已由 `03-graph` 保证可容纳）。
- per-atom 能量 / per-atom virial 的**生产**（本 spec 只把静默零改为报错；per-atom 能量的生产者是 `05-pbc`）。
- `num_models` 的 pair_style 文法与 `update_weights` 热重载调用面（随引入热重载消费者的 spec 一同发布；本 spec 只记录 `ModelRunner` 构造参数这一已敞开的扩展口）。
- 性能优化：pinned memory、异步拷贝、去排序、多 runner 并发调度。
- bridge / impls 下沉到 `interface/include`（等第二个引擎出现再议）。
- `.claude/notes/architecture.md` 对 `interface/` 的蓝图补齐（scope gap；链结束后一次性 `/mol:map`）。
