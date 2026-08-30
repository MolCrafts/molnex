---
title: GraphBridge 邻居图构建、reverse-comm 搬迁与 backend-neutral 执行策略
status: code-complete
created: 2026-08-30
slug: lammps-iface-03-graph
chain: lammps-iface
grilled: false
---

# GraphBridge 邻居图构建、reverse-comm 搬迁与 backend-neutral 执行策略

## Summary

把 `PairMolnex::compute()` 剩余的三块迁出为 bridge 组件：邻居表→`edge_index` 的构建（`pair_molnex.cpp:244-266`）、ghost 力的 reverse-comm pack/unpack（`cpp:359-382`）、以及模型执行本体（`cpp:268-317` 的 dynamic / CUDA-graph 两分支）。`LammpsView` 按 `02-view` 钉死的形状扩出邻居 span 与 `neigh_mask` 数据成员——其消费者 `GraphBridge` 于本 spec 到场。`GraphBridge` 明确镜像 `src/molix/batch/neighbor.py` 的**每步重算**语义（Edge Convention 的持有者），而非 `src/molix/md/neighbors.py` 的固定容量 live-range 语义；现有 `std::sort` 与 cutoff 重过滤**原样保留**（排序的处置是 `05-pbc` 的裁量）。reverse-comm 只做形状搬迁，逐字保留"直接 pack 共享的 `atom->f`"这一现行语义（该处正是 `05-pbc` 的靶子，本 spec 不得顺手修）。执行策略抽象为 **backend-neutral 的 `Exec` 接口**（输入 CPU 张量 → 输出 CPU 张量），`DynamicExec` / `GraphedExec` 纯委托 `ModelRunner::run` / `run_graphed`，capture 状态仍独占于 `ModelRunner`（`model_runner.cpp:160-188`），签名中不出现任何 AOTI 假设——closed spec `lammps-pair-molnex-01-aoti-force-export-wall` 记录的 TorchScript 运行时 backward 路线必须能作为同一接口的第三个实现接入。输出向量到 `ModelOutputs` 的转换收敛为**受校验的自由函数 `read_model_outputs`**，virial 的 `numel == 6` 校验由 facade 迁入执行层，使 `virial` 在构造上"良构或未定义"。`02-view` 推迟的 `pos.to(mdt)` 与 CPU/CUDA 设备搬运随 Exec 落地。facade 本体仍是单体（削薄属 `04-facade`）。除下述**经证明冗余**的删除外行为不变，parity 判据沿用 rel 1e-6。

## Design

### LammpsView 扩出邻居面（消费者到场）

`interface/lammps/bridge/lammps_view.h` 按 02 钉死的形状扩充，不改已有字段语义：

```
const int* ilist;              // (inum,)
const int* numneigh;           // 以 local i 索引
const int* const* firstneigh;  // firstneigh[i] -> (numneigh[i],)
int inum;
int neigh_mask;                // facade 从 LAMMPS 的 NEIGHMASK 填入
int neighbor(int i, int jj) const;   // firstneigh[i][jj] & neigh_mask
```

`int**` → `const int* const*` 是合法的限定转换，facade 无需 `const_cast`。掩码值以**数据成员**传入、由访问器施加，因此 bridge 层仍不复述任何 LAMMPS 常量，`NEIGHMASK` 这个 token 不得出现在 `bridge/` 下的任何文件里（`02-view` ac-002 的判据继续生效）。仍不含 `tag`（归未来的 `fixed-graph` 契约）与 `q`（无消费者）。

### GraphBridge —— 每步重算的 cutoff 图（有状态，类型）

`interface/lammps/bridge/graph_bridge.{h,cpp}`。状态为 `cutsq_`（构造自 `cutoff`，`settings()` 时一次性确定）与跨步复用的主机缓冲区 `edge_pairs_` / `edge_flat_`（`pair_molnex.h:98-99` 现已复用，**保留复用**——去掉它会是未经测量的性能回退）。有状态 ⇒ 类型，与 `02-view` 确立的形状规则一致。

`at::Tensor build(const LammpsView& view, int n_graph)` 逐字复刻 `cpp:244-266`：遍历 `ilist`；对每个中心原子 `i` 遍历其掩码后的邻居 `j`；丢弃 `j >= n_graph`；对存活的 `j` 以 `dx²+dy²+dz² < cutsq_` 重过滤；收集 `(i, j)`；`std::sort` 后展平为 `(ne, 2)` int64 CPU 张量。

四点必须在实现与评审中保持清醒：

1. **cutoff 重过滤是必需的，不是冗余**。LAMMPS 的邻居表按 cutoff + skin 构建，表内必然含超出模型 cutoff 的对；丢掉这道过滤会静默改变能量。
2. **`std::sort` 的语义是 `std::pair<int,int>` 的字典序，即按 (source, target) 排序**，正是 `cpp:236-239` 注释所述"molnex 聚合要求 edge_index 按 (source,target) 有序，与参考数据管线 `nonzero()` 的产出一致"。**`03-graph` 无条件保留它**；`lammps-iface-05-pbc` 将其**降级为 `requires_sorted_edges` 契约标志**（默认关闭，需要分段 / CSR 聚合的模型以 `true` 导出），排序代码保留但不再每步无条件执行。本 spec 内不得提前实施该降级。
3. **Edge Convention**：`edge_index[:,0]` = source = `ilist` 给出的中心原子 `i`，`[:,1]` = target = 邻居 `j`，与 `CLAUDE.md` 及 `src/molix/batch/neighbor.py` 一致。C++ 侧不计算 `edge_diff`（模型从 `pos` 自行重算），故本层只需保证列序。
4. **`ilist` 给出的中心原子索引被信任为 local。** `REQ_FULL` 下 LAMMPS 保证 `ilist` 只列出 `[0, nlocal)` 的原子，故 `edge_index[:,0]` 天然满足 source ∈ local。这里**不加运行时检查**——它是邻居表构造方的不变量，重复校验只会在热循环里买一份心理安慰。记录这句话的目的是让下一个读者知道该不变量是被依赖的，而不是被忽略的。（`05-pbc` 引入 ghost 遍历后此句改写：源索引域扩至 local∪ghost，正是对称闭包所要求。）

**入口处必须校验 `n_graph`**（同 `02-view` 的理由）：`build` 断言 `0 <= n_graph && n_graph <= view.nall()`，不满足即库侧 `throw`。`n_graph` 在此既是邻居过滤阈值又是 `view.x` 的索引上界，越界会读到 LAMMPS 拥有的内存之外。

**经证明冗余的删除（其一）**：现行 `cpp:253-254` 有两道过滤 `if (!supports_pbc && j >= nlocal) continue;` 与 `if (j >= n_graph) continue;`。因 `n_graph = supports_pbc ? nall : nlocal`（`cpp:219`），`supports_pbc=false` 时二者逐字等价，`supports_pbc=true` 时前者不激活；故单一 `j >= n_graph` 与原两行的行为**完全相同**。`GraphBridge` 只保留后者，不接受 `supports_pbc` 参数。

**空邻居表的正确性（iron law，本表面新发现）**：`ne == 0` 时现行 `cpp:261-266` 会对可能为 `nullptr` 的 `edge_flat_.data()` 调用 `at::from_blob(..., {0, 2}, ...)`。这是改写这一行时看到的、依赖空 `std::vector::data()` 返回值的未定义边缘。`GraphBridge::build` 必须在 `ne == 0` 时返回一个良构的空 `(0, 2)` int64 张量而不触碰空指针，并由 ctest 锁定。

### reverse-comm —— 只搬形状，不改语义

`interface/lammps/bridge/comm_reverse.{h,cpp}`，两个自由函数（零状态、零配置，形状规则同 `02-view` 的 `stage_nodes`）：

- `int pack_reverse_forces(const LammpsView&, int n, int first, double* buf)`
- `void unpack_reverse_forces(const LammpsView&, int n, const int* list, const double* buf)`

语义逐字取自 `cpp:359-382`：pack 从 `atom->f` 的 `[first, first+n)` 连续读出 `fx,fy,fz` 并返回写入的 double 数；unpack 以 `+=` 累加回 `f[list[i]]`。

**不得引入私有 ghost 力缓冲。** 原始需求把该组件称作 "CommBuf（私有 ghost 力缓冲）"，但现行代码根本没有私有缓冲——它直接 pack 共享的 `atom->f`，这正是 `05-pbc` 要修的缺陷之一。本 spec 是行为不变的搬迁，若在此引入私有缓冲就是把 `05-pbc` 的修复偷渡进 parity spec。因此组件按其**当下真实所为**命名（`comm_reverse` 而非 `CommBuf`），并在头注释中记录该缺陷与其归属 spec，使名字不替尚未发生的修复背书。

### Exec —— backend-neutral 的执行策略

`interface/lammps/bridge/exec.h` 定义抽象基类与输出聚合体：

```
struct ModelOutputs { at::Tensor energy; at::Tensor forces; at::Tensor virial; };  // virial.defined() 即存在性
class Exec {
 public:
  virtual ~Exec();
  virtual ModelOutputs run(const at::Tensor& Z,
                           const at::Tensor& pos_f64,
                           const at::Tensor& edge_index) = 0;
};
```

接口层级刻意定在**语义**而非裸张量向量：入参是 `stage_nodes` / `GraphBridge` 产出的 CPU 张量（`Z` int64、`pos` float64、`edge_index` int64），出参是 CPU float64 的能量/力/可选 virial。设备搬运、模型 dtype 转换、padding 与 mask、输出回搬全部是各实现的私有政策。签名里没有 `.pt2`、没有 `AOTIModelPackageLoader`、没有"权重已在导出期烘焙"的假设，因此 closed spec 记录的 `TorchScriptExec`（C++ 侧持 scripted module 并在运行时 `.backward()`）可以作为第三个实现接入而不动接口——这一点必须写进 `exec.h` 的 docstring。`ModelOutputs` 沿用 `src/molix/md/types.py:21` `ForceOutput` 的 attribute-only 聚合形状（无方法）。

**返回值的所有权契约（接口的一部分，必须写进 `exec.h` 的 docstring）：`run` 返回的张量是被调用方自有的 CPU 拷贝；实现不得别名任何运行时拥有或重放拥有的内存。** 这条不是风格建议：`ModelRunner::run_graphed` 的输出（`model_runner.cpp:178,188`）位于**固定的重放地址**上，下一次 `replay()` 会就地覆写它们。若 `GraphedExec` 把那些张量原样返回，调用方手里的"上一步结果"会在下一步被静默改写——症状是力场看起来"慢了一拍"，且只在 CUDA-graph 路径出现。现行 `cpp:299-304` 靠 `.to(at::kDouble).to(at::kCPU)` 恰好产生了拷贝，但那是巧合而非契约；组件化后必须把它升为接口承诺。

**输出向量 → `ModelOutputs` 的转换是一个受校验的自由函数。** 现行两条分支（`cpp:299-304` 与 `cpp:311-316`）各自重复同一段六行输出转换，且 `v_t.numel() < 6` 的 virial 长度校验孤零零地待在 facade 的 tally 里（`cpp:338-340`）——那是**错误的归属**：facade 拿到一个已 `defined()` 的 virial 却还要提防它读不出六个元素，等于让协议层替执行层擦屁股。

改为：`ModelOutputs read_model_outputs(const std::vector<at::Tensor>& raw, bool supports_virial)`，声明于 `exec.h`、定义于新增的 `exec.cpp`，两个 `Exec` 实现共用。零状态、零配置的纯转换，故按本链的形状规则落为自由函数（命名避开 `make_*` / `create_*`：它描述的是"读取"这个变换，不是某类型构造的别名）。它承担唯一的输出面不变量：

- `raw[0]` → `energy`，`raw[1]` → `forces`（`.to(kDouble).to(kCPU)`，forces 另加 `.contiguous()`），语义与现行逐字相同；
- `virial` **当且仅当** `supports_virial` 为真、`raw` 含对应元素、**且其 `numel() == 6`** 时被 `define`；长度不符即库侧 `throw`，文案给出实测 `numel`。

于是 `ModelOutputs::virial`（进而 `04-facade` 的 `EvalResult::virial`）**在构造上就是"良构或未定义"二者之一**，facade 永远不会收到一个已定义却读不出六个元素的 virial。现行 facade 的 `numel < 6` 检查随之消失——不是被删掉，是被搬到了它本该在的层。（docstring 记录一条形状守卫：本函数目前两个参数；若将来出现第三个契约开关，应改传一个小的 expectation struct，而不是继续堆布尔。）

- **`DynamicExec`**（`dynamic_exec.{h,cpp}`）：状态为非拥有的 `ModelRunner*`、`at::Device`、模型 `at::ScalarType`、`bool supports_virial`（皆在 `settings()` 时确定）。`run` 复刻 `cpp:307-317`：`{Z.to(dev), pos.to(mdt).to(dev), edge_index.to(dev)}` → `runner->run(...)` → `read_model_outputs(...)`。
- **`GraphedExec`**（`graphed_exec.{h,cpp}`）：追加 `n_fixed`、`e_max` 与持久设备缓冲 `Zg_/posg_/edgeg_/maskg_`（`pair_molnex.h:95`）。`run` 复刻 `cpp:273-304`：校验 `Z.size(0) == n_fixed`、`edge_index.size(0) <= e_max`（错误文案逐字保留），padding 后 `copy_` 进持久缓冲，调 `runner->run_graphed(...)` → `read_model_outputs(...)`。**capture 状态不得复制**——`graph_captured_` / `graph_` / `graph_outputs_` 全部留在 `ModelRunner`（`model_runner.cpp:160-188`），`GraphedExec` 只是调用方。**类声明无条件存在**（不置于 `#ifdef` 内），其构造函数在 `MOLNEX_INTERFACE_CUDA` 未定义时抛出"库为 CPU-only 构建"（照 `model_runner.cpp:99-105` 的既有方言与文案形状），`run` 的方法体在 `.cpp` 内以 `#ifdef` 守卫。这样 CPU-only 构建仍能编译、链接并实例化测试所需的全部符号，`#ifdef` 被关进一个 `.cpp`，而不会像现行 `cpp:272-306` 那样把预处理分支摊在 `compute()` 里——ac-011 的"`compute()` 内无 `#ifdef`"因此是可达的。
- **padding 抽为自由函数**：`struct PaddedEdges { at::Tensor edges; at::Tensor mask; };` 与 `PaddedEdges pad_edges(const at::Tensor& edge_index, int64_t e_max)`，声明于 `graphed_exec.h`、定义于 `graphed_exec.cpp`、**不置于 CUDA ifdef 内**。它是零状态的纯 CPU 张量运算，故为自由函数；抽出的理由是 CLAUDE.md 明文豁免的"unit test must target that unit"——`GraphedExec` 的端到端路径需要 CUDA 与定形导出产物，而 padding/mask 的正确性必须在 CPU-only 机器上可测。语义等价于 `cpp:283-291`：前 `ne` 行为真实边、其余为 `(0,0)`，`mask` 前 `ne` 项为真。

**经证明冗余的删除（其二）**：`cpp:274-275` 的 `nghost != 0` 校验被丢弃，因为 `settings()`（`cpp:141-143`）已拒绝 `cuda_graph && supports_pbc`，故 graphed 模式下 `supports_pbc == false`，而 `compute()` 开头 `cpp:202-206` 的守卫已对 `!supports_pbc && nghost != 0` 报错。该守卫留在 facade 且不变，因此删除是行为等价的。

**顺带消除的一次重复构建（padding 本身保留）。** 现行 graphed 分支在 `cpp:266` 先 clone 出 `edge_index`，随后在 `cpp:283-291` 从 `edge_pad_flat_` 把**同一批边**再摆一遍进定长数组，前者在该分支里从未被读取。新结构下 padding **依然存在且语义不变**，只是改由 `pad_edges` 直接消费 `GraphBridge` 已产出的 `(ne, 2)` 张量，因此那份从零重建的中间主机缓冲 `edge_pad_flat_`（`pair_molnex.h:100`）不再需要而删除。消失的是**重复的构建路径**，不是 padding、不是 mask、不是定长语义；数值逐位不变。

### 构建面：bridge 目标新增对 `molnex_interface` 的依赖

Exec 组件委托 `ModelRunner`，故 `molnex_lammps_bridge` 需 `target_link_libraries(... PUBLIC molnex_interface)`——这**修订 `02-view` Design 中"bridge 只依赖 torch"的措辞**，但不触动其验收判据：`molnex_interface` 本身是 LAMMPS-free 的，"未传 `-DLAMMPS_HEADER_DIR` / `-DLAMMPS_LIB` 也能构建并跑 ctest"继续成立；且 `meta_schema.h` / `lammps_contract.h` 仍不得被 `bridge/` 下任何文件包含（bridge 依赖运行时，不认识 meta schema）。链接它同时带入 PUBLIC 的 `MOLNEX_INTERFACE_CUDA` 编译定义。`interface/CMakeLists.txt` 中 `molnex_interface` 的定义先于 `add_subdirectory(lammps/bridge)`，顺序必须保持。父侧仍然只添加 `lammps/bridge` 一个子目录：`interface/CMakeLists.txt` 从不处理 `interface/lammps/CMakeLists.txt`（后者是独立 `project()` 且缺 LAMMPS 变量时 `FATAL_ERROR`，见 `interface/lammps/CMakeLists.txt:26-31`），因此**绝不可**用 `add_subdirectory(lammps)` 从父侧"修复"可见性。

### facade 侧改动（本阶段仍单体）

`compute()` 缩为：既有守卫与 `n_graph` 计算 → `stage_nodes` → `graph_bridge_->build(view, n_graph)` → `exec_->run(...)` → `force_scatter_->scatter(...)` → 既有的能量/virial tally 与 `reverse_comm`。`settings()` 依 `execution` 构造 `DynamicExec` 或 `GraphedExec` 存入 `std::unique_ptr<Exec>`（`use_cuda_graph` 布尔成员随之删除），并构造 `GraphBridge`。`pack_reverse_comm` / `unpack_reverse_comm` 两个 LAMMPS 回调缩为一行转发。`ev_init`、`eng_vdwl` 累加、virial 分支与 `virial_fdotr_compute`、`comm->reverse_comm` 调用点一律不动（归 `04-facade`）。

`pair_molnex.h` 删除 `edge_pairs_` / `edge_flat_` / `edge_pad_flat_` / `Zg_..maskg_` / `use_cuda_graph` / `n_fixed` / `e_max` 成员，**并删除 `fp64_model`**：模型 dtype 的唯一读者是执行路径（`cpp:269` 的 `mdt`），现已随 `pos.to(mdt)` 一并迁入 `Exec`；留在 facade 会出现两个可漂移的副本，与 `02-view` 对 `force_conv` 的处置同理。`cpp:152-159` 的 `logmesg` 仍要打印 dtype，故 `Exec` 暴露一个只读的 dtype 名字符串（`"float32"` / `"float64"`）供其取用——facade 只转述，不再自己判定。

**成员形状与初始化不变量。** `GraphBridge` 是廉价的完整类型，按 `02-view` 确立的先例用 `std::optional<GraphBridge>`（无堆分配，"尚未构造"表达在类型上）；`Exec` 是多态的，只能是 `std::unique_ptr<Exec>`。`02-view` 在 `compute()` 顶部设的那道 `error->all` 守卫在本阶段**扩为覆盖全部四个成员**——`SpeciesMap`、`ForceScatter`、`GraphBridge`、`Exec`；一条守卫确立此后四者皆已具值的不变量，之后直接解引用，`compute()` 内不再出现任何逐成员的空值判断。

### Reuse decision

- generalize 边收集 + 排序（`interface/lammps/pair_molnex.cpp:244-266` 与 `pair_molnex.h:98-99` 的复用缓冲）→ `GraphBridge`，并修掉 `ne == 0` 的空指针边缘。
- generalize `pack_reverse_comm` / `unpack_reverse_comm`（`cpp:359-382`）→ `pack_reverse_forces` / `unpack_reverse_forces`；语义逐字保留，缺陷记录在头注释并指向 `05-pbc`。
- generalize dynamic 与 graphed 两执行分支（`cpp:268-317`，含 `pair_molnex.h:92-95` 的持久缓冲）→ `Exec` / `DynamicExec` / `GraphedExec` / `pad_edges` / `read_model_outputs`（后者同时吸收 facade 侧 `cpp:338-340` 的 virial 长度校验——搬层，非删除）。
- generalize `LammpsView`（`02-view` 落地）→ 按其钉死的形状扩出邻居 span 与 `neigh_mask`。
- reuse `ModelRunner::run` / `run_graphed`（`interface/src/model_runner.cpp:128,149`）—— 纯委托，实现与 capture 状态一概不动、不重复实现。
- reuse `stage_nodes` / `ForceScatter` / `SpeciesMap`（`02-view`）—— 调用点从 `compute()` 内联片段变为与新组件并列，实现不改。
- reuse `interface/tests/generate_fixtures.py` 的 fixture 机制与 `interface/tests/CMakeLists.txt` 的双 ctest 列表（01 引入 no-fixture 列表）—— 扩展而非另起。
- pattern `src/molix/batch/neighbor.py`（每步重算、Edge Convention 持有者）→ `GraphBridge` 声明镜像的语义来源；**不**镜像 `src/molix/md/neighbors.py` 的固定容量 live-range。
- pattern `src/molix/md/types.py:21` `ForceOutput` → `ModelOutputs` / `PaddedEdges` 的 attribute-only 聚合形状。
- pattern `interface/src/model_runner.cpp:99-105` → CPU-only 构建下 CUDA 路径的报错方言与文案形状。
- new `Exec` 抽象 —— 现无任何类型承担"执行策略"职责；`cpp:272-306` 的 `#ifdef` + `if/else` 就是本 spec 要显式化的隐式分支，无可复用骨架。
- deferred `NeighborNeeds` / `EvalResult` / impl 派发 / `CutoffGraphImpl` / facade 削薄 / `n_graph` 归属 / `eflag_atom`-`vflag_atom` 报错 → `lammps-iface-04-facade`。
- deferred 排序处置（降级为契约标志）、私有 ghost 力缓冲、`REQ_GHOST`、双向边、newton 门控 → `lammps-iface-05-pbc`；tag↔local → 未来的 `fixed-graph` 契约 spec。

## Files to create or modify

- `interface/lammps/bridge/graph_bridge.h` (new)
- `interface/lammps/bridge/graph_bridge.cpp` (new)
- `interface/lammps/bridge/comm_reverse.h` (new)
- `interface/lammps/bridge/comm_reverse.cpp` (new)
- `interface/lammps/bridge/exec.h` (new)
- `interface/lammps/bridge/exec.cpp` (new)
- `interface/lammps/bridge/dynamic_exec.h` (new)
- `interface/lammps/bridge/dynamic_exec.cpp` (new)
- `interface/lammps/bridge/graphed_exec.h` (new)
- `interface/lammps/bridge/graphed_exec.cpp` (new)
- `interface/tests/test_bridge_graph.cpp` (new)
- `interface/lammps/bridge/lammps_view.h`
- `interface/lammps/bridge/CMakeLists.txt`
- `interface/tests/generate_fixtures.py`
- `interface/tests/CMakeLists.txt`
- `interface/lammps/pair_molnex.cpp`
- `interface/lammps/pair_molnex.h`
- `.github/workflows/ci.yml`

## Tasks

- [x] Write failing ctest cases for GraphBridge edge collection, cutoff refilter, sort order, n_graph boundary and empty-list handling over a synthetic LammpsView in interface/tests/test_bridge_graph.cpp (new), registered in the no-fixture list of interface/tests/CMakeLists.txt
- [x] Extend LammpsView with ilist, numneigh, firstneigh, inum, neigh_mask and the masked neighbor() accessor in interface/lammps/bridge/lammps_view.h
- [x] Implement GraphBridge in interface/lammps/bridge/graph_bridge.h and graph_bridge.cpp (new), keeping the existing std::sort and cutoff refilter and validating n_graph at entry
- [x] Write failing ctest cases for pack_reverse_forces and unpack_reverse_forces, and implement them in interface/lammps/bridge/comm_reverse.h and comm_reverse.cpp (new)
- [x] Write failing ctest cases for read_model_outputs, pad_edges and DynamicExec against a new three-input graph-potential fixture in interface/tests/test_bridge_graph.cpp and interface/tests/generate_fixtures.py
- [x] Implement the Exec interface, ModelOutputs and the validated read_model_outputs conversion in interface/lammps/bridge/exec.h (new) and exec.cpp (new), with docstrings per google style recording the backend-neutral and owned-CPU-copy contracts
- [x] Implement DynamicExec, GraphedExec and pad_edges in interface/lammps/bridge/dynamic_exec.{h,cpp} and graphed_exec.{h,cpp} (new), and link molnex_interface from interface/lammps/bridge/CMakeLists.txt
- [x] Migrate PairMolnex::settings, PairMolnex::compute and the reverse-comm callbacks onto GraphBridge, Exec and comm_reverse in interface/lammps/pair_molnex.cpp and pair_molnex.h
- [x] Extend the interface ctest job in .github/workflows/ci.yml to cover the new graph, comm, pad, read-outputs and dynamic-exec cases
- [x] Run full check + test suite (ctest 48/48; pytest test_molix 1289 green; ruff clean on own surface — one I001 in the concurrent mace-polar file noted)
- [x] Hygiene: /mol:simplify ran (janitor: 0 apply-able; 2 manual handoffs routed to 04-facade: settings() Exec-selection extraction lands with the dispatch layer, Exec-base member hoist folds into 04's construction work)
- [x] Docs Mode A: satisfied via janitor docstring audit (all exec.h/pad_edges/comm_reverse/model_dtype_name contracts present)

## Testing strategy

- **单元（C++ ctest，无 fixture，合成 `LammpsView`，不依赖 LAMMPS 亦不依赖 CUDA）**——注册进 01 引入的 no-fixture 列表：
  - `bridge_graph_edge_convention`：中心原子入 `[:,0]`、邻居入 `[:,1]`，形状 `(ne,2)`、dtype int64。
  - `bridge_graph_cutoff_refilter`：邻居表内含超出 cutoff 的对（模拟 LAMMPS 的 skin），断言其被丢弃、`ne` 等于手算值。这是本组件最危险的一条——丢掉重过滤会静默改能量。
  - `bridge_graph_sorted_by_source_target`：以乱序 `firstneigh` 输入，断言输出行按 (source, target) 字典序非降。
  - `bridge_graph_drops_beyond_n_graph`：`n_graph < nall` 时 `j >= n_graph` 的邻居被丢弃，锁住与原两道过滤的等价性。
  - `bridge_graph_applies_neigh_mask`：`firstneigh` 元素带高位标志位，断言被 `neigh_mask` 剥除后索引正确。
  - `bridge_graph_empty_neighbor_list`：无任何存活对时返回良构的 `(0,2)` int64 张量、不解引用空指针。
  - `bridge_graph_reuses_buffers_across_calls`：连续两次 `build`（第二次对数不同）结果各自正确，锁住跨步复用缓冲不串味。
  - `bridge_graph_rejects_n_graph_over_nall`：越界 `n_graph` 抛出而非越界读。
  - `bridge_comm_pack_reverse_layout`：pack 从 `[first, first+n)` 写出 `3n` 个 double、顺序为 `fx,fy,fz` 且返回值为 `3n`。
  - `bridge_comm_unpack_accumulates`：以**非零**初始 `f` 起测，断言 `f[list[i]] += buf`（累加而非覆盖）。
  - `bridge_pad_edges`：`e_max > ne` 时前 `ne` 行为真实边、其余为 `(0,0)`，`mask` 前 `ne` 为真其余为假，dtype 分别 int64 / bool；`ne == e_max` 与 `ne == 0` 两个边界同测。
  - `bridge_read_model_outputs_rejects_short_virial`：`supports_virial` 为真但第三个张量 `numel() == 3` 时抛出，文案含实测 numel。
  - `bridge_read_model_outputs_absent_virial_undefined`：`supports_virial` 为真但 `raw` 只有两项时，`virial` 保持 `!defined()` 且不抛出（与现行 `out.size() >= 3` 的宽容语义一致）。
  - `bridge_read_model_outputs_ignores_virial_when_unsupported`：`supports_virial` 为假时即便 `raw` 有第三项也不 `define` virial。
  （`read_model_outputs` 是纯张量变换，测试直接手工构造 `std::vector<at::Tensor>` 调用它，不需要 fixture、CUDA 或 `.pt2`——这正是把它抽成自由函数换来的可测性。）
- **单元（C++ ctest，需 fixture）**——注册进既有 `FIXTURES_REQUIRED` 列表：
  - `interface/tests/generate_fixtures.py` 新增一个三输入的微型图势 fixture：`(Z, pos, edge_index) -> (energy, forces)`，用**不含 autograd 的解析式**（如 `energy = Σ Z·|pos|²`、`forces` 由闭式给出）经**裸 `Exporter`** 导出。理由有二：`export_for_lammps` 的力导出路径在 CPU 上按设计拒绝（`RuntimeError "disabled on CPU"`，由 `tests/test_molix/test_engine.py::test_export_for_lammps_cpu_refused` 锁定；torch 2.12 CPU inductor 误编译 backward scatter——raise 发生在 `Exporter` 的力导出深处（`src/molix/export.py:153`），`export.py:141-142` 本身只做 `auto`→`cpu` 回落）；独立于此，裸 `Exporter` 是既有 fixture 路径（`generate_fixtures.py:61-75`），且 C++ 侧只需 `device` 字段——挂上 `lammps` 块反而把 fixture 耦合进部署 meta schema。现有 `TinyLinear` fixture 是单输入的，无法驱动三输入的 `Exec`。
  - `bridge_dynamic_exec_matches_reference`：`DynamicExec` 对该 fixture 的输出与 Python 侧存下的参考张量在 1e-12（fp64）内一致，且返回的 `energy` / `forces` 均为 CPU float64、`forces` 连续、`virial` 在 `supports_virial=false` 时 `!defined()`。
  - `bridge_graphed_exec_matches_dynamic`：CUDA 可用性处理沿用既有 `cuda_run_matches_reference` 用例的同一约定；`GraphedExec` 与 `DynamicExec` 在同一输入上的能量/力一致（rel 1e-6）。CPU-only 构建下另测构造 `GraphedExec` 抛出"CPU-only 构建"。
- **CI 门禁的显式过滤（本 spec 扩 01 引入的 `interface` job）**：除 no-fixture 列表外，job 另**显式选入** `bridge_dynamic_exec_matches_reference`。它是 `Exec` 抽取的**唯一数值 parity 证据**，而闭式三输入 fixture 在 CPU 上可导出、CI 已安装本项目与 CPU torch wheel，因此这条能且应当进 CI。`fixture_setup` 无需在过滤器里点名——它挂 `FIXTURES_SETUP`，ctest 在选中带 `FIXTURES_REQUIRED` 的用例时会自动拉起。过滤器必须**显式**，使 CUDA 门控的用例（`cuda_run_matches_reference`、`bridge_graphed_exec_matches_dynamic`）留在 CI 之外而非靠运行时跳过。
- **迁移不回归**：`interface/tests` 既有全部用例（`model_runner` 系、01/02 的 `meta_*` / `contract_*` / `bridge_*`）在本次迁移后仍绿。
- **parity**：本 spec 不引入任何数值变化。两处删除均已在 Design 中给出等价性证明，一处删除的是从未被读取的重复 padded-edge 构建。既有 fixture / 基准的能量与力判据沿用 rel 1e-6（非逐位相等）。排序、cutoff 重过滤、`+=` 累加、fp64 中间量与输出回搬顺序一律不变。
- **门禁归属（本 spec 无 `regressions/` 脚本，属记录在案的例外）**：理由与 `02-view` 相同——`regressions/README.md:3-6` 定义该目录为纯 Python 公开面场景脚本且禁止 subprocess，而本 spec 的全部产出是 C++。绑定门禁是 ctest 用例本身加上 `.github/workflows/ci.yml` 的 `interface` job。

## Out of scope

- **两处已命名的活缺陷，均归 `lammps-iface-05-pbc`（本 spec 只搬迁其相邻代码或形状，一律不修）**：
  1. **能量重复计入**：`pair_molnex.cpp:332-333` 在 `supports_pbc=true` 时把模型对 local + ghost 全部位点求得的总能量整体计入 `eng_vdwl`，ghost / 镜像位点被重复计入（`02-view` 已点名）。
  2. **ghost 力的双重通信**：`pair_molnex.cpp:351-352` 在 pair 层调 `comm->reverse_comm(this)`，而 `:359-382` 的 pack 直接读**共享的 `atom->f`**（不是本 pair style 自己的贡献缓冲）。当 `force->newton` 打开时，`Verlet::run` 在积分步内还会再调一次全局 `comm->reverse_comm()`，于是同一份 ghost 力贡献被送回属主**两次**——力偏大，且偏差随 ghost 数目而变。两个环节（pair 层自调 + pack 共享缓冲）任一单独存在都不足以致错，正是二者叠加构成缺陷，因此必须整体交给 `05-pbc` 决断（其正解是删掉 pair 层的自调、由积分器的全局 reverse comm 统一归约）。本 spec 把 pack/unpack 搬成自由函数时**逐字保留**这两点语义，不得顺手改动。
- `ContractImpl` / `NeighborNeeds` / `EvalResult` / 派发 / `CutoffGraphImpl` / facade 削薄、`n_graph` 归属迁入 impl、`ev_init` 与 tally 的重组、`eflag_atom`-`vflag_atom` 被请求而不提供时 `error->all`：`lammps-iface-04-facade`。
- `std::sort` 的降级（`requires_sorted_edges` 契约标志）、`REQ_GHOST`、双向边、per-atom energy 求和语义、newton 门控：`lammps-iface-05-pbc`。**本 spec 只搬形状，不得顺手修其中任何一项。**
- tag↔local 映射：未来的 `fixed-graph` 契约 spec。
- `TorchScriptExec` 本体（仅要求 `Exec` 接口能容纳它，并在 docstring 中记录该路线）。
- 性能优化：pinned memory、异步拷贝、去排序、跨步复用设备缓冲以外的任何新缓存、`GraphBridge` 的 `reserve` 启发式调整。
- bridge 组件下沉到 `interface/include`（等第二个引擎出现再议）。
- `LammpsView` 携带 `q` / 电荷（仍无消费者）。
- espaloma / topology-FF impl、`fixed-graph` 契约实现。
