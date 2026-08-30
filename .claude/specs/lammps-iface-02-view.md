---
title: LammpsView 视图与原子侧 bridge 组件（SpeciesMap / 节点 staging / 力散射）
status: approved
created: 2026-08-30
slug: lammps-iface-02-view
chain: lammps-iface
grilled: false
---

# LammpsView 视图与原子侧 bridge 组件（SpeciesMap / 节点 staging / 力散射）

## Summary

把 `PairMolnex::compute()` 中原子侧的三段循环（`pair_molnex.cpp:226-233` 的 type→Z + 位置 staging、`cpp:327-331` 的力散射）与 `pair_molnex.h:77` 的 `type2z` 表迁出为可独立单测的 bridge 组件，并引入 `LammpsView` 作为唯一的 LAMMPS 类型隔离层。新增 `interface/lammps/bridge/` 与一个**不含任何 LAMMPS 头**的静态目标 `molnex_lammps_bridge`，由 `interface/CMakeLists.txt` 父侧声明，使这些组件能用合成数据在 `interface/tests` 里跑 ctest，不需要 LAMMPS 构建。`compute()` / `coeff()` 立即改用这些组件，facade 本体在本阶段仍是单体（削薄属 `04-facade`）。同时落地本表面上已看见的静默失真（iron law）：`pair_molnex.cpp:163-178` 的 `pair_coeff` 对 Z **完全不校验**，越界或不在 meta `species` 内的元素会一路喂进模型产生垃圾能量而不报错——`SpeciesMap` 在构造时校验并列出合法 species。为此把 01 里按"消费者到场才落地"原则推迟的整数数组读取器一并补进 `MetaSchema` / `LammpsContract`。除该新增报错路径外**行为不变**（parity 容差沿用 rel 1e-6，不是逐位相等）。

## Design

### 前置条件（硬依赖）

本 spec 在 `lammps-iface-01-meta` 落地之前**不可实现**，且依赖的是 01 的三件产出，缺一不可：

1. `interface/include/molnex/interface/meta_schema.h` —— 通用 `MetaSchema`（含 `block(name)` 原语），本 spec 在其上补块内整数数组读取。
2. `interface/lammps/lammps_contract.h` —— `LammpsContract`，`SpeciesMap` 的校验数据源 `species()` 挂在它上面。
3. `.github/workflows/ci.yml` 的 `interface` job —— 本 spec 的全部绑定门禁都是 ctest 用例，没有这个 job 它们只是本地债（`02-view` 不再有 `regressions/` 脚本，见 Testing strategy 的记录在案例外）。

三者任一未持久化，本 spec 的 Tasks 一条都不能开工。

### LammpsView —— 唯一的 LAMMPS 类型隔离层

新增 header-only `interface/lammps/bridge/lammps_view.h`，POD 聚合体（无用户提供的构造函数，可聚合初始化），只借用 LAMMPS 拥有的内存，不拥有任何东西：

```
struct LammpsView {
  const double* const* x;   // atom->x, (nall, 3)
  double* const* f;         // atom->f, (nall, 3), 可写
  const int* type;          // atom->type, (nall,) 1-based
  int nlocal;
  int nghost;
  int nall() const;         // nlocal + nghost
};
```

`double**` → `const double* const*` 是 C++ 合法的限定转换，facade 侧无需 `const_cast`。

**本阶段视图只携带 02 组件真正消费的字段。** 具体地：不含 `tag`（tag↔local 映射不在本链内，见 `05-pbc` 的方案裁决）；不含 `q`（现行 `pair_molnex.cpp` 全文未读 `atom->q`，无消费者）；**不含 `ilist` / `numneigh` / `firstneigh` 邻居 span**——其唯一消费者 `GraphBridge` 在 `03-graph`，届时随消费者扩展该结构体。这与 01 中"整数数组读取等到 `SpeciesMap` 到场才落地"是同一条规则的同一次应用（CLAUDE.md "Inline until the second real use"；本 spec 正是那个消费者到场的时刻）。

**以下为非规范性的前瞻记录，不构成本 spec 的任何要求。** `03-graph` 届时将为该结构体添加邻居 span 与一个 `int neigh_mask` 数据成员（由 facade 从 LAMMPS 的 `NEIGHMASK` 填入，访问器逐元素施加），使掩码常量不必在 bridge 内以字面量复述。**`02-view` 不得添加其中任何一项**——本 spec 的 `LammpsView` 字段集以 ac-001 为准，前瞻记录只是把形状钉给 03 继承，不得被读作现在时的要求。

### SpeciesMap —— 身份/顺序映射（有状态，类型）

`interface/lammps/bridge/species_map.{h,cpp}`。状态是校验过的 type→Z 表，因此是真正的类型而非包装：

- 直接构造 `SpeciesMap(std::vector<int> z_by_type, const std::vector<int>& meta_species)`（无 `make_*` 工厂）。`z_by_type` 是 **0-based、长度恰为 `ntypes` 的稠密表**，**不含**任何占位槽——现行 `type2z` 用索引 0 作哑元（`pair_molnex.cpp:172`）既冗余又让"长度是 `ntypes` 还是 `ntypes+1`"成为读者必须猜的事。`z_of(int lammps_type)` 承担 1-based → 0-based 的换算（`z_by_type_[lammps_type - 1]`），对 `lammps_type <= 0` 或 `> ntypes()` 抛出。`ntypes()` 由表长直接给出，故签名里不需要再带一个 `ntypes` 参数与之互相校验。**构造即校验，不存在半构造状态**：每个 `Z` 必须 > 0；每个 `Z` 必须属于 meta `species` 集合，否则 `throw std::runtime_error`，文案给出出错的 LAMMPS type、其 Z、以及全部合法 species（错误文案照 `src/molix/engine/adapter.py:58-60` 的列全有效值风格）。
- `int z_of(int lammps_type) const` 做边界检查后返回 Z。每步每原子调用一次的 `O(1)` 检查相对神经网络前向是噪声量级；性能优化不在本链范围内，不为此牺牲可诊断性。

`meta_species` 以 `const std::vector<int>&` 传入，由 facade 先从 `LammpsContract::species()` 取出。这不是为了省一次包含，而是一条边界主张：**bridge 层不应知道 meta schema 的存在**。它需要的全部信息是"哪些 Z 是合法的"，那是一个整数集合；把 `LammpsContract` 这样一个承载 schema 版本、unit 表、契约枚举、输入输出清单的清单式类型递进来，等于让一个只做身份映射的组件绑定到导出格式的全部演化——schema 从 v2 走到 v3 时 `SpeciesMap` 会无谓地跟着动。**窄整数契约是比清单类型更小的缝**，这条理由与依赖图的形状无关，因而在依赖图变化后依然成立。

**以下为非规范性的前瞻记录。** `03-graph` 会把 `molnex_interface` 加进 `molnex_lammps_bridge` 的链接行（PUBLIC），因为其 `Exec` 组件委托 `ModelRunner`。那不削弱上述论证：本条主张的是 bridge **不认识 meta schema**，而不是 bridge 不链接任何库；`molnex_interface` 链接进来之后，`meta_schema.h` / `lammps_contract.h` 仍然不得被 `bridge/` 下的任何文件包含。`02-view` 本身不做这次链接改动。

### 节点 staging 与力散射 —— 形状按 shape check 分别裁决

责任三分（`SpeciesMap` / 节点 staging / 力散射）不变，但两者的 C++ 形状不同，理由是 CLAUDE.md 的 shape check 而非对称性：

- **`ForceScatter`（类型）**，`bridge/force_scatter.{h,cpp}`：状态是 `force_conv`——它在 `settings()` 时由 `UnitConversion::force` 一次性确定，之后每步复用，正是"construct → configure → one concern"的形状。方法 `void scatter(const LammpsView& view, const at::Tensor& forces, int n_graph) const`，语义与 `cpp:327-331` 逐字等价：`f[i][k] += force_conv * forces[i][k]`。散射前校验 `forces.size(0) == n_graph`（沿用 `cpp:319-321` 的检查，改为库侧 `throw`，facade 翻译成 `error->all`）。
- **`stage_nodes(...)`（自由函数）**，`bridge/node_staging.{h,cpp}`：`NodeTensors stage_nodes(const LammpsView&, const SpeciesMap&, int n_graph)`，返回 attribute-only 的 `struct NodeTensors { at::Tensor Z; at::Tensor pos; }`（照 `src/molix/md/types.py:21` `ForceOutput` 的聚合形状）。它**没有任何配置与状态**：一个零状态、单方法的类只是函数的包装，正是 CLAUDE.md 禁止的形状；而它也没有天然属主类型（`LammpsView` 是借用视图，给它挂建张量的方法会把 torch 拖进隔离层；`SpeciesMap` 不拥有位置）。故按"Module-level functions only for true free operations"落为自由函数。若将来某个性能 spec 引入跨步复用缓冲区，它才获得状态，届时再升为类型。

**入口处必须校验 `n_graph`。** `stage_nodes` 与 `ForceScatter::scatter` 都以 `n_graph` 为循环上界索引 `view.x` / `view.f`，而这两块内存归 LAMMPS 所有。二者在入口断言 `0 <= n_graph && n_graph <= view.nall()`，不满足即库侧 `throw`（facade 翻译为 `error->all`）。缺这道校验时 `n_graph > nall()` 是一次**越界写入 LAMMPS 拥有的 `atom->f`**——静默的堆破坏，症状会出现在与本组件毫无关系的地方。现行内联代码依赖 `n_graph` 由 `cpp:219` 就地算出这一巧合而无校验；组件化之后 `n_graph` 成为跨边界的入参，校验就不再是可选项。

**不引入缓冲区复用。** 现行代码每步 `at::empty` 新建 `Z` / `pos`（`cpp:224-225`），`stage_nodes` 保持同样的分配形状。把缓冲区复用夹带进一个以 parity 为验收标准的结构迁移 spec，是未经测量的性能改动伪装成重构；性能优化明确不在本链范围。

**dtype 转换不下移。** 现行代码以 fp64 构建 `pos`、在执行点才 `pos.to(mdt)`（`cpp:294`/`cpp:309`）。`stage_nodes` 同样返回 fp64 `pos`，转换留在原处，等 `03-graph` 的 Exec 组件接手。这样本阶段不产生任何数值差异。

### MetaSchema 补整数数组读取器（消费者到场）

`interface/include/molnex/interface/meta_schema.h` 新增块内整数数组读取，并由 `LammpsContract::species() -> std::vector<int>`（`interface/lammps/lammps_contract.h`）暴露。这是 01 里显式推迟的那一项，其唯一消费者（`SpeciesMap` 的校验）在本 spec 到场。数组元素非整数、含负值、或数组为空时抛出（meta 的 `species` 由 `export.py:187` 保证是排序去重的正整数列表）。

### 构建拓扑：LAMMPS-free 的 bridge 静态目标

新增 `interface/lammps/bridge/CMakeLists.txt`，定义 `molnex_lammps_bridge`（STATIC、PIC、`${TORCH_LIBRARIES}`、`TORCH_CXX_FLAGS`），**不引用 `LAMMPS_HEADER_DIR`**，源为 `species_map.cpp` / `node_staging.cpp` / `force_scatter.cpp`（`lammps_view.h` header-only），`target_include_directories` 以 `bridge/` 的父目录为 PUBLIC 使包含路径写作 `bridge/xxx.h`。

该目标由 `interface/CMakeLists.txt` **父侧** `add_subdirectory(lammps/bridge)` 声明。这是必需的方向，但**理由不是目录层级**：`interface/CMakeLists.txt` 根本不处理 `interface/lammps/CMakeLists.txt`——后者是独立的 `project(molnexplugin)`，且在缺 `-DLAMMPS_HEADER_DIR` / `-DLAMMPS_LIB` 时 `FATAL_ERROR`（`interface/lammps/CMakeLists.txt:26-31`）。因此**绝不可**用 `add_subdirectory(lammps)` 从父侧"修复"目标可见性——那会让整个 `interface/` 配置在没有 LAMMPS 的机器上直接失败，恰好摧毁本链赖以存在的 LAMMPS-free ctest 能力。父侧只添加 `lammps/bridge` 这一个自身不引用任何 LAMMPS 变量的子目录。文件位置**不动**（仍在 `interface/lammps/bridge/`，不下沉到 `interface/include`——按 librarian 裁定，等第二个引擎出现再谈下沉）。

`interface/lammps/CMakeLists.txt` 链接 `molnex_lammps_bridge` 并补 `bridge/` 的包含路径。它在 `:38-41` 已有 `if(NOT TARGET molnex_interface) add_subdirectory(../ ...)` 的守卫，父目录被拉入时同时带入 bridge 目标，故从 `interface/lammps` 独立配置与从上层配置两条路径都能拿到该目标，无需第二个守卫。

### facade 侧改动（本阶段仍单体）

- `PairMolnex::coeff()`：解析出 `z_by_type`（0-based，长度 `ntypes`）后构造 `SpeciesMap`，meta species 来自已在 `settings()` 缓存的 `LammpsContract`；若契约尚未加载（`pair_coeff` 先于 `pair_style` 的病态输入）则 `error->all`。构造抛出的异常按既有两方言约定（`cpp:95-99, 129-131`）翻译为 `error->all(FLERR, ...)`。
- `PairMolnex::settings()`：`UnitConversion` 取得后构造 `ForceScatter`。
- `PairMolnex::compute()`：`cpp:221-233` 的两段循环替换为一次 `stage_nodes(...)`；`cpp:326-331` 的散射循环替换为 `force_scatter_->scatter(...)`。`n_graph`（`cpp:219`）仍由 facade 计算并传入。`cpp:202-206` 的 nghost 守卫、能量/virial tally、CUDA-graph 分支一律不动。
- **成员用 `std::optional` 而非 `std::unique_ptr`。** `SpeciesMap` 与 `ForceScatter` 都是廉价的完整类型，堆分配换不来任何东西；`std::optional<SpeciesMap>` / `std::optional<ForceScatter>` 直接把"在 `settings()` / `coeff()` 之前尚未成立"这件事表达在类型上，且不引入所有权语义。`compute()` 顶部以一条 `error->all` 守卫（"pair_style / pair_coeff 未完成初始化"）把二者已具值确立为此后的不变量，之后直接解引用。
- `pair_molnex.h`：`std::vector<int> type2z` 删除。**`force_conv` 成员一并删除**——它唯一的读者是散射循环，现已随之迁入 `ForceScatter`，留在 facade 就会出现两个可漂移的副本。`energy_conv` **保留**：它仍被 `cpp:332`（`eng_vdwl`）与 `cpp:341-346`（virial）读取，且参与 `cpp:152-159` 的 `logmesg`。

### Reuse decision

- generalize `type2z`（`interface/lammps/pair_molnex.h:77` + `cpp:172-174, 229`）→ `SpeciesMap`，并补上原先完全缺失的 Z↔meta species 校验。
- generalize 节点 staging 循环（`interface/lammps/pair_molnex.cpp:226-233`）→ 自由函数 `stage_nodes` + `NodeTensors`。
- generalize 力散射循环（`interface/lammps/pair_molnex.cpp:327-331`）与力形状检查（`cpp:319-321`）→ `ForceScatter`。
- generalize `MetaSchema` / `LammpsContract`（`lammps-iface-01-meta` 落地）→ 补块内整数数组读取与 `species()` 访问器；01 已显式为此留位。
- reuse `interface/tests/CMakeLists.txt` 的 ctest 机制与 `01-meta` 引入的 **no-fixture 用例列表** —— 扩列表并新增一个可执行目标；bridge 用例不挂 `FIXTURES_REQUIRED`（合成数据，不需要导出产物）。`01-meta` 新增的 `.github/workflows/ci.yml` 的 `interface` job 自动覆盖新用例。
- reuse `interface/CMakeLists.txt` 既有的 `find_package(Torch)` / `TORCH_CXX_FLAGS` / PIC 设置 —— bridge 目标沿用同一套编译面，不另起一套 ABI 标志。
- pattern `src/molix/md/types.py:21` `ForceOutput` → `NodeTensors` 的 attribute-only 聚合形状。
- pattern `src/molix/engine/adapter.py:58-60` → `SpeciesMap` 校验失败时列全合法 species 的文案风格。
- pattern `interface/src/model_runner.cpp` 的 owning 类型惯例 —— snake_case 文件名、PascalCase 类型、尾下划线私有成员、库侧 `throw std::runtime_error`。
- new `LammpsView` —— 现无任何类型承担 LAMMPS 指针的隔离职责；`pair_molnex.cpp` 直接解引用 `atom->x` / `atom->f` 是本 spec 要消除的耦合，无可复用骨架。
- deferred 邻居 span + `neigh_mask` 字段、边收集与排序（`cpp:244-266`）、pack/unpack（`cpp:359-382`）、执行策略 → `lammps-iface-03-graph`。
- deferred `NeighborNeeds` / `EvalResult` / impl 派发 / `n_graph` 归属 / `eflag_atom`-`vflag_atom` 报错 → `lammps-iface-04-facade`。
- deferred tag↔local 映射 → 未来的 `fixed-graph` 契约 spec（本 spec 内不得出现）。

## Files to create or modify

- `interface/lammps/bridge/lammps_view.h` (new)
- `interface/lammps/bridge/species_map.h` (new)
- `interface/lammps/bridge/species_map.cpp` (new)
- `interface/lammps/bridge/node_staging.h` (new)
- `interface/lammps/bridge/node_staging.cpp` (new)
- `interface/lammps/bridge/force_scatter.h` (new)
- `interface/lammps/bridge/force_scatter.cpp` (new)
- `interface/lammps/bridge/CMakeLists.txt` (new)
- `interface/tests/test_bridge_atoms.cpp` (new)
- `interface/include/molnex/interface/meta_schema.h`
- `interface/lammps/lammps_contract.h`
- `interface/tests/test_lammps_contract.cpp`
- `interface/tests/CMakeLists.txt`
- `interface/CMakeLists.txt`
- `interface/lammps/CMakeLists.txt`
- `interface/lammps/pair_molnex.cpp`
- `interface/lammps/pair_molnex.h`
- `.github/workflows/ci.yml`

## Tasks

- [ ] Write failing ctest case for the meta species integer-array reader in interface/tests/test_lammps_contract.cpp
- [ ] Implement the block-scoped integer-array reader in interface/include/molnex/interface/meta_schema.h and `LammpsContract::species()` in interface/lammps/lammps_contract.h
- [ ] Write failing ctest cases for SpeciesMap, stage_nodes and ForceScatter over a synthetic LammpsView in interface/tests/test_bridge_atoms.cpp (new) and register them in the no-fixture list of interface/tests/CMakeLists.txt
- [ ] Implement LammpsView in interface/lammps/bridge/lammps_view.h (new) with docstrings per google style stating the borrowed-memory and no-tag/no-charge boundary
- [ ] Implement SpeciesMap in interface/lammps/bridge/species_map.h and species_map.cpp (new) with construction-time Z-versus-meta-species validation over a 0-based dense table
- [ ] Implement stage_nodes, NodeTensors and ForceScatter in interface/lammps/bridge/node_staging.{h,cpp} and force_scatter.{h,cpp} (new), each validating 0 <= n_graph <= view.nall() at entry
- [ ] Declare the LAMMPS-free molnex_lammps_bridge static target in interface/lammps/bridge/CMakeLists.txt (new), add it from interface/CMakeLists.txt, and link it from interface/lammps/CMakeLists.txt
- [ ] Migrate PairMolnex::settings, PairMolnex::coeff and PairMolnex::compute onto SpeciesMap, stage_nodes and ForceScatter in interface/lammps/pair_molnex.cpp and pair_molnex.h, holding both as std::optional and dropping the force_conv member
- [ ] Extend the interface ctest job in .github/workflows/ci.yml to cover the new bridge cases
- [ ] Run full check + test suite

## Testing strategy

- **单元（C++ ctest，`interface/tests/test_bridge_atoms.cpp`，合成 `LammpsView`，不依赖 LAMMPS、不依赖导出 fixture）**。合成视图构造方式：`std::vector<std::array<double,3>>` 承载坐标与力，配一组行指针 `std::vector<double*>` 作为 `x` / `f`，`std::vector<int>` 作为 `type`——这正是 `LammpsView` 存在的目的，使 bridge 可在无 LAMMPS 时被驱动。
  - happy path：`bridge_species_map_maps_types`（type→Z 与合成表逐项相等）；`bridge_stage_nodes_shapes_and_values`（`Z` 为 `int64` `(n_graph,)`、`pos` 为 `float64` `(n_graph,3)` 且逐元素等于视图坐标；`n_graph < nall` 时只取前 `n_graph` 个，锁住 `supports_pbc=false` 的 local-only 语义）；`bridge_force_scatter_accumulates`（`f` 为**累加**而非覆盖——用非零初始 `f` 验证，锁住 `+=` 语义；乘以 `force_conv` 后与手算字面量在 1e-12 内相等）。
  - edge cases：`bridge_species_map_rejects_unknown_z`（Z 不在 meta species 内时构造抛出，文案含该 type、该 Z 与全部合法 species——回归本 spec 修复的静默垃圾路径）；`bridge_species_map_rejects_bad_table_size`；`bridge_species_map_rejects_nonpositive_z`；`bridge_species_map_z_of_rejects_out_of_range_type`；`bridge_force_scatter_rejects_wrong_row_count`（`forces.size(0) != n_graph` 抛出）；`bridge_stage_nodes_rejects_n_graph_over_nall` 与 `bridge_force_scatter_rejects_n_graph_over_nall`（`n_graph` 越界抛出而非越界读写）。
  - `interface/tests/test_lammps_contract.cpp` 增 `contract_species_int_array`（读出完整整数数组而非首元素）与 `contract_species_rejects_non_integer`。
- **迁移不回归**：既有全部用例（`model_runner` 系、01 的 `meta_*` / `contract_*`）在本次迁移后仍绿；bridge 用例与它们并列注册进 no-fixture 列表。
- **构建面判据**：`molnex_lammps_bridge` 必须能在**未传 `-DLAMMPS_HEADER_DIR` / `-DLAMMPS_LIB`** 的 `interface/` 配置下构建并跑通其 ctest；bridge 的任一源/头不得 `#include` 任何 LAMMPS 头，也不得出现 `NEIGHMASK` 字面量。
- **parity**：本阶段唯一有意的行为变化是 `pair_coeff` 新增的 Z 校验报错路径与 `n_graph` 越界报错路径；用合法输入的既有 fixture / 基准时能量与力应与迁移前一致，判据沿用 rel 1e-6（非逐位相等）。`stage_nodes` 保持每步分配、`pos` 保持 fp64 直到执行点，故本阶段不引入任何数值或分配行为差异。
- **门禁归属（本 spec 无 `regressions/` 脚本，属记录在案的例外）**：`regressions/README.md:3-6` 把该目录定义为**纯 Python 公开面场景脚本**，明令禁止 subprocess，现存脚本无一例外。`02-view` 全部产出是 C++，没有任何 Python 公开面可供演练，靠 `regressions/` 脚本 shell out 去驱动 `ctest` 会直接违反该目录的契约。因此本 spec 的绑定门禁是 **ctest 用例本身 + `.github/workflows/ci.yml` 的 `interface` job**（由 `01-meta` 引入，本 spec 扩其用例覆盖），而非回归脚本。这条例外是显式记录的，不是遗漏。

## Out of scope

- **已命名的活缺陷，归 `lammps-iface-05-pbc`（本 spec 只命名，不修）**：`pair_molnex.cpp:332-333` 在 `supports_pbc=true` 时把模型对 **local + ghost** 全部位点求得的总能量整体计入 `eng_vdwl`，ghost / 镜像位点的位点能量被重复计入，多 rank 或周期体系的总能量因此偏高。该路径经 `export_for_lammps(..., supports_pbc=True)` 可达（`export.py:87`），是活缺陷而非假设。本 spec 迁移的正是其相邻代码（力散射与 staging），故按 iron law 在此**点名并留痕**；修复语义（per-atom energy 求和 / 只对 local 求和）与 `REQ_GHOST`、双向边、newton 门控一并属 `05-pbc`。
- 邻居 span 与 `neigh_mask` 字段、边收集与排序（`cpp:244-266`）、comm pack/unpack（`cpp:359-382`）、`DynamicExec` / `GraphedExec` 执行策略与 `pos.to(mdt)` 的归属：`lammps-iface-03-graph`。
- `ContractImpl` / `NeighborNeeds` / `EvalResult` / 派发 / `CutoffGraphImpl` / facade 削薄、`n_graph` 归属迁入 impl、`eflag_atom`-`vflag_atom` 被请求而不提供时 `error->all`：`lammps-iface-04-facade`。
- tag↔local 映射：未来的 `fixed-graph` 契约 spec。**本 spec 内不得出现任何 tag 相关字段或映射。**
- 性能优化：跨步复用 staging 缓冲区、pinned memory、异步拷贝、`z_of` 的边界检查消除。
- bridge 组件下沉到 `interface/include`（等第二个引擎出现再议）。
- `LammpsView` 携带 `q` / 电荷（现无消费者）。
- TorchScript 执行器、espaloma / topology-FF impl、`fixed-graph` 契约实现。
