---
title: meta.json schema v2 契约与唯一读取器 MetaSchema
status: code-complete
created: 2026-08-30
slug: lammps-iface-01-meta
chain: lammps-iface
grilled: false
---

# meta.json schema v2 契约与唯一读取器 MetaSchema

## Summary

把 `<name>.meta.json` 升级为受强校验的 v2 契约，并让 C++ 侧只剩一个 JSON 读取器。写入端 `molix.engine.export_for_lammps` 新增 `contract`、`execution`、`energy_conv` 换算表与 `length_unit`，删除 `cuda_graph` 布尔判别子，并把 `adapter` 降级为诚实的 provenance；读取端新增 LibTorch-free 且 LAMMPS-free 的通用 `interface/include/molnex/interface/meta_schema.h` 与承载 LAMMPS 词汇的 `interface/lammps/lammps_contract.h`，吸收 `model_runner.cpp:33` 的 `parse_device_field`，删除 `interface/lammps/lammps_meta.h`，用作用域块解析取代"键名全文唯一"这一无法强制的不变量，并在契约块构造时硬校验 `schema_version`。C++ 侧的换算浮点复述（`pair_molnex.cpp:41` 的 `0.0433641043`、`README.md:92-93` 的 ×0.0433641/×23.0605 两格）全部删除，改为 export 侧用 `molix.units`（pint）算好盖进 meta、C++ 按 `update->unit_style` 查表。本阶段**行为不变**（parity 容差见 Testing strategy），三层拆分本体属链上 `02-view`/`03-graph`/`04-facade`。顺带修复本表面上已看见的两处静默失真（iron law，不得留待后续）：`export.py:190` 在 `supports_virial=True` 时仍写 `outputs=["energy","forces"]`（meta 说谎）；`export.py:147-167` 在 `cuda_graph=True` 时静默忽略调用方传入的 `adapter` 参数并把不存在于 registry 的 `"molnex-tensordict-static"` 写进 meta。

**范围声明的两处例外（显式记录）**：其一，`.github/workflows/ci.yml` 新增 `interface` job 属于测试基建而非行为变更，是本链全部 C++ 判据能成为门禁而非本地债的前提，为**必需（required）**门而非 advisory；其二，`tests/test_molix/test_engine.py` 拆分为镜像包属于对本 spec 正在修改的文件上的硬规修复（iron law 的 fix-now 分支）。

## Domain basis

本阶段唯一的物理内容是能量/长度单位换算，且必须**由 `molpy.UnitSystem`（pint）求值，molnex 内不得复述任何换算浮点**（CLAUDE.md "Units"；`.claude/notes/learnable-classical-ff.md` §5）。

LAMMPS `real` 的能量单位是 kcal/mol，`metal` 是 eV。以 2019 SI 重新定义后的精确定义常量表达：

- 热化学卡路里定义值 \(1\,\mathrm{cal_{th}} \equiv 4.184\ \mathrm{J}\)（精确，NIST SP 811 §B.8）
- 阿伏伽德罗常数 \(N_A = 6.02214076\times10^{23}\ \mathrm{mol^{-1}}\)（精确定义值）
- 元电荷 \(e = 1.602176634\times10^{-19}\ \mathrm{C}\)（精确定义值），\(1\,\mathrm{eV} \equiv e\cdot 1\,\mathrm{V}\)

因此

\[
1\ \frac{\mathrm{kcal}}{\mathrm{mol}}
= \frac{4184\ \mathrm{J\,mol^{-1}}}{N_A\,e}\ \mathrm{eV}
\approx 4.3364\times10^{-2}\ \mathrm{eV},
\qquad
1\ \mathrm{eV} \approx 23.0605\ \frac{\mathrm{kcal}}{\mathrm{mol}}
\]

三个常数自 2019 年起皆为精确定义值，故该比值不携带实验不确定度；数值参考 CODATA 2018 评定（Tiesinga, Mohr, Newell, Taylor, *Rev. Mod. Phys.* **93**, 025010, 2019, DOI [10.1103/RevModPhys.93.025010](https://doi.org/10.1103/RevModPhys.93.025010)）与 NIST CUU（https://physics.nist.gov/cuu/Constants/）。

**力的换算不是能量换算的同义词。** \([F]=[E]/[L]\)：现行 `pair_molnex.cpp:118` 的 `force_conv = energy_conv` 只有在 \(L_{\text{model}} = L_{\text{LAMMPS}}\) 时成立。`real` 与 `metal` 的长度单位都是 Å，所以该等式当下为真，但**代码里没有任何东西检查它**——一个以 Bohr 导出的模型会被静默错缩 \(a_0 = 0.529177\ \text{Å}\) 倍且不报错。故 v2 必须盖进 `length_unit`（当前恒为 `"angstrom"`）并在 C++ 侧断言；断言通过才允许 `force = energy`。非 Å 导出需要 `schema_version` 3 与一张长度换算表，本阶段不预建（"inline until the second real use"）。

## Design

### 实体与所有权（C++ 侧）

读取端按**词汇归属**一分为二。`molnex_interface` 是引擎中立的库，其公开头里不得出现 LAMMPS 词汇（`lammps` 块名、unit style、pair style 契约），否则下一个引擎接入时会被迫链接一堆与它无关的概念。

**中立层** —— `interface/include/molnex/interface/meta_schema.h`，**纯 std，不含任何 `torch/` / `ATen/` / LAMMPS 头**：

- `class MetaSchema` —— 整份 meta 文档。`explicit MetaSchema(std::string json_text)` 直接构造；`static MetaSchema from_file(const std::string& path)` 是语义不同的替代构造器，符合 CLAUDE.md 对 `Foo.from_file` 的许可、不是 `make_*` 别名。公开的是**通用原语**：`device()`（吸收 `parse_device_field`，保留其现有错误文案形状）、`has_block(name)`、`block(name)` —— 后者返回一个作用域子文档（同为 `MetaSchema`），以及块内的字符串 / 字符串数组读取。**没有 `has_lammps()` / `lammps()`**：`"lammps"` 只是调用方传入的一个块名字符串，中立层不认识它。docstring 必须写明：`device()` 是**顶层专属字段**，在 `block(name)` 返回的子文档上调用会抛"missing 'device'"（作用域解析保证它不会静默解析到顶层值）——记录该边界即可，不为此另造 `MetaBlock` 类型去复制整套读取面。
- **构造不要求 `schema_version`**：`interface/tests/generate_fixtures.py:75` 用裸 `Exporter` 写出的 meta 只有 `device` / `model_class`，没有任何块，而 `ModelRunner` 必须继续加载它们。
- **作用域块解析取代全文搜索。** `lammps_meta.h:6-8` 承诺的"键名在整份文件里唯一"既无法强制也已被嵌套块打破，v2 直接**删除该不变量而非重新承诺**：扫描器按花括号深度前进并跳过字符串字面量，`block(name)` 在当前文档的顶层深度定位该键的 `{...}` 值域，块内查键同样限制在该块的顶层深度，故子对象里的同名键不会串味。该作用域性由 `meta_scoped_block_shadowing` ctest 锁死——它现在是共享库里的承重不变量，不再是插件本地头的口头约定。同时新增字符串数组读取（`inputs` / `outputs` 现为 JSON 数组，`get_string` 只回首元素是既有缺陷）。整数数组读取**不在本阶段加入**——其消费者（pair_coeff 的 Z 校验）在 `02-view`，届时随消费者落地。

**LAMMPS 词汇层** —— 新增 `interface/lammps/lammps_contract.h`，建立在 `MetaSchema::block("lammps")` 之上。它**同样不含任何 LAMMPS 头**（不 `#include "pair.h"` / `"update.h"`），只是承载 LAMMPS 侧的概念词汇，因此可以在没有 LAMMPS 构建的机器上被 ctest 驱动：

- `class LammpsContract` —— 由 `LammpsContract(const MetaSchema& doc)` 构造，内部取 `doc.block("lammps")`，**构造时硬校验 `schema_version == 2`**，不匹配即 `throw std::runtime_error`（文案含期望值与实际值）。它持有块文本的**自有拷贝**（meta.json 很小），生命周期不与调用方的 `MetaSchema` 耦合，不会悬垂。这是"schema_version 在构造时校验"这一裁决的忠实落点：校验发生在拥有该版本号的类型的构造上。
- `struct UnitConversion { double energy; double force; };` —— attribute-only 值类型，无方法（照 `src/molix/md/types.py:21` `ForceOutput` 的聚合形状）。由 `LammpsContract::units_for(const std::string& lammps_unit_style)` 返回：查 `energy_conv` 表得 `energy`；断言 `length_unit == "angstrom"`，通过后 `force = energy`。查不到该 unit style 时抛出并列出表中全部有效键（错误文案照 `src/molix/engine/adapter.py:58-60` 的 "valid …" 风格）。**不新增独立的 `Units` 组件**：按 CLAUDE.md shape check，这是"读一个以 unit style 为键的 meta 字段"，天然属主是契约类型，只有一个步骤、一个树内调用点。
- `enum class Contract { cutoff_graph, fixed_graph };` 与 `enum class Execution { dynamic, cuda_graph };` —— 闭集在类型层表达。字符串→枚举用 function-local `static const std::map` + `find` + 失败时列出全部有效名（照 `interface/tests/test_model_runner.cpp:244` 的显式表模式；无静态自注册）。**该模式的绑定消费者是 `lammps-iface-04-facade` 的 impl 派发**——本 spec 只是它的第一次使用。若整条链结束时 `fixed_graph` 仍无对应 impl 落地，该枚举值必须在 schema v3 下**删除**，而不是作为无人实现的占位符长期存在（此为链级待办，见 Out of scope）。头 docstring 必须写明 `fixed-graph` 未来承载的是 `bond_index` **`(2, N)` COO 共价拓扑**，而非 `edge_index` `(E, 2)` 邻居图——两者形状被 `CLAUDE.md` 刻意设计为不同，正是为了让一个永远不能被静默当作另一个（type-level anti-alias guard）。

`interface/lammps/CMakeLists.txt:59` 的 `${CMAKE_CURRENT_SOURCE_DIR}` 包含路径**保留**（注释从 `lammps_meta.h` 改指 `lammps_contract.h`）。

**测试放置。** 两个头各有一个 ctest 可执行目标，均注册进本 spec 新引入的 **no-fixture 列表**（不挂 `FIXTURES_REQUIRED`）：`test_meta_schema`（链接 `molnex_interface`）与 `test_lammps_contract`（同样链接 `molnex_interface`，另加一条指向 `interface/lammps/` 的**包含路径**以拿到 `lammps_contract.h`）。后者是纯包含路径，**不是** `add_subdirectory(lammps)`——见下方构建面说明。**已具名的债**：LAMMPS 侧头的测试因此住在中立库的测试目录里、经裸包含路径下探——这是构建系统限制（`interface/lammps/CMakeLists.txt` 是独立 `project()` 且缺 LAMMPS 变量即 `FATAL_ERROR`）造成的取位，不是设计意图。替代方案（把 `FATAL_ERROR` 守卫收窄到只护 `molnexplugin` 目标，使 `lammps/` 可 LAMMPS-free 配置、测试随其属主住进 `interface/lammps/tests/`）记录在此，**复检点为 `04-facade`**（届时 `impls/` 子目录落地，取位问题会再次出现）。

### meta.json v2 字段（写入端 `export_for_lammps`）

`LAMMPS_META_SCHEMA` 由 1 升到 2。`lammps` 块字段集精确为：`schema_version`、`contract`、`execution`、`cutoff`、`units`、`energy_unit`、`length_unit`、`energy_conv`、`species`、`model_dtype`、`inputs`、`outputs`、`adapter`、`supports_pbc`、`supports_virial`；当且仅当 `execution == "cuda-graph"` 时追加 `n_atoms`、`e_max`。

- `contract` ∈ {`cutoff-graph`, `fixed-graph`}：语义是**边集从哪来**（每步按 cutoff 重算 vs 由固定拓扑给定），不是网络名，故不违反 "no per-model C++"。当前两条导出路径都产出 `cutoff-graph`；`fixed-graph` 为拓扑型 FF 保留，本链不产出，C++ 侧遇到它必须显式报"尚无实现"而非回落。它由导出路径推导，**不作为 `export_for_lammps` 的入参**。`fixed-graph` 的绑定消费者是 `04-facade` 的派发；链结束仍无 impl 则于 schema v3 删除该值。
- `execution` ∈ {`dynamic`, `cuda-graph`}：**是 `cuda_graph` 布尔的重命名，不是新增**——`cuda_graph` 键从 meta 中删除。判别子净数不增：原先是两个重叠判别子（registry 中不存在的 `adapter_name` 变体 + `cuda_graph` 布尔），现在是两个正交判别子（`contract` 选 impl，`execution` 选 impl 内的执行策略）。Python 函数参数名 `cuda_graph=` 保持不变（用户可见 API 不动）。执行枚举是**策略选择器而非后端选择器**：closed spec `lammps-pair-molnex-01-aoti-force-export-wall` 已证 AOTI 无法导出 cuet 系的力，记录在案的路线是 TorchScript+libtorch 运行时 backward，因此未来会追加 `torchscript` 值；`MetaSchema` / `LammpsContract` 中任何字段语义都**不得写死 AOTI 假设**（`.pt2`/AOTI 字样只能出现在 provenance 注释里），扩展该枚举需要 `schema_version` 升到 3。此点必须写进头文件 docstring。
- `energy_conv`：`{style: 1 个模型能量单位在该 style 下的数值}`，键覆盖 `_UNIT_LABELS` 全集（`real`/`metal`），值由 `molix.units.preset_energy_ratio` 求得，`energy_conv[units] == 1.0`。
- `length_unit`：当前恒为 `"angstrom"`。
- `outputs`：`supports_virial=True` 时必须为 `["energy","forces","virial"]`（修复 `export.py:190` 的硬编码谎报）。
- `adapter`：**恒为 `EngineAdapter` registry 中真实存在的名字**。`cuda_graph` 分支不再写 `"molnex-tensordict-static"`；`StaticForward` 内联的正是 molnex-tensordict 约定，故记为 `"molnex-tensordict"`，且该分支传入非默认 `adapter` 时**抛 `ValueError` 而不是静默忽略**（`export_for_lammps` docstring 的 `Raises:` 块补该条目）。

`export.py` 内抽出私有 `_lammps_meta_block(...) -> dict` 承载该字段集。它是纯字典装配（无天然属主类型的自由操作，且为私有符号），抽出的理由有二：CLAUDE.md 明文豁免的"unit test must target that unit"——整段导出的力路径按 `test_export_for_lammps_cpu_refused` 在 CPU 上被设计拒绝（raise 在 `src/molix/export.py:153` 的 `Exporter` 力导出深处），只能 CUDA + `slow` 跑，而 meta 契约必须在 CPU CI 上被单测覆盖；其二，它同时是 fixture 生成器的调用点（见下）。

`molix.units` 新增 `preset_energy_ratio(source_style, target_style) -> float`，与既有 `preset_energy_to_md` 并列并共享 `UnitSystem.preset(style).base_units["energy"]` 的解析；二者目标单位空间不同（后者目标是 MD 复合单位而非某个 LAMMPS style），故不合并。

### 消费端改动（行为不变）

`model_runner.cpp` 删除 `parse_device_field` 与其私有 `read_file`（改用 `MetaSchema::from_file`），库内只剩一个 JSON 扫描器。`pair_molnex.cpp` 删除匿名命名空间的 `energy_unit_in_eV`，`settings()` 改为：`MetaSchema::from_file` → `LammpsContract` 构造（此处即完成 schema_version 校验）→ `units_for(update->unit_style)` 填 `energy_conv`/`force_conv` → `contract()` 校验（非 `cutoff_graph` 立即 `error->all`）→ `execution()` 映射到既有 `use_cuda_graph` 布尔。库侧 `throw` 与 LAMMPS 侧 `error->all(FLERR, ...)` 的两方言分工不变，翻译点沿用 `pair_molnex.cpp:95-99, 129-131` 的 try/catch 形状。`interface/lammps/lammps_meta.h` 删除；`interface/CMakeLists.txt` 把 header-only 的 `meta_schema.h` 列入 `molnex_interface` 目标源（仅供 IDE / 安装可见，不产生 TU）。

**构建面（措辞校正，防误修）。** `interface/tests/` 看不到 `interface/lammps/` 里定义的任何 CMake 目标，原因不是目录层级，而是 **`interface/CMakeLists.txt` 从不处理 `interface/lammps/CMakeLists.txt`**：后者是一个独立的 `project(molnexplugin)`，且在缺 `-DLAMMPS_HEADER_DIR` / `-DLAMMPS_LIB` 时 `FATAL_ERROR`（`interface/lammps/CMakeLists.txt:26-31`）。因此**绝不可**用 `add_subdirectory(lammps)` 从父侧"修复"可见性——那会让整个 `interface/` 配置在没有 LAMMPS 的机器上直接失败，正好摧毁本链所依赖的 LAMMPS-free ctest 能力。`test_lammps_contract` 拿到 `lammps_contract.h` 的方式只能是一条 `target_include_directories` 包含路径（header-only，无 TU、无目标）。

**CI 门（必需，非 advisory）。** `.github/workflows/ci.yml` 新增 `interface` job：复用既有两个 job 的 CPU torch wheel + `cmake>=4` + `ninja` 配置（`ci.yml:21-25,45-49` 的配方直接可用），配置 `interface/`（**不传** `LAMMPS_HEADER_DIR` / `LAMMPS_LIB`），运行 **过滤式** ctest——用 `-R`（或 label）显式选入 no-fixture 列表，**绝不可跑裸 `ctest`**：`interface/CMakeLists.txt:46-48` 默认 `MOLNEX_INTERFACE_BUILD_TESTS=ON`，裸跑会拉起挂 `FIXTURES_SETUP` 的 `fixture_setup`，即整套 Python AOTI 导出。该 job 是必需门：没有它，本链全部 C++ 判据只是本地债。

**一次性破坏性变更（记录在案）。** v1 插件**不得**加载 v2 导出：旧 `pair_molnex.cpp:124` 的 `get_bool(meta, "cuda_graph", false)` 在键消失后会静默回落为 `false`，把一个 CUDA-graph 导出当成 dynamic 路径跑。v2 侧由 `LammpsContract` 的 `schema_version` 硬校验拒绝旧 meta；反向（旧插件读新 meta）无法在旧二进制里拦截，因此必须写进 `interface/lammps/README.md`：插件与导出产物需成对升级，跨版本混用不受支持。

### Reuse decision

- generalize `parse_device_field`（`interface/src/model_runner.cpp:33`）→ 吸收为 `MetaSchema::device()`；原函数删除，库内不留第二个 JSON 扫描器。
- generalize `molnex::lammps_meta`（`interface/lammps/lammps_meta.h:21`）→ 拆升为 `MetaSchema`（中立层）与 `LammpsContract` / `UnitConversion` / `Contract` / `Execution`（LAMMPS 词汇层），补字符串数组读取与作用域块解析；原头文件删除，不留别名。
- generalize `LAMMPS_META_SCHEMA`（`src/molix/engine/export.py:39`）→ 由"写入但从不读"升为 v2 且在 `LammpsContract` 构造时硬校验。
- generalize `energy_unit_in_eV`（`interface/lammps/pair_molnex.cpp:39`）→ 删除；换算改由 export 侧 `molix.units` 求值盖表、C++ 查表，同时删掉 `README.md:92-93` 的第三处复述。
- generalize `preset_energy_to_md`（`src/molix/units.py:23`）→ 新增同源同文件的兄弟函数 `preset_energy_ratio`，共享 `UnitSystem.preset(...).base_units["energy"]` 解析；不合并、不改既有签名。
- generalize `adapter` provenance（`src/molix/engine/export.py:153`）→ 字段保留但值语义纠正为 registry 内的真实名字，并对被静默忽略的入参改为报错。
- reuse `ModelRunner::run` / `run_graphed`（`interface/src/model_runner.cpp:128,149`）—— 本阶段不触碰，capture 状态仍归 `ModelRunner` 独有。
- reuse `StaticForward`（`src/molix/engine/static.py:28`）—— 原样映射为 `execution == "cuda-graph"`，实现不改。
- reuse `interface/tests/CMakeLists.txt:27` 的 `_TESTS` 列表 + ctest 机制 —— 新增 no-fixture 列表与两个可执行目标；既有 `FIXTURES_REQUIRED` 列表保持不变。
- pattern `interface/tests/test_model_runner.cpp:244` 的 function-local `static const std::map` + `find` + 列全有效键 → 用于 `Contract` / `Execution` 的字符串→枚举解析；错误文案照 `src/molix/engine/adapter.py:58-60`。
- pattern `src/molix/md/types.py:21` `ForceOutput` → `UnitConversion` 采用同形 attribute-only 值类型。
- new `MetaSchema` / `LammpsContract` / `UnitConversion` / `Contract` / `Execution` —— 现有 `lammps_meta` 是自由函数集合，无可复用的类型骨架；上述为其类型化归宿。
- deferred `type2z` + staging/scatter（`pair_molnex.h:77`、`cpp:226-233,326-331`）→ `lammps-iface-02-view`；deferred 边收集+sort（`cpp:244-266`）与 pack/unpack（`cpp:359-382`）→ `lammps-iface-03-graph`；deferred impl 派发 / `NeighborNeeds` / `EvalResult` → `lammps-iface-04-facade`。

## Files to create or modify

- `interface/include/molnex/interface/meta_schema.h` (new)
- `interface/lammps/lammps_contract.h` (new)
- `interface/tests/test_meta_schema.cpp` (new)
- `interface/tests/test_lammps_contract.cpp` (new)
- `tests/test_molix/test_engine/__init__.py` (new)
- `tests/test_molix/test_engine/test_adapter.py` (new)
- `tests/test_molix/test_engine/test_export.py` (new)
- `tests/test_molix/test_engine/test_static.py` (new)
- `regressions/lammps-iface-01-meta.py` (new)
- `interface/src/model_runner.cpp`
- `interface/lammps/lammps_meta.h` （删除）
- `interface/lammps/pair_molnex.cpp`
- `interface/lammps/pair_molnex.h`
- `interface/lammps/CMakeLists.txt`
- `interface/lammps/README.md`
- `interface/CMakeLists.txt`
- `interface/tests/CMakeLists.txt`
- `interface/tests/generate_fixtures.py`
- `src/molix/units.py`
- `src/molix/engine/export.py`
- `tests/test_molix/test_units.py`
- `tests/test_molix/test_engine.py` （删除，拆入同名目录）
- `.github/workflows/ci.yml`

## Tasks

- [x] Write failing unit tests for `preset_energy_ratio` (tests/test_molix/test_units.py → TestPresetEnergyRatio)
- [x] Implement `preset_energy_ratio` in src/molix/units.py
- [x] Split tests/test_molix/test_engine.py into the tests/test_molix/test_engine/ package (__init__.py, test_adapter.py, test_export.py, test_static.py), preserving every existing case
- [x] Write failing unit tests for the schema-v2 meta block (tests/test_molix/test_engine/test_export.py → TestLammpsMetaBlock, TestExportForLammpsValidation)
- [x] Implement `_lammps_meta_block` and schema v2 in src/molix/engine/export.py, including the new Raises: entry for the adapter-with-cuda_graph ValueError
- [x] Write failing ctest cases for generic MetaSchema in interface/tests/test_meta_schema.cpp (new) and for LammpsContract, UnitConversion, Contract and Execution in interface/tests/test_lammps_contract.cpp (new), registering both in a new no-fixture list in interface/tests/CMakeLists.txt
- [x] Emit a real schema-v2 meta fixture from interface/tests/generate_fixtures.py by calling `_lammps_meta_block`
- [x] Implement generic MetaSchema in interface/include/molnex/interface/meta_schema.h and LammpsContract, UnitConversion, Contract and Execution in interface/lammps/lammps_contract.h (new), listing the neutral header in interface/CMakeLists.txt, with docstrings per google style
- [x] Migrate ModelRunner and PairMolnex onto the new readers: delete `parse_device_field` from interface/src/model_runner.cpp, retire interface/lammps/lammps_meta.h, retarget the include-path comment at interface/lammps/CMakeLists.txt:59, and rewrite the units and version-break sections of interface/lammps/README.md
- [x] Add a required `interface` ctest job to .github/workflows/ci.yml (CPU torch wheel, cmake, no LAMMPS) running the no-fixture ctest list via an explicit filter, never bare ctest
- [x] Add regression example regressions/lammps-iface-01-meta.py (public API only; hard-coded goldens, no third-party runtime, no subprocess)
- [x] Run full check + test suite  (CI-parity check green: ruff src/tests clean, ty 0 errors; interface ctest 19/19; target pytest sets green; full tree = 3745 passed / 15 skipped plus 22 pre-existing failures confined to tests/test_scripts/test_mm_param_learning — molpy-0.14 drift of the mm-param-val/molhub workstream, enumerated and routed, predating this chain)
- [x] Hygiene: /mol:simplify ran clean (2 dead includes removed; 3 manual handoffs recorded)

## Testing strategy

- **单元（Python，`tests/` 镜像，单函数/单方法，无 e2e）**
  - `tests/test_molix/test_units.py` → `TestPresetEnergyRatio`（目标 `molix.units.preset_energy_ratio`）：同 style 自比恒等于 `1.0`；`real→metal` 对全精度手算值 `0.043364104241800934` 在 rel 1e-6 内；`metal→real` 对 `23.06054783061903` 在 rel 1e-6 内（短写 `23.0605` 在 rel 1e-6 下不可满足——真实偏差 2.07e-6，测试用全精度 golden）；往返乘积与 1.0 差 < 1e-12；未知 style 抛错。硬编码期望值来自 Domain basis 的精确定义常量，不从被测代码反推。
  - `tests/test_molix/test_engine/test_export.py` → `TestLammpsMetaBlock`（目标 `molix.engine.export._lammps_meta_block`，CPU 可跑）：字段集与本 spec 列表逐字相等；`schema_version == 2`；`"cuda_graph"` 键不存在；`contract == "cutoff-graph"`；`execution` 随入参取 `dynamic`/`cuda-graph`；`energy_conv` 键集为 `{"real","metal"}` 且 `energy_conv[units] == 1.0`；`length_unit == "angstrom"`；`outputs` 含 `"virial"` 当且仅当 `supports_virial`；`n_atoms`/`e_max` 存在当且仅当 `execution == "cuda-graph"`；`adapter ∈ EngineAdapter.names()`。
  - `tests/test_molix/test_engine/test_export.py` → `TestExportForLammpsValidation`：`cuda_graph=True` 且 `adapter` 非默认时抛 `ValueError`（回归"静默忽略"）。既有 `test_export_for_lammps_roundtrip`（`slow` + CUDA）的断言同步到 v2 字段集。
- **单元（C++ ctest，两个可执行目标，JSON 用测内字面量或 fixture 产出的真 meta，不依赖 LAMMPS、不挂 `FIXTURES_REQUIRED`）**——本 spec 引入 `interface/tests/CMakeLists.txt` 的 **no-fixture 用例列表**（既有 `FIXTURES_REQUIRED` 列表保持不变），`02-view` / `03-graph` 后续扩展它。
  - `test_meta_schema`（中立层）：`meta_device_from_bare_export`（只有 `device`/`model_class` 的裸 `Exporter` meta 仍能取到 device，且 `has_block("lammps")` 为假——直接锁住 `generate_fixtures.py:75` 的既有产物）；`meta_scoped_block_shadowing`（顶层与块内同名键，取到块内值——锁死已删除的全文唯一不变量）；`meta_string_array`（数组读出全部元素而非首元素）；`meta_block_missing_returns_absent`。
  - `test_lammps_contract`（LAMMPS 词汇层）：`contract_schema_version_mismatch_throws`、`contract_unknown_contract_throws`、`contract_unknown_execution_throws`（文案含全部有效名）、`contract_unsupported_unit_style_throws`（文案列出表内全部键）、`contract_length_unit_not_angstrom_throws`、`contract_input_arity_mismatch_throws`（`execution=cuda-graph` 但 `inputs` 只有 3 项）、以及 `contract_reads_generated_v2_fixture`——最后一条消费 `generate_fixtures.py` 用 `_lammps_meta_block` 真正写出的 meta，**在测试时把 Python 侧的 2 与 C++ 侧的 2 绑在一起**；两侧任一漂移即红。
  - 迁移不回归：既有 `_TESTS` 全部 case（含 `load_bad_path_throws`）在 `ModelRunner` 换用 `MetaSchema` 后仍绿。
- **域校验（换算 parity，本 spec 唯一的数值行为变动）**：pint 求得的 `energy_conv` 与被删除的硬编码 `0.0433641043` 相对偏差 ≤ 1e-7。因此 parity 判据是**相对容差 1e-6，不是逐位相等**；任何要求 bit-exact 的判据在构造上不可满足。
- **CI 门**：见 Design 的"CI 门"段；过滤式 ctest、必需门。
- **回归示例**：`regressions/lammps-iface-01-meta.py`。**纯 Python 公开面，不启子进程**（`regressions/README.md:5-6` 明令禁止 subprocess，现存脚本无一例外）。断言两条**十进制字面量 golden**：`preset_energy_ratio("real","metal")` 对 `0.043364104`、`preset_energy_ratio("metal","real")` 对 `23.060548`，容差 rtol 1e-7；以及 `LAMMPS_META_SCHEMA == 2` 这一**字面量**（不得写成 `== export.LAMMPS_META_SCHEMA` 的自比）。goldens 来自 Domain basis 中由 SI 精确定义常量手算的值，**不是**用 molpy 现算再拿 molpy 去比（molpy-vs-molpy 自证毫无约束力）；往返乘积可作辅助自洽检查，但不得充当 golden。脚本头必须带 `regressions/oplsaa-parameterizer-02-heads.py:12-18` 形制的 Provenance 块，记录 formula / tool / capture command / date / tolerance。

## Out of scope

- 三层拆分本体：`LammpsView` / `SpeciesMap` / staging / scatter（`lammps-iface-02-view`）、`GraphBridge` / comm / `DynamicExec` / `GraphedExec`（`03-graph`）、`ContractImpl` / `NeighborNeeds` / `EvalResult` / 派发 / `CutoffGraphImpl` / facade 削薄（`04-facade`）。
- 同属后续的顺带修复：pair_coeff 的 Z 对 meta `species` 校验与整数数组读取（`02-view`）；`eflag_atom` / `vflag_atom` 报错、`num_models` 记录（`04-facade`）。
- PBC 正确性（`REQ_GHOST`、双向边、per-atom energy 求和语义、newton 门控、排序处置）：`lammps-iface-05-pbc`。
- TorchScript 执行器本体（仅在枚举 docstring 中记录其为预期扩展值）、espaloma / topology-FF impl、`fixed-graph` 契约的实现。
- 性能优化与 `n_atoms` / `e_max` 的语义变更。
- `export_for_lammps` 的用户可见参数签名变更（`cuda_graph=` 参数名保持不变）。
- `.claude/notes/architecture.md` 对 `interface/` 的蓝图补齐（scope gap，链结束后一次 `/mol:map`，不阻塞本 spec）。
- **链级待办（由链收尾检查，不随本 spec 关闭而消失）**：`Contract::fixed_graph` 在链结束时要么获得实现、要么按 schema 升版规则删除；`test_lammps_contract` 的取位债（见 Design"测试放置"）在 `04-facade` 复检。
