---
slug: mace-subpackage-restructure-06-wire
criteria:
  - id: ac-001
    summary: molzoo.mace is a self-describing flat re-export package
    type: code
    pass_when: |
      src/molzoo/mace/__init__.py 的模块 docstring 逐条列出 spec / geometry /
      encoder / potential / checkpoint / variants 各自的单一职责（同
      src/molzoo/pinet/__init__.py:1-31 的形态），文件除 `from .x import Y` 外无
      逻辑，且显式 __all__ 至少含 MACE, MACESpec, MACEMatpes, MACEOMol,
      load_matpes_state_dict, load_omol_state_dict;
      `python -c "import molzoo.mace as m; assert set(m.__all__) <= set(dir(m))"` 退出 0。
    status: pending
  - id: ac-002
    summary: top-level and deep import paths survive the cutover
    type: code
    pass_when: |
      干净解释器中 `from molzoo import MACE, MACESpec, MACEMatpes, MACEOMol,
      load_matpes_state_dict, load_omol_state_dict` 与 `from molzoo.mace import MACE`
      均成功;`python -m pytest tests/ --collect-only -q` 报 0 error;
      scripts/matpes_port/run_nve.py:45、benchmarks/bench_mace_matpes.py:38、
      benchmarks/bench_trainer_throughput.py:26 三处 import 行未被修改且可导入。
    status: pending
  - id: ac-003
    summary: one unified PEP 562 lazy policy, stated in the module docstring
    type: code
    pass_when: |
      `python -c "import sys, molzoo; assert 'cuequivariance_torch' not in sys.modules;
      molzoo.MACE; assert 'cuequivariance_torch' in sys.modules"` 退出 0;
      src/molzoo/__init__.py 模块级除 typing / TYPE_CHECKING 外无 import,_LAZY 覆盖
      Allegro, AllegroSpec, MACE, MACESpec, PiNet, PiNetSpec, MACEMatpes, MACEOMol,
      load_matpes_state_dict, load_omol_state_dict,存在 __dir__,且 docstring 用一句
      话写明"所有模型符号一律惰性导出"及其理由。
    status: pending
  - id: ac-004
    summary: MACEMatpes / MACEOMol aliases keep the consumed surface intact
    type: code
    pass_when: |
      tests/test_molzoo/test_mace/test_variants.py 通过:以 run_nve.py:148-164 的
      kwargs 形（缩小配置）构造 MACEMatpes 成功;实例暴露 forward(td)、
      energy_forces(...)、validate_elements(Z)、buffer z_table,并可**位置**调用
      _compute_energy(pos, Z, edge_index, batch, num_graphs, shifts);kwargs 形与
      spec 形实例在同一 batch 上能量差 == 0;MACEOMol 同样通过其 17 个 kwargs 构造
      并暴露 forward / energy_forces。
    status: pending
  - id: ac-005
    summary: flat MACE modules deleted with no dangling importer
    type: code
    pass_when: |
      src/molzoo/mace_matpes.py 与 src/molzoo/mace_omol.py 均不存在;
      `rg -n "molzoo\.mace_(matpes|omol)" src tests scripts benchmarks docs/api`
      零命中（.claude/ 与 src/molzoo/specs/ 的锚点归 07,不计入）。
    status: pending
  - id: ac-006
    summary: MACE tests folded into the mirror; helper debt cleared
    type: code
    pass_when: |
      test_mace_encoder.py、test_mace_matpes.py、test_mace_omol.py 三个文件均已删除
      (test_mace.py 已由 02 迁移删除),其用例出现在 tests/test_molzoo/test_mace/
      下的 test_variants.py / test_encoder.py / test_potential.py / test_checkpoint.py;
      `rg -n "_make_batch|_full_edges" tests/test_molzoo` 零命中;ProductHead 用例只
      存在于 tests/test_molrep/test_readout/ 的镜像文件且无重名类;
      tests/test_molrep/ 下无 molzoo import。
    status: pending
  - id: ac-007
    summary: test-mirror gate covers molzoo/mace/ and passes
    type: code
    pass_when: |
      scripts/check_test_mirror.py 的 PINET_SPINE 含 "molzoo/mace/";
      `python scripts/check_test_mirror.py --strict-pinet` 退出 0。
    status: pending
  - id: ac-008
    summary: full check + suite green with no new skips
    type: runtime
    pass_when: |
      `ruff check src/ && ruff format --check src/` 退出 0;
      `python -m pytest tests/ -q` 通过数 >= 切换前基线,且 0 新增 error / skip /
      xfail（切换前基线在任务开始时记录）。
    status: pending
  - id: ac-009
    summary: regression example reproduces hard-coded pre-cutover baselines
    type: runtime
    pass_when: |
      `python regressions/mace-subpackage-restructure-06-wire.py` 退出 0:断言完整导入
      面、惰性策略,并对小配置 MACEMatpes 与 MACEOMol 断言参数张量数与参数元素总数
      等于脚本内硬编码的切换前基线（注释记录采集命令、commit、日期）。脚本只用公开
      API,不 import 任何第三方 oracle（无 mace-torch / e3nn / ASE),不 subprocess
      调用外部工具。
    status: pending
  - id: ac-010
    summary: README + API page describe the new package layout
    type: docs
    pass_when: |
      src/molzoo/README.md 的 MACE 段落以与 PiNet 表同构的"模块 → 职责"表列出
      molzoo/mace/ 的各模块,含一句 MACESpec 语义说明,且不再出现与实际签名不符的
      `MACE(MACESpec(...))` 示例;docs/api/molzoo.md 将单条 `::: molzoo.mace` 展开为
      逐模块 `:::` 条目。
    status: pending
---
