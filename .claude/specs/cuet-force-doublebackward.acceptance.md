---
slug: cuet-force-doublebackward
criteria:
  - id: ac-001
    summary: Minimal cuet-only reproduction of the broken double-backward (RED)
    type: code
    pass_when: |
      scripts/cueq_db/minimal_repro.py is <=30 lines of body, depends only on
      cuequivariance / cuequivariance_torch / torch (no e3nn, no molnex model),
      builds a position-dependent input -> cuet.Linear with l>0 irreps -> scalar
      energy, computes g = dE/dx with create_graph=True, backprops g.pow(2), and
      asserts the Linear weight-gradient L1 == 0 on the CURRENT (molrep-style)
      instantiation. Running it prints BROKEN, reproducing the bug deterministically.
    status: pending
  - id: ac-002
    summary: Root cause isolated to a single instantiation knob (toggle BROKEN<->OK)
    type: code
    pass_when: |
      The repro contains two cuet.Linear(l>0) instances differing by exactly one
      construction variable (e.g. layout ir_mul vs mul_ir, shared/internal_weights,
      method, or weight organization reverse-engineered from the teacher's loaded
      cuet.Linear). One yields weight-grad L1 == 0, the other > 0. The differing
      knob is named and documented in the script header + spec. The teacher-probe
      script extracts the working construction signature from OMOL-cueq.model.
    status: pending
  - id: ac-003
    summary: Fix applied in molrep, pure cuet, repro flips RED->GREEN
    type: code
    pass_when: |
      cuet equivariant Linear (and any affected ChannelWiseTensorProduct /
      SymmetricContraction) instantiation in src/molrep/interaction/ is changed to
      the double-backward-capable construction, using ONLY cuequivariance (no e3nn).
      After the fix, the minimal repro on the molrep construction asserts weight-grad
      L1 > 0 (GREEN).
    status: pending
  - id: ac-004
    summary: Full MACEOMol force double-backward yields non-zero parameter grads
    type: code
    pass_when: |
      Building MACEOMol, computing forces and backpropagating a force loss yields
      non-zero gradient for a strict majority of parameters (>50%, ideally ~all),
      vs 0/73 before. Asserted by tests/test_molrep/test_cueq_double_backward.py
      and a full-model probe.
    status: pending
  - id: ac-005
    summary: Port consistency vs official OMOL preserved after the fix
    type: scientific
    pass_when: |
      With the fixed instantiation, MACEOMol still loads official OMOL weights and
      matches the teacher on isolated molecules to |dE| <= 1e-6 eV and
      max|dF| <= 1e-6 eV/Ang (scripts/omol_port/verify_e2e.py PASS; measured ~1e-9).
      If the fix changes weight layout, a weight-conversion is provided and the
      1e-6 bar still holds.
    status: pending
  - id: ac-006
    summary: Forces actually train; energy training not regressed
    type: scientific
    pass_when: |
      A short fine-tune (few epochs, small frame set, CP2K labels) shows the force
      MAE strictly decreasing across epochs (forces now learn, vs frozen before),
      and the energy MAE remains at the pre-fix level (<= a few meV/atom on the
      smoke set). Demonstrates the fix unblocks force-supervised training.
    status: pending
  - id: ac-007
    summary: Full check and test suite pass
    type: runtime
    pass_when: |
      `python -m pytest tests/ -v` runs green (with the documented LD_LIBRARY_PATH
      for fused ops). The new regression test is included. No prior test regresses.
    status: pending
---

# Acceptance criteria

- ac-001 / ac-002 是"复现优先"的硬核:先用 ≤30 行纯 cuet 片段确定性复现 BROKEN,再用单变量 toggle 把根因钉死(对照 teacher 反推出的可工作构造)。这是整个 spec 的钥匙。
- ac-003 / ac-004 是修复本体:molrep 内纯 cuet 改动,最小复现 RED→GREEN,全模型力二阶反传恢复。
- ac-005 是不退化护栏:修实例化不得动数值,仍对 teacher 1e-9。
- ac-006 证明修复确有实效(力真的能训、能量不退化)。
- ac-007 全量门禁。

科学类(ac-005/006)需人工/外部评估:一致性脚本与短训练曲线,不由 /mol:impl 自动判定。
