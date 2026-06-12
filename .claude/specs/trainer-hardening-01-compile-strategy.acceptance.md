---
slug: trainer-hardening-01-compile-strategy
criteria:
  - id: ac-001
    summary: every double-backward module is isolated by an eager fence
    type: code
    pass_when: |
      Every forward doing autograd.grad(..., create_graph=self.training)
      (force, stress, composer, sonata) routes that grad through a
      @torch.compiler.disable submodule; a test enumerates these sites and
      asserts the fence. No inline unfenced create_graph remains.
    status: pending
  - id: ac-002
    summary: plain torch.compile(model) is safe for force training
    type: runtime
    pass_when: |
      torch.compile(force_model) then train()+forces+loss.backward() does NOT
      raise the aot_autograd double-backward RuntimeError; the energy region
      runs eager when (training and compute_forces).
    status: pending
  - id: ac-003
    summary: user-facing compile_energy() removed, no CompilePolicy infra added
    type: code
    pass_when: |
      PiNet.compile_energy() and its benchmark call sites are gone; no
      molix.compile.CompilePolicy is introduced; maybe_compile and
      count_graph_breaks remain.
    status: pending
  - id: ac-004
    summary: energy graph-break gate + CUDA Graph gates documented
    type: docs
    pass_when: |
      cuda_graph_preconditions() returns unmet-gate names; docs/compile_strategy.md
      enumerates gate-1 (edge_index padding), gate-2 (double backward; dissolves
      under Phase-3), gate-3 (AMP get_scale host sync at steps/train.py:96,102),
      states profile-gated + opt-in.
    status: pending
  - id: ac-005
    summary: force numerics unchanged after Phase-1 refactor
    type: runtime
    pass_when: |
      forces from the fenced/refactored modules torch.allclose the
      pre-refactor baseline under backend="eager".
    status: pending
  - id: ac-006
    summary: Phase-3 functorch feasibility verdict recorded
    type: docs
    pass_when: |
      docs/compile_strategy.md Phase-3 records the spike result in gate order:
      (1) does torch.func.grad run eager through one MACE forward incl. cuEq;
      (2) does fullgraph trace without graph breaks on the target torch; with a
      go/no-go and the functional-encoder refactor cost estimate.
    status: pending
  - id: ac-007
    summary: full check and test suite pass
    type: runtime
    pass_when: |
      `ruff check src/ && ruff format --check src/` and
      `python -m pytest tests/ -v` both exit 0.
    status: pending
---

# Acceptance criteria
- **ac-001 (code)**：核心 —— 所有二阶反向模块统一 eager 围栏；当前 stress/composer/sonata 漏挂是要修的真 bug。
- **ac-002 (runtime)**：使用者无脑 `torch.compile(model)` 在 force 训练下不炸。
- **ac-003 (code)**：删 `compile_energy()` user API，不加 CompilePolicy 基建 —— 边界归开发者声明。
- **ac-004 (docs)**：CUDA Graph 三闸门 + profiler 门禁成文。
- **ac-005 (runtime)**：Phase-1 改造不改力数值。
- **ac-006 (docs)**：Phase-3 functorch 按门禁顺序的书面 go/no-go + 重构成本。
- **ac-007 (runtime)**：全量 check + 测试。
