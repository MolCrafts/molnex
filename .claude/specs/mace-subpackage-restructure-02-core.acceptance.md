---
slug: mace-subpackage-restructure-02-core
criteria:
  - id: ac-001
    summary: mace package shadows the flat module without breaking any import
    type: code
    pass_when: |
      After the move, all of `from molzoo import MACE, MACESpec`,
      `from molzoo.mace import MACE`, and
      `from molzoo.mace import EmbeddingBlock, InteractionBlock` succeed
      unmodified, `src/molzoo/mace.py` no longer exists,
      tests/test_molzoo/test_mace.py no longer exists (its cases live in the
      test_mace/ package and the 01 molrep mirrors; `pytest --collect-only`
      count is preserved), and `python -m pytest tests/test_molzoo -q` is green.
    status: pending
  - id: ac-002
    summary: spec.py carries no torch / cuEq / molrep dependency
    type: code
    pass_when: |
      An `ast.parse` of `src/molzoo/mace/spec.py` shows no import of torch,
      cuequivariance*, tensordict, molrep, molix or molzoo.*; and
      `MACEMatpesSpec(...).model_dump()` contains only JSON-native values
      (no `torch.Tensor`) — tests/test_molzoo/test_mace/test_spec.py passes.
    status: pending
  - id: ac-003
    summary: spec defaults reproduce the flat ctor defaults field by field
    type: code
    pass_when: |
      Every field default of MACEMatpesSpec equals the matching keyword default
      of `MACEMatpes.__init__` (mace_matpes.py:89-107, 17 kwargs) and every field
      default of MACEOMolSpec equals that of `MACEOMol.__init__`
      (mace_omol.py:65-85, 18 kwargs), asserted table-driven in
      tests/test_molzoo/test_mace/test_spec.py.
    status: pending
  - id: ac-004
    summary: spec validators reject contradictory config eagerly
    type: code
    pass_when: |
      ValueError is raised at construction for: len(atomic_energies) !=
      len(atomic_numbers); non-ascending or duplicated atomic_numbers;
      MACEMatpesSpec(num_interactions=1); MACEOMolSpec(l_max=0). No model is
      ever built from such a spec.
    status: pending
  - id: ac-005
    summary: importing a spec model does not import the encoder module
    type: code
    pass_when: |
      In a fresh subprocess, `import molzoo.mace; molzoo.mace.MACEMatpesSpec`
      leaves "molzoo.mace.encoder" absent from sys.modules.
    status: pending
  - id: ac-006
    summary: edge geometry is additive-PBC correct and differentiable
    type: code
    pass_when: |
      tests/test_molzoo/test_mace/test_geometry.py passes: edge_vectors matches
      hard-coded pos[target]-pos[source] values, equals the shift-free result
      plus `shifts` when given, `edge_lengths(..., keepdim=True)` has shape
      (E, 1), and autograd.grad w.r.t. pos equals the analytic +/- unit vectors
      with and without shifts.
    status: pending
  - id: ac-007
    summary: MACEEncoder state_dict is key- and shape-identical to the flat variants
    type: code
    pass_when: |
      For the tiny MatPES config (r_max=5.0, num_bessel=4, l_max=1,
      num_features=16, max_hidden_l=1, num_interactions=2, correlation=2,
      mlp_dim=8, radial_mlp=[8], atomic_numbers=[1,6,8]) and the tiny OMol
      config, `set(MACEEncoder(spec).state_dict())` equals the same-config flat
      model's key set and every shared key has equal shape and dtype
      (TestMACEEncoder::test_state_dict_parity_matpes / _omol).
    status: pending
  - id: ac-008
    summary: encoder hot path never reads the pydantic spec
    type: code
    pass_when: |
      `del encoder._spec` leaves every MACEEncoder primitive
      (node_attrs / angular_features / radial_features / layer_features) runnable,
      and an `ast.parse` of src/molzoo/mace/encoder.py finds no `self._spec`
      reference inside those method bodies.
    status: pending
  - id: ac-009
    summary: last-layer scalar features are translation and rotation invariant
    type: scientific
    pass_when: |
      In fp64 on CPU (use_fallback=True), the last entry of
      `MACEEncoder.layer_features(...)` for the tiny MatPES and tiny OMol specs
      changes by < 1e-10 under a rigid translation and under a random rotation
      applied to positions and to the PBC `shifts` vectors.
    status: pending
  - id: ac-010
    summary: regression example reproduces the embedded flat-variant goldens
    type: runtime
    pass_when: |
      `python regressions/mace-subpackage-restructure-02-core.py` exits 0,
      reproducing its hard-coded literal goldens (per-variant total energy and
      per-layer feature sum-of-squares, generated once from today's flat
      MACEMatpes / MACEOMol with deterministic linspace weights and literal
      coordinates) to |dE| <= 1e-9 eV and relative 1e-10 on the checksums,
      without importing molzoo.mace_matpes or molzoo.mace_omol at run time.
    status: pending
  - id: ac-011
    summary: README documents the MACESpec semantic migration
    type: docs
    pass_when: |
      src/molzoo/README.md:62-67 no longer shows `MACE(MACESpec(...))` with
      node_attr_specs; it states that `molzoo.mace.MACESpec` is now the shared
      foundation-variant config base, that the research-encoder config is
      `MACEResearchSpec`, and shows a `MACEMatpesSpec` / `MACEOMolSpec` example.
    status: pending
---

# Acceptance criteria

本合约覆盖 `mace-subpackage-restructure` 链的第 2 步（core）。ac-001 是链式前置闸门（包遮蔽同名模块后现有导入与测试必须全绿），ac-007 是贯穿全链的 `state_dict` 平价不变量，ac-010 是唯一的 `regressions/` 运行时判据（金标准以字面量内嵌，故 07-cleanup 删除 flat 模块后仍然有效）。
