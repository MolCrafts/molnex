---
slug: mace-polar-port-03-field
created: 2026-08-30
criteria:
  - id: ac-001
    summary: SparseUvu l=0 contraction is 6.0 and vector dots are z-rotation invariant
    type: code
    pass_when: |
      TestSparseUvuTensorProduct asserts 1x0e W=1, x1=2, x2=3 → 6.0;
      two independent 0e copies with identity W stay unmixed;
      1o×1o→0e equals a·b/√3 and is invariant under a z-rotation;
      1o×0e→1o scales the vector by scalar/√3; non-uvu and non-2-D inputs raise.
    status: verified
    last_checked: 2026-08-30
    verified_by: tests/test_molrep/test_interaction/test_polar/test_uvu.py
  - id: ac-002
    summary: Field update emits charges⊕2x0e with official submodule names and MLP widths
    type: code
    pass_when: |
      Tiny AgnosticEmbeddedOneBodyVariableUpdate (4x0e+4x1o, charges 2x0e+2x1o)
      outputs (N, 10); RadialMLP hs is [8, 64, 64, 64, 4]; readout linear_1
      irreps_out dim is 160 (32×Y_1 gate); submodule names include
      potential_embedding.{potential_linear,node_feats_linear,charge_embedding},
      source_embedding, dot_products, nonlinearity, tp_out, readout.{linear_1,
      linear_mid, linear_2}; 0e slices of the output are z-rotation invariant.
    status: verified
    last_checked: 2026-08-30
    verified_by: tests/test_molrep/test_interaction/test_polar/test_field.py
  - id: ac-003
    summary: Field readout is (N,) from q_induced+q_0 with official MLP width
    type: code
    pass_when: |
      OneBodyMLPFieldReadout outputs (N,); swapping charges_0 and
      charges_induced is a no-op; mlp.hs is [8, 128, 128, 128, 1]; submodule
      names are linear_up_q, linear_up_v, dot_products_q, dot_products_v, mlp.
    status: verified
    last_checked: 2026-08-30
    verified_by: tests/test_molrep/test_interaction/test_polar/test_field.py
  - id: ac-004
    summary: polar/ does not import molpot
    type: code
    pass_when: |
      AST walk of src/molrep/interaction/polar/*.py finds no molpot import.
    status: verified
    last_checked: 2026-08-30
    verified_by: tests/test_molrep/test_interaction/test_polar/test_field.py::TestPolarFieldPackage
  - id: ac-005
    summary: Regression pins dump MLP widths and tiny forward shapes
    type: runtime
    pass_when: |
      regressions/mace-polar-port-03-field.py exits 0 with hard-coded
      l=0 product 6.0, Polar-1-M n_0e=512 MLP lists [1024,64,64,64,512] and
      [1024,128,128,128,1], 32×Y_1 dim 128, charges⊕2x0e dim 10, tiny update
      (N,10) and readout (N,); no third-party MACE import.
    status: verified
    last_checked: 2026-08-30
    verified_by: regressions/mace-polar-port-03-field.py
out_of_scope:
  - GTO Hartree kernel and Fukui equilibration (04)
  - Polar potential wiring, POLAR_REMAP, cueq convert, official energy parity (05)
---

# Acceptance — mace-polar-port-03-field

本步完成 = `molrep.interaction.polar` 提供原生 `ir_mul` SparseUvu、单体场更新与场读出，子模块名对齐官方 remap，l=0 收缩为 6.0，0e 对 z 旋转不变。无能量、无 molpot。
