---
slug: mace-polar-port-01-backbone
created: 2026-08-30
criteria:
  - id: ac-001
    summary: MACEPolarSpec pins Polar-1-M dumped backbone defaults
    type: code
    pass_when: |
      TestMACEPolarSpec asserts defaults r_max=6.0, num_bessel=8, l_max=3,
      num_features=512, max_hidden_l=1, num_interactions=2, correlation=3,
      mlp_dim=16, edge_channels=128, keep_last_layer_irreps=True,
      last_readout_bias=True, interaction="residual", readout="per_layer",
      distance_transform="none", pair_repulsion="none", conditioning="none";
      Polar is not a MACEOMolSpec; l_max=0 and num_interactions=1 raise;
      spec.py stays torch-free.
    status: pending
  - id: ac-002
    summary: Residual stack requires edge_channels, not MACEOMolSpec
    type: code
    pass_when: |
      MACESpec(interaction="residual") without edge_channels raises TypeError
      matching edge_channels; MACEEncoder(MACEPolarSpec) succeeds; encoder
      residual branch does not isinstance-check MACEOMolSpec.
    status: pending
  - id: ac-003
    summary: Polar hidden and mid-edge share max_hidden_l not l_max-1
    type: code
    pass_when: |
      Tiny Polar (l_max=2, max_hidden_l=1, C=16, edge_channels=8) last-layer
      width is 64 (16x0e+16x1o); interactions[1].linear_up maps hidden to
      8x0e+8x1o (width 32), not _irreps(l_max-1=1 wait l_max-1=1 coincidentally
      — use l_max=3 tiny Polar last width 64 still, mid-edge 8x0e+8x1o dim 32
      not 8x0e+8x1o+8x2e dim 72).
    status: pending
  - id: ac-004
    summary: Polar last readout is NonLinearBiasReadout; last features are hidden
    type: code
    pass_when: |
      Tiny Polar readouts[0] is LinearReadout, readouts[1] is
      NonLinearBiasReadout; joint_embedding is None; last-layer feature width
      equals first-layer hidden width (64), not num_features (16).
    status: pending
  - id: ac-005
    summary: MatPES and OMOL last layers stay scalars
    type: code
    pass_when: |
      Tiny MatPES and tiny OMOL last-layer width equals num_features (16);
      test_variants.py::test_last_layer_keeps_scalars_only still passes.
    status: pending
  - id: ac-006
    summary: Polar last layer is translation-invariant at fp64
    type: code
    pass_when: |
      Tiny Polar layer_features[-1] is translation-invariant to atol=1e-10
      at fp64 on the cluster fixture (relative geometry).
    status: pending
  - id: ac-007
    summary: Regression pins Polar backbone dump literals without energy parity
    type: runtime
    pass_when: |
      regressions/mace-polar-port-01-backbone.py exits 0 with hard-coded
      r_max=6.0, num_features=512, max_hidden_l=1, E0[H]=-15.99000358581543,
      83*512=42496, tiny Polar last dim 64, tiny MatPES last dim 16; Polar
      is not MACEOMolSpec; no third-party MACE import.
    status: pending
out_of_scope:
  - Polar energy/force parity vs official checkpoint (05)
  - GTO, Fukui, cueq conversion
  - keep_last_layer_irreps on MACESpec base
---

# Acceptance — mace-polar-port-01-backbone

本步完成 = `MACEEncoder(MACEPolarSpec)` 建出与 dump 一致的残差骨干（`max_hidden_l=1` 的 hidden 与 mid-edge、末层完整 hidden、Linear 再 NonLinearBias 读出），且 MatPES/OMOL 末层仍为标量。无能量对照。
