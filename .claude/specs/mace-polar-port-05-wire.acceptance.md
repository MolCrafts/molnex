---
slug: mace-polar-port-05-wire
created: 2026-08-30
criteria:
  - id: ac-001
    summary: MACEPolarSpec dump GTO knobs match Polar-1-M
    type: code
    pass_when: |
      TestMACEPolarSpec defaults include num_recursion_steps=2, multipole_sigma=1.5,
      field_widths=[1.5, 3.0], field_norms=[20,20,0.5,0.5], add_local_electron_energy=True.
    status: verified
    last_checked: 2026-08-30
    verified_by: tests/test_molzoo/test_mace/test_spec.py::TestMACEPolarSpec
  - id: ac-002
    summary: PolarPotential builds, energy is finite and translation-invariant
    type: code
    pass_when: |
      Tiny PolarPotential writes graphs.energy (1,) finite eV; translating positions
      by 2.5 Å leaves energy within 1e-8; missing total_charge/spin defaults to Q=0 M=1;
      submodule names include lr_source_maps, field_dependent_charges_maps, fukui_source_map.
    status: verified
    last_checked: 2026-08-30
    verified_by: tests/test_molzoo/test_mace/test_polar.py::TestPolarPotential
  - id: ac-003
    summary: POLAR_REMAP fills parameters and returns unexpected aux keys
    type: code
    pass_when: |
      Synthetic official-dialect state (nested linear_mid.weight, kspace_cutoff dropped,
      aux_head unexpected) loads with no unfilled nn.Parameter.
    status: verified
    last_checked: 2026-08-30
    verified_by: tests/test_molzoo/test_mace/test_polar.py::TestPOLARRemap
  - id: ac-004
    summary: GTOFieldFeatures dump widths are ir_mul dim 8 and translation invariant
    type: code
    pass_when: |
      GTOFieldFeatures(σ=1.5, lmax=1, widths=[1.5,3.0]) out_dim=8; rigid translation
      of sources leaves features within 1e-8.
    status: verified
    last_checked: 2026-08-30
    verified_by: tests/test_molpot/test_potentials/test_gto/test_features.py
  - id: ac-005
    summary: Regression pins dump GTO knobs and tiny Polar forward
    type: runtime
    pass_when: |
      regressions/mace-polar-port-05-wire.py exits 0; dump literals; energy (1,) finite
      and translation-invariant; POLAR_REMAP on_unexpected=return; no mace/e3nn import.
    status: verified
    last_checked: 2026-08-30
    verified_by: regressions/mace-polar-port-05-wire.py
out_of_scope:
  - Official MACE-POLAR-1-M.model energy/force parity (needs out-of-tree cueq convert)
  - k-space PBC and applied electric field
---

# Acceptance — mace-polar-port-05-wire

本步完成 = `PolarPotential(MACEPolarSpec)` 在孤立分子上算出有限、平移不变的 eV 能量，子模块名对齐官方 remap，`POLAR_REMAP` 能装合成 cueq 键。不对照官方 `.model` 数值。
