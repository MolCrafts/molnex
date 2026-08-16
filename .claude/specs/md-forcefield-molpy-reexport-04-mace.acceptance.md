---
spec: md-forcefield-molpy-reexport-04-mace
created: 2026-08-16
criteria:
  - id: ac-001
    summary: edge_vectors consumes edge_diff as a detached remainder
    type: code
    pass_when: |
      edge_vectors(pos, edge_index, edge_diff=imaged) equals the supplied
      edge_diff (Å, atol=1e-12). geometry.py has no frac.round / MIC /
      pos[t]-pos[s]+shifts PBC path. edge_diff is keyword-only.
    status: pending
  - id: ac-002
    summary: L=10 MIC pair is molnex (-1,0,0) with autograd on pos
    type: scientific
    pass_when: |
      source (0.5,0,0), target (9.5,0,0), L=10: edge_diff=(-1,0,0),
      length=1, ∂d/∂pos_s=(1,0,0), ∂d/∂pos_t=(-1,0,0), atol=1e-12.
    status: pending
  - id: ac-003
    summary: Non-MIC supplied edge_diff is not re-wrapped
    type: code
    pass_when: |
      edge_diff=(+5,0,0) stays (+5,0,0), not MIC (-1,0,0).
    status: pending
  - id: ac-004
    summary: shifts is rejected as a PBC channel on every MACE seam
    type: code
    pass_when: |
      edge_vectors / energy_core / Matpes.energy_forces / OMol.energy_forces
      reject shifts=; a 6th positional tensor to energy_core raises TypeError.
    status: pending
  - id: ac-005
    summary: forward requires edges.edge_diff and ignores leftover shifts
    type: code
    pass_when: |
      Missing edge_diff raises ValueError; only-shifts does not silently
      image; both keys match edge_diff-only energy at 1e-12 eV.
    status: pending
  - id: ac-006
    summary: energy_core stays the compile seam with keyword-only edge_diff
    type: code
    pass_when: |
      Positional args stay (positions, Z, edge_index, batch, num_graphs);
      _compute_energy is the energy_core function object.
    status: pending
  - id: ac-007
    summary: Variant energy_forces takes edge_diff not shifts
    type: code
    pass_when: |
      energy_forces(..., edge_diff=) matches forward at 1e-12 eV; no
      production call in src/molzoo/mace/ passes shifts into energy_core.
    status: pending
  - id: ac-008
    summary: Regression reproduces MIC goldens without third-party runtime
    type: runtime
    pass_when: |
      regressions/md-forcefield-molpy-reexport-04-mace.py reproduces
      edge_diff=(-1,0,0), length=1, remainder=(-10,0,0), autograd signs;
      anti-MIC probe kept; shifts= and missing-edge_diff raise.
    status: pending
  - id: ac-009
    summary: Spec diff stays inside molzoo.mace plus its tests and regression
    type: code
    pass_when: |
      Diff touches only src/molzoo/mace/{geometry,potential,variants}.py,
      the three test modules, and the regression. No molix/molpot/mace-nve.
    status: pending
out_of_scope:
  - molix / molpot edits
  - mace-nve
  - Restoring shifts as PBC
---

# Acceptance — md-forcefield-molpy-reexport-04-mace

MACE 只信上游 `edge_diff`；`shifts` 不再是 PBC 通道；L=10 对是 molnex `(-1,0,0)`，力绑在 pos 上。
