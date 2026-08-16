---
spec: md-forcefield-molpy-reexport-02-nblist
created: 2026-08-16
criteria:
  - id: ac-001
    summary: NeighborList.build writes live edge_diff and edge_dist
    type: code
    pass_when: |
      After NeighborList.build(batch), batch["edges"] keys are exactly
      {edge_index, shifts, edge_diff, edge_dist}; edge_index/shifts are
      the list buffers by reference; edge_diff equals pos[target]-pos[source]+shifts
      on the same pos tensor (not detached).
    status: pending
  - id: ac-002
    summary: Hard-coded L=10 Å MIC pair matches textbook values
    type: scientific
    pass_when: |
      Cubic L=10 Å, r_s=(0.5,5,5), r_t=(9.5,5,5), cutoff=2 Å: directed
      edge s→t has edge_diff=(-1,0,0) Å, edge_dist=1 Å, rsq=1 Å²
      (atol=1e-12); raw is (9,0,0); reverse edge flips the sign.
    status: pending
  - id: ac-003
    summary: Live edge_diff carries pos autograd so F is nonzero
    type: scientific
    pass_when: |
      E=0.5*‖d_s→t‖² on the MIC pair yields F_s=(-1,0,0), F_t=(1,0,0)
      (atol=1e-12); detaching written edge_diff yields F=0.
    status: pending
  - id: ac-004
    summary: update(batch) rewrites geometry even without a rebuild
    type: code
    pass_when: |
      When update(batch) returns False after a rigid shift, edge_diff
      still equals pos[t]-pos[s]+S at the new positions.
    status: pending
  - id: ac-005
    summary: Integrator.eval_force consumes only ForceOutput
    type: code
    pass_when: |
      eval_force calls the bound pos->ForceOutput provider and casts to
      pos.dtype; Integrator has no rebuild attribute and never calls
      rebuild_neighbors.
    status: pending
  - id: ac-006
    summary: rebuilds_neighbors is removed from molix.md
    type: code
    pass_when: |
      grep of src/molix/md/ and tests/test_molix/test_md/ finds no
      identifier rebuilds_neighbors or rebuild_neighbors.
    status: pending
  - id: ac-007
    summary: MD.compute_force updates the list then calls pots.calc_*
    type: code
    pass_when: |
      compute_force(pos) calls nl.update on the working batch before
      pots.calc_energy/calc_forces; batch edges carry live MIC tensors.
    status: pending
  - id: ac-008
    summary: Regression reproduces the hard-coded MIC pair
    type: runtime
    pass_when: |
      regressions/md-forcefield-molpy-reexport-02-nblist.py asserts
      raw=(9,0,0), edge_diff=(-1,0,0), edge_dist=1, rsq=1 (atol=1e-12)
      plus F≠0; no third-party import.
    status: pending
  - id: ac-009
    summary: Two NeighborList types stay unmerged
    type: code
    pass_when: |
      molix.md.NeighborList is not molix.batch.neighbor.NeighborList.
    status: pending
  - id: ac-010
    summary: Google docstrings state Å units on the write/dispatch API
    type: docs
    pass_when: |
      NeighborList.build, NeighborList.update, MD.compute_force, and
      Integrator.eval_force have Google-style docstrings naming Å and
      stating edge_diff stays on the pos graph.
    status: pending
out_of_scope:
  - Potential-side PBC (03/04)
  - Merging the two NeighborList types
  - Verlet skin policy arithmetic
---

# Acceptance — md-forcefield-molpy-reexport-02-nblist

邻表写入 live MIC `edge_diff` / `edge_dist`；MD 先更新表再求值；积分器只吃 `ForceOutput`。教科书跨盒对 `L=10` → `edge_diff=(-1,0,0)`，且力绑在 `pos` 上。
