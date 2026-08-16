---
spec: md-forcefield-molpy-reexport-03-potentials
created: 2026-08-16
criteria:
  - id: ac-001
    summary: NeighborListBuilder prefers batch edge_diff or edge_dist
    type: code
    pass_when: |
      pos (0,0,0)/(9,0,0) Å, cell=10 I, edges with edge_dist=1 or
      edge_diff=(-1,0,0) return distances=1 Å not 9. Only edge_dist=2
      (no vectors) returns 2 Å. Bidirectional edges collapse to source<target.
    status: pending
  - id: ac-002
    summary: Placed-geometry path never runs a second PBC
    type: code
    pass_when: |
      With edge_diff or edge_dist present, NeighborListBuilder._periodic
      and molpy.NeighborQuery are not called.
    status: pending
  - id: ac-003
    summary: Missing displacement fields keep the original builder
    type: code
    pass_when: |
      build without edge_diff/edge_dist still matches today's _open/_periodic.
    status: pending
  - id: ac-004
    summary: LJCutPotential.energy consumes caller distances only
    type: code
    pass_when: |
      energy(..., distances=d) equals lj126_pair_energy at d even when d
      disagrees with ||pos_i-pos_j||; method does not read cell.
    status: pending
  - id: ac-005
    summary: Potentials LJ wrap energy is 16128 kcal/mol at r=1 Å
    type: scientific
    pass_when: |
      ForceField.def_pairstyle("lj/cut", cutoff=5).with_params(atoms
      epsilon=1, sigma=2).to_potentials().calc_energy on the wrap fixture
      with r=1 Å equals 16128 kcal/mol (rel_tol=1e-10).
    status: pending
  - id: ac-006
    summary: calc_forces stays attached to the pos leaf on placed edges
    type: code
    pass_when: |
      F_0=(+195072,0,0), F_1=(-195072,0,0) kcal mol⁻¹ Å⁻¹
      (atol=1e-6, rtol=1e-10); pair energy depends on the pos leaf.
    status: pending
  - id: ac-007
    summary: Regression script pins wrap LJ at r=1 not r=9
    type: runtime
    pass_when: |
      regressions/md-forcefield-molpy-reexport-03-potentials.py exits 0,
      energy equals literal 16128 kcal/mol and is not ≈-4.816e-4; no
      third-party import.
    status: pending
  - id: ac-008
    summary: Google docstrings state distances are already placed
    type: docs
    pass_when: |
      NeighborListBuilder.build, Potentials._scaled_pairs, and
      LJCutPotential.energy document Å / kcal/mol and no pair PBC.
    status: pending
out_of_scope:
  - molix NeighborList write surface (02)
  - molzoo MACE (04)
  - Changing LJCutPotential.energy signature
---

# Acceptance — md-forcefield-molpy-reexport-03-potentials

molpot 在 batch 已有位移/距离时禁止二次 PBC。跨面 LJ 用 r=1 得到 16128 kcal/mol，力绑在 pos 上。
