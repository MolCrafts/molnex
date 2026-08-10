---
slug: learnable-classical-ff-02-valence-topology
created: 2026-08-10
revised: 2026-08-10
criteria:
  - id: ac-001
    summary: Collate exposes angles as TensorDict columns atomi/atomj/atomk
    type: code
    pass_when: |
      After collate_molecules on two samples with local angles, the batch has
      batch["angles"]["atomi"] (or batch["angles", "atomi"]) and matching
      atomj/atomk as 1-D long tensors; second sample's indices equal local +
      n_atoms_0. Access form is nested TensorDict, not angle_index [3, N] as
      the primary leaf.
    status: pending
  - id: ac-002
    summary: Propers and impropers use atomi..atoml columns under namespaces
    type: code
    pass_when: |
      batch["propers"] has atomi, atomj, atomk, atoml; batch["impropers"] has
      the same four columns with atomi = center (molrs). Optional type column
      concatenates on dim 0 when present.
    status: pending
  - id: ac-003
    summary: PackedCache round-trips valence column buckets
    type: code
    pass_when: |
      PackedCache save/load of samples with angles/propers/impropers columns
      restores equal tensors; collate_packed agrees with collate_molecules
      leafwise on those columns. Single-file layout only.
    status: pending
  - id: ac-004
    summary: Batch topology is not a molpy Frame store
    type: code
    pass_when: |
      Collate/cache modules do not import molpy ForceField/Frame as the batch
      container; topology leaves are plain torch tensors inside TensorDict.
    status: pending
  - id: ac-005
    summary: Optional stack helper builds COO only for kernel call sites
    type: code
    pass_when: |
      A unit test stacks angles.atomi/j/k into [3, N] (and propers into [4, N])
      via a documented helper; the helper is not required by collate output.
    status: pending
  - id: ac-006
    summary: Existing bonds collate path remains green
    type: code
    pass_when: |
      Existing bond_index collate / packed tests still pass after valence work.
    status: pending
  - id: ac-007
    summary: Regression script pins nested column access and rebases
    type: runtime
    pass_when: |
      regressions/learnable-classical-ff-02-valence-topology.py (if present)
      asserts hard-coded rebased atomi/atomj/atomk tables via
      batch["angles"]["atomi"] form; exits 0 without third-party oracles.
    status: pending
  - id: ac-008
    summary: Full check and unit suite green
    type: runtime
    pass_when: |
      Project build.check and default pytest unit suite succeed after the change.
    status: pending
---

# Acceptance — learnable-classical-ff-02-valence-topology

Done means molix collate/cache carry valence connectivity as **nested
TensorDict column blocks** (`angles.atomi` …), rebased across molecules,
without using molpy as the runtime store and without making packed
`angle_index [3,N]` the public batch schema.
