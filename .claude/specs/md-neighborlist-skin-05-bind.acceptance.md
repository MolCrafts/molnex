---
slug: md-neighborlist-skin-05-bind
criteria:
  - id: ac-001
    summary: build(batch) returns the same batch and binds edges by reference
    type: code
    evaluator_hint: "pytest tests/test_molix/test_md/test_neighbors.py::TestNeighborListBind"
    pass_when: |
      nl.build(batch) is batch; batch["edges", "edge_index"] is nl.edge_index and
      batch["edges", "shifts"] is nl.shifts (object identity, not just equal
      values); batch["edges"].batch_size == torch.Size([nl.capacity]); and
      set(batch["edges"].keys()) == {"edge_index", "shifts"}.
    status: pending
  - id: ac-002
    summary: build replaces any pre-existing edges namespace wholesale
    type: code
    evaluator_hint: "pytest tests/test_molix/test_md/test_neighbors.py::TestNeighborListBind"
    pass_when: |
      Building into a batch whose "edges" already carries edge_diff, edge_dist and
      a shorter edge_index leaves exactly {"edge_index", "shifts"} at
      batch_size=[capacity], with both leaves identical (is) to the list buffers.
    status: pending
  - id: ac-003
    summary: build is a binding operation - rebuild_count untouched, ago reset
    type: code
    evaluator_hint: "pytest tests/test_molix/test_md/test_neighbors.py::TestNeighborListBind"
    pass_when: |
      nl.rebuild_count is unchanged across nl.build(batch) (0 on a freshly
      constructed list) while nl.ago == 0 after it, and building a batch that
      carries dilated positions leaves num_edges equal to the edge count at those
      positions rather than the constructor's.
    status: pending
  - id: ac-004
    summary: the by-reference tie survives in-place rebuild and update
    type: code
    evaluator_hint: "pytest tests/test_molix/test_md/test_neighbors.py::TestNeighborListBind"
    pass_when: |
      After nl.rebuild(compressed) and after nl.update(compressed_batch),
      batch["edges", "edge_index"] is still nl.edge_index, num_edges has changed,
      and torch.equal holds between the batch-read leaves and the list buffers for
      both edge_index and shifts. A bare nl.to(torch.float32) on a hand-bound batch
      does NOT keep the tie (documented consequence, asserted).
    status: pending
  - id: ac-005
    summary: build validates the batch cell against the constructor cell
    type: code
    evaluator_hint: "pytest tests/test_molix/test_md/test_neighbors.py::TestNeighborListBind"
    pass_when: |
      ("graphs", "cell") equal to the constructor cell builds cleanly as both (3,3)
      and (1,3,3); a cell perturbed by 0.01 A raises ValueError matching "cell";
      a (2,3,3) cell raises ValueError naming the batch size; and in every accepted
      case nl.cell is unchanged (the constructor cell remains the owner).
    status: pending
  - id: ac-006
    summary: build refuses a mismatched batch instead of silently casting
    type: code
    evaluator_hint: "pytest tests/test_molix/test_md/test_neighbors.py::TestNeighborListBind"
    pass_when: |
      A batch missing ("atoms", "pos") raises ValueError naming the key; an 8-atom
      pos against a 27-atom list raises ValueError containing both 8 and 27; a
      float32 pos against a float64 list raises ValueError naming both dtypes; a
      meta-device pos against a CPU list raises ValueError naming both devices. No
      case casts, and none reaches the neighbour kernel.
    status: pending
  - id: ac-007
    summary: update() dispatches on TensorDict vs tensor with identical outcomes
    type: code
    evaluator_hint: "pytest tests/test_molix/test_md/test_neighbors.py::TestNeighborListBind"
    pass_when: |
      Two identically constructed lists driven over the same displacement schedule,
      one via update(pos) and one via update(batch), produce identical update()
      return sequences and identical ago / rebuild_count / ndanger, with
      torch.equal edge_index and shifts and equal num_edges; update(batch) with a
      re-cast pos raises the same ValueError as build; and molix.md.neighbors
      exposes no update_td / update_pos twins.
    status: pending
  - id: ac-008
    summary: PeriodicPotentialForceField delegates binding to NeighborList.build
    type: code
    evaluator_hint: "pytest tests/test_molix/test_md/test_forcefield.py::TestPeriodicPotentialForceField"
    pass_when: |
      src/molix/md/forcefield.py contains no _bind_neighbors; the constructor and
      _apply both reach the binding through self.neighbors.build(self._work);
      test_dead_edge_padding_is_invisible and
      test_rebuild_is_visible_through_the_bound_buffers pass with no assertion,
      tolerance or expected-value edit (including rebuild_count == 1); a new test
      shows that after ff.to(torch.float32) a rebuild_neighbors still changes the
      energy; and LennardJonesCutForceField still reads neighbors.edge_index /
      .shifts directly.
    status: pending
  - id: ac-009
    summary: NeighborStrategy declares build and the widened update
    type: code
    evaluator_hint: "pytest tests/test_molix/test_md/test_neighbors.py tests/test_molix/test_md/test_forcefield.py"
    pass_when: |
      NeighborStrategy declares build(batch) -> TensorDict and annotates update as
      TensorDict | torch.Tensor -> bool; isinstance(nl, NeighborStrategy) is True;
      the whole suite passes with the _Recorder stub in test_forcefield.py
      unchanged.
    status: pending
  - id: ac-010
    summary: schema keys, user guide and docstrings record the new ownership
    type: docs
    pass_when: |
      CLAUDE.md's "Post-collate batch schema" block lists ("edges", "shifts")
      (E, 3) [optional] and ("graphs", "cell") (B, 3, 3) [optional] with their
      units (A) and their in-tree producers/consumers, plus the note that on the MD
      bind path edges.batch_size == [capacity] with live edges in [0, num_edges);
      no collate source file is modified. docs/molix/user-guide/md.md shows the
      two-statement nl.build(batch) / nl.update(batch) idiom. The neighbors.py
      docstrings state (a) that the list owns "edges" once bound and replaces it,
      (b) that to() severs the tie and the owner must re-bind, and (c) that
      nl.build(batch).update(batch) calls TensorDict.update, not the policy.
      CHANGELOG.md gains one [Unreleased]/Added bullet for NeighborList.build.
    status: pending
  - id: ac-011
    summary: regression script reproduces the hand-derived bind goldens
    type: runtime
    pass_when: |
      `PYTHONPATH=src python regressions/md-neighborlist-skin-05-bind.py` prints OK
      and exits 0, reproducing its hard-coded literals on the 64-atom lattice
      (spacing 3.0 A, 12 A cube, cutoff 3.5 A, skin 1.5 A, float64 CPU):
      r_build == 5.0, num_edges == 1152, capacity == 1556, build returns the same
      batch with edges leaves identical (is) to the list buffers at
      batch_size=[1556] and rebuild_count == 0; 20 update(batch) calls under a
      +0.2 A/step rigid translation return True exactly at updates {4, 8, 12, 16,
      20} with rebuild_count == 5 and ndanger == 0; reading through the batch after
      the run gives num_edges == 1152, a maximum reconstructed distance of 5.0 A
      and exactly 384 pairs within 3.5 A; and a second list driven with raw
      update(pos) ends torch.equal to the first. No third-party oracle is imported
      or subprocessed at runtime.
    status: pending
---

# Acceptance criteria

- **ac-001 / ac-002 / ac-003 — the bind contract.** Identity (`is`), not equality, is the
  binding bar: a TensorDict that copied on assignment would freeze the PES while every
  value-based assertion still passed. ac-002 pins "the list owns `edges` once bound" (and
  subsumes the stale `edge_diff` / `edge_dist` stripping the force field used to do by
  hand); ac-003 pins that `build` is a *binding* operation, which is exactly what keeps
  `.to()` re-syncs out of the `rebuild_count` diagnostic and keeps the existing
  force-field test green.
- **ac-004 — liveness.** The reason the surface exists: an in-place rebuild must be
  visible through the batch with no re-binding and no shape change. The negative half
  (a bare `to()` does not keep the tie) is pinned deliberately so the documented trade
  cannot silently become a lie.
- **ac-005 / ac-006 — refusal, not repair.** The constructor cell stays the owner and the
  list never casts a caller's positions; both failure modes convert a silently wrong
  energy (wrong cell, promoted dtype, wrong atom count) into a `ValueError` naming both
  sides. The `meta` device is what makes the device half testable without CUDA.
- **ac-007 — one method, two input types.** The dispatch is only worth having if the two
  paths are indistinguishable in outcome; the explicit "no `update_td` / `update_pos`
  twins" clause is part of the bar, since a twin pair would pass every behavioural
  assertion.
- **ac-008 / ac-009 — the ownership move.** The whole point of the link is that the
  binding knowledge exists in exactly one place. The unmodified-assertions clause is the
  guard: if either existing periodic test needs its expectations edited, the move changed
  semantics and must be reported, not accommodated. The cast test covers the one real
  regression risk (`_apply` re-sync).
- **ac-010 — documentation of live keys.** `("edges", "shifts")` and `("graphs", "cell")`
  are already produced and consumed in-tree; leaving them undocumented is how the next
  reader concludes the batch schema forbids them. The `TensorDict.update` collision is
  called out explicitly because `nl.build(batch).update(batch)` is a plausible-looking
  line that silently does nothing.
- **ac-011 — reproducibility.** The oracle-free, integer-golden record that the bind, the
  policy and the shift reconstruction all survive as one public-API scenario, read back
  entirely through the batch.
