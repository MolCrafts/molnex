---
slug: md-neighborlist-skin-02-prune
criteria:
  - id: ac-001
    summary: molix.nn export surface pins the removal and stays alphabetized
    type: code
    evaluator_hint: "pytest tests/test_molix/test_nn/test_init.py -v"
    pass_when: |
      tests/test_molix/test_nn/test_init.py exists and passes. It asserts, as a
      hard-coded literal, that molix.nn.__all__ == ["BatchAggregation",
      "KeyedMLP", "KeyedMLPSpec", "ScatterSum"]; that getattr(molix.nn, name)
      resolves for every entry; that the set of public non-module attributes of
      molix.nn equals set(molix.nn.__all__); that "NeighborList" is absent from
      both __all__ and hasattr(molix.nn, ...); and that
      importlib.util.find_spec("molix.nn.locality") is None.
    status: pending
  - id: ac-002
    summary: both dead neighbour-graph homonyms are gone from the tree
    type: code
    pass_when: |
      src/molix/nn/locality.py does not exist; src/molpot/graph/ does not exist
      (no radius.py, no leftover __pycache__); src/molix/nn/__init__.py contains
      no "from .locality import" line. A repo-wide grep for "radius_graph",
      "molpot.graph" and "nn.locality" over src/, tests/, benchmarks/, scripts/
      and docs/ returns no hit outside .claude/notes/architecture.md (blueprint,
      out of scope) and .claude/specs/ fossil text.
    status: pending
  - id: ac-003
    summary: molix.nn and molpot unit suites plus the full suite stay green
    type: code
    evaluator_hint: "pytest tests/test_molix/test_nn/ tests/test_molpot/ -v; then python -m pytest tests/ -v"
    pass_when: |
      python -m pytest tests/test_molix/test_nn/ tests/test_molpot/ -v exits 0
      with zero failures and zero new skips, and the full python -m pytest
      tests/ -v run has the same pass/fail/skip counts as before the change
      except for the tests added by ac-001. build.check (ruff check + ruff
      format --check + ty check) passes over src/ tests/ scripts/ regressions/.
    status: pending
  - id: ac-004
    summary: docstring and fossil-spec references point at live symbols only
    type: docs
    pass_when: |
      src/molpot/heads/charge_bond.py no longer mentions
      molix.nn.locality.NeighborList; its full_neighbor_list docstring cites
      :class:`molix.data.tasks.neighbor.NeighborList` and its symmetry=True
      default. In .claude/specs/mace-omol-port-02-pipeline-integration.md the
      unchecked optional item at line ~81 carries the parenthetical "(obsolete —
      molpot.graph removed by md-neighborlist-skin-02-prune; standalone edge
      sourcing is molix.md.NeighborList's job)", and `git diff` on that file
      shows no change to any [x] row or to any other section.
    status: pending
  - id: ac-005
    summary: regression script proves the prune and the surviving neighbour list
    type: runtime
    evaluator_hint: "PYTHONPATH=src python regressions/md-neighborlist-skin-02-prune.py"
    pass_when: |
      PYTHONPATH=src python regressions/md-neighborlist-skin-02-prune.py prints
      OK and exits 0. Using only public API and hard-coded literals (no
      third-party oracle, no network, no subprocess) it asserts: import molix.nn
      and import molpot succeed; molix.nn has no NeighborList attribute and none
      in __all__; find_spec("molix.nn.locality") is None; find_spec
      ("molpot.graph") is None; import molpot.graph.radius raises
      ModuleNotFoundError; molpot has no radius_graph attribute; and
      molix.data.tasks.neighbor.NeighborList(cutoff=1.5, symmetry=True).execute
      on a 3-atom chain at x = 0.0/1.0/2.0 Ang returns edge_index of shape (4, 2)
      with edge_dist allclose to 1.0 for all four entries (atol 1e-6).
    status: pending
---

# Acceptance criteria

**ac-001** pins the shrunken export list as a literal rather than a membership
check, so it fails both on a re-introduced `NeighborList` and on an `__all__`
that drifts out of alphabetical order (`notes.md:243`). The public-attribute
comparison filters `inspect.ismodule` — `molix.nn` defines no `__dir__`, so a
bare `set(dir()) == set(__all__)` would fail on the `mlp` / `scatter` submodule
attributes and is not the bar.

**ac-002** is the deletion itself. The grep clause tolerates exactly two
categories of surviving mention: the architecture blueprint (explicitly out of
scope, refreshed by `/mol:map`) and fossil spec prose, of which only the one
line named in ac-004 is edited.

**ac-003** encodes the "deletion-only" claim. A changed count anywhere in the
suite means the change was not, in fact, dead-code removal, and blocks the link.

**ac-004** covers the two prose edits. The `git diff` clause is load-bearing:
completed specs are historical record, so widening the annotation into a rewrite
of checked rows fails this criterion even if the parenthetical is correct.

**ac-005** is the standalone scenario. Its second half is the substantive claim
— the capability was duplicated, not lost — and its goldens are derivable on the
page (within a 1.5 Ang cutoff the 3-atom chain has half-pairs `(0,1)` and
`(1,2)`; the `(0,2)` separation of 2.0 Ang is outside; `symmetry=True` doubles
2 pairs to 4 edges, each 1.0 Ang), so no capture step and no oracle are
involved.
